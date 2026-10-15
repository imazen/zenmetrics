//! DMM (Chen et al., "Debiased Mapping for Full-Reference Image Quality
//! Assessment", IEEE TIP 2025; upstream `Baoliang93/DMM`, MIT, pip
//! `DMM_PyTorch`). Parity target: the upstream package, CPU, fp32.
//!
//! Upstream replaces VGG16's max-pools with L2 pooling (3×3 Hann-window RMS)
//! and takes relu3_3 (256ch @ h/4) and relu4_3 (512ch @ h/8) on a `[0,1]`
//! input. The exported trunk (`models/dmm/export.py`) is that VGG; the head
//! is host-side Rust because ONNX has no SVD: for each tap, 16×16 patches at
//! stride 4 are unfolded per channel and compared with
//! `torch.svd`-equivalent decompositions — the squared singular-value
//! distance weighted by the base-angle consistency `wt`, gated by an
//! SSIM-like channel-mean term `glb`, summed over the two taps.
//!
//! Preprocess (`prepare_image_adt`): `min(w,h) > 128` → PIL bilinear
//! short-side resize to `max(floor(min/48)·32, 128)` (`new_long =
//! int(new_short·long/short)`), then `ToTensor` → `[0,1]`. The resize is
//! Pillow's fixed-point u8 bilinear (`resize_bilinear_pil_u8`), not
//! `F.interpolate`.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, Tap};
use crate::preprocess::rgb8_to_planar_u8;
use crate::resize::resize_bilinear_pil_u8;
use crate::{Engine, Error, za};

const COLUMN: &str = "dmm";
const SCORE_REVISION: u32 = 1;
/// Patch kernel and stride of upstream's `nn.Unfold(16, stride=4)`.
const PATCH: usize = 16;
const STRIDE: usize = 4;

/// DMM (see module docs).
pub struct Dmm {
    backend: Backend,
    channels: Vec<usize>,
    min_size: usize,
}

/// One patch's SVD: `u` row-major 16×16, `s` the 16 singular values,
/// descending. Stored f32 — upstream computes the whole head in f32.
#[derive(Clone)]
struct PatchSvd {
    u: [f32; 256],
    s: [f32; 16],
}

/// One tap's per-patch decompositions plus its channel means (`y_mean`).
struct TapDecomp {
    /// `c * pt` patches, channel-major.
    patches: Vec<PatchSvd>,
    means: Vec<f32>,
    pt: usize,
}

impl Dmm {
    /// Load an exported model directory (`dmm_trunk.onnx`, `dmm.meta`, as
    /// written by `models/dmm/export.py`) on `engine`. Both engines run the
    /// trunk — the SVD head is host-side Rust either way.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join("dmm.meta"))?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(
                &dir.join("dmm_trunk.onnx"),
                channels.len(),
            )?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Trunk(Box::new(engine::ort_engine::OrtCudaTrunk::load(
                &dir.join("dmm_trunk.onnx"),
                "x",
                vec!["t3", "t4"],
            )?)),
            #[cfg(not(feature = "ort"))]
            Engine::OrtCuda => {
                return Err(Error::Engine("built without the ort feature".into()));
            }
        };
        Ok(Self {
            backend,
            channels,
            min_size,
        })
    }

    fn engine_name(&self) -> &'static str {
        let Backend::Trunk(t) = &self.backend else {
            return "none";
        };
        t.engine()
    }

    /// `prepare_image_adt`: conditional short-side PIL-bilinear resize, then
    /// `[0,1]`. Returns `(planar, h, w)` of the *resized* image.
    fn prep(img: &za::Rgb8In<'_>) -> (Vec<f32>, usize, usize) {
        let (h, w) = (img.height, img.width);
        let msize = h.min(w);
        let (planar, rh, rw) = if msize > 128 {
            let tar = (msize / 48 * 32).max(128);
            // torchvision `_compute_resized_output_size`: short->tar,
            // long = int(tar * long / short).
            let (short, long) = if w <= h { (w, h) } else { (h, w) };
            let new_long = tar * long / short;
            let (nw, nh) = if w <= h {
                (tar, new_long)
            } else {
                (new_long, tar)
            };
            let u8 = rgb8_to_planar_u8(img);
            let out = resize_bilinear_pil_u8(&u8, 3, h, w, nh, nw);
            (out, nh, nw)
        } else {
            (rgb8_to_planar_u8(img), h, w)
        };
        let f32 = planar
            .iter()
            .map(|&v| f32::from(v) / 255.0)
            .collect::<Vec<f32>>();
        (f32, rh, rw)
    }

    /// Trunk taps of a prepped image; checks the channel widths from the meta.
    fn taps(&self, planar: Vec<f32>, h: usize, w: usize) -> Result<Vec<Tap>, Error> {
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let taps = t.run(planar, h, w)?;
        if taps.len() != self.channels.len() {
            return Err(Error::BadModel(format!(
                "{} taps, expected {}",
                taps.len(),
                self.channels.len()
            )));
        }
        for (t, &c) in taps.iter().zip(&self.channels) {
            if t.c != c {
                return Err(Error::BadModel(format!(
                    "tap channel {}, expected {c}",
                    t.c
                )));
            }
            if t.h < PATCH || t.w < PATCH {
                return Err(Error::TooSmall { min: self.min_size });
            }
        }
        Ok(taps)
    }

    /// Unfold `PATCH`-sized patches at `STRIDE` per channel and SVD each —
    /// the upstream `(a, C, L, 16, 16)` layout flattened to `(c * pt)`.
    fn decompose(tap: &Tap) -> TapDecomp {
        let (c, h, w) = (tap.c, tap.h, tap.w);
        let ny = (h - PATCH) / STRIDE + 1;
        let nx = (w - PATCH) / STRIDE + 1;
        let pt = ny * nx;
        let mut patches = Vec::with_capacity(c * pt);
        let mut means = Vec::with_capacity(c);
        for ch in 0..c {
            let plane = &tap.data[ch * h * w..(ch + 1) * h * w];
            means.push(plane.iter().map(|&v| f64::from(v)).sum::<f64>() as f32
                / plane.len() as f32);
            for py in 0..ny {
                for px in 0..nx {
                    let mut m = [0f64; 256];
                    for y in 0..PATCH {
                        for x in 0..PATCH {
                            m[y * PATCH + x] =
                                f64::from(plane[(py * STRIDE + y) * w + px * STRIDE + x]);
                        }
                    }
                    let (u, s) = svd16(&m);
                    let mut uf = [0f32; 256];
                    let mut sf = [0f32; 16];
                    for i in 0..256 {
                        uf[i] = u[i] as f32;
                    }
                    for i in 0..16 {
                        sf[i] = s[i] as f32;
                    }
                    patches.push(PatchSvd { u: uf, s: sf });
                }
            }
        }
        TapDecomp {
            patches,
            means,
            pt,
        }
    }

    /// One tap's contribution: `mean_{c,pt}(diff_s·wt)·glb`.
    fn tap_score(r: &TapDecomp, d: &TapDecomp, c: usize) -> f64 {
        let n = r.pt * c;
        let mut acc = 0.0f64;
        for i in 0..n {
            let (pr, pd) = (&r.patches[i], &d.patches[i]);
            let mut diff_s = 0.0f64;
            for k in 0..16 {
                let ds = f64::from(pr.s[k]) - f64::from(pd.s[k]);
                diff_s += ds * ds;
            }
            let mut u_rd = [0f64; 16];
            for y in 0..16 {
                let mut a = 0.0f64;
                for x in 0..16 {
                    a += f64::from(pr.u[y * 16 + x]) * f64::from(pd.u[y * 16 + x]).abs();
                }
                u_rd[y] = a;
            }
            let mean = u_rd.iter().sum::<f64>() / 16.0;
            let var = u_rd.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 15.0;
            let wt = var.sqrt() / (mean + 1e-9);
            acc += diff_s * wt;
        }
        let avg = acc / n as f64;
        let mut s1 = 0.0f64;
        for ch in 0..c {
            let (xm, ym) = (f64::from(d.means[ch]), f64::from(r.means[ch]));
            s1 += (2.0 * xm * ym + 1e-6) / (xm * xm + ym * ym + 1e-6);
        }
        avg * (-2.0 * s1 / c as f64).exp()
    }

    fn score_taps(r: &[TapDecomp], d: &[TapDecomp]) -> f64 {
        r.iter()
            .zip(d)
            .map(|(rt, dt)| Self::tap_score(rt, dt, rt.means.len()))
            .sum()
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        if r.width.min(r.height) < self.min_size {
            return Err(Error::TooSmall {
                min: self.min_size,
            });
        }
        let (pr, rh, rw) = Self::prep(r);
        let (pd, _, _) = Self::prep(d);
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let (tr, td) = t.run_pair(pr, pd, rh, rw)?;
        let dr = tr.into_iter().map(|t| Self::decompose(&t)).collect::<Vec<_>>();
        let dd = td.into_iter().map(|t| Self::decompose(&t)).collect::<Vec<_>>();
        Ok(Self::score_taps(&dr, &dd))
    }
}

impl za::Metric for Dmm {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "DMM_PyTorch 0.1.2, prepare_image_adt preprocess; engine {}",
            self.engine_name()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Difference)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self
            .score_pair(&r, &d)
            .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        if view.width.min(view.height) < self.min_size {
            return Err(Error::TooSmall {
                min: self.min_size,
            })
            .metric_err_with(COLUMN, Error::kind);
        }
        let (pr, rh, rw) = Self::prep(&view);
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        let taps = self.taps(pr, rh, rw).metric_err_with(COLUMN, Error::kind)?;
        let _ = t;
        let reference = taps
            .iter()
            .map(Self::decompose)
            .collect::<Vec<TapDecomp>>();
        Ok(Box::new(WarmDmm {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference decomposed once: the patch SVDs and channel means.
struct WarmDmm<'m> {
    metric: &'m Dmm,
    reference: Vec<TapDecomp>,
    size: (usize, usize),
}

impl za::WarmReference for WarmDmm<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        if d.width.min(d.height) < self.metric.min_size {
            return Err(Error::TooSmall {
                min: self.metric.min_size,
            })
            .metric_err_with(COLUMN, Error::kind);
        }
        let (pd, dh, dw) = Dmm::prep(&d);
        let s = (|| -> Result<f64, Error> {
            let taps = self.metric.taps(pd, dh, dw)?;
            let dd = taps
                .into_iter()
                .map(|t| Dmm::decompose(&t))
                .collect::<Vec<_>>();
            Ok(Dmm::score_taps(&self.reference, &dd))
        })()
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

/// SVD of a 16×16 matrix via cyclic-Jacobi eigendecomposition of `MᵀM`:
/// `s_i = √λ_i` descending, `u_i = M·v_i / s_i`. For distinct singular
/// values the U columns are unique up to sign — and `|u_r ∘ u_d|` is
/// sign-free, so this matches `torch.svd`'s `u`, `s` for everything the head
/// reads. Near-zero `s_i` gets a deterministic Gram–Schmidt completion.
fn svd16(m: &[f64; 256]) -> ([f64; 256], [f64; 16]) {
    let mut g = [0f64; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut a = 0.0;
            for r in 0..16 {
                a += m[r * 16 + i] * m[r * 16 + j];
            }
            g[i * 16 + j] = a;
        }
    }
    let mut v = [0f64; 256];
    for i in 0..16 {
        v[i * 16 + i] = 1.0;
    }
    for _ in 0..60 {
        let mut off = 0.0f64;
        for p in 0..15 {
            for q in p + 1..16 {
                off += g[p * 16 + q] * g[p * 16 + q];
            }
        }
        if off <= 1e-28 {
            break;
        }
        for p in 0..15 {
            for q in p + 1..16 {
                let apq = g[p * 16 + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (g[q * 16 + q] - g[p * 16 + p]) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    1.0 / (theta - (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..16 {
                    let (gp, gq) = (g[i * 16 + p], g[i * 16 + q]);
                    g[i * 16 + p] = c * gp - s * gq;
                    g[i * 16 + q] = s * gp + c * gq;
                }
                for i in 0..16 {
                    let (gp, gq) = (g[p * 16 + i], g[q * 16 + i]);
                    g[p * 16 + i] = c * gp - s * gq;
                    g[q * 16 + i] = s * gp + c * gq;
                }
                for i in 0..16 {
                    let (vp, vq) = (v[i * 16 + p], v[i * 16 + q]);
                    v[i * 16 + p] = c * vp - s * vq;
                    v[i * 16 + q] = s * vp + c * vq;
                }
            }
        }
    }
    // Descending eigenvalues → singular values; u_i = M·v_i / s_i.
    let mut order: Vec<usize> = (0..16).collect();
    order.sort_by(|&a, &b| g[b * 16 + b].total_cmp(&g[a * 16 + a]));
    let max_s = g[order[0] * 16 + order[0]].max(0.0).sqrt();
    let mut s = [0f64; 16];
    let mut u = [0f64; 256];
    for (rank, &k) in order.iter().enumerate() {
        let sv = g[k * 16 + k].max(0.0).sqrt();
        s[rank] = sv;
        if sv > 1e-9 * max_s + 1e-300 {
            for i in 0..16 {
                let mut a = 0.0;
                for r in 0..16 {
                    a += m[i * 16 + r] * v[r * 16 + k];
                }
                u[i * 16 + rank] = a / sv;
            }
        } else {
            // Degenerate singular value: deterministic orthonormal
            // completion via Gram–Schmidt on the canonical basis.
            let mut e = [0f64; 16];
            e[rank] = 1.0;
            for prev in 0..rank {
                let mut dot = 0.0;
                for i in 0..16 {
                    dot += e[i] * u[i * 16 + prev];
                }
                for i in 0..16 {
                    e[i] -= dot * u[i * 16 + prev];
                }
            }
            let n = e.iter().map(|x| x * x).sum::<f64>().sqrt();
            let n = if n < 1e-12 {
                // exhausted the basis — pick the least-used canonical axis
                let mut pick = 0usize;
                let mut best = f64::MAX;
                for j in 0..16 {
                    let mut t = [0f64; 16];
                    t[j] = 1.0;
                    for prev in 0..rank {
                        let mut dot = 0.0;
                        for i in 0..16 {
                            dot += t[i] * u[i * 16 + prev];
                        }
                        for i in 0..16 {
                            t[i] -= dot * u[i * 16 + prev];
                        }
                    }
                    let nn = t.iter().map(|x| x * x).sum::<f64>();
                    if nn < best {
                        best = nn;
                        pick = j;
                    }
                }
                e = [0f64; 16];
                e[pick] = 1.0;
                for prev in 0..rank {
                    let mut dot = 0.0;
                    for i in 0..16 {
                        dot += e[i] * u[i * 16 + prev];
                    }
                    for i in 0..16 {
                        e[i] -= dot * u[i * 16 + prev];
                    }
                }
                e.iter().map(|x| x * x).sum::<f64>().sqrt()
            } else {
                n
            };
            for i in 0..16 {
                u[i * 16 + rank] = e[i] / n.max(1e-30);
            }
        }
    }
    (u, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// torch.svd references: np.linalg.svd on a fixed 16×16 matrix —
    /// singular values match to ~1e-10 regardless of sign convention.
    #[test]
    fn svd16_singular_values() {
        let mut m = [0f64; 256];
        // deterministic non-trivial matrix
        for i in 0..256 {
            m[i] = ((i * 37 + 11) % 97) as f64 / 97.0 - 0.5;
        }
        let (_, s) = svd16(&m);
        // np.linalg.svd of the same matrix
        let want = [
            4.894474977333818,
            4.311668685704951,
            4.047011757177965,
            3.901522766047755,
            3.723329429545972,
            3.5683409036209595,
            3.4457084360352185,
            3.3299838398704553,
            3.2152504370215305,
            3.0832494607267413,
            2.9558787838561444,
            2.7975025069653294,
            2.6405040318583014,
            2.4303839759790355,
            2.2041543922822446,
            1.8638384474239944,
        ];
        for (i, (&got, &w)) in s.iter().zip(&want).enumerate() {
            assert!((got - w).abs() < 1e-9, "s[{i}] = {got}, want {w}");
        }
    }

    #[test]
    fn svd16_identity_and_reconstruction() {
        let mut m = [0f64; 256];
        for i in 0..16 {
            m[i * 16 + i] = 2.0;
        }
        let (_, s) = svd16(&m);
        for &sv in &s {
            assert!((sv - 2.0).abs() < 1e-12);
        }
        // reconstruction: U·S·Vᵀ ≈ M on a generic matrix
        let mut m = [0f64; 256];
        for i in 0..256 {
            m[i] = ((i * 53 + 7) % 89) as f64 / 89.0 - 0.4;
        }
        let (u, s) = svd16(&m);
        // Vᵀ = Uᵀ·M / S  ⇒  check ‖u_i‖=1 and orthogonality
        for c in 0..16 {
            let n: f64 = (0..16).map(|r| u[r * 16 + c] * u[r * 16 + c]).sum();
            assert!((n - 1.0).abs() < 1e-9, "u_{c} norm {n}");
        }
        for c in 0..15 {
            for d in c + 1..16 {
                let dot: f64 = (0..16).map(|r| u[r * 16 + c] * u[r * 16 + d]).sum();
                assert!(dot.abs() < 1e-9, "u_{c}·u_{d} = {dot}");
            }
        }
    }
}
