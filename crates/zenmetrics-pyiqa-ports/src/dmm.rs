//! DMM as pyiqa 0.1.16 scores it (`pyiqa.create_metric('dmm')`,
//! `pyiqa/archs/dmm_arch.py`; the AIC2026 `DMM` column is close to it but
//! not exact).
//!
//! The one convention that differs from the public zenmetrics `dmm` port
//! (upstream DMM_PyTorch's packaged preprocessing) is `prepare_image_adt`:
//! when `min(h, w) > 128`, pyiqa resizes the `[0,1]` tensor to a **square**
//! `tar × tar`, `tar = max(⌊min/48⌋·32, 128)`, with `F.interpolate(…,
//! mode='bilinear', align_corners=False)` (no antialias), where the public
//! port keeps the aspect ratio with a Pillow short-side resize.
//!
//! The rest follows pyiqa: VGG16 (IMAGENET1K_V1, max-pools replaced by
//! 3×3 Hann L2 pooling, no input normalisation) taps relu3_3 and relu4_3
//! (`dmm_pyiqa_trunk.onnx`); per tap, 16×16 patches at stride 4 per
//! channel, SVD of each (pyiqa's `transpose(2, 4)` patch); per patch
//! `Σ(s_r − s_d)² · wt` with `wt = std(u_rd)/(mean(u_rd) + 1e-9)`,
//! `u_rd = Σ_j |u_r ∘ u_d|`; mean over patches and channels, times
//! `exp(−2·mean_c S1)` with the channel-mean SSIM term `S1`; summed over the
//! two taps. Lower is better; identical images score 0.
//!
//! The head (patch term, gate, 16×16 SVD) is the public port's
//! (`dmm_svd.rs`, and the functions marked below), whose solver follows
//! Reference-LAPACK; pyiqa runs `torch.svd` (MKL on CPU). `wt` depends on
//! the singular basis inside degenerate clusters, which solvers choose
//! differently, so scores are solver-dependent (see `models/dmm-pyiqa/card.toml`).
//!
//! The warm reference caches every reference patch SVD (hundreds of MB at
//! large sizes); the cold score streams them instead. Both sum the per-patch
//! terms per channel, then over channels in order, whatever the thread
//! count, so they are bit-identical.

use std::path::Path;

use zenmetrics_abstractions as za;
use zenmetrics_nn::Error;

use crate::Engine;
use crate::dmm_svd as svd;
use crate::engine::{self, Graph, Tensor, read_meta};
use crate::port::{Port, check_buffer, impl_metric};

const COLUMN: &str = "dmm_pyiqa";
const PATCH: usize = 16;
const STRIDE: usize = 4;
/// Largest side accepted before the resize.
const MAX_SIDE: usize = 16_384;

/// DMM, pyiqa convention (see the module docs).
pub struct DmmPyiqa {
    trunk: Box<dyn Graph>,
    channels: Vec<usize>,
    min_size: usize,
}

/// One `[c, h, w]` tap.
struct Tap {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

impl Tap {
    fn plane(&self, ch: usize) -> &[f32] {
        &self.data[ch * self.h * self.w..][..self.h * self.w]
    }
}

/// One patch's SVD: `u` row-major 16×16, the *right* singular vectors of
/// the spatial patch (pyiqa's `u` of the transposed patch), `s` descending.
#[derive(Clone)]
struct PatchSvd {
    u: [f32; 256],
    s: [f32; 16],
}

/// A reference image's per-tap patch SVDs (channel-major) and channel means.
pub(crate) struct WarmRef {
    w: usize,
    h: usize,
    taps: Vec<(Vec<PatchSvd>, Vec<f32>)>,
}

/// One axis of torch's `compute_indices_weights_linear` (align_corners
/// False, output size given, so scale = `in / out` in f32).
fn axis(input: usize, output: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|i| {
            let src = (scale * (i as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (src.floor() as usize).min(input - 1);
            let l1 = (src - i0 as f32).clamp(0.0, 1.0);
            let i1 = if i0 < input - 1 { i0 + 1 } else { i0 };
            (i0, i1, 1.0 - l1, l1)
        })
        .collect()
}

/// `prepare_image_adt` on packed RGB8: planar f32 `u8 / 255`, then the
/// square bilinear resize when `min(h, w) > 128`. Returns `(planes, side_h,
/// side_w)`.
fn prepare(rgb: &[u8], w: usize, h: usize, stride: usize) -> (Vec<f32>, usize, usize) {
    let plane = w * h;
    let x = crate::engine::rgb8_to_planar_01(&za::Rgb8In {
        pixels: rgb,
        width: w,
        height: h,
        stride,
    });
    let m = w.min(h);
    if m <= 128 {
        return (x, h, w);
    }
    let tar = (m / 48 * 32).max(128);
    let (ay, ax) = (axis(h, tar), axis(w, tar));
    let mut out = Vec::with_capacity(3 * tar * tar);
    for c in 0..3 {
        let p = &x[c * plane..][..plane];
        for &(y0, y1, wy0, wy1) in &ay {
            let (r0, r1) = (&p[y0 * w..][..w], &p[y1 * w..][..w]);
            for &(x0, x1, wx0, wx1) in &ax {
                let t0 = r0[x0] * wx0 + r0[x1] * wx1;
                let t1 = r1[x0] * wx0 + r1[x1] * wx1;
                out.push(t0 * wy0 + t1 * wy1);
            }
        }
    }
    (out, tar, tar)
}

/// The unfold patch grid `(ny, nx)` of an `h × w` tap.
/// (zenmetrics-nn `dmm.rs` `patch_grid`, e9cd38d2.)
fn patch_grid(h: usize, w: usize) -> (usize, usize) {
    ((h - PATCH) / STRIDE + 1, (w - PATCH) / STRIDE + 1)
}

/// `t.mean([2,3])` per channel, summed in f64, divided in f32.
/// (zenmetrics-nn `dmm.rs` `channel_means`, e9cd38d2.)
fn channel_means(t: &Tap) -> Vec<f32> {
    let n = t.h * t.w;
    (0..t.c)
        .map(|ch| t.plane(ch).iter().map(|&v| f64::from(v)).sum::<f64>() as f32 / n as f32)
        .collect()
}

/// (zenmetrics-nn `dmm.rs` `svd_patch`, e9cd38d2.)
fn svd_patch(plane: &[f32], w: usize, y0: usize, x0: usize) -> PatchSvd {
    let mut m = [0f32; 256];
    for y in 0..PATCH {
        for x in 0..PATCH {
            m[y * PATCH + x] = plane[(y0 + y) * w + x0 + x];
        }
    }
    let (u, s) = svd::svd16(&m);
    PatchSvd { u, s }
}

/// One patch pair's `Σ_k(s_r − s_d)² · wt`; `u_rd` takes `abs` of the
/// product, `std` is unbiased (`/15`).
/// (zenmetrics-nn `dmm.rs` `patch_term`, e9cd38d2.)
fn patch_term(pr: &PatchSvd, pd: &PatchSvd) -> f64 {
    let mut diff_s = 0.0f64;
    for k in 0..16 {
        let ds = f64::from(pr.s[k]) - f64::from(pd.s[k]);
        diff_s += ds * ds;
    }
    let mut u_rd = [0f64; 16];
    for (y, out) in u_rd.iter_mut().enumerate() {
        let mut a = 0.0f64;
        for x in 0..16 {
            a += (f64::from(pr.u[y * 16 + x]) * f64::from(pd.u[y * 16 + x])).abs();
        }
        *out = a;
    }
    let mean = u_rd.iter().sum::<f64>() / 16.0;
    let var = u_rd.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 15.0;
    diff_s * (var.sqrt() / (mean + 1e-9))
}

/// `mean_c((2·x̄_d·ȳ_r + 1e-6)/(x̄_d² + ȳ_r² + 1e-6))`.
/// (zenmetrics-nn `dmm.rs` `s1_gate`, e9cd38d2.)
fn s1_gate(means_r: &[f32], means_d: &[f32]) -> f64 {
    let mut s1 = 0.0f64;
    for ch in 0..means_r.len() {
        let (xm, ym) = (f64::from(means_d[ch]), f64::from(means_r[ch]));
        s1 += (2.0 * xm * ym + 1e-6) / (xm * xm + ym * ym + 1e-6);
    }
    s1 / means_r.len() as f64
}

/// Run `f(ch)` for every channel on `engine::workers()` threads and
/// return the results in channel order.
fn per_channel<T: Send>(c: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let n = engine::workers().min(c).max(1);
    let mut out: Vec<Option<T>> = (0..c).map(|_| None).collect();
    std::thread::scope(|s| {
        let f = &f;
        let chunks: Vec<_> = out.chunks_mut(c.div_ceil(n)).enumerate().collect();
        for (k, chunk) in chunks {
            s.spawn(move || {
                let base = k * c.div_ceil(n);
                for (i, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(f(base + i));
                }
            });
        }
    });
    out.into_iter()
        .map(|v| v.expect("every channel ran"))
        .collect()
}

/// Per-channel sums of the patch terms of one tap, then the tap's score
/// `mean(terms) · exp(−2·S1)`. `reference(ch, k)` yields the reference
/// patch SVD; both paths use this function, so they sum identically.
fn tap_score<R: Fn(usize, usize) -> PatchSvd + Sync>(
    d: &Tap,
    means_r: &[f32],
    reference: R,
) -> f64 {
    let (ny, nx) = patch_grid(d.h, d.w);
    let sums = per_channel(d.c, |ch| {
        let plane = d.plane(ch);
        let mut acc = 0.0f64;
        for py in 0..ny {
            for px in 0..nx {
                let pd = svd_patch(plane, d.w, py * STRIDE, px * STRIDE);
                acc += patch_term(&reference(ch, py * nx + px), &pd);
            }
        }
        acc
    });
    let total: f64 = sums.iter().sum();
    let s1 = s1_gate(means_r, &channel_means(d));
    total / (ny * nx * d.c) as f64 * (-2.0 * s1).exp()
}

impl DmmPyiqa {
    /// The trunk's weights: torchvision VGG16 IMAGENET1K_V1 (BSD-3-Clause
    /// torchvision); DMM itself has no fitted parameters.
    pub const WEIGHTS: zenmetrics_nn::WeightsTerms = zenmetrics_nn::WeightsTerms {
        model: COLUMN,
        licence: zenmetrics_nn::WeightsLicence::Permissive,
        terms_url: "https://github.com/pytorch/vision/blob/main/LICENSE",
    };

    /// Load `dmm_pyiqa_trunk.onnx` and `dmm_pyiqa.meta` from `dir` (written
    /// by `models/dmm-pyiqa/export.py`) on `engine`.
    ///
    /// # Errors
    /// I/O, malformed exports or engine load failures.
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join(format!("{COLUMN}.meta")))?;
        if channels != [256, 512] {
            return Err(Error::BadModel(format!(
                "{COLUMN}.meta: channels {channels:?}, expected [256, 512]"
            )));
        }
        Ok(Self {
            trunk: engine::load(&dir.join(format!("{COLUMN}_trunk.onnx")), engine)?,
            channels,
            min_size,
        })
    }

    fn taps(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Tap>, Error> {
        check_buffer(img)?;
        let (w, h) = (img.width, img.height);
        if w < self.min_size || h < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        if w > MAX_SIDE || h > MAX_SIDE {
            return Err(Error::TooLarge {
                limit: MAX_SIDE * MAX_SIDE,
            });
        }
        let (x, ph, pw) = prepare(img.pixels, w, h, img.stride);
        let out = self.trunk.run(vec![Tensor::new(vec![1, 3, ph, pw], x)])?;
        if out.len() != 2 {
            return Err(Error::BadModel("trunk must return t3 and t4".into()));
        }
        out.into_iter()
            .zip(&self.channels)
            .map(|(t, &c)| match *t.shape.as_slice() {
                [1, tc, th, tw] if tc == c => {
                    if th < PATCH || tw < PATCH {
                        return Err(Error::TooSmall { min: self.min_size });
                    }
                    Ok(Tap {
                        c,
                        h: th,
                        w: tw,
                        data: t.data,
                    })
                }
                _ => Err(Error::BadModel(format!("tap shape {:?}", t.shape))),
            })
            .collect()
    }
}

impl Port for DmmPyiqa {
    type Warm = WarmRef;
    const COLUMN: &'static str = COLUMN;
    const KIND: za::ScoreKind = za::ScoreKind::Difference;
    const REVISION: u32 = 1;

    fn reference(&self) -> String {
        format!(
            "pyiqa 0.1.16 dmm (square F.interpolate resize; LAPACK-based f32 SVD head); engine {}",
            self.trunk.engine()
        )
    }

    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<WarmRef, Error> {
        let taps = self
            .taps(r)?
            .into_iter()
            .map(|t| {
                let (ny, nx) = patch_grid(t.h, t.w);
                let per = per_channel(t.c, |ch| {
                    let plane = t.plane(ch);
                    (0..ny * nx)
                        .map(|k| svd_patch(plane, t.w, k / nx * STRIDE, k % nx * STRIDE))
                        .collect::<Vec<_>>()
                });
                (per.into_iter().flatten().collect(), channel_means(&t))
            })
            .collect();
        Ok(WarmRef {
            w: r.width,
            h: r.height,
            taps,
        })
    }

    fn warm_size(w: &WarmRef) -> (usize, usize) {
        (w.w, w.h)
    }

    /// Streams the reference's patch SVDs instead of caching them (the
    /// cache is ~1.5 GB at 2048×1536 and grows past 10 GB at 4096²).
    fn score_cold(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let rt = self.taps(r)?;
        let dt = self.taps(d)?;
        Ok(dt
            .iter()
            .zip(&rt)
            .map(|(dt, rt)| {
                let (_, nx) = patch_grid(rt.h, rt.w);
                tap_score(dt, &channel_means(rt), |ch, k| {
                    svd_patch(rt.plane(ch), rt.w, k / nx * STRIDE, k % nx * STRIDE)
                })
            })
            .sum())
    }

    fn score_warm(&self, r: &WarmRef, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let d = self.taps(d)?;
        Ok(d.iter()
            .zip(&r.taps)
            .map(|(dt, (patches, means))| {
                let (ny, nx) = patch_grid(dt.h, dt.w);
                tap_score(dt, means, |ch, k| patches[ch * ny * nx + k].clone())
            })
            .sum())
    }
}

impl_metric!(DmmPyiqa);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_resize_geometry() {
        let rgb = vec![7u8; 944 * 840 * 3];
        let (x, h, w) = prepare(&rgb, 944, 840, 944 * 3);
        assert_eq!((h, w), (544, 544));
        assert_eq!(x.len(), 3 * 544 * 544);
        let (_, h, w) = prepare(&vec![0u8; 100 * 120 * 3], 100, 120, 300);
        assert_eq!((h, w), (120, 100));
    }
}
