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
use za::ResultMetricExt as _;

const COLUMN: &str = "dmm";
const SCORE_REVISION: u32 = 2;
/// Patch kernel and stride of upstream's `nn.Unfold(16, stride=4)`.
const PATCH: usize = 16;
const STRIDE: usize = 4;
// Bound ingress before preprocessing or inference. The 4096² ladder remains
// supported; aspect ratios above 32:1 are refused before allocating planes.
const MAX_SIDE: usize = 4096;
const MAX_PIXELS: usize = if usize::BITS >= 64 {
    16_777_216
} else {
    1_048_576
};
const MAX_ASPECT: usize = 32;

fn validate_geometry(w: usize, h: usize) -> Result<(), Error> {
    if w.max(h) > MAX_SIDE {
        return Err(Error::TooLarge { limit: MAX_SIDE });
    }
    if w.checked_mul(h).is_none_or(|n| n > MAX_PIXELS) {
        return Err(Error::TooLarge { limit: MAX_PIXELS });
    }
    if w.min(h) == 0 || w.max(h) > w.min(h).saturating_mul(MAX_ASPECT) {
        return Err(Error::TooLarge { limit: MAX_ASPECT });
    }
    if w.min(h) < 128 {
        return Err(Error::TooSmall { min: 128 });
    }
    Ok(())
}

/// DMM (see module docs).
pub struct Dmm {
    backend: Backend,
    channels: Vec<usize>,
    min_size: usize,
}

/// One patch's SVD: `u` row-major 16×16 holding the patch's *right*
/// singular vectors (upstream's `u` of the transposed patch — see
/// `svd16`), `s` the 16 singular values, descending. Stored f32 —
/// upstream computes the whole head in f32.
#[derive(Clone)]
struct PatchSvd {
    u: [f32; 256],
    s: [f32; 16],
}

/// One tap's per-patch decompositions plus its channel means (`y_mean`).
/// Only the warm path materialises this — on a 2720² prepared image the two
/// taps hold ~11.5 GB of `PatchSvd`, so the cold score streams the head per
/// channel instead of keeping both sides resident (the sum over (c, pt) is
/// order-free).
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
            Engine::Rten => {
                Backend::Trunk(engine::load(&dir.join("dmm_trunk.onnx"), channels.len())?)
            }
            #[cfg(feature = "ort")]
            Engine::OrtCuda => {
                Backend::Trunk(Box::new(engine::ort_engine::OrtCudaTrunk::load_named(
                    &dir.join("dmm_trunk.onnx"),
                    "x",
                    vec!["t3", "t4"],
                )?))
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

    /// Trunk taps of `img` after `prep` as `(c, h, w, planar)` per tap.
    ///
    /// Tooling surface for `examples/dmm_tap_dump.rs`, which feeds
    /// `models/dmm/gen_netlib_goldens.py`: the netlib-`sgesdd` goldens
    /// that `nn-parity-dmm` gates on must be scored on the *same* taps
    /// this engine produces — the head amplifies tap noise ~100x into
    /// score units, so torch-tap goldens cannot be matched at gate.
    #[cfg(feature = "model-tests")]
    #[doc(hidden)]
    pub fn taps_for_export(
        &self,
        img: &za::Rgb8In<'_>,
    ) -> Result<Vec<(usize, usize, usize, Vec<f32>)>, Error> {
        let (planar, h, w) = Self::prep(img)?;
        Ok(self
            .taps(planar, h, w)?
            .into_iter()
            .map(|t| (t.c, t.h, t.w, t.data))
            .collect())
    }

    /// `prepare_image_adt`: conditional short-side PIL-bilinear resize, then
    /// `[0,1]`. Returns `(planar, h, w)` of the *resized* image.
    fn prep(img: &za::Rgb8In<'_>) -> Result<(Vec<f32>, usize, usize), Error> {
        validate_geometry(img.width, img.height)?;
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
        Ok((f32, rh, rw))
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
            if t.c != c || t.data.len() != t.c * t.plane() {
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
    /// Upstream's `transpose(2,4)` makes its patch `Mᵀ` of ours, so its
    /// `u` is our `v`: `PatchSvd.u` holds the *right* singular vectors
    /// (see `svd16`).
    fn decompose(tap: &Tap, opts: za::EvalOpts<'_>) -> za::MetricResult<TapDecomp> {
        let (c, h, w) = (tap.c, tap.h, tap.w);
        let (ny, nx) = patch_grid(h, w);
        let pt = ny * nx;
        let mut patches = Vec::new();
        let count = c
            .checked_mul(pt)
            .ok_or(Error::TooLarge { limit: MAX_PIXELS })
            .metric_err_with(COLUMN, Error::kind)?;
        patches
            .try_reserve_exact(count)
            .map_err(|e| Error::Engine(format!("DMM reference allocation: {e}")))
            .metric_err_with(COLUMN, Error::kind)?;
        for ch in 0..c {
            opts.checkpoint()?;
            let plane = &tap.data[ch * tap.plane()..(ch + 1) * tap.plane()];
            for py in 0..ny {
                for px in 0..nx {
                    patches.push(svd_patch(plane, w, py * STRIDE, px * STRIDE));
                }
            }
        }
        let mut means = Vec::new();
        means
            .try_reserve_exact(c)
            .map_err(|e| Error::Engine(format!("DMM means allocation: {e}")))
            .metric_err_with(COLUMN, Error::kind)?;
        means.extend((0..c).map(|ch| {
            let plane = &tap.data[ch * tap.plane()..(ch + 1) * tap.plane()];
            plane.iter().map(|&v| f64::from(v)).sum::<f64>() as f32 / plane.len() as f32
        }));
        Ok(TapDecomp { patches, means, pt })
    }

    /// One tap's contribution with both sides streamed: `mean_{c,pt}(diff_s·wt)·glb`.
    fn tap_score_cold(r: &Tap, d: &Tap, opts: za::EvalOpts<'_>) -> za::MetricResult<f64> {
        let (ny, nx) = patch_grid(r.h, r.w);
        let n = ny * nx * r.c;
        let mut acc = 0.0f64;
        for ch in 0..r.c {
            opts.checkpoint()?;
            let (pr, pd) = channel_planes(r, d, ch);
            for py in 0..ny {
                for px in 0..nx {
                    let pr = svd_patch(pr, r.w, py * STRIDE, px * STRIDE);
                    let pd = svd_patch(pd, d.w, py * STRIDE, px * STRIDE);
                    acc += patch_term(&pr, &pd);
                }
            }
        }
        let s1 = s1_gate(&channel_means(r), &channel_means(d));
        Ok(acc / n as f64 * (-2.0 * s1).exp())
    }

    /// One tap's contribution against a cached reference decomp.
    fn tap_score_warm(r: &TapDecomp, d: &Tap, opts: za::EvalOpts<'_>) -> za::MetricResult<f64> {
        let (ny, nx) = patch_grid(d.h, d.w);
        debug_assert_eq!(ny * nx, r.pt);
        let c = r.means.len();
        let n = r.pt * c;
        let mut acc = 0.0f64;
        for ch in 0..c {
            opts.checkpoint()?;
            let pd = &d.data[ch * d.plane()..(ch + 1) * d.plane()];
            for py in 0..ny {
                for px in 0..nx {
                    let pr = &r.patches[ch * r.pt + py * nx + px];
                    let pd = svd_patch(pd, d.w, py * STRIDE, px * STRIDE);
                    acc += patch_term(pr, &pd);
                }
            }
        }
        let s1 = s1_gate(&r.means, &channel_means(d));
        Ok(acc / n as f64 * (-2.0 * s1).exp())
    }

    fn score_taps(r: &[TapDecomp], d: &[Tap], opts: za::EvalOpts<'_>) -> za::MetricResult<f64> {
        r.iter()
            .zip(d)
            .map(|(rt, dt)| Self::tap_score_warm(rt, dt, opts))
            .sum()
    }

    fn score_pair(
        &self,
        r: &za::Rgb8In<'_>,
        d: &za::Rgb8In<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<f64> {
        let (tr, td) = (|| -> Result<_, Error> {
            if (r.width, r.height) != (d.width, d.height) {
                return Err(Error::SizeMismatch);
            }
            validate_geometry(r.width, r.height)?;
            if r.width.min(r.height) < self.min_size {
                return Err(Error::TooSmall { min: self.min_size });
            }
            let (pr, rh, rw) = Self::prep(r)?;
            let (pd, _, _) = Self::prep(d)?;
            let Backend::Trunk(t) = &self.backend else {
                return Err(Error::Engine("no trunk graph on this engine".into()));
            };
            let (tr, td) = t.run_pair(pr, pd, rh, rw)?;
            if tr.len() != self.channels.len() {
                return Err(Error::BadModel(format!(
                    "{} taps, expected {}",
                    tr.len(),
                    self.channels.len()
                )));
            }
            for (t, &c) in tr.iter().zip(&self.channels) {
                if t.c != c || t.data.len() != t.c * t.plane() {
                    return Err(Error::BadModel(format!(
                        "tap channel {}, expected {c}",
                        t.c
                    )));
                }
                if t.h < PATCH || t.w < PATCH {
                    return Err(Error::TooSmall { min: self.min_size });
                }
            }
            Ok((tr, td))
        })()
        .metric_err_with(COLUMN, Error::kind)?;
        tr.iter()
            .zip(&td)
            .map(|(rt, dt)| Self::tap_score_cold(rt, dt, opts))
            .sum()
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
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self.score_pair(&r, &d, opts)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        validate_geometry(view.width, view.height).metric_err_with(COLUMN, Error::kind)?;
        if view.width.min(view.height) < self.min_size {
            return Err(Error::TooSmall { min: self.min_size })
                .metric_err_with(COLUMN, Error::kind);
        }
        let (pr, rh, rw) = Self::prep(&view).metric_err_with(COLUMN, Error::kind)?;
        let taps = self.taps(pr, rh, rw).metric_err_with(COLUMN, Error::kind)?;
        let reference = taps
            .iter()
            .map(|t| Self::decompose(t, opts))
            .collect::<za::MetricResult<Vec<TapDecomp>>>()?;
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
        let (pd, dh, dw) = Dmm::prep(&d).metric_err_with(COLUMN, Error::kind)?;
        let dd = self
            .metric
            .taps(pd, dh, dw)
            .metric_err_with(COLUMN, Error::kind)?;
        let s = Dmm::score_taps(&self.reference, &dd, opts)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

/// The unfold patch grid `(ny, nx)` for an `h×w` tap.
fn patch_grid(h: usize, w: usize) -> (usize, usize) {
    ((h - PATCH) / STRIDE + 1, (w - PATCH) / STRIDE + 1)
}

/// The `ch`-th channel planes of taps `r`/`d`.
fn channel_planes<'a>(r: &'a Tap, d: &'a Tap, ch: usize) -> (&'a [f32], &'a [f32]) {
    (
        &r.data[ch * r.plane()..(ch + 1) * r.plane()],
        &d.data[ch * d.plane()..(ch + 1) * d.plane()],
    )
}

/// Upstream `t.mean([2,3])` — one f32 channel mean per channel.
fn channel_means(tap: &Tap) -> Vec<f32> {
    let n = tap.plane();
    (0..tap.c)
        .map(|ch| {
            tap.data[ch * n..(ch + 1) * n]
                .iter()
                .map(|&v| f64::from(v))
                .sum::<f64>() as f32
                / n as f32
        })
        .collect()
}

/// One `PATCH`×`PATCH` patch at `(y0, x0)` of a `w`-strided f32 plane,
/// SVD'd by [`svd16`].
fn svd_patch(plane: &[f32], w: usize, y0: usize, x0: usize) -> PatchSvd {
    let mut m = [0f32; 256];
    for y in 0..PATCH {
        for x in 0..PATCH {
            m[y * PATCH + x] = plane[(y0 + y) * w + x0 + x];
        }
    }
    let (u, s) = svd16(&m);
    PatchSvd { u, s }
}

/// One patch pair's `Σ_k(s_r−s_d)² · wt` term.
fn patch_term(pr: &PatchSvd, pd: &PatchSvd) -> f64 {
    let mut diff_s = 0.0f64;
    for k in 0..16 {
        let ds = f64::from(pr.s[k]) - f64::from(pd.s[k]);
        diff_s += ds * ds;
    }
    // upstream: |u_r ∘ u_d|.sum(-1) on ITS `u` (= our `v` — see `svd16`)
    // — abs on the PRODUCT, so u_rd >= 0; `u_r·|u_d|` instead can cancel
    // to ~0 and wt's mean+1e-9 guard blows up by ~1e9.
    let mut u_rd = [0f64; 16];
    for y in 0..16 {
        let mut a = 0.0f64;
        for x in 0..16 {
            a += (f64::from(pr.u[y * 16 + x]) * f64::from(pd.u[y * 16 + x])).abs();
        }
        u_rd[y] = a;
    }
    let mean = u_rd.iter().sum::<f64>() / 16.0;
    let var = u_rd.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 15.0;
    diff_s * (var.sqrt() / (mean + 1e-9))
}

/// Upstream `mean_c(S1)` where `S1 = (2·x̄_d·ȳ_r + 1e-6)/(x̄_d²+ȳ_r² + 1e-6)`
/// per channel — the SSIM-style gate's inner mean (before `exp(−2·)`).
fn s1_gate(means_r: &[f32], means_d: &[f32]) -> f64 {
    let mut s1 = 0.0f64;
    for ch in 0..means_r.len() {
        let (xm, ym) = (f64::from(means_d[ch]), f64::from(means_r[ch]));
        s1 += (2.0 * xm * ym + 1e-6) / (xm * xm + ym * ym + 1e-6);
    }
    s1 / means_r.len() as f64
}

// ---- LAPACK-based 16x16 SVD ------------------------------------------
// Adapted from Reference-LAPACK (BSD-3-Clause). Full copyright notices
// and licence terms are in LICENSE-lapack at the crate root.
//
// Upstream runs `torch.svd` on `unfold().view(a,b,k,k,-1).transpose(2,4)` —
// the `.transpose(2,4)` swaps the two k axes, so its patch matrix is `Mᵀ`
// and its `u` is *this* patch's V factor. The head therefore contracts
// `|v_r ∘ v_d|`, not `|u_r ∘ u_d|`.
//
// The score's `wt` term consumes the basis inside degenerate singular
// clusters. Solver implementations may choose different bases. This head
// follows the reference-LAPACK small-N SGESDD path in f32, but agreement
// is established per golden row, not inferred from the algorithm's name.
// See models/dmm/card.toml for the measured JND/pinned-row counts.
// Only the left (U) factor of the transposed patch is accumulated.
//
// `svd16(m)` therefore transposes `m` first (matching upstream's
// `transpose(2,4)`) and returns `u` of the transposed patch.

// SLAMCH('E') assumes rounding: EPSILON(0.0) / 2 = 2^-24.
// Reference-LAPACK v3.12.0 INSTALL/slamch.f, lines 97-104.
const D_EPS: f32 = f32::EPSILON * 0.5;
const D_SAFMIN: f32 = f32::MIN_POSITIVE;

fn dlapy2(x: f32, y: f32) -> f32 {
    let (x, y) = (x.abs(), y.abs());
    let (w, z) = if x < y { (y, x) } else { (x, y) };
    if w == 0.0 {
        0.0
    } else {
        w * (1.0 + (z / w) * (z / w)).sqrt()
    }
}

// Retain the classic scaled-f32 BLAS norm convention used by the
// qualified baseline. The f64 and modern three-accumulator experiments
// reassign degenerate bases and fail existing JND rows; see the receipt.
fn dnrm2(x: &[f32]) -> f32 {
    let (mut scale, mut ssq) = (0.0f32, 1.0f32);
    for &xi in x {
        if xi != 0.0 {
            let a = xi.abs();
            if scale < a {
                ssq = 1.0 + ssq * (scale / a) * (scale / a);
                scale = a;
            } else {
                ssq += (a / scale) * (a / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// DLARFG: householder reflector on (alpha; x). Returns (beta, tau); the
/// normalized tail is written back into `x`.
fn dlarfg(alpha: f32, x: &mut [f32]) -> (f32, f32) {
    if x.is_empty() {
        return (alpha, 0.0);
    }
    let mut alpha = alpha;
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let mut beta = -dlapy2(alpha, xnorm).copysign(alpha);
    let safmin = D_SAFMIN / D_EPS;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        while beta.abs() < safmin && knt < 20 {
            knt += 1;
            for xi in x.iter_mut() {
                *xi *= rsafmn;
            }
            beta *= rsafmn;
            alpha *= rsafmn;
        }
        let xnorm = dnrm2(x);
        beta = -dlapy2(alpha, xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let inv = 1.0 / (alpha - beta);
    for xi in x.iter_mut() {
        *xi *= inv;
    }
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// SLARTG, reference-LAPACK 3.12.0: nonnegative cosine and r signed
/// like f. SLARTGP instead forces r positive and is not SBDSQR's routine.
fn dlartg(f: f32, g: f32) -> (f32, f32, f32) {
    let f1 = f.abs();
    let g1 = g.abs();
    if g == 0.0 {
        return (1.0, 0.0, f);
    }
    if f == 0.0 {
        return (0.0, 1.0f32.copysign(g), g1);
    }
    let safmax = 1.0 / D_SAFMIN;
    let rtmin = D_SAFMIN.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let r = d.copysign(f);
        (f1 / d, g / r, r)
    } else {
        let u = safmax.min(D_SAFMIN.max(f1).max(g1));
        let fs = f / u;
        let gs = g / u;
        let d = (fs * fs + gs * gs).sqrt();
        let r = d.copysign(f);
        (fs.abs() / d, gs / r, r * u)
    }
}

/// DLAS2: singular values of the 2x2 lower-triangular [f 0; g h].
/// Returns (ssmin, ssmax).
fn dlas2(f: f32, g: f32, h: f32) -> (f32, f32) {
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let fhmn = fa.min(ha);
    let fhmx = fa.max(ha);
    let (ssmin, ssmax);
    if fhmn == 0.0 {
        ssmin = 0.0;
        ssmax = if fhmx == 0.0 {
            ga
        } else {
            fhmx.max(ga) * (1.0 + (fhmx.min(ga) / fhmx.max(ga)).powi(2)).sqrt()
        };
    } else if ga < fhmx {
        let as_ = 1.0 + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let c = 2.0 / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        ssmin = fhmn * c;
        ssmax = fhmx / c;
    } else {
        let au = fhmx / ga;
        if au == 0.0 {
            ssmin = (fhmn * fhmx) / ga;
            ssmax = ga;
        } else {
            let as_ = 1.0 + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = 1.0 / ((1.0 + (as_ * au).powi(2)).sqrt() + (1.0 + (at * au).powi(2)).sqrt());
            let mut s = (fhmn * c) * au;
            s += s;
            ssmin = s;
            ssmax = ga / (c + c);
        }
    }
    (ssmin, ssmax)
}

/// DLASV2: SVD of the 2x2 [f g; 0 h]. Returns (ssmin, ssmax, snr, csr, snl, csl).
fn dlasv2(f: f32, g: f32, h: f32) -> (f32, f32, f32, f32, f32, f32) {
    let (mut ft, mut fa) = (f, f.abs());
    let (mut ht, mut ha) = (h, h.abs());
    let mut pmax = 1;
    let swap = ha > fa;
    if swap {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
    }
    let gt = g;
    let ga = g.abs();
    let (clt, slt, crt, srt, ssmin, ssmax);
    if ga == 0.0 {
        ssmin = ha;
        ssmax = fa;
        clt = 1.0;
        slt = 0.0;
        crt = 1.0;
        srt = 0.0;
    } else {
        let mut gasmal = true;
        let mut clt_ = 1.0;
        let mut slt_ = 0.0;
        let mut crt_ = 1.0;
        let mut srt_ = 0.0;
        let mut ssmin_ = 0.0;
        let mut ssmax_ = 0.0;
        if ga > fa {
            pmax = 2;
            if fa / ga < D_EPS {
                gasmal = false;
                ssmax_ = ga;
                ssmin_ = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                clt_ = 1.0;
                slt_ = ht / gt;
                srt_ = 1.0;
                crt_ = ft / gt;
            }
        }
        if gasmal {
            let d = fa - ha;
            let l = if d == fa { 1.0 } else { d / fa };
            let m = gt / ft;
            let t = 2.0 - l;
            let mm = m * m;
            let tt = t * t;
            let s = (tt + mm).sqrt();
            let r = if l == 0.0 {
                m.abs()
            } else {
                (l * l + mm).sqrt()
            };
            let a = 0.5 * (s + r);
            ssmin_ = ha / a;
            ssmax_ = fa * a;
            let t = if mm == 0.0 {
                if l == 0.0 {
                    2.0f32.copysign(ft) * 1.0f32.copysign(gt)
                } else {
                    gt / d.copysign(ft) + m / t
                }
            } else {
                (m / (s + t) + m / (r + l)) * (1.0 + a)
            };
            let l = (t * t + 4.0).sqrt();
            crt_ = 2.0 / l;
            srt_ = t / l;
            clt_ = (crt_ + srt_ * m) / a;
            slt_ = (ht / ft) * srt_ / a;
        }
        clt = clt_;
        slt = slt_;
        crt = crt_;
        srt = srt_;
        ssmin = ssmin_;
        ssmax = ssmax_;
    }
    let (csl, snl, csr, snr) = if swap {
        (srt, crt, slt, clt)
    } else {
        (clt, slt, crt, srt)
    };
    let tsign = match pmax {
        1 => 1.0f32.copysign(csr) * 1.0f32.copysign(csl) * 1.0f32.copysign(f),
        2 => 1.0f32.copysign(snr) * 1.0f32.copysign(csl) * 1.0f32.copysign(g),
        _ => 1.0f32.copysign(snr) * 1.0f32.copysign(snl) * 1.0f32.copysign(h),
    };
    let ssmax = ssmax.copysign(tsign);
    let ssmin = ssmin.copysign(tsign * 1.0f32.copysign(f) * 1.0f32.copysign(h));
    (ssmin, ssmax, snr, csr, snl, csl)
}

/// DLARF 'Left': apply H = I - tau·v·vᵀ (v = [1, x]) to `a[r..16, c0..]`,
/// reference-BLAS accumulation order (row r0 first, then tail rows in order).
fn dlarf_left(r0: usize, x: &[f32], tau: f32, a: &mut [f32; 256], c0: usize) {
    if tau == 0.0 {
        return;
    }
    for c in c0..16 {
        let mut w = a[r0 * 16 + c]; // v[0] = 1 implicit
        for (i, &xi) in x.iter().enumerate() {
            w += xi * a[(r0 + 1 + i) * 16 + c];
        }
        let tw = tau * w;
        a[r0 * 16 + c] -= tw;
        for (i, &xi) in x.iter().enumerate() {
            a[(r0 + 1 + i) * 16 + c] -= xi * tw;
        }
    }
}

/// DLARF 'Right': apply H on the right to `a[r0.., c0..16]`; v = [1, x] sits
/// on row r0 starting at c0 (tail `x` is already stored there by the caller).
fn dlarf_right(r0: usize, c0: usize, x: &[f32], tau: f32, a: &mut [f32; 256]) {
    if tau == 0.0 {
        return;
    }
    for r in (r0 + 1)..16 {
        let mut w = a[r * 16 + c0];
        for (i, &xi) in x.iter().enumerate() {
            w += a[r * 16 + c0 + 1 + i] * xi;
        }
        // SLARF('Right') calls SGER with X=work and Y=v: each
        // column forms TEMP=(-tau)*v[c] before work[r]*TEMP.
        a[r * 16 + c0] += w * -tau;
        for (i, &xi) in x.iter().enumerate() {
            let temp = -tau * xi;
            a[r * 16 + c0 + 1 + i] += w * temp;
        }
    }
}

/// DGEBD2 on a 16x16 row-major matrix. On return `a` carries the left
/// reflector tails below the diagonal (LAPACK storage), `d`/`e` the
/// bidiagonal and `tq` the left taus. Right reflectors are applied (they
/// shape the bidiagonal) but not accumulated: the head never reads V.
fn dgebd2(a: &mut [f32; 256], d: &mut [f32; 16], e: &mut [f32; 15], tq: &mut [f32; 16]) {
    for j in 0..16 {
        let mut x = [0f32; 15];
        for r in 0..(15usize.saturating_sub(j)) {
            x[r] = a[(j + 1 + r) * 16 + j];
        }
        let (beta, tau) = dlarfg(a[j * 16 + j], &mut x[..15usize.saturating_sub(j)]);
        for r in 0..(15usize.saturating_sub(j)) {
            a[(j + 1 + r) * 16 + j] = x[r];
        }
        a[j * 16 + j] = beta;
        tq[j] = tau;
        if j < 15 {
            dlarf_left(j, &x[..15 - j], tau, a, j + 1);
        }
        d[j] = beta;
        if j < 15 {
            let mut xr = [0f32; 14];
            for c in 0..(14usize.saturating_sub(j)) {
                xr[c] = a[j * 16 + j + 2 + c];
            }
            let (beta2, tau2) = dlarfg(a[j * 16 + j + 1], &mut xr[..14usize.saturating_sub(j)]);
            for c in 0..(14usize.saturating_sub(j)) {
                a[j * 16 + j + 2 + c] = xr[c];
            }
            a[j * 16 + j + 1] = beta2;
            dlarf_right(j, j + 1, &xr[..14 - j], tau2, a);
            e[j] = beta2;
        }
    }
}

/// SORMBR('Q','L','N') -> SORM2R: apply the stored GEBD2 reflectors
/// directly to the bidiagonal left vectors. SGESDD uses this path rather
/// than materializing Q and multiplying Q*Ub, which rounds differently.
fn sorm2r_q(a: &[f32; 256], tq: &[f32; 16], u: &mut [f32; 256]) {
    for i in (0..16).rev() {
        let tau = tq[i];
        if tau == 0.0 {
            continue;
        }
        for c in 0..16 {
            let mut w = u[i * 16 + c];
            for r in (i + 1)..16 {
                w += a[r * 16 + i] * u[r * 16 + c];
            }
            let temp = -tau * w;
            u[i * 16 + c] += temp;
            for r in (i + 1)..16 {
                u[r * 16 + c] += a[r * 16 + i] * temp;
            }
        }
    }
}

/// Rotate U column pair (i, j): LAPACK DROT convention.
fn drot_cols(u: &mut [f32; 256], i: usize, j: usize, c: f32, s: f32) {
    for r in 0..16 {
        let (a, b) = (u[r * 16 + i], u[r * 16 + j]);
        u[r * 16 + i] = c * a + s * b;
        u[r * 16 + j] = c * b - s * a;
    }
}

/// DLASR 'R','V': apply a sequence of plane rotations to U column pairs
/// (c0+i, c0+i+1). forward: i = 0..k; backward: i = k-1..0.
fn dlasr_cols(u: &mut [f32; 256], c0: usize, cs: &[f32], sn: &[f32], forward: bool) {
    if forward {
        for i in 0..cs.len() {
            drot_cols(u, c0 + i, c0 + i + 1, cs[i], sn[i]);
        }
    } else {
        for i in (0..cs.len()).rev() {
            drot_cols(u, c0 + i, c0 + i + 1, cs[i], sn[i]);
        }
    }
}

/// DBDSQR, uplo='U', nru=16 (U accumulated), ncvt=0 (no Vt). Faithful port
/// of the reference control flow; `tol < 0` branches are dropped because
/// the caller always runs relative-accuracy (tol > 0).
fn dbdsqr(d: &mut [f32; 16], e: &mut [f32; 15], u: &mut [f32; 256]) {
    const N: usize = 16;
    let nm1 = N - 1;
    let nm12 = 2 * nm1;
    let nm13 = 3 * nm1;
    let mut idir = 0;
    let tolmult = (10.0f32).max((100.0f32).min(D_EPS.powf(-0.125)));
    let tol = tolmult * D_EPS;
    let mut smax = 0.0f32;
    for i in 0..N {
        smax = smax.max(d[i].abs());
    }
    for i in 0..N - 1 {
        smax = smax.max(e[i].abs());
    }
    // LAPACK inits SMIN=0; every path that reads it assigns smin_c first.
    #[allow(unused_assignments)]
    let mut smin = 0.0f32;
    // relative-accuracy threshold (tol > 0 path)
    let mut sminoa = d[0].abs();
    let thresh = if sminoa == 0.0 {
        // LAPACK goes through the absacc path here; threshold is unused
        // below eps-size anyway for our data
        6.0 * (N as f32 * (N as f32 * D_SAFMIN))
    } else {
        let mut mu = sminoa;
        for i in 1..N {
            mu = d[i].abs() * (mu / (mu + e[i - 1].abs()));
            sminoa = sminoa.min(mu);
            if sminoa == 0.0 {
                break;
            }
        }
        sminoa /= (N as f32).sqrt();
        (tol * sminoa).max(6.0 * (N as f32 * (N as f32 * D_SAFMIN)))
    };
    let maxitdivn = 6 * N;
    let mut iterdivn = 0;
    let mut iter = -1i64;
    let (mut oldll, mut oldm) = (-1i64, -1i64);
    let mut m = N; // 1-based position: active block is ll..m
    let mut work = [0f32; 4 * N];

    loop {
        // label 60
        if m <= 1 {
            break;
        }
        if iter >= N as i64 {
            iter -= N as i64;
            iterdivn += 1;
            if iterdivn >= maxitdivn {
                // LAPACK returns INFO>0 here: the caller (DMM head)
                // treats nonconvergence like convergence — the score
                // uses whatever basis was accumulated.
                return;
            }
        }
        // find block
        let mut smax2 = d[m - 1].abs();
        let mut ll = 0usize;
        for lll in 1..m {
            let llx = m - lll; // 1-based LL -> e index LL-1
            let abss = d[llx - 1].abs();
            let abse = if llx - 1 < N - 1 {
                e[llx - 1].abs()
            } else {
                0.0
            };
            if abse <= thresh {
                ll = llx;
                e[llx - 1] = 0.0;
                break;
            }
            smax2 = smax2.max(abss).max(abse);
        }
        if ll == m - 1 {
            m -= 1;
            continue;
        }
        ll += 1; // active block rows ll..m (1-based); 0-based ll-1..m-1
        if ll == m - 1 {
            // 2x2 block at (m-2, m-1) 0-based
            let (m1, m2) = (m - 2, m - 1);
            let (ssmin, ssmax, _sinr, _cosr, sinl, cosl) = dlasv2(d[m1], e[m1], d[m2]);
            d[m1] = ssmax;
            e[m1] = 0.0;
            d[m2] = ssmin;
            drot_cols(u, m1, m2, cosl, sinl);
            m -= 2;
            continue;
        }
        // shift direction
        if ll as i64 > oldm || (m as i64) < oldll {
            idir = if d[ll - 1].abs() >= d[m - 1].abs() {
                1
            } else {
                2
            };
        }
        // convergence tests
        if idir == 1 {
            if e[m - 2].abs() <= tol.abs() * d[m - 1].abs() {
                e[m - 2] = 0.0;
                continue;
            }
            let mut mu = d[ll - 1].abs();
            let mut smin_c = mu;
            let mut done = false;
            for lll in ll..m {
                if e[lll - 1].abs() <= tol * mu {
                    e[lll - 1] = 0.0;
                    done = true;
                    break;
                }
                mu = d[lll].abs() * (mu / (mu + e[lll - 1].abs()));
                smin_c = smin_c.min(mu);
            }
            if done {
                continue;
            }
            smin = smin_c;
        } else {
            if e[ll - 1].abs() <= tol.abs() * d[ll - 1].abs() {
                e[ll - 1] = 0.0;
                continue;
            }
            let mut mu = d[m - 1].abs();
            let mut smin_c = mu;
            let mut done = false;
            // LAPACK: LLL = M-1 downto LL over e index lll-1
            for lll in (ll..m).rev() {
                if e[lll - 1].abs() <= tol * mu {
                    e[lll - 1] = 0.0;
                    done = true;
                    break;
                }
                mu = d[lll - 1].abs() * (mu / (mu + e[lll - 1].abs()));
                smin_c = smin_c.min(mu);
            }
            if done {
                continue;
            }
            smin = smin_c;
        }
        oldll = ll as i64;
        oldm = m as i64;
        // shift
        let mut shift = 0.0;
        if tol * (N as f32) * (smin / smax2) <= D_EPS.max(0.01 * tol) {
            // tiny cluster: shift stays 0 (LAPACK's zero-shift path)
        } else {
            if idir == 1 {
                let sll = d[ll - 1].abs();
                let (s2, _r) = dlas2(d[m - 2], e[m - 2], d[m - 1]);
                shift = s2;
                if sll > 0.0 && (shift / sll).powi(2) < D_EPS {
                    shift = 0.0;
                }
            } else {
                let sll = d[m - 1].abs();
                let (s2, _r) = dlas2(d[ll - 1], e[ll - 1], d[ll]);
                shift = s2;
                if sll > 0.0 && (shift / sll).powi(2) < D_EPS {
                    shift = 0.0;
                }
            }
        }
        iter += (m - ll) as i64;
        if shift == 0.0 {
            if idir == 1 {
                let mut cs = 1.0;
                let mut oldcs = 1.0;
                let mut oldsn = 1.0;
                for i in (ll - 1)..(m - 1) {
                    let (csn, sn, r) = dlartg(d[i] * cs, e[i]);
                    cs = csn;
                    if i > ll - 1 {
                        e[i - 1] = oldsn * r;
                    }
                    let (ocs, osn, dd) = dlartg(oldcs * r, d[i + 1] * sn);
                    oldcs = ocs;
                    oldsn = osn;
                    d[i] = dd;
                    work[i - (ll - 1)] = cs;
                    work[i - (ll - 1) + nm1] = sn;
                    work[i - (ll - 1) + nm12] = oldcs;
                    work[i - (ll - 1) + nm13] = oldsn;
                }
                let hh = d[m - 1] * cs;
                d[m - 1] = hh * oldcs;
                e[m - 2] = hh * oldsn;
                dlasr_cols(
                    u,
                    ll - 1,
                    &work[nm12..nm12 + (m - ll)],
                    &work[nm13..nm13 + (m - ll)],
                    true,
                );
                if e[m - 2].abs() <= thresh {
                    e[m - 2] = 0.0;
                }
            } else {
                let mut cs = 1.0;
                let mut oldcs = 1.0;
                let mut oldsn = 1.0;
                for i in (ll..m).rev() {
                    // i 0-based from m-1 down to ll
                    let (csn, sn, r) = dlartg(d[i] * cs, e[i - 1]);
                    cs = csn;
                    if i < m - 1 {
                        e[i] = oldsn * r;
                    }
                    let (ocs, osn, dd) = dlartg(oldcs * r, d[i - 1] * sn);
                    oldcs = ocs;
                    oldsn = osn;
                    d[i] = dd;
                    work[i - ll] = cs;
                    work[i - ll + nm1] = -sn;
                    work[i - ll + nm12] = oldcs;
                    work[i - ll + nm13] = -oldsn;
                }
                let hh = d[ll - 1] * cs;
                d[ll - 1] = hh * oldcs;
                e[ll - 1] = hh * oldsn;
                dlasr_cols(
                    u,
                    ll - 1,
                    &work[..m - ll],
                    &work[nm1..nm1 + (m - ll)],
                    false,
                );
                if e[ll - 1].abs() <= thresh {
                    e[ll - 1] = 0.0;
                }
            }
        } else if idir == 1 {
            let mut f =
                (d[ll - 1].abs() - shift) * (1.0f32.copysign(d[ll - 1]) + shift / d[ll - 1]);
            let mut g = e[ll - 1];
            for i in (ll - 1)..(m - 1) {
                let (cosr, sinr, r) = dlartg(f, g);
                if i > ll - 1 {
                    e[i - 1] = r;
                }
                f = cosr * d[i] + sinr * e[i];
                e[i] = cosr * e[i] - sinr * d[i];
                g = sinr * d[i + 1];
                d[i + 1] *= cosr;
                let (cosl, sinl, r) = dlartg(f, g);
                d[i] = r;
                f = cosl * e[i] + sinl * d[i + 1];
                d[i + 1] = cosl * d[i + 1] - sinl * e[i];
                if i < m - 2 {
                    g = sinl * e[i + 1];
                    e[i + 1] *= cosl;
                }
                work[i - (ll - 1)] = cosr;
                work[i - (ll - 1) + nm1] = sinr;
                work[i - (ll - 1) + nm12] = cosl;
                work[i - (ll - 1) + nm13] = sinl;
            }
            e[m - 2] = f;
            dlasr_cols(
                u,
                ll - 1,
                &work[nm12..nm12 + (m - ll)],
                &work[nm13..nm13 + (m - ll)],
                true,
            );
            if e[m - 2].abs() <= thresh {
                e[m - 2] = 0.0;
            }
        } else {
            let mut f = (d[m - 1].abs() - shift) * (1.0f32.copysign(d[m - 1]) + shift / d[m - 1]);
            let mut g = e[m - 2];
            for i in (ll..m).rev() {
                let (cosr, sinr, r) = dlartg(f, g);
                if i < m - 1 {
                    e[i] = r;
                }
                f = cosr * d[i] + sinr * e[i - 1];
                e[i - 1] = cosr * e[i - 1] - sinr * d[i];
                g = sinr * d[i - 1];
                d[i - 1] *= cosr;
                let (cosl, sinl, r) = dlartg(f, g);
                d[i] = r;
                f = cosl * e[i - 1] + sinl * d[i - 1];
                d[i - 1] = cosl * d[i - 1] - sinl * e[i - 1];
                if i > ll {
                    g = sinl * e[i - 2];
                    e[i - 2] *= cosl;
                }
                work[i - ll] = cosr;
                work[i - ll + nm1] = -sinr;
                work[i - ll + nm12] = cosl;
                work[i - ll + nm13] = -sinl;
            }
            e[ll - 1] = f;
            if e[ll - 1].abs() <= thresh {
                e[ll - 1] = 0.0;
            }
            dlasr_cols(
                u,
                ll - 1,
                &work[..m - ll],
                &work[nm1..nm1 + (m - ll)],
                false,
            );
        }
    }
    // label 160: make positive (Vt sign-flip dropped: no Vt)
    for i in 0..N {
        if d[i] < 0.0 {
            d[i] = -d[i];
        }
    }
    // dbdsqr's own sort: move min to end, LE ties -> last occurrence
    for i in 1..N {
        let tgt = N - i;
        let mut isub = 0usize;
        let mut smin_ = d[0];
        for j in 1..=tgt {
            if d[j] <= smin_ {
                isub = j;
                smin_ = d[j];
            }
        }
        if isub != tgt {
            d[isub] = d[tgt];
            d[tgt] = smin_;
            for r in 0..16 {
                u.swap(r * 16 + isub, r * 16 + tgt);
            }
        }
    }
}

/// DLASDQ: dbdsqr + ascending selection sort on (d, U cols).
fn dlasdq(d: &mut [f32; 16], e: &mut [f32; 15], u: &mut [f32; 256]) {
    dbdsqr(d, e, u);
    for i in 0..16 {
        let mut isub = i;
        let mut smin_ = d[i];
        for j in (i + 1)..16 {
            if d[j] < smin_ {
                isub = j;
                smin_ = d[j];
            }
        }
        if isub != i {
            d[isub] = d[i];
            d[i] = smin_;
            for r in 0..16 {
                u.swap(r * 16 + isub, r * 16 + i);
            }
        }
    }
}

/// SVD of a 16×16 patch matrix via LAPACK GEBD2/BDSDC/ORM2R
/// stages (the `gesdd` path for n=16). `m` is the patch in natural
/// (y,x) layout; upstream feeds `Mᵀ` to `torch.svd`, so this transposes
/// first and returns `u` of the transposed matrix — the patch's own V.
fn svd16(m: &[f32; 256]) -> ([f32; 256], [f32; 16]) {
    let mut a = [0f32; 256];
    for i in 0..16 {
        for j in 0..16 {
            a[i * 16 + j] = m[j * 16 + i]; // P = Mᵀ
        }
    }
    let mut d = [0f32; 16];
    let mut e = [0f32; 15];
    let mut tq = [0f32; 16];
    dgebd2(&mut a, &mut d, &mut e, &mut tq);
    // Keep the reflector tails for SORMBR after the bidiagonal solve.
    // SBDSDC takes the N <= SMLSIZ (25) SLASDQ path BEFORE scaling.
    // At N=16 its bidiagonal must therefore remain unscaled.
    let mut ds = d;
    let mut es = e;
    let mut ub = [0f32; 256];
    for i in 0..16 {
        ub[i * 16 + i] = 1.0;
    }
    dlasdq(&mut ds, &mut es, &mut ub);
    // dbdsdc's descending selection sort with U column swaps
    for i in 0..16 {
        let mut kk = i;
        let mut p = ds[i];
        for j in (i + 1)..16 {
            if ds[j] > p {
                kk = j;
                p = ds[j];
            }
        }
        if kk != i {
            ds[kk] = ds[i];
            ds[i] = p;
            for r in 0..16 {
                ub.swap(r * 16 + kk, r * 16 + i);
            }
        }
    }
    sorm2r_q(&a, &tq, &mut ub);
    (ub, ds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_inputs_refuse_before_reading_pixels() {
        for (w, h) in [(4097, 128), (usize::MAX, 128), (4096, 127), (128, 4097)] {
            let img = za::Rgb8In {
                pixels: &[],
                width: w,
                height: h,
                stride: 0,
            };
            assert!(matches!(Dmm::prep(&img), Err(Error::TooLarge { .. })));
        }
        assert!(matches!(
            validate_geometry(0, 128),
            Err(Error::TooLarge { .. })
        ));
        assert!(matches!(
            validate_geometry(127, 128),
            Err(Error::TooSmall { .. })
        ));
        assert_eq!(validate_geometry(4096, 4096).is_ok(), usize::BITS >= 64);
        assert!(validate_geometry(1024, 1024).is_ok());
    }

    #[test]
    fn decomposition_reservation_overflow_is_an_error() {
        let tap = Tap {
            c: usize::MAX,
            h: 32,
            w: 32,
            data: Vec::new(),
        };
        let err = Dmm::decompose(&tap, za::EvalOpts::new())
            .err()
            .expect("overflow must refuse");
        assert_eq!(err.error().kind(), za::ErrorKind::InvalidInput);
    }

    #[test]
    fn upstream_resize_geometry_is_426_by_320() {
        let pixels = vec![0; 640 * 480 * 3];
        let img = za::Rgb8In {
            pixels: &pixels,
            width: 640,
            height: 480,
            stride: 640 * 3,
        };
        let (planar, h, w) = Dmm::prep(&img).unwrap();
        assert_eq!((w, h), (426, 320));
        assert_eq!(planar.len(), 3 * w * h);
    }

    #[test]
    fn decomposition_capacity_failure_is_recoverable() {
        // The patch count fits usize but its byte capacity cannot fit isize.
        let tap = Tap {
            c: usize::MAX / 2,
            h: 16,
            w: 16,
            data: Vec::new(),
        };
        let err = Dmm::decompose(&tap, za::EvalOpts::new())
            .err()
            .expect("capacity must refuse");
        assert_eq!(err.error().kind(), za::ErrorKind::Backend);
    }

    #[test]
    fn channel_work_observes_cancellation() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use za::how_far::{Stop, StopOnly, StopReason};
        struct StopAfterFirst(AtomicUsize);
        impl Stop for StopAfterFirst {
            fn check(&self) -> Result<(), StopReason> {
                if self.0.fetch_add(1, Ordering::Relaxed) >= 1 {
                    Err(StopReason::Cancelled)
                } else {
                    Ok(())
                }
            }
        }
        let tap = Tap {
            c: 2,
            h: 16,
            w: 16,
            data: vec![0.0; 512],
        };
        let stop = StopOnly::new(StopAfterFirst(AtomicUsize::new(0)));
        let err = Dmm::decompose(&tap, za::EvalOpts::new().with_pulse(&stop))
            .err()
            .expect("stop on second channel");
        assert_eq!(
            err.error().kind(),
            za::ErrorKind::Stopped(StopReason::Cancelled)
        );
        let stop = StopOnly::new(StopAfterFirst(AtomicUsize::new(0)));
        let err =
            Dmm::tap_score_cold(&tap, &tap, za::EvalOpts::new().with_pulse(&stop)).unwrap_err();
        assert_eq!(
            err.error().kind(),
            za::ErrorKind::Stopped(StopReason::Cancelled)
        );
        let reference = Dmm::decompose(&tap, za::EvalOpts::new()).unwrap();
        let stop = StopOnly::new(StopAfterFirst(AtomicUsize::new(0)));
        let err = Dmm::tap_score_warm(&reference, &tap, za::EvalOpts::new().with_pulse(&stop))
            .unwrap_err();
        assert_eq!(
            err.error().kind(),
            za::ErrorKind::Stopped(StopReason::Cancelled)
        );
    }

    #[test]
    fn lartg_preserves_reference_sign_convention() {
        assert_eq!(dlartg(-3.0, 4.0), (0.6, -0.8, -5.0));
        assert_eq!(dlartg(-3.0, 0.0), (1.0, 0.0, -3.0));
        assert_eq!(dlartg(0.0, -4.0), (0.0, -1.0, 4.0));
        assert_eq!(D_EPS, 2f32.powi(-24));
    }

    /// np.linalg.svd references on a fixed full-rank 16×16 matrix —
    /// singular values are convention-free, so the f64 references
    /// check the f32 solver at f32 accuracy (~1e-6 here).
    /// The matrix is filled by a splitmix64 hash of the flat index:
    /// linear/modular fills are low-rank (m[r,c] = f(16r+c) with f
    /// polynomial-ish has rank ≤ deg+1), which is how the earlier
    /// `(i*37+11)%97` constants went stale — its matrix was rank-3.
    #[test]
    fn svd16_singular_values() {
        let mut m = [0f32; 256];
        for (i, v) in m.iter_mut().enumerate() {
            let mut z = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            *v = (((z >> 11) & 0x1F_FFFF_FFFF_FFFF) as f64 / (1u64 << 53) as f64 - 0.5) as f32;
        }
        let (_, s) = svd16(&m);
        // np.linalg.svd of the same matrix
        let want = [
            2.1005899846614215,
            1.9556434753669627,
            1.5021354514513066,
            1.3943691120601851,
            1.3701256032148814,
            1.2917171886047512,
            1.2268008722835027,
            1.0570710744129788,
            0.8143173918116010,
            0.7853124261302923,
            0.6858541004635299,
            0.4762404932959022,
            0.4056329466236199,
            0.2141296450504027,
            0.1273009999997064,
            0.0431200927323836,
        ];
        for (i, (&got, &w)) in s.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(got) - w).abs() < 1e-5,
                "s[{i}] = {got}, want {w}"
            );
        }
    }

    #[test]
    fn svd16_identity_and_reconstruction() {
        let mut m = [0f32; 256];
        for i in 0..16 {
            m[i * 16 + i] = 2.0;
        }
        let (_, s) = svd16(&m);
        for &sv in &s {
            assert!((sv - 2.0).abs() < 1e-6);
        }
        // The returned "u" columns are the patch's right singular vectors
        // (v, upstream's u of the transposed patch) — check orthonormality.
        let mut m = [0f32; 256];
        for i in 0..256 {
            m[i] = (((i * 53 + 7) % 89) as f64 / 89.0 - 0.4) as f32;
        }
        let (u, _s) = svd16(&m);
        for c in 0..16 {
            let n: f64 = (0..16)
                .map(|r| f64::from(u[r * 16 + c]) * f64::from(u[r * 16 + c]))
                .sum();
            assert!((n - 1.0).abs() < 1e-5, "u_{c} norm {n}");
        }
        for c in 0..15 {
            for d in c + 1..16 {
                let dot: f64 = (0..16)
                    .map(|r| f64::from(u[r * 16 + c]) * f64::from(u[r * 16 + d]))
                    .sum();
                assert!(dot.abs() < 1e-5, "u_{c}·u_{d} = {dot}");
            }
        }
    }
}
