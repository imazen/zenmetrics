// Port of BasicSR's `basicsr/metrics/niqe.py`
// (https://github.com/XPixelGroup/BasicSR, Apache-2.0, XPixel Group; notice
// kept in `LICENSE-BasicSR`), itself a faithful port of LIVE's MATLAB
// `niqe_release`. The embedded pristine-model parameters in `params.rs` are
// LIVE's `modelparameters.mat` data (BasicSR's `niqe_pris_params.npz` is
// bit-identical to it — verified maxdiff 0.0); the LIVE redistribution
// notice is kept verbatim in `LICENSE-LIVE-niqe`.
//
// Reference:
//   A. Mittal, R. Soundararajan and A. C. Bovik, "Making a 'Completely
//   Blind' Image Quality Analyzer", IEEE Signal Processing Letters,
//   vol. 20, no. 3, pp. 209-212, March 2013. doi:10.1109/LSP.2012.2227726

//! NIQE — Natural Image Quality Evaluator (no-reference).
//!
//! A pure-Rust port of BasicSR's `niqe.py` (Apache-2.0) using LIVE's
//! pristine-model parameters. NIQE fits a 36-dimensional multivariate
//! Gaussian over per-block AGGD features of MSCN-normalized luminance at two
//! scales and returns the Mahalanobis distance to the pristine model —
//! higher means worse ([`ScoreKind::Distortion`](za::ScoreKind::Distortion)).
//!
//! Conventions matched to the oracle (`basicsr.metrics.niqe`,
//! `calculate_niqe(img, crop_border=0, input_order='HWC')`):
//!
//! - RGB8 → gray is MATLAB `rgb2gray`-style
//!   `Y ≈ 16 + 65.481·r + 128.553·g + 24.966·b` over `[0,1]`-normalized
//!   channels: the dot product runs in f64 on f32-normalized inputs, the
//!   result is divided by 255 (f64), cast to f32, multiplied by 255 (f32),
//!   then rounded (ties to even). GRAY8 input is the plane verbatim.
//! - The image is then cropped to the largest multiple of `48·scale_max`
//!   (= 96) pixels, dropping the leftover border — matching the oracle's
//!   `img = img[:num_block_h*96, :num_block_w*96]` (its `niqe()` helper
//!   computes `num_block_*` before cropping).
//! - Non-overlapping 96×96 blocks at scale 1; the image is halved by a
//!   MATLAB-compatible antialiased bicubic `imresize` (a = −0.5,
//!   kernel width 4/scale = 8, symmetric edge extension) and 48×48 blocks
//!   are used at scale 2.
//! - Per block: AGGD (α, mean, β_l, β_r) of the MSCN map plus four
//!   `np.roll`-shifted product maps (right, down, down-right, down-left) →
//!   18 features per scale, 36 total. Blocks/rows containing NaN are dropped
//!   from the covariance and skipped by `nanmean`, as in the oracle.
//! - `quality = sqrt(Δμ · pinv((Σ_pristine + Σ_dist)/2) · Δμᵀ)` with
//!   numpy's default pinv cutoff `1e-15·σ_max`.
//!
//! Deliberate divergences: images that leave fewer than two complete 96×96
//! blocks after `crop_border` report [`ErrorKind::TooSmall`] — the oracle
//! crashes there (`np.cov` of <2 rows is NaN → `pinv`: "SVD did not
//! converge"). Fully degenerate content (all-flat images) still propagates
//! NaN as the oracle does.
//!
//! [`ErrorKind::TooSmall`]: zenmetrics_abstractions::ErrorKind::TooSmall

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use za::zenpixels::PixelDescriptor;
use zenmetrics_abstractions as za;

mod params;
use params::{COV_PRIS, GAUSS_WINDOW, MU_PRIS};

/// Output revision of [`Niqe`] on the generic surface; bump when any output
/// can change.
const SCORE_REVISION: u32 = 1;

/// Analysis block size at scale 1 (BasicSR `sharpness_scale = 2` →
/// `48·scale_max = 96`; scale 2 blocks are 48×48).
const BLOCK: usize = 96;
/// Feature count per block: 2 + 4·4 per scale, × 2 scales.
const NFEAT: usize = 36;
/// `gam` grid: `np.arange(0.2, 10.001, 0.001)` → 9801 entries. numpy's arange
/// fill recomputes the step as `f64(start + step) − start` (verified
/// bit-identical to numpy 2.5), so `GAM_STEP` here is that derived step, not
/// the literal 0.001.
const GAM_LO: f64 = 0.2;
const GAM_STEP: f64 = (0.2 + 0.001) - 0.2;
const GAM_N: usize = 9801;

/// Estimated just-noticeable difference of the `niqe` score, measured on the
/// KonJND-1k corpus: `zenmetrics batch --metric niqe` over all 1,008 pairs
/// (`PairedNoReference` gives `niqe` and `niqe_ref` per pair), then JND =
/// median over pairs of `|niqe − niqe_ref|` = 0.602329 (JPEG 0.506864,
/// BPG 0.710947). See `benchmarks/nr_classic_2026-10-15/jnd.md` for the
/// full distribution.
pub const JND: f64 = 0.602329;

/// The NIQE metric as a generic no-reference scorer — accepts packed
/// `RGB8_SRGB` and `GRAY8`. Headline = `niqe`
/// ([`ScoreKind::Distortion`](za::ScoreKind::Distortion): higher is worse).
///
/// `crop_border` mirrors the oracle's `crop_border` argument: that many
/// pixels are trimmed from every edge before evaluation (default 0).
#[derive(Debug, Clone, Copy)]
pub struct Niqe {
    /// Pixels trimmed from each border before scoring (oracle `crop_border`).
    pub crop_border: u32,
}

impl Niqe {
    /// A new NIQE scorer with the oracle default (`crop_border = 0`).
    #[must_use]
    pub fn new() -> Self {
        Self { crop_border: 0 }
    }

    /// A scorer with an explicit border crop (oracle `crop_border`).
    #[must_use]
    pub fn with_crop_border(crop_border: u32) -> Self {
        Self { crop_border }
    }
}

impl Default for Niqe {
    fn default() -> Self {
        Self::new()
    }
}

impl za::NoReferenceMetric for Niqe {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("niqe", za::implementation!(SCORE_REVISION))
            .with_reference("BasicSR niqe.py + LIVE niqe_release params")
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("niqe", za::ScoreKind::Distortion)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        let d = format.packed();
        d == Some(PixelDescriptor::RGB8_SRGB) || d == Some(PixelDescriptor::GRAY8)
    }

    fn evaluate_image(
        &mut self,
        image: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        opts.checkpoint()?;
        opts.refuse_map("niqe")?;
        let (y, width, height) = gray_f64(&image, self.crop_border as usize)?;
        // The oracle needs ≥2 blocks for a computable covariance — below
        // that `np.cov` is all-NaN and `np.linalg.pinv` raises "SVD did not
        // converge". We report TooSmall instead (documented divergence).
        if (width / BLOCK) * (height / BLOCK) < 2 {
            return za::fail(
                za::MetricError::new(za::ErrorKind::TooSmall)
                    .with_metric("niqe")
                    .with_message(alloc::format!(
                        "image too small for NIQE: {width}x{height} usable after \
                         crop_border={} (needs two {BLOCK}x{BLOCK} blocks)",
                        self.crop_border
                    )),
            );
        }
        let score = niqe_quality(&y, width, height, opts)?;
        Ok(Box::new(za::ScalarEval::single("niqe", score)))
    }
}

/// Borrow the packed slice when its descriptor is `desc`.
fn packed_as<'a>(
    image: &'a za::Image<'a>,
    desc: PixelDescriptor,
) -> Option<(&'a [u8], usize, usize, usize)> {
    let za::Image::Packed(s) = image else {
        return None;
    };
    if s.descriptor() != desc {
        return None;
    }
    Some((
        s.as_strided_bytes(),
        s.width() as usize,
        s.rows() as usize,
        s.stride(),
    ))
}

/// The oracle's Y plane as f64 (values are the f32-rounded oracle output).
/// `crop` pixels are trimmed from every edge first.
fn gray_f64(image: &za::Image<'_>, crop: usize) -> za::MetricResult<(Vec<f64>, usize, usize)> {
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::RGB8_SRGB) {
        let (iw, ih) = (w.saturating_sub(2 * crop), h.saturating_sub(2 * crop));
        if iw == 0 || ih == 0 {
            return za::fail(
                za::MetricError::new(za::ErrorKind::TooSmall)
                    .with_metric("niqe")
                    .with_message(alloc::format!(
                        "image too small for NIQE: {w}x{h} minus crop_border={crop}"
                    )),
            );
        }
        let mut y = Vec::with_capacity(iw * ih);
        for r in 0..ih {
            let row = &px[(r + crop) * stride..(r + crop) * stride + w * 3];
            for c in 0..iw {
                let x = (c + crop) * 3;
                y.push(to_y(row[x], row[x + 1], row[x + 2]));
            }
        }
        return Ok((y, iw, ih));
    }
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::GRAY8) {
        let (iw, ih) = (w.saturating_sub(2 * crop), h.saturating_sub(2 * crop));
        if iw == 0 || ih == 0 {
            return za::fail(
                za::MetricError::new(za::ErrorKind::TooSmall)
                    .with_metric("niqe")
                    .with_message(alloc::format!(
                        "image too small for NIQE: {w}x{h} minus crop_border={crop}"
                    )),
            );
        }
        let mut y = Vec::with_capacity(iw * ih);
        for r in 0..ih {
            y.extend(
                px[(r + crop) * stride + crop..(r + crop) * stride + crop + iw]
                    .iter()
                    .map(|&v| f64::from(v)),
            );
        }
        return Ok((y, iw, ih));
    }
    za::fail(za::unsupported(
        &image.format(),
        "packed RGB8_SRGB or GRAY8",
    ))
}

/// `bgr2ycbcr(y_only)` + `*255` + `.round()` — the oracle's exact f32/f64
/// cast chain (see module docs). `r`,`g`,`b` are u8 samples.
fn to_y(r: u8, g: u8, b: u8) -> f64 {
    // to_y_channel: img.astype(f32)/255. — f32 division (weak scalar).
    let nf = |v: u8| f64::from(v as f32 / 255.0f32);
    // np.dot(f32 HWC, f64 [3]) promotes: f64 accumulation on f32-widened
    // inputs; +16.0; then _convert_output_type_range: f64 /= 255 → f32;
    // then to_y_channel's img*255. is an f32 multiply; .round() in f32.
    let d = 24.966 * nf(b) + 128.553 * nf(g) + 65.481 * nf(r) + 16.0;
    let y01 = (d / 255.0) as f32;
    f64::from((y01 * 255.0f32).round_ties_even())
}

/// Full NIQE pipeline on the Y plane (`niqe()` + `computequality`).
/// Returns NaN where the oracle does (degenerate content → NaN features).
fn niqe_quality(img: &[f64], w: usize, h: usize, opts: za::EvalOpts<'_>) -> za::MetricResult<f64> {
    // r_gam lookup, hoisted out of estimate_aggd_param: a pure function of
    // the fixed gam grid. The oracle computes
    // `square(gamma(gr*2)) / (gamma(gr) * gamma(gr*3))` with `gr = 1/gam[i]`
    // — the arguments are gr·2 and gr·3, not 2/g (one-ulp difference).
    let mut r_gam = Vec::with_capacity(GAM_N);
    for i in 0..GAM_N {
        let gr = 1.0 / (GAM_LO + i as f64 * GAM_STEP);
        let (g1, g2, g3) = (
            cephes_gamma(gr),
            cephes_gamma(gr * 2.0),
            cephes_gamma(gr * 3.0),
        );
        r_gam.push((g2 * g2) / (g1 * g3));
    }

    let nb_h = h / BLOCK;
    let nb_w = w / BLOCK;
    let nblocks = nb_h * nb_w;
    // np.concatenate([feat_s1, feat_s2], axis=1): row = one block, cols =
    // 18 scale-1 features then 18 scale-2 features.
    let mut distparam: Vec<f64> = vec![0.0; nblocks * NFEAT];

    // The oracle crops img to nb*96 once, then halves the cropped plane.
    // Everything through the feature extraction is f32: scipy's convolve
    // uses the input dtype (f32), and np.mean/sqrt stay f32 — only the
    // gamma evaluations and beta/mean products promote to f64.
    let (mut cw, mut ch) = (nb_w * BLOCK, nb_h * BLOCK);
    let mut cur: Vec<f32> = (0..ch)
        .flat_map(|r| img[r * w..r * w + cw].iter().map(|&v| v as f32))
        .collect();

    for scale_iter in 0..2u8 {
        if scale_iter == 1 {
            // imresize(img/255, 0.5, antialiasing) * 255 — all f32 ops.
            let scaled_in: Vec<f32> = cur.iter().map(|&v| v / 255.0).collect();
            cur = imresize(&scaled_in, cw, ch, 0.5);
            cw = (cw as f64 * 0.5).ceil() as usize;
            ch = (ch as f64 * 0.5).ceil() as usize;
            for v in &mut cur {
                *v *= 255.0;
            }
        }
        opts.checkpoint()?;

        let mu = conv_gauss(&cur, cw, ch);
        let sq: Vec<f32> = cur.iter().map(|&v| v * v).collect();
        let mu_sq = conv_gauss(&sq, cw, ch);
        let mut norm = vec![0.0f32; cw * ch];
        for i in 0..norm.len() {
            let sigma = (mu_sq[i] - mu[i] * mu[i]).abs().sqrt();
            norm[i] = (cur[i] - mu[i]) / (sigma + 1.0);
        }

        let bs = BLOCK >> scale_iter; // 96 at scale 1, 48 at scale 2
        // Same block iteration order both scales → consistent block index.
        for bw in 0..nb_w {
            for bh in 0..nb_h {
                let bi = bw * nb_h + bh;
                let (r0, c0) = (bh * bs, bw * bs);
                let block: Vec<f32> = (0..bs)
                    .flat_map(|r| {
                        norm[(r0 + r) * cw + c0..(r0 + r) * cw + c0 + bs]
                            .iter()
                            .copied()
                    })
                    .collect();
                let base = bi * NFEAT + scale_iter as usize * 18;
                compute_feature(&block, bs, &r_gam, &mut distparam[base..base + 18]);
            }
        }
    }

    let mu_dist = nanmean_cols(&distparam, nblocks, NFEAT);
    let cov_dist = cov_drop_nan(&distparam, nblocks, NFEAT);

    // invcov = pinv((cov_pris + cov_dist)/2), numpy rcond = eps·N·σ_max.
    let mut avg = [0.0f64; NFEAT * NFEAT];
    let mut finite = true;
    for i in 0..NFEAT * NFEAT {
        avg[i] = (COV_PRIS[i] + cov_dist[i]) / 2.0;
        finite &= avg[i].is_finite();
    }
    let invcov = if finite {
        pinv_sym(&avg)
    } else {
        [f64::NAN; NFEAT * NFEAT]
    };

    let mut q = 0.0f64;
    for i in 0..NFEAT {
        let di = MU_PRIS[i] - mu_dist[i];
        for j in 0..NFEAT {
            q += di * invcov[i * NFEAT + j] * (MU_PRIS[j] - mu_dist[j]);
        }
    }
    Ok(q.sqrt())
}

/// scipy `ndimage.convolve(img, gaussian_window, mode='nearest')` — true
/// convolution (the window is symmetric) over a clamped-edge input. scipy
/// widens the computation: both the f32 image and the f64 window are used
/// in f64, then the result is stored in the input's f32 dtype (verified
/// bitwise vs scipy 1.18: f64·f64 accumulate → f32 store matches exactly).
fn conv_gauss(img: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            let mut acc = 0.0f64;
            for a in 0..7usize {
                let rr = (r + 3).saturating_sub(a).min(h - 1);
                for b in 0..7usize {
                    let cc = (c + 3).saturating_sub(b).min(w - 1);
                    acc += f64::from(img[rr * w + cc]) * GAUSS_WINDOW[a * 7 + b];
                }
            }
            out[r * w + c] = acc as f32;
        }
    }
    out
}

/// MATLAB-compatible antialiased bicubic `imresize(img, scale)` for
/// `scale < 1` — port of BasicSR's `imresize`: `calculate_weights_indices`
/// (kernel width 4/scale, cubic a=−0.5, taps scaled by `scale`) + symmetric
/// extension + two 1-D passes (rows then columns of the transposed middle).
fn imresize(img: &[f32], w: usize, h: usize, scale: f64) -> Vec<f32> {
    // The oracle processes the H dimension first, then W — match that order.
    // Pass 1: resample along each column (transpose, resample rows of h).
    let mut t = vec![0.0f32; w * h];
    for r in 0..h {
        for c in 0..w {
            t[c * h + r] = img[r * w + c];
        }
    }
    let t1 = resample_rows(&t, h, w, scale, false); // w rows × out_h
    let out_h = (h as f64 * scale).ceil() as usize;
    // transpose back → out_h × w, then pass 2 along each row (length w).
    let mut tmp = vec![0.0f32; out_h * w];
    for r in 0..out_h {
        for c in 0..w {
            tmp[r * w + c] = t1[c * out_h + r];
        }
    }
    resample_rows(&tmp, w, out_h, scale, true)
}

/// One 1-D antialiased resample pass along rows: `rows` rows of length
/// `in_len` in `img` → `rows` rows of `ceil(in_len·scale)` out.
/// Returns row-major `rows × out_len`.
///
/// The oracle computes this in **f32** (torch `.float()` tensors): the
/// coordinates/weights/dot products are all f32. We reproduce the f32
/// arithmetic exactly and widen the result to f64 — that keeps the
/// quantization the oracle's scale-2 features see.
///
/// `dot8` selects the accumulation order the oracle's `Tensor.mv` uses. The
/// first pass's matrix is a transposed view (each output is a strided dot →
/// torch uses a plain f32 FMA chain). The second pass's matrix has contiguous
/// 8-element rows, and torch's row-dot reduces `w·v` as
/// `((p0+p1)+(p4+p5)) + ((p2+p3)+(p6+p7))` (hadd-style pairing) — verified
/// bitwise against torch on this host for both layouts.
fn resample_rows(img: &[f32], in_len: usize, rows: usize, scale: f64, dot8: bool) -> Vec<f32> {
    let out_len = (in_len as f64 * scale).ceil() as usize;
    // kernel_width = 4/scale; p = ceil(kernel_width) + 2 taps per output.
    let kernel_width = 4.0f32 / scale as f32;
    let p = (kernel_width as f64).ceil() as usize + 2;
    let mut indices = vec![0i64; out_len * p];
    let mut weights = vec![0.0f32; out_len * p];
    for k in 0..out_len {
        let x = (k + 1) as f32; // torch.linspace(1, out, out) in f32
        let u = x / scale as f32 + 0.5 * (1.0 - 1.0 / scale as f32);
        let left = (u - kernel_width / 2.0).floor() as i64;
        let mut wsum = 0.0f32;
        for j in 0..p {
            let ix = left + j as i64;
            // distance_to_center = u − indices (f32), cubic in f32, × scale.
            let wgt = scale as f32 * cubic32((u - ix as f32) * scale as f32);
            indices[k * p + j] = ix;
            weights[k * p + j] = wgt;
            wsum += wgt;
        }
        for j in 0..p {
            weights[k * p + j] /= wsum;
        }
    }
    // The oracle drops the first and last tap columns when they are all-zero
    // (`narrow(1, 1, p-2)`, then re-checks the new last). For scale 0.5 the
    // all-zero scan below produces the same cols 1..=p-2.
    let mut lo = 0usize;
    let mut hi = p;
    while lo < hi && (0..out_len).all(|k| weights[k * p + lo] == 0.0) {
        lo += 1;
    }
    while hi > lo && (0..out_len).all(|k| weights[k * p + hi - 1] == 0.0) {
        hi -= 1;
    }
    let ntap = hi - lo;
    let (mut min_idx, mut max_idx) = (i64::MAX, i64::MIN);
    for k in 0..out_len {
        for j in lo..hi {
            let ix = indices[k * p + j];
            min_idx = min_idx.min(ix);
            max_idx = max_idx.max(ix);
        }
    }
    let sym_s = (1 - min_idx).max(0) as usize; // −indices.min() + 1
    let sym_e = (max_idx - in_len as i64).max(0) as usize; // indices.max() − in_length

    let aug_len = sym_s + in_len + sym_e;
    let mut out = vec![0.0f32; rows * out_len];
    for r in 0..rows {
        let row = &img[r * in_len..(r + 1) * in_len];
        let mut aug = vec![0.0f32; aug_len];
        for i in 0..aug_len {
            aug[i] = if i < sym_s {
                row[sym_s - 1 - i]
            } else if i < sym_s + in_len {
                row[i - sym_s]
            } else {
                row[in_len - 1 - (i - sym_s - in_len)]
            };
        }
        for k in 0..out_len {
            let mut w8 = [0.0f32; 8];
            let mut v8 = [0.0f32; 8];
            for j in 0..ntap {
                // torch adds sym_len_s - 1 to the (1-based) indices.
                let ix = indices[k * p + lo + j] + sym_s as i64 - 1;
                w8[j] = weights[k * p + lo + j];
                v8[j] = aug[ix as usize];
            }
            out[r * out_len + k] = if dot8 && ntap == 8 {
                // torch row-dot for contiguous 8-element rows (see fn docs).
                let s01 = w8[0] * v8[0] + w8[1] * v8[1];
                let s23 = w8[2] * v8[2] + w8[3] * v8[3];
                let s45 = w8[4] * v8[4] + w8[5] * v8[5];
                let s67 = w8[6] * v8[6] + w8[7] * v8[7];
                (s01 + s45) + (s23 + s67)
            } else {
                // strided dot → f32 FMA chain, verified bitwise vs torch.
                let mut acc = 0.0f32;
                for j in 0..ntap {
                    acc = fmaf(w8[j], v8[j], acc);
                }
                acc
            };
        }
    }
    out
}

/// f32 fused multiply-add: `fma(w, v, acc)` — `f32::mul_add` under `std`,
/// `libm::fmaf` under `no_std` (identical semantics).
#[inline(always)]
fn fmaf(w: f32, v: f32, acc: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        w.mul_add(v, acc)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(w, v, acc)
    }
}

/// BasicSR `cubic` with a = −0.5 (MATLAB bicubic kernel) — evaluated in f32
/// as the oracle's torch path does.
fn cubic32(x: f32) -> f32 {
    let ax = x.abs();
    if ax <= 1.0 {
        1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0
    } else if ax <= 2.0 {
        -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0
    } else {
        0.0
    }
}

/// `estimate_aggd_param`: (α, _, β_l, β_r) — mean slot is computed by the
/// caller per the oracle's shifted-feature path. `block` is flat row-major
/// **f32** (the oracle's `img_nomalized` dtype): `np.mean`/`np.sqrt` on f32
/// stay f32 — `np.mean` accumulates with numpy's f32 pairwise summation,
/// scalar `**` is C `powf`, and the whole std/rhat chain is f32.
/// `gam`/`r_gam` are f64 and `f32 · f64` products promote, so βs are f64.
fn aggd(block: &[f32], r_gam: &[f64]) -> [f64; 4] {
    // `block[block<0]**2` / `block[block>0]**2`, `np.abs`, `block**2` — the
    // oracle materializes these as f32 arrays, then `np.mean` reduces them
    // with numpy's f32 pairwise summation. Scalar `**k` on np.float32 goes
    // through C `powf` (verified bitwise).
    let mut neg = Vec::with_capacity(block.len());
    let mut pos = Vec::with_capacity(block.len());
    let mut abs = Vec::with_capacity(block.len());
    let mut sq = Vec::with_capacity(block.len());
    for &v in block {
        let v2 = v * v;
        if v < 0.0 {
            neg.push(v2);
        } else if v > 0.0 {
            pos.push(v2);
        }
        abs.push(v.abs());
        sq.push(v2);
    }
    // np.mean on an empty array returns NaN (RuntimeWarning); np.sqrt
    // propagates it, so a block with no negatives → left_std = NaN.
    let mean32 = |a: &[f32]| {
        if a.is_empty() {
            f32::NAN
        } else {
            pairwise_sum_f32(a) / a.len() as f32
        }
    };
    let left_std = mean32(&neg).sqrt();
    let right_std = mean32(&pos).sqrt();
    let gammahat = left_std / right_std;
    let rhat = powf32(mean32(&abs), 2.0) / mean32(&sq);
    // (rhat·(γ̂³+1)·(γ̂+1)) / ((γ̂²+1)²) — left-associative f32, `**` = powf.
    let g2 = powf32(gammahat, 2.0) + 1.0;
    let rhatnorm = rhat * (powf32(gammahat, 3.0) + 1.0) * (gammahat + 1.0) / powf32(g2, 2.0);
    // np.argmin((r_gam − rhatnorm)**2): array `**2` is elementwise multiply;
    // NaN distances never compare below `best`, so all-NaN → position 0.
    let rhatnorm64 = f64::from(rhatnorm);
    let mut pos = 0usize;
    let mut best = f64::INFINITY;
    for (i, &r) in r_gam.iter().enumerate() {
        let d = (r - rhatnorm64) * (r - rhatnorm64);
        if d < best {
            best = d;
            pos = i;
        }
    }
    let alpha = GAM_LO + pos as f64 * GAM_STEP;
    // gamma(1/α)/gamma(3/α): division arguments, distinct from the gr·k
    // arguments in the r_gam table (one-ulp difference).
    let beta_norm = (cephes_gamma(1.0 / alpha) / cephes_gamma(3.0 / alpha)).sqrt();
    [
        alpha,
        0.0,
        f64::from(left_std) * beta_norm,
        f64::from(right_std) * beta_norm,
    ]
}

/// numpy `pairwise_sum` for f32 (from `_core/src/umath/loops_utils` /
/// `pairwise_sum_@TYPE@`): <8 sequential; ≤128 eight-accumulator unroll with
/// the `((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))` combine and sequential tail;
/// otherwise split at the largest multiple of 8 ≤ n/2 and recurse.
fn pairwise_sum_f32(a: &[f32]) -> f32 {
    const PW: usize = 128;
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0f32, |s, &v| s + v)
    } else if n <= PW {
        let mut r = [0.0f32; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let n2 = n / 2 - (n / 2) % 8;
        pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
    }
}

/// Same pairwise order for f64 (np.mean / np.cov's `average` on f64).
fn pairwise_sum_f64(a: &[f64]) -> f64 {
    const PW: usize = 128;
    let n = a.len();
    if n < 8 {
        a.iter().fold(0.0f64, |s, &v| s + v)
    } else if n <= PW {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        res
    } else {
        let n2 = n / 2 - (n / 2) % 8;
        pairwise_sum_f64(&a[..n2]) + pairwise_sum_f64(&a[n2..])
    }
}

/// C `powf` — `f32::powf` under `std` (glibc, matching the oracle bitwise),
/// `libm::powf` under `no_std`.
#[inline(always)]
fn powf32(x: f32, y: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.powf(y)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::powf(x, y)
    }
}

/// scipy `special.gamma` = cephes `Gamma` (scipy vendored xsf/cephes), ported
/// for the positive domain the oracle evaluates (arguments of `estimate_aggd_param`
/// are in (0, 16]). Nonpositive x is unreachable here; x > 33 uses the same
/// Stirling path cephes does.
fn cephes_gamma(x: f64) -> f64 {
    // Coefficients verbatim from scipy's xsf/cephes gamma.h.
    const P: [f64; 7] = [
        1.60119522476751861407E-4,
        1.19135147006586384913E-3,
        1.04213797561761569935E-2,
        4.76367800457137231464E-2,
        2.07448227648435975150E-1,
        4.94214826801497100753E-1,
        9.99999999999999996796E-1,
    ];
    const Q: [f64; 8] = [
        -2.31581873324120129819E-5,
        5.39605580493303397842E-4,
        -4.45641913851797240494E-3,
        1.18139785222060435552E-2,
        3.58236398605498653373E-2,
        -2.34591795718243348568E-1,
        7.14304917030273074085E-2,
        1.00000000000000000320E0,
    ];
    const STIR: [f64; 5] = [
        7.87311395793093628397E-4,
        -2.29549961613378126380E-4,
        -2.68132617805781232825E-3,
        3.47222221605458667310E-3,
        8.33333333333482257126E-2,
    ];
    fn polevl(x: f64, c: &[f64]) -> f64 {
        let mut a = c[0];
        for &k in &c[1..] {
            a = a * x + k;
        }
        a
    }
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::INFINITY.copysign(x);
    }
    let q = x.abs();
    if q > 33.0 {
        if x < 0.0 {
            // Negative-int/reflection branch — the oracle never reaches it
            // (arguments are > 0); fall through to the Stirling estimate.
            let w = 1.0 / q;
            let w = 1.0 + w * polevl(w, &STIR);
            let v = powf64(q, 0.5 * q - 0.25);
            let y = SQRT_PI * v * (v / exp64(q)) * w;
            return -(core::f64::consts::PI / (q * sinpi64(q - q.floor()) * y).abs());
        }
        // stirf(x)
        let w = 1.0 / x;
        let w = 1.0 + w * polevl(w, &STIR);
        let y = if x > 143.01608 {
            let v = powf64(x, 0.5 * x - 0.25);
            v * (v / exp64(x))
        } else {
            powf64(x, x - 0.5) / exp64(x)
        };
        return SQRT_PI * y * w;
    }
    let mut z = 1.0;
    let mut x = x;
    while x >= 3.0 {
        x -= 1.0;
        z *= x;
    }
    while x < 0.0 {
        if x > -1.0e-9 {
            return z / ((1.0 + 0.5772156649015329 * x) * x);
        }
        z /= x;
        x += 1.0;
    }
    while x < 2.0 {
        if x < 1.0e-9 {
            return z / ((1.0 + 0.5772156649015329 * x) * x);
        }
        z /= x;
        x += 1.0;
    }
    if x == 2.0 {
        return z;
    }
    let x = x - 2.0;
    z * polevl(x, &P) / polevl(x, &Q)
}

const SQRT_PI: f64 = 1.77245385090551602792983;

#[cfg(feature = "std")]
fn exp64(x: f64) -> f64 {
    x.exp()
}
#[cfg(not(feature = "std"))]
fn exp64(x: f64) -> f64 {
    libm::exp(x)
}
#[cfg(feature = "std")]
fn powf64(x: f64, y: f64) -> f64 {
    x.powf(y)
}
#[cfg(not(feature = "std"))]
fn powf64(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
fn sinpi64(x: f64) -> f64 {
    // scipy's sinpi: argument reduced to [0,2) then sin(π·x) on the reduced
    // value with exact quarter-point handling. Only reached for x<0 args
    // (unreachable in this metric); f64 sin approximation suffices.
    let mut x = x % 2.0;
    if x < 0.0 {
        x += 2.0;
    }
    match x {
        0.0 | 1.0 => 0.0f64.copysign(x),
        0.5 => 1.0,
        1.5 => -1.0,
        _ => {
            #[cfg(feature = "std")]
            {
                (core::f64::consts::PI * x).sin()
            }
            #[cfg(not(feature = "std"))]
            {
                libm::sin(core::f64::consts::PI * x)
            }
        }
    }
}

/// `compute_feature` for one block: [α, (βl+βr)/2] of the block itself,
/// then for np.roll shifts [[0,1],[1,0],[1,1],[1,−1]] on the shifted
/// product: [α, mean, βl, βr] with `mean = (βr−βl)·Γ(2/α)/Γ(1/α)`.
/// Writes 18 values into `out` (len ≥ 18).
fn compute_feature(block: &[f32], bs: usize, r_gam: &[f64], out: &mut [f64]) {
    let a0 = aggd(block, r_gam);
    out[0] = a0[0];
    out[1] = (a0[2] + a0[3]) / 2.0;
    let mut o = 2;
    for (dy, dx) in [(0i64, 1i64), (1, 0), (1, 1), (1, -1)] {
        let mut prod = vec![0.0f32; bs * bs];
        for r in 0..bs {
            for c in 0..bs {
                let sr = (r as i64 - dy).rem_euclid(bs as i64) as usize;
                let sc = (c as i64 - dx).rem_euclid(bs as i64) as usize;
                prod[r * bs + c] = block[r * bs + c] * block[sr * bs + sc];
            }
        }
        let a = aggd(&prod, r_gam);
        let mean = (a[3] - a[2]) * cephes_gamma(2.0 / a[0]) / cephes_gamma(1.0 / a[0]);
        out[o] = a[0];
        out[o + 1] = mean;
        out[o + 2] = a[2];
        out[o + 3] = a[3];
        o += 4;
    }
}

/// `np.nanmean(axis=0)` over a row-major `n × cols` matrix: per column,
/// NaN→0 then pairwise f64 sum over rows, `tot/cnt` via true_divide
/// (all-NaN column → NaN).
fn nanmean_cols(m: &[f64], n: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols];
    let mut col = Vec::with_capacity(n);
    for c in 0..cols {
        col.clear();
        let mut cnt = 0usize;
        for r in 0..n {
            let v = m[r * cols + c];
            if v.is_nan() {
                col.push(0.0);
            } else {
                col.push(v);
                cnt += 1;
            }
        }
        out[c] = if cnt == 0 {
            f64::NAN
        } else {
            pairwise_sum_f64(&col) / cnt as f64
        };
    }
    out
}

/// `np.cov(m[~isnan(m).any(1)], rowvar=False)` — ddof = 1. Fewer than two
/// clean rows yields an all-NaN covariance, as numpy produces. numpy computes
/// the column means with f64 pairwise summation and scales the product by
/// `true_divide(1, k−1)` — a reciprocal multiply, not a division.
fn cov_drop_nan(m: &[f64], n: usize, cols: usize) -> Vec<f64> {
    let keep: Vec<usize> = (0..n)
        .filter(|&r| (0..cols).all(|c| m[r * cols + c].is_finite()))
        .collect();
    let k = keep.len();
    let mut cov = vec![f64::NAN; cols * cols];
    if k < 2 {
        return cov;
    }
    let mut mu = vec![0.0f64; cols];
    let mut col = Vec::with_capacity(k);
    for (c, mu_c) in mu.iter_mut().enumerate() {
        col.clear();
        for &r in &keep {
            col.push(m[r * cols + c]);
        }
        *mu_c = pairwise_sum_f64(&col) / k as f64;
    }
    let fact = 1.0 / (k - 1) as f64;
    for i in 0..cols {
        for j in 0..=i {
            let mut s = 0.0f64;
            for &r in &keep {
                s += (m[r * cols + i] - mu[i]) * (m[r * cols + j] - mu[j]);
            }
            cov[i * cols + j] = s * fact;
            cov[j * cols + i] = cov[i * cols + j];
        }
    }
    cov
}

/// `np.linalg.pinv` of a symmetric `N×N` matrix (row-major) via cyclic
/// Jacobi eigendecomposition. Singular values are |eigenvalues|; numpy 2.x's
/// default cutoff `rcond = 1e-15·σ_max` applies — matching
/// `pinv((cov_pris + cov_dist)/2)`.
fn pinv_sym(a: &[f64; NFEAT * NFEAT]) -> [f64; NFEAT * NFEAT] {
    const N: usize = NFEAT;
    let mut m = *a;
    let mut v = [0.0f64; N * N];
    for i in 0..N {
        v[i * N + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0f64;
        for p in 0..N {
            for q in (p + 1)..N {
                off = off.max(m[p * N + q].abs());
            }
        }
        if off <= 1e-15 {
            break;
        }
        for p in 0..N {
            for q in (p + 1)..N {
                let apq = m[p * N + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (m[q * N + q] - m[p * N + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..N {
                    let mkp = m[k * N + p];
                    let mkq = m[k * N + q];
                    m[k * N + p] = c * mkp - s * mkq;
                    m[k * N + q] = s * mkp + c * mkq;
                }
                for k in 0..N {
                    let mpk = m[p * N + k];
                    let mqk = m[q * N + k];
                    m[p * N + k] = c * mpk - s * mqk;
                    m[q * N + k] = s * mpk + c * mqk;
                }
                for k in 0..N {
                    let vkp = v[k * N + p];
                    let vkq = v[k * N + q];
                    v[k * N + p] = c * vkp - s * vkq;
                    v[k * N + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let smax = (0..N).map(|i| m[i * N + i].abs()).fold(0.0f64, f64::max);
    let cutoff = 1e-15 * smax;
    let mut out = [0.0f64; N * N];
    for k in 0..N {
        let l = m[k * N + k];
        if !l.is_finite() || l.abs() <= cutoff {
            continue;
        }
        let inv = 1.0 / l;
        for i in 0..N {
            let vik = v[i * N + k];
            for j in 0..N {
                out[i * N + j] += vik * v[j * N + k] * inv;
            }
        }
    }
    out
}
