// Port of piq 0.8.0's `piq/brisque.py`
// (https://github.com/photosynthesis-team/piq, Apache-2.0,
// photosynthesis-team; notice kept in `LICENSE-piq`), itself a PyTorch
// port of LIVE's MATLAB `brisque.m`. The embedded SVR model in
// `params.bin` is LIVE's released `allmodel` — piq's
// `brisque_svm_weights.pt` is bit-identical to it (verified); the LIVE
// redistribution notice is kept verbatim in `LICENSE-LIVE-brisque`.
//
// Reference:
//   A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image Quality
//   Assessment in the Spatial Domain", IEEE Transactions on Image
//   Processing, vol. 21, no. 12, pp. 4695-4708, Dec. 2012.
//   doi:10.1109/TIP.2012.2214050

//! BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator.
//!
//! A pure-Rust port of piq 0.8.0's `brisque()`. MSCN normalization of
//! luminance at two scales (the second scale is a MATLAB-compatible
//! reflect-padded bicubic `imresize(0.5)`), GGD/AGGD features, SVM-scale
//! to `[-1,1]` via LIVE's `allrange`, epsilon-SVR predict against LIVE's
//! `allmodel` — higher means worse
//! ([`ScoreKind::Distortion`](za::ScoreKind::Distortion)).
//!
//! Numerical conventions follow the oracle (`piq.brisque(x, data_range=255)`,
//! `torch.set_num_threads(1)`) within the fixed JND/1000 score gate:
//!
//! - RGB8 → gray is piq's `rgb2yiq` Y channel rounded ties-to-even:
//!   `round(fma(B,.114, f32(.299R + .587G)))`, verified bitwise over all
//!   2^24 u8 triples. GRAY8 input is the plane verbatim.
//! - The 7×7 Gaussian `mu`/`std` convolutions are zero-padded sequential
//!   row-major FMA chains (torch conv2d's order).
//! - Map reductions use deterministic f64 fixed-lane accumulators (element
//!   `i` → lane `i & 7`, identical order on every backend); the resize dots
//!   keep the balanced pairwise f32 tap tree, evaluated lazily.
//! - Every `lgamma`/`exp` table in the pipeline is a fixed function of
//!   the 9801-entry `gam` grid, so the oracle-computed values are
//!   embedded verbatim (`R_GGD`, `R_AGGD`, `ETA_FACTOR`, `KERNEL49` in
//!   `params.bin`, decoded by `params.rs`). Integer squares and cubes use multiplication, as
//!   torch does, including left-associated `(x*x)*x`.
//! - The RBF exponential uses permissively licensed `libm::expf`.
//!   It is measured against torch's CPU exponential within the fixed JND/1000
//!   score gate; bitwise equality of final scores is not promised.
//! - `torch.arange(0.2, 10.001, 0.001)` reproduces ATen's per-8-block
//!   f64-base vectorized fill plus the f64 scalar tail.
//!
//! Deliberate divergences: inputs whose GGD sigma is ~0 (flat), whose
//! AGGD product maps are single-sign, or whose scaled features go
//! non-finite — all of which make the oracle raise — report
//! [`ErrorKind::Degenerate`](za::ErrorKind::Degenerate) instead.
//! `min(w,h)<8` (unsupported two-scale geometry) reports
//! [`ErrorKind::TooSmall`](za::ErrorKind::TooSmall).

#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use za::zenpixels::PixelDescriptor;
use zenmetrics_abstractions as za;

mod kernels;
mod params;
mod summation;

use params::{ETA_FACTOR, KERNEL49, R_AGGD, R_GGD, RHO, SV, SV_COEF};
use summation::{LaneSums8, inner_sum};

/// Output revision of [`Brisque`] on the generic surface; bump when any
/// output can change.
const SCORE_REVISION: u32 = 4;

/// `gam` grid count: `torch.arange(0.2, 10.001, 0.001)`.
const NGAM: usize = 9801;

/// `torch.arange`'s vectorized-loop boundary: full 16-element blocks take
/// the `Vec::arange` path, the remainder the f64 scalar tail (verified
/// bitwise against torch 2.14).
const ARANGE_VEC_BOUND: usize = NGAM - NGAM % 16;

/// piq's `_scale_features` ranges — verbatim from piq 0.8.0 (itself
/// LIVE's `allrange`), parsed to f32 the way `torch.tensor(literals,
/// dtype=f32)` does.
const FEATURE_RANGES: [[f32; 2]; 36] = [
    [0.338, 10.0],
    [0.017204, 0.806612],
    [0.236, 1.642],
    [-0.123884, 0.20293],
    [0.000155, 0.712298],
    [0.001122, 0.470257],
    [0.244, 1.641],
    [-0.123586, 0.179083],
    [0.000152, 0.710456],
    [0.000975, 0.470984],
    [0.249, 1.555],
    [-0.135687, 0.100858],
    [0.000174, 0.684173],
    [0.000913, 0.534174],
    [0.258, 1.561],
    [-0.143408, 0.100486],
    [0.000179, 0.685696],
    [0.000888, 0.536508],
    [0.471, 3.264],
    [0.012809, 0.703171],
    [0.218, 1.046],
    [-0.094876, 0.187459],
    [1.5e-005, 0.442057],
    [0.001272, 0.40803],
    [0.222, 1.042],
    [-0.115772, 0.162604],
    [1.6e-005, 0.444362],
    [0.001374, 0.40243],
    [0.227, 0.996],
    [-0.117188, 0.09832299999999999],
    [3e-005, 0.531903],
    [0.001122, 0.369589],
    [0.228, 0.99],
    [-0.12243, 0.098658],
    [2.8e-005, 0.530092],
    [0.001118, 0.370399],
];

#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::fmaf(a, b, c)
    }
}

#[inline(always)]
fn sqrt32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(x)
    }
}

#[inline(always)]
fn ceil64(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.ceil()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::ceil(x)
    }
}

#[inline(always)]
fn floor32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floorf(x)
    }
}

#[inline(always)]
fn rint32(x: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        x.round_ties_even()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::rintf(x)
    }
}

/// `torch.arange(0.2, 10.001, 0.001)` bitwise (see module docs).
fn gam_grid() -> Vec<f32> {
    let mut g = Vec::with_capacity(NGAM);
    for i in 0..NGAM {
        let v = if i < ARANGE_VEC_BOUND {
            // Vec::arange: per-8-block f32 base (f64-computed), lanes
            // f64(base) + lane·f64(step) → f32.
            let b = (0.2f64 + 0.001f64 * (i & !7) as f64) as f32;
            (b as f64 + (i & 7) as f64 * 0.001f64) as f32
        } else {
            (0.2f64 + 0.001f64 * i as f64) as f32
        };
        g.push(v);
    }
    g
}

/// Both MSCN 7×7 convs (x and x²), sequential row-major FMA chains
/// (torch conv2d). SIMD tiers keep the identical per-output FMA order on
/// a 3-px zero-padded copy (padded zeros contribute the same `k·0`
/// products the scalar bounds check selected), so every tier is
/// bitwise-identical to the scalar chain.
fn conv7_both(x: &[f32], w: usize, h: usize, k: &[f32; 49]) -> (Vec<f32>, Vec<f32>) {
    let (xp, pw) = kernels::pad3(x, w, h);
    let mut mu = vec![0f32; w * h];
    let mut std_c = vec![0f32; w * h];
    kernels::conv7_pair(&xp, pw, w, k, &mut mu, &mut std_c);
    (mu, std_c)
}

/// `_ggd_parameters` → (gam index, sigma). `x` is the MSCN map
/// (row-major w×h).
// `!(x > y)` is deliberate: the oracle's checks must also fire on NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn ggd_params(x: &[f32]) -> Result<(usize, f32), ()> {
    // One pass: E[x²] and E[|x|] in f64 fixed-lane sums — the same
    // quantities the f32 pairwise `inner_mean` produced, near-exact.
    let mut sx2 = LaneSums8::new();
    let mut sabs = LaneSums8::new();
    let n_e = x.len();
    let mut i = 0usize;
    while i + 8 <= n_e {
        let mut b = [0f64; 8];
        let mut m = [0f64; 8];
        for j in 0..8 {
            let v = x[i + j] as f64;
            b[j] = v * v;
            m[j] = v.abs();
        }
        for j in 0..8 {
            sx2.0[j] += b[j];
            sabs.0[j] += m[j];
        }
        i += 8;
    }
    while i < n_e {
        let v = x[i] as f64;
        sx2.add(i, v * v);
        sabs.add(i, v.abs());
        i += 1;
    }
    let n = x.len() as f64;
    let sigma_sq = (sx2.total() / n) as f32;
    let sigma = sqrt32(sigma_sq);
    // torch.isclose(sigma, 0) with default atol=1e-8 — oracle asserts
    // 'non zero variance' when this holds.
    if !(sigma.abs() > 1e-8 + 1e-5 * 0.0) {
        return Err(());
    }
    let e = (sabs.total() / n) as f32;
    let (rho, best) = ggd_ratio_argmin(sigma_sq, e);
    if !rho.is_finite() {
        return Err(());
    }
    Ok((best, sigma))
}

/// `_aggd_parameters` on `nr · roll(nr, dy, dx)` — product values are
/// computed inside the single stats pass (no rolled/prod copies), each
/// sum accumulated in f64 fixed lanes. Same quantities, same order on
/// every backend.
// `!(x > y)` is deliberate: the oracle's checks must also fire on NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn aggd_params(nr: &[f32], w: usize, h: usize, dy: i64, dx: i64) -> Result<(usize, f32, f32), ()> {
    let mut sl = LaneSums8::new();
    let mut sr = LaneSums8::new();
    let mut sabs = LaneSums8::new();
    let mut sx2 = LaneSums8::new();
    let mut count_l = 0usize;
    let mut count_r = 0usize;
    // sx = (c − dx) mod w = (c + base_off) mod w — one conditional wrap.
    let base_off = (w as i64 - dx).rem_euclid(w as i64) as usize;
    for r in 0..h {
        let sy = (r as i64 - dy).rem_euclid(h as i64) as usize;
        let row = &nr[r * w..r * w + w];
        let sh = &nr[sy * w..sy * w + w];
        let i0 = r * w;
        let mut c = 0usize;
        // head: until absolute index i is 8-aligned
        while c < w && (i0 + c) & 7 != 0 {
            let mut s = c + base_off;
            if s >= w {
                s -= w;
            }
            accum(
                &mut sl,
                &mut sr,
                &mut sabs,
                &mut sx2,
                &mut count_l,
                &mut count_r,
                i0 + c,
                row[c] * sh[s],
            );
            c += 1;
        }
        // aligned 8-blocks: element i0+c+j → accumulator lane j
        while c + 8 <= w {
            let mut v = [0f32; 8];
            for j in 0..8 {
                let mut s = c + j + base_off;
                if s >= w {
                    s -= w;
                }
                v[j] = row[c + j] * sh[s];
            }
            accum8(
                &mut sl,
                &mut sr,
                &mut sabs,
                &mut sx2,
                &mut count_l,
                &mut count_r,
                &v,
            );
            c += 8;
        }
        while c < w {
            let mut s = c + base_off;
            if s >= w {
                s -= w;
            }
            accum(
                &mut sl,
                &mut sr,
                &mut sabs,
                &mut sx2,
                &mut count_l,
                &mut count_r,
                i0 + c,
                row[c] * sh[s],
            );
            c += 1;
        }
    }
    let cl = count_l as f32;
    let cr = count_r as f32;
    if !(cl > 0.0) || !(cr > 0.0) {
        return Err(());
    }
    // (x*mask).pow(2).sum / count → sqrt
    let left_sigma = sqrt32((sl.total() / count_l as f64) as f32);
    let right_sigma = sqrt32((sr.total() / count_r as f64) as f32);
    if !(left_sigma > 0.0) || !(right_sigma > 0.0) {
        return Err(());
    }
    let n = (w * h) as f64;
    let absm = (sabs.total() / n) as f32;
    let x2m = (sx2.total() / n) as f32;
    let (ratio, best) = aggd_ratio_argmin(left_sigma, right_sigma, absm, x2m);
    if !ratio.is_finite() {
        return Err(());
    }
    Ok((best, left_sigma, right_sigma))
}

/// One product element of the AGGD pass (unaligned head/tail): sign-masked
/// squared sums (negatives → sl, positives → sr — the `if m {v} else {0}`
/// masked squares contribute nothing, including no `+0.0` reassociation),
/// |v|, v² and the side counts.
#[inline(always)]
fn accum(
    sl: &mut LaneSums8,
    sr: &mut LaneSums8,
    sabs: &mut LaneSums8,
    sx2: &mut LaneSums8,
    count_l: &mut usize,
    count_r: &mut usize,
    i: usize,
    v: f32,
) {
    let v64 = v as f64;
    if v < 0.0 {
        sl.add(i, v64 * v64);
        *count_l += 1;
    } else if v > 0.0 {
        sr.add(i, v64 * v64);
        *count_r += 1;
    }
    sabs.add(i, v64.abs());
    sx2.add(i, v64 * v64);
}

/// Eight aligned product elements — v[j] maps to lane j. Masked sides
/// contribute `+0.0` (an exact no-op on non-negative accumulators), so the
/// unconditional adds are branchless and vectorize cleanly.
#[inline(always)]
fn accum8(
    sl: &mut LaneSums8,
    sr: &mut LaneSums8,
    sabs: &mut LaneSums8,
    sx2: &mut LaneSums8,
    count_l: &mut usize,
    count_r: &mut usize,
    v: &[f32; 8],
) {
    let mut neg = [0f64; 8];
    let mut pos = [0f64; 8];
    let mut ab = [0f64; 8];
    let mut sq = [0f64; 8];
    for j in 0..8 {
        let x = v[j] as f64;
        let m = if x < 0.0 { x } else { 0.0 };
        neg[j] = m * m;
        let p = if x > 0.0 { x } else { 0.0 };
        pos[j] = p * p;
        ab[j] = x.abs();
        sq[j] = x * x;
    }
    for j in 0..8 {
        sl.0[j] += neg[j];
        sr.0[j] += pos[j];
        sabs.0[j] += ab[j];
        sx2.0[j] += sq[j];
        *count_l += (v[j] < 0.0) as usize;
        *count_r += (v[j] > 0.0) as usize;
    }
}

fn first_argmin(table: &[f32], value: f32) -> usize {
    let mut best = usize::MAX;
    let mut distance = f32::INFINITY;
    for (i, &t) in table.iter().enumerate() {
        let d = (value - t).abs();
        if d < distance {
            distance = d;
            best = i;
        }
    }
    best
}

// Integer powers follow torch multiplication, including left-associated cubes.
pub(crate) fn ggd_ratio_argmin(sigma_sq: f32, e: f32) -> (f32, usize) {
    let ratio = sigma_sq / (e * e);
    (ratio, first_argmin(&R_GGD, ratio))
}

pub(crate) fn aggd_ratio_argmin(left: f32, right: f32, absm: f32, x2m: f32) -> (f32, usize) {
    let gh = left / right;
    let ro_hat = (absm * absm) / x2m;
    let gh2 = gh * gh;
    let gh3 = gh2 * gh;
    let ratio = ((ro_hat * (gh3 + 1.0)) * (gh + 1.0)) / ((gh2 + 1.0) * (gh2 + 1.0));
    (ratio, first_argmin(&R_AGGD, ratio))
}

/// `_natural_scene_statistics` → 18 features, or Err on oracle asserts.
fn nss(x: &[f32], w: usize, h: usize, gam: &[f32]) -> Result<[f32; 18], ()> {
    let (mu, std_c) = conv7_both(x, w, h, &KERNEL49);
    let mut feats = [0f32; 18];
    let nr = kernels::normalize(x, &mu, &std_c, w * h);
    let (ai, sigma) = ggd_params(&nr)?;
    feats[0] = gam[ai];
    feats[1] = sigma * sigma;
    let shifts = [(0i64, 1i64), (1, 0), (1, 1), (-1, 1)];
    for (s, &(dy, dx)) in shifts.iter().enumerate() {
        let (ai, ls, rs) = aggd_params(&nr, w, h, dy, dx)?;
        // eta = (rs - ls)·exp(lg(2/α) − (lg(1/α)+lg(3/α))/2); the exp
        // factor is a fixed function of the gam-grid index.
        let eta = (rs - ls) * ETA_FACTOR[ai];
        feats[2 + 4 * s] = gam[ai];
        feats[3 + 4 * s] = eta;
        feats[4 + 4 * s] = ls * ls;
        feats[5 + 4 * s] = rs * rs;
    }
    Ok(feats)
}

/// `cubic_contribution` per tap (a = -0.5), already masked-selected.
fn cubic_weight(b: f32) -> f32 {
    let ax = b.abs();
    let ax2 = ax * ax;
    let ax3 = ax * ax2;
    let cont01 = (1.5 * ax3 - 2.5 * ax2 + 1.0) * if ax <= 1.0 { 1.0 } else { 0.0 };
    let cont12 = (((-0.5 * ax3) - (-2.5 * ax2)) + (-4.0 * ax) - (-2.0))
        * if ax > 1.0 && ax <= 2.0 { 1.0 } else { 0.0 };
    cont01 + cont12
}

/// `resize_1d` along one axis of a row-major image (piq's MATLAB-style
/// reflect padding, cubic kernel, antialiasing).
#[allow(clippy::too_many_arguments)]
fn resize_1d(
    x: &[f32],
    w: usize,
    h: usize,
    vertical: bool,
    out_size: usize,
) -> (Vec<f32>, usize, usize) {
    let in_size = if vertical { h } else { w };
    let scale = out_size as f64 / in_size as f64;
    // kernel_size=4 cubic; antialiased when downscaling.
    let mut ks = 4usize;
    let aaf = if scale < 1.0 {
        ks = ceil64(4.0 / scale) as usize;
        scale
    } else {
        1.0
    };
    ks += 2;
    let aaf32 = aaf as f32;
    let scale32 = scale as f32;

    // pos = (linspace(0, size-1) + 0.5)/scale - 0.5 — exact-int linspace.
    let mut base = Vec::with_capacity(out_size);
    let mut dist = Vec::with_capacity(out_size);
    for i in 0..out_size {
        let p = (i as f32 + 0.5) / scale32 - 0.5;
        let b = floor32(p) as i64 - (ks as i64 / 2) + 1;
        base.push(b);
        dist.push(p - b as f32);
    }
    // weight[idx][i] = cubic_contribution((dist[i]-idx)*aaf), normalized
    // per output column by an outer (dim-0) sum.
    let mut weight = vec![0f32; ks * out_size];
    for i in 0..out_size {
        for idx in 0..ks {
            let bpos = (dist[i] - idx as f32) * aaf32;
            weight[idx * out_size + i] = cubic_weight(bpos);
        }
    }
    // outer_sum's column tree, evaluated lazily per output column.
    for i in 0..out_size {
        let colsum = kernels::tap_tree(0, ks, &|idx| weight[idx * out_size + i]);
        for idx in 0..ks {
            weight[idx * out_size + i] /= colsum;
        }
    }
    // get_padding: shift base so taps fit, pad the remainder.
    let r_min = base.iter().copied().min().unwrap();
    let r_max = base.iter().copied().max().unwrap() + ks as i64 - 1;
    let pad_pre = if r_min <= 0 { (-r_min) as usize } else { 0 };
    for b in base.iter_mut() {
        *b += pad_pre as i64;
    }
    let pad_post = if r_max >= in_size as i64 {
        (r_max - in_size as i64 + 1) as usize
    } else {
        0
    };

    // reflect pad (MATLAB: edge element used twice) then gather+dot.
    let (out_w, out_h, pw, ph) = if vertical {
        (w, out_size, w, h + pad_pre + pad_post)
    } else {
        (out_size, h, w + pad_pre + pad_post, h)
    };
    let mut xpad = vec![0f32; pw * ph];
    if vertical {
        for r in 0..ph {
            let src = if r < pad_pre {
                pad_pre - 1 - r
            } else if r >= pad_pre + h {
                2 * h + pad_pre - 1 - r
            } else {
                r - pad_pre
            };
            xpad[r * pw..r * pw + pw].copy_from_slice(&x[src * w..src * w + w]);
        }
    } else {
        // Interior columns copy straight; only the pad_pre left and
        // pad_post right edge columns reflect (same values, less work).
        for r in 0..ph {
            let src = &x[r * w..r * w + w];
            let dst = &mut xpad[r * pw..r * pw + pw];
            dst[pad_pre..pad_pre + w].copy_from_slice(src);
            for c in 0..pad_pre {
                dst[c] = src[pad_pre - 1 - c];
            }
            for c in 0..pad_post {
                dst[pad_pre + w + c] = src[w - 1 - c];
            }
        }
    }

    // Each output is `outer_sum`'s column tree over tap products, evaluated
    // lazily (no prod materialization; identical combine order).
    let n_out = out_w * out_h;
    let mut out = vec![0f32; n_out];
    if vertical {
        for i in 0..out_size {
            let row = &mut out[i * w..i * w + w];
            kernels::vres_row_dispatch(&xpad, w, base[i] as usize, &weight, out_size, ks, i, row);
        }
    } else {
        for r in 0..h {
            for i in 0..out_size {
                let leaf = |idx: usize| {
                    xpad[r * pw + (base[i] + idx as i64) as usize] * weight[idx * out_size + i]
                };
                // ks=10 tree unrolled — same `tap_tree` combine order.
                out[r * out_size + i] = if ks == 10 {
                    ((leaf(0) + leaf(1)) + (leaf(2) + (leaf(3) + leaf(4))))
                        + ((leaf(5) + leaf(6)) + (leaf(7) + (leaf(8) + leaf(9))))
                } else {
                    kernels::tap_tree(0, ks, &leaf)
                };
            }
        }
    }
    (out, out_w, out_h)
}

/// `imresize(x, sizes=(h/2, w/2))` — height pass then width pass.
fn imresize_half(x: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (h2, w2) = (h / 2, w / 2);
    let (tmp, w_a, h2) = resize_1d(x, w, h, true, h2);
    let (out, w2, h_b) = resize_1d(&tmp, w_a, h2, false, w2);
    (out, w2, h_b)
}

fn gray(image: &za::Image<'_>) -> za::MetricResult<(Vec<f32>, usize, usize)> {
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::RGB8_SRGB) {
        return Ok((kernels::luma_plane(px, w, h, stride), w, h));
    }
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::GRAY8) {
        let mut y = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                y.push(px[r * stride + c] as f32);
            }
        }
        return Ok((y, w, h));
    }
    za::fail(za::unsupported(
        &image.format(),
        "packed RGB8_SRGB or GRAY8",
    ))
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

/// The solved torch `(1,774)@(774,)` dot: 8-lane FMA accumulators
/// (`i mod 8`) reduced by `((p0+p1)+(p4+p5)) + ((p2+p3)+(p6+p7))`.
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        acc[i & 7] = fma(x, y, acc[i & 7]);
    }
    ((acc[0] + acc[1]) + (acc[4] + acc[5])) + ((acc[2] + acc[3]) + (acc[6] + acc[7]))
}

/// Estimated just-noticeable difference of the `brisque` score, measured
/// on the KonJND-1k corpus: `zenmetrics batch --metric brisque` over all
/// 1,008 pairs (`PairedNoReference` gives `brisque` and `brisque_ref` per
/// pair), then JND = median over pairs of `|brisque − brisque_ref|` =
/// 15.138428 (JPEG 12.735596, BPG 17.310303). See
/// `benchmarks/nr_classic_2026-10-15/jnd.md` for the full distribution.
pub const JND: f64 = 15.138428;

/// BRISQUE scorer. `RGB8_SRGB` and `GRAY8`. Headline = `brisque`
/// ([`ScoreKind::Distortion`](za::ScoreKind::Distortion): higher is worse).
#[derive(Debug, Clone, Copy, Default)]
pub struct Brisque;

impl Brisque {
    /// A new BRISQUE scorer (oracle defaults: kernel 7, sigma 7/6).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Debug pipeline access: `(feats36, scaled36, dist774, dist_in, score)`
/// from a gray f32 plane — mirrors the oracle's intermediate tensors.
#[doc(hidden)]
#[allow(dead_code)] // Internal review examples compile this same implementation.
pub(crate) fn pipeline_debug(
    y: &[f32],
    w: usize,
    h: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, f32) {
    let gam = gam_grid();
    let mut feats = [0f32; 36];
    let mut x = y.to_vec();
    let (mut w, mut h) = (w, h);
    for scale in 0..2 {
        feats[scale * 18..scale * 18 + 18].copy_from_slice(&nss(&x, w, h, &gam).unwrap());
        let (nx, nw, nh) = imresize_half(&x, w, h);
        x = nx;
        w = nw;
        h = nh;
    }
    let mut scaled = vec![0f32; 36];
    for i in 0..36 {
        let (lo, hi) = (FEATURE_RANGES[i][0], FEATURE_RANGES[i][1]);
        scaled[i] = -1.0 + (2.0 * (feats[i] - lo)) / (hi - lo);
    }
    // The broadcast `sc - sv_t` output materializes with strides
    // (36,1,36) — dim2-major — so `sum(dim=1)` is a *contiguous* inner
    // reduce over each 36-element SV row.
    let mut dist_in = vec![0f32; 774 * 36];
    for (j, sv) in SV.iter().enumerate() {
        for (d, &s) in sv.iter().enumerate() {
            let dd = scaled[d] - s;
            dist_in[j * 36 + d] = dd * dd;
        }
    }
    let dist: Vec<f32> = (0..774)
        .map(|j| inner_sum(&dist_in[j * 36..j * 36 + 36]))
        .collect();
    let kf: Vec<f32> = dist.iter().map(|&d| libm::expf(-(d * 0.05))).collect();
    let score = dot8(&kf, &SV_COEF) - RHO;
    (feats.to_vec(), scaled, dist, dist_in, score)
}

fn degenerate(msg: &'static str) -> za::MetricResult<Box<dyn za::Evaluation>> {
    za::fail(
        za::MetricError::new(za::ErrorKind::Degenerate)
            .with_metric("brisque")
            .with_message(alloc::format!("BRISQUE degenerate input: {msg}")),
    )
}

impl za::NoReferenceMetric for Brisque {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("brisque", za::implementation!(SCORE_REVISION))
            .with_reference("piq 0.8.0 brisque.py + LIVE allmodel")
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] =
            &[za::ScoreSpec::new("brisque", za::ScoreKind::Distortion)];
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
        opts.refuse_map("brisque")?;
        let (x, w, h) = gray(&image)?;
        if w < 8 || h < 8 {
            return za::fail(
                za::MetricError::new(za::ErrorKind::TooSmall)
                    .with_metric("brisque")
                    .with_message(alloc::format!(
                        "image too small for BRISQUE: {w}x{h} (minimum 8 pixels on each axis)"
                    )),
            );
        }
        let gam = gam_grid();

        let mut x = x;
        let (mut w, mut h) = (w, h);
        let mut feats = [0f32; 36];
        for scale in 0..2 {
            let f = match nss(&x, w, h, &gam) {
                Ok(f) => f,
                Err(()) => {
                    return degenerate(match scale {
                        0 => "scale-1 statistics",
                        _ => "scale-2 statistics",
                    });
                }
            };
            feats[scale * 18..scale * 18 + 18].copy_from_slice(&f);
            let (nx, nw, nh) = imresize_half(&x, w, h);
            x = nx;
            w = nw;
            h = nh;
        }
        // scale to [-1,1]: lower + (upper-lower)·(f-lo)/(hi-lo)
        // tree = add(-1, div(mul(2, sub(f,lo)), sub(hi,lo))).
        let mut scaled = [0f32; 36];
        for i in 0..36 {
            let (lo, hi) = (FEATURE_RANGES[i][0], FEATURE_RANGES[i][1]);
            scaled[i] = -1.0 + (2.0 * (feats[i] - lo)) / (hi - lo);
        }
        if !scaled.iter().all(|v| v.is_finite()) {
            return degenerate("non-finite scaled features");
        }
        // dist[j] = Σ_d (f[d]-SV[j][d])² — the broadcast output
        // materializes dim2-major, making this a *contiguous* inner
        // reduce over each 36-element SV row.
        let mut dist_in = vec![0f32; 774 * 36];
        for (j, sv) in SV.iter().enumerate() {
            for (d, &s) in sv.iter().enumerate() {
                let dd = scaled[d] - s;
                dist_in[j * 36 + d] = dd * dd;
            }
        }
        let dist: Vec<f32> = (0..774)
            .map(|j| inner_sum(&dist_in[j * 36..j * 36 + 36]))
            .collect();
        // kernel_features = exp(-dist*0.05), via permissive libm.
        let kf: Vec<f32> = dist.iter().map(|&d| libm::expf(-(d * 0.05))).collect();
        let score = dot8(&kf, &SV_COEF) - RHO;
        if !score.is_finite() {
            return degenerate("non-finite score");
        }
        Ok(Box::new(za::ScalarEval::single("brisque", score as f64)))
    }
}

#[cfg(test)]
mod pow_tests {
    use super::*;
    #[test]
    fn torch_ratio_and_argmin_pin() {
        // torch 2.14.1 CPU, case from the 400k deterministic sweep.
        // Generator: benchmarks/brisque_review_2026-10-15/prepare.py.
        let a = [
            1146930109, 1067902915, 1143347281, 1116706703, 1090811246, 1121775981,
        ]
        .map(f32::from_bits);
        let (g, gi) = ggd_ratio_argmin(a[0], a[1]);
        let (r, ri) = aggd_ratio_argmin(a[2], a[3], a[4], a[5]);
        assert_eq!((g.to_bits(), gi), (1140971285, 0));
        assert_eq!((r.to_bits(), ri), (1059858852, 2463));
    }
}
