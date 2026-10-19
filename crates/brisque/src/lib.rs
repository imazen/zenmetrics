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
//! - Reductions use independently written balanced pairwise f32 summation.
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

mod params;
mod summation;

use params::{ETA_FACTOR, KERNEL49, R_AGGD, R_GGD, RHO, SV, SV_COEF};
use summation::{SUM_ABS, SUM_NEG2, SUM_POS2, SUM_SQ, inner_sum, outer_sum, quad_sums};

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

/// End column covered by [`conv7_core`]: 8-lane blocks start at `c = 3`
/// and step while `c + 11 <= w` (the rightmost tap reads `x[c + 10]`).
/// Returns 0 when no block fits — every pixel takes the scalar chain.
fn conv7_simd_col_end(w: usize) -> usize {
    if w >= 14 { 3 + 8 * ((w - 14) / 8 + 1) } else { 0 }
}

/// Interior of the 7×7 conv, 8 output columns per lane group.
///
/// Each lane runs the identical sequential 49-tap FMA chain as the
/// scalar path for its own output column — one fused rounding per tap
/// through `mul_add_portable` — so results are bit-identical on every
/// tier (the scalar tier emulates the same lanes). Rows outside
/// `[3, h-3)` and columns outside `[3, conv7_simd_col_end(w))` are left
/// for the scalar border code in [`conv7`].
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn conv7_core(
    token: Token,
    x: &[f32],
    w: usize,
    h: usize,
    k: &[f32; 49],
    out: &mut [f32],
) {
    let kv: [f32x8; 49] = core::array::from_fn(|t| f32x8::splat_t(token, k[t]));
    let zero = f32x8::zero_t(token);
    for r in 0..h {
        let mut c = 3;
        while c + 11 <= w {
            let mut acc = zero;
            let mut t = 0;
            for i in 0..7 {
                // `yy = r + i - 3` is uniform across lanes: out-of-range
                // rows contribute `k * 0` in the scalar chain — a zero
                // tap-vector keeps every lane's chain identical.
                let yy = r.wrapping_sub(3).wrapping_add(i);
                let row = if yy < h { &x[yy * w..] } else { &[] };
                for j in 0..7 {
                    let xv = if row.is_empty() {
                        zero
                    } else {
                        f32x8::from_slice_t(token, &row[c - 3 + j..c - 3 + j + 8])
                    };
                    acc = kv[t].mul_add_portable(xv, acc);
                    t += 1;
                }
            }
            out[r * w + c..r * w + c + 8].copy_from_slice(&acc.to_array());
            c += 8;
        }
    }
}

/// 7×7 zero-pad conv — sequential row-major FMA chain (torch conv2d).
/// Lane groups cover `[3, conv7_simd_col_end(w))` on every row (out-of-
/// range rows contribute zero tap-vectors, identical to the scalar
/// chain's `k*0`); only the left/right edge columns keep the scalar
/// chain.
fn conv7(x: &[f32], w: usize, h: usize, k: &[f32; 49]) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    let col_end = conv7_simd_col_end(w);
    if col_end > 0 {
        archmage::attuned!(
            conv7_core(x, w, h, k, &mut out),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
    }
    for r in 0..h {
        for c in (0..3.min(w)).chain(col_end.max(3).min(w)..w) {
            let mut acc = 0f32;
            for i in 0..7i64 {
                let yy = r as i64 + i - 3;
                for j in 0..7i64 {
                    let xx = c as i64 + j - 3;
                    let xv = if yy >= 0 && yy < h as i64 && xx >= 0 && xx < w as i64 {
                        x[yy as usize * w + xx as usize]
                    } else {
                        0.0
                    };
                    acc = fma(k[(i * 7 + j) as usize], xv, acc);
                }
            }
            out[r * w + c] = acc;
        }
    }
    out
}

/// `_ggd_parameters` → (gam index, sigma). `x` is the MSCN map
/// (row-major w×h).
// `!(x > y)` is deliberate: the oracle's checks must also fire on NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn ggd_params(x: &[f32]) -> Result<(usize, f32), ()> {
    // One tree walk covers the |x| and x² reductions (lanes 2/3); each
    // lane's association equals the scalar `inner_mean` tree.
    let q = quad_sums(x);
    let sigma_sq = q[SUM_SQ] / x.len() as f32;
    let sigma = sqrt32(sigma_sq);
    // torch.isclose(sigma, 0) with default atol=1e-8 — oracle asserts
    // 'non zero variance' when this holds.
    if !(sigma.abs() > 1e-8 + 1e-5 * 0.0) {
        return Err(());
    }
    let e = q[SUM_ABS] / x.len() as f32;
    let (rho, best) = ggd_ratio_argmin(sigma_sq, e);
    if !rho.is_finite() {
        return Err(());
    }
    Ok((best, sigma))
}

/// `_aggd_parameters` → (gam index, left_sigma, right_sigma) or Err on
/// the oracle's assert paths.
// `!(x > y)` is deliberate: the oracle's checks must also fire on NaN.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn aggd_params(x: &[f32]) -> Result<(usize, f32, f32), ()> {
    let n = x.len();
    let mut count_l = 0usize;
    let mut count_r = 0usize;
    for &v in x {
        if v < 0.0 {
            count_l += 1;
        } else if v > 0.0 {
            count_r += 1;
        }
    }
    let cl = count_l as f32;
    let cr = count_r as f32;
    if !(cl > 0.0) || !(cr > 0.0) {
        return Err(());
    }
    // (x*mask).pow(2).sum / count → sqrt. One tree walk covers all four
    // reductions — lane order [neg², pos², |x|, x²] — each lane's
    // association equals the scalar `inner_sum`/`inner_mean` tree over
    // its materialized array.
    let q = quad_sums(x);
    let n_f32 = n as f32;
    let left_sigma = sqrt32(q[SUM_NEG2] / cl);
    let right_sigma = sqrt32(q[SUM_POS2] / cr);
    if !(left_sigma > 0.0) || !(right_sigma > 0.0) {
        return Err(());
    }
    let absm = q[SUM_ABS] / n_f32;
    let x2m = q[SUM_SQ] / n_f32;
    let (ratio, best) = aggd_ratio_argmin(left_sigma, right_sigma, absm, x2m);
    if !ratio.is_finite() {
        return Err(());
    }
    Ok((best, left_sigma, right_sigma))
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

/// `torch.roll` shifts `(dy, dx)` on the last two dims (wrap-around).
/// Each output row is the source row `sy` rotated by `dx` — two
/// contiguous copies instead of per-element modulo indexing.
fn roll(x: &[f32], w: usize, h: usize, dy: i64, dx: i64) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    let dyu = dy.rem_euclid(h as i64) as usize;
    let dxu = dx.rem_euclid(w as i64) as usize;
    for r in 0..h {
        let sy = (r + h - dyu) % h;
        let src = &x[sy * w..sy * w + w];
        let dst = &mut out[r * w..r * w + w];
        dst[..dxu].copy_from_slice(&src[w - dxu..]);
        dst[dxu..].copy_from_slice(&src[..w - dxu]);
    }
    out
}

/// MSCN normalize: `nr = (x - mu) / (sqrt(|std_c - mu²|) + 1)`.
/// Every op is elementwise and correctly rounded (sub/mul/abs/sqrt/add/
/// div are each IEEE-exact), so lane order is irrelevant and the result
/// is bit-identical to the scalar loop.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn mscn_core(
    token: Token,
    x: &[f32],
    mu: &[f32],
    std_c: &[f32],
    nr: &mut [f32],
) {
    let one = f32x8::splat_t(token, 1.0);
    let n = x.len();
    let full = n & !7;
    for i in (0..full).step_by(8) {
        let xv = f32x8::from_slice_t(token, &x[i..i + 8]);
        let mv = f32x8::from_slice_t(token, &mu[i..i + 8]);
        let cv = f32x8::from_slice_t(token, &std_c[i..i + 8]);
        let std = (cv - mv * mv).abs().sqrt();
        let v = (xv - mv) / (std + one);
        nr[i..i + 8].copy_from_slice(&v.to_array());
    }
    for i in full..n {
        let std = sqrt32((std_c[i] - mu[i] * mu[i]).abs());
        nr[i] = (x[i] - mu[i]) / (std + 1.0);
    }
}

/// `_natural_scene_statistics` → 18 features, or Err on oracle asserts.
fn nss(x: &[f32], w: usize, h: usize, gam: &[f32]) -> Result<[f32; 18], ()> {
    let mu = conv7(x, w, h, &KERNEL49);
    let x2: Vec<f32> = x.iter().map(|&v| v * v).collect();
    let std_c = conv7(&x2, w, h, &KERNEL49);
    let mut feats = [0f32; 18];
    let mut nr = vec![0f32; w * h];
    archmage::attuned!(
        mscn_core(x, &mu, &std_c, &mut nr),
        [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
    );
    let (ai, sigma) = ggd_params(&nr)?;
    feats[0] = gam[ai];
    feats[1] = sigma * sigma;
    let shifts = [(0i64, 1i64), (1, 0), (1, 1), (-1, 1)];
    for (s, &(dy, dx)) in shifts.iter().enumerate() {
        let sh = roll(&nr, w, h, dy, dx);
        let prod: Vec<f32> = nr.iter().zip(&sh).map(|(&a, &b)| a * b).collect();
        let (ai, ls, rs) = aggd_params(&prod)?;
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
    let colsum = outer_sum(&weight, ks, out_size);
    for idx in 0..ks {
        for i in 0..out_size {
            weight[idx * out_size + i] /= colsum[i];
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
        for r in 0..ph {
            // Interior: contiguous copy of the source row.
            xpad[r * pw + pad_pre..r * pw + pad_pre + w]
                .copy_from_slice(&x[r * w..r * w + w]);
            for c in 0..pad_pre {
                xpad[r * pw + c] = x[r * w + pad_pre - 1 - c];
            }
            for c in pad_pre + w..pw {
                xpad[r * pw + c] = x[r * w + 2 * w + pad_pre - 1 - c];
            }
        }
    }

    // sample*weight then sum(dim=1): outer-sum over ks taps per output.
    let n_out = out_w * out_h;
    let mut prod = vec![0f32; ks * n_out];
    if vertical {
        for i in 0..out_size {
            for idx in 0..ks {
                let row = (base[i] + idx as i64) as usize;
                for c in 0..w {
                    prod[idx * n_out + i * w + c] = xpad[row * w + c] * weight[idx * out_size + i];
                }
            }
        }
    } else {
        for r in 0..h {
            for i in 0..out_size {
                for idx in 0..ks {
                    prod[idx * n_out + r * out_size + i] =
                        xpad[r * pw + (base[i] + idx as i64) as usize] * weight[idx * out_size + i];
                }
            }
        }
    }
    let out = outer_sum(&prod, ks, n_out);
    // out is column-major over (ks, n_out) → out[j] is output j.
    let mut reshaped = vec![0f32; n_out];
    if vertical {
        for i in 0..out_size {
            for c in 0..w {
                reshaped[i * w + c] = out[i * w + c];
            }
        }
    } else {
        reshaped.copy_from_slice(&out);
    }
    (reshaped, out_w, out_h)
}

/// `imresize(x, sizes=(h/2, w/2))` — height pass then width pass.
fn imresize_half(x: &[f32], w: usize, h: usize) -> (Vec<f32>, usize, usize) {
    let (h2, w2) = (h / 2, w / 2);
    let (tmp, w_a, h2) = resize_1d(x, w, h, true, h2);
    let (out, w2, h_b) = resize_1d(&tmp, w_a, h2, false, w2);
    (out, w2, h_b)
}

/// piq `rgb2yiq` Y row: `round(fma(B,.114, f32(.299R + .587G)))`,
/// ties-to-even (bitwise-verified over all u8 triples).
#[inline]
fn to_y(r: u8, g: u8, b: u8) -> f32 {
    let m01 = r as f32 * 0.299 + g as f32 * 0.587;
    rint32(fma(b as f32, 0.114, m01))
}

/// piq `rgb2yiq` Y for one row of packed RGB8:
/// `round_ties_even(fma(B,.114, .299R + .587G))` — u8→f32 converts are
/// lane gathers, the arithmetic keeps the scalar op order (two muls,
/// one add, one fused fma, ties-even round) → bit-identical.
#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn gray_row_core(token: Token, row: &[u8], w: usize, y: &mut [f32]) {
    let c299 = f32x8::splat_t(token, 0.299);
    let c587 = f32x8::splat_t(token, 0.587);
    let c114 = f32x8::splat_t(token, 0.114);
    let full = w & !7;
    for c in (0..full).step_by(8) {
        let rv = f32x8::from_array_t(
            token,
            core::array::from_fn(|l| row[(c + l) * 3] as f32),
        );
        let gv = f32x8::from_array_t(
            token,
            core::array::from_fn(|l| row[(c + l) * 3 + 1] as f32),
        );
        let bv = f32x8::from_array_t(
            token,
            core::array::from_fn(|l| row[(c + l) * 3 + 2] as f32),
        );
        let v = c114.mul_add_portable(bv, c299 * rv + c587 * gv).round();
        y[c..c + 8].copy_from_slice(&v.to_array());
    }
    for c in full..w {
        y[c] = to_y(row[c * 3], row[c * 3 + 1], row[c * 3 + 2]);
    }
}

fn gray(image: &za::Image<'_>) -> za::MetricResult<(Vec<f32>, usize, usize)> {
    if let Some((px, w, h, stride)) = packed_as(image, PixelDescriptor::RGB8_SRGB) {
        let mut y = vec![0f32; w * h];
        for r in 0..h {
            let row = &px[r * stride..r * stride + w * 3];
            archmage::attuned!(
                gray_row_core(row, w, &mut y[r * w..r * w + w]),
                [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
            );
        }
        return Ok((y, w, h));
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
mod conv7_tests {
    use super::*;

    /// The original scalar implementation, kept as the semantic
    /// reference: the interior may now be filled through lane groups,
    /// but every output pixel must equal this bit-for-bit.
    fn ref_conv7(x: &[f32], w: usize, h: usize, k: &[f32; 49]) -> Vec<f32> {
        let mut out = vec![0f32; w * h];
        for r in 0..h {
            for c in 0..w {
                let mut acc = 0f32;
                for i in 0..7i64 {
                    let yy = r as i64 + i - 3;
                    for j in 0..7i64 {
                        let xx = c as i64 + j - 3;
                        let xv = if yy >= 0 && yy < h as i64 && xx >= 0 && xx < w as i64 {
                            x[yy as usize * w + xx as usize]
                        } else {
                            0.0
                        };
                        acc = fma(k[(i * 7 + j) as usize], xv, acc);
                    }
                }
                out[r * w + c] = acc;
            }
        }
        out
    }

    /// Deterministic texture covering a wide value range (xorshift64).
    fn genimg(w: usize, h: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..w * h)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as f32 / u64::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn conv7_matches_scalar_reference_bit_for_bit() {
        // Sizes exercise every coverage shape: w<14 fits no lane group,
        // w=14 fits exactly one, ragged widths leave mid-row tails, and
        // h<7 forces the all-scalar border path on every row.
        let k: [f32; 49] = core::array::from_fn(|i| ((i * 37 % 13) as f32 - 6.0) / 7.0);
        for &(w, h) in &[
            (5, 5),
            (8, 8),
            (13, 9),
            (14, 14),
            (15, 15),
            (22, 16),
            (33, 17),
            (64, 48),
            (255, 129),
            (64, 6),
        ] {
            let x = genimg(w, h, 0x9e3779b97f4a7c15 ^ (w as u64) << 32 ^ h as u64);
            let got = conv7(&x, w, h, &k);
            let want = ref_conv7(&x, w, h, &k);
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "conv7 {w}x{h} pixel {i} differs");
            }
        }
    }

    /// `gray`'s SIMD row must match `to_y` element-for-element on every
    /// dispatch tier.
    #[cfg(all(feature = "_dev", target_arch = "x86_64"))]
    #[test]
    fn gray_row_tiers_match_to_y() {
        use archmage::SimdToken;
        let w = 61usize; // exercises the w%8 scalar tail
        let mut s = 0x853c49e6748fea9bu64;
        let row: Vec<u8> = (0..w * 3)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 33) as u8
            })
            .collect();
        let want: Vec<f32> = (0..w)
            .map(|c| to_y(row[c * 3], row[c * 3 + 1], row[c * 3 + 2]))
            .collect();
        let mut scalar = vec![f32::NAN; w];
        gray_row_core_scalar_t(archmage::ScalarToken, &row, w, &mut scalar);
        if let Some(t) = archmage::X64V3Token::summon() {
            let mut v3 = vec![f32::NAN; w];
            gray_row_core_v3_t(t, &row, w, &mut v3);
            for c in 0..w {
                assert_eq!(v3[c].to_bits(), want[c].to_bits(), "v3 gray {c} differs");
            }
        }
        for c in 0..w {
            assert_eq!(scalar[c].to_bits(), want[c].to_bits(), "scalar gray {c} differs");
        }
    }

    /// Every dispatch tier must produce the identical interior: each
    /// lane runs the same fixed-order FMA chain as the scalar loop.
    #[cfg(all(feature = "_dev", target_arch = "x86_64"))]
    #[test]
    fn conv7_kernel_tiers_identical() {
        use archmage::SimdToken;
        let (w, h) = (37usize, 29usize);
        let x = genimg(w, h, 0xdeadbeefcafebabe);
        let k: [f32; 49] = core::array::from_fn(|i| ((i * 11 % 7) as f32 - 3.0) / 5.0);
        let mut scalar = vec![f32::NAN; w * h];
        conv7_core_scalar_t(archmage::ScalarToken, &x, w, h, &k, &mut scalar);
        if let Some(t) = archmage::X64V3Token::summon() {
            let mut v3 = vec![f32::NAN; w * h];
            conv7_core_v3_t(t, &x, w, h, &k, &mut v3);
            for (i, (a, b)) in scalar.iter().zip(&v3).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "v3 interior pixel {i} differs");
            }
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4xToken::summon() {
            let mut v4 = vec![f32::NAN; w * h];
            conv7_core_v4x_t(t, &x, w, h, &k, &mut v4);
            for (i, (a, b)) in scalar.iter().zip(&v4).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "v4x interior pixel {i} differs");
            }
        }
    }
}

#[cfg(test)]
mod summation_tests {
    use super::*;

    /// Pre-SIMD scalar `outer_sum` — per-column balanced tree, kept as
    /// the semantic reference for the lane-per-column kernel.
    fn ref_outer_sum(values: &[f32], rows: usize, columns: usize) -> Vec<f32> {
        fn column(values: &[f32], stride: usize, index: usize, first: usize, end: usize) -> f32 {
            match end - first {
                0 => 0.0,
                1 => values[first * stride + index],
                n => {
                    let middle = first + n / 2;
                    column(values, stride, index, first, middle)
                        + column(values, stride, index, middle, end)
                }
            }
        }
        (0..columns)
            .map(|i| column(values, columns, i, 0, rows))
            .collect()
    }

    fn genimg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s as f32 / u64::MAX as f32) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn outer_sum_matches_column_reference_bit_for_bit() {
        // rows mirror the resize kernel sizes (4..~20); column counts
        // cover full groups, tails, and fewer than 8.
        for &(rows, columns) in &[(6, 3), (6, 512), (9, 37), (11, 1024), (1, 8), (2, 13)] {
            let v = genimg(rows * columns, 0x243f6a8885a308d3 ^ (rows as u64) << 16 ^ columns as u64);
            let got = outer_sum(&v, rows, columns);
            let want = ref_outer_sum(&v, rows, columns);
            for (i, (g, e)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "outer_sum {rows}x{columns} column {i} differs"
                );
            }
        }
    }

    #[test]
    fn quad_sums_match_scalar_trees_bit_for_bit() {
        for &n in &[0usize, 1, 7, 8, 9, 33, 36, 255, 1024, 4099] {
            let x = genimg(n, 0x452821e638d01377 ^ n as u64);
            let q = quad_sums(&x);
            let tree = |f: &dyn Fn(f32) -> f32| inner_sum(&x.iter().map(|&v| f(v)).collect::<Vec<_>>());
            let want = [
                tree(&|v| {
                    let p = if v < 0.0 { v } else { 0.0 };
                    p * p
                }),
                tree(&|v| {
                    let p = if v > 0.0 { v } else { 0.0 };
                    p * p
                }),
                tree(&|v| v.abs()),
                tree(&|v| v * v),
            ];
            for (k, (g, e)) in q.iter().zip(&want).enumerate() {
                assert_eq!(g.to_bits(), e.to_bits(), "quad_sums n={n} lane {k} differs");
            }
        }
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
