// Copyright (c) Imazen LLC and the JPEG XL Project Authors.
// Licensed under AGPL-3.0-or-later. Commercial licenses at https://www.imazen.io/pricing

//! LUT-based separable Gaussian blur.
//!
//! Mathematically identical to [`super::blur`] / [`super::blur_3ch`], but
//! reads pre-computed weights and integral-table values from a small
//! GPU buffer instead of evaluating `exp` per tap.
//!
//! The two-fold win:
//!
//! 1. **No transcendentals in the hot loop.** The existing kernels call
//!    `f32::powf(2.0, x*LOG2_E)` once per tap per pixel; for the σ=7.16
//!    LF blur that's ~33 `powf` calls per output pixel × 12 MP × 2
//!    passes ≈ 800M transcendentals per blur direction. The LUT kernel
//!    replaces each with a single small-array load.
//!
//! 2. **O(1) edge weight via integral table** (vship-style — see
//!    `~/work/refs/Vship/src/HIP/butter/gaussianblur.hpp:8-22`). Instead
//!    of summing weights with the partial-window loop, we look up
//!    `integral[end_offset + 1] - integral[begin_offset]` — one
//!    subtraction. Removes a `wsum += weight` per tap.
//!
//! Inspired by vship's `GaussianHandle` pattern but the GPU code is
//! rewritten in CubeCL idioms (no line-for-line port, vship is MIT NON-AI).
//!
//! ## Weight table layout
//!
//! For a given sigma the host computes `radius = max(1, floor(2.25 * sigma))`
//! and writes:
//!
//! - `weights[0..=2R]`        — Gaussian weights at offsets `-R..=R`.
//!   Stored UN-normalized so the kernel's per-output normalization
//!   (`sum/wsum`) is bit-equivalent to the on-the-fly path.
//! - `integral[0..=2R+1]`     — `integral[k] = Σ_{i<k} weights[i]`.
//!
//! Both tables are packed into a single `Array<f32>` of length `4R+3`:
//! `weights` occupy `[0..=2R]`, `integral` occupies `[2R+1..=4R+2]`.
//! Use the helpers below to lay them out.

#![allow(clippy::assign_op_pattern)]

use cubecl::prelude::*;

/// Kernel-extent multiplier — matches libjxl's `M = 2.25` (same as
/// `super::blur`). Public so the host helper agrees on radius.
pub const M: f32 = 2.25;

/// Compute `radius = max(1, floor(M * sigma))` host-side.
pub fn radius_for(sigma: f32) -> usize {
    let raw = (M * sigma) as u32;
    raw.max(1) as usize
}

/// Compute the un-normalized Gaussian weights + their integral table on
/// the host. The layout matches what the kernels below expect. Returns
/// `(packed_table, radius)`.
///
/// Weight formula: `gauss(d, s) = exp(-0.5 * (d/s)^2)`. Matches
/// [`super::blur::gauss`] / [`super::blur_3ch::gauss`] which use the
/// equivalent `exp(x) = 2^(x*log2(e))` substitution; the resulting
/// floats are within ulp on every backend the CUDA toolchain targets.
pub fn make_table(sigma: f32) -> (Vec<f32>, usize) {
    let r = radius_for(sigma);
    let inv_s = 1.0_f32 / sigma;
    let mut table = vec![0.0_f32; 4 * r + 3];
    let mut acc = 0.0_f32;
    // weights[0..=2R]
    for k in 0..=(2 * r) {
        let d = (k as i32 - r as i32) as f32;
        let z = d * inv_s;
        let w = (-0.5_f32 * z * z).exp();
        table[k] = w;
    }
    // integral[0..=2R+1] follows the weights region.
    let integ_off = 2 * r + 1;
    for k in 0..=(2 * r + 1) {
        table[integ_off + k] = acc;
        if k <= 2 * r {
            acc += table[k];
        }
    }
    (table, r)
}

/// Horizontal Gaussian blur with precomputed weight LUT + integral table.
///
/// `radius` is the half-window size (same definition as
/// [`super::blur::horizontal_blur_kernel`]'s `radius_us`). `table` is
/// the packed `[weights || integral]` array; weights occupy
/// `[0..=2R]`, integral occupies `[2R+1..=4R+2]`.
#[cube(launch_unchecked)]
pub fn horizontal_blur_lut_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let row = idx / w;
    let x = idx - row * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(x, r);
    let end = u32::min((x + r) as u32, (w - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    // Edge-clamped weight sum via integral table:
    //   wsum = integral[(end - x) + r + 1] - integral[(begin - x) + r]
    // (begin-x can be negative; adding r shifts to non-negative.)
    let a = begin + r - x;
    let b = end + r + 1 - x;
    let wsum = table[integ_off + b] - table[integ_off + a];

    let mut sum = 0.0f32;
    let row_off = row * w;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - x];
        sum += src[row_off + i] * weight;
        i += 1;
    }
    // sum / wsum (NOT sum * (1/wsum)) — bit-rounding agreement with
    // the original blur kernel matters for downstream tie-breakers.
    dst[idx] = sum / wsum;
}

/// Vertical Gaussian blur with precomputed weight LUT + integral table.
#[cube(launch_unchecked)]
pub fn vertical_blur_lut_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(y, r);
    let end = u32::min((y + r) as u32, (h - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - y;
    let b = end + r + 1 - y;
    let wsum = table[integ_off + b] - table[integ_off + a];

    let mut sum = 0.0f32;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - y];
        sum += src[i * w + x] * weight;
        i += 1;
    }
    dst[idx] = sum / wsum;
}

/// 3-channel fused horizontal blur (LUT variant).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn horizontal_blur_3ch_lut_kernel(
    src_x: &Array<f32>,
    src_y: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_b: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let row = idx / w;
    let x = idx - row * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(x, r);
    let end = u32::min((x + r) as u32, (w - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - x;
    let b = end + r + 1 - x;
    let wsum = table[integ_off + b] - table[integ_off + a];

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let row_off = row * w;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - x];
        let off = row_off + i;
        sum_x += src_x[off] * weight;
        sum_y += src_y[off] * weight;
        sum_b += src_b[off] * weight;
        i += 1;
    }
    dst_x[idx] = sum_x / wsum;
    dst_y[idx] = sum_y / wsum;
    dst_b[idx] = sum_b / wsum;
}

/// 3-channel fused vertical blur + opsin dynamics (LUT variant).
///
/// Combines the σ=1.2 vertical blur with the opsin-dynamics XYB
/// conversion into a single launch. Eliminates the intermediate
/// `blur_*` buffer write/read pair (3 channels × n × 4 B = 144 MB at
/// 12 MP) that the standalone `vertical_blur_3ch_lut_kernel` +
/// `opsin_dynamics_planar_kernel` pair generates.
///
/// Inputs:
/// - `h_src_*`: horizontal-blurred linear RGB (output of the H-pass).
/// - `orig_*`: original linear RGB (pre-blur). Read-only here.
/// - `table`, `width`, `height`, `radius`: same shape as
///   [`vertical_blur_3ch_lut_kernel`].
/// - `intensity_multiplier`: same as [`super::colors::opsin_dynamics_planar_kernel`].
///
/// Output:
/// - `xyb_*`: planar XYB after opsin. Same `(sx-sy, sx+sy, sz)` layout
///   as `opsin_dynamics_planar_kernel`.
///
/// Math matches the explicit two-kernel sequence bit-for-bit (same
/// f32 op tree, same FMA-vs-mul boundaries).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn vertical_blur_3ch_opsin_lut_kernel(
    h_src_x: &Array<f32>,
    h_src_y: &Array<f32>,
    h_src_b: &Array<f32>,
    orig_x: &Array<f32>,
    orig_y: &Array<f32>,
    orig_b: &Array<f32>,
    xyb_x: &mut Array<f32>,
    xyb_y: &mut Array<f32>,
    xyb_z: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
    intensity_multiplier: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(y, r);
    let end = u32::min((y + r) as u32, (h - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - y;
    let b_idx = end + r + 1 - y;
    let wsum = table[integ_off + b_idx] - table[integ_off + a];

    // ── vertical blur of the horizontally-pre-blurred inputs ──
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - y];
        let off = i * w + x;
        sum_x += h_src_x[off] * weight;
        sum_y += h_src_y[off] * weight;
        sum_b += h_src_b[off] * weight;
        i += 1;
    }
    let br = (sum_x / wsum) * intensity_multiplier;
    let bg = (sum_y / wsum) * intensity_multiplier;
    let bb = (sum_b / wsum) * intensity_multiplier;

    // ── opsin dynamics (clamp-absorbance path on the blurred sample) ──
    // CPU butteraugli's exact matrix + bias-floor + gamma + sensitivity.
    let bx_pre = 0.299_565_5_f32 * br + 0.633_730_9 * bg + 0.077_705_614 * bb + 1.755_748_4;
    let by_pre = 0.221_586_91 * br + 0.693_913_9 * bg + 0.098_731_36 * bb + 1.755_748_4;
    let bz_pre = 0.02 * br + 0.02 * bg + 0.204_801_29 * bb + 12.226_455;
    let bx = f32::max(f32::max(bx_pre, 1.755_748_4), 1e-4);
    let by = f32::max(f32::max(by_pre, 1.755_748_4), 1e-4);
    let bz = f32::max(f32::max(bz_pre, 12.226_455), 1e-4);
    let sens_x = f32::max(gamma(bx) / bx, 1e-4);
    let sens_y = f32::max(gamma(by) / by, 1e-4);
    let sens_z = f32::max(gamma(bz) / bz, 1e-4);

    // ── original-sample absorbance (no clamp) ──
    let or = orig_x[idx] * intensity_multiplier;
    let og = orig_y[idx] * intensity_multiplier;
    let ob = orig_b[idx] * intensity_multiplier;
    let sx_pre = 0.299_565_5_f32 * or + 0.633_730_9 * og + 0.077_705_614 * ob + 1.755_748_4;
    let sy_pre = 0.221_586_91 * or + 0.693_913_9 * og + 0.098_731_36 * ob + 1.755_748_4;
    let sz_pre = 0.02 * or + 0.02 * og + 0.204_801_29 * ob + 12.226_455;

    let mut sx = sx_pre * sens_x;
    let mut sy = sy_pre * sens_y;
    let mut sz = sz_pre * sens_z;
    sx = f32::max(sx, 1.755_748_4);
    sy = f32::max(sy, 1.755_748_4);
    sz = f32::max(sz, 12.226_455);

    xyb_x[idx] = sx - sy;
    xyb_y[idx] = sx + sy;
    xyb_z[idx] = sz;
}

/// Butteraugli's `gamma` — matches `super::colors::gamma` exactly.
#[cube]
fn gamma(v: f32) -> f32 {
    19.245_014_f32 * f32::ln(v + 9.971_064) - 23.160_463
}

/// 3-channel fused vertical blur + MF subtract + xyb_low_freq_to_vals
/// for the LF separation stage.
///
/// Eliminates the `subtract_arrays(lin, LF) → MF` triple-launch and the
/// `xyb_low_freq_to_vals(LF)` launch by doing both inside the V-blur:
///
///   for each output pixel (x, y):
///     lf_x = Σ_i  table[i] · h_src_x[(y+i)·w + x]   (vertical blur)
///     lf_y = ...                                                "
///     lf_b = ...                                                "
///     mf_x = orig_x[idx] - lf_x  → write mf_x_out[idx]
///     mf_y = orig_y[idx] - lf_y  → write mf_y_out[idx]
///     mf_b = orig_b[idx] - lf_b  → write mf_b_out[idx]
///     // xyb_low_freq_to_vals on the LF triple:
///     lf_b += Y_TO_B_MULI · lf_y
///     lf_b *= BMULI
///     lf_x *= XMULI
///     lf_y *= YMULI
///     write (lf_x, lf_y, lf_b) → lf_*_out[idx]
///
/// No R/W hazard: `orig_*` reads are pointwise (`[idx]` only) and don't
/// touch the V-blur window; MF outputs are independent from LF outputs.
///
/// Replaces 5 separate launches (V-blur 3ch + 3× subtract + 1× xyb-mul)
/// with one. At 12 MP each saved kernel was ~220 µs; net saving per
/// `compute()` call is ~5 × (215+215+215+445) µs ≈ 5.5 ms (across both
/// sides). The fused V-blur itself stays at the V-blur's existing cost
/// since the post-blur math is cheap pointwise FMAs.
///
/// Constants below MUST match
/// [`crate::kernels::frequency::xyb_low_freq_to_vals_kernel`]:
/// `XMULI = 33.832837`, `YMULI = 14.458268`, `BMULI = 49.879845`,
/// `Y_TO_B_MULI = -0.36226705`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn vertical_blur_3ch_lf_split_lut_kernel(
    h_src_x: &Array<f32>,
    h_src_y: &Array<f32>,
    h_src_b: &Array<f32>,
    orig_x: &Array<f32>,
    orig_y: &Array<f32>,
    orig_b: &Array<f32>,
    lf_x_out: &mut Array<f32>,
    lf_y_out: &mut Array<f32>,
    lf_b_out: &mut Array<f32>,
    mf_x_out: &mut Array<f32>,
    mf_y_out: &mut Array<f32>,
    mf_b_out: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(y, r);
    let end = u32::min((y + r) as u32, (h - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - y;
    let b_idx = end + r + 1 - y;
    let wsum = table[integ_off + b_idx] - table[integ_off + a];

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - y];
        let off = i * w + x;
        sum_x += h_src_x[off] * weight;
        sum_y += h_src_y[off] * weight;
        sum_b += h_src_b[off] * weight;
        i += 1;
    }
    let lf_x_raw = sum_x / wsum;
    let lf_y_raw = sum_y / wsum;
    let lf_b_raw = sum_b / wsum;

    // MF = orig - LF (pre-xyb-mul). Read orig_*[idx] once; that read
    // doesn't overlap the V-blur window so it's race-free.
    mf_x_out[idx] = orig_x[idx] - lf_x_raw;
    mf_y_out[idx] = orig_y[idx] - lf_y_raw;
    mf_b_out[idx] = orig_b[idx] - lf_b_raw;

    // xyb_low_freq_to_vals on the LF triple (in-the-same-kernel,
    // matches the standalone kernel bit-for-bit).
    let lf_b_mixed = (lf_b_raw + (-0.362_267_05_f32) * lf_y_raw) * 49.879_845;
    lf_x_out[idx] = lf_x_raw * 33.832_837;
    lf_y_out[idx] = lf_y_raw * 14.458_268;
    lf_b_out[idx] = lf_b_mixed;
}

/// 3-channel fused vertical blur + MF/HF split for the
/// SIGMA_HF separation step.
///
/// Replaces the V-pass of `vertical_blur_3ch_lut_kernel` + 3 downstream
/// split kernels:
///
///   X: blur(MF_X) → blurred_X
///      → HF_X = orig_X - blurred_X                       (write freq[1][0])
///      → MF_X = remove_range(blurred_X, REMOVE_MF_RANGE) (write freq[2][0])
///   Y: blur(MF_Y) → blurred_Y
///      → HF_Y = orig_Y - blurred_Y                       (write freq[1][1])
///      → MF_Y = amplify_range(blurred_Y, ADD_MF_RANGE)   (write freq[2][1])
///   B: blur(MF_B) → blurred_B
///      → MF_B = blurred_B                                (write freq[2][2])
///
/// 5 outputs total (HF_X, HF_Y, MF_X, MF_Y, MF_B). Per-thread reads
/// of `orig_*[idx]` are pointwise (no overlap with V-blur window) so
/// reading and writing freq[2][ch] within the same thread is safe.
///
/// Saves 3 split-kernel launches per `separate_frequencies` HF step.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn vertical_blur_3ch_hf_split_lut_kernel(
    h_src_x: &Array<f32>,
    h_src_y: &Array<f32>,
    h_src_b: &Array<f32>,
    orig_x: &Array<f32>,
    orig_y: &Array<f32>,
    out_hf_x: &mut Array<f32>,
    out_hf_y: &mut Array<f32>,
    out_mf_x: &mut Array<f32>,
    out_mf_y: &mut Array<f32>,
    out_mf_b: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
    remove_mf_range: f32,
    add_mf_range: f32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(y, r);
    let end = u32::min((y + r) as u32, (h - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - y;
    let b_idx = end + r + 1 - y;
    let wsum = table[integ_off + b_idx] - table[integ_off + a];

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - y];
        let off = i * w + x;
        sum_x += h_src_x[off] * weight;
        sum_y += h_src_y[off] * weight;
        sum_b += h_src_b[off] * weight;
        i += 1;
    }
    let bx = sum_x / wsum;
    let by = sum_y / wsum;
    let bb = sum_b / wsum;

    // X: HF + MF (remove-range MF).
    let ox = orig_x[idx];
    out_hf_x[idx] = ox - bx;
    let mx = if bx > remove_mf_range {
        bx - remove_mf_range
    } else if bx < -remove_mf_range {
        bx + remove_mf_range
    } else {
        f32::new(0.0)
    };
    out_mf_x[idx] = mx;

    // Y: HF + MF (amplify-range MF).
    let oy = orig_y[idx];
    out_hf_y[idx] = oy - by;
    let my = if by > add_mf_range {
        by + add_mf_range
    } else if by < -add_mf_range {
        by - add_mf_range
    } else {
        f32::new(2.0) * by
    };
    out_mf_y[idx] = my;

    // B: MF only (no HF accumulated for B).
    out_mf_b[idx] = bb;
}

/// 3-channel fused vertical blur (LUT variant).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn vertical_blur_3ch_lut_kernel(
    src_x: &Array<f32>,
    src_y: &Array<f32>,
    src_b: &Array<f32>,
    dst_x: &mut Array<f32>,
    dst_y: &mut Array<f32>,
    dst_b: &mut Array<f32>,
    table: &Array<f32>,
    width: u32,
    height: u32,
    radius: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (width * height) as usize;
    if idx >= total {
        terminate!();
    }
    let w = width as usize;
    let h = height as usize;
    let y = idx / w;
    let x = idx - y * w;

    let r = radius as usize;
    let begin = usize::saturating_sub(y, r);
    let end = u32::min((y + r) as u32, (h - 1) as u32) as usize;

    let integ_off = 2 * r + 1;
    let a = begin + r - y;
    let b = end + r + 1 - y;
    let wsum = table[integ_off + b] - table[integ_off + a];

    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_b = 0.0f32;
    let mut i = begin;
    while i <= end {
        let weight = table[i + r - y];
        let off = i * w + x;
        sum_x += src_x[off] * weight;
        sum_y += src_y[off] * weight;
        sum_b += src_b[off] * weight;
        i += 1;
    }
    dst_x[idx] = sum_x / wsum;
    dst_y[idx] = sum_y / wsum;
    dst_b[idx] = sum_b / wsum;
}
