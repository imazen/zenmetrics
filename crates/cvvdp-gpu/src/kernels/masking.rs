//! Contrast masking — cvvdp v0.5.4's `mult-mutual` model with
//! cross-channel masking on.
//!
//! Per pixel per channel cc, given CSF-weighted contrasts `T_p[cc]`
//! and `R_p[cc]` (the test and reference, after `T * S * ch_gain`):
//!
//! ```text
//! M_mm[cc] = phase_uncertainty(min(|T_p[cc]|, |R_p[cc]|))
//! M[cc]    = sum_in xcm[in, cc] * safe_pow(|M_mm[in]|, q[in])
//! D[cc]    = clamp_diffs(safe_pow(|T_p[cc] - R_p[cc]|, p) / (1 + M[cc]))
//! ```
//!
//! Parameters (from cvvdp v0.5.4's `cvvdp_parameters.json`):
//!
//! - [`MASK_P`] — `mask_p` exponent (single scalar)
//! - [`MASK_Q`] — `mask_q[0..3]` per-channel exponents
//! - [`MASK_C`] — phase-uncertainty `mask_c` (log10)
//! - [`D_MAX`]  — clamp ceiling (log10)
//! - [`XCM_3X3`] — first 3 rows × 3 cols of `2^xcm_weights` reshaped
//!   to 4×4, which is the cross-channel pooling matrix for the 3-
//!   channel still-image path.
//!
//! Phase-uncertainty Gaussian blur (sigma = `pu_dilate = 3`)
//! ships via `pu_blur_h_kernel` + `pu_blur_v_kernel` (single-
//! channel spec reference) and the production 3-channel fused
//! variants `pu_blur_h_3ch_kernel` + `pu_blur_v_3ch_scaled_kernel`
//! (the v-pass folds the post-blur `* 10^MASK_C` scale into the
//! output). cvvdp skips the blur for bands smaller than
//! `pu_padsize = 6`; `compute_dkl_d_bands` routes the small-band
//! case to `mult_mutual_3ch_no_blur_kernel`.

use cubecl::prelude::*;

/// Per-channel gain `[1, 1.45, 1]` that cvvdp's `mult-mutual` path
/// applies to `T_p` and `R_p` together with the CSF sensitivity `S`,
/// BEFORE masking:
///
/// ```text
/// T_p = T * S * CH_GAIN
/// R_p = R * S * CH_GAIN
/// ```
///
/// In cvvdp v0.5.4 the 4-channel tensor is `[1, 1.45, 1, 1]`; the
/// still-image 3-channel pipeline slices to `[1, 1.45, 1]`. Apply
/// at the masking call site — the CSF kernel itself doesn't bake
/// this in.
pub const CH_GAIN: [f32; 3] = [1.0, 1.45, 1.0];

/// `mask_p` exponent from cvvdp v0.5.4.
pub const MASK_P: f32 = 2.264_355_2;

/// `mask_q[0..3]` for the still-image 3-channel pipeline.
pub const MASK_Q: [f32; 3] = [1.302_622_7, 2.888_590_8, 3.680_771_3];

/// `mask_c` for phase-uncertainty scaling: applied as `10^mask_c`.
pub const MASK_C: f32 = -0.795_497_12;

/// `d_max` for soft clamp ceiling: applied as `10^d_max`.
pub const D_MAX: f32 = 2.564_245_5;

/// The 3×3 cross-channel masking matrix derived from cvvdp v0.5.4's
/// `xcm_weights` (16 values reshaped 4×4, first 3 rows × 3 cols,
/// elementwise `2^x`). Row index = input channel; column index =
/// output channel:
///
/// ```text
/// M[out] = sum_in XCM_3X3[in][out] * mask_term[in]
/// ```
pub const XCM_3X3: [[f32; 3]; 3] = [
    // 2^(-0.189501), 2^(-5.962151), 2^(-4.318346)
    [0.876_968, 0.016_103_15, 0.050_159_38],
    // 2^2.565559,  2^0.344067, 2^(-2.719646)
    [5.918_792, 1.269_323, 0.152_080_92],
    // 2^3.811837,  2^(-1.005171), 2^(-0.519338)
    [14.041_055, 0.498_209_6, 0.697_756_55],
];

/// cvvdp's `safe_pow(x, p) = (x + eps)^p - eps^p`, with `eps = 1e-5`.
/// Avoids zero-derivative singularities at `x = 0`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::safe_pow;
///
/// // safe_pow(0, p) = (eps)^p - eps^p = 0 exactly.
/// assert_eq!(safe_pow(0.0, 2.0), 0.0);
/// assert_eq!(safe_pow(0.0, 0.5), 0.0);
///
/// // Above the eps regularization point, behaves like x^p within
/// // a small bias.
/// let v = safe_pow(2.0, 2.0);
/// assert!((v - 4.0).abs() < 0.01, "safe_pow(2, 2) = {v}, expected ≈ 4");
///
/// // Monotonically increasing in x for any positive p.
/// assert!(safe_pow(1.0, 2.5) < safe_pow(2.0, 2.5));
/// ```
#[inline]
#[must_use]
pub fn safe_pow(x: f32, p: f32) -> f32 {
    let eps: f32 = 1e-5;
    (x + eps).powf(p) - eps.powf(p)
}

/// Soft clamp matching cvvdp v0.5.4's `dclamp_type = "soft"`:
/// `D_max * D / (D_max + D)` with `D_max = 10^d_max`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::{D_MAX, clamp_diff_soft};
///
/// // f(0) = 0.
/// assert_eq!(clamp_diff_soft(0.0), 0.0);
///
/// // f(d_max) = d_max / 2 (the half-saturation point).
/// let d_max = 10.0_f32.powf(D_MAX);
/// let half = clamp_diff_soft(d_max);
/// assert!((half - d_max / 2.0).abs() / d_max < 1e-5);
///
/// // Asymptotic: for d ≫ d_max, output stays strictly below d_max.
/// assert!(clamp_diff_soft(1e9) < d_max);
/// ```
#[inline]
#[must_use]
pub fn clamp_diff_soft(d: f32) -> f32 {
    let d_max = 10.0_f32.powf(D_MAX);
    d_max * d / (d_max + d)
}

/// Phase uncertainty for the "small band, no blur" path: multiply
/// by `10^mask_c`. cvvdp uses this branch when the band's smallest
/// dimension is ≤ `PU_PADSIZE`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::{MASK_C, phase_uncertainty_no_blur};
///
/// // Pure scaling — output = input × 10^MASK_C ≈ input × 0.1603.
/// let scale = 10.0_f32.powf(MASK_C);
/// assert_eq!(phase_uncertainty_no_blur(1.0), scale);
/// assert_eq!(phase_uncertainty_no_blur(0.0), 0.0);
///
/// // Scale factor lands in [0.15, 0.17] for the canonical
/// // MASK_C = -0.7955.
/// assert!(scale > 0.15 && scale < 0.17);
/// ```
#[inline]
#[must_use]
pub fn phase_uncertainty_no_blur(m: f32) -> f32 {
    m * 10.0_f32.powf(MASK_C)
}

/// 1D Gaussian kernel for phase-uncertainty blur. Matches the
/// kernel `torchvision.transforms.GaussianBlur(13, 3.0)` builds —
/// σ = 3 px, 13 taps, normalized to sum 1.
#[rustfmt::skip]
pub const PU_BLUR_KERNEL_1D: [f32; 13] = [
    1.854_402_2e-2, 3.416_694_2e-2, 5.633_176_4e-2, 8.310_854e-2,
    1.097_193e-1,   1.296_180_3e-1, 1.370_228_2e-1, 1.296_180_3e-1,
    1.097_193e-1,   8.310_854e-2,   5.633_176_4e-2, 3.416_694_2e-2,
    1.854_402_2e-2,
];

/// Reflect index `i` into `[0, n)` for the 13-tap PU blur. Matches
/// torchvision's `F.pad(..., mode='reflect')` behaviour: indices
/// outside the range mirror around the boundary without repeating
/// the edge pixel.
fn reflect_idx_for_blur(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    debug_assert!(n_i > 0);
    let mut j = i;
    // Loop in case the radius exceeds the dimension (small bands).
    while j < 0 || j >= n_i {
        if j < 0 {
            j = -j;
        }
        if j >= n_i {
            j = 2 * n_i - 2 - j;
        }
    }
    j as usize
}

/// Apply the σ=3 separable Gaussian blur from cvvdp's
/// `phase_uncertainty`. Allocates the intermediate buffer; not for
/// hot paths. Returns a `w × h` `Vec<f32>`. Reflect padding at the
/// boundary, matching torchvision.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::gaussian_blur_sigma3;
///
/// // Output length matches input.
/// let src = vec![0.0_f32; 16 * 16];
/// let out = gaussian_blur_sigma3(&src, 16, 16);
/// assert_eq!(out.len(), 256);
///
/// // DC preservation: a uniform input passes through unchanged
/// // (the 13-tap kernel sums to 1).
/// let uniform = vec![5.0_f32; 16 * 16];
/// let out = gaussian_blur_sigma3(&uniform, 16, 16);
/// for &v in &out {
///     assert!((v - 5.0).abs() < 1e-5);
/// }
/// ```
#[must_use]
pub fn gaussian_blur_sigma3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), w * h);
    let k = PU_BLUR_KERNEL_1D;
    let half = 6_isize; // (13 - 1) / 2

    // Horizontal pass.
    let mut h_pass = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0_f32;
            for t in 0..13 {
                let sx = reflect_idx_for_blur(x as isize + t as isize - half, w);
                s += k[t] * src[y * w + sx];
            }
            h_pass[y * w + x] = s;
        }
    }

    // Vertical pass.
    let mut out = vec![0.0_f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0_f32;
            for t in 0..13 {
                let sy = reflect_idx_for_blur(y as isize + t as isize - half, h);
                s += k[t] * h_pass[sy * w + x];
            }
            out[y * w + x] = s;
        }
    }
    out
}

/// Horizontal pass of the σ=3 separable Gaussian blur, with reflect
/// padding. Per-output-pixel thread. Input `src` is `w × h`
/// row-major; output `dst` is the same shape. Caller multiplies the
/// kernel coefficients by `10^MASK_C` separately (or chains a
/// scalar multiply kernel) since the blur itself is sum-to-1.
///
/// Boundary handling: reflect (no edge repeat), matching
/// torchvision's `F.pad(..., mode='reflect')`.
#[cube(launch)]
pub fn pu_blur_h_kernel(src: &Array<f32>, dst: &mut Array<f32>, w: u32, h: u32) {
    let idx = ABSOLUTE_POS;
    let total = (w * h) as usize;
    if idx >= total {
        terminate!();
    }
    let wu = w as usize;
    let y = idx / wu;
    let x = idx - y * wu;
    let w_i = w as i32;
    let x_i = x as i32;
    let half = 6_i32;

    // 13-tap Gaussian (σ=3), normalized — matches PU_BLUR_KERNEL_1D.
    let k0 = f32::new(1.854_402_2e-2);
    let k1 = f32::new(3.416_694_2e-2);
    let k2 = f32::new(5.633_176_4e-2);
    let k3 = f32::new(8.310_854e-2);
    let k4 = f32::new(1.097_193e-1);
    let k5 = f32::new(1.296_180_3e-1);
    let k6 = f32::new(1.370_228_2e-1);
    let k7 = f32::new(1.296_180_3e-1);
    let k8 = f32::new(1.097_193e-1);
    let k9 = f32::new(8.310_854e-2);
    let k10 = f32::new(5.633_176_4e-2);
    let k11 = f32::new(3.416_694_2e-2);
    let k12 = f32::new(1.854_402_2e-2);

    // Reflect-12 unroll, inline. For each tap t in 0..13, source
    // index is x + (t - half), reflected into [0, w).
    let s0 = reflect_pu_idx(x_i - half, w_i);
    let s1 = reflect_pu_idx(x_i + 1 - half, w_i);
    let s2 = reflect_pu_idx(x_i + 2 - half, w_i);
    let s3 = reflect_pu_idx(x_i + 3 - half, w_i);
    let s4 = reflect_pu_idx(x_i + 4 - half, w_i);
    let s5 = reflect_pu_idx(x_i + 5 - half, w_i);
    let s6 = reflect_pu_idx(x_i + 6 - half, w_i);
    let s7 = reflect_pu_idx(x_i + 7 - half, w_i);
    let s8 = reflect_pu_idx(x_i + 8 - half, w_i);
    let s9 = reflect_pu_idx(x_i + 9 - half, w_i);
    let s10 = reflect_pu_idx(x_i + 10 - half, w_i);
    let s11 = reflect_pu_idx(x_i + 11 - half, w_i);
    let s12 = reflect_pu_idx(x_i + 12 - half, w_i);

    let row_off = y * wu;
    dst[idx] = k0 * src[row_off + s0]
        + k1 * src[row_off + s1]
        + k2 * src[row_off + s2]
        + k3 * src[row_off + s3]
        + k4 * src[row_off + s4]
        + k5 * src[row_off + s5]
        + k6 * src[row_off + s6]
        + k7 * src[row_off + s7]
        + k8 * src[row_off + s8]
        + k9 * src[row_off + s9]
        + k10 * src[row_off + s10]
        + k11 * src[row_off + s11]
        + k12 * src[row_off + s12];
}

/// 3-channel `|T_p_dis[c] - T_p_ref[c]|` per pixel. Used by the
/// baseband branch of `compute_dkl_d_bands` — cvvdp bypasses the
/// mult-mutual masker on the baseband and emits the raw absolute
/// difference of the CSF-weighted T_p values. Single launch per
/// non-masked band.
#[cube(launch)]
pub fn diff_abs_3ch_kernel(
    t_p_dis_a: &Array<f32>,
    t_p_dis_rg: &Array<f32>,
    t_p_dis_vy: &Array<f32>,
    t_p_ref_a: &Array<f32>,
    t_p_ref_rg: &Array<f32>,
    t_p_ref_vy: &Array<f32>,
    d_a: &mut Array<f32>,
    d_rg: &mut Array<f32>,
    d_vy: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let zero = f32::new(0.0);

    let da = t_p_dis_a[idx] - t_p_ref_a[idx];
    d_a[idx] = if da < zero { -da } else { da };
    let drg = t_p_dis_rg[idx] - t_p_ref_rg[idx];
    d_rg[idx] = if drg < zero { -drg } else { drg };
    let dvy = t_p_dis_vy[idx] - t_p_ref_vy[idx];
    d_vy[idx] = if dvy < zero { -dvy } else { dvy };
}

/// 3-channel horizontal pass of the σ=3 separable Gaussian blur.
/// Reads `src_a` / `src_rg` / `src_vy` at the same `(x, y)` and
/// writes the corresponding three blurred outputs in one launch.
/// Saves 2 kernel launches per non-baseband pyramid level vs the
/// per-channel `pu_blur_h_kernel`.
#[cube(launch)]
pub fn pu_blur_h_3ch_kernel(
    src_a: &Array<f32>,
    src_rg: &Array<f32>,
    src_vy: &Array<f32>,
    dst_a: &mut Array<f32>,
    dst_rg: &mut Array<f32>,
    dst_vy: &mut Array<f32>,
    w: u32,
    h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (w * h) as usize;
    if idx >= total {
        terminate!();
    }
    let wu = w as usize;
    let y = idx / wu;
    let x = idx - y * wu;
    let w_i = w as i32;
    let x_i = x as i32;
    let half = 6_i32;

    let k0 = f32::new(1.854_402_2e-2);
    let k1 = f32::new(3.416_694_2e-2);
    let k2 = f32::new(5.633_176_4e-2);
    let k3 = f32::new(8.310_854e-2);
    let k4 = f32::new(1.097_193e-1);
    let k5 = f32::new(1.296_180_3e-1);
    let k6 = f32::new(1.370_228_2e-1);
    let k7 = f32::new(1.296_180_3e-1);
    let k8 = f32::new(1.097_193e-1);
    let k9 = f32::new(8.310_854e-2);
    let k10 = f32::new(5.633_176_4e-2);
    let k11 = f32::new(3.416_694_2e-2);
    let k12 = f32::new(1.854_402_2e-2);

    let s0 = reflect_pu_idx(x_i - half, w_i);
    let s1 = reflect_pu_idx(x_i + 1 - half, w_i);
    let s2 = reflect_pu_idx(x_i + 2 - half, w_i);
    let s3 = reflect_pu_idx(x_i + 3 - half, w_i);
    let s4 = reflect_pu_idx(x_i + 4 - half, w_i);
    let s5 = reflect_pu_idx(x_i + 5 - half, w_i);
    let s6 = reflect_pu_idx(x_i + 6 - half, w_i);
    let s7 = reflect_pu_idx(x_i + 7 - half, w_i);
    let s8 = reflect_pu_idx(x_i + 8 - half, w_i);
    let s9 = reflect_pu_idx(x_i + 9 - half, w_i);
    let s10 = reflect_pu_idx(x_i + 10 - half, w_i);
    let s11 = reflect_pu_idx(x_i + 11 - half, w_i);
    let s12 = reflect_pu_idx(x_i + 12 - half, w_i);

    let row_off = y * wu;
    dst_a[idx] = k0 * src_a[row_off + s0]
        + k1 * src_a[row_off + s1]
        + k2 * src_a[row_off + s2]
        + k3 * src_a[row_off + s3]
        + k4 * src_a[row_off + s4]
        + k5 * src_a[row_off + s5]
        + k6 * src_a[row_off + s6]
        + k7 * src_a[row_off + s7]
        + k8 * src_a[row_off + s8]
        + k9 * src_a[row_off + s9]
        + k10 * src_a[row_off + s10]
        + k11 * src_a[row_off + s11]
        + k12 * src_a[row_off + s12];
    dst_rg[idx] = k0 * src_rg[row_off + s0]
        + k1 * src_rg[row_off + s1]
        + k2 * src_rg[row_off + s2]
        + k3 * src_rg[row_off + s3]
        + k4 * src_rg[row_off + s4]
        + k5 * src_rg[row_off + s5]
        + k6 * src_rg[row_off + s6]
        + k7 * src_rg[row_off + s7]
        + k8 * src_rg[row_off + s8]
        + k9 * src_rg[row_off + s9]
        + k10 * src_rg[row_off + s10]
        + k11 * src_rg[row_off + s11]
        + k12 * src_rg[row_off + s12];
    dst_vy[idx] = k0 * src_vy[row_off + s0]
        + k1 * src_vy[row_off + s1]
        + k2 * src_vy[row_off + s2]
        + k3 * src_vy[row_off + s3]
        + k4 * src_vy[row_off + s4]
        + k5 * src_vy[row_off + s5]
        + k6 * src_vy[row_off + s6]
        + k7 * src_vy[row_off + s7]
        + k8 * src_vy[row_off + s8]
        + k9 * src_vy[row_off + s9]
        + k10 * src_vy[row_off + s10]
        + k11 * src_vy[row_off + s11]
        + k12 * src_vy[row_off + s12];
}

/// Vertical pass of the σ=3 separable Gaussian blur.
#[cube(launch)]
pub fn pu_blur_v_kernel(src: &Array<f32>, dst: &mut Array<f32>, w: u32, h: u32) {
    let idx = ABSOLUTE_POS;
    let total = (w * h) as usize;
    if idx >= total {
        terminate!();
    }
    let wu = w as usize;
    let y = idx / wu;
    let x = idx - y * wu;
    let h_i = h as i32;
    let y_i = y as i32;
    let half = 6_i32;

    let k0 = f32::new(1.854_402_2e-2);
    let k1 = f32::new(3.416_694_2e-2);
    let k2 = f32::new(5.633_176_4e-2);
    let k3 = f32::new(8.310_854e-2);
    let k4 = f32::new(1.097_193e-1);
    let k5 = f32::new(1.296_180_3e-1);
    let k6 = f32::new(1.370_228_2e-1);
    let k7 = f32::new(1.296_180_3e-1);
    let k8 = f32::new(1.097_193e-1);
    let k9 = f32::new(8.310_854e-2);
    let k10 = f32::new(5.633_176_4e-2);
    let k11 = f32::new(3.416_694_2e-2);
    let k12 = f32::new(1.854_402_2e-2);

    let s0 = reflect_pu_idx(y_i - half, h_i);
    let s1 = reflect_pu_idx(y_i + 1 - half, h_i);
    let s2 = reflect_pu_idx(y_i + 2 - half, h_i);
    let s3 = reflect_pu_idx(y_i + 3 - half, h_i);
    let s4 = reflect_pu_idx(y_i + 4 - half, h_i);
    let s5 = reflect_pu_idx(y_i + 5 - half, h_i);
    let s6 = reflect_pu_idx(y_i + 6 - half, h_i);
    let s7 = reflect_pu_idx(y_i + 7 - half, h_i);
    let s8 = reflect_pu_idx(y_i + 8 - half, h_i);
    let s9 = reflect_pu_idx(y_i + 9 - half, h_i);
    let s10 = reflect_pu_idx(y_i + 10 - half, h_i);
    let s11 = reflect_pu_idx(y_i + 11 - half, h_i);
    let s12 = reflect_pu_idx(y_i + 12 - half, h_i);

    dst[idx] = k0 * src[s0 * wu + x]
        + k1 * src[s1 * wu + x]
        + k2 * src[s2 * wu + x]
        + k3 * src[s3 * wu + x]
        + k4 * src[s4 * wu + x]
        + k5 * src[s5 * wu + x]
        + k6 * src[s6 * wu + x]
        + k7 * src[s7 * wu + x]
        + k8 * src[s8 * wu + x]
        + k9 * src[s9 * wu + x]
        + k10 * src[s10 * wu + x]
        + k11 * src[s11 * wu + x]
        + k12 * src[s12 * wu + x];
}

/// 3-channel vertical pass of the σ=3 separable Gaussian blur with
/// the `* pu_scale` post-multiply folded in. Replaces three
/// `pu_blur_v_kernel` launches plus three `weight_band_kernel`
/// launches per non-baseband pyramid level: 6→1 launches.
///
/// `pu_scale` is the host-supplied `10^MASK_C` value applied by
/// cvvdp's `phase_uncertainty` after the sum-to-1 blur (the σ=3
/// Gaussian kernel coefficients here are already normalized).
#[cube(launch)]
pub fn pu_blur_v_3ch_scaled_kernel(
    src_a: &Array<f32>,
    src_rg: &Array<f32>,
    src_vy: &Array<f32>,
    dst_a: &mut Array<f32>,
    dst_rg: &mut Array<f32>,
    dst_vy: &mut Array<f32>,
    pu_scale: f32,
    w: u32,
    h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (w * h) as usize;
    if idx >= total {
        terminate!();
    }
    let wu = w as usize;
    let y = idx / wu;
    let x = idx - y * wu;
    let h_i = h as i32;
    let y_i = y as i32;
    let half = 6_i32;

    let k0 = f32::new(1.854_402_2e-2);
    let k1 = f32::new(3.416_694_2e-2);
    let k2 = f32::new(5.633_176_4e-2);
    let k3 = f32::new(8.310_854e-2);
    let k4 = f32::new(1.097_193e-1);
    let k5 = f32::new(1.296_180_3e-1);
    let k6 = f32::new(1.370_228_2e-1);
    let k7 = f32::new(1.296_180_3e-1);
    let k8 = f32::new(1.097_193e-1);
    let k9 = f32::new(8.310_854e-2);
    let k10 = f32::new(5.633_176_4e-2);
    let k11 = f32::new(3.416_694_2e-2);
    let k12 = f32::new(1.854_402_2e-2);

    let s0 = reflect_pu_idx(y_i - half, h_i);
    let s1 = reflect_pu_idx(y_i + 1 - half, h_i);
    let s2 = reflect_pu_idx(y_i + 2 - half, h_i);
    let s3 = reflect_pu_idx(y_i + 3 - half, h_i);
    let s4 = reflect_pu_idx(y_i + 4 - half, h_i);
    let s5 = reflect_pu_idx(y_i + 5 - half, h_i);
    let s6 = reflect_pu_idx(y_i + 6 - half, h_i);
    let s7 = reflect_pu_idx(y_i + 7 - half, h_i);
    let s8 = reflect_pu_idx(y_i + 8 - half, h_i);
    let s9 = reflect_pu_idx(y_i + 9 - half, h_i);
    let s10 = reflect_pu_idx(y_i + 10 - half, h_i);
    let s11 = reflect_pu_idx(y_i + 11 - half, h_i);
    let s12 = reflect_pu_idx(y_i + 12 - half, h_i);

    dst_a[idx] = (k0 * src_a[s0 * wu + x]
        + k1 * src_a[s1 * wu + x]
        + k2 * src_a[s2 * wu + x]
        + k3 * src_a[s3 * wu + x]
        + k4 * src_a[s4 * wu + x]
        + k5 * src_a[s5 * wu + x]
        + k6 * src_a[s6 * wu + x]
        + k7 * src_a[s7 * wu + x]
        + k8 * src_a[s8 * wu + x]
        + k9 * src_a[s9 * wu + x]
        + k10 * src_a[s10 * wu + x]
        + k11 * src_a[s11 * wu + x]
        + k12 * src_a[s12 * wu + x])
        * pu_scale;
    dst_rg[idx] = (k0 * src_rg[s0 * wu + x]
        + k1 * src_rg[s1 * wu + x]
        + k2 * src_rg[s2 * wu + x]
        + k3 * src_rg[s3 * wu + x]
        + k4 * src_rg[s4 * wu + x]
        + k5 * src_rg[s5 * wu + x]
        + k6 * src_rg[s6 * wu + x]
        + k7 * src_rg[s7 * wu + x]
        + k8 * src_rg[s8 * wu + x]
        + k9 * src_rg[s9 * wu + x]
        + k10 * src_rg[s10 * wu + x]
        + k11 * src_rg[s11 * wu + x]
        + k12 * src_rg[s12 * wu + x])
        * pu_scale;
    dst_vy[idx] = (k0 * src_vy[s0 * wu + x]
        + k1 * src_vy[s1 * wu + x]
        + k2 * src_vy[s2 * wu + x]
        + k3 * src_vy[s3 * wu + x]
        + k4 * src_vy[s4 * wu + x]
        + k5 * src_vy[s5 * wu + x]
        + k6 * src_vy[s6 * wu + x]
        + k7 * src_vy[s7 * wu + x]
        + k8 * src_vy[s8 * wu + x]
        + k9 * src_vy[s9 * wu + x]
        + k10 * src_vy[s10 * wu + x]
        + k11 * src_vy[s11 * wu + x]
        + k12 * src_vy[s12 * wu + x])
        * pu_scale;
}

/// Reflect index `i` into `[0, n)` for the PU blur. Matches
/// torchvision's `F.pad(..., mode='reflect')` (no edge repeat).
/// Inline-able from #[cube] bodies.
#[cube]
fn reflect_pu_idx(i: i32, n: i32) -> usize {
    let mut j = i;
    // Up to four folds cover the kernel-radius-6 range we use. For
    // bands ≥ 7 px (the only sizes where this kernel runs) one or
    // two folds suffice; the extra branches are cheap and keep
    // the function defined for all inputs.
    if j < 0 {
        j = -j;
    }
    if j >= n {
        j = 2 * n - 2 - j;
    }
    if j < 0 {
        j = -j;
    }
    if j >= n {
        j = 2 * n - 2 - j;
    }
    j as usize
}

/// cvvdp's `phase_uncertainty` for an entire band. If both
/// dimensions exceed `PU_PADSIZE = 6`, applies the σ=3 separable
/// Gaussian blur; otherwise just scales by `10^mask_c`.
#[must_use]
pub fn phase_uncertainty_band(m: &[f32], w: usize, h: usize) -> Vec<f32> {
    let scale = 10.0_f32.powf(MASK_C);
    if w > PU_PADSIZE && h > PU_PADSIZE {
        let blurred = gaussian_blur_sigma3(m, w, h);
        blurred.into_iter().map(|v| v * scale).collect()
    } else {
        m.iter().map(|v| v * scale).collect()
    }
}

/// Band-size threshold below which `phase_uncertainty` skips the
/// Gaussian blur. cvvdp computes `pu_padsize = int(pu_dilate * 2) = 6`.
pub const PU_PADSIZE: usize = 6;

/// Cross-channel mask pooling at one pixel: given `term[ch]` for
/// each of the 3 channels (where `term[ch] = safe_pow(|M_mm[ch]|, q[ch])`),
/// returns `m_per_out[cc] = sum_in XCM_3X3[in][cc] * term[in]`.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::{XCM_3X3, mask_pool_pixel};
///
/// // Zero input → zero output.
/// assert_eq!(mask_pool_pixel([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
///
/// // Unit-basis input [1, 0, 0] recovers row 0 of XCM_3X3.
/// let r0 = mask_pool_pixel([1.0, 0.0, 0.0]);
/// assert_eq!(r0, XCM_3X3[0]);
/// ```
#[inline]
#[must_use]
pub fn mask_pool_pixel(term: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for cc in 0..3 {
        out[cc] = XCM_3X3[0][cc] * term[0] + XCM_3X3[1][cc] * term[1] + XCM_3X3[2][cc] * term[2];
    }
    out
}

/// One-pixel masking for the cvvdp "mult-mutual" model with xchannel
/// masking on. Inputs are CSF-weighted contrasts (`T_p = T * S *
/// ch_gain`, similarly `R_p`). The CSF + ch_gain composition happens
/// in the caller; this function handles only the masking step.
///
/// Returns `D[cc]` for each of the 3 channels.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::kernels::masking::mult_mutual_pixel;
///
/// // T == R → no perceptible difference → D = [0, 0, 0].
/// let same = [0.5_f32, -0.3, 1.2];
/// assert_eq!(mult_mutual_pixel(same, same), [0.0, 0.0, 0.0]);
///
/// // Symmetric in arguments: f(T, R) == f(R, T) (min and abs both
/// // commute over the operands).
/// let t = [0.5_f32, -0.3, 1.2];
/// let r = [0.1_f32, 0.4, -0.8];
/// assert_eq!(mult_mutual_pixel(t, r), mult_mutual_pixel(r, t));
///
/// // D is always non-negative (safe_pow + clamp_diff_soft preserve sign).
/// for v in mult_mutual_pixel(t, r) {
///     assert!(v >= 0.0);
/// }
/// ```
#[must_use]
pub fn mult_mutual_pixel(t_p: [f32; 3], r_p: [f32; 3]) -> [f32; 3] {
    // Per-channel: M_mm = phase_uncertainty(min(|T_p|, |R_p|)).
    let m_mm = [
        phase_uncertainty_no_blur(t_p[0].abs().min(r_p[0].abs())),
        phase_uncertainty_no_blur(t_p[1].abs().min(r_p[1].abs())),
        phase_uncertainty_no_blur(t_p[2].abs().min(r_p[2].abs())),
    ];

    // term[in] = safe_pow(|M_mm[in]|, q[in]) — q is per input channel.
    let term = [
        safe_pow(m_mm[0].abs(), MASK_Q[0]),
        safe_pow(m_mm[1].abs(), MASK_Q[1]),
        safe_pow(m_mm[2].abs(), MASK_Q[2]),
    ];

    // M[cc] = cross-channel-pooled term.
    let m = mask_pool_pixel(term);

    // D[cc] = clamp_diffs(safe_pow(|T_p[cc] - R_p[cc]|, p) / (1 + M[cc])).
    let mut d = [0.0_f32; 3];
    for cc in 0..3 {
        let diff = (t_p[cc] - r_p[cc]).abs();
        let d_u = safe_pow(diff, MASK_P) / (1.0 + m[cc]);
        d[cc] = clamp_diff_soft(d_u);
    }
    d
}

/// Full-band masking for the cvvdp "mult-mutual" model with
/// xchannel masking on. Inputs are CSF-weighted contrasts including
/// `CH_GAIN` (so `t_p_per_ch[c][i] = T[c][i] * S[c] * CH_GAIN[c]`).
///
/// Applies the band-level `phase_uncertainty` (with the σ=3 blur
/// when the band is large enough) before the cross-channel pool,
/// matching cvvdp exactly. Returns per-channel per-pixel D values.
#[must_use]
pub fn mult_mutual_band(
    t_p_per_ch: &[Vec<f32>; 3],
    r_p_per_ch: &[Vec<f32>; 3],
    w: usize,
    h: usize,
) -> [Vec<f32>; 3] {
    let n = w * h;
    debug_assert_eq!(t_p_per_ch[0].len(), n);

    // Step 1: M_mm[ch][i] = min(|T_p[ch][i]|, |R_p[ch][i]|).
    let m_mm_raw: [Vec<f32>; 3] = [
        (0..n)
            .map(|i| t_p_per_ch[0][i].abs().min(r_p_per_ch[0][i].abs()))
            .collect(),
        (0..n)
            .map(|i| t_p_per_ch[1][i].abs().min(r_p_per_ch[1][i].abs()))
            .collect(),
        (0..n)
            .map(|i| t_p_per_ch[2][i].abs().min(r_p_per_ch[2][i].abs()))
            .collect(),
    ];

    // Step 2: phase_uncertainty per channel (blur if band large).
    let m_mm: [Vec<f32>; 3] = [
        phase_uncertainty_band(&m_mm_raw[0], w, h),
        phase_uncertainty_band(&m_mm_raw[1], w, h),
        phase_uncertainty_band(&m_mm_raw[2], w, h),
    ];

    // Step 3: term[ch][i] = safe_pow(|M_mm[ch][i]|, q[ch]).
    let term: [Vec<f32>; 3] = [
        m_mm[0]
            .iter()
            .map(|v| safe_pow(v.abs(), MASK_Q[0]))
            .collect(),
        m_mm[1]
            .iter()
            .map(|v| safe_pow(v.abs(), MASK_Q[1]))
            .collect(),
        m_mm[2]
            .iter()
            .map(|v| safe_pow(v.abs(), MASK_Q[2]))
            .collect(),
    ];

    // Step 4: cross-channel pool per pixel + masked diff.
    let mut d: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let term_i = [term[0][i], term[1][i], term[2][i]];
        let m_pool = mask_pool_pixel(term_i);
        for cc in 0..3 {
            let diff = (t_p_per_ch[cc][i] - r_p_per_ch[cc][i]).abs();
            let d_u = safe_pow(diff, MASK_P) / (1.0 + m_pool[cc]);
            d[cc][i] = clamp_diff_soft(d_u);
        }
    }
    d
}

/// Per-pixel `M_mm_raw[c] = min(|T_p[c]|, |R_p[c]|)` for all 3
/// channels. First step of cvvdp's `mult-mutual` masking — the raw
/// per-channel masker before phase-uncertainty (either the σ=3
/// Gaussian blur for large bands or just `* 10^MASK_C` for small
/// bands).
///
/// Used by callers that need to chain `min_abs → pu_blur_h →
/// pu_blur_v → mult_mutual_3ch_with_blurred` on bands larger than
/// `PU_PADSIZE`. For small bands the no-blur kernel inlines this
/// step.
#[cube(launch)]
pub fn min_abs_3ch_kernel(
    t_p_a: &Array<f32>,
    t_p_rg: &Array<f32>,
    t_p_vy: &Array<f32>,
    r_p_a: &Array<f32>,
    r_p_rg: &Array<f32>,
    r_p_vy: &Array<f32>,
    m_mm_a: &mut Array<f32>,
    m_mm_rg: &mut Array<f32>,
    m_mm_vy: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let zero = f32::new(0.0);

    let ta = t_p_a[idx];
    let abs_ta = if ta < zero { -ta } else { ta };
    let ra = r_p_a[idx];
    let abs_ra = if ra < zero { -ra } else { ra };
    m_mm_a[idx] = if abs_ta < abs_ra { abs_ta } else { abs_ra };

    let trg = t_p_rg[idx];
    let abs_trg = if trg < zero { -trg } else { trg };
    let rrg = r_p_rg[idx];
    let abs_rrg = if rrg < zero { -rrg } else { rrg };
    m_mm_rg[idx] = if abs_trg < abs_rrg { abs_trg } else { abs_rrg };

    let tvy = t_p_vy[idx];
    let abs_tvy = if tvy < zero { -tvy } else { tvy };
    let rvy = r_p_vy[idx];
    let abs_rvy = if rvy < zero { -rvy } else { rvy };
    m_mm_vy[idx] = if abs_tvy < abs_rvy { abs_tvy } else { abs_rvy };
}

/// Full mult-mutual + xchannel masking for a 3-channel band, no
/// phase-uncertainty blur. Mirrors `mult_mutual_band` for the small-
/// band branch (`w ≤ PU_PADSIZE || h ≤ PU_PADSIZE`):
///
/// ```text
/// M_mm[c]  = min(|T_p[c]|, |R_p[c]|) * 10^mask_c
/// term[c]  = safe_pow(|M_mm[c]|, q[c])
/// M[c]     = sum_in xcm[in, c] * term[in]
/// D[c]     = clamp_diffs(safe_pow(|T_p[c] - R_p[c]|, p) / (1 + M[c]))
/// ```
///
/// Inputs `t_p_*`, `r_p_*` are CSF-weighted contrasts already
/// scaled by `S * CH_GAIN`. Outputs `d_*` are per-pixel post-clamp
/// masked differences.
///
/// Constants `MASK_P`, `MASK_Q`, `MASK_C`, `D_MAX`, `XCM_3X3` are
/// baked at compile time via `f32::new` literals. Same numeric
/// outputs as `mult_mutual_band` when called with the same inputs
/// at a small band size.
///
/// For bands larger than `PU_PADSIZE = 6`, callers run the σ=3
/// Gaussian blur first (production uses the 3-channel fused
/// `pu_blur_h_3ch_kernel` + `pu_blur_v_3ch_scaled_kernel` — the
/// v-pass folds the `* 10^MASK_C` post-scale into its output) and
/// feed the blurred M_mm tensor to
/// `mult_mutual_3ch_with_blurred_kernel`. This no-blur form covers
/// the deepest pyramid levels where the blur is skipped.
#[cube(launch)]
pub fn mult_mutual_3ch_no_blur_kernel(
    t_p_a: &Array<f32>,
    t_p_rg: &Array<f32>,
    t_p_vy: &Array<f32>,
    r_p_a: &Array<f32>,
    r_p_rg: &Array<f32>,
    r_p_vy: &Array<f32>,
    d_a: &mut Array<f32>,
    d_rg: &mut Array<f32>,
    d_vy: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    // cvvdp v0.5.4 constants — bake to match host scalar exactly.
    let mask_p = f32::new(2.264_355_2);
    let mask_q_0 = f32::new(1.302_622_7);
    let mask_q_1 = f32::new(2.888_590_8);
    let mask_q_2 = f32::new(3.680_771_3);
    let pu_scale = f32::new(0.160_188_4); // 10^MASK_C with MASK_C = -0.79549712
    let d_max_lin = f32::new(366.732_25); // 10^D_MAX with D_MAX = 2.5642455

    // XCM_3X3 (row-major: [in][out]) baked as scalar consts.
    let xcm_00 = f32::new(0.876_968);
    let xcm_01 = f32::new(0.016_103_15);
    let xcm_02 = f32::new(0.050_159_38);
    let xcm_10 = f32::new(5.918_792);
    let xcm_11 = f32::new(1.269_323);
    let xcm_12 = f32::new(0.152_080_92);
    let xcm_20 = f32::new(14.041_055);
    let xcm_21 = f32::new(0.498_209_6);
    let xcm_22 = f32::new(0.697_756_55);

    let eps = f32::new(1e-5);

    let t_a = t_p_a[idx];
    let t_rg = t_p_rg[idx];
    let t_vy = t_p_vy[idx];
    let r_a = r_p_a[idx];
    let r_rg = r_p_rg[idx];
    let r_vy = r_p_vy[idx];

    let abs_t_a = if t_a < f32::new(0.0) { -t_a } else { t_a };
    let abs_t_rg = if t_rg < f32::new(0.0) { -t_rg } else { t_rg };
    let abs_t_vy = if t_vy < f32::new(0.0) { -t_vy } else { t_vy };
    let abs_r_a = if r_a < f32::new(0.0) { -r_a } else { r_a };
    let abs_r_rg = if r_rg < f32::new(0.0) { -r_rg } else { r_rg };
    let abs_r_vy = if r_vy < f32::new(0.0) { -r_vy } else { r_vy };

    // M_mm = min(|T_p|, |R_p|) * 10^mask_c.
    let mm_a = if abs_t_a < abs_r_a { abs_t_a } else { abs_r_a };
    let mm_rg = if abs_t_rg < abs_r_rg {
        abs_t_rg
    } else {
        abs_r_rg
    };
    let mm_vy = if abs_t_vy < abs_r_vy {
        abs_t_vy
    } else {
        abs_r_vy
    };
    let m_mm_a = mm_a * pu_scale;
    let m_mm_rg = mm_rg * pu_scale;
    let m_mm_vy = mm_vy * pu_scale;

    // term[c] = safe_pow(|M_mm[c]|, q[c]). safe_pow(x, p) = (x+eps)^p - eps^p.
    let term_a = f32::powf(m_mm_a + eps, mask_q_0) - f32::powf(eps, mask_q_0);
    let term_rg = f32::powf(m_mm_rg + eps, mask_q_1) - f32::powf(eps, mask_q_1);
    let term_vy = f32::powf(m_mm_vy + eps, mask_q_2) - f32::powf(eps, mask_q_2);

    // M[c] = sum_in xcm[in, c] * term[in].
    let m_a_pool = xcm_00 * term_a + xcm_10 * term_rg + xcm_20 * term_vy;
    let m_rg_pool = xcm_01 * term_a + xcm_11 * term_rg + xcm_21 * term_vy;
    let m_vy_pool = xcm_02 * term_a + xcm_12 * term_rg + xcm_22 * term_vy;

    // D[c] = clamp_diffs(safe_pow(|T_p[c] - R_p[c]|, p) / (1 + M[c])).
    let diff_a = t_a - r_a;
    let abs_diff_a = if diff_a < f32::new(0.0) {
        -diff_a
    } else {
        diff_a
    };
    let diff_rg = t_rg - r_rg;
    let abs_diff_rg = if diff_rg < f32::new(0.0) {
        -diff_rg
    } else {
        diff_rg
    };
    let diff_vy = t_vy - r_vy;
    let abs_diff_vy = if diff_vy < f32::new(0.0) {
        -diff_vy
    } else {
        diff_vy
    };

    let sp_a = f32::powf(abs_diff_a + eps, mask_p) - f32::powf(eps, mask_p);
    let sp_rg = f32::powf(abs_diff_rg + eps, mask_p) - f32::powf(eps, mask_p);
    let sp_vy = f32::powf(abs_diff_vy + eps, mask_p) - f32::powf(eps, mask_p);

    let d_u_a = sp_a / (f32::new(1.0) + m_a_pool);
    let d_u_rg = sp_rg / (f32::new(1.0) + m_rg_pool);
    let d_u_vy = sp_vy / (f32::new(1.0) + m_vy_pool);

    d_a[idx] = d_max_lin * d_u_a / (d_max_lin + d_u_a);
    d_rg[idx] = d_max_lin * d_u_rg / (d_max_lin + d_u_rg);
    d_vy[idx] = d_max_lin * d_u_vy / (d_max_lin + d_u_vy);
}

/// Companion to `min_abs_3ch_kernel` for the large-band path. Takes
/// the pre-blurred + pre-scaled `M_mm` (i.e. `pu_blur(min(|T_p|,
/// |R_p|)) * 10^MASK_C` per channel; caller chains `min_abs →
/// pu_blur_h → pu_blur_v → multiply-by-10^MASK_C`) and finishes the
/// mult-mutual + xchannel + soft-clamp formula. Same math as
/// `mult_mutual_3ch_no_blur_kernel` from step 3 onward (`term[c] =
/// safe_pow(|M_mm|, q[c])`, cross-channel pool, masked diff with
/// `safe_pow(|T_p-R_p|, p) / (1+M)`, soft clamp).
///
/// 9 input arrays (`T_p\[3\]`, `R_p\[3\]`, `M_mm_blurred_scaled\[3\]`)
/// and 3 output arrays (`D\[3\]`). Same constants baked.
#[cube(launch)]
pub fn mult_mutual_3ch_with_blurred_kernel(
    t_p_a: &Array<f32>,
    t_p_rg: &Array<f32>,
    t_p_vy: &Array<f32>,
    r_p_a: &Array<f32>,
    r_p_rg: &Array<f32>,
    r_p_vy: &Array<f32>,
    m_mm_a: &Array<f32>,
    m_mm_rg: &Array<f32>,
    m_mm_vy: &Array<f32>,
    d_a: &mut Array<f32>,
    d_rg: &mut Array<f32>,
    d_vy: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }

    let mask_p = f32::new(2.264_355_2);
    let mask_q_0 = f32::new(1.302_622_7);
    let mask_q_1 = f32::new(2.888_590_8);
    let mask_q_2 = f32::new(3.680_771_3);
    let d_max_lin = f32::new(366.732_25);

    let xcm_00 = f32::new(0.876_968);
    let xcm_01 = f32::new(0.016_103_15);
    let xcm_02 = f32::new(0.050_159_38);
    let xcm_10 = f32::new(5.918_792);
    let xcm_11 = f32::new(1.269_323);
    let xcm_12 = f32::new(0.152_080_92);
    let xcm_20 = f32::new(14.041_055);
    let xcm_21 = f32::new(0.498_209_6);
    let xcm_22 = f32::new(0.697_756_55);

    let eps = f32::new(1e-5);

    let m_a = m_mm_a[idx];
    let m_rg = m_mm_rg[idx];
    let m_vy = m_mm_vy[idx];

    let abs_m_a = if m_a < f32::new(0.0) { -m_a } else { m_a };
    let abs_m_rg = if m_rg < f32::new(0.0) { -m_rg } else { m_rg };
    let abs_m_vy = if m_vy < f32::new(0.0) { -m_vy } else { m_vy };

    let term_a = f32::powf(abs_m_a + eps, mask_q_0) - f32::powf(eps, mask_q_0);
    let term_rg = f32::powf(abs_m_rg + eps, mask_q_1) - f32::powf(eps, mask_q_1);
    let term_vy = f32::powf(abs_m_vy + eps, mask_q_2) - f32::powf(eps, mask_q_2);

    let m_a_pool = xcm_00 * term_a + xcm_10 * term_rg + xcm_20 * term_vy;
    let m_rg_pool = xcm_01 * term_a + xcm_11 * term_rg + xcm_21 * term_vy;
    let m_vy_pool = xcm_02 * term_a + xcm_12 * term_rg + xcm_22 * term_vy;

    let t_a = t_p_a[idx];
    let t_rg = t_p_rg[idx];
    let t_vy = t_p_vy[idx];
    let r_a = r_p_a[idx];
    let r_rg = r_p_rg[idx];
    let r_vy = r_p_vy[idx];

    let diff_a = t_a - r_a;
    let abs_diff_a = if diff_a < f32::new(0.0) {
        -diff_a
    } else {
        diff_a
    };
    let diff_rg = t_rg - r_rg;
    let abs_diff_rg = if diff_rg < f32::new(0.0) {
        -diff_rg
    } else {
        diff_rg
    };
    let diff_vy = t_vy - r_vy;
    let abs_diff_vy = if diff_vy < f32::new(0.0) {
        -diff_vy
    } else {
        diff_vy
    };

    let sp_a = f32::powf(abs_diff_a + eps, mask_p) - f32::powf(eps, mask_p);
    let sp_rg = f32::powf(abs_diff_rg + eps, mask_p) - f32::powf(eps, mask_p);
    let sp_vy = f32::powf(abs_diff_vy + eps, mask_p) - f32::powf(eps, mask_p);

    let d_u_a = sp_a / (f32::new(1.0) + m_a_pool);
    let d_u_rg = sp_rg / (f32::new(1.0) + m_rg_pool);
    let d_u_vy = sp_vy / (f32::new(1.0) + m_vy_pool);

    d_a[idx] = d_max_lin * d_u_a / (d_max_lin + d_u_a);
    d_rg[idx] = d_max_lin * d_u_rg / (d_max_lin + d_u_rg);
    d_vy[idx] = d_max_lin * d_u_vy / (d_max_lin + d_u_vy);
}
