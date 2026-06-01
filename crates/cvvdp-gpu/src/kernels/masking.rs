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

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;

// Phase 8c.1-C: scalar items (CH_GAIN / MASK_P / MASK_Q / MASK_C /
// D_MAX / XCM_3X3 / PU_BLUR_KERNEL_1D / PU_PADSIZE constants plus the
// safe_pow / clamp_diff_soft / phase_uncertainty_no_blur /
// gaussian_blur_sigma3 / phase_uncertainty_band / mask_pool_pixel /
// mult_mutual_pixel / mult_mutual_band host-scalar helpers) live in
// `cvvdp::kernels::masking` so the CPU crate owns the canonical
// scalar implementation. Re-export the surface so existing
// `cvvdp_gpu::kernels::masking::*` callsites resolve unchanged.
//
// The cube-macro `#[cube(launch)]` kernels below use inline
// `f32::new(...)` literals for the masking constants (MASK_P, MASK_Q,
// MASK_C-derived 10^c scale, D_MAX-derived 10^d_max, XCM_3X3 matrix
// elements, PU_BLUR_KERNEL_1D 13-tap blur coefficients) rather than
// referencing the moved constants by name in the cube body. The
// scalar helper `reflect_idx_for_blur` was private and only used by
// `gaussian_blur_sigma3` (also moved) — no callers remain. No
// cube-macro name-resolution interaction.
pub use cvvdp::kernels::masking::{
    CH_GAIN, D_MAX, MASK_C, MASK_P, MASK_Q, PU_BLUR_KERNEL_1D, PU_PADSIZE, XCM_3X3,
    clamp_diff_soft, gaussian_blur_sigma3, mask_pool_pixel, mult_mutual_band, mult_mutual_pixel,
    phase_uncertainty_band, phase_uncertainty_no_blur, safe_pow,
};

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

/// Strip-aware sibling of [`pu_blur_h_3ch_kernel`] for Mode E
/// Phase 3 **API uniformity**.
///
/// The σ=3 Gaussian blur's horizontal pass only reflects along the
/// X axis (`reflect_pu_idx(x_i + t - half, w_i)`) — it does no
/// Y-axis arithmetic that depends on the logical image height. Strip
/// processing slices Y, not X, so an H-blur dispatched on a strip
/// buffer produces bit-exact body rows from a `w × h` view of the
/// full plane without any reflection-target change.
///
/// `body_offset_y` and `logical_h` are accepted **for API
/// uniformity** with [`pu_blur_v_3ch_scaled_strip_aware_kernel`] so
/// the future strip walker can dispatch both kernels with the same
/// call shape (and so callers don't have to maintain two different
/// dispatch-arg helpers). They are **unused** in the kernel body —
/// the existing `(w, h)` parameterisation already covers the strip
/// buffer's footprint.
///
/// When called with any `body_offset_y, logical_h`, this is bit-
/// exact identical to [`pu_blur_h_3ch_kernel`] (the extra params
/// don't enter the per-pixel arithmetic).
#[cube(launch)]
pub fn pu_blur_h_3ch_strip_aware_kernel(
    src_a: &Array<f32>,
    src_rg: &Array<f32>,
    src_vy: &Array<f32>,
    dst_a: &mut Array<f32>,
    dst_rg: &mut Array<f32>,
    dst_vy: &mut Array<f32>,
    w: u32,
    h: u32,
    body_offset_y: u32,
    logical_h: u32,
) {
    // Body offset / logical height are deliberately unused — H-blur
    // does not touch the Y axis. See doc comment above.
    let _ = body_offset_y;
    let _ = logical_h;

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

/// Strip-aware sibling of [`pu_blur_v_3ch_scaled_kernel`] for Mode E
/// Phase 3. Same separable σ=3 Gaussian blur with `* pu_scale` post-
/// multiply, but reflects the Y-axis taps against the **logical**
/// image height instead of the strip-buffer height, so a strip
/// dispatch on a sub-band of the full image produces bit-exact body
/// rows against a full-image dispatch.
///
/// Buffer model:
/// - `src_*` and `dst_*` are **strip-buffer-sized** 3-channel f32
///   planes of shape `w × h` (where `h` is the strip-buffer height,
///   typically `body_h + halo_top + halo_bot`).
/// - `body_offset_y` is the **global y of strip-buffer row 0**
///   (i.e. the global y of the topmost halo row, NOT the body's
///   first row). For full-image dispatch pass `body_offset_y = 0`.
/// - `logical_h` is the height of the underlying logical image (the
///   reflection target). For full-image dispatch pass
///   `logical_h = h`.
///
/// Per output thread `idx` (1D flat index into the `w × h` strip
/// buffer):
/// ```text
/// (x, y_strip) = (idx % w, idx / w)
/// y_global     = y_strip + body_offset_y
/// for tap t in 0..13:
///     g_t      = y_global + (t - 6)           // global tap index
///     ref_g_t  = reflect_pu_idx(g_t, logical_h) // reflect against logical edges
///     local_t  = ref_g_t - body_offset_y       // strip-buffer row
///     dst[idx] += k[t] * src[local_t * w + x]
/// dst[idx] *= pu_scale
/// ```
///
/// Correctness gate: `local_t` MUST lie in `[0, h)` for every tap
/// the body rows touch. The caller (strip walker) is responsible
/// for sizing halos so reflection lands inside the strip buffer.
/// If `local_t` lands out of range, the read is OOB — this is a
/// strip-coverage bug in the caller, not a kernel bug.
///
/// When called with `body_offset_y = 0, logical_h = h`, this is
/// bit-exact identical to [`pu_blur_v_3ch_scaled_kernel`] (the
/// reflect target collapses to the buffer height and the offset is
/// a no-op).
#[cube(launch)]
pub fn pu_blur_v_3ch_scaled_strip_aware_kernel(
    src_a: &Array<f32>,
    src_rg: &Array<f32>,
    src_vy: &Array<f32>,
    dst_a: &mut Array<f32>,
    dst_rg: &mut Array<f32>,
    dst_vy: &mut Array<f32>,
    pu_scale: f32,
    w: u32,
    h: u32,
    body_offset_y: u32,
    logical_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (w * h) as usize;
    if idx >= total {
        terminate!();
    }
    let wu = w as usize;
    let y_strip = idx / wu;
    let x = idx - y_strip * wu;

    let logical_h_i = logical_h as i32;
    let body_off_i = body_offset_y as i32;
    let y_global_i = y_strip as i32 + body_off_i;
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

    // Reflect each tap's global y against logical_h, then translate
    // back to a strip-buffer row by subtracting body_offset_y.
    // (reflect_pu_idx returns usize; cast to i32 for the subtract.)
    let s0 = (reflect_pu_idx(y_global_i - half, logical_h_i) as i32 - body_off_i) as usize;
    let s1 = (reflect_pu_idx(y_global_i + 1 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s2 = (reflect_pu_idx(y_global_i + 2 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s3 = (reflect_pu_idx(y_global_i + 3 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s4 = (reflect_pu_idx(y_global_i + 4 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s5 = (reflect_pu_idx(y_global_i + 5 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s6 = (reflect_pu_idx(y_global_i + 6 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s7 = (reflect_pu_idx(y_global_i + 7 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s8 = (reflect_pu_idx(y_global_i + 8 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s9 = (reflect_pu_idx(y_global_i + 9 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s10 = (reflect_pu_idx(y_global_i + 10 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s11 = (reflect_pu_idx(y_global_i + 11 - half, logical_h_i) as i32 - body_off_i) as usize;
    let s12 = (reflect_pu_idx(y_global_i + 12 - half, logical_h_i) as i32 - body_off_i) as usize;

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
