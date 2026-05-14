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
//! Phase-uncertainty Gaussian blur (sigma = `pu_dilate = 3`) is
//! **out of scope for this tick**. cvvdp skips the blur for bands
//! smaller than `pu_padsize = 6`, so small-band parity is already
//! exact; the blur path lands when whole-image parity (large
//! coarse-level bands at standard_4k resolution) is wired through.

use cubecl::prelude::*;

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
#[inline]
pub fn safe_pow(x: f32, p: f32) -> f32 {
    let eps: f32 = 1e-5;
    (x + eps).powf(p) - eps.powf(p)
}

/// Soft clamp matching cvvdp v0.5.4's `dclamp_type = "soft"`:
/// `D_max * D / (D_max + D)` with `D_max = 10^d_max`.
#[inline]
pub fn clamp_diff_soft(d: f32) -> f32 {
    let d_max = 10.0_f32.powf(D_MAX);
    d_max * d / (d_max + d)
}

/// Phase uncertainty for the "small band, no blur" path: multiply
/// by `10^mask_c`. The blur path (Gaussian sigma = 3 for bands
/// larger than 6 px) is not yet ported.
#[inline]
pub fn phase_uncertainty_no_blur(m: f32) -> f32 {
    m * 10.0_f32.powf(MASK_C)
}

/// Cross-channel mask pooling at one pixel: given `term[ch]` for
/// each of the 3 channels (where `term[ch] = safe_pow(|M_mm[ch]|, q[ch])`),
/// returns `m_per_out[cc] = sum_in XCM_3X3[in][cc] * term[in]`.
#[inline]
pub fn mask_pool_pixel(term: [f32; 3]) -> [f32; 3] {
    let mut out = [0.0_f32; 3];
    for cc in 0..3 {
        out[cc] = XCM_3X3[0][cc] * term[0]
            + XCM_3X3[1][cc] * term[1]
            + XCM_3X3[2][cc] * term[2];
    }
    out
}

/// One-pixel masking for the cvvdp "mult-mutual" model with xchannel
/// masking on. Inputs are CSF-weighted contrasts (`T_p = T * S *
/// ch_gain`, similarly `R_p`). The CSF + ch_gain composition happens
/// in the caller; this function handles only the masking step.
///
/// Returns `D[cc]` for each of the 3 channels.
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

/// Per-pixel masked difference. **Stub**.
#[cube(launch)]
#[allow(unused_variables)]
pub fn masked_diff_kernel(
    ref_band: &Array<f32>,
    dist_band: &Array<f32>,
    masker: &Array<f32>,
    out: &mut Array<f32>,
    p: f32,
    q: f32,
    k: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    out[idx] = 0.0;
}
