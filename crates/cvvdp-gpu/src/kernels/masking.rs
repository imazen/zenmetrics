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
/// by `10^mask_c`. cvvdp uses this branch when the band's smallest
/// dimension is ≤ `PU_PADSIZE`.
#[inline]
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
/// hot paths. Returns a `w × h` Vec<f32>. Reflect padding at the
/// boundary, matching torchvision.
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

/// cvvdp's `phase_uncertainty` for an entire band. If both
/// dimensions exceed `PU_PADSIZE = 6`, applies the σ=3 separable
/// Gaussian blur; otherwise just scales by `10^mask_c`.
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

/// Full-band masking for the cvvdp "mult-mutual" model with
/// xchannel masking on. Inputs are CSF-weighted contrasts including
/// `CH_GAIN` (so `t_p_per_ch[c][i] = T[c][i] * S[c] * CH_GAIN[c]`).
///
/// Applies the band-level `phase_uncertainty` (with the σ=3 blur
/// when the band is large enough) before the cross-channel pool,
/// matching cvvdp exactly. Returns per-channel per-pixel D values.
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
        m_mm[0].iter().map(|v| safe_pow(v.abs(), MASK_Q[0])).collect(),
        m_mm[1].iter().map(|v| safe_pow(v.abs(), MASK_Q[1])).collect(),
        m_mm[2].iter().map(|v| safe_pow(v.abs(), MASK_Q[2])).collect(),
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
