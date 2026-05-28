//! Scalar contrast-masking helpers — cvvdp v0.5.4's `mult-mutual`
//! model with cross-channel masking on.
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::masking` so
//! the CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths. GPU-side `#[cube(launch)]`
//! kernels remain in `cvvdp-gpu::kernels::masking`.

use alloc::vec;
use alloc::vec::Vec;

/// Per-channel gain `[1, 1.45, 1]` applied to `T_p` and `R_p` BEFORE masking.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::CH_GAIN;
///
/// assert_eq!(CH_GAIN.len(), 3);
/// assert_eq!(CH_GAIN[0], 1.0);
/// assert_eq!(CH_GAIN[1], 1.45);
/// assert_eq!(CH_GAIN[2], 1.0);
/// ```
pub const CH_GAIN: [f32; 3] = [1.0, 1.45, 1.0];

/// `mask_p` exponent from cvvdp v0.5.4.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{MASK_P, MASK_Q, MASK_C, D_MAX};
///
/// assert!(MASK_P > 0.0);
/// assert!((MASK_P - 2.264_355_2).abs() < 1e-6);
///
/// assert_eq!(MASK_Q.len(), 3);
/// assert!(MASK_Q[0] < MASK_Q[1] && MASK_Q[1] < MASK_Q[2]);
///
/// assert!(MASK_C < 0.0);
/// assert!(10.0_f32.powf(MASK_C) < 1.0);
///
/// assert!(D_MAX > 0.0);
/// assert!(10.0_f32.powf(D_MAX) > 100.0);
/// ```
pub const MASK_P: f32 = 2.264_355_2;

/// `mask_q[0..3]` for the still-image 3-channel pipeline.
pub const MASK_Q: [f32; 3] = [1.302_622_7, 2.888_590_8, 3.680_771_3];

/// `mask_c` for phase-uncertainty scaling: applied as `10^mask_c`.
pub const MASK_C: f32 = -0.795_497_12;

/// `d_max` for soft clamp ceiling: applied as `10^d_max`.
pub const D_MAX: f32 = 2.564_245_5;

/// Cross-channel masking 3×3 matrix derived from cvvdp v0.5.4's `xcm_weights`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::XCM_3X3;
///
/// assert_eq!(XCM_3X3.len(), 3);
/// for row in &XCM_3X3 {
///     assert_eq!(row.len(), 3);
///     for &v in row {
///         assert!(v > 0.0 && v.is_finite());
///     }
/// }
/// assert!(XCM_3X3[0][0] > 0.5);
/// ```
pub const XCM_3X3: [[f32; 3]; 3] = [
    [0.876_968, 0.016_103_15, 0.050_159_38],
    [5.918_792, 1.269_323, 0.152_080_92],
    [14.041_055, 0.498_209_6, 0.697_756_55],
];

/// cvvdp's `safe_pow(x, p) = (x + eps)^p - eps^p`, with `eps = 1e-5`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::safe_pow;
///
/// assert_eq!(safe_pow(0.0, 2.0), 0.0);
/// assert_eq!(safe_pow(0.0, 0.5), 0.0);
///
/// let v = safe_pow(2.0, 2.0);
/// assert!((v - 4.0).abs() < 1e-3, "safe_pow(2, 2) = {v}, expected ≈ 4");
///
/// assert!(safe_pow(1.0, 2.5) < safe_pow(2.0, 2.5));
/// ```
#[inline]
#[must_use]
pub fn safe_pow(x: f32, p: f32) -> f32 {
    let eps: f32 = 1e-5;
    (x + eps).powf(p) - eps.powf(p)
}

/// Soft clamp matching cvvdp v0.5.4's `dclamp_type = "soft"`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{D_MAX, clamp_diff_soft};
///
/// assert_eq!(clamp_diff_soft(0.0), 0.0);
///
/// let d_max = 10.0_f32.powf(D_MAX);
/// let half = clamp_diff_soft(d_max);
/// assert!((half - d_max / 2.0).abs() / d_max < 1e-5);
///
/// assert!(clamp_diff_soft(1e9) < d_max);
/// ```
#[inline]
#[must_use]
pub fn clamp_diff_soft(d: f32) -> f32 {
    let d_max = 10.0_f32.powf(D_MAX);
    d_max * d / (d_max + d)
}

/// Phase uncertainty for the "small band, no blur" path: multiply by `10^mask_c`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{MASK_C, phase_uncertainty_no_blur};
///
/// let scale = 10.0_f32.powf(MASK_C);
/// assert_eq!(phase_uncertainty_no_blur(1.0), scale);
/// assert_eq!(phase_uncertainty_no_blur(0.0), 0.0);
///
/// assert!(scale > 0.15 && scale < 0.17);
/// ```
#[inline]
#[must_use]
pub fn phase_uncertainty_no_blur(m: f32) -> f32 {
    m * 10.0_f32.powf(MASK_C)
}

/// 1D Gaussian kernel for phase-uncertainty blur.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::PU_BLUR_KERNEL_1D;
///
/// assert_eq!(PU_BLUR_KERNEL_1D.len(), 13);
/// for i in 0..6 {
///     assert_eq!(PU_BLUR_KERNEL_1D[i].to_bits(), PU_BLUR_KERNEL_1D[12 - i].to_bits());
/// }
/// let sum: f32 = PU_BLUR_KERNEL_1D.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
///
/// let center = PU_BLUR_KERNEL_1D[6];
/// let edge = PU_BLUR_KERNEL_1D[0];
/// assert!(center > edge * 5.0);
/// ```
#[rustfmt::skip]
pub const PU_BLUR_KERNEL_1D: [f32; 13] = [
    1.854_402_2e-2, 3.416_694_2e-2, 5.633_176_4e-2, 8.310_854e-2,
    1.097_193e-1,   1.296_180_3e-1, 1.370_228_2e-1, 1.296_180_3e-1,
    1.097_193e-1,   8.310_854e-2,   5.633_176_4e-2, 3.416_694_2e-2,
    1.854_402_2e-2,
];

/// Reflect index `i` into `[0, n)` for the 13-tap PU blur.
/// Matches torchvision's `F.pad(..., mode='reflect')` (no edge repeat).
fn reflect_idx_for_blur(i: isize, n: usize) -> usize {
    let n_i = n as isize;
    debug_assert!(n_i > 0);
    let mut j = i;
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

/// Apply the σ=3 separable Gaussian blur from cvvdp's `phase_uncertainty`.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::gaussian_blur_sigma3;
///
/// let src = vec![0.0_f32; 16 * 16];
/// let out = gaussian_blur_sigma3(&src, 16, 16);
/// assert_eq!(out.len(), 256);
///
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
    let half = 6_isize;

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

/// Band-size threshold below which `phase_uncertainty` skips the Gaussian blur.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{PU_PADSIZE, phase_uncertainty_band};
///
/// assert_eq!(PU_PADSIZE, 6);
///
/// let small = vec![1.0_f32; 36];
/// let out_small = phase_uncertainty_band(&small, 6, 6);
/// assert_eq!(out_small.len(), 36);
/// ```
pub const PU_PADSIZE: usize = 6;

/// cvvdp's `phase_uncertainty` for an entire band.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{MASK_C, phase_uncertainty_band};
///
/// let scale = 10.0_f32.powf(MASK_C);
/// let small = vec![1.0_f32, 2.0, 3.0, 4.0];
/// let out = phase_uncertainty_band(&small, 2, 2);
/// for (i, &v) in out.iter().enumerate() {
///     assert!((v - small[i] * scale).abs() < 1e-6);
/// }
///
/// let large = vec![1.0_f32; 8 * 8];
/// let out_large = phase_uncertainty_band(&large, 8, 8);
/// assert_eq!(out_large.len(), 64);
/// ```
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

/// Cross-channel mask pooling at one pixel.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::{XCM_3X3, mask_pool_pixel};
///
/// assert_eq!(mask_pool_pixel([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
///
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

/// One-pixel masking for the cvvdp "mult-mutual" model.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::mult_mutual_pixel;
///
/// let same = [0.5_f32, -0.3, 1.2];
/// assert_eq!(mult_mutual_pixel(same, same), [0.0, 0.0, 0.0]);
///
/// let t = [0.5_f32, -0.3, 1.2];
/// let r = [0.1_f32, 0.4, -0.8];
/// assert_eq!(mult_mutual_pixel(t, r), mult_mutual_pixel(r, t));
///
/// for v in mult_mutual_pixel(t, r) {
///     assert!(v >= 0.0);
/// }
/// ```
#[must_use]
pub fn mult_mutual_pixel(t_p: [f32; 3], r_p: [f32; 3]) -> [f32; 3] {
    let m_mm = [
        phase_uncertainty_no_blur(t_p[0].abs().min(r_p[0].abs())),
        phase_uncertainty_no_blur(t_p[1].abs().min(r_p[1].abs())),
        phase_uncertainty_no_blur(t_p[2].abs().min(r_p[2].abs())),
    ];

    let term = [
        safe_pow(m_mm[0].abs(), MASK_Q[0]),
        safe_pow(m_mm[1].abs(), MASK_Q[1]),
        safe_pow(m_mm[2].abs(), MASK_Q[2]),
    ];

    let m = mask_pool_pixel(term);

    let mut d = [0.0_f32; 3];
    for cc in 0..3 {
        let diff = (t_p[cc] - r_p[cc]).abs();
        let d_u = safe_pow(diff, MASK_P) / (1.0 + m[cc]);
        d[cc] = clamp_diff_soft(d_u);
    }
    d
}

/// Full-band masking for the cvvdp "mult-mutual" model.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::masking::mult_mutual_band;
///
/// let (w, h) = (8_usize, 8_usize);
/// let plane: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.01).collect();
/// let t = [plane.clone(), plane.clone(), plane.clone()];
/// let d = mult_mutual_band(&t, &t, w, h);
/// for cc in 0..3 {
///     assert_eq!(d[cc].len(), w * h);
///     for &v in &d[cc] {
///         assert_eq!(v.to_bits(), 0.0_f32.to_bits());
///     }
/// }
/// ```
#[must_use]
pub fn mult_mutual_band(
    t_p_per_ch: &[Vec<f32>; 3],
    r_p_per_ch: &[Vec<f32>; 3],
    w: usize,
    h: usize,
) -> [Vec<f32>; 3] {
    let n = w * h;
    debug_assert_eq!(t_p_per_ch[0].len(), n);

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

    let m_mm: [Vec<f32>; 3] = [
        phase_uncertainty_band(&m_mm_raw[0], w, h),
        phase_uncertainty_band(&m_mm_raw[1], w, h),
        phase_uncertainty_band(&m_mm_raw[2], w, h),
    ];

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
