//! SSIM contrast-structure (cs) + luminance (l) maps via the 11×11
//! Gaussian (σ=1.5) window. Matches the Python reference's
//! `scale_qualty_maps`.
//!
//! Conventions:
//!
//! - Window applied as **valid** mode (no padding): output dimensions
//!   are `(h - 10, w - 10)`. This matches `F.conv2d(img, ms_win)` in
//!   the reference (PyTorch's default is no-padding).
//! - 11×11 Gaussian is separable → 1D 11-tap horizontal then 1D 11-tap
//!   vertical. The `filters::SSIM_WIN_1D` constant is pre-normalized.
//! - SSIM constants: `C1 = (0.01·255)²`, `C2 = (0.03·255)²` (Wang &
//!   Bovik 2002).
//! - Σ² clamped at 0 (matches `torch.max(zeros, σ²)`).

use alloc::vec::Vec;

use crate::filters::SSIM_WIN_LEN;
use crate::simd_kernels::{cs_combine_into, mul_into, square_into, ssim_gauss_h_pass, ssim_gauss_v_pass};

const SSIM_L: f32 = 255.0;
const SSIM_K1: f32 = 0.01;
const SSIM_K2: f32 = 0.03;
pub(crate) const SSIM_C1: f32 = (SSIM_K1 * SSIM_L) * (SSIM_K1 * SSIM_L);
pub(crate) const SSIM_C2: f32 = (SSIM_K2 * SSIM_L) * (SSIM_K2 * SSIM_L);

/// 11×11 Gaussian (σ=1.5) **valid** convolution via separable SIMD
/// passes. Output dims are `(h - 10, w - 10)`.
pub(crate) fn gaussian_11x11_valid(
    src: &[f32],
    h: usize,
    w: usize,
    dst_h: usize,
    dst_w: usize,
    h_scratch: &mut [f32],
    dst: &mut [f32],
) {
    debug_assert_eq!(src.len(), h * w);
    debug_assert_eq!(dst.len(), dst_h * dst_w);
    debug_assert!(dst_w + (SSIM_WIN_LEN - 1) == w);
    debug_assert!(dst_h + (SSIM_WIN_LEN - 1) == h);
    debug_assert_eq!(h_scratch.len(), h * dst_w);
    // Stage 1: horizontal pass. `(h, w)` → `(h, dst_w)`.
    ssim_gauss_h_pass(src, h, w, dst_w, h_scratch);
    // Stage 2: vertical pass. `(h, dst_w)` → `(dst_h, dst_w)`.
    ssim_gauss_v_pass(h_scratch, h, dst_h, dst_w, dst);
}

/// SSIM stats for one pyramid level.
pub(crate) struct CsStats {
    /// Output cs/l dims (`cs_w × cs_h = (w - 10) × (h - 10)`).
    pub cs_w: usize,
    pub cs_h: usize,
    /// `mu1 = blur(ref)` — retained for diagnostic inspection.
    #[allow(dead_code)]
    pub mu1: Vec<f32>,
    /// `mu2 = blur(dis)` — retained for diagnostic inspection.
    #[allow(dead_code)]
    pub mu2: Vec<f32>,
    /// `cs = (2σ₁₂ + C₂) / (σ₁² + σ₂² + C₂)`.
    pub cs: Vec<f32>,
}

/// Compute the cs map for one pyramid scale + (optionally) the
/// luminance term `l = (2µ₁µ₂ + C₁) / (µ₁² + µ₂² + C₁)`.
///
/// Returns the cs stats. If `with_luminance == true`, also multiplies
/// `cs` by the luminance map in place (matching the upstream's
/// `cs_map[s] * l_map` combination at the top scale).
pub(crate) fn compute_cs(
    img_ref: &[f32],
    img_dis: &[f32],
    h: usize,
    w: usize,
    with_luminance: bool,
) -> CsStats {
    assert_eq!(img_ref.len(), h * w);
    assert_eq!(img_dis.len(), h * w);
    let cs_h = h - (SSIM_WIN_LEN - 1);
    let cs_w = w - (SSIM_WIN_LEN - 1);
    let n_cs = cs_h * cs_w;

    let mut h_scratch = alloc::vec![0.0_f32; h * cs_w];
    let mut mu1 = alloc::vec![0.0_f32; n_cs];
    let mut mu2 = alloc::vec![0.0_f32; n_cs];
    let mut sigma1_sq = alloc::vec![0.0_f32; n_cs];
    let mut sigma2_sq = alloc::vec![0.0_f32; n_cs];
    let mut sigma12 = alloc::vec![0.0_f32; n_cs];

    // mu1, mu2.
    gaussian_11x11_valid(img_ref, h, w, cs_h, cs_w, &mut h_scratch, &mut mu1);
    gaussian_11x11_valid(img_dis, h, w, cs_h, cs_w, &mut h_scratch, &mut mu2);

    // E[x²], E[y²], E[x·y].
    let mut sq_buf = alloc::vec![0.0_f32; h * w];
    square_into(img_ref, &mut sq_buf);
    gaussian_11x11_valid(&sq_buf, h, w, cs_h, cs_w, &mut h_scratch, &mut sigma1_sq);
    square_into(img_dis, &mut sq_buf);
    gaussian_11x11_valid(&sq_buf, h, w, cs_h, cs_w, &mut h_scratch, &mut sigma2_sq);
    mul_into(img_ref, img_dis, &mut sq_buf);
    gaussian_11x11_valid(&sq_buf, h, w, cs_h, cs_w, &mut h_scratch, &mut sigma12);

    // σ² and σ₁₂ from raw moments + luminance (top scale).
    let mut cs = alloc::vec![0.0_f32; n_cs];
    cs_combine_into(&mu1, &mu2, &sigma1_sq, &sigma2_sq, &sigma12, &mut cs, with_luminance);

    CsStats {
        cs_w,
        cs_h,
        mu1,
        mu2,
        cs,
    }
}
