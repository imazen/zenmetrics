//! Per-pixel SSIM combine: build `cs` and (top-scale only) `cs · l`
//! from the blurred buffers.
//!
//! All five inputs are the post-Gaussian (11×11 σ=1.5, valid-mode)
//! statistics at scale `j`. Output is `(in_h − 10) × (in_w − 10)` — same
//! shape as the inputs.
//!
//! ```text
//! σ₁²    = max(0, blur(x²) − µ₁²)
//! σ₂²    = max(0, blur(y²) − µ₂²)
//! σ_{12} = blur(x·y) − µ₁·µ₂
//! cs     = (2σ_{12} + C₂) / (σ₁² + σ₂² + C₂)
//! l      = (2µ₁·µ₂   + C₁) / (µ₁² + µ₂²  + C₁)   (top scale only)
//! ```
//!
//! `C₁ = (0.01 · L)²`, `C₂ = (0.03 · L)²`, `L = 255` — matches both
//! reference implementations exactly.

use cubecl::prelude::*;

// L=255 dynamic range. C1 / C2 baked at compile time so the kernel
// is pure compute.
const C1: f32 = 6.502_5_f32; // (0.01 * 255)^2
const C2: f32 = 58.522_5_f32; // (0.03 * 255)^2

/// Compute `cs_map` only (scales 0..3). Five inputs, one output, all
/// of length `n`.
#[cube(launch_unchecked)]
pub fn ssim_cs_kernel(
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    m11: &Array<f32>, // blur(x²)
    m22: &Array<f32>, // blur(y²)
    m12: &Array<f32>, // blur(x·y)
    cs_out: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = cs_out.len();
    if idx >= n {
        terminate!();
    }
    let u1 = mu1[idx];
    let u2 = mu2[idx];
    let mut s11 = m11[idx] - u1 * u1;
    let mut s22 = m22[idx] - u2 * u2;
    let s12 = m12[idx] - u1 * u2;
    if s11 < 0.0_f32 {
        s11 = 0.0_f32;
    }
    if s22 < 0.0_f32 {
        s22 = 0.0_f32;
    }
    cs_out[idx] = (2.0_f32 * s12 + C2) / (s11 + s22 + C2);
}

/// Compute `cs · l` at the top scale — what the per-scale pooling
/// step actually needs (scale 4 has no IW weight, so the reduction
/// averages this product directly).
#[cube(launch_unchecked)]
pub fn ssim_cs_l_kernel(
    mu1: &Array<f32>,
    mu2: &Array<f32>,
    m11: &Array<f32>,
    m22: &Array<f32>,
    m12: &Array<f32>,
    out: &mut Array<f32>,
) {
    let idx = ABSOLUTE_POS;
    let n = out.len();
    if idx >= n {
        terminate!();
    }
    let u1 = mu1[idx];
    let u2 = mu2[idx];
    let mut s11 = m11[idx] - u1 * u1;
    let mut s22 = m22[idx] - u2 * u2;
    let s12 = m12[idx] - u1 * u2;
    if s11 < 0.0_f32 {
        s11 = 0.0_f32;
    }
    if s22 < 0.0_f32 {
        s22 = 0.0_f32;
    }
    let cs = (2.0_f32 * s12 + C2) / (s11 + s22 + C2);
    let l = (2.0_f32 * u1 * u2 + C1) / (u1 * u1 + u2 * u2 + C1);
    out[idx] = cs * l;
}
