//! Scalar diffmap helpers — bilinear sampling + per-pixel channel pool.
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels::diffmap` so the
//! CPU crate owns the canonical scalar implementation; cvvdp-gpu
//! continues to re-export the same paths. GPU-side `#[cube(launch)]`
//! kernels remain in `cvvdp-gpu::kernels::diffmap`.

/// Host-side scalar reference for a bilinear sample (matches the GPU
/// `bilinear_upsample_kernel` per-pixel math at f32 precision).
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::diffmap::bilinear_sample_scalar;
///
/// // Identity (src_dims == dst_dims) returns the source pixel.
/// let src: Vec<f32> = (0..16).map(|i| i as f32).collect();
/// let v = bilinear_sample_scalar(&src, 4, 4, 1, 2, 4, 4);
/// assert!((v - 9.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn bilinear_sample_scalar(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    x_dst: u32,
    y_dst: u32,
    dst_w: u32,
    dst_h: u32,
) -> f32 {
    let src_w_f = src_w as f32;
    let src_h_f = src_h as f32;
    let dst_w_f = dst_w as f32;
    let dst_h_f = dst_h as f32;

    let fx = (x_dst as f32 + 0.5) * (src_w_f / dst_w_f) - 0.5;
    let fy = (y_dst as f32 + 0.5) * (src_h_f / dst_h_f) - 0.5;

    let fx_c = fx.clamp(0.0, src_w_f - 1.0);
    let fy_c = fy.clamp(0.0, src_h_f - 1.0);

    let x0_f = fx_c.floor();
    let y0_f = fy_c.floor();
    let dx = fx_c - x0_f;
    let dy = fy_c - y0_f;

    let x0 = x0_f as u32;
    let y0 = y0_f as u32;
    let src_w_m1 = src_w.saturating_sub(1);
    let src_h_m1 = src_h.saturating_sub(1);
    let x1 = if x0 < src_w_m1 { x0 + 1 } else { src_w_m1 };
    let y1 = if y0 < src_h_m1 { y0 + 1 } else { src_h_m1 };

    let i00 = (y0 * src_w + x0) as usize;
    let i01 = (y0 * src_w + x1) as usize;
    let i10 = (y1 * src_w + x0) as usize;
    let i11 = (y1 * src_w + x1) as usize;

    let w00 = (1.0 - dx) * (1.0 - dy);
    let w01 = dx * (1.0 - dy);
    let w10 = (1.0 - dx) * dy;
    let w11 = dx * dy;

    w00 * src[i00] + w01 * src[i01] + w10 * src[i10] + w11 * src[i11]
}

/// Host-side scalar reference for the per-pixel channel pool.
///
/// # Examples
///
/// ```
/// use cvvdp::kernels::diffmap::channel_pool_scalar;
///
/// assert_eq!(channel_pool_scalar(0.0, 0.0, 0.0, 4.0), 0.0);
///
/// let v = channel_pool_scalar(2.0, 0.0, 0.0, 4.0);
/// assert!((v - 2.0).abs() < 1e-6, "got {v}, expected = 2.0");
///
/// let pos = channel_pool_scalar(2.0, 0.0, 0.0, 4.0);
/// let neg = channel_pool_scalar(-2.0, 0.0, 0.0, 4.0);
/// assert_eq!(neg, 0.0);
/// assert!(pos > 0.0);
/// ```
#[must_use]
pub fn channel_pool_scalar(a: f32, rg: f32, vy: f32, beta: f32) -> f32 {
    let a_pos = a.max(0.0);
    let rg_pos = rg.max(0.0);
    let vy_pos = vy.max(0.0);

    let acc = a_pos.powf(beta) + rg_pos.powf(beta) + vy_pos.powf(beta);
    acc.powf(1.0 / beta)
}
