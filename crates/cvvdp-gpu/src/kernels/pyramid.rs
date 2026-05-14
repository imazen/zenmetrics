//! Laplacian pyramid analysis (still-image cvvdp).
//!
//! For each of the 3 DKL channels, produces `n_levels` band buffers:
//!
//! - `band[k]` for `k < n_levels - 1` = `gauss[k] - upscale(gauss[k+1])`
//! - `band[n_levels - 1]` = the coarsest gaussian (residual)
//!
//! cvvdp uses a 5-tap binomial Gaussian filter for the analysis and
//! reconstruction kernels (Burt-Adelson). Edge handling is **replicate**
//! in the reference; we must match this — not reflect, not zero.
//!
//! Kernels in this module:
//! - `downscale_kernel`: 5-tap separable Gaussian + 2× decimation.
//! - `upscale_kernel`: 2× zero-insertion + 5-tap separable Gaussian
//!   (×4 gain for the inserted-zero rebalancing).
//! - `subtract_kernel`: `band = fine - upscale(coarse)`.
//!
//! Compiling stubs; bodies land once goldens are captured.

use cubecl::prelude::*;

/// 2× downscale with the cvvdp 5-tap Gaussian (binomial [1, 4, 6, 4, 1] / 16).
#[cube(launch)]
#[allow(unused_variables)]
pub fn downscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    dst[idx] = 0.0;
}

/// 2× upscale with the cvvdp 5-tap Gaussian (× 4 reconstruction gain).
#[cube(launch)]
#[allow(unused_variables)]
pub fn upscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    dst[idx] = 0.0;
}

/// `band = fine - upscaled_coarse`, written into `band`.
#[cube(launch)]
#[allow(unused_variables)]
pub fn subtract_kernel(
    fine: &Array<f32>,
    upscaled_coarse: &Array<f32>,
    band: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band[idx] = 0.0;
}
