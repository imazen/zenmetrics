//! 11-tap separable Gaussian (σ=1.5) in **valid** mode for the SSIM
//! statistics step.
//!
//! Two passes: horizontal then vertical. Each pass drops `RADIUS`
//! pixels from each side of the convolved axis so the final result is
//! `(H − 10) × (W − 10)`. This matches `F.conv2d(imgo, ms_win)` in
//! `IW_SSIM_PyTorch.py`, which is PyTorch's `valid` default.
//!
//! Six variants are useful per scale: filter on `ref`, `dist`, `ref²`,
//! `dist²`, `ref·dist`, and an optional pass over the cs/l results.
//! All can be expressed via:
//!
//! - `gauss11_h_kernel` — input plane → horizontally-filtered, width
//!   shrunk by `10`. Computes `f(p)` where `f` is one of `id`, `²`,
//!   `· q`. We split into three variants to avoid extra material reads
//!   on the hot path.

use cubecl::prelude::*;

use crate::filters;

const TAPS: usize = 11;

/// `out = conv1d(src, gauss11_1d)` along x; valid mode (out_w = in_w − 10).
#[cube(launch_unchecked)]
pub fn gauss11_h_kernel(src: &Array<f32>, dst: &mut Array<f32>, in_h: u32, in_w: u32, out_w: u32) {
    let idx = ABSOLUTE_POS;
    let total = (in_h * out_w) as usize;
    if idx >= total {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = idx / out_w_us;
    let ox = idx - oy * out_w_us;
    let in_w_us = in_w as usize;
    let row_off = oy * in_w_us;
    // VALID: input start x = ox (since output spans [R, in_w − R)).
    let base = row_off + ox;
    let mut acc = 0.0_f32;
    #[unroll]
    for k in 0..TAPS {
        acc += filters::SSIM_WIN_1D[k] * src[base + k];
    }
    dst[idx] = acc;
}

/// `out = conv1d(src², gauss11_1d)` along x; valid mode. Saves one
/// pointwise-square kernel by fusing the square into the read.
#[cube(launch_unchecked)]
pub fn gauss11_h_sq_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    in_h: u32,
    in_w: u32,
    out_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (in_h * out_w) as usize;
    if idx >= total {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = idx / out_w_us;
    let ox = idx - oy * out_w_us;
    let in_w_us = in_w as usize;
    let row_off = oy * in_w_us;
    let base = row_off + ox;
    let mut acc = 0.0_f32;
    #[unroll]
    for k in 0..TAPS {
        let v = src[base + k];
        acc += filters::SSIM_WIN_1D[k] * (v * v);
    }
    dst[idx] = acc;
}

/// `out = conv1d(a·b, gauss11_1d)` along x; valid mode.
#[cube(launch_unchecked)]
pub fn gauss11_h_prod_kernel(
    a: &Array<f32>,
    b: &Array<f32>,
    dst: &mut Array<f32>,
    in_h: u32,
    in_w: u32,
    out_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (in_h * out_w) as usize;
    if idx >= total {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = idx / out_w_us;
    let ox = idx - oy * out_w_us;
    let in_w_us = in_w as usize;
    let row_off = oy * in_w_us;
    let base = row_off + ox;
    let mut acc = 0.0_f32;
    #[unroll]
    for k in 0..TAPS {
        acc += filters::SSIM_WIN_1D[k] * (a[base + k] * b[base + k]);
    }
    dst[idx] = acc;
}

/// `out = conv1d(src, gauss11_1d)` along y; valid mode (out_h = in_h − 10).
/// Input width has already been shrunk by the horizontal pass.
#[cube(launch_unchecked)]
pub fn gauss11_v_kernel(src: &Array<f32>, dst: &mut Array<f32>, out_h: u32, in_h: u32, w: u32) {
    let _ = in_h; // unused; kept for API symmetry with the H kernels.
    let idx = ABSOLUTE_POS;
    let total = (out_h * w) as usize;
    if idx >= total {
        terminate!();
    }
    let w_us = w as usize;
    let oy = idx / w_us;
    let ox = idx - oy * w_us;
    let base = oy * w_us + ox; // start y = oy (out covers [R, in_h − R)).
    let stride = w_us;
    let mut acc = 0.0_f32;
    #[unroll]
    for k in 0..TAPS {
        acc += filters::SSIM_WIN_1D[k] * src[base + k * stride];
    }
    dst[idx] = acc;
}
