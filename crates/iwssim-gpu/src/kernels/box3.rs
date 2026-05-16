//! 3×3 box-filter statistics for the IW-SSIM info-content path.
//!
//! Faithful to `info_content_weight_map`: each of the 5 statistics is
//! a `'same'`-mode (zero-padded) 3×3 mean of the corresponding
//! product, then combined into `(g, vv)` per pixel.
//!
//! ```text
//! mean_x = box3(x)          mean_y = box3(y)
//! cov_xy = box3(x · y) − mean_x · mean_y
//! ss_x   = max(0, box3(x²) − mean_x²)
//! ss_y   = max(0, box3(y²) − mean_y²)
//!
//! g  = cov_xy / (ss_x + tol)
//! vv = ss_y − g · cov_xy
//!
//! if ss_x < tol: g = 0; vv = ss_y; ss_x = 0
//! if ss_y < tol: g = 0; vv = 0
//! ```
//!
//! `tol = 1e-15` per the reference.

use cubecl::prelude::*;

const TOL: f32 = 1.0e-15_f32;
const SCALE: f32 = 1.0_f32 / 9.0_f32;

/// `box3` mean on `x` and `y`; emits the four per-pixel quantities
/// `mean_x`, `mean_y`, `g`, `vv`. Plus `ss_x` because the eigendecomp
/// neighborhood code reads `LP[scale]` (= x), and the masking on
/// `ss_x < tol` is only used to zero `g`/`vv` — we don't need `ss_x`
/// downstream.
///
/// `w_i32`, `h_i32` give us cheap reflect-free zero padding: any
/// out-of-range (kx, ky) contributes 0.
#[cube(launch_unchecked)]
pub fn box3_gv_kernel(
    x: &Array<f32>,
    y: &Array<f32>,
    g_out: &mut Array<f32>,
    vv_out: &mut Array<f32>,
    w: u32,
    h: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = (w * h) as usize;
    if idx >= n {
        terminate!();
    }
    let w_us = w as usize;
    let py = idx / w_us;
    let px = idx - py * w_us;
    let w_i = w as i32;
    let h_i = h as i32;

    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    let mut sxy = 0.0_f32;
    let mut sxx = 0.0_f32;
    let mut syy = 0.0_f32;

    // 3×3 patch, zero padding outside the image (matches torch's
    // F.conv2d default and MATLAB's filter2 with `'same'`).
    for dy in -1_i32..=1 {
        for dx in -1_i32..=1 {
            let qy = (py as i32) + dy;
            let qx = (px as i32) + dx;
            if qx >= 0 && qx < w_i && qy >= 0 && qy < h_i {
                let off = (qy as usize) * w_us + (qx as usize);
                let xv = x[off];
                let yv = y[off];
                sx += xv;
                sy += yv;
                sxy += xv * yv;
                sxx += xv * xv;
                syy += yv * yv;
            }
        }
    }

    let mean_x = sx * SCALE;
    let mean_y = sy * SCALE;
    let cov_xy = sxy * SCALE - mean_x * mean_y;
    let mut ss_x = sxx * SCALE - mean_x * mean_x;
    let mut ss_y = syy * SCALE - mean_y * mean_y;
    if ss_x < 0.0_f32 {
        ss_x = 0.0_f32;
    }
    if ss_y < 0.0_f32 {
        ss_y = 0.0_f32;
    }

    // The Python sequence (transcribed verbatim — order matters for
    // edge cases where both ss_x and ss_y are sub-tol):
    //   g  = cov_xy / (ss_x + tol)
    //   vv = ss_y − g · cov_xy
    //   if ss_x < tol: g = 0; vv = ss_y; (ss_x = 0, not used past here)
    //   if ss_y < tol: g = 0; vv = 0
    let mut g = cov_xy / (ss_x + TOL);
    let mut vv = ss_y - g * cov_xy;
    if ss_x < TOL {
        g = 0.0_f32;
        vv = ss_y;
    }
    if ss_y < TOL {
        g = 0.0_f32;
        vv = 0.0_f32;
    }
    g_out[idx] = g;
    vv_out[idx] = vv;
}
