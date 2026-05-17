//! `imenlarge2`: MATLAB-style ×2 bilinear upsample with 1-pixel linear
//! edge extrapolation, then decimate-by-2. Faithfully reproduces the
//! `imenlarge2.m` / Python `imenlarge2` reference.
//!
//! Closed-form per-output-pixel:
//!
//! ```text
//! For axis y, output index `oy` reads:
//!   if oy == 0:        src[0] * 1.25 + src[1] * (-0.25)
//!   if oy == 2M - 1:   src[M-2] * (-0.25) + src[M-1] * 1.25
//!   else:
//!     t1_row = 2*oy - 1                 // ∈ {1, 3, 5, ..., 4M-5}
//!     y0     = t1_row / 4
//!     rem    = t1_row - 4*y0            // ∈ {1, 3}
//!     frac   = rem / 4.0                // = 0.25 or 0.75
//!     weights= (1 - frac, frac) at (y0, y0+1)
//! ```
//!
//! Symmetric for axis x. We compute weights/indices for each axis
//! using a branchless reformulation:
//!
//! - Set defaults to the interior case (works for `oy ≥ 1`).
//! - Override top edge (`oy == 0`) and bottom edge (`oy == 2M-1`) at
//!   the end with simple conditional writes.
//!
//! cubecl 0.10's `#[cube]` macro is picky about let-mut reassignment
//! across branches with mixed types; this kernel uses only u32 index
//! math and f32 weight math, with one final guarded select per axis.

use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn imenlarge2_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (out_h * out_w) as usize;
    if idx >= total {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = (idx / out_w_us) as u32;
    let ox = (idx - (oy as usize) * out_w_us) as u32;

    // ---- Y axis ----
    let two_m_y = 2 * in_h - 1;
    // Interior defaults (oy ≥ 1, oy < 2M-1). For oy = 0 we override
    // below; the interior path safely reads two consecutive source
    // rows so it's fine to compute on the oy=0 case as a placeholder.
    let oy_for_interior = u32::max(oy, 1);
    let t1_row_y = 2 * oy_for_interior - 1;
    let y0_int = t1_row_y / 4;
    let rem_y = t1_row_y - 4 * y0_int;
    let frac_y = (rem_y as f32) * 0.25_f32;
    let mut y0 = y0_int;
    let mut y1 = y0_int + 1;
    let mut wy0 = 1.0_f32 - frac_y;
    let mut wy1 = frac_y;
    if oy == 0 {
        y0 = 0;
        y1 = 1;
        wy0 = 1.25_f32;
        wy1 = -0.25_f32;
    }
    if oy == two_m_y {
        y0 = in_h - 2;
        y1 = in_h - 1;
        wy0 = -0.25_f32;
        wy1 = 1.25_f32;
    }

    // ---- X axis ----
    let two_m_x = 2 * in_w - 1;
    let ox_for_interior = u32::max(ox, 1);
    let t1_col_x = 2 * ox_for_interior - 1;
    let x0_int = t1_col_x / 4;
    let rem_x = t1_col_x - 4 * x0_int;
    let frac_x = (rem_x as f32) * 0.25_f32;
    let mut x0 = x0_int;
    let mut x1 = x0_int + 1;
    let mut wx0 = 1.0_f32 - frac_x;
    let mut wx1 = frac_x;
    if ox == 0 {
        x0 = 0;
        x1 = 1;
        wx0 = 1.25_f32;
        wx1 = -0.25_f32;
    }
    if ox == two_m_x {
        x0 = in_w - 2;
        x1 = in_w - 1;
        wx0 = -0.25_f32;
        wx1 = 1.25_f32;
    }

    let stride = in_w as usize;
    let p00 = src[(y0 as usize) * stride + (x0 as usize)];
    let p01 = src[(y0 as usize) * stride + (x1 as usize)];
    let p10 = src[(y1 as usize) * stride + (x0 as usize)];
    let p11 = src[(y1 as usize) * stride + (x1 as usize)];

    dst[idx] = wy0 * (wx0 * p00 + wx1 * p01) + wy1 * (wx0 * p10 + wx1 * p11);
}
