//! Laplacian pyramid build, faithful to `pyrtools.pyramids.LaplacianPyramid`.
//!
//! pyrtools uses the `binom5` separable filter (`sqrt(2)·[1,4,6,4,1]/16`)
//! with `reflect1` boundary (reflection through the edge pixels — the
//! edge pixel itself is **not** duplicated). The pyramid is built
//! separably: filter horizontal with step (1,2), then filter vertical
//! with step (2,1). Each downsample halves both axes via `ceil(W/2)`,
//! `ceil(H/2)`.
//!
//! Expansion (for the Laplacian residual `LP_j = G_j − expand(G_{j+1})`)
//! is the same filter applied via zero-insertion upsampling. pyrtools'
//! `upConv` semantics: insert one zero between every sample, convolve
//! with the same filter, scale by 2 (one axis) or 4 (two axes) to
//! preserve DC — but pyrtools' `binom5` already carries the
//! `sqrt(2)` factor, so we apply the filter twice without extra
//! scaling and get the same result.
//!
//! ## Kernel structure
//!
//! Two single-axis kernels:
//!
//! - `corr_dn_axis_kernel` — correlate with `binom5` along one axis,
//!   then decimate by 2. `step_axis` parameter selects horizontal vs
//!   vertical work; output buffer has the smaller dimension on that
//!   axis.
//! - `up_conv_axis_kernel` — zero-insert × 2 along one axis, correlate
//!   with `binom5`. Output buffer has 2× (or 2N−1, controlled by
//!   `out_axis_len`) on that axis.
//!
//! Two host-side passes per pyramid stage = one full 2D operation.
//! Same pattern as pyrtools' `_build_next` and `_recon_prev`.

use cubecl::prelude::*;

use crate::filters;

/// reflect1: index `i` outside `[0, n)` mirrors through the edge
/// pixels — `−1 → 1`, `−2 → 2`, `n → n−2`, `n+1 → n−3`, etc.
///
/// Implemented as a fold rather than a closed form so cubecl emits
/// straight-line code without a loop bound that depends on filter
/// radius. Filter radius is `≤ 2` for binom5, so 2 reflections are
/// enough — verified by running the wrap up-to-2-times branch on
/// every (i, n) ∈ {−2,−1,0,n−1,n,n+1} for typical n.
#[cube]
fn reflect1(i: i32, n: i32) -> i32 {
    let mut k = i;
    if k < 0 {
        k = -k;
    }
    if k >= n {
        k = 2 * (n - 1) - k;
    }
    if k < 0 {
        k = -k;
    }
    k
}

/// Correlate with `binom5` along one axis, then decimate by 2.
///
/// - `axis == 0`: filter rows (vertical), input is `(in_h, w)`, output
///   is `(out_h, w)` with `out_h = ceil(in_h / 2)`.
/// - `axis == 1`: filter columns (horizontal), input is `(h, in_w)`,
///   output is `(h, out_w)` with `out_w = ceil(in_w / 2)`.
///
/// `axis` is a CubeCL compile-time constant via the kernel's
/// `comptime_axis` — encoded as 0/1 explicitly in the kernel body.
/// `start_offset` selects between centered-on-0 and centered-on-1
/// decimation; pyrtools defaults to `(0,0)` start which yields
/// "input index 2*out_idx" — we match that.
#[cube(launch_unchecked)]
pub fn corr_dn_horizontal_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    h: u32,
    in_w: u32,
    out_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = (h * out_w) as usize;
    if idx >= n {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = idx / out_w_us;
    let ox = idx - oy * out_w_us;
    // Source index: in_x = 2 * out_x (filter centered there).
    let in_x_center = (ox * 2) as i32;
    let in_w_i = in_w as i32;
    let mut acc = 0.0_f32;
    let r = filters::BINOM5_RADIUS;
    // Manual unroll: 5 taps.
    let row_off = oy * (in_w as usize);
    let i0 = reflect1(in_x_center - r, in_w_i) as usize;
    let i1 = reflect1(in_x_center - r + 1, in_w_i) as usize;
    let i2 = reflect1(in_x_center - r + 2, in_w_i) as usize;
    let i3 = reflect1(in_x_center - r + 3, in_w_i) as usize;
    let i4 = reflect1(in_x_center - r + 4, in_w_i) as usize;
    acc += filters::BINOM5[0] * src[row_off + i0];
    acc += filters::BINOM5[1] * src[row_off + i1];
    acc += filters::BINOM5[2] * src[row_off + i2];
    acc += filters::BINOM5[3] * src[row_off + i3];
    acc += filters::BINOM5[4] * src[row_off + i4];
    dst[idx] = acc;
}

#[cube(launch_unchecked)]
pub fn corr_dn_vertical_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    out_h: u32,
    in_h: u32,
    w: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = (out_h * w) as usize;
    if idx >= n {
        terminate!();
    }
    let w_us = w as usize;
    let oy = idx / w_us;
    let ox = idx - oy * w_us;
    let in_y_center = (oy * 2) as i32;
    let in_h_i = in_h as i32;
    let mut acc = 0.0_f32;
    let r = filters::BINOM5_RADIUS;
    let i0 = reflect1(in_y_center - r, in_h_i) as usize;
    let i1 = reflect1(in_y_center - r + 1, in_h_i) as usize;
    let i2 = reflect1(in_y_center - r + 2, in_h_i) as usize;
    let i3 = reflect1(in_y_center - r + 3, in_h_i) as usize;
    let i4 = reflect1(in_y_center - r + 4, in_h_i) as usize;
    acc += filters::BINOM5[0] * src[i0 * w_us + ox];
    acc += filters::BINOM5[1] * src[i1 * w_us + ox];
    acc += filters::BINOM5[2] * src[i2 * w_us + ox];
    acc += filters::BINOM5[3] * src[i3 * w_us + ox];
    acc += filters::BINOM5[4] * src[i4 * w_us + ox];
    dst[idx] = acc;
}

/// Reflect-onto-extended-axis helper for upConv: the zero-stuffed
/// signal has length `2 · in_axis`. `i` may be negative or
/// `≥ 2·in_axis` and we reflect through the edge samples to bring it
/// in-bounds, then return the half-index `q/2` only when `q` lands
/// on an even position (i.e., a real source sample, not an inserted
/// zero) — caller passes a sentinel `usize::MAX` to mean "skip".
#[cube]
fn reflect_expanded(i: i32, in_axis: i32) -> (bool, i32) {
    let two_n = 2 * in_axis;
    let mut q = i;
    if q < 0 {
        q = -q;
    }
    if q >= two_n {
        q = 2 * (two_n - 1) - q;
    }
    if q < 0 {
        q = -q;
    }
    if q >= two_n {
        q = 2 * (two_n - 1) - q;
    }
    // q is now in [0, 2·in_axis). Even = real sample, odd = inserted zero.
    let active = (q & 1) == 0;
    let sx = q / 2;
    (active, sx)
}

/// Zero-insert × 2 horizontally then correlate with `binom5`. Output
/// `out_w` may be `2*in_w` (even target) or `2*in_w − 1` (odd target).
/// Boundary: reflect1 on the expanded axis (length `2·in_w`).
#[cube(launch_unchecked)]
pub fn up_conv_horizontal_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    h: u32,
    in_w: u32,
    out_w: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = (h * out_w) as usize;
    if idx >= n {
        terminate!();
    }
    let out_w_us = out_w as usize;
    let oy = idx / out_w_us;
    let ox = idx - oy * out_w_us;
    let row_off = oy * (in_w as usize);
    let mut acc = 0.0_f32;
    let r = filters::BINOM5_RADIUS;
    let in_w_i = in_w as i32;
    // Unroll over 5 taps.
    let p0 = (ox as i32) - r;
    let p1 = p0 + 1;
    let p2 = p0 + 2;
    let p3 = p0 + 3;
    let p4 = p0 + 4;
    let (a0, s0) = reflect_expanded(p0, in_w_i);
    let (a1, s1) = reflect_expanded(p1, in_w_i);
    let (a2, s2) = reflect_expanded(p2, in_w_i);
    let (a3, s3) = reflect_expanded(p3, in_w_i);
    let (a4, s4) = reflect_expanded(p4, in_w_i);
    if a0 {
        acc += filters::BINOM5[0] * src[row_off + s0 as usize];
    }
    if a1 {
        acc += filters::BINOM5[1] * src[row_off + s1 as usize];
    }
    if a2 {
        acc += filters::BINOM5[2] * src[row_off + s2 as usize];
    }
    if a3 {
        acc += filters::BINOM5[3] * src[row_off + s3 as usize];
    }
    if a4 {
        acc += filters::BINOM5[4] * src[row_off + s4 as usize];
    }
    // pyrtools' upConv applies the same `binom5 · sqrt(2)` filter once
    // per axis and relies on the sqrt(2) factor + zero insertion to
    // preserve DC; multiplying by 2 per axis re-introduces the DC that
    // the zero-stuffing dropped.
    dst[idx] = acc * 2.0_f32;
}

#[cube(launch_unchecked)]
pub fn up_conv_vertical_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    out_h: u32,
    in_h: u32,
    w: u32,
) {
    let idx = ABSOLUTE_POS;
    let n = (out_h * w) as usize;
    if idx >= n {
        terminate!();
    }
    let w_us = w as usize;
    let oy = idx / w_us;
    let ox = idx - oy * w_us;
    let mut acc = 0.0_f32;
    let r = filters::BINOM5_RADIUS;
    let in_h_i = in_h as i32;
    let p0 = (oy as i32) - r;
    let p1 = p0 + 1;
    let p2 = p0 + 2;
    let p3 = p0 + 3;
    let p4 = p0 + 4;
    let (a0, s0) = reflect_expanded(p0, in_h_i);
    let (a1, s1) = reflect_expanded(p1, in_h_i);
    let (a2, s2) = reflect_expanded(p2, in_h_i);
    let (a3, s3) = reflect_expanded(p3, in_h_i);
    let (a4, s4) = reflect_expanded(p4, in_h_i);
    if a0 {
        acc += filters::BINOM5[0] * src[(s0 as usize) * w_us + ox];
    }
    if a1 {
        acc += filters::BINOM5[1] * src[(s1 as usize) * w_us + ox];
    }
    if a2 {
        acc += filters::BINOM5[2] * src[(s2 as usize) * w_us + ox];
    }
    if a3 {
        acc += filters::BINOM5[3] * src[(s3 as usize) * w_us + ox];
    }
    if a4 {
        acc += filters::BINOM5[4] * src[(s4 as usize) * w_us + ox];
    }
    dst[idx] = acc * 2.0_f32;
}

/// Pointwise `lap = gauss_curr − expanded_next`. Used to build the
/// Laplacian band from the same-level Gaussian and the upsampled
/// next-level Gaussian.
#[cube(launch_unchecked)]
pub fn pointwise_sub_kernel(curr: &Array<f32>, expanded: &Array<f32>, dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    let n = dst.len();
    if idx >= n {
        terminate!();
    }
    dst[idx] = curr[idx] - expanded[idx];
}
