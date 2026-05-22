//! Per-scale weighted-pool reductions.
//!
//! For scales `j ∈ 0..3`:
//! - `Σ(cs_j ⊙ iw_j[cropped by bound1])`
//! - `Σ(iw_j[cropped by bound1])`
//!
//! For scale `j = 4` (top): `Σ(cs_4 ⊙ l_4)`. The denominator is the
//! pixel count of the cs-shaped buffer at scale 4 (host-side scalar).
//!
//! The cropped iw extent matches cs's `(h_LP − 10, w_LP − 10)` shape:
//!
//! - iw is computed on `(nblv, nblh) = (h_LP − 2, w_LP − 2)`.
//! - bound1 = `5 − 1 = 4` for `winsize = 11`, `blSzX = 3`.
//! - Cropped iw is `(nblv − 8, nblh − 8) = (h_LP − 10, w_LP − 10)`,
//!   reading `iw[bound1 .. nblv − bound1, bound1 .. nblh − bound1]`.
//!
//! Both reductions are simple `Σ` over the same pixel range — we
//! launch one grid-strided kernel per (slot, source-buffer) pair.

use cubecl::prelude::*;

/// Number of threads per cube.
pub const BLOCK_SIZE: u32 = 256;
/// Number of cubes per reduction launch.
pub const NUM_BLOCKS: u32 = 16;
/// Total threads (= partials per slot) — must match the finalizer
/// kernel's compile-time bound.
pub const THREADS_PER_REDUCTION: u32 = NUM_BLOCKS * BLOCK_SIZE;

/// Grid-strided sum of `cs[py, px] * iw[py + b1, px + b1]` for
/// `py ∈ [cs_y_start, cs_y_end)`, `px ∈ [0, cs_w)`. cs is laid out at
/// the cropped shape `(cs_h, cs_w)`; iw at the full `(iw_h, iw_w)`.
/// Output goes to `partials[base + tid]`.
///
/// `cs_y_start` / `cs_y_end` restrict the pooling to a row range of
/// the cs buffer — used by strip processing so per-strip pools only
/// include the strip's body rows (halo rows are computed but not
/// summed). For the whole-image path the caller passes
/// `cs_y_start = 0`, `cs_y_end = cs_h`, recovering the original
/// full-buffer reduction.
#[cube(launch_unchecked)]
pub fn weighted_sum_kernel(
    cs: &Array<f32>,
    iw: &Array<f32>,
    partials: &mut Array<f32>,
    cs_h: u32,
    cs_w: u32,
    iw_h: u32,
    iw_w: u32,
    bound1: u32,
    partials_base: u32,
    cs_y_start: u32,
    cs_y_end: u32,
) {
    let _ = iw_h;
    let _ = cs_h;
    let tid = ABSOLUTE_POS;
    // Manual CUBE_COUNT aggregation (= X*Y*Z). The aggregated CUBE_COUNT
    // builtin is unimplemented on `cubecl-cpu` (silently panics inside the
    // worker, kernel produces 0 — see `cov.rs` header). Cube launches in
    // this file are all 1D in X, so the product is = CUBE_COUNT_X, but we
    // compute the full product for correctness in case a future change
    // makes a launch 2D/3D.
    let stride = ((CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z) as usize) * (CUBE_DIM_X as usize);
    let cs_w_us = cs_w as usize;
    let iw_w_us = iw_w as usize;
    let b1 = bound1 as usize;
    let y_start = cs_y_start as usize;
    let y_end = cs_y_end as usize;
    let rows = y_end - y_start;
    let n = rows * cs_w_us;

    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        let local_py = i / cs_w_us;
        let px = i - local_py * cs_w_us;
        let py = local_py + y_start;
        let cs_v = cs[py * cs_w_us + px];
        let iw_v = iw[(py + b1) * iw_w_us + (px + b1)];
        s += cs_v * iw_v;
        i += stride;
    }
    partials[(partials_base as usize) + tid] = s;
}

/// Grid-strided sum of `iw[py + b1, px + b1]` over a row range
/// `[cs_y_start, cs_y_end)` of the cropped iw extent. Same row-range
/// semantics as [`weighted_sum_kernel`].
#[cube(launch_unchecked)]
pub fn iw_sum_kernel(
    iw: &Array<f32>,
    partials: &mut Array<f32>,
    cs_h: u32,
    cs_w: u32,
    iw_h: u32,
    iw_w: u32,
    bound1: u32,
    partials_base: u32,
    cs_y_start: u32,
    cs_y_end: u32,
) {
    let _ = iw_h;
    let _ = cs_h;
    let tid = ABSOLUTE_POS;
    let stride = ((CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z) as usize) * (CUBE_DIM_X as usize);
    let cs_w_us = cs_w as usize;
    let iw_w_us = iw_w as usize;
    let b1 = bound1 as usize;
    let y_start = cs_y_start as usize;
    let y_end = cs_y_end as usize;
    let rows = y_end - y_start;
    let n = rows * cs_w_us;

    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        let local_py = i / cs_w_us;
        let px = i - local_py * cs_w_us;
        let py = local_py + y_start;
        let iw_v = iw[(py + b1) * iw_w_us + (px + b1)];
        s += iw_v;
        i += stride;
    }
    partials[(partials_base as usize) + tid] = s;
}

/// Grid-strided sum of `src[i]` over a row-range slice. `src` is laid
/// out at `(src_h, src_w)` row-major; the reduction sums
/// `src[y, x]` for `y ∈ [y_start, y_end)`, `x ∈ [0, src_w)`. Used for
/// the top scale's `Σ(cs · l)` (cs · l is a separate kernel output)
/// and as the generic single-buffer fold.
///
/// For the whole-image path, pass `y_start = 0` and `y_end = src_h`
/// to recover a sum over the full buffer. Strip processing passes
/// the strip's body range (in the top scale's coordinate system) so
/// only body rows are pooled.
#[cube(launch_unchecked)]
pub fn plain_sum_kernel(
    src: &Array<f32>,
    partials: &mut Array<f32>,
    partials_base: u32,
    src_w: u32,
    y_start: u32,
    y_end: u32,
) {
    let tid = ABSOLUTE_POS;
    let stride = ((CUBE_COUNT_X * CUBE_COUNT_Y * CUBE_COUNT_Z) as usize) * (CUBE_DIM_X as usize);
    let w_us = src_w as usize;
    let y_lo = y_start as usize;
    let y_hi = y_end as usize;
    let rows = y_hi - y_lo;
    let n = rows * w_us;
    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        let local_py = i / w_us;
        let px = i - local_py * w_us;
        let py = local_py + y_lo;
        s += src[py * w_us + px];
        i += stride;
    }
    partials[(partials_base as usize) + tid] = s;
}

/// Stage-2 finalizer: one cube per (slot, sum) pair sums the partials
/// from `THREADS_PER_REDUCTION` workers into a single f32 at `dst[slot]`.
#[cube(launch_unchecked)]
pub fn finalize_kernel(partials: &Array<f32>, dst: &mut Array<f32>) {
    let slot = CUBE_POS_X;
    let n_threads = NUM_BLOCKS * BLOCK_SIZE;
    let mut s = 0.0_f32;
    let mut k: u32 = 0;
    while k < n_threads {
        s += partials[(slot * n_threads + k) as usize];
        k += 1;
    }
    dst[slot as usize] = s;
}

/// Convenience: launch a `Σ(cs · iw)` reduction into the given slot.
///
/// `cs_y_start` / `cs_y_end` restrict the sum to a row range of the
/// cs buffer. For whole-image reduction pass `(0, cs_h)`; for strip
/// reduction pass the body row range (in cs coordinates) so halo
/// rows on either side are skipped.
#[allow(clippy::too_many_arguments)]
pub fn launch_weighted_sum<R: Runtime>(
    client: &ComputeClient<R>,
    cs: cubecl::server::Handle,
    cs_len: usize,
    iw: cubecl::server::Handle,
    iw_len: usize,
    partials: cubecl::server::Handle,
    partials_len: usize,
    cs_h: u32,
    cs_w: u32,
    iw_h: u32,
    iw_w: u32,
    bound1: u32,
    slot: u32,
    cs_y_start: u32,
    cs_y_end: u32,
) {
    let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
    let partials_base = slot * THREADS_PER_REDUCTION;
    unsafe {
        weighted_sum_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(cs, cs_len),
            ArrayArg::from_raw_parts(iw, iw_len),
            ArrayArg::from_raw_parts(partials, partials_len),
            cs_h,
            cs_w,
            iw_h,
            iw_w,
            bound1,
            partials_base,
            cs_y_start,
            cs_y_end,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_iw_sum<R: Runtime>(
    client: &ComputeClient<R>,
    iw: cubecl::server::Handle,
    iw_len: usize,
    partials: cubecl::server::Handle,
    partials_len: usize,
    cs_h: u32,
    cs_w: u32,
    iw_h: u32,
    iw_w: u32,
    bound1: u32,
    slot: u32,
    cs_y_start: u32,
    cs_y_end: u32,
) {
    let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
    let partials_base = slot * THREADS_PER_REDUCTION;
    unsafe {
        iw_sum_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(iw, iw_len),
            ArrayArg::from_raw_parts(partials, partials_len),
            cs_h,
            cs_w,
            iw_h,
            iw_w,
            bound1,
            partials_base,
            cs_y_start,
            cs_y_end,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_plain_sum<R: Runtime>(
    client: &ComputeClient<R>,
    src: cubecl::server::Handle,
    src_len: usize,
    partials: cubecl::server::Handle,
    partials_len: usize,
    slot: u32,
    src_w: u32,
    y_start: u32,
    y_end: u32,
) {
    let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
    let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
    let partials_base = slot * THREADS_PER_REDUCTION;
    unsafe {
        plain_sum_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src, src_len),
            ArrayArg::from_raw_parts(partials, partials_len),
            partials_base,
            src_w,
            y_start,
            y_end,
        );
    }
}

pub fn launch_finalize<R: Runtime>(
    client: &ComputeClient<R>,
    partials: cubecl::server::Handle,
    partials_len: usize,
    dst: cubecl::server::Handle,
    dst_len: usize,
    num_slots: u32,
) {
    let cube_count = CubeCount::Static(num_slots, 1, 1);
    let cube_dim = CubeDim::new_1d(1);
    unsafe {
        finalize_kernel::launch_unchecked::<R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(partials, partials_len),
            ArrayArg::from_raw_parts(dst, dst_len),
        );
    }
}
