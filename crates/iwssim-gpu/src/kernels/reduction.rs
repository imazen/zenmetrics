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
/// `py ∈ [0, nblv − 2·b1)`, `px ∈ [0, nblh − 2·b1)`. cs is laid out at
/// the cropped shape `(nblv − 2·b1, nblh − 2·b1)`; iw at the full
/// `(nblv, nblh)`. Output goes to `partials[base + tid]`.
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
) {
    let _ = iw_h;
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = (cs_h * cs_w) as usize;
    let cs_w_us = cs_w as usize;
    let iw_w_us = iw_w as usize;
    let b1 = bound1 as usize;

    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        let py = i / cs_w_us;
        let px = i - py * cs_w_us;
        let cs_v = cs[i];
        let iw_v = iw[(py + b1) * iw_w_us + (px + b1)];
        s += cs_v * iw_v;
        i += stride;
    }
    partials[(partials_base as usize) + tid] = s;
}

/// Grid-strided sum of `iw[py + b1, px + b1]` over the cropped range
/// `(nblv − 2·b1, nblh − 2·b1) = (cs_h, cs_w)`.
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
) {
    let _ = iw_h;
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = (cs_h * cs_w) as usize;
    let cs_w_us = cs_w as usize;
    let iw_w_us = iw_w as usize;
    let b1 = bound1 as usize;

    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        let py = i / cs_w_us;
        let px = i - py * cs_w_us;
        let iw_v = iw[(py + b1) * iw_w_us + (px + b1)];
        s += iw_v;
        i += stride;
    }
    partials[(partials_base as usize) + tid] = s;
}

/// Grid-strided sum of `src[i]` over `n` elements — used for the top
/// scale's `Σ(cs · l)` (cs · l is a separate kernel output) and as the
/// generic single-buffer fold.
#[cube(launch_unchecked)]
pub fn plain_sum_kernel(src: &Array<f32>, partials: &mut Array<f32>, partials_base: u32) {
    let tid = ABSOLUTE_POS;
    let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
    let n = src.len();
    let mut s = 0.0_f32;
    let mut i = tid;
    while i < n {
        s += src[i];
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
        );
    }
}

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
        );
    }
}

pub fn launch_plain_sum<R: Runtime>(
    client: &ComputeClient<R>,
    src: cubecl::server::Handle,
    src_len: usize,
    partials: cubecl::server::Handle,
    partials_len: usize,
    slot: u32,
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
