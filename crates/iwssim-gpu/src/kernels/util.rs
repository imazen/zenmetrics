//! Small utility kernels — zero-out and copy. Used to avoid the
//! per-call `create_from_slice` allocation churn (each
//! `create_from_slice` performs `cudaMalloc` + `cudaMemcpy`; for
//! the ≤ 100-element `cu_atomic` reset, a single-launch zero kernel
//! is materially cheaper).

use cubecl::prelude::*;

/// `dst[i] = 0` for `i < dst.len()`. Run as one launch over the
/// target's full length.
#[cube(launch_unchecked)]
pub fn zero_kernel(dst: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    dst[idx] = 0.0_f32;
}
