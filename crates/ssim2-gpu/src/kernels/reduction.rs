//! Per-plane (Σd, Σd⁴) reduction.
//!
//! For each error-map plane the SSIMULACRA2 score wants two numbers:
//! - mean: `(1/N) · Σ d`
//! - p4 norm: `((1/N) · Σ d⁴)^(1/4)`
//!
//! ## Two reduction implementations
//!
//! Selected at compile time via the `fast-reduction` feature flag:
//!
//! - **`fast-reduction` (default-on)** — single-pass `Atomic<f32>::fetch_add`.
//!   ~3× faster on CUDA at small image sizes; works correctly on CUDA but
//!   silently no-ops on cubecl-wgpu's Metal backend (the runtime reports
//!   `Atomic<f32> = LoadStore|Add` as supported but the codegen drops it).
//!   For CUDA-only or HIP deployments this is the right choice.
//! - **`fast-reduction` off** — two-stage path: each grid-strided thread
//!   writes its own (sum, p4) partial to a scratch region, then a
//!   one-cube-per-slot finalizer kernel folds the 4096 partials into one
//!   pair of f32s. ~2-3× slower on CUDA but actually works on every
//!   cubecl backend.
//!
//! Build for cross-vendor (wgpu / Metal):
//! ```bash
//! cargo build -p ssim2-gpu --no-default-features --features wgpu
//! ```

use cubecl::prelude::*;

/// Threads per cube for the reduction kernels.
pub const BLOCK_SIZE: u32 = 256;
/// Cubes per reduction (16 × 256 = 4096 grid-strided workers).
pub const NUM_BLOCKS: u32 = 16;
/// Total threads per reduction.
pub const THREADS_PER_REDUCTION: usize = (NUM_BLOCKS * BLOCK_SIZE) as usize;

/// Floats this slot consumes in the on-device "partials" buffer.
/// In fast mode this IS the final (sum, p4) pair (atomic writes go
/// straight there); in portable mode it's per-thread partials that a
/// finalizer kernel folds into the small sums buffer.
#[cfg(feature = "fast-reduction")]
pub const PARTIALS_PER_REDUCTION: usize = 2;
#[cfg(not(feature = "fast-reduction"))]
pub const PARTIALS_PER_REDUCTION: usize = THREADS_PER_REDUCTION * 2;

// =====================================================================
// Fast path — single-pass Atomic<f32>::fetch_add. CUDA only.
// =====================================================================

#[cfg(feature = "fast-reduction")]
mod fast {
    use super::*;

    #[cube(launch_unchecked)]
    fn fused_sum_p4_kernel(plane: &Array<f32>, output_sums: &mut Array<Atomic<f32>>, slot: u32) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;

        let mut i = tid;
        while i < n {
            let v = plane[i];
            local_sum += v;
            let v2 = v * v;
            local_p4 += v2 * v2;
            i += stride;
        }

        let off = (slot * 2) as usize;
        output_sums[off].fetch_add(local_sum);
        output_sums[off + 1].fetch_add(local_p4);
    }

    /// Strip-aware variant of `fused_sum_p4_kernel`.
    ///
    /// `plane` is laid out as `width × plane_h` row-major (where
    /// `width = plane.len() / plane_h` is the per-row stride). Only
    /// elements whose column index `col = i % width` falls in
    /// `[body_col_start, body_col_end)` are summed. Halo elements are
    /// dropped from both Σ and Σ⁴.
    ///
    /// Used by the strip-processing path (`compute_stripped`). For
    /// ssim2-gpu's transposed error-map orientation, the buffer's
    /// "column" axis corresponds to the original frame's Y axis, so
    /// the body-column range here is the body **row range** of the
    /// untransposed strip — see `pipeline.rs::compute_stripped` for
    /// the orientation mapping.
    #[cube(launch_unchecked)]
    fn fused_sum_p4_rows_kernel(
        plane: &Array<f32>,
        output_sums: &mut Array<Atomic<f32>>,
        slot: u32,
        width: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;

        let mut i = tid;
        while i < n {
            let col = (i as u32) % width;
            if col >= body_col_start && col < body_col_end {
                let v = plane[i];
                local_sum += v;
                let v2 = v * v;
                local_p4 += v2 * v2;
            }
            i += stride;
        }

        let off = (slot * 2) as usize;
        output_sums[off].fetch_add(local_sum);
        output_sums[off + 1].fetch_add(local_p4);
    }

    #[cube(launch_unchecked)]
    fn fused_sum_p4_batched_kernel(
        plane: &Array<f32>,
        output_sums: &mut Array<Atomic<f32>>,
        plane_stride: u32,
        num_slots: u32,
        slot: u32,
    ) {
        let batch_idx = CUBE_POS_Y;
        let tid_in_plane = UNIT_POS_X + CUBE_POS_X * (CUBE_DIM_X as u32);
        let stride_per_plane = CUBE_COUNT_X as u32 * (CUBE_DIM_X as u32);
        let plane_off = (batch_idx * plane_stride) as usize;
        let plane_us = plane_stride as usize;

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;
        let mut i = tid_in_plane as usize;
        while i < plane_us {
            let v = plane[plane_off + i];
            local_sum += v;
            let v2 = v * v;
            local_p4 += v2 * v2;
            i += stride_per_plane as usize;
        }

        let off = ((batch_idx * num_slots + slot) * 2) as usize;
        output_sums[off].fetch_add(local_sum);
        output_sums[off + 1].fetch_add(local_p4);
    }

    pub fn launch_sum_p4<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        unsafe {
            fused_sum_p4_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot,
            );
        }
    }

    /// Strip-aware launcher — sums only elements whose column index
    /// (in the `width`-strided plane) falls in
    /// `[body_col_start, body_col_end)`.
    pub fn launch_sum_p4_rows<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
        width: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        unsafe {
            fused_sum_p4_rows_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot,
                width,
                body_col_start,
                body_col_end,
            );
        }
    }

    pub fn launch_sum_p4_batched<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        plane_stride: u32,
        batch_size: u32,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        num_slots: u32,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, batch_size, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        unsafe {
            fused_sum_p4_batched_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    plane_handle,
                    (plane_stride as usize) * (batch_size as usize),
                ),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                plane_stride,
                num_slots,
                slot,
            );
        }
    }

    /// In fast mode the partials buffer already holds the final
    /// `(slot, sum, p4)` layout — `launch_sum_p4` writes there
    /// directly via atomic_add. We just copy partials → sums so the
    /// pipeline's read-back path is identical to portable mode.
    #[cube(launch_unchecked)]
    fn copy_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
        let i = ABSOLUTE_POS;
        if i >= dst.len() {
            terminate!();
        }
        dst[i] = src[i];
    }

    pub fn launch_finalize<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        _num_slots: u32,
    ) {
        const TPB: u32 = 64;
        let cubes = ((output_len as u32) + TPB - 1) / TPB;
        unsafe {
            copy_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes.max(1), 1, 1),
                CubeDim::new_1d(TPB),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
            );
        }
    }

    pub fn launch_finalize_batched<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        _batch_size: u32,
        _num_slots: u32,
    ) {
        const TPB: u32 = 64;
        let cubes = ((output_len as u32) + TPB - 1) / TPB;
        unsafe {
            copy_kernel::launch_unchecked::<R>(
                client,
                CubeCount::Static(cubes.max(1), 1, 1),
                CubeDim::new_1d(TPB),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
            );
        }
    }
}

// =====================================================================
// Portable path — per-thread partials + on-device finalizer kernel.
//
// **Default since task #52 (2026-05-26).** Each grid-strided thread
// writes its `(local_sum, local_p4)` partial to its own scratch slot
// (indexed by `slot_offset + tid * 2`), then `finalize_sum_p4_kernel`
// sums all 4096 partials in a fixed `k = 0..n_threads` order from
// a single cube. f32 add isn't associative but the summation order
// IS deterministic — so two runs of the same input produce
// bit-identical scores, vs. the ~5e-5 reorder noise the
// fast-reduction `Atomic<f32>::fetch_add` path leaks.
//
// This also INDIRECTLY restores Metal support: cubecl-wgpu's Metal
// backend reports `Atomic<f32> = LoadStore|Add` as supported, but
// the codegen silently no-ops `fetch_add` at execution time —
// every reduction returned zero, every score collapsed to ~100.
// The portable path uses plain stores, so it works on Metal as
// well as any backend that supports plain f32 array writes.
//
// Trade-off: the partials buffer is `THREADS_PER_REDUCTION * 2` =
// 8192 f32s per slot vs. 2 f32s per slot in fast mode (~4096×
// memory amplification), and the single-cube finalize is sequential.
// On CUDA this is measurably slower than the atomic-add path for
// tiny images but disappears in the noise above 256×256.
// =====================================================================

#[cfg(not(feature = "fast-reduction"))]
mod portable {
    use super::*;

    /// u32 mirrors of the usize constants for use inside `#[cube]`
    /// kernel bodies (cubecl 0.10's macro fights with `usize` const-expr
    /// arithmetic inside kernels).
    const PARTIALS_PER_REDUCTION_U32: u32 = NUM_BLOCKS * BLOCK_SIZE * 2;
    const THREADS_PER_REDUCTION_U32: u32 = NUM_BLOCKS * BLOCK_SIZE;

    #[cube(launch_unchecked)]
    fn thread_sum_p4_kernel(plane: &Array<f32>, output: &mut Array<f32>, slot_offset: u32) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;
        let mut i = tid;
        while i < n {
            let v = plane[i];
            local_sum += v;
            let v2 = v * v;
            local_p4 += v2 * v2;
            i += stride;
        }

        let off = (slot_offset as usize) + tid * 2;
        output[off] = local_sum;
        output[off + 1] = local_p4;
    }

    /// Strip-aware portable variant. Writes per-thread partials,
    /// then a separate finalize launch folds them.
    ///
    /// IMPORTANT: in the portable path, `launch_sum_p4_rows` must be
    /// chained with a partials-zero-fill before the strip *and*
    /// accumulated host-side across strips into the same slot. The
    /// current implementation reuses the same partials slot — so the
    /// driver runs the finalize once per strip into a host-side
    /// accumulator. See `pipeline.rs::compute_stripped` for the
    /// host-side accumulation loop.
    #[cube(launch_unchecked)]
    fn thread_sum_p4_rows_kernel(
        plane: &Array<f32>,
        output: &mut Array<f32>,
        slot_offset: u32,
        width: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        let tid = ABSOLUTE_POS;
        let stride = CUBE_COUNT * (CUBE_DIM_X as usize);
        let n = plane.len();

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;
        let mut i = tid;
        while i < n {
            let col = (i as u32) % width;
            if col >= body_col_start && col < body_col_end {
                let v = plane[i];
                local_sum += v;
                let v2 = v * v;
                local_p4 += v2 * v2;
            }
            i += stride;
        }

        let off = (slot_offset as usize) + tid * 2;
        output[off] = local_sum;
        output[off + 1] = local_p4;
    }

    #[cube(launch_unchecked)]
    fn thread_sum_p4_batched_kernel(
        plane: &Array<f32>,
        output: &mut Array<f32>,
        plane_stride: u32,
        image_stride: u32,
        slot_offset: u32,
    ) {
        let tid_in_plane = UNIT_POS_X + CUBE_POS_X * CUBE_DIM_X;
        let stride_per_image = CUBE_COUNT_X * CUBE_DIM_X;
        let batch_idx = CUBE_POS_Y;
        let plane_us = plane_stride as usize;
        let plane_off = (batch_idx * plane_stride) as usize;
        let img_off = (batch_idx * image_stride + slot_offset) as usize;

        let mut local_sum = 0.0_f32;
        let mut local_p4 = 0.0_f32;
        let mut i = tid_in_plane as usize;
        while i < plane_us {
            let v = plane[plane_off + i];
            local_sum += v;
            let v2 = v * v;
            local_p4 += v2 * v2;
            i += stride_per_image as usize;
        }

        let out_idx = img_off + (tid_in_plane as usize) * 2;
        output[out_idx] = local_sum;
        output[out_idx + 1] = local_p4;
    }

    #[cube(launch_unchecked)]
    fn finalize_sum_p4_kernel(partials: &Array<f32>, output: &mut Array<f32>) {
        let slot = CUBE_POS_X;
        let in_off = (slot * PARTIALS_PER_REDUCTION_U32) as usize;
        let out_off = (slot * 2) as usize;
        let mut sum = 0.0_f32;
        let mut p4 = 0.0_f32;
        let mut k: u32 = 0;
        while k < THREADS_PER_REDUCTION_U32 {
            sum += partials[in_off + (k as usize) * 2];
            p4 += partials[in_off + (k as usize) * 2 + 1];
            k += 1;
        }
        output[out_off] = sum;
        output[out_off + 1] = p4;
    }

    #[cube(launch_unchecked)]
    fn finalize_sum_p4_batched_kernel(
        partials: &Array<f32>,
        output: &mut Array<f32>,
        num_slots: u32,
    ) {
        let slot = CUBE_POS_X;
        let batch_idx = CUBE_POS_Y;
        let img_off = batch_idx * num_slots;
        let in_off = ((img_off + slot) * PARTIALS_PER_REDUCTION_U32) as usize;
        let out_off = ((img_off + slot) * 2) as usize;
        let mut sum = 0.0_f32;
        let mut p4 = 0.0_f32;
        let mut k: u32 = 0;
        while k < THREADS_PER_REDUCTION_U32 {
            sum += partials[in_off + (k as usize) * 2];
            p4 += partials[in_off + (k as usize) * 2 + 1];
            k += 1;
        }
        output[out_off] = sum;
        output[out_off + 1] = p4;
    }

    pub fn launch_sum_p4<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);
        unsafe {
            thread_sum_p4_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot_offset,
            );
        }
    }

    /// Strip-aware portable launcher — sums only body-column elements.
    pub fn launch_sum_p4_rows<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        n_pixels: usize,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        slot: u32,
        width: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, 1, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);
        unsafe {
            thread_sum_p4_rows_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(plane_handle, n_pixels),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                slot_offset,
                width,
                body_col_start,
                body_col_end,
            );
        }
    }

    pub fn launch_sum_p4_batched<R: Runtime>(
        client: &ComputeClient<R>,
        plane_handle: cubecl::server::Handle,
        plane_stride: u32,
        batch_size: u32,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        num_slots: u32,
        slot: u32,
    ) {
        let cube_count = CubeCount::Static(NUM_BLOCKS, batch_size, 1);
        let cube_dim = CubeDim::new_1d(BLOCK_SIZE);
        let image_stride = num_slots * (PARTIALS_PER_REDUCTION as u32);
        let slot_offset = slot * (PARTIALS_PER_REDUCTION as u32);
        unsafe {
            thread_sum_p4_batched_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(
                    plane_handle,
                    (plane_stride as usize) * (batch_size as usize),
                ),
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                plane_stride,
                image_stride,
                slot_offset,
            );
        }
    }

    pub fn launch_finalize<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        num_slots: u32,
    ) {
        let cube_count = CubeCount::Static(num_slots, 1, 1);
        let cube_dim = CubeDim::new_1d(1);
        unsafe {
            finalize_sum_p4_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
            );
        }
    }

    pub fn launch_finalize_batched<R: Runtime>(
        client: &ComputeClient<R>,
        partials_handle: cubecl::server::Handle,
        partials_len: usize,
        output_handle: cubecl::server::Handle,
        output_len: usize,
        batch_size: u32,
        num_slots: u32,
    ) {
        let cube_count = CubeCount::Static(num_slots, batch_size, 1);
        let cube_dim = CubeDim::new_1d(1);
        unsafe {
            finalize_sum_p4_batched_kernel::launch_unchecked::<R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(partials_handle, partials_len),
                ArrayArg::from_raw_parts(output_handle, output_len),
                num_slots,
            );
        }
    }
}

#[cfg(feature = "fast-reduction")]
pub use fast::{
    launch_finalize, launch_finalize_batched, launch_sum_p4, launch_sum_p4_batched,
    launch_sum_p4_rows,
};

#[cfg(not(feature = "fast-reduction"))]
pub use portable::{
    launch_finalize, launch_finalize_batched, launch_sum_p4, launch_sum_p4_batched,
    launch_sum_p4_rows,
};

// =====================================================================
// On-device zero-fill (T_x.A). Avoids per-call host→device upload of a
// fresh `vec![0.0_f32; PARTIALS_LEN]` (~1.77 MB for the 54-slot
// partials buffer). On CUDA this saves a cuMemcpyHtoDAsync per call.
// =====================================================================

#[cube(launch_unchecked)]
fn zero_fill_f32_kernel(buf: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= buf.len() {
        terminate!();
    }
    buf[i] = 0.0_f32;
}

/// Zero a flat f32 buffer on-device. Equivalent to uploading a
/// `vec![0.0_f32; len]` from the host but doesn't touch host memory or
/// the PCIe bus.
pub fn launch_zero_fill_f32<R: Runtime>(
    client: &ComputeClient<R>,
    buf: cubecl::server::Handle,
    len: usize,
) {
    const TPB: u32 = 256;
    let cubes = (len as u32).div_ceil(TPB);
    unsafe {
        zero_fill_f32_kernel::launch_unchecked::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(TPB),
            ArrayArg::from_raw_parts(buf, len),
        );
    }
}
