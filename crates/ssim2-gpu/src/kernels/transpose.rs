//! Plane transpose.
//!
//! Used between the two recursive-Gaussian passes so the IIR walks
//! columns both times — exactly how the CUDA reference structures its
//! 2D blur.
//!
//! ## Tiled implementation (T_x.B, 2026-05-17)
//!
//! `transpose_kernel` uses the classic 32×32 LDS-tiled transpose with
//! a 1-column pad on the shared-memory buffer to avoid 32-way bank
//! conflicts on the column reads. Both global loads and stores are
//! coalesced. Re-derived from NVIDIA's well-known efficient-matrix-
//! transpose sample, rewritten in CubeCL idioms.

use cubecl::prelude::*;

/// Side length of the LDS tile (each cube transposes a TILE_DIM × TILE_DIM
/// block). 32 = one warp wide on every NVIDIA/AMD shipping GPU; keeps
/// each row's coalesced loads inside one transaction.
pub const TILE_DIM: u32 = 32;
const PAD: u32 = 1;
const TILE_DIM_USIZE: usize = TILE_DIM as usize;
const ROW_STRIDE: usize = TILE_DIM_USIZE + PAD as usize;
const PADDED: usize = TILE_DIM_USIZE * ROW_STRIDE;
/// Threads per cube on each axis. Match TILE_DIM so each cube has 32×32
/// = 1024 threads, the maximum on every modern NVIDIA SM.
pub const TPB_X: u32 = TILE_DIM;
pub const TPB_Y: u32 = TILE_DIM;

/// Transpose `src` (`width × height`, row-major) into `dst`
/// (`height × width`, row-major). Cube grid is
/// `(width.div_ceil(TILE_DIM), height.div_ceil(TILE_DIM), 1)`.
#[cube(launch_unchecked)]
pub fn transpose_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let mut tile = SharedMemory::<f32>::new(PADDED);

    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let bx = CUBE_POS_X * TILE_DIM;
    let by = CUBE_POS_Y * TILE_DIM;

    // 1) Coalesced read: row (by + ty), column (bx + tx). Adjacent
    //    threads in the warp read adjacent src cells.
    let x_in = bx + tx;
    let y_in = by + ty;
    if x_in < width && y_in < height {
        let src_idx = (y_in as usize) * (width as usize) + (x_in as usize);
        let lds_idx = (ty as usize) * ROW_STRIDE + (tx as usize);
        tile[lds_idx] = src[src_idx];
    }

    sync_cube();

    // 2) Coalesced write: dst is `height × width`. Output row (bx + ty),
    //    column (by + tx). Adjacent threads (varying tx) again write
    //    adjacent dst cells. Tile read at `[tx, ty]` — column-direction
    //    access into LDS, conflict-free only because of the +1 pad.
    let x_out = by + tx;
    let y_out = bx + ty;
    if x_out < height && y_out < width {
        let lds_idx = (tx as usize) * ROW_STRIDE + (ty as usize);
        let v = tile[lds_idx];
        let dst_idx = (y_out as usize) * (height as usize) + (x_out as usize);
        dst[dst_idx] = v;
    }
}

/// Per-image transpose for batched buffers. Each plane is
/// `width × height`, stored at `plane_stride` floats apart in `src`
/// and `dst`.
#[cube(launch_unchecked)]
pub fn transpose_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= dst.len() {
        terminate!();
    }
    let pl = plane_stride as usize;
    let batch_idx = idx / pl;
    let local = idx - batch_idx * pl;
    let h = height as usize;
    let w = width as usize;
    if local >= w * h {
        terminate!();
    }
    let yt = local / h;
    let xt = local - yt * h;
    let plane_off = batch_idx * pl;
    dst[idx] = src[plane_off + xt * w + yt];
}
