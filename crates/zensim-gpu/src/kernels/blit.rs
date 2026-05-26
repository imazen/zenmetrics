//! Row-range device-to-device blit kernel.
//!
//! Used by Mode E (cached-ref + strip) to slice a sub-range of rows
//! out of a full-image-sized buffer into a strip-sized buffer ahead
//! of the per-strip fused-features + masked-IW passes. The downscale
//! pyramid is deterministic on consecutive 2-row pairs, so when the
//! strip's image-coord `up_lo` is a multiple of `2^(SCALES-1)`
//! (= [`crate::pipeline::STRIP_ALIGN`]), strip-buffer scale-s row r
//! corresponds bit-exactly to full-buffer scale-s row `(up_lo/2^s + r)`.
//!
//! Mirrors `dssim-gpu::pipeline::copy_rows_kernel`.

use cubecl::prelude::*;

/// Copies `n_rows × width` f32 values from row `src_row_start` of `src`
/// into row 0 of `dst`. `src` is laid out as
/// `src_total_rows × width` row-major (caller passes the **buffer
/// length** in elements so cubecl's typed bounds check accepts the
/// dispatch — only the indexed region is read). `dst` is the
/// strip-sized destination; threads beyond `n_rows × width` exit.
#[cube(launch_unchecked)]
pub fn copy_rows_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    n_rows: u32,
    src_row_start: u32,
) {
    let i = ABSOLUTE_POS;
    let w = width as usize;
    let limit = (n_rows as usize) * w;
    if i >= limit {
        terminate!();
    }
    let row = i / w;
    let col = i - row * w;
    let src_idx = ((src_row_start as usize) + row) * w + col;
    dst[i] = src[src_idx];
}
