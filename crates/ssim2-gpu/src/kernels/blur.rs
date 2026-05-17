//! Gaussian blur kernels for SSIMULACRA2 at σ = 1.5.
//!
//! ## T_y.B.1 (2026-05-17): separable FIR D=5
//!
//! Replaces the prior recursive Charalampidis IIR (kept here as
//! `blur_pass_kernel` / `blur_pass_batched_kernel` for reference and
//! A/B benching) with a **separable 5-tap truncated-Gaussian FIR**
//! per Kanetaka et al., "Fast Implementation of SSIMULACRA2 for Image
//! Quality Assessment", IWAIT 2026.
//!
//! Why D=5:
//!
//! - **Speed.** Recursive IIR is one thread per column walking all
//!   rows sequentially with 6 floats of state and a ring buffer for
//!   the lookback term. Per-thread parallelism is column-count, and
//!   each thread serialises height-many steps. The separable FIR is
//!   one thread per OUTPUT pixel — full parallelism, K reads per
//!   output, no inter-step state. On a 5-tap kernel this is a strict
//!   win at any image size where launch overhead is small vs total
//!   work.
//! - **Accuracy.** Per Kanetaka Table 2 on CID22 (512×512), D=5 hits
//!   SROCC 0.890387 vs the libjxl reference's 0.889297 — i.e. the
//!   truncated-Gaussian 5-tap is **more accurate** than the
//!   "constant-time" recursive Gaussian we're replacing. D=3 loses
//!   0.005 SROCC; D=11 starts losing again. D=5 is the optimum.
//! - **No transpose between passes.** The FIR runs in input
//!   orientation directly — `blur_h_fir5_kernel` (horizontal,
//!   one-output-per-thread, K reads along row) followed by
//!   `blur_v_fir5_kernel` (vertical, K reads down column). Output
//!   stays row-major in input orientation. The current pipeline's
//!   IIR-era intermediate transpose between vertical passes is no
//!   longer needed.
//!
//! ## Boundary handling — ZERO PADDING
//!
//! The libjxl SSIMULACRA2 recursive Gaussian (`ssimulacra2::Blur`,
//! the Charalampidis truncated-cosine IIR our CPU reference uses)
//! **does NOT reflect-pad**. It seeds the IIR state to zero, walks the
//! signal in-place, and produces a darkened "halo" of width ~3σ at the
//! borders (verified empirically: blurring a uniform 0.5 plane at
//! σ = 1.5 yields ~0.20 at the corners, not 0.5). The downstream
//! SSIM stat computation is calibrated for this convention — any
//! border-preserving FIR would shift the per-pixel SSIM-error map by
//! a large amount in the (proportionally significant) border region,
//! and the final score by tens of points.
//!
//! To preserve the SSIMULACRA2 score's libjxl-IIR-compatible
//! magnitude AND benefit from the paper's accuracy improvement (Table
//! 2, D=5 SROCC 0.890387 > libjxl 0.889297), the FIR uses the same
//! **zero-padding** convention: out-of-frame samples contribute zero
//! to the accumulator. This is the simplest convention to evaluate
//! (a single branch per sample) AND the one that matches the libjxl
//! reference for direct score comparability.
//!
//! Implementation: for offset `k ∈ {-2,…,2}` from output position
//! `(x, y)`, sample at `(x + k_dx, y + k_dy)` if in-frame, else 0.
//!
//! ## Coefficients
//!
//! Generated at build time by `build.rs` from
//! `g(x) = exp(-x² / (2σ²))` with σ = 1.5, normalized to sum 1:
//!
//! ```text
//! [0.12016, 0.23383, 0.29203, 0.23383, 0.12016]
//! ```
//!
//! See the file-level doc on the old IIR kernels below for the prior
//! design discussion.

use cubecl::prelude::*;

mod consts {
    #![allow(clippy::unreadable_literal, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

// IIR constants (still re-exported for any external consumer; the
// pipeline no longer uses them after T_y.B.1).
pub use consts::RADIUS;
pub const RADIUS_U32: u32 = consts::RADIUS as u32;

const RING_SIZE: u32 = RADIUS_U32 * 2 + 1;
const RING_SIZE_USIZE: usize = consts::RADIUS * 2 + 1;
const TWO_N: u32 = 2 * RADIUS_U32;

// ───────────────── T_y.B.1 separable FIR D=5 ─────────────────

/// FIR kernel radius (taps on each side of centre). 2 means 5 taps total.
pub const FIR_RADIUS: u32 = consts::FIR_RADIUS as u32;
/// FIR kernel diameter (total taps).
pub const FIR_TAPS: u32 = consts::FIR_TAPS as u32;
/// Compile-time check that the build-side `FIR_RADIUS` is 2 (5 taps).
/// If you change the kernel diameter, the `reflect_idx_*` clamp logic
/// below needs to be regeneralised.
const _: () = assert!(consts::FIR_RADIUS == 2);

/// Horizontal 5-tap FIR pass with ZERO padding (libjxl-IIR-compatible).
///
/// One thread per output pixel. Reads 5 samples along the row (clamped
/// + reflected at the left/right edges) and accumulates a normalized
/// Gaussian convolution. Output is row-major, same orientation as
/// input.
///
/// Launch geometry: `cube_count_1d(width * height)`, `cube_dim_1d(256)`.
/// The kernel's per-thread index `idx` decomposes into `(y, x)` via
/// `y = idx / width`, `x = idx % width`.
#[cube(launch_unchecked)]
pub fn blur_h_fir5_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let n = (width * height) as usize;
    if idx >= n {
        terminate!();
    }
    let idx_u = idx as u32;
    let y = idx_u / width;
    let x = idx_u % width;
    let row_base = (y * width) as usize;

    // Unrolled 5-tap H pass: x-2, x-1, x, x+1, x+2 with ZERO padding.
    // Out-of-frame samples contribute 0 (libjxl IIR convention).
    let s_m2 = if x >= 2u32 {
        src[row_base + ((x - 2u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_m1 = if x >= 1u32 {
        src[row_base + ((x - 1u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_0 = src[row_base + (x as usize)];
    let s_p1 = if x + 1u32 < width {
        src[row_base + ((x + 1u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_p2 = if x + 2u32 < width {
        src[row_base + ((x + 2u32) as usize)]
    } else {
        f32::new(0.0)
    };

    // Symmetric taps: |k|=2 → FIR_TAP_0, |k|=1 → FIR_TAP_1, |k|=0 → FIR_TAP_2.
    let acc = s_0 * consts::FIR_TAP_2
        + (s_m1 + s_p1) * consts::FIR_TAP_1
        + (s_m2 + s_p2) * consts::FIR_TAP_0;
    dst[idx] = acc;
}

/// Vertical 5-tap FIR pass with ZERO padding (libjxl-IIR-compatible).
///
/// One thread per output pixel. Reads 5 samples down the column
/// (clamped + reflected at the top/bottom edges) and accumulates a
/// normalized Gaussian convolution. Output is row-major, same
/// orientation as input.
#[cube(launch_unchecked)]
pub fn blur_v_fir5_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let idx = ABSOLUTE_POS;
    let n = (width * height) as usize;
    if idx >= n {
        terminate!();
    }
    let idx_u = idx as u32;
    let y = idx_u / width;
    let x = idx_u % width;
    let w = width;

    let s_m2 = if y >= 2u32 {
        src[(((y - 2u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_m1 = if y >= 1u32 {
        src[(((y - 1u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_0 = src[((y * w) + x) as usize];
    let s_p1 = if y + 1u32 < height {
        src[(((y + 1u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_p2 = if y + 2u32 < height {
        src[(((y + 2u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };

    let acc = s_0 * consts::FIR_TAP_2
        + (s_m1 + s_p1) * consts::FIR_TAP_1
        + (s_m2 + s_p2) * consts::FIR_TAP_0;
    dst[idx] = acc;
}

/// Batched horizontal 5-tap FIR. `plane_stride = width * height`;
/// `batch_size` planes packed contiguously in `src` / `dst`. Launch
/// geometry: `cube_count = (ceil(plane_stride / 256), batch_size, 1)`
/// — same per-image work, batched by `CUBE_POS_Y`.
#[cube(launch_unchecked)]
pub fn blur_h_fir5_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let local = UNIT_POS_X + CUBE_POS_X * CUBE_DIM_X;
    if local >= plane_stride {
        terminate!();
    }
    let y = local / width;
    let x = local % width;
    let plane_off = (batch_idx * plane_stride) as usize;
    let row_base = (y * width) as usize;

    let s_m2 = if x >= 2u32 {
        src[plane_off + row_base + ((x - 2u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_m1 = if x >= 1u32 {
        src[plane_off + row_base + ((x - 1u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_0 = src[plane_off + row_base + (x as usize)];
    let s_p1 = if x + 1u32 < width {
        src[plane_off + row_base + ((x + 1u32) as usize)]
    } else {
        f32::new(0.0)
    };
    let s_p2 = if x + 2u32 < width {
        src[plane_off + row_base + ((x + 2u32) as usize)]
    } else {
        f32::new(0.0)
    };

    let acc = s_0 * consts::FIR_TAP_2
        + (s_m1 + s_p1) * consts::FIR_TAP_1
        + (s_m2 + s_p2) * consts::FIR_TAP_0;
    dst[plane_off + (local as usize)] = acc;

    // height kept in signature for API parity with the V kernel; unused.
    let _ = height;
}

/// Batched vertical 5-tap FIR. See `blur_h_fir5_batched_kernel` doc.
#[cube(launch_unchecked)]
pub fn blur_v_fir5_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let local = UNIT_POS_X + CUBE_POS_X * CUBE_DIM_X;
    if local >= plane_stride {
        terminate!();
    }
    let y = local / width;
    let x = local % width;
    let plane_off = (batch_idx * plane_stride) as usize;
    let w = width;

    let s_m2 = if y >= 2u32 {
        src[plane_off + (((y - 2u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_m1 = if y >= 1u32 {
        src[plane_off + (((y - 1u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_0 = src[plane_off + ((y * w) + x) as usize];
    let s_p1 = if y + 1u32 < height {
        src[plane_off + (((y + 1u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };
    let s_p2 = if y + 2u32 < height {
        src[plane_off + (((y + 2u32) * w) + x) as usize]
    } else {
        f32::new(0.0)
    };

    let acc = s_0 * consts::FIR_TAP_2
        + (s_m1 + s_p1) * consts::FIR_TAP_1
        + (s_m2 + s_p2) * consts::FIR_TAP_0;
    dst[plane_off + (local as usize)] = acc;
}

// ───────────────── Legacy Charalampidis IIR (pre-T_y.B.1) ─────────────────
//
// Kept compilable for reference. No longer wired into the pipeline.
// The IIR carries six floats of state per column-walker
// (`prev_{1,3,5}`, `prev2_{1,3,5}`) and a `2·N + 1` ring buffer for
// the lookback term `y − N − 1`. One thread = one column.

/// (Legacy IIR) One thread = one column. Walks `y` from `−N + 1` to
/// `height − 1` and emits one output per non-negative `y`.
#[cube(launch_unchecked)]
pub fn blur_pass_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    let x = ABSOLUTE_POS;
    if x >= width as usize {
        terminate!();
    }

    let mut ring = SharedMemory::<f32>::new(BLOCK_TIMES_RING_USIZE);
    let tx = UNIT_POS_X as usize;
    let ring_base = tx * RING_SIZE_USIZE;

    let mut k: u32 = 0;
    while k < RING_SIZE {
        ring[ring_base + (k as usize)] = f32::new(0.0);
        k += 1;
    }

    let mut prev_1 = 0.0_f32;
    let mut prev_3 = 0.0_f32;
    let mut prev_5 = 0.0_f32;
    let mut prev2_1 = 0.0_f32;
    let mut prev2_3 = 0.0_f32;
    let mut prev2_5 = 0.0_f32;

    let h = height;
    let w = width as usize;

    let span = h + RADIUS_U32 - 1;
    let mut i: u32 = 0;
    while i < span {
        let right = i;
        let left_present = i >= TWO_N;
        let y_emit = i + 1 >= RADIUS_U32;

        let right_val = if right < h {
            src[(right as usize) * w + x]
        } else {
            f32::new(0.0)
        };

        let left_val = if left_present {
            let slot = (i - TWO_N) % RING_SIZE;
            ring[ring_base + (slot as usize)]
        } else {
            f32::new(0.0)
        };

        let sum = left_val + right_val;

        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 += consts::MUL_PREV2_1 * prev2_1;
        out_3 += consts::MUL_PREV2_3 * prev2_3;
        out_5 += consts::MUL_PREV2_5 * prev2_5;
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 += consts::MUL_PREV_1 * prev_1;
        out_3 += consts::MUL_PREV_3 * prev_3;
        out_5 += consts::MUL_PREV_5 * prev_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if y_emit {
            let y = i + 1 - RADIUS_U32;
            if y < h {
                dst[(y as usize) * w + x] = out_1 + out_3 + out_5;
            }
        }

        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}

/// (Legacy IIR) Threads-per-block for the IIR kernel.
pub const BLOCK_WIDTH: u32 = 96;
const BLOCK_WIDTH_USIZE: usize = 96;
const BLOCK_TIMES_RING_USIZE: usize = BLOCK_WIDTH_USIZE * RING_SIZE_USIZE;

/// (Legacy IIR) Batched recursive Gaussian column walk.
#[cube(launch_unchecked)]
pub fn blur_pass_batched_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    width: u32,
    height: u32,
    plane_stride: u32,
) {
    let batch_idx = CUBE_POS_Y;
    let x = UNIT_POS_X + CUBE_POS_X * CUBE_DIM_X;
    if x >= width {
        terminate!();
    }
    let plane_off = (batch_idx * plane_stride) as usize;

    let mut ring = SharedMemory::<f32>::new(BLOCK_TIMES_RING_USIZE);
    let tx = UNIT_POS_X as usize;
    let ring_base = tx * RING_SIZE_USIZE;

    let mut k: u32 = 0;
    while k < RING_SIZE {
        ring[ring_base + (k as usize)] = f32::new(0.0);
        k += 1;
    }

    let mut prev_1 = 0.0_f32;
    let mut prev_3 = 0.0_f32;
    let mut prev_5 = 0.0_f32;
    let mut prev2_1 = 0.0_f32;
    let mut prev2_3 = 0.0_f32;
    let mut prev2_5 = 0.0_f32;

    let h = height;
    let w = width as usize;

    let span = h + RADIUS_U32 - 1;
    let mut i: u32 = 0;
    while i < span {
        let right = i;
        let left_present = i >= TWO_N;
        let y_emit = i + 1 >= RADIUS_U32;

        let right_val = if right < h {
            src[plane_off + (right as usize) * w + (x as usize)]
        } else {
            f32::new(0.0)
        };

        let left_val = if left_present {
            let slot = (i - TWO_N) % RING_SIZE;
            ring[ring_base + (slot as usize)]
        } else {
            f32::new(0.0)
        };

        let sum = left_val + right_val;

        let mut out_1 = sum * consts::MUL_IN_1;
        let mut out_3 = sum * consts::MUL_IN_3;
        let mut out_5 = sum * consts::MUL_IN_5;

        out_1 += consts::MUL_PREV2_1 * prev2_1;
        out_3 += consts::MUL_PREV2_3 * prev2_3;
        out_5 += consts::MUL_PREV2_5 * prev2_5;
        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;

        out_1 += consts::MUL_PREV_1 * prev_1;
        out_3 += consts::MUL_PREV_3 * prev_3;
        out_5 += consts::MUL_PREV_5 * prev_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if y_emit {
            let y = i + 1 - RADIUS_U32;
            if y < h {
                dst[plane_off + (y as usize) * w + (x as usize)] = out_1 + out_3 + out_5;
            }
        }

        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}
