//! Gaussian blur kernels for SSIMULACRA2 at σ = 1.5.
//!
//! Two implementations live here:
//!
//! - **Recursive IIR** (default, `Ssim2Blur::Iir`). The Charalampidis
//!   [2016] truncated-cosine sliding-DCT recursive Gaussian — bit-identical
//!   to the published `ssimulacra2` CPU crate and to the `ssimulacra2-cuda`
//!   GPU reference. Kernels: `blur_pass_kernel`, `blur_pass_batched_kernel`.
//!
//! - **Separable 5-tap FIR** (opt-in, `Ssim2Blur::Fir`). The truncated
//!   Gaussian D=5 from Kanetaka et al., "Fast Implementation of
//!   SSIMULACRA2 for Image Quality Assessment", IWAIT 2026 (DOI
//!   10.1117/12.3100969). Per the paper's Table 2 on CID22, D=5 hits
//!   SROCC 0.890387 — slightly **higher** than the libjxl reference's
//!   0.889297 — but the per-image score values are NOT the same scale
//!   as the IIR (the FIR's effective impulse-response support is
//!   narrower). Treat as a **distinct metric**, not a faster/equivalent
//!   reimplementation. Kernels: `blur_h_fir5_kernel`,
//!   `blur_h_fir5_batched_kernel`.
//!
//! Both modes use the same `vpass → transpose → vpass` pipeline
//! structure, just with different per-pass kernels. The FIR's
//! horizontal pass is reused as the "vertical" pass via the
//! intermediate transpose — see `pipeline.rs::blur_plane_two_pass`.
//!
//! ## FIR boundary handling — ZERO PADDING
//!
//! The libjxl SSIMULACRA2 recursive Gaussian (the CPU reference our
//! IIR path matches bit-for-bit) **does NOT reflect-pad**. It seeds
//! the IIR state to zero and walks in-place, producing a darkened
//! "halo" at borders. The FIR uses the same **zero-padding**
//! convention: out-of-frame samples contribute zero. This is the
//! simplest convention to evaluate (single branch per sample) AND the
//! one most consistent with the libjxl SSIM stat normalization
//! downstream. The remaining IIR-vs-FIR per-image divergence is
//! algorithmic (5-tap truncated Gaussian vs 4-radius recursive
//! Gaussian impulse response), not a boundary-handling artefact.
//!
//! ## IIR file-level notes (legacy detail)
//!
//! The recursive Gaussian carries six floats of state per column-walker
//! (`prev_{1,3,5}`, `prev2_{1,3,5}`) and a `2·N + 1` ring buffer for
//! the lookback term `y − N − 1`.
//!
//! ## Two passes, one direction
//!
//! Each kernel invocation does a single top-to-bottom column walk
//! ("vertical pass"). To get a 2D blur the host runs:
//!
//! 1. `blur_pass_kernel` on the source (walks columns) → `_v`.
//! 2. Transpose `_v` → `_vt`.
//! 3. `blur_pass_kernel` on `_vt` (walks columns of the transposed —
//!    i.e. rows of the original).
//!
//! The output stays in transposed coords, which suits the rest of the
//! pipeline (compute_error_maps + reduction are orientation-agnostic).
//!
//! That's exactly the structure `ssimulacra2-cuda/src/lib.rs::process_scale`
//! uses (the CUDA crate just folds 5 source planes into one launch via
//! `block_idx_y`; we drop that fanout — cubecl's grid_y for "pick which
//! src/dst pair" is awkward — and launch once per plane).
//!
//! ## Coefficients
//!
//! Generated at build time by `build.rs` (Charalampidis recurrence on
//! `SIGMA = 1.5`); identical values to the CPU `ssimulacra2` and CUDA
//! `ssimulacra2-cuda-kernel` build scripts. Imported via `consts::*`.
//!
//! ## Recurrence form
//!
//! The kernel uses the "single-step" formulation that matches the
//! CUDA `blur_plane_pass_fused`:
//!
//! ```text
//! out_k = sum * MUL_IN_k + MUL_PREV2_k * prev2_k + MUL_PREV_k * prev_k
//! prev2_k = prev_k
//! prev_k  = out_k
//! ```
//!
//! The "vertical" pass in the CPU `RecursiveGaussian::vertical_pass`
//! uses an algebraically-equivalent variant with VERT_MUL_* constants
//! (sign flipped: `out = sum*VERT_MUL_IN - VERT_MUL_PREV*prev - prev2`).
//! Either form yields the same f32 sequence at single-step granularity;
//! we use the MUL_IN_*/MUL_PREV*_* form so we share build.rs constants
//! 1:1 with the CUDA crate.

use cubecl::prelude::*;

mod consts {
    #![allow(clippy::unreadable_literal, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/recursive_gaussian.rs"));
}

pub use consts::RADIUS;
pub const RADIUS_U32: u32 = consts::RADIUS as u32;

const RING_SIZE: u32 = RADIUS_U32 * 2 + 1;
const RING_SIZE_USIZE: usize = consts::RADIUS * 2 + 1;
const TWO_N: u32 = 2 * RADIUS_U32;

/// FIR kernel radius (taps on each side of centre). 2 means 5 taps total.
#[cfg(feature = "fir")]
pub const FIR_RADIUS: u32 = consts::FIR_RADIUS as u32;
/// FIR kernel diameter (total taps).
#[cfg(feature = "fir")]
pub const FIR_TAPS: u32 = consts::FIR_TAPS as u32;
/// Compile-time check that the build-side `FIR_RADIUS` is 2 (5 taps).
/// If you change the kernel diameter, the boundary-clamp logic in the
/// FIR kernels below needs to be regeneralised.
#[cfg(feature = "fir")]
const _: () = assert!(consts::FIR_RADIUS == 2);

/// Threads-per-block for the FIR kernels. 256 is a round-warp count
/// that matches the rest of the pipeline's pointwise kernels and gives
/// good occupancy on every modern GPU (32-wide SIMD = 8 warps per cube).
#[cfg(feature = "fir")]
pub const FIR_BLOCK_WIDTH: u32 = 256;

/// One thread = one column. Walks `y` from `−N + 1` to `height − 1` and
/// emits one output per non-negative `y`.
///
/// `src` and `dst` are single-plane f32 of length `width × height`,
/// row-major. Coefficients come from `consts` (build.rs).
#[cube(launch_unchecked)]
pub fn blur_pass_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    // x = absolute column index across the launch grid.
    let x = ABSOLUTE_POS;
    if x >= width as usize {
        terminate!();
    }

    // Per-thread ring buffer. 1 row per thread × RING_SIZE columns;
    // flatten into a 1D shared array indexed by `tx * RING_SIZE + slot`.
    // (cubecl SharedMemory is 1D and `usize`-indexed.)
    let mut ring = SharedMemory::<f32>::new(BLOCK_TIMES_RING_USIZE);
    let tx = UNIT_POS_X as usize;
    let ring_base = tx * RING_SIZE_USIZE;

    // Zero the ring slots for this thread.
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

    // y ranges over `(-N + 1) ..= (height - 1)`. Use u32 with a phase
    // offset of `N - 1` so the loop variable is non-negative; the
    // mathematical y is `i - (N - 1)`.
    let span = h + RADIUS_U32 - 1; // = height + N - 1 iterations
    let mut i: u32 = 0;
    while i < span {
        // Mathematical y = i - (N - 1); we stay in u32 by tracking
        // `right = y + N - 1 = i` and `left = y - N - 1 = i - 2N`.
        let right = i; // = y + N - 1
        let left_present = i >= TWO_N; // y - N - 1 >= 0  <=>  i >= 2N
        let y_emit = i + 1 >= RADIUS_U32; // y >= 0  <=>  i + 1 >= N

        let right_val = if right < h {
            src[(right as usize) * w + x]
        } else {
            f32::new(0.0)
        };

        let left_val = if left_present {
            // left = i - 2N
            let slot = (i - TWO_N) % RING_SIZE;
            ring[ring_base + (slot as usize)]
        } else {
            f32::new(0.0)
        };

        let sum = left_val + right_val;

        // Three IIR taps; parallel; six FMA-able terms.
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
            // y = i - (N - 1), and y_emit means y >= 0; we still need to
            // mask `y < height` since the IIR keeps stepping for N-1
            // extra iterations after the last input row.
            let y = i + 1 - RADIUS_U32;
            if y < h {
                dst[(y as usize) * w + x] = out_1 + out_3 + out_5;
            }
        }

        // Push right_val into the ring at slot `right % RING_SIZE`.
        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}

/// Threads-per-block for the blur kernel. Must match the launch dim
/// chosen on the host. 96 = 3 warps of 32, the same value the CUDA
/// reference uses (`BLOCK_WIDTH = 3 * 32`).
pub const BLOCK_WIDTH: u32 = 96;
const BLOCK_WIDTH_USIZE: usize = 96;
const BLOCK_TIMES_RING_USIZE: usize = BLOCK_WIDTH_USIZE * RING_SIZE_USIZE;

/// One thread = one row. Walks `x` from `-N + 1` to `width - 1` and
/// emits one output per non-negative `x`.
///
/// `src` and `dst` are single-plane f32 of length `width × height`,
/// row-major. The H-pass companion to [`blur_pass_kernel`] — same
/// recursive IIR Gaussian coefficients (Charalampidis 2016), same
/// boundary handling (state seeded to zero, zero-padding outside the
/// row), just walking horizontally rather than vertically.
///
/// Together with `blur_pass_kernel` this produces a separable 2D
/// Gaussian: `h_pass(v_pass(src))` is bit-equivalent (up to summation
/// order, which doesn't apply since the order of v then h is the same
/// as v then transpose then v under separability) to the current
/// `v_pass + transpose + v_pass` three-step. Eliminates the explicit
/// `transpose_kernel` launch and the `t_scratch` plane family per
/// scale.
///
/// Launch geometry: cubes `(ceil(height / BLOCK_WIDTH), 1, 1)`, cube
/// dim `(BLOCK_WIDTH, 1, 1)`. Each thread owns one row `y = ABSOLUTE_POS`.
///
/// **Output orientation: untransposed.** Caller no longer needs to
/// pre-transpose the input; the fully-blurred output is in the same
/// `(y, x)` row-major layout as `src`. Downstream consumers that expect
/// the transposed layout (today's `error_maps_kernel`) need to be
/// updated to read untransposed — see the fused-features kernel work
/// in `docs/SSIM2_FIX_ASSESSMENT.md` for the planned consumer.
#[cube(launch_unchecked)]
pub fn blur_h_pass_kernel(src: &Array<f32>, dst: &mut Array<f32>, width: u32, height: u32) {
    // y = absolute row index across the launch grid.
    let y = ABSOLUTE_POS;
    if y >= height as usize {
        terminate!();
    }

    // Per-thread ring buffer — same shape as the v-pass: BLOCK_WIDTH
    // threads × RING_SIZE entries each. Each entry stores one column's
    // pixel from this row's history window.
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

    let w = width;
    let w_us = width as usize;

    // x ranges over `(-N + 1) ..= (width - 1)`. Phase offset by `N - 1`
    // so the loop variable is non-negative; the mathematical x is
    // `i - (N - 1)`.
    let span = w + RADIUS_U32 - 1;
    let row_base = y * w_us;
    let mut i: u32 = 0;
    while i < span {
        let right = i; // = x + N - 1
        let left_present = i >= TWO_N; // x - N - 1 >= 0  <=>  i >= 2N
        let x_emit = i + 1 >= RADIUS_U32; // x >= 0  <=>  i + 1 >= N

        let right_val = if right < w {
            src[row_base + (right as usize)]
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

        // Three IIR taps; same coefficients as v-pass.
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

        if x_emit {
            let x = i + 1 - RADIUS_U32;
            if x < w {
                dst[row_base + (x as usize)] = out_1 + out_3 + out_5;
            }
        }

        let slot = right % RING_SIZE;
        ring[ring_base + (slot as usize)] = right_val;

        i += 1;
    }
}

/// Batched recursive Gaussian column walk.
///
/// `src` and `dst` hold `batch_size` planes of `width × height` packed
/// contiguously at `plane_stride` floats apart. Launch geometry:
/// `cube_count = (ceil(width / BLOCK_WIDTH), batch_size, 1)`,
/// `cube_dim = (BLOCK_WIDTH, 1, 1)`. `CUBE_POS_Y` selects which image
/// in the batch this cube walks; the IIR's height clamp stays within
/// each plane's local height — see G4.8 in CUBECL_GOTCHAS.md for why
/// we can't just reuse the unbatched kernel on a tall stacked buffer
/// (the column walk would bleed across image boundaries).
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

// ───────────────── Ssim2Blur::Fir — separable D=5 FIR ─────────────────
//
// **Gated behind the `fir` Cargo feature** — off by default, the IIR
// path above is the only blur surface.
//
// One thread per output pixel; 5 reads along the row; symmetric taps
// folded; ZERO padding at the borders (libjxl-IIR convention).
//
// Used in pairs: H-pass → transpose → H-pass(transposed) yields a 2D
// blur in transposed orientation (the rest of the pipeline expects
// transposed). See pipeline.rs::blur_plane_two_pass for orchestration.

/// Horizontal 5-tap FIR pass with ZERO padding (libjxl-IIR convention).
///
/// One thread per output pixel. Reads 5 samples along the row (zero
/// outside the frame) and accumulates a normalized Gaussian
/// convolution. Output is row-major, same orientation as input.
///
/// Launch geometry: `cube_count_1d(width * height)`,
/// `cube_dim_1d(FIR_BLOCK_WIDTH)`. The kernel's per-thread `idx`
/// decomposes into `(y, x)` via `y = idx / width`, `x = idx % width`.
#[cfg(feature = "fir")]
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

/// Batched horizontal 5-tap FIR. `plane_stride = width * height`;
/// `batch_size` planes packed contiguously in `src` / `dst`. Launch
/// geometry: `cube_count = (ceil(plane_stride / FIR_BLOCK_WIDTH),
/// batch_size, 1)` — same per-image work, batched by `CUBE_POS_Y`.
#[cfg(feature = "fir")]
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

    // height is in the signature for symmetry with the IIR's batched
    // kernel — the FIR doesn't need it because it walks one output per
    // thread and the boundary clamp uses width only.
    let _ = height;
}
