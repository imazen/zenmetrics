//! Pyramid decomposition for still-image cvvdp (Weber contrast on
//! non-baseband levels, gaussian residual on the baseband).
//!
//! Per DKL channel, `weber_contrast_pyr_dec_scalar` produces
//! `n_levels` bands:
//!
//! - `band[k]` for `k < n_levels - 1` = Weber contrast of the
//!   `(gauss[k] - expand(gauss[k+1]))` layer relative to the
//!   per-pixel achromatic `L_bkg` plane (`expand(gauss_l_bkg[k+1])`,
//!   floored at 0.01, clipped to ±1000).
//! - `band[n_levels - 1]` = the coarsest gaussian (residual); the
//!   host bypasses Weber contrast for the baseband and feeds it
//!   directly into pooling.
//!
//! cvvdp v0.5.4 uses a 5-tap separable Gaussian (the "Burt-Adelson
//! kernel" with `a = 0.4`):
//!
//! ```text
//! K[a] = [0.25 - a/2, 0.25, a, 0.25, 0.25 - a/2]
//! ```
//!
//! At `a = 0.4` that's `[0.05, 0.25, 0.40, 0.25, 0.05]`. Applied
//! separably (vertical then horizontal) at stride 2 in each direction
//! for the reduce step; zero-interleaved at stride 2 + filtered for
//! the expand step (with the ×4 reconstruction gain split between the
//! two passes).
//!
//! Edge handling: **symmetric padding**. cvvdp's `gausspyr_reduce`
//! uses `F.conv2d` with `padding=2` (zero-pad) and then patches the
//! first/last rows/cols with explicit reflection terms. For the
//! scalar reference here we collapse those patches into a single
//! reflect-index helper; numerical equivalence is verified against
//! pycvvdp goldens in `tests/pyramid_scalar.rs`.
//!
//! Kernels in this module (all live, all parity-tested in
//! `tests/pyramid_kernel.rs`):
//!
//! - `downscale_kernel` — 5-tap separable Gaussian + 2× decimation
//!   (gauss-pyramid reduce step).
//! - `upscale_v_kernel` + `upscale_h_kernel` — separable vertical
//!   then horizontal 2× zero-insertion + 5-tap Gaussian (gauss-pyramid
//!   expand step), with reconstruction gain ×4 split as ×2 per pass.
//! - `subtract_kernel` — `band = fine - upscaled_coarse`. Still
//!   used by `compute_dkl_laplacian_pyramid` (vanilla Laplacian)
//!   and `compute_dkl_csf_weighted_bands` via the shared
//!   `_dispatch_laplacian_pyramid_gpu` helper; the Weber path
//!   went through the fused subtract+weber kernel below.
//! - `weber_contrast_compute_kernel` — per-pixel `layer / L_bkg`
//!   with cvvdp's clamps + `log10(L_bkg)` emission for the CSF
//!   lookup. Spec reference for the per-pixel math; production
//!   uses the 3-channel variants below.
//! - `weber_contrast_compute_3ch_kernel` — fused 3-channel weber
//!   compute with shared `log10(L_bkg)` write. One launch per
//!   non-baseband level instead of three.
//! - `subtract_weber_3ch_kernel` — further fuses the subtract step
//!   into the weber compute. Reads `fine[c]` + `upscaled[c]` for
//!   3 channels and writes `band[c] = clamp((fine[c] - upscaled[c])
//!   / L_bkg)` + shared `log_l_bkg`. Production weber kernel
//!   (replaces 3× `subtract_kernel` + 1× weber per level).

// Tick 514: silence missing_docs warnings on items emitted by the
// #[cube(launch)] macro (see kernels/color.rs for full rationale).
#![allow(missing_docs)]

use cubecl::prelude::*;


// Phase 8c.1-C: scalar items (KERNEL_A + GAUSS5 constants, the Band
// + WeberPyramid structs, band_frequencies + gausspyr_reduce_scalar +
// gausspyr_expand_scalar + laplacian_pyramid_dec_scalar +
// weber_contrast_pyr_dec_scalar host helpers) live in
// `cvvdp::kernels::pyramid` so the CPU crate owns the canonical scalar
// implementation. Re-export the surface so existing
// `cvvdp_gpu::kernels::pyramid::*` callsites resolve unchanged.
//
// The cube-macro `#[cube(launch)]` kernels below
// (`downscale_kernel`, `downscale_strip_kernel`,
// `downscale_tiled_kernel`, `upscale_v_kernel`,
// `upscale_v_strip_kernel`, `upscale_h_kernel`,
// `upscale_h_strip_kernel`, `subtract_kernel`,
// `weber_contrast_compute_kernel`,
// `weber_contrast_compute_3ch_kernel`, `subtract_weber_3ch_kernel`,
// `subtract_weber_3ch_strip_kernel`, `baseband_divide_3ch_kernel`)
// reference inline `f32::new(...)` constants for the Burt-Adelson
// taps (= -0.05, 0.25, 0.5, etc.) rather than the moved `GAUSS5`
// array. The `DOWNSCALE_TILED_*` workgroup-tile constants stay
// declared in this file (they parameterise the cube kernels'
// SharedMemory sizes and CubeDim::new_2d launch shape). No
// cube-macro name-resolution interaction with the moved items.
pub use cvvdp::kernels::pyramid::{
    Band, GAUSS5, KERNEL_A, WeberPyramid, band_frequencies, gausspyr_expand_scalar,
    gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar, weber_contrast_pyr_dec_scalar,
};


/// 2× downscale with the cvvdp 5-tap Gaussian. Per-output-pixel
/// thread; each thread reads 25 source pixels (5 × 5 reflected
/// taps) and emits one f32. Equivalent to two-pass separable conv
/// with symmetric reflection.
///
/// Bug-compatible with pycvvdp's `gausspyr_reduce` (lpyr_dec.py:186):
/// upstream uses zero-pad + parity-aware boundary patches. The
/// horizontal-pass right-column patch checks `x.shape[-2]` (INPUT
/// ROW parity) where the comments say "odd number of columns" —
/// the check is using rows. For mismatched-parity inputs (sw and
/// sh have different parity) the right column gets the wrong patch
/// from pycvvdp's perspective, but since we MATCH that pycvvdp
/// behavior our goldens align. Pure symmetric reflection (what
/// this kernel computes interior) matches pycvvdp's boundary
/// behavior for ALL same-parity inputs (256², 4000×3000 etc.);
/// for mixed-parity inputs we apply a delta correction at the
/// right column to switch from "reflect" to "pycvvdp's bug
/// branch". See `docs/CHROMA_DRIFT_INVESTIGATION.md` tick 206.
#[cube(launch)]
pub fn downscale_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let dy = idx / dw;
    let dx = idx - dy * dw;

    let cy = 2 * (dy as i32);
    let cx = 2 * (dx as i32);
    let sw = src_w as usize;
    let sh = src_h as usize;
    let sh_i = src_h as i32;
    let sw_i = src_w as i32;

    let k0 = f32::new(0.05);
    let k1 = f32::new(0.25);
    let k2 = f32::new(0.40);
    let k3 = f32::new(0.25);
    let k4 = f32::new(0.05);

    // Symmetric reflection at boundaries [0, n). For kernel-radius-2
    // accesses near the edge: one fold covers all cases (the most
    // extreme is cy-2 = -2 → 1, never re-reflects). Same for the
    // upper edge: cy+2 = sh+1 → sh-2.
    let r0_i = if cy - 2 < 0 { -(cy - 2) - 1 } else { cy - 2 };
    let r0 = r0_i as usize;
    let r1_i = if cy - 1 < 0 { -(cy - 1) - 1 } else { cy - 1 };
    let r1 = r1_i as usize;
    let r2 = cy as usize;
    let r3_i = if cy + 1 >= sh_i {
        2 * sh_i - (cy + 1) - 1
    } else {
        cy + 1
    };
    let r3 = r3_i as usize;
    let r4_i = if cy + 2 >= sh_i {
        2 * sh_i - (cy + 2) - 1
    } else {
        cy + 2
    };
    let r4 = r4_i as usize;

    let sx0_i = if cx - 2 < 0 { -(cx - 2) - 1 } else { cx - 2 };
    let sx0 = sx0_i as usize;
    let sx1_i = if cx - 1 < 0 { -(cx - 1) - 1 } else { cx - 1 };
    let sx1 = sx1_i as usize;
    let sx2 = cx as usize;
    let sx3_i = if cx + 1 >= sw_i {
        2 * sw_i - (cx + 1) - 1
    } else {
        cx + 1
    };
    let sx3 = sx3_i as usize;
    let sx4_i = if cx + 2 >= sw_i {
        2 * sw_i - (cx + 2) - 1
    } else {
        cx + 2
    };
    let sx4 = sx4_i as usize;

    let col0 = k0 * src[r0 * sw + sx0]
        + k1 * src[r1 * sw + sx0]
        + k2 * src[r2 * sw + sx0]
        + k3 * src[r3 * sw + sx0]
        + k4 * src[r4 * sw + sx0];
    let col1 = k0 * src[r0 * sw + sx1]
        + k1 * src[r1 * sw + sx1]
        + k2 * src[r2 * sw + sx1]
        + k3 * src[r3 * sw + sx1]
        + k4 * src[r4 * sw + sx1];
    let col2 = k0 * src[r0 * sw + sx2]
        + k1 * src[r1 * sw + sx2]
        + k2 * src[r2 * sw + sx2]
        + k3 * src[r3 * sw + sx2]
        + k4 * src[r4 * sw + sx2];
    let col3 = k0 * src[r0 * sw + sx3]
        + k1 * src[r1 * sw + sx3]
        + k2 * src[r2 * sw + sx3]
        + k3 * src[r3 * sw + sx3]
        + k4 * src[r4 * sw + sx3];
    let col4 = k0 * src[r0 * sw + sx4]
        + k1 * src[r1 * sw + sx4]
        + k2 * src[r2 * sw + sx4]
        + k3 * src[r3 * sw + sx4]
        + k4 * src[r4 * sw + sx4];

    let mut total_v = k0 * col0 + k1 * col1 + k2 * col2 + k3 * col3 + k4 * col4;

    // Tick 206 bug-compat delta. At the right column (dx = dw-1),
    // pycvvdp picks the horizontal patch branch by INPUT ROW
    // parity (sh) — its comment says "columns" but the code uses
    // rows. When sw and sh have the same parity the patch matches
    // what reflect computes; when they differ we add a delta to
    // switch from reflect to pycvvdp's bug branch. Closes the
    // 73×91 odd-dim residual.
    if dx == dw - 1 && sw >= 2 {
        // vscratch values at the right two columns. Use the same
        // reflect-based vertical conv (matches pycvvdp regardless
        // of sh parity for the vertical pass; see analysis in
        // docs/CHROMA_DRIFT_INVESTIGATION.md).
        let vs_last = k0 * src[r0 * sw + sw - 1]
            + k1 * src[r1 * sw + sw - 1]
            + k2 * src[r2 * sw + sw - 1]
            + k3 * src[r3 * sw + sw - 1]
            + k4 * src[r4 * sw + sw - 1];
        let vs_last2 = k0 * src[r0 * sw + sw - 2]
            + k1 * src[r1 * sw + sw - 2]
            + k2 * src[r2 * sw + sw - 2]
            + k3 * src[r3 * sw + sw - 2]
            + k4 * src[r4 * sw + sw - 2];

        let sw_odd = sw % 2 == 1;
        let sh_odd = sh % 2 == 1;
        if sw_odd && !sh_odd {
            // Reflect gave the "odd-W" patch result; pycvvdp picks
            // even-W (using sh's parity). Delta = pycvvdp_even -
            // reflect_odd = -0.05*vs_last2 - 0.20*vs_last.
            total_v += f32::new(-0.05) * vs_last2 + f32::new(-0.20) * vs_last;
        } else if !sw_odd && sh_odd {
            // Reflect gave "even-W"; pycvvdp picks odd-W.
            // Delta = +0.05*vs_last2 + 0.20*vs_last.
            total_v += f32::new(0.05) * vs_last2 + f32::new(0.20) * vs_last;
        }
    }

    dst[idx] = total_v;
}

/// Strip-aware sibling of [`downscale_kernel`] (Mode E Phase 3,
/// task #79 follow-on). Functionally equivalent to [`downscale_kernel`]
/// on the BODY rows of a strip; reflects against the **logical**
/// image dimensions (not the strip buffer dims) and applies the
/// pycvvdp bug-compat parity delta using `logical_src_w % 2` /
/// `logical_src_h % 2` so the JOD remains bit-exact against the
/// full-image path.
///
/// **Buffer convention.** The destination strip buffer is sized
/// exactly to the body region (no dst halo); buffer-local dst row
/// 0 corresponds to logical dst row `body_offset_y`. The source
/// strip buffer includes halo rows; buffer-local src row 0
/// corresponds to logical src row `src_strip_offset`. Caller is
/// responsible for populating the source strip so every logical
/// row needed by reflection over `[2·body_offset_y − 2,
/// 2·(body_offset_y + dst_h − 1) + 2]` is present.
///
/// **Reflection.** All reflection happens in logical-src
/// coordinates against `logical_src_h`; the kernel then translates
/// the post-reflect logical row to buffer-local by subtracting
/// `src_strip_offset`. X-axis reflection is unchanged (strips
/// partition only Y; `src_w` IS `logical_src_w`).
///
/// **Parity delta.** The tick-206 bug-compat delta at the right
/// column uses `src_w % 2` (= logical_src_w % 2) and
/// `logical_src_h % 2`. The legacy kernel's `sh` parameter refers
/// to the buffer height, which on a strip is NOT the logical
/// image height. Using the logical dim here keeps the delta
/// firing identically to the full-image path — without this, an
/// odd-height logical image striped into even-height buffers
/// would silently drop the parity correction.
///
/// **Legacy equivalence.** Calling this with `body_offset_y = 0`,
/// `src_strip_offset = 0`, `logical_src_h = src_h`, and
/// `logical_dst_h = dst_h` produces output bit-identical to
/// [`downscale_kernel`] on the same input.
#[cube(launch)]
pub fn downscale_strip_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    body_offset_y: u32,
    src_strip_offset: u32,
    logical_src_h: u32,
    logical_dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let dy_local = idx / dw;
    let dx = idx - dy_local * dw;

    // Logical dst row this thread emits — used to compute the
    // logical src center row. The buffer-local dst row index
    // `dy_local` writes to `dst[idx]` directly.
    let dy_logical = (dy_local as u32) + body_offset_y;

    let cy = 2 * (dy_logical as i32);
    let cx = 2 * (dx as i32);
    let sw = src_w as usize;
    let sh_buf = src_h as usize;
    let lsh_i = logical_src_h as i32;
    let sw_i = src_w as i32;
    let src_off = src_strip_offset as i32;

    let k0 = f32::new(0.05);
    let k1 = f32::new(0.25);
    let k2 = f32::new(0.40);
    let k3 = f32::new(0.25);
    let k4 = f32::new(0.05);

    // Reflect against LOGICAL src dims (not buffer). One fold is
    // enough for kernel-radius-2 accesses: `cy-2 = -2 → 1`, and
    // `cy+2 = lsh+1 → lsh-2`. Translate back to buffer-local by
    // subtracting `src_strip_offset`.
    let r0_l = if cy - 2 < 0 { -(cy - 2) - 1 } else { cy - 2 };
    let r0 = (r0_l - src_off) as usize;
    let r1_l = if cy - 1 < 0 { -(cy - 1) - 1 } else { cy - 1 };
    let r1 = (r1_l - src_off) as usize;
    let r2_l = cy;
    let r2 = (r2_l - src_off) as usize;
    let r3_l = if cy + 1 >= lsh_i {
        2 * lsh_i - (cy + 1) - 1
    } else {
        cy + 1
    };
    let r3 = (r3_l - src_off) as usize;
    let r4_l = if cy + 2 >= lsh_i {
        2 * lsh_i - (cy + 2) - 1
    } else {
        cy + 2
    };
    let r4 = (r4_l - src_off) as usize;

    // X-axis reflection is unchanged — strips partition only Y, so
    // `src_w` equals `logical_src_w` and column indexing is direct.
    let sx0_i = if cx - 2 < 0 { -(cx - 2) - 1 } else { cx - 2 };
    let sx0 = sx0_i as usize;
    let sx1_i = if cx - 1 < 0 { -(cx - 1) - 1 } else { cx - 1 };
    let sx1 = sx1_i as usize;
    let sx2 = cx as usize;
    let sx3_i = if cx + 1 >= sw_i {
        2 * sw_i - (cx + 1) - 1
    } else {
        cx + 1
    };
    let sx3 = sx3_i as usize;
    let sx4_i = if cx + 2 >= sw_i {
        2 * sw_i - (cx + 2) - 1
    } else {
        cx + 2
    };
    let sx4 = sx4_i as usize;

    let col0 = k0 * src[r0 * sw + sx0]
        + k1 * src[r1 * sw + sx0]
        + k2 * src[r2 * sw + sx0]
        + k3 * src[r3 * sw + sx0]
        + k4 * src[r4 * sw + sx0];
    let col1 = k0 * src[r0 * sw + sx1]
        + k1 * src[r1 * sw + sx1]
        + k2 * src[r2 * sw + sx1]
        + k3 * src[r3 * sw + sx1]
        + k4 * src[r4 * sw + sx1];
    let col2 = k0 * src[r0 * sw + sx2]
        + k1 * src[r1 * sw + sx2]
        + k2 * src[r2 * sw + sx2]
        + k3 * src[r3 * sw + sx2]
        + k4 * src[r4 * sw + sx2];
    let col3 = k0 * src[r0 * sw + sx3]
        + k1 * src[r1 * sw + sx3]
        + k2 * src[r2 * sw + sx3]
        + k3 * src[r3 * sw + sx3]
        + k4 * src[r4 * sw + sx3];
    let col4 = k0 * src[r0 * sw + sx4]
        + k1 * src[r1 * sw + sx4]
        + k2 * src[r2 * sw + sx4]
        + k3 * src[r3 * sw + sx4]
        + k4 * src[r4 * sw + sx4];

    let mut total_v = k0 * col0 + k1 * col1 + k2 * col2 + k3 * col3 + k4 * col4;

    // Tick-206 bug-compat delta — uses LOGICAL dims. The legacy
    // kernel reads `sh % 2`; on a strip with smaller buffer height
    // that parity is wrong. `logical_src_h % 2` keeps the delta
    // firing identically to the full-image path.
    if dx == dw - 1 && sw >= 2 {
        let vs_last = k0 * src[r0 * sw + sw - 1]
            + k1 * src[r1 * sw + sw - 1]
            + k2 * src[r2 * sw + sw - 1]
            + k3 * src[r3 * sw + sw - 1]
            + k4 * src[r4 * sw + sw - 1];
        let vs_last2 = k0 * src[r0 * sw + sw - 2]
            + k1 * src[r1 * sw + sw - 2]
            + k2 * src[r2 * sw + sw - 2]
            + k3 * src[r3 * sw + sw - 2]
            + k4 * src[r4 * sw + sw - 2];

        let sw_odd = sw % 2 == 1;
        let lsh_odd = (logical_src_h as usize) % 2 == 1;
        if sw_odd && !lsh_odd {
            total_v += f32::new(-0.05) * vs_last2 + f32::new(-0.20) * vs_last;
        } else if !sw_odd && lsh_odd {
            total_v += f32::new(0.05) * vs_last2 + f32::new(0.20) * vs_last;
        }
    }

    // Silence unused warnings — `sh_buf` and `logical_dst_h` are
    // part of the strip-aware API contract for callers (and
    // forward symmetry with future strip kernels that may need
    // them) but the kernel body does not read them.
    let _ = sh_buf;
    let _ = logical_dst_h;

    dst[idx] = total_v;
}

/// Workgroup size (output pixels per side) for the LDS-tiled downscale.
pub const DOWNSCALE_TILED_BLOCK_DIM: u32 = 16;
const DOWNSCALE_TILED_BLOCK_DIM_USIZE: usize = 16;
/// Input tile width = `2 · BLOCK + 4` (5-tap stencil halo on each side).
const DOWNSCALE_TILED_TILE_DIM_USIZE: usize = 2 * DOWNSCALE_TILED_BLOCK_DIM_USIZE + 4;
const DOWNSCALE_TILED_TILE_DIM_U32: u32 = 2 * DOWNSCALE_TILED_BLOCK_DIM + 4;
const DOWNSCALE_TILED_TILE_LEN_USIZE: usize =
    DOWNSCALE_TILED_TILE_DIM_USIZE * DOWNSCALE_TILED_TILE_DIM_USIZE;
const DOWNSCALE_TILED_TILE_LEN_U32: u32 =
    DOWNSCALE_TILED_TILE_DIM_U32 * DOWNSCALE_TILED_TILE_DIM_U32;
const DOWNSCALE_TILED_BLOCK_LIN_U32: u32 =
    DOWNSCALE_TILED_BLOCK_DIM * DOWNSCALE_TILED_BLOCK_DIM;
/// Loads per thread to cover the full tile (256 threads × 6 = 1536 ≥ 1296).
const DOWNSCALE_TILED_LOAD_ITERS: u32 =
    DOWNSCALE_TILED_TILE_LEN_U32.div_ceil(DOWNSCALE_TILED_BLOCK_LIN_U32);

/// LDS-tiled 2× downscale (T1.B). Functionally equivalent to
/// [`downscale_kernel`] including the tick-206 bug-compat delta;
/// trades 25-load-per-thread global access for a shared-memory
/// tile load followed by 25 LDS reads.
///
/// **Workgroup**: 16×16 = 256 threads. Each thread emits one output
/// pixel at `(dx, dy) = (CUBE_POS_X·16 + UNIT_POS_X, CUBE_POS_Y·16 +
/// UNIT_POS_Y)`. A 36×36 input tile (= 2·16 + 4 halo per side, 5.2 KB
/// per workgroup) is loaded cooperatively into shared memory with
/// symmetric reflection at borders, then each output thread runs a
/// 5×5 unrolled stencil from LDS.
///
/// **Launch**:
///
/// ```text
/// cube_dim   = CubeDim::new_2d(BLOCK_DIM, BLOCK_DIM)
/// cube_count = (dst_w.div_ceil(BLOCK_DIM), dst_h.div_ceil(BLOCK_DIM), 1)
/// ```
///
/// Boundary handling matches [`downscale_kernel`] exactly: reflect
/// during LDS load (`-i-1` low side, `2·n - i - 1` high side); at
/// `dx == dst_w - 1` apply the parity-mismatched delta correction
/// for pycvvdp's `gausspyr_reduce` bug-compat.
#[cube(launch)]
pub fn downscale_tiled_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) {
    let tx = UNIT_POS_X;
    let ty = UNIT_POS_Y;
    let dx_base = CUBE_POS_X * DOWNSCALE_TILED_BLOCK_DIM;
    let dy_base = CUBE_POS_Y * DOWNSCALE_TILED_BLOCK_DIM;

    // First source col/row covered by this workgroup's input tile.
    // Each output `dx` reads `2·dx ± 2`, so the workgroup spans
    // `[2·dx_base - 2, 2·(dx_base + 15) + 2]` (35 unique cols; we
    // allocate 36 for `2·BLOCK + 4` alignment).
    let sx_base_i = 2 * (dx_base as i32) - 2;
    let sy_base_i = 2 * (dy_base as i32) - 2;

    let sw = src_w as usize;
    let sw_i = src_w as i32;
    let sh_i = src_h as i32;

    let mut tile = SharedMemory::<f32>::new(DOWNSCALE_TILED_TILE_LEN_USIZE);

    // Cooperative tile load with symmetric reflection. 256 threads ×
    // 6 iterations = 1536 ≥ 1296 tile pixels.
    let lid = ty * DOWNSCALE_TILED_BLOCK_DIM + tx;
    let mut iter: u32 = 0;
    while iter < DOWNSCALE_TILED_LOAD_ITERS {
        let i = lid + iter * DOWNSCALE_TILED_BLOCK_LIN_U32;
        if i < DOWNSCALE_TILED_TILE_LEN_U32 {
            let tile_y = i / DOWNSCALE_TILED_TILE_DIM_U32;
            let tile_x = i - tile_y * DOWNSCALE_TILED_TILE_DIM_U32;

            let sy_raw = sy_base_i + tile_y as i32;
            let sx_raw = sx_base_i + tile_x as i32;

            let sy_i = if sy_raw < 0 {
                -sy_raw - 1
            } else if sy_raw >= sh_i {
                2 * sh_i - sy_raw - 1
            } else {
                sy_raw
            };
            let sx_i = if sx_raw < 0 {
                -sx_raw - 1
            } else if sx_raw >= sw_i {
                2 * sw_i - sx_raw - 1
            } else {
                sx_raw
            };

            tile[i as usize] = src[(sy_i as usize) * sw + (sx_i as usize)];
        }
        iter += 1;
    }
    sync_cube();

    let dx = dx_base + tx;
    let dy = dy_base + ty;
    if dx >= dst_w || dy >= dst_h {
        terminate!();
    }

    // Tile-local center for this thread's output. The tile starts
    // at sx_base = 2·dx_base - 2 in source coords; the center input
    // for output `dx = dx_base + tx` is `2·(dx_base+tx) = 2·dx_base
    // + 2·tx`, which is tile-col `2·tx + 2` (since the tile offsets
    // by -2).
    let tcx = 2 * tx as usize + 2;
    let tcy = 2 * ty as usize + 2;
    let stride = DOWNSCALE_TILED_TILE_DIM_USIZE;

    let k0 = f32::new(0.05);
    let k1 = f32::new(0.25);
    let k2 = f32::new(0.40);
    let k3 = f32::new(0.25);
    let k4 = f32::new(0.05);

    let r0 = tcy - 2;
    let r1 = tcy - 1;
    let r2 = tcy;
    let r3 = tcy + 1;
    let r4 = tcy + 2;
    let c0 = tcx - 2;
    let c1 = tcx - 1;
    let c2 = tcx;
    let c3 = tcx + 1;
    let c4 = tcx + 2;

    let col0 = k0 * tile[r0 * stride + c0]
        + k1 * tile[r1 * stride + c0]
        + k2 * tile[r2 * stride + c0]
        + k3 * tile[r3 * stride + c0]
        + k4 * tile[r4 * stride + c0];
    let col1 = k0 * tile[r0 * stride + c1]
        + k1 * tile[r1 * stride + c1]
        + k2 * tile[r2 * stride + c1]
        + k3 * tile[r3 * stride + c1]
        + k4 * tile[r4 * stride + c1];
    let col2 = k0 * tile[r0 * stride + c2]
        + k1 * tile[r1 * stride + c2]
        + k2 * tile[r2 * stride + c2]
        + k3 * tile[r3 * stride + c2]
        + k4 * tile[r4 * stride + c2];
    let col3 = k0 * tile[r0 * stride + c3]
        + k1 * tile[r1 * stride + c3]
        + k2 * tile[r2 * stride + c3]
        + k3 * tile[r3 * stride + c3]
        + k4 * tile[r4 * stride + c3];
    let col4 = k0 * tile[r0 * stride + c4]
        + k1 * tile[r1 * stride + c4]
        + k2 * tile[r2 * stride + c4]
        + k3 * tile[r3 * stride + c4]
        + k4 * tile[r4 * stride + c4];

    let mut total_v = k0 * col0 + k1 * col1 + k2 * col2 + k3 * col3 + k4 * col4;

    // Tick-206 bug-compat delta. Same logic as `downscale_kernel`.
    // At dx == dst_w - 1 with mismatched (sw, sh) parity, pycvvdp's
    // gausspyr_reduce uses the wrong patch branch; we reproduce it.
    // The patch needs blur5 of cols sw-1 and sw-2 — these are valid
    // (non-reflected) source columns, and they fall inside the LDS
    // tile when the workgroup contains dx == dst_w - 1 (always true
    // since dx_base ≤ dst_w-1 < dx_base+16 implies sw - 1 ≤
    // 2·dx_base + 32 + bounded margin). Read from LDS.
    if dx == dst_w - 1 && src_w >= 2 {
        let last_tile_col_i = (sw_i - 1) - sx_base_i;
        let last2_tile_col_i = (sw_i - 2) - sx_base_i;
        let tc_last = last_tile_col_i as usize;
        let tc_last2 = last2_tile_col_i as usize;

        let vs_last = k0 * tile[r0 * stride + tc_last]
            + k1 * tile[r1 * stride + tc_last]
            + k2 * tile[r2 * stride + tc_last]
            + k3 * tile[r3 * stride + tc_last]
            + k4 * tile[r4 * stride + tc_last];
        let vs_last2 = k0 * tile[r0 * stride + tc_last2]
            + k1 * tile[r1 * stride + tc_last2]
            + k2 * tile[r2 * stride + tc_last2]
            + k3 * tile[r3 * stride + tc_last2]
            + k4 * tile[r4 * stride + tc_last2];

        let sw_odd = src_w % 2 == 1;
        let sh_odd = src_h % 2 == 1;
        if sw_odd && !sh_odd {
            total_v += f32::new(-0.05) * vs_last2 + f32::new(-0.20) * vs_last;
        } else if !sw_odd && sh_odd {
            total_v += f32::new(0.05) * vs_last2 + f32::new(0.20) * vs_last;
        }
    }

    let dw = dst_w as usize;
    let dy_u = dy as usize;
    let dx_u = dx as usize;
    dst[dy_u * dw + dx_u] = total_v;
}

/// Vertical pass of the cvvdp expand. Produces a `src_w × dst_h`
/// buffer from a `src_w × src_h` input. Each output pixel runs a
/// 5-tap conv of the implicit zero-interleaved column with cvvdp's
/// `interleave_zeros_and_pad` edge-replication scheme:
///
/// - `z = 0`                            → `src[0]` (front edge)
/// - `z = dst_h + 2 + (dst_h & 1)`      → `src[src_h - 1]` (back edge)
/// - `z = 2 + 2k` for `0 ≤ k < src_h`   → `src[k]`
/// - else                                → sparse zero
///
/// Output gain is ×2 here; the horizontal kernel applies the other
/// ×2 for the full ×4 reconstruction gain.
///
/// Validity branch is dodged by mask-multiplying the coefficient:
/// invalid taps contribute 0 to the sum, and the read index falls
/// back to 0 to avoid OOB.
//
// The `0u32.into()` calls in the body bridge between native `u32`
// literals and the cubecl IR type that the `#[cube(launch)]` macro
// expects on each branch of the `if`/`else` chain; clippy flags
// them as `useless_conversion` but removing them breaks the macro.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_v_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    src_h: u32,
    dst_h: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (src_w * dst_h) as usize;
    if idx >= total {
        terminate!();
    }
    let sw = src_w as usize;
    let y = idx / sw;
    let x = idx - y * sw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_v = (dst_h as i32) + 2 + ((dst_h as i32) & 1);
    let sh_i = src_h as i32;
    let zy_base = y as i32;

    let z0 = zy_base;
    let z1 = zy_base + 1;
    let z2 = zy_base + 2;
    let z3 = zy_base + 3;
    let z4 = zy_base + 4;

    let v0 = z0 == 0 || z0 == back_v || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sh_i);
    let v1 = z1 == 0 || z1 == back_v || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sh_i);
    let v2 = z2 == 0 || z2 == back_v || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sh_i);
    let v3 = z3 == 0 || z3 == back_v || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sh_i);
    let v4 = z4 == 0 || z4 == back_v || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sh_i);

    let y0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_v {
        src_h - 1
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_v {
        src_h - 1
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_v {
        src_h - 1
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_v {
        src_h - 1
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let y4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_v {
        src_h - 1
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    dst[idx] = (k0 * m0) * src[y0 as usize * sw + x]
        + (k1 * m1) * src[y1 as usize * sw + x]
        + (k2 * m2) * src[y2 as usize * sw + x]
        + (k3 * m3) * src[y3 as usize * sw + x]
        + (k4 * m4) * src[y4 as usize * sw + x];
}

/// Strip-aware variant of [`upscale_v_kernel`]. Computes the same
/// vertical expand math but writes only the body slice
/// `[body_offset_y, body_offset_y + body_h)` of the logical
/// `src_w × logical_dst_h` output, into a strip buffer sized
/// `src_w × body_h`. The strip height (`body_h`) is implicit in the
/// dispatch grid: each thread covers one body output pixel
/// (`total_threads = src_w * body_h`).
///
/// Edge reflection runs against `logical_src_h` (the full image's
/// source height) and `logical_dst_h` (the full output height), so a
/// strip that touches y=0 or y=logical_dst_h-1 still reflects the
/// same way as the full-image kernel would.
///
/// Setting `body_offset_y = 0`, `src_strip_offset = 0`, and
/// `logical_dst_h = body_h` reduces this kernel to
/// [`upscale_v_kernel`] for the same input.
///
/// **Source-strip semantics (Path-A Phase 1).** `src_strip_offset`
/// translates the post-reflect logical source row to a buffer-local
/// source row, so callers can pass in a strip-local source buffer
/// that holds rows `[src_strip_offset, src_strip_offset + src_buf_h)`
/// of the full logical source. With `src_strip_offset = 0` the
/// source buffer is interpreted as full-image (existing behavior).
/// Mirrors the [`downscale_strip_kernel`] `src_strip_offset` pattern.
/// Caller is responsible for populating the source strip so every
/// logical row needed by reflection over the body rows is present.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_v_strip_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    logical_src_h: u32,
    logical_dst_h: u32,
    body_offset_y: u32,
    body_h: u32,
    src_strip_offset: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (src_w * body_h) as usize;
    if idx >= total {
        terminate!();
    }
    let sw = src_w as usize;
    let local_y = idx / sw;
    let x = idx - local_y * sw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_v = (logical_dst_h as i32) + 2 + ((logical_dst_h as i32) & 1);
    let sh_i = logical_src_h as i32;
    // Map the strip-local output row back to the full-image
    // destination row; this is what the original V-kernel called `y`.
    let zy_base = (local_y as i32) + (body_offset_y as i32);

    let z0 = zy_base;
    let z1 = zy_base + 1;
    let z2 = zy_base + 2;
    let z3 = zy_base + 3;
    let z4 = zy_base + 4;

    let v0 = z0 == 0 || z0 == back_v || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sh_i);
    let v1 = z1 == 0 || z1 == back_v || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sh_i);
    let v2 = z2 == 0 || z2 == back_v || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sh_i);
    let v3 = z3 == 0 || z3 == back_v || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sh_i);
    let v4 = z4 == 0 || z4 == back_v || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sh_i);

    // Reflection above maps logical-dst-row to logical-src-row via
    // `((z - 2) >> 1)`. We then translate the logical source row to
    // buffer-local by subtracting `src_strip_offset`. With
    // `src_strip_offset = 0` (existing callers) the translation is a
    // no-op and the buffer-local index equals the logical row.
    let src_off = src_strip_offset;
    let y0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_v {
        logical_src_h - 1 - src_off
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32 - src_off
    } else {
        0u32.into()
    };
    let y1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_v {
        logical_src_h - 1 - src_off
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32 - src_off
    } else {
        0u32.into()
    };
    let y2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_v {
        logical_src_h - 1 - src_off
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32 - src_off
    } else {
        0u32.into()
    };
    let y3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_v {
        logical_src_h - 1 - src_off
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32 - src_off
    } else {
        0u32.into()
    };
    let y4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_v {
        logical_src_h - 1 - src_off
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32 - src_off
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    dst[idx] = (k0 * m0) * src[y0 as usize * sw + x]
        + (k1 * m1) * src[y1 as usize * sw + x]
        + (k2 * m2) * src[y2 as usize * sw + x]
        + (k3 * m3) * src[y3 as usize * sw + x]
        + (k4 * m4) * src[y4 as usize * sw + x];
}

/// Horizontal pass of the cvvdp expand. Consumes the vertical
/// kernel's output (`src_w × in_h`) and produces the full
/// `dst_w × in_h` result. The other ×2 of the ×4 reconstruction
/// gain lives here.
//
// See `upscale_v_kernel` for the rationale on the `useless_conversion`
// allow — the `0u32.into()` branches are required by the
// `#[cube(launch)]` macro.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_h_kernel(src: &Array<f32>, dst: &mut Array<f32>, src_w: u32, dst_w: u32, in_h: u32) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * in_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let y = idx / dw;
    let x = idx - y * dw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_h = (dst_w as i32) + 2 + ((dst_w as i32) & 1);
    let sw_i = src_w as i32;
    let zx_base = x as i32;

    let z0 = zx_base;
    let z1 = zx_base + 1;
    let z2 = zx_base + 2;
    let z3 = zx_base + 3;
    let z4 = zx_base + 4;

    let v0 = z0 == 0 || z0 == back_h || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sw_i);
    let v1 = z1 == 0 || z1 == back_h || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sw_i);
    let v2 = z2 == 0 || z2 == back_h || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sw_i);
    let v3 = z3 == 0 || z3 == back_h || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sw_i);
    let v4 = z4 == 0 || z4 == back_h || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sw_i);

    let x0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_h {
        src_w - 1
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_h {
        src_w - 1
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_h {
        src_w - 1
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_h {
        src_w - 1
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_h {
        src_w - 1
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    let base = y * sw;
    dst[idx] = (k0 * m0) * src[base + x0 as usize]
        + (k1 * m1) * src[base + x1 as usize]
        + (k2 * m2) * src[base + x2 as usize]
        + (k3 * m3) * src[base + x3 as usize]
        + (k4 * m4) * src[base + x4 as usize];
}

/// Strip-aware variant of [`upscale_h_kernel`]. The H-axis expand
/// has no Y-direction state: each output row is computed
/// independently from the same input row. The strip-aware signature
/// adds `body_offset_y` and `logical_dst_h` purely for API
/// uniformity with the V-axis strip kernel and to make a future
/// strip-walker pipeline self-describing — the H kernel doesn't
/// reflect against Y, so neither param affects the math.
///
/// Dispatch grid covers `src_w × in_h` output pixels just like the
/// non-strip kernel; the H kernel's only Y dependency is the input
/// row index `y = idx / dst_w`, which is identical between strip
/// and full layouts because the V kernel already placed the
/// strip's body rows contiguously at the top of `src`. The
/// `logical_dst_h` and `body_offset_y` arguments document the
/// strip the caller intends to fill but never enter the per-pixel
/// math.
#[cube(launch)]
#[allow(clippy::useless_conversion)]
pub fn upscale_h_strip_kernel(
    src: &Array<f32>,
    dst: &mut Array<f32>,
    src_w: u32,
    dst_w: u32,
    in_h: u32,
    _logical_dst_h: u32,
    _body_offset_y: u32,
) {
    let idx = ABSOLUTE_POS;
    let total = (dst_w * in_h) as usize;
    if idx >= total {
        terminate!();
    }
    let dw = dst_w as usize;
    let sw = src_w as usize;
    let y = idx / dw;
    let x = idx - y * dw;

    let k0 = f32::new(0.1);
    let k1 = f32::new(0.5);
    let k2 = f32::new(0.8);
    let k3 = f32::new(0.5);
    let k4 = f32::new(0.1);

    let back_h = (dst_w as i32) + 2 + ((dst_w as i32) & 1);
    let sw_i = src_w as i32;
    let zx_base = x as i32;

    let z0 = zx_base;
    let z1 = zx_base + 1;
    let z2 = zx_base + 2;
    let z3 = zx_base + 3;
    let z4 = zx_base + 4;

    let v0 = z0 == 0 || z0 == back_h || (z0 >= 2 && (z0 & 1) == 0 && ((z0 - 2) >> 1) < sw_i);
    let v1 = z1 == 0 || z1 == back_h || (z1 >= 2 && (z1 & 1) == 0 && ((z1 - 2) >> 1) < sw_i);
    let v2 = z2 == 0 || z2 == back_h || (z2 >= 2 && (z2 & 1) == 0 && ((z2 - 2) >> 1) < sw_i);
    let v3 = z3 == 0 || z3 == back_h || (z3 >= 2 && (z3 & 1) == 0 && ((z3 - 2) >> 1) < sw_i);
    let v4 = z4 == 0 || z4 == back_h || (z4 >= 2 && (z4 & 1) == 0 && ((z4 - 2) >> 1) < sw_i);

    let x0 = if z0 == 0 {
        0u32.into()
    } else if z0 == back_h {
        src_w - 1
    } else if z0 >= 2 && (z0 & 1) == 0 {
        ((z0 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x1 = if z1 == 0 {
        0u32.into()
    } else if z1 == back_h {
        src_w - 1
    } else if z1 >= 2 && (z1 & 1) == 0 {
        ((z1 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x2 = if z2 == 0 {
        0u32.into()
    } else if z2 == back_h {
        src_w - 1
    } else if z2 >= 2 && (z2 & 1) == 0 {
        ((z2 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x3 = if z3 == 0 {
        0u32.into()
    } else if z3 == back_h {
        src_w - 1
    } else if z3 >= 2 && (z3 & 1) == 0 {
        ((z3 - 2) >> 1) as u32
    } else {
        0u32.into()
    };
    let x4 = if z4 == 0 {
        0u32.into()
    } else if z4 == back_h {
        src_w - 1
    } else if z4 >= 2 && (z4 & 1) == 0 {
        ((z4 - 2) >> 1) as u32
    } else {
        0u32.into()
    };

    let m0 = if v0 { f32::new(1.0) } else { f32::new(0.0) };
    let m1 = if v1 { f32::new(1.0) } else { f32::new(0.0) };
    let m2 = if v2 { f32::new(1.0) } else { f32::new(0.0) };
    let m3 = if v3 { f32::new(1.0) } else { f32::new(0.0) };
    let m4 = if v4 { f32::new(1.0) } else { f32::new(0.0) };

    let base = y * sw;
    dst[idx] = (k0 * m0) * src[base + x0 as usize]
        + (k1 * m1) * src[base + x1 as usize]
        + (k2 * m2) * src[base + x2 as usize]
        + (k3 * m3) * src[base + x3 as usize]
        + (k4 * m4) * src[base + x4 as usize];
}

/// `band = fine - upscaled_coarse`.
#[cube(launch)]
pub fn subtract_kernel(
    fine: &Array<f32>,
    upscaled_coarse: &Array<f32>,
    band: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band[idx] = fine[idx] - upscaled_coarse[idx];
}

/// Per-pixel finishing step of the Weber-contrast pyramid for one
/// non-baseband band of one channel. Mirrors the inner body of
/// `weber_contrast_pyr_dec_scalar`:
///
/// ```text
/// L_bkg     = max(expanded_lbkg, 0.01)
/// contrast  = clamp(layer / L_bkg, max = 1000)
/// log_l_bkg = log10(L_bkg)
/// ```
///
/// Inputs:
/// - `layer`         — `gauss_img[k] - expand(gauss_img[k+1])` for the
///                     channel of interest. Caller produces this via
///                     `upscale_v` + `upscale_h` + `subtract` kernels.
/// - `expanded_lbkg` — `expand(gauss_l_bkg[k+1])` (achromatic L_bkg
///                     plane, expanded to the band's spatial size).
///
/// Outputs:
/// - `contrast` — Weber-contrast band ready for CSF weighting +
///                masking.
/// - `log_l_bkg` — per-pixel log10 background luminance for the CSF
///                lookup. All 3 DKL channels share the same field
///                produced by the achromatic-channel run.
///
/// The baseband case (scalar mean L_bkg) is handled separately by
/// host code; the per-band per-pixel mean reduction wouldn't gain
/// from a per-pixel kernel.
#[cube(launch)]
pub fn weber_contrast_compute_kernel(
    layer: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    let c_raw = layer[idx] / l;
    let c_hi = if c_raw > l_max { l_max } else { c_raw };
    let c_clamped = if c_hi < l_min_neg { l_min_neg } else { c_hi };

    contrast[idx] = c_clamped;
    // log10(x) via the natural log. cubecl 0.10's `f32::log` is
    // base-2 (per butteraugli-gpu's PORT_STATUS notes); `f32::ln` is
    // natural log. log10(x) = ln(x) * log10(e) = ln(x) * (1/ln(10)).
    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

/// 3-channel fused weber-contrast compute. Single launch produces
/// `contrast` for all three DKL channels plus the shared
/// `log_l_bkg`. Replaces three separate `weber_contrast_compute_kernel`
/// launches per non-baseband pyramid level — the per-pixel
/// `l_bkg_fine → log10` math is now computed once instead of three
/// times.
///
/// Inputs:
/// - `layer_a` / `layer_rg` / `layer_vy` — per-channel Laplacian
///   layers (`fine - upscaled_coarse`) for the three DKL channels.
/// - `expanded_lbkg` — per-pixel achromatic L_bkg (upscaled from
///   `gauss[k+1]` to fine resolution).
/// - `n` — pixel count (must match all four input arrays + outputs).
///
/// Outputs:
/// - `contrast_a` / `contrast_rg` / `contrast_vy` — per-channel
///   Weber-contrast bands.
/// - `log_l_bkg` — per-pixel `log10(max(L_bkg, 0.01))` shared by all
///   three channels in the downstream CSF lookup.
#[cube(launch)]
pub fn weber_contrast_compute_3ch_kernel(
    layer_a: &Array<f32>,
    layer_rg: &Array<f32>,
    layer_vy: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast_a: &mut Array<f32>,
    contrast_rg: &mut Array<f32>,
    contrast_vy: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    // Three layers / l with the same upper+lower clamp pattern.
    let c_a_raw = layer_a[idx] / l;
    let c_a_hi = if c_a_raw > l_max { l_max } else { c_a_raw };
    let c_a = if c_a_hi < l_min_neg {
        l_min_neg
    } else {
        c_a_hi
    };
    contrast_a[idx] = c_a;

    let c_rg_raw = layer_rg[idx] / l;
    let c_rg_hi = if c_rg_raw > l_max { l_max } else { c_rg_raw };
    let c_rg = if c_rg_hi < l_min_neg {
        l_min_neg
    } else {
        c_rg_hi
    };
    contrast_rg[idx] = c_rg;

    let c_vy_raw = layer_vy[idx] / l;
    let c_vy_hi = if c_vy_raw > l_max { l_max } else { c_vy_raw };
    let c_vy = if c_vy_hi < l_min_neg {
        l_min_neg
    } else {
        c_vy_hi
    };
    contrast_vy[idx] = c_vy;

    // log10(x) via the natural log — once per pixel rather than
    // three times.
    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

/// Fused subtract + 3-channel Weber-contrast compute.
///
/// Replaces three `subtract_kernel` launches and one
/// `weber_contrast_compute_3ch_kernel` launch per non-baseband
/// pyramid level — and eliminates the per-channel `layer_c`
/// intermediate buffer (the Laplacian-style layer never has to
/// materialize).
///
/// Per-pixel math, per channel `c ∈ {A, RG, VY}`:
///
/// ```text
/// L_bkg       = max(expanded_lbkg, 0.01)
/// contrast[c] = clamp((fine[c] - upscaled_coarse[c]) / L_bkg,
///                     [-1000, 1000])
/// log_l_bkg   = log10(L_bkg)   // shared across all three channels
/// ```
///
/// Inputs:
/// - `fine_a` / `fine_rg` / `fine_vy` — `gauss_ref[k]` planes (the
///   fine-resolution side of the Laplacian).
/// - `upsc_a` / `upsc_rg` / `upsc_vy` — upscaled coarse planes
///   produced by `upscale_v_kernel` + `upscale_h_kernel`.
/// - `expanded_lbkg` — per-pixel achromatic L_bkg from the
///   A-channel upscale.
/// - `n` — pixel count.
///
/// Outputs:
/// - `contrast_a` / `contrast_rg` / `contrast_vy` — per-channel
///   Weber-contrast bands ready for CSF weighting.
/// - `log_l_bkg` — per-pixel `log10(L_bkg)` shared by all three
///   channels.
#[cube(launch)]
pub fn subtract_weber_3ch_kernel(
    fine_a: &Array<f32>,
    fine_rg: &Array<f32>,
    fine_vy: &Array<f32>,
    upsc_a: &Array<f32>,
    upsc_rg: &Array<f32>,
    upsc_vy: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast_a: &mut Array<f32>,
    contrast_rg: &mut Array<f32>,
    contrast_vy: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    let layer_a = fine_a[idx] - upsc_a[idx];
    let c_a_raw = layer_a / l;
    let c_a_hi = if c_a_raw > l_max { l_max } else { c_a_raw };
    let c_a = if c_a_hi < l_min_neg {
        l_min_neg
    } else {
        c_a_hi
    };
    contrast_a[idx] = c_a;

    let layer_rg = fine_rg[idx] - upsc_rg[idx];
    let c_rg_raw = layer_rg / l;
    let c_rg_hi = if c_rg_raw > l_max { l_max } else { c_rg_raw };
    let c_rg = if c_rg_hi < l_min_neg {
        l_min_neg
    } else {
        c_rg_hi
    };
    contrast_rg[idx] = c_rg;

    let layer_vy = fine_vy[idx] - upsc_vy[idx];
    let c_vy_raw = layer_vy / l;
    let c_vy_hi = if c_vy_raw > l_max { l_max } else { c_vy_raw };
    let c_vy = if c_vy_hi < l_min_neg {
        l_min_neg
    } else {
        c_vy_hi
    };
    contrast_vy[idx] = c_vy;

    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);
}

/// Strip-aware sibling of [`subtract_weber_3ch_kernel`] (Mode E
/// Phase 3, task #79 follow-on). Same per-pixel math; processes
/// only the **body rows** of an input strip buffer with halo,
/// writing to the same-layout output buffers.
///
/// **Buffer convention.** All input AND output buffers share the
/// same strip-buffer layout `(strip_h_buf × width)` — they include
/// any halo rows the caller chose. The kernel iterates `body_h`
/// rows starting at buffer-local row `body_offset_y`. Halo rows
/// in the output buffers are left untouched (caller initialises
/// them if it cares about their contents).
///
/// **Why no reflection.** The Weber-contrast subtract step is
/// strictly per-pixel: `band = (fine − upsc) / max(lbkg, 0.01)`
/// with clamps. No neighbouring-pixel access, so the kernel never
/// needs to reflect against logical-image edges. `logical_h` is
/// taken as a parameter purely for API symmetry with the other
/// strip-aware pyramid kernels (and as a documentation contract
/// for callers — the strip is a slice of a `logical_h`-row
/// logical image). It is not used in the kernel math.
///
/// **Legacy equivalence.** Calling this with `body_offset_y = 0`
/// and `body_h = strip_h_buf` (and `logical_h` arbitrary)
/// produces output bit-identical to [`subtract_weber_3ch_kernel`]
/// over the same buffers with `n = strip_h_buf × width`.
///
/// **Source-strip semantics (Path-A Phase 1b).** `src_strip_offset`
/// translates the logical buffer row `body_offset_y + dy_local` to a
/// buffer-local row by subtracting `src_strip_offset`. This applies
/// uniformly to BOTH input reads (`fine_*`, `upsc_*`,
/// `expanded_lbkg`) AND output writes (`contrast_*`, `log_l_bkg`),
/// so callers can pass strip-local sliced handles for any subset of
/// the input/output buffers — the buffer's row 0 represents the
/// logical row `src_strip_offset`. With `src_strip_offset = 0`
/// (legacy / default), the buffer is interpreted as full-image and
/// reads/writes happen at full-image-relative `(body_offset_y +
/// dy_local) * w + dx`. Mirrors the
/// [`upscale_v_strip_kernel`] `src_strip_offset` pattern.
///
/// **Per-buffer offset NOT supported.** All buffers share the same
/// `src_strip_offset`. Callers mixing full-image inputs with
/// strip-local inputs must pre-slice the full-image handles via
/// `offset_start(src_strip_offset * width * 4)` so every buffer
/// presents the same row-0 origin.
#[cube(launch)]
pub fn subtract_weber_3ch_strip_kernel(
    fine_a: &Array<f32>,
    fine_rg: &Array<f32>,
    fine_vy: &Array<f32>,
    upsc_a: &Array<f32>,
    upsc_rg: &Array<f32>,
    upsc_vy: &Array<f32>,
    expanded_lbkg: &Array<f32>,
    contrast_a: &mut Array<f32>,
    contrast_rg: &mut Array<f32>,
    contrast_vy: &mut Array<f32>,
    log_l_bkg: &mut Array<f32>,
    width: u32,
    body_h: u32,
    body_offset_y: u32,
    logical_h: u32,
    src_strip_offset: u32,
) {
    let tid = ABSOLUTE_POS;
    let total = (width * body_h) as usize;
    if tid >= total {
        terminate!();
    }
    let w = width as usize;
    let dy_local = tid / w;
    let dx = tid - dy_local * w;
    // Translate the logical body row (`body_offset_y + dy_local`) to
    // a buffer-local row by subtracting `src_strip_offset`. With
    // `src_strip_offset = 0` this is a no-op and matches the legacy
    // full-image-relative indexing. With `src_strip_offset =
    // body_offset_y` (Phase 1b strip mode) the buffer-local row is
    // `dy_local`, i.e. the buffer's row 0 corresponds to the start
    // of the strip body.
    let buf_y = (body_offset_y as usize) + dy_local - (src_strip_offset as usize);
    let idx = buf_y * w + dx;

    let l_min = f32::new(0.01);
    let l_max = f32::new(1000.0);
    let l_min_neg = f32::new(-1000.0);

    let raw_lbkg = expanded_lbkg[idx];
    let l = if raw_lbkg < l_min { l_min } else { raw_lbkg };

    let layer_a = fine_a[idx] - upsc_a[idx];
    let c_a_raw = layer_a / l;
    let c_a_hi = if c_a_raw > l_max { l_max } else { c_a_raw };
    let c_a = if c_a_hi < l_min_neg {
        l_min_neg
    } else {
        c_a_hi
    };
    contrast_a[idx] = c_a;

    let layer_rg = fine_rg[idx] - upsc_rg[idx];
    let c_rg_raw = layer_rg / l;
    let c_rg_hi = if c_rg_raw > l_max { l_max } else { c_rg_raw };
    let c_rg = if c_rg_hi < l_min_neg {
        l_min_neg
    } else {
        c_rg_hi
    };
    contrast_rg[idx] = c_rg;

    let layer_vy = fine_vy[idx] - upsc_vy[idx];
    let c_vy_raw = layer_vy / l;
    let c_vy_hi = if c_vy_raw > l_max { l_max } else { c_vy_raw };
    let c_vy = if c_vy_hi < l_min_neg {
        l_min_neg
    } else {
        c_vy_hi
    };
    contrast_vy[idx] = c_vy;

    log_l_bkg[idx] = f32::ln(l) * f32::new(core::f32::consts::LOG10_E);

    // `logical_h` is part of the strip-aware API contract for
    // forward symmetry with the pyramid downscale/upscale kernels
    // (which DO reflect against the logical image). The subtract
    // is per-pixel and never needs it, but callers always pass it
    // so this kernel can be swapped in without signature changes
    // when a future variant adds reflection (e.g. for a halo-fill
    // pass).
    let _ = logical_h;
}

// Note (tick 159): I tried adding `upscale_v_3ch_kernel` and
// `upscale_h_3ch_kernel` that read/write 3 channels per thread with
// shared index/mask math. The intent was to halve the upscale
// launch count per level (6 → 2). Result: a ~4% jod regression at
// 12 MP on RTX-class CUDA — the 3ch kernel's per-thread work and
// register footprint reduced warp-level latency hiding more than
// launch overhead was costing us. Kept as a doc breadcrumb so this
// path isn't re-tried without a different angle (e.g. shared-memory
// tiling that actually changes the memory access pattern).

/// Baseband finishing step: scale each of the 3 coarsest Gaussian
/// planes by `inv_l_bkg_mean` (= 1 / mean(max(gauss_a, 0.01))) and
/// emit the 3 baseband bands. Replaces the host-side per-channel
/// read-back → divide → re-upload that the prior baseband path did
/// in `_dispatch_weber_pyramid_gpu`.
///
/// The host still reads back `gauss_a` once to compute the mean
/// (small buffer — ~192 pixels at MAX_LEVELS=9 / 12 MP, single
/// synchronous drain), but the 3 per-channel readbacks +
/// 3 per-channel reuploads become this single GPU launch.
#[cube(launch)]
pub fn baseband_divide_3ch_kernel(
    gauss_a: &Array<f32>,
    gauss_rg: &Array<f32>,
    gauss_vy: &Array<f32>,
    band_a: &mut Array<f32>,
    band_rg: &mut Array<f32>,
    band_vy: &mut Array<f32>,
    inv_l_bkg_mean: f32,
    n: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= n as usize {
        terminate!();
    }
    band_a[idx] = gauss_a[idx] * inv_l_bkg_mean;
    band_rg[idx] = gauss_rg[idx] * inv_l_bkg_mean;
    band_vy[idx] = gauss_vy[idx] * inv_l_bkg_mean;
}

