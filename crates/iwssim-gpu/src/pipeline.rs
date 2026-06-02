//! IW-SSIM pipeline orchestration.
//!
//! Wires the kernels together into the full IW-SSIM algorithm:
//!
//! ```text
//!   gray u8/f32 ──► LP pyramid (5 levels, pyrtools binom5 + reflect1)
//!     ├─► [per scale]  11×11 Gaussian (valid) → µ, σ stats → cs / l
//!     └─► [scales 0..3]  3×3 box stats → g, vv
//!                        imenlarge2(LP[s+1])  → parent band
//!                        cov accumulate (atomic) → C_u
//!                        CPU eigendecomp + invert → C_u_inv, λ_k
//!                        per-pixel quadratic form + infow → iw_map
//!   Σ(cs·iw)/Σ(iw) and mean(cs·l)  →  wmcs[s]
//!   score = Π |wmcs[s]|^β[s]
//! ```
//!
//! Per `Iwssim::new` we pre-allocate every device buffer the pipeline
//! needs at the configured `(width, height)`. Subsequent
//! `compute_gray` / `compute_rgb` calls upload the inputs and re-use
//! the pre-allocated buffers — no per-call allocation on the GPU.
//!
//! Buffer naming follows the IW-SSIM reference:
//! - `lp_ref[s]` / `lp_dis[s]` — Laplacian band at scale `s`.
//! - `cs[s]` — contrast-structure SSIM map at scale `s` (shape
//!   `(h_s − 10, w_s − 10)`).
//! - `iw[s]` — info-content weight map at scale `s` (shape
//!   `(h_s − 2, w_s − 2)`).

use cubecl::prelude::*;

use crate::eig;
use crate::filters;
use crate::kernels::{
    box3, cov, gauss11, imenlarge2, infow, lap_pyramid, reduction, rgb2gray, ssim_combine,
};
use crate::{
    Error, GpuIwssimResult, IwssimConfig, IwssimStrategy, MIN_NATIVE_DIM, NUM_SCALES, Result,
};

/// Reflect-pad index map (`reflect1` boundary convention, matching
/// pyrtools): index `i` outside `[0, n)` is folded back via mirror
/// reflection without repeating the boundary sample. For `n == 1`
/// returns 0.
///
/// Examples for `n = 5`:
/// - `i = -1` → 1, `i = -2` → 2, `i = -3` → 3
/// - `i = 5` → 3, `i = 6` → 2, `i = 7` → 1
///
/// The mapping is the standard "ping-pong" reflection along
/// `period = 2 * (n - 1)`.
#[inline]
pub(crate) fn reflect_index(i: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let n_i = n as isize;
    let period = 2 * (n_i - 1);
    // Modulo-into-period.
    let mut r = i.rem_euclid(period);
    if r >= n_i {
        r = period - r;
    }
    r as usize
}

/// Reflect-pad a tightly-packed `sw × sh` f32 image to `dw × dh`,
/// where `dw ≥ sw` and `dh ≥ sh`. Returns a fresh `Vec<f32>` of
/// length `dw * dh`. The native image lives in the top-left corner
/// `[0..sh, 0..sw]`; the trailing rows/columns are filled with the
/// reflect1 mapping.
pub(crate) fn reflect_pad_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), sw * sh);
    debug_assert!(dw >= sw && dh >= sh);
    let mut out = vec![0.0_f32; dw * dh];
    // For each destination row, compute the source row via
    // reflect_index along the height axis, then fill columns.
    for dy in 0..dh {
        let sy = if dy < sh {
            dy
        } else {
            reflect_index(dy as isize, sh)
        };
        let src_row = &src[sy * sw..sy * sw + sw];
        let dst_row = &mut out[dy * dw..dy * dw + dw];
        // Copy the in-range columns directly, fill the rest via
        // reflect_index along the width axis.
        dst_row[..sw].copy_from_slice(src_row);
        for dx in sw..dw {
            let sx = reflect_index(dx as isize, sw);
            dst_row[dx] = src_row[sx];
        }
    }
    out
}

/// Tile a tightly-packed `sw × sh` f32 image to `dw × dh` by repeating
/// the source content (toroidal wrap). For `dw == sw` and `dh == sh`
/// this is identity. The native image starts at the top-left corner;
/// trailing rows/columns repeat from `(dy mod sh, dx mod sw)`.
///
/// Empirically the best small-image strategy on the validation corpus
/// (`benchmarks/iwssim_smallimg/README.md`): the pyramid sees a
/// periodic signal whose boundary statistics match the interior.
pub(crate) fn tile_pad_f32(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    debug_assert_eq!(src.len(), sw * sh);
    debug_assert!(dw >= sw && dh >= sh);
    let mut out = vec![0.0_f32; dw * dh];
    for dy in 0..dh {
        let sy = dy % sh;
        let src_row = &src[sy * sw..sy * sw + sw];
        let dst_row = &mut out[dy * dw..dy * dw + dw];
        let mut dx = 0;
        // Bulk-copy whole source rows where they fit.
        while dx + sw <= dw {
            dst_row[dx..dx + sw].copy_from_slice(src_row);
            dx += sw;
        }
        // Wrap the trailing partial column block.
        for k in dx..dw {
            dst_row[k] = src_row[k % sw];
        }
    }
    out
}

/// Tile a tightly-packed `sw × sh × 3` u8 RGB image to `dw × dh × 3`.
/// Same boundary convention as [`tile_pad_f32`].
pub(crate) fn tile_pad_rgb_u8(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    debug_assert_eq!(src.len(), sw * sh * 3);
    debug_assert!(dw >= sw && dh >= sh);
    let mut out = vec![0_u8; dw * dh * 3];
    for dy in 0..dh {
        let sy = dy % sh;
        let src_row_start = sy * sw * 3;
        let src_row = &src[src_row_start..src_row_start + sw * 3];
        let dst_row_start = dy * dw * 3;
        let dst_row = &mut out[dst_row_start..dst_row_start + dw * 3];
        let mut dx = 0;
        while dx + sw <= dw {
            dst_row[dx * 3..(dx + sw) * 3].copy_from_slice(src_row);
            dx += sw;
        }
        for k in dx..dw {
            let sx = k % sw;
            dst_row[k * 3] = src_row[sx * 3];
            dst_row[k * 3 + 1] = src_row[sx * 3 + 1];
            dst_row[k * 3 + 2] = src_row[sx * 3 + 2];
        }
    }
    out
}

/// Reflect-pad a tightly-packed `sw × sh × 3` u8 RGB image to
/// `dw × dh × 3`. Same boundary convention as
/// [`reflect_pad_f32`]. Returned buffer length is `dw * dh * 3`.
pub(crate) fn reflect_pad_rgb_u8(
    src: &[u8],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<u8> {
    debug_assert_eq!(src.len(), sw * sh * 3);
    debug_assert!(dw >= sw && dh >= sh);
    let mut out = vec![0_u8; dw * dh * 3];
    for dy in 0..dh {
        let sy = if dy < sh {
            dy
        } else {
            reflect_index(dy as isize, sh)
        };
        let src_row_start = sy * sw * 3;
        let src_row = &src[src_row_start..src_row_start + sw * 3];
        let dst_row_start = dy * dw * 3;
        let dst_row = &mut out[dst_row_start..dst_row_start + dw * 3];
        // In-range columns: byte-copy.
        dst_row[..sw * 3].copy_from_slice(src_row);
        // Out-of-range columns: per-pixel reflect on x.
        for dx in sw..dw {
            let sx = reflect_index(dx as isize, sw);
            let s = sx * 3;
            let d = dx * 3;
            dst_row[d] = src_row[s];
            dst_row[d + 1] = src_row[s + 1];
            dst_row[d + 2] = src_row[s + 2];
        }
    }
    out
}

/// MS-SSIM Gaussian window radius — used to compute `bound1` (the
/// crop applied to `iw_j` so it aligns with `cs_j`).
const BOUND: u32 = 5;
/// blSzX = 3 in the reference → floor((3 − 1) / 2) = 1.
const BLK_HALF: u32 = 1;
/// Cropped offset applied to `iw_j` before pooling against `cs_j`.
const BOUND1: u32 = BOUND - BLK_HALF; // = 4

/// Compute the cs-row range that owns "body" pixels at scale `s` for
/// one strip. `body_lp_top` / `body_lp_bot` are body row bounds at
/// scale 0 in strip-local coordinates. `cs_h_s` is the scale-s cs
/// buffer height (= `strip_h_s − 10`). Returned range is clamped to
/// `[0, cs_h_s]`; an empty range (start ≥ end) means the strip's
/// body doesn't pool any cs rows at this scale.
///
/// Row mapping at scale `s`:
///   LP row `y_s = y_0 / 2^s`        (integer division)
///   cs row = LP row − 5             (11×11 valid blur, 5-row crop)
fn body_cs_range(body_lp_top: i64, body_lp_bot: i64, scale: usize, cs_h_s: u32) -> (u32, u32) {
    let denom: i64 = 1 << (scale as u32);
    // Use round-half-up division: body_top contributes from the first
    // LP row that exceeds the body's start, body_bot from the last
    // LP row strictly within the body. Mismatch by ±1 here is
    // tolerable — the body row count at scale s drifts by < 1 row
    // out of (h_body / 2^s), well within strip-overlap tolerances.
    // Use floor for both ends; that's exact when body_lp_top is a
    // multiple of 2^s (guaranteed in interior strips by construction).
    let lp_top_s: i64 = body_lp_top.div_euclid(denom);
    let lp_bot_s: i64 = body_lp_bot.div_euclid(denom);
    let cs_top: i64 = (lp_top_s - 5).max(0).min(cs_h_s as i64);
    let cs_bot: i64 = (lp_bot_s - 5).max(0).min(cs_h_s as i64);
    (cs_top as u32, cs_bot as u32)
}

/// iw-row variant of [`body_cs_range`]. iw at scale `s` has shape
/// `(h_s − 2, w_s − 2)`; cov_accum / iw_sum read iw rows starting
/// at LP row 1 (box3 crops 1 row from each side). So iw row
/// `y_iw = y_LP − 1`.
fn body_iw_range(body_lp_top: i64, body_lp_bot: i64, scale: usize, iw_h_s: u32) -> (u32, u32) {
    let denom: i64 = 1 << (scale as u32);
    let lp_top_s: i64 = body_lp_top.div_euclid(denom);
    let lp_bot_s: i64 = body_lp_bot.div_euclid(denom);
    let iw_top: i64 = (lp_top_s - 1).max(0).min(iw_h_s as i64);
    let iw_bot: i64 = (lp_bot_s - 1).max(0).min(iw_h_s as i64);
    (iw_top as u32, iw_bot as u32)
}

/// Per-scale device buffer set.
struct Scale {
    /// LP shape.
    h: u32,
    w: u32,
    /// `(h − 10, w − 10)` — SSIM cs/l shape at this scale.
    cs_h: u32,
    cs_w: u32,
    /// `(h − 2, w − 2)` — IW shape at this scale.
    iw_h: u32,
    iw_w: u32,

    // LP coefficients at this scale (both sides).
    lp_ref: cubecl::server::Handle,
    lp_dis: cubecl::server::Handle,

    // Gaussian pyramid (working buffers used during LP build).
    g_ref: cubecl::server::Handle,
    g_dis: cubecl::server::Handle,

    // SSIM 11×11 valid-mode intermediates.
    // gh_* = post horizontal pass (h × (w − 10)); g_* = post vertical
    // pass ((h − 10) × (w − 10)).
    gh_ref: cubecl::server::Handle,
    gh_dis: cubecl::server::Handle,
    gh_ref2: cubecl::server::Handle,
    gh_dis2: cubecl::server::Handle,
    gh_refdis: cubecl::server::Handle,
    mu1: cubecl::server::Handle,
    mu2: cubecl::server::Handle,
    m11: cubecl::server::Handle,
    m22: cubecl::server::Handle,
    m12: cubecl::server::Handle,

    /// cs_map (j < 4) or `cs · l` (j = 4).
    cs: cubecl::server::Handle,

    // IW path (allocated for j < 4; allocated at full LP shape for
    // simplicity — j=4 buffers are unused).
    g_buf: cubecl::server::Handle,
    vv_buf: cubecl::server::Handle,
    parent_band: cubecl::server::Handle,
    iw: cubecl::server::Handle,

    // C_u accumulator (always 10*10 = 100 **f64** — wastes 19 entries when
    // N=9 but cheap and avoids two-variant allocation). Receives the
    // 100-cell output of `cov_finalize_kernel`; was previously written
    // by atomic-add inside the cov_accum kernels. The atomic version
    // panicked silently on `cubecl-cpu` (no `atomic<f32>` lowering in
    // the MLIR backend) and produced score=0; the partials + finalize
    // path works on every backend.
    //
    // Promoted to f64 in tighten-tolerances pass 2026-05-22: the
    // finalize kernel sums 16384 f32 partials per cell, and the f32
    // round-off floor √N · ε ≈ 7.7e-6 per cell propagated to a 2-3e-4
    // drift in the multi-strip parity gate. f64 accumulation in
    // finalize drops that to ~1e-15 per cell; cells stay f64 through
    // host readback for the cleanest precision path.
    cu: cubecl::server::Handle,
    cu_inv_dev: cubecl::server::Handle,
    lambda_dev: cubecl::server::Handle,
}

fn alloc<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// f64-typed allocator for buffers that need higher precision than
/// f32 (e.g., the cov_finalize accumulator). Byte-length is `n * 8`.
fn alloc_f64<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f64::as_bytes(&vec![0.0_f64; n]))
}

impl Scale {
    fn new<R: Runtime>(client: &ComputeClient<R>, h: u32, w: u32) -> Self {
        let n_lp = (h as usize) * (w as usize);
        let cs_h = h - 10;
        let cs_w = w - 10;
        let n_cs = (cs_h as usize) * (cs_w as usize);
        let iw_h = h - 2;
        let iw_w = w - 2;
        let n_iw = (iw_h as usize) * (iw_w as usize);
        // Horizontal-pass intermediates have width (w − 10) and full
        // height h. Vertical-pass output has shape ((h − 10), (w − 10)).
        let n_h = (h as usize) * ((w - 10) as usize);
        Self {
            h,
            w,
            cs_h,
            cs_w,
            iw_h,
            iw_w,
            lp_ref: alloc(client, n_lp),
            lp_dis: alloc(client, n_lp),
            g_ref: alloc(client, n_lp),
            g_dis: alloc(client, n_lp),
            gh_ref: alloc(client, n_h),
            gh_dis: alloc(client, n_h),
            gh_ref2: alloc(client, n_h),
            gh_dis2: alloc(client, n_h),
            gh_refdis: alloc(client, n_h),
            mu1: alloc(client, n_cs),
            mu2: alloc(client, n_cs),
            m11: alloc(client, n_cs),
            m22: alloc(client, n_cs),
            m12: alloc(client, n_cs),
            cs: alloc(client, n_cs),
            g_buf: alloc(client, n_lp),
            vv_buf: alloc(client, n_lp),
            parent_band: alloc(client, n_lp),
            iw: alloc(client, n_iw),
            cu: alloc_f64(client, 100),
            cu_inv_dev: alloc(client, 100),
            lambda_dev: alloc(client, 10),
        }
    }
}

// Cube count / dim used by `cov_accum_*_kernel`. Total threads per
// launch = COV_CUBE_COUNT * COV_CUBE_DIM. Kept in sync with the values
// passed to `launch_unchecked` below — and used to size the cov
// partials buffer (one f32 per (cell, thread)) plus to pass `n_threads`
// into both the accumulator (which strides by it) and the finalizer
// (which knows how many partials to fold per cell).
const COV_CUBE_COUNT: u32 = 64;
const COV_CUBE_DIM: u32 = 256;
const COV_N_THREADS: u32 = COV_CUBE_COUNT * COV_CUBE_DIM;
/// Maximum cells per cov matrix (10×10 with-parent; the no-parent
/// kernel writes 81 cells, ignoring 19; the partials buffer is sized
/// for the max).
const COV_MAX_CELLS: u32 = 100;

/// Default scale-0 halo (rows) per side for strip processing.
///
/// Picked to comfortably cover the 5-level pipeline's worst-case
/// cumulative reach (~180 rows, see `docs/STRIP_PROCESSING.md`). The
/// halo MUST be a multiple of `2^(NUM_SCALES − 1) = 16` so it shrinks
/// to an integer count at every pyramid level — 256 satisfies that
/// and matches the design-doc default.
pub const STRIP_DEFAULT_HALO: u32 = 256;

/// Default body rows per strip at scale 0. Combined with the default
/// halo this gives a maximum strip height of 1536 — the sweet spot
/// per `docs/STRIP_PROCESSING.md` (50% halo overhead, ~460 MB working
/// set on a 24 MP image).
pub const STRIP_DEFAULT_BODY: u32 = 1024;

/// Strip-mode state. Present when the pipeline was constructed via
/// [`Iwssim::new_strip`].
#[derive(Debug, Clone, Copy)]
struct StripState {
    /// Full source image height (rows).
    image_h: u32,
    /// Per-strip body rows at scale 0 (the contribution each strip
    /// owns to the final per-scale reductions).
    h_body: u32,
    /// Halo rows per side at scale 0 (image rows pulled in past the
    /// body region for stencil reach + cross-scale dependency).
    halo: u32,
    /// Maximum strip height at scale 0 = `h_body + 2 * halo`. Used
    /// as the allocation height of every per-scale buffer.
    /// Cached for debug / introspection; the strip loop derives the
    /// per-strip actual_h from the upload range.
    #[allow(dead_code)]
    strip_alloc_h: u32,
}

impl StripState {
    /// Yield `(body_start, body_end, upload_start, upload_end)` for
    /// each strip, all in scale-0 image rows. `upload_*` are clamped
    /// to `[0, image_h]`. The strip's actual GPU height is
    /// `upload_end − upload_start` (may be < `strip_alloc_h` for
    /// boundary strips).
    fn strips(&self) -> Vec<(u32, u32, u32, u32)> {
        let mut out = Vec::new();
        let mut body_start = 0u32;
        while body_start < self.image_h {
            let body_end = (body_start + self.h_body).min(self.image_h);
            let upload_start = body_start.saturating_sub(self.halo);
            let upload_end = (body_end + self.halo).min(self.image_h);
            out.push((body_start, body_end, upload_start, upload_end));
            body_start = body_end;
        }
        out
    }
}

/// Per-strip cached reference state — used only when
/// [`Iwssim::set_reference_stripped`] has populated it. See the
/// `cached_strip_ref` field on [`Iwssim`] for the design rationale.
///
/// The cache holds **one independent device handle per (strip, scale)**
/// for the ref-side Laplacian pyramid, plus the global eigendecomposed
/// C_u_inv + lambda per scale. After this is populated,
/// [`Iwssim::compute_with_reference_stripped`] only needs to: upload
/// the dis-side strip, build the dis-side LP, and run pass-2
/// (ssim_stats + iw_box3_parent + iw_infow + reductions) using the
/// cached ref-side state. Pass 1 (cov accumulation) is fully elided.
struct CachedStripRefState {
    /// Per-strip per-scale LP-band handles for the cached reference.
    /// `lp_ref[strip_idx][scale]` is independent device memory — each
    /// strip's LP pyramid was built into its own handle during
    /// `set_reference_stripped`, so they survive across compute calls.
    lp_ref: Vec<Vec<cubecl::server::Handle>>,
    /// Per-scale inverted C_u matrices (host f32, packed at the
    /// matching `n_dim` for the scale: 10×10 for s ∈ 0..2,
    /// 9×9 for s = 3). Kept on host for diagnostics / reuse in
    /// `set_rgb_reference_stripped`'s pyramid rebuild path; the
    /// per-call hot path uses `cu_inv_dev_per_scale` instead.
    #[allow(dead_code)]
    cu_inv_per_scale: Vec<Vec<f32>>,
    /// Per-scale lambda eigenvalues (host f32, length `n_dim`).
    #[allow(dead_code)]
    lambda_per_scale: Vec<Vec<f32>>,
    /// **Per-scale device handles** holding `cu_inv_per_scale` and
    /// `lambda_per_scale` already uploaded to GPU. Filled ONCE in
    /// `set_reference_stripped` (mirroring the `lp_ref` cache pattern
    /// for the ref-side LP pyramid). Subsequent
    /// `compute_with_reference_stripped` /
    /// `compute_rgb_with_reference_stripped_native` calls clone these
    /// handles into the active `scales[s].cu_inv_dev` /
    /// `scales[s].lambda_dev` slots instead of re-uploading the host
    /// vectors per call.
    ///
    /// Eliminates `2 * n_scales_iw` HtoDs per cached-ref strip call
    /// (typically 8 small HtoDs at 4 IW scales). The data is constant
    /// across all dist-side calls for a given cached reference — it's
    /// derived from the ref-side C_u accumulator alone — so it's the
    /// same "static-given-reference" pattern that already applies to
    /// the per-strip LP handle cache above.
    cu_inv_dev_per_scale: Vec<cubecl::server::Handle>,
    /// Companion to `cu_inv_dev_per_scale` — see that field's doc.
    lambda_dev_per_scale: Vec<cubecl::server::Handle>,
    /// Strip layout snapshot at the time `set_reference_stripped` was
    /// called. Subsequent `compute_with_reference_stripped` calls MUST
    /// see the same strip count; we recompute strips each time from
    /// the live `StripState` and validate it matches.
    strip_count: usize,
}

/// Per-instance allocations + per-call orchestration. Construct once
/// for a given `(width, height)`, reuse across many image pairs.
pub struct Iwssim<R: Runtime> {
    client: ComputeClient<R>,
    /// Native (caller-supplied) image width.
    width: u32,
    /// Native (caller-supplied) image height.
    height: u32,
    /// Padded width fed to the GPU pipeline. Equal to `width` for
    /// stock-size inputs (≥ MIN_NATIVE_DIM on both axes). When the
    /// strategy is non-`Reject` and the native width is below
    /// `MIN_NATIVE_DIM`, `pad_width = MIN_NATIVE_DIM`; the
    /// `compute_*` entry points apply the chosen padding strategy
    /// from `width` to `pad_width` on the host before upload.
    pad_width: u32,
    /// Padded height fed to the GPU pipeline. Same contract as
    /// `pad_width`.
    pad_height: u32,
    /// How sub-176 inputs are padded to fill `(pad_width, pad_height)`.
    /// `Reject` is never observed at this point — construction would
    /// have failed earlier. Stored to dispatch the right host-side
    /// pad helper at every `compute_*` entry.
    strategy: IwssimStrategy,

    /// sRGB u8 staging — re-uploaded per RGB call. None when the
    /// instance has only ever consumed grayscale planes.
    src_u32_a: cubecl::server::Handle,
    src_u32_b: cubecl::server::Handle,
    // T_x.O (2026-05-17): `pack_scratch: Vec<u32>` removed. The
    // upload path now packs u8×3 → u32 directly into the pinned
    // staging buffer reserved per call (`client.reserve_staging`),
    // collapsing two host-side passes (pack to pageable + memcpy to
    // pinned) into one. Mirrors butter T_x.O (10a5b996).
    /// One Scale per pyramid level. `scales.len() == NUM_SCALES` for
    /// validly-sized inputs.
    scales: Vec<Scale>,

    /// 11 reduction slots — 4 × (Σ(cs·iw), Σ(iw)) for j ∈ 0..3, plus
    /// 1 × Σ(cs·l) for j = 4 plus 2 unused. Plain `f32` partials with
    /// the portable two-stage reduction pattern.
    partials: cubecl::server::Handle,
    sums: cubecl::server::Handle,

    /// Per-thread cov accumulator partials. Layout
    /// `[cell × COV_N_THREADS + tid]` — one f32 per (cell, thread).
    /// Shared across scales (each scale's cov_accum + cov_finalize pair
    /// completes before the next scale's pair starts). Replaces the
    /// `Atomic<f32>` accumulator that the cov_accum kernels used to
    /// write directly into `Scale::cu_atomic` — see Scale::cu docstring
    /// for the rationale.
    cov_partials: cubecl::server::Handle,

    /// `set_reference` populates `scales[s].lp_ref` for every scale
    /// and flips this flag. Subsequent `compute_with_reference` calls
    /// skip the ref-side LP pyramid build.
    has_reference: bool,

    /// Strip-mode state. `Some(_)` when the pipeline was built via
    /// [`Iwssim::new_strip`] — `compute_gray_stripped` walks the
    /// strip layout, runs the existing per-strip pipeline, and folds
    /// per-strip partial sums on the host. `None` for the historical
    /// whole-image path.
    strip: Option<StripState>,

    /// Per-strip cached reference state. `Some(_)` after
    /// [`Self::set_reference_stripped`] has populated the cache;
    /// subsequent [`Self::compute_with_reference_stripped`] calls
    /// reuse this state and skip the ref-side LP build + the entire
    /// pass-1 cov accumulation.
    ///
    /// Whole-image cached-ref state lives in `scales[s].lp_ref`
    /// directly (one LP pyramid per scale, no per-strip dimension);
    /// strip mode needs ONE pyramid per strip per scale because the
    /// strip walker mutates `scales[s].lp_ref` each iteration. The
    /// cache holds independent device handles that survive across
    /// strip iterations.
    cached_strip_ref: Option<CachedStripRefState>,
}

/// Slot layout in the partials / sums buffer. Indices match the order
/// in which the host reads sums back.
const SLOT_CSIW_BASE: u32 = 0; // 4 slots: j ∈ 0..3
const SLOT_IW_BASE: u32 = 4; // 4 slots: j ∈ 0..3
const SLOT_CSL: u32 = 8; // 1 slot: j = 4
const NUM_SLOTS: u32 = 9;

impl<R: Runtime> Iwssim<R> {
    /// Allocate the pipeline for the given image dimensions with the
    /// default config (reject inputs below `MIN_NATIVE_DIM` on either
    /// axis). Returns `Err(InvalidImageSize)` if either dimension is
    /// too small for a 5-level pyramid with 11×11 valid-mode SSIM
    /// stats at the coarsest scale.
    ///
    /// Equivalent to `Self::with_config(client, width, height,
    /// IwssimConfig::default())`.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        Self::with_config(client, width, height, IwssimConfig::default())
    }

    /// Allocate the pipeline for the given image dimensions with an
    /// explicit configuration.
    ///
    /// When `cfg.allow_small` is false (the default), behaves exactly
    /// like the historical `new`: inputs below `MIN_NATIVE_DIM` on
    /// either axis return `Err(InvalidImageSize)` and the pipeline
    /// runs at `(width, height)` (zero overhead vs the pre-feature
    /// build).
    ///
    /// When `cfg.allow_small` is true and either axis is below
    /// `MIN_NATIVE_DIM`, the pipeline is constructed at the **padded**
    /// dimensions `(max(width, MIN_NATIVE_DIM), max(height,
    /// MIN_NATIVE_DIM))`. Every subsequent `compute_*` call
    /// reflect-pads the input on the host from the native dimensions
    /// to the padded dimensions before upload. The resulting score is
    /// the IW-SSIM of the **padded** image — it is **not bit-exact
    /// stock IW-SSIM** for the native input, since the pyramid sees
    /// reflected content past the native border. Treat the score as
    /// informational / monotonic for small inputs (still suitable for
    /// codec sweeps where the same pair of distortions is being
    /// compared) but do not compare it against scores from a true
    /// stock-size run on the same content.
    pub fn with_config(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        cfg: IwssimConfig,
    ) -> Result<Self> {
        // Coarsest scale needs at least 11 pixels per axis for a
        // valid-mode 11×11 conv. With 5 pyramid levels that's
        // 11 · 2^(NUM_SCALES − 1) = 11 · 16 = 176 at the input.
        let (pad_width, pad_height) = if width < MIN_NATIVE_DIM || height < MIN_NATIVE_DIM {
            match cfg.strategy {
                IwssimStrategy::Reject => return Err(Error::InvalidImageSize),
                IwssimStrategy::Tile | IwssimStrategy::ReflectPad => {
                    // Pad the short axis up to MIN_NATIVE_DIM. The
                    // long axis (if already ≥ MIN_NATIVE_DIM) stays native.
                    (width.max(MIN_NATIVE_DIM), height.max(MIN_NATIVE_DIM))
                }
            }
        } else {
            (width, height)
        };

        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut h = pad_height;
        let mut w = pad_width;
        for _ in 0..NUM_SCALES {
            dims.push((h, w));
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        let scales: Vec<Scale> = dims
            .iter()
            .map(|&(h, w)| Scale::new(&client, h, w))
            .collect();

        // T4.L (2026-05-16): pack 3 sRGB bytes per pixel into ONE u32
        // (R | G<<8 | B<<16). Length = n_pixels, not n_pixels × 3.
        // Cuts per-call host→device upload from 12 B/pixel to 4 B/pixel.
        let n_pixels_usize = (pad_width * pad_height) as usize;
        let src_u32_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_usize]));
        let src_u32_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_usize]));

        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;
        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; partials_len]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; NUM_SLOTS as usize]));

        // Cov partials: one f32 per (cell, thread). The accumulators
        // overwrite (not accumulate-into), so this buffer never needs
        // zeroing between calls — we still init zero-filled for
        // determinism in case a future change re-introduces partial
        // writes.
        let cov_partials_len = (COV_MAX_CELLS * COV_N_THREADS) as usize;
        let cov_partials =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; cov_partials_len]));

        Ok(Self {
            client,
            width,
            height,
            pad_width,
            pad_height,
            strategy: cfg.strategy,
            src_u32_a,
            src_u32_b,
            scales,
            partials,
            sums,
            cov_partials,
            has_reference: false,
            strip: None,
            cached_strip_ref: None,
        })
    }

    /// Unified [`MemoryMode`](crate::MemoryMode) constructor.
    /// iwssim-gpu is **NOT strip-preferred** — strip mode is ~1.7×
    /// slower than whole-image (the cached-reference strip path is
    /// deferred; see `docs/STRIP_PROCESSING.md`). Auto picks Full
    /// whenever it fits the VRAM cap.
    ///
    /// - `MemoryMode::Auto`: Full if it fits, else Strip (when
    ///   `min(w, h) ≥ MIN_NATIVE_DIM = 176`; small images that can't
    ///   use strip surface [`crate::Error::TooBigForFull`]).
    /// - `MemoryMode::Full`: constructs via [`Self::new`].
    /// - `MemoryMode::Strip { h_body }`: constructs via
    ///   [`Self::new_strip`]. `h_body == None` auto-sizes within the
    ///   cap.
    /// - `MemoryMode::Tile {..}` returns
    ///   [`crate::Error::ModeUnsupported`].
    pub fn new_with_memory_mode(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        use crate::MemoryMode;
        use crate::memory_mode::{ResolvedMode, resolve_auto, vram_cap_bytes};
        match mode {
            MemoryMode::Full => Self::new(client, width, height),
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, cap)
                });
                Self::new_strip(client, width, height, body)
            }
            MemoryMode::Tile { .. } => Err(crate::Error::ModeUnsupported("Tile")),
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                match resolve_auto(width, height, cap)? {
                    ResolvedMode::Full => Self::new(client, width, height),
                    ResolvedMode::Strip { h_body } => {
                        Self::new_strip(client, width, height, h_body)
                    }
                }
            }
        }
    }

    /// Construct a strip-processing pipeline for an `image_w × image_h`
    /// image, with each strip carrying `h_body` body rows + the default
    /// halo per side ([`STRIP_DEFAULT_HALO`]). Per-scale GPU buffers
    /// are sized for a single strip of `h_body + 2 * halo` rows, not
    /// the full image — peak working set drops from `O(image_h)` to
    /// `O(strip_alloc_h)`. See `docs/STRIP_PROCESSING.md` for the
    /// memory analysis.
    ///
    /// Use `image_w` and `image_h` from the input you intend to score;
    /// `h_body` defaults to [`STRIP_DEFAULT_BODY`] (1024 rows). The
    /// `compute_gray_stripped` entry point loops over strips, runs
    /// the existing whole-image pipeline on each strip (so existing
    /// kernels are reused unchanged), and accumulates per-strip
    /// partial sums on the host. The final score is the IW-SSIM of
    /// the full image.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImageSize`] if either image axis is
    /// below [`MIN_NATIVE_DIM`], or if `h_body` is too small (must be
    /// at least 16 rows for the scale-4 strip to fit the 11×11 valid
    /// blur), or if the resulting strip allocation height is below
    /// [`MIN_NATIVE_DIM`]. Reflects the historical contract: strip
    /// mode is only well-defined on stock-size inputs; the small-image
    /// adaptive path stays whole-image.
    ///
    /// **Backwards-compatible:** [`Iwssim::new`] / [`Iwssim::with_config`]
    /// continue to allocate a whole-image pipeline.
    pub fn new_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
    ) -> Result<Self> {
        Self::new_strip_with_halo(client, image_w, image_h, h_body, STRIP_DEFAULT_HALO)
    }

    /// Like [`Iwssim::new_strip`] but lets the caller pick the halo
    /// rows per side. Use this when the default halo is too generous
    /// (e.g. small but still ≥ 176 px tall images where 256-row halo
    /// would force the entire image into a single strip). Halo MUST
    /// be a non-zero multiple of 16 (`2^(NUM_SCALES − 1)`) — see
    /// `docs/STRIP_PROCESSING.md`.
    pub fn new_strip_with_halo(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
        halo: u32,
    ) -> Result<Self> {
        if image_w < MIN_NATIVE_DIM || image_h < MIN_NATIVE_DIM {
            return Err(Error::InvalidImageSize);
        }
        // Halo must respect the pyramid downsampling factor so it
        // shrinks to an integer row count at every scale.
        let pyr_factor: u32 = 1 << (NUM_SCALES - 1); // 16
        if halo == 0 || !halo.is_multiple_of(pyr_factor) {
            return Err(Error::InvalidImageSize);
        }
        if h_body == 0 || !h_body.is_multiple_of(pyr_factor) {
            return Err(Error::InvalidImageSize);
        }
        let strip_alloc_h = h_body + 2 * halo;
        // The allocation strip must satisfy the same 5-level pyramid
        // floor as a whole image. A 176-row floor at scale 0 leaves
        // 11 rows at scale 4 — exactly the 11×11 valid-blur radius.
        if strip_alloc_h < MIN_NATIVE_DIM {
            return Err(Error::InvalidImageSize);
        }
        // Allocate per-scale buffers sized for the MAX strip
        // (h_body + 2 halo). Boundary strips with fewer rows pass a
        // smaller actual_h into the kernels — buffers have extra
        // capacity at the tail.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut h = strip_alloc_h;
        let mut w = image_w;
        for _ in 0..NUM_SCALES {
            dims.push((h, w));
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        let scales: Vec<Scale> = dims
            .iter()
            .map(|&(h, w)| Scale::new(&client, h, w))
            .collect();
        let n_pixels_alloc = (strip_alloc_h * image_w) as usize;
        let src_u32_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_alloc]));
        let src_u32_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n_pixels_alloc]));

        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;
        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; partials_len]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; NUM_SLOTS as usize]));

        let cov_partials_len = (COV_MAX_CELLS * COV_N_THREADS) as usize;
        let cov_partials =
            client.create_from_slice(f32::as_bytes(&vec![0.0_f32; cov_partials_len]));

        Ok(Self {
            client,
            width: image_w,
            height: image_h,
            pad_width: image_w,
            pad_height: image_h,
            strategy: IwssimStrategy::Reject,
            src_u32_a,
            src_u32_b,
            scales,
            partials,
            sums,
            cov_partials,
            has_reference: false,
            strip: Some(StripState {
                image_h,
                h_body,
                halo,
                strip_alloc_h,
            }),
            cached_strip_ref: None,
        })
    }

    /// True if this pipeline was constructed via [`Iwssim::new_strip`].
    pub fn is_strip_mode(&self) -> bool {
        self.strip.is_some()
    }

    /// True if [`Self::set_reference_stripped`] has populated the
    /// per-strip cached-reference state. Strip mode only; whole-image
    /// callers should use [`Self::has_reference`] instead.
    pub fn has_cached_reference_stripped(&self) -> bool {
        self.cached_strip_ref.is_some()
    }

    /// Drop any cached-reference strip state. Subsequent
    /// [`Self::compute_with_reference_stripped`] calls will fail with
    /// [`Error::NoCachedReference`] until a fresh
    /// [`Self::set_reference_stripped`] is run.
    pub fn clear_reference_stripped(&mut self) {
        self.cached_strip_ref = None;
    }

    /// Native `(width, height)` the caller supplied to `new` /
    /// `with_config`. Use this for input buffer length checks.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Padded `(width, height)` actually fed to the GPU pipeline.
    /// Equal to `dimensions()` when not in `allow_small` mode, or
    /// when both native axes already meet `MIN_NATIVE_DIM`.
    pub fn padded_dimensions(&self) -> (u32, u32) {
        (self.pad_width, self.pad_height)
    }

    /// True when the pipeline is padding native inputs to reach
    /// `MIN_NATIVE_DIM` on at least one axis.
    pub fn is_padded(&self) -> bool {
        self.pad_width != self.width || self.pad_height != self.height
    }

    /// Which small-image strategy this instance is using. `Reject`
    /// is impossible at this point — construction would have failed.
    pub fn strategy(&self) -> IwssimStrategy {
        self.strategy
    }

    /// Dispatch host-side f32 padding by strategy. Caller guarantees
    /// `src.len() == sw*sh`, `dw >= sw`, `dh >= sh`.
    fn pad_f32(&self, src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
        match self.strategy {
            IwssimStrategy::Reject => {
                // Should never reach here; with Reject we don't pad.
                debug_assert_eq!(sw, dw);
                debug_assert_eq!(sh, dh);
                src.to_vec()
            }
            IwssimStrategy::Tile => tile_pad_f32(src, sw, sh, dw, dh),
            IwssimStrategy::ReflectPad => reflect_pad_f32(src, sw, sh, dw, dh),
        }
    }

    /// Dispatch host-side RGB-u8 padding by strategy.
    fn pad_rgb_u8(&self, src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
        match self.strategy {
            IwssimStrategy::Reject => {
                debug_assert_eq!(sw, dw);
                debug_assert_eq!(sh, dh);
                src.to_vec()
            }
            IwssimStrategy::Tile => tile_pad_rgb_u8(src, sw, sh, dw, dh),
            IwssimStrategy::ReflectPad => reflect_pad_rgb_u8(src, sw, sh, dw, dh),
        }
    }
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }
    pub fn has_reference(&self) -> bool {
        self.has_reference
    }
    /// Drop any cached reference state. `compute_with_reference` will
    /// fail with `NoCachedReference` until a fresh `set_reference` is
    /// run. Also clears any cached strip-mode state (so re-uploading
    /// a new reference is a single call regardless of mode).
    pub fn clear_reference(&mut self) {
        self.has_reference = false;
        self.cached_strip_ref = None;
    }

    /// Upload `ref_gray` and pre-compute the reference-side Laplacian
    /// pyramid. Subsequent `compute_with_reference` calls reuse the
    /// cached `lp_ref[s]` at every scale, skipping the ref-side
    /// downsample + upConv work.
    ///
    /// Saves roughly half the LP-pyramid build time per call (and at
    /// 4096² the much larger reference upload), with no parity impact:
    /// the rest of the pipeline reads `lp_ref` exactly as before.
    ///
    /// `ref_gray.len()` must equal `width * height` (native dims). If
    /// `is_padded()` is true, the buffer is reflect-padded on the host
    /// to `pad_width × pad_height` before upload.
    pub fn set_reference(&mut self, ref_gray: &[f32]) -> Result<()> {
        if self.strip.is_some() {
            return Err(Error::CachedRefNotSupportedInStripMode);
        }
        let expected = (self.width * self.height) as usize;
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        let uploaded = if self.is_padded() {
            let padded = self.pad_f32(
                ref_gray,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            self.client.create_from_slice(f32::as_bytes(&padded))
        } else {
            self.client.create_from_slice(f32::as_bytes(ref_gray))
        };
        self.scales[0].g_ref = uploaded;
        // Build only the ref-side pyramid; the dis-side will be built
        // in `compute_with_reference`.
        self.build_laplacian_pyramid(true);
        self.has_reference = true;
        Ok(())
    }

    /// Score one distortion against the cached reference. Returns
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called.
    ///
    /// `dis_gray.len()` must equal `width * height` (native dims).
    /// Reflect-padded when `is_padded()` is true (same contract as
    /// [`set_reference`]).
    pub fn compute_with_reference(&mut self, dis_gray: &[f32]) -> Result<GpuIwssimResult> {
        if self.strip.is_some() {
            return Err(Error::CachedRefNotSupportedInStripMode);
        }
        if !self.has_reference {
            return Err(Error::NoCachedReference);
        }
        let expected = (self.width * self.height) as usize;
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        let uploaded = if self.is_padded() {
            let padded = self.pad_f32(
                dis_gray,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            self.client.create_from_slice(f32::as_bytes(&padded))
        } else {
            self.client.create_from_slice(f32::as_bytes(dis_gray))
        };
        self.scales[0].g_dis = uploaded;
        // Skip the ref-side pyramid; only build dis-side.
        self.build_laplacian_pyramid(false);
        // Then the rest of the pipeline reads both `lp_ref[s]` (cached)
        // and `lp_dis[s]` (just built) — same as `run_pipeline`'s
        // post-pyramid stages.
        self.run_pipeline_post_pyramid()
    }

    /// Score one RGB-u8 pair. Both buffers must be `width × height × 3`
    /// in RGB byte order (native dimensions). The pipeline performs the
    /// BT.601 rgb→gray + half-up rounding step on the GPU.
    ///
    /// When `is_padded()` is true, the inputs are host-side
    /// reflect-padded RGB to `pad_width × pad_height × 3` before being
    /// packed and uploaded.
    pub fn compute_rgb(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<GpuIwssimResult> {
        // Strip-mode instances need a separate dispatch path — the
        // strip walker requires f32 gray inputs (the on-device
        // rgb→gray kernel only sees the strip-sized staging buffer).
        // Convert host-side using BT.601 rounded (matches the on-
        // device `rgb_u32_to_gray_kernel`).
        if self.strip.is_some() {
            let expected = (self.width * self.height * 3) as usize;
            if ref_rgb.len() != expected {
                return Err(Error::DimensionMismatch {
                    expected,
                    got: ref_rgb.len(),
                });
            }
            if dis_rgb.len() != expected {
                return Err(Error::DimensionMismatch {
                    expected,
                    got: dis_rgb.len(),
                });
            }
            let ref_gray = rgb_u8_to_gray_bt601(ref_rgb);
            let dis_gray = rgb_u8_to_gray_bt601(dis_rgb);
            return self.compute_gray_stripped(&ref_gray, &dis_gray);
        }
        let expected = (self.width * self.height * 3) as usize;
        if ref_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_rgb.len(),
            });
        }
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        // T_x.O (2026-05-17): pack u8×3 → u32 directly into the
        // pinned staging buffer (one host-side pass instead of two).
        // Previously we packed into `self.pack_scratch` and then
        // `create_from_slice_pinned` copied that scratch into a
        // pinned buffer — two ~48 MB host writes per upload. The
        // reserve_staging path lets us produce the packed bytes
        // straight into the pinned buffer. T4.M's pinned-DMA fast
        // path is preserved (handle from `client.create`).
        //
        // Layout (unchanged from T4.L): 4 bytes per pixel — R | G<<8
        // | B<<16 (alpha unused). Reader (`rgb_u32_to_gray_kernel`)
        // sees the same `[u32]` packing.
        if self.is_padded() {
            let ref_pad = self.pad_rgb_u8(
                ref_rgb,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            let dis_pad = self.pad_rgb_u8(
                dis_rgb,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            self.src_u32_a = Self::pack_into_pinned(&self.client, &ref_pad);
            self.src_u32_b = Self::pack_into_pinned(&self.client, &dis_pad);
        } else {
            self.src_u32_a = Self::pack_into_pinned(&self.client, ref_rgb);
            self.src_u32_b = Self::pack_into_pinned(&self.client, dis_rgb);
        }

        self.rgb_u32_to_gray_from_packed();
        self.run_pipeline()
    }

    /// Pack a `width × height × 3` sRGB-u8 buffer into the packed-u32
    /// device handle layout that [`Self::compute_handles`] expects.
    /// Uses the same pinned-staging fast path as the internal upload.
    ///
    /// When `is_padded()` is true the host-side reflect-pad is applied
    /// before packing; the returned handle has length `pad_width *
    /// pad_height` u32s, not `width * height`.
    ///
    /// Returns `Err(DimensionMismatch)` if `srgb.len() != width *
    /// height * 3`.
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        let expected = (self.width * self.height * 3) as usize;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        if self.is_padded() {
            let padded = self.pad_rgb_u8(
                srgb,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            Ok(Self::pack_into_pinned(&self.client, &padded))
        } else {
            Ok(Self::pack_into_pinned(&self.client, srgb))
        }
    }

    /// Compute against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. Equivalent to
    /// [`Self::compute_rgb`] but skips the internal byte pack/upload.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this `Iwssim<R>`.
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuIwssimResult> {
        self.src_u32_a = ref_handle.clone();
        self.src_u32_b = dis_handle.clone();
        self.rgb_u32_to_gray_from_packed();
        self.run_pipeline()
    }

    /// Run the packed-u32 → grayscale kernel on whichever handles
    /// currently sit in `src_u32_a` / `src_u32_b`. Split out of
    /// [`Self::compute_rgb`] so [`Self::compute_handles`] can reuse
    /// the dispatch step without re-packing bytes.
    ///
    /// Uses padded dimensions — the packed buffers are sized for the
    /// padded image, and scale-0 `g_ref`/`g_dis` are sized for the
    /// padded image (see `Scale::new`).
    fn rgb_u32_to_gray_from_packed(&self) {
        let n_pixels = (self.pad_width * self.pad_height) as usize;
        let s0 = &self.scales[0];
        unsafe {
            rgb2gray::rgb_u32_to_gray_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_pixels),
                Self::cube_dim_1d(),
                // T4.L: one u32 per pixel.
                ArrayArg::from_raw_parts(self.src_u32_a.clone(), n_pixels),
                ArrayArg::from_raw_parts(s0.g_ref.clone(), n_pixels),
            );
            rgb2gray::rgb_u32_to_gray_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_pixels),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(self.src_u32_b.clone(), n_pixels),
                ArrayArg::from_raw_parts(s0.g_dis.clone(), n_pixels),
            );
        }
    }

    /// Score one grayscale-f32 pair. Both buffers must be `width × height`
    /// floats in the 0..=255 range (matches the reference convention
    /// of `L = 255` for the SSIM constants).
    ///
    /// When `is_padded()` is true, both inputs are reflect-padded on
    /// the host from native to padded dims before upload.
    pub fn compute_gray(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let expected = (self.width * self.height) as usize;
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }
        // Direct upload into g_ref / g_dis (scale-0 working Gaussian).
        // Replace the handle so the new contents are visible.
        let t = std::time::Instant::now();
        let (h_ref, h_dis) = if self.is_padded() {
            let r = self.pad_f32(
                ref_gray,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            let d = self.pad_f32(
                dis_gray,
                self.width as usize,
                self.height as usize,
                self.pad_width as usize,
                self.pad_height as usize,
            );
            (
                self.client.create_from_slice(f32::as_bytes(&r)),
                self.client.create_from_slice(f32::as_bytes(&d)),
            )
        } else {
            (
                self.client.create_from_slice(f32::as_bytes(ref_gray)),
                self.client.create_from_slice(f32::as_bytes(dis_gray)),
            )
        };
        // Swap handles into scale-0. Earlier g_ref/g_dis is dropped.
        self.scales[0].g_ref = h_ref;
        self.scales[0].g_dis = h_dis;
        if profile {
            cubecl::future::block_on(self.client.sync()).expect("client.sync");
            eprintln!(
                "    stage 'upload': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        self.run_pipeline()
    }

    /// Score one grayscale-f32 pair via the strip-processing path.
    /// Only valid on instances constructed with [`Iwssim::new_strip`];
    /// returns [`Error::NotStripMode`] when called on a whole-image
    /// instance.
    ///
    /// Both buffers must be `image_w × image_h` floats (native dims).
    /// The implementation slices the image into strips, runs the
    /// existing whole-image pipeline on each strip, accumulates
    /// per-strip partial sums on the host, and finalizes once at the
    /// end. Peak GPU working set is bounded by a single strip's
    /// allocation (~ `strip_alloc_h × image_w × 4 B × scale_factor`),
    /// not by the full image.
    ///
    /// # Reduction order
    ///
    /// f32 sums are reordered per-strip vs the whole-image path; the
    /// drift is ~1e-5 rel — well below the cross-backend tolerance
    /// (5e-4) and well below the parity-test tolerance (5e-3 against
    /// the Python reference). The cached-reference strip path is NOT
    /// implemented in this pass (see `docs/STRIP_PROCESSING.md` —
    /// follow-up work).
    pub fn compute_gray_stripped(
        &mut self,
        ref_gray: &[f32],
        dis_gray: &[f32],
    ) -> Result<GpuIwssimResult> {
        let strip_state = match self.strip {
            Some(s) => s,
            None => return Err(Error::NotStripMode),
        };
        let expected = (self.width * self.height) as usize;
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }

        let image_w = self.width;
        let strips = strip_state.strips();
        let n_scales_iw = self.scales.len() - 1;

        // ── Pass 1: build LP, accumulate per-scale raw Σ Yᵀ Y over
        //   each strip's BODY iw row range only (no halo overlap).
        //   Sum the raw matrices on host; once all strips finish,
        //   divide by total nexp, eigendecompose, upload C_u_inv +
        //   lambda to per-scale device buffers.
        //
        // The matrix is at most 10×10, so per-scale host accumulators
        // are tiny — keep them in a 5-element vec of fixed 100-cell
        // f64 buffers.
        let mut acc_cu = vec![vec![0.0_f64; 100]; n_scales_iw];
        let mut total_nexp = vec![0_u64; n_scales_iw];
        let mut n_dim_per_scale = vec![0_usize; n_scales_iw];
        let mut has_parent_per_scale = vec![false; n_scales_iw];

        for &(body_lo, body_hi, up_lo, up_hi) in strips.iter() {
            let actual_strip_h = up_hi - up_lo;
            self.set_scale_dims_for_strip(actual_strip_h, image_w);
            self.upload_strip_gray(ref_gray, dis_gray, up_lo, up_hi);
            self.build_laplacian_pyramid(true);
            self.build_laplacian_pyramid(false);

            // Per scale: box3 + parent_band + cov_accum (body iw range).
            let body_lp_top = (body_lo - up_lo) as i64;
            let body_lp_bot = (body_hi - up_lo) as i64;
            for s in 0..n_scales_iw {
                let iw_h = self.scales[s].iw_h;
                let (py_lo, py_hi) = body_iw_range(body_lp_top, body_lp_bot, s, iw_h);
                if py_hi <= py_lo {
                    // Strip's body contributes no iw rows at this
                    // scale — skip cov accum to preserve sc.cu from
                    // a previous strip / leave it empty for the
                    // host accumulator.
                    continue;
                }
                self.run_iw_box3_parent(s);
                self.run_iw_cov_accum_range(s, py_lo, py_hi);
                let (cu_raw, n_dim, has_parent) = self.read_cu_raw(s);
                n_dim_per_scale[s] = n_dim;
                has_parent_per_scale[s] = has_parent;
                let device_stride = if has_parent { 10 } else { 9 };
                // Accumulate raw Σ Yᵀ Y (NOT yet divided by nexp).
                // device buffer is 10×10; we read the top-left
                // n_dim × n_dim block. cu_raw is already f64 (the
                // cov_finalize kernel sums in f64).
                for i in 0..n_dim {
                    for j in 0..n_dim {
                        acc_cu[s][i * 10 + j] += cu_raw[i * device_stride + j];
                    }
                }
                let strip_nexp = (py_hi - py_lo) as u64 * (self.scales[s].iw_w as u64);
                total_nexp[s] += strip_nexp;
            }
        }

        // ── Between passes: divide acc_cu by total_nexp, eigendecompose,
        //    upload C_u_inv + lambda once per scale. These device handles
        //    are referenced by every strip's infow launch in pass 2.
        for s in 0..n_scales_iw {
            if total_nexp[s] == 0 {
                // No strip contributed — set C_u_inv to identity-ish
                // (eigvals = 1) so infow doesn't divide by zero.
                let n_dim = if s < self.scales.len() - 2 { 10 } else { 9 };
                let mut cu_f64 = vec![0.0_f64; n_dim * n_dim];
                for i in 0..n_dim {
                    cu_f64[i * n_dim + i] = 1.0;
                }
                self.eig_and_upload(s, &cu_f64, n_dim, s < self.scales.len() - 2);
                continue;
            }
            let n_dim = n_dim_per_scale[s];
            let nexp_f64 = total_nexp[s] as f64;
            // Pack raw Σ Yᵀ Y (top-left n_dim block of the 10×10
            // accumulator) into a tight n_dim × n_dim f64 matrix
            // and divide by total_nexp.
            let mut cu_f64 = vec![0.0_f64; n_dim * n_dim];
            for i in 0..n_dim {
                for j in 0..n_dim {
                    cu_f64[i * n_dim + j] = acc_cu[s][i * 10 + j] / nexp_f64;
                }
            }
            self.eig_and_upload(s, &cu_f64, n_dim, has_parent_per_scale[s]);
        }

        // ── Pass 2: rebuild LP, run ssim_stats, box3+parent_band, infow
        //   (with the now-uploaded global C_u_inv), reductions on the
        //   body row range, accumulate scalar sums on host.
        let mut acc_csiw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_iw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_csl: f64 = 0.0;
        let mut top_pool_count: u64 = 0;
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;

        for &(body_lo, body_hi, up_lo, up_hi) in strips.iter() {
            let actual_strip_h = up_hi - up_lo;
            self.set_scale_dims_for_strip(actual_strip_h, image_w);
            self.upload_strip_gray(ref_gray, dis_gray, up_lo, up_hi);
            self.build_laplacian_pyramid(true);
            self.build_laplacian_pyramid(false);
            for s in 0..self.scales.len() {
                self.run_ssim_stats(s);
            }
            for s in 0..n_scales_iw {
                self.run_iw_box3_parent(s);
                self.run_iw_infow(s);
            }

            // Per-strip reductions over the body row range.
            let body_lp_top = (body_lo - up_lo) as i64;
            let body_lp_bot = (body_hi - up_lo) as i64;
            for s in 0..n_scales_iw {
                let sc = &self.scales[s];
                let cs_n = (sc.cs_h as usize) * (sc.cs_w as usize);
                let iw_n = (sc.iw_h as usize) * (sc.iw_w as usize);
                let (cs_y_start, cs_y_end) = body_cs_range(body_lp_top, body_lp_bot, s, sc.cs_h);
                if cs_y_end <= cs_y_start {
                    self.zero_partial_slot(SLOT_CSIW_BASE + s as u32);
                    self.zero_partial_slot(SLOT_IW_BASE + s as u32);
                    continue;
                }
                reduction::launch_weighted_sum::<R>(
                    &self.client,
                    sc.cs.clone(),
                    cs_n,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_CSIW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
                reduction::launch_iw_sum::<R>(
                    &self.client,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_IW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
            }
            let top = self.scales.len() - 1;
            let sc_top = &self.scales[top];
            let cs_top_n = (sc_top.cs_h as usize) * (sc_top.cs_w as usize);
            let (top_y_start, top_y_end) =
                body_cs_range(body_lp_top, body_lp_bot, top, sc_top.cs_h);
            if top_y_end > top_y_start {
                reduction::launch_plain_sum::<R>(
                    &self.client,
                    sc_top.cs.clone(),
                    cs_top_n,
                    self.partials.clone(),
                    partials_len,
                    SLOT_CSL,
                    sc_top.cs_w,
                    top_y_start,
                    top_y_end,
                );
                top_pool_count += (top_y_end - top_y_start) as u64 * sc_top.cs_w as u64;
            } else {
                self.zero_partial_slot(SLOT_CSL);
            }

            reduction::launch_finalize::<R>(
                &self.client,
                self.partials.clone(),
                partials_len,
                self.sums.clone(),
                NUM_SLOTS as usize,
                NUM_SLOTS,
            );

            let bytes = self.client.read_one(self.sums.clone()).expect("read sums");
            let sums = f32::from_bytes(&bytes);
            debug_assert_eq!(sums.len(), NUM_SLOTS as usize);
            for s in 0..n_scales_iw {
                acc_csiw[s] += sums[(SLOT_CSIW_BASE + s as u32) as usize] as f64;
                acc_iw[s] += sums[(SLOT_IW_BASE + s as u32) as usize] as f64;
            }
            acc_csl += sums[SLOT_CSL as usize] as f64;
        }

        // Combine accumulated per-strip sums into the final score.
        let mut per_scale = [1.0_f64; NUM_SCALES];
        for s in 0..n_scales_iw {
            let num = acc_csiw[s];
            let den = acc_iw[s];
            per_scale[s] = if den == 0.0 || !den.is_finite() {
                1.0
            } else {
                num / den
            };
        }
        let top = self.scales.len() - 1;
        per_scale[top] = if top_pool_count == 0 || !acc_csl.is_finite() {
            1.0
        } else {
            acc_csl / (top_pool_count as f64)
        };

        let mut score = 1.0_f64;
        for s in 0..self.scales.len() {
            let b = filters::SCALE_WEIGHTS[s] as f64;
            let v = per_scale[s].abs();
            score *= v.powf(b);
        }
        Ok(GpuIwssimResult { score, per_scale })
    }

    /// Upload `ref_gray` and pre-compute the per-strip reference-side
    /// state (Laplacian pyramid + eigendecomposed C_u_inv + lambda per
    /// scale) used by [`Self::compute_with_reference_stripped`]. Only
    /// valid on instances constructed with [`Iwssim::new_strip`]; the
    /// whole-image equivalent is [`Self::set_reference`].
    ///
    /// For RD-search workloads scoring one reference against many
    /// distorted images, this elides:
    ///   * the **ref-side LP pyramid build** per strip per
    ///     `compute_with_reference_stripped` call, AND
    ///   * the entire **pass-1 cov accumulation** (which depends only
    ///     on the reference's LP at each scale).
    ///
    /// `compute_with_reference_stripped` then runs only pass-2 on each
    /// strip: dis-side LP build, ssim_stats, iw_box3_parent, iw_infow,
    /// reductions. Empirically the per-call cost is ~halved for the
    /// 1-ref × N-dist hot loop — `benchmarks/iwssim_cachedref_strip_*.csv`.
    ///
    /// `ref_gray.len()` must equal `image_w × image_h` (native dims).
    /// On error, the previous cached-strip state (if any) is preserved.
    ///
    /// # Errors
    ///
    /// - [`Error::NotStripMode`] when called on a whole-image instance
    ///   (constructed via [`Iwssim::new`] / [`Iwssim::with_config`]).
    /// - [`Error::DimensionMismatch`] when `ref_gray.len()` doesn't
    ///   match the configured `image_w × image_h`.
    pub fn set_reference_stripped(&mut self, ref_gray: &[f32]) -> Result<()> {
        let strip_state = match self.strip {
            Some(s) => s,
            None => return Err(Error::NotStripMode),
        };
        let expected = (self.width * self.height) as usize;
        if ref_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_gray.len(),
            });
        }

        let image_w = self.width;
        let strips = strip_state.strips();
        let n_strips = strips.len();
        let n_scales_iw = self.scales.len() - 1;

        // Reset `scales[s].lp_ref` to a fresh full-strip allocation.
        // A prior `compute_with_reference_stripped` call (line 1809 era)
        // replaces `scales[s].lp_ref` with the cached handle for the
        // last strip processed (which is sized to that strip's
        // `actual_strip_h × image_w`, not the maximum `strip_alloc_h`).
        // If we don't reset, the next `build_laplacian_pyramid` writes
        // through `pointwise_sub_kernel(... lp_cur)` to a buffer that
        // may be smaller than the current strip's `sc.h × sc.w`,
        // producing a kernel out-of-bounds + a panic at the readback
        // below where `bytes[..n*4]` overruns.
        let max_strip_h = strip_state.h_body + 2 * strip_state.halo;
        {
            let mut h = max_strip_h;
            let mut w = image_w;
            for s in 0..self.scales.len() {
                let n = (h as usize) * (w as usize);
                self.scales[s].lp_ref = alloc(&self.client, n);
                h = h.div_ceil(2);
                w = w.div_ceil(2);
            }
        }

        // Build per-strip per-scale lp_ref handles. Each strip's LP
        // pyramid lives in INDEPENDENT device memory so subsequent
        // strips don't trample it via `scales[s].lp_ref`. We allocate
        // fresh handles per (strip, scale) here, build into them, and
        // park them in `lp_ref_cache` for `compute_with_reference_stripped`
        // to swap into `scales[s].lp_ref` per iteration.
        //
        // Capacity vs perf: at 1024² body=256 halo=256 (1536 max strip
        // height, 4 strips), per-strip per-scale LP storage is
        // strip_alloc_h × image_w × 4 B summed across the 5-level
        // pyramid = ~14 MB per strip × 4 strips = ~56 MB. Trivial
        // relative to the whole-image working set we're already paying
        // for, and amortized across many `compute_with_reference_stripped`
        // calls in the RD-search hot loop.
        let mut lp_ref_cache: Vec<Vec<cubecl::server::Handle>> = Vec::with_capacity(n_strips);

        // Per-scale raw Σ Yᵀ Y accumulator and total iw row count.
        // These follow `compute_gray_stripped`'s pass-1 logic exactly
        // — we're just running it once with the ref input and caching
        // the results.
        let mut acc_cu = vec![vec![0.0_f64; 100]; n_scales_iw];
        let mut total_nexp = vec![0_u64; n_scales_iw];
        let mut n_dim_per_scale = vec![0_usize; n_scales_iw];
        let mut has_parent_per_scale = vec![false; n_scales_iw];

        for &(body_lo, body_hi, up_lo, up_hi) in strips.iter() {
            let actual_strip_h = up_hi - up_lo;
            self.set_scale_dims_for_strip(actual_strip_h, image_w);

            // Upload ref strip to scale-0 g_ref (only — we don't need
            // dis here). Build ONLY the ref-side pyramid.
            let row_stride = self.width as usize;
            let ref_strip: &[f32] =
                &ref_gray[(up_lo as usize) * row_stride..(up_hi as usize) * row_stride];
            self.scales[0].g_ref = self.client.create_from_slice(f32::as_bytes(ref_strip));
            self.build_laplacian_pyramid(true);

            // Snapshot each scale's lp_ref handle into the cache. The
            // build wrote into `scales[s].lp_ref`; that handle clones
            // cheaply (Arc-style refcount) and the underlying buffer
            // lives as long as we hold a clone. The next strip's
            // `build_laplacian_pyramid(true)` will REASSIGN
            // `scales[s].lp_ref` to a different handle (because the
            // top-scale path does `self.scales[top].lp_ref = sc.g_ref.clone()`
            // which writes a new field value), so our cached handles
            // are not overwritten.
            //
            // CAVEAT: at the top scale, `lp_ref` is just an alias of
            // `g_ref` (no separate LP buffer at the residual). We need
            // an independent copy so the next strip's g_ref write
            // doesn't poison it. Read the top scale's bytes back and
            // re-upload to a fresh handle — at the top scale the data
            // is tiny (strip_alloc_h/16 × image_w/16 × 4 B ≈ tens of
            // KB at 1024², ≪ 1 MB at 6000² × 1536).
            //
            // For scales s < top, `lp_ref` is a per-scale allocation
            // distinct from `g_ref` (see `Scale::new`), so cloning the
            // handle is sufficient — but the SHARED scale-buffer is
            // also reused per strip. We must take an INDEPENDENT copy
            // for every scale to survive across strips.
            //
            // Easiest correct approach: read each scale's lp_ref back
            // to host and re-upload to a fresh device handle. At
            // strip_alloc_h × image_w bytes worst-case (scale 0) this
            // is a few MB per strip; the readback + reupload happens
            // once per `set_reference_stripped` call, NOT per
            // `compute_with_reference_stripped`, so it's amortized.
            let mut strip_lp: Vec<cubecl::server::Handle> = Vec::with_capacity(self.scales.len());
            for s in 0..self.scales.len() {
                let sc = &self.scales[s];
                let n = (sc.h as usize) * (sc.w as usize);
                let bytes = self
                    .client
                    .read_one(sc.lp_ref.clone())
                    .expect("read lp_ref strip cache");
                // `bytes.len()` is the underlying allocation size in
                // bytes (cubecl's `read_one` returns the handle's full
                // `size_in_used()`). The fresh-alloc reset above
                // guarantees `bytes.len() >= max_strip_h × image_w × 4`
                // at scale 0, so `bytes[..n*4]` is always in-range.
                let active = &bytes[..n * 4];
                strip_lp.push(self.client.create_from_slice(active));
            }
            lp_ref_cache.push(strip_lp);

            // Pass-1 cov accumulation per scale (body iw range only).
            let body_lp_top = (body_lo - up_lo) as i64;
            let body_lp_bot = (body_hi - up_lo) as i64;
            for s in 0..n_scales_iw {
                let iw_h = self.scales[s].iw_h;
                let (py_lo, py_hi) = body_iw_range(body_lp_top, body_lp_bot, s, iw_h);
                if py_hi <= py_lo {
                    continue;
                }
                self.run_iw_box3_parent_ref_only(s);
                self.run_iw_cov_accum_range(s, py_lo, py_hi);
                let (cu_raw, n_dim, has_parent) = self.read_cu_raw(s);
                n_dim_per_scale[s] = n_dim;
                has_parent_per_scale[s] = has_parent;
                let device_stride = if has_parent { 10 } else { 9 };
                // cu_raw is f64 (cov_finalize accumulates in f64).
                for i in 0..n_dim {
                    for j in 0..n_dim {
                        acc_cu[s][i * 10 + j] += cu_raw[i * device_stride + j];
                    }
                }
                let strip_nexp = (py_hi - py_lo) as u64 * (self.scales[s].iw_w as u64);
                total_nexp[s] += strip_nexp;
            }
        }

        // Eigendecompose + cache per-scale C_u_inv + lambda on host.
        // `compute_with_reference_stripped` uploads these once per
        // call (NOT per strip — the global Cu is image-wide).
        let mut cu_inv_per_scale: Vec<Vec<f32>> = Vec::with_capacity(n_scales_iw);
        let mut lambda_per_scale: Vec<Vec<f32>> = Vec::with_capacity(n_scales_iw);
        for s in 0..n_scales_iw {
            if total_nexp[s] == 0 {
                let n_dim = if s < self.scales.len() - 2 { 10 } else { 9 };
                let mut cu_f64 = vec![0.0_f64; n_dim * n_dim];
                for i in 0..n_dim {
                    cu_f64[i * n_dim + i] = 1.0;
                }
                let eig_result = eig::decompose_and_invert(&cu_f64, n_dim);
                cu_inv_per_scale.push(eig_result.c_u_inv[..n_dim * n_dim].to_vec());
                lambda_per_scale.push(eig_result.lambda[..n_dim].to_vec());
                continue;
            }
            let n_dim = n_dim_per_scale[s];
            let nexp_f64 = total_nexp[s] as f64;
            let mut cu_f64 = vec![0.0_f64; n_dim * n_dim];
            for i in 0..n_dim {
                for j in 0..n_dim {
                    cu_f64[i * n_dim + j] = acc_cu[s][i * 10 + j] / nexp_f64;
                }
            }
            let eig_result = eig::decompose_and_invert(&cu_f64, n_dim);
            cu_inv_per_scale.push(eig_result.c_u_inv[..n_dim * n_dim].to_vec());
            lambda_per_scale.push(eig_result.lambda[..n_dim].to_vec());
        }

        // Upload the per-scale eig outputs to GPU ONCE here, holding
        // the resulting handles in the cache. Each per-call
        // `compute_with_reference_stripped` / `_rgb_*_native` then
        // clones these handles into the active `scales[s].cu_inv_dev`
        // / `scales[s].lambda_dev` slots — no per-call HtoD for these
        // constants. Same "deterministic given the cached reference"
        // pattern that `lp_ref_cache` above already uses.
        let mut cu_inv_dev_per_scale: Vec<cubecl::server::Handle> = Vec::with_capacity(n_scales_iw);
        let mut lambda_dev_per_scale: Vec<cubecl::server::Handle> = Vec::with_capacity(n_scales_iw);
        for s in 0..n_scales_iw {
            cu_inv_dev_per_scale.push(
                self.client
                    .create_from_slice(f32::as_bytes(&cu_inv_per_scale[s])),
            );
            lambda_dev_per_scale.push(
                self.client
                    .create_from_slice(f32::as_bytes(&lambda_per_scale[s])),
            );
        }

        // Replace any previous cache atomically — only after every
        // step above completed successfully, so a mid-call failure
        // leaves the previous cache intact.
        self.cached_strip_ref = Some(CachedStripRefState {
            lp_ref: lp_ref_cache,
            cu_inv_per_scale,
            lambda_per_scale,
            cu_inv_dev_per_scale,
            lambda_dev_per_scale,
            strip_count: n_strips,
        });
        Ok(())
    }

    /// Score one distortion against the per-strip cached reference.
    /// Equivalent in result to
    /// [`Self::compute_gray_stripped`]`(cached_ref, dis_gray)` but
    /// skips the ref-side LP pyramid build and the pass-1 cov
    /// accumulation per strip — both can be pre-computed once via
    /// [`Self::set_reference_stripped`] and reused.
    ///
    /// `dis_gray.len()` must equal `image_w × image_h`. Returns
    /// [`Error::NoCachedReference`] if `set_reference_stripped` hasn't
    /// been called (or [`Self::clear_reference_stripped`] dropped the
    /// cache), [`Error::NotStripMode`] on whole-image instances.
    ///
    /// # Reduction order
    ///
    /// Same reduction-order drift as
    /// [`Self::compute_gray_stripped`] vs the whole-image path —
    /// per-strip f32 sums reorder vs a single global pass. Bounded at
    /// ~1e-5 rel. The cached-ref path adds no additional drift: it's
    /// numerically identical to running `compute_gray_stripped` on
    /// the same `(ref, dis)` pair.
    pub fn compute_with_reference_stripped(&mut self, dis_gray: &[f32]) -> Result<GpuIwssimResult> {
        let strip_state = match self.strip {
            Some(s) => s,
            None => return Err(Error::NotStripMode),
        };
        let cached = match self.cached_strip_ref.as_ref() {
            Some(c) => c,
            None => return Err(Error::NoCachedReference),
        };
        let expected = (self.width * self.height) as usize;
        if dis_gray.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_gray.len(),
            });
        }

        let image_w = self.width;
        let strips = strip_state.strips();
        if strips.len() != cached.strip_count {
            // Shouldn't happen — StripState is immutable post-
            // construction — but guard anyway to keep `unwrap` indices
            // safe.
            return Err(Error::DimensionMismatch {
                expected: cached.strip_count,
                got: strips.len(),
            });
        }
        let n_scales_iw = self.scales.len() - 1;

        // Bind cached C_u_inv + lambda device handles into each
        // scale's active slot. These were uploaded ONCE by
        // `set_reference_stripped`; cloning the handle is just an
        // Arc-style refcount bump, no HtoD traffic. Replaces the
        // per-call `create_from_slice(cached.cu_inv_per_scale[s])`
        // pair that previously fired 2*n_scales_iw small HtoDs per
        // call. Mirrors the cvvdp-gpu Option C constants lift
        // (152a6924) and the lp_ref handle cache that already lives
        // in this same struct.
        for s in 0..n_scales_iw {
            self.scales[s].cu_inv_dev = cached.cu_inv_dev_per_scale[s].clone();
            self.scales[s].lambda_dev = cached.lambda_dev_per_scale[s].clone();
        }

        let mut acc_csiw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_iw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_csl: f64 = 0.0;
        let mut top_pool_count: u64 = 0;
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;

        for (strip_idx, &(body_lo, body_hi, up_lo, up_hi)) in strips.iter().enumerate() {
            let actual_strip_h = up_hi - up_lo;
            self.set_scale_dims_for_strip(actual_strip_h, image_w);

            // Upload dis strip + build ONLY the dis-side LP pyramid.
            let row_stride = self.width as usize;
            let dis_strip: &[f32] =
                &dis_gray[(up_lo as usize) * row_stride..(up_hi as usize) * row_stride];
            self.scales[0].g_dis = self.client.create_from_slice(f32::as_bytes(dis_strip));
            self.build_laplacian_pyramid(false);

            // Restore cached ref-side LP handles for this strip. After
            // this loop iteration ends the handles stay where the
            // cache put them — they aren't reassigned by the next
            // strip's dis-side build (which only writes lp_dis / g_dis).
            //
            // NOTE: we hold cached.lp_ref through the iteration via the
            // outer borrow; index into it without re-borrowing self.
            for s in 0..self.scales.len() {
                self.scales[s].lp_ref =
                    self.cached_strip_ref.as_ref().unwrap().lp_ref[strip_idx][s].clone();
            }

            // Pass-2: ssim_stats + iw_box3_parent + iw_infow + reductions.
            for s in 0..self.scales.len() {
                self.run_ssim_stats(s);
            }
            for s in 0..n_scales_iw {
                self.run_iw_box3_parent(s);
                self.run_iw_infow(s);
            }

            let body_lp_top = (body_lo - up_lo) as i64;
            let body_lp_bot = (body_hi - up_lo) as i64;
            for s in 0..n_scales_iw {
                let sc = &self.scales[s];
                let cs_n = (sc.cs_h as usize) * (sc.cs_w as usize);
                let iw_n = (sc.iw_h as usize) * (sc.iw_w as usize);
                let (cs_y_start, cs_y_end) = body_cs_range(body_lp_top, body_lp_bot, s, sc.cs_h);
                if cs_y_end <= cs_y_start {
                    self.zero_partial_slot(SLOT_CSIW_BASE + s as u32);
                    self.zero_partial_slot(SLOT_IW_BASE + s as u32);
                    continue;
                }
                reduction::launch_weighted_sum::<R>(
                    &self.client,
                    sc.cs.clone(),
                    cs_n,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_CSIW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
                reduction::launch_iw_sum::<R>(
                    &self.client,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_IW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
            }
            let top = self.scales.len() - 1;
            let sc_top = &self.scales[top];
            let cs_top_n = (sc_top.cs_h as usize) * (sc_top.cs_w as usize);
            let (top_y_start, top_y_end) =
                body_cs_range(body_lp_top, body_lp_bot, top, sc_top.cs_h);
            if top_y_end > top_y_start {
                reduction::launch_plain_sum::<R>(
                    &self.client,
                    sc_top.cs.clone(),
                    cs_top_n,
                    self.partials.clone(),
                    partials_len,
                    SLOT_CSL,
                    sc_top.cs_w,
                    top_y_start,
                    top_y_end,
                );
                top_pool_count += (top_y_end - top_y_start) as u64 * sc_top.cs_w as u64;
            } else {
                self.zero_partial_slot(SLOT_CSL);
            }

            reduction::launch_finalize::<R>(
                &self.client,
                self.partials.clone(),
                partials_len,
                self.sums.clone(),
                NUM_SLOTS as usize,
                NUM_SLOTS,
            );

            let bytes = self.client.read_one(self.sums.clone()).expect("read sums");
            let sums = f32::from_bytes(&bytes);
            debug_assert_eq!(sums.len(), NUM_SLOTS as usize);
            for s in 0..n_scales_iw {
                acc_csiw[s] += sums[(SLOT_CSIW_BASE + s as u32) as usize] as f64;
                acc_iw[s] += sums[(SLOT_IW_BASE + s as u32) as usize] as f64;
            }
            acc_csl += sums[SLOT_CSL as usize] as f64;
        }

        let mut per_scale = [1.0_f64; NUM_SCALES];
        for s in 0..n_scales_iw {
            let num = acc_csiw[s];
            let den = acc_iw[s];
            per_scale[s] = if den == 0.0 || !den.is_finite() {
                1.0
            } else {
                num / den
            };
        }
        let top = self.scales.len() - 1;
        per_scale[top] = if top_pool_count == 0 || !acc_csl.is_finite() {
            1.0
        } else {
            acc_csl / (top_pool_count as f64)
        };

        let mut score = 1.0_f64;
        for s in 0..self.scales.len() {
            let b = filters::SCALE_WEIGHTS[s] as f64;
            let v = per_scale[s].abs();
            score *= v.powf(b);
        }
        Ok(GpuIwssimResult { score, per_scale })
    }

    /// RGB-u8 variant of [`Self::compute_gray_stripped`]. Both inputs
    /// must be `image_w × image_h × 3` in packed RGB byte order
    /// (native dims). The pipeline performs the BT.601 rgb→gray
    /// conversion + half-up rounding on the **host** (matching the
    /// on-device `rgb_u32_to_gray_kernel`), then routes through
    /// [`Self::compute_gray_stripped`].
    ///
    /// Why host-side conversion: the strip walker uploads one strip's
    /// f32 gray plane to scale-0 `g_ref`/`g_dis` per iteration; the
    /// on-device packed-u32 → gray kernel expects a whole-image
    /// staging buffer (`src_u32_a`/`src_u32_b`) sized for the strip
    /// allocation, not per-strip uploads. Doing the conversion on the
    /// host once for the full image gives a single tight inner loop;
    /// strip-by-strip on-device conversion would require per-strip
    /// packing + extra kernel launches that don't recover the upload
    /// savings.
    ///
    /// # Errors
    ///
    /// - [`Error::NotStripMode`] on whole-image instances.
    /// - [`Error::DimensionMismatch`] when either buffer's length
    ///   doesn't match `image_w × image_h × 3`.
    pub fn compute_rgb_stripped(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<GpuIwssimResult> {
        if self.strip.is_none() {
            return Err(Error::NotStripMode);
        }
        let expected = (self.width * self.height * 3) as usize;
        if ref_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_rgb.len(),
            });
        }
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        let ref_gray = rgb_u8_to_gray_bt601(ref_rgb);
        let dis_gray = rgb_u8_to_gray_bt601(dis_rgb);
        self.compute_gray_stripped(&ref_gray, &dis_gray)
    }

    /// RGB-u8 variant of [`Self::set_reference_stripped`] — converts
    /// the reference to grayscale via host-side BT.601 (matching the
    /// on-device `rgb_u32_to_gray_kernel`) and delegates. Pairs with
    /// [`Self::compute_rgb_with_reference_stripped`] for RD-search
    /// workloads where the reference is held in sRGB form.
    pub fn set_rgb_reference_stripped(&mut self, ref_rgb: &[u8]) -> Result<()> {
        if self.strip.is_none() {
            return Err(Error::NotStripMode);
        }
        let expected = (self.width * self.height * 3) as usize;
        if ref_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_rgb.len(),
            });
        }
        let ref_gray = rgb_u8_to_gray_bt601(ref_rgb);
        self.set_reference_stripped(&ref_gray)
    }

    /// RGB-u8 variant of [`Self::compute_with_reference_stripped`].
    pub fn compute_rgb_with_reference_stripped(
        &mut self,
        dis_rgb: &[u8],
    ) -> Result<GpuIwssimResult> {
        if self.strip.is_none() {
            return Err(Error::NotStripMode);
        }
        let expected = (self.width * self.height * 3) as usize;
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }
        let dis_gray = rgb_u8_to_gray_bt601(dis_rgb);
        self.compute_with_reference_stripped(&dis_gray)
    }

    /// Native-RGB variant of [`Self::compute_rgb_with_reference_stripped`]
    /// — same external semantics (sRGB-u8 in, IW-SSIM score out, cached
    /// reference reused) but **skips the host-side BT.601 rgb→gray
    /// conversion** for the distorted image. Per strip:
    ///
    /// 1. Slice the dist sRGB rows covered by `[up_lo..up_hi)`.
    /// 2. Pack 3 sRGB bytes per pixel → one packed u32 directly into a
    ///    pinned staging buffer (no FP math, no full-image f32 alloc).
    /// 3. Upload the packed strip + launch
    ///    [`crate::kernels::rgb2gray::rgb_u32_to_gray_kernel`] to write
    ///    the strip-sized gray-f32 into `scales[0].g_dis` on the GPU.
    /// 4. Run pass-2 identically to the gray-input strip walker.
    ///
    /// The motivation comes from the `native_rgb_perf_probe` benchmark
    /// (`benchmarks/iwssim_native_rgb_perf_2026-05-26.csv`): at 1024²
    /// through 4096², host-side `rgb_u8_to_gray_bt601` consumed 35–41%
    /// of per-call wall time. Replacing it with strip-by-strip
    /// pack-and-launch saves that host work without altering the GPU
    /// math (the existing `rgb_u32_to_gray_kernel` is reused as-is).
    ///
    /// # Parity guarantee
    ///
    /// Scores are numerically identical to
    /// [`Self::compute_rgb_with_reference_stripped`] (the host and
    /// device BT.601 paths produce the same gray-f32 values by
    /// construction; see [`rgb_u8_to_gray_bt601`]). A regression test
    /// at `tests/rgb_strip_native.rs` locks this in.
    ///
    /// # Errors
    ///
    /// - [`Error::NotStripMode`] when called on a whole-image instance.
    /// - [`Error::NoCachedReference`] when no cached reference has been
    ///   set via [`Self::set_reference_stripped`] /
    ///   [`Self::set_rgb_reference_stripped`].
    /// - [`Error::DimensionMismatch`] when `dis_rgb.len() != width *
    ///   height * 3`.
    pub fn compute_rgb_with_reference_stripped_native(
        &mut self,
        dis_rgb: &[u8],
    ) -> Result<GpuIwssimResult> {
        let strip_state = match self.strip {
            Some(s) => s,
            None => return Err(Error::NotStripMode),
        };
        let cached = match self.cached_strip_ref.as_ref() {
            Some(c) => c,
            None => return Err(Error::NoCachedReference),
        };
        let expected = (self.width * self.height * 3) as usize;
        if dis_rgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dis_rgb.len(),
            });
        }

        let image_w = self.width;
        let strips = strip_state.strips();
        if strips.len() != cached.strip_count {
            return Err(Error::DimensionMismatch {
                expected: cached.strip_count,
                got: strips.len(),
            });
        }
        let n_scales_iw = self.scales.len() - 1;

        // Bind cached C_u_inv + lambda device handles into each
        // scale's active slot — handle clones, no HtoD. Same fix as
        // `compute_with_reference_stripped` above; see that site's
        // comment for the rationale.
        for s in 0..n_scales_iw {
            self.scales[s].cu_inv_dev = cached.cu_inv_dev_per_scale[s].clone();
            self.scales[s].lambda_dev = cached.lambda_dev_per_scale[s].clone();
        }

        let mut acc_csiw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_iw = [0.0_f64; NUM_SCALES - 1];
        let mut acc_csl: f64 = 0.0;
        let mut top_pool_count: u64 = 0;
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;

        let rgb_row_stride = image_w as usize * 3;

        for (strip_idx, &(body_lo, body_hi, up_lo, up_hi)) in strips.iter().enumerate() {
            let actual_strip_h = up_hi - up_lo;
            self.set_scale_dims_for_strip(actual_strip_h, image_w);

            // ===== Native-RGB strip upload =====
            // 1. Slice the dist RGB strip.
            let dis_strip_rgb: &[u8] =
                &dis_rgb[(up_lo as usize) * rgb_row_stride..(up_hi as usize) * rgb_row_stride];
            // 2. Pack to pinned packed-u32 (3 B → 4 B, no FP math).
            let strip_pack = Self::pack_into_pinned(&self.client, dis_strip_rgb);
            // 3. Allocate a strip-sized gray-f32 device buffer for the
            //    kernel to write into, then dispatch the kernel. Uses
            //    `create_from_slice` of a zero vec to match the
            //    existing alloc pattern (e.g., `Scale::new` line 328);
            //    the kernel overwrites every pixel so init contents
            //    don't matter — the zero vec is just to size the
            //    handle.
            let n_strip_pixels = (actual_strip_h as usize) * (image_w as usize);
            let g_dis_strip = self
                .client
                .create_from_slice(f32::as_bytes(&vec![0.0_f32; n_strip_pixels]));
            unsafe {
                rgb2gray::rgb_u32_to_gray_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_strip_pixels),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(strip_pack, n_strip_pixels),
                    ArrayArg::from_raw_parts(g_dis_strip.clone(), n_strip_pixels),
                );
            }
            self.scales[0].g_dis = g_dis_strip;
            // ===== End native-RGB strip upload =====

            self.build_laplacian_pyramid(false);

            // Restore cached ref-side LP handles for this strip.
            for s in 0..self.scales.len() {
                self.scales[s].lp_ref =
                    self.cached_strip_ref.as_ref().unwrap().lp_ref[strip_idx][s].clone();
            }

            // Pass-2: ssim_stats + iw_box3_parent + iw_infow + reductions.
            for s in 0..self.scales.len() {
                self.run_ssim_stats(s);
            }
            for s in 0..n_scales_iw {
                self.run_iw_box3_parent(s);
                self.run_iw_infow(s);
            }

            let body_lp_top = (body_lo - up_lo) as i64;
            let body_lp_bot = (body_hi - up_lo) as i64;
            for s in 0..n_scales_iw {
                let sc = &self.scales[s];
                let cs_n = (sc.cs_h as usize) * (sc.cs_w as usize);
                let iw_n = (sc.iw_h as usize) * (sc.iw_w as usize);
                let (cs_y_start, cs_y_end) = body_cs_range(body_lp_top, body_lp_bot, s, sc.cs_h);
                if cs_y_end <= cs_y_start {
                    self.zero_partial_slot(SLOT_CSIW_BASE + s as u32);
                    self.zero_partial_slot(SLOT_IW_BASE + s as u32);
                    continue;
                }
                reduction::launch_weighted_sum::<R>(
                    &self.client,
                    sc.cs.clone(),
                    cs_n,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_CSIW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
                reduction::launch_iw_sum::<R>(
                    &self.client,
                    sc.iw.clone(),
                    iw_n,
                    self.partials.clone(),
                    partials_len,
                    sc.cs_h,
                    sc.cs_w,
                    sc.iw_h,
                    sc.iw_w,
                    BOUND1,
                    SLOT_IW_BASE + s as u32,
                    cs_y_start,
                    cs_y_end,
                );
            }
            let top = self.scales.len() - 1;
            let sc_top = &self.scales[top];
            let cs_top_n = (sc_top.cs_h as usize) * (sc_top.cs_w as usize);
            let (top_y_start, top_y_end) =
                body_cs_range(body_lp_top, body_lp_bot, top, sc_top.cs_h);
            if top_y_end > top_y_start {
                reduction::launch_plain_sum::<R>(
                    &self.client,
                    sc_top.cs.clone(),
                    cs_top_n,
                    self.partials.clone(),
                    partials_len,
                    SLOT_CSL,
                    sc_top.cs_w,
                    top_y_start,
                    top_y_end,
                );
                top_pool_count += (top_y_end - top_y_start) as u64 * sc_top.cs_w as u64;
            } else {
                self.zero_partial_slot(SLOT_CSL);
            }

            reduction::launch_finalize::<R>(
                &self.client,
                self.partials.clone(),
                partials_len,
                self.sums.clone(),
                NUM_SLOTS as usize,
                NUM_SLOTS,
            );

            let bytes = self.client.read_one(self.sums.clone()).expect("read sums");
            let sums = f32::from_bytes(&bytes);
            debug_assert_eq!(sums.len(), NUM_SLOTS as usize);
            for s in 0..n_scales_iw {
                acc_csiw[s] += sums[(SLOT_CSIW_BASE + s as u32) as usize] as f64;
                acc_iw[s] += sums[(SLOT_IW_BASE + s as u32) as usize] as f64;
            }
            acc_csl += sums[SLOT_CSL as usize] as f64;
        }

        let mut per_scale = [1.0_f64; NUM_SCALES];
        for s in 0..n_scales_iw {
            let num = acc_csiw[s];
            let den = acc_iw[s];
            per_scale[s] = if den == 0.0 || !den.is_finite() {
                1.0
            } else {
                num / den
            };
        }
        let top = self.scales.len() - 1;
        per_scale[top] = if top_pool_count == 0 || !acc_csl.is_finite() {
            1.0
        } else {
            acc_csl / (top_pool_count as f64)
        };

        let mut score = 1.0_f64;
        for s in 0..self.scales.len() {
            let b = filters::SCALE_WEIGHTS[s] as f64;
            let v = per_scale[s].abs();
            score *= v.powf(b);
        }
        Ok(GpuIwssimResult { score, per_scale })
    }

    /// `run_iw_box3_parent` variant that only needs `lp_ref` —
    /// matches the existing helper exactly today (it already only
    /// reads `lp_ref` / `lp_dis` indirectly via `box3_gv_kernel` and
    /// the parent_band gather, but **the box3 kernel reads BOTH
    /// `lp_ref` AND `lp_dis`** to compute the joint 3×3 box stats).
    /// For the cached-ref-strip pass-1 path where we only care about
    /// the cov accumulator (which itself only reads `lp_ref` and
    /// `parent_band`), the box3 stage is unnecessary. We still run it
    /// — but with a benign `lp_dis` source — because the cov kernels
    /// don't read `g_buf` / `vv_buf` (those feed `infow`, which we
    /// don't run in pass-1). This means in pass-1 of
    /// `set_reference_stripped`, `box3_gv_kernel` reads `lp_dis` (a
    /// zero-or-stale buffer from `Scale::new`'s alloc) — harmless
    /// because its outputs are discarded.
    ///
    /// In other words: this helper is a thin wrapper that **could**
    /// skip the box3 launch entirely for pass-1, but doesn't, because
    /// the box3 launch is cheap (≤ 1% of strip cost) and skipping
    /// would require splitting the box3+parent gather, which is more
    /// trouble than the saving.
    fn run_iw_box3_parent_ref_only(&mut self, s: usize) {
        self.run_iw_box3_parent(s);
    }

    /// Upload one strip's slice of the input gray buffers into
    /// scale-0 `g_ref` / `g_dis`. Internal helper for
    /// `compute_gray_stripped`'s two passes.
    fn upload_strip_gray(&mut self, ref_gray: &[f32], dis_gray: &[f32], up_lo: u32, up_hi: u32) {
        let row_stride = self.width as usize;
        let ref_strip: &[f32] =
            &ref_gray[(up_lo as usize) * row_stride..(up_hi as usize) * row_stride];
        let dis_strip: &[f32] =
            &dis_gray[(up_lo as usize) * row_stride..(up_hi as usize) * row_stride];
        self.scales[0].g_ref = self.client.create_from_slice(f32::as_bytes(ref_strip));
        self.scales[0].g_dis = self.client.create_from_slice(f32::as_bytes(dis_strip));
    }

    /// Mutate every Scale's dimension fields (`h`, `w`, `cs_h`,
    /// `cs_w`, `iw_h`, `iw_w`) so subsequent pipeline calls operate
    /// on a strip-sized region. Backing GPU buffers are unchanged —
    /// they were sized for `strip_alloc_h` at construction and the
    /// new dims are guaranteed ≤ allocation.
    fn set_scale_dims_for_strip(&mut self, strip_h: u32, image_w: u32) {
        let mut h = strip_h;
        let mut w = image_w;
        for s in 0..self.scales.len() {
            self.scales[s].h = h;
            self.scales[s].w = w;
            self.scales[s].cs_h = h - 10;
            self.scales[s].cs_w = w - 10;
            self.scales[s].iw_h = h - 2;
            self.scales[s].iw_w = w - 2;
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
    }

    /// Zero one slot's partials region in the device buffer. Used by
    /// the strip path when a scale has no body rows to contribute
    /// (last strip's deep scale collapsing under `body_cs_range`'s
    /// floor + 5-row crop). Without this, the slot would retain the
    /// previous strip's partial sum and double-count it.
    fn zero_partial_slot(&self, slot: u32) {
        let zeros = vec![0.0_f32; reduction::THREADS_PER_REDUCTION as usize];
        // Replace the matching range of `self.partials` by writing
        // zeros into the offset. cubecl's API doesn't expose a
        // sub-range write directly; instead we just allocate a small
        // host buffer and rely on the finalize kernel reading all of
        // `partials` — easiest correct fix is to launch a
        // plain_sum_kernel over a zero-length range (which writes
        // zero into the slot) by passing y_start == y_end. A future
        // change could add a dedicated "zero this slot" kernel.
        let _ = zeros;
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;
        // Use plain_sum with src == cs[0] (dummy) and y_start == y_end == 0.
        // src_w == 0 too — the kernel's `n` becomes 0 so the loop
        // immediately falls through, writing 0 into every thread's
        // partial. We need a real (non-empty) src array to satisfy
        // cubecl's bounds checks. Reuse self.sums as a tiny f32 array.
        reduction::launch_plain_sum::<R>(
            &self.client,
            self.sums.clone(),
            NUM_SLOTS as usize,
            self.partials.clone(),
            partials_len,
            slot,
            1, // src_w = 1 so we don't divide-by-zero in the kernel
            0,
            0,
        );
    }

    // ───────────────────────── helpers ─────────────────────────

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = (n as u32).div_ceil(TPB);
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }
    /// T_x.O (2026-05-17): pack 3 sRGB bytes per pixel into ONE u32
    /// (R | G<<8 | B<<16), writing the packed bytes directly into a
    /// freshly-reserved pinned staging buffer and handing the buffer
    /// to `client.create` for DMA. One host-side pass for the data,
    /// no intermediate `Vec<u32>` allocation.
    ///
    /// Layout is little-endian by construction: the kernel reader
    /// `rgb_u32_to_gray_kernel` sees one u32 per pixel.
    fn pack_into_pinned(client: &ComputeClient<R>, src: &[u8]) -> cubecl::server::Handle {
        debug_assert!(src.len().is_multiple_of(3));
        let n_pixels = src.len() / 3;
        let pinned_len = n_pixels * 4;
        let mut staging = client.reserve_staging(&[pinned_len]);
        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        client.create(bytes)
    }

    fn run_pipeline(&mut self) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let t0 = std::time::Instant::now();
        // 1. Build Gaussian pyramid + extract LP bands.
        self.build_laplacian_pyramid(true);
        self.build_laplacian_pyramid(false);
        if profile {
            cubecl::future::block_on(self.client.sync()).expect("client.sync");
            eprintln!(
                "    stage 'lp_pyramid': {:.3} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        self.run_pipeline_post_pyramid()
    }

    /// Stages 2-6: SSIM stats → IW path → reductions → score. Called
    /// after `lp_ref[s]` and `lp_dis[s]` are populated at every scale
    /// (either by the full `run_pipeline` flow or by the cached
    /// `compute_with_reference` flow).
    fn run_pipeline_post_pyramid(&mut self) -> Result<GpuIwssimResult> {
        let profile = std::env::var("IWSSIM_PROFILE").is_ok();
        let total_t = std::time::Instant::now();

        // Optional per-scale pyramid stats (set `IWSSIM_DEBUG=1`). Kept
        // because the upConv DC scaling is the most failure-prone piece
        // of the port and these prints are the fastest way to catch a
        // regression.
        if std::env::var("IWSSIM_DEBUG").is_ok() {
            self.debug_pyramid_stats();
        }

        // 2. Per-scale SSIM stats + combine.
        let t = std::time::Instant::now();
        for s in 0..self.scales.len() {
            self.run_ssim_stats(s);
        }
        if profile {
            cubecl::future::block_on(self.client.sync()).expect("client.sync");
            eprintln!(
                "    stage 'ssim_stats': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // 3. Per-scale IW path (j = 0..3).
        let t = std::time::Instant::now();
        for s in 0..(self.scales.len() - 1) {
            self.run_iw_scale(s);
            if std::env::var("IWSSIM_DEBUG").is_ok() {
                self.debug_iw_stats(s);
            }
        }
        if profile {
            cubecl::future::block_on(self.client.sync()).expect("client.sync");
            eprintln!(
                "    stage 'iw_scales': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // Debug-only: bypass the IW weighting (treat all weights as 1).
        // Reproduces the reference's `iw_flag=False` mode for sanity-
        // checking the CS path independent of the IW path.
        if std::env::var("IWSSIM_NO_IW").is_ok() {
            for s in 0..(self.scales.len() - 1) {
                let sc = &self.scales[s];
                let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);
                let ones = vec![1.0_f32; n_iw];
                self.scales[s].iw = self.client.create_from_slice(f32::as_bytes(&ones));
            }
        }

        // 4. Reductions per scale.
        // partials and sums are pre-allocated in `new()`. Each
        // weighted_sum / iw_sum / plain_sum thread writes to its own
        // slot (no accumulation), and the finalizer overwrites sums.
        // So nothing needs clearing between calls.
        let partials_len = (NUM_SLOTS * reduction::NUM_BLOCKS * reduction::BLOCK_SIZE) as usize;

        let t = std::time::Instant::now();
        for s in 0..(self.scales.len() - 1) {
            let sc = &self.scales[s];
            let cs_n = (sc.cs_h as usize) * (sc.cs_w as usize);
            let iw_n = (sc.iw_h as usize) * (sc.iw_w as usize);
            reduction::launch_weighted_sum::<R>(
                &self.client,
                sc.cs.clone(),
                cs_n,
                sc.iw.clone(),
                iw_n,
                self.partials.clone(),
                partials_len,
                sc.cs_h,
                sc.cs_w,
                sc.iw_h,
                sc.iw_w,
                BOUND1,
                SLOT_CSIW_BASE + s as u32,
                0,
                sc.cs_h,
            );
            reduction::launch_iw_sum::<R>(
                &self.client,
                sc.iw.clone(),
                iw_n,
                self.partials.clone(),
                partials_len,
                sc.cs_h,
                sc.cs_w,
                sc.iw_h,
                sc.iw_w,
                BOUND1,
                SLOT_IW_BASE + s as u32,
                0,
                sc.cs_h,
            );
        }
        // Top scale: Σ(cs · l) over its native shape.
        let top = self.scales.len() - 1;
        let sc_top = &self.scales[top];
        let cs_top_n = (sc_top.cs_h as usize) * (sc_top.cs_w as usize);
        reduction::launch_plain_sum::<R>(
            &self.client,
            sc_top.cs.clone(),
            cs_top_n,
            self.partials.clone(),
            partials_len,
            SLOT_CSL,
            sc_top.cs_w,
            0,
            sc_top.cs_h,
        );

        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            partials_len,
            self.sums.clone(),
            NUM_SLOTS as usize,
            NUM_SLOTS,
        );
        if profile {
            cubecl::future::block_on(self.client.sync()).expect("client.sync");
            eprintln!(
                "    stage 'reductions': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }

        // 5. Read back and finish on host.
        let t = std::time::Instant::now();
        let bytes = self.client.read_one(self.sums.clone()).expect("read sums");
        let sums = f32::from_bytes(&bytes);
        debug_assert_eq!(sums.len(), NUM_SLOTS as usize);

        // Degenerate-pair handling: on truly identical (ref, dis) pairs,
        // the per-scale IW-weighting collapses — Σ(iw) can be 0 (or
        // non-finite via underflow) when the reference LP signal carries
        // negligible information content at that scale, and the CS map
        // is exactly 1 everywhere by construction (σ_{12} = σ_1² = σ_2²
        // → cs = (2σ + C₂)/(2σ + C₂) = 1). In that regime the Σ(cs·iw)
        // / Σ(iw) ratio is 0/0 and the per-scale wmcs is undefined. The
        // correct IW-SSIM value for an identical pair is 1.0 (every
        // component of the product Π |wmcs_j|^β_j → 1); treat the
        // degenerate slot as 1.0 so the final score lands on 1.0
        // instead of collapsing through 0.0^β = 0 or NaN.
        let mut per_scale = [1.0_f64; NUM_SCALES];
        for s in 0..(self.scales.len() - 1) {
            let num = sums[(SLOT_CSIW_BASE + s as u32) as usize] as f64;
            let den = sums[(SLOT_IW_BASE + s as u32) as usize] as f64;
            // Reference Python:  wmcs[s] = Σ(cs·iw) / Σ(iw)
            per_scale[s] = if den == 0.0 || !den.is_finite() {
                1.0
            } else {
                num / den
            };
        }
        let top_sum = sums[SLOT_CSL as usize] as f64;
        let top_n = (sc_top.cs_h as usize * sc_top.cs_w as usize) as f64;
        per_scale[top] = if top_n == 0.0 || !top_sum.is_finite() {
            1.0
        } else {
            top_sum / top_n
        };

        // 6. Final product: score = Π |wmcs[s]|^β[s]
        let mut score = 1.0_f64;
        for s in 0..self.scales.len() {
            let b = filters::SCALE_WEIGHTS[s] as f64;
            let v = per_scale[s].abs();
            score *= v.powf(b);
        }
        if profile {
            eprintln!(
                "    stage 'readback+score': {:.3} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
            eprintln!(
                "    >>> TOTAL pipeline: {:.3} ms",
                total_t.elapsed().as_secs_f64() * 1e3
            );
        }
        Ok(GpuIwssimResult { score, per_scale })
    }

    /// Build the 5-level Laplacian pyramid for one side (ref or dis).
    /// On entry, `scales[0].g_{ref|dis}` already holds the grayscale
    /// input; on exit, `scales[s].lp_{ref|dis}` holds the LP band at
    /// each scale, and `scales[s].g_{ref|dis}` holds the matching
    /// Gaussian band.
    fn build_laplacian_pyramid(&mut self, is_ref: bool) {
        let n_levels = self.scales.len();

        // 1. Downsample chain: g[s+1] = corrDn_v(corrDn_h(g[s]), binom5).
        for s in 0..(n_levels - 1) {
            let (h_cur, w_cur) = (self.scales[s].h, self.scales[s].w);
            let (h_nxt, w_nxt) = (self.scales[s + 1].h, self.scales[s + 1].w);
            // Reuse the per-scale gh_ref / gh_dis buffer (sized for
            // an `h_cur × (w_cur − 10)` SSIM intermediate, which is
            // strictly larger than `h_cur × w_nxt` we need here — fits).
            let scratch = if is_ref {
                self.scales[s].gh_ref.clone()
            } else {
                self.scales[s].gh_dis.clone()
            };
            let g_cur = if is_ref {
                self.scales[s].g_ref.clone()
            } else {
                self.scales[s].g_dis.clone()
            };
            let g_nxt = if is_ref {
                self.scales[s + 1].g_ref.clone()
            } else {
                self.scales[s + 1].g_dis.clone()
            };
            let n_scratch = (h_cur as usize) * (w_nxt as usize);
            let n_nxt = (h_nxt as usize) * (w_nxt as usize);
            unsafe {
                lap_pyramid::corr_dn_horizontal_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_scratch),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_cur, (h_cur * w_cur) as usize),
                    ArrayArg::from_raw_parts(scratch.clone(), n_scratch),
                    h_cur,
                    w_cur,
                    w_nxt,
                );
                lap_pyramid::corr_dn_vertical_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_nxt),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch, n_scratch),
                    ArrayArg::from_raw_parts(g_nxt, n_nxt),
                    h_nxt,
                    h_cur,
                    w_nxt,
                );
            }
        }

        // 2. LP residual at top scale.
        let top = n_levels - 1;
        {
            let sc = &self.scales[top];
            let n_top = (sc.h as usize) * (sc.w as usize);
            // LP_top = g_top.  Single copy via pointwise_sub against zero
            // doesn't exist; just clone the handle.
            if is_ref {
                self.scales[top].lp_ref = sc.g_ref.clone();
            } else {
                self.scales[top].lp_dis = sc.g_dis.clone();
            }
            let _ = n_top;
        }

        // 3. Upsample chain: LP[s] = g[s] − upConv_v(upConv_h(g[s+1])).
        for s in (0..(n_levels - 1)).rev() {
            let (h_cur, w_cur) = (self.scales[s].h, self.scales[s].w);
            let (_, w_nxt) = (self.scales[s + 1].h, self.scales[s + 1].w);
            let scratch = if is_ref {
                self.scales[s].gh_ref2.clone()
            } else {
                self.scales[s].gh_dis2.clone()
            };
            let g_nxt = if is_ref {
                self.scales[s + 1].g_ref.clone()
            } else {
                self.scales[s + 1].g_dis.clone()
            };
            let g_cur = if is_ref {
                self.scales[s].g_ref.clone()
            } else {
                self.scales[s].g_dis.clone()
            };
            let lp_cur = if is_ref {
                self.scales[s].lp_ref.clone()
            } else {
                self.scales[s].lp_dis.clone()
            };
            let h_nxt = self.scales[s + 1].h;
            let n_h_scratch = (h_nxt as usize) * (w_cur as usize);
            let n_cur = (h_cur as usize) * (w_cur as usize);
            // expanded: insert zeros + binom5 along width, output (h_nxt, w_cur).
            unsafe {
                lap_pyramid::up_conv_horizontal_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_h_scratch),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_nxt, (h_nxt * w_nxt) as usize),
                    ArrayArg::from_raw_parts(scratch.clone(), n_h_scratch),
                    h_nxt,
                    w_nxt,
                    w_cur,
                );
            }
            // Borrow the per-scale parent_band buffer as scratch for
            // the second pass: it's sized for (h, w) at this scale,
            // which is exactly what we need.
            let scratch2 = self.scales[s].parent_band.clone();
            unsafe {
                lap_pyramid::up_conv_vertical_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cur),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(scratch, n_h_scratch),
                    ArrayArg::from_raw_parts(scratch2.clone(), n_cur),
                    h_cur,
                    h_nxt,
                    w_cur,
                );
                lap_pyramid::pointwise_sub_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cur),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(g_cur, n_cur),
                    ArrayArg::from_raw_parts(scratch2, n_cur),
                    ArrayArg::from_raw_parts(lp_cur, n_cur),
                );
            }
        }
    }

    /// SSIM stats at one scale: 11×11 separable Gaussian on
    /// `LP_ref`, `LP_dis`, `LP_ref²`, `LP_dis²`, `LP_ref · LP_dis`,
    /// then combine into `cs` (or `cs · l` at the top scale).
    fn run_ssim_stats(&mut self, s: usize) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let cs_w = sc.cs_w;
        let cs_h = sc.cs_h;
        let n_lp = (h as usize) * (w as usize);
        let n_h = (h as usize) * (cs_w as usize);
        let n_cs = (cs_h as usize) * (cs_w as usize);

        unsafe {
            // Horizontal passes: 5 inputs → 5 hstrip outputs.
            // mu1, mu2 (identity)
            gauss11::gauss11_h_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_ref.clone(), n_h),
                h,
                w,
                cs_w,
            );
            gauss11::gauss11_h_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_dis.clone(), n_h),
                h,
                w,
                cs_w,
            );
            // m11, m22 (squared)
            gauss11::gauss11_h_sq_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_ref2.clone(), n_h),
                h,
                w,
                cs_w,
            );
            gauss11::gauss11_h_sq_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_dis2.clone(), n_h),
                h,
                w,
                cs_w,
            );
            // m12 (product)
            gauss11::gauss11_h_prod_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_h),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.gh_refdis.clone(), n_h),
                h,
                w,
                cs_w,
            );

            // Vertical passes: 5 hstrip inputs → 5 cs-shaped outputs.
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_ref.clone(), n_h),
                ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_dis.clone(), n_h),
                ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_ref2.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_dis2.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );
            gauss11::gauss11_v_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_cs),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.gh_refdis.clone(), n_h),
                ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                cs_h,
                h,
                cs_w,
            );

            // Combine. Top scale: `cs · l`. Others: `cs` only.
            let is_top = s == self.scales.len() - 1;
            if is_top {
                ssim_combine::ssim_cs_l_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cs),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.cs.clone(), n_cs),
                );
            } else {
                ssim_combine::ssim_cs_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_cs),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.mu1.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.mu2.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m11.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m22.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.m12.clone(), n_cs),
                    ArrayArg::from_raw_parts(sc.cs.clone(), n_cs),
                );
            }
        }
    }

    /// Probe pyramid coefficients at every scale — called when
    /// `IWSSIM_DEBUG=1`. Prints G[s] and LP[s] mean/RMS so the upConv
    /// DC scaling can be sanity-checked against the pyrtools reference
    /// (`LP mean` should be ~0 for `s < top`; `LP[top] = G[top]`).
    fn debug_pyramid_stats(&self) {
        for s in 0..self.scales.len() {
            let sc = &self.scales[s];
            let n_lp = (sc.h as usize) * (sc.w as usize);
            let lp_bytes = self.client.read_one(sc.lp_ref.clone()).expect("lp read");
            let lp = f32::from_bytes(&lp_bytes);
            let lp_active = &lp[..n_lp];
            let lp_mean: f64 = lp_active.iter().map(|&v| v as f64).sum::<f64>() / (n_lp as f64);
            let lp_rms = (lp_active
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / n_lp as f64)
                .sqrt();
            let g_bytes = self.client.read_one(sc.g_ref.clone()).expect("g read");
            let g = f32::from_bytes(&g_bytes);
            let g_active = &g[..n_lp];
            let g_mean: f64 = g_active.iter().map(|&v| v as f64).sum::<f64>() / (n_lp as f64);
            let g_rms = (g_active
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum::<f64>()
                / n_lp as f64)
                .sqrt();
            eprintln!(
                "PYR | s={} (h={},w={}) | G mean={:.4} rms={:.4} | LP mean={:.4} rms={:.4}",
                s, sc.h, sc.w, g_mean, g_rms, lp_mean, lp_rms,
            );
        }
    }

    /// Probe iw values for the most recent IW scale — called at the
    /// end of `run_iw_scale` when IWSSIM_DEBUG is set.
    fn debug_iw_stats(&self, s: usize) {
        let sc = &self.scales[s];
        let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);
        let iw_bytes = self.client.read_one(sc.iw.clone()).expect("iw read");
        let iw_arr = f32::from_bytes(&iw_bytes);
        let active = &iw_arr[..n_iw];
        let any_inf = active.iter().any(|v| v.is_infinite());
        let any_nan = active.iter().any(|v| v.is_nan());
        let iw_max = active.iter().fold(f32::NEG_INFINITY, |a, &v| a.max(v));
        let iw_min = active.iter().fold(f32::INFINITY, |a, &v| a.min(v));
        let iw_mean: f64 = active.iter().map(|&v| v as f64).sum::<f64>() / (n_iw as f64);
        eprintln!(
            "scale {} | iw min={:.3e} max={:.3e} mean={:.3e} any_inf={} any_nan={}",
            s, iw_min, iw_max, iw_mean, any_inf, any_nan
        );
    }

    /// Per-scale IW path for scale `s ∈ 0..NUM_SCALES − 1`.
    fn run_iw_scale(&mut self, s: usize) {
        self.run_iw_box3_parent(s);
        self.run_iw_cov_accum(s);
        let (cu_raw, n_dim, has_parent) = self.read_cu_raw(s);
        let nexp = (self.scales[s].iw_h as f64) * (self.scales[s].iw_w as f64);
        let cu_f64 = scale_cu(&cu_raw, n_dim, has_parent, nexp);
        self.eig_and_upload(s, &cu_f64, n_dim, has_parent);
        self.run_iw_infow(s);
    }

    /// Step 1/3 of the IW path: 3×3 box stats + parent band gather.
    /// Leaves `g_buf`, `vv_buf`, and (when `s < NUM_SCALES − 2`)
    /// `parent_band` populated for the cov + infow stages.
    fn run_iw_box3_parent(&mut self, s: usize) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let n_lp = (h as usize) * (w as usize);

        // 1. 3×3 box stats → g, vv at LP shape.
        unsafe {
            box3::box3_gv_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_lp),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.lp_dis.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                w,
                h,
            );
        }

        // 2. Parent band? Present when s < Nsc − 2 (matches Python's
        //    `parent and scale < Nsc − 1` with 1-indexed scale).
        let has_parent = s < self.scales.len() - 2;

        if has_parent {
            // imenlarge2(LP[s+1]) cropped to (h, w).
            let nxt = &self.scales[s + 1];
            unsafe {
                imenlarge2::imenlarge2_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_lp),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(nxt.lp_ref.clone(), (nxt.h * nxt.w) as usize),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    nxt.h,
                    nxt.w,
                    h,
                    w,
                );
            }
        }
    }

    /// Step 2/3: cov accumulation + finalize. Leaves `sc.cu` holding
    /// the raw Σ Yᵀ Y over iw rows `[py_start, py_end)` (NOT yet
    /// divided by `nexp`). Whole-image path passes `(0, iw_h)`;
    /// strip path passes the strip's body iw row range so per-strip
    /// contributions sum cleanly to the global Σ Yᵀ Y without halo
    /// overlap.
    fn run_iw_cov_accum_range(&mut self, s: usize, py_start: u32, py_end: u32) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let n_lp = (h as usize) * (w as usize);

        let has_parent = s < self.scales.len() - 2;
        let n_cells = if has_parent { 100_u32 } else { 81_u32 };
        let cov_partials_len = (COV_MAX_CELLS * COV_N_THREADS) as usize;

        unsafe {
            if has_parent {
                cov::cov_accum_with_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(COV_CUBE_COUNT, 1, 1),
                    CubeDim::new_1d(COV_CUBE_DIM),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    ArrayArg::from_raw_parts(self.cov_partials.clone(), cov_partials_len),
                    h,
                    w,
                    COV_N_THREADS,
                    py_start,
                    py_end,
                );
            } else {
                cov::cov_accum_no_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    CubeCount::Static(COV_CUBE_COUNT, 1, 1),
                    CubeDim::new_1d(COV_CUBE_DIM),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(self.cov_partials.clone(), cov_partials_len),
                    h,
                    w,
                    COV_N_THREADS,
                    py_start,
                    py_end,
                );
            }
            cov::cov_finalize_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(n_cells, 1, 1),
                CubeDim::new_1d(1),
                ArrayArg::from_raw_parts(self.cov_partials.clone(), cov_partials_len),
                ArrayArg::from_raw_parts(sc.cu.clone(), 100),
                COV_N_THREADS,
            );
        }
    }

    /// Whole-image / whole-strip cov accum — sum over the entire iw
    /// shape. Equivalent to `run_iw_cov_accum_range(s, 0, iw_h)`.
    fn run_iw_cov_accum(&mut self, s: usize) {
        let iw_h = self.scales[s].iw_h;
        self.run_iw_cov_accum_range(s, 0, iw_h);
    }

    /// Read the raw Σ Yᵀ Y matrix back to host (no division). Returns
    /// the raw cells (f64 — the cov_finalize kernel accumulates in
    /// f64), `n_dim`, and `has_parent`. Callers divide and
    /// (optionally) accumulate across strips before eigendecomp.
    fn read_cu_raw(&self, s: usize) -> (Vec<f64>, usize, bool) {
        let has_parent = s < self.scales.len() - 2;
        let n_dim = if has_parent { 10 } else { 9 };
        let cu_bytes = self
            .client
            .read_one(self.scales[s].cu.clone())
            .expect("read C_u");
        let cu_f64 = f64::from_bytes(&cu_bytes).to_vec();
        (cu_f64, n_dim, has_parent)
    }

    /// Eigendecompose, invert, and upload C_u_inv + lambdas to the
    /// per-scale device buffers (`cu_inv_dev`, `lambda_dev`). Used by
    /// both the whole-image path (one C_u per scale per call) and the
    /// strip path (one global C_u per scale after summing per-strip
    /// raw contributions).
    fn eig_and_upload(&mut self, s: usize, cu_f64: &[f64], n_dim: usize, has_parent: bool) {
        let _ = has_parent;
        let eig_result = eig::decompose_and_invert(cu_f64, n_dim);
        let lambda_slice = &eig_result.lambda[..n_dim];
        let cu_inv_slice = &eig_result.c_u_inv[..n_dim * n_dim];
        self.scales[s].lambda_dev = self.client.create_from_slice(f32::as_bytes(lambda_slice));
        self.scales[s].cu_inv_dev = self.client.create_from_slice(f32::as_bytes(cu_inv_slice));
    }

    /// Step 3/3: launch the infow kernel using whatever C_u_inv +
    /// lambda are currently uploaded to `self.scales[s].cu_inv_dev`
    /// and `lambda_dev`. Caller is responsible for ensuring those
    /// were populated (via `eig_and_upload`) before invoking.
    fn run_iw_infow(&self, s: usize) {
        let sc = &self.scales[s];
        let h = sc.h;
        let w = sc.w;
        let n_lp = (h as usize) * (w as usize);
        let n_iw = (sc.iw_h as usize) * (sc.iw_w as usize);
        let has_parent = s < self.scales.len() - 2;
        unsafe {
            if has_parent {
                infow::infow_with_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_iw),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.parent_band.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_inv_dev.clone(), 100),
                    ArrayArg::from_raw_parts(sc.lambda_dev.clone(), 10),
                    ArrayArg::from_raw_parts(sc.iw.clone(), n_iw),
                    h,
                    w,
                    0.4_f32,
                );
            } else {
                infow::infow_no_parent_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_iw),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(sc.lp_ref.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.g_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.vv_buf.clone(), n_lp),
                    ArrayArg::from_raw_parts(sc.cu_inv_dev.clone(), 81),
                    ArrayArg::from_raw_parts(sc.lambda_dev.clone(), 9),
                    ArrayArg::from_raw_parts(sc.iw.clone(), n_iw),
                    h,
                    w,
                    0.4_f32,
                );
            }
        }
    }
}

/// Divide the raw Σ Yᵀ Y matrix by `nexp` and pack into a `n_dim × n_dim`
/// f64 matrix in row-major order. Handles the 10×10 vs 9×9 device
/// layout split (the device buffer is always allocated 10×10; for
/// `n_dim == 9` we read only the top-left 9×9 block).
fn scale_cu(cu_raw: &[f64], n_dim: usize, has_parent: bool, nexp: f64) -> Vec<f64> {
    let device_stride = if has_parent { 10 } else { 9 };
    let mut out = vec![0.0_f64; n_dim * n_dim];
    for i in 0..n_dim {
        for j in 0..n_dim {
            let src_idx = i * device_stride + j;
            out[i * n_dim + j] = cu_raw[src_idx] / nexp;
        }
    }
    out
}

/// Host-side BT.601 rgb→gray (rounded) matching
/// `crate::kernels::rgb2gray::rgb_u32_to_gray_kernel`. Used by the
/// strip-mode dispatch in [`Iwssim::compute_rgb`] when the instance
/// was built via [`Iwssim::new_strip`] — the strip walker only takes
/// f32 gray, so we convert on the host before handing off.
pub(crate) fn rgb_u8_to_gray_bt601(rgb: &[u8]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let y = 0.2989_f32 * r + 0.5870_f32 * g + 0.1140_f32 * b;
        out.push((y + 0.5_f32).floor());
    }
    out
}

#[cfg(test)]
mod reflect_pad_tests {
    use super::*;

    #[test]
    fn reflect_index_inside_range_is_identity() {
        for n in 1..=8usize {
            for i in 0..n {
                assert_eq!(reflect_index(i as isize, n), i);
            }
        }
    }

    #[test]
    fn reflect_index_n1_returns_zero() {
        assert_eq!(reflect_index(-3, 1), 0);
        assert_eq!(reflect_index(0, 1), 0);
        assert_eq!(reflect_index(5, 1), 0);
    }

    #[test]
    fn reflect_index_ping_pong_n5() {
        // n=5, period=8, expected sequence around the boundary:
        // ..., 3, 2, 1, [0, 1, 2, 3, 4], 3, 2, 1, 0, 1, 2, ...
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(-3, 5), 3);
        assert_eq!(reflect_index(-4, 5), 4);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(7, 5), 1);
        assert_eq!(reflect_index(8, 5), 0);
        assert_eq!(reflect_index(9, 5), 1);
    }

    #[test]
    fn reflect_pad_f32_identity_when_sizes_match() {
        let src: Vec<f32> = (0..(3 * 4)).map(|i| i as f32).collect();
        let out = reflect_pad_f32(&src, 4, 3, 4, 3);
        assert_eq!(out, src);
    }

    #[test]
    fn reflect_pad_f32_extends_width() {
        // Row major: source 3×2 = [[0,1,2],[3,4,5]], pad to width 5.
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = reflect_pad_f32(&src, 3, 2, 5, 2);
        // Expected per row: [0, 1, 2, reflect(3, 3)=1, reflect(4, 3)=0]
        assert_eq!(out, vec![0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn reflect_pad_f32_extends_height() {
        // Source 3×2; pad to height 4.
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = reflect_pad_f32(&src, 3, 2, 3, 4);
        // Row 0: src row 0.
        // Row 1: src row 1.
        // Row 2: reflect_index(2, 2) = 0 (period=2, 2%2=0, 0<2 → 0)? wait
        //        period=2*(2-1)=2, r=2%2=0, 0<2 → row 0. So row 2 = src row 0.
        // Row 3: reflect_index(3, 2): period=2, r=3%2=1, 1<2 → row 1.
        assert_eq!(
            out,
            vec![
                0.0, 1.0, 2.0, // row 0
                3.0, 4.0, 5.0, // row 1
                0.0, 1.0, 2.0, // row 2 reflected back to src 0
                3.0, 4.0, 5.0, // row 3 reflected back to src 1
            ]
        );
    }

    #[test]
    fn reflect_pad_f32_extends_both_axes() {
        // 2×2 source extended to 4×4.
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let out = reflect_pad_f32(&src, 2, 2, 4, 4);
        // reflect_index(2, 2) = 0; reflect_index(3, 2) = 1.
        // Row 0: 1 2 | reflect col 2 = src col 0 = 1, reflect col 3 = src col 1 = 2.
        // Row 1: 3 4 | 3 4.
        // Row 2: reflect_index row 2 = 0 → row 0 contents.
        // Row 3: reflect_index row 3 = 1 → row 1 contents.
        assert_eq!(
            out,
            vec![
                1.0, 2.0, 1.0, 2.0, // row 0
                3.0, 4.0, 3.0, 4.0, // row 1
                1.0, 2.0, 1.0, 2.0, // row 2
                3.0, 4.0, 3.0, 4.0, // row 3
            ]
        );
    }

    #[test]
    fn reflect_pad_rgb_u8_extends_width() {
        // Source 2 px wide × 1 high RGB: [(10,20,30),(40,50,60)]. Pad to 4 wide.
        let src: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let out = reflect_pad_rgb_u8(&src, 2, 1, 4, 1);
        // Width 2 → period 2, reflect(2,2)=0, reflect(3,2)=1.
        // Row 0: (10,20,30) (40,50,60) | (10,20,30) (40,50,60)
        assert_eq!(out, vec![10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn reflect_pad_rgb_u8_extends_height_3x2() {
        // 2px × 2 rows RGB: row 0 = [(1,2,3),(4,5,6)]; row 1 = [(7,8,9),(10,11,12)].
        // Pad to height 4.
        let src: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let out = reflect_pad_rgb_u8(&src, 2, 2, 2, 4);
        // Same reflect map as f32 test: rows [0,1,0,1].
        assert_eq!(
            out,
            vec![
                1, 2, 3, 4, 5, 6, // row 0
                7, 8, 9, 10, 11, 12, // row 1
                1, 2, 3, 4, 5, 6, // row 2 reflected to 0
                7, 8, 9, 10, 11, 12, // row 3 reflected to 1
            ]
        );
    }

    // ─── tile_pad tests ───

    #[test]
    fn tile_pad_f32_identity_when_sizes_match() {
        let src: Vec<f32> = (0..(3 * 4)).map(|i| i as f32).collect();
        let out = tile_pad_f32(&src, 4, 3, 4, 3);
        assert_eq!(out, src);
    }

    #[test]
    fn tile_pad_f32_extends_width_by_repeat() {
        // Source 3×2 = [[0,1,2],[3,4,5]], pad to width 5.
        // Tile wraps so x=3 → src col 0, x=4 → src col 1.
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = tile_pad_f32(&src, 3, 2, 5, 2);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0, 5.0, 3.0, 4.0]);
    }

    #[test]
    fn tile_pad_f32_extends_height_by_repeat() {
        // Source 3×2 → pad to height 4. Rows 0,1,0,1.
        let src: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = tile_pad_f32(&src, 3, 2, 3, 4);
        assert_eq!(
            out,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn tile_pad_f32_extends_both_axes() {
        // 2×2 source → 5×5 tile. Row mapping: 0,1,0,1,0; col: 0,1,0,1,0.
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let out = tile_pad_f32(&src, 2, 2, 5, 5);
        // Build expected manually.
        let mut expected = Vec::with_capacity(25);
        for dy in 0..5 {
            for dx in 0..5 {
                let sy = dy % 2;
                let sx = dx % 2;
                expected.push(src[sy * 2 + sx]);
            }
        }
        assert_eq!(out, expected);
    }

    #[test]
    fn tile_pad_rgb_u8_extends_width() {
        // 2×1 RGB: [(10,20,30),(40,50,60)] → tile to width 5.
        let src: Vec<u8> = vec![10, 20, 30, 40, 50, 60];
        let out = tile_pad_rgb_u8(&src, 2, 1, 5, 1);
        // Cols 0..5: (10,20,30) (40,50,60) (10,20,30) (40,50,60) (10,20,30)
        assert_eq!(
            out,
            vec![10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60, 10, 20, 30]
        );
    }
}
