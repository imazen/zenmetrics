//! cvvdp pipeline orchestration.
//!
//! Wires the kernels in [`crate::kernels`] into a still-image
//! ColorVideoVDP scorer.
//!
//! Public entry points:
//! - [`Cvvdp::new`] + [`Cvvdp::score`] — one-shot scoring of a
//!   (reference, distorted) pair.
//! - [`Cvvdp::set_reference`] + [`Cvvdp::score_with_reference`] —
//!   reference-side cache for encoder loops that compare many
//!   candidates to the same source.
//!
//! ## Algorithm overview (per call)
//!
//! 1. Upload sRGB-u8 bytes for both sides (or skip reference side
//!    when cached).
//! 2. Run `color::srgb_to_dkl_kernel` once per side → 3 planar DKL
//!    buffers each (achromatic + RG + VY).
//! 3. Build per-channel Weber-contrast pyramids: each pyramid level
//!    runs `pyramid::downscale_kernel` (Gaussian reduce) then
//!    `pyramid::upscale_v_kernel` + `pyramid::upscale_h_kernel`
//!    (separable expand), followed by the fused
//!    `pyramid::subtract_weber_3ch_kernel` that emits all three
//!    channel bands plus the shared `log10(L_bkg)` from one launch.
//!    Yields `n_levels` Weber-contrast bands per channel per side
//!    plus a per-pixel `log10(L_bkg)` map for step 4. The coarsest
//!    band (the gaussian base) bypasses Weber contrast and feeds
//!    directly into step 5's baseband bypass.
//! 4. Per-pixel CSF apply via `csf::csf_apply_6ch_kernel` — a single
//!    launch per non-baseband level runs CSF for both REF and DIST
//!    sides (the per-pixel LUT bracket math is shared). Per-band
//!    `rho` resolved via `csf::precompute_logs_row`. Output `T_p` =
//!    Weber × S(rho, L_bkg, channel) × CH_GAIN.
//! 5. Multi-channel mult-mutual masking:
//!    - Non-baseband bands: `masking::min_abs_3ch_kernel` →
//!      `masking::pu_blur_h_3ch_kernel` →
//!      `masking::pu_blur_v_3ch_scaled_kernel` (folds the
//!      `* 10^MASK_C` post-scale) →
//!      `masking::mult_mutual_3ch_with_blurred_kernel`. Falls back
//!      to `masking::mult_mutual_3ch_no_blur_kernel` when either
//!      band dimension is ≤ `PU_PADSIZE`.
//!    - Baseband: `masking::diff_abs_3ch_kernel` writes
//!      `|T_p_dis - T_p_ref|` (cvvdp's baseband bypass).
//! 6. Per-band Minkowski accumulation
//!    (`pool::pool_band_3ch_kernel`, fused 3-channel launch per
//!    level) → per-band f32 partials (one `f32` per (level,
//!    channel) in a shared GPU buffer).
//! 7. Host-side fold: read back the `n_levels × 3` partials Vec,
//!    `pool_band_finalize` per (level, channel), then the 3-stage
//!    Minkowski pool + `pool::met2jod` piecewise.
//!
//! ## Buffer layout
//!
//! Storage is GPU-resident and pre-allocated by `Cvvdp::new` for a
//! fixed `(width, height)`. No SIMD-pad columns (cvvdp's reference
//! doesn't pad).
//!
//! - `bands_ref: Vec<Level>` — one `Handle` per (channel, level)
//!   for the Weber-contrast bands. Content alternates between REF
//!   and DIST data inside `compute_dkl_d_bands`: the per-side weber
//!   pyramid dispatch overwrites it, and the band loop reads
//!   directly from these handles for the downstream CSF apply.
//! - `gauss_ref: Vec<Level>` — Gaussian pyramid handles per
//!   (channel, level), used by the weber pyramid dispatch to build
//!   each non-baseband level. Shared between REF/DIST passes the
//!   same way `bands_ref` is.
//! - `weber_scratch[k]` (per non-baseband level) — the
//!   `l_bkg_fine`, `vscratch_a`, `log_l_bkg`, plus per-channel
//!   `vscratch_c` / `upscaled_c` handles consumed by the fused
//!   `subtract_weber_3ch_kernel`.
//! - `d_scratch[k]` (per level, including baseband) — the final
//!   `d` output handles, per-channel. Every level's D plane lives in
//!   `d_scratch[k].d[c]` regardless of whether the band ran through
//!   the masker (`mult_mutual_3ch_*`) or the baseband bypass
//!   (`diff_abs_3ch`). The per-band transient intermediates
//!   (`t_p_ref`, `t_p_dis`, `m_raw`, `m_mid`, `m_blur`) are
//!   allocated lazily inside the band loop as a
//!   [`DBandsTransient`] and dropped at the end of each band
//!   iteration — only the current band's intermediates are GPU-
//!   resident at any time, vs the previous all-levels-up-front
//!   allocation (tick "Path B lazy-transient", 2026-05-26).
//! - `logs_row[k][c]` — pre-uploaded 32-entry CSF sensitivity LUT
//!   row per (level, channel); stable across calls since `rho_k`
//!   is fixed for this Cvvdp.
//!
//! Total per-Cvvdp budget at 4000×3000 with 8 pyramid levels is
//! a few hundred MB of GPU memory. The `DBandsTransient`
//! contribution is one band at a time (largest = base level) plus
//! the persistent `d` output planes; the previous all-levels-up-
//! front allocation of `t_p_*` + `m_*` was eliminated in the lazy-
//! transient refactor. The remaining persistent allocations happen
//! once in `Cvvdp::new`; the hot path does only `create_from_slice`
//! for input bytes and reads back small results.

use cubecl::prelude::*;

use crate::kernels::color::{srgb_to_dkl_kernel, SRGB8_TO_LINEAR_LUT};
use crate::kernels::csf::{
    csf_apply_3ch_kernel, csf_apply_6ch_kernel, flatten_band_weights, precompute_logs_row,
    precomputed_band_weights, weight_band_kernel, CsfChannel,
};
use crate::kernels::diffmap::{
    diffmap_band_accumulate_kernel, diffmap_channel_pool_kernel, diffmap_zero_kernel,
    linear_rgb_planes_to_dkl_kernel,
};
use crate::kernels::masking::{
    diff_abs_3ch_kernel, min_abs_3ch_kernel, mult_mutual_3ch_no_blur_kernel,
    mult_mutual_3ch_with_blurred_kernel, pu_blur_h_3ch_kernel, pu_blur_h_3ch_strip_aware_kernel,
    pu_blur_v_3ch_scaled_kernel, pu_blur_v_3ch_scaled_strip_aware_kernel, CH_GAIN, MASK_C,
    PU_PADSIZE,
};
use crate::kernels::pool::{
    copy_f32_kernel, do_pooling_and_jod_still_3ch, fill_f32_kernel, lp_norm_mean,
    pool_band_3ch_kernel, pool_band_3ch_lds_kernel, pool_band_3ch_offset_kernel,
    pool_band_finalize, BASEBAND_W, BETA_CH, BETA_SPATIAL, PER_CH_W, POOL_LDS_BLOCK_DIM,
};
use crate::kernels::pyramid::{
    band_frequencies, baseband_divide_3ch_kernel, downscale_strip_kernel,
    downscale_tiled_kernel, subtract_kernel, subtract_weber_3ch_kernel,
    subtract_weber_3ch_strip_kernel, upscale_h_kernel, upscale_h_strip_kernel, upscale_v_kernel,
    upscale_v_strip_kernel, DOWNSCALE_TILED_BLOCK_DIM,
};
use crate::params::CvvdpParams;
use crate::{Error, Result, MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM};

/// Return shape of [`Cvvdp::compute_dkl_weber_pyramid`].
///
/// - `.0` — `levels[k] = [a, rg, vy]` Weber-contrast bands. Same
///   layout as `compute_dkl_laplacian_pyramid`'s output.
/// - `.1` — `levels[k]` per-pixel `log10(L_bkg)` plane for non-
///   baseband levels, replicated scalar for the baseband. Matches
///   `host_scalar::WeberPyramid::log_l_bkg`.
pub type WeberPyramidGpu = (Vec<[Vec<f32>; 3]>, Vec<Vec<f32>>);

/// One pyramid level: a `width × height` planar f32 buffer per channel.
struct Level {
    w: u32,
    h: u32,
    /// One f32 plane per DKL channel.
    planes: [cubecl::server::Handle; N_CHANNELS],
}

/// Per-level persistent output buffer reused by
/// `compute_dkl_d_bands`. Only the final `d` (masked-difference)
/// planes need to survive past their band's masking dispatch — the
/// pool / diffmap / `compute_dkl_d_bands` consumers all read
/// `d_scratch[k].d[c]` after the band loop completes.
///
/// The per-band transient buffers (`t_p_ref`, `t_p_dis`, `m_raw`,
/// `m_mid`, `m_blur`) used to live on this struct too — they were
/// allocated for every level up front, peaking at 15/18 of the
/// d_scratch volume even though only one band's worth was ever
/// live at a time. The lazy-transient refactor moved them into
/// [`DBandsTransient`], which is allocated inside the band loop
/// and dropped at the end of each iteration.
struct DBandsScratch {
    /// Per-band masked-difference output (consumed by host
    /// `lp_norm_mean` after read-back, or by the GPU pool /
    /// diffmap dispatch). Stays GPU-resident for every level so
    /// the post-band-loop pool stage can read all bands' D planes.
    ///
    /// **Path A Phase 1d (2026-05-26):** `None` for non-baseband
    /// levels in `StripMode::Pair` (Mode B), where
    /// [`Self::d_strip`] owns one strip's worth of d per
    /// (level, strip) and the pool is dispatched **inline** by
    /// [`Cvvdp::_run_band_masking_strip_walker`] before the next
    /// strip overwrites the buffer. The baseband level still
    /// allocates `d` in Mode B because the baseband bypasses the
    /// strip walker (it uses the full-band `diff_abs_3ch_kernel`),
    /// and at the deepest level the band is small enough that
    /// per-strip shrinking has no value.
    ///
    /// Full and CachedRef modes (Mode E) keep `Some(...)` at every
    /// level — Mode E's post-band-loop pool reads strip-by-strip
    /// from the full `d` plane.
    d: Option<[cubecl::server::Handle; N_CHANNELS]>,
    /// Per-strip d buffer (Mode B only, non-baseband levels). Sized
    /// `(bw_k × strip_h_at_k × f32)` per channel — `strip_h_at_k =
    /// (h_body >> k).max(1)`. The masking strip walker writes one
    /// strip's d into this buffer at offset 0, then immediately
    /// dispatches the pool kernel to accumulate the strip's
    /// contribution into `partials_h` before the next strip
    /// overwrites this buffer.
    ///
    /// `None` outside `StripMode::Pair`, and `None` for the
    /// baseband level even in Mode B (see [`Self::d`]).
    d_strip: Option<[cubecl::server::Handle; N_CHANNELS]>,
}

/// Per-band transient scratch for the CSF + masking chain, allocated
/// lazily at the top of each band-loop iteration in
/// [`Cvvdp::_run_d_bands_band_loop`] and dropped at the bottom.
/// Holding only one band's worth of these buffers (vs the previous
/// all-levels-up-front allocation) is the bulk of the Path-B memory
/// reduction: at 12 MP these are ~120 MB per level × 8 levels (the
/// finest band alone is ~60 MB), and dropping after the masking
/// stage lets cubecl's memory pool recycle those pages for the next
/// band's allocation.
struct DBandsTransient {
    /// CSF-applied bands per channel for ref and dist sides.
    /// `compute_dkl_d_bands` runs `csf_apply_per_pixel_kernel` into
    /// these (one launch per side per channel).
    t_p_ref: [cubecl::server::Handle; N_CHANNELS],
    t_p_dis: [cubecl::server::Handle; N_CHANNELS],
    /// Masking-chain scratch (non-baseband levels only). Allocated
    /// unconditionally — baseband bypasses the masking kernels but
    /// keeping the alloc unconditional keeps the band-loop control
    /// flow uniform across levels. At the baseband size these are
    /// tiny vs the finest band.
    m_raw: [cubecl::server::Handle; N_CHANNELS],
    m_mid: [cubecl::server::Handle; N_CHANNELS],
    m_blur: [cubecl::server::Handle; N_CHANNELS],
}

impl DBandsTransient {
    /// Allocate the per-band CSF + masking transients for a band of
    /// `n` pixels. cubecl's memory pool will recycle pages on Drop,
    /// so the per-iter alloc cost is the pool's bookkeeping, not a
    /// full GPU buffer create.
    fn new<R: Runtime>(client: &ComputeClient<R>, n: usize) -> Self {
        Self {
            t_p_ref: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            t_p_dis: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_raw: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_mid: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
            m_blur: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
        }
    }

    /// **P2.4 (2026-05-27):** allocate strip-shaped t_p_* + m_*
    /// transients sized at `n_strip = bw × R_k` per channel. Used by
    /// the strip-major outer band loop for shallow non-baseband
    /// levels (`k < k_split`) — each strip iteration overwrites the
    /// buffer in place before the masking chain consumes it within
    /// the SAME (s, k) iteration.
    ///
    /// Caller must pass `body_off_kernel = top_global` to the
    /// strip-aware masking kernels AND skip the `offset_start`
    /// byte slices (the buffer's row 0 IS the strip window's row 0
    /// at dispatch time). See `_run_band_masking_strip_s_for_level`'s
    /// `transients_strip_local: bool` parameter.
    fn new_strip<R: Runtime>(client: &ComputeClient<R>, n_strip: usize) -> Self {
        // Same shape as `new`, just sized at strip pixels instead of
        // full-band pixels.
        Self::new(client, n_strip)
    }
}

fn alloc_zeros_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// Per-channel CSF gain for a pyramid level. Non-baseband bands get
/// `band_mul * CH_GAIN[c]`; the baseband bypasses `CH_GAIN` so the
/// downstream `|T_p_dis - T_p_ref|` subtraction reproduces cvvdp's
/// `apply_masking_model` baseband formula exactly.
fn ch_gain_for_band(is_baseband: bool, band_mul: f32) -> [f32; N_CHANNELS] {
    if is_baseband {
        [1.0, 1.0, 1.0]
    } else {
        [
            band_mul * CH_GAIN[0],
            band_mul * CH_GAIN[1],
            band_mul * CH_GAIN[2],
        ]
    }
}

/// Per-level scratch buffers reused by `compute_dkl_weber_pyramid`.
/// At 12 MP the function would otherwise allocate ~140 MB of
/// transient GPU buffers per call (l_bkg_fine, vscratch_a, log_l_bkg
/// per level + vscratch_c/upscaled_c per (level, channel)).
/// Called twice per `compute_dkl_d_bands` so doubled per d_bands call.
///
/// `coarse_w * fine_h` shape (n_v) is the vertical-pass scratch for
/// upscale_v_kernel; `fine_w * fine_h` shape (n_fine) is everything
/// else.
struct WeberScratch {
    /// Expanded achromatic L_bkg, shared across channels (n_fine).
    ///
    /// **P2.6 (2026-05-27):** in `StripMode::Pair` for shallow levels
    /// (`k < k_split`), this is allocated at `fine_w × R_k` instead of
    /// `n_fine = fine_w × fine_h`. Each (s, k) iteration overwrites
    /// l_bkg_fine in place (REF stage 1 then DIST stage 1); REF and
    /// DIST stage 3 each read the value they just wrote. No
    /// cross-strip data dependency. Deep levels keep `n_fine`.
    l_bkg_fine: cubecl::server::Handle,
    /// Vertical-pass scratch for achromatic L_bkg expand (n_v).
    ///
    /// **P2.6 (2026-05-27):** in `StripMode::Pair` for shallow levels,
    /// allocated at `coarse_w × R_k` instead of `coarse_w × fine_h`.
    /// Same strip-local overwrite-per-(s,k) pattern as `l_bkg_fine`.
    vscratch_a: cubecl::server::Handle,
    /// Per-pixel log10(L_bkg) plane for the REF side.
    ///
    /// **P2.6 (2026-05-27):** in `StripMode::Pair` for shallow levels
    /// (`k < k_split`), sized `fine_w × R_k`. The REF strip helper
    /// writes it body+halo; the CSF helper reads it body+halo, both
    /// within the same `(s, k)` iteration. The next strip overwrites
    /// it. Deep levels (`k >= k_split`) keep `n_fine` because the
    /// level-major dispatch path reads them at full level dims.
    log_l_bkg: cubecl::server::Handle,
    /// Throwaway destination for DIST's log_l_bkg write — same
    /// shape as `log_l_bkg`. cvvdp's weber_g1 rule uses REF's
    /// log_l_bkg for both sides, so DIST's value is computed but
    /// discarded; this lets the DIST dispatch write somewhere
    /// without clobbering REF.
    ///
    /// **P2.6 (2026-05-27):** same shallow/deep sizing rule as
    /// `log_l_bkg`. The dispatch writes here but nothing downstream
    /// reads it — keeping it strip-sized just saves wasted memory.
    log_l_bkg_dis: cubecl::server::Handle,
    /// Per-channel vertical/horizontal expand scratch (n_v, n_fine).
    /// The previous `layer_c` intermediate is gone — tick 91 fuses
    /// `subtract + weber` into a single 3-channel kernel that reads
    /// `fine` + `upscaled_c` directly.
    vscratch_c: [cubecl::server::Handle; N_CHANNELS],
    /// Per-channel separable upscale destination at full-image
    /// resolution `(fine_w × fine_h × f32)`. Written by the Full /
    /// CachedRef Weber-finalize path's `upscale_h_kernel` and read by
    /// `subtract_weber_3ch_kernel` on the same level.
    ///
    /// **Path A Phase 1c (2026-05-26):** `None` in `StripMode::Pair`,
    /// where the Mode B walker writes to the strip-sized
    /// [`Self::upscaled_c_strip`] instead. Skipping the allocation
    /// here is where the actual memory shrink lives — the strip buf
    /// alone would have been *additional* memory, not less. Full and
    /// CachedRef modes still allocate this (their walkers index it
    /// every level / strip).
    upscaled_c: Option<[cubecl::server::Handle; N_CHANNELS]>,
    /// Per-strip upscaled scratch (Phase 1b origin; **P2.1b 2026-05-27
    /// resizes to back-projected R_k**). Allocated only when the
    /// owning `Cvvdp` instance is in `StripMode::Pair`.
    ///
    /// **Sizing** (P2.1b): per channel, `fine_w × R_k × f32` where
    /// `R_k = mode_b_strip_h_at_level(k, h_body, k_split).min(fine_h)`
    /// for shallow levels (k < k_split), or `(h_body >> k).max(1).min(fine_h)`
    /// for deep levels (k ≥ k_split). The back-projection ensures the
    /// buffer can hold body+halo of `t_p_*[k]` per the level-k CSF
    /// dispatch window — even with the level-major-outer caller still
    /// active, the per-strip walker writes body+halo redundantly and
    /// the level-(k-1) reduce can read from this strip's halo without
    /// looking at sibling strip data.
    ///
    /// The Weber-finalize strip walker writes the strip's upscale
    /// output here over a body+halo window, then
    /// `subtract_weber_3ch_strip_kernel` reads it with
    /// `src_strip_offset = top_global` (P2.1b semantics — buffer row
    /// 0 corresponds to the halo top, not the body top).
    ///
    /// `None` outside `StripMode::Pair` so the Full and CachedRef
    /// dispatch paths keep using the full-image [`Self::upscaled_c`]
    /// without paying the Phase 1b allocation.
    upscaled_c_strip: Option<[cubecl::server::Handle; N_CHANNELS]>,
    /// Per-strip bands_dis scratch (Phase 1b origin; **P2.1b 2026-05-27
    /// resizes to back-projected R_k**). Allocated only when the
    /// owning `Cvvdp` instance is in `StripMode::Pair`.
    ///
    /// **Sizing** (P2.1b): per channel, `fine_w × R_k × f32` — identical
    /// to [`Self::upscaled_c_strip`]. See that field's docstring for
    /// the R_k derivation. The buffer holds the body+halo window of
    /// `bands_dis[k]` for one strip; subsequent strips overwrite it
    /// before the next CSF dispatch reads it.
    ///
    /// **Reader/writer wiring (P2.1b):** the per-strip CSF walker
    /// writes this buffer over a body+halo window via
    /// `subtract_weber_3ch_strip_kernel` (stage 3) and reads it back
    /// in the same window via `csf_apply_6ch_kernel` (stage 4). The
    /// full-image `bands_dis[k].planes` remain zero-sized in Mode B
    /// (skipped at construction).
    ///
    /// `None` outside `StripMode::Pair` so the Full and CachedRef
    /// dispatch paths skip the allocation entirely.
    bands_dis_strip: Option<[cubecl::server::Handle; N_CHANNELS]>,
    /// Per-strip bands_ref scratch (**P2.3, 2026-05-27**). Allocated only
    /// when the owning `Cvvdp` instance is in `StripMode::Pair` AND this
    /// level is shallow (`k < k_split`). Mirrors [`Self::bands_dis_strip`]
    /// for the REF-side weber-pyramid output.
    ///
    /// **Sizing** (P2.3): per channel, `fine_w × R_k × f32` — identical
    /// to [`Self::bands_dis_strip`]. Holds the body+halo window of
    /// `bands_ref[k]` for one strip; subsequent strips overwrite it
    /// before the next CSF dispatch reads it.
    ///
    /// **Reader/writer wiring (P2.3):** the per-strip REF helper
    /// (analog to the per-strip DIST helper) writes this buffer in the
    /// strip-major-outer band loop, and the per-strip CSF helper
    /// reads it in the same `(s, k)` iteration before the next strip
    /// overwrites it. The full-image `bands_ref[k].planes` are zero-
    /// sized in Mode B for shallow non-baseband levels (skipped at
    /// construction); deep levels (`k >= k_split`) and the baseband
    /// keep their full-image allocation because the level-major-outer
    /// dispatch path reads them.
    ///
    /// `None` outside `StripMode::Pair`, and `None` for deep
    /// non-baseband levels even in Mode B (those keep the full-image
    /// `bands_ref[k]` allocation; the strip-major outer only iterates
    /// shallow levels).
    bands_ref_strip: Option<[cubecl::server::Handle; N_CHANNELS]>,
}


/// GPU scratch for the per-pixel diffmap pipeline. Allocated lazily on
/// the first `score_with_diffmap` / `score_from_linear_planes_with_diffmap`
/// call — callers that never request a diffmap pay zero memory.
///
/// `acc[c]` are base-resolution `W × H` accumulator planes (one per
/// DKL channel) that the per-band upsample step writes into;
/// `out` is the per-pixel diffmap result the channel-pool step
/// fills before host readback.
///
/// Memory cost: `4 * W * H * 4 bytes = 16 * W * H bytes`
/// (~50 MB at 12 MP). Allocated once and reused across calls; the
/// pool / channel-pool dispatch zeros + overwrites it per call.
struct DiffmapScratch {
    acc: [cubecl::server::Handle; N_CHANNELS],
    out: cubecl::server::Handle,
}

/// Three planar f32 buffers (R, G, B) reused across
/// `score_from_linear_planes*` calls to avoid per-iter
/// `client.create_from_slice` allocations. Layout matches what
/// `linear_rgb_planes_to_dkl_kernel` expects: tightly-packed
/// row-major `W × H` linear-light unit-scaled sRGB primaries.
///
/// Allocated lazily on the first `from_linear_planes` call so
/// callers that only use sRGB-byte inputs don't pay the 12 MB
/// (per side) at 1 MP / 144 MB at 12 MP cost.
struct LinearPlanesUpload {
    /// `W × H` linear-RGB upload buffers. Reused across REF and DIST
    /// dispatches — each call uploads fresh bytes before the kernel
    /// launch, so a single triple is enough.
    planes: [cubecl::server::Handle; N_CHANNELS],
}

fn build_weber_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
    strip_pair_h_body: Option<u32>,
) -> Vec<WeberScratch> {
    let mut out = Vec::with_capacity(n_levels.saturating_sub(1));
    let mut fine_w = width;
    let mut fine_h = height;
    // **P2.1b (2026-05-27):** in StripMode::Pair the per-strip
    // `upscaled_c_strip` / `bands_dis_strip` buffers are sized at
    // **back-projected `R_k`** rows, not body-only. This is the
    // required buffer growth that lets the per-strip CSF walker
    // dispatch over body+halo of `t_p_*[k]` so subsequent masking
    // strip-major-outer dispatch (P2.1c) can read its own halo from
    // its own CSF dispatch instead of relying on sibling strips'
    // contributions to a full-image `bands_dis`. `mode_b_strip_h_at_level`
    // returns `body + 2·halo_band` clamped against the cross-level
    // reduce chain — see its docstring for the derivation. The level-k
    // image height clamp is necessary because near-degenerate
    // (h_body, image_h) combos can have R_k > fine_h.
    //
    // Outside StripMode::Pair both fields are `None` — Full / CachedRef
    // paths read from full-image `upscaled_c` / `bands_dis` and the
    // strip allocation is irrelevant.
    let k_split = strip_pair_h_body
        .map(|hb| mode_b_k_split(hb, n_levels as u32))
        .unwrap_or(0);
    // Only non-baseband levels need scratch (baseband bypasses the
    // expand/subtract/weber chain).
    for k in 0..n_levels.saturating_sub(1) {
        // Ceil-div halving — matches cvvdp's `gausspyr_reduce`
        // boundary semantics so the GPU pyramid stays bit-stable
        // against the host scalar reference at all sizes (not just
        // even-dim corpora). See `gausspyr_reduce_scalar` in
        // kernels/pyramid.rs (which already uses div_ceil(2)).
        let coarse_w = fine_w.div_ceil(2);
        let coarse_h = fine_h.div_ceil(2);
        let n_fine = (fine_w as usize) * (fine_h as usize);
        let n_v = (coarse_w as usize) * (fine_h as usize);

        // P2.1b back-projected strip height. Falls back to
        // `min(body_only, fine_h)` for deep levels (k >= k_split) where
        // `mode_b_strip_h_at_level` returns 0 — deep levels stay
        // level-major so body-only sizing is sufficient. For shallow
        // levels (k < k_split), R_k is the body+2·halo back-projected
        // through the reduce chain.
        let strip_h_at_k = strip_pair_h_body.map(|hb| {
            let r_k_back = mode_b_strip_h_at_level(k as u32, hb, k_split);
            if r_k_back == 0 {
                // Deep level — use body-only (halved per level, clamped to 1).
                (hb >> k).max(1).min(fine_h)
            } else {
                // Shallow level — back-projected R_k, clamped to image height.
                r_k_back.min(fine_h)
            }
        });

        // P2.1b: in StripMode::Pair allocate a per-strip-sized
        // upscale destination at `fine_w × R_k`. The per-strip CSF
        // walker writes body+halo rows (using src_strip_offset =
        // top_global to anchor the buffer's row 0 at the halo top).
        // None outside StripMode::Pair so the Full / CachedRef paths
        // skip the allocation entirely (they read from upscaled_c
        // which IS full-image-sized).
        let upscaled_c_strip = strip_h_at_k.map(|h_strip| {
            let n_strip = (fine_w as usize) * (h_strip as usize);
            [
                alloc_zeros_f32(client, n_strip),
                alloc_zeros_f32(client, n_strip),
                alloc_zeros_f32(client, n_strip),
            ]
        });

        // P2.1b bands_dis_strip: same R_k strip-buffer shape as
        // upscaled_c_strip — fine_w × R_k per channel. Allocated only
        // in StripMode::Pair; the fused csf-in-walker path reads it
        // directly with strip-local indexing over the body+halo
        // window so the per-strip CSF dispatch can write t_p_*[k]
        // body+halo rows (the level-major-outer caller does redundant
        // halo writes per strip — deterministic, JOD bit-identical;
        // P2.1c flips the outer loop to strip-major and the halo
        // becomes the load-bearing read path).
        let bands_dis_strip = strip_h_at_k.map(|h_strip| {
            let n_strip = (fine_w as usize) * (h_strip as usize);
            [
                alloc_zeros_f32(client, n_strip),
                alloc_zeros_f32(client, n_strip),
                alloc_zeros_f32(client, n_strip),
            ]
        });

        // P2.3 bands_ref_strip (2026-05-27): allocate ONLY for shallow
        // levels (k < k_split) where the strip-major-outer band loop
        // reads bands_ref from the strip buffer. Deep levels keep
        // bands_ref full-image (the level-major-outer dispatch path
        // reads them at full level dims). Outside Mode B both fields
        // stay `None`. Sizing matches `bands_dis_strip` exactly so the
        // CSF helper can swap the read source without launch-geometry
        // changes.
        let bands_ref_strip = strip_h_at_k.and_then(|h_strip| {
            if (k as u32) < k_split {
                let n_strip = (fine_w as usize) * (h_strip as usize);
                Some([
                    alloc_zeros_f32(client, n_strip),
                    alloc_zeros_f32(client, n_strip),
                    alloc_zeros_f32(client, n_strip),
                ])
            } else {
                None
            }
        });

        // Path A Phase 1c: skip the full-image `upscaled_c` allocation
        // when the strip variant is live. Mode B's Weber-finalize
        // walker writes to `upscaled_c_strip` exclusively (verified by
        // the strip-walker reads in `_finalize_weber_pyramid_strip_walker`).
        // The Full / CachedRef path still allocates the full buffer —
        // those code paths index `upscaled_c[c]` every level. With the
        // byte-upload `_dispatch_weber_pyramid_gpu` now routed through
        // `_finalize_weber_pyramid_after_gauss`, Mode B byte callers
        // hit the strip walker too, so the full alloc is genuinely
        // unused under StripMode::Pair.
        let upscaled_c = if strip_h_at_k.is_some() {
            None
        } else {
            Some([
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
            ])
        };

        // P2.6 (2026-05-27): in StripMode::Pair for shallow levels
        // (k < k_split), shrink l_bkg_fine / log_l_bkg / log_l_bkg_dis
        // to `fine_w × R_k` instead of `n_fine`. Each (s, k) iteration
        // overwrites them in place; no cross-strip read. Same shrink
        // logic for vscratch_a / vscratch_c (`coarse_w × R_k`).
        //
        // Deep levels (k >= k_split) keep full-image sizing because
        // the level-major dispatch path reads them at full dims.
        let p26_shallow = strip_h_at_k.is_some() && (k as u32) < k_split;
        let (n_fine_alloc, n_v_alloc) = if p26_shallow {
            let h_strip = strip_h_at_k.unwrap() as usize;
            (
                (fine_w as usize) * h_strip,
                (coarse_w as usize) * h_strip,
            )
        } else {
            (n_fine, n_v)
        };
        out.push(WeberScratch {
            l_bkg_fine: alloc_zeros_f32(client, n_fine_alloc),
            vscratch_a: alloc_zeros_f32(client, n_v_alloc),
            log_l_bkg: alloc_zeros_f32(client, n_fine_alloc),
            log_l_bkg_dis: alloc_zeros_f32(client, n_fine_alloc),
            vscratch_c: [
                alloc_zeros_f32(client, n_v_alloc),
                alloc_zeros_f32(client, n_v_alloc),
                alloc_zeros_f32(client, n_v_alloc),
            ],
            upscaled_c,
            upscaled_c_strip,
            bands_dis_strip,
            bands_ref_strip,
        });
        fine_w = coarse_w;
        fine_h = coarse_h;
        // P2.1b (2026-05-27): strip_h_at_k is now derived per-level
        // from `mode_b_strip_h_at_level(k, h_body, k_split)` inside the
        // loop, so the prior `strip_h_at_k = strip_h_at_k.map(|hb|
        // (hb >> 1).max(1));` cascade is unused. Removed to keep the
        // sizing model single-sourced from the back-projection helper.
    }
    out
}

fn build_d_bands_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
    strip_pair_h_body: Option<u32>,
) -> Vec<DBandsScratch> {
    // After the lazy-transient refactor, this only allocates the
    // persistent `d` output planes per level. The 5 transient
    // buffer kinds (`t_p_ref`, `t_p_dis`, `m_raw`, `m_mid`,
    // `m_blur`) are now allocated inside the band loop via
    // [`DBandsTransient::new`] and dropped at the end of each band.
    //
    // **Path A Phase 1d (2026-05-26):** under `StripMode::Pair`
    // (Mode B) the non-baseband levels skip the full-image `d`
    // allocation and instead allocate `d_strip` sized per-strip
    // (`bw_k × strip_h_at_k`). The masking strip walker writes one
    // strip's d into the strip buffer at offset 0 and dispatches the
    // pool kernel inline before the next strip overwrites it. The
    // baseband level retains full-image `d` because its diff_abs
    // dispatch bypasses the strip walker (and the band is small).
    let mut out = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    let mut strip_h_at_k = strip_pair_h_body;
    for k in 0..n_levels {
        let n = (w as usize) * (h as usize);
        let is_baseband = k == n_levels - 1;
        // The masking strip walker writes `bw × min(strip_h_at_k, bh)`
        // pixels per dispatch when `use_blur` (large bands). For small
        // bands (`bw ≤ PU_PADSIZE || bh ≤ PU_PADSIZE`) the band loop
        // takes the no-blur path which writes `n_px = bw × bh` in a
        // single dispatch — Mode B's strip buffer must cover that
        // worst case at deep levels. Most savings come from shallow
        // levels where `bw × strip_h_at_k ≪ n_px`.
        let use_blur = w as usize > PU_PADSIZE && h as usize > PU_PADSIZE;
        // Mode B non-baseband: per-strip d_strip, no full d.
        // Mode B baseband: full d (small), no d_strip.
        // Other modes: full d at every level, no d_strip.
        let (d, d_strip) = match (strip_h_at_k, is_baseband) {
            (Some(h_body), false) => {
                let strip_h = (h_body as usize).min(h as usize);
                let n_strip_buf = if use_blur {
                    // Strip walker dispatches: each write is bw × body_h
                    // where body_h ≤ strip_h_at_k. Cap at full band when
                    // h_body would exceed the band height.
                    (w as usize) * strip_h
                } else {
                    // No-blur fallback kernel writes the full band at
                    // n_px. d_strip must cover that — at PU_PADSIZE-or-
                    // smaller bands the savings are negligible anyway.
                    n
                };
                let d_strip = [
                    alloc_zeros_f32(client, n_strip_buf),
                    alloc_zeros_f32(client, n_strip_buf),
                    alloc_zeros_f32(client, n_strip_buf),
                ];
                (None, Some(d_strip))
            }
            _ => {
                let d = [
                    alloc_zeros_f32(client, n),
                    alloc_zeros_f32(client, n),
                    alloc_zeros_f32(client, n),
                ];
                (Some(d), None)
            }
        };
        out.push(DBandsScratch { d, d_strip });
        // Ceil-div halving — see WeberScratch comment.
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        // Halve the strip body for the next (deeper) level. Clamp to
        // 1 so deep levels still allocate a usable strip buffer (the
        // walker's `strip_h_at_k` uses the same `.max(1)` clamp).
        strip_h_at_k = strip_h_at_k.map(|hb| (hb >> 1).max(1));
    }
    out
}

/// Dedicated ref-side full-image state populated by
/// [`Cvvdp::set_reference_strip`] when running in [`MemoryMode::Strip`].
/// The shared `bands_ref` / `weber_scratch[k].log_l_bkg` scratch lives
/// in the `Cvvdp` struct and gets clobbered by one-shot dispatches; the
/// strip-mode cached-ref path needs its own copy of the ref pyramid +
/// log_l_bkg planes so per-strip dist scoring can read ref data
/// without the every-call REF dispatch re-running.
///
/// Phase 2 (task #79) introduces this struct; Phase 3 will wire the
/// per-strip dist walker that reads from these buffers via row-slab
/// copies into the strip-sized working-set buffers.
///
/// Layout mirrors what [`Cvvdp::warm_reference`] writes into
/// `self.bands_ref[k]` + `self.weber_scratch[k].log_l_bkg`:
///
/// - `bands[k][c]` — per-level Weber-contrast band, `w_k × h_k` f32
///   plane, one per DKL channel.
/// - `log_l_bkg[k]` — per-non-baseband-level `log10(L_bkg)`,
///   `w_k × h_k` f32 plane (consumed by the per-band CSF apply).
/// - `baseband_gauss[c]` — baseband gauss pyramid level (read by the
///   band loop's baseband path when computing `inv_l_bkg_mean`). One
///   per DKL channel.
/// - `baseband_log_l_bkg_scalar` — host-side scalar mirroring the
///   value [`Cvvdp::warm_reference`] caches in
///   `warm_ref_baseband_log_l_bkg`.
struct RefFullState {
    /// Per-level Weber-contrast bands: `bands[k][c]` is one `w_k × h_k`
    /// f32 plane per DKL channel.
    bands: Vec<[cubecl::server::Handle; N_CHANNELS]>,
    /// Per-non-baseband-level `log10(L_bkg)` plane. Indexed `0..n_levels - 1`.
    log_l_bkg: Vec<cubecl::server::Handle>,
    /// Per-DKL-channel baseband gauss level. Used by the band loop's
    /// baseband path (the masking baseband bypass reads gauss[last]
    /// directly via `baseband_divide_3ch_kernel`).
    baseband_gauss: [cubecl::server::Handle; N_CHANNELS],
    /// Host-side baseband `log10(L_bkg)` scalar, mirroring
    /// `warm_ref_baseband_log_l_bkg` in Full mode.
    baseband_log_l_bkg_scalar: f32,
}

/// Strip-processing configuration, present iff this [`Cvvdp<R>`] was
/// constructed via [`Cvvdp::new_strip`] (or
/// [`Cvvdp::new_with_memory_mode`] resolving to
/// [`crate::ResolvedMode::Strip`]). Drives the strip-walker dispatch
/// in [`Cvvdp::compute_dkl_jod_with_warm_ref`] when the strip cached-
/// ref state is set.
///
/// Phase 1 (task #79) introduces the type with `h_body` storage. The
/// strip walker dispatch lands in Phase 3 — until then,
/// `compute_dkl_jod_with_warm_ref` surfaces a not-yet-implemented
/// error when called in strip mode (the public API enforces "set
/// reference, then call compute" so callers see the error before
/// any dist work has been queued).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StripMode {
    /// Mode E: ref-full cached on device, dist walks in strips.
    /// Per-DIST cost: only dist pyramid + masking work.
    CachedRef,
    /// Mode B: both ref and dist walk in strips together, no ref cache.
    /// Per-DIST cost: full ref+dist pipeline for every dist.
    Pair,
}

#[derive(Debug, Clone, Copy)]
struct StripConfig {
    /// Dist-side strip body height in rows at scale 0.
    h_body: u32,
    /// Walker variant — CachedRef (Mode E) or Pair (Mode B).
    mode: StripMode,
}

/// Reference-side state kept across `score_with_reference` calls.
///
/// Stashes the raw sRGB bytes; every `score_with_reference` call
/// re-runs the full GPU pipeline (`compute_dkl_jod`) against the
/// cached bytes — matches `Cvvdp::score(ref, dist)` byte-for-byte.
///
/// The dedicated warm-ref fast path that materialises the REF
/// Weber pyramid on the GPU once and skips it on subsequent DIST
/// calls lives at [`Cvvdp::warm_reference`] +
/// [`Cvvdp::compute_dkl_jod_with_warm_ref`] (~1.8× per-DIST
/// throughput at 12 MP — see `lib.rs` Status). The host-pool variants
/// `compute_dkl_jod_host_pool_with_warm_ref` give the same
/// optimisation on the cpu cubecl backend.
struct CachedReference {
    /// Cached reference sRGB bytes (length `width * height * 3`).
    ref_srgb: Vec<u8>,
}

/// ColorVideoVDP scorer.
///
/// Allocates GPU buffers up front for a fixed image size and reuses
/// them across calls. To score images of a different size, construct
/// a new `Cvvdp`.
///
/// # Examples
///
/// `ignore` because `cubecl::cuda::CudaRuntime` requires a live
/// CUDA driver — docs.rs builds in a sandbox without one. The
/// runtime-test counterpart in `tests/pipeline_score.rs` exercises
/// this exact pattern under the `cuda` feature.
///
/// ```ignore
/// use cubecl::{Runtime, cuda::CudaRuntime};
/// use cvvdp_gpu::{Cvvdp, CvvdpParams};
///
/// // Allocate buffers for a 256² image. Reuse `cvvdp` for any
/// // number of (ref, dist) pairs at THIS size — the GPU buffers
/// // are owned by `Cvvdp` and reused per call.
/// let client = CudaRuntime::client(&Default::default());
/// let mut cvvdp =
///     Cvvdp::<CudaRuntime>::new(client, 256, 256, CvvdpParams::PLACEHOLDER)?;
///
/// // One-shot scoring (re-uploads ref every call):
/// let ref_bytes = vec![128_u8; 256 * 256 * 3];
/// let dist_bytes = vec![100_u8; 256 * 256 * 3];
/// let jod = cvvdp.score(&ref_bytes, &dist_bytes)?;
/// assert!(jod <= 10.0 && jod >= 0.0);
///
/// // Cached-reference scoring for many DISTs vs one REF: see
/// // `set_reference` + `score_with_reference`, or the faster
/// // `warm_reference` + `compute_dkl_jod_with_warm_ref`.
/// # Ok::<(), cvvdp_gpu::Error>(())
/// ```
pub struct Cvvdp<R: Runtime> {
    client: ComputeClient<R>,
    params: CvvdpParams,
    /// Viewing geometry — drives PPD (= cy/deg) for the CSF lookup.
    /// Independent of `width`/`height` (the image dimensions) since
    /// cvvdp's PPD is a display property, not an image one.
    geometry: crate::params::DisplayGeometry,
    width: u32,
    height: u32,
    n_levels: u32,

    /// sRGB byte upload scratch. The GPU helpers reuse this slot
    /// for both ref and dist (writing one side, running the
    /// pipeline, reading back, then overwriting for the other
    /// side). A second `_dis` slot was originally allocated but
    /// went unused — kept on one buffer to save ~3 MB at 256×256.
    src_ref: cubecl::server::Handle,

    /// Persistent host-side `Vec<u32>` reused across every
    /// `_dispatch_dkl_planes_gpu` call to widen the input
    /// `&[u8]` sRGB bytes to `u32` slots (one byte per slot —
    /// `srgb_to_dkl_kernel` reads `Array<u32>` because the LUT
    /// indexing wants the byte as an integer). Tick 234 replaces
    // T_x.O (2026-05-17): `src_u32_scratch: Vec<u32>` removed. The
    // upload path now packs u8×3 → u32 directly into the pinned
    // staging buffer reserved per call (`client.reserve_staging`),
    // collapsing two host-side passes (pack to pageable + memcpy to
    // pinned) into one. Saves a ~48 MB host write per upload at 12 MP.

    /// 256-entry sRGB→linear LUT, uploaded once.
    srgb_lut: cubecl::server::Handle,

    /// Gaussian pyramid buffers (per channel, per level). Reused
    /// for both sides — each `compute_dkl_*` call overwrites these
    /// for the side it's currently processing then reads back.
    gauss_ref: Vec<Level>,

    /// Pyramid-band buffers for the REFERENCE side (per channel,
    /// per level). The REF weber-pyramid dispatch writes here; the
    /// band loop's CSF reads here for REF inputs. Coarsest level
    /// shares storage with the coarsest gaussian for the Weber
    /// baseband path.
    ///
    /// **P2.3 (2026-05-27):** in `StripMode::Pair`, shallow non-
    /// baseband levels (`k < k_split`) are allocated as zero-size
    /// handles. The REF strip helper in `_run_d_bands_strip_major_shallow`
    /// writes [`WeberScratch::bands_ref_strip`] per `(s, k)` instead,
    /// in lockstep with the DIST CSF dispatch that reads it. Deep
    /// non-baseband levels and the baseband keep full-image
    /// allocations because the level-major dispatch path reads them
    /// at full level dims.
    bands_ref: Vec<Level>,

    /// Alternate full-image Gaussian-pyramid buffer (`StripMode::Pair`
    /// only). `Some(...)` only in Mode B; `None` elsewhere.
    ///
    /// **P2.3 (2026-05-27):** the REF and DIST sides write into a
    /// SHARED `gauss_ref` buffer (REF first, DIST clobbers). Mode B's
    /// strip-major-outer band loop needs REF gauss data to persist
    /// through DIST dispatch so the per-strip REF weber helper can
    /// read it. We allocate this alt-buffer once at construction and
    /// `mem::swap` `gauss_ref` ↔ `gauss_alt` after REF weber finalize:
    /// post-swap `gauss_alt` holds REF gauss data, `gauss_ref` is the
    /// (overwritten) prior REF buffer (garbage), and the next DIST
    /// gauss dispatch writes to `gauss_ref` without disturbing REF
    /// state in `gauss_alt`.
    ///
    /// **Memory cost (P2.3):** at 4096² adds ~268 MiB (one full-image
    /// pyramid). Net P2.3 win: bands_ref shrink frees ~680 MiB,
    /// minus this 268 MiB = ~412 MiB net reduction. P2.7 will strip-
    /// shape both gauss_ref and gauss_alt to recover this overhead.
    gauss_alt: Option<Vec<Level>>,

    /// Pyramid-band buffers for the DISTORTED side. Same shape as
    /// `bands_ref`; separate storage so both sides' weber-pyramid
    /// data can live on GPU simultaneously through the d_bands
    /// band loop (avoiding host upload from `dist_weber` Vec).
    bands_dis: Vec<Level>,

    /// Per-level scratch for `compute_dkl_d_bands`'s CSF, masking,
    /// and D output buffers. Pre-allocated so the hot loop doesn't
    /// churn GPU allocations per band (~1.5 GB worth at 12 MP).
    d_scratch: Vec<DBandsScratch>,

    /// Per-non-baseband-level scratch for `compute_dkl_weber_pyramid`'s
    /// expand/subtract/weber chain. Pre-allocated; reused per side
    /// per call. ~176 MB worth at 12 MP per call.
    weber_scratch: Vec<WeberScratch>,

    /// Pre-allocated per-pixel log_l_bkg buffer for the baseband level.
    /// `subtract_weber_3ch_kernel` doesn't run at the baseband, so
    /// `weber_scratch[last].log_l_bkg` (which doesn't exist — `weber_scratch`
    /// only spans non-baseband levels) can't hold the baseband value.
    /// `_dispatch_d_bands_into_scratch` fills this with the host-computed
    /// scalar `log_l_bkg_baseband` via `fill_f32_kernel` per JOD call.
    /// Tick 168 replaces the per-call `vec![scalar; n_baseband]` host
    /// alloc + upload with a single GPU launch.
    baseband_log_l_bkg: cubecl::server::Handle,

    /// Pre-allocated `n_levels × N_CHANNELS` partials buffer that
    /// `pool_band_3ch_kernel` accumulates into via `Atomic<f32>::fetch_add`.
    /// `_pool_and_finalize_jod` zero-fills via `fill_f32_kernel` per call
    /// (one tiny launch, ~144 bytes worth at MAX_LEVELS=9) instead of
    /// allocating a fresh GPU buffer + uploading host zeros every JOD
    /// call. Tick 227: replaces the per-call `create_from_slice` host
    /// alloc + GPU upload.
    partials_h: cubecl::server::Handle,

    /// Pre-built clones of `weber_scratch[k].log_l_bkg` and
    /// `weber_scratch[k].log_l_bkg_dis` handles, one per non-baseband
    /// level. `_dispatch_ref_weber_pyramid_only` and
    /// `_dispatch_dist_weber_pyramid_only` pass these by reference
    /// to `_dispatch_weber_pyramid_gpu` instead of building a fresh
    /// `Vec<Handle>` + `n_levels - 1` handle ref-bumps per JOD-side
    /// dispatch. Tick 240.
    log_l_bkg_ref_dests: Vec<cubecl::server::Handle>,
    log_l_bkg_dis_dests: Vec<cubecl::server::Handle>,

    /// Pre-uploaded logs_row buffers for the CSF per-pixel apply.
    /// Indexed `[level][channel]`. Each holds the 32-entry
    /// `precompute_logs_row(rho_k, channel)` result. rho_k depends
    /// on `geometry.pixels_per_degree()` which is fixed per Cvvdp
    /// — so these are stable across calls and reuploading per band
    /// is pure waste (was 24 uploads of 128 B per call).
    logs_row: Vec<[cubecl::server::Handle; N_CHANNELS]>,

    /// Reference-side cache (used by `score_with_reference`).
    cached: Option<CachedReference>,

    /// GPU-warm reference state for batch scoring. `Some(scalar)`
    /// means `warm_reference` was called and `bands_ref` +
    /// `weber_scratch[k].log_l_bkg` hold a valid REF state; the
    /// scalar is the baseband `log10(L_bkg)` returned by the REF
    /// weber dispatch (needed by the band loop's baseband CSF
    /// path). Reset to `None` whenever a method that dispatches
    /// REF weber runs (compute_dkl_jod, compute_dkl_d_bands, etc.),
    /// since those overwrite bands_ref and weber_scratch.log_l_bkg
    /// with the new REF's data.
    warm_ref_baseband_log_l_bkg: Option<f32>,

    /// Diffmap pipeline scratch (3 base-res accumulator planes + 1
    /// output plane). Lazy-allocated on the first
    /// `score_with_diffmap` / `score_from_linear_planes_with_diffmap`
    /// call; `None` until then. Callers that only request the JOD
    /// scalar pay zero memory for diffmap support.
    diffmap_scratch: Option<DiffmapScratch>,

    /// Linear-RGB-planes upload scratch (3 planes × `W * H * 4 bytes`
    /// each). Lazy-allocated on the first `from_linear_planes`-family
    /// call; `None` until then. Skipped entirely on the sRGB-byte
    /// upload path.
    linear_planes_upload: Option<LinearPlanesUpload>,

    /// Strip-mode configuration. `Some(_)` iff this `Cvvdp<R>` was
    /// built via [`Cvvdp::new_strip`] / `new_with_memory_mode` resolving
    /// to [`crate::ResolvedMode::Strip`]. Drives the cached-ref strip
    /// walker dispatch (Phase 3+).
    strip_config: Option<StripConfig>,

    /// Dedicated full-image ref-side state for strip mode. Populated
    /// by [`Cvvdp::set_reference_strip`] when in strip mode; `None`
    /// before set_reference and in non-strip configurations. See
    /// [`RefFullState`] for the layout.
    ref_full_state: Option<RefFullState>,

    /// Cumulative count of strip iterations the Mode E pool walker
    /// has dispatched since construction. Incremented by
    /// `_pool_and_finalize_jod_strip` per outer strip-iteration in
    /// the band loop. Visible via [`Self::strip_dispatch_counter`]
    /// (hidden accessor) so the Phase 3 parity test can assert that
    /// the walker actually partitioned (N >= 2 strips) at large
    /// sizes, distinguishing a real strip dispatch from a single
    /// full-image Full-mode fallback.
    ///
    /// Phase 3 (task #79): only the pool stage of the band loop is
    /// strip-aware at this point. The dist weber pyramid + CSF +
    /// masking still run full-image; the strip walker partitions the
    /// per-band `d_scratch[k].d[c]` pixel range and dispatches the
    /// `pool_band_3ch_offset_kernel` per slab. Atomic-add into
    /// `partials_h` is associative across slabs, so the JOD scalar is
    /// bit-exact against Full-mode `_pool_and_finalize_jod`. The
    /// architectural foundation generalises to strip-aware CSF +
    /// masking (when the kernels are ported to logical-image
    /// reflection).
    strip_dispatch_counter: core::sync::atomic::AtomicU32,
}

fn pyramid_levels(ppd: f32, width: u32, height: u32) -> u32 {
    // Defer to band_frequencies — same as host_scalar (see
    // host_scalar::predict_jod_still_3ch) and same as pycvvdp's
    // n_bands. Tick 181: previously we additionally capped by a
    // size-based loop with PYRAMID_MIN_DIM=4 (stopping when
    // cur < 8), which under-counted at small inputs (5 vs 6 at
    // 73×91, 4 vs 5 at 32×32, 7 vs 8 at 256×256) and drove a
    // ~0.6 JOD drift vs pycvvdp at sub-megapixel sizes. The
    // band_frequencies cutoff (~0.2 cpd) is the authoritative
    // limit; MAX_LEVELS still caps the alloc count.
    let band_count = crate::kernels::pyramid::band_frequencies(ppd, width as usize, height as usize)
        .len() as u32;
    band_count.min(MAX_LEVELS as u32)
}

/// Validation predicate for the `h_body` strip-walker parameter.
///
/// Accepts any positive power of two. Powers of two halve cleanly
/// through every Weber pyramid level (`h_body >> k` stays an
/// integer ≥ 1 for `k < log2(h_body)`, with the `.max(1)` clamp in
/// the pool/walker handling deeper levels), so this is exactly the
/// alignment the strip walker actually needs. The legacy
/// [`crate::memory_mode::STRIP_ALIGN`] constant (= 256) baked in a
/// worst-case alignment for `MAX_LEVELS = 9` images; that's too
/// strict for small inputs (e.g. 128² has only 6 levels, so
/// `h_body = 32` aligns cleanly), and rejecting those left the
/// Mode B walker un-testable at small sizes.
///
/// Rejects: zero, non-power-of-two values (e.g. 100, 200, 300).
/// Accepts: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, …
#[inline]
fn is_valid_strip_h_body(h_body: u32) -> bool {
    h_body > 0 && h_body.is_power_of_two()
}

/// Static-analysis predictor for the GPU memory `Cvvdp::new` will
/// allocate for an image of `(width, height)` under the standard 4K
/// viewing geometry. Sums every persistent buffer enumerated in
/// `Cvvdp::new` (source bytes, three full pyramids, d_scratch,
/// weber_scratch, partials, baseband log_l_bkg, srgb_lut, logs_row)
/// using ceil-div halving per level — matches the actual allocator
/// layout (tick 175's ceil-div pyramid + tick 208's d_scratch + tick
/// 240's pre-bundled handles).
///
/// Use this to **cap concurrency** when running many Cvvdp instances
/// against a shared GPU: divide free GPU memory (via
/// `cudaMemGetInfo` or the equivalent) by `1.5 × estimate` (1.5×
/// safety factor for transient kernel allocations + cubecl runtime
/// overhead) to derive a safe parallel-instance count.
///
/// Returns `None` if `(width, height)` is below the
/// [`PYRAMID_MIN_DIM`] × 2 threshold — same precondition as
/// [`Cvvdp::new`].
///
/// # Caveats
///
/// - Counts only the buffers visible at `Cvvdp::new` time; transient
///   per-call uploads (the per-DIST `srgb` byte buffer, per-band
///   readback bytes when callers use `compute_dkl_*_bands`, etc.)
///   are excluded. Add ~`width * height * 4` for one warm dist
///   buffer if you `score` in a loop.
/// - Uses `DisplayGeometry::STANDARD_4K` to derive `n_levels`. Other
///   geometries shift PPD which shifts `band_frequencies`'s cutoff;
///   for typical 4K-class viewing the level count is the same
///   ±1 across realistic geometries.
/// - The cubecl runtime adds its own metadata + page alignment
///   overhead per buffer (~hundreds of bytes per allocation × ~50
///   allocations = single-digit MB). Bake this into the safety
///   factor at the call site.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::estimate_gpu_memory_bytes;
///
/// // Too small — below PYRAMID_MIN_DIM × 2 = 8.
/// assert!(estimate_gpu_memory_bytes(4, 4).is_none());
/// assert!(estimate_gpu_memory_bytes(7, 8).is_none());
///
/// // 1 MP estimate is on the order of 200 MB — the float planes
/// // (3 pyramids × 3 channels at the fine level alone) dominate.
/// let bytes_1mp = estimate_gpu_memory_bytes(1024, 1024).expect("1MP");
/// assert!(bytes_1mp > 100_000_000);
/// assert!(bytes_1mp < 300_000_000);
///
/// // 4 MP (2048²) has 4× the pixels of 1 MP (1024²). The ratio
/// // should be in `[3.6, 4.4]` — pinned by
/// // `tests/pipeline_score.rs::estimate_gpu_memory_scales_with_pixel_count`.
/// // The tolerance band absorbs the ceil-div pyramid sum overhead +
/// // fixed-cost dilution at small sizes (srgb_lut + partials +
/// // logs_row are constant-ish).
/// let bytes_4mp = estimate_gpu_memory_bytes(2048, 2048).expect("4MP");
/// let ratio = bytes_4mp as f64 / bytes_1mp as f64;
/// assert!(ratio > 3.6 && ratio < 4.4, "ratio = {ratio}");
/// ```
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> Option<usize> {
    if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
        return None;
    }
    let ppd = crate::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let n_levels = pyramid_levels(ppd, width, height) as usize;
    let n0 = (width as usize) * (height as usize);

    // src_ref: u32 array of length n0 * 3 = 12 bytes/pixel.
    let src_bytes: usize = n0 * 3 * 4;
    // srgb_lut: 256 entries × f32.
    let srgb_lut_bytes: usize = 256 * 4;
    // Persistent partials buffer: n_levels × N_CHANNELS × f32.
    let partials_bytes: usize = n_levels * crate::N_CHANNELS * 4;
    // logs_row: per (level, channel) a length-N_L_BKG row of f32.
    let logs_row_bytes: usize = n_levels * crate::N_CHANNELS * crate::kernels::csf::N_L_BKG * 4;

    // Per-level pixel count (ceil-div halving).
    let mut level_pixels: Vec<usize> = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    for _ in 0..n_levels {
        level_pixels.push((w as usize) * (h as usize));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let sum_level_pixels: usize = level_pixels.iter().sum();

    // gauss_ref + bands_ref + bands_dis: each is 3 channels × sum
    // of per-level pixel counts × f32.
    let pyramid_bytes: usize = 3 * 3 * sum_level_pixels * 4;

    // d_scratch: post-Path-B layout. The persistent allocation
    // holds only the `d` output planes (1 buffer kind × 3 channels)
    // for every level. The 5 transient buffer kinds (t_p_ref,
    // t_p_dis, m_raw, m_mid, m_blur) are allocated inside the band
    // loop one band at a time and dropped after that band's
    // masking dispatch — so peak working set adds just one band's
    // transient at the largest level (= base resolution n0). At
    // 1MP with 8 levels this drops the d_scratch contribution by
    // ~21% vs the prior all-bands-at-once layout.
    let d_scratch_persistent_bytes: usize = 3 * sum_level_pixels * 4;
    let d_scratch_transient_peak_bytes: usize = 5 * 3 * n0 * 4;
    let d_scratch_bytes: usize = d_scratch_persistent_bytes + d_scratch_transient_peak_bytes;

    // weber_scratch: only non-baseband levels (n_levels - 1).
    // Per level: 3 fine-sized planes (l_bkg_fine, log_l_bkg,
    // log_l_bkg_dis) + 3 fine-sized upscaled_c + 1 v-scratch
    // (vscratch_a, half-width) + 3 v-scratch (vscratch_c, half-width).
    // v-scratch size = ceil(W/2) × H per level.
    let mut weber_bytes: usize = 0;
    let mut fw = width;
    let mut fh = height;
    for _ in 0..n_levels.saturating_sub(1) {
        let n_fine = (fw as usize) * (fh as usize);
        let cw = fw.div_ceil(2);
        let n_v = (cw as usize) * (fh as usize);
        let fine_planes = 6_usize; // l_bkg_fine + log_l_bkg + log_l_bkg_dis + 3 × upscaled_c
        let v_planes = 4_usize; // vscratch_a + 3 × vscratch_c
        weber_bytes += (fine_planes * n_fine + v_planes * n_v) * 4;
        fw = cw;
        fh = fh.div_ceil(2);
    }

    // Baseband log_l_bkg buffer: pixels at the coarsest level × f32.
    let baseband_bytes: usize = level_pixels.last().copied().unwrap_or(0) * 4;

    Some(
        src_bytes
            + srgb_lut_bytes
            + partials_bytes
            + logs_row_bytes
            + pyramid_bytes
            + d_scratch_bytes
            + weber_bytes
            + baseband_bytes,
    )
}

/// Capped-pyramid GPU-memory estimator. Returns the working-set bytes
/// when the cvvdp pipeline is constructed via
/// [`Cvvdp::new_capped_pyramid`] with the given `levels` cap. The
/// caller-supplied `levels` is clamped against the natural pyramid
/// depth for `(width, height)` under STANDARD_4K geometry; the
/// returned bytes mirror exactly what [`estimate_gpu_memory_bytes`]
/// would compute if it operated on the capped depth.
///
/// Returns `None` when `(width, height)` is below the pyramid minimum
/// or `levels == 0`.
///
/// **Not JOD-bit-identical** to Full mode; the capped pyramid drops
/// the deepest bands. See [`crate::MemoryMode::CappedPyramid`].
#[must_use]
pub fn estimate_gpu_memory_bytes_capped(
    width: u32,
    height: u32,
    levels: u32,
) -> Option<usize> {
    if levels == 0 {
        return None;
    }
    if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
        return None;
    }
    let ppd = crate::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let natural = pyramid_levels(ppd, width, height) as usize;
    let n_levels = natural.min(levels as usize).max(1);

    let n0 = (width as usize) * (height as usize);
    let src_bytes: usize = n0 * 3 * 4;
    let srgb_lut_bytes: usize = 256 * 4;
    let partials_bytes: usize = n_levels * crate::N_CHANNELS * 4;
    let logs_row_bytes: usize = n_levels * crate::N_CHANNELS * crate::kernels::csf::N_L_BKG * 4;

    let mut level_pixels: Vec<usize> = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    for _ in 0..n_levels {
        level_pixels.push((w as usize) * (h as usize));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let sum_level_pixels: usize = level_pixels.iter().sum();
    let pyramid_bytes: usize = 3 * 3 * sum_level_pixels * 4;
    // Same lazy-transient accounting as estimate_gpu_memory_bytes.
    let n0_capped = level_pixels.first().copied().unwrap_or(0);
    let d_scratch_persistent_bytes: usize = 3 * sum_level_pixels * 4;
    let d_scratch_transient_peak_bytes: usize = 5 * 3 * n0_capped * 4;
    let d_scratch_bytes: usize = d_scratch_persistent_bytes + d_scratch_transient_peak_bytes;

    let mut weber_bytes: usize = 0;
    let mut fw = width;
    let mut fh = height;
    for _ in 0..n_levels.saturating_sub(1) {
        let n_fine = (fw as usize) * (fh as usize);
        let cw = fw.div_ceil(2);
        let n_v = (cw as usize) * (fh as usize);
        let fine_planes = 6_usize;
        let v_planes = 4_usize;
        weber_bytes += (fine_planes * n_fine + v_planes * n_v) * 4;
        fw = cw;
        fh = fh.div_ceil(2);
    }

    let baseband_bytes: usize = level_pixels.last().copied().unwrap_or(0) * 4;

    Some(
        src_bytes
            + srgb_lut_bytes
            + partials_bytes
            + logs_row_bytes
            + pyramid_bytes
            + d_scratch_bytes
            + weber_bytes
            + baseband_bytes,
    )
}

/// Strip-mode (Mode E) GPU-memory estimator. Returns the working-set
/// bytes when the cvvdp pipeline is constructed via
/// [`Cvvdp::new_strip`]:
///
/// - Full ref state on device (dedicated buffers, identical layout to
///   [`Cvvdp::warm_reference`] but stored in a separate [`RefFullState`]
///   so the shared `bands_ref` / `weber_scratch` scratch can house
///   per-strip dist-side traffic without clobbering the ref).
/// - Strip-sized dist working set: per-level pyramids sized for one
///   `(h_body + 2 × halo)` strip rather than the full image.
///
/// Returns `None` if `(width, height)` is below the pyramid minimum or
/// if `h_body` is zero / mis-aligned. Mirrors
/// [`estimate_gpu_memory_bytes`]'s caveats (geometry-derived `n_levels`,
/// transient overhead excluded).
///
/// This estimator is intentionally **conservative** in this initial
/// landing (task #79 Phase 1): it bounds the strip footprint by the
/// full-image footprint **plus** a small fixed delta for the dedicated
/// ref cache. As the strip walker (Phase 3) shrinks the dist working
/// set, this estimator will tighten. Today the value is suitable for
/// "Strip is at most this much" decisions in `resolve_auto`, not for
/// fine-grained capacity planning.
#[must_use]
pub fn estimate_gpu_memory_bytes_strip(width: u32, height: u32, h_body: u32) -> Option<usize> {
    if !is_valid_strip_h_body(h_body) {
        return None;
    }
    let full_bytes = estimate_gpu_memory_bytes(width, height)?;
    let ppd = crate::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let n_levels = pyramid_levels(ppd, width, height) as usize;

    // Dedicated ref-full state: 3 channels × sum-of-per-level pixels ×
    // f32 for ref bands + 3 channels × full-image pixels × f32 for the
    // baseband gauss (read by the band loop's baseband path) + per-
    // non-baseband-level log_l_bkg (one fine-sized f32 plane per level).
    let mut sum_level_pixels: usize = 0;
    let mut log_l_bkg_pixels: usize = 0;
    let mut w = width;
    let mut h = height;
    for k in 0..n_levels {
        let n = (w as usize) * (h as usize);
        sum_level_pixels += n;
        if k < n_levels - 1 {
            log_l_bkg_pixels += n;
        }
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let ref_full_bands_bytes = crate::N_CHANNELS * sum_level_pixels * 4;
    let ref_full_log_l_bkg_bytes = log_l_bkg_pixels * 4;
    let ref_full_baseband_gauss_bytes =
        crate::N_CHANNELS * width.div_ceil(1 << (n_levels - 1)) as usize
            * height.div_ceil(1 << (n_levels - 1)) as usize
            * 4;
    let ref_full_bytes =
        ref_full_bands_bytes + ref_full_log_l_bkg_bytes + ref_full_baseband_gauss_bytes;

    // h_body is currently unused by the (not-yet-implemented) strip
    // dispatch — Phase 3 will replace this conservative bound with a
    // strip-sized dist-working-set estimate.
    let _ = h_body;

    Some(full_bytes.saturating_add(ref_full_bytes))
}

/// Mode B (StripPair) GPU-memory estimator. Returns the working-set
/// bytes when the cvvdp pipeline is constructed via
/// [`Cvvdp::new_strip_pair`]:
///
/// - No full ref cache — only the strip-pair walker's per-strip
///   working set is kept on device.
/// - Strip-sized ref+dist working set: per-level pyramids sized for
///   one `(h_body + 2 × halo)` strip rather than the full image.
///
/// Returns `None` if `(width, height)` is below the pyramid minimum or
/// if `h_body` is zero / mis-aligned. Mirrors
/// [`estimate_gpu_memory_bytes`]'s caveats.
///
/// The estimate models the hybrid K_SPLIT walker
/// ([`mode_b_k_split`]): bands shallower than K_SPLIT use strip-sized
/// `(h_body + 2 × halo)` storage (halved per level), bands at K_SPLIT
/// and deeper keep full-image storage (tiny at deep levels — level 8
/// at 4096² is 16×16 = 256 f32 per channel). The dist-side scratch
/// (`d_scratch.t_p_*`, `d_scratch.m_*`, etc.) follows the same
/// per-level split. `gauss_ref` carries the ref-side state through the
/// walker; sized identically to the dist side so the per-strip
/// dispatch can run REF + DIST through the same buffer geometry.
#[must_use]
pub fn estimate_gpu_memory_bytes_strip_pair(
    width: u32,
    height: u32,
    h_body: u32,
) -> Option<usize> {
    if !is_valid_strip_h_body(h_body) {
        return None;
    }
    if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
        return None;
    }
    let ppd = crate::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let n_levels = pyramid_levels(ppd, width, height) as usize;
    let n0 = (width as usize) * (height as usize);

    // Fixed-cost buffers (same in Full and Strip-Pair modes).
    let src_bytes: usize = n0 * 3 * 4;
    let srgb_lut_bytes: usize = 256 * 4;
    let partials_bytes: usize = n_levels * crate::N_CHANNELS * 4;
    let logs_row_bytes: usize = n_levels * crate::N_CHANNELS * crate::kernels::csf::N_L_BKG * 4;

    // Per-level buffer pixel count. Shallow bands (k < K_SPLIT) use
    // a strip-sized buffer of `(W>>k, body_h>>k + 2*halo>>k)`. Deep
    // bands (k >= K_SPLIT) use full-image-sized `(W>>k, H>>k)` —
    // small in absolute terms (level 8 at 4096² is 256 px).
    let k_split = mode_b_k_split(h_body, n_levels as u32) as usize;
    let mut sum_level_pixels: usize = 0;
    let mut sum_level_pixels_v: usize = 0; // for vscratch (half-width fine_h)
    let mut buf_w = width;
    let mut buf_h = height; // updated each level
    for k in 0..n_levels {
        // Level pixel count. Strip buffer height comes from the
        // back-projected helper `mode_b_strip_h_at_level` (which
        // accounts for the reduce chain's source-row requirements
        // recursively); we clamp it to the level's full-image height
        // because the strip can't legitimately exceed the whole image
        // (happens at small (h, h_body) pairs — e.g. 1024² h_body=512
        // back-projects to 1148 > 1024 at level 0, in which case the
        // strip degenerates to full-image storage). Deep levels
        // (k >= k_split) use full-image dims unconditionally.
        let n_lvl = if k < k_split {
            let strip_h = mode_b_strip_h_at_level(k as u32, h_body, k_split as u32).min(buf_h);
            (buf_w as usize) * (strip_h as usize)
        } else {
            (buf_w as usize) * (buf_h as usize)
        };
        sum_level_pixels += n_lvl;

        // vscratch is coarse_w * fine_h — half-width of the level's
        // output dimensions, used by upscale_v. Same shallow/deep split.
        if k < n_levels.saturating_sub(1) {
            let cw = buf_w.div_ceil(2);
            let fine_h_eff = if k < k_split {
                mode_b_strip_h_at_level(k as u32, h_body, k_split as u32).min(buf_h)
            } else {
                buf_h
            };
            sum_level_pixels_v += (cw as usize) * (fine_h_eff as usize);
        }

        // Advance buffer dims for next level.
        buf_w = buf_w.div_ceil(2);
        buf_h = buf_h.div_ceil(2);
    }

    // gauss_ref + bands_ref + bands_dis: 3 pyramids × 3 channels each.
    let pyramid_bytes: usize = 3 * 3 * sum_level_pixels * 4;
    // d_scratch (lazy-transient layout): persistent `d` only +
    // peak transient = one band's worth (5 kinds × 3 channels at
    // the largest band buffer size, which under the K_SPLIT
    // walker is the strip-buffer-sized fine band).
    let largest_level_pixels = {
        // Compute the largest per-level buffer pixel count the way
        // sum_level_pixels was computed above (strip-buf for k <
        // k_split, full-image for k >= k_split). The fine band
        // (k = 0) always dominates because back-projection accumulates
        // halo as you climb up the pyramid. Clamp to full-image height
        // for the small-image-large-h_body degenerate case.
        let strip_h0 = mode_b_strip_h_at_level(0, h_body, k_split as u32).min(height);
        (width as usize) * (strip_h0 as usize)
    };
    let d_scratch_persistent_bytes: usize = 3 * sum_level_pixels * 4;
    let d_scratch_transient_peak_bytes: usize = 5 * 3 * largest_level_pixels * 4;
    let d_scratch_bytes: usize = d_scratch_persistent_bytes + d_scratch_transient_peak_bytes;

    // weber_scratch: only non-baseband levels. Per level: 3 fine-sized
    // planes (l_bkg_fine, log_l_bkg, log_l_bkg_dis) + 3 upscaled_c_strip
    // (which is strip-sized in Mode B — but `sum_level_pixels` here is
    // already the strip-aware sum, so we count them the same way) +
    // 1 + 3 v-scratch (half-width).
    // Approx: 6 fine + 4 vscratch — Path A Phase 1c (2026-05-26) makes
    // the full-image `upscaled_c` an Option that's None in Mode B, so
    // the "3 upscaled_c" line above refers to `upscaled_c_strip`, which
    // is strip-sized and already accounted for in `sum_level_pixels`.
    let weber_fine: usize = 6 * sum_level_pixels.saturating_sub(
        // Subtract the deepest level's contribution since weber_scratch
        // doesn't carry the baseband.
        {
            let mut last_pixels = 0usize;
            let mut bw = width;
            let mut bh = height;
            for k in 0..n_levels {
                let n_lvl = if k < k_split {
                    let strip_h =
                        mode_b_strip_h_at_level(k as u32, h_body, k_split as u32).min(bh);
                    (bw as usize) * (strip_h as usize)
                } else {
                    (bw as usize) * (bh as usize)
                };
                if k == n_levels - 1 {
                    last_pixels = n_lvl;
                }
                bw = bw.div_ceil(2);
                bh = bh.div_ceil(2);
            }
            last_pixels
        },
    ) * 4;
    let weber_vscratch: usize = 4 * sum_level_pixels_v * 4;
    let weber_bytes: usize = weber_fine + weber_vscratch;

    // Baseband log_l_bkg buffer: deep level (full-image pixels at deepest level).
    let baseband_bytes: usize = {
        let mut w = width;
        let mut h = height;
        for _ in 0..n_levels - 1 {
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        (w as usize) * (h as usize) * 4
    };

    Some(
        src_bytes
            + srgb_lut_bytes
            + partials_bytes
            + logs_row_bytes
            + pyramid_bytes
            + d_scratch_bytes
            + weber_bytes
            + baseband_bytes,
    )
}

/// Pick the hybrid K_SPLIT for Mode B given `h_body` and the natural
/// pyramid level count. Bands at level `k < K_SPLIT` are processed
/// per-strip; bands at level `k >= K_SPLIT` are processed full-image.
///
/// The split is chosen so the strip body at the band's resolution
/// (`h_body >> k`) is at least `MODE_B_DEEP_THRESHOLD` rows — beyond
/// that the PU-blur halo (±6 rows at the band's resolution) approaches
/// or exceeds the body, so striping yields no benefit. Returns the
/// smaller of `n_levels` and the computed split.
///
/// For `h_body = 512` and 9 pyramid levels: K_SPLIT = 5 (bands 0-4
/// strip-aware, bands 5-8 full-image). For smaller `h_body` the split
/// shifts down accordingly.
#[doc(hidden)]
#[must_use]
pub fn mode_b_k_split(h_body: u32, n_levels: u32) -> u32 {
    const MODE_B_DEEP_THRESHOLD: u32 = 12;
    let mut k_split = 0;
    while k_split < n_levels && (h_body >> k_split) >= MODE_B_DEEP_THRESHOLD {
        k_split += 1;
    }
    k_split.min(n_levels)
}

/// Per-level **band-resolution** halo for the Mode B strip walker.
///
/// This is the halo a single level needs *at its own resolution* for
/// its own band-loop (PU blur reads ±6, pyramid downscale reads ±2,
/// so 8 rows covers both). It does NOT account for back-projection
/// through the reduce chain — see [`mode_b_strip_h_at_level`] for
/// the correct buffer height accounting cross-level reduce halos.
///
/// Kept as a separate helper because some callers (e.g., the band-
/// loop's own per-strip halo math) need the level-local value.
/// Allocator + estimator callers should prefer
/// [`mode_b_strip_h_at_level`].
#[doc(hidden)]
#[must_use]
pub fn mode_b_halo_at_level(k: u32, k_split: u32) -> u32 {
    if k >= k_split {
        0 // deep levels use full-image storage, no halo padding
    } else {
        // PU blur radius (6) + 2-tap downscale slack at this level.
        8
    }
}

/// Strip buffer height at level `k` for the Mode B walker
/// (back-projected through the reduce chain).
///
/// `downscale_strip_kernel` reads `±2` source rows around `2·dy_logical`,
/// so producing `R_{k+1}` valid level-(k+1) output rows from level-k
/// source requires `2·R_{k+1} + 4` level-k source rows. The level-k
/// buffer must satisfy two constraints simultaneously:
///
/// 1. Its own band loop reads body+halo at level k:
///    `R_k ≥ body_k + 2·halo_k = (h_body >> k) + 16`.
/// 2. It must feed the level-(k+1) reduce:
///    `R_k ≥ 2·R_{k+1} + 4` for `k < k_split − 1`.
///
/// The recursion runs deepest-shallow → shallowest. At `h_body = 512,
/// k_split = 6`:
///
/// | k | body_k | R_k                              |
/// |---|--------|----------------------------------|
/// | 5 | 16     | 32                               |
/// | 4 | 32     | max(48, 2·32+4) = 68             |
/// | 3 | 64     | max(80, 2·68+4) = 140            |
/// | 2 | 128    | max(144, 2·140+4) = 284          |
/// | 1 | 256    | max(272, 2·284+4) = 572          |
/// | 0 | 512    | max(528, 2·572+4) = **1148**     |
///
/// Compared to the band-resolution-only model (which would give 528 at
/// level 0), back-projection roughly doubles the level-0 buffer at
/// h_body=512. The level-0 strip is `1148·W·4 ≈ 18.8 MiB` per channel
/// at 4096² fine_w vs `4096·W·4 = 64 MiB` for full-image — a ~71%
/// per-buffer saving at level 0.
///
/// Returns 0 for `k >= k_split` since deep levels use full-image
/// storage (caller substitutes the level-k full image dim).
#[doc(hidden)]
#[must_use]
pub fn mode_b_strip_h_at_level(k: u32, h_body: u32, k_split: u32) -> u32 {
    if k >= k_split {
        return 0; // caller uses full-image dim instead
    }
    // Iterate deepest-shallow → up. Track R_{k+1} as we walk down to k.
    // Deepest shallow level (k_split - 1): only the body+halo constraint
    // applies (no further reduce to feed).
    let halo_band = 8_u32;
    let deepest = k_split - 1;
    let mut r_deeper: u32 = (h_body >> deepest).saturating_add(2 * halo_band);
    if k == deepest {
        return r_deeper;
    }
    // Walk from k_split - 2 down to k.
    for ki in (k..deepest).rev() {
        let body_ki = h_body >> ki;
        let own = body_ki.saturating_add(2 * halo_band);
        let from_reduce = r_deeper.saturating_mul(2).saturating_add(4);
        r_deeper = own.max(from_reduce);
    }
    r_deeper
}

/// Safety factor applied by [`recommend_parallel`] on top of the raw
/// [`estimate_gpu_memory_bytes`] prediction. Covers:
///
/// - Per-call transient uploads (`src_dist` byte buffer ≈ +0.33×
///   `src_bytes`, per-band readback buffers when callers use
///   `compute_dkl_*_bands`).
/// - cubecl runtime metadata + page alignment per buffer (~hundreds
///   of bytes × ~50 allocations = single-digit MB).
/// - NVRTC PTX cache + kernel-module memory on the CUDA runtime
///   (one-time ~50-100 MB per process; amortizes across instances).
/// - Driver-side reservations that aren't reported by
///   `cudaMemGetInfo` as "free" but are also not visible as "used".
///
/// `1.5` is conservative for typical sweep workloads (one-shot
/// `score` calls). Tighten to ~1.2 when the caller batches with
/// `warm_reference` (no per-DIST allocator churn). Loosen to ~2.0
/// if the calling process also runs CPU-side decode/encode in the
/// same memory namespace.
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::{PARALLEL_SAFETY_FACTOR, estimate_gpu_memory_bytes, recommend_parallel};
///
/// // The safety factor sits in [1.0, 3.0] — below 1.0 leaves no
/// // transient slack; above 3.0 wastes GPU memory. Pinned to the
/// // documented value 1.5 by tests/pipeline_score.rs.
/// assert_eq!(PARALLEL_SAFETY_FACTOR, 1.5);
///
/// // Worked example: budget = free / (safety × per-instance estimate).
/// // For an 8 GB GPU at 1024² scoring, the formula matches
/// // `recommend_parallel`'s own answer.
/// let free = 8_u64 * 1024 * 1024 * 1024;
/// let est = estimate_gpu_memory_bytes(1024, 1024).unwrap() as f64;
/// let manual = (free as f64 / (PARALLEL_SAFETY_FACTOR * est)).floor() as u32;
/// let helper = recommend_parallel(free, 1024, 1024);
/// assert_eq!(manual, helper);
/// ```
pub const PARALLEL_SAFETY_FACTOR: f64 = 1.5;

/// Recommend a `PARALLEL` instance count for running many
/// [`Cvvdp::new`] instances against a shared GPU. Combines
/// [`estimate_gpu_memory_bytes`] with [`PARALLEL_SAFETY_FACTOR`]
/// so callers don't have to maintain the safety constant
/// themselves (or forget it).
///
/// Inputs:
/// - `free_gpu_bytes`: free GPU memory in bytes, typically from
///   `cudaMemGetInfo` (or `wgpu::Limits::max_buffer_size` for the
///   wgpu backend). Pass `0` to get a definitive 0 (caller does
///   the boundary handling).
/// - `(width, height)`: target image dimensions.
///
/// Returns the maximum number of `Cvvdp` instances that should fit
/// concurrently, capped at `u32::MAX`. Always returns at least 1
/// if the image dimensions are valid (a single instance always
/// gets to run; OOM after that is the caller's signal to back off
/// to host-pool or a smaller image).
///
/// Returns 0 only when:
/// - `(width, height)` is below the [`PYRAMID_MIN_DIM`] × 2
///   threshold (same as `Cvvdp::new` would reject), OR
/// - `free_gpu_bytes` is literally 0 (no memory to allocate against).
///
/// # Examples
///
/// ```
/// use cvvdp_gpu::pipeline::recommend_parallel;
///
/// // RTX 3070 (8 GB) running 1024² scoring:
/// let p = recommend_parallel(8 * 1024 * 1024 * 1024, 1024, 1024);
/// // At 1 MP per instance after the 1.5× safety factor, 8 GB
/// // fits 10-40 concurrent. The wide band is the safety factor's
/// // play vs. the actual per-instance cost (~330 MB at 1 MP).
/// // Pinned by `recommend_parallel_matches_documented_examples`
/// // in tests/pipeline_score.rs.
/// assert!((10..=40).contains(&p));
///
/// // 24 GB / 12 MP (4096×3072) — RTX 3090/4090 class:
/// let p_4090_12mp = recommend_parallel(24 * 1024 * 1024 * 1024, 4096, 3072);
/// // 12 MP per instance is ~2.5 GB after safety, so 24 GB fits ~3-10.
/// assert!((3..=10).contains(&p_4090_12mp));
/// ```
#[must_use]
pub fn recommend_parallel(free_gpu_bytes: u64, width: u32, height: u32) -> u32 {
    if free_gpu_bytes == 0 {
        return 0;
    }
    let Some(est) = estimate_gpu_memory_bytes(width, height) else {
        return 0;
    };
    if est == 0 {
        return 0;
    }
    let budgeted = free_gpu_bytes as f64 / (PARALLEL_SAFETY_FACTOR * est as f64);
    // Round down to the nearest integer, clamp to [1, u32::MAX].
    // Returning 0 when budget < 1 would mask the per-instance
    // overrun; the caller should see "1 instance, may OOM" and
    // back off explicitly rather than treat 0 as "no work".
    (budgeted.floor() as u32).max(1)
}

impl<R: Runtime> Cvvdp<R> {
    /// Allocate GPU buffers for a fixed `width × height` image and the
    /// given parameter bundle. Uses
    /// [`crate::params::DisplayGeometry::STANDARD_4K`] as the viewing
    /// geometry — equivalent to `new_with_geometry(..., STANDARD_4K)`.
    /// Override via `new_with_geometry` for non-4K displays.
    ///
    /// **Of the `params` fields, only `display` and `perf_mode` are
    /// consumed** — the `csf`/`masking`/`pooling`/`jod` sub-bundles
    /// are unused because the per-stage cvvdp v0.5.4 numbers are
    /// inlined as `const`s in the kernels module. `perf_mode` is
    /// stored on `Self` so future stage-level fast paths can gate on
    /// it; today it's a no-op (Strict and Fast produce identical
    /// output). Pass [`CvvdpParams::PLACEHOLDER`] unless you
    /// specifically need to override the display model or opt into
    /// `PerfMode::Fast`. See the `CvvdpParams::PLACEHOLDER` and
    /// [`crate::params::PerfMode`] docstrings for the full picture.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`PYRAMID_MIN_DIM`] × 2 (no usable pyramid),
    /// or if a GPU buffer allocation / kernel dispatch fails.
    ///
    /// # Examples
    ///
    /// `ignore` — needs a live GPU runtime; docs.rs builds in a
    /// sandbox without one. The runtime-test counterpart lives in
    /// `tests/pipeline_score.rs`.
    ///
    /// ```ignore
    /// use cubecl::{Runtime, cuda::CudaRuntime};
    /// use cvvdp_gpu::{Cvvdp, CvvdpParams};
    ///
    /// let client = CudaRuntime::client(&Default::default());
    /// // Construct for 1 MP; PLACEHOLDER selects STANDARD_4K display
    /// // and PerfMode::Strict (the parity-calibrated baseline).
    /// let cvvdp = Cvvdp::<CudaRuntime>::new(client, 1024, 1024, CvvdpParams::PLACEHOLDER)?;
    /// # Ok::<(), cvvdp_gpu::Error>(())
    /// ```
    pub fn new(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
    ) -> Result<Self> {
        Self::new_with_geometry(
            client,
            width,
            height,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
        )
    }

    /// Unified [`MemoryMode`](crate::MemoryMode) constructor.
    ///
    /// - `Full` → standard full-image pipeline ([`Self::new`]).
    /// - `Strip { h_body }` → Mode E pipeline ([`Self::new_strip`]).
    ///   `h_body = None` resolves to
    ///   [`crate::memory_mode::STRIP_H_BODY_DEFAULT`].
    /// - `Auto` → Full when it fits the cap, else Strip with the
    ///   default `h_body`.
    ///
    /// See [`crate::memory_mode`] module docs for the mode-E
    /// JOD-preservation rationale.
    pub fn new_with_memory_mode(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        use crate::memory_mode::{resolve_auto, vram_cap_bytes, ResolvedMode, STRIP_H_BODY_DEFAULT};
        use crate::MemoryMode;
        match mode {
            MemoryMode::Full => Self::new(client, width, height, params),
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or(STRIP_H_BODY_DEFAULT);
                Self::new_strip(client, width, height, body, params)
            }
            MemoryMode::StripPair { h_body } => {
                let body = h_body.unwrap_or(STRIP_H_BODY_DEFAULT);
                Self::new_strip_pair(client, width, height, body, params)
            }
            MemoryMode::CappedPyramid { levels } => {
                Self::new_capped_pyramid(client, width, height, params, levels)
            }
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                match resolve_auto(width, height, cap)? {
                    ResolvedMode::Full => Self::new(client, width, height, params),
                    ResolvedMode::Strip { h_body } => {
                        Self::new_strip(client, width, height, h_body, params)
                    }
                }
            }
        }
    }

    /// Configured image `(width, height)`. Matches the values passed
    /// to [`Self::new`] / [`Self::new_with_geometry`].
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Internal accessor used by [`crate::opaque::CvvdpOpaque`] to
    /// fetch the construction-time PPD without reaching into the
    /// private `geometry` field. Same value returned by
    /// `self.geometry.pixels_per_degree()`; provided as a stable
    /// in-crate API surface so [`crate::opaque`] never imports
    /// `crate::params::DisplayGeometry` paths.
    #[doc(hidden)]
    pub fn geometry_ppd_for_warm_ref(&self) -> f32 {
        self.geometry.pixels_per_degree()
    }

    /// Allocate GPU buffers + record a custom viewing geometry. The
    /// geometry is used by `score` to derive PPD (and thus the
    /// per-band spatial frequencies the CSF table is queried with).
    ///
    /// Same `params` caveat as [`Cvvdp::new`]: only `params.display`
    /// and `params.perf_mode` are consumed; the
    /// `csf`/`masking`/`pooling`/`jod` sub-bundles are ignored
    /// (per-stage numbers inlined as `const`s in the kernels
    /// module). See [`crate::params::PerfMode`] for the parity-vs-perf
    /// opt-in.
    ///
    /// # Examples
    ///
    /// Construct against a phone-class geometry (1080p at 0.4 m
    /// viewing distance on a 5.5″ panel — ≈340 ppd vs STANDARD_4K's
    /// ≈75 ppd). `ignore` for the same reason as [`Cvvdp::new`]'s
    /// doctest: docs.rs has no GPU and the no-default-features build
    /// path doesn't include any `cubecl` runtime feature.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    ///
    /// let client = Backend::client(&Default::default());
    /// let phone = DisplayGeometry {
    ///     resolution_w: 1920,
    ///     resolution_h: 1080,
    ///     distance_m: 0.40,
    ///     diagonal_inches: 5.5,
    /// };
    /// let cvvdp = Cvvdp::<Backend>::new_with_geometry(
    ///     client, 64, 64, CvvdpParams::PLACEHOLDER, phone,
    /// )
    /// .expect("Cvvdp::new_with_geometry");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`PYRAMID_MIN_DIM`] × 2 (no usable pyramid),
    /// or if a GPU buffer allocation / kernel dispatch fails.
    pub fn new_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        Self::new_with_geometry_inner(client, width, height, params, geometry, None, None)
    }

    /// Capped-pyramid variant (Option B safety net). Constructs a
    /// scorer whose pyramid depth is clamped to `levels` from above,
    /// in addition to the natural per-image depth derived from the
    /// viewing geometry. **NOT JOD-bit-identical** to [`Self::new`] —
    /// see [`crate::MemoryMode::CappedPyramid`] for the metric-value
    /// tradeoff. Uses the standard 4K viewing geometry; for a custom
    /// geometry use [`Self::new_capped_pyramid_with_geometry`].
    ///
    /// `levels >= 1` required; values larger than the natural pyramid
    /// depth are silently clamped (the natural depth is the upper
    /// bound).
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidImageSize`] if either dimension is below the
    ///   [`PYRAMID_MIN_DIM`] × 2 minimum.
    /// - [`Error::ModeUnsupported`] if `levels == 0`.
    pub fn new_capped_pyramid(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        levels: u32,
    ) -> Result<Self> {
        Self::new_capped_pyramid_with_geometry(
            client,
            width,
            height,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
            levels,
        )
    }

    /// Capped-pyramid variant with custom display geometry. See
    /// [`Self::new_capped_pyramid`] for the metric-value tradeoff and
    /// [`Self::new_with_geometry`] for the display-geometry semantics.
    ///
    /// # Errors
    ///
    /// Same as [`Self::new_capped_pyramid`].
    pub fn new_capped_pyramid_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
        levels: u32,
    ) -> Result<Self> {
        if levels == 0 {
            return Err(Error::ModeUnsupported("CappedPyramid { levels=0 }"));
        }
        Self::new_with_geometry_inner(client, width, height, params, geometry, Some(levels), None)
    }

    /// Inner constructor that backs both
    /// [`Self::new_with_geometry`] and
    /// [`Self::new_capped_pyramid_with_geometry`]. `cap_levels` clamps
    /// the pyramid depth from above; `None` selects the natural depth
    /// (the standard [`Self::new`] behaviour).
    ///
    /// `strip_pair_h_body` is `Some(h_body)` when the caller is
    /// constructing for `StripMode::Pair` (Mode B). The Phase 1b
    /// allocator uses this to size the per-strip
    /// `weber_scratch[k].upscaled_c_strip` buffer; passing `None`
    /// skips that allocation and the existing full-image
    /// `upscaled_c` is used. The strip_config field itself is still
    /// set by the calling `new_strip_pair_with_geometry`.
    fn new_with_geometry_inner(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
        cap_levels: Option<u32>,
        strip_pair_h_body: Option<u32>,
    ) -> Result<Self> {
        if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
            return Err(Error::InvalidImageSize);
        }
        let natural_n_levels = pyramid_levels(geometry.pixels_per_degree(), width, height);
        let n_levels = match cap_levels {
            Some(c) => natural_n_levels.min(c).max(1),
            None => natural_n_levels,
        };

        let n0 = (width as usize) * (height as usize);
        // Source-byte buffers are u32-slot arrays of length `n0 * 3`
        // T4.L (2026-05-16): pack 3 sRGB bytes per pixel into one u32
        // for upload (R | G<<8 | B<<16). Length = n0, not n0*3.
        let src_ref = client.create_from_slice(u32::as_bytes(&vec![0u32; n0]));
        // T_x.O (2026-05-17): the per-call pack now writes directly
        // into a pinned staging buffer reserved via
        // `client.reserve_staging` (see `_dispatch_dkl_planes_gpu`),
        // so the persistent `src_u32_scratch: Vec<u32>` is gone.
        let srgb_lut = client.create_from_slice(f32::as_bytes(&SRGB8_TO_LINEAR_LUT));

        let build_pyramid = |client: &ComputeClient<R>| -> Vec<Level> {
            let mut out = Vec::with_capacity(n_levels as usize);
            let mut w = width;
            let mut h = height;
            for _ in 0..n_levels {
                let n = (w as usize) * (h as usize);
                out.push(Level {
                    w,
                    h,
                    planes: [
                        alloc_zeros_f32(client, n),
                        alloc_zeros_f32(client, n),
                        alloc_zeros_f32(client, n),
                    ],
                });
                // Ceil-div halving — matches cvvdp's `gausspyr_reduce`
                // boundary semantics (tick 175). Was floor-div, which
                // caused 0.586 JOD drift vs pycvvdp at 4000×3000 due
                // to off-by-one shapes at levels 4+ on odd-dim inputs.
                w = w.div_ceil(2);
                h = h.div_ceil(2);
            }
            out
        };

        let gauss_ref = build_pyramid(&client);

        // P2.3 gauss_alt (2026-05-27): in StripMode::Pair, allocate a
        // second gauss pyramid so REF gauss data can survive past DIST
        // dispatch (see field docstring). Outside Mode B, `None`.
        //
        // P2.7 partial (2026-05-27): only SHALLOW levels are
        // full-image; deep levels are zero-size. The swap is shallow-
        // only (see `_maybe_swap_gauss_alt_post_ref`), so deep
        // gauss_alt is never read. This saves the deep-pyramid bytes
        // from gauss_alt at construction. The shallow full-image
        // allocation remains a temporary cost — a full P2.7 (strip-
        // shape shallow gauss_alt + strip-shape gauss_ref + per-strip
        // DIST gauss reduce) would retire it entirely, but that's a
        // major restructure beyond P2.x's mechanical-shrink scope.
        let gauss_alt: Option<Vec<Level>> = if let Some(h_body) = strip_pair_h_body {
            let k_split = mode_b_k_split(h_body, n_levels) as usize;
            let mut out = Vec::with_capacity(n_levels as usize);
            let mut w = width;
            let mut h = height;
            for k in 0..n_levels as usize {
                // The shallow REF strip helper for level k reads
                // `gauss_alt[k+1]` (coarser source for the upscale).
                // For the last shallow level `k = k_split - 1`, that's
                // `gauss_alt[k_split]` — so gauss_alt needs full-image
                // at level `k_split` too. Allocate full-image for
                // `k <= k_split`; zero-size for `k > k_split` (those
                // are unread post-shallow-swap because REF deep
                // weber finalize completes before swap and DIST deep
                // reads `gauss_ref` directly).
                let n_alloc = if k <= k_split {
                    (w as usize) * (h as usize)
                } else {
                    0
                };
                out.push(Level {
                    w,
                    h,
                    planes: [
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                    ],
                });
                w = w.div_ceil(2);
                h = h.div_ceil(2);
            }
            Some(out)
        } else {
            None
        };

        // P2.3 bands_ref shrink (2026-05-27): in StripMode::Pair,
        // allocate non-baseband levels k<k_split as ZERO-SIZE handles
        // (the strip-major outer in the band loop reads bands_ref
        // from per-strip `weber_scratch[k].bands_ref_strip` instead).
        // Deep levels (k >= k_split, non-baseband) and the baseband
        // keep full-image allocations because the level-major dispatch
        // path reads them at full level dims. Mirrors the bands_dis
        // skip pattern (2026-05-26).
        //
        // Full / CachedRef modes use `build_pyramid` (every level
        // full-image-sized).
        let bands_ref: Vec<Level> = if let Some(h_body) = strip_pair_h_body {
            let k_split = mode_b_k_split(h_body, n_levels);
            let mut out = Vec::with_capacity(n_levels as usize);
            let mut w = width;
            let mut h = height;
            for k in 0..n_levels as usize {
                let is_baseband = k == n_levels as usize - 1;
                let is_shallow = (k as u32) < k_split;
                let n_alloc = if !is_baseband && is_shallow {
                    0
                } else {
                    (w as usize) * (h as usize)
                };
                out.push(Level {
                    w,
                    h,
                    planes: [
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                    ],
                });
                w = w.div_ceil(2);
                h = h.div_ceil(2);
            }
            out
        } else {
            build_pyramid(&client)
        };

        // Path A bands_dis shrink (2026-05-26): in StripMode::Pair,
        // allocate non-baseband levels' planes as ZERO-SIZE handles.
        // The actual DIST data flows through the per-strip
        // `WeberScratch.bands_dis_strip` buffer + fused csf dispatch
        // inside the Mode B band loop (Weber strip writes
        // bands_dis_strip → csf strip reads bands_dis_strip → writes
        // t_p_*; next strip overwrites bands_dis_strip — the buffer
        // never holds more than one strip at a time, so the band
        // loop can't read at full level dims and instead dispatches
        // csf per-strip in lockstep with the Weber walker).
        //
        // The baseband level keeps a full-image allocation — the
        // baseband-divide kernel writes the coarsest-level DIST data
        // there (no strip walker at the baseband) and the band loop
        // reads it at baseband resolution via the standard
        // csf_apply_6ch path.
        //
        // Full / CachedRef modes use `build_pyramid` (every level
        // full-image-sized).
        let bands_dis: Vec<Level> = if strip_pair_h_body.is_some() {
            let mut out = Vec::with_capacity(n_levels as usize);
            let mut w = width;
            let mut h = height;
            for k in 0..n_levels as usize {
                let is_baseband = k == n_levels as usize - 1;
                let n_alloc = if is_baseband {
                    (w as usize) * (h as usize)
                } else {
                    0
                };
                out.push(Level {
                    w,
                    h,
                    planes: [
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                        alloc_zeros_f32(&client, n_alloc),
                    ],
                });
                w = w.div_ceil(2);
                h = h.div_ceil(2);
            }
            out
        } else {
            build_pyramid(&client)
        };
        let d_scratch = build_d_bands_scratch(
            &client,
            n_levels as usize,
            width,
            height,
            strip_pair_h_body,
        );
        let weber_scratch =
            build_weber_scratch(&client, n_levels as usize, width, height, strip_pair_h_body);

        // Baseband log_l_bkg buffer. Size matches `gauss_ref[last]`
        // which `build_pyramid` allocated with ceil-div halving
        // (tick 175). Allocated once; filled per-JOD via `fill_f32_kernel`.
        let last = n_levels as usize - 1;
        let baseband_w = gauss_ref[last].w as usize;
        let baseband_h = gauss_ref[last].h as usize;
        let baseband_n = baseband_w * baseband_h;
        let baseband_log_l_bkg = alloc_zeros_f32(&client, baseband_n);

        // Persistent `n_levels * N_CHANNELS` atomic-pool partials buffer
        // — reused across every `_pool_and_finalize_jod` call. Zero-fill
        // happens per call via `fill_f32_kernel`; the per-call
        // `create_from_slice` host alloc + upload is eliminated (tick 227).
        let partials_h = alloc_zeros_f32(&client, (n_levels as usize) * N_CHANNELS);

        // Persistent `Vec<Handle>` slices for the per-non-baseband
        // log_l_bkg destinations. _dispatch_*_weber_pyramid_only used
        // to .collect() these per call; pre-building once eliminates
        // a small but real per-JOD allocator round-trip plus a handle
        // ref-bump per level. Tick 240.
        let log_l_bkg_ref_dests: Vec<cubecl::server::Handle> =
            weber_scratch.iter().map(|s| s.log_l_bkg.clone()).collect();
        let log_l_bkg_dis_dests: Vec<cubecl::server::Handle> = weber_scratch
            .iter()
            .map(|s| s.log_l_bkg_dis.clone())
            .collect();

        // Pre-upload logs_row per (level, channel) — depends only on
        // (rho_k, channel) which are fixed for this Cvvdp.
        //
        // Tick 204 fix: the baseband uses `CSF_BASEBAND_RHO` (0.1
        // cy/deg) instead of the geometric `band_frequencies` value,
        // matching pycvvdp `process_block_of_frames`'s
        // `rho_band[bb] = 0.1` override at the last band
        // (cvvdp_metric.py:628). This closes the chroma_shift drift
        // tracked through ticks 191-203 — the baseband S lookup for
        // chromatic channels (RG, VY) at the LOW-rho regime is what
        // gives the chroma response its right magnitude.
        let ppd = geometry.pixels_per_degree();
        let freqs = band_frequencies(ppd, width as usize, height as usize);
        let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];
        let mut logs_row: Vec<[cubecl::server::Handle; N_CHANNELS]> =
            Vec::with_capacity(n_levels as usize);
        for k in 0..n_levels as usize {
            let is_baseband = k == n_levels as usize - 1;
            let rho_k = if is_baseband {
                crate::kernels::csf::CSF_BASEBAND_RHO
            } else {
                freqs[k]
            };
            logs_row.push([
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(rho_k, channels[0]))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(rho_k, channels[1]))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(rho_k, channels[2]))),
            ]);
        }

        Ok(Self {
            client,
            params,
            geometry,
            width,
            height,
            n_levels,
            src_ref,
            srgb_lut,
            gauss_ref,
            gauss_alt,
            bands_ref,
            bands_dis,
            d_scratch,
            weber_scratch,
            baseband_log_l_bkg,
            partials_h,
            log_l_bkg_ref_dests,
            log_l_bkg_dis_dests,
            logs_row,
            cached: None,
            warm_ref_baseband_log_l_bkg: None,
            diffmap_scratch: None,
            linear_planes_upload: None,
            strip_config: None,
            ref_full_state: None,
            strip_dispatch_counter: core::sync::atomic::AtomicU32::new(0),
        })
    }

    /// Allocate GPU buffers for strip-mode (Mode E) processing.
    ///
    /// In strip mode:
    /// - One-shot scoring ([`Self::score`]) still runs the standard
    ///   full-image dispatch — same memory profile as
    ///   [`Self::new`]. Strip mode does NOT change the one-shot path
    ///   because that path's working set IS the dist working set
    ///   that mode E aims to shrink (and we can't shrink it without
    ///   the strip walker, which is meaningful only for cached-ref).
    /// - Cached-ref scoring ([`Self::set_reference_strip`] +
    ///   [`Self::compute_with_cached_reference_strip`]) lands the
    ///   ref-side state into a dedicated [`RefFullState`]
    ///   (full-image-sized, populated once), then walks the dist
    ///   side in `(h_body + halo)` strips.
    ///
    /// **Phase 1 landing (task #79):** the constructor allocates the
    /// strip config + dedicated ref-state storage. The strip-walker
    /// dispatch itself (Phase 3) is not yet wired —
    /// `compute_with_cached_reference_strip` surfaces a
    /// [`Error::ModeUnsupported`] until Phase 3 lands. Callers can
    /// detect this at construction time via [`Self::is_strip_mode`]
    /// and fall back to Full mode at the application layer.
    ///
    /// `h_body` must be a positive power of two so the per-level
    /// halving in the strip walker halves cleanly. See
    /// [`crate::memory_mode::STRIP_H_BODY_DEFAULT`] for the recommended
    /// default.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`crate::PYRAMID_MIN_DIM`] × 2, or
    /// [`Error::ModeUnsupported`] if `h_body` is zero or not a power of two.
    pub fn new_strip(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        h_body: u32,
        params: CvvdpParams,
    ) -> Result<Self> {
        Self::new_strip_with_geometry(
            client,
            width,
            height,
            h_body,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
        )
    }

    /// Strip-mode constructor with a custom display geometry. See
    /// [`Self::new_strip`] for the strip-mode semantics and
    /// [`Self::new_with_geometry`] for the display-geometry semantics.
    pub fn new_strip_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        h_body: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        if !is_valid_strip_h_body(h_body) {
            return Err(Error::ModeUnsupported("Strip { h_body=invalid }"));
        }
        let mut this = Self::new_with_geometry(client, width, height, params, geometry)?;
        this.strip_config = Some(StripConfig {
            h_body,
            mode: StripMode::CachedRef,
        });
        Ok(this)
    }

    /// Allocate GPU buffers for Mode B (StripPair) one-shot strip-pair
    /// processing. Both ref and dist sides walk in strips together; no
    /// full ref cache is kept on device.
    ///
    /// Use this when scoring one (ref, dist) pair at a time without a
    /// batch workflow — peak memory ≈ 2× per-strip working set, which
    /// is the right tradeoff for CLI / one-shot scoring on large
    /// images. For batch workloads with many DISTs per REF, prefer
    /// [`Self::new_strip`] (Mode E) so the REF pyramid is built once.
    ///
    /// `h_body` must be a positive power of two so the per-level
    /// halving in the strip walker halves cleanly. Uses the standard
    /// 4K viewing geometry; for a custom geometry use
    /// [`Self::new_strip_pair_with_geometry`].
    ///
    /// # Errors
    ///
    /// Same as [`Self::new_strip`].
    pub fn new_strip_pair(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        h_body: u32,
        params: CvvdpParams,
    ) -> Result<Self> {
        Self::new_strip_pair_with_geometry(
            client,
            width,
            height,
            h_body,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
        )
    }

    /// Mode B (StripPair) constructor with a custom display geometry.
    /// See [`Self::new_strip_pair`] for the strip-mode semantics.
    pub fn new_strip_pair_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        h_body: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        if !is_valid_strip_h_body(h_body) {
            return Err(Error::ModeUnsupported("StripPair { h_body=invalid }"));
        }
        // Route through `new_with_geometry_inner` with
        // `strip_pair_h_body = Some(h_body)` so the Phase 1b allocator
        // sizes `weber_scratch[k].upscaled_c_strip` per-strip rather
        // than full-image. The `strip_config` field is still set
        // post-construction below so existing call sites that check
        // `strip_config` for Mode B dispatch behave identically.
        let mut this = Self::new_with_geometry_inner(
            client, width, height, params, geometry, None, Some(h_body),
        )?;
        this.strip_config = Some(StripConfig {
            h_body,
            mode: StripMode::Pair,
        });
        Ok(this)
    }

    /// `true` if this scorer was built for Mode B (StripPair) one-shot
    /// strip-pair processing. See [`Self::new_strip_pair`].
    pub fn is_strip_pair_mode(&self) -> bool {
        matches!(
            self.strip_config,
            Some(StripConfig {
                mode: StripMode::Pair,
                ..
            })
        )
    }

    /// `true` if this scorer was built for strip-mode processing.
    /// See [`Self::new_strip`].
    pub fn is_strip_mode(&self) -> bool {
        self.strip_config.is_some()
    }

    /// Strip-mode dist body height in rows, or `None` if not in strip
    /// mode. Returns the value passed to [`Self::new_strip`] (or
    /// resolved by [`Self::new_with_memory_mode`]).
    pub fn strip_h_body(&self) -> Option<u32> {
        self.strip_config.as_ref().map(|c| c.h_body)
    }

    /// `true` if a warm reference state is currently cached on device,
    /// `false` if [`Self::warm_reference`] has not been called or the
    /// warm state was invalidated by an intervening REF-dispatching
    /// method (see [`Self::warm_reference`] for the invalidation
    /// contract).
    ///
    /// In strip mode the warm state lives in the dedicated
    /// [`RefFullState`] buffers — which survive intervening
    /// dispatches because the shared scratch (which gets clobbered)
    /// isn't where strip-mode's cached state lives. So strip-mode
    /// `has_warm_reference()` stays `true` as long as
    /// [`Self::warm_reference`] was called at least once on this
    /// instance.
    pub fn has_warm_reference(&self) -> bool {
        if self.strip_config.is_some() {
            self.ref_full_state.is_some()
        } else {
            self.warm_ref_baseband_log_l_bkg.is_some()
        }
    }

    /// Test-only accessor for the Mode E Phase 3 strip-iteration
    /// counter. Returns the cumulative number of pool-strip
    /// dispatches `_pool_and_finalize_jod_strip` has launched since
    /// this `Cvvdp` was constructed.
    ///
    /// Used by `tests/strip_mode_e_phase3.rs` to confirm the walker
    /// actually partitioned at large sizes (N >= 2) rather than
    /// dropping back to a single Full-mode dispatch. Not part of the
    /// stable public API — exposed via `#[doc(hidden)]`.
    #[doc(hidden)]
    pub fn strip_dispatch_counter(&self) -> u32 {
        self.strip_dispatch_counter
            .load(core::sync::atomic::Ordering::Relaxed)
    }

    /// Test-only counter reset. Mirrors
    /// [`Self::strip_dispatch_counter`]'s semantics; allows the
    /// parity test to zero the counter between sub-tests without
    /// reconstructing the scorer.
    #[doc(hidden)]
    pub fn reset_strip_dispatch_counter(&self) {
        self.strip_dispatch_counter
            .store(0, core::sync::atomic::Ordering::Relaxed);
    }

    /// Pyramid level `k`'s spatial dimensions as `(bw, bh, n_px)`.
    /// Per-level (bw, bh, n_px) for the GPU pyramid. Reads the actual
    /// stored `gauss_ref[k]` dimensions which since tick 175 use
    /// ceil-div halving (`(n + 1) / 2`) matching pycvvdp's
    /// `gausspyr_reduce`. Was previously `width >> k` (floor-div bit
    /// shift), which silently underprocessed ~1 row per odd-dim level
    /// in the band loop.
    fn level_dims(&self, k: usize) -> (usize, usize, usize) {
        let bw = self.gauss_ref[k].w as usize;
        let bh = self.gauss_ref[k].h as usize;
        (bw, bh, bw * bh)
    }

    /// Debug-only sanity check that the caller-passed `ppd` matches
    /// the geometry baked into this `Cvvdp` at construction.
    ///
    /// Several public methods (`compute_dkl_jod`, `compute_dkl_d_bands`,
    /// `compute_dkl_t_p_bands`, `compute_dkl_jod_with_warm_ref`, plus
    /// the host_pool variants and `compute_dkl_csf_weighted_bands`)
    /// take a `ppd: f32` parameter that the implementation
    /// **silently ignores** — `logs_row` is pre-uploaded at
    /// construction time against `self.geometry.pixels_per_degree()`,
    /// so passing a different ppd does NOT re-tune the CSF lookup.
    /// The parameter remains in the signatures for source-
    /// compatibility (changing it would break the public API).
    ///
    /// Pre-tick-243 there was no surfaced sanity check: a caller who
    /// constructed Cvvdp with `STANDARD_4K` (75.4 PPD) then called
    /// `compute_dkl_jod(ref, dist, phone_ppd)` (110 PPD) would get
    /// results scored against 75.4 PPD with no warning. Tick 243
    /// adds a debug_assert at the public boundary so the mismatch
    /// fires in debug builds. Release builds preserve the silent-
    /// ignore behavior to avoid changing observable semantics.
    ///
    /// Tolerance: `1e-3` PPD absolute — slack enough for f32-noise
    /// roundtrips (e.g. derived-from-geometry via
    /// `DisplayGeometry::pixels_per_degree()`) but tight enough to
    /// catch a caller passing PHONE PPD when STANDARD_4K is baked
    /// in.
    #[inline]
    fn debug_assert_ppd_matches_geometry(&self, ppd: f32) {
        debug_assert!(
            (ppd - self.geometry.pixels_per_degree()).abs() < 1e-3,
            "ppd={} mismatched with self.geometry.pixels_per_degree()={}; \
             the GPU CSF logs_row LUT is pre-uploaded against the geometry \
             at Cvvdp::new and the per-call ppd is silently ignored. \
             Reconstruct Cvvdp with `new_with_geometry` if you need a \
             different display geometry.",
            ppd,
            self.geometry.pixels_per_degree(),
        );
    }

    /// Run only the color stage: upload sRGB bytes, launch the
    /// `srgb_to_dkl_kernel`, and read back three planar `f32` buffers
    /// (A, RG, VY) in row-major order.
    ///
    /// Used by integration tests + downstream stages that consume DKL
    /// planes. Equivalent to:
    ///
    /// ```text
    /// for pixel in srgb { srgb_byte_to_dkl_scalar(pixel, display) }
    /// ```
    ///
    /// but executed on the GPU.
    ///
    /// # Examples
    ///
    /// Read back the three DKL planes (A, RG, VY) for a 64×64
    /// mid-gray buffer. `ignore` because docs.rs has no GPU and
    /// the no-default-features build path doesn't resolve cubecl
    /// runtime types.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let [a, rg, vy] = cvvdp.compute_dkl_planes(&srgb).expect("compute_dkl_planes");
    /// assert_eq!(a.len(), (w * h) as usize);
    /// assert_eq!(rg.len(), (w * h) as usize);
    /// assert_eq!(vy.len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails.
    pub fn compute_dkl_planes(&mut self, srgb: &[u8]) -> Result<[Vec<f32>; 3]> {
        self._dispatch_dkl_planes_gpu(srgb)?;

        let a_handle = self.gauss_ref[0].planes[0].clone();
        let rg_handle = self.gauss_ref[0].planes[1].clone();
        let vy_handle = self.gauss_ref[0].planes[2].clone();

        let a_bytes = self
            .client
            .read_one(a_handle)
            .map_err(|_| Error::InvalidImageSize)?;
        let rg_bytes = self
            .client
            .read_one(rg_handle)
            .map_err(|_| Error::InvalidImageSize)?;
        let vy_bytes = self
            .client
            .read_one(vy_handle)
            .map_err(|_| Error::InvalidImageSize)?;

        Ok([
            f32::from_bytes(&a_bytes).to_vec(),
            f32::from_bytes(&rg_bytes).to_vec(),
            f32::from_bytes(&vy_bytes).to_vec(),
        ])
    }

    /// Dispatch-only version of `compute_dkl_planes`: uploads sRGB
    /// bytes, launches the color kernel, leaves `gauss_ref[0].planes[c]`
    /// populated on GPU. No host readback. Internal helper used by
    /// `compute_dkl_planes` and by downstream pipeline stages that
    /// only need the GPU handles (gauss pyramid, weber pyramid).
    fn _dispatch_dkl_planes_gpu(&mut self, srgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        let n0 = (self.width as usize) * (self.height as usize);

        // T_x.O (2026-05-17): pack u8×3 → u32 directly into the
        // pinned staging buffer (one host-side pass instead of two).
        // Previously we packed into `self.src_u32_scratch` and then
        // `create_from_slice_pinned` copied that scratch into a
        // pinned buffer — two full ~48 MB host writes for the same
        // data at 12 MP. `reserve_staging` lets us produce the
        // packed bytes straight into the pinned buffer.
        //
        // Layout (unchanged from T4.L): 4 bytes per pixel — R | G<<8
        // | B<<16 (alpha unused). Reader
        // (`srgb_to_dkl_kernel`) sees the same `[u32]` packing.
        let pinned_len = n0 * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(srgb.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        // T4.M (2026-05-16): pinned-host fast path — direct DMA
        // (12-25 GB/s on PCIe 4.0 vs 5-6 GB/s from pageable).
        // T_x.O: skipping the per-call scratch intermediate saves
        // one ~48 MB host write per upload.
        self.src_ref = self.client.create(bytes);

        self._launch_srgb_to_dkl_from_src_ref();
        Ok(())
    }

    /// Dispatch the sRGB→DKL color kernel reading from whatever
    /// packed-u32 handle currently sits in `self.src_ref`. Split out
    /// of [`Self::_dispatch_dkl_planes_gpu`] so [`Self::compute_handles`]
    /// (Phase 4 upload-once path) can reuse the dispatch step without
    /// re-uploading bytes.
    ///
    /// In Mode B (StripPair), the dispatch is **partitioned into
    /// `ceil(height / h_body)` row strips** using `Handle::offset_start`
    /// to slice both `src_ref` and the level-0 output planes per strip.
    /// `srgb_to_dkl_kernel` is pointwise (`out[idx] = f(src[idx])`), so
    /// dispatching it over disjoint row ranges produces bit-identical
    /// output to a single full-image launch. The strip-walked dispatch
    /// increments [`Self::strip_dispatch_counter`] by one per outer
    /// strip iteration, proving the Mode B walker partitioned the work
    /// rather than bypassing to Full. Cross-strip data dependencies in
    /// the downstream Gauss / Weber / Masking stages are handled at
    /// those stages' dispatchers — DKL itself has no cross-pixel
    /// reads so per-strip dispatch is always safe.
    fn _launch_srgb_to_dkl_from_src_ref(&self) {
        let display = self.params.display;
        let (eotf_tag, gamma_exp) =
            crate::kernels::color::eotf_tag_and_gamma(display.eotf);
        let hlg_gamma =
            crate::params::hlg_system_gamma(display.y_peak, display.e_ambient_lux);
        let m = display.primaries.linear_rgb_to_dkl();
        let cube_dim = CubeDim::new_1d(64);

        // Per-strip iteration: build (body_offset_y, body_h) slabs and
        // dispatch the same pointwise kernel against handles sliced
        // through `Handle::offset_start`. For Full mode the loop has
        // a single iteration covering the whole image (body_offset_y=0,
        // body_h=height).
        let strip_h_body = match self.strip_config {
            Some(StripConfig { mode: StripMode::Pair, h_body }) => h_body.min(self.height),
            _ => self.height,
        };
        let n_strips = self.height.div_ceil(strip_h_body);
        let w = self.width;

        for s in 0..n_strips {
            let body_offset_y = s * strip_h_body;
            let body_h = (self.height - body_offset_y).min(strip_h_body);
            let n_strip = (w as usize) * (body_h as usize);
            let byte_off: u64 = u64::from(body_offset_y) * u64::from(w) * 4;
            // For src_ref the packed-u32 layout is 4 bytes/pixel, same
            // as the f32 output planes — both indexed by row × width.
            let src_strip = self.src_ref.clone().offset_start(byte_off);
            let a_strip = self.gauss_ref[0].planes[0].clone().offset_start(byte_off);
            let rg_strip = self.gauss_ref[0].planes[1].clone().offset_start(byte_off);
            let vy_strip = self.gauss_ref[0].planes[2].clone().offset_start(byte_off);

            let cube_count = CubeCount::Static((n_strip as u32).div_ceil(64), 1, 1);
            unsafe {
                srgb_to_dkl_kernel::launch::<R>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(src_strip, n_strip),
                    ArrayArg::from_raw_parts(self.srgb_lut.clone(), SRGB8_TO_LINEAR_LUT.len()),
                    ArrayArg::from_raw_parts(a_strip, n_strip),
                    ArrayArg::from_raw_parts(rg_strip, n_strip),
                    ArrayArg::from_raw_parts(vy_strip, n_strip),
                    w,
                    body_h,
                    display.y_peak,
                    display.y_black,
                    display.y_refl,
                    eotf_tag,
                    gamma_exp,
                    hlg_gamma,
                    m[0][0],
                    m[0][1],
                    m[0][2],
                    m[1][0],
                    m[1][1],
                    m[1][2],
                    m[2][0],
                    m[2][1],
                    m[2][2],
                );
            }

            // Mode B: count this DKL strip dispatch toward the walker's
            // iteration counter so the parity test can observe the
            // strip-by-strip partitioning. Full mode never enters this
            // branch (n_strips = 1 with strip_h_body = self.height,
            // but strip_config is None so we still skip the increment).
            if matches!(
                self.strip_config,
                Some(StripConfig { mode: StripMode::Pair, .. }),
            ) {
                self.strip_dispatch_counter
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Install a caller-supplied packed-u32 device handle as
    /// `self.src_ref` (the input slot read by
    /// [`Self::_launch_srgb_to_dkl_from_src_ref`]) and dispatch the
    /// color kernel. Internal helper for the upload-once
    /// `compute_handles` path.
    fn _install_src_ref_and_dispatch_dkl(&mut self, handle: &cubecl::server::Handle) {
        self.src_ref = handle.clone();
        self._launch_srgb_to_dkl_from_src_ref();
    }

    /// Validate three planar `W × H` linear-RGB f32 buffers and return
    /// the per-plane length. Used at the boundary of every
    /// `from_linear_planes*` method to give the caller a precise
    /// dimension-mismatch error if any plane is the wrong size.
    fn _validate_linear_planes(&self, r: &[f32], g: &[f32], b: &[f32]) -> Result<usize> {
        let expected = (self.width as usize) * (self.height as usize);
        for (label, plane) in [("r", r), ("g", g), ("b", b)] {
            if plane.len() != expected {
                let _ = label; // surfaced via DimensionMismatch
                return Err(Error::DimensionMismatch {
                    expected,
                    got: plane.len(),
                });
            }
        }
        Ok(expected)
    }

    /// Upload three planar `W × H` linear-RGB f32 buffers (unit-
    /// scaled sRGB primaries) into the lazy `linear_planes_upload`
    /// scratch and dispatch [`linear_rgb_planes_to_dkl_kernel`] into
    /// `self.gauss_ref[0].planes[c]` — the same output slot
    /// [`Self::_dispatch_dkl_planes_gpu`] writes to.
    ///
    /// Skips the sRGB→linear LUT lookup that the sRGB-byte path runs
    /// in `srgb_to_dkl_kernel`. Callers using this path MUST pre-
    /// linearise their RGB; the kernel reads each plane as
    /// already-linear-light. The display-model step (`y_peak`,
    /// `y_black`, `y_refl`) and the DKL matrix multiply still run on
    /// GPU.
    fn _dispatch_dkl_planes_gpu_from_linear_planes(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
    ) -> Result<()> {
        let n0 = self._validate_linear_planes(r, g, b)?;
        self._ensure_linear_planes_upload();

        // Upload R/G/B into the scratch buffers via the cubecl-
        // standard pinned-staging path. Reusing the existing handles
        // means each iteration overwrites the buffer rather than
        // allocating a fresh GPU buffer per call.
        let upload = self.linear_planes_upload.as_ref().expect("ensured above");
        let r_handle = upload.planes[0].clone();
        let g_handle = upload.planes[1].clone();
        let b_handle = upload.planes[2].clone();
        // create_from_slice replaces the prior contents of the slot.
        // (Same pattern as the sRGB path's `self.src_ref = self.client.create(bytes)`
        // line — cubecl handles dedicate-by-replace correctly for
        // long-lived slot bindings.)
        let new_r = self.client.create_from_slice(f32::as_bytes(r));
        let new_g = self.client.create_from_slice(f32::as_bytes(g));
        let new_b = self.client.create_from_slice(f32::as_bytes(b));
        // Update the scratch-slot bindings so subsequent calls see the
        // last-installed buffer. We can't just `.clone()` because
        // create_from_slice returns a brand-new handle.
        if let Some(upload_mut) = self.linear_planes_upload.as_mut() {
            upload_mut.planes[0] = new_r.clone();
            upload_mut.planes[1] = new_g.clone();
            upload_mut.planes[2] = new_b.clone();
        }
        let _ = (r_handle, g_handle, b_handle); // older handles drop here

        let a_handle = self.gauss_ref[0].planes[0].clone();
        let rg_handle = self.gauss_ref[0].planes[1].clone();
        let vy_handle = self.gauss_ref[0].planes[2].clone();

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n0 as u32).div_ceil(64), 1, 1);
        let display = self.params.display;
        let m = display.primaries.linear_rgb_to_dkl();
        unsafe {
            linear_rgb_planes_to_dkl_kernel::launch::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(new_r, n0),
                ArrayArg::from_raw_parts(new_g, n0),
                ArrayArg::from_raw_parts(new_b, n0),
                ArrayArg::from_raw_parts(a_handle, n0),
                ArrayArg::from_raw_parts(rg_handle, n0),
                ArrayArg::from_raw_parts(vy_handle, n0),
                self.width,
                self.height,
                display.y_peak,
                display.y_black,
                display.y_refl,
                m[0][0],
                m[0][1],
                m[0][2],
                m[1][0],
                m[1][1],
                m[1][2],
                m[2][0],
                m[2][1],
                m[2][2],
            );
        }
        Ok(())
    }

    /// Gaussian-pyramid build starting from linear-RGB planar input
    /// (instead of packed sRGB bytes). Mirrors
    /// [`Self::_dispatch_gauss_pyramid_gpu`].
    fn _dispatch_gauss_pyramid_gpu_from_linear_planes(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
    ) -> Result<()> {
        self._dispatch_dkl_planes_gpu_from_linear_planes(r, g, b)?;
        self._reduce_gauss_pyramid_from_level0();
        Ok(())
    }

    /// Weber-pyramid build from linear-RGB planar input. Mirrors
    /// [`Self::_dispatch_weber_pyramid_gpu`].
    fn _dispatch_weber_pyramid_gpu_from_linear_planes(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        log_l_bkg_dest: &[cubecl::server::Handle],
        dest_is_dis: bool,
    ) -> Result<f32> {
        self._dispatch_gauss_pyramid_gpu_from_linear_planes(r, g, b)?;
        self._finalize_weber_pyramid_after_gauss(log_l_bkg_dest, dest_is_dis)
    }

    /// REF weber pyramid only, from linear-RGB planar input. Mirrors
    /// [`Self::_dispatch_ref_weber_pyramid_only`].
    fn _dispatch_ref_weber_pyramid_only_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<f32> {
        self.warm_ref_baseband_log_l_bkg = None;
        let dests = std::mem::take(&mut self.log_l_bkg_ref_dests);
        let result =
            self._dispatch_weber_pyramid_gpu_from_linear_planes(ref_r, ref_g, ref_b, &dests, false);
        self.log_l_bkg_ref_dests = dests;
        self._maybe_swap_gauss_alt_post_ref();
        result
    }

    /// DIST weber pyramid only, from linear-RGB planar input. Mirrors
    /// [`Self::_dispatch_dist_weber_pyramid_only`].
    fn _dispatch_dist_weber_pyramid_only_from_linear_planes(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<()> {
        let dests = std::mem::take(&mut self.log_l_bkg_dis_dests);
        let result = self
            ._dispatch_weber_pyramid_gpu_from_linear_planes(dist_r, dist_g, dist_b, &dests, true);
        self.log_l_bkg_dis_dests = dests;
        result.map(|_| ())
    }

    /// Full D-bands dispatch from linear-RGB planar inputs. Mirrors
    /// [`Self::_dispatch_d_bands_into_scratch`] but uses the planar
    /// f32 entry points throughout.
    fn _dispatch_d_bands_into_scratch_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_ref = std::time::Instant::now();
        let log_l_bkg_baseband =
            self._dispatch_ref_weber_pyramid_only_from_linear_planes(ref_r, ref_g, ref_b)?;
        if trace {
            eprintln!("[trace] weber(ref):  {:?}", t_weber_ref.elapsed());
        }
        self._dispatch_d_bands_dist_and_band_loop_from_linear_planes(
            dist_r,
            dist_g,
            dist_b,
            log_l_bkg_baseband,
        )
    }

    /// DIST weber + band loop, from linear-RGB planar input. Mirrors
    /// [`Self::_dispatch_d_bands_dist_and_band_loop`].
    fn _dispatch_d_bands_dist_and_band_loop_from_linear_planes(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        log_l_bkg_baseband: f32,
    ) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_dis = std::time::Instant::now();
        self._dispatch_dist_weber_pyramid_only_from_linear_planes(dist_r, dist_g, dist_b)?;
        if trace {
            eprintln!("[trace] weber(dist): {:?}", t_weber_dis.elapsed());
        }
        self._run_d_bands_band_loop(log_l_bkg_baseband)
    }

    /// Run color stage + Gaussian-pyramid reduce loop. Returns the
    /// pyramid as `levels[k] = [a, rg, vy]` planar f32 vecs, with
    /// `levels[0]` at base resolution and each subsequent level
    /// halved (cvvdp's `div_ceil(2)` convention).
    ///
    /// # Examples
    ///
    /// Read back the gaussian pyramid for a 64×64 buffer; level 0 is
    /// 64×64 = 4096 pixels per channel, level 1 is 32×32 = 1024, etc.
    /// `ignore` because docs.rs has no GPU (same constraint as the
    /// other `Cvvdp::*` examples).
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let levels = cvvdp.compute_dkl_gauss_pyramid(&srgb).expect("compute_dkl_gauss_pyramid");
    /// assert!(!levels.is_empty());
    /// // levels[0] is base resolution: 64 * 64 = 4096 per channel.
    /// assert_eq!(levels[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails anywhere in the
    /// color → gauss-pyramid chain.
    pub fn compute_dkl_gauss_pyramid(&mut self, srgb: &[u8]) -> Result<Vec<[Vec<f32>; 3]>> {
        self._dispatch_gauss_pyramid_gpu(srgb)?;

        // Read back every level × every channel.
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(self.n_levels as usize);
        for k in 0..(self.n_levels as usize) {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.gauss_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Dispatch-only version of `compute_dkl_gauss_pyramid`: chains
    /// `_dispatch_dkl_planes_gpu` (color stage) with the per-level
    /// `downscale_kernel` reduce. Leaves `gauss_ref[k].planes[c]`
    /// populated on GPU for `k = 0..n_levels`. No host readback.
    fn _dispatch_gauss_pyramid_gpu(&mut self, srgb: &[u8]) -> Result<()> {
        self._dispatch_dkl_planes_gpu(srgb)?;
        self._reduce_gauss_pyramid_from_level0();
        Ok(())
    }

    /// Reduce the Gaussian pyramid starting from already-populated
    /// `gauss_ref[0].planes[*]`. Split out of
    /// [`Self::_dispatch_gauss_pyramid_gpu`] so the upload-once
    /// `compute_handles` path can populate level 0 from a caller-
    /// supplied packed-u32 handle and then run the same downscale
    /// chain bit-for-bit.
    ///
    /// Full / Mode E path: LDS-tiled `downscale_tiled_kernel`. T1.B
    /// (2026-05-16) — 16×16 workgroup; 36×36 input tile in shared
    /// memory; 25-tap stencil from LDS. Functionally equivalent to
    /// the scalar `downscale_kernel`, including the tick-206
    /// bug-compat delta.
    ///
    /// Mode B (StripPair) path: per-level strip walker over
    /// `downscale_strip_kernel`. For each level `k = 1..n_levels`:
    ///   1. Partition the level's `bh` rows into `ceil(bh / strip_h_at_k)`
    ///      strips where `strip_h_at_k = max(h_body >> k, 1)`.
    ///   2. Dispatch the strip-aware kernel once per (strip, channel),
    ///      reading from the FULL level-`k-1` buffer (already fully
    ///      populated by the prior level's strip walk) and writing
    ///      strip-body rows of the level-`k` buffer via
    ///      `Handle::offset_start`.
    ///   3. Output is bit-identical to the tiled kernel: the strip
    ///      kernel is the scalar reference path, and the tiled kernel
    ///      is pinned bit-exact against it by
    ///      `strip_kernel_parity_pyramid` tests.
    ///
    /// Cross-strip data dependency: each strip at level `k` reads
    /// level `k-1` rows in a small neighbourhood around `2·body_offset_y_k`,
    /// which were ALL written by level `k-1`'s strip walk before this
    /// level started. Level-major iteration order keeps the walker
    /// correct. The `strip_dispatch_counter` increments once per
    /// (strip, channel) dispatch so a test can observe the walker
    /// partitioned the work.
    /// **P2.3 (2026-05-27):** swap `gauss_ref` ↔ `gauss_alt` so REF
    /// gauss data persists past a subsequent DIST gauss dispatch. No-op
    /// outside `StripMode::Pair`. See [`Self::gauss_alt`] docstring for
    /// the architecture rationale.
    ///
    /// After this swap:
    ///   - `gauss_alt.as_ref().unwrap()` holds REF gauss data.
    ///   - `gauss_ref` holds garbage (the prior call's leftover, or
    ///     the zero-initialised buffer on first call). The next DIST
    ///     gauss dispatch overwrites it before any read.
    ///
    /// The swap happens AFTER REF weber finalize (incl. baseband
    /// mean) so the finalize path can read `gauss_ref` as usual.
    ///
    /// **P2.7 partial (2026-05-27):** swap is now SHALLOW-ONLY
    /// (`k < k_split`). Deep levels of `gauss_alt` are zero-size
    /// (never read for REF — REF weber finalize writes deep bands_ref
    /// from gauss_ref before this swap, then nothing reads REF deep
    /// gauss until next call). After the per-level swap:
    ///   - `gauss_alt[k]` shallow: holds REF gauss for shallow k.
    ///   - `gauss_alt[k]` deep: zero-size (was already zero-size pre-swap).
    ///   - `gauss_ref[k]` shallow: garbage (will be overwritten by DIST).
    ///   - `gauss_ref[k]` deep: holds REF data — the DIST gauss reduce
    ///     overwrites it (DIST baseband path reads gauss_ref[last]
    ///     where last >= k_split, so DIST data is what's wanted there).
    fn _maybe_swap_gauss_alt_post_ref(&mut self) {
        let mode_b = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );
        if !mode_b {
            return;
        }
        let h_body = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return,
        };
        let n_levels_u32 = self.n_levels;
        let k_split = mode_b_k_split(h_body, n_levels_u32) as usize;
        // Swap shallow levels + level k_split (the "coarse source" for
        // the last shallow REF strip helper read). Levels > k_split
        // stay unswapped because gauss_alt is zero-sized for them.
        let swap_upto = (k_split + 1).min(self.n_levels as usize);
        if let Some(alt) = self.gauss_alt.as_mut() {
            for k in 0..swap_upto {
                std::mem::swap(&mut self.gauss_ref[k], &mut alt[k]);
            }
        }
    }

    fn _reduce_gauss_pyramid_from_level0(&self) {
        let mode_b = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );
        if mode_b {
            self._reduce_gauss_pyramid_strip_walker();
        } else {
            self._reduce_gauss_pyramid_tiled();
        }
    }

    /// Tiled (LDS) downscale dispatch — the Full / Mode E path.
    fn _reduce_gauss_pyramid_tiled(&self) {
        let cube_dim = CubeDim::new_2d(DOWNSCALE_TILED_BLOCK_DIM, DOWNSCALE_TILED_BLOCK_DIM);
        for k in 1..(self.n_levels as usize) {
            let prev_w = self.gauss_ref[k - 1].w;
            let prev_h = self.gauss_ref[k - 1].h;
            let curr_w = self.gauss_ref[k].w;
            let curr_h = self.gauss_ref[k].h;
            let n_curr = (curr_w * curr_h) as usize;
            let n_prev = (prev_w * prev_h) as usize;
            let cube_count = CubeCount::Static(
                curr_w.div_ceil(DOWNSCALE_TILED_BLOCK_DIM),
                curr_h.div_ceil(DOWNSCALE_TILED_BLOCK_DIM),
                1,
            );

            for c in 0..N_CHANNELS {
                let src = self.gauss_ref[k - 1].planes[c].clone();
                let dst = self.gauss_ref[k].planes[c].clone();
                unsafe {
                    downscale_tiled_kernel::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(src, n_prev),
                        ArrayArg::from_raw_parts(dst, n_curr),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                    );
                }
            }
        }
    }

    /// Per-level strip walker for Mode B. Uses `downscale_strip_kernel`
    /// (scalar reference path) with full-image src and `body_offset_y`-
    /// offset dst handles so each strip dispatch writes exactly its
    /// body rows of the per-level output buffer.
    fn _reduce_gauss_pyramid_strip_walker(&self) {
        let cube_dim = CubeDim::new_1d(64);
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return,
        };

        for k in 1..(self.n_levels as usize) {
            let prev_w = self.gauss_ref[k - 1].w;
            let prev_h = self.gauss_ref[k - 1].h;
            let curr_w = self.gauss_ref[k].w;
            let curr_h = self.gauss_ref[k].h;
            let n_prev = (prev_w * prev_h) as usize;
            // Strip body height at this level: scale-0 body halved per
            // level, clamped to 1. At deep levels the strip count
            // collapses to 1 (single dispatch covering all rows).
            let strip_h_at_k = (h_body_at_0 >> k).max(1);
            let n_strips = if curr_h <= strip_h_at_k {
                1
            } else {
                curr_h.div_ceil(strip_h_at_k)
            };

            for s in 0..n_strips {
                let body_offset_y = s * strip_h_at_k;
                let body_h = (curr_h - body_offset_y).min(strip_h_at_k);
                let n_strip = (curr_w as usize) * (body_h as usize);
                let byte_off: u64 = u64::from(body_offset_y) * u64::from(curr_w) * 4;
                let cube_count = CubeCount::Static((n_strip as u32).div_ceil(64), 1, 1);

                for c in 0..N_CHANNELS {
                    let src = self.gauss_ref[k - 1].planes[c].clone();
                    let dst_strip = self.gauss_ref[k].planes[c].clone().offset_start(byte_off);
                    unsafe {
                        downscale_strip_kernel::launch::<R>(
                            &self.client,
                            cube_count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(src, n_prev),
                            ArrayArg::from_raw_parts(dst_strip, n_strip),
                            prev_w,
                            prev_h,
                            curr_w,
                            body_h,
                            body_offset_y,
                            0,         // src_strip_offset: src is FULL prev-level buffer
                            prev_h,    // logical_src_h
                            curr_h,    // logical_dst_h
                        );
                    }
                    self.strip_dispatch_counter
                        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }
            }
        }
    }

    /// Mode B per-level strip walker for the Weber-pyramid finalize
    /// (non-baseband levels k = 0..n_levels-1). Mirrors
    /// [`Self::_reduce_gauss_pyramid_strip_walker`]'s shape:
    /// level-major iteration, per (strip, channel) dispatch.
    ///
    /// Each level k:
    ///   1. Per strip, per channel: separable upscale of
    ///      `gauss_ref[k+1]` (FULL coarse buffer) → vscratch (strip
    ///      body of full vscratch buffer) → upscaled_c (strip body of
    ///      full upscaled buffer). Uses `upscale_v_strip_kernel` +
    ///      `upscale_h_strip_kernel`.
    ///   2. Per strip: fused `subtract_weber_3ch_strip_kernel` reads
    ///      from FULL fine/upsc/l_bkg buffers and writes to body rows
    ///      of `bands_*[k]` + `log_l_bkg`. The strip-aware kernel
    ///      uses absolute indexing `(body_offset_y + dy_local) * w +
    ///      dx`; pre-allocated buffers are full-image-sized so the
    ///      body rows land in the same place a Full-mode dispatch
    ///      would have written.
    ///
    /// Strip body height halves per level. At deep levels the strip
    /// count collapses to 1 (single dispatch covers all rows).
    ///
    /// Cross-strip data dependency: each strip's upscale reads
    /// `gauss_ref[k+1]` rows in a small neighbourhood around
    /// `body_offset_y / 2` (the upscale's reflection); those rows are
    /// either in the FULL coarse buffer (cross-strip data inherited
    /// from the gauss reduce) OR the body region of `vscratch` (for
    /// the upscale_h reading upscale_v's strip output). The body
    /// region is well-defined because v + h run in the same strip.
    ///
    /// The `strip_dispatch_counter` increments per (level, strip,
    /// channel) dispatch + per (level, strip) subtract dispatch so a
    /// test can observe the walker partitioned the work.
    fn _finalize_weber_pyramid_strip_walker(
        &self,
        log_l_bkg_dest: &[cubecl::server::Handle],
        dest_is_dis: bool,
    ) {
        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return,
        };
        let n_levels_u32 = self.n_levels;
        let k_split = mode_b_k_split(h_body_at_0, n_levels_u32) as usize;

        for k in 0..n_levels.saturating_sub(1) {
            // P2.3 REF shrink (2026-05-27): for the REF side
            // (`dest_is_dis = false`), defer shallow-level (k < k_split)
            // writes to the band-loop's strip-major-outer dispatch.
            // The full-image `bands_ref[k].planes` are zero-size for
            // shallow levels under StripMode::Pair so writing here
            // would crash; the per-strip REF helper invoked by
            // `_run_d_bands_band_loop` writes
            // `weber_scratch[k].bands_ref_strip` instead, in lockstep
            // with the per-strip DIST CSF dispatch that consumes it.
            // Deep levels (k >= k_split) keep full-image bands_ref and
            // run through this strip walker as before.
            //
            // The DIST side keeps its existing behaviour (full-image
            // bands_dis is zero-size only for non-baseband levels in
            // Mode B; the DIST-side `_finalize_weber_pyramid_after_gauss`
            // routes Mode B through a separate "defer entirely" branch
            // upstream of this walker, so DIST never reaches this
            // codepath for non-baseband levels in Mode B).
            if !dest_is_dis && k < k_split {
                continue;
            }
            let coarse_w = self.gauss_ref[k + 1].w;
            let coarse_h = self.gauss_ref[k + 1].h;
            let fine_w = self.gauss_ref[k].w;
            let fine_h = self.gauss_ref[k].h;
            // `n_v` (coarse_w × fine_h) would be the FULL vscratch
            // buffer's element count; the strip path computes
            // `n_strip_v` per strip below from `coarse_w × body_h`.
            // The full count isn't referenced here — it's kept as a
            // doc breadcrumb to mirror the legacy weber finalize.
            let _n_v_full = (coarse_w * fine_h) as usize;
            let n_fine = (fine_w * fine_h) as usize;
            let n_coarse = (coarse_w * coarse_h) as usize;

            // Strip body height at this level: scale-0 body halved per
            // level, clamped to 1. Matches gauss strip walker.
            let strip_h_at_k = (h_body_at_0 >> k).max(1);
            let n_strips = if fine_h <= strip_h_at_k {
                1
            } else {
                fine_h.div_ceil(strip_h_at_k)
            };

            let scratch = &self.weber_scratch[k];
            let bands_dest = if dest_is_dis {
                &self.bands_dis
            } else {
                &self.bands_ref
            };

            for s in 0..n_strips {
                let body_offset_y = s * strip_h_at_k;
                let body_h = (fine_h - body_offset_y).min(strip_h_at_k);
                // Strip-shaped element counts at the fine and v-pass
                // resolutions. The v-pass writes `coarse_w × body_h`
                // rows; the h-pass and subtract write `fine_w ×
                // body_h` rows. Offsets into the full vscratch and
                // l_bkg_fine / upscaled / band buffers point at the
                // body row block.
                let n_strip_v = (coarse_w as usize) * (body_h as usize);
                let n_strip_fine = (fine_w as usize) * (body_h as usize);
                let count_v_strip = CubeCount::Static((n_strip_v as u32).div_ceil(64), 1, 1);
                let count_fine_strip = CubeCount::Static((n_strip_fine as u32).div_ceil(64), 1, 1);
                let byte_off_v: u64 = u64::from(body_offset_y) * u64::from(coarse_w) * 4;
                let byte_off_fine: u64 = u64::from(body_offset_y) * u64::from(fine_w) * 4;

                // Stage 1: separable upscale of coarse A → l_bkg_fine
                // body. The A plane drives the per-pixel L_bkg used by
                // every channel's Weber-contrast division below.
                let coarse_a = self.gauss_ref[k + 1].planes[0].clone();
                let vscratch_a_strip =
                    scratch.vscratch_a.clone().offset_start(byte_off_v);
                let l_bkg_fine_strip =
                    scratch.l_bkg_fine.clone().offset_start(byte_off_fine);
                unsafe {
                    upscale_v_strip_kernel::launch::<R>(
                        &self.client,
                        count_v_strip.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(coarse_a, n_coarse),
                        ArrayArg::from_raw_parts(vscratch_a_strip.clone(), n_strip_v),
                        coarse_w,
                        coarse_h,        // logical_src_h
                        fine_h,          // logical_dst_h
                        body_offset_y,   // body_offset_y in dst rows
                        body_h,          // body_h in dst rows
                        0,               // src_strip_offset: source is FULL coarse
                    );
                    upscale_h_strip_kernel::launch::<R>(
                        &self.client,
                        count_fine_strip.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(vscratch_a_strip, n_strip_v),
                        ArrayArg::from_raw_parts(l_bkg_fine_strip, n_strip_fine),
                        coarse_w,
                        fine_w,
                        body_h,          // in_h = strip body height
                        fine_h,          // logical_dst_h (unused by H-pass)
                        body_offset_y,   // body_offset_y (unused by H-pass)
                    );
                }
                self.strip_dispatch_counter
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

                // Stage 2: per-channel separable upscale of coarse →
                // upscaled_c body. Same shape as the A plane upscale
                // but targeting the per-channel scratch buffers.
                //
                // Path A Phase 1b: when `upscaled_c_strip` is allocated
                // (StripMode::Pair builds it), write to the
                // per-strip-sized buffer directly with NO byte_off slice
                // — the buffer is the strip (sized `fine_w *
                // strip_h_at_k * 4`). Each strip's iteration
                // overwrites the buffer; the subtract_weber kernel
                // reads it back below (same iteration) so no
                // cross-strip dependency exists. When Phase 1b is
                // disabled (legacy / non-Mode-B), fall back to the
                // sliced `upscaled_c` (full-image-sized) at
                // `byte_off_fine`.
                let use_phase1b_upsc = scratch.upscaled_c_strip.is_some();
                for c in 0..N_CHANNELS {
                    let coarse = self.gauss_ref[k + 1].planes[c].clone();
                    let vscratch_c_strip =
                        scratch.vscratch_c[c].clone().offset_start(byte_off_v);
                    let (upscaled_c_strip_h, upscaled_c_strip_n) = if let Some(strips) =
                        scratch.upscaled_c_strip.as_ref()
                    {
                        // Phase 1b: per-strip buffer sized
                        // `fine_w * strip_h_at_k`. The kernel writes
                        // strip-local rows [0..body_h) of this buffer;
                        // rows [body_h..strip_h_at_k) (if any, on the
                        // last strip of an unaligned fine_h) keep
                        // their prior contents but subtract_weber's
                        // launch geometry (`count_fine_strip`) ensures
                        // we only iterate `body_h` rows so stale
                        // contents above body_h are never read.
                        let n_strip_buf = (fine_w as usize) * (strip_h_at_k as usize);
                        (strips[c].clone(), n_strip_buf)
                    } else {
                        // Legacy / Full path: slice the full-image
                        // `upscaled_c` at the strip's byte offset so
                        // the kernel writes to the body row block of
                        // the full buffer (identical to pre-Phase-1b
                        // behaviour). Path A Phase 1c (2026-05-26)
                        // makes `upscaled_c` an `Option` that's `None`
                        // in StripMode::Pair. The strip walker only
                        // dispatches in Mode B (verified by the
                        // `_finalize_weber_pyramid_after_gauss` and
                        // `_reduce_gauss_pyramid_from_level0` branches),
                        // so `upscaled_c` here would have to be
                        // `Some(...)` to mean Mode B's strip alloc was
                        // disabled — a config we don't support. Treat
                        // it as a hard invariant; `expect` so a future
                        // misconfiguration surfaces loudly rather than
                        // silently writing to a stale buffer.
                        let upscaled_full = scratch
                            .upscaled_c
                            .as_ref()
                            .expect(
                                "upscaled_c full buffer is None in strip-walker legacy fallback; \
                                 StripMode::Pair must allocate upscaled_c_strip instead",
                            );
                        (
                            upscaled_full[c].clone().offset_start(byte_off_fine),
                            n_strip_fine,
                        )
                    };
                    unsafe {
                        upscale_v_strip_kernel::launch::<R>(
                            &self.client,
                            count_v_strip.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(coarse, n_coarse),
                            ArrayArg::from_raw_parts(vscratch_c_strip.clone(), n_strip_v),
                            coarse_w,
                            coarse_h,
                            fine_h,
                            body_offset_y,
                            body_h,
                            0,               // src_strip_offset: source is FULL coarse
                        );
                        upscale_h_strip_kernel::launch::<R>(
                            &self.client,
                            count_fine_strip.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(vscratch_c_strip, n_strip_v),
                            ArrayArg::from_raw_parts(upscaled_c_strip_h, upscaled_c_strip_n),
                            coarse_w,
                            fine_w,
                            body_h,
                            fine_h,
                            body_offset_y,
                        );
                    }
                    self.strip_dispatch_counter
                        .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
                }

                // Stage 3: fused subtract + Weber-contrast + log_l_bkg.
                //
                // Path A Phase 1b (when `upscaled_c_strip` is set):
                //   - `upsc_*` reads from per-strip buffers
                //     (`upscaled_c_strip[c]`); the buffer's row 0
                //     corresponds to logical row `body_offset_y`.
                //   - Every OTHER input/output buffer
                //     (fine_*, l_bkg_fine, bands_*, log_l_bkg) is
                //     full-image-sized; we slice each via
                //     `offset_start(byte_off_fine)` so its row 0
                //     ALSO corresponds to logical row body_offset_y.
                //   - The kernel's `src_strip_offset = body_offset_y`
                //     translates the logical body row
                //     `body_offset_y + dy_local` to buffer-local
                //     `dy_local`, which is the right index for every
                //     buffer in this configuration.
                //
                // Legacy path (when `upscaled_c_strip` is None):
                //   - Inputs and outputs are FULL-image buffers; the
                //     kernel reads/writes at full-image-relative
                //     `(body_offset_y + dy_local) * w + dx`. Pass
                //     `src_strip_offset = 0`.
                let fine_a_full = self.gauss_ref[k].planes[0].clone();
                let fine_rg_full = self.gauss_ref[k].planes[1].clone();
                let fine_vy_full = self.gauss_ref[k].planes[2].clone();
                let l_bkg_fine_full = scratch.l_bkg_fine.clone();
                let log_l_bkg_full = log_l_bkg_dest[k].clone();
                let band_a_full = bands_dest[k].planes[0].clone();
                let band_rg_full = bands_dest[k].planes[1].clone();
                let band_vy_full = bands_dest[k].planes[2].clone();

                let (
                    fine_a_h,
                    fine_rg_h,
                    fine_vy_h,
                    upsc_a_h,
                    upsc_rg_h,
                    upsc_vy_h,
                    l_bkg_fine_h,
                    band_a_h,
                    band_rg_h,
                    band_vy_h,
                    log_l_bkg_h,
                    src_strip_off,
                    buf_n,
                ) = if let Some(strips) = scratch.upscaled_c_strip.as_ref() {
                    // Phase 1b: all buffers strip-local. Slice the
                    // full-image fine/lbkg/bands/log_l_bkg handles at
                    // `byte_off_fine`; upsc_* is the per-strip buffer
                    // (no slice). The kernel's read/write index is
                    // `dy_local * w + dx` for ALL of them. The bound
                    // we pass to `ArrayArg::from_raw_parts` is
                    // `n_strip_fine = body_h * fine_w`, the kernel's
                    // exact iteration count — every buffer in this
                    // configuration has at least that many remaining
                    // elements from its row-0 origin:
                    //   * upsc_* strip buf is `strip_h_at_k * fine_w
                    //     >= body_h * fine_w`.
                    //   * sliced full-image handles have remaining
                    //     `(fine_h - body_offset_y) * fine_w >= body_h
                    //     * fine_w`.
                    (
                        fine_a_full.offset_start(byte_off_fine),
                        fine_rg_full.offset_start(byte_off_fine),
                        fine_vy_full.offset_start(byte_off_fine),
                        strips[0].clone(),
                        strips[1].clone(),
                        strips[2].clone(),
                        l_bkg_fine_full.offset_start(byte_off_fine),
                        band_a_full.offset_start(byte_off_fine),
                        band_rg_full.offset_start(byte_off_fine),
                        band_vy_full.offset_start(byte_off_fine),
                        log_l_bkg_full.offset_start(byte_off_fine),
                        body_offset_y,
                        n_strip_fine,
                    )
                } else {
                    // Legacy: all buffers FULL-image; the kernel uses
                    // full-image-relative indexing. Same expect-or-die
                    // contract as the Stage 2 fallback above: this
                    // branch is structurally unreachable in
                    // StripMode::Pair (which is the only mode that
                    // dispatches the strip walker), so a `None` here
                    // signals a config that was rejected upstream.
                    let upscaled_full = scratch
                        .upscaled_c
                        .as_ref()
                        .expect(
                            "upscaled_c full buffer is None in strip-walker subtract-weber legacy \
                             fallback; StripMode::Pair must allocate upscaled_c_strip instead",
                        );
                    (
                        fine_a_full,
                        fine_rg_full,
                        fine_vy_full,
                        upscaled_full[0].clone(),
                        upscaled_full[1].clone(),
                        upscaled_full[2].clone(),
                        l_bkg_fine_full,
                        band_a_full,
                        band_rg_full,
                        band_vy_full,
                        log_l_bkg_full,
                        0,
                        n_fine,
                    )
                };
                let _ = use_phase1b_upsc;

                unsafe {
                    subtract_weber_3ch_strip_kernel::launch::<R>(
                        &self.client,
                        count_fine_strip.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(fine_a_h, buf_n),
                        ArrayArg::from_raw_parts(fine_rg_h, buf_n),
                        ArrayArg::from_raw_parts(fine_vy_h, buf_n),
                        ArrayArg::from_raw_parts(upsc_a_h, buf_n),
                        ArrayArg::from_raw_parts(upsc_rg_h, buf_n),
                        ArrayArg::from_raw_parts(upsc_vy_h, buf_n),
                        ArrayArg::from_raw_parts(l_bkg_fine_h, buf_n),
                        ArrayArg::from_raw_parts(band_a_h, buf_n),
                        ArrayArg::from_raw_parts(band_rg_h, buf_n),
                        ArrayArg::from_raw_parts(band_vy_h, buf_n),
                        ArrayArg::from_raw_parts(log_l_bkg_h, buf_n),
                        fine_w,
                        body_h,
                        body_offset_y,
                        fine_h, // logical_h (carried for API symmetry; unused)
                        src_strip_off,
                    );
                }
                self.strip_dispatch_counter
                    .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    /// Equivalent to [`Self::_dispatch_gauss_pyramid_gpu`] but starts
    /// from a caller-supplied packed-u32 device handle (one `u32`
    /// per pixel, `R | G<<8 | B<<16`, length `width × height`).
    fn _dispatch_gauss_pyramid_gpu_from_handle(&mut self, packed_u32: &cubecl::server::Handle) {
        self._install_src_ref_and_dispatch_dkl(packed_u32);
        self._reduce_gauss_pyramid_from_level0();
    }

    /// Run color + full Laplacian-pyramid decomposition. Returns
    /// `levels[k] = [a, rg, vy]` planar f32 bands matching cvvdp's
    /// `lpyr_dec.laplacian_pyramid_dec`:
    ///
    /// - `levels[k]` for `k < n_levels - 1` = `gauss[k] - expand(gauss[k+1])`
    /// - `levels[n_levels - 1]` = `gauss[n_levels - 1]` (coarse residual)
    ///
    /// Per-level temp buffers are allocated per call (no scratch
    /// pool yet). Future ticks can extend `Cvvdp::new` to allocate
    /// these once.
    ///
    /// # Examples
    ///
    /// Read back the Laplacian pyramid for a 64×64 buffer; the last
    /// level is the coarse residual (cvvdp's baseband-Laplacian
    /// convention). `ignore` for the standard `Cvvdp::*` reason.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let bands = cvvdp.compute_dkl_laplacian_pyramid(&srgb)
    ///     .expect("compute_dkl_laplacian_pyramid");
    /// assert!(!bands.is_empty());
    /// // bands[0] is the finest level (full base resolution).
    /// assert_eq!(bands[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails anywhere in the
    /// color → gauss → laplacian chain.
    pub fn compute_dkl_laplacian_pyramid(&mut self, srgb: &[u8]) -> Result<Vec<[Vec<f32>; 3]>> {
        // _dispatch_laplacian_pyramid_gpu overwrites bands_ref[k] with
        // Laplacian bands (not the Weber bands the warm-ref state was
        // built on). Invalidate the cached scalar so a subsequent
        // compute_dkl_jod_with_warm_ref surfaces NoWarmReference
        // instead of silently mixing Laplacian bands against the
        // cached Weber-baseband scalar. Same shape as the tick-236
        // fix for compute_dkl_weber_pyramid / compute_dkl_t_p_bands.
        self.warm_ref_baseband_log_l_bkg = None;
        self._dispatch_laplacian_pyramid_gpu(srgb)?;

        let n_levels = self.n_levels as usize;
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Dispatch-only version of `compute_dkl_laplacian_pyramid`:
    /// builds the Gaussian pyramid on GPU then emits the Laplacian
    /// bands into `bands_ref[k].planes[c]`. No host readback. Used by
    /// `compute_dkl_csf_weighted_bands` so we don't pay for a pyramid
    /// readback whose result is immediately discarded.
    fn _dispatch_laplacian_pyramid_gpu(&mut self, srgb: &[u8]) -> Result<()> {
        self._dispatch_gauss_pyramid_gpu(srgb)?;

        let cube_dim = CubeDim::new_1d(64);

        // Now produce Laplacian bands top-down. For each level k <
        // n_levels - 1: expand gauss[k+1] → temp, then subtract
        // (gauss[k] - temp) → bands_ref[k].
        for k in 0..(self.n_levels as usize - 1) {
            let coarse_w = self.gauss_ref[k + 1].w;
            let coarse_h = self.gauss_ref[k + 1].h;
            let fine_w = self.gauss_ref[k].w;
            let fine_h = self.gauss_ref[k].h;
            let n_v = (coarse_w * fine_h) as usize;
            let n_fine = (fine_w * fine_h) as usize;

            // Per-channel: upscale_v(coarse → vscratch), upscale_h(vscratch →
            // upscaled), subtract(fine, upscaled → band).
            for c in 0..N_CHANNELS {
                let coarse = self.gauss_ref[k + 1].planes[c].clone();
                let fine = self.gauss_ref[k].planes[c].clone();
                let band = self.bands_ref[k].planes[c].clone();

                let vscratch = alloc_zeros_f32(&self.client, n_v);
                let upscaled = alloc_zeros_f32(&self.client, n_fine);

                let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
                let count_h = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);
                let count_sub = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);
                let n_coarse = (coarse_w * coarse_h) as usize;

                unsafe {
                    upscale_v_kernel::launch::<R>(
                        &self.client,
                        count_v,
                        cube_dim,
                        ArrayArg::from_raw_parts(coarse, n_coarse),
                        ArrayArg::from_raw_parts(vscratch.clone(), n_v),
                        coarse_w,
                        coarse_h,
                        fine_h,
                    );
                    upscale_h_kernel::launch::<R>(
                        &self.client,
                        count_h,
                        cube_dim,
                        ArrayArg::from_raw_parts(vscratch, n_v),
                        ArrayArg::from_raw_parts(upscaled.clone(), n_fine),
                        coarse_w,
                        fine_w,
                        fine_h,
                    );
                    subtract_kernel::launch::<R>(
                        &self.client,
                        count_sub,
                        cube_dim,
                        ArrayArg::from_raw_parts(fine, n_fine),
                        ArrayArg::from_raw_parts(upscaled, n_fine),
                        ArrayArg::from_raw_parts(band, n_fine),
                        n_fine as u32,
                    );
                }
            }
        }

        // Coarsest band = coarsest gauss (no subtraction). Read it
        // directly from gauss_ref. For symmetry with the rest, copy
        // into bands_ref[last] via a host trip — small buffer.
        let n_levels = self.n_levels as usize;
        let last = n_levels - 1;
        for c in 0..N_CHANNELS {
            let g = self.gauss_ref[last].planes[c].clone();
            let bytes = self
                .client
                .read_one(g)
                .map_err(|_| Error::InvalidImageSize)?;
            // Re-upload as bands_ref[last] so the read-back loop is
            // uniform across levels.
            self.bands_ref[last].planes[c] = self.client.create_from_slice(&bytes);
        }

        Ok(())
    }

    /// Run color + Weber-contrast pyramid on GPU. Matches what
    /// `host_scalar::weber_contrast_pyr_dec_scalar` builds for each
    /// of the 3 DKL channels, using each image's own achromatic
    /// channel as the `L_bkg` source (cvvdp's `weber_g1` rule).
    ///
    /// For non-baseband levels `k < N-1`:
    /// 1. `layer_c = gauss_c[k] - expand(gauss_c[k+1])` per channel
    ///    (built via `upscale_v_kernel` + `upscale_h_kernel` +
    ///    `subtract_kernel`, sharing the expand of `gauss_A[k+1]`
    ///    across channels for the L_bkg pathway).
    /// 2. `L_bkg = expand(gauss_A[k+1])`, clamped to ≥ 0.01 inside
    ///    `weber_contrast_compute_kernel`.
    /// 3. `contrast_c = clamp(layer_c / L_bkg, ±1000)` and
    ///    `log_l_bkg = log10(L_bkg)` via
    ///    `weber_contrast_compute_kernel`.
    ///
    /// For the baseband level `k = N-1`, cvvdp uses a SCALAR mean
    /// of `max(gauss_A[N-1], 0.01)`. The mean is computed host-side
    /// from a read-back of the achromatic baseband (≤16 pixels at
    /// 1024² × 7 levels), then each channel's baseband is divided
    /// by that scalar host-side. Avoids a GPU reduction for tiny
    /// data; the per-pixel divide is also tiny.
    ///
    /// Returns `(bands, log_l_bkg)`:
    /// - `bands[k] = [a, rg, vy]` Weber-contrast planar f32 vecs,
    ///   matching the shape of `compute_dkl_laplacian_pyramid`.
    /// - `log_l_bkg[k]` is a per-pixel `log10(L_bkg)` plane for
    ///   non-baseband levels and a scalar (replicated 1×1) for the
    ///   baseband. Same shape convention as
    ///   `WeberPyramid::log_l_bkg` in host_scalar.
    ///
    /// GPU-only Weber pyramid dispatch. Writes:
    /// - `self.bands_ref[k].planes[c]` — Weber-contrast bands per
    ///   level per channel (non-baseband levels). Baseband level
    ///   gets the per-channel `gauss[last]` divided by the achromatic
    ///   baseband's scalar mean (host-side).
    /// - `self.weber_scratch[k].log_l_bkg` — per-pixel
    ///   `log10(L_bkg)` plane per non-baseband level.
    ///
    /// Returns the baseband's scalar `log10(L_bkg)` since that's
    /// computed host-side (small reduction over the achromatic
    /// baseband). Callers handle the per-pixel readback themselves
    /// — this function does NO readback of the per-level band /
    /// log_l_bkg data.
    ///
    /// Used by `compute_dkl_weber_pyramid` (which wraps with
    /// readback to host Vecs) and by `_dispatch_d_bands_into_scratch`
    /// (which feeds the dispatch's per-band CSF + masking chain).
    /// The caller supplies the `log_l_bkg_dest` slice — typically
    /// the per-level handles from `self.weber_scratch` — so the
    /// output destination is decoupled from this helper.
    ///
    /// `log_l_bkg_dest` must have length `n_levels - 1` (one handle
    /// per non-baseband level). `dest_is_dis = false` writes weber
    /// bands to `self.bands_ref` (the REF side); `true` writes to
    /// `self.bands_dis` so both sides can persist on GPU through
    /// the d_bands band loop.
    fn _dispatch_weber_pyramid_gpu(
        &mut self,
        srgb: &[u8],
        log_l_bkg_dest: &[cubecl::server::Handle],
        dest_is_dis: bool,
    ) -> Result<f32> {
        // Build Gaussian pyramids on GPU. The dispatch-only helper
        // leaves `self.gauss_ref[k].planes[c]` populated for
        // `k = 0..n_levels` without paying for a full-pyramid
        // host readback (~190 MB at 12 MP) that we'd discard.
        self._dispatch_gauss_pyramid_gpu(srgb)?;

        // Path A Phase 1c (skip-full-alloc): byte-upload path now
        // defers to the unified `_finalize_weber_pyramid_after_gauss`
        // helper. The handle-upload sibling
        // (`_dispatch_weber_pyramid_gpu_from_handle`) already does
        // this; the byte path previously inlined a duplicate Full-mode
        // walker that ignored `strip_config`, so Mode B byte callers
        // never hit the strip walker and `WeberScratch.upscaled_c_strip`
        // was unused. Routing through the shared finalize lets the
        // Mode B branch inside `_finalize_weber_pyramid_after_gauss`
        // fire — and, in turn, lets us drop the now-unused full-image
        // `upscaled_c` allocation in StripMode::Pair (see
        // `build_weber_scratch`).
        self._finalize_weber_pyramid_after_gauss(log_l_bkg_dest, dest_is_dis)
    }

    /// Run color → Gaussian pyramid → Weber-contrast pyramid for one
    /// side, then read back the per-band data to host.
    ///
    /// Pipeline (GPU):
    /// - `srgb_to_dkl_kernel` writes 3 DKL planes into `gauss_ref[0]`.
    /// - `downscale_kernel` builds the Gaussian pyramid into
    ///   `gauss_ref[1..n_levels]`.
    /// - For each non-baseband level: separable upscale of
    ///   `gauss_ref[k+1]` + fused `subtract_weber_3ch_kernel` →
    ///   `bands_ref[k]` (per-channel Weber-contrast) and
    ///   `weber_scratch[k].log_l_bkg` (per-pixel `log10(L_bkg)`).
    /// - Baseband: scalar `L_bkg` = mean of `max(gauss_A[N-1], 0.01)`
    ///   computed host-side from a small read-back; each channel's
    ///   baseband band is `gauss[N-1][c] / L_bkg`.
    ///
    /// Side effect: `self.bands_ref[k].planes[c]` GPU handles are
    /// overwritten with the just-computed weber bands. Callers that
    /// need to compose REF + DIST sides (`compute_dkl_d_bands`,
    /// `compute_dkl_jod`) capture the host-side return Vecs OR read
    /// the handles directly between the REF and DIST pyramid calls.
    ///
    /// Returns `(bands, log_l_bkg)`:
    /// - `bands[k][c]` — Weber-contrast band, one `Vec<f32>` per
    ///   `(level, channel)`. Same shape as
    ///   `compute_dkl_laplacian_pyramid` / `compute_dkl_csf_weighted_bands`.
    /// - `log_l_bkg[k]` — per-pixel `log10(L_bkg)` plane for
    ///   non-baseband levels; a `Vec<f32>` of length `baseband_n`
    ///   filled with the scalar `log10(L_bkg_baseband_mean)` for the
    ///   baseband entry (replicated 1×1 shape convention matching
    ///   `host_scalar::WeberPyramid::log_l_bkg`).
    ///
    /// `CVVDP_TRACE_WEBER=1` env-var enables stderr instrumentation
    /// of the GPU dispatch vs read-back split — zero cost when
    /// unset.
    ///
    /// # Examples
    ///
    /// Read back Weber bands + log_l_bkg planes for a 64×64 buffer.
    /// `ignore` for the standard `Cvvdp::*` reason.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let (bands, log_l_bkg) = cvvdp.compute_dkl_weber_pyramid(&srgb)
    ///     .expect("compute_dkl_weber_pyramid");
    /// assert_eq!(bands.len(), log_l_bkg.len());
    /// // The first level matches base resolution per channel.
    /// assert_eq!(bands[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails anywhere in the
    /// color → weber-pyramid chain.
    pub fn compute_dkl_weber_pyramid(&mut self, srgb: &[u8]) -> Result<WeberPyramidGpu> {
        let trace_weber = std::env::var_os("CVVDP_TRACE_WEBER").is_some();
        let t_dispatch = std::time::Instant::now();

        // _dispatch_weber_pyramid_gpu overwrites `bands_ref[k]` +
        // `weber_scratch[k].log_l_bkg` with `srgb`'s data. The
        // warm-ref state cached different bytes; invalidate the
        // scalar so a subsequent `compute_dkl_jod_with_warm_ref`
        // surfaces `Error::NoWarmReference` instead of producing a
        // stale-mixed JOD. The `Cvvdp::warm_reference` docstring
        // already promised this for `compute_dkl_weber_pyramid`;
        // tick 236 closed the gap where the promise wasn't kept.
        self.warm_ref_baseband_log_l_bkg = None;

        // Build dests Vec (cloned from self.weber_scratch[*].log_l_bkg).
        let dests: Vec<cubecl::server::Handle> = self
            .weber_scratch
            .iter()
            .map(|s| s.log_l_bkg.clone())
            .collect();
        let log_l_bkg_baseband = self._dispatch_weber_pyramid_gpu(srgb, &dests, false)?;

        let n_levels = self.n_levels as usize;
        let last = n_levels - 1;
        let baseband_n = (self.gauss_ref[last].w as usize) * (self.gauss_ref[last].h as usize);

        if trace_weber {
            eprintln!(
                "[weber-trace] GPU dispatch + baseband host (before readback): {:?}",
                t_dispatch.elapsed()
            );
        }
        let t_readback = std::time::Instant::now();

        // Read back every band × every channel for return.
        let mut bands_out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            bands_out.push(planes);
        }

        if trace_weber {
            eprintln!(
                "[weber-trace] bands readback ({n_levels} levels): {:?}",
                t_readback.elapsed()
            );
        }
        let t_log_readback = std::time::Instant::now();

        // Read back log_l_bkg per band: non-baseband from GPU
        // (reconstruct handle from self.weber_scratch[k].log_l_bkg
        // since _dispatch_weber_pyramid_gpu left the data there),
        // baseband as replicated scalar matching host_scalar's
        // WeberPyramid shape.
        let mut log_l_bkg_out: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
        for k in 0..n_levels.saturating_sub(1) {
            let log_h = self.weber_scratch[k].log_l_bkg.clone();
            let bytes = self
                .client
                .read_one(log_h)
                .map_err(|_| Error::InvalidImageSize)?;
            log_l_bkg_out.push(f32::from_bytes(&bytes).to_vec());
        }
        log_l_bkg_out.push(vec![log_l_bkg_baseband; baseband_n]);

        if trace_weber {
            eprintln!(
                "[weber-trace] log_l_bkg readback: {:?}",
                t_log_readback.elapsed()
            );
        }

        Ok((bands_out, log_l_bkg_out))
    }

    /// Run color + Weber-contrast pyramid + per-pixel CSF apply on
    /// GPU. Returns the `T_p` bands that the masking stage consumes:
    ///
    /// ```text
    /// T_p[k][c][i] = band_mul[k] * weber[k][c][i] * S(rho_k, log_l_bkg[k][i], c) * CH_GAIN_eff
    /// ```
    ///
    /// where:
    /// - `band_mul = 1.0` for the first level (`k == 0`) and baseband
    ///   (`k == N-1`), `2.0` otherwise. Matches cvvdp's
    ///   `lpyr.get_band` ×2 band-readout gain on non-edge levels.
    /// - `S` is the per-pixel CSF sensitivity (with the
    ///   `sensitivity_correction` log offset baked in) from the
    ///   `csf_lut_weber_fixed_size` LUT. The kernel interpolates
    ///   `logs_row[rho_k, c]` along the per-pixel `log10(L_bkg)` axis.
    /// - `CH_GAIN_eff = CH_GAIN[c] = [1, 1.45, 1]` for non-baseband
    ///   levels. For the baseband, cvvdp's `apply_masking_model`
    ///   bypasses `CH_GAIN`, so this helper sets `CH_GAIN_eff = 1.0`
    ///   on the baseband — the caller can still subtract sides
    ///   directly to obtain the per-channel `D` (cvvdp's baseband
    ///   formula is `|T_p - R_p|` with `CH_GAIN` absorbed only in
    ///   `T_p` / `R_p` of non-baseband bands).
    ///
    /// `ppd` is pixels-per-degree. **The per-call `ppd` is silently
    /// ignored** — `logs_row` was pre-uploaded at `Cvvdp::new` /
    /// `Cvvdp::new_with_geometry` time against
    /// `self.geometry.pixels_per_degree()`, so the per-band `rho_k`
    /// (and thus the CSF LUT lookup) is fixed for this `Cvvdp` instance.
    /// Pass it consistent with the construction-time geometry for clarity;
    /// debug builds verify the match via `debug_assert_ppd_matches_geometry`
    /// (tick 243). Reconstruct with `new_with_geometry` if you need a
    /// different display geometry.
    ///
    /// Returns `levels[k] = [a, rg, vy]` planar f32 vecs, same shape
    /// as `compute_dkl_weber_pyramid`'s `.0`.
    ///
    /// # Examples
    ///
    /// Read back CSF-weighted bands for a 64×64 buffer. `ignore`
    /// for the standard `Cvvdp::*` reason.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let bands = cvvdp.compute_dkl_t_p_bands(&srgb, ppd)
    ///     .expect("compute_dkl_t_p_bands");
    /// assert!(!bands.is_empty());
    /// assert_eq!(bands[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails anywhere in the
    /// color → weber → CSF chain.
    pub fn compute_dkl_t_p_bands(&mut self, srgb: &[u8], ppd: f32) -> Result<Vec<[Vec<f32>; 3]>> {
        self.debug_assert_ppd_matches_geometry(ppd);

        // _dispatch_weber_pyramid_gpu below overwrites bands_ref +
        // weber_scratch[k].log_l_bkg with the new srgb's data. The
        // warm-ref state cached different bytes; invalidate the
        // scalar so a subsequent compute_dkl_jod_with_warm_ref
        // surfaces Error::NoWarmReference instead of producing a
        // stale-mixed JOD. Closes the same docstring-gap as
        // compute_dkl_weber_pyramid (tick 236).
        self.warm_ref_baseband_log_l_bkg = None;

        // Build Weber bands + log_l_bkg on GPU. Side effect leaves
        // weber bands resident in self.bands_ref and log_l_bkg in
        // weber_scratch[k].log_l_bkg handles.
        //
        // Tick 101: fused 3-channel CSF apply (was 3 per-channel
        // launches per level) AND read weber from `self.bands_ref`
        // handles directly (was re-uploading from the host Vec
        // returned by compute_dkl_weber_pyramid). Per non-baseband
        // level: 3 host uploads + 3 kernel launches → 0 uploads +
        // 1 launch.
        //
        // Tick 163: dispatch directly via `_dispatch_weber_pyramid_gpu`
        // and read back log_l_bkg only (the per-level planes the CSF
        // kernel needs), skipping the public wrapper's ~190 MB bands
        // host-alloc. Mirrors the fix tick 156 applied in
        // `_dispatch_d_bands_into_scratch`.
        let n_levels = self.n_levels as usize;
        let ref_log_l_bkg_dests: Vec<cubecl::server::Handle> = self
            .weber_scratch
            .iter()
            .map(|s| s.log_l_bkg.clone())
            .collect();
        let log_l_bkg_baseband =
            self._dispatch_weber_pyramid_gpu(srgb, &ref_log_l_bkg_dests, false)?;
        let mut log_l_bkg: Vec<Vec<f32>> = Vec::with_capacity(n_levels);
        for k in 0..n_levels.saturating_sub(1) {
            let (_, _, n_px_k) = self.level_dims(k);
            let h = self.weber_scratch[k].log_l_bkg.clone();
            let bytes = self
                .client
                .read_one(h)
                .map_err(|_| Error::InvalidImageSize)?;
            log_l_bkg.push(f32::from_bytes(&bytes).to_vec());
            debug_assert_eq!(log_l_bkg[k].len(), n_px_k);
        }
        let (_, _, baseband_n) = self.level_dims(n_levels - 1);
        log_l_bkg.push(vec![log_l_bkg_baseband; baseband_n]);

        let cube_dim = CubeDim::new_1d(64);

        let mut t_p_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let is_baseband = k == n_levels - 1;
            // band_mul = 2.0 on every level except the finest (k=0) and the
            // baseband — those use 1.0 per cvvdp's `lpyr.get_band` contract.
            let band_mul: f32 = if k == 0 || is_baseband { 1.0 } else { 2.0 };
            let (_, _, n_px) = self.level_dims(k);
            debug_assert_eq!(log_l_bkg[k].len(), n_px);

            let log_l_bkg_h = self.client.create_from_slice(f32::as_bytes(&log_l_bkg[k]));
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);

            let [ch_gain_a, ch_gain_rg, ch_gain_vy] = ch_gain_for_band(is_baseband, band_mul);

            let t_p_a_h = alloc_zeros_f32(&self.client, n_px);
            let t_p_rg_h = alloc_zeros_f32(&self.client, n_px);
            let t_p_vy_h = alloc_zeros_f32(&self.client, n_px);

            unsafe {
                csf_apply_3ch_kernel::launch::<R>(
                    &self.client,
                    count.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[0].clone(), n_px),
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[1].clone(), n_px),
                    ArrayArg::from_raw_parts(self.bands_ref[k].planes[2].clone(), n_px),
                    ArrayArg::from_raw_parts(log_l_bkg_h, n_px),
                    ArrayArg::from_raw_parts(self.logs_row[k][0].clone(), 32),
                    ArrayArg::from_raw_parts(self.logs_row[k][1].clone(), 32),
                    ArrayArg::from_raw_parts(self.logs_row[k][2].clone(), 32),
                    ArrayArg::from_raw_parts(t_p_a_h.clone(), n_px),
                    ArrayArg::from_raw_parts(t_p_rg_h.clone(), n_px),
                    ArrayArg::from_raw_parts(t_p_vy_h.clone(), n_px),
                    ch_gain_a,
                    ch_gain_rg,
                    ch_gain_vy,
                    n_px as u32,
                );
            }

            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for (c, h) in [t_p_a_h, t_p_rg_h, t_p_vy_h].into_iter().enumerate() {
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            t_p_bands.push(planes);
        }

        Ok(t_p_bands)
    }

    /// Run the full per-band D dispatch and leave each level's per-
    /// channel D plane in `self.d_scratch[k].d[c]`.
    ///
    /// Pipeline (all GPU after tick 96):
    /// - Color: sRGB → DKL (cached source bytes).
    /// - Weber pyramid: per-level upscale + fused
    ///   `subtract_weber_3ch_kernel` writing all three channels +
    ///   shared `log_l_bkg` in one launch.
    /// - Per-pixel CSF: `csf_apply_6ch_kernel` runs REF and DIST in
    ///   a single launch per non-baseband level (the LUT bracket
    ///   math is shared across all 6 outputs). Per cvvdp's
    ///   `weber_g1` contract, REF's `log10(L_bkg)` is used for
    ///   both sides.
    /// - Masking:
    ///   - Non-baseband bands: `min_abs_3ch_kernel →
    ///     pu_blur_h_3ch_kernel → pu_blur_v_3ch_scaled_kernel
    ///     (folds `* 10^MASK_C`) → mult_mutual_3ch_with_blurred_kernel`
    ///     (or `mult_mutual_3ch_no_blur_kernel` when `bw ≤ PU_PADSIZE`
    ///     or `bh ≤ PU_PADSIZE`).
    ///   - Baseband: `diff_abs_3ch_kernel` writes `|T_p_dis - T_p_ref|`
    ///     for all three channels in one launch (since tick 94 every
    ///     level's D plane lives in the same `d_scratch_d.d[k][c]` slot).
    ///
    /// No GPU→host readback inside this helper. Callers that want
    /// the host-side `Vec<[Vec<f32>; 3]>` snapshot use
    /// [`Cvvdp::compute_dkl_d_bands`]; callers that pool on GPU
    /// (`Cvvdp::compute_dkl_jod`) read straight from the resident
    /// handles via `pool_band_3ch_kernel` (one fused 3-channel
    /// launch per band).
    /// REF-side weber pyramid only. Dispatches color +
    /// `_dispatch_weber_pyramid_gpu` writing into `bands_ref` and
    /// `weber_scratch[k].log_l_bkg`. Returns the scalar baseband
    /// `log10(L_bkg)` so callers can store it for the band-loop's
    /// baseband CSF path.
    ///
    /// Skips the ~190 MB bands host readback that
    /// `compute_dkl_weber_pyramid` would do — the CSF dispatch
    /// reads `self.bands_ref[k]` GPU handles directly (tick 155);
    /// log_l_bkg planes stay resident on GPU (tick 166).
    ///
    /// Lives in its own helper so a future `warm_reference` /
    /// `compute_dkl_jod_with_warm_ref` fast path can dispatch the
    /// REF side once and reuse the GPU-resident state across many
    /// DIST candidates.
    fn _dispatch_ref_weber_pyramid_only(&mut self, ref_srgb: &[u8]) -> Result<f32> {
        // Overwriting REF state invalidates any prior warm cache —
        // the new REF's bands_ref / log_l_bkg / scalar are now
        // resident. `warm_reference` re-sets the scalar after this
        // returns; everyone else lets it stay None.
        self.warm_ref_baseband_log_l_bkg = None;
        // Temporarily move the pre-built dests Vec out via mem::take
        // so we can pass it as `&[Handle]` without conflicting with
        // the `&mut self` borrow on _dispatch_weber_pyramid_gpu.
        // Restored on the path back. Tick 240 amortises the per-call
        // Vec alloc + handle ref-bumps across construction time.
        let dests = std::mem::take(&mut self.log_l_bkg_ref_dests);
        let result = self._dispatch_weber_pyramid_gpu(ref_srgb, &dests, false);
        self.log_l_bkg_ref_dests = dests;
        self._maybe_swap_gauss_alt_post_ref();
        result
    }

    /// DIST-side weber pyramid only. Writes log_l_bkg to the
    /// throwaway `log_l_bkg_dis` handles so REF's data on
    /// `weber_scratch[k].log_l_bkg` survives. cvvdp's weber_g1
    /// rule uses REF's log_l_bkg for both sides, so DIST's value
    /// is computed-then-discarded.
    ///
    /// The DIST CSF kernel reads from `self.bands_dis[k]` GPU
    /// handles directly (tick 154 split); no host transfer.
    fn _dispatch_dist_weber_pyramid_only(&mut self, dist_srgb: &[u8]) -> Result<()> {
        // mem::take the pre-built dests Vec to pass it as &[Handle]
        // without conflicting with _dispatch_weber_pyramid_gpu's
        // &mut self. Restored on the path back. Mirrors tick 240's
        // fix in _dispatch_ref_weber_pyramid_only.
        let dests = std::mem::take(&mut self.log_l_bkg_dis_dests);
        let result = self._dispatch_weber_pyramid_gpu(dist_srgb, &dests, true);
        self.log_l_bkg_dis_dests = dests;
        result.map(|_| ())
    }

    /// Variant of [`Self::_dispatch_weber_pyramid_gpu`] that starts
    /// from a caller-supplied packed-u32 device handle instead of
    /// host bytes. The handle's layout MUST match
    /// [`Self::_dispatch_gauss_pyramid_gpu_from_handle`] — one
    /// `u32` per pixel, `R | G<<8 | B<<16`, length `width × height`.
    /// Internal helper for the upload-once `compute_handles` path.
    fn _dispatch_weber_pyramid_gpu_from_handle(
        &mut self,
        packed_u32: &cubecl::server::Handle,
        log_l_bkg_dest: &[cubecl::server::Handle],
        dest_is_dis: bool,
    ) -> Result<f32> {
        // Replace the byte-upload step with a handle install + reduce.
        // Everything past gauss-pyramid-build is identical to the
        // byte-flavored path; we inline the body so we can swap the
        // first stage without duplicating the rest of the kernel
        // graph. See `_dispatch_weber_pyramid_gpu` for the running
        // commentary on each stage — this method mirrors it 1:1.
        self._dispatch_gauss_pyramid_gpu_from_handle(packed_u32);
        self._finalize_weber_pyramid_after_gauss(log_l_bkg_dest, dest_is_dis)
    }

    /// Weber-pyramid stage that assumes `self.gauss_ref[k].planes[c]`
    /// is already populated for `k = 0..n_levels`. Shared body of
    /// [`Self::_dispatch_weber_pyramid_gpu`] (byte-upload path) and
    /// [`Self::_dispatch_weber_pyramid_gpu_from_handle`] (Phase 4
    /// upload-once path). Returns the scalar `log10(L_bkg)` baseband
    /// value the band loop needs.
    fn _finalize_weber_pyramid_after_gauss(
        &mut self,
        log_l_bkg_dest: &[cubecl::server::Handle],
        dest_is_dis: bool,
    ) -> Result<f32> {
        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;

        // Mode B (StripPair): route per-level non-baseband finalize
        // through the strip-aware walker. Mirrors
        // `_reduce_gauss_pyramid_strip_walker`'s shape: level-major
        // iteration, per (strip, channel) dispatch, reads from FULL
        // coarse/fine buffers and writes to body-offset rows of the
        // per-level scratch + bands buffers. Baseband (k = n_levels-1)
        // is one global-mean reduction; handled below identically to
        // the Full path because it needs all-pixels-of-A anyway.
        //
        // **Path A bands_dis shrink (2026-05-26):** For the DIST side
        // (`dest_is_dis = true`) in Mode B, the non-baseband Weber
        // finalize is DEFERRED to the band loop
        // (`_run_d_bands_band_loop`) where it runs in lock-step with
        // csf strip dispatch (Weber strip writes
        // `bands_dis_strip[k]` → csf strip reads it → writes
        // `t_p_*[k]` body, all within one strip iteration). The
        // full-image `bands_dis[k].planes` are zero-size in Mode B
        // for non-baseband levels, so the strip walker's body-offset
        // writes would crash. We still need the baseband path below
        // because the baseband-divide kernel writes the full-size
        // `bands_dis[last].planes` and the band loop reads them at
        // baseband resolution.
        //
        // The REF side (`dest_is_dis = false`) keeps the standard
        // strip walker call — bands_ref is allocated full-image and
        // the band loop reads it at full level dims for the
        // csf_apply_6ch reference inputs.
        let mode_b = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );
        if mode_b && !dest_is_dis {
            self._finalize_weber_pyramid_strip_walker(log_l_bkg_dest, dest_is_dis);
        } else if mode_b && dest_is_dis {
            // DIST + Mode B: skip non-baseband finalize here. The
            // band loop's per-strip Weber+csf fusion handles it.
            // Baseband still runs below (same path as Full mode).
        } else {
            for k in 0..n_levels.saturating_sub(1) {
                let coarse_w = self.gauss_ref[k + 1].w;
                let coarse_h = self.gauss_ref[k + 1].h;
                let fine_w = self.gauss_ref[k].w;
                let fine_h = self.gauss_ref[k].h;
                let n_v = (coarse_w * fine_h) as usize;
                let n_fine = (fine_w * fine_h) as usize;
                let n_coarse = (coarse_w * coarse_h) as usize;

                let count_v = CubeCount::Static((n_v as u32).div_ceil(64), 1, 1);
                let count_fine = CubeCount::Static((n_fine as u32).div_ceil(64), 1, 1);

                let scratch = &self.weber_scratch[k];
                let l_bkg_fine = scratch.l_bkg_fine.clone();
                let vscratch_a = scratch.vscratch_a.clone();
                let coarse_a = self.gauss_ref[k + 1].planes[0].clone();
                unsafe {
                    upscale_v_kernel::launch::<R>(
                        &self.client,
                        count_v.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(coarse_a, n_coarse),
                        ArrayArg::from_raw_parts(vscratch_a.clone(), n_v),
                        coarse_w,
                        coarse_h,
                        fine_h,
                    );
                    upscale_h_kernel::launch::<R>(
                        &self.client,
                        count_fine.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(vscratch_a, n_v),
                        ArrayArg::from_raw_parts(l_bkg_fine.clone(), n_fine),
                        coarse_w,
                        fine_w,
                        fine_h,
                    );
                }
                let log_l_bkg = log_l_bkg_dest[k].clone();
                // Path A Phase 1c: `upscaled_c` is `Some` here because
                // this branch is the non-Mode-B walker (Mode B took the
                // `mode_b` branch above and returned). The
                // `build_weber_scratch` allocator only sets `None` when
                // `strip_pair_h_body.is_some()` (i.e., Mode B); every
                // other construction path keeps the full alloc, so
                // `expect` documents the invariant and panics loudly if
                // a future refactor breaks it.
                let upscaled_c_full = scratch
                    .upscaled_c
                    .as_ref()
                    .expect(
                        "upscaled_c is None in Full / CachedRef Weber finalize path; \
                         StripMode::Pair should have routed through the strip walker",
                    );
                for c in 0..N_CHANNELS {
                    let coarse = self.gauss_ref[k + 1].planes[c].clone();
                    let vscratch_c = scratch.vscratch_c[c].clone();
                    let upscaled_c = upscaled_c_full[c].clone();
                    unsafe {
                        upscale_v_kernel::launch::<R>(
                            &self.client,
                            count_v.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(coarse, n_coarse),
                            ArrayArg::from_raw_parts(vscratch_c.clone(), n_v),
                            coarse_w,
                            coarse_h,
                            fine_h,
                        );
                        upscale_h_kernel::launch::<R>(
                            &self.client,
                            count_fine.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(vscratch_c, n_v),
                            ArrayArg::from_raw_parts(upscaled_c, n_fine),
                            coarse_w,
                            fine_w,
                            fine_h,
                        );
                    }
                }
                let fine_a = self.gauss_ref[k].planes[0].clone();
                let fine_rg = self.gauss_ref[k].planes[1].clone();
                let fine_vy = self.gauss_ref[k].planes[2].clone();
                let upsc_a = upscaled_c_full[0].clone();
                let upsc_rg = upscaled_c_full[1].clone();
                let upsc_vy = upscaled_c_full[2].clone();
                let bands_dest = if dest_is_dis {
                    &self.bands_dis
                } else {
                    &self.bands_ref
                };
                let band_a = bands_dest[k].planes[0].clone();
                let band_rg = bands_dest[k].planes[1].clone();
                let band_vy = bands_dest[k].planes[2].clone();
                unsafe {
                    subtract_weber_3ch_kernel::launch::<R>(
                        &self.client,
                        count_fine.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(fine_a, n_fine),
                        ArrayArg::from_raw_parts(fine_rg, n_fine),
                        ArrayArg::from_raw_parts(fine_vy, n_fine),
                        ArrayArg::from_raw_parts(upsc_a, n_fine),
                        ArrayArg::from_raw_parts(upsc_rg, n_fine),
                        ArrayArg::from_raw_parts(upsc_vy, n_fine),
                        ArrayArg::from_raw_parts(l_bkg_fine, n_fine),
                        ArrayArg::from_raw_parts(band_a, n_fine),
                        ArrayArg::from_raw_parts(band_rg, n_fine),
                        ArrayArg::from_raw_parts(band_vy, n_fine),
                        ArrayArg::from_raw_parts(log_l_bkg, n_fine),
                        n_fine as u32,
                    );
                }
            }
        }

        // Baseband — bit-identical to `_dispatch_weber_pyramid_gpu`.
        let last = n_levels - 1;
        let baseband_w = self.gauss_ref[last].w as usize;
        let baseband_h = self.gauss_ref[last].h as usize;
        let baseband_n = baseband_w * baseband_h;

        let gauss_a_last = self.gauss_ref[last].planes[0].clone();
        let bytes_a = self
            .client
            .read_one(gauss_a_last)
            .map_err(|_| Error::InvalidImageSize)?;
        let gauss_a_data: &[f32] = f32::from_bytes(&bytes_a);
        let l_bkg_sum: f32 = gauss_a_data.iter().map(|v| v.max(0.01)).sum();
        let l_bkg_mean = l_bkg_sum / baseband_n as f32;
        let log_l_bkg_baseband = l_bkg_mean.log10();

        let inv_l_bkg_mean = 1.0_f32 / l_bkg_mean;
        let gauss_a = self.gauss_ref[last].planes[0].clone();
        let gauss_rg = self.gauss_ref[last].planes[1].clone();
        let gauss_vy = self.gauss_ref[last].planes[2].clone();
        let bands_dest = if dest_is_dis {
            &self.bands_dis
        } else {
            &self.bands_ref
        };
        let band_a = bands_dest[last].planes[0].clone();
        let band_rg = bands_dest[last].planes[1].clone();
        let band_vy = bands_dest[last].planes[2].clone();
        let baseband_count = CubeCount::Static((baseband_n as u32).div_ceil(64), 1, 1);
        unsafe {
            baseband_divide_3ch_kernel::launch::<R>(
                &self.client,
                baseband_count,
                cube_dim,
                ArrayArg::from_raw_parts(gauss_a, baseband_n),
                ArrayArg::from_raw_parts(gauss_rg, baseband_n),
                ArrayArg::from_raw_parts(gauss_vy, baseband_n),
                ArrayArg::from_raw_parts(band_a, baseband_n),
                ArrayArg::from_raw_parts(band_rg, baseband_n),
                ArrayArg::from_raw_parts(band_vy, baseband_n),
                inv_l_bkg_mean,
                baseband_n as u32,
            );
        }

        Ok(log_l_bkg_baseband)
    }

    /// Handle-flavored sibling of [`Self::_dispatch_ref_weber_pyramid_only`]
    /// — takes a packed-u32 device handle instead of host bytes.
    fn _dispatch_ref_weber_pyramid_only_from_handle(
        &mut self,
        ref_handle: &cubecl::server::Handle,
    ) -> Result<f32> {
        self.warm_ref_baseband_log_l_bkg = None;
        let dests = std::mem::take(&mut self.log_l_bkg_ref_dests);
        let result = self._dispatch_weber_pyramid_gpu_from_handle(ref_handle, &dests, false);
        self.log_l_bkg_ref_dests = dests;
        self._maybe_swap_gauss_alt_post_ref();
        result
    }

    /// Handle-flavored sibling of [`Self::_dispatch_dist_weber_pyramid_only`].
    fn _dispatch_dist_weber_pyramid_only_from_handle(
        &mut self,
        dist_handle: &cubecl::server::Handle,
    ) -> Result<()> {
        let dests = std::mem::take(&mut self.log_l_bkg_dis_dests);
        let result = self._dispatch_weber_pyramid_gpu_from_handle(dist_handle, &dests, true);
        self.log_l_bkg_dis_dests = dests;
        result.map(|_| ())
    }

    fn _dispatch_d_bands_into_scratch(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_ref = std::time::Instant::now();
        let log_l_bkg_baseband = self._dispatch_ref_weber_pyramid_only(ref_srgb)?;
        if trace {
            eprintln!("[trace] weber(ref):  {:?}", t_weber_ref.elapsed());
        }
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband)
    }

    /// Handle-flavored sibling of [`Self::_dispatch_d_bands_into_scratch`].
    /// Both inputs are caller-uploaded packed-u32 device handles. Used
    /// by the Phase 4 [`Self::compute_handles`] upload-once path.
    fn _dispatch_d_bands_into_scratch_from_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dist_handle: &cubecl::server::Handle,
    ) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_ref = std::time::Instant::now();
        let log_l_bkg_baseband = self._dispatch_ref_weber_pyramid_only_from_handle(ref_handle)?;
        if trace {
            eprintln!("[trace] weber(ref):  {:?}", t_weber_ref.elapsed());
        }
        self._dispatch_d_bands_dist_and_band_loop_from_handle(dist_handle, log_l_bkg_baseband)
    }

    /// DIST weber + band loop. Reads REF-side state from
    /// `bands_ref[k]` + `weber_scratch[k].log_l_bkg` (populated by
    /// `_dispatch_ref_weber_pyramid_only` either earlier in
    /// `_dispatch_d_bands_into_scratch` or by `warm_reference`).
    ///
    /// Leaves the per-band D planes resident in
    /// `self.d_scratch[k].d[c]`. No host readback.
    ///
    /// Lifted out of `_dispatch_d_bands_into_scratch` so a
    /// `warm_reference` + `compute_dkl_jod_with_warm_ref` fast path
    /// can dispatch the REF side once and reuse the GPU-resident
    /// state across many DIST candidates.
    fn _dispatch_d_bands_dist_and_band_loop(
        &mut self,
        dist_srgb: &[u8],
        log_l_bkg_baseband: f32,
    ) -> Result<()> {
        // CVVDP_TRACE=1 enables per-phase eprintln timings so we can
        // see where the dispatch spends its time without committing
        // instrumentation. Zero cost when unset.
        let trace = std::env::var_os("CVVDP_TRACE").is_some();

        let t_weber_dis = std::time::Instant::now();
        self._dispatch_dist_weber_pyramid_only(dist_srgb)?;
        if trace {
            eprintln!("[trace] weber(dist): {:?}", t_weber_dis.elapsed());
        }

        self._run_d_bands_band_loop(log_l_bkg_baseband)
    }

    /// Handle-flavored sibling of [`Self::_dispatch_d_bands_dist_and_band_loop`].
    /// Takes a packed-u32 device handle for the dist side instead of
    /// host bytes; both methods then run the same band loop after the
    /// dist-weber dispatch lands `self.bands_dis[*]`.
    fn _dispatch_d_bands_dist_and_band_loop_from_handle(
        &mut self,
        dist_handle: &cubecl::server::Handle,
        log_l_bkg_baseband: f32,
    ) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_dis = std::time::Instant::now();
        self._dispatch_dist_weber_pyramid_only_from_handle(dist_handle)?;
        if trace {
            eprintln!("[trace] weber(dist): {:?}", t_weber_dis.elapsed());
        }
        self._run_d_bands_band_loop(log_l_bkg_baseband)
    }

    /// Per-level CSF + masking band loop. Both REF and DIST weber
    /// pyramids must already be resident in `self.bands_ref[*]` /
    /// `self.bands_dis[*]`. Shared body of
    /// [`Self::_dispatch_d_bands_dist_and_band_loop`] (byte path) and
    /// [`Self::_dispatch_d_bands_dist_and_band_loop_from_handle`]
    /// (handle path).
    ///
    /// **Dispatch shape (level-major outer + strip-major inner).** The
    /// outer `for k in 0..n_levels` loop iterates pyramid levels in
    /// order; the per-level helpers
    /// ([`Self::_dispatch_dist_weber_csf_strip_walker_for_level`] and
    /// [`Self::_run_band_masking_strip_walker`]) strip-walk INSIDE a
    /// single level. This is **not** the strip-major outer ordering
    /// that Mode B's Phase 2 buffer shrink ultimately needs — see
    /// `docs/STRIP_PROCESSING.md#phase-1-structural-strip-major-walker-investigation-2026-05-26`
    /// for the structural blocker (cross-strip halo dependencies on
    /// pyramid + masking V-blur) and the Phase 2 recipe that resolves
    /// it (per-strip body+halo-shaped `bands_dis_strip` / `t_p_*` /
    /// `m_*` transients + K_SPLIT hybrid for deep levels).
    fn _run_d_bands_band_loop(&mut self, log_l_bkg_baseband: f32) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let n_levels = self.n_levels as usize;
        let cube_dim = CubeDim::new_1d(64);
        // `10^MASK_C` post-blur scale for the PU stage — constant
        // per Cvvdp config, so compute once outside the band loop.
        let pu_scale = 10.0_f32.powf(MASK_C);

        // Path A Phase 1d (2026-05-26): in Mode B (StripMode::Pair),
        // non-baseband bands dispatch the pool kernel inline (per
        // strip in the strip walker, or once per band on the no-blur
        // fallback). The atomic-adds accumulate into `partials_h`,
        // which must be zero before any band writes. Zero it once
        // here; the post-band-loop pool finalize then only needs to
        // dispatch over the baseband (which still uses full `d`).
        // Mode E and Mode Full continue to zero in
        // `_pool_and_finalize_jod*` per their own dispatches.
        let mode_b_outer = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );
        if mode_b_outer {
            let n_partials = n_levels * N_CHANNELS;
            unsafe {
                fill_f32_kernel::launch::<R>(
                    &self.client,
                    CubeCount::Static((n_partials as u32).div_ceil(64), 1, 1),
                    cube_dim,
                    ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                    0.0,
                    n_partials as u32,
                );
            }
        }

        // Mode E (StripMode::CachedRef) reads REF-side band data + REF
        // log_l_bkg straight from `RefFullState` — the dedicated buffer
        // populated once by `warm_reference`. Modes Full and B both
        // dispatch a fresh REF Weber pyramid per call into the shared
        // `bands_ref` / `weber_scratch[k].log_l_bkg` scratch, so they
        // read from there. The mode-dependent source resolution is
        // hoisted out of the per-band hot loop into a const closure /
        // bool so the band loop body stays uniform.
        let mode_e = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::CachedRef, .. }),
        );
        // **Path A bands_dis shrink (2026-05-26):** Mode B
        // (StripPair) defers non-baseband DIST Weber finalize to this
        // band loop where each level's iteration runs Weber strip +
        // csf strip in lockstep over `bands_dis_strip` (one strip's
        // worth of storage per level). The full-image
        // `bands_dis[k].planes` are zero-size for non-baseband levels
        // in Mode B (skipped at construction) — touching them would
        // crash. The mode_b_pair flag below routes Mode B non-baseband
        // through the fused walker; everything else (Full / CachedRef
        // / baseband) goes through the standard csf_apply_6ch path.
        let mode_b_pair = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );

        // P2.1c outer-loop inversion (2026-05-27): in Mode B
        // (StripPair) the shallow levels (k < k_split) run strip-
        // major-outer: each strip's CSF + masking + pool fire across
        // all shallow levels in sequence, before the next strip
        // starts. P2.1b's body+halo CSF dispatch lets each strip's
        // masking find valid t_p_*[k] data in its halo window — the
        // CSF for strip s writes t_p_*[k]'s body+halo rows, and
        // masking for the same strip reads them. Deep levels (k >=
        // k_split) and the baseband continue level-major-outer in
        // the existing `for k in 0..n_levels` loop below — Mode B
        // skips already-handled shallow levels via `if k < k_split
        // && mode_b_pair && shallow_done { continue }`.
        let k_split_us = if mode_b_pair {
            let n_levels_u32 = n_levels as u32;
            let h_body = match self.strip_config {
                Some(StripConfig { h_body, .. }) => h_body,
                None => 0, // unreachable — mode_b_pair implies strip_config
            };
            // Cap at non-baseband-level count. `mode_b_k_split` can
            // return `n_levels` for tiny images where h_body covers
            // every level — but the baseband (k = n_levels - 1) is
            // handled by the level-major-outer loop below (it uses
            // `diff_abs_3ch` not csf/masking, so the strip-major-outer
            // dispatch doesn't apply). At minimum we exclude the
            // baseband from the strip-major-outer phase.
            let k_raw = mode_b_k_split(h_body, n_levels_u32) as usize;
            k_raw.min(n_levels.saturating_sub(1))
        } else {
            0
        };
        // Run strip-major-outer for shallow levels (Mode B only).
        // The deep-level + baseband loop below picks up at k_split.
        if mode_b_pair && k_split_us > 0 {
            self._run_d_bands_strip_major_shallow(k_split_us)?;
        }

        let t_band_loop = std::time::Instant::now();
        for k in 0..n_levels {
            // P2.1c: Mode B already ran shallow levels strip-major-
            // outer above; skip them in the level-major-outer loop.
            if mode_b_pair && k < k_split_us {
                continue;
            }
            let is_baseband = k == n_levels - 1;
            // band_mul = 2.0 on every level except the finest (k=0) and the
            // baseband — those use 1.0 per cvvdp's `lpyr.get_band` contract.
            let band_mul: f32 = if k == 0 || is_baseband { 1.0 } else { 2.0 };
            let (bw, bh, n_px) = self.level_dims(k);

            let t_band = std::time::Instant::now();

            // log_l_bkg source:
            // - Non-baseband bands:
            //   * Modes Full and B: read directly from the GPU-resident
            //     `weber_scratch[k].log_l_bkg` handle (REF data, written
            //     during the REF weber dispatch above). Tick 166 skips
            //     the host roundtrip — was reading back ~64 MB at 12 MP
            //     then re-uploading the same bytes per band.
            //   * Mode E: read from `ref_full_state.log_l_bkg[k]` — the
            //     dedicated buffer populated by `warm_reference`. Avoids
            //     the per-call copy from `RefFullState` back into
            //     `weber_scratch` that the Phase 2 snapshot/restore
            //     fallback used to do.
            // - Baseband: GPU fill the pre-allocated
            //   `self.baseband_log_l_bkg` buffer with the scalar
            //   `log_l_bkg_baseband`. Tick 168 replaces the per-JOD
            //   `vec![scalar; n]` host alloc + upload with a single
            //   GPU launch — keeps the JOD hot path entirely GPU-
            //   resident for log_l_bkg data.
            let t_log_upload = std::time::Instant::now();
            let log_l_bkg_h = if is_baseband {
                let fill_count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
                unsafe {
                    fill_f32_kernel::launch::<R>(
                        &self.client,
                        fill_count,
                        cube_dim,
                        ArrayArg::from_raw_parts(self.baseband_log_l_bkg.clone(), n_px),
                        log_l_bkg_baseband,
                        n_px as u32,
                    );
                }
                self.baseband_log_l_bkg.clone()
            } else if mode_e {
                // Mode E: read from RefFullState. Safe to unwrap — the
                // outer dispatcher (`compute_dkl_jod_with_warm_ref` etc)
                // surfaces `Error::NoWarmReference` when `ref_full_state`
                // is None before reaching here.
                self.ref_full_state
                    .as_ref()
                    .expect("Mode E: ref_full_state must be Some before band loop")
                    .log_l_bkg[k]
                    .clone()
            } else {
                self.weber_scratch[k].log_l_bkg.clone()
            };
            if trace {
                eprintln!(
                    "[trace] L{k} log_l_bkg source ({bw}×{bh}): {:?}",
                    t_log_upload.elapsed()
                );
            }
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);

            // Path-B lazy transient: allocate the per-band CSF +
            // masking intermediates here (drops at end of iteration
            // when `transient` goes out of scope, freeing the
            // 15 GPU buffers back to cubecl's memory pool for the
            // next band to reuse). `self.d_scratch[k].d[c]` is the
            // only persistent allocation now — downstream pool /
            // diffmap / `compute_dkl_d_bands` consumers read those
            // after the band loop completes.
            let transient = DBandsTransient::new(&self.client, n_px);
            let scratch_d = &self.d_scratch[k];
            let t_p_ref_h: [cubecl::server::Handle; 3] = [
                transient.t_p_ref[0].clone(),
                transient.t_p_ref[1].clone(),
                transient.t_p_ref[2].clone(),
            ];
            let t_p_dis_h: [cubecl::server::Handle; 3] = [
                transient.t_p_dis[0].clone(),
                transient.t_p_dis[1].clone(),
                transient.t_p_dis[2].clone(),
            ];

            let t_csf = std::time::Instant::now();

            // Fused 3-channel CSF apply — one launch per side instead
            // of three. The per-pixel LUT bracket math is shared across
            // the A/RG/VY channels.
            let [ch_gain_a, ch_gain_rg, ch_gain_vy] = ch_gain_for_band(is_baseband, band_mul);

            // Fused 6-channel CSF apply: one launch runs both sides
            // (REF + DIST) and shares the per-pixel LUT bracket math.
            // After tick 154's bands_ref/bands_dis split, both
            // sides' weber data lives on GPU at band-loop time:
            // - Modes Full and B: REF in `self.bands_ref[k]` (re-dispatched
            //   per call by `_dispatch_ref_weber_pyramid_only`).
            // - Mode E: REF in `self.ref_full_state.bands[k]` (snapshotted
            //   once by `warm_reference`). The kernel reads the same
            //   shape and stride, only the source handle differs.
            // DIST in `self.bands_dis[k]` regardless.
            // No host upload needed.
            let band_ref_a = if mode_e {
                self.ref_full_state
                    .as_ref()
                    .expect("Mode E: ref_full_state must be Some")
                    .bands[k][0]
                    .clone()
            } else {
                self.bands_ref[k].planes[0].clone()
            };
            let band_ref_rg = if mode_e {
                self.ref_full_state.as_ref().expect("checked above").bands[k][1].clone()
            } else {
                self.bands_ref[k].planes[1].clone()
            };
            let band_ref_vy = if mode_e {
                self.ref_full_state.as_ref().expect("checked above").bands[k][2].clone()
            } else {
                self.bands_ref[k].planes[2].clone()
            };
            if mode_b_pair && !is_baseband {
                // Path A bands_dis shrink: fused DIST Weber strip +
                // csf strip dispatch over the per-level strip loop.
                // Writes `t_p_ref[k]` and `t_p_dis[k]` body rows;
                // the masking strip walker below reads those.
                //
                // Sources:
                //   - bands_ref[k].planes[c] — full-image, sliced at
                //     body offset for each strip s.
                //   - bands_dis_strip[k] — strip-local buffer freshly
                //     written by the Weber kernel ONE LINE earlier.
                //   - log_l_bkg (REF's) — full-image, sliced at body
                //     offset. (Note: `log_l_bkg_h` above resolves to
                //     `weber_scratch[k].log_l_bkg` for Mode B; we
                //     slice it per strip below.)
                self._dispatch_dist_weber_csf_strip_walker_for_level(
                    k,
                    band_ref_a.clone(),
                    band_ref_rg.clone(),
                    band_ref_vy.clone(),
                    log_l_bkg_h.clone(),
                    &t_p_ref_h,
                    &t_p_dis_h,
                    [ch_gain_a, ch_gain_rg, ch_gain_vy],
                    band_mul,
                )?;
            } else {
                unsafe {
                    csf_apply_6ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(band_ref_a, n_px),
                        ArrayArg::from_raw_parts(band_ref_rg, n_px),
                        ArrayArg::from_raw_parts(band_ref_vy, n_px),
                        ArrayArg::from_raw_parts(self.bands_dis[k].planes[0].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_dis[k].planes[1].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_dis[k].planes[2].clone(), n_px),
                        ArrayArg::from_raw_parts(log_l_bkg_h.clone(), n_px),
                        ArrayArg::from_raw_parts(self.logs_row[k][0].clone(), 32),
                        ArrayArg::from_raw_parts(self.logs_row[k][1].clone(), 32),
                        ArrayArg::from_raw_parts(self.logs_row[k][2].clone(), 32),
                        ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                        ch_gain_a,
                        ch_gain_rg,
                        ch_gain_vy,
                        n_px as u32,
                    );
                }
            }
            // Tick 166 removed the per-band host Vec entirely —
            // non-baseband bands read log_l_bkg from GPU directly,
            // baseband does a one-shot scalar fill. Nothing to drop.
            if trace {
                eprintln!("[trace] L{k} csf 1 fused launch:    {:?}", t_csf.elapsed());
            }

            let t_mask = std::time::Instant::now();
            // Path A Phase 1d (2026-05-26): pick the d destination
            // for this level. The baseband always uses `d` (full-band,
            // small at deep levels). Non-baseband Mode B uses
            // `d_strip` (per-strip-sized); other modes use the full
            // `d` plane. Routing the handle selection through this
            // closure keeps the rest of the band loop body uniform.
            let mode_b = matches!(
                self.strip_config,
                Some(StripConfig { mode: StripMode::Pair, .. }),
            );
            if is_baseband {
                // Baseband: cvvdp's `|T_p_dis - T_p_ref|` bypass. Tick
                // 94 — GPU fused 3-channel diff into scratch.d so the
                // baseband output lives in d_scratch_d.d[k][c] like every
                // other level (prep for GPU pool in tick 95). Mode B
                // keeps the full-image baseband alloc because the
                // baseband band is small (~16×16 at 4K) and the diff_abs
                // path bypasses the strip walker entirely.
                let d_full = scratch_d.d.as_ref().expect(
                    "DBandsScratch.d must be Some at baseband (allocated by build_d_bands_scratch)",
                );
                let d_h: [cubecl::server::Handle; 3] = [
                    d_full[0].clone(),
                    d_full[1].clone(),
                    d_full[2].clone(),
                ];
                unsafe {
                    diff_abs_3ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                        ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                        n_px as u32,
                    );
                }
                let _ = d_h; // baseband result lives in d_scratch[k].d[c]
            } else {
                // GPU masking. D output + masking-chain scratch all
                // come from the pre-allocated d_scratch[k] (no
                // per-band alloc_zeros_f32 churn).
                //
                // Mode B routes to `d_strip` (per-strip-sized; the
                // strip walker writes one strip at a time at offset 0
                // and dispatches the pool inline before the next
                // strip overwrites it). Other modes use the full
                // `d` plane (Mode E still strip-walks the pool
                // post-band-loop over the full plane).
                let d_h: [cubecl::server::Handle; 3] = if mode_b {
                    let d_strip = scratch_d.d_strip.as_ref().expect(
                        "DBandsScratch.d_strip must be Some at non-baseband levels in StripMode::Pair",
                    );
                    [d_strip[0].clone(), d_strip[1].clone(), d_strip[2].clone()]
                } else {
                    let d_full = scratch_d.d.as_ref().expect(
                        "DBandsScratch.d must be Some at non-baseband levels outside StripMode::Pair",
                    );
                    [d_full[0].clone(), d_full[1].clone(), d_full[2].clone()]
                };
                let use_blur = bw > PU_PADSIZE && bh > PU_PADSIZE;
                // Mode B (StripPair) and Mode E (CachedRef): per-band
                // per-strip dispatch of the masking chain (min_abs →
                // pu_blur_h → pu_blur_v → mult_mutual). The V-blur is
                // the only kernel with Y-axis dependency; we dispatch
                // the whole chain over a halo-padded strip window so
                // the per-strip V-blur reads correct rows of m_mid
                // (populated by per-strip h-blur on the same window).
                // mult_mutual processes ONLY body rows so the d band
                // buffer stays valid. See _run_band_masking_strip_walker
                // docstring for the halo-padded buffer convention.
                //
                // Both Mode B and Mode E share this walker — the only
                // per-mode difference is the REF source (Mode B writes
                // bands_ref per call via the ref weber dispatch; Mode
                // E reads from `RefFullState` populated once by
                // `warm_reference`). That difference lives in
                // `_band_loop_ref_handles` above; the masking chain
                // consumes the same t_p_ref / t_p_dis handles regardless.
                let any_strip_mode = self.strip_config.is_some();
                unsafe {
                    if use_blur && any_strip_mode {
                        let m_raw_h: [cubecl::server::Handle; 3] = [
                            transient.m_raw[0].clone(),
                            transient.m_raw[1].clone(),
                            transient.m_raw[2].clone(),
                        ];
                        let m_mid_h: [cubecl::server::Handle; 3] = [
                            transient.m_mid[0].clone(),
                            transient.m_mid[1].clone(),
                            transient.m_mid[2].clone(),
                        ];
                        let m_blur_h: [cubecl::server::Handle; 3] = [
                            transient.m_blur[0].clone(),
                            transient.m_blur[1].clone(),
                            transient.m_blur[2].clone(),
                        ];
                        self._run_band_masking_strip_walker(
                            k,
                            bw,
                            bh,
                            n_px,
                            pu_scale,
                            &t_p_ref_h,
                            &t_p_dis_h,
                            &m_raw_h,
                            &m_mid_h,
                            &m_blur_h,
                            &d_h,
                        );
                    } else if use_blur {
                        // min_abs → pu_blur_h → pu_blur_v → mult_mutual_3ch_with_blurred.
                        let m_raw_h: [cubecl::server::Handle; 3] = [
                            transient.m_raw[0].clone(),
                            transient.m_raw[1].clone(),
                            transient.m_raw[2].clone(),
                        ];
                        min_abs_3ch_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[2].clone(), n_px),
                            n_px as u32,
                        );
                        // PU blur: 3-channel h pass → 3-channel v pass
                        // with `* 10^MASK_C` post-scale folded in (tick
                        // 92). One launch each = 2 launches total
                        // instead of 9 (3× pu_blur_h + 3× pu_blur_v +
                        // 3× weight_band).
                        let m_mid_h: [cubecl::server::Handle; 3] = [
                            transient.m_mid[0].clone(),
                            transient.m_mid[1].clone(),
                            transient.m_mid[2].clone(),
                        ];
                        let m_blur_h: [cubecl::server::Handle; 3] = [
                            transient.m_blur[0].clone(),
                            transient.m_blur[1].clone(),
                            transient.m_blur[2].clone(),
                        ];
                        pu_blur_h_3ch_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(m_raw_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_raw_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[2].clone(), n_px),
                            bw as u32,
                            bh as u32,
                        );
                        pu_blur_v_3ch_scaled_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(m_mid_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_mid_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[2].clone(), n_px),
                            pu_scale,
                            bw as u32,
                            bh as u32,
                        );
                        mult_mutual_3ch_with_blurred_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(m_blur_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                            n_px as u32,
                        );
                    } else {
                        // Small band: inline no-blur masker (band ≤ PU_PADSIZE).
                        // In Mode B `d_h` points at `d_strip` which is
                        // allocated to hold `n_px` at this level (the
                        // build_d_bands_scratch `use_blur=false` branch).
                        mult_mutual_3ch_no_blur_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(t_p_dis_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_dis_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(t_p_ref_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                            n_px as u32,
                        );
                    }
                }
                // Path A Phase 1d: Mode B's inline pool dispatch for
                // the non-strip-walker masking paths (no_blur small
                // bands; also the Full/CachedRef use_blur paths that
                // don't fire in Mode B but defensive isolation keeps
                // the conditional uniform). The strip walker already
                // dispatches its own per-strip pool inside the loop;
                // here we only need to cover the cases where the band
                // was written as a single full-band dispatch.
                //
                // Mode B's only non-strip-walker path is the small-
                // band no_blur branch (use_blur=false). In that case
                // d_strip was sized at n_px so the kernel above wrote
                // n_px elements; we now atomic-add the whole band's
                // contribution into partials_h[k * 3 .. + 3].
                if mode_b && !use_blur {
                    let partial_idx_a = (k * N_CHANNELS) as u32;
                    let partial_idx_rg = (k * N_CHANNELS + 1) as u32;
                    let partial_idx_vy = (k * N_CHANNELS + 2) as u32;
                    unsafe {
                        pool_band_3ch_offset_kernel::launch::<R>(
                            &self.client,
                            count.clone(),
                            cube_dim,
                            ArrayArg::from_raw_parts(d_h[0].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[1].clone(), n_px),
                            ArrayArg::from_raw_parts(d_h[2].clone(), n_px),
                            ArrayArg::from_raw_parts(
                                self.partials_h.clone(),
                                (self.n_levels as usize) * N_CHANNELS,
                            ),
                            BETA_SPATIAL,
                            partial_idx_a,
                            partial_idx_rg,
                            partial_idx_vy,
                            0_u32,
                            n_px as u32,
                            n_px as u32,
                        );
                    }
                }
                let _ = d_h; // non-baseband result lives in d_scratch[k].d[c]
            }
            if trace {
                eprintln!(
                    "[trace] L{k} mask:                  {:?}   (band total: {:?})",
                    t_mask.elapsed(),
                    t_band.elapsed()
                );
            }
        }
        if trace {
            eprintln!(
                "[trace] band loop total ({n_levels} levels): {:?}",
                t_band_loop.elapsed()
            );
        }

        Ok(())
    }

    /// P2.1c strip-major-outer band loop for shallow levels in Mode
    /// B. (2026-05-27)
    ///
    /// **Dispatch shape** (vs the existing level-major-outer loop in
    /// `_run_d_bands_band_loop`):
    ///
    /// ```text
    /// Allocate full-image DBandsTransient for each shallow level k in
    /// 0..k_split (k_split levels alive simultaneously).
    /// For each strip s in 0..n_strips_at_level_0:
    ///     For each k in 0..k_split:
    ///         CSF strip helper (P2.1b) writes body+halo of t_p_*[k].
    ///         Masking strip helper (P2.1a) reads body+halo of t_p_*[k]
    ///                                       writes body of d_strip[k]
    ///                                       inline pool into partials_h.
    /// Drop all shallow transients.
    /// ```
    ///
    /// **Why strip-major outer.** The masking V-blur reads m_mid rows
    /// in the body+halo window of `t_p_*[k]` (chained through
    /// min_abs/H-blur from t_p_*[k]). P2.1b's body+halo CSF dispatch
    /// populates `t_p_*[k]` for the strip's body+halo rows, so the
    /// masking helper reads its own halo from its own CSF dispatch
    /// instead of relying on sibling strips' contributions — the
    /// strict precondition for strip-major-outer correctness.
    ///
    /// **JOD invariant.** Bit-identical to today's level-major-outer
    /// dispatch because:
    ///   1. CSF for each (s, k) writes deterministic values at every
    ///      global row (band_ref + log_l_bkg are full-image; the
    ///      strip-aware upscale + subtract_weber are pure functions
    ///      of the global row index via `zy_base = local_y +
    ///      body_offset_y`). Halo overlap rows receive bit-identical
    ///      writes from any strip touching them.
    ///   2. Masking strip helper's halo m_mid reads pull from t_p_*[k]
    ///      that this strip just wrote (body+halo from CSF), so the
    ///      read pattern is identical to today's where each strip's
    ///      masking ran after ALL strips' CSF at level k.
    ///   3. mult_mutual writes only body rows of d_strip[k]; the
    ///      inline pool dispatch reads body rows and atomic-adds into
    ///      partials_h. Atomic-f32 reduction order can vary across
    ///      strips but the values at each global row are identical
    ///      to the level-major-outer dispatch.
    ///
    /// **Memory cost.** k_split full-image DBandsTransients alive
    /// simultaneously. At 4096² h_body=256 that's 5 transients
    /// (k_split=5), ~250 MiB additional vs the lazy-per-level path.
    /// P2.4-P2.5 will shrink these transients to per-strip sizes,
    /// recovering the memory plus more. P2.1c is the enabling
    /// refactor; the user-visible memory drop arrives with those
    /// shrinks.
    ///
    /// **Constraints**:
    ///   - `k_split` ≥ 1 (caller checks; we still bail safely).
    ///   - Mode B `strip_config.h_body` is Some and h_body > 0
    ///     (enforced by `Cvvdp::new_strip_pair`).
    ///   - Per-strip d_strip buffer exists at every shallow level
    ///     (`build_d_bands_scratch` allocates under StripPair).
    ///
    /// # Errors
    ///
    /// Propagates errors from
    /// [`Self::_dispatch_dist_weber_csf_strip_s_for_level`] (the only
    /// fallible op in the chain). Masking helper is infallible.
    fn _run_d_bands_strip_major_shallow(&mut self, k_split_us: usize) -> Result<()> {
        // PU blur post-scale (10^MASK_C) — constant per Cvvdp config.
        let pu_scale = 10.0_f32.powf(MASK_C);

        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return Err(Error::InvalidImageSize),
        };

        let n_strips_at_0 = {
            let fine_h_0 = self.gauss_ref[0].h;
            if fine_h_0 <= h_body_at_0 {
                1
            } else {
                fine_h_0.div_ceil(h_body_at_0)
            }
        };

        // Allocate full-image DBandsTransients for every shallow
        // level upfront. These live for the whole strip-major-outer
        // phase; each strip's CSF + masking writes into the level-k
        // transient. After the phase, the transients drop (cubecl
        // pool recycles).
        let mut transients: Vec<DBandsTransient> = Vec::with_capacity(k_split_us);
        // P2.4 (2026-05-27): allocate transients at strip size
        // (`bw × R_k`) instead of full band (`bw × bh`). Each (s, k)
        // iteration overwrites the buffer in place before the masking
        // chain consumes it within the same iteration — no
        // cross-strip data dependency on t_p_* / m_*. Saves ~`(bh -
        // R_k) / bh` per level at shallow levels (e.g. at 4096²
        // h_body=256 level 0: bh=4096, R_k=572 → ~86% per-level shrink
        // of 5 buffer kinds × 3 channels = ~660 MiB total across
        // shallow levels).
        let n_levels_u32 = self.n_levels;
        let k_split = mode_b_k_split(h_body_at_0, n_levels_u32);
        for k in 0..k_split_us {
            let (bw, bh, _n_px) = self.level_dims(k);
            // R_k strip height — matches build_weber_scratch sizing.
            let r_k = {
                let r_back = mode_b_strip_h_at_level(k as u32, h_body_at_0, k_split);
                if r_back == 0 {
                    ((h_body_at_0 as usize) >> k).max(1).min(bh)
                } else {
                    (r_back as usize).min(bh)
                }
            };
            let n_strip = bw * r_k;
            transients.push(DBandsTransient::new_strip(&self.client, n_strip));
        }

        // Resolve REF source per shallow level once (mode-dependent,
        // but uniform across strips within this phase). For Mode B
        // (StripPair) the REF data lives in `self.bands_ref[k].planes`
        // and `self.weber_scratch[k].log_l_bkg`. Mode E (CachedRef)
        // does NOT use this strip-major-outer path — see caller's
        // mode_b_pair gate.
        // Resolve the (ch_gain_a, ch_gain_rg, ch_gain_vy) tuple per
        // level once. `is_baseband = false` for k < k_split (k_split
        // never reaches n_levels-1 for h_body satisfying the design
        // table — design table caps k_split at log2(h_body / 12)+1).
        struct ShallowLevelInputs {
            band_ref_a: cubecl::server::Handle,
            band_ref_rg: cubecl::server::Handle,
            band_ref_vy: cubecl::server::Handle,
            log_l_bkg: cubecl::server::Handle,
            ch_gain: [f32; N_CHANNELS],
            bw: usize,
            bh: usize,
        }
        let mut per_level_inputs: Vec<ShallowLevelInputs> =
            Vec::with_capacity(k_split_us);
        for k in 0..k_split_us {
            let (bw, bh, _n_px) = self.level_dims(k);
            // band_mul = 1.0 at k=0, else 2.0 (shallow levels never
            // include baseband).
            let band_mul: f32 = if k == 0 { 1.0 } else { 2.0 };
            let ch_gain = ch_gain_for_band(false, band_mul);
            // P2.3 (2026-05-27): for shallow non-baseband levels in
            // Mode B, bands_ref[k].planes is zero-size — the REF
            // strip helper writes to `weber_scratch[k].bands_ref_strip`
            // per (s, k). The CSF helper reads from these strip-local
            // handles via `band_ref_strip_local: true`. Hold strip
            // handles in `per_level_inputs` so they outlive the
            // strip-major outer loop without per-iteration clones.
            let bands_ref_strip = self.weber_scratch[k].bands_ref_strip.as_ref().expect(
                "WeberScratch.bands_ref_strip must be Some at shallow non-baseband levels \
                 (k < k_split) in StripMode::Pair — build_weber_scratch allocates them.",
            );
            per_level_inputs.push(ShallowLevelInputs {
                band_ref_a: bands_ref_strip[0].clone(),
                band_ref_rg: bands_ref_strip[1].clone(),
                band_ref_vy: bands_ref_strip[2].clone(),
                log_l_bkg: self.weber_scratch[k].log_l_bkg.clone(),
                ch_gain,
                bw,
                bh,
            });
        }

        // Strip-major outer over shallow levels.
        for s in 0..n_strips_at_0 {
            for k in 0..k_split_us {
                let inp = &per_level_inputs[k];
                let band_ref_a = &inp.band_ref_a;
                let band_ref_rg = &inp.band_ref_rg;
                let band_ref_vy = &inp.band_ref_vy;
                let log_l_bkg_full = &inp.log_l_bkg;
                let ch_gain = &inp.ch_gain;
                let bw = &inp.bw;
                let bh = &inp.bh;

                // Strip index at level k is `s` because all shallow
                // levels' strips share the level-0 strip partition
                // (strip s at level k covers rows
                // `[s*strip_h_at_k, s*strip_h_at_k + strip_h_at_k)`
                // where `strip_h_at_k = (h_body >> k).max(1)`).
                let t_p_ref_h: [cubecl::server::Handle; 3] = [
                    transients[k].t_p_ref[0].clone(),
                    transients[k].t_p_ref[1].clone(),
                    transients[k].t_p_ref[2].clone(),
                ];
                let t_p_dis_h: [cubecl::server::Handle; 3] = [
                    transients[k].t_p_dis[0].clone(),
                    transients[k].t_p_dis[1].clone(),
                    transients[k].t_p_dis[2].clone(),
                ];

                // Stage A0: P2.3 REF strip helper writes
                // `bands_ref_strip[k]` body+halo from `gauss_alt`
                // (REF gauss data preserved post-swap). Runs once per
                // (s, k) strictly before the CSF helper that reads
                // bands_ref_strip in lockstep. Mode B only — Mode E
                // doesn't take this path.
                self._dispatch_ref_weber_strip_s_for_level(s, k)?;

                // Stage A: P2.1b body+halo CSF strip helper. P2.3:
                // band_ref handles are strip-local (point at
                // bands_ref_strip[c] populated by Stage A0).
                self._dispatch_dist_weber_csf_strip_s_for_level(
                    s,
                    k,
                    band_ref_a,
                    band_ref_rg,
                    band_ref_vy,
                    log_l_bkg_full,
                    &t_p_ref_h,
                    &t_p_dis_h,
                    *ch_gain,
                    true, // band_ref_strip_local (P2.3)
                )?;

                // Stage B: P2.1a masking strip helper. Reads body+halo
                // of t_p_*[k] (populated by Stage A), writes body of
                // d_strip[k], inline-pools into partials_h.
                let scratch_d = &self.d_scratch[k];
                let d_strip = scratch_d.d_strip.as_ref().expect(
                    "DBandsScratch.d_strip must be Some at non-baseband levels in StripMode::Pair",
                );
                let d_h: [cubecl::server::Handle; 3] = [
                    d_strip[0].clone(),
                    d_strip[1].clone(),
                    d_strip[2].clone(),
                ];

                // Masking transients for this level.
                let m_raw_h: [cubecl::server::Handle; 3] = [
                    transients[k].m_raw[0].clone(),
                    transients[k].m_raw[1].clone(),
                    transients[k].m_raw[2].clone(),
                ];
                let m_mid_h: [cubecl::server::Handle; 3] = [
                    transients[k].m_mid[0].clone(),
                    transients[k].m_mid[1].clone(),
                    transients[k].m_mid[2].clone(),
                ];
                let m_blur_h: [cubecl::server::Handle; 3] = [
                    transients[k].m_blur[0].clone(),
                    transients[k].m_blur[1].clone(),
                    transients[k].m_blur[2].clone(),
                ];

                // Masking helper requires `use_blur` semantics. For
                // shallow levels at production sizes, bw and bh are
                // always > PU_PADSIZE (k_split is chosen so all
                // shallow levels have body dim ≥ 12). If somehow a
                // tiny shallow level lands here, fall back to the
                // no-blur dispatch with single full-band launch +
                // pool. The strip helper itself is the use_blur path.
                let use_blur = *bw > PU_PADSIZE && *bh > PU_PADSIZE;
                if use_blur {
                    self._run_band_masking_strip_s_for_level(
                        s as usize,
                        k,
                        *bw,
                        *bh,
                        pu_scale,
                        &t_p_ref_h,
                        &t_p_dis_h,
                        &m_raw_h,
                        &m_mid_h,
                        &m_blur_h,
                        &d_h,
                        true, // P2.4: transients are strip-local
                    );
                } else {
                    // Small shallow level (shouldn't happen at
                    // production sizes given k_split's design table,
                    // but stays correct as a defensive fallback).
                    // Run the no-blur dispatch + pool on this strip's
                    // body rows. n_px here equals bw*bh — the full
                    // band — because the no_blur fallback dispatches
                    // over the whole band rather than per-strip. So
                    // we only fire this branch once per band (when s
                    // == 0) to avoid double-counting.
                    if s == 0 {
                        self._run_band_masking_strip_s_for_level(
                            s as usize,
                            k,
                            *bw,
                            *bh,
                            pu_scale,
                            &t_p_ref_h,
                            &t_p_dis_h,
                            &m_raw_h,
                            &m_mid_h,
                            &m_blur_h,
                            &d_h,
                            true, // P2.4: transients are strip-local
                        );
                    }
                }
            }
        }

        // Drop transients implicitly here (Vec goes out of scope at
        // function return). cubecl pool reclaims the GPU memory.
        let _ = transients;
        Ok(())
    }

    /// **Path A bands_dis shrink — fused DIST Weber + csf strip
    /// walker for a single non-baseband level in Mode B.** Replaces
    /// the standard pre-band-loop DIST Weber finalize (which would
    /// write to the now-zero-size `bands_dis[k].planes`) with a
    /// per-strip pipeline that:
    ///
    /// 1. Runs the Weber strip kernels for level `k` (upscale +
    ///    subtract+weber) writing one strip's worth of band data
    ///    into the strip-local [`WeberScratch::bands_dis_strip`].
    /// 2. Runs `csf_apply_6ch_kernel` over the same strip's body
    ///    rows, reading `bands_dis_strip` (strip-local) + bands_ref
    ///    (full, sliced at body offset) + log_l_bkg (full, sliced)
    ///    and writing the per-strip body of the full-image
    ///    `t_p_ref` / `t_p_dis` transients passed in.
    ///
    /// The two kernels must run in lockstep per strip — the next
    /// strip's Weber dispatch overwrites `bands_dis_strip`, so csf
    /// must consume it before the loop advances. The masking strip
    /// walker dispatched after this function returns reads the
    /// full-image `t_p_*` (now populated for all strips of level k)
    /// and writes per-strip body rows of `d_scratch[k].d[c]`.
    ///
    /// `band_ref_a/rg/vy` are full-image handles for level k's REF
    /// bands. `log_l_bkg_full` is the full-image log_l_bkg for the
    /// REF side (cvvdp uses REF's log_l_bkg for both csf inputs).
    /// `t_p_ref_h`, `t_p_dis_h` are full-image-sized per-channel
    /// transient buffers — this walker writes strip body rows; the
    /// masking walker reads them.
    #[allow(clippy::too_many_arguments)]
    fn _dispatch_dist_weber_csf_strip_walker_for_level(
        &self,
        k: usize,
        band_ref_a: cubecl::server::Handle,
        band_ref_rg: cubecl::server::Handle,
        band_ref_vy: cubecl::server::Handle,
        log_l_bkg_full: cubecl::server::Handle,
        t_p_ref_h: &[cubecl::server::Handle; 3],
        t_p_dis_h: &[cubecl::server::Handle; 3],
        ch_gain: [f32; 3],
        _band_mul: f32,
    ) -> Result<()> {
        // P2.1a refactor (2026-05-27): the body of this walker is now
        // the per-strip helper `_dispatch_dist_weber_csf_strip_s_for_level`.
        // The wrapper iterates `s in 0..n_strips` and dispatches each
        // strip individually. This factorisation is JOD bit-identical
        // (same kernel launches in the same order) and positions the
        // strip-major outer dispatch (P2.1b/c) to call the per-strip
        // helper directly without re-implementing stages 1-4. See
        // docs/STRIP_PROCESSING.md#phase-2--p21-implementation-analysis-2026-05-27
        // for the canary decomposition.
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => {
                // Caller (the band loop) only invokes this when Mode
                // B is active; if strip_config is somehow None here
                // the buffers we'd use are not allocated. Treat as
                // hard invariant and bail.
                return Err(Error::InvalidImageSize);
            }
        };

        let fine_h = self.gauss_ref[k].h;
        let strip_h_at_k = (h_body_at_0 >> k).max(1);
        let n_strips = if fine_h <= strip_h_at_k {
            1
        } else {
            fine_h.div_ceil(strip_h_at_k)
        };

        for s in 0..n_strips {
            self._dispatch_dist_weber_csf_strip_s_for_level(
                s,
                k,
                &band_ref_a,
                &band_ref_rg,
                &band_ref_vy,
                &log_l_bkg_full,
                t_p_ref_h,
                t_p_dis_h,
                ch_gain,
                false, // band_ref_strip_local: legacy caller passes full-image handles
            )?;
        }

        Ok(())
    }

    /// P2.1a per-strip helper (2026-05-27). Runs stages 1-4 of the
    /// fused DIST Weber + csf strip walker for a SINGLE strip
    /// `s` at non-baseband level `k` in Mode B.
    ///
    /// Stages (mirrors the previous in-loop body of
    /// `_dispatch_dist_weber_csf_strip_walker_for_level`):
    /// 1. Upscale_v/h of coarse A → `l_bkg_fine` strip body.
    /// 2. Per-channel upscale of coarse → `upscaled_c_strip`
    ///    (strip-local; overwritten each strip iteration).
    /// 3. Fused `subtract_weber_3ch_strip` → writes per-strip
    ///    `bands_dis_strip[c]` + body rows of full-image
    ///    `log_l_bkg_dis` (throwaway).
    /// 4. Fused `csf_apply_6ch` → writes body rows of full-image
    ///    `t_p_ref[c]` / `t_p_dis[c]`.
    ///
    /// **JOD invariant.** This helper is the inner-loop body of the
    /// old strip walker, extracted verbatim. Two consecutive calls
    /// with `s = 0` then `s = 1` produce bit-identical state to the
    /// old wrapper's `for s in 0..2`. Strip-major-outer callers can
    /// interleave `(s, k)` ordering as long as the per-`(s, k)` body
    /// order (stages 1→2→3→4) is preserved within each call.
    ///
    /// `s` must be `< n_strips_at_level_k`. The caller is responsible
    /// for that check; the helper does not validate.
    ///
    /// `band_ref_*` and `log_l_bkg_full` are borrowed (not owned) so
    /// the per-strip caller can pass through to many `(s, k)` calls
    /// without per-call handle clones bumping refcounts at the
    /// cubecl layer.
    ///
    /// **P2.1b body+halo dispatch (2026-05-27).** This helper now
    /// dispatches stages 1-4 over the **body+halo window** at level k
    /// for shallow levels (`k < k_split`). The window is
    /// `[top_global, bot_global)` where `top_global = body_offset_y -
    /// halo_band` (clamped to ≥0) and `bot_global = body_offset_y +
    /// body_h + halo_band` (clamped to ≤ fine_h), with `halo_band =
    /// mode_b_halo_at_level(k, k_split) = 8` for shallow levels.
    ///
    /// **Why dispatch wider than body?** When the strip-major-outer
    /// caller (P2.1c) runs masking per (s, k), the masking V-blur
    /// reads m_mid rows from this strip's body+6-halo window, which
    /// chains back to needing `t_p_*[k]` body+6-halo. We populate
    /// `t_p_*[k]` over body+8 (slight slack vs the 6-tap V-blur half)
    /// per strip's CSF dispatch so masking finds valid data in its own
    /// halo window without depending on sibling strips' csf calls.
    ///
    /// For deep levels (`k >= k_split`), `mode_b_halo_at_level`
    /// returns 0, so the halo is zero and dispatch reduces to body-
    /// only — matching today's behaviour (deep levels stay level-major-
    /// outer and rely on full-image t_p_*[k] population by all strips
    /// at level k via the level-major caller).
    ///
    /// **JOD invariant under level-major-outer caller.** Until P2.1c
    /// flips the outer loop, the existing
    /// `_dispatch_dist_weber_csf_strip_walker_for_level` calls this
    /// helper for `s in 0..n_strips`. Each strip s's CSF now writes
    /// body+halo of `t_p_*[k]`, overlapping strip s±1's body writes.
    /// Determinism: the CSF kernel reads `band_ref_*` + `log_l_bkg`
    /// (full-image, same data per global row), and `bands_dis_strip`
    /// (per-strip — but for the same global row, stages 1-3 produce
    /// bit-identical values across strips because the upscale + weber
    /// kernels are pure functions of the global row index). Final
    /// write wins on overlap rows; all writes produce the same value.
    ///
    /// **Strip-local buffer indexing (P2.1b).** `upscaled_c_strip` and
    /// `bands_dis_strip` are sized at `fine_w × R_k` (R_k =
    /// `mode_b_strip_h_at_level`). For this strip's body+halo window,
    /// the buffer's row 0 corresponds to global row `top_global`. All
    /// strip-aware kernels accept `src_strip_offset = top_global` so
    /// they translate `body_offset_y + dy_local → top_global +
    /// local_y` correctly.
    ///
    /// `s` must be `< n_strips_at_level_k`. The caller is responsible
    /// for that check; the helper does not validate.
    ///
    /// `band_ref_*` and `log_l_bkg_full` are borrowed (not owned) so
    /// the per-strip caller can pass through to many `(s, k)` calls
    /// without per-call handle clones bumping refcounts at the
    /// cubecl layer.
    #[allow(clippy::too_many_arguments)]
    fn _dispatch_dist_weber_csf_strip_s_for_level(
        &self,
        s: u32,
        k: usize,
        band_ref_a: &cubecl::server::Handle,
        band_ref_rg: &cubecl::server::Handle,
        band_ref_vy: &cubecl::server::Handle,
        log_l_bkg_full: &cubecl::server::Handle,
        t_p_ref_h: &[cubecl::server::Handle; 3],
        t_p_dis_h: &[cubecl::server::Handle; 3],
        ch_gain: [f32; 3],
        // P2.3 (2026-05-27): when `true`, the `band_ref_*` handles are
        // strip-local with row 0 corresponding to global row
        // `top_global` (matching the `bands_dis_strip` /
        // `upscaled_c_strip` shape). When `false` (legacy), they are
        // full-image handles that need slicing at
        // `byte_off_fine_window` for the strip's body+halo window.
        band_ref_strip_local: bool,
    ) -> Result<()> {
        let cube_dim = CubeDim::new_1d(64);
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return Err(Error::InvalidImageSize),
        };

        let coarse_w = self.gauss_ref[k + 1].w;
        let coarse_h = self.gauss_ref[k + 1].h;
        let fine_w = self.gauss_ref[k].w;
        let fine_h = self.gauss_ref[k].h;
        let n_coarse = (coarse_w * coarse_h) as usize;
        let n_levels = self.n_levels;
        let k_split = mode_b_k_split(h_body_at_0, n_levels);

        let strip_h_at_k = (h_body_at_0 >> k).max(1);

        let scratch = &self.weber_scratch[k];
        let log_l_bkg_dis_dest = scratch.log_l_bkg_dis.clone();

        let bands_dis_strip = scratch
            .bands_dis_strip
            .as_ref()
            .expect(
                "bands_dis_strip is None in Mode B fused walker; \
                 build_weber_scratch must allocate it under StripMode::Pair",
            );

        let upscaled_c_strip = scratch
            .upscaled_c_strip
            .as_ref()
            .expect(
                "upscaled_c_strip is None in Mode B fused walker; \
                 build_weber_scratch must allocate it under StripMode::Pair",
            );

        let body_offset_y = s * strip_h_at_k;
        let body_h = (fine_h - body_offset_y).min(strip_h_at_k);

        // P2.1b body+halo dispatch window. For shallow levels
        // (k < k_split) halo_band = 8 covers PU blur ±6 + 2 rows of
        // pyramid downscale slack. For deep levels halo_band = 0
        // (deep levels rely on level-major full-image t_p_* state).
        // The strip-local buffer's row 0 maps to global row top_global
        // via src_strip_offset = top_global.
        let halo_band = mode_b_halo_at_level(k as u32, k_split);
        let top_global = body_offset_y.saturating_sub(halo_band);
        let bot_global = (body_offset_y + body_h + halo_band).min(fine_h);
        let strip_window_h = bot_global - top_global;
        // R_k size of the per-strip buffers (must match
        // build_weber_scratch's sizing model exactly so ArrayArg
        // length checks don't fire). Deep levels (k >= k_split) get
        // body-only sizing; shallow levels get back-projected R_k.
        let r_k = {
            let r_back = mode_b_strip_h_at_level(k as u32, h_body_at_0, k_split);
            if r_back == 0 {
                strip_h_at_k.min(fine_h)
            } else {
                r_back.min(fine_h)
            }
        };
        let n_strip_buf = (fine_w as usize) * (r_k as usize);
        debug_assert!(
            strip_window_h <= r_k,
            "strip_window_h={} exceeds R_k={} for (k={}, h_body={}, k_split={})",
            strip_window_h,
            r_k,
            k,
            h_body_at_0,
            k_split,
        );

        let n_strip_v_window = (coarse_w as usize) * (strip_window_h as usize);
        let n_strip_window = (fine_w as usize) * (strip_window_h as usize);
        let count_v_window = CubeCount::Static((n_strip_v_window as u32).div_ceil(64), 1, 1);
        let count_window = CubeCount::Static((n_strip_window as u32).div_ceil(64), 1, 1);

        // `byte_off_*_full` slice FULL-image buffers (gauss_ref, plus
        // any weber_scratch buffer that was NOT strip-shaped this
        // commit). Always required: gauss_ref is full-image in P2.3
        // and only retires to strip in P2.7.
        let byte_off_v_full: u64 = u64::from(top_global) * u64::from(coarse_w) * 4;
        let byte_off_fine_full: u64 = u64::from(top_global) * u64::from(fine_w) * 4;

        // P2.6 (2026-05-27): when `band_ref_strip_local`, the caller
        // also allocated `vscratch_a` / `vscratch_c` / `l_bkg_fine` /
        // `log_l_bkg` / `log_l_bkg_dis` at strip dims (per shallow
        // level). These buffers no longer need slicing — buffer row 0
        // IS the strip start. The strip-aware kernels still get
        // `body_off_kernel = top_global` so the reflection math is
        // unchanged.
        let (byte_off_v_window, byte_off_fine_window) = if band_ref_strip_local {
            (0_u64, 0_u64)
        } else {
            (byte_off_v_full, byte_off_fine_full)
        };

        // Stage 1: upscale_v/h of coarse A → l_bkg_fine body+halo.
        // l_bkg_fine is full-image; we slice it at top_global and
        // write strip_window_h rows. The strip-aware upscale kernel
        // takes body_offset_y to anchor its destination rows; we
        // pass top_global so the kernel iterates global rows
        // [top_global, top_global + strip_window_h).
        let coarse_a = self.gauss_ref[k + 1].planes[0].clone();
        let vscratch_a_strip = scratch.vscratch_a.clone().offset_start(byte_off_v_window);
        let l_bkg_fine_strip = scratch
            .l_bkg_fine
            .clone()
            .offset_start(byte_off_fine_window);
        unsafe {
            upscale_v_strip_kernel::launch::<R>(
                &self.client,
                count_v_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(coarse_a, n_coarse),
                ArrayArg::from_raw_parts(vscratch_a_strip.clone(), n_strip_v_window),
                coarse_w,
                coarse_h,
                fine_h,
                top_global,
                strip_window_h,
                0,
            );
            upscale_h_strip_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(vscratch_a_strip, n_strip_v_window),
                ArrayArg::from_raw_parts(l_bkg_fine_strip, n_strip_window),
                coarse_w,
                fine_w,
                strip_window_h,
                fine_h,
                top_global,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 2: per-channel upscale → upscaled_c_strip (R_k rows).
        // Buffer row 0 corresponds to global row top_global per
        // src_strip_offset semantics. The strip-aware kernels here
        // accept `src_strip_offset = 0` for inputs (they read coarse
        // = full-image) but the dest stride is the strip-local
        // R_k-sized buffer. The horizontal kernel doesn't care about
        // y-axis offsets so it processes strip_window_h rows from
        // vscratch_c (also strip-local).
        for c in 0..N_CHANNELS {
            let coarse = self.gauss_ref[k + 1].planes[c].clone();
            let vscratch_c_strip = scratch.vscratch_c[c].clone().offset_start(byte_off_v_window);
            unsafe {
                upscale_v_strip_kernel::launch::<R>(
                    &self.client,
                    count_v_window.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(coarse, n_coarse),
                    ArrayArg::from_raw_parts(vscratch_c_strip.clone(), n_strip_v_window),
                    coarse_w,
                    coarse_h,
                    fine_h,
                    top_global,
                    strip_window_h,
                    0,
                );
                upscale_h_strip_kernel::launch::<R>(
                    &self.client,
                    count_window.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(vscratch_c_strip, n_strip_v_window),
                    ArrayArg::from_raw_parts(upscaled_c_strip[c].clone(), n_strip_buf),
                    coarse_w,
                    fine_w,
                    strip_window_h,
                    fine_h,
                    top_global,
                );
            }
            self.strip_dispatch_counter
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }

        // Stage 3: subtract_weber → writes bands_dis_strip over the
        // body+halo window. The kernel uses a single src_strip_offset
        // shared across all buffers; we pass `top_global` so the
        // strip-local buffer's row 0 corresponds to global row
        // top_global. Full-image inputs (`fine_a/rg/vy`, `l_bkg_fine`,
        // `log_l_bkg_dis`) are pre-sliced by `byte_off_fine_window`
        // so each buffer's row 0 also lands at top_global.
        let fine_a_full = self.gauss_ref[k].planes[0].clone();
        let fine_rg_full = self.gauss_ref[k].planes[1].clone();
        let fine_vy_full = self.gauss_ref[k].planes[2].clone();
        unsafe {
            subtract_weber_3ch_strip_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                // gauss_ref planes are full-image; always slice at full top_global.
                ArrayArg::from_raw_parts(
                    fine_a_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(
                    fine_rg_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(
                    fine_vy_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(upscaled_c_strip[0].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(upscaled_c_strip[1].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(upscaled_c_strip[2].clone(), n_strip_buf),
                // P2.6: l_bkg_fine + log_l_bkg_dis are weber_scratch;
                // strip-shaped (offset 0) under strip-major outer,
                // full-image (offset top_global) otherwise.
                ArrayArg::from_raw_parts(
                    scratch
                        .l_bkg_fine
                        .clone()
                        .offset_start(byte_off_fine_window),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(bands_dis_strip[0].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(bands_dis_strip[1].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(bands_dis_strip[2].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(
                    log_l_bkg_dis_dest
                        .clone()
                        .offset_start(byte_off_fine_window),
                    n_strip_window,
                ),
                fine_w,
                strip_window_h,
                top_global,
                fine_h,
                top_global, // src_strip_offset — strip-local row 0 = global row top_global
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 4: csf_apply_6ch on this strip's body+halo window.
        // csf_apply_6ch is per-pixel (no neighborhood reads), takes a
        // flat element count `n`. We dispatch `n_strip_window`
        // threads and slice every full-image input by
        // byte_off_fine_window so row 0 corresponds to top_global.
        // bands_dis_strip / t_p_*_strip are strip-local; the t_p_*
        // writes hit the body+halo rows of the full-image transient.
        // P2.3 (2026-05-27): when band_ref handles are strip-local
        // (row 0 = top_global), skip the slice. Strip-local handles
        // come from `bands_ref_strip` at the caller (shallow Mode B);
        // full-image handles come from `self.bands_ref[k].planes` and
        // need slicing to position row 0 at top_global.
        let (band_ref_a_strip, band_ref_rg_strip, band_ref_vy_strip) = if band_ref_strip_local {
            (band_ref_a.clone(), band_ref_rg.clone(), band_ref_vy.clone())
        } else {
            (
                band_ref_a.clone().offset_start(byte_off_fine_window),
                band_ref_rg.clone().offset_start(byte_off_fine_window),
                band_ref_vy.clone().offset_start(byte_off_fine_window),
            )
        };
        let log_l_bkg_strip = log_l_bkg_full.clone().offset_start(byte_off_fine_window);
        // P2.4 (2026-05-27): when `band_ref_strip_local`, the
        // strip-major outer caller passes strip-local t_p_* as well
        // (they're allocated together by `DBandsTransient::new_strip`).
        // The csf_apply_6ch kernel is per-pixel (no row math), so
        // skipping the slice merely changes which buffer rows the
        // writes land in — strip-local row 0 = top_global maps the
        // writes to the correct band+halo region of the strip buffer.
        // The masking helper then reads from those same rows.
        let (t_p_ref_a_strip, t_p_ref_rg_strip, t_p_ref_vy_strip,
             t_p_dis_a_strip, t_p_dis_rg_strip, t_p_dis_vy_strip) = if band_ref_strip_local {
            (
                t_p_ref_h[0].clone(),
                t_p_ref_h[1].clone(),
                t_p_ref_h[2].clone(),
                t_p_dis_h[0].clone(),
                t_p_dis_h[1].clone(),
                t_p_dis_h[2].clone(),
            )
        } else {
            (
                t_p_ref_h[0].clone().offset_start(byte_off_fine_window),
                t_p_ref_h[1].clone().offset_start(byte_off_fine_window),
                t_p_ref_h[2].clone().offset_start(byte_off_fine_window),
                t_p_dis_h[0].clone().offset_start(byte_off_fine_window),
                t_p_dis_h[1].clone().offset_start(byte_off_fine_window),
                t_p_dis_h[2].clone().offset_start(byte_off_fine_window),
            )
        };
        unsafe {
            csf_apply_6ch_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(band_ref_a_strip, n_strip_window),
                ArrayArg::from_raw_parts(band_ref_rg_strip, n_strip_window),
                ArrayArg::from_raw_parts(band_ref_vy_strip, n_strip_window),
                ArrayArg::from_raw_parts(bands_dis_strip[0].clone(), n_strip_window),
                ArrayArg::from_raw_parts(bands_dis_strip[1].clone(), n_strip_window),
                ArrayArg::from_raw_parts(bands_dis_strip[2].clone(), n_strip_window),
                ArrayArg::from_raw_parts(log_l_bkg_strip, n_strip_window),
                ArrayArg::from_raw_parts(self.logs_row[k][0].clone(), 32),
                ArrayArg::from_raw_parts(self.logs_row[k][1].clone(), 32),
                ArrayArg::from_raw_parts(self.logs_row[k][2].clone(), 32),
                ArrayArg::from_raw_parts(t_p_ref_a_strip, n_strip_window),
                ArrayArg::from_raw_parts(t_p_ref_rg_strip, n_strip_window),
                ArrayArg::from_raw_parts(t_p_ref_vy_strip, n_strip_window),
                ArrayArg::from_raw_parts(t_p_dis_a_strip, n_strip_window),
                ArrayArg::from_raw_parts(t_p_dis_rg_strip, n_strip_window),
                ArrayArg::from_raw_parts(t_p_dis_vy_strip, n_strip_window),
                ch_gain[0],
                ch_gain[1],
                ch_gain[2],
                n_strip_window as u32,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let _ = strip_h_at_k; // P2.1b: width derived from R_k; legacy strip_h_at_k unused

        Ok(())
    }

    /// **P2.3 (2026-05-27): per-strip REF weber finalize helper.**
    ///
    /// Mirrors stages 1-3 of [`Self::_dispatch_dist_weber_csf_strip_s_for_level`]
    /// (which run the DIST side + CSF) but writes only REF outputs:
    ///   - Stage 1: upscale_v/h of `gauss_alt[k+1] A` → `l_bkg_fine`
    ///     body+halo. **gauss_alt** holds REF gauss data (preserved
    ///     past the post-REF-weber swap; see
    ///     [`Self::_maybe_swap_gauss_alt_post_ref`]).
    ///   - Stage 2: per-channel separable upscale of
    ///     `gauss_alt[k+1] c` → `upscaled_c_strip[c]` body+halo.
    ///   - Stage 3: fused subtract+Weber → writes `bands_ref_strip[c]`
    ///     body+halo + `weber_scratch[k].log_l_bkg` body+halo (the
    ///     REF per-pixel log10(L_bkg) that the CSF stage 4 reads).
    ///
    /// The buffer layout is identical to the DIST helper: strip-local
    /// `bands_ref_strip` + `upscaled_c_strip` (row 0 = top_global);
    /// full-image-sliced `l_bkg_fine` + `log_l_bkg` (at
    /// `byte_off_fine_window`).
    ///
    /// **Lockstep contract:** caller invokes this helper for `(s, k)`
    /// immediately before [`Self::_dispatch_dist_weber_csf_strip_s_for_level`]
    /// for the same `(s, k)`. The CSF helper reads `bands_ref_strip`
    /// (just written by this helper) + `bands_dis_strip` (written by
    /// the DIST helper's stage 3) + `log_l_bkg` body+halo (written by
    /// this helper's stage 3). The next `s` overwrites
    /// `bands_ref_strip` / `bands_dis_strip` / `upscaled_c_strip` —
    /// strict lockstep is required.
    ///
    /// **gauss_alt dependency.** This helper hard-requires
    /// `self.gauss_alt = Some(_)` (allocated only in `StripMode::Pair`)
    /// AND that `_maybe_swap_gauss_alt_post_ref` has run after the
    /// most-recent REF weber finalize so `gauss_alt` holds REF gauss
    /// data. The `mode_b_pair` gate in `_run_d_bands_band_loop`
    /// ensures both invariants.
    ///
    /// Increments `strip_dispatch_counter` by 1 (stage 1) + N_CHANNELS
    /// (stage 2 per channel) + 1 (stage 3) = 5 per call.
    fn _dispatch_ref_weber_strip_s_for_level(&self, s: u32, k: usize) -> Result<()> {
        let cube_dim = CubeDim::new_1d(64);
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body,
            None => return Err(Error::InvalidImageSize),
        };

        let coarse_w = self.gauss_ref[k + 1].w;
        let coarse_h = self.gauss_ref[k + 1].h;
        let fine_w = self.gauss_ref[k].w;
        let fine_h = self.gauss_ref[k].h;
        let n_coarse = (coarse_w * coarse_h) as usize;
        let n_levels = self.n_levels;
        let k_split = mode_b_k_split(h_body_at_0, n_levels);

        let strip_h_at_k = (h_body_at_0 >> k).max(1);

        let scratch = &self.weber_scratch[k];
        // REF log_l_bkg destination — full-image, sliced. Mirrors the
        // pre-P2.3 `_finalize_weber_pyramid_strip_walker` REF path.
        let log_l_bkg_dest = scratch.log_l_bkg.clone();

        let bands_ref_strip = scratch.bands_ref_strip.as_ref().expect(
            "bands_ref_strip is None in Mode B per-strip REF helper; \
             build_weber_scratch must allocate it for k < k_split under StripMode::Pair",
        );
        let upscaled_c_strip = scratch.upscaled_c_strip.as_ref().expect(
            "upscaled_c_strip is None in Mode B per-strip REF helper; \
             build_weber_scratch must allocate it under StripMode::Pair",
        );

        // gauss_alt holds REF gauss data after the post-REF-weber swap.
        let gauss_alt = self.gauss_alt.as_ref().expect(
            "gauss_alt is None in Mode B per-strip REF helper; \
             new_with_geometry_inner must allocate it under StripMode::Pair",
        );

        let body_offset_y = s * strip_h_at_k;
        let body_h = (fine_h - body_offset_y).min(strip_h_at_k);

        // P2.1b body+halo dispatch window — identical to DIST helper.
        let halo_band = mode_b_halo_at_level(k as u32, k_split);
        let top_global = body_offset_y.saturating_sub(halo_band);
        let bot_global = (body_offset_y + body_h + halo_band).min(fine_h);
        let strip_window_h = bot_global - top_global;
        let r_k = {
            let r_back = mode_b_strip_h_at_level(k as u32, h_body_at_0, k_split);
            if r_back == 0 {
                strip_h_at_k.min(fine_h)
            } else {
                r_back.min(fine_h)
            }
        };
        let n_strip_buf = (fine_w as usize) * (r_k as usize);
        debug_assert!(
            strip_window_h <= r_k,
            "strip_window_h={} exceeds R_k={} for (k={}, h_body={}, k_split={})",
            strip_window_h,
            r_k,
            k,
            h_body_at_0,
            k_split,
        );

        let n_strip_v_window = (coarse_w as usize) * (strip_window_h as usize);
        let n_strip_window = (fine_w as usize) * (strip_window_h as usize);
        let count_v_window = CubeCount::Static((n_strip_v_window as u32).div_ceil(64), 1, 1);
        let count_window = CubeCount::Static((n_strip_window as u32).div_ceil(64), 1, 1);
        // gauss_alt is full-image: always slice at top_global * w * 4.
        // (`byte_off_v_full` unused — vscratch_a in stage 1 reads from
        // strip-shaped weber_scratch.vscratch_a with offset 0; gauss_alt
        // is the SRC for upscale_v, not a sliced read.)
        let _byte_off_v_full: u64 = u64::from(top_global) * u64::from(coarse_w) * 4;
        let byte_off_fine_full: u64 = u64::from(top_global) * u64::from(fine_w) * 4;
        // P2.6 (2026-05-27): REF strip helper always runs at shallow
        // Mode B (caller gates `k < k_split` in
        // `_run_d_bands_strip_major_shallow`), so the weber_scratch
        // buffers (vscratch_a, l_bkg_fine, log_l_bkg, vscratch_c) are
        // ALWAYS strip-shaped here — no slice needed. Byte offsets
        // are 0 for weber_scratch buffers, `byte_off_fine_full` for
        // gauss_alt stage-3 fine reads.
        let byte_off_v_window: u64 = 0;
        let byte_off_fine_window: u64 = 0;

        // Stage 1: upscale_v/h of gauss_alt[k+1] A → l_bkg_fine body+halo.
        // l_bkg_fine is full-image scratch shared with the DIST helper;
        // each strip iteration overwrites it sequentially (REF then DIST).
        let coarse_a = gauss_alt[k + 1].planes[0].clone();
        let vscratch_a_strip = scratch.vscratch_a.clone().offset_start(byte_off_v_window);
        let l_bkg_fine_strip = scratch
            .l_bkg_fine
            .clone()
            .offset_start(byte_off_fine_window);
        unsafe {
            upscale_v_strip_kernel::launch::<R>(
                &self.client,
                count_v_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(coarse_a, n_coarse),
                ArrayArg::from_raw_parts(vscratch_a_strip.clone(), n_strip_v_window),
                coarse_w,
                coarse_h,
                fine_h,
                top_global,
                strip_window_h,
                0,
            );
            upscale_h_strip_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(vscratch_a_strip, n_strip_v_window),
                ArrayArg::from_raw_parts(l_bkg_fine_strip, n_strip_window),
                coarse_w,
                fine_w,
                strip_window_h,
                fine_h,
                top_global,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 2: per-channel separable upscale → upscaled_c_strip
        // body+halo (R_k rows). Strip-local; row 0 = top_global.
        for c in 0..N_CHANNELS {
            let coarse = gauss_alt[k + 1].planes[c].clone();
            let vscratch_c_strip = scratch.vscratch_c[c].clone().offset_start(byte_off_v_window);
            unsafe {
                upscale_v_strip_kernel::launch::<R>(
                    &self.client,
                    count_v_window.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(coarse, n_coarse),
                    ArrayArg::from_raw_parts(vscratch_c_strip.clone(), n_strip_v_window),
                    coarse_w,
                    coarse_h,
                    fine_h,
                    top_global,
                    strip_window_h,
                    0,
                );
                upscale_h_strip_kernel::launch::<R>(
                    &self.client,
                    count_window.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(vscratch_c_strip, n_strip_v_window),
                    ArrayArg::from_raw_parts(upscaled_c_strip[c].clone(), n_strip_buf),
                    coarse_w,
                    fine_w,
                    strip_window_h,
                    fine_h,
                    top_global,
                );
            }
            self.strip_dispatch_counter
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }

        // Stage 3: subtract_weber → writes bands_ref_strip (strip-local)
        // + log_l_bkg (strip-shaped at P2.6, offset 0). The strip-aware
        // kernel uses `src_strip_offset = top_global` so every buffer's
        // row 0 corresponds to global row top_global — identical to
        // the DIST helper's stage 3 layout. gauss_alt fine_* are full-
        // image; slice at `byte_off_fine_full`. weber_scratch buffers
        // (l_bkg_fine, log_l_bkg) are strip-shaped; no slice.
        let fine_a_full = gauss_alt[k].planes[0].clone();
        let fine_rg_full = gauss_alt[k].planes[1].clone();
        let fine_vy_full = gauss_alt[k].planes[2].clone();
        unsafe {
            subtract_weber_3ch_strip_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(
                    fine_a_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(
                    fine_rg_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(
                    fine_vy_full.offset_start(byte_off_fine_full),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(upscaled_c_strip[0].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(upscaled_c_strip[1].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(upscaled_c_strip[2].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(
                    scratch
                        .l_bkg_fine
                        .clone()
                        .offset_start(byte_off_fine_window),
                    n_strip_window,
                ),
                ArrayArg::from_raw_parts(bands_ref_strip[0].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(bands_ref_strip[1].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(bands_ref_strip[2].clone(), n_strip_buf),
                ArrayArg::from_raw_parts(
                    log_l_bkg_dest.offset_start(byte_off_fine_window),
                    n_strip_window,
                ),
                fine_w,
                strip_window_h,
                top_global,
                fine_h,
                top_global, // src_strip_offset
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        let _ = strip_h_at_k;
        Ok(())
    }

    /// Mode B per-strip masking dispatch within a single non-baseband
    /// band (k = 0..n_levels-1). The full chain
    /// `min_abs → pu_blur_h → pu_blur_v_scaled → mult_mutual` is
    /// dispatched per strip over a halo-padded window of the band
    /// buffers.
    ///
    /// # Halo-padded strip convention
    ///
    /// V-blur uses a 13-tap kernel (half = 6), so each strip needs
    /// `HALO = 6` rows of context above and below its body rows.
    /// For a body at `[body_offset_y, body_offset_y + body_h)`:
    ///   `top_global = max(0, body_offset_y - HALO)`
    ///   `bot_global = min(bh, body_offset_y + body_h + HALO)`
    ///   `strip_window_h = bot_global - top_global`
    ///
    /// The first three chain stages (min_abs / pu_blur_h /
    /// pu_blur_v_strip) dispatch over the WHOLE halo-padded window —
    /// they all use offset handles starting at `top_global * bw * 4`.
    /// V-blur's strip-aware kernel computes
    /// `y_global = y_strip + body_off_kernel` and uses `body_off_kernel
    /// = top_global` so reflections against `logical_h = bh` resolve
    /// to absolute rows; `s[i] = reflect - top_global` then indexes
    /// into the offset src buffer correctly because that buffer's
    /// row 0 IS global row top_global.
    ///
    /// Halo rows of m_raw / m_mid / m_blur receive garbage values
    /// (V-blur reflection-vs-buffer-edge math doesn't reach valid
    /// data) but those rows are NOT read by anything downstream:
    /// mult_mutual processes ONLY body rows (offset handles by
    /// `body_offset_y * bw * 4`, n = body_h * bw), so the d band
    /// buffer only ever sees correct values written to body rows.
    ///
    /// Sequential strip dispatch order is critical: GPU stream FIFO
    /// guarantees strip k's chain completes before strip k+1's
    /// starts, so strip overlap in m_raw / m_mid / m_blur is safe
    /// (each strip's chain is self-contained in its window). d band
    /// rows are partitioned non-overlappingly across strips by body
    /// offsets so no cross-strip race exists.
    ///
    /// `strip_dispatch_counter` increments by 4 per strip (one per
    /// kernel launch in the chain).
    #[allow(clippy::too_many_arguments)]
    fn _run_band_masking_strip_walker(
        &self,
        k: usize,
        bw: usize,
        bh: usize,
        n_px: usize,
        pu_scale: f32,
        t_p_ref_h: &[cubecl::server::Handle; 3],
        t_p_dis_h: &[cubecl::server::Handle; 3],
        m_raw_h: &[cubecl::server::Handle; 3],
        m_mid_h: &[cubecl::server::Handle; 3],
        m_blur_h: &[cubecl::server::Handle; 3],
        d_h: &[cubecl::server::Handle; 3],
    ) {
        // P2.1a refactor (2026-05-27): the body of this walker is now
        // the per-strip helper `_run_band_masking_strip_s_for_level`.
        // The wrapper iterates `s in 0..n_strips` and dispatches each
        // strip individually. JOD bit-identical refactor; positions
        // the strip-major outer dispatch (P2.1b/c) to call the per-
        // strip helper. See docs/STRIP_PROCESSING.md#phase-2--p21-implementation-analysis-2026-05-27.
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body as usize,
            None => return,
        };

        // Per-band strip body height: scale-0 body halved per level,
        // clamped to 1. Matches the gauss + Weber strip walkers.
        let strip_h_at_k = (h_body_at_0 >> k).max(1);
        let n_strips = if bh <= strip_h_at_k {
            1
        } else {
            bh.div_ceil(strip_h_at_k)
        };

        for s in 0..n_strips {
            self._run_band_masking_strip_s_for_level(
                s,
                k,
                bw,
                bh,
                pu_scale,
                t_p_ref_h,
                t_p_dis_h,
                m_raw_h,
                m_mid_h,
                m_blur_h,
                d_h,
                false, // legacy caller: full-image transients
            );
        }

        // `n_px` is the level's full-band element count — the wrapper
        // doesn't use it directly. The single-strip helper computes
        // its own per-strip body/window counts from `(s, k, bw, bh)`.
        let _ = n_px;
    }

    /// P2.1a per-strip helper (2026-05-27). Runs stages 1-5 of the
    /// fused masking chain
    /// (`min_abs → pu_blur_h → pu_blur_v → mult_mutual` + optional
    /// Mode B inline pool) for a SINGLE strip `s` at non-baseband
    /// level `k`.
    ///
    /// Stages (mirrors the previous in-loop body of
    /// `_run_band_masking_strip_walker`):
    /// 1. min_abs over halo-padded window of t_p_*.
    /// 2. pu_blur_h_3ch_strip_aware over halo-padded window of m_raw.
    /// 3. pu_blur_v_3ch_scaled_strip_aware over halo-padded window of m_mid.
    /// 4. mult_mutual_3ch_with_blurred over body rows.
    /// 5. (Mode B only) pool_band_3ch_offset_kernel over body rows.
    ///
    /// **JOD invariant.** The helper is the inner-loop body extracted
    /// verbatim; consecutive calls with `s = 0, 1, 2, ...` produce
    /// bit-identical state to the wrapper's `for s in 0..n_strips`
    /// loop.
    ///
    /// **Cross-strip halo dependency** (unchanged from the prior in-
    /// loop body): the V-blur halo reads m_mid rows above and below
    /// this strip's body, which were populated by H-blur in the SAME
    /// strip call (the H-blur was dispatched over the halo-padded
    /// window). m_raw is similarly populated by min_abs in this
    /// strip's stage 1 over the halo-padded window. So all per-strip
    /// reads are self-contained.
    ///
    /// **Caller invariant** for strip-major-outer dispatch: t_p_*[k]
    /// must already be populated for the body+halo rows of this
    /// strip. Today's level-major-outer caller (`_run_d_bands_band_loop`)
    /// satisfies this by running the csf walker for ALL strips at
    /// level k BEFORE any masking strip runs. A future strip-major-
    /// outer caller must extend csf to write t_p_*'s halo rows per
    /// strip (P2.1b work) — until then, calling this helper in
    /// strip-major-outer order silently mixes valid + stale t_p data
    /// in the V-blur halo reads, drifting JOD.
    #[allow(clippy::too_many_arguments)]
    fn _run_band_masking_strip_s_for_level(
        &self,
        s: usize,
        k: usize,
        bw: usize,
        bh: usize,
        pu_scale: f32,
        t_p_ref_h: &[cubecl::server::Handle; 3],
        t_p_dis_h: &[cubecl::server::Handle; 3],
        m_raw_h: &[cubecl::server::Handle; 3],
        m_mid_h: &[cubecl::server::Handle; 3],
        m_blur_h: &[cubecl::server::Handle; 3],
        d_h: &[cubecl::server::Handle; 3],
        // P2.4 (2026-05-27): when `true`, the `t_p_*` and `m_*`
        // handles are strip-local (size `bw × R_k`, row 0 = top_global).
        // The body+halo offsets are 0 (no `offset_start` slice). The
        // strip-aware kernels still receive `body_off_kernel =
        // top_global` so y_global computation works correctly; the
        // buffer-relative index `reflect - top_global` is the same
        // whether the buffer is full-image-sliced or strip-local.
        // When `false` (legacy), handles are full-image and are
        // sliced at `top_global * bw * 4`.
        transients_strip_local: bool,
    ) {
        // PU blur radius — `pu_blur_v_3ch_scaled_strip_aware_kernel`
        // is a 13-tap (half = 6) kernel.
        const HALO: usize = 6;
        let cube_dim = CubeDim::new_1d(64);
        let h_body_at_0 = match self.strip_config {
            Some(StripConfig { h_body, .. }) => h_body as usize,
            None => return,
        };

        let strip_h_at_k = (h_body_at_0 >> k).max(1);

        let body_offset_y = s * strip_h_at_k;
        let body_h = (bh - body_offset_y).min(strip_h_at_k);

        // Halo-padded window for the H/V blur chain.
        let top_global = body_offset_y.saturating_sub(HALO);
        let bot_global = (body_offset_y + body_h + HALO).min(bh);
        let strip_window_h = bot_global - top_global;
        let n_strip_window = bw * strip_window_h;
        let byte_off_window: u64 = (top_global as u64) * (bw as u64) * 4;
        let count_window =
            CubeCount::Static((n_strip_window as u32).div_ceil(64), 1, 1);

        // mult_mutual processes ONLY body rows (no halo).
        let n_strip_body = bw * body_h;
        let byte_off_body: u64 = (body_offset_y as u64) * (bw as u64) * 4;
        let count_body =
            CubeCount::Static((n_strip_body as u32).div_ceil(64), 1, 1);

        // P2.4 (2026-05-27): when transients are strip-local
        // (`bw × R_k`), the buffer's row 0 IS top_global at this
        // dispatch. So:
        //   - Window slices (Stage 1-3) use offset 0 instead of
        //     top_global * bw * 4.
        //   - Body slices (Stage 4) point at body's offset within the
        //     strip buffer: `(body_offset_y - top_global) * bw * 4 =
        //     HALO * bw * 4` for interior strips, or 0 for strip 0
        //     where top_global == body_offset_y == 0.
        // The strip-aware kernels still receive `body_off_kernel =
        // top_global` so reflection math against `logical_h = bh` is
        // unchanged — only the buffer-relative index translates.
        let (tp_m_off_window, tp_m_off_body) = if transients_strip_local {
            let body_off_in_strip: u64 =
                (body_offset_y.saturating_sub(top_global) as u64) * (bw as u64) * 4;
            (0_u64, body_off_in_strip)
        } else {
            (byte_off_window, byte_off_body)
        };

        // Stage 1: min_abs over halo-padded window.
        // Per-pixel, no reflection — offset handles + n = window size.
        let t_p_dis_a_w = t_p_dis_h[0].clone().offset_start(tp_m_off_window);
        let t_p_dis_rg_w = t_p_dis_h[1].clone().offset_start(tp_m_off_window);
        let t_p_dis_vy_w = t_p_dis_h[2].clone().offset_start(tp_m_off_window);
        let t_p_ref_a_w = t_p_ref_h[0].clone().offset_start(tp_m_off_window);
        let t_p_ref_rg_w = t_p_ref_h[1].clone().offset_start(tp_m_off_window);
        let t_p_ref_vy_w = t_p_ref_h[2].clone().offset_start(tp_m_off_window);
        let m_raw_a_w = m_raw_h[0].clone().offset_start(tp_m_off_window);
        let m_raw_rg_w = m_raw_h[1].clone().offset_start(tp_m_off_window);
        let m_raw_vy_w = m_raw_h[2].clone().offset_start(tp_m_off_window);
        unsafe {
            min_abs_3ch_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(t_p_dis_a_w, n_strip_window),
                ArrayArg::from_raw_parts(t_p_dis_rg_w, n_strip_window),
                ArrayArg::from_raw_parts(t_p_dis_vy_w, n_strip_window),
                ArrayArg::from_raw_parts(t_p_ref_a_w, n_strip_window),
                ArrayArg::from_raw_parts(t_p_ref_rg_w, n_strip_window),
                ArrayArg::from_raw_parts(t_p_ref_vy_w, n_strip_window),
                ArrayArg::from_raw_parts(m_raw_a_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_raw_rg_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_raw_vy_w.clone(), n_strip_window),
                n_strip_window as u32,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 2: pu_blur_h_3ch_strip_aware over halo-padded window.
        // H-blur is X-only; body_offset_y / logical_h are passed
        // for API uniformity but ignored by the kernel.
        let m_mid_a_w = m_mid_h[0].clone().offset_start(tp_m_off_window);
        let m_mid_rg_w = m_mid_h[1].clone().offset_start(tp_m_off_window);
        let m_mid_vy_w = m_mid_h[2].clone().offset_start(tp_m_off_window);
        unsafe {
            pu_blur_h_3ch_strip_aware_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(m_raw_a_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_raw_rg_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_raw_vy_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_mid_a_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_mid_rg_w.clone(), n_strip_window),
                ArrayArg::from_raw_parts(m_mid_vy_w.clone(), n_strip_window),
                bw as u32,
                strip_window_h as u32,
                top_global as u32, // body_offset_y (unused by H-pass)
                bh as u32,         // logical_h (unused by H-pass)
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 3: pu_blur_v_3ch_scaled_strip_aware over halo-padded
        // window. The strip-aware V kernel resolves reflection
        // against `logical_h = bh` and reads src via
        // `s[i] = reflect(y_global, bh) - body_off_kernel`. We pass
        // body_off_kernel = top_global so that:
        //   - y_global = y_strip + top_global  (correct absolute
        //     row of every dispatched output element)
        //   - s[i] = reflect - top_global      (correct row index
        //     into the offset src buffer whose row 0 is global
        //     row top_global)
        //
        // Reflections from interior body rows stay within the
        // strip window. Reflections from halo rows can resolve to
        // src rows outside [0, strip_window_h) which underflow
        // usize and write garbage; this is fine because:
        //   - Halo rows of dst (m_blur) are never read downstream
        //     — mult_mutual only reads body rows.
        //   - Halo rows belong to neighbouring strips' bodies, and
        //     each neighbouring strip recomputes its body
        //     correctly on its own dispatch (sequential GPU stream
        //     ordering preserves this).
        let m_blur_a_w = m_blur_h[0].clone().offset_start(tp_m_off_window);
        let m_blur_rg_w = m_blur_h[1].clone().offset_start(tp_m_off_window);
        let m_blur_vy_w = m_blur_h[2].clone().offset_start(tp_m_off_window);
        unsafe {
            pu_blur_v_3ch_scaled_strip_aware_kernel::launch::<R>(
                &self.client,
                count_window.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(m_mid_a_w, n_strip_window),
                ArrayArg::from_raw_parts(m_mid_rg_w, n_strip_window),
                ArrayArg::from_raw_parts(m_mid_vy_w, n_strip_window),
                ArrayArg::from_raw_parts(m_blur_a_w, n_strip_window),
                ArrayArg::from_raw_parts(m_blur_rg_w, n_strip_window),
                ArrayArg::from_raw_parts(m_blur_vy_w, n_strip_window),
                pu_scale,
                bw as u32,
                strip_window_h as u32,
                top_global as u32,
                bh as u32,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 4: mult_mutual_3ch_with_blurred — process body
        // rows only. Reads body rows of m_blur (correct from V-blur)
        // + body rows of t_p/r_p (correct from CSF full-band
        // dispatch) and writes body rows of d.
        //
        // Path A Phase 1d (2026-05-26): in Mode B the `d_h`
        // buffer passed in is `d_strip` (per-strip-sized, allocated
        // by build_d_bands_scratch for non-baseband levels). Each
        // strip dispatch writes into position 0 of d_strip — the
        // inline pool dispatch below then accumulates that strip's
        // contribution into `partials_h` before the next strip
        // iteration overwrites the buffer. Mode E keeps the full-
        // image d plane and writes at `byte_off_body`.
        let mode_b = matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        );
        let d_byte_off: u64 = if mode_b { 0 } else { byte_off_body };
        let t_p_dis_a_b = t_p_dis_h[0].clone().offset_start(tp_m_off_body);
        let t_p_dis_rg_b = t_p_dis_h[1].clone().offset_start(tp_m_off_body);
        let t_p_dis_vy_b = t_p_dis_h[2].clone().offset_start(tp_m_off_body);
        let t_p_ref_a_b = t_p_ref_h[0].clone().offset_start(tp_m_off_body);
        let t_p_ref_rg_b = t_p_ref_h[1].clone().offset_start(tp_m_off_body);
        let t_p_ref_vy_b = t_p_ref_h[2].clone().offset_start(tp_m_off_body);
        let m_blur_a_b = m_blur_h[0].clone().offset_start(tp_m_off_body);
        let m_blur_rg_b = m_blur_h[1].clone().offset_start(tp_m_off_body);
        let m_blur_vy_b = m_blur_h[2].clone().offset_start(tp_m_off_body);
        let d_a_b = d_h[0].clone().offset_start(d_byte_off);
        let d_rg_b = d_h[1].clone().offset_start(d_byte_off);
        let d_vy_b = d_h[2].clone().offset_start(d_byte_off);
        unsafe {
            mult_mutual_3ch_with_blurred_kernel::launch::<R>(
                &self.client,
                count_body.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(t_p_dis_a_b, n_strip_body),
                ArrayArg::from_raw_parts(t_p_dis_rg_b, n_strip_body),
                ArrayArg::from_raw_parts(t_p_dis_vy_b, n_strip_body),
                ArrayArg::from_raw_parts(t_p_ref_a_b, n_strip_body),
                ArrayArg::from_raw_parts(t_p_ref_rg_b, n_strip_body),
                ArrayArg::from_raw_parts(t_p_ref_vy_b, n_strip_body),
                ArrayArg::from_raw_parts(m_blur_a_b, n_strip_body),
                ArrayArg::from_raw_parts(m_blur_rg_b, n_strip_body),
                ArrayArg::from_raw_parts(m_blur_vy_b, n_strip_body),
                ArrayArg::from_raw_parts(d_a_b.clone(), n_strip_body),
                ArrayArg::from_raw_parts(d_rg_b.clone(), n_strip_body),
                ArrayArg::from_raw_parts(d_vy_b.clone(), n_strip_body),
                n_strip_body as u32,
            );
        }
        self.strip_dispatch_counter
            .fetch_add(1, core::sync::atomic::Ordering::Relaxed);

        // Stage 5 (Mode B only): inline pool dispatch over this
        // strip's d (which lives in `d_strip` at offset 0). The
        // pool kernel atomic-adds into `self.partials_h[k *
        // N_CHANNELS + c]`; subsequent strips of this band (and
        // every other band) accumulate into the same partials
        // entries, so the final `partials_h` matches the
        // post-band-loop pool result by atomic associativity.
        //
        // Mode E keeps the full-d post-band-loop pool path
        // (`_pool_and_finalize_jod_strip`); only Mode B interleaves
        // because only Mode B owns a per-strip-sized d_strip
        // buffer that the next strip iteration is about to
        // overwrite.
        if mode_b {
            let partial_idx_a = (k * N_CHANNELS) as u32;
            let partial_idx_rg = (k * N_CHANNELS + 1) as u32;
            let partial_idx_vy = (k * N_CHANNELS + 2) as u32;
            unsafe {
                pool_band_3ch_offset_kernel::launch::<R>(
                    &self.client,
                    count_body.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(d_h[0].clone(), n_strip_body),
                    ArrayArg::from_raw_parts(d_h[1].clone(), n_strip_body),
                    ArrayArg::from_raw_parts(d_h[2].clone(), n_strip_body),
                    ArrayArg::from_raw_parts(
                        self.partials_h.clone(),
                        (self.n_levels as usize) * N_CHANNELS,
                    ),
                    BETA_SPATIAL,
                    partial_idx_a,
                    partial_idx_rg,
                    partial_idx_vy,
                    0_u32,
                    n_strip_body as u32,
                    n_strip_body as u32,
                );
            }
            self.strip_dispatch_counter
                .fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Host-side readback wrapper around the GPU D-bands dispatch.
    /// Runs the full GPU dispatch (color → weber → CSF → masking)
    /// into `self.d_scratch[k].d[c]` then copies each band's D plane
    /// out into a `Vec<[Vec<f32>; 3]>`. Use this when you need the
    /// raw band values (parity checks, debugging, downstream host
    /// scalar processing); use [`Cvvdp::compute_dkl_jod`] directly
    /// when you want the JOD scalar — that path pools on GPU and
    /// avoids the full ~432 MB per-band readback at 12 MP.
    ///
    /// `ppd` is silently ignored — see [`Cvvdp::compute_dkl_jod`].
    /// Pass it consistent with the construction-time geometry; debug
    /// builds verify the match (tick 243).
    ///
    /// # Examples
    ///
    /// Read back the per-band masked-difference planes for a 64×64
    /// (ref, dist) pair. `ignore` for the standard `Cvvdp::*` reason.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let ref_bytes = vec![128u8; (w * h * 3) as usize];
    /// let dist_bytes: Vec<u8> = ref_bytes.iter().map(|b| b.saturating_add(8)).collect();
    /// let bands = cvvdp.compute_dkl_d_bands(&ref_bytes, &dist_bytes, ppd)
    ///     .expect("compute_dkl_d_bands");
    /// assert!(!bands.is_empty());
    /// // bands[0] is finest level, base resolution per channel.
    /// assert_eq!(bands[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either input buffer's
    /// length doesn't match `width × height × 3`, or
    /// [`Error::InvalidImageSize`] if a GPU readback / kernel
    /// dispatch fails anywhere in the color → weber → CSF →
    /// masking chain.
    pub fn compute_dkl_d_bands(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        self.debug_assert_ppd_matches_geometry(ppd);
        // Path A Phase 1d (2026-05-26): Mode B's non-baseband d
        // buffers are per-strip-sized and the strip walker
        // overwrites them across strip iterations, so a single
        // post-band-loop readback can't reconstruct the full-band
        // planes. compute_dkl_d_bands is a parity/debug helper that
        // production never calls; reject the mode loudly rather than
        // returning truncated buffers.
        if matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        ) {
            return Err(Error::InvalidImageSize);
        }
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb)?;

        let n_levels = self.n_levels as usize;
        let mut d_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            // Empty Vecs — `read_one` + `to_vec` allocates a fresh
            // host buffer per channel; the previous `vec![0.0; n_px]`
            // alloc + zero-fill was discarded by `planes[c] =
            // f32::from_bytes(&bytes).to_vec()` on the next line.
            // Matches `compute_dkl_gauss_pyramid`'s readback shape.
            let mut planes: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            let d_full = self.d_scratch[k].d.as_ref().expect(
                "DBandsScratch.d must be Some in compute_dkl_d_bands (Mode B was rejected above)",
            );
            for c in 0..N_CHANNELS {
                let bytes = self
                    .client
                    .read_one(d_full[c].clone())
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            d_bands.push(planes);
        }
        Ok(d_bands)
    }

    /// Final JOD for a (reference, distorted) sRGB pair, computed
    /// through the full GPU composition:
    ///
    /// ```text
    /// sRGB → DKL (GPU)
    ///      → Weber pyramid (GPU, fused subtract+weber 3ch per level)
    ///      → per-pixel CSF apply (GPU, fused REF+DIST 6ch per level)
    ///      → mult-mutual masking (GPU, fused min_abs + pu_blur 3ch +
    ///        mult_mutual_3ch_with_blurred per level — baseband uses
    ///        diff_abs_3ch)
    ///      → spatial pool (GPU, pool_band_3ch_kernel — one fused
    ///        3-channel launch per band, atomic-f32 accumulation
    ///        into a partials Vec)
    ///      → 3-stage Minkowski fold + met2jod (host scalar — operates
    ///        on the `n_levels × N_CHANNELS` partials Vec, ~144 bytes
    ///        total, sub-microsecond regardless of image size).
    /// ```
    ///
    /// Only the GPU→host readback of the partials Vec touches host
    /// memory in proportion to anything other than the pyramid depth
    /// — and that readback is tiny (≤ 36 floats for typical 4K
    /// imagery). The full per-band D Vec readback was removed in
    /// tick 96; callers that still want the host-side band Vecs use
    /// [`Cvvdp::compute_dkl_d_bands`] directly.
    ///
    /// Returns JOD on cvvdp's 0–10 scale (10 = imperceptible).
    ///
    /// `ppd` is silently ignored — the GPU CSF LUT was pre-uploaded
    /// at construction time against `self.geometry.pixels_per_degree()`.
    /// Pass it consistent with the construction-time geometry; debug
    /// builds verify the match via `debug_assert_ppd_matches_geometry`
    /// (tick 243). Use `Cvvdp::new_with_geometry` if you need a
    /// different display geometry.
    ///
    /// `Cvvdp::score` routes through this method as of tick 213
    /// (post-tick-207's tightened 0.005 JOD manifest-parity
    /// tolerance). Parity tests:
    /// - `compute_dkl_jod_matches_host_scalar` (GPU vs the
    ///   all-host reference, f32 precision)
    /// - `shadow_jod_gpu_runs_and_is_close_to_manifest_on_corpus`
    ///   (GPU vs pycvvdp v1 R2 manifest, ≤ 0.005 JOD)
    /// - `compute_dkl_jod_host_pool_matches_compute_dkl_jod` (GPU
    ///   atomic pool vs host pool, 0.000000 diff)
    ///
    /// # Backend support
    ///
    /// This method dispatches `pool_band_3ch_kernel`, which uses
    /// `Atomic<f32>::fetch_add`. The two known traps:
    ///
    /// - **`cubecl-cpu` (0.10.x): the kernel panics at launch** with
    ///   "not yet implemented: This type is not implemented yet.
    ///   `atomic<f32>`". The panic is NOT surfaced as
    ///   [`Error::InvalidImageSize`] — it unwinds through the
    ///   caller. Use [`Cvvdp::compute_dkl_jod_host_pool`] on
    ///   `cubecl-cpu` instead; it reads D bands back and folds
    ///   on host (same JOD output at f32 precision).
    /// - **Metal (via `cubecl-wgpu`): the kernel succeeds but
    ///   silently no-ops** on the `Atomic<f32>::fetch_add`,
    ///   producing all-zero partials and thus JOD = 10 (the
    ///   identity-pair value) regardless of input. Use
    ///   [`Cvvdp::compute_dkl_jod_host_pool`] there too.
    ///
    /// CUDA, Vulkan, DX12, and HIP backends support
    /// `Atomic<f32>::fetch_add` correctly and produce the
    /// canonical JOD.
    ///
    /// # Examples
    ///
    /// Canonical one-shot scoring at 64×64. `ignore` because docs.rs
    /// has no GPU and the no-default-features build path doesn't
    /// resolve cubecl runtime types (same as the rest of the
    /// `Cvvdp::*` examples).
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let ref_bytes = vec![128u8; (w * h * 3) as usize];
    /// let dist_bytes: Vec<u8> = ref_bytes.iter().map(|b| b.saturating_add(8)).collect();
    /// let jod: f32 = cvvdp.compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
    ///     .expect("compute_dkl_jod");
    /// assert!(jod.is_finite() && (0.0..=10.0 + 1e-3).contains(&jod));
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either input buffer's
    /// length doesn't match `width × height × 3`, or
    /// [`Error::InvalidImageSize`] if a GPU readback / kernel
    /// dispatch fails anywhere in the color → weber → CSF →
    /// masking → pool chain. **Note:** the `cubecl-cpu` panic
    /// described in the Backend support section above is NOT
    /// surfaced via this error path; it unwinds.
    pub fn compute_dkl_jod(&mut self, ref_srgb: &[u8], dist_srgb: &[u8], ppd: f32) -> Result<f32> {
        self.debug_assert_ppd_matches_geometry(ppd);

        // Run the full D-bands GPU dispatch (color → weber → CSF →
        // masking). `_dispatch_d_bands_into_scratch` leaves the
        // per-band D planes resident in `self.d_scratch[k].d[c]` for
        // every level (baseband included since tick 94's
        // diff_abs_3ch_kernel) and does no host read-back.
        //
        // The JOD path then pools via `pool_band_3ch_kernel` on each
        // resident D handle, accumulating into an `n_levels ×
        // N_CHANNELS` partials buffer that's the only data read back
        // to host. `compute_dkl_d_bands` (the parity-test helper)
        // adds a per-band readback on top, paying ~432 MB of
        // GPU→host transfer at 12 MP — JOD skips that.
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb)?;
        // Mode B (StripPair) and Mode E (CachedRef via warm_ref) route
        // the pool stage through the strip-aware walker that
        // partitions each band's per-pixel pool into row-strips and
        // dispatches `pool_band_3ch_offset_kernel` per slab. Atomic
        // adds are associative across slabs, so JOD is bit-exact
        // against `_pool_and_finalize_jod`. The strip dispatch
        // counter increments by one per (level, strip) so tests can
        // verify the walker actually partitioned.
        if self.strip_config.is_some() {
            self._pool_and_finalize_jod_strip()
        } else {
            self._pool_and_finalize_jod()
        }
    }

    /// Pack the caller's `width × height × 3` sRGB-u8 bytes into a
    /// `width × height` packed-u32 device handle (`R | G<<8 | B<<16`),
    /// using the same pinned-staging fast path the internal upload
    /// uses. Cheaper than [`Self::score`] / [`Self::compute_dkl_jod`]
    /// when scoring the same pair through multiple metrics — pack
    /// once via [`Self::pack_srgb_into_packed_u32_handle`] on any one
    /// metric's client, then thread the handle through
    /// [`Self::compute_handles`] on every metric that shares the same
    /// client.
    ///
    /// Returns `Err(DimensionMismatch)` if `srgb.len() != width *
    /// height * 3`.
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        let n0 = (self.width as usize) * (self.height as usize);
        let pinned_len = n0 * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(srgb.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        Ok(self.client.create(bytes))
    }

    /// Handle-flavored sibling of [`Self::compute_dkl_jod`] —
    /// upload-once Phase 4 entry point. Skips the
    /// `client.reserve_staging` + byte-pack work that
    /// [`Self::compute_dkl_jod`] does internally, letting one
    /// `(ref, dist)` upload feed several metrics on the same client.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this `Cvvdp<R>`; sharing handles across clients
    /// is undefined behaviour at the cubecl layer and is not
    /// validated here.
    pub fn compute_dkl_jod_from_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dist_handle: &cubecl::server::Handle,
        ppd: f32,
    ) -> Result<f32> {
        self.debug_assert_ppd_matches_geometry(ppd);
        self._dispatch_d_bands_into_scratch_from_handles(ref_handle, dist_handle)?;
        self._pool_and_finalize_jod()
    }

    /// Score from caller-supplied packed-u32 device handles — the
    /// upload-once Phase 4 entry point matching the layout produced
    /// by [`Self::pack_srgb_into_packed_u32_handle`]. Equivalent to
    /// [`Self::score`] but skips the host-to-device upload (use it
    /// when one packed-pair feeds several metrics on the same
    /// client). Returns the JOD score as `f64` to match
    /// [`Self::score`].
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<f64> {
        let ppd = self.geometry.pixels_per_degree();
        let jod = self.compute_dkl_jod_from_handles(ref_handle, dis_handle, ppd)?;
        Ok(f64::from(jod))
    }

    /// Portable-backend variant of [`Cvvdp::compute_dkl_jod`] that
    /// avoids the GPU `Atomic<f32>::fetch_add` trap.
    ///
    /// Same JOD result, but uses a host-side spatial pool instead
    /// of `pool_band_3ch_kernel`. That GPU kernel uses
    /// `Atomic<f32>::fetch_add`, which `cubecl-cpu` panics on and
    /// Metal silently no-ops; this variant reads D bands back via
    /// [`Cvvdp::compute_dkl_d_bands`] and pools them with the
    /// host-scalar `lp_norm_mean`, so it runs on every cubecl
    /// runtime — including `cubecl-cpu` and Metal-via-`cubecl-wgpu`.
    ///
    /// Tradeoff: the readback is `O(n_pixels × n_channels × n_levels
    /// × 4/3)` bytes (geometric series on band sizes). At 12 MP that's
    /// ≈ 432 MB GPU→host transfer per call, swamping the GPU pool's
    /// few-microsecond kernel time. **Use this on `cubecl-cpu` and
    /// Metal**; for CUDA / Vulkan / DX12 / HIP runtimes prefer
    /// [`Cvvdp::compute_dkl_jod`], which keeps everything GPU-
    /// resident and produces canonical JOD via the working
    /// atomic-reduction path. See the "Backend support" section
    /// on [`Cvvdp::compute_dkl_jod`] for the full atomic-f32 story.
    ///
    /// Output matches `compute_dkl_jod` to f32 noise on all backends
    /// where both run (the GPU pool's atomic reduction and the host
    /// `lp_norm_mean` compute the same `safe_pow`-form Minkowski norm).
    ///
    /// `ppd` is silently ignored — see [`Cvvdp::compute_dkl_jod`].
    /// Pass it consistent with the construction-time geometry; debug
    /// builds verify the match (tick 243).
    ///
    /// # Examples
    ///
    /// CPU-runtime scoring of a byte-identical 64×64 pair (max JOD = 10).
    /// Doctest body is gated on `feature = "cpu"` so non-cpu builds
    /// (e.g. CI's `--no-default-features --features wgpu` doctest pass)
    /// skip the body while still type-checking the example shape.
    ///
    /// ```
    /// # #[cfg(feature = "cpu")] {
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<cubecl::cpu::CpuRuntime>::new(
    ///     client, w, h, CvvdpParams::PLACEHOLDER,
    /// ).expect("Cvvdp::new");
    /// let bytes = vec![128u8; (w * h * 3) as usize];
    /// let jod = cvvdp.compute_dkl_jod_host_pool(&bytes, &bytes, ppd)
    ///     .expect("compute_dkl_jod_host_pool");
    /// assert!((jod - 10.0).abs() < 1e-3, "expected JOD ≈ 10, got {jod}");
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either input buffer's
    /// length doesn't match `width × height × 3`, or
    /// [`Error::InvalidImageSize`] if a GPU readback / dispatch
    /// fails inside [`Cvvdp::compute_dkl_d_bands`] (the GPU stages
    /// up to the per-band readback).
    pub fn compute_dkl_jod_host_pool(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<f32> {
        self.debug_assert_ppd_matches_geometry(ppd);
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb)?;
        self._host_pool_and_finalize_jod()
    }

    /// Warm-reference companion to [`Cvvdp::compute_dkl_jod_host_pool`].
    ///
    /// Same algorithm and same JOD output as
    /// [`Cvvdp::compute_dkl_jod_with_warm_ref`] but pools the per-band
    /// D values on the host instead of via the GPU atomic kernel —
    /// runs on every cubecl runtime, including `cubecl-cpu` and
    /// Metal-via-`cubecl-wgpu`. Useful for batch CPU/Metal scoring
    /// (one warm REF, many DIST candidates) where the GPU pool path
    /// panics or silently no-ops. See the "Backend support" section
    /// on [`Cvvdp::compute_dkl_jod`] for the underlying
    /// `Atomic<f32>::fetch_add` trap.
    ///
    /// Same `Error::NoWarmReference` semantics as the GPU warm-ref
    /// variant: requires a prior [`Cvvdp::warm_reference`] call, and
    /// any intervening REF-dispatching method invalidates the warm
    /// state.
    ///
    /// `ppd` is silently ignored — see [`Cvvdp::compute_dkl_jod`].
    /// Pass it consistent with the construction-time geometry; debug
    /// builds verify the match (tick 243).
    ///
    /// # Examples
    ///
    /// Score N distorted candidates against one warm REF on the cpu
    /// runtime — common batch-CPU pattern for sweep workers:
    ///
    /// ```
    /// # #[cfg(feature = "cpu")] {
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// let client = cubecl::cpu::CpuRuntime::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<cubecl::cpu::CpuRuntime>::new(
    ///     client, w, h, CvvdpParams::PLACEHOLDER,
    /// ).expect("Cvvdp::new");
    ///
    /// let ref_bytes = vec![128u8; (w * h * 3) as usize];
    /// cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    ///
    /// // Score each candidate; REF weber runs once via warm_reference
    /// // above instead of being re-dispatched per call.
    /// for shift in [0u8, 8, 16] {
    ///     let dist_bytes: Vec<u8> = ref_bytes
    ///         .iter()
    ///         .map(|b| b.saturating_add(shift))
    ///         .collect();
    ///     let jod = cvvdp
    ///         .compute_dkl_jod_host_pool_with_warm_ref(&dist_bytes, ppd)
    ///         .expect("warm host_pool");
    ///     assert!((0.0..=10.0).contains(&jod));
    /// }
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`Error::DimensionMismatch`] if `dist_srgb.len() !=
    ///   width × height × 3` (checked first, per the tick-248
    ///   precedence rule shared with
    ///   [`Cvvdp::compute_dkl_jod_with_warm_ref`]).
    /// - [`Error::NoWarmReference`] if `warm_reference` wasn't
    ///   called or the warm state was invalidated by an
    ///   intervening REF-dispatching method (see
    ///   [`Cvvdp::warm_reference`] for the documented set).
    /// - [`Error::InvalidImageSize`] if a GPU readback / dispatch
    ///   in the DIST chain fails before the per-band readback.
    pub fn compute_dkl_jod_host_pool_with_warm_ref(
        &mut self,
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<f32> {
        self.debug_assert_ppd_matches_geometry(ppd);
        // Tick 248: validate dist length before checking warm state.
        // See compute_dkl_jod_with_warm_ref for the same ordering
        // rationale.
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        // Mode E (task #79): in strip mode, restore ref state from
        // `ref_full_state`. See compute_dkl_jod_with_warm_ref for the
        // full rationale.
        let log_l_bkg_baseband = self._warm_ref_baseband_log_l_bkg_for_dispatch()?;
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband)?;
        self._host_pool_and_finalize_jod()
    }

    /// Host-side spatial pool + 3-stage Minkowski fold over the D
    /// planes resident in `self.d_scratch[k].d[c]`. Used by both
    /// `compute_dkl_jod_host_pool` and `compute_dkl_jod_host_pool_with_warm_ref`
    /// — the dispatch path that landed the D bands differs, but
    /// the pool tail is identical.
    fn _host_pool_and_finalize_jod(&mut self) -> Result<f32> {
        // Path A Phase 1d (2026-05-26): Mode B's d_strip planes are
        // per-strip-sized and overwritten across strip iterations —
        // a post-band-loop readback would only contain the LAST
        // strip's data per band. Reject Mode B here too; production
        // routes Mode B through `compute_dkl_jod` (GPU pool) which
        // sums correctly via the inline pool.
        if matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        ) {
            return Err(Error::InvalidImageSize);
        }
        let n_levels = self.n_levels as usize;
        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            let d_full = self.d_scratch[k].d.as_ref().expect(
                "DBandsScratch.d must be Some in _host_pool_and_finalize_jod (Mode B rejected above)",
            );
            for c in 0..N_CHANNELS {
                let bytes = self
                    .client
                    .read_one(d_full[c].clone())
                    .map_err(|_| Error::InvalidImageSize)?;
                let d_vec: &[f32] = f32::from_bytes(&bytes);
                debug_assert_eq!(d_vec.len(), n_px);
                q[c] = lp_norm_mean(d_vec, BETA_SPATIAL);
            }
            q_per_ch.push(q);
        }
        Ok(do_pooling_and_jod_still_3ch(&q_per_ch))
    }

    /// Pre-dispatch the REF weber pyramid + cache state for batch
    /// scoring. Subsequent calls to
    /// [`Cvvdp::compute_dkl_jod_with_warm_ref`] skip the REF half of
    /// the JOD pipeline — measured ~1.8× per-DIST throughput at
    /// 12 MP on CUDA (cold ~62 ns/px → warm ~34 ns/px, ~45% saved)
    /// per the post-tick-175 numbers in `lib.rs`. Pre-tick-175
    /// timings (36/21 ns/px) reflected a 0.586 JOD drift vs pycvvdp;
    /// the current numbers are slower but bit-stable with the
    /// pycvvdp reference.
    ///
    /// Any call to [`Cvvdp::compute_dkl_jod`],
    /// [`Cvvdp::compute_dkl_jod_host_pool`],
    /// [`Cvvdp::compute_dkl_d_bands`],
    /// [`Cvvdp::compute_dkl_weber_pyramid`],
    /// [`Cvvdp::compute_dkl_t_p_bands`],
    /// [`Cvvdp::compute_dkl_laplacian_pyramid`], or
    /// [`Cvvdp::compute_dkl_csf_weighted_bands`] invalidates the
    /// warm state (their REF dispatches overwrite the shared GPU
    /// scratch — either bands_ref via the weber chain, or bands_ref
    /// via the Laplacian chain; `compute_dkl_jod_host_pool` routes
    /// through the same `_dispatch_d_bands_into_scratch` →
    /// `_dispatch_ref_weber_pyramid_only` path the GPU jod uses,
    /// clearing the cached scalar at the inner-REF call). Call
    /// `warm_reference` again to re-arm.
    ///
    /// [`Cvvdp::score`] and [`Cvvdp::score_with_reference`] also
    /// invalidate — they route through `compute_dkl_jod` since
    /// tick 213. [`Cvvdp::set_reference`] does NOT invalidate
    /// (it only stashes host-side bytes; no GPU dispatch).
    /// [`Cvvdp::compute_dkl_jod_host_pool_with_warm_ref`] only
    /// reads the cached scalar (via `.ok_or(NoWarmReference)`),
    /// never writes it — so it does NOT invalidate either.
    ///
    /// Validates that `ref_srgb.len() == width × height × 3`.
    ///
    /// # Examples
    ///
    /// Warm against a REF once, then score N DIST candidates
    /// against the cached state. `ignore` for the same reason as
    /// the rest of the `Cvvdp::*` doctests — docs.rs has no GPU
    /// and the no-default-features build path doesn't resolve
    /// cubecl runtime types.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// // One REF, one warm dispatch.
    /// let ref_bytes = vec![128u8; (w * h * 3) as usize];
    /// cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    ///
    /// // Score many DIST candidates without re-dispatching REF.
    /// for shift in [0u8, 4, 8, 16] {
    ///     let dist: Vec<u8> = ref_bytes.iter().map(|b| b.saturating_add(shift)).collect();
    ///     let jod = cvvdp.compute_dkl_jod_with_warm_ref(&dist, ppd)
    ///         .expect("warm-ref JOD");
    ///     assert!(jod.is_finite() && (0.0..=10.0 + 1e-3).contains(&jod));
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `ref_srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a GPU
    /// readback / dispatch in the REF weber-pyramid pass fails.
    pub fn warm_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_srgb.len(),
            });
        }
        let log_l_bkg_baseband = self._dispatch_ref_weber_pyramid_only(ref_srgb)?;
        self.warm_ref_baseband_log_l_bkg = Some(log_l_bkg_baseband);

        // Mode E (task #79): when running in strip mode, snapshot the
        // ref-side state from the shared `bands_ref` / `weber_scratch`
        // scratch into the dedicated `ref_full_state` buffers so the
        // cached state survives across any intervening one-shot
        // dispatches (which clobber the shared scratch). Phase 3 will
        // add a strip-walker dispatch that reads directly from
        // `ref_full_state` per strip.
        if self.strip_config.is_some() {
            self._snapshot_ref_state_to_full(log_l_bkg_baseband)?;
        }
        Ok(())
    }

    /// Allocate the dedicated [`RefFullState`] buffers if absent. Lazy
    /// because Full-mode constructions don't need them and the
    /// allocation cost is non-trivial at 4 MP+ (`~3 × N_CHANNELS × W ×
    /// H × 4 bytes ≈ 144 MB` at 12 MP for the pyramid).
    fn _ensure_ref_full_state(&mut self) {
        if self.ref_full_state.is_some() {
            return;
        }
        let n_levels = self.n_levels as usize;
        let mut bands: Vec<[cubecl::server::Handle; N_CHANNELS]> = Vec::with_capacity(n_levels);
        let mut log_l_bkg: Vec<cubecl::server::Handle> = Vec::with_capacity(n_levels - 1);
        let mut w = self.width;
        let mut h = self.height;
        for k in 0..n_levels {
            let n = (w as usize) * (h as usize);
            bands.push([
                alloc_zeros_f32(&self.client, n),
                alloc_zeros_f32(&self.client, n),
                alloc_zeros_f32(&self.client, n),
            ]);
            if k < n_levels - 1 {
                log_l_bkg.push(alloc_zeros_f32(&self.client, n));
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        // Baseband gauss has its own slot: 3 channels × baseband size.
        let last = n_levels - 1;
        let bb_w = self.gauss_ref[last].w as usize;
        let bb_h = self.gauss_ref[last].h as usize;
        let bb_n = bb_w * bb_h;
        let baseband_gauss = [
            alloc_zeros_f32(&self.client, bb_n),
            alloc_zeros_f32(&self.client, bb_n),
            alloc_zeros_f32(&self.client, bb_n),
        ];
        self.ref_full_state = Some(RefFullState {
            bands,
            log_l_bkg,
            baseband_gauss,
            baseband_log_l_bkg_scalar: 0.0,
        });
    }

    /// Snapshot the ref-side state from the shared scratch
    /// (`bands_ref` + `weber_scratch[k].log_l_bkg` + baseband gauss in
    /// `gauss_ref[last]`) into the dedicated [`RefFullState`] buffers.
    /// Called at the end of [`Self::warm_reference`] when running in
    /// strip mode.
    ///
    /// Uses a per-level [`copy_f32_kernel`] launch per channel per
    /// data plane. Per-strip dist scoring later reads from these
    /// dedicated buffers.
    fn _snapshot_ref_state_to_full(&mut self, log_l_bkg_baseband: f32) -> Result<()> {
        self._ensure_ref_full_state();
        let n_levels = self.n_levels as usize;
        let cube_dim = CubeDim::new_1d(64);
        let mut w = self.width;
        let mut h = self.height;
        let state = self
            .ref_full_state
            .as_mut()
            .expect("ensured above");
        for k in 0..n_levels {
            let n = (w as usize) * (h as usize);
            let count = CubeCount::Static((n as u32).div_ceil(64), 1, 1);
            for c in 0..N_CHANNELS {
                let src = self.bands_ref[k].planes[c].clone();
                let dst = state.bands[k][c].clone();
                unsafe {
                    copy_f32_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(src, n),
                        ArrayArg::from_raw_parts(dst, n),
                        n as u32,
                    );
                }
            }
            if k < n_levels - 1 {
                let src = self.weber_scratch[k].log_l_bkg.clone();
                let dst = state.log_l_bkg[k].clone();
                unsafe {
                    copy_f32_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(src, n),
                        ArrayArg::from_raw_parts(dst, n),
                        n as u32,
                    );
                }
            }
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        // Baseband gauss: 3 channels of gauss_ref[last].
        let last = n_levels - 1;
        let bb_n = (self.gauss_ref[last].w as usize) * (self.gauss_ref[last].h as usize);
        let bb_count = CubeCount::Static((bb_n as u32).div_ceil(64), 1, 1);
        for c in 0..N_CHANNELS {
            let src = self.gauss_ref[last].planes[c].clone();
            let dst = state.baseband_gauss[c].clone();
            unsafe {
                copy_f32_kernel::launch::<R>(
                    &self.client,
                    bb_count.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(src, bb_n),
                    ArrayArg::from_raw_parts(dst, bb_n),
                    bb_n as u32,
                );
            }
        }
        state.baseband_log_l_bkg_scalar = log_l_bkg_baseband;
        Ok(())
    }

    // Tick 263 (Mode E walker): `_restore_ref_state_from_full` was
    // removed. The Phase 2 snapshot/restore loop copied REF state from
    // `ref_full_state` back into the shared `bands_ref` /
    // `weber_scratch.log_l_bkg` / `gauss_ref[last]` buffers ahead of
    // every DIST dispatch — a per-call ~3× pyramid + 1× log_l_bkg
    // pyramid copy (~144 MB at 12 MP).
    //
    // The Mode E walker now reads REF bands and non-baseband log_l_bkg
    // straight from `ref_full_state` inside `_run_d_bands_band_loop`,
    // and `_warm_ref_baseband_log_l_bkg_for_dispatch` returns the
    // cached baseband scalar without touching the shared scratch.
    // No restore is needed; the cached state survives intervening
    // one-shot dispatches because it lives in dedicated buffers that
    // are never written by REF weber dispatchers (only `warm_reference`
    // touches them via `_snapshot_ref_state_to_full`).

    /// Score a DIST candidate against the GPU-warmed REF. Same JOD
    /// output as [`Cvvdp::compute_dkl_jod`] but skips the REF weber
    /// pyramid — useful for batch workflows where one reference
    /// is scored against many distorted candidates (codec quality
    /// sweeps, fixture-based testing).
    ///
    /// Returns [`Error::NoWarmReference`] if `warm_reference` was
    /// not called, or if the warm state was invalidated by an
    /// intervening REF-dispatching method.
    ///
    /// `ppd` is silently ignored — see [`Cvvdp::compute_dkl_jod`].
    /// Pass it consistent with the construction-time geometry; debug
    /// builds verify the match (tick 243).
    ///
    /// # Examples
    ///
    /// Score N distorted candidates against one warm REF on the CUDA
    /// backend — canonical batch-GPU pattern for sweep workers.
    /// `no_run` because docs.rs has no GPU; mirrors the cpu-runtime
    /// example on [`Cvvdp::compute_dkl_jod_host_pool_with_warm_ref`]:
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let ref_bytes = vec![128u8; (w * h * 3) as usize];
    /// cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    ///
    /// // REF weber runs once above; each compute_dkl_jod_with_warm_ref
    /// // call skips it. ~1.8× per-DIST throughput at 12 MP vs cold.
    /// for shift in [0u8, 8, 16] {
    ///     let dist_bytes: Vec<u8> = ref_bytes
    ///         .iter()
    ///         .map(|b| b.saturating_add(shift))
    ///         .collect();
    ///     let jod = cvvdp
    ///         .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
    ///         .expect("warm GPU jod");
    ///     assert!((0.0..=10.0).contains(&jod));
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`Error::DimensionMismatch`] if `dist_srgb.len() !=
    ///   width × height × 3` (checked first, ahead of the warm-state
    ///   check, per the tick-248 precedence audit).
    /// - [`Error::NoWarmReference`] if `warm_reference` wasn't called
    ///   first, or the warm state was invalidated by an intervening
    ///   REF-dispatching method (see the documented set on
    ///   [`Cvvdp::warm_reference`]).
    /// - [`Error::InvalidImageSize`] if a GPU readback / dispatch
    ///   in the DIST weber → CSF → masking → pool chain fails.
    ///
    /// # Backend support
    ///
    /// Dispatches `pool_band_3ch_kernel` like
    /// [`Cvvdp::compute_dkl_jod`]; same constraints. `cubecl-cpu`
    /// callers must use
    /// [`Cvvdp::compute_dkl_jod_host_pool_with_warm_ref`] instead;
    /// Metal callers should too. See the "Backend support"
    /// section on [`Cvvdp::compute_dkl_jod`] for the full
    /// `Atomic<f32>::fetch_add` story.
    pub fn compute_dkl_jod_with_warm_ref(&mut self, dist_srgb: &[u8], ppd: f32) -> Result<f32> {
        self.debug_assert_ppd_matches_geometry(ppd);
        // Tick 248: validate dist length before checking warm state.
        // If a caller has both problems, the wrong-size buffer is the
        // more actionable error — they need to fix the buffer regardless
        // of whether warm state is set. Pre-tick-248 ordering reported
        // NoWarmReference first, masking the dim mismatch until they
        // re-armed.
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        // Mode E (task #79): in strip mode the cached REF state lives
        // in the dedicated `ref_full_state` buffers populated by
        // `warm_reference`. The walker reads REF bands and non-baseband
        // log_l_bkg straight from `ref_full_state` inside
        // `_run_d_bands_band_loop` and `_warm_ref_baseband_log_l_bkg_for_dispatch`
        // returns the cached baseband scalar without copying anything
        // back into the shared scratch. The Phase 2 snapshot/restore
        // intermediate was removed in tick 263 — the walker shape now
        // matches Mode B (strip-aware masking + strip-aware pool) with
        // the only per-mode difference being the REF source.
        let log_l_bkg_baseband = self._warm_ref_baseband_log_l_bkg_for_dispatch()?;
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband)?;
        // Mode B (StripPair) and Mode E (CachedRef) both route the
        // per-band pool through the strip-aware walker that partitions
        // each band's per-pixel pool into row-strips and dispatches
        // `pool_band_3ch_offset_kernel` per slab. Atomic-adds across
        // slabs are associative so JOD is bit-identical to Full mode
        // (within the same per-call ordering noise band as Full's own
        // repeated calls). The masking chain in Mode E now also runs
        // through the strip-aware walker (`_run_band_masking_strip_walker`)
        // — same dispatch shape as Mode B with the REF source pulled
        // from `ref_full_state`.
        if self.strip_config.is_some() {
            self._pool_and_finalize_jod_strip()
        } else {
            self._pool_and_finalize_jod()
        }
    }

    /// GPU pool + host fold for the per-band D planes resident in
    /// `self.d_scratch[k].d[c]`. Used by both `compute_dkl_jod` and
    /// `compute_dkl_jod_with_warm_ref` — the dispatch path that
    /// landed the bands differs, but the pool/fold tail is identical.
    fn _pool_and_finalize_jod(&mut self) -> Result<f32> {
        let n_levels = self.n_levels as usize;
        let n_partials = n_levels * N_CHANNELS;
        let cube_dim = CubeDim::new_1d(64);

        // Zero the persistent partials buffer before atomic-add
        // accumulation. One tiny GPU launch replaces the per-call
        // host alloc + create_from_slice upload (tick 227).
        unsafe {
            fill_f32_kernel::launch::<R>(
                &self.client,
                CubeCount::Static((n_partials as u32).div_ceil(64), 1, 1),
                cube_dim,
                ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                0.0,
                n_partials as u32,
            );
        }

        // T1.C + T4.K (2026-05-16): per-size pool dispatch. The
        // LDS-reduction kernel (`pool_band_3ch_lds_kernel`, 256-thread
        // workgroup, pointer-jumping reduce, 1 atomic per workgroup
        // per channel) wins at large bands by cutting atomic traffic
        // ~255×; at tiny bands (≤ ~16 K pixels) its 8-sync overhead
        // exceeds the per-pixel-atomic cost. POOL_LDS_MIN_PIXELS sets
        // the crossover; benched on RTX 5070 (256² regressed under
        // unconditional LDS; 1 MP and 12 MP win cleanly).
        const POOL_LDS_MIN_PIXELS: usize = 16_384;
        let pool_lds_cube_dim = CubeDim::new_1d(POOL_LDS_BLOCK_DIM);
        let pool_atomic_cube_dim = CubeDim::new_1d(64);
        for k in 0..n_levels {
            let (_, _, n_px) = self.level_dims(k);
            // Full / CachedRef modes own a `Some(d)` at every level —
            // this method is never called from Mode B's hot path
            // (`compute_dkl_jod` routes Mode B to `_pool_and_finalize_jod_strip`).
            let d_full = self.d_scratch[k].d.as_ref().expect(
                "DBandsScratch.d must be Some in _pool_and_finalize_jod (Full / CachedRef)",
            );
            let d_a = d_full[0].clone();
            let d_rg = d_full[1].clone();
            let d_vy = d_full[2].clone();
            let partial_idx_a = (k * N_CHANNELS) as u32;
            let partial_idx_rg = (k * N_CHANNELS + 1) as u32;
            let partial_idx_vy = (k * N_CHANNELS + 2) as u32;
            if n_px >= POOL_LDS_MIN_PIXELS {
                let count = CubeCount::Static((n_px as u32).div_ceil(POOL_LDS_BLOCK_DIM), 1, 1);
                unsafe {
                    pool_band_3ch_lds_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        pool_lds_cube_dim,
                        ArrayArg::from_raw_parts(d_a, n_px),
                        ArrayArg::from_raw_parts(d_rg, n_px),
                        ArrayArg::from_raw_parts(d_vy, n_px),
                        ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                        BETA_SPATIAL,
                        partial_idx_a,
                        partial_idx_rg,
                        partial_idx_vy,
                        n_px as u32,
                    );
                }
            } else {
                let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
                unsafe {
                    pool_band_3ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        pool_atomic_cube_dim,
                        ArrayArg::from_raw_parts(d_a, n_px),
                        ArrayArg::from_raw_parts(d_rg, n_px),
                        ArrayArg::from_raw_parts(d_vy, n_px),
                        ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                        BETA_SPATIAL,
                        partial_idx_a,
                        partial_idx_rg,
                        partial_idx_vy,
                        n_px as u32,
                    );
                }
            }
        }

        let bytes = self
            .client
            .read_one(self.partials_h.clone())
            .map_err(|_| Error::InvalidImageSize)?;
        let partials_data: &[f32] = f32::from_bytes(&bytes);

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px_k) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            for c in 0..N_CHANNELS {
                q[c] = pool_band_finalize(partials_data[k * N_CHANNELS + c], n_px_k, BETA_SPATIAL);
            }
            q_per_ch.push(q);
        }

        Ok(do_pooling_and_jod_still_3ch(&q_per_ch))
    }

    /// Mode E Phase 3 strip-aware variant of [`Self::_pool_and_finalize_jod`].
    ///
    /// Partitions each band's per-pixel pool into `n_strips` row-strips
    /// and dispatches [`pool_band_3ch_offset_kernel`] per strip. The
    /// atomic-add into `partials_h` is associative across strips, so
    /// the final `partials_h[k * N_CHANNELS + c]` value equals the
    /// single-shot pool dispatch result to f32 atomic-ordering noise
    /// (which is the same drift band Full mode already produces
    /// across repeated calls — see
    /// `compute_dkl_jod_is_deterministic_across_repeated_calls` in
    /// `tests/pipeline_score.rs`).
    ///
    /// Per-band strip count is computed from `strip_config.h_body` at
    /// the band's resolution: a band of height `bh` is partitioned
    /// into `ceil(bh / strip_body_h)` strips. Deep bands whose height
    /// is shorter than the configured strip body fall through to a
    /// single dispatch (effectively the same as the Full path) —
    /// these bands carry negligible pool work so there's no value in
    /// forcing multi-strip dispatch.
    ///
    /// Increments [`Self::strip_dispatch_counter`] by one per outer
    /// strip-iteration (NOT per kernel launch; one strip iteration
    /// dispatches the kernel once per `n_levels`). Test-only via
    /// `#[doc(hidden)]` accessor.
    ///
    /// Caller contract: only invoke when `self.strip_config.is_some()`
    /// (Mode E). Full-mode callers should keep using the canonical
    /// [`Self::_pool_and_finalize_jod`] — single-dispatch is faster
    /// when partitioning isn't needed.
    fn _pool_and_finalize_jod_strip(&mut self) -> Result<f32> {
        let n_levels = self.n_levels as usize;
        let n_partials = n_levels * N_CHANNELS;
        let cube_dim = CubeDim::new_1d(64);

        // Strip body height at scale 0. Mode E's strip_config is
        // guaranteed Some(_) by the caller; the unwrap-equivalent is
        // safe but we defensively read via expect to surface a clear
        // message if a future caller forgets to gate.
        let cfg = self
            .strip_config
            .as_ref()
            .expect("_pool_and_finalize_jod_strip requires strip_config");
        let strip_h_body = cfg.h_body as usize;
        let mode_b = cfg.mode == StripMode::Pair;

        // Mode E: zero partials_h here (the post-band-loop pool owns
        // the accumulation).
        // Mode B: partials_h was zeroed at the top of
        // `_run_d_bands_band_loop` and the non-baseband pool was
        // dispatched inline by the strip walker / no-blur fallback.
        // Re-zeroing here would wipe the accumulated partials —
        // skip the fill in Mode B.
        if !mode_b {
            unsafe {
                fill_f32_kernel::launch::<R>(
                    &self.client,
                    CubeCount::Static((n_partials as u32).div_ceil(64), 1, 1),
                    cube_dim,
                    ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                    0.0,
                    n_partials as u32,
                );
            }
        }

        let pool_atomic_cube_dim = CubeDim::new_1d(64);

        // Track per-iteration strip count. We dispatch the offset
        // kernel once per (level, strip); the outer loop iterates
        // strips across levels so a single test can see N >= 2 strip
        // iterations from a single warm-ref call when the image is
        // tall enough.
        let mut outer_strip_iters: u32 = 0;

        for k in 0..n_levels {
            let (bw, bh, n_px) = self.level_dims(k);
            let is_baseband = k == n_levels - 1;

            // Mode B's non-baseband bands were already pooled inline
            // by the strip walker / no-blur fallback at band-loop
            // time; skip them here. The baseband (always full `d`
            // in Mode B too) still needs its pool dispatch.
            if mode_b && !is_baseband {
                continue;
            }

            // Both Mode B (baseband only at this point) and Mode E
            // read from full `d` at this level. Mode B's d_strip is
            // not addressable here — it was per-strip-sized and the
            // strip walker already consumed it.
            let d_full = self.d_scratch[k].d.as_ref().expect(
                "DBandsScratch.d must be Some for any pool dispatch in _pool_and_finalize_jod_strip \
                 (Mode B non-baseband levels are skipped via the inline pool above)",
            );
            let d_a = d_full[0].clone();
            let d_rg = d_full[1].clone();
            let d_vy = d_full[2].clone();
            let partial_idx_a = (k * N_CHANNELS) as u32;
            let partial_idx_rg = (k * N_CHANNELS + 1) as u32;
            let partial_idx_vy = (k * N_CHANNELS + 2) as u32;

            // Per-band strip body height: scale 0 strip body halved
            // to match the band's resolution. Bands whose body
            // shrinks below 1 row clamp to bh (single-strip dispatch).
            let strip_h_at_band = (strip_h_body >> k).max(1);
            let n_strips_band = if bh <= strip_h_at_band {
                1
            } else {
                bh.div_ceil(strip_h_at_band)
            };

            for s in 0..n_strips_band {
                let row_start = s * strip_h_at_band;
                let row_count = (bh - row_start).min(strip_h_at_band);
                let start_offset = row_start * bw;
                let slab_n = row_count * bw;
                let count = CubeCount::Static((slab_n as u32).div_ceil(64), 1, 1);
                unsafe {
                    pool_band_3ch_offset_kernel::launch::<R>(
                        &self.client,
                        count,
                        pool_atomic_cube_dim,
                        ArrayArg::from_raw_parts(d_a.clone(), n_px),
                        ArrayArg::from_raw_parts(d_rg.clone(), n_px),
                        ArrayArg::from_raw_parts(d_vy.clone(), n_px),
                        ArrayArg::from_raw_parts(self.partials_h.clone(), n_partials),
                        BETA_SPATIAL,
                        partial_idx_a,
                        partial_idx_rg,
                        partial_idx_vy,
                        start_offset as u32,
                        slab_n as u32,
                        n_px as u32,
                    );
                }
                outer_strip_iters += 1;
            }
        }

        self.strip_dispatch_counter
            .fetch_add(outer_strip_iters, core::sync::atomic::Ordering::Relaxed);

        let bytes = self
            .client
            .read_one(self.partials_h.clone())
            .map_err(|_| Error::InvalidImageSize)?;
        let partials_data: &[f32] = f32::from_bytes(&bytes);

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px_k) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            for c in 0..N_CHANNELS {
                q[c] = pool_band_finalize(partials_data[k * N_CHANNELS + c], n_px_k, BETA_SPATIAL);
            }
            q_per_ch.push(q);
        }

        Ok(do_pooling_and_jod_still_3ch(&q_per_ch))
    }

    /// Lazy-allocate the diffmap GPU scratch (one-time alloc, reused
    /// across calls). See [`DiffmapScratch`] for the buffer layout +
    /// memory cost.
    fn _ensure_diffmap_scratch(&mut self) {
        if self.diffmap_scratch.is_some() {
            return;
        }
        let n0 = (self.width as usize) * (self.height as usize);
        let acc = [
            alloc_zeros_f32(&self.client, n0),
            alloc_zeros_f32(&self.client, n0),
            alloc_zeros_f32(&self.client, n0),
        ];
        let out = alloc_zeros_f32(&self.client, n0);
        self.diffmap_scratch = Some(DiffmapScratch { acc, out });
    }

    /// Lazy-allocate the linear-RGB upload scratch (one-time alloc,
    /// reused across calls). See [`LinearPlanesUpload`] for the
    /// buffer layout + memory cost.
    fn _ensure_linear_planes_upload(&mut self) {
        if self.linear_planes_upload.is_some() {
            return;
        }
        let n0 = (self.width as usize) * (self.height as usize);
        let planes = [
            alloc_zeros_f32(&self.client, n0),
            alloc_zeros_f32(&self.client, n0),
            alloc_zeros_f32(&self.client, n0),
        ];
        self.linear_planes_upload = Some(LinearPlanesUpload { planes });
    }

    /// Drop the per-band masked-difference planes (`d_scratch[k].d[c]`)
    /// through the diffmap accumulator, then run the channel pool and
    /// read back the W·H f32 plane into `diffmap_out`.
    ///
    /// Pre-condition: `d_scratch[k].d[c]` is GPU-resident for every
    /// `(k, c)` (set up by `_dispatch_d_bands_into_scratch` /
    /// `_dispatch_d_bands_dist_and_band_loop`).
    ///
    /// Post-condition: `diffmap_out` has length `width * height` and
    /// contains the diffmap per the recipe in
    /// [`crate::kernels::diffmap`]'s module docs.
    ///
    /// `diffmap_out`'s capacity is grown as needed; existing content
    /// is overwritten via `clear` + extend so callers can reuse a
    /// long-lived `Vec`.
    fn _compute_diffmap_into(&mut self, diffmap_out: &mut Vec<f32>) -> Result<()> {
        // Path A Phase 1d (2026-05-26): Mode B's per-strip d buffers
        // are overwritten across strip iterations — the diffmap
        // accumulator needs the FULL d plane at every level. Reject
        // Mode B here; production score_with_diffmap callers are
        // expected to keep using Full / CachedRef modes (the diffmap
        // path has never been wired through the strip walker).
        if matches!(
            self.strip_config,
            Some(StripConfig { mode: StripMode::Pair, .. }),
        ) {
            return Err(Error::InvalidImageSize);
        }
        self._ensure_diffmap_scratch();
        let n_levels = self.n_levels as usize;
        let n0 = (self.width as usize) * (self.height as usize);

        // Take ownership of the scratch handles briefly so we can pass
        // them by Clone without conflicting with `&mut self`. The
        // borrow on `diffmap_scratch` is dropped before the kernel
        // launches (which take only `&self.client` immutably).
        let scratch = self.diffmap_scratch.as_ref().expect("ensured above");
        let acc_a = scratch.acc[0].clone();
        let acc_rg = scratch.acc[1].clone();
        let acc_vy = scratch.acc[2].clone();
        let out = scratch.out.clone();

        let cube_dim = CubeDim::new_1d(64);
        let count_base = CubeCount::Static((n0 as u32).div_ceil(64), 1, 1);

        // Step 1: zero the 3 accumulator planes.
        for handle in [acc_a.clone(), acc_rg.clone(), acc_vy.clone()] {
            unsafe {
                diffmap_zero_kernel::launch::<R>(
                    &self.client,
                    count_base.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(handle, n0),
                    n0 as u32,
                );
            }
        }

        // Step 2: for each band, upsample D[k][c] to base res via
        // bilinear sampling and add `per_sband_w * PER_CH_W` weighted
        // sample into the matching accumulator plane.
        let dst_w = self.width;
        let dst_h = self.height;
        for k in 0..n_levels {
            let bw = self.gauss_ref[k].w;
            let bh = self.gauss_ref[k].h;
            let n_k = (bw as usize) * (bh as usize);
            let is_baseband = k == n_levels - 1;
            let w_a = PER_CH_W[0] * if is_baseband { BASEBAND_W[0] } else { 1.0 };
            let w_rg = PER_CH_W[1] * if is_baseband { BASEBAND_W[1] } else { 1.0 };
            let w_vy = PER_CH_W[2] * if is_baseband { BASEBAND_W[2] } else { 1.0 };
            let d_full = self.d_scratch[k].d.as_ref().expect(
                "DBandsScratch.d must be Some in _compute_diffmap_into (Mode B rejected above)",
            );
            let d_a = d_full[0].clone();
            let d_rg = d_full[1].clone();
            let d_vy = d_full[2].clone();
            unsafe {
                diffmap_band_accumulate_kernel::launch::<R>(
                    &self.client,
                    count_base.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(d_a, n_k),
                    ArrayArg::from_raw_parts(d_rg, n_k),
                    ArrayArg::from_raw_parts(d_vy, n_k),
                    ArrayArg::from_raw_parts(acc_a.clone(), n0),
                    ArrayArg::from_raw_parts(acc_rg.clone(), n0),
                    ArrayArg::from_raw_parts(acc_vy.clone(), n0),
                    bw,
                    bh,
                    dst_w,
                    dst_h,
                    w_a,
                    w_rg,
                    w_vy,
                );
            }
        }

        // Step 3: per-pixel Minkowski-p pool across the 3 DKL channels.
        unsafe {
            diffmap_channel_pool_kernel::launch::<R>(
                &self.client,
                count_base.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(acc_a, n0),
                ArrayArg::from_raw_parts(acc_rg, n0),
                ArrayArg::from_raw_parts(acc_vy, n0),
                ArrayArg::from_raw_parts(out.clone(), n0),
                BETA_CH,
                n0 as u32,
            );
        }

        // Step 4: read back the per-pixel diffmap into the caller's Vec.
        let bytes = self
            .client
            .read_one(out)
            .map_err(|_| Error::InvalidImageSize)?;
        let data: &[f32] = f32::from_bytes(&bytes);
        debug_assert_eq!(data.len(), n0);
        diffmap_out.clear();
        diffmap_out.extend_from_slice(data);
        Ok(())
    }

    /// Pool + finalize the JOD scalar AND fill the per-pixel diffmap
    /// buffer. The scalar fold and the diffmap fold both consume the
    /// per-band masked-difference planes resident in
    /// `self.d_scratch[k].d[c]` — running both folds in one helper
    /// avoids re-dispatching the upstream band loop.
    fn _pool_and_finalize_jod_with_diffmap(&mut self, diffmap_out: &mut Vec<f32>) -> Result<f32> {
        let jod = self._pool_and_finalize_jod()?;
        self._compute_diffmap_into(diffmap_out)?;
        Ok(jod)
    }

    /// Run color + Laplacian-pyramid + per-band CSF weighting.
    ///
    /// `ppd` is pixels-per-degree. **Unlike the JOD-path helpers,
    /// this function genuinely consumes `ppd`** — `precomputed_band_weights`
    /// runs on host with the caller-passed value to compute per-band
    /// `rho_k` for the weight LUT. Pass a `ppd` whose
    /// `band_frequencies(ppd, w, h)` length matches the construction-
    /// time `n_levels`, or the weights buffer will mismatch the
    /// kernel's expectations. Tick 246 reverted a misplaced tick-243
    /// debug_assert here.
    /// `l_bkg` is the scalar background-luminance approximation used
    /// for every pyramid band — typically a per-image mean or
    /// display-peak / 2. The per-pixel L_bkg form (cvvdp's exact
    /// behaviour) lands once we wire the achromatic `gauss\[1\]`
    /// read path into the kernel.
    ///
    /// Returns the same shape as `compute_dkl_laplacian_pyramid`:
    /// `levels[k] = [a, rg, vy]` planar f32 vecs, with each pixel
    /// already multiplied by `sensitivity_corrected_scalar(rho_k,
    /// l_bkg, channel)`.
    ///
    /// # Examples
    ///
    /// Read back CSF-weighted Laplacian bands for a 64×64 buffer at
    /// the standard-4K ppd, with `l_bkg = 100.0` cd/m² (a typical
    /// display midtone). `ignore` for the standard `Cvvdp::*` reason.
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let srgb = vec![128u8; (w * h * 3) as usize];
    /// let bands = cvvdp.compute_dkl_csf_weighted_bands(&srgb, ppd, 100.0)
    ///     .expect("compute_dkl_csf_weighted_bands");
    /// assert!(!bands.is_empty());
    /// // bands[0] is finest level, base resolution.
    /// assert_eq!(bands[0][0].len(), (w * h) as usize);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `srgb.len() !=
    /// width × height × 3`, or [`Error::InvalidImageSize`] if a
    /// GPU readback / kernel dispatch fails anywhere in the
    /// color → gauss → laplacian → CSF chain.
    pub fn compute_dkl_csf_weighted_bands(
        &mut self,
        srgb: &[u8],
        ppd: f32,
        l_bkg: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        // Note: unlike the Weber-chain helpers, this function genuinely
        // consumes the caller-passed `ppd` (via `precomputed_band_weights`
        // below). Tick 246 reverts the tick-243 debug_assert that was
        // added in error here. Pyramid shape is still fixed by
        // construction-time geometry (n_levels is set by Cvvdp::new),
        // so passing a `ppd` whose `band_frequencies(ppd, w, h)` length
        // differs from the construction-time n_levels would mismatch
        // the weights buffer — caller must ensure consistency at the
        // pyramid-shape level, not just the PPD scalar.

        // Overwrites bands_ref[k] (first with Laplacian bands via
        // _dispatch_laplacian_pyramid_gpu, then in place via the per-
        // (level, channel) weight_band_kernel loop below). Either
        // way the warm-ref Weber bands are gone; invalidate the
        // cached scalar so a subsequent compute_dkl_jod_with_warm_ref
        // surfaces NoWarmReference. Same shape as the tick-236 fix
        // for compute_dkl_weber_pyramid.
        self.warm_ref_baseband_log_l_bkg = None;

        // Leaves the un-weighted Laplacian bands in
        // self.bands_ref[k].planes[c]. Uses the dispatch-only helper
        // so we don't pay for a full-pyramid host readback we'd
        // immediately discard.
        self._dispatch_laplacian_pyramid_gpu(srgb)?;

        let weights_per_level =
            precomputed_band_weights(ppd, self.width as usize, self.height as usize, l_bkg);
        let n_levels = self.n_levels as usize;
        // Pyramid shape is fixed by Cvvdp::new; the per-level
        // weight_band_kernel loop reads `weight_idx = k * N_CHANNELS + c`
        // into the flat weights buffer for `k = 0..n_levels`. If the
        // caller's ppd produces fewer band frequencies than
        // construction-time n_levels, those higher-k kernel launches
        // would read past `flat_weights.len()`. Debug-assert the match;
        // release builds will silently OOB-read on a violation (which
        // is the function's documented precondition). Tick 247.
        debug_assert_eq!(
            weights_per_level.len(),
            n_levels,
            "precomputed_band_weights(ppd={}, w={}, h={}) yielded {} bands but \
             Cvvdp construction-time n_levels = {}; \
             the caller-passed ppd implies a different pyramid shape — \
             reconstruct the Cvvdp instance against the new geometry instead.",
            ppd,
            self.width,
            self.height,
            weights_per_level.len(),
            n_levels,
        );
        let flat_weights = flatten_band_weights(&weights_per_level);
        let weights_handle = self.client.create_from_slice(f32::as_bytes(&flat_weights));

        let cube_dim = CubeDim::new_1d(64);
        for k in 0..n_levels {
            let n_px = (self.bands_ref[k].w * self.bands_ref[k].h) as usize;
            let cube_count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
            for c in 0..N_CHANNELS {
                let weight_idx = (k * N_CHANNELS + c) as u32;
                let band = self.bands_ref[k].planes[c].clone();
                unsafe {
                    weight_band_kernel::launch::<R>(
                        &self.client,
                        cube_count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(band, n_px),
                        ArrayArg::from_raw_parts(weights_handle.clone(), flat_weights.len()),
                        weight_idx,
                        n_px as u32,
                    );
                }
            }
        }

        // Read back every band × every channel.
        let mut out: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let mut planes = [Vec::new(), Vec::new(), Vec::new()];
            for c in 0..N_CHANNELS {
                let h = self.bands_ref[k].planes[c].clone();
                let bytes = self
                    .client
                    .read_one(h)
                    .map_err(|_| Error::InvalidImageSize)?;
                planes[c] = f32::from_bytes(&bytes).to_vec();
            }
            out.push(planes);
        }
        Ok(out)
    }

    /// Score a (reference, distorted) sRGB pair, returning JOD on
    /// the cvvdp scale (0–10; 10 = imperceptible).
    ///
    /// Routes through the full GPU composition path
    /// ([`Cvvdp::compute_dkl_jod`]): color → Weber pyramid → CSF →
    /// masking → spatial pool → host fold. Tick 213 switched from
    /// the host-scalar reference path now that all 6 v1 R2 manifest
    /// q-levels match pycvvdp at ≤ 0.005 JOD on the GPU pipeline
    /// (the canonical parity tolerance; see
    /// `shadow_jod_gpu_runs_and_is_close_to_manifest_on_corpus`).
    ///
    /// Output matches the prior host-scalar path to f32 noise
    /// (verified by `compute_dkl_jod_matches_host_scalar`,
    /// `compute_dkl_jod_host_pool_matches_compute_dkl_jod`); callers
    /// that need the all-host reference for any reason can still
    /// invoke [`crate::host_scalar::predict_jod_still_3ch`] directly,
    /// or [`Cvvdp::compute_dkl_jod_host_pool`] for the cpu-runtime
    /// host-pool path.
    ///
    /// The viewing geometry comes from `self.geometry` — set via
    /// `Cvvdp::new_with_geometry` or defaulted to STANDARD_4K by
    /// `Cvvdp::new`.
    ///
    /// # Examples
    ///
    /// Score a 64×64 byte-identical pair on the CUDA backend (max JOD = 10).
    /// `no_run` because docs.rs has no GPU; the call shape compiles against
    /// every cubecl backend with a working atomic-f32 pool (cuda, wgpu,
    /// hip — see [`Cvvdp::compute_dkl_jod_host_pool`] for the cpu runtime):
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    /// let bytes = vec![128u8; (w * h * 3) as usize];
    /// let jod = cvvdp.score(&bytes, &bytes).expect("score");
    /// assert!((jod - 10.0).abs() < 1e-3, "expected JOD ≈ 10, got {jod}");
    /// ```
    ///
    /// # Backend support
    ///
    /// See the "Backend support" section on [`Cvvdp::compute_dkl_jod`]
    /// — `score` inherits its constraints. `cubecl-cpu` callers
    /// must route through [`Cvvdp::compute_dkl_jod_host_pool`]
    /// instead; Metal callers should too.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either input buffer's
    /// length doesn't match `width × height × 3`, or
    /// [`Error::InvalidImageSize`] if a GPU readback or dispatch
    /// fails inside the underlying [`Cvvdp::compute_dkl_jod`].
    pub fn score(&mut self, reference_srgb: &[u8], distorted_srgb: &[u8]) -> Result<f64> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if reference_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: reference_srgb.len(),
            });
        }
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        // Mode B (StripPair): currently routes through the Full
        // pipeline for JOD parity. The per-strip walker that delivers
        // memory reduction at runtime is the multi-day port documented
        // in `docs/STRIP_PROCESSING.md` — the strip-aware kernels are
        // already in place; wiring them through the walker is the
        // remaining engineering. The estimator
        // ([`estimate_gpu_memory_bytes_strip_pair`]) models the
        // post-port memory profile (~32% of Full at 1024², ~12% at
        // 4096²); the constructor today still allocates Full-mode
        // buffers, so runtime savings are zero until Chunk 2 lands.
        let ppd = self.geometry.pixels_per_degree();
        let jod = self.compute_dkl_jod(reference_srgb, distorted_srgb, ppd)?;
        Ok(f64::from(jod))
    }

    /// Cache the reference side for repeated `score_with_reference`
    /// calls against many distorted candidates.
    ///
    /// Stashes the raw sRGB bytes; each subsequent
    /// `score_with_reference` re-runs the full GPU
    /// `compute_dkl_jod` against them. For the dedicated warm-ref
    /// fast path that materialises the REF Weber pyramid on the
    /// GPU once and skips it per DIST call, use
    /// [`Cvvdp::warm_reference`] +
    /// [`Cvvdp::compute_dkl_jod_with_warm_ref`] (~1.8× per-DIST
    /// throughput at 12 MP — see `lib.rs` Status).
    ///
    /// Calling `set_reference` a second time **replaces** the prior
    /// cached reference; only the most recent stash is used by
    /// subsequent `score_with_reference` calls. Pinned by
    /// `set_reference_replaces_prior_cache` (tick 249). Does NOT
    /// disturb the separate warm-ref state — see
    /// `set_reference_does_not_invalidate_warm_state` (tick 238).
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if `reference_srgb.len()
    /// != width × height × 3`. No GPU dispatch happens here — the
    /// bytes are stashed host-side until the next
    /// `score_with_reference` call.
    ///
    /// # Examples
    ///
    /// `ignore` for the same reason as [`Cvvdp::new`] — runtime needs
    /// a live GPU. Pattern is exercised in `tests/state_machine_independence.rs`.
    ///
    /// ```ignore
    /// use cubecl::{Runtime, cuda::CudaRuntime};
    /// use cvvdp_gpu::{Cvvdp, CvvdpParams};
    ///
    /// let client = CudaRuntime::client(&Default::default());
    /// let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, 256, 256, CvvdpParams::PLACEHOLDER)?;
    /// let ref_bytes = vec![128_u8; 256 * 256 * 3];
    /// cvvdp.set_reference(&ref_bytes)?;
    ///
    /// // Now multiple DISTs can score against the cached REF —
    /// // each call re-runs the full GPU pipeline (matches
    /// // `score(ref, dist)` bit-for-bit; the faster path is
    /// // `warm_reference` + `compute_dkl_jod_with_warm_ref`).
    /// for dist_bytes in &[vec![100_u8; 256 * 256 * 3], vec![120_u8; 256 * 256 * 3]] {
    ///     let _jod = cvvdp.score_with_reference(dist_bytes)?;
    /// }
    /// # Ok::<(), cvvdp_gpu::Error>(())
    /// ```
    pub fn set_reference(&mut self, reference_srgb: &[u8]) -> Result<()> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if reference_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: reference_srgb.len(),
            });
        }
        self.cached = Some(CachedReference {
            ref_srgb: reference_srgb.to_vec(),
        });
        Ok(())
    }

    /// Score a distorted candidate against the cached reference.
    /// Matches `score(ref, dist)` exactly (same GPU pipeline as of
    /// tick 213); the dedicated warm-ref fast path that skips re-
    /// running the REF weber pyramid per call lives at
    /// [`Cvvdp::warm_reference`] + [`Cvvdp::compute_dkl_jod_with_warm_ref`]
    /// — `score_with_reference` keeps the simple ref-stashing
    /// contract from v0 and re-dispatches REF on every call.
    ///
    /// Returns [`Error::NoCachedReference`] if [`Cvvdp::set_reference`]
    /// wasn't called first.
    ///
    /// # Examples
    ///
    /// Batch-score N distorted candidates against one stashed reference
    /// on the CUDA backend (max JOD = 10 for the byte-identical pair).
    /// `ignore` because docs.rs has no GPU AND the no-default-features
    /// build path doesn't include any `cubecl` runtime feature, so
    /// the `Backend` type alias below would fail to resolve:
    ///
    /// ```ignore
    /// use cvvdp_gpu::Cvvdp;
    /// use cvvdp_gpu::params::CvvdpParams;
    /// use cubecl::Runtime;
    ///
    /// # #[cfg(feature = "cuda")]
    /// type Backend = cubecl::cuda::CudaRuntime;
    /// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    /// # type Backend = cubecl::wgpu::WgpuRuntime;
    /// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
    /// # type Backend = cubecl::cpu::CpuRuntime;
    /// let client = Backend::client(&Default::default());
    /// let (w, h) = (64u32, 64u32);
    /// let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
    ///     .expect("Cvvdp::new");
    ///
    /// let reference = vec![128u8; (w * h * 3) as usize];
    /// cvvdp.set_reference(&reference).expect("set_reference");
    ///
    /// // Score the same buffer against itself — perceptually identical.
    /// let jod = cvvdp.score_with_reference(&reference).expect("score_with_reference");
    /// assert!((jod - 10.0).abs() < 1e-3, "expected JOD ≈ 10, got {jod}");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`Error::NoCachedReference`] if `set_reference` wasn't
    ///   called first.
    /// - [`Error::DimensionMismatch`] if `distorted_srgb.len() !=
    ///   width × height × 3`.
    /// - [`Error::InvalidImageSize`] if the underlying
    ///   [`Cvvdp::compute_dkl_jod`] dispatch fails.
    pub fn score_with_reference(&mut self, distorted_srgb: &[u8]) -> Result<f64> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        let ref_srgb = self
            .cached
            .as_ref()
            .ok_or(Error::NoCachedReference)?
            .ref_srgb
            .clone();
        let ppd = self.geometry.pixels_per_degree();
        let jod = self.compute_dkl_jod(&ref_srgb, distorted_srgb, ppd)?;
        Ok(f64::from(jod))
    }

    // ===================================================================
    // Diffmap + linear-planes API (see kernels::diffmap module docs for
    // the recipe contract; see docs/DIFFMAP_DIVERGENCES.md for the note
    // on the relationship between the per-pixel diffmap and the scalar
    // JOD).
    // ===================================================================

    /// One-shot score from sRGB-byte inputs that ALSO fills a
    /// per-pixel diffmap.
    ///
    /// Same JOD scalar as [`Self::score`] (and same numerical
    /// pipeline; the diffmap fold runs alongside the scalar fold,
    /// it doesn't replace it). `diffmap_out` is overwritten via
    /// `clear` + extend so callers can reuse a long-lived `Vec`.
    /// On return, `diffmap_out.len() == width * height` and the
    /// values are non-negative f32 row-major.
    ///
    /// Returns the JOD on cvvdp's 0–10 scale (10 = identical pair).
    ///
    /// See [`crate::kernels::diffmap`] module docs for the recipe
    /// the diffmap follows, and `docs/DIFFMAP_DIVERGENCES.md` for
    /// the note on its relationship to the scalar JOD.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if either input buffer's
    ///   length doesn't match `width × height × 3`.
    /// - [`Error::InvalidImageSize`] on GPU dispatch / readback
    ///   failure anywhere in the pipeline.
    pub fn score_with_diffmap(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if ref_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_srgb.len(),
            });
        }
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb)?;
        self._pool_and_finalize_jod_with_diffmap(diffmap_out)
    }

    /// Warm-ref variant of [`Self::score_with_diffmap`]. Requires a
    /// prior [`Self::warm_reference`] (or
    /// [`Self::warm_reference_from_linear_planes`]) call; skips the
    /// REF half of the pipeline per the same warm-state contract as
    /// [`Self::compute_dkl_jod_with_warm_ref`].
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `dist_srgb.len() != width × height × 3`.
    /// - [`Error::NoWarmReference`] if no warm REF state is cached.
    /// - [`Error::InvalidImageSize`] on GPU dispatch failure.
    pub fn score_with_warm_ref_diffmap(
        &mut self,
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        let log_l_bkg_baseband = self._warm_ref_baseband_log_l_bkg_for_dispatch()?;
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband)?;
        self._pool_and_finalize_jod_with_diffmap(diffmap_out)
    }

    /// Returns the baseband `log10(L_bkg)` scalar for the cached ref
    /// state. Shared helper across every warm-ref dispatcher (sRGB-byte,
    /// linear-planes, diffmap variants).
    ///
    /// Strip mode (Mode E) reads the cached scalar from
    /// `ref_full_state.baseband_log_l_bkg_scalar` — the dedicated REF
    /// state populated by `warm_reference`. The band loop (see
    /// [`Self::_run_d_bands_band_loop`]) reads REF bands and
    /// non-baseband `log_l_bkg` planes directly from
    /// [`RefFullState`] too, so this scalar is the only ref-side
    /// payload returned to the caller.
    ///
    /// Full mode reads from `warm_ref_baseband_log_l_bkg`.
    ///
    /// Returns [`Error::NoWarmReference`] if no warm REF state is
    /// cached (Full mode: `warm_ref_baseband_log_l_bkg` is `None`;
    /// strip mode: `ref_full_state` is `None`).
    fn _warm_ref_baseband_log_l_bkg_for_dispatch(&mut self) -> Result<f32> {
        if self.strip_config.is_some() {
            self.ref_full_state
                .as_ref()
                .map(|s| s.baseband_log_l_bkg_scalar)
                .ok_or(Error::NoWarmReference)
        } else {
            self.warm_ref_baseband_log_l_bkg
                .ok_or(Error::NoWarmReference)
        }
    }

    /// One-shot score from three planar `W × H` linear-RGB f32
    /// buffers (one per primary, unit-scaled sRGB linear-light).
    /// Skips the host-side sRGB-byte upload pack + sRGB→linear LUT
    /// kernel — direct path from caller-owned linear-light buffers
    /// to the DKL pipeline. Mirrors butteraugli-gpu's
    /// `compute_with_reference_from_linear_planes`
    /// (W44-PHASE3-B4 in the jxl-encoder repo).
    ///
    /// Display model (`y_peak`, `y_black`, `y_refl`) and the DKL
    /// matrix still apply on GPU; the caller is responsible for
    /// linearising sRGB only.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane length differs
    ///   from `width × height`.
    /// - [`Error::InvalidImageSize`] on GPU dispatch failure.
    pub fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        self._dispatch_d_bands_into_scratch_from_linear_planes(
            ref_r, ref_g, ref_b, dist_r, dist_g, dist_b,
        )?;
        self._pool_and_finalize_jod()
    }

    /// As [`Self::score_from_linear_planes`], plus per-pixel diffmap.
    #[allow(clippy::too_many_arguments)] // 6 planar f32 slices + W*H out — natural shape for "ref + dist" linear-RGB.
    pub fn score_from_linear_planes_with_diffmap(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self._dispatch_d_bands_into_scratch_from_linear_planes(
            ref_r, ref_g, ref_b, dist_r, dist_g, dist_b,
        )?;
        self._pool_and_finalize_jod_with_diffmap(diffmap_out)
    }

    /// Warm the REF side using three planar linear-RGB f32 buffers.
    /// Mirrors [`Self::warm_reference`] but uses the
    /// linear-planes upload path. Subsequent calls to
    /// [`Self::score_from_linear_planes_with_warm_ref`] /
    /// [`Self::score_from_linear_planes_with_warm_ref_diffmap`]
    /// reuse the cached REF state.
    ///
    /// Warm-state invalidation rules are identical to
    /// [`Self::warm_reference`] — any subsequent REF-dispatching
    /// method overwrites `bands_ref`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane length differs
    ///   from `width × height`.
    /// - [`Error::InvalidImageSize`] on GPU dispatch failure.
    pub fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        // _validate runs once host-side; the dispatch helper also
        // validates but we surface the boundary error first per the
        // same ordering as warm_reference / set_reference.
        let _ = self._validate_linear_planes(ref_r, ref_g, ref_b)?;
        let log_l_bkg_baseband =
            self._dispatch_ref_weber_pyramid_only_from_linear_planes(ref_r, ref_g, ref_b)?;
        self.warm_ref_baseband_log_l_bkg = Some(log_l_bkg_baseband);

        // Mode E (task #79): snapshot for the strip-mode cached-ref
        // contract — see `warm_reference` for the rationale.
        if self.strip_config.is_some() {
            self._snapshot_ref_state_to_full(log_l_bkg_baseband)?;
        }
        Ok(())
    }

    /// Score a DIST candidate (planar linear-RGB f32) against the
    /// warm-cached REF state. Returns the JOD scalar only.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane length differs
    ///   from `width × height` (checked first, ahead of the warm
    ///   state, per the tick-248 precedence audit).
    /// - [`Error::NoWarmReference`] if the warm state is missing or
    ///   was invalidated by an intervening REF dispatch.
    /// - [`Error::InvalidImageSize`] on GPU dispatch failure.
    pub fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        let _ = self._validate_linear_planes(dist_r, dist_g, dist_b)?;
        let log_l_bkg_baseband = self._warm_ref_baseband_log_l_bkg_for_dispatch()?;
        self._dispatch_d_bands_dist_and_band_loop_from_linear_planes(
            dist_r,
            dist_g,
            dist_b,
            log_l_bkg_baseband,
        )?;
        // Phase 3 strip-aware pool routes for Mode E. See
        // `compute_dkl_jod_with_warm_ref` for the rationale.
        if self.strip_config.is_some() {
            self._pool_and_finalize_jod_strip()
        } else {
            self._pool_and_finalize_jod()
        }
    }

    /// Score a DIST candidate (planar linear-RGB f32) against the
    /// warm-cached REF state and fill a per-pixel diffmap.
    ///
    /// # Errors
    ///
    /// Same as [`Self::score_from_linear_planes_with_warm_ref`].
    pub fn score_from_linear_planes_with_warm_ref_diffmap(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let _ = self._validate_linear_planes(dist_r, dist_g, dist_b)?;
        let log_l_bkg_baseband = self._warm_ref_baseband_log_l_bkg_for_dispatch()?;
        self._dispatch_d_bands_dist_and_band_loop_from_linear_planes(
            dist_r,
            dist_g,
            dist_b,
            log_l_bkg_baseband,
        )?;
        self._pool_and_finalize_jod_with_diffmap(diffmap_out)
    }
}
