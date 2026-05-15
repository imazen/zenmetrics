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
//! 6. Per-band Minkowski accumulation (`pool::pool_band_kernel`) →
//!    per-band f32 partials (one `f32` per (level, channel) in a
//!    shared GPU buffer).
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
//! - `d_scratch[k]` (per level, including baseband) — `t_p_ref`,
//!   `t_p_dis`, masking-chain (`m_raw`, `m_mid`, `m_blur`), and
//!   the final `d` output handles, all per-channel. Every level's
//!   D plane lives in `d_scratch[k].d[c]` regardless of whether
//!   the band ran through the masker (`mult_mutual_3ch_*`) or the
//!   baseband bypass (`diff_abs_3ch`).
//! - `logs_row[k][c]` — pre-uploaded 32-entry CSF sensitivity LUT
//!   row per (level, channel); stable across calls since `rho_k`
//!   is fixed for this Cvvdp.
//!
//! Total per-Cvvdp budget at 4000×3000 with 8 pyramid levels is
//! ~700 MB of GPU memory (dominated by `d_scratch.t_p_*` and the
//! masking-chain scratch buffers). All allocations happen once in
//! `Cvvdp::new`; the hot path does only `create_from_slice` for
//! input bytes and reads back small results.

use cubecl::prelude::*;

use crate::kernels::color::{SRGB8_TO_LINEAR_LUT, srgb_to_dkl_kernel};
use crate::kernels::csf::{
    CsfChannel, csf_apply_3ch_kernel, csf_apply_6ch_kernel, flatten_band_weights,
    precompute_logs_row, precomputed_band_weights, weight_band_kernel,
};
use crate::kernels::masking::{
    CH_GAIN, MASK_C, PU_PADSIZE, diff_abs_3ch_kernel, min_abs_3ch_kernel,
    mult_mutual_3ch_no_blur_kernel, mult_mutual_3ch_with_blurred_kernel, pu_blur_h_3ch_kernel,
    pu_blur_v_3ch_scaled_kernel,
};
use crate::kernels::pool::{
    BETA_SPATIAL, do_pooling_and_jod_still_3ch, fill_f32_kernel, lp_norm_mean,
    pool_band_3ch_kernel, pool_band_finalize,
};
use crate::kernels::pyramid::{
    band_frequencies, baseband_divide_3ch_kernel, downscale_kernel, subtract_kernel,
    subtract_weber_3ch_kernel, upscale_h_kernel, upscale_v_kernel,
};
use crate::params::CvvdpParams;
use crate::{Error, MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM, Result};

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

/// Per-level scratch buffers reused by `compute_dkl_d_bands` so the
/// hot loop doesn't allocate per band. At 12 MP the function would
/// otherwise allocate ~1.5 GB of transient GPU buffers per call (3
/// channels × 2 sides × 6 buffer kinds × per-level size). Pre-
/// allocating once on `Cvvdp::new` keeps the steady-state cost off
/// the per-frame budget.
struct DBandsScratch {
    /// CSF-applied bands per channel for ref and dist sides.
    /// `compute_dkl_d_bands` runs `csf_apply_per_pixel_kernel` into
    /// these (one launch per side per channel).
    t_p_ref: [cubecl::server::Handle; N_CHANNELS],
    t_p_dis: [cubecl::server::Handle; N_CHANNELS],
    /// Masking-chain scratch (non-baseband levels only).
    m_raw: [cubecl::server::Handle; N_CHANNELS],
    m_mid: [cubecl::server::Handle; N_CHANNELS],
    m_blur: [cubecl::server::Handle; N_CHANNELS],
    /// Per-band masked-difference output (consumed by host
    /// `lp_norm_mean` after read-back).
    d: [cubecl::server::Handle; N_CHANNELS],
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
    l_bkg_fine: cubecl::server::Handle,
    /// Vertical-pass scratch for achromatic L_bkg expand (n_v).
    vscratch_a: cubecl::server::Handle,
    /// Per-pixel log10(L_bkg) plane for the REF side (n_fine).
    /// Persists through the band loop so the CSF kernel can read
    /// it directly without a host roundtrip (tick 166).
    log_l_bkg: cubecl::server::Handle,
    /// Throwaway destination for DIST's log_l_bkg write — same
    /// shape as `log_l_bkg`. cvvdp's weber_g1 rule uses REF's
    /// log_l_bkg for both sides, so DIST's value is computed but
    /// discarded; this lets the DIST dispatch write somewhere
    /// without clobbering REF.
    log_l_bkg_dis: cubecl::server::Handle,
    /// Per-channel vertical/horizontal expand scratch (n_v, n_fine).
    /// The previous `layer_c` intermediate is gone — tick 91 fuses
    /// `subtract + weber` into a single 3-channel kernel that reads
    /// `fine` + `upscaled_c` directly.
    vscratch_c: [cubecl::server::Handle; N_CHANNELS],
    upscaled_c: [cubecl::server::Handle; N_CHANNELS],
}

fn build_weber_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
) -> Vec<WeberScratch> {
    let mut out = Vec::with_capacity(n_levels.saturating_sub(1));
    let mut fine_w = width;
    let mut fine_h = height;
    // Only non-baseband levels need scratch (baseband bypasses the
    // expand/subtract/weber chain).
    for _ in 0..n_levels.saturating_sub(1) {
        // Ceil-div halving — matches cvvdp's `gausspyr_reduce`
        // boundary semantics so the GPU pyramid stays bit-stable
        // against the host scalar reference at all sizes (not just
        // even-dim corpora). See `gausspyr_reduce_scalar` in
        // kernels/pyramid.rs (which already uses div_ceil(2)).
        let coarse_w = (fine_w + 1) / 2;
        let coarse_h = (fine_h + 1) / 2;
        let n_fine = (fine_w as usize) * (fine_h as usize);
        let n_v = (coarse_w as usize) * (fine_h as usize);
        out.push(WeberScratch {
            l_bkg_fine: alloc_zeros_f32(client, n_fine),
            vscratch_a: alloc_zeros_f32(client, n_v),
            log_l_bkg: alloc_zeros_f32(client, n_fine),
            log_l_bkg_dis: alloc_zeros_f32(client, n_fine),
            vscratch_c: [
                alloc_zeros_f32(client, n_v),
                alloc_zeros_f32(client, n_v),
                alloc_zeros_f32(client, n_v),
            ],
            upscaled_c: [
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
                alloc_zeros_f32(client, n_fine),
            ],
        });
        fine_w = coarse_w;
        fine_h = coarse_h;
    }
    out
}

fn build_d_bands_scratch<R: Runtime>(
    client: &ComputeClient<R>,
    n_levels: usize,
    width: u32,
    height: u32,
) -> Vec<DBandsScratch> {
    let mut out = Vec::with_capacity(n_levels);
    let mut w = width;
    let mut h = height;
    for _ in 0..n_levels {
        let n = (w as usize) * (h as usize);
        out.push(DBandsScratch {
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
            d: [
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
                alloc_zeros_f32(client, n),
            ],
        });
        // Ceil-div halving — see WeberScratch comment.
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
    out
}

/// Reference-side state kept across `score_with_reference` calls.
///
/// Stashes the raw sRGB bytes so the host-scalar pipeline can re-run
/// end-to-end per distorted candidate — same bytes that `score()`
/// would have re-uploaded, just kept around. Exact-parity with
/// `score(ref, dist)`.
///
/// The fast path that materializes the reference's CSF-weighted
/// pyramid bands once (`Vec<Vec<Handle>>`, indexed `[level][channel]`)
/// is the obvious next optimization but isn't wired yet — every
/// `score_with_reference` call still re-runs the full host pipeline.
/// The GPU helpers (`compute_dkl_weber_pyramid` and friends) exist
/// and could be retargeted here once `Cvvdp::score` itself routes
/// through the GPU composition.
struct CachedReference {
    /// Cached reference sRGB bytes (length `width * height * 3`).
    ref_srgb: Vec<u8>,
}

/// ColorVideoVDP scorer.
///
/// Allocates GPU buffers up front for a fixed image size and reuses
/// them across calls. To score images of a different size, construct
/// a new `Cvvdp`.
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
    bands_ref: Vec<Level>,

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
    let band_count = crate::kernels::pyramid::band_frequencies(
        ppd,
        width as usize,
        height as usize,
    )
    .len() as u32;
    band_count.min(MAX_LEVELS as u32)
}

impl<R: Runtime> Cvvdp<R> {
    /// Allocate GPU buffers for a fixed `width × height` image and the
    /// given parameter bundle. Uses
    /// [`crate::params::DisplayGeometry::STANDARD_4K`] as the viewing
    /// geometry — equivalent to `new_with_geometry(..., STANDARD_4K)`.
    /// Override via `new_with_geometry` for non-4K displays.
    ///
    /// Returns [`Error::InvalidImageSize`] if either dimension is
    /// smaller than [`PYRAMID_MIN_DIM`] × 2 (no usable pyramid).
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

    /// Allocate GPU buffers + record a custom viewing geometry. The
    /// geometry is used by `score` to derive PPD (and thus the
    /// per-band spatial frequencies the CSF table is queried with).
    pub fn new_with_geometry(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2 {
            return Err(Error::InvalidImageSize);
        }
        let n_levels = pyramid_levels(geometry.pixels_per_degree(), width, height);

        let n0 = (width as usize) * (height as usize);
        // Source-byte buffers are u32-slot arrays of length `n0 * 3`
        // — one byte per slot, RGBRGB row-major. Matches what
        // `srgb_to_dkl_kernel` expects.
        let src_ref = client.create_from_slice(u32::as_bytes(&vec![0u32; n0 * 3]));
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
                w = (w + 1) / 2;
                h = (h + 1) / 2;
            }
            out
        };

        let gauss_ref = build_pyramid(&client);
        let bands_ref = build_pyramid(&client);
        let bands_dis = build_pyramid(&client);
        let d_scratch = build_d_bands_scratch(&client, n_levels as usize, width, height);
        let weber_scratch = build_weber_scratch(&client, n_levels as usize, width, height);

        // Baseband log_l_bkg buffer. Size matches `gauss_ref[last]`
        // which `build_pyramid` allocated with ceil-div halving
        // (tick 175). Allocated once; filled per-JOD via `fill_f32_kernel`.
        let last = n_levels as usize - 1;
        let baseband_w = gauss_ref[last].w as usize;
        let baseband_h = gauss_ref[last].h as usize;
        let baseband_n = baseband_w * baseband_h;
        let baseband_log_l_bkg = alloc_zeros_f32(&client, baseband_n);

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
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[0],
                ))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[1],
                ))),
                client.create_from_slice(f32::as_bytes(&precompute_logs_row(
                    rho_k,
                    channels[2],
                ))),
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
            bands_ref,
            bands_dis,
            d_scratch,
            weber_scratch,
            baseband_log_l_bkg,
            logs_row,
            cached: None,
            warm_ref_baseband_log_l_bkg: None,
        })
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

        let src_u32: Vec<u32> = srgb.iter().map(|&b| b as u32).collect();
        self.src_ref = self.client.create_from_slice(u32::as_bytes(&src_u32));

        let a_handle = self.gauss_ref[0].planes[0].clone();
        let rg_handle = self.gauss_ref[0].planes[1].clone();
        let vy_handle = self.gauss_ref[0].planes[2].clone();

        let cube_dim = CubeDim::new_1d(64);
        let cube_count = CubeCount::Static((n0 as u32).div_ceil(64), 1, 1);

        let display = self.params.display;
        unsafe {
            srgb_to_dkl_kernel::launch::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(self.src_ref.clone(), n0 * 3),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), SRGB8_TO_LINEAR_LUT.len()),
                ArrayArg::from_raw_parts(a_handle, n0),
                ArrayArg::from_raw_parts(rg_handle, n0),
                ArrayArg::from_raw_parts(vy_handle, n0),
                self.width,
                self.height,
                display.y_peak,
                display.y_black,
                display.y_refl,
            );
        }
        Ok(())
    }

    /// Run color stage + Gaussian-pyramid reduce loop. Returns the
    /// pyramid as `levels[k] = [a, rg, vy]` planar f32 vecs, with
    /// `levels[0]` at base resolution and each subsequent level
    /// halved (cvvdp's `div_ceil(2)` convention).
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

        let cube_dim = CubeDim::new_1d(64);
        for k in 1..(self.n_levels as usize) {
            let prev_w = self.gauss_ref[k - 1].w;
            let prev_h = self.gauss_ref[k - 1].h;
            let curr_w = self.gauss_ref[k].w;
            let curr_h = self.gauss_ref[k].h;
            let n_curr = (curr_w * curr_h) as usize;
            let n_prev = (prev_w * prev_h) as usize;
            let cube_count = CubeCount::Static((n_curr as u32).div_ceil(64), 1, 1);

            for c in 0..N_CHANNELS {
                let src = self.gauss_ref[k - 1].planes[c].clone();
                let dst = self.gauss_ref[k].planes[c].clone();
                unsafe {
                    downscale_kernel::launch::<R>(
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
        Ok(())
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
    pub fn compute_dkl_laplacian_pyramid(&mut self, srgb: &[u8]) -> Result<Vec<[Vec<f32>; 3]>> {
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
    ///   gets the per-channel gauss[last] divided by the achromatic
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

        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;

        // Non-baseband levels: build layers + expanded L_bkg, then
        // launch weber_contrast_compute_kernel per channel.
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

            // Pre-allocated per-level scratch (Cvvdp.weber_scratch).
            // Reuses the same handles across calls + across both sides
            // of compute_dkl_d_bands. Each call writes-then-reads-back
            // before the next call overwrites, so the read-back captures
            // the data correctly.
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

            // Per channel: upscale coarse → fine (separable v + h).
            // Tried fusing into 3-channel kernels (see git history,
            // tick 159) — that regressed perf at 12 MP on RTX-class
            // CUDA by ~4% jod. Hypothesis: the 3ch kernel's register
            // pressure / per-thread work limited warp-level
            // parallelism; the 6 small launches per level give the
            // CUDA scheduler more in-flight warps to hide latency.
            // The 3ch fusion may still win on backends with higher
            // launch overhead (wgpu/hip) but isn't worth the
            // launch-count saving on CUDA.
            //
            // Subtract + Weber-contrast + log_l_bkg are fused into a
            // single 3-channel launch (tick 91) below — eliminates 3
            // subtract_kernel launches per level + the `layer_c`
            // intermediate Vec materialization step.
            let log_l_bkg = log_l_bkg_dest[k].clone();
            for c in 0..N_CHANNELS {
                let coarse = self.gauss_ref[k + 1].planes[c].clone();
                let vscratch_c = scratch.vscratch_c[c].clone();
                let upscaled_c = scratch.upscaled_c[c].clone();

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
            // Fused subtract + 3-channel weber-contrast. One launch
            // does `band[c] = clamp((fine[c] - upscaled[c]) / L_bkg)`
            // for all three channels plus the shared log_l_bkg.
            let fine_a = self.gauss_ref[k].planes[0].clone();
            let fine_rg = self.gauss_ref[k].planes[1].clone();
            let fine_vy = self.gauss_ref[k].planes[2].clone();
            let upsc_a = scratch.upscaled_c[0].clone();
            let upsc_rg = scratch.upscaled_c[1].clone();
            let upsc_vy = scratch.upscaled_c[2].clone();
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

        // Baseband: scalar L_bkg = mean of max(gauss_A[N-1], 0.01).
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

        // GPU divide: bands_{ref,dis}[last].planes[c] = gauss[last][c] /
        // l_bkg_mean. Replaces 3 channel readbacks + 3 reuploads with
        // one launch using the host-computed mean as a scalar uniform.
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
    pub fn compute_dkl_weber_pyramid(&mut self, srgb: &[u8]) -> Result<WeberPyramidGpu> {
        let trace_weber = std::env::var_os("CVVDP_TRACE_WEBER").is_some();
        let t_dispatch = std::time::Instant::now();

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
    /// `ppd` is pixels-per-degree (from `DisplayGeometry::pixels_per_degree()`).
    /// Each level's `rho_k` is resolved via
    /// [`crate::kernels::pyramid::band_frequencies`].
    ///
    /// Returns `levels[k] = [a, rg, vy]` planar f32 vecs, same shape
    /// as `compute_dkl_weber_pyramid`'s `.0`.
    pub fn compute_dkl_t_p_bands(&mut self, srgb: &[u8], ppd: f32) -> Result<Vec<[Vec<f32>; 3]>> {
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
        // ppd unused — logs_row is pre-uploaded against the geometry
        // baked into Cvvdp::new. compute_dkl_t_p_bands still takes
        // ppd in the signature for source-compatibility.
        let _ = ppd;

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
    ///     level's D plane lives in the same `d_scratch.d[k][c]` slot).
    ///
    /// No GPU→host readback inside this helper. Callers that want
    /// the host-side `Vec<[Vec<f32>; 3]>` snapshot use
    /// [`Cvvdp::compute_dkl_d_bands`]; callers that pool on GPU
    /// (`Cvvdp::compute_dkl_jod`) read straight from the resident
    /// handles via `pool_band_kernel`.
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
        let ref_log_l_bkg_dests: Vec<cubecl::server::Handle> = self
            .weber_scratch
            .iter()
            .map(|s| s.log_l_bkg.clone())
            .collect();
        self._dispatch_weber_pyramid_gpu(ref_srgb, &ref_log_l_bkg_dests, false)
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
        let dist_log_l_bkg_dests: Vec<cubecl::server::Handle> = self
            .weber_scratch
            .iter()
            .map(|s| s.log_l_bkg_dis.clone())
            .collect();
        let _ = self._dispatch_weber_pyramid_gpu(dist_srgb, &dist_log_l_bkg_dests, true)?;
        Ok(())
    }

    fn _dispatch_d_bands_into_scratch(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<()> {
        let trace = std::env::var_os("CVVDP_TRACE").is_some();
        let t_weber_ref = std::time::Instant::now();
        let log_l_bkg_baseband = self._dispatch_ref_weber_pyramid_only(ref_srgb)?;
        if trace {
            eprintln!("[trace] weber(ref):  {:?}", t_weber_ref.elapsed());
        }
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband, ppd)
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
        ppd: f32,
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

        let n_levels = self.n_levels as usize;
        // ppd unused — logs_row is pre-uploaded against the geometry
        // baked into Cvvdp::new. compute_dkl_d_bands keeps ppd in the
        // signature for source-compatibility.
        let _ = ppd;
        let cube_dim = CubeDim::new_1d(64);
        // `10^MASK_C` post-blur scale for the PU stage — constant
        // per Cvvdp config, so compute once outside the band loop.
        let pu_scale = 10.0_f32.powf(MASK_C);

        let t_band_loop = std::time::Instant::now();
        for k in 0..n_levels {
            let is_baseband = k == n_levels - 1;
            // band_mul = 2.0 on every level except the finest (k=0) and the
            // baseband — those use 1.0 per cvvdp's `lpyr.get_band` contract.
            let band_mul: f32 = if k == 0 || is_baseband { 1.0 } else { 2.0 };
            let (bw, bh, n_px) = self.level_dims(k);

            let t_band = std::time::Instant::now();

            // log_l_bkg source:
            // - Non-baseband bands: read directly from the GPU-resident
            //   `weber_scratch[k].log_l_bkg` handle (REF data, written
            //   during the REF weber dispatch above). Tick 166 skips
            //   the host roundtrip — was reading back ~64 MB at 12 MP
            //   then re-uploading the same bytes per band.
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

            // Reuse the pre-allocated per-level scratch (Cvvdp.d_scratch).
            // T_p / m_* / d handles are kept resident so the masking kernels
            // can consume them without a round-trip to host AND without
            // per-band alloc_zeros_f32 churn (~1.5 GB worth at 12 MP).
            let scratch = &self.d_scratch[k];
            let t_p_ref_h: [cubecl::server::Handle; 3] = [
                scratch.t_p_ref[0].clone(),
                scratch.t_p_ref[1].clone(),
                scratch.t_p_ref[2].clone(),
            ];
            let t_p_dis_h: [cubecl::server::Handle; 3] = [
                scratch.t_p_dis[0].clone(),
                scratch.t_p_dis[1].clone(),
                scratch.t_p_dis[2].clone(),
            ];

            let t_csf = std::time::Instant::now();

            // Fused 3-channel CSF apply — one launch per side instead
            // of three. The per-pixel LUT bracket math is shared across
            // the A/RG/VY channels.
            let [ch_gain_a, ch_gain_rg, ch_gain_vy] = ch_gain_for_band(is_baseband, band_mul);

            // Fused 6-channel CSF apply: one launch runs both sides
            // (REF + DIST) and shares the per-pixel LUT bracket math.
            // After tick 154's bands_ref/bands_dis split, both
            // sides' weber data lives on GPU at band-loop time —
            // REF in `self.bands_ref[k]`, DIST in `self.bands_dis[k]`.
            // No host upload needed.
            {
                unsafe {
                    csf_apply_6ch_kernel::launch::<R>(
                        &self.client,
                        count.clone(),
                        cube_dim,
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[0].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[1].clone(), n_px),
                        ArrayArg::from_raw_parts(self.bands_ref[k].planes[2].clone(), n_px),
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
            if is_baseband {
                // Baseband: cvvdp's `|T_p_dis - T_p_ref|` bypass. Tick
                // 94 — GPU fused 3-channel diff into scratch.d so the
                // baseband output lives in d_scratch.d[k][c] like every
                // other level (prep for GPU pool in tick 95).
                let d_h: [cubecl::server::Handle; 3] = [
                    scratch.d[0].clone(),
                    scratch.d[1].clone(),
                    scratch.d[2].clone(),
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
                let d_h: [cubecl::server::Handle; 3] = [
                    scratch.d[0].clone(),
                    scratch.d[1].clone(),
                    scratch.d[2].clone(),
                ];
                let use_blur = bw > PU_PADSIZE && bh > PU_PADSIZE;
                unsafe {
                    if use_blur {
                        // min_abs → pu_blur_h → pu_blur_v → mult_mutual_3ch_with_blurred.
                        let m_raw_h: [cubecl::server::Handle; 3] = [
                            scratch.m_raw[0].clone(),
                            scratch.m_raw[1].clone(),
                            scratch.m_raw[2].clone(),
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
                            scratch.m_mid[0].clone(),
                            scratch.m_mid[1].clone(),
                            scratch.m_mid[2].clone(),
                        ];
                        let m_blur_h: [cubecl::server::Handle; 3] = [
                            scratch.m_blur[0].clone(),
                            scratch.m_blur[1].clone(),
                            scratch.m_blur[2].clone(),
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

    /// Host-side readback wrapper around the GPU D-bands dispatch.
    /// Runs the full GPU dispatch (color → weber → CSF → masking)
    /// into `self.d_scratch[k].d[c]` then copies each band's D plane
    /// out into a `Vec<[Vec<f32>; 3]>`. Use this when you need the
    /// raw band values (parity checks, debugging, downstream host
    /// scalar processing); use [`Cvvdp::compute_dkl_jod`] directly
    /// when you want the JOD scalar — that path pools on GPU and
    /// avoids the full ~432 MB per-band readback at 12 MP.
    pub fn compute_dkl_d_bands(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb, ppd)?;

        let n_levels = self.n_levels as usize;
        let mut d_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px) = self.level_dims(k);
            let mut planes: [Vec<f32>; 3] =
                [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            for c in 0..N_CHANNELS {
                let bytes = self
                    .client
                    .read_one(self.d_scratch[k].d[c].clone())
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
    ///      → spatial pool (GPU, pool_band_kernel per (band, channel),
    ///        atomic-f32 accumulation into a partials Vec)
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
    /// The shadow_jod test still pins the public `Cvvdp::score`
    /// path through `host_scalar::predict_jod_still_3ch` against
    /// the v1 R2 manifest (≤ 0.006 JOD). This helper exposes the
    /// GPU-composed path so its parity vs the host scalar can be
    /// measured independently — see
    /// `tests/pipeline_color.rs::compute_dkl_jod_matches_host_scalar`,
    /// `tests/pipeline_score.rs::compute_dkl_jod_on_v1_manifest_corpus`,
    /// and the drift sweep `compute_dkl_jod_vs_host_scalar_on_corpus`.
    /// Once the GPU JOD parity vs the host scalar is locked at
    /// f32-precision tolerance, `Cvvdp::score` will switch to this
    /// helper and the manifest-parity test will retarget.
    pub fn compute_dkl_jod(&mut self, ref_srgb: &[u8], dist_srgb: &[u8], ppd: f32) -> Result<f32> {
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
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb, ppd)?;
        self._pool_and_finalize_jod()
    }

    /// CPU-backend-compatible variant of [`Cvvdp::compute_dkl_jod`].
    ///
    /// Same JOD result, but uses a host-side spatial pool instead
    /// of `pool_band_3ch_kernel`. That GPU kernel uses
    /// `Atomic<f32>::fetch_add`, which `cubecl-cpu` doesn't support;
    /// this variant reads D bands back via
    /// [`Cvvdp::compute_dkl_d_bands`] and pools them with the
    /// host-scalar `lp_norm_mean`, so it runs on every cubecl
    /// runtime — including `cubecl-cpu`.
    ///
    /// Tradeoff: the readback is `O(n_pixels × n_channels × n_levels
    /// × 4/3)` bytes (geometric series on band sizes). At 12 MP that's
    /// ≈ 432 MB GPU→host transfer per call, swamping the GPU pool's
    /// few-microsecond kernel time. **Use this only on the CPU
    /// backend** — for `cuda` / `wgpu` / `hip` runtimes prefer
    /// [`Cvvdp::compute_dkl_jod`], which keeps everything GPU-
    /// resident.
    ///
    /// Output matches `compute_dkl_jod` to f32 noise on all backends
    /// where both run (the GPU pool's atomic reduction and the host
    /// `lp_norm_mean` compute the same `safe_pow`-form Minkowski norm).
    pub fn compute_dkl_jod_host_pool(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<f32> {
        self._dispatch_d_bands_into_scratch(ref_srgb, dist_srgb, ppd)?;
        self._host_pool_and_finalize_jod()
    }

    /// Warm-reference companion to [`Cvvdp::compute_dkl_jod_host_pool`].
    ///
    /// Same algorithm and same JOD output as
    /// [`Cvvdp::compute_dkl_jod_with_warm_ref`] but pools the per-band
    /// D values on the host instead of via the GPU atomic kernel —
    /// runs on every cubecl runtime, including `cubecl-cpu`. Useful
    /// for batch CPU scoring (one warm REF, many DIST candidates)
    /// where the GPU pool path isn't available.
    ///
    /// Same `Error::NoWarmReference` semantics as the GPU warm-ref
    /// variant: requires a prior [`Cvvdp::warm_reference`] call, and
    /// any intervening REF-dispatching method invalidates the warm
    /// state.
    pub fn compute_dkl_jod_host_pool_with_warm_ref(
        &mut self,
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<f32> {
        let log_l_bkg_baseband = self
            .warm_ref_baseband_log_l_bkg
            .ok_or(Error::NoWarmReference)?;
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband, ppd)?;
        self._host_pool_and_finalize_jod()
    }

    /// Host-side spatial pool + 3-stage Minkowski fold over the D
    /// planes resident in `self.d_scratch[k].d[c]`. Used by both
    /// `compute_dkl_jod_host_pool` and `compute_dkl_jod_host_pool_with_warm_ref`
    /// — the dispatch path that landed the D bands differs, but
    /// the pool tail is identical.
    fn _host_pool_and_finalize_jod(&mut self) -> Result<f32> {
        let n_levels = self.n_levels as usize;
        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            for c in 0..N_CHANNELS {
                let bytes = self
                    .client
                    .read_one(self.d_scratch[k].d[c].clone())
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
    /// the JOD pipeline, halving the GPU compute per DIST candidate.
    ///
    /// Any call to [`Cvvdp::compute_dkl_jod`],
    /// [`Cvvdp::compute_dkl_d_bands`],
    /// [`Cvvdp::compute_dkl_weber_pyramid`], or
    /// [`Cvvdp::compute_dkl_t_p_bands`] invalidates the warm state
    /// (their REF dispatches overwrite the shared GPU scratch). Call
    /// `warm_reference` again to re-arm.
    ///
    /// Validates that `ref_srgb.len() == width × height × 3`.
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
        Ok(())
    }

    /// Score a DIST candidate against the GPU-warmed REF. Same JOD
    /// output as [`Cvvdp::compute_dkl_jod`] but skips the REF weber
    /// pyramid — useful for batch workflows where one reference
    /// is scored against many distorted candidates (codec quality
    /// sweeps, fixture-based testing).
    ///
    /// Returns [`Error::NoWarmReference`] if `warm_reference` was
    /// not called, or if the warm state was invalidated by an
    /// intervening REF-dispatching method.
    pub fn compute_dkl_jod_with_warm_ref(
        &mut self,
        dist_srgb: &[u8],
        ppd: f32,
    ) -> Result<f32> {
        let log_l_bkg_baseband = self
            .warm_ref_baseband_log_l_bkg
            .ok_or(Error::NoWarmReference)?;
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }
        self._dispatch_d_bands_dist_and_band_loop(dist_srgb, log_l_bkg_baseband, ppd)?;
        self._pool_and_finalize_jod()
    }

    /// GPU pool + host fold for the per-band D planes resident in
    /// `self.d_scratch[k].d[c]`. Used by both `compute_dkl_jod` and
    /// `compute_dkl_jod_with_warm_ref` — the dispatch path that
    /// landed the bands differs, but the pool/fold tail is identical.
    fn _pool_and_finalize_jod(&mut self) -> Result<f32> {
        let n_levels = self.n_levels as usize;
        let partials_init = [0.0_f32; MAX_LEVELS * N_CHANNELS];
        let partials_h = self
            .client
            .create_from_slice(f32::as_bytes(&partials_init[..n_levels * N_CHANNELS]));
        let cube_dim = CubeDim::new_1d(64);
        for k in 0..n_levels {
            let (_, _, n_px) = self.level_dims(k);
            let count = CubeCount::Static((n_px as u32).div_ceil(64), 1, 1);
            let d_a = self.d_scratch[k].d[0].clone();
            let d_rg = self.d_scratch[k].d[1].clone();
            let d_vy = self.d_scratch[k].d[2].clone();
            let partial_idx_a = (k * N_CHANNELS) as u32;
            let partial_idx_rg = (k * N_CHANNELS + 1) as u32;
            let partial_idx_vy = (k * N_CHANNELS + 2) as u32;
            unsafe {
                pool_band_3ch_kernel::launch::<R>(
                    &self.client,
                    count.clone(),
                    cube_dim,
                    ArrayArg::from_raw_parts(d_a, n_px),
                    ArrayArg::from_raw_parts(d_rg, n_px),
                    ArrayArg::from_raw_parts(d_vy, n_px),
                    ArrayArg::from_raw_parts(partials_h.clone(), n_levels * N_CHANNELS),
                    BETA_SPATIAL,
                    partial_idx_a,
                    partial_idx_rg,
                    partial_idx_vy,
                    n_px as u32,
                );
            }
        }

        let bytes = self
            .client
            .read_one(partials_h)
            .map_err(|_| Error::InvalidImageSize)?;
        let partials_data: &[f32] = f32::from_bytes(&bytes);

        let mut q_per_ch: Vec<[f32; 3]> = Vec::with_capacity(n_levels);
        for k in 0..n_levels {
            let (_, _, n_px_k) = self.level_dims(k);
            let mut q = [0.0_f32; 3];
            for c in 0..N_CHANNELS {
                q[c] = pool_band_finalize(
                    partials_data[k * N_CHANNELS + c],
                    n_px_k,
                    BETA_SPATIAL,
                );
            }
            q_per_ch.push(q);
        }

        Ok(do_pooling_and_jod_still_3ch(&q_per_ch))
    }

    /// Run color + Laplacian-pyramid + per-band CSF weighting.
    ///
    /// `ppd` is pixels-per-degree (from `DisplayGeometry::pixels_per_degree()`).
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
    pub fn compute_dkl_csf_weighted_bands(
        &mut self,
        srgb: &[u8],
        ppd: f32,
        l_bkg: f32,
    ) -> Result<Vec<[Vec<f32>; 3]>> {
        // Leaves the un-weighted Laplacian bands in
        // self.bands_ref[k].planes[c]. Uses the dispatch-only helper
        // so we don't pay for a full-pyramid host readback we'd
        // immediately discard.
        self._dispatch_laplacian_pyramid_gpu(srgb)?;

        let weights_per_level =
            precomputed_band_weights(ppd, self.width as usize, self.height as usize, l_bkg);
        let flat_weights = flatten_band_weights(&weights_per_level);
        let weights_handle = self.client.create_from_slice(f32::as_bytes(&flat_weights));

        let cube_dim = CubeDim::new_1d(64);
        let n_levels = self.n_levels as usize;
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
    /// Currently routes through the parity-locked host scalar
    /// (`host_scalar::predict_jod_still_3ch`). The full GPU
    /// composition path is implemented and parity-tested as
    /// [`Cvvdp::compute_dkl_jod`] (color → pyramid → CSF → masking →
    /// `pool_band_kernel` → host fold); `score` will retarget once
    /// the v1 R2 manifest parity is held by the GPU path through a
    /// `shadow_jod`-style anchor.
    ///
    /// Score matches pycvvdp v0.5.4 on the v1 R2 manifest within
    /// 0.006 JOD across q1–q90.
    ///
    /// The viewing geometry comes from `self.geometry` — set via
    /// `Cvvdp::new_with_geometry` or defaulted to STANDARD_4K by
    /// `Cvvdp::new`.
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
        let ppd = self.geometry.pixels_per_degree();
        let jod = crate::host_scalar::predict_jod_still_3ch(
            reference_srgb,
            distorted_srgb,
            self.width as usize,
            self.height as usize,
            self.params.display,
            ppd,
        );
        Ok(jod as f64)
    }

    /// Cache the reference side for repeated scoring against many
    /// distorted candidates.
    ///
    /// Today this just stashes the sRGB bytes (the host-scalar path
    /// re-runs the reference side per call); the planned GPU
    /// composition will materialise the CSF-weighted pyramid here so
    /// the reference work happens once per `set_reference`.
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
    /// Matches `score(ref, dist)` exactly — the fast path lands when
    /// GPU composition stops re-running the reference side.
    pub fn score_with_reference(&mut self, distorted_srgb: &[u8]) -> Result<f64> {
        let cached = self.cached.as_ref().ok_or(Error::NoCachedReference)?;
        let expected = (self.width as usize) * (self.height as usize) * 3;
        if distorted_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: distorted_srgb.len(),
            });
        }
        let ppd = self.geometry.pixels_per_degree();
        let jod = crate::host_scalar::predict_jod_still_3ch(
            &cached.ref_srgb,
            distorted_srgb,
            self.width as usize,
            self.height as usize,
            self.params.display,
            ppd,
        );
        Ok(jod as f64)
    }
}
