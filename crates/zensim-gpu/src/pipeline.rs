//! zensim pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` into the 4-scale zensim feature
//! extractor.
//!
//! Public entry points:
//! - [`Zensim::new`] + [`Zensim::compute_features`] — extract the
//!   228-feature vector from one (ref, dist) pair.
//! - [`Zensim::set_reference`] + [`Zensim::compute_with_reference`] —
//!   cache the reference-side pyramid and score many distorted images
//!   against it (encoder-loop friendly).
//!
//! ## Algorithm (per scale, faithful to `zensim-cuda`'s `compute_features`)
//!
//! 1. sRGB packed-u8 → planar positive-XYB (Halley `cbrtf_fast`).
//! 2. Mirror-fill SIMD-padding columns `[logical_w..padded_w)`.
//! 3. 2× planar downscale to build the pyramid (4 levels).
//! 4. Per scale, per channel:
//!    a. Fused H-blur (mu1, mu2, sigma_sq, sigma12).
//!    b. Fused V-blur + per-pixel features → 17 f64 + 3 f32 per column.
//!    c. Host-side fold across columns → per-channel feature row.
//! 5. Pack into the 228-entry vector matching CPU layout (basic block
//!    of 156 + peaks block of 72).
//!
//! ## Buffer layout
//!
//! Buffers are flat `padded_w × height` planar f32 arrays — no
//! pitched-2D padding. CPU zensim doesn't depend on alignment within
//! the row beyond the SIMD-pad columns we explicitly emit, so flat
//! storage matches its math without translation.

use cubecl::prelude::*;

use crate::kernels::{self, blit, color, diffmap, downscale, fused, masked_iw_strip, reduce};
use zensim::{
    DiffmapOptions, PrecomputedReference, Zensim as ZensimCpu, ZensimError, ZensimProfile,
};
// `masked_iw` retained for back-compat in tests; not used in the
// production extended path.
#[allow(unused_imports)]
use crate::kernels::masked_iw;
use crate::{
    Error, FEATURES_PER_CHANNEL_BASIC, FEATURES_PER_CHANNEL_IW, FEATURES_PER_CHANNEL_MASKED,
    FEATURES_PER_CHANNEL_PEAKS, Result, SCALES, TOTAL_FEATURES, ZensimFeatureRegime,
    simd_padded_width,
};

// `logical_w` and `partials_*_per_scale` are bookkeeping kept for
// future debug tooling (per-channel intermediate dump). The
// pre-fused-kernel `h_mu1..h_sigma12` H-blur scratch planes were
// removed in T_z.B (2026-05-16): the tile-fused `fused_features_kernel`
// allocates its working set in shared memory, so 12 padded-f32 planes
// per scale (~576 MB of zero-fill traffic at 12 MP) were dead weight.
#[allow(dead_code)]
struct Scale {
    logical_w: u32,
    padded_w: u32,
    h: u32,
    n_padded: usize,
    n_strips: u32,
    /// Strip count for the **extended (masked + IW) kernel**. Differs
    /// from `n_strips` (which is tuned for GPU occupancy of the basic
    /// kernel) because the masked path reproduces CPU's CPU-shaped
    /// strip layout for parity. Equals `ceil(h / STRIP_INNER)` from
    /// [`kernels::masked_iw_strip::cpu_strip_count`].
    n_strips_ext: u32,

    /// Three planar XYB planes per side at `padded_w × h`. Allocated
    /// `empty()` — `srgb_to_positive_xyb_kernel` writes every pixel
    /// in `[0, padded_w) × [0, h)` (including the mirror-padded
    /// columns) so zero-fill on the host side is unnecessary.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Mirror-offset table (one u32 per padding column). `None` when
    /// `padded_w == logical_w`.
    mirror_offsets: Option<cubecl::server::Handle>,
    pad_count: u32,

    /// Offset (in f64 / f32 units) of this scale's partials within the
    /// big shared `partials_*` buffers. Layout per scale:
    /// `[ch0 strip0 col0 .. col(pw-1) | ch0 strip1 ... | ch1 strip0 ... | ...]`
    /// with 17 f64 (or 3 f32) per slot.
    partials_f64_off: usize,
    partials_max_off: usize,
    partials_f64_per_scale: usize, // = pw × n_strips × 3 channels × 17
    partials_max_per_scale: usize, // = pw × n_strips × 3 channels × 3

    /// Offset of this scale's masked + IW partials within the
    /// `partials_ext_f64` buffer. `0` (unused) when regime == Basic.
    partials_ext_off: usize,
    partials_ext_per_scale: usize, // = pw × n_strips_ext × 3 channels × 12
}

/// Allocate an uninitialised f32 plane on-device. Use only when the
/// caller writes every element before the next kernel reads any —
/// the fused features pipeline matches that contract for every plane
/// (xyb produced by `srgb_to_positive_xyb_kernel`, downscale outputs
/// produced by `downscale_2x_3ch_kernel`, partials overwritten by
/// `fused_features_kernel`'s per-thread store).
fn alloc_empty_f32<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.empty(n * core::mem::size_of::<f32>())
}
fn alloc_empty_f64<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.empty(n * core::mem::size_of::<f64>())
}
/// Choose a per-scale strip count to keep V-blur GPU-occupied at all
/// resolutions. The kernel's parallelism is `padded_w × n_strips × 3
/// channels`. RTX-5070-class GPUs want ≥ 16 K resident threads to
/// hide latency.
fn pick_n_strips(padded_w: u32, height: u32) -> u32 {
    if height <= 64 {
        1
    } else if height >= 1024 {
        8
    } else if padded_w >= 256 {
        4
    } else {
        2
    }
}

impl Scale {
    fn new<R: Runtime>(
        client: &ComputeClient<R>,
        logical_w: u32,
        padded_w: u32,
        h: u32,
        partials_f64_off: usize,
        partials_max_off: usize,
        partials_ext_off: usize,
    ) -> Self {
        let n = (padded_w as usize) * (h as usize);
        let alloc3_empty = || -> [cubecl::server::Handle; 3] {
            [
                alloc_empty_f32(client, n),
                alloc_empty_f32(client, n),
                alloc_empty_f32(client, n),
            ]
        };
        let pad_count = padded_w - logical_w;
        let n_strips = pick_n_strips(padded_w, h);
        // Extended kernel uses CPU's strip layout for parity.
        let n_strips_ext = kernels::masked_iw_strip::cpu_strip_count(h);

        // Mirror-offset table matching CPU zensim
        // (streaming.rs:591-601):
        //   period = 2 * (logical_w - 1)
        //   for i in 0..pad_count:
        //     m = (logical_w + i) % period
        //     offset = if m < logical_w { m } else { period - m }
        let mirror_offsets = if pad_count > 0 {
            let lw = logical_w as usize;
            let pc = pad_count as usize;
            let period = 2 * (lw - 1);
            let host: Vec<u32> = (0..pc)
                .map(|i| {
                    let m = (lw + i) % period;
                    let off = if m < lw { m } else { period - m };
                    off as u32
                })
                .collect();
            Some(client.create_from_slice(u32::as_bytes(&host)))
        } else {
            None
        };

        Self {
            logical_w,
            padded_w,
            h,
            n_padded: n,
            ref_xyb: alloc3_empty(),
            dis_xyb: alloc3_empty(),
            mirror_offsets,
            pad_count,
            partials_f64_off,
            partials_max_off,
            partials_f64_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 17,
            partials_max_per_scale: (padded_w as usize) * (n_strips as usize) * 3 * 3,
            partials_ext_off,
            partials_ext_per_scale: (padded_w as usize) * (n_strips_ext as usize) * 3 * 12,
            n_strips,
            n_strips_ext,
        }
    }
}

/// Strip-mode state. Present when the pipeline was constructed via
/// [`Zensim::new_strip`] / [`Zensim::new_strip_with_halo`].
///
/// `image_h` is the full image height the caller intends to score;
/// per-scale buffers (in `Zensim::scales`) are sized for
/// `strip_alloc_h = h_body + 2 × halo` instead of `image_h`, so peak
/// working set drops from `O(image_h)` to `O(strip_alloc_h)`.
#[derive(Debug, Clone, Copy)]
struct StripState {
    /// Full source image height (rows). Equal to `Zensim::height` in
    /// strip mode — kept here for clarity at the strip-walker call
    /// sites that need the image-level dimension distinct from the
    /// per-scale strip allocation.
    #[allow(dead_code)]
    image_h: u32,
    /// Body rows per strip at scale 0 (each strip's contribution to
    /// the per-feature scalar sum). Multiple of [`STRIP_ALIGN`].
    h_body: u32,
    /// Halo rows per side at scale 0 (image rows pulled in past the
    /// body region for V-blur reach + cross-scale halo). Multiple of
    /// [`STRIP_ALIGN`].
    halo: u32,
    /// Maximum strip allocation height at scale 0 = `h_body + 2 × halo`.
    /// All per-scale buffers are sized to `strip_alloc_h / 2^s`.
    strip_alloc_h: u32,
}

impl StripState {
    /// Yield `(body_start, body_end, upload_start, upload_end)` for
    /// each strip, all in scale-0 image rows. `upload_*` is the actual
    /// region uploaded to the GPU (body region expanded by halo on
    /// each side, clamped to image bounds).
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

/// Per-scale full-image cached reference XYB pyramid (Mode E
/// refinement; task #75, 2026-05-26). Populated by [`Zensim::set_reference`]
/// when in strip mode AND the device cache is enabled (default).
///
/// The strip walker installs a row-range slice from these full-image
/// XYB planes into the strip-sized `Scale.ref_xyb` buffers via the
/// `copy_rows_kernel` blit, skipping per-strip ref re-upload + ref
/// xyb pyramid rebuild on every `compute_with_reference` call.
///
/// Memory cost: 3 channels × `Σ pyramid_pixels_at_full_h` × 4 bytes.
/// For 4096² Basic that's ~130 MB on top of strip-mode's ~290 MB —
/// still within typical VRAM caps. Callers who need to opt out (very
/// tight VRAM, few dist iterations) can use
/// [`Zensim::set_reference_host_cached_only`] instead.
struct RefFullXybState {
    /// Per-scale `(padded_w, full_h)` dims. `padded_w[s]` matches
    /// `Scale::padded_w[s]` exactly; `full_h[s]` is the FULL image
    /// height halved per scale (= what Full-mode would allocate).
    dims: Vec<(u32, u32)>,
    /// Per-scale 3-channel XYB planes, each of length
    /// `padded_w[s] × full_h[s]`. Filled by running the standard
    /// `srgb_to_positive_xyb_kernel` + `downscale_2x_3ch_kernel`
    /// chain on full-image-sized scratch buffers.
    xyb: Vec<[cubecl::server::Handle; 3]>,
}

/// Pyramid alignment factor: must divide `h_body` and `halo` cleanly so
/// the strip's pyramid halves to integer row counts at every scale.
/// `2^(SCALES - 1)` = 2^3 = 8 for zensim (4 scales).
pub const STRIP_ALIGN: u32 = 1u32 << (SCALES as u32 - 1);

/// Default halo rows per side at scale 0 = 40 (multiple of [`STRIP_ALIGN`]).
/// At the smallest scale (scale 3) this shrinks to `40 / 8 = 5` rows —
/// just enough to cover the 11×11 valid-blur radius `R = 5`. Picking a
/// smaller halo would mean the V-blur at body rows of the smallest
/// scale would invoke the strip-local mirror (introducing artifacts
/// vs the full-image path at strip boundaries). 40 is the minimum that
/// reproduces full-image V-blur at every scale.
pub const STRIP_DEFAULT_HALO: u32 = 40;

/// Default body rows per strip at scale 0 = 256 (multiple of
/// [`STRIP_ALIGN`]). Chosen as a balance between GPU occupancy
/// (larger body = more parallelism per launch) and memory savings
/// (smaller body = lower peak working set). Tunable via
/// [`Zensim::new_strip_with_halo`].
pub const STRIP_DEFAULT_BODY: u32 = 256;

/// One per-resolution zensim pipeline. Allocate once with
/// [`Zensim::new`] for a (width, height); reuse across many image pairs
/// of that resolution.
pub struct Zensim<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    pixels: usize,

    /// `Some(_)` when constructed via [`Zensim::new_strip`] — every
    /// per-scale buffer in `scales` is sized for `strip_alloc_h`
    /// rather than the full image height.
    strip: Option<StripState>,
    /// Host-side cached reference sRGB bytes for strip mode. Filled
    /// by `set_reference` in strip mode whenever the device-cache
    /// path is disabled. Empty when `ref_full_xyb` is `Some` (the
    /// device cache supersedes the host cache).
    ///
    /// Kept as a fallback for callers that opt out of the device
    /// cache via [`Zensim::set_reference_host_cached_only`]; the
    /// strip walker uses host bytes when `ref_full_xyb` is `None`.
    cached_ref_strip_srgb: Vec<u8>,

    /// Per-scale full-image ref XYB pyramid cached on device. Populated
    /// by [`Zensim::set_reference`] in strip mode (task #75 mode-E
    /// refinement); `None` in Full mode and on the host-cached-only
    /// strip fallback.
    ref_full_xyb: Option<RefFullXybState>,

    /// sRGB-u8 staging buffer sized for the full image (image_w ×
    /// image_h). Only allocated in strip mode when the device-cache
    /// `set_reference` path is used; lives behind an `Option` to
    /// avoid the alloc cost on Full mode + on strip mode callers
    /// who never set a reference. Reused across set_reference calls.
    src_u8_full: Option<cubecl::server::Handle>,
    /// Persistent host pack scratch for `src_u8_full` (one u32 per
    /// pixel). Empty Vec when the device cache isn't in use.
    pack_scratch_full: Vec<u32>,

    /// Persistent host-side packing scratch (one u32 per pixel = R |
    /// G<<8 | B<<16). Reused across uploads to avoid the alloc + iter
    /// per `compute_with_reference`.
    pack_scratch: Vec<u32>,

    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    /// Device copy of `color::SRGB8_TO_LINEARF32_LUT`.
    srgb_lut: cubecl::server::Handle,

    scales: Vec<Scale>,

    /// Persistent partials buffers. Sized to fit all (scale, channel)
    /// per-column slots. Avoids the per-channel alloc-then-read churn
    /// that dominated the warm-path cost in the original pipeline.
    partials_f64: cubecl::server::Handle,
    partials_max: cubecl::server::Handle,
    partials_f64_len: usize,
    partials_max_len: usize,

    /// Final per-(scale, channel, slot) sums after the on-device
    /// reduction pass — small enough that host read-back is sub-µs.
    finals_f64: cubecl::server::Handle,
    finals_max: cubecl::server::Handle,

    has_reference: bool,

    // ───────── Diffmap + linear-planes API (Phase 1) ─────────
    /// Lazy CPU diffmap state. Allocated on first call to a diffmap or
    /// linear-planes entry-point (so callers using only the scalar
    /// feature API pay zero cost).
    ///
    /// In Phase 1 the per-pixel diffmap is delegated to the canonical
    /// CPU `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`
    /// path (see `crate::kernels::diffmap` module docs +
    /// `docs/DIFFMAP_DIVERGENCES.md` for rationale). This state caches
    /// the CPU `Zensim` driver + a `PrecomputedReference` so warm-ref
    /// callers don't pay the ref-XYB pyramid build per iter.
    diffmap_state: Option<DiffmapState>,

    // ───────── Extended / WithIw regime support ─────────
    regime: ZensimFeatureRegime,
    /// Per-scale per-channel mu1/mu2/ssq/s12 persist planes — laid out
    /// `[ch0_pixels | ch1_pixels | ch2_pixels]` per scale per side. One
    /// pair (ref + dist) per scale; only the ref-side mu1 is consumed
    /// by the masked-IW activity blur, but we persist all four for
    /// matching CPU's masked SSIM math (mu1 + mu2 + ssq + s12 at the
    /// SAME pixel). Each entry is `pad_total × 3` f32. Empty Vec on
    /// Basic regime — zero memory cost on the fast path.
    ///
    /// Indexed by `scales[s].partials_ext_off` is misleading — we
    /// instead keep per-scale per-channel handles directly because the
    /// pad_total varies per scale.
    persist_planes_ref: Vec<[cubecl::server::Handle; 4]>,

    /// Masked + IW per-(col, strip, ch) partials buffer. Length =
    /// Σ per-scale (pw × n_strips × 3 × 12). Empty handle on Basic.
    partials_ext_f64: cubecl::server::Handle,
    partials_ext_f64_len: usize,

    /// Reduced masked + IW finals: per-(scale, channel, slot in [0,12)).
    finals_ext_f64: cubecl::server::Handle,

    /// Lazy **pure-GPU diffmap scratch** (Phase 1b). Allocated on first
    /// call to a GPU-native diffmap path. Holds per-scale mu1/mu2/ssq/s12
    /// persist planes (independent of the outer `regime` — the diffmap
    /// path always needs them even on Basic) plus the base-resolution
    /// upsample accumulator + the trimmed output buffer + the linear-RGB
    /// upload planes. See [`GpuDiffmapScratch`] + the
    /// `gpu_diffmap_*` methods.
    gpu_diffmap_scratch: Option<GpuDiffmapScratch<R>>,
}

/// Pure-GPU diffmap scratch (Phase 1b). Lazily allocated on first GPU
/// diffmap call; reused across calls so the warm buttloop pays the
/// alloc once.
///
/// Strategy: an inner WithIw-regime [`Zensim<R>`] runs the full
/// 372-feature GPU pipeline (which also writes the per-scale
/// mu1/mu2/ssq/s12 persist planes the diffmap kernels need). The
/// 372 features feed the CPU V0_3 MLP for the scalar score (proven
/// bit-equivalent to the CPU `score_features_with_profile` path by
/// `tests/opaque_default_weights_v03.rs`); the persist planes feed
/// the GPU diffmap kernel chain ([`diffmap::per_scale_weighted_ssim_kernel`]
/// → [`diffmap::pow2x_upsample_add_kernel`] → [`diffmap::diffmap_trim_padded_kernel`]).
///
/// All buffers are sized from the inner `Zensim`'s `scales` geometry
/// at construction time — the diffmap path never resizes them.
struct GpuDiffmapScratch<R: Runtime> {
    /// Inner WithIw-regime pipeline. Owns the XYB pyramid + persist
    /// planes + 372-feature reduction. Boxed to keep `Zensim<R>`'s own
    /// size bounded (it's stored behind an `Option` field on `Zensim`).
    inner: Box<Zensim<R>>,
    /// Per-scale weighted-SSIM plane (`pad_total × 1` f32) — output of
    /// [`diffmap::per_scale_weighted_ssim_kernel`] for one scale,
    /// consumed by [`diffmap::pow2x_upsample_add_kernel`].
    scale_dm: Vec<cubecl::server::Handle>,
    /// Base-resolution upsample accumulator (`padded_w0 × height` f32).
    acc: cubecl::server::Handle,
    /// Trimmed `width × height` f32 output buffer (drops pad columns).
    out: cubecl::server::Handle,
    /// Distorted-side linear-RGB upload planes (`width × height` each).
    /// Filled host-side from the caller's planes, uploaded once per
    /// distorted candidate.
    dist_lin: [cubecl::server::Handle; 3],
    /// Reference-side linear-RGB upload planes — used by the one-shot
    /// / cold path to seed the inner pipeline's reference XYB pyramid.
    ref_lin: [cubecl::server::Handle; 3],
    /// Per-scale (per-channel SSIM weight) and per-scale blend weight
    /// for the default `DiffmapOptions` Trained path, precomputed once
    /// from [`diffmap::trained_multiscale_ssim_weights_default`].
    /// Cached so the warm loop doesn't recompute the f64 reduction.
    per_scale_w: Vec<[f32; 3]>,
    scale_blend: Vec<f32>,
    /// CPU scorer bound to V0_3. The scalar score comes from the CPU
    /// canonical `compute_with_ref_and_diffmap_linear_planar` path
    /// (byte-exact to Phase 1) — NOT from the GPU-feature → MLP path,
    /// which is catastrophically wrong on the pinned zensim 0.3.0 (a
    /// documented pre-existing WithIw-feature/V0_3-MLP parity bug; see
    /// `docs/DIFFMAP_DIVERGENCES.md` §9 + §2 and the Phase 1 memo).
    /// The GPU produces the diffmap; the CPU produces the score.
    cpu_scorer: ZensimCpu,
    /// Cached CPU `PrecomputedReference` for the warm-ref score path.
    cpu_ref: Option<PrecomputedReference>,
    /// Profile the score is computed against.
    profile: ZensimProfile,
}

impl<R: Runtime> Zensim<R> {
    /// Allocate every per-resolution buffer up front. `width` and
    /// `height` must each be ≥ 8 — zensim's pyramid collapses below
    /// that. Default regime is [`ZensimFeatureRegime::Basic`] (228
    /// features) — backwards-compatible with the pre-372 GPU output.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        Self::new_with_regime(client, width, height, ZensimFeatureRegime::Basic)
    }

    /// Unified [`MemoryMode`](crate::MemoryMode) constructor.
    ///
    /// - `Full` allocates the whole-image pipeline.
    /// - `Strip { h_body }` allocates a strip-walker pipeline with
    ///   `h_body` body rows per strip (defaults to
    ///   [`STRIP_DEFAULT_BODY`] when `None`).
    /// - `Tile { .. }` returns [`Error::ModeUnsupported`] (not
    ///   implemented).
    /// - `Auto` picks Full when it fits the VRAM cap, else falls back
    ///   to Strip with an auto-sized body. Returns
    ///   [`Error::TooBigForFull`] when neither fits.
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
                    crate::memory_mode::auto_strip_body_for(
                        width,
                        height,
                        ZensimFeatureRegime::Basic,
                        cap,
                    )
                });
                Self::new_strip(client, width, height, body)
            }
            MemoryMode::Tile { .. } => Err(crate::Error::ModeUnsupported("Tile")),
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                // This entry point defaults to Basic regime (see `new()`).
                match resolve_auto(width, height, ZensimFeatureRegime::Basic, cap)? {
                    ResolvedMode::Full => Self::new(client, width, height),
                    ResolvedMode::Strip { h_body } => {
                        Self::new_strip(client, width, height, h_body)
                    }
                }
            }
        }
    }

    /// Construct a strip-mode pipeline for an `image_w × image_h`
    /// image, with each strip carrying `h_body` body rows + the default
    /// halo per side ([`STRIP_DEFAULT_HALO`]). Per-scale GPU buffers
    /// are sized for a single strip of `h_body + 2 × halo` rows, not
    /// the full image — peak working set drops from `O(image_h)` to
    /// `O(strip_alloc_h)`.
    ///
    /// `h_body` and the halo MUST be multiples of [`STRIP_ALIGN`] (=8,
    /// the pyramid alignment factor for 4 scales). [`STRIP_DEFAULT_BODY`]
    /// (256) satisfies this.
    ///
    /// Default regime is [`ZensimFeatureRegime::Basic`]. Use
    /// [`Self::new_strip_with_halo_and_regime`] for explicit regime
    /// control.
    pub fn new_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
    ) -> Result<Self> {
        Self::new_strip_with_halo_and_regime(
            client,
            image_w,
            image_h,
            h_body,
            STRIP_DEFAULT_HALO,
            ZensimFeatureRegime::Basic,
        )
    }

    /// Strip-mode constructor with explicit halo and feature regime.
    /// Both `h_body` and `halo` must be multiples of [`STRIP_ALIGN`]
    /// (=8). `halo` must be ≥ 40 (= 5 × 8) so the smallest scale's
    /// halo covers the 11×11 V-blur radius — smaller halos invoke the
    /// strip-local mirror at body rows of scale 3 and produce wrong
    /// V-blur outputs there.
    pub fn new_strip_with_halo_and_regime(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
        halo: u32,
        regime: ZensimFeatureRegime,
    ) -> Result<Self> {
        if image_w < 8 || image_h < 8 {
            return Err(Error::InvalidImageSize);
        }
        if h_body == 0 || !h_body.is_multiple_of(STRIP_ALIGN) {
            return Err(Error::InvalidImageSize);
        }
        if halo < STRIP_DEFAULT_HALO || !halo.is_multiple_of(STRIP_ALIGN) {
            return Err(Error::InvalidImageSize);
        }
        let strip_alloc_h = h_body + 2 * halo;
        if strip_alloc_h < 8 {
            return Err(Error::InvalidImageSize);
        }
        let strip = StripState {
            image_h,
            h_body,
            halo,
            strip_alloc_h,
        };
        Self::new_with_regime_strip_budget(
            client,
            image_w,
            image_h,
            regime,
            usize::MAX,
            Some(strip),
        )
    }

    /// True if this pipeline was constructed via [`Self::new_strip`].
    pub fn is_strip_mode(&self) -> bool {
        self.strip.is_some()
    }

    /// Construct with an explicit feature regime.
    ///
    /// `regime == Extended` adds the 72 masked features (`228..300`).
    /// `regime == WithIw` adds the 72 IW features on top (`300..372`).
    /// See [`ZensimFeatureRegime`] for the slot map.
    ///
    /// **Memory cost**: Extended / WithIw both allocate 4 persist
    /// planes × 3 channels × 2 sides (ref/dis is needed for the IW
    /// activity that uses `src - mu1` so we hold mu1 on the *ref* side
    /// only — see `launch_masked_iw`). The plane footprint is dominated
    /// by scale 0; at 12 MP that's ~600 MB. Use
    /// [`Zensim::new_with_regime_budget`] to fail fast when the budget
    /// is unacceptable.
    pub fn new_with_regime(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        regime: ZensimFeatureRegime,
    ) -> Result<Self> {
        // usize::MAX disables the budget gate; if you care, use
        // `new_with_regime_budget` directly.
        Self::new_with_regime_budget(client, width, height, regime, usize::MAX)
    }

    /// Construct with an explicit feature regime AND an explicit cap on
    /// the extended-regime persist-plane memory footprint (in bytes).
    /// Returns [`Error::ExtendedPlaneBudgetExceeded`] if the regime
    /// requires more than `max_extended_plane_bytes`. The cap is
    /// ignored on `Basic`.
    pub fn new_with_regime_budget(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        regime: ZensimFeatureRegime,
        max_extended_plane_bytes: usize,
    ) -> Result<Self> {
        Self::new_with_regime_strip_budget(
            client,
            width,
            height,
            regime,
            max_extended_plane_bytes,
            None,
        )
    }

    /// Like [`Self::new_with_regime_budget`] but accepts an optional
    /// strip-mode descriptor. When `strip` is `Some`, per-scale buffers
    /// are sized for `strip_alloc_h = h_body + 2 × halo` instead of the
    /// full image height; the pipeline then uses strip-walker variants
    /// of every `compute_*` entry point. Internal helper backing
    /// [`Self::new_strip_with_halo_and_regime`] — public callers should
    /// use the strip-specific constructors.
    fn new_with_regime_strip_budget(
        client: ComputeClient<R>,
        width: u32,
        height: u32,
        regime: ZensimFeatureRegime,
        max_extended_plane_bytes: usize,
        strip: Option<StripState>,
    ) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let pixels = (width as usize) * (height as usize);

        // Per-scale allocation height = strip_alloc_h in strip mode,
        // image height otherwise. The kernel's mirror semantics work
        // on whatever buffer height we configure (mirror is *within*
        // the buffer — strip mode supplies image-correct halo rows).
        let effective_h = strip.map(|s| s.strip_alloc_h).unwrap_or(height);

        let mut scales = Vec::with_capacity(SCALES);
        let mut logical_w = width;
        let mut padded_w = simd_padded_width(width as usize) as u32;
        let mut h = effective_h;
        let mut plan: Vec<(u32, u32, u32)> = Vec::with_capacity(SCALES);
        for _ in 0..SCALES {
            if logical_w < 8 || h < 8 {
                break;
            }
            plan.push((logical_w, padded_w, h));
            logical_w = logical_w.div_ceil(2);
            padded_w /= 2;
            h = h.div_ceil(2);
        }
        let mut partials_f64_total: usize = 0;
        let mut partials_max_total: usize = 0;
        let mut partials_ext_total: usize = 0;
        for &(_, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            let ns_ext = kernels::masked_iw_strip::cpu_strip_count(ph) as usize;
            partials_f64_total += (pw as usize) * ns * 3 * 17;
            partials_max_total += (pw as usize) * ns * 3 * 3;
            partials_ext_total += (pw as usize) * ns_ext * 3 * 12;
        }
        let mut f64_off: usize = 0;
        let mut max_off: usize = 0;
        let mut ext_off: usize = 0;
        for &(lw, pw, ph) in &plan {
            let ns = pick_n_strips(pw, ph) as usize;
            let ns_ext = kernels::masked_iw_strip::cpu_strip_count(ph) as usize;
            scales.push(Scale::new(&client, lw, pw, ph, f64_off, max_off, ext_off));
            f64_off += (pw as usize) * ns * 3 * 17;
            max_off += (pw as usize) * ns * 3 * 3;
            ext_off += (pw as usize) * ns_ext * 3 * 12;
        }

        // Budget check: 4 planes × 3 channels × padded_pixels × 4 bytes
        // per scale (ref side only — see persist_planes_ref docstring).
        let needs_planes = regime.needs_extended_kernel();
        let extended_plane_bytes: usize = if needs_planes {
            plan.iter()
                .map(|&(_, pw, ph)| (pw as usize) * (ph as usize) * 3 * 4 * 4)
                .sum()
        } else {
            0
        };
        if needs_planes && extended_plane_bytes > max_extended_plane_bytes {
            return Err(Error::ExtendedPlaneBudgetExceeded {
                needed_bytes: extended_plane_bytes,
                max_bytes: max_extended_plane_bytes,
            });
        }

        // src_u8 buffer size: in strip mode we only ever upload one strip
        // at a time (strip_alloc_h rows × image width), so size for the
        // strip rather than the full image.
        let upload_pixels = if let Some(s) = strip {
            (width as usize) * (s.strip_alloc_h as usize)
        } else {
            pixels
        };
        let src_u8_a = client.empty(upload_pixels * core::mem::size_of::<u32>());
        let src_u8_b = client.empty(upload_pixels * core::mem::size_of::<u32>());

        let srgb_lut = client.create_from_slice(f32::as_bytes(
            &crate::kernels::color::SRGB8_TO_LINEARF32_LUT,
        ));

        let partials_f64 = alloc_empty_f64(&client, partials_f64_total);
        let partials_max = alloc_empty_f32(&client, partials_max_total);
        let n_finals_f64 = scales.len() * 3 * 17;
        let n_finals_max = scales.len() * 3 * 3;
        let finals_f64 = alloc_empty_f64(&client, n_finals_f64);
        let finals_max = alloc_empty_f32(&client, n_finals_max);

        // Extended regime allocations — only when needed. The persist
        // planes layout per scale is `[ch0 | ch1 | ch2]` flat, with
        // `pad_total` f32s per channel, one allocation per
        // (scale, plane). The masked-IW kernel reads ref-side
        // mu1/mu2/ssq/s12; CPU zensim uses ref-side only.
        let mut persist_planes_ref: Vec<[cubecl::server::Handle; 4]> = Vec::new();
        let partials_ext_f64: cubecl::server::Handle;
        let finals_ext_f64: cubecl::server::Handle;
        if needs_planes {
            for sc in scales.iter() {
                let plane_len = (sc.padded_w as usize) * (sc.h as usize) * 3;
                let alloc_planes = || -> [cubecl::server::Handle; 4] {
                    [
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                        alloc_empty_f32(&client, plane_len),
                    ]
                };
                persist_planes_ref.push(alloc_planes());
            }
            partials_ext_f64 = alloc_empty_f64(&client, partials_ext_total);
            finals_ext_f64 = alloc_empty_f64(&client, scales.len() * 3 * 12);
        } else {
            // Basic regime: tiny no-op placeholders. cubecl needs every
            // ArrayArg handle to be valid even if the kernel using it
            // is never launched; placeholders are 1-element rather
            // than 0-element to dodge any backend-side zero-len checks.
            partials_ext_f64 = alloc_empty_f64(&client, 1);
            finals_ext_f64 = alloc_empty_f64(&client, 1);
            // Empty placeholder vecs — code paths that read these check
            // `regime.needs_extended_kernel()` before indexing.
        }

        Ok(Self {
            client,
            width,
            height,
            pixels,
            strip,
            cached_ref_strip_srgb: Vec::new(),
            ref_full_xyb: None,
            src_u8_full: None,
            pack_scratch_full: Vec::new(),
            pack_scratch: vec![0_u32; upload_pixels],
            src_u8_a,
            src_u8_b,
            srgb_lut,
            scales,
            partials_f64,
            partials_max,
            partials_f64_len: partials_f64_total,
            partials_max_len: partials_max_total,
            finals_f64,
            finals_max,
            has_reference: false,
            diffmap_state: None,
            regime,
            persist_planes_ref,
            partials_ext_f64,
            partials_ext_f64_len: partials_ext_total.max(1),
            finals_ext_f64,
            gpu_diffmap_scratch: None,
        })
    }

    /// Which regime this pipeline was constructed for.
    pub fn regime(&self) -> ZensimFeatureRegime {
        self.regime
    }

    /// Debug-only: read back the persist-plane `mu1` at the given scale
    /// and channel. Returns a `Vec<f32>` of length `padded_w * height`
    /// (the plane's stride). `plane_idx` selects between mu1 (0), mu2
    /// (1), ssq (2), s12 (3). Returns an empty vec if the regime
    /// doesn't allocate persist planes.
    pub fn debug_read_persist_plane(
        &self,
        scale: usize,
        channel: usize,
        plane_idx: usize,
    ) -> Vec<f32> {
        if self.persist_planes_ref.is_empty() {
            return Vec::new();
        }
        let s = &self.scales[scale];
        let plane = &self.persist_planes_ref[scale][plane_idx];
        let bytes = self
            .client
            .read_one(plane.clone())
            .expect("read persist plane");
        let all = f32::from_bytes(&bytes);
        // Extract this channel's slice.
        let pt = s.n_padded;
        let ch_start = channel * pt;
        all[ch_start..ch_start + pt].to_vec()
    }

    /// Debug-only: read back the per-scale, per-channel `ref_xyb` (or
    /// `dis_xyb`) plane after `set_reference` / `compute_with_reference`
    /// has been called. Returns a `Vec<f32>` of length `padded_w * height`.
    pub fn debug_read_xyb(&self, scale: usize, channel: usize, ref_side: bool) -> Vec<f32> {
        let s = &self.scales[scale];
        let plane = if ref_side {
            &s.ref_xyb[channel]
        } else {
            &s.dis_xyb[channel]
        };
        let bytes = self.client.read_one(plane.clone()).expect("read xyb plane");
        f32::from_bytes(&bytes).to_vec()
    }

    /// Debug-only: get the padded width and image height for a scale.
    pub fn debug_scale_dims(&self, scale: usize) -> (u32, u32) {
        let s = &self.scales[scale];
        (s.padded_w, s.h)
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// Compute the 228-feature vector for one (reference, distorted)
    /// pair. **Only valid for `regime == Basic`.** When the pipeline was
    /// constructed with a wider regime, this returns the first 228 slots
    /// (`compute_features_vec` returns the full vector).
    pub fn compute_features(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        let v = self.compute_features_vec(ref_srgb, dist_srgb)?;
        let mut out = [0.0_f64; TOTAL_FEATURES];
        out.copy_from_slice(&v[..TOTAL_FEATURES]);
        Ok(out)
    }

    /// Compute the regime-appropriate feature vector for one (reference,
    /// distorted) pair. Length matches `self.regime().total_features()`:
    /// 228 / 300 / 372.
    ///
    /// This is a **cold one-shot** entry: it sets the reference and then
    /// runs exactly one `compute_with_reference_vec`. It is never the
    /// warm-loop entry point (a warm loop calls
    /// [`Self::set_reference`] once and then
    /// [`Self::compute_with_reference_vec`] / [`Self::compute_with_reference`]
    /// repeatedly). Because of that, in strip mode we route the cold
    /// reference through [`Self::set_reference_host_cached_only`] rather
    /// than [`Self::set_reference`]: the device-cached full-image ref
    /// XYB pyramid (task #75) earns its keep only across MANY warm dist
    /// iterations, so building it for a single dist call is redundant
    /// device-side work. The host-cached-only path rebuilds the ref XYB
    /// per strip — bit-identical to the device-cache row-slice (aligned
    /// strip starts; see `STRIP_PROCESSING.md` "Cached reference (Mode E
    /// device cache)") — and avoids allocating the full-image
    /// `src_u8_full` + per-scale ref XYB planes that Full-height device
    /// state would pin.
    ///
    /// In Full (non-strip) mode `set_reference_host_cached_only` is
    /// exactly equivalent to `set_reference` (no host cache is used), so
    /// this routing is a strip-only behaviour change.
    pub fn compute_features_vec(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<Vec<f64>> {
        if self.strip.is_some() {
            // Cold one-shot strip: skip the redundant full-image device
            // ref cache; the per-strip ref rebuild is bit-exact.
            self.set_reference_host_cached_only(ref_srgb)?;
        } else {
            self.set_reference(ref_srgb)?;
        }
        self.compute_with_reference_vec(dist_srgb)
    }

    /// Cache the reference pyramid; subsequent
    /// [`Zensim::compute_with_reference`] calls reuse it.
    ///
    /// **Phase 1 diffmap state**: if the diffmap state has been
    /// initialised (by any earlier call to a diffmap or linear-planes
    /// entry-point), this call also populates the CPU-side
    /// `PrecomputedReference` so the subsequent
    /// [`Self::score_with_warm_ref_diffmap`] path works against the
    /// SAME reference content. If the diffmap state has not been
    /// initialised, `set_reference` only updates the GPU pyramid —
    /// callers using the warm-ref-diffmap path must either invoke a
    /// diffmap entry-point first (which lazy-allocates the state) OR
    /// use [`Self::warm_reference_from_linear_planes`].
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        if self.strip.is_some() {
            // Strip mode: build the full-image ref XYB pyramid ONCE on
            // device (task #75 mode-E refinement; 2026-05-26). Per-strip
            // `compute_with_reference` then only copies the relevant
            // row range from the cached planes into the strip's
            // `Scale.ref_xyb`, skipping ref re-upload + ref XYB rebuild
            // per strip.
            //
            // The host-side `cached_ref_strip_srgb` fallback is cleared
            // here so the strip walker takes the device-cache code path.
            self.cached_ref_strip_srgb.clear();
            self.build_full_ref_xyb_pyramid(ref_srgb)?;
            self.has_reference = true;
        } else {
            self.upload_u8(true, ref_srgb);
            self.run_xyb_pyramid(true);
            self.has_reference = true;
        }

        // Mirror the reference into the diffmap state's warm cache so
        // sRGB-byte warm-ref-diffmap callers see the same reference
        // content the GPU side just uploaded.
        if let Some(state) = self.diffmap_state.as_mut() {
            let w = self.width as usize;
            let h = self.height as usize;
            let lut = srgb_lut_256();
            srgb_u8_to_linear_planes_tight(ref_srgb, w, h, &mut state.ref_linear_planes, &lut);
            let ref_views: [&[f32]; 3] = [
                &state.ref_linear_planes[0],
                &state.ref_linear_planes[1],
                &state.ref_linear_planes[2],
            ];
            let pre = state
                .cpu_zensim
                .precompute_reference_linear_planar(ref_views, w, h, w)
                .map_err(map_zensim_error)?;
            state.warm_ref = Some(pre);
        }
        // Phase 1b: invalidate the GPU diffmap scratch reference so the
        // next warm-ref-diffmap call re-warms it from the freshly
        // decoded `state.ref_linear_planes`. (We DON'T rebuild the GPU
        // ref pyramid eagerly here — only diffmap callers need it, and
        // the lazy rewarm in `score_with_warm_ref_diffmap` keeps the
        // scalar-only fast path free of the GPU diffmap cost.)
        if let Some(g) = self.gpu_diffmap_scratch.as_mut() {
            g.inner.has_reference = false;
        }
        Ok(())
    }

    pub fn clear_reference(&mut self) {
        self.has_reference = false;
        self.cached_ref_strip_srgb.clear();
        // Drop device-cached ref XYB planes — handles get refcount-
        // decremented; cubecl's memory pool reclaims when the last
        // handle goes away. The `src_u8_full` staging buffer stays
        // allocated so the next set_reference call doesn't pay the
        // alloc.
        self.ref_full_xyb = None;
        if let Some(state) = self.diffmap_state.as_mut() {
            state.warm_ref = None;
        }
        if let Some(g) = self.gpu_diffmap_scratch.as_mut() {
            g.inner.has_reference = false;
        }
    }

    /// Opt out of the device-cached ref XYB pyramid added in task #75.
    ///
    /// In strip mode only, this caches the reference sRGB bytes
    /// host-side instead of building the full-image XYB pyramid on
    /// device. Each `compute_with_reference` then re-uploads the
    /// appropriate ref strip + rebuilds the ref XYB pyramid per strip
    /// — the pre-task-#75 behaviour.
    ///
    /// Use when VRAM pressure dominates and ref-side rebuild cost
    /// across N dist iterations is acceptable. The default path
    /// ([`Self::set_reference`]) is faster for warm-loop usage.
    ///
    /// In Full mode this is equivalent to [`Self::set_reference`]
    /// (no host cache is used).
    pub fn set_reference_host_cached_only(&mut self, ref_srgb: &[u8]) -> Result<()> {
        self.check_dims(ref_srgb)?;
        if self.strip.is_some() {
            // Drop any device cache from a prior call.
            self.ref_full_xyb = None;
            self.cached_ref_strip_srgb.clear();
            self.cached_ref_strip_srgb.extend_from_slice(ref_srgb);
            self.has_reference = true;
        } else {
            self.upload_u8(true, ref_srgb);
            self.run_xyb_pyramid(true);
            self.has_reference = true;
        }

        // Mirror to diffmap state (same as set_reference).
        if let Some(state) = self.diffmap_state.as_mut() {
            let w = self.width as usize;
            let h = self.height as usize;
            let lut = srgb_lut_256();
            srgb_u8_to_linear_planes_tight(ref_srgb, w, h, &mut state.ref_linear_planes, &lut);
            let ref_views: [&[f32]; 3] = [
                &state.ref_linear_planes[0],
                &state.ref_linear_planes[1],
                &state.ref_linear_planes[2],
            ];
            let pre = state
                .cpu_zensim
                .precompute_reference_linear_planar(ref_views, w, h, w)
                .map_err(map_zensim_error)?;
            state.warm_ref = Some(pre);
        }
        Ok(())
    }

    pub fn has_reference(&self) -> bool {
        self.has_reference
    }

    /// Compute the 228-feature vector for one distorted image against
    /// the cached reference. **Only valid for `regime == Basic`.**
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<[f64; TOTAL_FEATURES]> {
        let v = self.compute_with_reference_vec(dist_srgb)?;
        let mut out = [0.0_f64; TOTAL_FEATURES];
        out.copy_from_slice(&v[..TOTAL_FEATURES]);
        Ok(out)
    }

    /// Compute the regime-appropriate feature vector for one distorted
    /// image against the cached reference. Returns
    /// [`Error::NoCachedReference`] if [`Zensim::set_reference`] hasn't
    /// been called.
    pub fn compute_with_reference_vec(&mut self, dist_srgb: &[u8]) -> Result<Vec<f64>> {
        if !self.has_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;
        // Strip mode runs a separate walker that loops over image
        // strips; the full-image fast path stays inline so the warm
        // hot loop pays no extra dispatch cost.
        if self.strip.is_some() {
            return self.compute_with_reference_vec_strip(dist_srgb);
        }
        self.upload_u8(false, dist_srgb);
        self.run_xyb_pyramid(false);

        let n_scales = self.scales.len();
        let needs_ext = self.regime.needs_extended_kernel();

        // Phase 1: launch the per-scale fused features kernel. Two
        // variants: persist (writes mu1/mu2/ssq/s12 planes) on the
        // Extended/WithIw regime, plain on Basic. Both produce the
        // same per-column 17 f64 + 3 f32 partials.
        for s in 0..n_scales {
            if needs_ext {
                self.launch_blur_and_features_persist(s);
            } else {
                self.launch_blur_and_features(s);
            }
        }

        // Phase 1b: masked + IW pooling pass. Needs the persist planes
        // from Phase 1.
        if needs_ext {
            for s in 0..n_scales {
                self.launch_masked_iw(s);
            }
        }

        // Phase 2: on-device reduction. Basic partials reduce as before;
        // masked + IW partials reduce via `reduce_ext_kernel`.
        self.launch_reduction();
        if needs_ext {
            self.launch_reduction_ext();
        }

        // Phase 3: ONE read for the basic finals; one more for masked +
        // IW finals when needed. cubecl serialises both reads behind
        // the reduction launches.
        let f64_bytes = self
            .client
            .read_one(self.finals_f64.clone())
            .expect("read finals_f64");
        let max_bytes = self
            .client
            .read_one(self.finals_max.clone())
            .expect("read finals_max");
        let finals_f64 = f64::from_bytes(&f64_bytes);
        let finals_max = f32::from_bytes(&max_bytes);

        let ext_bytes_storage;
        let finals_ext_f64: &[f64] = if needs_ext {
            ext_bytes_storage = self
                .client
                .read_one(self.finals_ext_f64.clone())
                .expect("read finals_ext_f64");
            f64::from_bytes(&ext_bytes_storage)
        } else {
            &[]
        };

        // Phase 4: host packs the regime-appropriate feature vector.
        //
        // Per-scale image height: in Full mode, `self.scales[s].h`. In
        // strip mode (NOT this branch — strip walker dispatches earlier
        // in this function), we'd use the **full image height** halved
        // per scale.
        let mut scale_image_h: [u32; SCALES] = [0; SCALES];
        let mut hs = self.height;
        for s in 0..n_scales {
            scale_image_h[s] = hs;
            hs = hs.div_ceil(2);
        }

        Ok(self.pack_feature_vector(
            finals_f64,
            finals_max,
            finals_ext_f64,
            &scale_image_h[..n_scales],
        ))
    }

    /// Host-side packing of raw per-(scale, ch, slot) accumulators
    /// into the regime-appropriate feature vector. Shared between Full
    /// and Strip modes.
    ///
    /// `finals_f64.len() == n_scales × 3 × 17`,
    /// `finals_max.len() == n_scales × 3 × 3`,
    /// `finals_ext_f64.len() == n_scales × 3 × 12` (or empty on Basic).
    ///
    /// `scale_image_h[s]` is the image height at scale s — `self.height`
    /// halved per scale; equal to `self.scales[s].h` in Full mode.
    /// Strip mode passes the full-image scale heights (NOT the strip
    /// allocation heights, which differ).
    fn pack_feature_vector(
        &self,
        finals_f64: &[f64],
        finals_max: &[f32],
        finals_ext_f64: &[f64],
        scale_image_h: &[u32],
    ) -> Vec<f64> {
        let needs_ext = self.regime.needs_extended_kernel();
        let n_scales = self.scales.len();
        let total = self.regime.total_features();
        let mut out = vec![0.0_f64; total];
        let basic_total = n_scales * FEATURES_PER_CHANNEL_BASIC * 3;
        let peaks_total = n_scales * FEATURES_PER_CHANNEL_PEAKS * 3;
        let masked_block_off = basic_total + peaks_total;
        let iw_block_off = masked_block_off + n_scales * FEATURES_PER_CHANNEL_MASKED * 3;

        for s in 0..n_scales {
            for ch in 0..3 {
                let final_f64_base = (s * 3 + ch) * 17;
                let final_max_base = (s * 3 + ch) * 3;
                let mut sums = [0.0_f64; 17];
                sums.copy_from_slice(&finals_f64[final_f64_base..final_f64_base + 17]);
                let mut peaks = [0.0_f32; 3];
                peaks.copy_from_slice(&finals_max[final_max_base..final_max_base + 3]);

                let pad_w = self.scales[s].padded_w as usize;
                let h_dim = scale_image_h[s] as usize;
                let inv_n = 1.0_f64 / (pad_w as f64 * h_dim as f64);
                let var_src = sums[10] * inv_n;
                let mad_src = sums[12] * inv_n;
                let (hf_energy_loss, hf_energy_gain) = if var_src > 1e-10 {
                    let r = sums[11] / sums[10];
                    ((1.0 - r).max(0.0), (r - 1.0).max(0.0))
                } else {
                    (0.0, 0.0)
                };
                let hf_mag_loss = if mad_src > 1e-10 {
                    (1.0 - sums[13] / sums[12]).max(0.0)
                } else {
                    0.0
                };

                // Basic block: 13 features per channel.
                let bb = s * 3 * FEATURES_PER_CHANNEL_BASIC + ch * FEATURES_PER_CHANNEL_BASIC;
                out[bb] = (sums[0] * inv_n).abs();
                out[bb + 1] = (sums[1] * inv_n).max(0.0).powf(0.25);
                out[bb + 2] = (sums[2] * inv_n).max(0.0).sqrt();
                out[bb + 3] = (sums[3] * inv_n).abs();
                out[bb + 4] = (sums[4] * inv_n).max(0.0).powf(0.25);
                out[bb + 5] = (sums[5] * inv_n).max(0.0).sqrt();
                out[bb + 6] = (sums[6] * inv_n).abs();
                out[bb + 7] = (sums[7] * inv_n).max(0.0).powf(0.25);
                out[bb + 8] = (sums[8] * inv_n).max(0.0).sqrt();
                out[bb + 9] = sums[9] * inv_n;
                out[bb + 10] = hf_energy_loss;
                out[bb + 11] = hf_mag_loss;
                out[bb + 12] = hf_energy_gain;

                // Peaks block.
                let pb = basic_total
                    + s * 3 * FEATURES_PER_CHANNEL_PEAKS
                    + ch * FEATURES_PER_CHANNEL_PEAKS;
                out[pb] = peaks[0] as f64;
                out[pb + 1] = peaks[1] as f64;
                out[pb + 2] = peaks[2] as f64;
                out[pb + 3] = (sums[14] * inv_n).max(0.0).powf(0.125);
                out[pb + 4] = (sums[15] * inv_n).max(0.0).powf(0.125);
                out[pb + 5] = (sums[16] * inv_n).max(0.0).powf(0.125);

                // Masked + IW blocks.
                if needs_ext {
                    let ext_base = (s * 3 + ch) * 12;
                    let mut ext_sums = [0.0_f64; 12];
                    ext_sums.copy_from_slice(&finals_ext_f64[ext_base..ext_base + 12]);

                    if self.regime.needs_masked() {
                        let mb = masked_block_off
                            + s * 3 * FEATURES_PER_CHANNEL_MASKED
                            + ch * FEATURES_PER_CHANNEL_MASKED;
                        out[mb] = (ext_sums[0] * inv_n).abs();
                        out[mb + 1] = (ext_sums[1] * inv_n).max(0.0).powf(0.25);
                        out[mb + 2] = (ext_sums[2] * inv_n).max(0.0).sqrt();
                        out[mb + 3] = (ext_sums[3] * inv_n).max(0.0).powf(0.25);
                        out[mb + 4] = (ext_sums[4] * inv_n).max(0.0).powf(0.25);
                        out[mb + 5] = ext_sums[5] * inv_n;
                    }
                    if self.regime.needs_iw() {
                        let ib = iw_block_off
                            + s * 3 * FEATURES_PER_CHANNEL_IW
                            + ch * FEATURES_PER_CHANNEL_IW;
                        out[ib] = (ext_sums[6] * inv_n).abs();
                        out[ib + 1] = (ext_sums[7] * inv_n).max(0.0).powf(0.25);
                        out[ib + 2] = (ext_sums[8] * inv_n).max(0.0).sqrt();
                        out[ib + 3] = (ext_sums[9] * inv_n).max(0.0).powf(0.25);
                        out[ib + 4] = (ext_sums[10] * inv_n).max(0.0).powf(0.25);
                        out[ib + 5] = ext_sums[11] * inv_n;
                    }
                }
            }
        }
        out
    }

    /// Strip-mode entry point. Walks the image in strips, runs the
    /// existing per-scale kernels on strip-sized buffers with body-row
    /// gating, accumulates raw per-(scale, ch, slot) sums across strips
    /// on the host, then packs the feature vector using the full image
    /// height for the per-pixel normaliser.
    fn compute_with_reference_vec_strip(&mut self, dist_srgb: &[u8]) -> Result<Vec<f64>> {
        // Caller guarantees `self.strip.is_some()` and validated dims.
        let strip_state = self.strip.expect("strip mode");
        let n_scales = self.scales.len();
        let needs_ext = self.regime.needs_extended_kernel();

        // Host-side raw-sum accumulators per (scale, channel, slot).
        // Mirror the device finals' layout so `pack_feature_vector`
        // sees the same shape it does in Full mode.
        let mut acc_f64 = vec![0.0_f64; n_scales * 3 * 17];
        let mut acc_max = vec![f32::NEG_INFINITY; n_scales * 3 * 3];
        let mut acc_ext_f64 = vec![0.0_f64; if needs_ext { n_scales * 3 * 12 } else { 0 }];

        let strips = strip_state.strips();
        for &(body_lo, body_hi, up_lo, up_hi) in strips.iter() {
            // Run one strip through the pipeline.
            self.run_one_strip_ref(dist_srgb, body_lo, body_hi, up_lo, up_hi)?;

            // Read back per-(scale, ch, slot) finals — already
            // body-gated by the kernel.
            let f64_bytes = self
                .client
                .read_one(self.finals_f64.clone())
                .expect("read finals_f64");
            let max_bytes = self
                .client
                .read_one(self.finals_max.clone())
                .expect("read finals_max");
            let finals_f64 = f64::from_bytes(&f64_bytes);
            let finals_max = f32::from_bytes(&max_bytes);
            for i in 0..acc_f64.len() {
                acc_f64[i] += finals_f64[i];
            }
            for i in 0..acc_max.len() {
                if finals_max[i] > acc_max[i] {
                    acc_max[i] = finals_max[i];
                }
            }
            if needs_ext {
                let ext_bytes = self
                    .client
                    .read_one(self.finals_ext_f64.clone())
                    .expect("read finals_ext_f64");
                let finals_ext = f64::from_bytes(&ext_bytes);
                for i in 0..acc_ext_f64.len() {
                    acc_ext_f64[i] += finals_ext[i];
                }
            }
        }

        // Replace any never-touched peak slots (image too small for
        // body to overlap any strip — should never happen in practice
        // since at least one strip's body row range is non-empty) with
        // 0.0 so the host pack doesn't see -inf.
        for v in acc_max.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }

        // Build scale_image_h from FULL image height — body rows
        // partition `[0, image_h)` perfectly, so per-scale body pixels
        // = `pw × image_h_at_scale`. (Strip allocation height is
        // irrelevant to the per-feature normaliser.)
        let mut scale_image_h: [u32; SCALES] = [0; SCALES];
        let mut hs = self.height;
        for s in 0..n_scales {
            scale_image_h[s] = hs;
            hs = hs.div_ceil(2);
        }
        Ok(self.pack_feature_vector(&acc_f64, &acc_max, &acc_ext_f64, &scale_image_h[..n_scales]))
    }

    /// Run one strip end-to-end: upload, xyb pyramid, fused features,
    /// (optional) masked-IW, reduction. Leaves the per-(scale, ch, slot)
    /// finals in `self.finals_f64 / finals_max / finals_ext_f64` ready
    /// for the caller to read back.
    ///
    /// `body_lo..body_hi` is the body row range in image coordinates
    /// (the rows this strip "owns"); `up_lo..up_hi` is the actual
    /// uploaded region (body + halo, clamped to image bounds). The
    /// uploaded region maps onto the strip-sized device buffers at
    /// rows `[0, up_hi - up_lo)`; the body row range in strip-local
    /// coords is `[body_lo - up_lo, body_hi - up_lo)`.
    fn run_one_strip_ref(
        &mut self,
        dist_srgb: &[u8],
        body_lo: u32,
        body_hi: u32,
        up_lo: u32,
        up_hi: u32,
    ) -> Result<()> {
        let needs_ext = self.regime.needs_extended_kernel();
        let actual_strip_h = up_hi - up_lo;
        // Body in strip-local coords (rows in `[0, actual_strip_h)`).
        let body_local_lo = body_lo - up_lo;
        let body_local_hi = body_hi - up_lo;

        // 1. Update per-scale h_dim to reflect the actual strip height
        //    (boundary strips may be < strip_alloc_h).
        self.set_scale_h_for_strip(actual_strip_h);

        // 2-3. Reference side: prefer the device-cached full-image
        //    ref XYB pyramid (task #75; built by `set_reference`)
        //    when available — slice rows into the strip's
        //    `Scale.ref_xyb` and skip ref re-upload + ref XYB rebuild.
        //    Fallback path (only used after `set_reference_host_cached_only`)
        //    uploads the ref strip then runs the pyramid build.
        if self.ref_full_xyb.is_some() {
            self.install_ref_xyb_from_full_cache(up_lo);
        } else {
            self.upload_u8_strip(true, up_lo as usize, up_hi as usize, None);
            self.run_xyb_pyramid(true);
        }
        // Distorted side always uploads + builds the strip's pyramid
        // (no warm-cache equivalent — every dist call is unique).
        self.upload_u8_strip(false, up_lo as usize, up_hi as usize, Some(dist_srgb));
        self.run_xyb_pyramid(false);

        // 4. Fused features (persist variant when masked/IW needed).
        //
        // Map body rows from scale-0 strip-local to scale s. Both ends
        // use `div_ceil` so consecutive strips' body ranges at scale s
        // are contiguous (strip k's body_hi at scale s = strip k+1's
        // body_lo at scale s). Using floor for body_hi would lose the
        // last partial row at the bottom-boundary strip when image_h
        // isn't aligned to 2^(SCALES - 1).
        for s in 0..self.scales.len() {
            let denom = 1u32 << s;
            let body_s_lo = body_local_lo.div_ceil(denom);
            let body_s_hi = body_local_hi.div_ceil(denom);
            // Clamp to actual scale h so the kernel's body gate doesn't
            // try to count rows past the strip buffer.
            let scale_h = self.scales[s].h;
            let body_s_hi_clamped = body_s_hi.min(scale_h);
            if needs_ext {
                self.launch_blur_and_features_persist_with_body(s, body_s_lo, body_s_hi_clamped);
            } else {
                self.launch_blur_and_features_with_body(s, body_s_lo, body_s_hi_clamped);
            }
        }

        // 5. Masked + IW pooling.
        if needs_ext {
            for s in 0..self.scales.len() {
                let denom = 1u32 << s;
                let body_s_lo = body_local_lo.div_ceil(denom);
                let body_s_hi = body_local_hi.div_ceil(denom);
                let scale_h = self.scales[s].h;
                let body_s_hi_clamped = body_s_hi.min(scale_h);
                self.launch_masked_iw_with_body(s, body_s_lo, body_s_hi_clamped);
            }
        }

        // 6. Reduction — partials → finals on device.
        self.launch_reduction();
        if needs_ext {
            self.launch_reduction_ext();
        }
        Ok(())
    }

    /// Reset every `Scale.h` to `actual_strip_h / 2^s` for the current
    /// strip. Buffers are sized for `strip_alloc_h / 2^s` so this only
    /// changes the kernels' iteration bounds; boundary strips with
    /// `actual_strip_h < strip_alloc_h` reuse the same allocations.
    fn set_scale_h_for_strip(&mut self, actual_strip_h: u32) {
        let mut h = actual_strip_h;
        for s in self.scales.iter_mut() {
            s.h = h;
            // n_strips_ext recomputed to keep the CPU-shape strip
            // count in sync with the actual strip height.
            s.n_strips_ext = kernels::masked_iw_strip::cpu_strip_count(h);
            h = h.div_ceil(2);
        }
    }

    /// Build the full-image ref XYB pyramid on device (task #75
    /// mode-E refinement). Allocates `src_u8_full` + the per-scale
    /// XYB plane handles on first call; reuses across subsequent
    /// `set_reference` invocations.
    ///
    /// Mirrors the existing `upload_u8 + run_xyb_pyramid(is_a=true)`
    /// chain but operates on full-image-sized buffers rather than the
    /// strip-sized `src_u8_a` + `Scale.ref_xyb`. The downscale is
    /// bit-identical to what the strip walker would produce on a
    /// pyramid-aligned strip (consecutive 2-row downscale; aligned
    /// strip start preserves the row pairing). See `STRIP_PROCESSING.md`
    /// "Cached reference (Mode E device cache)" for the parity proof.
    fn build_full_ref_xyb_pyramid(&mut self, ref_srgb: &[u8]) -> Result<()> {
        // Per-scale `(padded_w, full_h)` plan — matches what Full mode
        // would allocate for this (width, height) pair, NOT the strip
        // allocation. simd_padded_width is monotone in width, so
        // padded_w per scale matches `Scale.padded_w` exactly (the
        // strip and full pipelines descend the pyramid the same way).
        let mut plan_full: Vec<(u32, u32)> = Vec::with_capacity(SCALES);
        let mut pw = simd_padded_width(self.width as usize) as u32;
        let mut h_at_s = self.height;
        for _ in 0..SCALES {
            if pw < 8 || h_at_s < 8 {
                break;
            }
            plan_full.push((pw, h_at_s));
            pw /= 2;
            h_at_s = h_at_s.div_ceil(2);
        }

        // Allocate per-scale XYB planes on first use (or after a
        // resolution-changing reset — currently we never resize a
        // Zensim, so this is a one-shot allocation). When already
        // allocated and dims unchanged, reuse.
        let needs_alloc = match self.ref_full_xyb.as_ref() {
            Some(state) => state.dims != plan_full,
            None => true,
        };
        if needs_alloc {
            let xyb: Vec<[cubecl::server::Handle; 3]> = plan_full
                .iter()
                .map(|&(pw, h)| {
                    let n = (pw as usize) * (h as usize);
                    [
                        alloc_empty_f32(&self.client, n),
                        alloc_empty_f32(&self.client, n),
                        alloc_empty_f32(&self.client, n),
                    ]
                })
                .collect();
            self.ref_full_xyb = Some(RefFullXybState {
                dims: plan_full.clone(),
                xyb,
            });
        }

        // Allocate src_u8_full + pack_scratch_full on first use.
        let full_pixels = (self.width as usize) * (self.height as usize);
        if self.src_u8_full.is_none() {
            self.src_u8_full = Some(self.client.empty(full_pixels * core::mem::size_of::<u32>()));
        }
        if self.pack_scratch_full.len() != full_pixels {
            self.pack_scratch_full.resize(full_pixels, 0);
        }

        // Pack u8×3 → u32 into the persistent host scratch.
        for (dst, chunk) in self
            .pack_scratch_full
            .iter_mut()
            .zip(ref_srgb.chunks_exact(3))
        {
            *dst = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        }
        let bytes = u32::as_bytes(&self.pack_scratch_full);
        // create_from_slice_pinned returns a new handle each call; we
        // overwrite the cached handle so the prior allocation can be
        // reclaimed.
        self.src_u8_full = Some(self.client.create_from_slice_pinned(bytes));

        // Run sRGB → positive XYB on full-image-sized buffers, then
        // downscale through the pyramid. Mirrors the inner body of
        // `run_xyb_pyramid` but uses `plan_full` dims + `ref_full_xyb`
        // handles rather than `Scale.ref_xyb`.
        self.run_ref_full_xyb_pyramid_kernels(&plan_full);
        Ok(())
    }

    /// Kernel-launch body of [`Self::build_full_ref_xyb_pyramid`].
    /// Split out so the parameter packing + handle allocation stay
    /// close to the cache state, while the unsafe kernel launches
    /// stay near the rest of the kernel-dispatch code.
    fn run_ref_full_xyb_pyramid_kernels(&self, plan_full: &[(u32, u32)]) {
        let absorbance_bias_neg = -color::cbrtf_fast_host(color::K_B0);
        let ref_state = self
            .ref_full_xyb
            .as_ref()
            .expect("ref_full_xyb allocated by caller");
        let src = self
            .src_u8_full
            .as_ref()
            .expect("src_u8_full allocated by caller");

        // Scale 0: sRGB → positive XYB on the full image.
        let (pw0, h0) = plan_full[0];
        let n_padded0 = (pw0 as usize) * (h0 as usize);
        let pixels0 = (self.width as usize) * (h0 as usize);
        let mirror_arg = match self.scales[0].mirror_offsets.as_ref() {
            Some(mo) => (mo.clone(), self.scales[0].pad_count as usize),
            None => (self.srgb_lut.clone(), 1),
        };
        unsafe {
            color::srgb_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n_padded0),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), pixels0),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), 256),
                ArrayArg::from_raw_parts(mirror_arg.0, mirror_arg.1),
                ArrayArg::from_raw_parts(ref_state.xyb[0][0].clone(), n_padded0),
                ArrayArg::from_raw_parts(ref_state.xyb[0][1].clone(), n_padded0),
                ArrayArg::from_raw_parts(ref_state.xyb[0][2].clone(), n_padded0),
                self.width,
                h0,
                pw0,
                absorbance_bias_neg,
            );
        }

        // Pyramid: 2× downscale per scale.
        for s in 1..plan_full.len() {
            let (prev_pw, prev_h) = plan_full[s - 1];
            let (curr_pw, curr_h) = plan_full[s];
            let prev_n = (prev_pw as usize) * (prev_h as usize);
            let curr_n = (curr_pw as usize) * (curr_h as usize);
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(curr_n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(ref_state.xyb[s - 1][0].clone(), prev_n),
                    ArrayArg::from_raw_parts(ref_state.xyb[s - 1][1].clone(), prev_n),
                    ArrayArg::from_raw_parts(ref_state.xyb[s - 1][2].clone(), prev_n),
                    ArrayArg::from_raw_parts(ref_state.xyb[s][0].clone(), curr_n),
                    ArrayArg::from_raw_parts(ref_state.xyb[s][1].clone(), curr_n),
                    ArrayArg::from_raw_parts(ref_state.xyb[s][2].clone(), curr_n),
                    prev_pw,
                    prev_h,
                    curr_pw,
                    curr_h,
                );
            }
        }
    }

    /// Copy the strip's row range from the cached full-image ref XYB
    /// pyramid into the strip-sized `Scale.ref_xyb` buffers.
    ///
    /// `up_lo` is the strip's upload start in scale-0 image rows
    /// (multiple of [`STRIP_ALIGN`]). At scale s, the source row
    /// range is `[up_lo >> s, (up_lo >> s) + Scale.h)` — the
    /// strip-buffer height at scale s equals `Scale.h` because
    /// [`set_scale_h_for_strip`] was called immediately before this.
    ///
    /// Bit-exact parity vs the strip-walker's own pyramid build:
    /// the 2× downscale operates on consecutive row pairs `(2r, 2r+1)`,
    /// so when `up_lo` is a multiple of `2^(SCALES-1)`, scale-s row r
    /// of the strip buffer equals scale-s row `((up_lo >> s) + r)` of
    /// the full buffer.
    fn install_ref_xyb_from_full_cache(&self, up_lo: u32) {
        let ref_state = self
            .ref_full_xyb
            .as_ref()
            .expect("install_ref_xyb_from_full_cache requires populated cache");
        for s in 0..self.scales.len() {
            let (full_pw, full_h) = ref_state.dims[s];
            // Source row range at scale s.
            let src_row_start = up_lo >> (s as u32);
            // Active rows for this scale = strip's Scale.h (already
            // set by set_scale_h_for_strip), clamped to the actual
            // remaining rows in the full ref scale plane.
            let scale = &self.scales[s];
            let n_rows = scale.h.min(full_h.saturating_sub(src_row_start));
            if n_rows == 0 {
                continue;
            }
            let src_total_n = (full_pw as usize) * (full_h as usize);
            let dst_total_n = scale.n_padded;
            for ch in 0..3 {
                self.launch_copy_rows(
                    &ref_state.xyb[s][ch],
                    &scale.ref_xyb[ch],
                    full_pw,
                    src_total_n,
                    dst_total_n,
                    n_rows,
                    src_row_start,
                );
            }
        }
    }

    /// Launch the row-range copy kernel — `dst[0..n_rows*width]` =
    /// `src[src_row_start*width..(src_row_start+n_rows)*width]`.
    /// Mirrors `dssim-gpu::pipeline::Dssim::launch_copy_rows`.
    #[allow(clippy::too_many_arguments)]
    fn launch_copy_rows(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        width: u32,
        src_total_n: usize,
        dst_total_n: usize,
        n_rows: u32,
        src_row_start: u32,
    ) {
        let total = (n_rows as usize) * (width as usize);
        if total == 0 {
            return;
        }
        unsafe {
            blit::copy_rows_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(total),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), src_total_n),
                ArrayArg::from_raw_parts(dst.clone(), dst_total_n),
                width,
                n_rows,
                src_row_start,
            );
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.pixels * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        let cubes = (n as u32).div_ceil(TPB);
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }

    fn upload_u8(&mut self, is_a: bool, srgb: &[u8]) {
        // T4.L (pre-dates this session): pack 3 u8 bytes into one u32
        // per pixel: R | G<<8 | B<<16. Kernel masks the bytes back
        // out; on-device math is unchanged. 3× H2D bandwidth saving
        // vs the older "widen each u8 to its own u32" layout —
        // significant on WSL2 where PCIe is virtualised to ~3 GB/s.
        for (dst, chunk) in self.pack_scratch.iter_mut().zip(srgb.chunks_exact(3)) {
            *dst = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        }
        // T4.M (2026-05-16): pinned-host upload via the lilith/cubecl
        // feat/pinned-upload fork — DMAs at 12-25 GB/s on PCIe 4.0 vs
        // 5-6 GB/s pageable. See docs/CUBECL_GOTCHAS.md G6.5.
        let bytes = u32::as_bytes(&self.pack_scratch);
        if is_a {
            self.src_u8_a = self.client.create_from_slice_pinned(bytes);
        } else {
            self.src_u8_b = self.client.create_from_slice_pinned(bytes);
        }
    }

    /// Strip-mode upload helper: pack rows `[up_lo, up_hi)` of either
    /// the cached ref bytes (when `is_a == true` and `dist_srgb` is
    /// `None`) or the supplied dist bytes (when `is_a == false`) into
    /// `pack_scratch`, then upload as packed u32 pixels.
    ///
    /// The src buffer (src_u8_a / src_u8_b) is sized for one strip's
    /// allocation (`width * strip_alloc_h`). Boundary strips may have
    /// `up_hi - up_lo < strip_alloc_h`; we pack only that many rows.
    fn upload_u8_strip(
        &mut self,
        is_a: bool,
        up_lo: usize,
        up_hi: usize,
        dist_srgb: Option<&[u8]>,
    ) {
        let src_bytes: &[u8] = if is_a {
            &self.cached_ref_strip_srgb
        } else {
            dist_srgb.expect("dist_srgb required when is_a == false")
        };
        let row_bytes = (self.width as usize) * 3;
        let strip_rows = up_hi - up_lo;
        let strip_pixels = strip_rows * (self.width as usize);
        let row_start = up_lo * row_bytes;
        let row_end = up_hi * row_bytes;
        let strip_slice = &src_bytes[row_start..row_end];
        // Pack into the persistent pack_scratch (already sized to
        // `width * strip_alloc_h` per `new_with_regime_strip_budget`).
        for (dst, chunk) in self
            .pack_scratch
            .iter_mut()
            .take(strip_pixels)
            .zip(strip_slice.chunks_exact(3))
        {
            *dst = (chunk[0] as u32) | ((chunk[1] as u32) << 8) | ((chunk[2] as u32) << 16);
        }
        // Upload only the populated prefix to avoid pinning unused
        // tail rows for boundary strips. cubecl::Handle returned by
        // create_from_slice_pinned wraps the byte length internally.
        let bytes = u32::as_bytes(&self.pack_scratch[..strip_pixels]);
        if is_a {
            self.src_u8_a = self.client.create_from_slice_pinned(bytes);
        } else {
            self.src_u8_b = self.client.create_from_slice_pinned(bytes);
        }
    }

    /// sRGB → positive XYB at scale 0, mirror-fill padding, then
    /// downscale through the pyramid. Operates on either the reference
    /// or distorted side based on `is_a`.
    ///
    /// In strip mode, `self.scales[*].h` reflects the current strip's
    /// actual_strip_h (set via [`set_scale_h_for_strip`]); the kernel
    /// processes `padded_w × h_strip` pixels.
    fn run_xyb_pyramid(&self, is_a: bool) {
        let s0 = &self.scales[0];
        let src = if is_a { &self.src_u8_a } else { &self.src_u8_b };
        let xyb = if is_a { &s0.ref_xyb } else { &s0.dis_xyb };
        let absorbance_bias_neg = -color::cbrtf_fast_host(color::K_B0);
        let mirror_arg = match s0.mirror_offsets.as_ref() {
            Some(mo) => (mo.clone(), s0.pad_count as usize),
            None => (self.srgb_lut.clone(), 1),
        };
        // Pixel count for THIS pass — in strip mode the kernel only
        // touches the first `s0.h` rows even though the buffer is sized
        // for `strip_alloc_h`. The xyb buffer length passed to the
        // kernel is still the full allocation so cubecl's bounds check
        // doesn't reject the call.
        let this_h = s0.h;
        let this_pixels = (self.width as usize) * (this_h as usize);
        let this_padded_pixels = (s0.padded_w as usize) * (this_h as usize);
        unsafe {
            color::srgb_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(this_padded_pixels),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), this_pixels),
                ArrayArg::from_raw_parts(self.srgb_lut.clone(), 256),
                ArrayArg::from_raw_parts(mirror_arg.0, mirror_arg.1),
                ArrayArg::from_raw_parts(xyb[0].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[1].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[2].clone(), s0.n_padded),
                self.width,
                this_h,
                s0.padded_w,
                absorbance_bias_neg,
            );
        }
        // Build pyramid via 2× planar downscale, all 3 channels per launch.
        for s in 1..self.scales.len() {
            let prev = &self.scales[s - 1];
            let curr = &self.scales[s];
            let prev_xyb = if is_a { &prev.ref_xyb } else { &prev.dis_xyb };
            let curr_xyb = if is_a { &curr.ref_xyb } else { &curr.dis_xyb };
            // Cube count over the current scale's *active* pixels
            // (`padded_w × h`); buffer size stays at `n_padded` for
            // strip mode boundary strips where `h` is < strip_alloc_h.
            let curr_active = (curr.padded_w as usize) * (curr.h as usize);
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(curr_active),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(prev_xyb[0].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[1].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[2].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[0].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[1].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[2].clone(), curr.n_padded),
                    prev.padded_w,
                    prev.h,
                    curr.padded_w,
                    curr.h,
                );
            }
        }
    }

    /// Launch the **tile-fused H-blur + V-blur + features** kernel for
    /// one scale. Grid `(ceil(pw/64), n_strips, 3)`; block dim 64.
    /// One launch per scale (was 2 with the separate H-blur path).
    /// Eliminates the 12 H-blur scratch planes from DRAM — H-blur
    /// outputs live in shared memory across the V-blur slide.
    fn launch_blur_and_features(&self, scale: usize) {
        let s = &self.scales[scale];
        let h = s.h;
        self.launch_blur_and_features_with_body(scale, 0, h);
    }

    /// Body-aware variant: only rows `[y_body_start, y_body_end)`
    /// contribute to the per-(col, strip, ch) partials. Halo rows
    /// drive the V-blur sliding window so per-pixel mu1/mu2/ssq/s12
    /// at body rows are buffer-correct (which means
    /// image-correct in strip mode when halo rows ship the image's
    /// rows adjacent to the body region).
    fn launch_blur_and_features_with_body(&self, scale: usize, y_body_start: u32, y_body_end: u32) {
        // Keep in sync with `kernels::fused::TX`.
        const TX: u32 = 64;
        let s = &self.scales[scale];
        let pad_total = s.n_padded;

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        unsafe {
            fused::fused_features_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                s.padded_w,
                s.h,
                s.n_strips,
                s.partials_f64_off as u32,
                s.partials_max_off as u32,
                y_body_start,
                y_body_end,
            );
        }
    }

    /// Persist-planes variant of `launch_blur_and_features`. Same math
    /// + same per-column partials, plus per-pixel writes of mu1/mu2/
    /// ssq/s12 into the appropriate side's persist planes. Run once for
    /// `is_a == true` then once for `false` is NOT what happens — the
    /// existing pipeline already runs the basic kernel once per pair
    /// (combining ref + dist via the (src_a/dst_a) channel arrays).
    /// The persist variant takes the same combined inputs but ALSO
    /// writes the persist planes. We persist to the `ref` side's planes
    /// (the masked-IW kernel needs them for `activity = blur(|src - mu1|)`
    /// which uses the REFERENCE side per CPU).
    fn launch_blur_and_features_persist(&self, scale: usize) {
        let s = &self.scales[scale];
        let h = s.h;
        self.launch_blur_and_features_persist_with_body(scale, 0, h);
    }

    fn launch_blur_and_features_persist_with_body(
        &self,
        scale: usize,
        y_body_start: u32,
        y_body_end: u32,
    ) {
        const TX: u32 = 64;
        let s = &self.scales[scale];
        let pad_total = s.n_padded;
        let plane_len = pad_total * 3;
        let planes = &self.persist_planes_ref[scale];

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        unsafe {
            fused::fused_features_kernel_persist::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                ArrayArg::from_raw_parts(planes[0].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[1].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[2].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[3].clone(), plane_len),
                s.padded_w,
                s.h,
                s.n_strips,
                s.partials_f64_off as u32,
                s.partials_max_off as u32,
                pad_total as u32,
                y_body_start,
                y_body_end,
            );
        }
    }

    /// Launch the masked + IW pooling kernel for one scale. Requires
    /// the persist planes to already be filled by
    /// `launch_blur_and_features_persist`.
    ///
    /// Uses the **strip-local** variant
    /// [`masked_iw_strip::masked_iw_strip_kernel`] which reproduces
    /// CPU's per-strip semantics. Per the 2026-05-17 principled
    /// per-channel H-blur redesign, the activity computation uses
    /// on-the-fly `H_blur(src[channel])` instead of any cross-channel
    /// cascade — see the kernel's docstring and zensim's
    /// `docs/PRINCIPLED_ACTIVITY.md`. No carryover plane needed.
    fn launch_masked_iw(&self, scale: usize) {
        let s = &self.scales[scale];
        let h = s.h;
        self.launch_masked_iw_with_body(scale, 0, h);
    }

    fn launch_masked_iw_with_body(&self, scale: usize, y_body_start: u32, y_body_end: u32) {
        const TX: u32 = 64;

        let s = &self.scales[scale];
        let pad_total = s.n_padded;
        let plane_len = pad_total * 3;
        let planes = &self.persist_planes_ref[scale];

        let cube_x = s.padded_w.div_ceil(TX).max(1);
        let cube_count = CubeCount::Static(cube_x, s.n_strips_ext, 3);
        let cube_dim = CubeDim::new_3d(TX, 1, 1);
        let do_ext = if self.regime.needs_masked() {
            1u32
        } else {
            0u32
        };
        let do_iw = if self.regime.needs_iw() { 1u32 } else { 0u32 };
        unsafe {
            masked_iw_strip::masked_iw_strip_kernel::launch_unchecked::<R>(
                &self.client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(s.ref_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[0].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[1].clone(), pad_total),
                ArrayArg::from_raw_parts(s.ref_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(s.dis_xyb[2].clone(), pad_total),
                ArrayArg::from_raw_parts(planes[0].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[1].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[2].clone(), plane_len),
                ArrayArg::from_raw_parts(planes[3].clone(), plane_len),
                ArrayArg::from_raw_parts(self.partials_ext_f64.clone(), self.partials_ext_f64_len),
                s.padded_w,
                s.h,
                s.n_strips_ext,
                pad_total as u32,
                s.partials_ext_off as u32,
                do_ext,
                do_iw,
                y_body_start,
                y_body_end,
            );
        }
    }

    /// On-device reduction of the masked + IW per-(col, strip, ch)
    /// partials into per-(scale, ch, slot) finals. One launch per
    /// scale, grid `(36, 1, 1)` (= 3 channels × 12 slots).
    fn launch_reduction_ext(&self) {
        let n_scales = self.scales.len();
        let cube_dim = CubeDim::new_1d(256);
        let n_finals = n_scales * 3 * 12;
        for s in 0..n_scales {
            let sc = &self.scales[s];
            let pw = sc.padded_w as usize;
            // Use the **extended kernel's** strip count, NOT the basic
            // kernel's GPU-occupancy strip count. The strip-local
            // masked + IW kernel writes `pw × n_strips_ext` partials
            // per channel.
            let ns = sc.n_strips_ext as usize;
            let n_partials_per_ch = (pw * ns) as u32;
            let cube_count = CubeCount::Static(36, 1, 1);
            unsafe {
                reduce::reduce_ext_kernel::launch_unchecked::<R>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(
                        self.partials_ext_f64.clone(),
                        self.partials_ext_f64_len,
                    ),
                    ArrayArg::from_raw_parts(self.finals_ext_f64.clone(), n_finals),
                    sc.partials_ext_off as u32,
                    n_partials_per_ch,
                    (s * 3 * 12) as u32,
                );
            }
        }
    }

    /// On-device reduction of per-(col, strip, channel) partials into
    /// per-(scale, channel, slot) finals. One launch per scale (4
    /// total at SCALES = 4); each launch fires a 60-cube grid (3
    /// channels × 20 slot kinds) so the entire pyramid's reduction
    /// costs ~4 launches plus 240 fast cube-level tree reduces.
    fn launch_reduction(&self) {
        let n_scales = self.scales.len();
        let n_finals_f64 = n_scales * 3 * 17;
        let n_finals_max = n_scales * 3 * 3;
        let cube_dim = CubeDim::new_1d(256);
        for s in 0..n_scales {
            let sc = &self.scales[s];
            let pw = sc.padded_w as usize;
            let ns = sc.n_strips as usize;
            let n_partials_per_ch = (pw * ns) as u32;
            let cube_count = CubeCount::Static(60, 1, 1);
            unsafe {
                reduce::reduce_scale_kernel::launch_unchecked::<R>(
                    &self.client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts(self.partials_f64.clone(), self.partials_f64_len),
                    ArrayArg::from_raw_parts(self.partials_max.clone(), self.partials_max_len),
                    ArrayArg::from_raw_parts(self.finals_f64.clone(), n_finals_f64),
                    ArrayArg::from_raw_parts(self.finals_max.clone(), n_finals_max),
                    sc.partials_f64_off as u32,
                    sc.partials_max_off as u32,
                    n_partials_per_ch,
                    (s * 3 * 17) as u32,
                    (s * 3 * 3) as u32,
                );
            }
        }
    }

    /// Host-side fold of one (scale, channel)'s partials. The kernel
    /// laid out per (col, strip, channel) slots; we sum across cols ×
    /// strips for this channel.
    #[allow(dead_code)]
    fn fold_partials(
        &self,
        scale: usize,
        channel: usize,
        parts_all: &[f64],
        maxs_all: &[f32],
    ) -> ([f64; 17], [f32; 3]) {
        let s = &self.scales[scale];
        let pw = s.padded_w as usize;
        let ns = s.n_strips as usize;
        // Slot index: ch × ns × pw + strip × pw + col.
        let f64_ch_base = s.partials_f64_off + channel * ns * pw * 17;
        let max_ch_base = s.partials_max_off + channel * ns * pw * 3;

        let mut sums = [0.0_f64; 17];
        let mut peaks = [0.0_f32; 3];
        for strip in 0..ns {
            let f64_strip_base = f64_ch_base + strip * pw * 17;
            let max_strip_base = max_ch_base + strip * pw * 3;
            for col in 0..pw {
                for i in 0..17 {
                    sums[i] += parts_all[f64_strip_base + col * 17 + i];
                }
                for i in 0..3 {
                    let v = maxs_all[max_strip_base + col * 3 + i];
                    if v > peaks[i] {
                        peaks[i] = v;
                    }
                }
            }
        }
        (sums, peaks)
    }
}

// ============================================================
// Diffmap + linear-planes + warm-ref-diffmap API (Phase 1).
//
// Public entry points mirror cvvdp-gpu's `8b658b40` commit shape:
//
// - score_with_diffmap                      (sRGB-byte inputs)
// - score_with_warm_ref_diffmap             (sRGB-byte distorted vs warm ref)
// - score_from_linear_planes                (3 linear-f32 planes × 2)
// - score_from_linear_planes_with_diffmap   (+ diffmap)
// - warm_reference_from_linear_planes
// - score_from_linear_planes_with_warm_ref
// - score_from_linear_planes_with_warm_ref_diffmap
//
// Phase 1 strategy: the diffmap *production* is delegated to the
// canonical CPU `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`
// path. The GPU path remains the feature/scoring fast-path; diffmap
// uses CPU. See `crate::kernels::diffmap` module docs +
// `docs/DIFFMAP_DIVERGENCES.md` for rationale.
// ============================================================

/// Lazy CPU diffmap state — built on first use of any Phase 1 diffmap
/// or linear-planes entry-point on a given [`Zensim<R>`].
///
/// This avoids paying any cost (driver build, plane allocs) for callers
/// who only use the scalar/feature-vector GPU fast path. Once
/// constructed, `cpu_zensim` is reused across all subsequent calls;
/// the `warm_ref` field caches the most recent reference so the
/// `set_reference → score_with_warm_ref_diffmap × N` warm pattern
/// works the same way it does for the GPU feature pipeline.
struct DiffmapState {
    /// CPU `Zensim` driver bound to the profile recorded on first use.
    cpu_zensim: ZensimCpu,
    /// Profile bound at first call. All subsequent diffmap calls on
    /// this state assume the same profile — switching profiles
    /// mid-stream is a misuse pattern (would silently change the
    /// scalar score direction). Phase 1 ships with the default
    /// `PreviewV0_3` per RFC #2 §5. Kept as a diagnostic + Phase 1b
    /// hook for surfacing the active profile to opaque-API callers.
    #[allow(dead_code)]
    profile: ZensimProfile,
    /// Cached precomputed reference. Populated by `warm_reference_from_linear_planes`,
    /// the warm-ref-diffmap entry-points, or by the one-shot diffmap
    /// entry-points (which build a fresh PrecomputedReference per
    /// call). Cleared whenever `Zensim::set_reference` runs on the GPU
    /// side so the two reference states never drift.
    warm_ref: Option<PrecomputedReference>,
    /// Distorted-side `f32` linear-RGB scratch planes used by the
    /// sRGB-byte path's host LUT decode. Three planes of length
    /// `width * height`; resized to fit on first use, reused across
    /// calls.
    dist_linear_planes: [Vec<f32>; 3],
    /// Reference-side `f32` linear-RGB scratch planes for the
    /// sRGB-byte one-shot path. Reused across calls.
    ref_linear_planes: [Vec<f32>; 3],
}

impl DiffmapState {
    fn new(profile: ZensimProfile) -> Self {
        Self {
            cpu_zensim: ZensimCpu::new(profile),
            profile,
            warm_ref: None,
            dist_linear_planes: [Vec::new(), Vec::new(), Vec::new()],
            ref_linear_planes: [Vec::new(), Vec::new(), Vec::new()],
        }
    }
}

/// Host-side sRGB-u8 → linear-f32 lookup table.
///
/// Standard sRGB EOTF (IEC 61966-2-1), evaluated at 256 integer code
/// points. Output in `[0, 1]` linear light. Identical to the LUT
/// `zensim` itself uses internally; we duplicate it here so the GPU
/// crate doesn't need to import `zensim`'s private helpers.
fn srgb_lut_256() -> [f32; 256] {
    let mut lut = [0.0_f32; 256];
    for (i, slot) in lut.iter_mut().enumerate() {
        let v = (i as f32) / 255.0;
        *slot = if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        };
    }
    lut
}

/// Decode a packed sRGB-u8 RGB buffer (length `3 * width * height`,
/// row-major `[r0, g0, b0, r1, g1, b1, ...]`) into three planar f32
/// linear-RGB planes, tight stride = `width`. Each plane is resized
/// to `width * height` and overwritten.
fn srgb_u8_to_linear_planes_tight(
    src: &[u8],
    width: usize,
    height: usize,
    out: &mut [Vec<f32>; 3],
    lut: &[f32; 256],
) {
    let n = width * height;
    for plane in out.iter_mut() {
        if plane.len() < n {
            plane.resize(n, 0.0);
        } else {
            plane.truncate(n);
        }
    }
    debug_assert_eq!(src.len(), n * 3);
    let (r_plane, gb_planes) = out.split_first_mut().unwrap();
    let (g_plane, b_planes) = gb_planes.split_first_mut().unwrap();
    let b_plane = &mut b_planes[0];
    for (i, pixel) in src.chunks_exact(3).enumerate() {
        r_plane[i] = lut[pixel[0] as usize];
        g_plane[i] = lut[pixel[1] as usize];
        b_plane[i] = lut[pixel[2] as usize];
    }
}

/// Whether the **pure-GPU diffmap kernel chain** (Phase 1b) is enabled
/// for diffmap production.
///
/// Default: `false` — the production diffmap path keeps the Phase 1
/// CPU pipeline (score + diffmap both from
/// `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar`). This
/// is a **deliberate honest-stop**: the GPU diffmap *kernels* are
/// validated bit-close to the CPU canonical (≤ 2.08e-4 pointwise, see
/// `tests/cpu_gpu_diffmap_parity.rs`), BUT they cannot yet replace the
/// CPU path on the wall axis because the SCALAR SCORE must still come
/// from the CPU canonical: the GPU-feature → V0_3 MLP score path is
/// catastrophically wrong on the pinned zensim 0.3.0 (a pre-existing
/// WithIw-feature / V0_3-MLP-sensitivity bug — see
/// `docs/DIFFMAP_DIVERGENCES.md` §2 + §9). Running GPU-diffmap +
/// CPU-score is strictly slower than Phase 1 (CPU does the full
/// pipeline anyway), so the default stays on CPU until chunk N+1 fixes
/// the GPU score path.
///
/// Setting env `ZENSIM_GPU_DIFFMAP=1` routes diffmap production through
/// the GPU kernel chain (score still from CPU canonical) — for A/B
/// validation + as the wired infrastructure the score-path fix
/// unblocks.
fn gpu_diffmap_enabled() -> bool {
    std::env::var_os("ZENSIM_GPU_DIFFMAP").is_some_and(|v| v == "1")
}

/// Map a [`ZensimError`] from the CPU side into the GPU crate's
/// [`Error`] type. Unmappable variants fold to `InvalidImageSize`
/// with the error preserved in `Display`.
fn map_zensim_error(err: ZensimError) -> Error {
    match err {
        ZensimError::ImageTooSmall => Error::InvalidImageSize,
        ZensimError::DimensionMismatch => Error::DimensionMismatch {
            expected: 0,
            got: 0,
        },
        // Other variants (InvalidStride, InvalidDataLength, ImageTooLarge,
        // UnsupportedFormat, etc.) are surfaced as InvalidImageSize for
        // the buttloop's purposes. The buttloop never inspects the inner
        // ZensimError; it logs + falls back to butteraugli.
        _ => Error::InvalidImageSize,
    }
}

impl<R: Runtime> Zensim<R> {
    /// One-shot zensim score from sRGB-byte inputs that ALSO fills a
    /// per-pixel diffmap.
    ///
    /// Returns the **butteraugli-direction normalized score**:
    /// `(100.0 - zensim_score)` clamped to `[0, 100]`. zensim's
    /// native score is 0..100 where 100 = identical; the buttloop
    /// contract is smaller=better at the trait boundary (per RFC #1
    /// §1.1), so this entry-point normalizes for the buttloop.
    ///
    /// `diffmap_out` is overwritten via `clear` + extend so callers
    /// can reuse a long-lived `Vec`. On return,
    /// `diffmap_out.len() == width * height` and the values are
    /// non-negative f32, row-major (no padding), zensim-native
    /// units (per-pixel SSIM error fused across pyramid scales).
    ///
    /// **Phase 1**: the diffmap is produced by zensim's CPU pipeline.
    /// See `docs/DIFFMAP_DIVERGENCES.md`.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if either input buffer's length
    ///   doesn't match `width × height × 3`.
    /// - [`Error::InvalidImageSize`] on dispatch / shape failure
    ///   inside zensim CPU.
    pub fn score_with_diffmap(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;

        let w = self.width as usize;
        let h = self.height as usize;
        let lut = srgb_lut_256();

        self.ensure_diffmap_state();
        // Borrow state in a confined scope so we can call self.* helpers.
        {
            let state = self.diffmap_state.as_mut().expect("ensured");
            srgb_u8_to_linear_planes_tight(ref_srgb, w, h, &mut state.ref_linear_planes, &lut);
            srgb_u8_to_linear_planes_tight(dist_srgb, w, h, &mut state.dist_linear_planes, &lut);
        }

        if !gpu_diffmap_enabled() {
            // Default: Phase 1 CPU score + CPU diffmap (zero regression).
            return self.compute_diffmap_from_linear_planes_into(w, h, true, diffmap_out);
        }

        // Opt-in (ZENSIM_GPU_DIFFMAP=1): GPU diffmap + CPU score. Move
        // the decoded linear planes out of the diffmap state into owned
        // scratch so the borrow doesn't alias `self` during the GPU call.
        let (rp, dp) = {
            let state = self.diffmap_state.as_mut().expect("ensured");
            (
                [
                    core::mem::take(&mut state.ref_linear_planes[0]),
                    core::mem::take(&mut state.ref_linear_planes[1]),
                    core::mem::take(&mut state.ref_linear_planes[2]),
                ],
                [
                    core::mem::take(&mut state.dist_linear_planes[0]),
                    core::mem::take(&mut state.dist_linear_planes[1]),
                    core::mem::take(&mut state.dist_linear_planes[2]),
                ],
            )
        };
        let score = self.gpu_diffmap_linear_into(
            Some([&rp[0], &rp[1], &rp[2]]),
            [&dp[0], &dp[1], &dp[2]],
            true,
            diffmap_out,
        );
        // Return the decoded planes to the state (reuse buffers).
        if let Some(state) = self.diffmap_state.as_mut() {
            state.ref_linear_planes = rp;
            state.dist_linear_planes = dp;
        }
        score
    }

    /// Warm-ref variant of [`Self::score_with_diffmap`]. Requires a
    /// prior [`Self::set_reference`] OR
    /// [`Self::warm_reference_from_linear_planes`] call — both prime
    /// the diffmap state's cached `PrecomputedReference`.
    ///
    /// Skips the REF-side sRGB decode + XYB pyramid build. Mirrors the
    /// cvvdp-gpu warm-ref-diffmap pattern.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if `dist_srgb.len() != width × height × 3`.
    /// - [`Error::NoCachedReference`] if no warm reference is cached
    ///   in the diffmap state.
    /// - [`Error::InvalidImageSize`] on dispatch / shape failure
    ///   inside zensim CPU.
    pub fn score_with_warm_ref_diffmap(
        &mut self,
        dist_srgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.check_dims(dist_srgb)?;

        let w = self.width as usize;
        let h = self.height as usize;
        let lut = srgb_lut_256();

        self.ensure_diffmap_state();
        {
            let state = self.diffmap_state.as_mut().expect("ensured");
            if state.warm_ref.is_none() {
                return Err(Error::NoCachedReference);
            }
            srgb_u8_to_linear_planes_tight(dist_srgb, w, h, &mut state.dist_linear_planes, &lut);
        }

        if !gpu_diffmap_enabled() {
            // Default: Phase 1 CPU score + CPU diffmap (zero regression).
            return self.compute_diffmap_from_linear_planes_into(w, h, false, diffmap_out);
        }

        // Opt-in: warm the GPU scratch reference (if not already) from
        // the cached ref linear planes that `set_reference` decoded,
        // then run the GPU diffmap on the distorted planes.
        self.ensure_gpu_diffmap_scratch()?;
        let gpu_ref_warmed = self
            .gpu_diffmap_scratch
            .as_ref()
            .map(|s| s.inner.has_reference)
            .unwrap_or(false);
        if !gpu_ref_warmed {
            let rp = {
                let state = self.diffmap_state.as_mut().expect("ensured");
                [
                    core::mem::take(&mut state.ref_linear_planes[0]),
                    core::mem::take(&mut state.ref_linear_planes[1]),
                    core::mem::take(&mut state.ref_linear_planes[2]),
                ]
            };
            let res = self.gpu_diffmap_warm_ref_linear(&rp[0], &rp[1], &rp[2]);
            if let Some(state) = self.diffmap_state.as_mut() {
                state.ref_linear_planes = rp;
            }
            res?;
        }

        let dp = {
            let state = self.diffmap_state.as_mut().expect("ensured");
            [
                core::mem::take(&mut state.dist_linear_planes[0]),
                core::mem::take(&mut state.dist_linear_planes[1]),
                core::mem::take(&mut state.dist_linear_planes[2]),
            ]
        };
        let score =
            self.gpu_diffmap_linear_into(None, [&dp[0], &dp[1], &dp[2]], false, diffmap_out);
        if let Some(state) = self.diffmap_state.as_mut() {
            state.dist_linear_planes = dp;
        }
        score
    }

    /// One-shot zensim score from 6 planar linear-RGB f32 buffers
    /// (3 reference + 3 distorted; each plane has length
    /// `width * height` tight-strided).
    ///
    /// Returns the **butteraugli-direction normalized score**
    /// (smaller=better, identity→0) per [`Self::score_with_diffmap`].
    ///
    /// Mirrors butteraugli-gpu's W44-PHASE3-B4 + cvvdp-gpu's
    /// `score_from_linear_planes` shape — skips the host-side
    /// sRGB-u8 pack + GPU-side sRGB→linear kernel. Direct path from
    /// caller-owned linear-light buffers to zensim's CPU pipeline.
    ///
    /// **Phase 1**: Both score and (when requested) diffmap come from
    /// zensim's CPU pipeline. The GPU-side feature pipeline is NOT
    /// used on the linear-planes entry-points in Phase 1 — Phase 1b
    /// will fuse the linear-planes upload with the existing
    /// `srgb_to_positive_xyb_kernel` → XYB pyramid build.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane's length differs
    ///   from `width × height`.
    /// - [`Error::InvalidImageSize`] on shape failure.
    pub fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        let w = self.width as usize;
        let h = self.height as usize;
        validate_linear_planes(ref_r, ref_g, ref_b, w, h)?;
        validate_linear_planes(dist_r, dist_g, dist_b, w, h)?;

        self.ensure_diffmap_state();
        let stride = w;
        let state = self.diffmap_state.as_mut().expect("ensured");
        let pre = state
            .cpu_zensim
            .precompute_reference_linear_planar([ref_r, ref_g, ref_b], w, h, stride)
            .map_err(map_zensim_error)?;
        // Reuse the CPU `compute_with_ref` (no diffmap) — it is the
        // cheapest path that returns a `ZensimResult`.
        // We replicate the CPU `Zensim::compute_with_ref` plane-path
        // by going through `precompute_reference_linear_planar` + the
        // public `compute_with_ref_and_diffmap_linear_planar` API; we
        // just don't pass the diffmap output through to the caller.
        let res = state
            .cpu_zensim
            .compute_with_ref_and_diffmap_linear_planar(
                &pre,
                [dist_r, dist_g, dist_b],
                w,
                h,
                stride,
                DiffmapOptions::default(),
            )
            .map_err(map_zensim_error)?;
        Ok(normalize_zensim_score(res.score()))
    }

    /// As [`Self::score_from_linear_planes`], plus a per-pixel diffmap.
    #[allow(clippy::too_many_arguments)]
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
        let w = self.width as usize;
        let h = self.height as usize;
        validate_linear_planes(ref_r, ref_g, ref_b, w, h)?;
        validate_linear_planes(dist_r, dist_g, dist_b, w, h)?;

        if !gpu_diffmap_enabled() {
            // Default: Phase 1 CPU score + CPU diffmap (zero regression).
            self.ensure_diffmap_state();
            let stride = w;
            let state = self.diffmap_state.as_mut().expect("ensured");
            let pre = state
                .cpu_zensim
                .precompute_reference_linear_planar([ref_r, ref_g, ref_b], w, h, stride)
                .map_err(map_zensim_error)?;
            let res = state
                .cpu_zensim
                .compute_with_ref_and_diffmap_linear_planar(
                    &pre,
                    [dist_r, dist_g, dist_b],
                    w,
                    h,
                    stride,
                    DiffmapOptions::default(),
                )
                .map_err(map_zensim_error)?;
            write_diffmap_into(diffmap_out, res.diffmap());
            return Ok(normalize_zensim_score(res.score()));
        }

        // Opt-in: GPU diffmap + CPU canonical score.
        self.gpu_diffmap_linear_into(
            Some([ref_r, ref_g, ref_b]),
            [dist_r, dist_g, dist_b],
            true,
            diffmap_out,
        )
    }

    /// Warm the diffmap-state's `PrecomputedReference` from three
    /// planar linear-RGB f32 buffers.
    ///
    /// Subsequent [`Self::score_from_linear_planes_with_warm_ref`] /
    /// [`Self::score_from_linear_planes_with_warm_ref_diffmap`] reuse
    /// the cached reference. Mirrors cvvdp-gpu's
    /// `warm_reference_from_linear_planes` + invalidation rules:
    /// any call to `set_reference` on the GPU side also clears the
    /// warm-ref to keep the two reference states aligned.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane's length differs
    ///   from `width × height`.
    /// - [`Error::InvalidImageSize`] on shape failure inside zensim
    ///   CPU.
    pub fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        let w = self.width as usize;
        let h = self.height as usize;
        validate_linear_planes(ref_r, ref_g, ref_b, w, h)?;

        self.ensure_diffmap_state();
        let stride = w;
        {
            let state = self.diffmap_state.as_mut().expect("ensured");
            let pre = state
                .cpu_zensim
                .precompute_reference_linear_planar([ref_r, ref_g, ref_b], w, h, stride)
                .map_err(map_zensim_error)?;
            state.warm_ref = Some(pre);
        }
        // Phase 1b (opt-in): also warm the GPU diffmap scratch's
        // reference XYB pyramid so subsequent warm-ref-diffmap calls
        // run the GPU kernel chain. Skipped by default to avoid the
        // GPU scratch alloc on the CPU-default path.
        if gpu_diffmap_enabled() {
            self.gpu_diffmap_warm_ref_linear(ref_r, ref_g, ref_b)?;
        }
        Ok(())
    }

    /// Score a distorted planar linear-RGB f32 candidate against the
    /// warm-cached diffmap-state reference. Returns the
    /// butteraugli-direction normalized score.
    ///
    /// # Errors
    ///
    /// - [`Error::DimensionMismatch`] if any plane's length differs
    ///   from `width × height`.
    /// - [`Error::NoCachedReference`] if the warm reference is
    ///   missing.
    /// - [`Error::InvalidImageSize`] on shape failure.
    pub fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        let w = self.width as usize;
        let h = self.height as usize;
        validate_linear_planes(dist_r, dist_g, dist_b, w, h)?;

        self.ensure_diffmap_state();
        let stride = w;
        let state = self.diffmap_state.as_mut().expect("ensured");
        let pre = state.warm_ref.as_ref().ok_or(Error::NoCachedReference)?;
        let res = state
            .cpu_zensim
            .compute_with_ref_and_diffmap_linear_planar(
                pre,
                [dist_r, dist_g, dist_b],
                w,
                h,
                stride,
                DiffmapOptions::default(),
            )
            .map_err(map_zensim_error)?;
        Ok(normalize_zensim_score(res.score()))
    }

    /// As [`Self::score_from_linear_planes_with_warm_ref`], plus a
    /// per-pixel diffmap.
    pub fn score_from_linear_planes_with_warm_ref_diffmap(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let w = self.width as usize;
        let h = self.height as usize;
        validate_linear_planes(dist_r, dist_g, dist_b, w, h)?;

        if !gpu_diffmap_enabled() {
            // Default: Phase 1 CPU score + CPU diffmap (zero regression).
            self.ensure_diffmap_state();
            let stride = w;
            let state = self.diffmap_state.as_mut().expect("ensured");
            let pre = state.warm_ref.as_ref().ok_or(Error::NoCachedReference)?;
            let res = state
                .cpu_zensim
                .compute_with_ref_and_diffmap_linear_planar(
                    pre,
                    [dist_r, dist_g, dist_b],
                    w,
                    h,
                    stride,
                    DiffmapOptions::default(),
                )
                .map_err(map_zensim_error)?;
            write_diffmap_into(diffmap_out, res.diffmap());
            return Ok(normalize_zensim_score(res.score()));
        }

        // Opt-in: GPU warm-ref diffmap + CPU canonical score. Requires
        // the GPU scratch reference to have been warmed via
        // [`Self::warm_reference_from_linear_planes`].
        let gpu_ref_warmed = self
            .gpu_diffmap_scratch
            .as_ref()
            .map(|s| s.inner.has_reference)
            .unwrap_or(false);
        if !gpu_ref_warmed {
            return Err(Error::NoCachedReference);
        }
        self.gpu_diffmap_linear_into(None, [dist_r, dist_g, dist_b], false, diffmap_out)
    }

    // ─────────────────────── private helpers ───────────────────────

    /// Allocate `diffmap_state` lazily on first use. Default profile is
    /// `ZensimProfile::A` per `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §5.
    /// Callers needing a non-default profile should construct via the
    /// `ZensimOpaque::with_profile` path (Phase 1b will surface a
    /// `Zensim::with_diffmap_profile` setter if needed).
    //
    // Migrated from the `PreviewV0_3` deprecated alias to the canonical
    // `ZensimProfile::A`. Provably score-neutral: in zensim
    // `Self::A | Self::PreviewV0_3 => &PROFILE_A` — both resolve to the
    // identical `PROFILE_A` params (only the profile *name* string
    // differs). Upstream dropped the `PreviewV0_3` alias, so `A` is also
    // the only form that compiles against current zensim.
    fn ensure_diffmap_state(&mut self) {
        if self.diffmap_state.is_none() {
            self.diffmap_state = Some(DiffmapState::new(ZensimProfile::A));
        }
    }

    /// Shared implementation for sRGB-byte diffmap paths: assumes the
    /// state's linear planes have been populated host-side.
    ///
    /// When `build_fresh_ref` is true, this function builds a fresh
    /// `PrecomputedReference` from `state.ref_linear_planes` and
    /// discards it after use (one-shot diffmap). When false, it uses
    /// the warm-cached reference (which must exist; caller's
    /// responsibility to check).
    ///
    /// **Phase 1b**: this is the DEFAULT diffmap production path
    /// (CPU score + CPU diffmap, zero regression vs Phase 1). The
    /// opt-in GPU diffmap path ([`Self::gpu_diffmap_linear_into`],
    /// `ZENSIM_GPU_DIFFMAP=1`) keeps the CPU score but produces the
    /// diffmap on GPU — see [`gpu_diffmap_enabled`] for why the GPU
    /// path is opt-in (broken GPU V0_3 score on the pinned zensim).
    fn compute_diffmap_from_linear_planes_into(
        &mut self,
        w: usize,
        h: usize,
        build_fresh_ref: bool,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let stride = w;
        let state = self.diffmap_state.as_mut().expect("ensured by caller");

        let ref_views: [&[f32]; 3] = [
            &state.ref_linear_planes[0],
            &state.ref_linear_planes[1],
            &state.ref_linear_planes[2],
        ];
        let dist_views: [&[f32]; 3] = [
            &state.dist_linear_planes[0],
            &state.dist_linear_planes[1],
            &state.dist_linear_planes[2],
        ];

        let res = if build_fresh_ref {
            let pre = state
                .cpu_zensim
                .precompute_reference_linear_planar(ref_views, w, h, stride)
                .map_err(map_zensim_error)?;
            state
                .cpu_zensim
                .compute_with_ref_and_diffmap_linear_planar(
                    &pre,
                    dist_views,
                    w,
                    h,
                    stride,
                    DiffmapOptions::default(),
                )
                .map_err(map_zensim_error)?
        } else {
            let pre = state.warm_ref.as_ref().ok_or(Error::NoCachedReference)?;
            state
                .cpu_zensim
                .compute_with_ref_and_diffmap_linear_planar(
                    pre,
                    dist_views,
                    w,
                    h,
                    stride,
                    DiffmapOptions::default(),
                )
                .map_err(map_zensim_error)?
        };
        write_diffmap_into(diffmap_out, res.diffmap());
        Ok(normalize_zensim_score(res.score()))
    }
}

// ============================================================
// Pure-GPU diffmap pipeline (Phase 1b).
//
// Replaces the Phase 1 CPU-delegation for the default
// `DiffmapOptions` path. An inner WithIw-regime `Zensim<R>` runs the
// full 372-feature GPU pipeline (XYB pyramid + per-scale persist
// planes + masked/IW + reduction). From one GPU feature pass we get:
//   • the per-scale mu1/mu2/ssq/s12 persist planes → fed to the
//     chunk-1/2 diffmap kernels to produce the multi-scale SSIM
//     diffmap on-device;
//   • the 372 features → fed to the CPU V0_3 MLP for the scalar
//     score (bit-equivalent to `score_features_with_profile_and_codec`
//     per `tests/opaque_default_weights_v03.rs`, which is the same
//     score the Phase 1 CPU path produced).
//
// See `docs/DIFFMAP_DIVERGENCES.md` §2 for the strategy + the residual
// CPU dependency (the V0_3 MLP forward pass, ~µs).
// ============================================================

impl<R: Runtime> Zensim<R> {
    /// Upload three tight `width × height` linear-RGB f32 planes into
    /// the scale-0 XYB planes (ref or dist side) via
    /// [`color::linear_to_positive_xyb_kernel`], then build the 2× box
    /// downscale pyramid (reusing [`Self::run_xyb_pyramid`]'s downscale
    /// loop). After this returns, `scales[*].ref_xyb` (or `dis_xyb`)
    /// hold the positive-XYB pyramid for the linear input.
    ///
    /// Mirrors [`Self::run_xyb_pyramid`] but ingests linear f32 planes
    /// instead of decoding packed sRGB-u8 through the LUT — bit-exact
    /// to CPU zensim's `linear_to_positive_xyb_planar_into` path.
    fn run_xyb_pyramid_linear(&mut self, is_ref: bool, r: &[f32], g: &[f32], b: &[f32]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let n = w * h;
        // Upload the three tight planes (pinned for fast DMA).
        let r_h = self.client.create_from_slice_pinned(f32::as_bytes(&r[..n]));
        let g_h = self.client.create_from_slice_pinned(f32::as_bytes(&g[..n]));
        let b_h = self.client.create_from_slice_pinned(f32::as_bytes(&b[..n]));

        let s0 = &self.scales[0];
        let xyb = if is_ref { &s0.ref_xyb } else { &s0.dis_xyb };
        let absorbance_bias_neg = -color::cbrtf_fast_host(color::K_B0);
        // Mirror table: same fallback the sRGB path uses (a 1-element
        // dummy when there are no pad columns — the kernel never reads
        // it then because `x < width` for every pixel).
        let mirror_arg = match s0.mirror_offsets.as_ref() {
            Some(mo) => (mo.clone(), s0.pad_count as usize),
            None => (self.srgb_lut.clone(), 1),
        };
        let this_padded_pixels = (s0.padded_w as usize) * h;
        unsafe {
            color::linear_to_positive_xyb_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(this_padded_pixels),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(r_h, n),
                ArrayArg::from_raw_parts(g_h, n),
                ArrayArg::from_raw_parts(b_h, n),
                ArrayArg::from_raw_parts(mirror_arg.0, mirror_arg.1),
                ArrayArg::from_raw_parts(xyb[0].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[1].clone(), s0.n_padded),
                ArrayArg::from_raw_parts(xyb[2].clone(), s0.n_padded),
                self.width,
                self.height,
                s0.padded_w,
                absorbance_bias_neg,
            );
        }
        // Build the rest of the pyramid via 2× planar downscale —
        // identical to the tail of `run_xyb_pyramid`.
        for s in 1..self.scales.len() {
            let prev = &self.scales[s - 1];
            let curr = &self.scales[s];
            let prev_xyb = if is_ref { &prev.ref_xyb } else { &prev.dis_xyb };
            let curr_xyb = if is_ref { &curr.ref_xyb } else { &curr.dis_xyb };
            let curr_active = (curr.padded_w as usize) * (curr.h as usize);
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(curr_active),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(prev_xyb[0].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[1].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(prev_xyb[2].clone(), prev.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[0].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[1].clone(), curr.n_padded),
                    ArrayArg::from_raw_parts(curr_xyb[2].clone(), curr.n_padded),
                    prev.padded_w,
                    prev.h,
                    curr.padded_w,
                    curr.h,
                );
            }
        }
    }

    /// Run the WithIw feature pipeline assuming both `ref_xyb` and
    /// `dis_xyb` pyramids are already built (by
    /// [`Self::run_xyb_pyramid_linear`]). Writes the per-scale persist
    /// planes (mu1/mu2/ssq/s12) AND reduces the 372-feature vector.
    /// Requires the WithIw regime (persist planes allocated).
    ///
    /// Returns the packed 372-feature vector. Mirrors the body of
    /// [`Self::compute_with_reference_vec`] from the post-pyramid
    /// point onward.
    fn compute_withiw_features_from_built_xyb(&mut self) -> Vec<f64> {
        debug_assert!(self.regime.needs_extended_kernel());
        let n_scales = self.scales.len();

        for s in 0..n_scales {
            self.launch_blur_and_features_persist(s);
        }
        for s in 0..n_scales {
            self.launch_masked_iw(s);
        }
        self.launch_reduction();
        self.launch_reduction_ext();

        let f64_bytes = self
            .client
            .read_one(self.finals_f64.clone())
            .expect("read finals_f64");
        let max_bytes = self
            .client
            .read_one(self.finals_max.clone())
            .expect("read finals_max");
        let finals_f64 = f64::from_bytes(&f64_bytes);
        let finals_max = f32::from_bytes(&max_bytes);
        let ext_bytes = self
            .client
            .read_one(self.finals_ext_f64.clone())
            .expect("read finals_ext_f64");
        let finals_ext_f64 = f64::from_bytes(&ext_bytes);

        let mut scale_image_h: [u32; SCALES] = [0; SCALES];
        let mut hs = self.height;
        for sh in scale_image_h.iter_mut().take(n_scales) {
            *sh = hs;
            hs = hs.div_ceil(2);
        }
        self.pack_feature_vector(
            finals_f64,
            finals_max,
            finals_ext_f64,
            &scale_image_h[..n_scales],
        )
    }

    /// Run the GPU diffmap kernel chain on the per-scale persist planes
    /// currently resident in `persist_planes_ref` (written by the last
    /// [`Self::compute_withiw_features_from_built_xyb`] call). Produces
    /// the multi-scale weighted-SSIM diffmap on-device, trims pad
    /// columns, and reads it back into `diffmap_out` (length
    /// `width × height`).
    ///
    /// `scale_dm[s]` are per-scale weighted-SSIM scratch planes;
    /// `acc` is the base-resolution accumulator; `out` is the trimmed
    /// destination. `per_scale_w[s] = [w_x, w_y, w_b]` and
    /// `scale_blend[s]` come from
    /// [`diffmap::trained_multiscale_ssim_weights_default`].
    #[allow(clippy::too_many_arguments)]
    fn run_gpu_diffmap_chain(
        &self,
        scale_dm: &[cubecl::server::Handle],
        acc: &cubecl::server::Handle,
        out: &cubecl::server::Handle,
        per_scale_w: &[[f32; 3]],
        scale_blend: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) {
        let n_scales = self.scales.len();
        let width = self.width;
        let height = self.height;
        let base_padded_w = self.scales[0].padded_w;
        let base_n = (base_padded_w as usize) * (height as usize);

        // Step 1: zero the base accumulator.
        unsafe {
            diffmap::diffmap_zero_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(base_n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(acc.clone(), base_n),
                base_n as u32,
            );
        }

        // Step 2: per scale, compute the weighted-SSIM plane then
        // upsample-add it into the base accumulator with the per-scale
        // blend weight. factor = 1 << scale (scale 0 = identity copy).
        for s in 0..n_scales {
            let blend = scale_blend.get(s).copied().unwrap_or(0.0);
            if blend <= 0.0 {
                continue;
            }
            let sc = &self.scales[s];
            let pad_total = sc.n_padded;
            let plane_len = pad_total * 3;
            let planes = &self.persist_planes_ref[s];
            let w = per_scale_w.get(s).copied().unwrap_or([1.0 / 3.0; 3]);

            // Per-scale weighted SSIM error → scale_dm[s].
            unsafe {
                diffmap::per_scale_weighted_ssim_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(pad_total),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(planes[0].clone(), plane_len),
                    ArrayArg::from_raw_parts(planes[1].clone(), plane_len),
                    ArrayArg::from_raw_parts(planes[2].clone(), plane_len),
                    ArrayArg::from_raw_parts(planes[3].clone(), plane_len),
                    ArrayArg::from_raw_parts(scale_dm[s].clone(), pad_total),
                    sc.padded_w,
                    sc.h,
                    pad_total as u32,
                    w[0],
                    w[1],
                    w[2],
                );
            }

            // Upsample-add scale_dm[s] into the base accumulator.
            unsafe {
                diffmap::pow2x_upsample_add_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(base_n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(scale_dm[s].clone(), pad_total),
                    ArrayArg::from_raw_parts(acc.clone(), base_n),
                    sc.padded_w,
                    sc.h,
                    base_padded_w,
                    height,
                    s as u32,
                    blend,
                );
            }
        }

        // Step 3: trim padded accumulator → tight width × height output.
        let tight_n = (width as usize) * (height as usize);
        if base_padded_w == width {
            // No pad columns — read the accumulator directly (its first
            // `width × height` slots are exactly the tight output).
            let bytes = self.client.read_one(acc.clone()).expect("read acc");
            let data = f32::from_bytes(&bytes);
            diffmap_out.clear();
            diffmap_out.extend_from_slice(&data[..tight_n]);
        } else {
            unsafe {
                diffmap::diffmap_trim_padded_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(tight_n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(acc.clone(), base_n),
                    ArrayArg::from_raw_parts(out.clone(), tight_n),
                    width,
                    base_padded_w,
                    height,
                );
            }
            let bytes = self.client.read_one(out.clone()).expect("read out");
            let data = f32::from_bytes(&bytes);
            diffmap_out.clear();
            diffmap_out.extend_from_slice(&data[..tight_n]);
        }
    }

    /// Lazily allocate the [`GpuDiffmapScratch`] (inner WithIw
    /// pipeline + accumulator + per-scale dm planes + linear upload
    /// planes + cached trained weights). One-time alloc; reused across
    /// calls so the warm buttloop pays it once.
    ///
    /// The inner pipeline binds to `ZensimProfile::A` (== V0_3) — the
    /// same profile the Phase 1 CPU path used.
    fn ensure_gpu_diffmap_scratch(&mut self) -> Result<()> {
        if self.gpu_diffmap_scratch.is_some() {
            return Ok(());
        }
        let profile = ZensimProfile::A;
        // Inner WithIw pipeline on the same client + dims. WithIw is
        // required so the persist planes (mu1/mu2/ssq/s12) the diffmap
        // kernels consume are allocated, AND so the 372-feature vector
        // for the V0_3 MLP score is produced.
        let inner = Box::new(Zensim::new_with_regime(
            self.client.clone(),
            self.width,
            self.height,
            ZensimFeatureRegime::WithIw,
        )?);

        let n_scales = inner.scales.len();
        let base_padded_w = inner.scales[0].padded_w as usize;
        let height = inner.height as usize;
        let width = inner.width as usize;
        let base_n = base_padded_w * height;
        let tight_n = width * height;

        let scale_dm: Vec<cubecl::server::Handle> = inner
            .scales
            .iter()
            .map(|s| alloc_empty_f32(&self.client, s.n_padded))
            .collect();
        let acc = alloc_empty_f32(&self.client, base_n);
        let out = alloc_empty_f32(&self.client, tight_n);
        let dist_lin = [
            alloc_empty_f32(&self.client, tight_n),
            alloc_empty_f32(&self.client, tight_n),
            alloc_empty_f32(&self.client, tight_n),
        ];
        let ref_lin = [
            alloc_empty_f32(&self.client, tight_n),
            alloc_empty_f32(&self.client, tight_n),
            alloc_empty_f32(&self.client, tight_n),
        ];

        // Precompute the default-options Trained multi-scale weights
        // from the canonical V0_2 weight table (the diffmap weighting
        // is profile-independent — it's pure SSIM from
        // WEIGHTS_PREVIEW_V0_2, matching CPU `trained_multiscale_weights`
        // with edge_mse + hf disabled).
        let weights_f64: Vec<f64> = crate::weights::WEIGHTS_PREVIEW_V0_2.to_vec();
        let (per_scale_w, scale_blend) =
            diffmap::trained_multiscale_ssim_weights_default(&weights_f64, n_scales);

        self.gpu_diffmap_scratch = Some(GpuDiffmapScratch {
            inner,
            scale_dm,
            acc,
            out,
            dist_lin,
            ref_lin,
            per_scale_w,
            scale_blend,
            cpu_scorer: ZensimCpu::new(profile),
            cpu_ref: None,
            profile,
        });
        Ok(())
    }

    /// Pure-GPU diffmap + score from linear-RGB f32 planes.
    ///
    /// `build_ref`: when true, builds the reference XYB pyramid from
    /// `ref_planes` (one-shot / cold path). When false, reuses the
    /// inner pipeline's cached reference pyramid (warm-ref path) —
    /// `ref_planes` is ignored.
    ///
    /// Returns the butteraugli-direction normalized score; fills
    /// `diffmap_out` with the `width × height` multi-scale SSIM diffmap.
    fn gpu_diffmap_linear_into(
        &mut self,
        ref_planes: Option<[&[f32]; 3]>,
        dist_planes: [&[f32]; 3],
        build_ref: bool,
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.ensure_gpu_diffmap_scratch()?;
        let (w, h) = (self.width as usize, self.height as usize);

        // Pull the scratch out so we can mutably drive the inner
        // pipeline + read the cached weights without aliasing `self`.
        let mut scratch = self.gpu_diffmap_scratch.take().expect("ensured");

        // ── Reference setup ──
        // GPU side: build the reference XYB pyramid on the inner
        // pipeline (cold path). CPU side: build / reuse the cached
        // PrecomputedReference for the SCORE.
        if build_ref {
            let rp = ref_planes.ok_or(Error::NoCachedReference)?;
            scratch
                .inner
                .run_xyb_pyramid_linear(true, rp[0], rp[1], rp[2]);
            scratch.inner.has_reference = true;
            // CPU reference for the score path.
            let pre = match scratch.cpu_scorer.precompute_reference_linear_planar(
                [rp[0], rp[1], rp[2]],
                w,
                h,
                w,
            ) {
                Ok(p) => p,
                Err(e) => {
                    self.gpu_diffmap_scratch = Some(scratch);
                    return Err(map_zensim_error(e));
                }
            };
            scratch.cpu_ref = Some(pre);
        } else if !scratch.inner.has_reference || scratch.cpu_ref.is_none() {
            self.gpu_diffmap_scratch = Some(scratch);
            return Err(Error::NoCachedReference);
        }

        // ── Score (CPU canonical) ──
        // The score MUST come from the CPU canonical path: the
        // GPU-feature → V0_3 MLP path is catastrophically wrong on the
        // pinned zensim 0.3.0 (pre-existing WithIw-feature / V0_3-MLP
        // parity bug, documented in DIFFMAP_DIVERGENCES.md §9). The CPU
        // call also yields the canonical diffmap, but we DISCARD it —
        // the diffmap is produced on GPU below (the future win, once
        // the score-path bug is fixed in chunk N+1, is to drop this CPU
        // call entirely).
        let raw_score = {
            let cpu_ref = scratch.cpu_ref.as_ref().expect("set above");
            match scratch
                .cpu_scorer
                .compute_with_ref_and_diffmap_linear_planar(
                    cpu_ref,
                    dist_planes,
                    w,
                    h,
                    w,
                    DiffmapOptions::default(),
                ) {
                Ok(res) => res.score(),
                Err(e) => {
                    self.gpu_diffmap_scratch = Some(scratch);
                    return Err(map_zensim_error(e));
                }
            }
        };

        // ── Diffmap (GPU) ──
        // Build distorted XYB pyramid + WithIw features on the inner
        // pipeline (writes the mu1/mu2/ssq/s12 persist planes the
        // diffmap kernels consume). We do NOT use the 372-feature
        // vector for scoring (see above) — only the persist planes
        // matter here. The reduction still runs (it's cheap) so the
        // inner pipeline state stays self-consistent.
        scratch
            .inner
            .run_xyb_pyramid_linear(false, dist_planes[0], dist_planes[1], dist_planes[2]);
        let _features = scratch.inner.compute_withiw_features_from_built_xyb();

        {
            let inner = scratch.inner.as_ref();
            inner.run_gpu_diffmap_chain(
                &scratch.scale_dm,
                &scratch.acc,
                &scratch.out,
                &scratch.per_scale_w,
                &scratch.scale_blend,
                diffmap_out,
            );
        }

        let _ = (&scratch.ref_lin, &scratch.dist_lin, scratch.profile);

        self.gpu_diffmap_scratch = Some(scratch);
        Ok(normalize_zensim_score(raw_score))
    }

    /// Warm the GPU diffmap reference pyramid + the CPU score reference
    /// from linear-RGB planes.
    fn gpu_diffmap_warm_ref_linear(&mut self, r: &[f32], g: &[f32], b: &[f32]) -> Result<()> {
        self.ensure_gpu_diffmap_scratch()?;
        let (w, h) = (self.width as usize, self.height as usize);
        let mut scratch = self.gpu_diffmap_scratch.take().expect("ensured");
        scratch.inner.run_xyb_pyramid_linear(true, r, g, b);
        scratch.inner.has_reference = true;
        let pre = match scratch
            .cpu_scorer
            .precompute_reference_linear_planar([r, g, b], w, h, w)
        {
            Ok(p) => p,
            Err(e) => {
                self.gpu_diffmap_scratch = Some(scratch);
                return Err(map_zensim_error(e));
            }
        };
        scratch.cpu_ref = Some(pre);
        self.gpu_diffmap_scratch = Some(scratch);
        Ok(())
    }
}

/// Convert zensim's higher-is-better 0..100 score to a
/// butteraugli-direction normalised score (smaller=better,
/// identity→0). Per RFC #1 §1.1 + `RFC_ZENSIM_BUTTLOOP_AUDIT.md` §1.1.
fn normalize_zensim_score(score: f64) -> f32 {
    let v = (100.0 - score).clamp(0.0, 100.0);
    v as f32
}

/// Copy the CPU diffmap's `&[f32]` into the caller-owned `Vec<f32>`
/// via clear + extend (preserves capacity).
fn write_diffmap_into(diffmap_out: &mut Vec<f32>, src: &[f32]) {
    diffmap_out.clear();
    diffmap_out.extend_from_slice(src);
}

/// Validate three linear-RGB f32 planes against the expected
/// `width × height`. Each plane must contain AT LEAST `w * h`
/// elements (zensim's CPU API accepts extra padding, but we
/// reject under-supply to surface the contract failure cleanly).
fn validate_linear_planes(r: &[f32], g: &[f32], b: &[f32], w: usize, h: usize) -> Result<()> {
    let expected = w * h;
    for plane in [r, g, b] {
        if plane.len() < expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: plane.len(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod diffmap_helpers_tests {
    use super::*;

    #[test]
    fn normalize_zensim_score_identity_zero() {
        assert_eq!(normalize_zensim_score(100.0), 0.0);
    }

    #[test]
    fn normalize_zensim_score_max_clamps_at_100() {
        assert_eq!(normalize_zensim_score(0.0), 100.0);
        // Below zero (shouldn't happen but defend) clamps at 100.
        assert_eq!(normalize_zensim_score(-1.0), 100.0);
    }

    #[test]
    fn normalize_zensim_score_above_100_clamps_at_zero() {
        // Numerical noise from f32 path: zensim can return 100.0001;
        // we clamp at 0 so the buttloop's accept_bound math doesn't
        // see a negative.
        assert_eq!(normalize_zensim_score(100.0001), 0.0);
    }

    #[test]
    fn write_diffmap_into_clears_and_extends() {
        let mut buf = vec![99.0, 98.0, 97.0];
        let src = vec![1.0, 2.0];
        write_diffmap_into(&mut buf, &src);
        assert_eq!(buf, vec![1.0, 2.0]);
    }

    #[test]
    fn validate_linear_planes_rejects_short_plane() {
        let r = vec![0.0; 100];
        let g = vec![0.0; 100];
        let b = vec![0.0; 99]; // short
        let err = validate_linear_planes(&r, &g, &b, 10, 10).unwrap_err();
        match err {
            Error::DimensionMismatch { expected, got } => {
                assert_eq!(expected, 100);
                assert_eq!(got, 99);
            }
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn srgb_lut_endpoints_match_spec() {
        let lut = srgb_lut_256();
        assert_eq!(lut[0], 0.0);
        assert!((lut[255] - 1.0).abs() < 1e-6, "lut[255] = {}", lut[255]);
    }

    #[test]
    fn srgb_u8_to_linear_planes_separates_channels() {
        let lut = srgb_lut_256();
        // 2×1 image: [(255, 0, 128), (0, 255, 64)]
        let src = vec![255, 0, 128, 0, 255, 64];
        let mut planes = [Vec::new(), Vec::new(), Vec::new()];
        srgb_u8_to_linear_planes_tight(&src, 2, 1, &mut planes, &lut);
        assert_eq!(planes[0].len(), 2);
        assert_eq!(planes[1].len(), 2);
        assert_eq!(planes[2].len(), 2);
        assert!((planes[0][0] - lut[255]).abs() < 1e-7);
        assert!((planes[0][1] - lut[0]).abs() < 1e-7);
        assert!((planes[1][0] - lut[0]).abs() < 1e-7);
        assert!((planes[1][1] - lut[255]).abs() < 1e-7);
        assert!((planes[2][0] - lut[128]).abs() < 1e-7);
        assert!((planes[2][1] - lut[64]).abs() < 1e-7);
    }
}
