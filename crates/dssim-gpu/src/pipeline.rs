//! DSSIM pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` into the 5-scale DSSIM algorithm.
//! Public entry points:
//!
//! - [`Dssim::new`] + [`Dssim::compute`] — score one image pair from sRGB.
//! - [`Dssim::set_reference`] + [`Dssim::compute_with_reference`] — cache
//!   reference-side state and score many distorted images against it.
//!
//! Algorithm (per scale, faithful to `dssim-cuda`'s `compute_sync_inner`):
//!   1. Linear-RGB pyramid: sRGB→linear at scale 0, then 2×2 box
//!      downscale planar between scales.
//!   2. Per scale: linear RGB → planar Lab.
//!   3. Pre-blur the chroma (a, b) twice each — matches `dssim-core`'s
//!      chroma blur step.
//!   4. For every channel ∈ {L, a, b}:
//!      - mu = blur(blur(channel))
//!      - sq_blur = blur(blur_squared(channel))
//!      - cross = blur(blur_product(ref_channel, dis_channel))
//!   5. Fused 15-input SSIM map.
//!   6. Σ ssim → mean_ssim; avg = mean_ssim ^ (0.5 ^ scale_idx);
//!      Σ |ssim - avg| → mad; score_for_scale = 1 - mad.
//!   7. Final = Σ (score_for_scale · weight) / Σ weight, then
//!      `1 / ssim - 1` to convert SSIM → DSSIM.
//!
//! Buffer aliasing follows the `blur_plane_via(src, dst, scratch)` rule
//! from `CUBECL_GOTCHAS.md` — `temp1` is the only mutable scratch and
//! it's never the source or dest of the second pass.

use cubecl::prelude::*;

use crate::kernels::{blur, downscale, lab, reduction, srgb, ssim};
use crate::{Error, GpuDssimResult, NUM_SCALES, Result, SCALE_WEIGHTS};

/// Per-scale buffer set. All planes are `width × height` planar f32.
/// We keep `temp1` and `temp2` separate so the two-pass blur sequence
/// (`blur → blur` for mu, `blur_squared → blur` for variance,
/// `blur_product → blur` for covariance) never aliases its source and
/// destination through the same scratch.
struct Scale {
    width: u32,
    height: u32,
    n: usize,

    /// Linear RGB at this scale (planar).
    ref_lin: [cubecl::server::Handle; 3],
    dis_lin: [cubecl::server::Handle; 3],

    /// Lab planes (post conversion + chroma pre-blur).
    ref_lab: [cubecl::server::Handle; 3],
    dis_lab: [cubecl::server::Handle; 3],

    /// Cached reference-side fully-blurred outputs (used by
    /// `compute_with_reference`).
    ref_mu: [cubecl::server::Handle; 3],
    ref_sq_blur: [cubecl::server::Handle; 3],

    /// Distorted-side and cross outputs (overwritten per call).
    dis_mu: [cubecl::server::Handle; 3],
    dis_sq_blur: [cubecl::server::Handle; 3],
    cross_blur: [cubecl::server::Handle; 3],

    /// Two scratch planes for the two-pass-blur shape.
    temp1: cubecl::server::Handle,
    temp2: cubecl::server::Handle,

    /// Per-pixel SSIM map (also reused as input for `abs_diff_scalar`).
    ssim_map: cubecl::server::Handle,
    /// Per-pixel |ssim - avg| map (input to the second reduction).
    mad_map: cubecl::server::Handle,
}

fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
}

/// Five-scale pyramid dims for a `width × height` image. Mirrors
/// [`Dssim::new`]'s dim computation: `div_ceil(2)` per descent,
/// clamped at 8 per axis. Shared with [`RefFullState`] allocation in
/// strip-mode `set_reference`.
fn full_pyramid_dims(width: u32, height: u32) -> Vec<(u32, u32)> {
    let mut dims = Vec::with_capacity(NUM_SCALES);
    let mut w = width;
    let mut h = height;
    for _ in 0..NUM_SCALES {
        dims.push((w, h));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        if w < 8 {
            w = 8;
        }
        if h < 8 {
            h = 8;
        }
    }
    dims
}

fn alloc_3<R: Runtime>(client: &ComputeClient<R>, n: usize) -> [cubecl::server::Handle; 3] {
    [
        alloc_plane(client, n),
        alloc_plane(client, n),
        alloc_plane(client, n),
    ]
}

impl Scale {
    fn new<R: Runtime>(client: &ComputeClient<R>, width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            n,
            ref_lin: alloc_3(client, n),
            dis_lin: alloc_3(client, n),
            ref_lab: alloc_3(client, n),
            dis_lab: alloc_3(client, n),
            ref_mu: alloc_3(client, n),
            ref_sq_blur: alloc_3(client, n),
            dis_mu: alloc_3(client, n),
            dis_sq_blur: alloc_3(client, n),
            cross_blur: alloc_3(client, n),
            temp1: alloc_plane(client, n),
            temp2: alloc_plane(client, n),
            ssim_map: alloc_plane(client, n),
            mad_map: alloc_plane(client, n),
        }
    }
}

/// Per-strip geometry, recomputed each iteration of the strip loop.
/// All row coordinates are at scale 0.
#[derive(Debug, Clone, Copy)]
struct StripPlan {
    /// First image row read into this strip (inclusive). Equals
    /// `body_start - halo`, clamped to 0 at the top.
    #[allow(dead_code)] // useful for debugging; logged via Debug
    read_start_in_image: u32,
    /// One past the last image row read into this strip. Equals
    /// `body_end + halo`, clamped to image_h at the bottom.
    #[allow(dead_code)]
    read_end_in_image: u32,
    /// Offset of the body region within the strip buffer (in scale-0
    /// rows). 0 for the top strip, == halo for interior strips.
    body_offset_in_strip_at_0: u32,
    /// Height of the body region within the strip at scale 0. Equals
    /// `h_body` for full strips; may be less for the bottom strip.
    body_h_in_strip_at_0: u32,
    /// Actual strip-buffer rows in use this iteration (may be less
    /// than the allocated strip_h for top/bottom strips). Tail rows
    /// beyond this are zeroed.
    strip_h_actual: u32,
}

/// Compute the per-scale `[start_idx, end_idx)` element range that
/// covers `plan`'s body rows at scale `s`, given the scale's buffer
/// dimensions.
///
/// The pyramid uses `div_ceil(2)` per descent (with each axis clamped
/// to ≥ 8). Body offset and body height at scale `s` are
/// `body_offset_at_0.div_ceil(2^s)` and
/// `body_h_at_0.div_ceil(2^s)`. We clamp the end against the scale's
/// allocated `height` so the kernel never reads past the buffer.
fn scale_row_range(
    plan: &StripPlan,
    scale: usize,
    width_at_s: u32,
    height_at_s: u32,
) -> (u32, u32) {
    let divisor = 1u32 << (scale as u32);
    let body_offset_at_s = plan.body_offset_in_strip_at_0 / divisor;
    let body_h_at_s = plan.body_h_in_strip_at_0.div_ceil(divisor).max(1);
    let body_end_at_s = (body_offset_at_s + body_h_at_s).min(height_at_s);
    let start_idx = body_offset_at_s * width_at_s;
    let end_idx = body_end_at_s * width_at_s;
    (start_idx, end_idx)
}

/// Per-scale full-image-sized reference state cached in strip mode
/// after a successful [`Dssim::set_reference`] call. The strip walker
/// in [`Dssim::compute_with_reference`] copies the per-strip row
/// region from these full-image buffers into the strip-sized
/// [`Scale::ref_lab`] / [`Scale::ref_mu`] / [`Scale::ref_sq_blur`]
/// buffers before running cross_blur + ssim_map for that strip.
///
/// Dimensions per scale match the **full image** pyramid (mirroring
/// [`Dssim::new`]'s scale grid), not the strip buffer.
///
/// Mode E (task #73) cached-ref-in-strip: keeps the ref-side mu /
/// sigma at full image size on device, the dist side walks in
/// strip-sized working buffers reusing the same cached ref state
/// across many distortions.
struct RefFullState {
    /// Per-scale full-image dimensions `(width_at_s, height_at_s)`.
    /// Mirrors the pyramid grid that [`Dssim::new`] would allocate
    /// for the same `(image_w, image_h)`.
    dims: Vec<(u32, u32)>,
    /// Per-scale Lab planes (post chroma pre-blur). Each entry is
    /// `[L, a, b]` of `width_at_s × height_at_s` f32 planes.
    ref_lab: Vec<[cubecl::server::Handle; 3]>,
    /// Per-scale ref_mu planes (output of `blur(blur(ref_lab))`).
    ref_mu: Vec<[cubecl::server::Handle; 3]>,
    /// Per-scale ref_sq_blur planes (output of
    /// `blur(blur_squared(ref_lab))`).
    ref_sq_blur: Vec<[cubecl::server::Handle; 3]>,
}

/// Strip-processing configuration. Present iff the `Dssim<R>` was
/// constructed via [`Dssim::new_strip`]; absent in whole-image mode.
///
/// In strip mode the per-scale buffers are sized for one (body + halo)
/// strip rather than the full image. `compute_stripped` walks the
/// image as `n_strips` vertical strips, populating the scale-0 linear
/// planes with the strip's rows (plus halo rows from neighboring
/// strips) and accumulating per-scale `Σ ssim` / `Σ mad` over the
/// body region only.
#[derive(Debug, Clone)]
struct StripConfig {
    /// Full image width (kept on `Dssim::width` too — duplicated for
    /// clarity at strip-driver call sites).
    image_w: u32,
    /// Full image height — the strip driver loops over this.
    image_h: u32,
    /// Number of body rows per strip at scale 0. Must be divisible by
    /// `2^(NUM_SCALES - 1)` so the body region maps cleanly through
    /// every pyramid level.
    h_body: u32,
    /// Halo rows above + below the body at scale 0. Sized to cover the
    /// worst-case blur reach across all 5 scales.
    halo: u32,
    /// Strip total height = `h_body + 2 * halo`, clamped at the
    /// top/bottom strips so we never index past the image boundary.
    /// (The Scale buffers are allocated for the worst-case
    /// `h_body + 2*halo`; smaller boundary strips just leave the
    /// extra rows unused for that strip iteration.)
    #[allow(dead_code)]
    strip_h: u32,
}

/// Per-instance allocations + per-call orchestration of the DSSIM
/// pipeline. Construct once for a given resolution; reuse across many
/// image pairs of that resolution.
pub struct Dssim<R: Runtime> {
    client: ComputeClient<R>,
    /// Sub-minimum reflect-pad plan: holds both the caller's logical
    /// extent (reported by `dimensions()` in whole-image mode) and the
    /// padded extent (`max(requested, MIN_PAD_DIM)` per axis) that all
    /// buffers/scales build for. Inputs reflect-pad logical → padded at
    /// the upload boundary. No-op at ≥8px (so the opaque shim's own pad,
    /// which already builds the inner pipeline at ≥8px, stays the sole
    /// padder when called through it).
    pad: zenmetrics_gpu_core::PadPlan,
    /// `n_pixels` at scale 0 (padded extent).
    n: usize,

    /// sRGB u8 staging (uploaded as u32 because WGSL has no `u8` type).
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    // T_x.O (2026-05-17): `pack_scratch: Vec<u32>` removed. The
    // upload path now packs u8×3 → u32 directly into the pinned
    // staging buffer reserved per call (`client.reserve_staging`),
    // collapsing two host-side passes (pack to pageable + memcpy to
    // pinned) into one. Mirrors butter T_x.O (10a5b996).
    /// Per-scale buffer sets.
    scales: Vec<Scale>,

    /// Per-thread (or per-slot, in fast mode) reduction partials.
    /// Layout: 2 reductions per scale × `NUM_SCALES` scales.
    /// Slot encoding: `scale * 2 + (0 = ssim_sum, 1 = mad_sum)`.
    partials: cubecl::server::Handle,
    /// Final per-slot scalars folded by the finalizer kernel.
    sums: cubecl::server::Handle,

    has_reference: bool,

    /// Full-image cached reference state (mode E). Populated only by
    /// [`Dssim::set_reference`] when in strip mode; `None` for
    /// whole-image instances (which cache directly in
    /// `self.scales[*].ref_*`). Per-strip dist scoring copies the
    /// relevant row range from here into the strip-sized Scale
    /// buffers.
    ref_full: Option<RefFullState>,

    /// `Some(_)` iff constructed via [`Dssim::new_strip`]. Drives the
    /// strip-loop in [`Dssim::compute_stripped`] / friends.
    strip_config: Option<StripConfig>,

    /// Per-scale "actual data height" for the current strip
    /// iteration. Empty outside strip mode; populated by
    /// [`Dssim::set_strip_data_extents`] at the top of each strip
    /// iteration to make blur / downscale kernels clamp at the
    /// strip's data edge instead of the (larger) buffer edge.
    strip_data_h: Vec<u32>,
}

const NUM_SLOTS: usize = NUM_SCALES * 2; // 10
const PARTIALS_LEN: usize = NUM_SLOTS * reduction::PARTIALS_PER_REDUCTION;
const SUMS_LEN: usize = NUM_SLOTS;

/// Minimum per-axis dimension DSSIM's 5-scale pyramid needs (the typed
/// pipeline rejects `< 8×8`). Sub-`MIN_PAD_DIM` requests are
/// reflect(mirror)-padded up to it (shared [`zenmetrics_gpu_core::PadPlan`]),
/// so the typed `Dssim<R>` — like `DssimOpaque` — scores down to 1×1
/// instead of erroring. NO-OP at ≥8px.
pub const MIN_PAD_DIM: u32 = 8;

impl<R: Runtime> Dssim<R> {
    /// Allocate every per-instance buffer for the given image size.
    /// Returns `Err(InvalidImageSize)` for images smaller than 8×8.
    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        // Reflect-pad sub-MIN_PAD_DIM up to the pyramid floor: build for
        // the padded extent, store the plan, report logical via dims(),
        // reflect-pad inputs at the upload boundary. 0-dim → padded 0 →
        // rejected below. NO-OP at ≥8px.
        let pad = zenmetrics_gpu_core::PadPlan::to_min(width, height, MIN_PAD_DIM);
        let (width, height) = pad.padded();
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let n = (width as usize) * (height as usize);

        // Pyramid dims: 5 levels, halving each axis. dssim-cuda clamps
        // each axis to a minimum of 8 to keep the SSIM window valid;
        // we mirror that.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = width;
        let mut h = height;
        for _ in 0..NUM_SCALES {
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            if w < 8 {
                w = 8;
            }
            if h < 8 {
                h = 8;
            }
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        // T4.L (2026-05-16): one packed u32 per pixel (R | G<<8 | B<<16).
        // Length = n, not n × 3. Cuts upload 3× per call.
        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));

        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        let sums = client.create_from_slice(f32::as_bytes(&[0.0_f32; SUMS_LEN]));

        Ok(Self {
            client,
            pad,
            n,
            src_u8_a,
            src_u8_b,
            scales,
            partials,
            sums,
            has_reference: false,
            ref_full: None,
            strip_config: None,
            strip_data_h: Vec::new(),
        })
    }

    /// Unified [`MemoryMode`](crate::MemoryMode) constructor.
    /// dssim-gpu is **NOT strip-preferred** — Strip is 2-5× slower
    /// than Full on this crate, so Auto picks Full whenever it fits
    /// the VRAM cap. Set `ZENMETRICS_VRAM_CAP_BYTES` to override the
    /// 8 GB default cap.
    ///
    /// - `MemoryMode::Auto`: picks Full when it fits, Strip when
    ///   not, errors with [`crate::Error::TooBigForFull`] when
    ///   neither fits.
    /// - `MemoryMode::Full`: constructs via [`Self::new`].
    /// - `MemoryMode::Strip { h_body }`: constructs via
    ///   [`Self::new_strip`]. `h_body == None` auto-sizes within the
    ///   cap; `Some(n)` pins to `n` (must satisfy `new_strip`'s
    ///   pyramid-alignment contract).
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

    /// Strip-processing constructor. Allocates working set for a
    /// single `(h_body + 2 * halo) × image_w` strip rather than the
    /// full image; reuses across strips for the same
    /// `(image_w, image_h, h_body)` configuration.
    ///
    /// Halo size is fixed at 256 rows per side — enough to cover the
    /// worst-case 4-pass 3×3 blur reach across all 5 pyramid scales,
    /// plus the 2×2 box-downscale halo accumulated through each
    /// pyramid descent (see `STRIP_PROCESSING.md` for the math).
    ///
    /// Constraints:
    /// - `image_w` and `image_h` ≥ 8 (same as whole-image path).
    /// - `h_body` must be a positive multiple of `2^(NUM_SCALES - 1)`
    ///   so the body region maps cleanly through every pyramid level.
    /// - If `h_body + 2 * halo >= image_h` the strip path degenerates
    ///   to a single full-image strip (still works, but skip strips
    ///   in that case — `new()` is cheaper).
    pub fn new_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
    ) -> Result<Self> {
        // Sub-MIN_PAD_DIM: strip mode is meaningless on a tiny image —
        // build the Full pipeline, which reflect-pads to the floor.
        // (0-dim falls through to `new`'s rejection.)
        if image_w < MIN_PAD_DIM || image_h < MIN_PAD_DIM {
            return Self::new(client, image_w, image_h);
        }
        if image_w < 8 || image_h < 8 {
            return Err(Error::InvalidImageSize);
        }
        // Pyramid alignment: halve 4 times = factor 16.
        const PYRAMID_ALIGN: u32 = 1 << (NUM_SCALES as u32 - 1);
        if h_body == 0 || !h_body.is_multiple_of(PYRAMID_ALIGN) {
            return Err(Error::InvalidImageSize);
        }
        const HALO: u32 = 256;
        debug_assert_eq!(HALO % PYRAMID_ALIGN, 0);

        let strip_h = (h_body + 2 * HALO).min(image_h);
        // Reject configs where the strip would shrink below 8 at any
        // scale (matches `Dssim::new`'s "image must be at least 8×8"
        // contract applied to the strip buffers).
        {
            let mut h = strip_h;
            for _ in 0..NUM_SCALES {
                if h < 8 {
                    return Err(Error::InvalidImageSize);
                }
                h = h.div_ceil(2);
                if h < 8 {
                    h = 8;
                }
            }
        }
        let n = (image_w as usize) * (strip_h as usize);

        // Pyramid dims — width is image_w, height is strip_h.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = image_w;
        let mut h = strip_h;
        for _ in 0..NUM_SCALES {
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
            if w < 8 {
                w = 8;
            }
            if h < 8 {
                h = 8;
            }
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        // sRGB staging at strip-pixel count (n = image_w × strip_h).
        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));

        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        let sums = client.create_from_slice(f32::as_bytes(&[0.0_f32; SUMS_LEN]));

        Ok(Self {
            client,
            // ≥MIN here (sub-min routed to `new`): no-op plan whose
            // logical = the image dims. Strip-mode `dimensions()` reads
            // `strip_config` instead, so this is only carried for the
            // `pack_srgb_into_packed_u32_handle` / upload validators.
            pad: zenmetrics_gpu_core::PadPlan::to_min(image_w, image_h, MIN_PAD_DIM),
            n,
            src_u8_a,
            src_u8_b,
            scales,
            partials,
            sums,
            has_reference: false,
            ref_full: None,
            strip_config: Some(StripConfig {
                image_w,
                image_h,
                h_body,
                halo: HALO,
                strip_h,
            }),
            strip_data_h: vec![0; NUM_SCALES],
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        match &self.strip_config {
            Some(cfg) => (cfg.image_w, cfg.image_h),
            // Caller-requested (logical) extent. For a sub-MIN_PAD_DIM
            // whole-image instance this differs from the padded extent
            // the buffers were built for; for ≥8px it's the same.
            None => self.pad.logical(),
        }
    }

    /// Number of active pyramid scales.
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// `true` iff this `Dssim` was constructed via [`Self::new_strip`]
    /// and routes scoring through the strip-processing path.
    pub fn is_strip_mode(&self) -> bool {
        self.strip_config.is_some()
    }

    /// Effective height at scale `s` for kernel `width/height`
    /// arguments — the buffer height (`scales[s].height`) for
    /// whole-image mode, or the strip's actual data height at
    /// scale `s` (≤ buffer height) for strip mode.
    fn effective_h(&self, scale: usize) -> u32 {
        if self.strip_config.is_some() && !self.strip_data_h.is_empty() {
            self.strip_data_h[scale]
        } else {
            self.scales[scale].height
        }
    }

    /// Pack the caller's `width × height × 3` sRGB-u8 bytes into a
    /// `width × height` packed-u32 device handle (`R | G<<8 | B<<16`),
    /// using the same pinned-staging fast path the internal upload
    /// uses. See [`Self::compute_handles`] for the umbrella-batch
    /// rationale.
    ///
    /// Returns `Err(DimensionMismatch)` if `srgb.len() != width *
    /// height * 3`.
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        // Validate the LOGICAL extent, then reflect-pad up to the padded
        // extent so the packed handle matches the padded pipeline (and
        // `compute_handles` works on it unchanged). No-op at ≥8px.
        let expected = self.pad.logical_len(3);
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
        let plan = self.pad;
        let srgb = plan.pad(srgb, 3);
        let srgb: &[u8] = &srgb;
        let pinned_len = self.n * 4;
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

    /// Compute against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. Skips the
    /// `client.reserve_staging` + byte-pack work that
    /// [`Self::compute`] does internally, letting one
    /// `(ref, dist)` upload feed several metrics on the same client.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this `Dssim<R>`.
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuDssimResult> {
        self.install_packed_handle(true, ref_handle);
        self.install_packed_handle(false, dis_handle);
        self.compute_post_srgb()
    }

    /// Score one image pair, both sRGB packed RGB u8 of length
    /// `width × height × 3`.
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        if self.strip_config.is_some() {
            // Route image-sized buffers through the strip driver
            // automatically so backwards-compatibility is preserved
            // for callers that only know about `compute()`.
            return self.compute_stripped(ref_srgb, dist_srgb);
        }
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;

        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.upload_and_srgb_to_linear(false, dist_srgb);

        self.compute_post_srgb()
    }

    /// Pipeline body shared by [`Self::compute`] and
    /// [`Self::compute_handles`]. Both entry points have already
    /// populated `scales[0].{ref,dis}_lin` (via byte upload or via
    /// the install-packed-handle path) before reaching here.
    fn compute_post_srgb(&mut self) -> Result<GpuDssimResult> {
        // Build linear pyramid first (dssim-core downsamples in linear
        // RGB, NOT in Lab — Lab is per-scale).
        self.build_linear_pyramid(true);
        self.build_linear_pyramid(false);

        self.zero_partials();

        for s in 0..self.scales.len() {
            self.run_lab(s, true);
            self.run_lab(s, false);
            self.run_chroma_preblur(s, true);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, true);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
            self.run_sum_ssim(s);
        }
        self.run_finalize();

        // Read sums → compute mean_ssim per scale → re-launch
        // |ssim - avg| → second reduction → final score.
        let sums_host = self.read_sums();
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let ssim_sum = sums_host[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            self.run_abs_diff_and_sum(s, avg as f32);
        }
        self.run_finalize();
        let sums_host_pass2 = self.read_sums();

        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let mad_sum = sums_host_pass2[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        let dssim = ssim_to_dssim(ssim);

        Ok(GpuDssimResult { score: dssim })
    }

    /// Cache reference-side state for many comparisons against a fixed
    /// reference. Subsequent `compute_with_reference` calls skip the
    /// reference-side pyramid + Lab + reference-blur work.
    ///
    /// **Strip mode (mode E, task #73)**: when this `Dssim` was built
    /// via [`Self::new_strip`], `set_reference` allocates a separate
    /// full-image-sized [`RefFullState`] and populates it by running
    /// the ref-side pipeline on the full image (one-shot — the
    /// reference state is then reused across every per-strip distorted
    /// scoring call). [`Self::compute_with_reference`] in strip mode
    /// walks the dist image in strips and slices the relevant row
    /// range from the cached full-image ref state into the strip
    /// buffers before running cross_blur + ssim_map per strip.
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        if self.strip_config.is_some() {
            return self.set_reference_full_for_strip(ref_srgb);
        }
        self.check_dims(ref_srgb)?;
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.build_linear_pyramid(true);
        for s in 0..self.scales.len() {
            self.run_lab(s, true);
            self.run_chroma_preblur(s, true);
            self.run_blur_stats(s, true);
        }
        self.has_reference = true;
        Ok(())
    }

    /// Build per-scale full-image-sized ref state for a strip-mode
    /// `Dssim` (mode E, task #73). Allocates a sibling set of
    /// per-scale "full-image" `ref_lin` / `ref_lab` / `ref_mu` /
    /// `ref_sq_blur` planes and scratch (`temp1` / `temp2`), uploads
    /// the full reference image, runs the ref-side pipeline on those
    /// full-size planes, then stores `ref_lab` / `ref_mu` /
    /// `ref_sq_blur` into [`Self::ref_full`]. The transient
    /// `ref_lin` and scratch planes drop at the end of this call
    /// (their handles aren't retained), so the persistent footprint
    /// is `9 planes/scale × pyramid` plus the strip-sized working set
    /// the dist side already uses.
    fn set_reference_full_for_strip(&mut self, ref_srgb: &[u8]) -> Result<()> {
        let cfg = self
            .strip_config
            .as_ref()
            .expect("set_reference_full_for_strip requires strip-mode instance")
            .clone();
        let expected = (cfg.image_w as usize) * (cfg.image_h as usize) * 3;
        if ref_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_srgb.len(),
            });
        }

        // Full-image pyramid dims, matching `Dssim::new`'s grid.
        let dims = full_pyramid_dims(cfg.image_w, cfg.image_h);

        // Allocate per-scale full-image planes. Persistent: ref_lab,
        // ref_mu, ref_sq_blur. Transient (dropped at end): ref_lin,
        // temp1, temp2.
        let mut ref_lin: Vec<[cubecl::server::Handle; 3]> = Vec::with_capacity(dims.len());
        let mut ref_lab: Vec<[cubecl::server::Handle; 3]> = Vec::with_capacity(dims.len());
        let mut ref_mu: Vec<[cubecl::server::Handle; 3]> = Vec::with_capacity(dims.len());
        let mut ref_sq_blur: Vec<[cubecl::server::Handle; 3]> = Vec::with_capacity(dims.len());
        let mut temp1_full: Vec<cubecl::server::Handle> = Vec::with_capacity(dims.len());
        let mut temp2_full: Vec<cubecl::server::Handle> = Vec::with_capacity(dims.len());
        for &(w, h) in &dims {
            let n = (w as usize) * (h as usize);
            ref_lin.push(alloc_3(&self.client, n));
            ref_lab.push(alloc_3(&self.client, n));
            ref_mu.push(alloc_3(&self.client, n));
            ref_sq_blur.push(alloc_3(&self.client, n));
            temp1_full.push(alloc_plane(&self.client, n));
            temp2_full.push(alloc_plane(&self.client, n));
        }

        // Full-image scale-0 staging buffer (packed u32, one per
        // pixel). Sized for the full image, not the strip.
        let n0 = (cfg.image_w as usize) * (cfg.image_h as usize);
        let pinned_len = n0 * 4;
        let staging_handle = {
            let mut staging = self.client.reserve_staging(&[pinned_len]);
            let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
            {
                let dst: &mut [u8] = &mut bytes;
                debug_assert_eq!(dst.len(), pinned_len);
                for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(ref_srgb.chunks_exact(3)) {
                    chunk_out[0] = triple[0];
                    chunk_out[1] = triple[1];
                    chunk_out[2] = triple[2];
                    chunk_out[3] = 0;
                }
            }
            self.client.create(bytes)
        };

        // sRGB → linear on full-image scale-0 ref_lin.
        unsafe {
            srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n0),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(staging_handle.clone(), n0),
                ArrayArg::from_raw_parts(ref_lin[0][0].clone(), n0),
                ArrayArg::from_raw_parts(ref_lin[0][1].clone(), n0),
                ArrayArg::from_raw_parts(ref_lin[0][2].clone(), n0),
            );
        }

        // Build linear pyramid through full-image scales.
        for s in 1..dims.len() {
            let (prev_w, prev_h) = dims[s - 1];
            let (curr_w, curr_h) = dims[s];
            let n_prev = (prev_w as usize) * (prev_h as usize);
            let n_curr = (curr_w as usize) * (curr_h as usize);
            for ch in 0..3 {
                unsafe {
                    downscale::downscale_2x_plane_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n_curr),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(ref_lin[s - 1][ch].clone(), n_prev),
                        ArrayArg::from_raw_parts(ref_lin[s][ch].clone(), n_curr),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                    );
                }
            }
        }

        // Per-scale: linear → Lab → chroma pre-blur → blur_stats (mu,
        // sq_blur), all on full-image-sized planes.
        for s in 0..dims.len() {
            let (w, h) = dims[s];
            let n = (w as usize) * (h as usize);
            // run_lab equivalent
            unsafe {
                lab::linear_to_lab_planar_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(ref_lin[s][0].clone(), n),
                    ArrayArg::from_raw_parts(ref_lin[s][1].clone(), n),
                    ArrayArg::from_raw_parts(ref_lin[s][2].clone(), n),
                    ArrayArg::from_raw_parts(ref_lab[s][0].clone(), n),
                    ArrayArg::from_raw_parts(ref_lab[s][1].clone(), n),
                    ArrayArg::from_raw_parts(ref_lab[s][2].clone(), n),
                );
            }

            // Chroma pre-blur on channels 1, 2 — two-pass blur via
            // temp1_full, temp2_full, with dst == src.
            for ch in [1usize, 2] {
                self.full_blur_two_pass(
                    &ref_lab[s][ch],
                    &ref_lab[s][ch],
                    &temp1_full[s],
                    &temp2_full[s],
                    w,
                    h,
                );
            }

            // Blur stats: mu = blur(blur(ref_lab)); sq_blur =
            // blur(blur_squared(ref_lab)).
            for ch in 0..3 {
                // mu
                self.full_blur_two_pass(
                    &ref_lab[s][ch],
                    &ref_mu[s][ch],
                    &temp1_full[s],
                    &temp2_full[s],
                    w,
                    h,
                );
                // sq_blur: blur_squared(ref_lab[ch]) → temp1_full, then
                // blur → sq_blur_dst.
                unsafe {
                    blur::blur_squared_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(ref_lab[s][ch].clone(), n),
                        ArrayArg::from_raw_parts(temp1_full[s].clone(), n),
                        w,
                        h,
                    );
                    blur::blur_3x3_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(temp1_full[s].clone(), n),
                        ArrayArg::from_raw_parts(ref_sq_blur[s][ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }

        self.ref_full = Some(RefFullState {
            dims,
            ref_lab,
            ref_mu,
            ref_sq_blur,
        });
        self.has_reference = true;
        // ref_lin, temp1_full, temp2_full drop here.
        Ok(())
    }

    /// Full-image two-pass blur for mode E reference build. Mirrors
    /// [`Self::blur_two_pass`] but uses an explicit `(width, height)`
    /// pair (full-image dims at scale s) rather than a `Scale` index.
    fn full_blur_two_pass(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch_a: &cubecl::server::Handle,
        scratch_b: &cubecl::server::Handle,
        width: u32,
        height: u32,
    ) {
        let n = (width as usize) * (height as usize);
        unsafe {
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), n),
                ArrayArg::from_raw_parts(scratch_a.clone(), n),
                width,
                height,
            );
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_a.clone(), n),
                ArrayArg::from_raw_parts(scratch_b.clone(), n),
                width,
                height,
            );
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_b.clone(), n),
                ArrayArg::from_raw_parts(dst.clone(), n),
            );
        }
    }

    /// Drop any cached reference state.
    pub fn clear_reference(&mut self) {
        self.has_reference = false;
        self.ref_full = None;
    }

    pub fn has_reference(&self) -> bool {
        self.has_reference
    }

    /// Compute against the cached reference. Returns
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called.
    ///
    /// In strip mode (mode E, task #73), this routes to
    /// [`Self::compute_with_reference_stripped`] — the cached full-image
    /// ref state lives in [`Self::ref_full`] and is sliced per-strip
    /// while the dist side walks in strip-sized buffers.
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        if !self.has_reference {
            return Err(Error::NoCachedReference);
        }
        if self.strip_config.is_some() {
            return self.compute_with_reference_stripped(dist_srgb);
        }
        self.check_dims(dist_srgb)?;

        self.upload_and_srgb_to_linear(false, dist_srgb);
        self.build_linear_pyramid(false);

        self.zero_partials();

        for s in 0..self.scales.len() {
            self.run_lab(s, false);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
            self.run_sum_ssim(s);
        }
        self.run_finalize();

        let sums_host = self.read_sums();
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let ssim_sum = sums_host[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            self.run_abs_diff_and_sum(s, avg as f32);
        }
        self.run_finalize();
        let sums_host_pass2 = self.read_sums();

        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.scales[s].n as f64;
            let mad_sum = sums_host_pass2[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        Ok(GpuDssimResult {
            score: ssim_to_dssim(ssim),
        })
    }

    /// Mode-E (task #73) cached-ref + strip combined driver. Required
    /// path when [`Self::compute_with_reference`] is called on a
    /// strip-mode instance.
    ///
    /// Walks the distorted image in strips (same geometry as
    /// [`Self::compute_stripped`]); for each strip:
    /// 1. Copy the strip's row range from the full-image cached
    ///    [`RefFullState`] into the strip-sized `Scale.ref_lab` /
    ///    `Scale.ref_mu` / `Scale.ref_sq_blur` buffers.
    /// 2. Upload the dist strip; build dist linear pyramid.
    /// 3. Per scale: dist Lab + chroma pre-blur + blur_stats +
    ///    cross_blur + ssim_map, then sum body rows.
    /// 4. Read pass-1 sums, compute per-scale `avg`, then run a
    ///    second strip pass for `|ssim - avg|` body sums.
    ///
    /// Returns parity with the whole-image cached-ref path (within
    /// f32 reordering noise on the boundary 3x3 blur — body rows are
    /// bit-identical because the same kernels run on the same data;
    /// halo rows that affect blur reach are sliced from the full
    /// ref state and the dist side re-derives them per strip exactly
    /// as it would in whole-image mode).
    fn compute_with_reference_stripped(&mut self, dist_srgb: &[u8]) -> Result<GpuDssimResult> {
        let cfg = self
            .strip_config
            .as_ref()
            .expect("compute_with_reference_stripped requires strip-mode instance")
            .clone();
        if self.ref_full.is_none() {
            return Err(Error::NoCachedReference);
        }
        let expected = (cfg.image_w as usize) * (cfg.image_h as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }

        // Pass 1: Σ ssim per scale across body rows of every strip.
        self.zero_partials();
        for strip in 0..self.n_strips() {
            let plan = self.strip_plan(strip);
            self.set_strip_data_extents(&plan);
            // Install ref state for this strip — copies the relevant
            // row range from RefFullState into the strip-sized Scale
            // ref_lab / ref_mu / ref_sq_blur buffers.
            self.install_ref_state_for_strip(&plan);
            // Dist-side strip pipeline: upload, pyramid, Lab,
            // chroma pre-blur, blur_stats, cross_blur, ssim_map.
            self.upload_strip(false, dist_srgb, &plan);
            self.run_strip_dist_only_to_ssim_map();
            self.sum_ssim_body(&plan);
        }
        self.run_finalize();
        let pass1_sums = self.read_sums();

        // Compute per-scale avg from full-image mean_ssim.
        let mut avg_per_scale = [0.0_f32; NUM_SCALES];
        for s in 0..self.scales.len() {
            let n_pix = self.body_pixels_at_scale(s, &cfg) as f64;
            let ssim_sum = pass1_sums[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            avg_per_scale[s] = avg as f32;
        }

        // Pass 2: re-run dist strip pipeline, but this time compute
        // |ssim - avg| and sum the mad map over body rows.
        self.zero_partials();
        for strip in 0..self.n_strips() {
            let plan = self.strip_plan(strip);
            self.set_strip_data_extents(&plan);
            self.install_ref_state_for_strip(&plan);
            self.upload_strip(false, dist_srgb, &plan);
            self.run_strip_dist_only_to_ssim_map();
            for s in 0..self.scales.len() {
                self.run_abs_diff_only(s, avg_per_scale[s]);
                self.sum_mad_body(s, &plan);
            }
        }
        self.run_finalize();
        let pass2_sums = self.read_sums();

        // Final score.
        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.body_pixels_at_scale(s, &cfg) as f64;
            let mad_sum = pass2_sums[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        Ok(GpuDssimResult {
            score: ssim_to_dssim(ssim),
        })
    }

    /// Slice this strip's row range out of the cached full-image
    /// [`RefFullState`] into the strip-sized `Scale.ref_lab` /
    /// `Scale.ref_mu` / `Scale.ref_sq_blur` buffers. Per-scale row
    /// range derives from the strip plan's `read_start_in_image` /
    /// `read_end_in_image` mapped through the pyramid `div_ceil(2)`
    /// descent.
    fn install_ref_state_for_strip(&self, plan: &StripPlan) {
        let ref_full = self
            .ref_full
            .as_ref()
            .expect("install_ref_state_for_strip requires cached ref");
        for s in 0..self.scales.len() {
            let (full_w, full_h) = ref_full.dims[s];
            // Pyramid uses div_ceil(2) per descent. The strip's
            // read_start_in_image is a multiple of PYRAMID_ALIGN
            // (h_body and HALO both are) so the per-scale offset
            // divides cleanly. read_end_in_image may not align; we
            // clamp against the full scale-s plane height.
            let divisor = 1u32 << (s as u32);
            let src_row_start = plan.read_start_in_image / divisor;
            // n_rows is the strip's data height at scale s — what
            // the kernels will treat as the strip's valid data
            // edge. set_strip_data_extents must have been called
            // first.
            let n_rows = self.strip_data_h[s].min(full_h.saturating_sub(src_row_start));
            let strip_scale = &self.scales[s];
            for ch in 0..3 {
                self.launch_copy_rows(
                    &ref_full.ref_lab[s][ch],
                    &strip_scale.ref_lab[ch],
                    full_w,
                    full_w * full_h,
                    strip_scale.n,
                    n_rows,
                    src_row_start,
                );
                self.launch_copy_rows(
                    &ref_full.ref_mu[s][ch],
                    &strip_scale.ref_mu[ch],
                    full_w,
                    full_w * full_h,
                    strip_scale.n,
                    n_rows,
                    src_row_start,
                );
                self.launch_copy_rows(
                    &ref_full.ref_sq_blur[s][ch],
                    &strip_scale.ref_sq_blur[ch],
                    full_w,
                    full_w * full_h,
                    strip_scale.n,
                    n_rows,
                    src_row_start,
                );
            }
        }
    }

    /// Launch the row-range copy kernel `dst[0..n_rows*width]`
    /// = `src[src_row_start*width..(src_row_start+n_rows)*width]`.
    /// `src_total_n` and `dst_total_n` are the underlying buffer
    /// lengths used to satisfy `ArrayArg::from_raw_parts`'s typed
    /// length requirement (the kernel itself reads only within the
    /// row range).
    #[allow(clippy::too_many_arguments)]
    fn launch_copy_rows(
        &self,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        width: u32,
        src_total_n: u32,
        dst_total_n: usize,
        n_rows: u32,
        src_row_start: u32,
    ) {
        let total = (n_rows as usize) * (width as usize);
        if total == 0 {
            return;
        }
        unsafe {
            copy_rows_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(total),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), src_total_n as usize),
                ArrayArg::from_raw_parts(dst.clone(), dst_total_n),
                width,
                n_rows,
                src_row_start,
            );
        }
    }

    /// Dist-only per-strip pipeline up to and including the SSIM
    /// map for every scale. Companion to
    /// [`Self::run_strip_to_ssim_map`] but assumes ref-side state
    /// is already installed via [`Self::install_ref_state_for_strip`]
    /// — skips every `_is_a=true` kernel.
    fn run_strip_dist_only_to_ssim_map(&mut self) {
        self.build_linear_pyramid(false);
        for s in 0..self.scales.len() {
            self.run_lab(s, false);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
        }
    }

    // ───────────────────────── strip processing ─────────────────────────

    /// Strip-mode pair scoring. Splits the image into vertical strips
    /// of `h_body` body rows + `halo` rows above/below for stencil
    /// reach, runs the full DSSIM pipeline per strip on strip-sized
    /// working buffers, and accumulates body-row partial sums into
    /// per-scale totals.
    ///
    /// Returns the same `GpuDssimResult` as [`Self::compute`] — strip
    /// vs whole-image agrees within f32 reordering noise (typically
    /// 1e-5 rel on accumulated sums).
    ///
    /// Returns `Err(NoCachedReference)` if called on an instance not
    /// constructed via [`Self::new_strip`].
    pub fn compute_stripped(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuDssimResult> {
        if self.strip_config.is_none() {
            // Borrow the same error variant — strip mode wasn't
            // requested at construction time. (The
            // `InvalidImageSize` variant would be wrong here; use
            // `NoCachedReference` as the closest "wrong API path"
            // signal — but better to introduce a dedicated variant
            // later if this becomes user-facing.)
            return Err(Error::NoCachedReference);
        }
        self.check_dims_image(ref_srgb)?;
        self.check_dims_image(dist_srgb)?;

        let cfg = self.strip_config.clone().unwrap();

        // Pass 1: sum Σ ssim per scale across all strips.
        self.zero_partials();
        for strip in 0..self.n_strips() {
            let plan = self.strip_plan(strip);
            self.set_strip_data_extents(&plan);
            self.upload_strip(true, ref_srgb, &plan);
            self.upload_strip(false, dist_srgb, &plan);
            self.run_strip_to_ssim_map();
            self.sum_ssim_body(&plan);
        }
        self.run_finalize();
        let pass1_sums = self.read_sums();

        // Compute per-scale avg from full-image mean_ssim.
        let mut avg_per_scale = [0.0_f32; NUM_SCALES];
        for s in 0..self.scales.len() {
            let n_pix = self.body_pixels_at_scale(s, &cfg) as f64;
            let ssim_sum = pass1_sums[s * 2] as f64;
            let mean_ssim = ssim_sum / n_pix;
            let avg = mean_ssim.max(0.0).powf(0.5_f64.powi(s as i32));
            avg_per_scale[s] = avg as f32;
        }

        // Pass 2: re-run pipeline per strip, but now also compute
        // abs_diff_scalar against the per-scale avg and sum the mad
        // map over body rows.
        self.zero_partials();
        for strip in 0..self.n_strips() {
            let plan = self.strip_plan(strip);
            self.set_strip_data_extents(&plan);
            self.upload_strip(true, ref_srgb, &plan);
            self.upload_strip(false, dist_srgb, &plan);
            self.run_strip_to_ssim_map();
            for s in 0..self.scales.len() {
                self.run_abs_diff_only(s, avg_per_scale[s]);
                self.sum_mad_body(s, &plan);
            }
        }
        self.run_finalize();
        let pass2_sums = self.read_sums();

        // Final score.
        let mut weighted = 0.0_f64;
        let mut weight_sum = 0.0_f64;
        for s in 0..self.scales.len() {
            let n_pix = self.body_pixels_at_scale(s, &cfg) as f64;
            let mad_sum = pass2_sums[s * 2 + 1] as f64;
            let mad = mad_sum / n_pix;
            let scale_score = 1.0 - mad;
            weighted += scale_score * SCALE_WEIGHTS[s];
            weight_sum += SCALE_WEIGHTS[s];
        }
        let ssim = weighted / weight_sum;
        Ok(GpuDssimResult {
            score: ssim_to_dssim(ssim),
        })
    }

    fn n_strips(&self) -> u32 {
        let cfg = self.strip_config.as_ref().expect("strip mode required");
        cfg.image_h.div_ceil(cfg.h_body)
    }

    /// Populate `self.strip_data_h` with the actual data height per
    /// scale for the strip described by `plan`. After this call,
    /// `effective_h(s)` returns the strip's data height at scale `s`,
    /// and subsequent kernel launches for this strip clamp at the
    /// data edge (matching whole-image's boundary semantics).
    fn set_strip_data_extents(&mut self, plan: &StripPlan) {
        // Per-scale data height. At scale 0 it's `strip_h_actual`;
        // at scale s it's `div_ceil(strip_h_at_(s-1), 2)`. Mirrors
        // the pyramid build's div_ceil(2) pattern.
        let mut h = plan.strip_h_actual;
        for s in 0..NUM_SCALES {
            self.strip_data_h[s] = h;
            h = h.div_ceil(2);
        }
    }

    /// Plan for one strip: row coordinates within the image, and the
    /// in-strip body offset / body height at scale 0.
    fn strip_plan(&self, strip_idx: u32) -> StripPlan {
        let cfg = self.strip_config.as_ref().expect("strip mode required");
        let body_start_in_image = strip_idx * cfg.h_body;
        let body_end_in_image = (body_start_in_image + cfg.h_body).min(cfg.image_h);
        let read_start_in_image = body_start_in_image.saturating_sub(cfg.halo);
        let read_end_in_image = (body_end_in_image + cfg.halo).min(cfg.image_h);
        let body_offset_in_strip = body_start_in_image - read_start_in_image;
        let body_h_in_strip = body_end_in_image - body_start_in_image;
        let strip_h_actual = read_end_in_image - read_start_in_image;
        StripPlan {
            read_start_in_image,
            read_end_in_image,
            body_offset_in_strip_at_0: body_offset_in_strip,
            body_h_in_strip_at_0: body_h_in_strip,
            strip_h_actual,
        }
    }

    /// Number of body pixels summed across all strips at scale `s`.
    /// Used to normalize the accumulated `Σ ssim` / `Σ mad` into a
    /// mean.
    ///
    /// Subtlety: `image_h` may not be exactly divisible by
    /// `2^(NUM_SCALES - 1)`. The pyramid uses `div_ceil(2)` per
    /// descent and clamps each axis to ≥ 8; we mirror that here by
    /// computing the per-scale total height as `div_ceil` of
    /// `image_h` by `2^s` (and the per-scale width similarly).
    /// Per-strip body heights at scale `s` are
    /// `body_h_in_strip_at_0 / 2^s` for full strips, with the final
    /// (partial) strip reduced proportionally.
    fn body_pixels_at_scale(&self, scale: usize, cfg: &StripConfig) -> u64 {
        // Width at scale s with clamp to 8.
        let mut w = cfg.image_w;
        for _ in 0..scale {
            w = w.div_ceil(2);
            if w < 8 {
                w = 8;
            }
        }
        // Sum body heights at scale s across all strips. Each strip's
        // body at scale 0 is `h_body` rows (except possibly the last);
        // at scale s that's `h_body / 2^s` (require divisibility,
        // enforced in `new_strip`). The last strip's body may be
        // shorter, mapped via div_ceil.
        let divisor = 1u32 << (scale as u32);
        let n_strips = self.n_strips();
        let mut total_h: u64 = 0;
        for k in 0..n_strips {
            let body_start = k * cfg.h_body;
            let body_end = (body_start + cfg.h_body).min(cfg.image_h);
            let body_h_at_0 = body_end - body_start;
            // Scale-s mapping: body at scale 0 starts at
            // `body_start` and is `body_h_at_0` rows. The
            // corresponding scale-s body row count is the number of
            // scale-s rows whose "footprint" lies in the scale-0
            // body. With div_ceil pyramid, this is `div_ceil(body_h,
            // 2^s)` modulo clamp.
            let body_h_at_s = body_h_at_0.div_ceil(divisor).max(1);
            total_h += body_h_at_s as u64;
        }
        // Cap by the scale-s height (we never sum more rows than
        // the scale-s plane actually has).
        let mut h_scale = cfg.image_h;
        for _ in 0..scale {
            h_scale = h_scale.div_ceil(2);
            if h_scale < 8 {
                h_scale = 8;
            }
        }
        total_h = total_h.min(h_scale as u64);
        total_h * (w as u64)
    }

    /// Copy strip rows from `image_srgb` into the scale-0 staging
    /// buffer and launch sRGB→linear conversion. Mirrors
    /// `upload_and_srgb_to_linear` but for a strip slice.
    ///
    /// **Edge handling**: tail rows beyond the actual strip data
    /// height are zero-filled. The per-strip kernel launches use the
    /// actual data heights (per-scale) for `width`/`height` clamp
    /// arguments so blurs and downscales clamp at the data edge, not
    /// the buffer edge. This keeps boundary semantics identical to
    /// the whole-image path: blur at the last data row reads
    /// `[y - 1, y, y]` because `min(y + 1, data_h - 1) = y`.
    fn upload_strip(&mut self, is_a: bool, image_srgb: &[u8], plan: &StripPlan) {
        let cfg = self
            .strip_config
            .as_ref()
            .expect("strip mode required")
            .clone();
        let row_bytes = (cfg.image_w as usize) * 3;
        let strip_n = (cfg.image_w as usize) * (plan.strip_h_actual as usize);
        let buffer_h = self.scales[0].height as usize;
        let pinned_len = (cfg.image_w as usize) * buffer_h * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            // Zero-fill the buffer. Tail rows past strip_h_actual
            // are never read in strip mode because per-scale kernel
            // launches pass the actual data height as `height`.
            for b in dst.iter_mut() {
                *b = 0;
            }
            let src_start = (plan.read_start_in_image as usize) * row_bytes;
            let src_slice = &image_srgb[src_start..src_start + strip_n * 3];
            for (chunk_out, triple) in dst.chunks_exact_mut(4).zip(src_slice.chunks_exact(3)) {
                chunk_out[0] = triple[0];
                chunk_out[1] = triple[1];
                chunk_out[2] = triple[2];
                chunk_out[3] = 0;
            }
        }
        let handle = self.client.create(bytes);
        if is_a {
            self.src_u8_a = handle;
        } else {
            self.src_u8_b = handle;
        }
        // sRGB→linear runs on the full buffer (n pixels); tail
        // pixels with sRGB=0 just produce linear=0 in the tail.
        self.srgb_to_linear_from_packed(is_a);
    }

    /// Per-strip pipeline up to and including the SSIM map for every
    /// scale. Same shape as `compute_post_srgb`'s scale loop but
    /// skips the final reduction (caller handles body-only summation
    /// after this returns).
    fn run_strip_to_ssim_map(&mut self) {
        // Build linear pyramid + linear pyramid for dis.
        self.build_linear_pyramid(true);
        self.build_linear_pyramid(false);

        for s in 0..self.scales.len() {
            self.run_lab(s, true);
            self.run_lab(s, false);
            self.run_chroma_preblur(s, true);
            self.run_chroma_preblur(s, false);
            self.run_blur_stats(s, true);
            self.run_blur_stats(s, false);
            self.run_cross_blur(s);
            self.run_ssim_map(s);
        }
    }

    /// Sum the scale-s SSIM map's body rows into the per-scale
    /// `ssim_sum` slot. The body row span at scale s is
    /// `body_offset_in_strip_at_s × width_at_s` ..
    /// `(body_offset + body_h) × width_at_s`.
    fn sum_ssim_body(&self, plan: &StripPlan) {
        for s in 0..self.scales.len() {
            let scale = &self.scales[s];
            let (start_idx, end_idx) = scale_row_range(plan, s, scale.width, scale.height);
            let slot = (s * 2) as u32; // ssim_sum slot
            reduction::launch_sum_range::<R>(
                &self.client,
                scale.ssim_map.clone(),
                scale.n,
                self.partials.clone(),
                PARTIALS_LEN,
                slot,
                start_idx,
                end_idx,
            );
        }
    }

    /// Compute `|ssim - avg|` into `mad_map` for the given scale.
    /// Separated from `run_abs_diff_and_sum` because in strip mode
    /// we want to sum only body rows, not the whole strip.
    fn run_abs_diff_only(&self, scale: usize, avg: f32) {
        let s = &self.scales[scale];
        unsafe {
            ssim::abs_diff_scalar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ssim_map.clone(), s.n),
                ArrayArg::from_raw_parts(s.mad_map.clone(), s.n),
                avg,
            );
        }
    }

    fn sum_mad_body(&self, scale: usize, plan: &StripPlan) {
        let s = &self.scales[scale];
        let (start_idx, end_idx) = scale_row_range(plan, scale, s.width, s.height);
        let slot = (scale * 2 + 1) as u32; // mad_sum slot
        reduction::launch_sum_range::<R>(
            &self.client,
            s.mad_map.clone(),
            s.n,
            self.partials.clone(),
            PARTIALS_LEN,
            slot,
            start_idx,
            end_idx,
        );
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        // Validate against the LOGICAL extent — sub-MIN_PAD_DIM callers
        // pass logical-sized buffers, which `upload_and_srgb_to_linear`
        // reflect-pads up to the padded pipeline. No-op vs `self.n * 3`
        // at ≥8px (logical == padded).
        let expected = self.pad.logical_len(3);
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Image-level dimension check for the strip path — `srgb` must
    /// be `image_w × image_h × 3` bytes (not strip-sized).
    fn check_dims_image(&self, srgb: &[u8]) -> Result<()> {
        let cfg = self
            .strip_config
            .as_ref()
            .expect("check_dims_image called outside strip mode");
        let expected = (cfg.image_w as usize) * (cfg.image_h as usize) * 3;
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

    fn upload_and_srgb_to_linear(&mut self, is_a: bool, srgb: &[u8]) {
        // Reflect-pad a sub-MIN_PAD_DIM logical input up to the padded
        // pyramid floor before packing. Borrows unchanged at ≥8px
        // (logical == padded). Callers validate the logical length via
        // `check_dims` first.
        let plan = self.pad;
        let srgb = plan.pad(srgb, 3);
        let srgb: &[u8] = &srgb;
        let n_bytes = self.n * 3;
        debug_assert_eq!(srgb.len(), n_bytes);
        // T_x.O (2026-05-17): pack u8×3 → u32 directly into the
        // pinned staging buffer (one host-side pass instead of two).
        // Previously we packed into `self.pack_scratch` and then
        // `create_from_slice_pinned` copied that scratch into a
        // pinned buffer — two ~48 MB host writes at 12 MP. The
        // reserve_staging path lets us produce the packed bytes
        // straight into the pinned buffer.
        //
        // Layout (unchanged from T4.L): 4 bytes per pixel — R | G<<8
        // | B<<16 (alpha unused). Reader
        // (`srgb_u8_to_linear_planar_kernel`) sees the same `[u32]`
        // packing.
        let pinned_len = self.n * 4;
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
        // T4.M (2026-05-16): pinned-host upload — DMAs at 12-25 GB/s
        // on PCIe 4.0 vs 5-6 GB/s pageable.
        // T_x.O: skipping the pack_scratch intermediate saves one
        // ~48 MB host write per upload at 12 MP.
        let handle = self.client.create(bytes);
        if is_a {
            self.src_u8_a = handle;
        } else {
            self.src_u8_b = handle;
        }
        self.srgb_to_linear_from_packed(is_a);
    }

    /// Run the sRGB-u8 → linear-planar conversion from whichever
    /// packed-u32 handle currently sits in `src_u8_a` / `src_u8_b`.
    /// Split out of [`Self::upload_and_srgb_to_linear`] so that
    /// [`Self::compute_handles`] (Phase 4 upload-once path) can skip
    /// the byte-copy step and reuse a caller-supplied device buffer.
    fn srgb_to_linear_from_packed(&self, is_a: bool) {
        let (src, lin) = if is_a {
            (&self.src_u8_a, &self.scales[0].ref_lin)
        } else {
            (&self.src_u8_b, &self.scales[0].dis_lin)
        };
        unsafe {
            srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(self.n),
                Self::cube_dim_1d(),
                // T4.L: one u32 per pixel.
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
            );
        }
    }

    /// Install a caller-supplied packed-u32 device handle as the
    /// ref/dist input. Handle layout MUST match what
    /// [`Self::upload_and_srgb_to_linear`] produces: `width × height`
    /// `u32`s, each `R | G<<8 | B<<16` (alpha unused). After return
    /// the sRGB→linear kernel has been dispatched and scale-0 linear
    /// planes are populated.
    fn install_packed_handle(&mut self, is_a: bool, handle: &cubecl::server::Handle) {
        if is_a {
            self.src_u8_a = handle.clone();
        } else {
            self.src_u8_b = handle.clone();
        }
        self.srgb_to_linear_from_packed(is_a);
    }

    fn build_linear_pyramid(&self, is_a: bool) {
        for s in 1..self.scales.len() {
            let prev_w = self.scales[s - 1].width;
            let curr_w = self.scales[s].width;
            // Effective heights: actual data heights in strip mode,
            // buffer heights in whole-image mode. Downscale clamps at
            // src_h - 1 so passing the data height keeps the
            // boundary behaviour identical between modes.
            let prev_h = self.effective_h(s - 1);
            let curr_h = self.effective_h(s);
            let (prev_lin, curr_lin) = if is_a {
                (&self.scales[s - 1].ref_lin, &self.scales[s].ref_lin)
            } else {
                (&self.scales[s - 1].dis_lin, &self.scales[s].dis_lin)
            };
            let n_curr = self.scales[s].n;
            let n_prev = self.scales[s - 1].n;
            for ch in 0..3 {
                unsafe {
                    downscale::downscale_2x_plane_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n_curr),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(prev_lin[ch].clone(), n_prev),
                        ArrayArg::from_raw_parts(curr_lin[ch].clone(), n_curr),
                        prev_w,
                        prev_h,
                        curr_w,
                        curr_h,
                    );
                }
            }
        }
    }

    fn run_lab(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let (lin, lab_buf) = if is_a {
            (&s.ref_lin, &s.ref_lab)
        } else {
            (&s.dis_lin, &s.dis_lab)
        };
        unsafe {
            lab::linear_to_lab_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), s.n),
                ArrayArg::from_raw_parts(lin[1].clone(), s.n),
                ArrayArg::from_raw_parts(lin[2].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[0].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[1].clone(), s.n),
                ArrayArg::from_raw_parts(lab_buf[2].clone(), s.n),
            );
        }
    }

    /// Two-pass `blur_3x3` of the chroma channels (a, b) in place via
    /// scratch. Matches `dssim-cuda`'s pre-SSIM chroma blur step.
    fn run_chroma_preblur(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let lab_buf = if is_a { &s.ref_lab } else { &s.dis_lab };
        for ch in [1usize, 2] {
            self.blur_two_pass(scale, &lab_buf[ch], &lab_buf[ch], &s.temp1, &s.temp2);
        }
    }

    /// Two-pass blur via `src → temp_a → temp_b` then copy `temp_b → dst`.
    /// `dst` and `src` may be the same handle. Aliasing rule: the two
    /// scratches must be distinct from each other AND from src/dst.
    fn blur_two_pass(
        &self,
        scale: usize,
        src: &cubecl::server::Handle,
        dst: &cubecl::server::Handle,
        scratch_a: &cubecl::server::Handle,
        scratch_b: &cubecl::server::Handle,
    ) {
        let s = &self.scales[scale];
        let h = self.effective_h(scale);
        unsafe {
            // pass 1: src → scratch_a
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(src.clone(), s.n),
                ArrayArg::from_raw_parts(scratch_a.clone(), s.n),
                s.width,
                h,
            );
            // pass 2: scratch_a → scratch_b
            blur::blur_3x3_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_a.clone(), s.n),
                ArrayArg::from_raw_parts(scratch_b.clone(), s.n),
                s.width,
                h,
            );
            // copy scratch_b → dst (allows dst == src). Use the
            // `blur_3x3` kernel here would corrupt the result; use a
            // pointwise copy via the abs_diff_scalar(0.0) hack? No —
            // use an explicit copy kernel.
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(scratch_b.clone(), s.n),
                ArrayArg::from_raw_parts(dst.clone(), s.n),
            );
        }
    }

    /// For each channel: `mu = blur(blur(src))`,
    /// `sq_blur = blur(blur_squared(src))`. Stores into `ref_mu` /
    /// `ref_sq_blur` (or dis variants).
    fn run_blur_stats(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let h = self.effective_h(scale);
        let (src_lab, mu_dst, sq_dst) = if is_a {
            (&s.ref_lab, &s.ref_mu, &s.ref_sq_blur)
        } else {
            (&s.dis_lab, &s.dis_mu, &s.dis_sq_blur)
        };
        for ch in 0..3 {
            // mu pipeline.
            self.blur_two_pass(scale, &src_lab[ch], &mu_dst[ch], &s.temp1, &s.temp2);

            // sq_blur pipeline: blur_squared(src) → temp1, then blur → sq_dst.
            unsafe {
                blur::blur_squared_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(src_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    s.width,
                    h,
                );
                blur::blur_3x3_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    ArrayArg::from_raw_parts(sq_dst[ch].clone(), s.n),
                    s.width,
                    h,
                );
            }
        }
    }

    /// `cross_blur[ch] = blur(blur_product(ref_lab[ch], dis_lab[ch]))`.
    fn run_cross_blur(&self, scale: usize) {
        let s = &self.scales[scale];
        let h = self.effective_h(scale);
        for ch in 0..3 {
            unsafe {
                blur::blur_product_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.ref_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.dis_lab[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    s.width,
                    h,
                );
                blur::blur_3x3_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.temp1.clone(), s.n),
                    ArrayArg::from_raw_parts(s.cross_blur[ch].clone(), s.n),
                    s.width,
                    h,
                );
            }
        }
    }

    fn run_ssim_map(&self, scale: usize) {
        let s = &self.scales[scale];
        unsafe {
            ssim::ssim_lab_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ref_mu[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_mu[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_mu[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_mu[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.ref_sq_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.dis_sq_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[0].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[1].clone(), s.n),
                ArrayArg::from_raw_parts(s.cross_blur[2].clone(), s.n),
                ArrayArg::from_raw_parts(s.ssim_map.clone(), s.n),
            );
        }
    }

    fn run_sum_ssim(&self, scale: usize) {
        let s = &self.scales[scale];
        let slot = (scale * 2) as u32; // ssim_sum slot
        reduction::launch_sum::<R>(
            &self.client,
            s.ssim_map.clone(),
            s.n,
            self.partials.clone(),
            PARTIALS_LEN,
            slot,
        );
    }

    /// Compute `mad_map = |ssim_map - avg|` then run the second
    /// reduction into the slot reserved for this scale.
    fn run_abs_diff_and_sum(&self, scale: usize, avg: f32) {
        let s = &self.scales[scale];
        unsafe {
            ssim::abs_diff_scalar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(s.ssim_map.clone(), s.n),
                ArrayArg::from_raw_parts(s.mad_map.clone(), s.n),
                avg,
            );
        }
        let slot = (scale * 2 + 1) as u32; // mad_sum slot
        reduction::launch_sum::<R>(
            &self.client,
            s.mad_map.clone(),
            s.n,
            self.partials.clone(),
            PARTIALS_LEN,
            slot,
        );
    }

    fn run_finalize(&self) {
        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            PARTIALS_LEN,
            self.sums.clone(),
            SUMS_LEN,
            NUM_SLOTS as u32,
        );
    }

    /// Reset both reduction buffers. Required because the fast-mode
    /// path uses `Atomic<f32>::fetch_add` and inherits the previous
    /// call's accumulator otherwise.
    fn zero_partials(&mut self) {
        self.partials = self
            .client
            .create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        self.sums = self
            .client
            .create_from_slice(f32::as_bytes(&[0.0_f32; SUMS_LEN]));
    }

    fn read_sums(&mut self) -> Vec<f32> {
        let bytes = self
            .client
            .read_one(self.sums.clone())
            .expect("read sums buffer");
        f32::from_bytes(&bytes).to_vec()
    }
}

/// Pointwise copy kernel `dst[i] = src[i]`. Used in the chroma
/// pre-blur double-pass to allow `dst == src`.
#[cube(launch_unchecked)]
pub fn copy_kernel(src: &Array<f32>, dst: &mut Array<f32>) {
    let i = ABSOLUTE_POS;
    if i >= dst.len() {
        terminate!();
    }
    dst[i] = src[i];
}

/// Row-range copy kernel: copies the slice of `src` covering rows
/// `[src_row_start, src_row_start + n_rows)` (each `width` elements
/// wide) into `dst` starting at row 0. Used by mode E (cached-ref in
/// strip mode) to slice the per-strip row range out of the
/// full-image-sized ref state buffers into the strip-sized Scale
/// buffers ahead of cross_blur / ssim_map for the current strip.
///
/// `src` is laid out as `src_total_rows × width` row-major; the
/// kernel reads `src[(src_row_start + r) * width + x]` for
/// `r ∈ [0, n_rows)`, `x ∈ [0, width)` and writes to
/// `dst[r * width + x]`. Threads beyond `n_rows * width` exit.
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

/// Convert SSIM (0-1, higher better) to DSSIM (0+, lower better).
/// Verbatim from `dssim-cuda::ssim_to_dssim`.
fn ssim_to_dssim(ssim: f64) -> f64 {
    1.0 / ssim.max(f64::EPSILON) - 1.0
}
