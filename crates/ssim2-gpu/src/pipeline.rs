//! SSIMULACRA2 pipeline orchestration.
//!
//! Wires the kernels in `kernels::*` together into the 6-octave
//! SSIMULACRA2 algorithm. Public entry points:
//!
//! - [`Ssim2::new`] + [`Ssim2::compute`] — score one image pair from sRGB.
//! - [`Ssim2::set_reference`] + [`Ssim2::compute_with_reference`] — cache
//!   reference-side state and score many distorted images against it
//!   (encoder rate-distortion search).
//!
//! Algorithm (per scale, per channel):
//!   1. linear RGB pyramid (downscale_2x).
//!   2. linear → positive XYB.
//!   3. sigma11 = ref²; sigma22 = dis²; sigma12 = ref·dis.
//!   4. Blur 5 planes: ref, dis, sigma11, sigma22, sigma12.
//!      Implementation: vertical IIR pass → transpose → vertical IIR pass.
//!      Output is in transposed orientation (saves a final transpose);
//!      compute_error_maps + reduction don't care about orientation.
//!   5. Transpose ref/dis (raw, unblurred) so error_maps can read them
//!      alongside the already-transposed mu1/mu2/sigma planes.
//!   6. compute_error_maps → ssim, artifact, detail_loss (per channel).
//!   7. Reduce each → (Σ, Σ⁴) per (scale, channel, error-map).
//!
//! Final score: weighted dot product of all 108 sub-stats with the
//! published SSIMULACRA2 weights, then the standard sigmoid-like remap.
//! Mirrors `ssimulacra2::Msssim::score` line-for-line.

use cubecl::prelude::*;

#[cfg(feature = "fir")]
use crate::Ssim2Blur;
use crate::kernels::{blur, downscale, error_maps, reduction, srgb, transpose, xyb};
use crate::skipmap::{Ssim2Mode, skip_error_map, skip_reduction, skip_scale};
use crate::{Error, GpuSsim2Result, NUM_SCALES, Result};

/// Strip-processing metadata. See `Ssim2::strip` for full docs.
#[derive(Debug, Clone, Copy)]
struct StripMeta {
    /// Full-frame width and height the caller passes to `compute_stripped`.
    image_w: u32,
    image_h: u32,
    /// Body rows per strip (excluding halo). Constructor clamps this
    /// against `image_h` so tiny images degenerate to single-strip mode.
    h_body: u32,
    /// Halo rows per side at the finest scale. Always
    /// [`crate::memory_mode::STRIP_HALO_ROWS`] currently; carried in
    /// the struct so per-scale halos derive from it consistently.
    halo: u32,
}

/// Cached reference state for strip-mode mode E (task #46). Holds
/// **full-image-sized** ref-side state per scale: the raw transposed
/// XYB pyramid + the fully-blurred mu1 + the fully-blurred ref²
/// (sigma11), all in transposed orientation (inner stride = full
/// image height at the scale). When a strip-mode `Ssim2` has this
/// populated, `compute_with_reference` walks the distorted side in
/// strips and reads ref values out of these full buffers via the
/// `error_maps_strip_from_full_ref` kernel.
///
/// Each scale's `(width, full_h, n)` reflects the full-image pyramid
/// size at that scale — NOT the strip pyramid size.
struct StripCachedRefScale {
    /// Width in original-frame X at this pyramid scale.
    width: u32,
    /// Height in original-frame Y at this pyramid scale. The
    /// transposed buffers' inner stride equals this.
    full_h: u32,
    /// `width × full_h` total floats per plane.
    n: usize,
    /// Raw **pre-transpose** reference XYB at this scale. Used by
    /// strip-mode compute to build `sigma12_strip = ref · dis` via
    /// the `pointwise_mul_offset` kernel, slicing this full plane
    /// with the strip's flat-index offset.
    ref_xyb_full: [cubecl::server::Handle; 3],
    /// Raw transposed reference XYB (input to error_maps `source`).
    ref_xyb_t_full: [cubecl::server::Handle; 3],
    /// Fully-blurred reference XYB (input to error_maps `mu1`).
    mu1_full_full: [cubecl::server::Handle; 3],
    /// Fully-blurred reference² (input to error_maps `sigma11`).
    sigma11_full_full: [cubecl::server::Handle; 3],
}

impl StripCachedRefScale {
    fn new<R: Runtime>(client: &ComputeClient<R>, width: u32, full_h: u32) -> Self {
        let n = (width as usize) * (full_h as usize);
        Self {
            width,
            full_h,
            n,
            ref_xyb_full: alloc_3(client, n),
            ref_xyb_t_full: alloc_3(client, n),
            mu1_full_full: alloc_3(client, n),
            sigma11_full_full: alloc_3(client, n),
        }
    }
}

/// Per-scale buffer set. Each plane is `width × height` f32, planar
/// (one buffer per channel of a 3-channel image).
struct Scale {
    width: u32,
    height: u32,
    n: usize,

    /// Linear RGB at this scale (planar). Kept across calls so
    /// `set_reference` can rebuild the pyramid without re-uploading.
    ref_lin: [cubecl::server::Handle; 3],
    dis_lin: [cubecl::server::Handle; 3],

    /// Positive XYB after linear→XYB conversion.
    ref_xyb: [cubecl::server::Handle; 3],
    dis_xyb: [cubecl::server::Handle; 3],

    /// Pointwise products (re-used as scratch each call).
    sigma11_in: [cubecl::server::Handle; 3], // ref·ref
    sigma22_in: [cubecl::server::Handle; 3], // dis·dis
    sigma12_in: [cubecl::server::Handle; 3], // ref·dis

    /// Rolling scratch buffers for the two-pass blur, shared across all
    /// 5 plane blurs (sigma11/22/12/mu1/mu2) within this scale.
    ///
    /// Phase 1 (2026-05-22) aliasing: previously this struct carried
    /// 30 separate plane buffers — `{sigma11,sigma22,sigma12,mu1,mu2}_v`
    /// and `{sigma11,sigma22,sigma12,mu1,mu2}_t` (5 plane names × 3
    /// channels × 2 orientations). Each `*_v` was dead the moment its
    /// `*_t` was written by the transpose; each `*_t` was dead the
    /// moment its `*_full` was written by the second blur pass. With
    /// in-order GPU launches the same `(v_scratch[ch], t_scratch[ch])`
    /// pair safely cycles across all 5 blurs of one channel.
    ///
    /// The batched pipeline (`pipeline_batch.rs::BatchScale`) already
    /// uses this idiom — see its `v_scratch`/`t_scratch` fields. This
    /// change brings the unbatched pipeline in line.
    ///
    /// Net saving per scale: 30 → 6 plane handles for the blur
    /// intermediates. At 24 MP that's ~570 MB of working set returned
    /// to the runtime.
    v_scratch: [cubecl::server::Handle; 3],
    t_scratch: [cubecl::server::Handle; 3],

    /// Second-pass (vertical-walk on transposed) outputs. Orientation
    /// is "transposed" — for a `width × height` source these now live
    /// in `height × width` row-major.
    sigma11_full: [cubecl::server::Handle; 3],
    sigma22_full: [cubecl::server::Handle; 3],
    sigma12_full: [cubecl::server::Handle; 3],
    mu1_full: [cubecl::server::Handle; 3],
    mu2_full: [cubecl::server::Handle; 3],

    /// Raw XYB transposed (input to compute_error_maps' `source`/`distorted`).
    ref_xyb_t: [cubecl::server::Handle; 3],
    dis_xyb_t: [cubecl::server::Handle; 3],

    /// Error maps (in transposed orientation).
    ssim: [cubecl::server::Handle; 3],
    artifact: [cubecl::server::Handle; 3],
    detail: [cubecl::server::Handle; 3],
}

fn alloc_plane<R: Runtime>(client: &ComputeClient<R>, n: usize) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(&vec![0.0_f32; n]))
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
            ref_xyb: alloc_3(client, n),
            dis_xyb: alloc_3(client, n),
            sigma11_in: alloc_3(client, n),
            sigma22_in: alloc_3(client, n),
            sigma12_in: alloc_3(client, n),
            v_scratch: alloc_3(client, n),
            t_scratch: alloc_3(client, n),
            sigma11_full: alloc_3(client, n),
            sigma22_full: alloc_3(client, n),
            sigma12_full: alloc_3(client, n),
            mu1_full: alloc_3(client, n),
            mu2_full: alloc_3(client, n),
            ref_xyb_t: alloc_3(client, n),
            dis_xyb_t: alloc_3(client, n),
            ssim: alloc_3(client, n),
            artifact: alloc_3(client, n),
            detail: alloc_3(client, n),
        }
    }
}

/// Per-instance allocations + per-call orchestration of the full
/// SSIMULACRA2 pipeline. Construct once for a given resolution; reuse
/// across many image pairs of that resolution.
pub struct Ssim2<R: Runtime> {
    client: ComputeClient<R>,
    width: u32,
    height: u32,
    /// `n_pixels` at scale 0.
    n: usize,

    /// sRGB u8 staging — re-uploaded per call.
    src_u8_a: cubecl::server::Handle,
    src_u8_b: cubecl::server::Handle,

    // T_x.O (2026-05-17): `pack_scratch: Vec<u32>` removed. The
    // upload path now packs u8×3 → u32 directly into the pinned
    // staging buffer reserved per call (`client.reserve_staging`),
    // collapsing two host-side passes (pack to pageable + memcpy to
    // pinned) into one. Same shape as butter T_x.O (10a5b996).
    /// Per-scale buffer sets.
    scales: Vec<Scale>,

    /// Per-thread partials scratch — `NUM_SLOTS * PARTIALS_PER_REDUCTION`
    /// floats. Used in stage 1 of the two-stage reduction. Never read
    /// by the host; only the much smaller `sums` buffer crosses the
    /// device→host boundary.
    partials: cubecl::server::Handle,
    /// Final (Σ, Σ⁴) per slot — `NUM_SLOTS * 2` floats. Populated by
    /// the finalizer kernel and read once per `compute()`.
    /// Slot index: `(scale * 3 + channel) * 3 + map_type`
    ///   - map_type: 0=ssim, 1=artifact, 2=detail
    sums: cubecl::server::Handle,

    has_cached_reference: bool,

    /// Whether the cached reference's blur outputs are in untransposed
    /// orientation (IIR fast path, default) or transposed orientation
    /// (FIR fallback). `true` until `set_reference` is called with FIR
    /// mode active. Read by `compute_with_reference_with_mode` to pick
    /// the right error_maps inputs.
    cached_ref_is_untransposed: bool,

    /// Strip-processing metadata. `None` for whole-image instances
    /// (constructed via `new` / `new_with_memory_mode { Full | Auto→Full }`);
    /// `Some` for strip-mode instances constructed via `new_strip`. When
    /// `Some`, `compute_with_mode` is illegal (use `compute_stripped`);
    /// when `None`, `compute_stripped` is illegal.
    ///
    /// Records `(image_w, image_h, h_body)` so the strip driver can:
    /// - reject `(ref, dist)` whose dimensions don't match `image_w×image_h`,
    /// - compute strip start/end + body row ranges per strip,
    /// - and validate that `set_reference` isn't being misused.
    ///
    /// The `Scale` buffers' `width × height` reflect the **strip**
    /// dimensions (image_w × (h_body + 2*halo)), not the full image.
    strip: Option<StripMeta>,

    /// Strip-mode mode-E cached reference state — full-image-sized
    /// ref-side buffers per pyramid scale. Populated by
    /// [`Self::set_reference`] when invoked on a strip-mode instance;
    /// consumed by [`Self::compute_with_reference`] which walks the
    /// distorted side in strips and reads ref maps from the cached
    /// full buffers via `error_maps_strip_from_full_ref`. Lazy-allocated
    /// on the first `set_reference` call. `None` for whole-image
    /// instances (those use the per-scale buffers directly).
    strip_cached_ref: Option<Vec<StripCachedRefScale>>,

    /// Selected blur kernel. Defaults to `Ssim2Blur::Iir` (the canonical
    /// libjxl recursive Gaussian — bit-identical to the pre-T_y.B
    /// behaviour). Set via `with_blur` / `set_blur`. The non-default
    /// `Ssim2Blur::Fir` is the separable 5-tap truncated Gaussian D=5
    /// from Kanetaka et al. IWAIT 2026 — a **distinct metric** with a
    /// different per-image score scale; tag via [`crate::SSIM2_FIR_COLUMN_NAME`].
    ///
    /// Field is gated behind the `fir` Cargo feature — without the
    /// feature the IIR path is the only blur and there's no per-instance
    /// blur state to carry.
    #[cfg(feature = "fir")]
    blur: Ssim2Blur,
}

const NUM_SLOTS: usize = NUM_SCALES * 3 * 3; // 54
const PARTIALS_LEN: usize = NUM_SLOTS * reduction::PARTIALS_PER_REDUCTION;
const SUMS_LEN: usize = NUM_SLOTS * 2;

impl<R: Runtime> Ssim2<R> {
    /// Allocate every per-instance buffer for the given image size.
    /// Returns `Err(InvalidImageSize)` for images smaller than 8×8.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let s = Ssim2::<WgpuRuntime>::new(client, 1024, 768)?;
    /// assert_eq!(s.dimensions(), (1024, 768));
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    /// Unified [`MemoryMode`](crate::MemoryMode) constructor. Phase 2
    /// (2026-05-22): `Strip` now ships — constructs via `new_strip`
    /// with the requested body height (defaulting to
    /// [`crate::memory_mode::STRIP_H_BODY_DEFAULT`]). `Tile` still
    /// returns [`crate::Error::ModeUnsupported`]. Auto picks Full
    /// when it fits the cap and Strip otherwise; if even Strip
    /// exceeds the cap, surfaces [`crate::Error::TooBigForFull`].
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
                let h = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, cap)
                });
                Self::new_strip(client, width, height, h)
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

    pub fn new(client: ComputeClient<R>, width: u32, height: u32) -> Result<Self> {
        if width < 8 || height < 8 {
            return Err(Error::InvalidImageSize);
        }
        let n = (width as usize) * (height as usize);

        // Pyramid dimensions: 6 levels, each ceil(prev/2). Stop early
        // if a level would shrink below 8×8 — same as the CPU crate.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = width;
        let mut h = height;
        for _ in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        // T4.L (2026-05-16): pack 3 sRGB bytes per pixel into one u32
        // (R | G<<8 | B<<16). Length = n, not n*3. Cuts the per-call
        // host→device upload from `n_pixels × 12 B` to `n_pixels × 4 B`.
        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));

        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; SUMS_LEN]));

        Ok(Self {
            client,
            width,
            height,
            n,
            src_u8_a,
            src_u8_b,
            scales,
            partials,
            sums,
            has_cached_reference: false,
            cached_ref_is_untransposed: true,
            strip: None,
            strip_cached_ref: None,
            #[cfg(feature = "fir")]
            blur: Ssim2Blur::default(),
        })
    }

    /// Strip-processing constructor (Phase 2, 2026-05-22). Allocates
    /// working-set buffers sized for one strip
    /// (`image_w × (h_body + 2 × STRIP_HALO_ROWS)`) and configures
    /// the instance so [`Self::compute_stripped`] can loop strips with
    /// halo overlap, accumulating per-strip partial sums host-side.
    ///
    /// Memory cost is a function of `h_body`, not `image_h` — see
    /// [`crate::memory_mode::estimate_strip_gpu_memory_bytes`] for the
    /// per-strip estimator. The whole-image API (`compute` /
    /// `compute_with_mode`) is unavailable on strip-mode instances and
    /// will return [`Error::DimensionMismatch`] (the strip-sized scale-0
    /// buffer can't hold a full-frame upload).
    ///
    /// Task #46 (2026-05-26): strip-mode `set_reference` is now
    /// supported via mode E (ref full on device, dist walks in
    /// strips). The reference-side state is built once at full image
    /// resolution; subsequent `compute_with_reference` calls walk the
    /// distorted side in strips and read ref values from the cached
    /// full buffers. Memory cost adds ~3 × full-image plane size per
    /// scale on top of the strip working set.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new_strip(client, 6000, 4000, 1024)?;
    /// let r = vec![0_u8; 6000 * 4000 * 3];
    /// let d = vec![0_u8; 6000 * 4000 * 3];
    /// let score = s.compute_stripped(&r, &d)?.score;
    /// # let _ = score;
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn new_strip(
        client: ComputeClient<R>,
        image_w: u32,
        image_h: u32,
        h_body: u32,
    ) -> Result<Self> {
        if image_w < 8 || image_h < 8 {
            return Err(Error::InvalidImageSize);
        }
        if h_body == 0 {
            return Err(Error::InvalidImageSize);
        }
        let halo = crate::memory_mode::STRIP_HALO_ROWS;
        // Clamp h_body so a single-strip computation works on small
        // images. If image_h ≤ h_body + 2*halo we just allocate enough
        // for one whole-image-sized strip.
        let h_body_eff = h_body.min(image_h);
        // The strip-0 height — what the per-scale buffers must hold.
        // Cap at image_h so we don't over-allocate when image_h is
        // smaller than h_body + 2*halo.
        let strip_h0 = h_body_eff
            .saturating_add(2 * halo)
            .min(image_h.saturating_add(2 * halo));
        // For *truly* tiny images we want at least image_h rows; one
        // strip suffices and halo regions are simply empty (the IIR
        // zero-pad already handles this).
        let alloc_h = strip_h0.max(image_h.min(strip_h0));
        let n = (image_w as usize) * (alloc_h as usize);

        // Pyramid dimensions: scale s has w = ceil(image_w / 2^s),
        // h = ceil(alloc_h / 2^s). Stop early when below 8×8.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = image_w;
        let mut h = alloc_h;
        for _ in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let scales = dims
            .iter()
            .map(|&(w, h)| Scale::new(&client, w, h))
            .collect::<Vec<_>>();

        let src_u8_a = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));
        let src_u8_b = client.create_from_slice(u32::as_bytes(&vec![0_u32; n]));

        let partials = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; PARTIALS_LEN]));
        let sums = client.create_from_slice(f32::as_bytes(&vec![0.0_f32; SUMS_LEN]));

        Ok(Self {
            client,
            width: image_w,
            height: alloc_h,
            n,
            src_u8_a,
            src_u8_b,
            scales,
            partials,
            sums,
            has_cached_reference: false,
            cached_ref_is_untransposed: true,
            strip: Some(StripMeta {
                image_w,
                image_h,
                h_body: h_body_eff,
                halo,
            }),
            strip_cached_ref: None,
            #[cfg(feature = "fir")]
            blur: Ssim2Blur::default(),
        })
    }

    pub fn dimensions(&self) -> (u32, u32) {
        // Strip-mode instances report the IMAGE dimensions (not the
        // strip dimensions) so downstream callers see the size the
        // caller passed to `new_strip` — matches the contract that
        // dimensions() echoes the constructor's input.
        if let Some(m) = self.strip {
            return (m.image_w, m.image_h);
        }
        (self.width, self.height)
    }

    /// True if this instance was constructed via [`Self::new_strip`].
    /// Strip-mode and whole-image methods are mutually exclusive:
    /// strip-mode rejects `compute` / `compute_with_mode` /
    /// `set_reference`; whole-image rejects `compute_stripped`.
    pub fn is_strip_mode(&self) -> bool {
        self.strip.is_some()
    }

    /// Builder-style blur selector — **gated behind the `fir` Cargo
    /// feature**. Returns `self` so callers can chain:
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::{Ssim2, Ssim2Blur};
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?
    ///     .with_blur(Ssim2Blur::Iir);
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    ///
    /// **Switching the blur mode invalidates any cached reference.**
    /// Subsequent `compute_with_reference` calls will fail with
    /// `Error::NoCachedReference` until `set_reference` is called again
    /// — the cached pyramid + XYB + ref²-blur state is blur-specific.
    #[cfg(feature = "fir")]
    pub fn with_blur(mut self, blur: Ssim2Blur) -> Self {
        self.set_blur(blur);
        self
    }

    /// In-place blur selector — **gated behind the `fir` Cargo feature**.
    /// See `with_blur` for semantics and the cache-invalidation note.
    #[cfg(feature = "fir")]
    pub fn set_blur(&mut self, blur: Ssim2Blur) {
        if blur != self.blur {
            self.has_cached_reference = false;
        }
        self.blur = blur;
    }

    /// Currently-selected blur mode — **gated behind the `fir` Cargo
    /// feature**.
    #[cfg(feature = "fir")]
    pub fn blur(&self) -> Ssim2Blur {
        self.blur
    }

    /// Number of active pyramid scales (≤ NUM_SCALES; smaller for
    /// images that shrink below 8×8 before reaching the 6th level).
    pub fn n_scales(&self) -> usize {
        self.scales.len()
    }

    /// `(width, height, n_pixels)` at scale `s`.
    pub fn scale_dims(&self, s: usize) -> (u32, u32, usize) {
        let sc = &self.scales[s];
        (sc.width, sc.height, sc.n)
    }

    /// Cached reference handles needed by `Ssim2Batch`. After
    /// `set_reference`, these hold:
    /// - `ref_xyb_t[ch]`: transposed raw reference XYB (`source` input
    ///   to `error_maps`).
    /// - `mu1_full[ch]`: fully-blurred reference XYB (transposed
    ///   orientation, `mu1` input to `error_maps`).
    /// - `sigma11_full[ch]`: fully-blurred ref·ref (`sigma11` input).
    /// - `ref_xyb[ch]`: raw reference XYB (used by `Ssim2Batch` for
    ///   the broadcast `sigma12 = ref_xyb · dis_xyb_batched` mul).
    pub(crate) fn cached_ref_xyb_t(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].ref_xyb_t
    }
    pub(crate) fn cached_mu1_full(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].mu1_full
    }
    pub(crate) fn cached_sigma11_full(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].sigma11_full
    }
    pub(crate) fn cached_ref_xyb(&self, s: usize) -> &[cubecl::server::Handle; 3] {
        &self.scales[s].ref_xyb
    }
    pub(crate) fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Score one image pair, both sRGB packed RGB u8 of length
    /// `width × height × 3`.
    ///
    /// Returns `Err(DimensionMismatch)` if either buffer's length
    /// doesn't match the configured image size.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// let r = vec![0_u8; 256 * 256 * 3];
    /// let d = vec![0_u8; 256 * 256 * 3];
    /// let score = s.compute(&r, &d)?.score;
    /// assert!((score - 100.0).abs() < 0.1); // identical → ~100
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn compute(&mut self, ref_srgb: &[u8], dist_srgb: &[u8]) -> Result<GpuSsim2Result> {
        self.compute_with_mode(Ssim2Mode::default(), ref_srgb, dist_srgb)
    }

    /// Pack the caller's `width × height × 3` sRGB-u8 bytes into a
    /// `width × height` packed-u32 device handle (`R | G<<8 | B<<16`),
    /// using the same pinned-staging fast path the internal upload
    /// uses. Cheaper than [`Self::compute`] when scoring the same
    /// pair through multiple metrics — pack once via
    /// [`Self::pack_srgb_into_packed_u32_handle`] on any one metric's
    /// client, then thread the handle through
    /// [`Self::compute_handles`] on every metric that shares the
    /// same client.
    ///
    /// Returns `Err(DimensionMismatch)` if `srgb.len() != width *
    /// height * 3`.
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        let expected = self.n * 3;
        if srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            });
        }
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
    /// handle is also expected to live on the same cubecl client
    /// that constructed this `Ssim2<R>`; sharing handles across
    /// clients is undefined behaviour at the cubecl layer and is
    /// not validated here.
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuSsim2Result> {
        self.compute_handles_with_mode(Ssim2Mode::default(), ref_handle, dis_handle)
    }

    /// Mode-explicit counterpart of [`Self::compute_handles`] — same
    /// skip-map semantics as [`Self::compute_with_mode`].
    pub fn compute_handles_with_mode(
        &mut self,
        mode: Ssim2Mode,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<GpuSsim2Result> {
        if self.strip.is_some() {
            return Err(Error::ModeUnsupported(
                "compute_handles is whole-image only; strip-mode instances must use compute_stripped",
            ));
        }
        // Same zero-fill discipline as `compute_with_mode`. See that
        // method's comment for rationale.
        reduction::launch_zero_fill_f32(&self.client, self.partials.clone(), PARTIALS_LEN);

        self.install_packed_handle(true, ref_handle);
        self.install_packed_handle(false, dis_handle);

        let last_active = (0..self.scales.len())
            .rev()
            .find(|&s| !skip_scale(mode, s))
            .unwrap_or(0);
        self.build_linear_pyramid_until(true, last_active);
        self.build_linear_pyramid_until(false, last_active);

        for s in 0..self.scales.len() {
            if skip_scale(mode, s) {
                continue;
            }
            self.process_scale(s, mode);
        }
        self.run_finalizer();

        Ok(GpuSsim2Result {
            score: self.read_and_aggregate(),
        })
    }

    /// Score one image pair under the chosen [`Ssim2Mode`]. Identical
    /// to [`Ssim2::compute`] but with explicit control over the
    /// skip-map dispatch — `Ssim2Mode::Full` matches the pre-skip-map
    /// behaviour bit-for-bit; the more aggressive modes skip cells
    /// whose contribution to the final score is bounded below the
    /// mode's threshold.
    ///
    /// See `crates/ssim2-gpu/docs/SKIP_MAP_AUDIT.md` for the per-cell
    /// audit and the threshold rationale.
    pub fn compute_with_mode(
        &mut self,
        mode: Ssim2Mode,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result> {
        if self.strip.is_some() {
            // Strip-mode instances allocate strip-sized scale-0 buffers
            // (image_w × (h_body + 2*halo)); they can't hold a full-frame
            // upload. Route the caller to compute_stripped instead.
            return self.compute_stripped_with_mode(mode, ref_srgb, dist_srgb);
        }
        self.check_dims(ref_srgb)?;
        self.check_dims(dist_srgb)?;

        // Per-call zero-fill of partials so:
        // (a) Skipped reduction slots in non-Full mode contribute
        //     exactly 0 to the host's weighted-sum fold.
        // (b) Fast-reduction `Atomic<f32>::fetch_add` accumulates on
        //     top of a clean zero (otherwise the prior call's per-slot
        //     sum would carry over).
        // Promoting this to a per-call pre-step subsumes the previous
        // end-of-call zero in `read_and_aggregate` (now removed) and
        // makes the portable-reduction path safe for skip-map dispatch
        // as well — neither mode is sensitive to whether the prior
        // call zeroed partials or not.
        reduction::launch_zero_fill_f32(&self.client, self.partials.clone(), PARTIALS_LEN);

        // Upload + sRGB → linear for both sides into scale-0 buffers.
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.upload_and_srgb_to_linear(false, dist_srgb);

        // Build linear pyramid only up to the deepest non-skipped
        // scale. The downscale chain is recursive, so if scale `S+1`
        // is the deepest active scale we still need to downscale into
        // `S+1`; if scale `S+2` and beyond are all skip-scale, those
        // downscales are wasted compute.
        let last_active = (0..self.scales.len())
            .rev()
            .find(|&s| !skip_scale(mode, s))
            .unwrap_or(0);
        self.build_linear_pyramid_until(true, last_active);
        self.build_linear_pyramid_until(false, last_active);

        // Per-scale processing — populates per-thread partials.
        for s in 0..self.scales.len() {
            if skip_scale(mode, s) {
                continue;
            }
            self.process_scale(s, mode);
        }
        // Stage-2 finalizer folds partials → small (slot, sum, p4) buffer.
        self.run_finalizer();

        Ok(GpuSsim2Result {
            score: self.read_and_aggregate(),
        })
    }

    /// Cache reference-side state for many comparisons against a fixed
    /// reference. Subsequent `compute_with_reference` calls skip the
    /// reference-side pyramid + XYB + ref²-blur work.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::Ssim2;
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// let r = vec![0_u8; 256 * 256 * 3];
    /// s.set_reference(&r)?;
    /// assert!(s.has_cached_reference());
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn set_reference(&mut self, ref_srgb: &[u8]) -> Result<()> {
        if self.strip.is_some() {
            return self.set_reference_strip_mode(ref_srgb);
        }
        self.check_dims(ref_srgb)?;
        self.upload_and_srgb_to_linear(true, ref_srgb);
        self.build_linear_pyramid(true);
        let mut blur_untransposed = true;
        for s in 0..self.scales.len() {
            self.run_xyb(s, true);
            self.run_self_products(s, true);
            blur_untransposed = self.run_blur_pair(s, true) && blur_untransposed;
            // Pre-transpose the raw reference XYB into ref_xyb_t. The
            // single-image cached-ref path (compute_with_reference_with_mode)
            // doesn't need this when blurs are untransposed (IIR), but
            // the batched path (Ssim2Batch::compute_batch) reads ref_xyb_t
            // via the `cached_ref_xyb_t` accessor in
            // error_maps_broadcast_batched_kernel — that path is still on
            // the transposed-orientation contract. set_reference runs
            // ONCE per encoder loop so the per-scale transpose is cheap
            // (called once, then thousands of dist-side compute calls
            // benefit from the cache). Migrating Ssim2Batch to the
            // untransposed orientation is queued as a follow-on.
            self.run_transpose_raw_xyb_pair(s, true, false);
        }
        self.cached_ref_is_untransposed = blur_untransposed;
        self.has_cached_reference = true;
        Ok(())
    }

    /// Strip-mode mode-E set_reference (task #46). Builds the
    /// **full-image** reference-side state (raw transposed XYB, blurred
    /// mu1, blurred sigma11) per pyramid scale into the
    /// `strip_cached_ref` slot, then subsequent
    /// [`Self::compute_with_reference`] walks the distorted side in
    /// strips and reads ref values from these full buffers.
    ///
    /// Uses temporary full-image-sized scratch buffers for the linear
    /// pyramid + XYB + sigma11_in + the two-pass-blur scratch. The
    /// scratch is dropped at the end of this call. Only the three
    /// "final" ref maps per scale × channel persist in
    /// `strip_cached_ref`.
    fn set_reference_strip_mode(&mut self, ref_srgb: &[u8]) -> Result<()> {
        let meta = self.strip.expect("strip mode required");
        let expected = (meta.image_w as usize) * (meta.image_h as usize) * 3;
        if ref_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: ref_srgb.len(),
            });
        }

        // Build per-scale full-image dimensions.
        let mut dims = Vec::with_capacity(NUM_SCALES);
        let mut w = meta.image_w;
        let mut h = meta.image_h;
        for _ in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            dims.push((w, h));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        // Lazily allocate the cached-ref store. The cache shape is
        // fixed by the strip-mode instance's (image_w, image_h), so a
        // single allocation is reused across `set_reference` calls.
        if self
            .strip_cached_ref
            .as_ref()
            .map(|v| v.len() != dims.len())
            .unwrap_or(true)
        {
            let cache = dims
                .iter()
                .map(|&(w, h)| StripCachedRefScale::new(&self.client, w, h))
                .collect::<Vec<_>>();
            self.strip_cached_ref = Some(cache);
        }

        // Temporary full-image scratch per scale, just for the ref-side
        // pipeline. Allocated on a per-scale basis so peak working-set
        // is one scale's scratch at a time.
        //
        // Layout per scale: 3-channel ref_lin (input to this scale's
        // XYB), 3-channel ref_xyb, 3-channel sigma11_in (ref²),
        // 3-channel v_scratch, 3-channel t_scratch. Final 3-channel
        // outputs (ref_xyb_t_full, mu1_full_full, sigma11_full_full)
        // live in `strip_cached_ref`.
        //
        // To build the pyramid we keep ref_lin live across all scales
        // — each scale's downscale reads from scale s-1 and writes
        // into scale s. The other buffers (XYB, sigma11_in, scratch)
        // can be dropped scale-by-scale, but the simplest path
        // pre-allocates one full pyramid's worth of ref_lin and
        // reuses per-scale temp buffers across calls.

        // 1. Upload the reference + sRGB→linear at scale 0 into a
        //    full-image-sized temp ref_lin buffer.
        let n0 = (meta.image_w as usize) * (meta.image_h as usize);
        let pinned_len = n0 * 4;
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
        let packed_handle = self.client.create(bytes);

        // Build full ref_lin pyramid.
        let mut full_ref_lin: Vec<[cubecl::server::Handle; 3]> = Vec::with_capacity(dims.len());
        for &(w, h) in &dims {
            let n = (w as usize) * (h as usize);
            full_ref_lin.push(alloc_3(&self.client, n));
        }
        // sRGB→linear into scale 0.
        unsafe {
            srgb::srgb_u8_to_linear_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(n0),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(packed_handle.clone(), n0),
                ArrayArg::from_raw_parts(full_ref_lin[0][0].clone(), n0),
                ArrayArg::from_raw_parts(full_ref_lin[0][1].clone(), n0),
                ArrayArg::from_raw_parts(full_ref_lin[0][2].clone(), n0),
            );
        }
        // Downscale chain — one launch per scale transition via the
        // 3-channel-fused kernel.
        for s in 1..dims.len() {
            let (pw, ph) = dims[s - 1];
            let (cw, ch_) = dims[s];
            let n_prev = (pw as usize) * (ph as usize);
            let n_curr = (cw as usize) * (ch_ as usize);
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_curr),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(full_ref_lin[s - 1][0].clone(), n_prev),
                    ArrayArg::from_raw_parts(full_ref_lin[s - 1][1].clone(), n_prev),
                    ArrayArg::from_raw_parts(full_ref_lin[s - 1][2].clone(), n_prev),
                    ArrayArg::from_raw_parts(full_ref_lin[s][0].clone(), n_curr),
                    ArrayArg::from_raw_parts(full_ref_lin[s][1].clone(), n_curr),
                    ArrayArg::from_raw_parts(full_ref_lin[s][2].clone(), n_curr),
                    pw,
                    ph,
                    cw,
                    ch_,
                );
            }
        }

        // Per-scale: XYB → sigma11_in (ref²) → blur(sigma11) + blur(ref_xyb)
        // → transpose(ref_xyb). Write outputs into `strip_cached_ref`.
        for s in 0..dims.len() {
            let (w, h) = dims[s];
            let n = (w as usize) * (h as usize);

            // Allocate per-scale temp working buffers. sigma11_in and
            // the two blur scratches are dropped at end of this scale;
            // the XYB output we write directly into the cached
            // `ref_xyb_full` slot.
            let sigma11_in_temp = alloc_3(&self.client, n);
            let v_scratch_temp = alloc_3(&self.client, n);
            let t_scratch_temp = alloc_3(&self.client, n);

            // linear → XYB (3-in, 3-out fused kernel). Output lands
            // directly in the cached pre-transpose ref_xyb_full slot
            // so the strip-mode compute can read it without a second
            // copy. Borrow the cache via the `clone()`-into-handle
            // idiom that the rest of the pipeline uses.
            let cache_s_handles = {
                let cache_s = &self.strip_cached_ref.as_ref().unwrap()[s];
                debug_assert_eq!(cache_s.n, n);
                (
                    [
                        cache_s.ref_xyb_full[0].clone(),
                        cache_s.ref_xyb_full[1].clone(),
                        cache_s.ref_xyb_full[2].clone(),
                    ],
                    [
                        cache_s.ref_xyb_t_full[0].clone(),
                        cache_s.ref_xyb_t_full[1].clone(),
                        cache_s.ref_xyb_t_full[2].clone(),
                    ],
                    [
                        cache_s.mu1_full_full[0].clone(),
                        cache_s.mu1_full_full[1].clone(),
                        cache_s.mu1_full_full[2].clone(),
                    ],
                    [
                        cache_s.sigma11_full_full[0].clone(),
                        cache_s.sigma11_full_full[1].clone(),
                        cache_s.sigma11_full_full[2].clone(),
                    ],
                )
            };
            let (ref_xyb_full_h, ref_xyb_t_full_h, mu1_full_h, sigma11_full_h) = cache_s_handles;

            unsafe {
                xyb::linear_to_xyb_planar_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(full_ref_lin[s][0].clone(), n),
                    ArrayArg::from_raw_parts(full_ref_lin[s][1].clone(), n),
                    ArrayArg::from_raw_parts(full_ref_lin[s][2].clone(), n),
                    ArrayArg::from_raw_parts(ref_xyb_full_h[0].clone(), n),
                    ArrayArg::from_raw_parts(ref_xyb_full_h[1].clone(), n),
                    ArrayArg::from_raw_parts(ref_xyb_full_h[2].clone(), n),
                );
            }

            // sigma11_in = ref · ref (per channel).
            for chi in 0..3 {
                unsafe {
                    pointwise_mul_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::cube_count_1d(n),
                        Self::cube_dim_1d(),
                        ArrayArg::from_raw_parts(ref_xyb_full_h[chi].clone(), n),
                        ArrayArg::from_raw_parts(ref_xyb_full_h[chi].clone(), n),
                        ArrayArg::from_raw_parts(sigma11_in_temp[chi].clone(), n),
                    );
                }
            }

            for chi in 0..3 {
                // sigma11_full = blur(ref²)
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &sigma11_in_temp[chi],
                    &v_scratch_temp[chi],
                    &t_scratch_temp[chi],
                    &sigma11_full_h[chi],
                );
                // mu1_full = blur(ref_xyb)
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &ref_xyb_full_h[chi],
                    &v_scratch_temp[chi],
                    &t_scratch_temp[chi],
                    &mu1_full_h[chi],
                );
                // ref_xyb_t_full = transpose(ref_xyb)
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::transpose_cube_count(w, h),
                        Self::transpose_cube_dim(),
                        ArrayArg::from_raw_parts(ref_xyb_full_h[chi].clone(), n),
                        ArrayArg::from_raw_parts(ref_xyb_t_full_h[chi].clone(), n),
                        w,
                        h,
                    );
                }
            }
            // temp handles drop here at scope end — cubecl returns the
            // memory to the runtime allocator. ref_lin we keep across
            // all scales (the downscale chain reads from scale s-1
            // when building scale s).
        }

        self.has_cached_reference = true;
        Ok(())
    }

    /// Drop any cached reference state.
    pub fn clear_reference(&mut self) {
        self.has_cached_reference = false;
        // For strip mode, also release the cached ref buffers. For
        // whole-image mode the per-scale buffers stay live (they're
        // pre-allocated as part of the instance) and just the flag
        // flips to false.
        self.strip_cached_ref = None;
    }

    pub fn has_cached_reference(&self) -> bool {
        self.has_cached_reference
    }

    /// Compute against the cached reference. Returns
    /// `Err(NoCachedReference)` if `set_reference` hasn't been called,
    /// `Err(DimensionMismatch)` if the buffer's length doesn't match
    /// the configured image size.
    ///
    /// ```no_run
    /// use cubecl::Runtime;
    /// use cubecl::wgpu::WgpuRuntime;
    /// use ssim2_gpu::{Error, Ssim2};
    ///
    /// let client = WgpuRuntime::client(&Default::default());
    /// let mut s = Ssim2::<WgpuRuntime>::new(client, 256, 256)?;
    /// // Without set_reference first:
    /// let d = vec![0_u8; 256 * 256 * 3];
    /// assert!(matches!(
    ///     s.compute_with_reference(&d),
    ///     Err(Error::NoCachedReference)
    /// ));
    /// # Ok::<(), ssim2_gpu::Error>(())
    /// ```
    pub fn compute_with_reference(&mut self, dist_srgb: &[u8]) -> Result<GpuSsim2Result> {
        self.compute_with_reference_with_mode(Ssim2Mode::default(), dist_srgb)
    }

    /// Score against the cached reference under the chosen
    /// [`Ssim2Mode`]. Same skip-map semantics as
    /// [`Ssim2::compute_with_mode`]. `set_reference` caches every
    /// scale × channel so a single cached reference can be re-used
    /// across calls with different modes.
    pub fn compute_with_reference_with_mode(
        &mut self,
        mode: Ssim2Mode,
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result> {
        if self.strip.is_some() {
            return self.compute_with_reference_strip_with_mode(mode, dist_srgb);
        }
        if !self.has_cached_reference {
            return Err(Error::NoCachedReference);
        }
        self.check_dims(dist_srgb)?;

        // See `compute_with_mode` for the rationale on per-call zeroing.
        reduction::launch_zero_fill_f32(&self.client, self.partials.clone(), PARTIALS_LEN);

        self.upload_and_srgb_to_linear(false, dist_srgb);

        let last_active = (0..self.scales.len())
            .rev()
            .find(|&s| !skip_scale(mode, s))
            .unwrap_or(0);
        self.build_linear_pyramid_until(false, last_active);

        // Cached-ref path: the ref-side cache (mu1_full, sigma11_full,
        // ref_xyb_t) is committed to TRANSPOSED orientation today (so
        // Ssim2Batch's broadcast_batched_kernel keeps working — see
        // `run_blur_pair` for the rationale). The dis-side blurs MUST
        // match. Use the legacy transposed-blur path for parity. The
        // process_scale (non-cached compute) path uses the untransposed
        // fast path independently.
        for s in 0..self.scales.len() {
            if skip_scale(mode, s) {
                continue;
            }
            self.run_xyb_masked(s, false, mode);
            self.run_self_products_masked(s, false, mode); // sigma22
            self.run_cross_product_masked(s, mode); // sigma12
            self.run_blur_dis_only_masked_transposed(s, mode);
            // Cached-ref dis-side transpose: pair of raw transposes.
            self.run_transpose_raw_xyb_pair_masked(s, false, true, mode);
            self.run_error_maps_masked_oriented(s, mode, false);
            self.run_reductions_masked(s, mode);
        }
        self.run_finalizer();

        Ok(GpuSsim2Result {
            score: self.read_and_aggregate(),
        })
    }

    // ────────────────── strip processing (cached-ref mode E) ──────────────────

    /// Strip-mode mode-E `compute_with_reference_with_mode` (task #46).
    /// Walks the distorted side in strips; reads reference-side state
    /// from the full-image `strip_cached_ref` buffers populated by
    /// [`Self::set_reference_strip_mode`].
    fn compute_with_reference_strip_with_mode(
        &mut self,
        mode: Ssim2Mode,
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result> {
        if !self.has_cached_reference || self.strip_cached_ref.is_none() {
            return Err(Error::NoCachedReference);
        }
        let meta = self.strip.expect("strip-mode required");
        let expected = (meta.image_w as usize) * (meta.image_h as usize) * 3;
        if dist_srgb.len() != expected {
            return Err(Error::DimensionMismatch {
                expected,
                got: dist_srgb.len(),
            });
        }

        let h_body = meta.h_body;
        let halo = meta.halo;
        let image_h = meta.image_h;
        let image_w = meta.image_w;
        let strip_h0_alloc = self.scales[0].height;

        let n_slots = NUM_SLOTS;
        let mut acc_sum = vec![0.0_f64; n_slots];
        let mut acc_p4 = vec![0.0_f64; n_slots];

        let mut body_start = 0u32;
        while body_start < image_h {
            let body_end = (body_start + h_body).min(image_h);
            let strip_top = body_start.saturating_sub(halo);
            let strip_bot = (body_end + halo).min(image_h);
            let strip_h_active = strip_bot - strip_top;
            let body_col_start = body_start - strip_top;
            let body_col_end = body_end - strip_top;

            // Upload only the distorted side. Reference is already
            // resident in `strip_cached_ref` at full-image resolution.
            self.upload_strip_slice(
                false,
                dist_srgb,
                image_w,
                image_h,
                strip_top,
                strip_h_active,
                strip_h0_alloc,
            );

            reduction::launch_zero_fill_f32(&self.client, self.partials.clone(), PARTIALS_LEN);

            // Build dist-side linear pyramid only.
            let last_active = (0..self.scales.len())
                .rev()
                .find(|&s| !skip_scale(mode, s))
                .unwrap_or(0);
            self.build_linear_pyramid_until(false, last_active);

            for s in 0..self.scales.len() {
                if skip_scale(mode, s) {
                    continue;
                }
                let s_u = s as u32;
                let scale_strip_h = self.scales[s].height;
                let scale_body_start = body_col_start >> s_u;
                let scale_body_end = ((body_col_end + (1 << s_u) - 1) >> s_u).min(scale_strip_h);
                // The strip's top edge in full-image scale-s row coords.
                // Scale-s row offset = floor(strip_top / 2^s) — same
                // semantics as the scale-0 → scale-s downscale chain
                // uses (which is div_ceil for height, but the body
                // alignment uses floor on the top edge to keep the
                // first body sample in scope).
                let strip_top_at_s = strip_top >> s_u;
                self.process_scale_strip_cached_ref(
                    s,
                    mode,
                    scale_strip_h,
                    scale_body_start,
                    scale_body_end,
                    strip_top_at_s,
                );
            }

            self.run_finalizer();
            let bytes = self
                .client
                .read_one(self.sums.clone())
                .expect("read sums buffer (strip cached-ref)");
            let raw = f32::from_bytes(&bytes);
            debug_assert_eq!(raw.len(), SUMS_LEN);
            for slot in 0..n_slots {
                acc_sum[slot] += raw[slot * 2] as f64;
                acc_p4[slot] += raw[slot * 2 + 1] as f64;
            }
            body_start = body_end;
        }

        Ok(GpuSsim2Result {
            score: self.aggregate_from_accumulators(&acc_sum, &acc_p4, meta),
        })
    }

    /// Per-scale strip processing for mode E. Builds dist-side state
    /// only; uses cached full-image ref state for sigma12 (via the
    /// offset-mul kernel) and error_maps (via the
    /// `error_maps_strip_from_full_ref` kernel).
    fn process_scale_strip_cached_ref(
        &self,
        scale: usize,
        mode: Ssim2Mode,
        scale_strip_h: u32,
        body_col_start: u32,
        body_col_end: u32,
        strip_top_at_s: u32,
    ) {
        // dist-side XYB
        self.run_xyb_masked(scale, false, mode);
        // Zero the pad-row region of dis_xyb at this scale. Strip's
        // upload zeros the linear-RGB pad rows, but XYB(linear 0) =
        // (0.42, 0.01, 0.55) (non-zero bias), so without this clear
        // the subsequent blur at body-bottom rows would see bias values
        // at pad rows. Whole-image computation has no pad rows and the
        // blur reads "out of bounds = 0" at image bottom — clearing
        // the dis-side pad rows here makes the strip's body-bottom
        // blur match the whole-image body-bottom blur.
        //
        // For non-bottom strips, the strip's body bottom is `halo`
        // rows above any pad — the IIR's radius-4 reach can't see
        // pad rows from the body, so this clear is a no-op in
        // practice for those strips. For the bottom strip (where
        // strip_h_active = image_h - strip_top), the pad region IS
        // adjacent to body bottom and this clear is load-bearing.
        self.zero_dis_pad_rows(scale, strip_top_at_s, mode);
        // sigma22 = dis · dis
        self.run_self_products_masked(scale, false, mode);
        // sigma12 = ref_full_slice · dis, where the ref slice is the
        // strip's body+halo region of the cached full ref_xyb plane.
        // Pre-transpose orientation: flat layout `image_h_s` rows of
        // `image_w_s` cols, so the strip's slice starts at flat index
        // `strip_top_at_s * image_w_s`.
        self.run_cross_product_strip_cached_ref(scale, mode, strip_top_at_s);
        // Strip-mode mode-E uses transposed orientation throughout (the
        // ref-side cached `*_t_full` slots are transposed). Use the
        // legacy v+t+v blur path for dis-side consistency.
        self.run_blur_dis_only_masked_transposed(scale, mode);
        // Transpose raw dist XYB (ref's transpose is already cached).
        self.run_transpose_raw_xyb_pair_masked(scale, false, true, mode);
        // Error maps: ref-side inputs from the cached full buffers,
        // dist-side from the strip buffers. Output is strip-shape.
        self.run_error_maps_strip_cached_ref(scale, mode, strip_top_at_s);
        // Standard strip-mode reduction over output ssim/artifact/detail
        // planes, filtered to body inner indices.
        self.run_reductions_strip_masked(scale, mode, scale_strip_h, body_col_start, body_col_end);
    }

    /// Strip-mode mode-E: zero out the `dis_xyb` pad-row region for
    /// this scale. Pad rows are strip-buffer rows beyond the actual
    /// image data (`image_h_at_s - strip_top_at_s`). For the bottom
    /// strip this is non-trivial since alloc extends past image_h;
    /// for non-bottom strips the pad region is far from body and
    /// clearing it is a no-op for the score.
    fn zero_dis_pad_rows(&self, scale: usize, strip_top_at_s: u32, mode: Ssim2Mode) {
        let s = &self.scales[scale];
        let cache_s = &self
            .strip_cached_ref
            .as_ref()
            .expect("strip_cached_ref populated")[scale];
        // image_h_at_s = cache_s.full_h. Real-data rows in the strip
        // at this scale = image_h_at_s - strip_top_at_s, clamped to
        // strip's own height (a non-bottom strip's full strip range
        // is entirely within image).
        let image_h_at_s = cache_s.full_h;
        let real_rows = image_h_at_s.saturating_sub(strip_top_at_s).min(s.height);
        if real_rows >= s.height {
            // No pad rows to clear.
            return;
        }
        // Flat-index start of pad region in pre-transpose layout
        // (strip is `s.height rows × s.width cols` row-major).
        let start_idx = real_rows * s.width;
        for chi in 0..3 {
            if skip_error_map(mode, scale, chi) {
                continue;
            }
            unsafe {
                crate::kernels::error_maps::zero_tail_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(s.dis_xyb[chi].clone(), s.n),
                    start_idx,
                );
            }
        }
    }

    /// Strip-mode mode-E sigma12 cross-product: `sigma12_in_strip =
    /// ref_xyb_full[offset..] · dis_xyb_strip`. The offset is the
    /// pre-transpose flat index where the strip's first row lands in
    /// the full ref plane.
    fn run_cross_product_strip_cached_ref(
        &self,
        scale: usize,
        mode: Ssim2Mode,
        strip_top_at_s: u32,
    ) {
        let s = &self.scales[scale];
        let cache_s = &self
            .strip_cached_ref
            .as_ref()
            .expect("strip_cached_ref populated")[scale];
        // Offset in elements (flat-idx units) into the pre-transpose
        // full ref buffer at this scale.
        let a_offset = strip_top_at_s * cache_s.width;
        for chi in 0..3 {
            if skip_error_map(mode, scale, chi) {
                continue;
            }
            unsafe {
                crate::kernels::error_maps::pointwise_mul_offset_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(cache_s.ref_xyb_full[chi].clone(), cache_s.n),
                    ArrayArg::from_raw_parts(s.dis_xyb[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma12_in[chi].clone(), s.n),
                    a_offset,
                );
            }
        }
    }

    /// Strip-mode mode-E error_maps: ref inputs from full buffers
    /// (transposed orientation, inner stride = `full_h`); dist inputs
    /// + outputs in strip-shape transposed (inner stride = `strip_h`).
    /// `strip_top_at_s` is the row offset where the strip starts in
    /// the full transposed buffer's inner-index axis.
    fn run_error_maps_strip_cached_ref(&self, scale: usize, mode: Ssim2Mode, strip_top_at_s: u32) {
        let s = &self.scales[scale];
        let cache_s = &self
            .strip_cached_ref
            .as_ref()
            .expect("strip_cached_ref populated")[scale];
        let strip_inner_stride = s.height; // transposed strip inner dim
        let full_inner_stride = cache_s.full_h;
        for chi in 0..3 {
            if skip_error_map(mode, scale, chi) {
                continue;
            }
            unsafe {
                crate::kernels::error_maps::error_maps_strip_from_full_ref_kernel::launch_unchecked::<
                    R,
                >(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(cache_s.ref_xyb_t_full[chi].clone(), cache_s.n),
                    ArrayArg::from_raw_parts(s.dis_xyb_t[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(cache_s.mu1_full_full[chi].clone(), cache_s.n),
                    ArrayArg::from_raw_parts(s.mu2_full[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(cache_s.sigma11_full_full[chi].clone(), cache_s.n),
                    ArrayArg::from_raw_parts(s.sigma22_full[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma12_full[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(s.ssim[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(s.artifact[chi].clone(), s.n),
                    ArrayArg::from_raw_parts(s.detail[chi].clone(), s.n),
                    strip_inner_stride,
                    full_inner_stride,
                    strip_top_at_s,
                );
            }
        }
    }

    // ───────────────────────── strip processing ─────────────────────────

    /// Strip-processing driver. Public entry point for strip-mode
    /// instances (`new_strip`). Slices the input into strips with halo
    /// overlap, runs the pipeline per strip with the body row range
    /// passed to the reduction kernel, and accumulates partial sums
    /// host-side.
    ///
    /// Returns `Err(DimensionMismatch)` if either buffer's length doesn't
    /// match `image_w × image_h × 3` (the dimensions passed to `new_strip`).
    pub fn compute_stripped(
        &mut self,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result> {
        self.compute_stripped_with_mode(Ssim2Mode::default(), ref_srgb, dist_srgb)
    }

    /// Mode-explicit strip driver — same skip-map semantics as
    /// [`Self::compute_with_mode`] but operates per-strip.
    pub fn compute_stripped_with_mode(
        &mut self,
        mode: Ssim2Mode,
        ref_srgb: &[u8],
        dist_srgb: &[u8],
    ) -> Result<GpuSsim2Result> {
        let meta = self.strip.ok_or(Error::ModeUnsupported(
            "compute_stripped requires strip-mode instance",
        ))?;
        let expected = (meta.image_w as usize) * (meta.image_h as usize) * 3;
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

        // Plan strips. Strip i covers body rows
        // `[i*h_body, min((i+1)*h_body, image_h))`. Around each body
        // we attach halo rows clamped to [0, image_h). The strip
        // buffer is image_w × strip_h0 (allocation size); for strips
        // shorter than that, the trailing rows are zero-padded by the
        // upload (left over from the previous strip — we zero-fill
        // them explicitly via upload).
        let h_body = meta.h_body;
        let halo = meta.halo;
        let image_h = meta.image_h;
        let image_w = meta.image_w;
        let strip_h0_alloc = self.scales[0].height; // = allocation height

        // Accumulators for (Σ, Σ⁴) per slot, in f64 to absorb f32 noise
        // across many strips.
        let n_slots = NUM_SLOTS;
        let mut acc_sum = vec![0.0_f64; n_slots];
        let mut acc_p4 = vec![0.0_f64; n_slots];

        let mut strip_idx = 0usize;
        let mut body_start = 0u32;
        while body_start < image_h {
            let body_end = (body_start + h_body).min(image_h);
            // Halo: extend halo rows above body_start and below body_end,
            // clamped to image bounds.
            let strip_top = body_start.saturating_sub(halo);
            let strip_bot = (body_end + halo).min(image_h);
            let strip_h_active = strip_bot - strip_top;
            // Within the strip-local coord system, body rows are at
            // [body_start - strip_top, body_end - strip_top).
            let body_col_start = body_start - strip_top;
            let body_col_end = body_end - strip_top;

            // Upload this strip's slice of (ref, dist) into the
            // pre-allocated strip buffers. We tightly pack into
            // image_w × strip_h_active and zero-fill the remainder up
            // to strip_h0_alloc (the trailing rows must be zero so they
            // don't contaminate the IIR boundary on shorter strips).
            self.upload_strip_slice(
                true,
                ref_srgb,
                image_w,
                image_h,
                strip_top,
                strip_h_active,
                strip_h0_alloc,
            );
            self.upload_strip_slice(
                false,
                dist_srgb,
                image_w,
                image_h,
                strip_top,
                strip_h_active,
                strip_h0_alloc,
            );

            // Run the per-strip pipeline. We re-zero the partials buffer
            // each strip; per-strip results are then read back and
            // accumulated host-side into `acc_sum` / `acc_p4`.
            reduction::launch_zero_fill_f32(&self.client, self.partials.clone(), PARTIALS_LEN);

            // Build linear pyramid over the strip dimensions (which
            // match the scale buffer dims).
            let last_active = (0..self.scales.len())
                .rev()
                .find(|&s| !skip_scale(mode, s))
                .unwrap_or(0);
            self.build_linear_pyramid_until(true, last_active);
            self.build_linear_pyramid_until(false, last_active);

            // Process each scale with the body row range.
            for s in 0..self.scales.len() {
                if skip_scale(mode, s) {
                    continue;
                }
                // Per-scale body column range (after transpose; see
                // `kernels::reduction` docstring): start = body row start
                // at scale 0, divided by 2^s. The downscale uses ceiling
                // semantics, so we use the same ceiling for the strip
                // endpoint. We use floor for the start so we don't drop
                // any active body pixels.
                let s_u = s as u32;
                let scale_strip_h = self.scales[s].height; // transposed width
                let scale_body_start = body_col_start >> s_u;
                let scale_body_end = ((body_col_end + (1 << s_u) - 1) >> s_u).min(scale_strip_h);
                self.process_scale_strip(s, mode, scale_strip_h, scale_body_start, scale_body_end);
            }

            // Stage-2 finalize → small sums buffer.
            self.run_finalizer();
            // Read sums back, accumulate.
            let bytes = self
                .client
                .read_one(self.sums.clone())
                .expect("read sums buffer (strip)");
            let raw = f32::from_bytes(&bytes);
            debug_assert_eq!(raw.len(), SUMS_LEN);
            for slot in 0..n_slots {
                acc_sum[slot] += raw[slot * 2] as f64;
                acc_p4[slot] += raw[slot * 2 + 1] as f64;
            }

            strip_idx += 1;
            body_start = body_end;
        }
        let _ = strip_idx;

        // Final aggregation. Re-uses the same WEIGHT table /
        // sigmoid as `read_and_aggregate` but driven from the
        // host-side accumulators instead of the on-device sums buffer.
        Ok(GpuSsim2Result {
            score: self.aggregate_from_accumulators(&acc_sum, &acc_p4, meta),
        })
    }

    /// Upload `image_w × strip_h_active` rows starting at row
    /// `image_y_start` from `srgb` into the scale-0 ref or dist buffer.
    /// Trailing rows up to `strip_h0_alloc` are zero-filled to keep
    /// the IIR boundary clean.
    fn upload_strip_slice(
        &mut self,
        is_a: bool,
        srgb: &[u8],
        image_w: u32,
        _image_h: u32,
        image_y_start: u32,
        strip_h_active: u32,
        strip_h0_alloc: u32,
    ) {
        let n_alloc = (image_w as usize) * (strip_h0_alloc as usize);
        let pinned_len = n_alloc * 4;
        let mut staging = self.client.reserve_staging(&[pinned_len]);
        let mut bytes = staging.pop().expect("reserve_staging returned no buffers");
        {
            let dst: &mut [u8] = &mut bytes;
            debug_assert_eq!(dst.len(), pinned_len);
            let row_stride_bytes = (image_w as usize) * 4;
            let src_row_stride = (image_w as usize) * 3;
            // 1) Active rows: pack u8×3 → u32 (R | G<<8 | B<<16).
            for sy in 0..strip_h_active as usize {
                let image_y = (image_y_start as usize) + sy;
                let src_row = &srgb[image_y * src_row_stride..(image_y + 1) * src_row_stride];
                let dst_row =
                    &mut dst[sy * row_stride_bytes..sy * row_stride_bytes + row_stride_bytes];
                for (chunk_out, triple) in dst_row.chunks_exact_mut(4).zip(src_row.chunks_exact(3))
                {
                    chunk_out[0] = triple[0];
                    chunk_out[1] = triple[1];
                    chunk_out[2] = triple[2];
                    chunk_out[3] = 0;
                }
            }
            // 2) Trailing padding rows: zero.
            let active_bytes = (strip_h_active as usize) * row_stride_bytes;
            if active_bytes < pinned_len {
                dst[active_bytes..].fill(0);
            }
        }
        let handle = self.client.create(bytes);
        if is_a {
            self.src_u8_a = handle;
        } else {
            self.src_u8_b = handle;
        }
        self.srgb_to_linear_from_packed(is_a);
    }

    /// Per-scale processing for strip mode. Mirrors `process_scale`
    /// but routes reductions through the row-range launcher with the
    /// supplied body column range.
    fn process_scale_strip(
        &self,
        scale: usize,
        mode: Ssim2Mode,
        scale_strip_h: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        self.run_xyb_masked(scale, true, mode);
        self.run_xyb_masked(scale, false, mode);
        self.run_self_products_masked(scale, true, mode);
        self.run_self_products_masked(scale, false, mode);
        self.run_cross_product_masked(scale, mode);
        // run_blur_full_masked returns false (transposed) until the
        // batched/cached-ref orientation migration lands. Use legacy
        // transposed-orientation error_maps inputs.
        self.run_blur_full_masked(scale, mode);
        self.run_transpose_raw_xyb_pair_masked(scale, true, true, mode);
        self.run_error_maps_masked_oriented(scale, mode, false);
        self.run_reductions_strip_masked(scale, mode, scale_strip_h, body_col_start, body_col_end);
    }

    /// Strip-aware reduction launcher.
    fn run_reductions_strip_masked(
        &self,
        scale: usize,
        mode: Ssim2Mode,
        scale_strip_h: u32,
        body_col_start: u32,
        body_col_end: u32,
    ) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            let plane_handles = [&s.ssim[ch], &s.artifact[ch], &s.detail[ch]];
            for map_type in 0..3 {
                if skip_reduction(mode, scale, ch, map_type) {
                    continue;
                }
                let slot = ((scale * 3 + ch) * 3 + map_type) as u32;
                reduction::launch_sum_p4_rows::<R>(
                    &self.client,
                    plane_handles[map_type].clone(),
                    s.n,
                    self.partials.clone(),
                    PARTIALS_LEN,
                    slot,
                    scale_strip_h,
                    body_col_start,
                    body_col_end,
                );
            }
        }
    }

    /// Fold host-side accumulators through the SSIMULACRA2 weight table.
    /// Same algebra as `read_and_aggregate` but with f64 accumulators
    /// summed across strips and the n_pix divisor taken from `meta` (the
    /// **full image** pixel count at each scale, not the per-strip
    /// count — every strip's body sums add up to one whole-image sum).
    fn aggregate_from_accumulators(&self, acc_sum: &[f64], acc_p4: &[f64], meta: StripMeta) -> f64 {
        let mut avg_ssim = vec![[0.0_f64; 6]; NUM_SCALES];
        let mut avg_edgediff = vec![[0.0_f64; 12]; NUM_SCALES];

        // Whole-image pixel count per scale (matches what `Ssim2::new`
        // would compute for the same image_w / image_h).
        let mut w = meta.image_w;
        let mut h = meta.image_h;
        let mut scale_npix = Vec::with_capacity(NUM_SCALES);
        for _ in 0..NUM_SCALES {
            if w < 8 || h < 8 {
                break;
            }
            scale_npix.push((w as f64) * (h as f64));
            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }
        // Match the per-scale loop in `read_and_aggregate`.
        let n_scales = scale_npix.len();

        for scale in 0..n_scales {
            let n_pix = scale_npix[scale];
            let one_per_pixels = 1.0 / n_pix;
            for ch in 0..3 {
                let s_slot = (scale * 3 + ch) * 3;
                let a_slot = s_slot + 1;
                let d_slot = s_slot + 2;

                avg_ssim[scale][ch * 2] = one_per_pixels * acc_sum[s_slot];
                avg_ssim[scale][ch * 2 + 1] = (one_per_pixels * acc_p4[s_slot]).sqrt().sqrt();

                avg_edgediff[scale][ch * 4] = one_per_pixels * acc_sum[a_slot];
                avg_edgediff[scale][ch * 4 + 1] = (one_per_pixels * acc_p4[a_slot]).sqrt().sqrt();
                avg_edgediff[scale][ch * 4 + 2] = one_per_pixels * acc_sum[d_slot];
                avg_edgediff[scale][ch * 4 + 3] = (one_per_pixels * acc_p4[d_slot]).sqrt().sqrt();
            }
        }
        score_from_stats(&avg_ssim, &avg_edgediff, n_scales)
    }

    // ───────────────────────── helpers ─────────────────────────

    fn check_dims(&self, srgb: &[u8]) -> Result<()> {
        let expected = self.n * 3;
        if srgb.len() != expected {
            Err(Error::DimensionMismatch {
                expected,
                got: srgb.len(),
            })
        } else {
            Ok(())
        }
    }

    /// Cube grid for 1D pointwise kernels (one thread per pixel).
    ///
    /// **Why 2D-when-large**: the WebGPU portable-floor limit on
    /// `max_compute_workgroups_per_dimension` is 65535
    /// (`Limits::downlevel_defaults`), so a 1D dispatch saturates at
    /// 65535 × TPB = ~16.7M threads. The scale-0 kernel grid at 4096²
    /// has exactly 16,777,216 pixels → 65,536 cubes, which is +1 over
    /// the wgpu cap and the runtime rejects the dispatch.
    ///
    /// The kernels read `ABSOLUTE_POS` as a flat linear index. Cubecl
    /// computes that builtin as
    /// `(absolute_pos_y * cube_count_x * cube_dim_x) + absolute_pos_x`
    /// (see `cubecl-cpp::shared::kernel::compile_cube_builtin_bindings_decl`
    /// and the wgsl `body.rs` `let id = ...` template), so splitting
    /// the 1D `cubes` count into a 2D `(x, y)` grid with the same total
    /// product preserves the linear index that each thread sees — the
    /// kernel needs no changes.
    ///
    /// Tile shape: keep `x` close to 32768 (well under both wgpu's
    /// 65535 and CUDA's 2^31 limits) and stretch `y` as needed. The
    /// kernels bounds-check `ABSOLUTE_POS >= len()` so trailing
    /// no-op threads from rounding are safe.
    fn cube_count_1d(n: usize) -> CubeCount {
        const TPB: u32 = 256;
        const MAX_CUBES_PER_DIM: u32 = 32768;
        let cubes = (n as u32).div_ceil(TPB).max(1);
        if cubes <= MAX_CUBES_PER_DIM {
            CubeCount::Static(cubes, 1, 1)
        } else {
            // 2D split. Choose x = MAX_CUBES_PER_DIM and y = ceil(cubes / x).
            // Total = x * y >= cubes; the kernel's `if idx >= len { terminate; }`
            // guard makes the overshoot a no-op.
            let x = MAX_CUBES_PER_DIM;
            let y = cubes.div_ceil(x);
            CubeCount::Static(x, y, 1)
        }
    }
    fn cube_dim_1d() -> CubeDim {
        CubeDim::new_1d(256)
    }
    fn blur_cube_count(width: u32) -> CubeCount {
        let cubes = width.div_ceil(blur::BLOCK_WIDTH);
        CubeCount::Static(cubes.max(1), 1, 1)
    }
    fn blur_cube_dim() -> CubeDim {
        CubeDim::new_1d(blur::BLOCK_WIDTH)
    }
    /// T_x.B (2026-05-17): 2D launch geometry for the tiled transpose.
    fn transpose_cube_count(width: u32, height: u32) -> CubeCount {
        let x_cubes = width.div_ceil(transpose::TILE_DIM).max(1);
        let y_cubes = height.div_ceil(transpose::TILE_DIM).max(1);
        CubeCount::Static(x_cubes, y_cubes, 1)
    }
    fn transpose_cube_dim() -> CubeDim {
        CubeDim::new_2d(transpose::TPB_X, transpose::TPB_Y)
    }

    fn upload_and_srgb_to_linear(&mut self, is_a: bool, srgb: &[u8]) {
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
        // T4.M (2026-05-16): pinned-host fast path — direct DMA (12-25
        // GB/s on PCIe 4.0 vs 5-6 GB/s from pageable).
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
                ArrayArg::from_raw_parts(src.clone(), self.n),
                ArrayArg::from_raw_parts(lin[0].clone(), self.n),
                ArrayArg::from_raw_parts(lin[1].clone(), self.n),
                ArrayArg::from_raw_parts(lin[2].clone(), self.n),
            );
        }
    }

    /// Install a caller-supplied packed-u32 device handle as the
    /// ref/dist input. The handle layout MUST match what
    /// [`Self::upload_and_srgb_to_linear`] produces: `width × height`
    /// `u32`s, each `R | G<<8 | B<<16` (alpha unused). After this
    /// returns the sRGB→linear kernel has been dispatched and the
    /// pipeline can run from scale-0 linear planes onwards.
    fn install_packed_handle(&mut self, is_a: bool, handle: &cubecl::server::Handle) {
        if is_a {
            self.src_u8_a = handle.clone();
        } else {
            self.src_u8_b = handle.clone();
        }
        self.srgb_to_linear_from_packed(is_a);
    }

    fn build_linear_pyramid(&self, is_a: bool) {
        let last = self.scales.len().saturating_sub(1);
        self.build_linear_pyramid_until(is_a, last);
    }

    /// Build linear pyramid up to (and including) `last_scale`. Saves
    /// downscale launches for scales beyond `last_scale` when the
    /// skip-map elides them. `last_scale` is inclusive — must be in
    /// `0..self.scales.len()`.
    ///
    /// Uses the 3-channel-fused [`downscale::downscale_2x_3ch_kernel`]:
    /// one launch per scale transition (down from three pre-fix). Output
    /// is bit-identical to the per-plane variant — same clamp math, same
    /// `* 0.25` box-average. See `SSIM2_FIX_ASSESSMENT.md` for the
    /// per-line audit of the kernel's portability.
    fn build_linear_pyramid_until(&self, is_a: bool, last_scale: usize) {
        let stop = last_scale.min(self.scales.len().saturating_sub(1));
        for s in 1..=stop {
            let (prev_w, prev_h) = (self.scales[s - 1].width, self.scales[s - 1].height);
            let (curr_w, curr_h) = (self.scales[s].width, self.scales[s].height);
            let (prev_lin, curr_lin) = if is_a {
                (&self.scales[s - 1].ref_lin, &self.scales[s].ref_lin)
            } else {
                (&self.scales[s - 1].dis_lin, &self.scales[s].dis_lin)
            };
            let n_curr = self.scales[s].n;
            let n_prev = self.scales[s - 1].n;
            unsafe {
                downscale::downscale_2x_3ch_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n_curr),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(prev_lin[0].clone(), n_prev),
                    ArrayArg::from_raw_parts(prev_lin[1].clone(), n_prev),
                    ArrayArg::from_raw_parts(prev_lin[2].clone(), n_prev),
                    ArrayArg::from_raw_parts(curr_lin[0].clone(), n_curr),
                    ArrayArg::from_raw_parts(curr_lin[1].clone(), n_curr),
                    ArrayArg::from_raw_parts(curr_lin[2].clone(), n_curr),
                    prev_w,
                    prev_h,
                    curr_w,
                    curr_h,
                );
            }
        }
    }

    fn run_xyb(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        let (lin, xyb_buf) = if is_a {
            (&s.ref_lin, &s.ref_xyb)
        } else {
            (&s.dis_lin, &s.dis_xyb)
        };
        unsafe {
            xyb::linear_to_xyb_planar_kernel::launch_unchecked::<R>(
                &self.client,
                Self::cube_count_1d(s.n),
                Self::cube_dim_1d(),
                ArrayArg::from_raw_parts(lin[0].clone(), s.n),
                ArrayArg::from_raw_parts(lin[1].clone(), s.n),
                ArrayArg::from_raw_parts(lin[2].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[0].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[1].clone(), s.n),
                ArrayArg::from_raw_parts(xyb_buf[2].clone(), s.n),
            );
        }
    }

    /// Pointwise product `a · b → out` for one scale × all 3 channels.
    fn pointwise_mul(
        &self,
        scale: usize,
        a: &[cubecl::server::Handle; 3],
        b: &[cubecl::server::Handle; 3],
        out: &[cubecl::server::Handle; 3],
    ) {
        let n = self.scales[scale].n;
        for ch in 0..3 {
            unsafe {
                pointwise_mul_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(a[ch].clone(), n),
                    ArrayArg::from_raw_parts(b[ch].clone(), n),
                    ArrayArg::from_raw_parts(out[ch].clone(), n),
                );
            }
        }
    }

    /// Compute self-products: `sigma11 = ref²` (if is_a) or
    /// `sigma22 = dis²` (else).
    fn run_self_products(&self, scale: usize, is_a: bool) {
        let s = &self.scales[scale];
        if is_a {
            self.pointwise_mul(scale, &s.ref_xyb, &s.ref_xyb, &s.sigma11_in);
        } else {
            self.pointwise_mul(scale, &s.dis_xyb, &s.dis_xyb, &s.sigma22_in);
        }
    }

    /// One-plane two-pass blur: `src → pass-0 → tiled-transpose → t_buf →
    /// pass-1 → full`. Caller supplies all 4 same-channel buffers.
    ///
    /// Without the `fir` Cargo feature this calls the IIR path
    /// directly (no per-instance blur knob). With the feature enabled
    /// it dispatches on `self.blur`:
    /// - `Ssim2Blur::Iir`: the column-walking Charalampidis recursive
    ///   Gaussian (default — bit-identical to the published CPU
    ///   `ssimulacra2` reference).
    /// - `Ssim2Blur::Fir`: the separable 5-tap truncated Gaussian D=5
    ///   per Kanetaka et al. IWAIT 2026. Distinct metric — see the
    ///   file-level doc on `kernels::blur` and the `Ssim2Blur::Fir`
    ///   variant doc on `crate::Ssim2Blur`.
    fn blur_plane_two_pass(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        t_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) {
        #[cfg(feature = "fir")]
        {
            match self.blur {
                Ssim2Blur::Iir => {
                    self.blur_plane_two_pass_iir(width, height, n, src, v_buf, t_buf, full)
                }
                Ssim2Blur::Fir => {
                    self.blur_plane_two_pass_fir(width, height, n, src, v_buf, t_buf, full)
                }
            }
        }
        #[cfg(not(feature = "fir"))]
        {
            self.blur_plane_two_pass_iir(width, height, n, src, v_buf, t_buf, full)
        }
    }

    /// Untransposed-output variant of [`Self::blur_plane_two_pass`].
    ///
    /// **Not yet wired into the production paths** — preserved as a
    /// building block for the follow-on migration where Ssim2Batch's
    /// `error_maps_broadcast_batched_kernel` is ported to untransposed
    /// orientation, at which point `set_reference` + all error_maps
    /// callers can flip together to the v+h fast path.
    #[allow(dead_code)]
    ///
    /// IIR path: `v + h` (2 launches, output in untransposed orientation
    /// via the new [`blur::blur_h_pass_kernel`]).
    ///
    /// FIR path: falls back to [`Self::blur_plane_two_pass_fir`] (the
    /// pre-fix v+t+v shape). The FIR output stays in transposed
    /// orientation. Callers that consume FIR-blurred values must
    /// therefore still use the transposed-orientation consumers (and
    /// the `_t` raw xyb pair). The perf-fix work to make FIR
    /// untransposed is queued as a follow-on; FIR is opt-in (gated by
    /// the `fir` Cargo feature) and rarely adopted in production
    /// sweeps, so the optimisation lives on the IIR path first.
    ///
    /// Whole-image and whole-image-cached-ref paths call this when the
    /// blur is IIR. Strip-mode mode-E keeps using the transposed-output
    /// [`Self::blur_plane_two_pass`] because its cached layout is
    /// committed to transposed orientation.
    ///
    /// Returns `true` if the output landed in untransposed orientation
    /// (IIR), `false` if it stayed in transposed orientation (FIR
    /// fallback). The caller uses this to decide whether to feed
    /// untransposed or transposed inputs to the error_maps kernel.
    #[cfg_attr(not(feature = "fir"), allow(clippy::unnecessary_wraps))]
    fn blur_plane_two_pass_untransposed(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        t_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) -> bool {
        #[cfg(feature = "fir")]
        {
            match self.blur {
                Ssim2Blur::Iir => {
                    let _ = t_buf;
                    self.blur_plane_two_pass_iir_untransposed(width, height, n, src, v_buf, full);
                    true
                }
                Ssim2Blur::Fir => {
                    self.blur_plane_two_pass_fir(width, height, n, src, v_buf, t_buf, full);
                    false
                }
            }
        }
        #[cfg(not(feature = "fir"))]
        {
            let _ = t_buf;
            self.blur_plane_two_pass_iir_untransposed(width, height, n, src, v_buf, full);
            true
        }
    }

    /// Default IIR path (Charalampidis recursive Gaussian).
    ///
    /// Output orientation: TRANSPOSED (`height × width` row-major).
    /// This is the original "v-pass + transpose + v-pass" three-step;
    /// it's still used by strip-mode mode-E (`set_reference_strip_mode`
    /// + `compute_with_reference_strip_with_mode`) because the cached
    /// `ref_xyb_t_full`, `mu1_full_full`, `sigma11_full_full` slots are
    /// in transposed orientation and the `error_maps_strip_from_full_ref_kernel`
    /// indexes them under the `full_inner_stride = full_h` (transposed
    /// inner-dim) convention.
    ///
    /// For the whole-image and whole-image-cached-ref paths use
    /// [`Self::blur_plane_two_pass_iir_untransposed`] instead — it's
    /// `v + h` (2 launches), no transpose, output in untransposed
    /// orientation matching the raw XYB. That path is the perf
    /// optimisation; this one stays for strip-mode-E migration backlog.
    fn blur_plane_two_pass_iir(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        t_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) {
        unsafe {
            // 1. v-pass on src (walks columns of width × height) → v_buf.
            blur::blur_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(width),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(src.clone(), n),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                width,
                height,
            );
            // 2. tiled transpose v_buf → t_buf (now height × width).
            transpose::transpose_kernel::launch_unchecked::<R>(
                &self.client,
                Self::transpose_cube_count(width, height),
                Self::transpose_cube_dim(),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                width,
                height,
            );
            // 3. v-pass on t_buf → full (transposed orientation).
            blur::blur_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(height),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                ArrayArg::from_raw_parts(full.clone(), n),
                height,
                width,
            );
        }
    }

    /// Untransposed-output IIR path (`v + h` via
    /// [`blur::blur_h_pass_kernel`]) — the perf-optimised version added
    /// in the 2026-05-27 work.
    ///
    /// Output orientation: UNTRANSPOSED (`width × height` row-major,
    /// same as `src`). Mathematically equivalent (bit-identical, per
    /// `examples/blur_h_pass_parity`) to `v + transpose + v` because
    /// the IIR is separable. Saves one transpose launch per blur AND
    /// eliminates the `t_buf` write/read DRAM roundtrip.
    ///
    /// Use this in the whole-image / whole-image-cached-ref paths.
    /// Strip-mode mode-E still uses [`Self::blur_plane_two_pass_iir`]
    /// because its caching layout is committed to transposed
    /// orientation.
    ///
    /// **Not yet wired into production paths** — see
    /// `blur_plane_two_pass_untransposed`'s docstring for the gate on
    /// the follow-on Ssim2Batch migration. Kept as a building block
    /// + parity-tested via `examples/blur_h_pass_parity`.
    #[allow(dead_code)]
    fn blur_plane_two_pass_iir_untransposed(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) {
        unsafe {
            // 1. v-pass on src (walks columns of width × height) → v_buf.
            blur::blur_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(width),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(src.clone(), n),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                width,
                height,
            );
            // 2. h-pass on v_buf (walks rows of width × height) → full.
            //    Output is in untransposed orientation.
            blur::blur_h_pass_kernel::launch_unchecked::<R>(
                &self.client,
                Self::blur_cube_count(height),
                Self::blur_cube_dim(),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                ArrayArg::from_raw_parts(full.clone(), n),
                width,
                height,
            );
        }
    }

    /// Opt-in FIR D=5 path (Kanetaka et al. IWAIT 2026) — **gated
    /// behind the `fir` Cargo feature**.
    ///
    /// Uses the horizontal 5-tap FIR for both passes: the second pass
    /// runs on the transposed intermediate, so the kernel's horizontal
    /// walk corresponds to a vertical walk in the original frame. The
    /// 2D blur result lands in transposed orientation in `full`, exactly
    /// matching the IIR path's output convention.
    #[cfg(feature = "fir")]
    fn blur_plane_two_pass_fir(
        &self,
        width: u32,
        height: u32,
        n: usize,
        src: &cubecl::server::Handle,
        v_buf: &cubecl::server::Handle,
        t_buf: &cubecl::server::Handle,
        full: &cubecl::server::Handle,
    ) {
        unsafe {
            // 1. H-FIR on src (one thread per output pixel, 5 reads
            //    along the row, zero-padded at borders) → v_buf.
            blur::blur_h_fir5_kernel::launch_unchecked::<R>(
                &self.client,
                Self::fir_cube_count(n),
                Self::fir_cube_dim(),
                ArrayArg::from_raw_parts(src.clone(), n),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                width,
                height,
            );
            // 2. Tiled transpose v_buf → t_buf (now height × width).
            transpose::transpose_kernel::launch_unchecked::<R>(
                &self.client,
                Self::transpose_cube_count(width, height),
                Self::transpose_cube_dim(),
                ArrayArg::from_raw_parts(v_buf.clone(), n),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                width,
                height,
            );
            // 3. H-FIR on t_buf → full. Note: the transposed buffer's
            //    "width" is the original height. This second H-FIR is
            //    a vertical FIR in original coordinates.
            blur::blur_h_fir5_kernel::launch_unchecked::<R>(
                &self.client,
                Self::fir_cube_count(n),
                Self::fir_cube_dim(),
                ArrayArg::from_raw_parts(t_buf.clone(), n),
                ArrayArg::from_raw_parts(full.clone(), n),
                height,
                width,
            );
        }
    }

    #[cfg(feature = "fir")]
    fn fir_cube_count(n: usize) -> CubeCount {
        // Same 2D-when-large split as `cube_count_1d`. See that helper
        // for the rationale (wgpu's 65535-per-dim cap blocks 4096²).
        const MAX_CUBES_PER_DIM: u32 = 32768;
        let cubes = (n as u32).div_ceil(blur::FIR_BLOCK_WIDTH).max(1);
        if cubes <= MAX_CUBES_PER_DIM {
            CubeCount::Static(cubes, 1, 1)
        } else {
            let x = MAX_CUBES_PER_DIM;
            let y = cubes.div_ceil(x);
            CubeCount::Static(x, y, 1)
        }
    }
    #[cfg(feature = "fir")]
    fn fir_cube_dim() -> CubeDim {
        CubeDim::new_1d(blur::FIR_BLOCK_WIDTH)
    }

    /// Pointwise transpose for the `ref_xyb` / `dis_xyb` planes (raw,
    /// unblurred — used as `source` / `distorted` inputs to compute_error_maps).
    fn run_transpose_raw_xyb_pair(&self, scale: usize, do_ref: bool, do_dis: bool) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        if do_ref {
            for ch in 0..3 {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::transpose_cube_count(w, h),
                        Self::transpose_cube_dim(),
                        ArrayArg::from_raw_parts(s.ref_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.ref_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }
        if do_dis {
            for ch in 0..3 {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::transpose_cube_count(w, h),
                        Self::transpose_cube_dim(),
                        ArrayArg::from_raw_parts(s.dis_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.dis_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }
    }

    /// Reference-only blur pair (sigma11 + mu1). Used by `set_reference`
    /// — populates `sigma11_full` and `mu1_full`.
    ///
    /// Uses the legacy v+t+v path (transposed-output) for **all** modes.
    /// Why: the `Ssim2Batch` path (`pipeline_batch.rs`) consumes
    /// `cached_mu1_full(s)` / `cached_sigma11_full(s)` via
    /// `error_maps_broadcast_batched_kernel` which is committed to the
    /// transposed-orientation contract. Migrating Ssim2Batch's blur +
    /// error_maps kernels to untransposed is the obvious follow-on; for
    /// now keep set_reference on the legacy path so the batched-path
    /// parity stays unbroken.
    ///
    /// Returns `false` (transposed) always to signal the orientation
    /// to the caller.
    fn run_blur_pair(&self, scale: usize, is_a: bool) -> bool {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        if is_a {
            for ch in 0..3 {
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &s.sigma11_in[ch],
                    &s.v_scratch[ch],
                    &s.t_scratch[ch],
                    &s.sigma11_full[ch],
                );
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &s.ref_xyb[ch],
                    &s.v_scratch[ch],
                    &s.t_scratch[ch],
                    &s.mu1_full[ch],
                );
            }
        } else {
            for ch in 0..3 {
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &s.sigma22_in[ch],
                    &s.v_scratch[ch],
                    &s.t_scratch[ch],
                    &s.sigma22_full[ch],
                );
                self.blur_plane_two_pass(
                    w,
                    h,
                    n,
                    &s.dis_xyb[ch],
                    &s.v_scratch[ch],
                    &s.t_scratch[ch],
                    &s.mu2_full[ch],
                );
            }
        }
        false
    }

    // (Unmasked variants `run_blur_dis_only`, `run_blur_full`,
    // `run_error_maps`, `run_reductions`, `run_cross_product` were
    // removed when the skip-map pass landed — the masked variants
    // below subsume them. `Ssim2Mode::Full` selects the no-skip
    // behaviour bit-for-bit if needed by tests.)

    // ───────────────────── skip-map masked variants ─────────────────────
    //
    // These wrap the per-channel launch loops with `skip_error_map` and
    // `skip_reduction` predicates so masked channels never pay for the
    // upstream blur / transpose / pointwise-mul that feeds them. See
    // `crate::skipmap` for the per-cell skip table.

    fn run_xyb_masked(&self, scale: usize, is_a: bool, mode: Ssim2Mode) {
        // XYB is a 3-in 3-out fused kernel. Skip only if EVERY channel
        // at this scale is `skip_error_map` — no downstream consumer
        // anywhere. (`skip_scale` already gates the whole scale at
        // the caller, so this triggers when `skip_scale` is false but
        // every channel happens to be inactive, which currently never
        // happens — kept for completeness.)
        if (0..3).all(|c| skip_error_map(mode, scale, c)) {
            return;
        }
        self.run_xyb(scale, is_a);
    }

    /// Pointwise product `a · b → out` for one scale × selected channels.
    fn pointwise_mul_masked(
        &self,
        scale: usize,
        a: &[cubecl::server::Handle; 3],
        b: &[cubecl::server::Handle; 3],
        out: &[cubecl::server::Handle; 3],
        mode: Ssim2Mode,
    ) {
        let n = self.scales[scale].n;
        for ch in 0..3 {
            if skip_error_map(mode, scale, ch) {
                continue;
            }
            unsafe {
                pointwise_mul_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(a[ch].clone(), n),
                    ArrayArg::from_raw_parts(b[ch].clone(), n),
                    ArrayArg::from_raw_parts(out[ch].clone(), n),
                );
            }
        }
    }

    fn run_self_products_masked(&self, scale: usize, is_a: bool, mode: Ssim2Mode) {
        let s = &self.scales[scale];
        if is_a {
            self.pointwise_mul_masked(scale, &s.ref_xyb, &s.ref_xyb, &s.sigma11_in, mode);
        } else {
            self.pointwise_mul_masked(scale, &s.dis_xyb, &s.dis_xyb, &s.sigma22_in, mode);
        }
    }

    fn run_cross_product_masked(&self, scale: usize, mode: Ssim2Mode) {
        let s = &self.scales[scale];
        self.pointwise_mul_masked(scale, &s.ref_xyb, &s.dis_xyb, &s.sigma12_in, mode);
    }

    fn run_transpose_raw_xyb_pair_masked(
        &self,
        scale: usize,
        do_ref: bool,
        do_dis: bool,
        mode: Ssim2Mode,
    ) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        for ch in 0..3 {
            if skip_error_map(mode, scale, ch) {
                continue;
            }
            if do_ref {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::transpose_cube_count(w, h),
                        Self::transpose_cube_dim(),
                        ArrayArg::from_raw_parts(s.ref_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.ref_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
            if do_dis {
                unsafe {
                    transpose::transpose_kernel::launch_unchecked::<R>(
                        &self.client,
                        Self::transpose_cube_count(w, h),
                        Self::transpose_cube_dim(),
                        ArrayArg::from_raw_parts(s.dis_xyb[ch].clone(), n),
                        ArrayArg::from_raw_parts(s.dis_xyb_t[ch].clone(), n),
                        w,
                        h,
                    );
                }
            }
        }
    }

    /// Run all 5 blurs (sigma11/22/12 + ref + dis = mu1/mu2) for one
    /// scale × selected channels.
    ///
    /// Currently produces TRANSPOSED outputs (legacy v+t+v path) — the
    /// shared `_full` plane slots between this path (non-cached compute)
    /// and `set_reference` (cached compute) must stay in matching
    /// orientation, and `set_reference` is committed to transposed for
    /// Ssim2Batch compatibility. The h-pass IIR kernel and the
    /// `blur_plane_two_pass_iir_untransposed` helper exist but aren't
    /// wired into this path yet — the follow-on migration is to (a)
    /// port Ssim2Batch to untransposed, then (b) flip this path AND
    /// set_reference to untransposed simultaneously.
    ///
    /// Returns `false` (transposed) for now to signal callers.
    fn run_blur_full_masked(&self, scale: usize, mode: Ssim2Mode) -> bool {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        for ch in 0..3 {
            if skip_error_map(mode, scale, ch) {
                continue;
            }
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.sigma11_in[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.sigma11_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.ref_xyb[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.mu1_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.sigma22_in[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.sigma22_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.dis_xyb[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.mu2_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.sigma12_in[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.sigma12_full[ch],
            );
        }
        false
    }

    /// Run the 3 dist-side blurs only via the LEGACY transposed-output
    /// path (`v + transpose + v`). Used by `compute_with_reference_with_mode`
    /// because the ref-side cache is committed to transposed orientation.
    /// Migrating the cache + Ssim2Batch to untransposed is the obvious
    /// follow-on — at that point this function becomes dead code and
    /// the optimised `run_blur_dis_only_masked` becomes the only path.
    fn run_blur_dis_only_masked_transposed(&self, scale: usize, mode: Ssim2Mode) {
        let s = &self.scales[scale];
        let n = s.n;
        let w = s.width;
        let h = s.height;
        for ch in 0..3 {
            if skip_error_map(mode, scale, ch) {
                continue;
            }
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.sigma22_in[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.sigma22_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.dis_xyb[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.mu2_full[ch],
            );
            self.blur_plane_two_pass(
                w,
                h,
                n,
                &s.sigma12_in[ch],
                &s.v_scratch[ch],
                &s.t_scratch[ch],
                &s.sigma12_full[ch],
            );
        }
    }

    // `run_blur_dis_only_masked` (the untransposed-output variant) was
    // landed-and-removed in the 2026-05-27 wiring attempt — when the
    // cached-ref + Ssim2Batch paths force transposed-orientation cache
    // (set_reference contract), the untransposed dis-only variant has
    // no callers. Keeping it around as dead code is misleading. The
    // h-pass kernel + blur_plane_two_pass_iir_untransposed are
    // available for the eventual full migration; the wiring landing
    // is gated on Ssim2Batch's broadcast_batched_kernel being ported
    // to untransposed orientation. See SSIM2_FIX_ASSESSMENT.md.

    /// Orientation-aware error_maps dispatch. `untransposed_raw` selects
    /// whether to feed the raw `ref_xyb` / `dis_xyb` directly (true,
    /// IIR fast path) or the pre-transposed `ref_xyb_t` / `dis_xyb_t`
    /// (false, FIR fallback). The blurred sigma/mu inputs are taken from
    /// `*_full` slots regardless — those are written by the same blur
    /// kernel that produced the orientation in the first place, so
    /// orientations align.
    fn run_error_maps_masked_oriented(
        &self,
        scale: usize,
        mode: Ssim2Mode,
        untransposed_raw: bool,
    ) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            if skip_error_map(mode, scale, ch) {
                continue;
            }
            let (src_h, dis_h) = if untransposed_raw {
                (s.ref_xyb[ch].clone(), s.dis_xyb[ch].clone())
            } else {
                (s.ref_xyb_t[ch].clone(), s.dis_xyb_t[ch].clone())
            };
            unsafe {
                error_maps::error_maps_kernel::launch_unchecked::<R>(
                    &self.client,
                    Self::cube_count_1d(s.n),
                    Self::cube_dim_1d(),
                    ArrayArg::from_raw_parts(src_h, s.n),
                    ArrayArg::from_raw_parts(dis_h, s.n),
                    ArrayArg::from_raw_parts(s.mu1_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.mu2_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma11_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma22_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.sigma12_full[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.ssim[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.artifact[ch].clone(), s.n),
                    ArrayArg::from_raw_parts(s.detail[ch].clone(), s.n),
                );
            }
        }
    }

    /// Legacy non-oriented entry point — defaults to the IIR untransposed
    /// orientation. Used by `pipeline_batch.rs` and other callers that
    /// don't track the per-call blur orientation. Such callers are
    /// expected to run in IIR mode (the default).
    #[allow(dead_code)]
    fn run_error_maps_masked(&self, scale: usize, mode: Ssim2Mode) {
        self.run_error_maps_masked_oriented(scale, mode, true);
    }

    fn run_reductions_masked(&self, scale: usize, mode: Ssim2Mode) {
        let s = &self.scales[scale];
        for ch in 0..3 {
            let plane_handles = [&s.ssim[ch], &s.artifact[ch], &s.detail[ch]];
            for map_type in 0..3 {
                if skip_reduction(mode, scale, ch, map_type) {
                    continue;
                }
                // Slot encoding: (scale * 3 + ch) * 3 + map_type ∈ [0, 54).
                let slot = ((scale * 3 + ch) * 3 + map_type) as u32;
                reduction::launch_sum_p4::<R>(
                    &self.client,
                    plane_handles[map_type].clone(),
                    s.n,
                    self.partials.clone(),
                    PARTIALS_LEN,
                    slot,
                );
            }
        }
    }

    /// Stage-2 finalizer: fold per-thread partials into a small
    /// `(slot, sum, p4)` buffer the host reads back. One launch per
    /// `compute()` instead of 16× the readback the partials buffer would
    /// require if the host folded directly.
    fn run_finalizer(&self) {
        reduction::launch_finalize::<R>(
            &self.client,
            self.partials.clone(),
            PARTIALS_LEN,
            self.sums.clone(),
            SUMS_LEN,
            NUM_SLOTS as u32,
        );
    }

    /// Per-scale processing for `compute()`: XYB, products, blurs,
    /// error maps, reduction. Called for every non-skipped pyramid
    /// scale. The `mode` selects which `(channel, map_type)` cells
    /// can be skipped — see `crate::skipmap` for the per-cell table.
    fn process_scale(&self, scale: usize, mode: Ssim2Mode) {
        // 1. linear → XYB for both sides (XYB is fused per-channel,
        //    cannot be skipped at sub-channel granularity — but if
        //    NO channel at this scale is active, the entire scale was
        //    already skipped by `compute_with_mode`'s outer guard).
        self.run_xyb_masked(scale, true, mode);
        self.run_xyb_masked(scale, false, mode);
        // 2. Pointwise products: sigma11 = ref²; sigma22 = dis²; sigma12 = ref·dis.
        self.run_self_products_masked(scale, true, mode);
        self.run_self_products_masked(scale, false, mode);
        self.run_cross_product_masked(scale, mode);
        // 3. Blur all 5. Stays on the legacy v+t+v path while the
        //    cached-ref + Ssim2Batch contract for transposed-orientation
        //    cache holds — see `run_blur_full_masked` docstring.
        self.run_blur_full_masked(scale, mode);
        // 4. Transpose raw XYB so error_maps reads them in the same
        //    (transposed) orientation as the blurred buffers.
        self.run_transpose_raw_xyb_pair_masked(scale, true, true, mode);
        // 5. Per-pixel error maps.
        self.run_error_maps_masked_oriented(scale, mode, false);
        // 6. Reduce to (Σ, Σ⁴) per (channel × map type).
        self.run_reductions_masked(scale, mode);
    }

    /// Read the sums buffer back to host and compute the final SSIMULACRA2
    /// score. Mirrors `ssimulacra2::Msssim::score` exactly (same WEIGHT
    /// table, same sigmoid).
    fn read_and_aggregate(&mut self) -> f64 {
        let bytes = self
            .client
            .read_one(self.sums.clone())
            .expect("read sums buffer");
        let raw = f32::from_bytes(&bytes);
        debug_assert_eq!(raw.len(), SUMS_LEN);

        // T_y.A (2026-05-17): the per-call zero-fill moved to the
        // START of `compute_with_mode` / `compute_with_reference_with_mode`
        // (subsumes both the prior post-call zero used by the
        // fast-reduction path AND the need-to-zero for skip-map
        // dispatch in portable mode).

        // Layout post-finalizer: `raw[slot * 2]` = Σ, `raw[slot * 2 + 1]` = Σ⁴.
        // Total length = NUM_SLOTS * 2 = 108 floats. The 4096 per-thread
        // partials per slot were already folded on-device by the
        // finalizer kernel — much less device→host bandwidth than reading
        // the full partials buffer.
        let mut avg_ssim = vec![[0.0_f64; 6]; NUM_SCALES]; // [scale][c*2 + n]
        let mut avg_edgediff = vec![[0.0_f64; 12]; NUM_SCALES]; // [scale][c*4 + n]

        let fold_slot =
            |slot: usize| -> (f64, f64) { (raw[slot * 2] as f64, raw[slot * 2 + 1] as f64) };

        for scale in 0..self.scales.len() {
            let n_pix = self.scales[scale].n as f64;
            let one_per_pixels = 1.0 / n_pix;
            for ch in 0..3 {
                let s_slot = (scale * 3 + ch) * 3; // ssim
                let a_slot = s_slot + 1; // artifact
                let d_slot = s_slot + 2; // detail
                let (s_sum, s_p4) = fold_slot(s_slot);
                let (a_sum, a_p4) = fold_slot(a_slot);
                let (d_sum, d_p4) = fold_slot(d_slot);

                avg_ssim[scale][ch * 2] = one_per_pixels * s_sum;
                avg_ssim[scale][ch * 2 + 1] = (one_per_pixels * s_p4).sqrt().sqrt();

                avg_edgediff[scale][ch * 4] = one_per_pixels * a_sum;
                avg_edgediff[scale][ch * 4 + 1] = (one_per_pixels * a_p4).sqrt().sqrt();
                avg_edgediff[scale][ch * 4 + 2] = one_per_pixels * d_sum;
                avg_edgediff[scale][ch * 4 + 3] = (one_per_pixels * d_p4).sqrt().sqrt();
            }
        }

        score_from_stats(&avg_ssim, &avg_edgediff, self.scales.len())
    }
}

/// Pointwise multiply kernel `out = a · b` (single plane). Exposed
/// to the batched pipeline because the unbatched kernel works fine on
/// flat batched arrays — `Ssim2Batch::process_scale_batched` uses it
/// for `sigma22 = dis · dis`.
#[cube(launch_unchecked)]
pub fn pointwise_mul_kernel(a: &Array<f32>, b: &Array<f32>, out: &mut Array<f32>) {
    let idx = ABSOLUTE_POS;
    if idx >= out.len() {
        terminate!();
    }
    out[idx] = a[idx] * b[idx];
}

/// Final score: weighted dot-product of 108 stats then sigmoid remap.
/// Verbatim from `ssimulacra2::Msssim::score` — for fewer-than-6 actual
/// scales, missing entries are treated as 0 (matches the CPU which
/// `break`s the per-scale loop early but still iterates all 6 weight
/// rows; slots beyond `n_scales` stay at their `0.0` default).
pub(crate) fn score_from_stats(
    avg_ssim: &[[f64; 6]],
    avg_edgediff: &[[f64; 12]],
    n_scales: usize,
) -> f64 {
    const WEIGHT: [f64; 108] = [
        0.0,
        0.000_737_660_670_740_658_6,
        0.0,
        0.0,
        0.000_779_348_168_286_730_9,
        0.0,
        0.0,
        0.000_437_115_573_010_737_9,
        0.0,
        1.104_172_642_665_734_6,
        0.000_662_848_341_292_71,
        0.000_152_316_327_837_187_52,
        0.0,
        0.001_640_643_745_659_975_4,
        0.0,
        1.842_245_552_053_929_8,
        11.441_172_603_757_666,
        0.0,
        0.000_798_910_943_601_516_3,
        0.000_176_816_438_078_653,
        0.0,
        1.878_759_497_954_638_7,
        10.949_069_906_051_42,
        0.0,
        0.000_728_934_699_150_807_2,
        0.967_793_708_062_683_3,
        0.0,
        0.000_140_034_242_854_358_84,
        0.998_176_697_785_496_7,
        0.000_319_497_559_344_350_53,
        0.000_455_099_211_379_206_3,
        0.0,
        0.0,
        0.001_364_876_616_324_339_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        7.466_890_328_078_848,
        0.0,
        17.445_833_984_131_262,
        0.000_623_560_163_404_146_6,
        0.0,
        0.0,
        6.683_678_146_179_332,
        0.000_377_244_079_796_112_96,
        1.027_889_937_768_264,
        225.205_153_008_492_74,
        0.0,
        0.0,
        19.213_238_186_143_016,
        0.001_140_152_458_661_836_1,
        0.001_237_755_635_509_985,
        176.393_175_984_506_94,
        0.0,
        0.0,
        24.433_009_998_704_76,
        0.285_208_026_121_177_57,
        0.000_448_543_692_383_340_8,
        0.0,
        0.0,
        0.0,
        34.779_063_444_837_72,
        44.835_625_328_877_896,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_868_055_657_329_169_8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_531_319_187_435_874_7,
        0.0,
        0.000_165_338_141_613_791_12,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000_417_917_180_325_133_6,
        0.001_729_082_823_472_283_3,
        0.0,
        0.002_082_700_584_663_643_7,
        0.0,
        0.0,
        8.826_982_764_996_862,
        23.192_433_439_989_26,
        0.0,
        95.108_049_881_108_6,
        0.986_397_803_440_068_2,
        0.983_438_279_246_535_3,
        0.001_228_640_504_827_849_3,
        171.266_725_589_730_7,
        0.980_785_887_243_537_9,
        0.0,
        0.0,
        0.0,
        0.000_513_006_458_899_067_9,
        0.0,
        0.000_108_540_578_584_115_37,
    ];

    let mut ssim = 0.0_f64;
    let mut i = 0_usize;
    for c in 0..3 {
        for scale_idx in 0..NUM_SCALES {
            for n in 0..2 {
                let avg_s = if scale_idx < n_scales {
                    avg_ssim[scale_idx][c * 2 + n]
                } else {
                    0.0
                };
                let avg_a = if scale_idx < n_scales {
                    avg_edgediff[scale_idx][c * 4 + n]
                } else {
                    0.0
                };
                let avg_d = if scale_idx < n_scales {
                    avg_edgediff[scale_idx][c * 4 + n + 2]
                } else {
                    0.0
                };
                ssim = WEIGHT[i].mul_add(avg_s.abs(), ssim);
                i += 1;
                ssim = WEIGHT[i].mul_add(avg_a.abs(), ssim);
                i += 1;
                ssim = WEIGHT[i].mul_add(avg_d.abs(), ssim);
                i += 1;
            }
        }
    }

    ssim *= 0.956_238_261_683_484_4_f64;
    ssim = (6.248_496_625_763_138e-5 * ssim * ssim).mul_add(
        ssim,
        2.326_765_642_916_932_f64.mul_add(ssim, -0.020_884_521_182_843_837 * ssim * ssim),
    );

    if ssim > 0.0 {
        ssim = ssim.powf(0.627_633_646_783_138_7).mul_add(-10.0, 100.0)
    } else {
        ssim = 100.0;
    }

    ssim
}
