//! Uniform opaque API for `butteraugli-gpu`.
//!
//! See `dssim-gpu/src/opaque.rs` for the full design rationale —
//! every metric crate ships the same shape so Phase 3's umbrella
//! crate (`zenmetrics-api`) can dispatch through a single trait.
//!
//! The shim hides the cubecl `Runtime` generic behind a [`Backend`]
//! enum. `compute_srgb_u8` and `compute_pixels` always return the
//! max-norm score in [`Score::value`]; the libjxl 3-norm aggregation
//! (also produced by the same reduction kernel) is currently dropped
//! by the opaque path. Callers that need both must use the typed
//! API behind `cubecl-types`.

#[cfg(feature = "pixels")]
use crate::Error;
use crate::pipeline::Butteraugli;
use crate::{ButteraugliParams, Result};

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// The backend selector and uniform score type are shared verbatim
/// across all six `*-gpu` metric crates — see [`zenmetrics_gpu_core`].
/// Re-exported here so `crate::Backend` / `crate::opaque::Score` keep
/// resolving.
pub use zenmetrics_gpu_core::{Backend, Score};

trait ButteraugliInner: Send {
    fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<Score>;
    /// Like `compute_srgb_u8` but also returns the libjxl 3-norm
    /// aggregation (`pnorm_3`). Both numbers come from the same
    /// fused reduction kernel — there's no extra GPU work; the opaque
    /// path simply drops `pnorm_3` after the score is produced.
    fn compute_srgb_u8_with_pnorm3(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<(Score, f64)>;
    fn dims(&self) -> (u32, u32);
    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
        params: &ButteraugliParams,
    ) -> Result<Score>;
    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle>;
    /// Cache the reference image's opsin pyramid + blur cascade on a
    /// whole-image instance. The opaque wrapper only routes here for
    /// whole-image instances; strip-mode instances use the wrapper-held
    /// buffer-replay path (task #160) and never call this. On a
    /// single-res strip instance this still runs the Mode-E
    /// (`ref_cache_full`) path; on a multires-strip instance it returns
    /// [`crate::Error::StripModeUnsupported`] (the wrapper never reaches
    /// it for strip instances).
    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8], params: &ButteraugliParams) -> Result<()>;
    /// Score one candidate against the cached reference.
    fn compute_with_reference_srgb_u8(
        &mut self,
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<Score>;
    /// Drop cached reference state.
    fn clear_reference(&mut self);
    /// Whether a reference has been cached.
    fn has_reference(&self) -> bool;
    /// True iff the underlying instance was constructed in strip mode
    /// (`new_strip` / `new_multires_strip` / Auto-resolved-to-Strip).
    /// The opaque wrapper uses this to drive the buffer-replay
    /// cached-reference path for strip instances (task #160).
    fn is_strip_mode(&self) -> bool;
}

impl<R> ButteraugliInner for Butteraugli<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<Score> {
        // Route to the strip-mode entry point on strip-mode instances
        // (constructed via MemoryMode::Strip or Auto-resolved-to-Strip).
        let r = if self.is_strip_mode() {
            Butteraugli::compute_strip_with_options(self, ref_rgb, dis_rgb, params)?
        } else {
            Butteraugli::compute_with_options(self, ref_rgb, dis_rgb, params)?
        };
        Ok(Score {
            value: r.score as f64,
            metric_name: "butter",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn compute_srgb_u8_with_pnorm3(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<(Score, f64)> {
        let r = if self.is_strip_mode() {
            Butteraugli::compute_strip_with_options(self, ref_rgb, dis_rgb, params)?
        } else {
            Butteraugli::compute_with_options(self, ref_rgb, dis_rgb, params)?
        };
        let score = Score {
            value: r.score as f64,
            metric_name: "butter",
            metric_version: env!("CARGO_PKG_VERSION"),
        };
        Ok((score, r.pnorm_3 as f64))
    }

    fn dims(&self) -> (u32, u32) {
        Butteraugli::dimensions(self)
    }

    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
        params: &ButteraugliParams,
    ) -> Result<Score> {
        let r = Butteraugli::compute_handles_with_options(self, ref_handle, dis_handle, params)?;
        Ok(Score {
            value: r.score as f64,
            metric_name: "butter",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        Butteraugli::pack_srgb_into_packed_u32_handle(self, srgb)
    }

    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8], params: &ButteraugliParams) -> Result<()> {
        Butteraugli::set_reference_with_options(self, ref_rgb, params)
    }

    fn compute_with_reference_srgb_u8(
        &mut self,
        dis_rgb: &[u8],
        _params: &ButteraugliParams,
    ) -> Result<Score> {
        let r = Butteraugli::compute_with_reference(self, dis_rgb)?;
        Ok(Score {
            value: r.score as f64,
            metric_name: "butter",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn clear_reference(&mut self) {
        Butteraugli::clear_reference(self)
    }

    fn has_reference(&self) -> bool {
        Butteraugli::has_reference(self)
    }

    fn is_strip_mode(&self) -> bool {
        Butteraugli::is_strip_mode(self)
    }
}

/// Opaque butteraugli scorer.
pub struct ButteraugliOpaque {
    inner: Box<dyn ButteraugliInner + Send>,
    params: ButteraugliParams,
    #[allow(dead_code)]
    backend: Backend,
    /// Held reference sRGB buffer for the strip-mode cached-reference
    /// path (task #160). Strip-mode instances (single-res or multires)
    /// don't carry a cached-ref *state* on device the way whole-image
    /// instances do; instead this wrapper holds an owned copy of the
    /// reference bytes after `set_reference_srgb_u8` and replays the
    /// pair-strip compute on `(held_ref, dist)` in
    /// `compute_with_reference_srgb_u8`. Because that is exactly
    /// the one-shot `compute_srgb_u8(ref, dist)` with the reference
    /// held, the cached-ref score is **identical** to the one-shot
    /// score — no parity / score-shift risk. `None` until
    /// `set_reference_srgb_u8` is called on a strip-mode instance, and
    /// always `None` on whole-image instances (which use the on-device
    /// cached-ref path in `inner`).
    cached_ref_strip: Option<Vec<u8>>,
}

impl ButteraugliOpaque {
    /// Construct an opaque butteraugli scorer. Uses the multi-
    /// resolution pipeline (matches CPU butteraugli's default
    /// non-`single_resolution` mode). Backwards-compatible alias for
    /// `new_with_memory_mode(.., MemoryMode::Auto)`.
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: ButteraugliParams,
    ) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque butteraugli scorer with an explicit
    /// [`MemoryMode`](crate::MemoryMode). butteraugli-gpu is
    /// **strip-preferred** — see
    /// [`Butteraugli::new_with_memory_mode`](crate::pipeline::Butteraugli::new_with_memory_mode).
    ///
    /// ALL modes engage the multi-resolution path (full-res + half-res
    /// supersample), matching CPU butteraugli's default. `Full`/`Auto→
    /// Full` use `new_multires`; `Strip`/`Auto→Strip` use
    /// `new_multires_strip` so a Strip score is score-identical to the
    /// Full score (within the f64 reduction-noise band) on all content,
    /// including aggressive high-frequency input. Earlier revisions
    /// dropped Strip to single-resolution, which silently omitted the
    /// half-res band and shifted the score several percent vs Full on
    /// HF content (task #158).
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: ButteraugliParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        // Resolve the mode once on the host (resolution doesn't depend
        // on the runtime) so the three backends agree on what they
        // allocate.
        let cap = crate::memory_mode::vram_cap_bytes();
        let resolved = match mode {
            crate::MemoryMode::Full => crate::ResolvedMode::Full,
            crate::MemoryMode::Strip { h_body } => crate::ResolvedMode::Strip {
                h_body: h_body
                    .unwrap_or_else(|| crate::memory_mode::auto_strip_body_for(width, height, cap)),
            },
            crate::MemoryMode::Tile { .. } => {
                return Err(crate::Error::ModeUnsupported("Tile"));
            }
            crate::MemoryMode::Auto => crate::memory_mode::resolve_auto(width, height, cap)?,
        };
        let inner: Box<dyn ButteraugliInner + Send> = match (backend, resolved) {
            #[cfg(feature = "cuda")]
            (Backend::Cuda, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Butteraugli::<cubecl::cuda::CudaRuntime>::new_multires(
                    client, width, height,
                ))
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(
                    Butteraugli::<cubecl::cuda::CudaRuntime>::new_multires_strip(
                        client, width, height, h_body,
                    ),
                )
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Butteraugli::<cubecl::wgpu::WgpuRuntime>::new_multires(
                    client, width, height,
                ))
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(
                    Butteraugli::<cubecl::wgpu::WgpuRuntime>::new_multires_strip(
                        client, width, height, h_body,
                    ),
                )
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Butteraugli::<cubecl::cpu::CpuRuntime>::new_multires(
                    client, width, height,
                ))
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Butteraugli::<cubecl::cpu::CpuRuntime>::new_multires_strip(
                    client, width, height, h_body,
                ))
            }
            #[allow(unreachable_patterns)]
            _ => {
                let _ = (width, height);
                return Err(crate::Error::ModeUnsupported("no-backend-enabled"));
            }
        };
        Ok(Self {
            inner,
            params,
            backend,
            cached_ref_strip: None,
        })
    }

    /// Build a [`ButteraugliOpaque`] from a caller-supplied cubecl
    /// client (which may be bound to an explicit stream). Internal
    /// plumbing for [`crate::session::new_opaque_on_stream`]. Mirrors
    /// [`Self::new_with_memory_mode`]'s host-side mode resolution +
    /// multires/strip selection, but on the supplied generic client.
    #[cfg(feature = "cubecl-types")]
    pub(crate) fn build_from_client<R: cubecl::Runtime>(
        client: cubecl::prelude::ComputeClient<R>,
        backend: Backend,
        width: u32,
        height: u32,
        params: ButteraugliParams,
        mode: crate::MemoryMode,
    ) -> Result<Self>
    where
        Butteraugli<R>: Send + 'static,
    {
        let cap = crate::memory_mode::vram_cap_bytes();
        let resolved = match mode {
            crate::MemoryMode::Full => crate::ResolvedMode::Full,
            crate::MemoryMode::Strip { h_body } => crate::ResolvedMode::Strip {
                h_body: h_body
                    .unwrap_or_else(|| crate::memory_mode::auto_strip_body_for(width, height, cap)),
            },
            crate::MemoryMode::Tile { .. } => return Err(crate::Error::ModeUnsupported("Tile")),
            crate::MemoryMode::Auto => crate::memory_mode::resolve_auto(width, height, cap)?,
        };
        let inner: Box<dyn ButteraugliInner + Send> = match resolved {
            crate::ResolvedMode::Full => {
                Box::new(Butteraugli::<R>::new_multires(client, width, height))
            }
            crate::ResolvedMode::Strip { h_body } => Box::new(
                Butteraugli::<R>::new_multires_strip(client, width, height, h_body),
            ),
        };
        Ok(Self {
            inner,
            params,
            backend,
            cached_ref_strip: None,
        })
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Score one sRGB RGB8 pair.
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb, &self.params)
    }

    /// Score one sRGB RGB8 pair and also return the libjxl `pnorm_3`
    /// aggregation. The CubeCL fused reduction kernel produces both
    /// the max-norm `Score` and `pnorm_3` in one pass — this entry
    /// point exposes both without re-running the kernel.
    ///
    /// Callers that already use [`Self::compute_srgb_u8`] keep working
    /// unchanged; this is a strictly additive surface.
    pub fn compute_srgb_u8_with_pnorm3(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<(Score, f64)> {
        self.inner
            .compute_srgb_u8_with_pnorm3(ref_rgb, dis_rgb, &self.params)
    }

    /// Score from [`PixelSlice`] inputs. See `dssim-gpu`'s
    /// `compute_pixels` for the fast-path / conversion-path
    /// semantics; identical here.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_srgb_u8(&ref_buf, &dis_buf, &self.params)
    }

    /// Score against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. See per-crate typed
    /// `Butteraugli::compute_handles` for the layout contract.
    #[cfg(feature = "cubecl-types")]
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        self.inner
            .compute_handles(ref_handle, dis_handle, &self.params)
    }

    /// Pack a `width × height × 3` sRGB-u8 buffer into the packed-u32
    /// device handle layout that [`Self::compute_handles`] expects.
    #[cfg(feature = "cubecl-types")]
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        self.inner.pack_srgb(srgb)
    }

    /// Cache the reference image so subsequent
    /// [`Self::compute_with_reference_srgb_u8`] calls can score
    /// many distorted candidates against the same reference.
    ///
    /// - **Whole-image instances** cache the reference's opsin / blur /
    ///   masking state on device (subsequent cached-ref calls skip the
    ///   ref-side pyramid build).
    /// - **Strip-mode instances** (single-res or multires, including the
    ///   Auto-resolved-to-Strip case the umbrella builds) hold an owned
    ///   copy of the reference bytes (task #160) and replay the
    ///   pair-strip compute on `(held_ref, dist)` per cached-ref call.
    ///   That replay is exactly the one-shot `compute_srgb_u8(ref,
    ///   dist)` with the reference held, so the cached-ref score is
    ///   **identical** to the one-shot score. This makes
    ///   `set_reference` universally supported regardless of the
    ///   resolved memory mode.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::DimensionMismatch`] if `ref_rgb.len()` doesn't
    ///   match the configured `width × height × 3`.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        if self.inner.is_strip_mode() {
            // Strip-mode cached reference (task #160): hold the reference
            // bytes and replay the pair-strip compute later. Validate the
            // dimensions up front so a wrong-sized reference fails here
            // (same contract as the whole-image on-device path) rather
            // than on the first compute call.
            let (w, h) = self.inner.dims();
            let expected = (w as usize) * (h as usize) * 3;
            if ref_rgb.len() != expected {
                return Err(crate::Error::DimensionMismatch {
                    expected,
                    got: ref_rgb.len(),
                });
            }
            self.cached_ref_strip = Some(ref_rgb.to_vec());
            return Ok(());
        }
        self.inner.set_reference_srgb_u8(ref_rgb, &self.params)
    }

    /// Score a distorted candidate against the cached reference set
    /// by [`Self::set_reference_srgb_u8`]. Returns
    /// [`crate::Error::NoCachedReference`] if no reference is cached.
    ///
    /// For strip-mode instances this replays the pair-strip compute on
    /// the held reference and `dis_rgb` (task #160), producing a score
    /// **identical** to the one-shot `compute_srgb_u8` path.
    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        if self.inner.is_strip_mode() {
            // Replay the pair-strip compute on the held reference. We
            // clone the held buffer out of `self` so the `&mut self`
            // borrow needed by `compute_srgb_u8` doesn't conflict with
            // the immutable borrow of `cached_ref_strip` (cubecl `Handle`
            // re-upload is the dominant cost; the host Vec clone is
            // negligible beside it).
            let held = match self.cached_ref_strip.as_ref() {
                Some(buf) => buf.clone(),
                None => return Err(crate::Error::NoCachedReference),
            };
            return self.inner.compute_srgb_u8(&held, dis_rgb, &self.params);
        }
        self.inner
            .compute_with_reference_srgb_u8(dis_rgb, &self.params)
    }

    /// Drop cached reference state (both the strip-mode held buffer and
    /// any whole-image on-device cache).
    pub fn clear_reference(&mut self) {
        self.cached_ref_strip = None;
        self.inner.clear_reference();
    }

    /// `true` if a reference has been cached.
    pub fn has_reference(&self) -> bool {
        if self.inner.is_strip_mode() {
            return self.cached_ref_strip.is_some();
        }
        self.inner.has_reference()
    }
}

#[cfg(feature = "pixels")]
pub(crate) fn to_srgb_rgb8(
    s: &PixelSlice<'_>,
    expected_w: u32,
    expected_h: u32,
) -> Result<Vec<u8>> {
    if s.width() != expected_w || s.rows() != expected_h {
        let expected = (expected_w as usize) * (expected_h as usize) * 3;
        let got = (s.width() as usize) * (s.rows() as usize) * 3;
        return Err(Error::DimensionMismatch { expected, got });
    }
    let target = zenpixels::PixelDescriptor::RGB8_SRGB;
    if s.descriptor() == target {
        return Ok(s.contiguous_bytes().into_owned());
    }
    zenmetrics_gpu_core::convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w as usize) * (expected_h as usize) * 3,
        got: (s.width() as usize) * (s.rows() as usize) * 3,
    })
}
