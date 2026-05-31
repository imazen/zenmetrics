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

/// Selects the GPU/CPU backend the opaque shim dispatches to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend (NVIDIA, requires the `cuda` Cargo feature).
    #[cfg(feature = "cuda")]
    Cuda,
    /// WGPU backend (cross-vendor, requires the `wgpu` Cargo feature).
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU reference backend (requires the `cpu` Cargo feature). The
    /// butteraugli-gpu kernels use `Atomic<f32>` reductions that the
    /// cubecl-cpu backend does not support — `Cpu` is accepted for
    /// API uniformity but kernels will panic at first dispatch.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The numeric score (butteraugli max-norm).
    pub value: f64,
    /// Short metric identifier (`"butter"`).
    pub metric_name: &'static str,
    /// Implementation version tag.
    pub metric_version: &'static str,
}

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
    /// Cache the reference image's opsin pyramid + blur cascade.
    /// Strip-mode instances allocate a whole-image cache sibling on
    /// first call (Mode E — task #45 / issue #15) and run the
    /// reference-side pipeline on it. Returns
    /// [`crate::Error::StripModeUnsupported`] only for the multires-
    /// strip case, which doesn't have a Mode E port yet.
    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8], params: &ButteraugliParams) -> Result<()>;
    /// Score one candidate against the cached reference.
    fn compute_with_cached_reference_srgb_u8(
        &mut self,
        dis_rgb: &[u8],
        params: &ButteraugliParams,
    ) -> Result<Score>;
    /// Drop cached reference state.
    fn clear_reference(&mut self);
    /// Whether a reference has been cached.
    fn has_cached_reference(&self) -> bool;
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

    fn compute_with_cached_reference_srgb_u8(
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

    fn has_cached_reference(&self) -> bool {
        Butteraugli::has_cached_reference(self)
    }
}

/// Opaque butteraugli scorer.
pub struct ButteraugliOpaque {
    inner: Box<dyn ButteraugliInner + Send>,
    params: ButteraugliParams,
    #[allow(dead_code)]
    backend: Backend,
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

    /// Cache the reference image's opsin / blur / masking state on
    /// device. Subsequent [`Self::compute_with_cached_reference_srgb_u8`]
    /// calls skip the ref-side pyramid build.
    ///
    /// Strip-mode instances allocate a whole-image cache sibling on
    /// first call (Mode E — task #45 / issue #15) and run the
    /// reference-side pipeline on it; subsequent
    /// `compute_with_cached_reference_srgb_u8` calls walk dist strips
    /// while blitting cached ref planes per strip.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::StripModeUnsupported`] when invoked on a
    ///   multires-strip instance — the half-res strip cached-ref
    ///   path is not yet implemented.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        self.inner.set_reference_srgb_u8(ref_rgb, &self.params)
    }

    /// Score a distorted candidate against the cached reference set
    /// by [`Self::set_reference_srgb_u8`]. Returns
    /// [`crate::Error::NoCachedReference`] if no reference is cached.
    pub fn compute_with_cached_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        self.inner
            .compute_with_cached_reference_srgb_u8(dis_rgb, &self.params)
    }

    /// Drop cached reference state.
    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    /// `true` if a reference has been cached.
    pub fn has_cached_reference(&self) -> bool {
        self.inner.has_cached_reference()
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
    convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w as usize) * (expected_h as usize) * 3,
        got: (s.width() as usize) * (s.rows() as usize) * 3,
    })
}

#[cfg(feature = "pixels")]
fn convert_to_srgb_rgb8(
    s: &PixelSlice<'_>,
    target: zenpixels::PixelDescriptor,
) -> core::result::Result<Vec<u8>, zenpixels_convert::ConvertError> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let plan = ConvertPlan::new(s.descriptor(), target).map_err(|e| e.decompose().0)?;
    let w = s.width();
    let h = s.rows();
    let row_bytes = (w as usize) * target.bytes_per_pixel();
    let mut out = vec![0u8; row_bytes * (h as usize)];
    for y in 0..h {
        let src_row = s.row(y);
        let start = (y as usize) * row_bytes;
        let dst_row = &mut out[start..start + row_bytes];
        convert_row(&plan, src_row, dst_row, w);
    }
    Ok(out)
}
