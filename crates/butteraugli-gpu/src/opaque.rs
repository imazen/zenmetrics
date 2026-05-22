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

use crate::pipeline::Butteraugli;
#[cfg(feature = "pixels")]
use crate::Error;
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
    /// `MemoryMode::Full` and `MemoryMode::Auto` (when Auto picks
    /// Full) engage the multi-resolution sibling; `MemoryMode::Strip`
    /// and Auto-resolved-to-Strip drop to single-resolution since the
    /// half-res strip walker isn't implemented yet.
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
                h_body: h_body.unwrap_or_else(|| {
                    crate::memory_mode::auto_strip_body_for(width, height, cap)
                }),
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
                Box::new(Butteraugli::<cubecl::cuda::CudaRuntime>::new_strip(
                    client, width, height, h_body,
                ))
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
                Box::new(Butteraugli::<cubecl::wgpu::WgpuRuntime>::new_strip(
                    client, width, height, h_body,
                ))
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
                Box::new(Butteraugli::<cubecl::cpu::CpuRuntime>::new_strip(
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

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Score one sRGB RGB8 pair.
    pub fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb, &self.params)
    }

    /// Score from [`PixelSlice`] inputs. See `dssim-gpu`'s
    /// `compute_pixels` for the fast-path / conversion-path
    /// semantics; identical here.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
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
    pub fn pack_srgb_into_packed_u32_handle(
        &self,
        srgb: &[u8],
    ) -> Result<cubecl::server::Handle> {
        self.inner.pack_srgb(srgb)
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
