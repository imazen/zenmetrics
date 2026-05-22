//! Uniform opaque API for `dssim-gpu`.
//!
//! The opaque shim erases the `<R: Runtime>` generic from
//! [`crate::pipeline::Dssim`], so downstream consumers can construct a
//! metric instance by passing a runtime [`Backend`] enum value rather
//! than a cubecl-runtime type parameter. The shim is the surface
//! Phase 3's umbrella crate (`zenmetrics-api`) will unify on; every
//! metric crate ships the same shape.
//!
//! See the crate root for the typed-generic alternative
//! (gated behind the `cubecl-types` feature).

use crate::pipeline::Dssim;
#[cfg(feature = "pixels")]
use crate::Error;
use crate::{NUM_SCALES, Result};

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
    /// CPU reference backend (requires the `cpu` Cargo feature). Note
    /// the dssim-gpu kernels rely on `CUBE_COUNT` builtins and
    /// `Atomic<f32>` which `cubecl-cpu` does NOT support — `Cpu` is
    /// accepted for API uniformity but kernels will panic at first
    /// dispatch. Use `Cuda` or `Wgpu` instead.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim. Carries
/// the underlying `f64` value plus identifying metadata so multiple
/// metric implementations can be compared side-by-side.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The numeric score (interpretation is per-metric — see
    /// `metric_name`).
    pub value: f64,
    /// Short metric identifier (`"dssim"`, `"ssim2"`, etc).
    pub metric_name: &'static str,
    /// Implementation version tag (currently the crate's
    /// `CARGO_PKG_VERSION`).
    pub metric_version: &'static str,
}

/// Configuration for [`DssimOpaque`]. Currently empty — `dssim-gpu`
/// has no user-tunable knobs. Exists for API uniformity with the
/// other metric crates' `<Metric>Params` types.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct DssimParams;

impl DssimParams {
    /// Default parameter bundle (unit struct — placeholder for
    /// future per-scale weight tuning).
    pub const DEFAULT: Self = Self;
}

/// Internal trait — erases the cubecl-runtime generic from the
/// typed `Dssim<R>` so different backends can live behind a single
/// `Box<dyn DssimInner + Send>` handle.
trait DssimInner: Send {
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score>;
    fn dims(&self) -> (u32, u32);
    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score>;
    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle>;
}

impl<R> DssimInner for Dssim<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        let r = Dssim::compute(self, ref_rgb, dis_rgb)?;
        Ok(Score {
            value: r.score,
            metric_name: "dssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn dims(&self) -> (u32, u32) {
        Dssim::dimensions(self)
    }

    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        let r = Dssim::compute_handles(self, ref_handle, dis_handle)?;
        Ok(Score {
            value: r.score,
            metric_name: "dssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        Dssim::pack_srgb_into_packed_u32_handle(self, srgb)
    }
}

/// Opaque DSSIM scorer. Uniform constructor + score API across all six
/// `*-gpu` metric crates. The cubecl `Runtime` generic is hidden
/// behind the [`Backend`] enum, so downstream consumers don't pin to a
/// specific cubecl version through their public types.
///
/// ```no_run
/// use dssim_gpu::{DssimOpaque, Backend, DssimParams};
///
/// let mut d = DssimOpaque::new(Backend::Cuda, 256, 256, DssimParams::DEFAULT)?;
/// let ref_buf = vec![128u8; 256 * 256 * 3];
/// let dis_buf = vec![100u8; 256 * 256 * 3];
/// let score = d.compute_srgb_u8(&ref_buf, &dis_buf)?;
/// println!("{} = {:.6} (impl {})", score.metric_name, score.value, score.metric_version);
/// # Ok::<(), dssim_gpu::Error>(())
/// ```
pub struct DssimOpaque {
    inner: Box<dyn DssimInner + Send>,
    #[allow(dead_code)]
    backend: Backend,
}

impl DssimOpaque {
    /// Construct an opaque DSSIM scorer for `width × height` images on
    /// the requested [`Backend`]. The `_params` argument is reserved
    /// for future use (currently unit-typed).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`Error`] if the GPU allocation fails or
    /// the image is smaller than the minimum 8×8 size required for the
    /// 5-scale pyramid.
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: DssimParams,
    ) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque DSSIM scorer with an explicit
    /// [`MemoryMode`](crate::MemoryMode). dssim-gpu is **NOT
    /// strip-preferred** — see
    /// [`Dssim::new_with_memory_mode`](crate::pipeline::Dssim::new_with_memory_mode).
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        _params: DssimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        let inner: Box<dyn DssimInner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Dssim::<cubecl::cuda::CudaRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Dssim::<cubecl::wgpu::WgpuRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Dssim::<cubecl::cpu::CpuRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?)
            }
        };
        Ok(Self { inner, backend })
    }

    /// Return the configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Number of pyramid scales used internally — exposed for parity
    /// with the typed `Dssim::n_scales`.
    pub fn n_scales(&self) -> usize {
        NUM_SCALES
    }

    /// Score one reference / distorted pair, both packed sRGB
    /// `R, G, B, R, G, B, …` of length `width × height × 3`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either buffer's length
    /// differs from `width × height × 3`.
    pub fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb)
    }

    /// Score one reference / distorted pair from [`PixelSlice`]
    /// inputs.
    ///
    /// # Fast path
    ///
    /// If both inputs are `PixelDescriptor::RGB8_SRGB` with tightly-
    /// packed rows (stride == `width * 3`), the slice bytes are passed
    /// to the kernel without allocation or conversion.
    ///
    /// # Conversion path
    ///
    /// Otherwise — different format (Rgba8, Bgr8, OklabF32, …), wrong
    /// transfer, or stride padding — the slice is converted into a
    /// per-call sRGB-RGB8 buffer via the per-row converter from
    /// `zenpixels-convert`. This allocates `width × height × 3` bytes
    /// per call. Document this in callers' hot-path benchmarks.
    ///
    /// # Errors
    ///
    /// Returns [`Error::DimensionMismatch`] if either slice's
    /// `(width, rows)` doesn't match the scorer's configured
    /// `(width, height)`, or if `zenpixels-convert` cannot build a
    /// conversion path to sRGB RGB8 (rare — most non-CMYK formats
    /// convert).
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_srgb_u8(&ref_buf, &dis_buf)
    }

    /// Score against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point.
    ///
    /// Handle layout MUST be the packed-u32 form produced by
    /// [`Self::pack_srgb_into_packed_u32_handle`] (one `u32` per
    /// pixel, `R | G<<8 | B<<16`, length `width × height`). The
    /// handle is expected to live on the same cubecl client that
    /// constructed this opaque; sharing handles across clients is
    /// undefined behaviour at the cubecl layer and not validated.
    #[cfg(feature = "cubecl-types")]
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        self.inner.compute_handles(ref_handle, dis_handle)
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

/// Convert a [`PixelSlice`] into a tightly-packed sRGB RGB8 buffer.
/// Fast path: if the slice already matches and is contiguous, returns
/// a `Vec<u8>` cloned from the borrowed bytes (the inner kernel only
/// accepts `&[u8]`, so we always materialise a contiguous buffer).
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
        // Fast path — already the right format. Strip stride padding
        // if any (per CLAUDE.md "Pixel Buffer APIs": natively support
        // strided rows).
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
