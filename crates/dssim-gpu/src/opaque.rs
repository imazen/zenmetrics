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

#[cfg(feature = "pixels")]
use crate::Error;
use crate::pipeline::Dssim;
use crate::{NUM_SCALES, Result};

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// The backend selector and uniform score type are shared verbatim
/// across all six `*-gpu` metric crates — see [`zenmetrics_gpu_core`].
/// Re-exported here so `crate::Backend` / `crate::opaque::Score` keep
/// resolving.
pub use zenmetrics_gpu_core::{Backend, Score};

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
    /// Cache the reference image's linear-RGB pyramid.
    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()>;
    /// Score one candidate against the cached reference.
    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score>;
    /// Drop cached reference state.
    fn clear_reference(&mut self);
    /// Whether a reference has been cached.
    fn has_reference(&self) -> bool;
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

    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        Dssim::set_reference(self, ref_rgb)
    }

    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        let r = Dssim::compute_with_reference(self, dis_rgb)?;
        Ok(Score {
            value: r.score,
            metric_name: "dssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn clear_reference(&mut self) {
        Dssim::clear_reference(self)
    }

    fn has_reference(&self) -> bool {
        Dssim::has_reference(self)
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
/// // `Backend` variants are feature-gated; pick whichever this build has
/// // (the CI doctest job runs under `wgpu`, not `cuda`).
/// # #[cfg(feature = "cuda")] let backend = Backend::Cuda;
/// # #[cfg(all(feature = "wgpu", not(feature = "cuda")))] let backend = Backend::Wgpu;
/// # #[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))] let backend = Backend::Cpu;
/// let mut d = DssimOpaque::new(backend, 256, 256, DssimParams::DEFAULT)?;
/// let ref_buf = vec![128u8; 256 * 256 * 3];
/// let dis_buf = vec![100u8; 256 * 256 * 3];
/// let score = d.compute_srgb_u8(&ref_buf, &dis_buf)?;
/// println!("{} = {:.6} (impl {})", score.metric_name, score.value, score.metric_version);
/// # Ok::<(), dssim_gpu::Error>(())
/// ```
/// Minimum per-axis dimension DSSIM's 5-scale pyramid needs (the typed
/// pipeline rejects `< 8×8`). Sub-`MIN_DIM` inputs are reflect(mirror)-
/// padded up to this floor so the scorer returns a finite score down to
/// 1×1 instead of `InvalidImageSize`. NO-OP at ≥8px.
const MIN_DIM: u32 = 8;

pub struct DssimOpaque {
    inner: Box<dyn DssimInner + Send>,
    #[allow(dead_code)]
    backend: Backend,
    /// Caller-requested logical dims. Smaller than the inner pipeline's
    /// dims when the request was sub-8px; compute entries reflect-pad
    /// inputs up to that size. Equal at ≥8px (pad becomes a no-op borrow).
    logical_w: u32,
    logical_h: u32,
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
    pub fn new(backend: Backend, width: u32, height: u32, params: DssimParams) -> Result<Self> {
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
        // Reflect-pad sub-8px requests up to the pyramid floor; inner is
        // built padded, compute entries pad inputs. `logical_*` records
        // the caller's request so `dims()` stays honest.
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_DIM);
        let height = height.max(MIN_DIM);
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
        Ok(Self {
            inner,
            backend,
            logical_w,
            logical_h,
        })
    }

    /// Build a [`DssimOpaque`] from a caller-supplied cubecl client
    /// (which may be bound to an explicit stream). Internal plumbing for
    /// [`crate::session::new_opaque_on_stream`].
    #[cfg(feature = "cubecl-types")]
    pub(crate) fn build_from_client<R: cubecl::Runtime>(
        client: cubecl::prelude::ComputeClient<R>,
        backend: Backend,
        width: u32,
        height: u32,
        _params: DssimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self>
    where
        Dssim<R>: Send + 'static,
    {
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_DIM);
        let height = height.max(MIN_DIM);
        let inner: Box<dyn DssimInner + Send> = Box::new(Dssim::<R>::new_with_memory_mode(
            client, width, height, mode,
        )?);
        Ok(Self {
            inner,
            backend,
            logical_w,
            logical_h,
        })
    }

    /// Caller-requested logical `(width, height)`. For sub-8px images
    /// this is smaller than the internal padded pipeline size; compute
    /// entries reflect-pad inputs up to that size transparently.
    pub fn dims(&self) -> (u32, u32) {
        (self.logical_w, self.logical_h)
    }

    /// `true` when the inner pipeline was built larger than the logical
    /// image (sub-8px request needing reflect-pad). No-op fast path at ≥8px.
    #[inline]
    fn is_padded(&self) -> bool {
        let (pw, ph) = self.inner.dims();
        pw != self.logical_w || ph != self.logical_h
    }

    /// Reflect(mirror)-pad a packed `RGB8` buffer from the logical extent
    /// up to the inner pipeline's padded extent. Borrows unchanged at
    /// ≥8px. Validates the input length against the logical extent.
    fn pad_rgb<'a>(&self, src: &'a [u8]) -> Result<std::borrow::Cow<'a, [u8]>> {
        if !self.is_padded() {
            return Ok(std::borrow::Cow::Borrowed(src));
        }
        let (lw, lh) = (self.logical_w as usize, self.logical_h as usize);
        if src.len() != lw * lh * 3 {
            return Err(crate::Error::DimensionMismatch {
                expected: lw * lh * 3,
                got: src.len(),
            });
        }
        let (pw, ph) = self.inner.dims();
        Ok(std::borrow::Cow::Owned(zenmetrics_gpu_core::reflect_pad(
            src, lw, lh, pw as usize, ph as usize, 3,
        )))
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
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_srgb_u8(&r, &d)
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
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        let rp = self.pad_rgb(&ref_buf)?;
        let dp = self.pad_rgb(&dis_buf)?;
        self.inner.compute_srgb_u8(&rp, &dp)
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
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        let s = self.pad_rgb(srgb)?;
        self.inner.pack_srgb(&s)
    }

    /// Cache the reference image's linear-RGB pyramid on device.
    /// Subsequent [`Self::compute_with_reference_srgb_u8`]
    /// calls skip the ref-side pyramid build.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        let r = self.pad_rgb(ref_rgb)?;
        self.inner.set_reference_srgb_u8(&r)
    }

    /// Score a distorted candidate against the cached reference.
    /// Returns [`crate::Error::NoCachedReference`] if no reference
    /// is cached.
    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_with_reference_srgb_u8(&d)
    }

    /// Drop cached reference state.
    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    /// `true` if a reference has been cached.
    pub fn has_reference(&self) -> bool {
        self.inner.has_reference()
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
    zenmetrics_gpu_core::convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w as usize) * (expected_h as usize) * 3,
        got: (s.width() as usize) * (s.rows() as usize) * 3,
    })
}
