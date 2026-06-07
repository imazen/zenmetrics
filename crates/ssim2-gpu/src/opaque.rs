//! Uniform opaque API for `ssim2-gpu`.
//!
//! See `dssim-gpu/src/opaque.rs` for the full design rationale.

#[cfg(feature = "pixels")]
use crate::Error;
use crate::Result;
use crate::pipeline::Ssim2;
use crate::skipmap::Ssim2Mode;

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// The backend selector and uniform score type are shared verbatim
/// across all six `*-gpu` metric crates — see [`zenmetrics_gpu_core`].
/// Re-exported here so `crate::Backend` / `crate::opaque::Score` keep
/// resolving.
pub use zenmetrics_gpu_core::{Backend, Score};

/// Configuration for [`Ssim2Opaque`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct Ssim2Params {
    /// Skip-map dispatch mode — see [`Ssim2Mode`]. Default is
    /// `Faster` (matches the typed `Ssim2::compute` default).
    pub mode: Ssim2Mode,
    /// Blur kernel selector (only meaningful when the `fir` Cargo
    /// feature is enabled). Off-`fir` builds ignore this field.
    #[cfg(feature = "fir")]
    pub blur: crate::Ssim2Blur,
}

impl Ssim2Params {
    /// Default parameter bundle (Faster skip-map, default blur).
    pub const DEFAULT: Self = Self {
        mode: Ssim2Mode::Faster,
        #[cfg(feature = "fir")]
        blur: crate::Ssim2Blur::Iir,
    };
}

trait Ssim2Inner: Send {
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8], mode: Ssim2Mode)
    -> Result<Score>;
    fn dims(&self) -> (u32, u32);
    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
        mode: Ssim2Mode,
    ) -> Result<Score>;
    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle>;
    /// Cache the reference image. Subsequent
    /// `compute_with_reference_srgb_u8` calls skip the
    /// ref-side multi-scale Gaussian + XYB-pyramid build.
    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()>;
    /// Score one distorted candidate against the cached reference.
    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score>;
    /// Drop cached reference state.
    fn clear_reference(&mut self);
    /// Whether a reference has been cached.
    fn has_reference(&self) -> bool;
}

impl<R> Ssim2Inner for Ssim2<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        mode: Ssim2Mode,
    ) -> Result<Score> {
        let r = Ssim2::compute_with_mode(self, mode, ref_rgb, dis_rgb)?;
        Ok(Score {
            value: r.score,
            metric_name: "ssim2",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn dims(&self) -> (u32, u32) {
        Ssim2::dimensions(self)
    }

    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
        mode: Ssim2Mode,
    ) -> Result<Score> {
        let r = Ssim2::compute_handles_with_mode(self, mode, ref_handle, dis_handle)?;
        Ok(Score {
            value: r.score,
            metric_name: "ssim2",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        Ssim2::pack_srgb_into_packed_u32_handle(self, srgb)
    }

    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        Ssim2::set_reference(self, ref_rgb)
    }

    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        let r = Ssim2::compute_with_reference(self, dis_rgb)?;
        Ok(Score {
            value: r.score,
            metric_name: "ssim2",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn clear_reference(&mut self) {
        Ssim2::clear_reference(self)
    }

    fn has_reference(&self) -> bool {
        Ssim2::has_reference(self)
    }
}

/// Minimum per-axis dimension SSIMULACRA2's pyramid needs (the typed
/// pipeline rejects `< 8×8`). Sub-`MIN_DIM` inputs are reflect(mirror)-
/// padded up to this floor so the scorer returns a finite score down to
/// 1×1 instead of `InvalidImageSize`. NO-OP at ≥8px.
const MIN_DIM: u32 = 8;

/// Opaque SSIMULACRA2 scorer.
pub struct Ssim2Opaque {
    inner: Box<dyn Ssim2Inner + Send>,
    params: Ssim2Params,
    #[allow(dead_code)]
    backend: Backend,
    /// Caller-requested logical dims. Smaller than the inner pipeline's
    /// dims when the request was sub-8px (inner built for the padded
    /// size); compute entries reflect-pad inputs up to that size. Equal
    /// at ≥8px (pad helpers become no-op borrows).
    logical_w: u32,
    logical_h: u32,
}

impl Ssim2Opaque {
    /// Construct an opaque SSIMULACRA2 scorer. Equivalent to
    /// `new_with_memory_mode(.., MemoryMode::Auto)`.
    pub fn new(backend: Backend, width: u32, height: u32, params: Ssim2Params) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque SSIMULACRA2 scorer with an explicit
    /// [`MemoryMode`](crate::MemoryMode). ssim2-gpu has no Strip
    /// implementation yet — `MemoryMode::Strip` / `Tile` return
    /// [`crate::Error::ModeUnsupported`].
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: Ssim2Params,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        // Reflect-pad sub-8px requests up to the pyramid floor; the inner
        // pipeline is built for the padded size and compute entries pad
        // their inputs. `logical_*` records the caller's request so
        // `dims()` stays honest.
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_DIM);
        let height = height.max(MIN_DIM);
        let inner: Box<dyn Ssim2Inner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::cuda::CudaRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::wgpu::WgpuRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::cpu::CpuRuntime>::new_with_memory_mode(
                    client, width, height, mode,
                )?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
            }
        };
        Ok(Self {
            inner,
            params,
            backend,
            logical_w,
            logical_h,
        })
    }

    /// Build an [`Ssim2Opaque`] from a caller-supplied cubecl client
    /// (which may be bound to an explicit stream). Internal plumbing for
    /// [`crate::session::new_opaque_on_stream`]; the default-stream
    /// constructor inlines the equivalent per-backend `Ssim2::<R>::new`
    /// call so it can pick the runtime type without a generic boundary.
    #[cfg(feature = "cubecl-types")]
    pub(crate) fn build_from_client<R: cubecl::Runtime>(
        client: cubecl::prelude::ComputeClient<R>,
        backend: Backend,
        width: u32,
        height: u32,
        params: Ssim2Params,
        mode: crate::MemoryMode,
    ) -> Result<Self>
    where
        Ssim2<R>: Send + 'static,
    {
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_DIM);
        let height = height.max(MIN_DIM);
        let s = Ssim2::<R>::new_with_memory_mode(client, width, height, mode)?;
        #[cfg(feature = "fir")]
        let s = s.with_blur(params.blur);
        Ok(Self {
            inner: Box::new(s),
            params,
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

    /// Score one sRGB RGB8 pair. Sub-8px inputs are reflect-padded.
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_srgb_u8(&r, &d, self.params.mode)
    }

    /// Score from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        let rp = self.pad_rgb(&ref_buf)?;
        let dp = self.pad_rgb(&dis_buf)?;
        self.inner
            .compute_srgb_u8(&rp, &dp, self.params.mode)
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
        self.inner
            .compute_handles(ref_handle, dis_handle, self.params.mode)
    }

    /// Pack a `width × height × 3` sRGB-u8 buffer into the packed-u32
    /// device handle layout that [`Self::compute_handles`] expects. A
    /// sub-8px buffer is reflect-padded first, so the resulting handle
    /// matches the padded inner pipeline (and `compute_handles` works on
    /// it unchanged).
    #[cfg(feature = "cubecl-types")]
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        let s = self.pad_rgb(srgb)?;
        self.inner.pack_srgb(&s)
    }

    /// Cache the reference image's SSIMULACRA2 state on device.
    /// Subsequent [`Self::compute_with_reference_srgb_u8`]
    /// calls skip the ref-side multi-scale Gaussian + XYB pyramid
    /// build.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        let r = self.pad_rgb(ref_rgb)?;
        self.inner.set_reference_srgb_u8(&r)
    }

    /// Score a distorted candidate against the cached reference set
    /// by [`Self::set_reference_srgb_u8`]. Returns
    /// [`crate::Error::NoCachedReference`] if no reference is cached.
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
