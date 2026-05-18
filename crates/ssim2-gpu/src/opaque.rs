//! Uniform opaque API for `ssim2-gpu`.
//!
//! See `dssim-gpu/src/opaque.rs` for the full design rationale.

use crate::pipeline::Ssim2;
use crate::skipmap::Ssim2Mode;
#[cfg(feature = "pixels")]
use crate::Error;
use crate::Result;

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
    /// ssim2-gpu reduction uses `Atomic<f32>` which the cubecl-cpu
    /// backend does not support — `Cpu` is accepted for API uniformity
    /// but kernels will panic at first dispatch.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// SSIMULACRA2 score (0..~100; higher = better, 100 = identical).
    pub value: f64,
    /// Short metric identifier (`"ssim2"`).
    pub metric_name: &'static str,
    /// Implementation version tag.
    pub metric_version: &'static str,
}

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
    fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        mode: Ssim2Mode,
    ) -> Result<Score>;
    fn dims(&self) -> (u32, u32);
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
}

/// Opaque SSIMULACRA2 scorer.
pub struct Ssim2Opaque {
    inner: Box<dyn Ssim2Inner + Send>,
    params: Ssim2Params,
    #[allow(dead_code)]
    backend: Backend,
}

impl Ssim2Opaque {
    /// Construct an opaque SSIMULACRA2 scorer.
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: Ssim2Params,
    ) -> Result<Self> {
        let inner: Box<dyn Ssim2Inner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::cuda::CudaRuntime>::new(client, width, height)?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::wgpu::WgpuRuntime>::new(client, width, height)?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                let s = Ssim2::<cubecl::cpu::CpuRuntime>::new(client, width, height)?;
                #[cfg(feature = "fir")]
                let s = s.with_blur(params.blur);
                Box::new(s)
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
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb, self.params.mode)
    }

    /// Score from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner
            .compute_srgb_u8(&ref_buf, &dis_buf, self.params.mode)
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
