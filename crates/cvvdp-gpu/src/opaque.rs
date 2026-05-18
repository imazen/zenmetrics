//! Uniform opaque API for `cvvdp-gpu`.
//!
//! See `dssim-gpu/src/opaque.rs` for the full design rationale.

use crate::params::CvvdpParams;
use crate::pipeline::Cvvdp;
#[cfg(feature = "pixels")]
use crate::Error;
use crate::Result;

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// Selects the GPU/CPU backend the opaque shim dispatches to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend.
    #[cfg(feature = "cuda")]
    Cuda,
    /// WGPU backend (note: cvvdp-gpu's `score` path is not currently
    /// supported on wgpu — see `Cvvdp::score`'s "Backend support"
    /// section. Use `Cuda` for production scoring).
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU reference backend (use `compute_dkl_jod_host_pool` only —
    /// the GPU `score` path doesn't run on cubecl-cpu).
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// ColorVideoVDP JOD score (10 = identical, lower = worse;
    /// useful range typically 3..10 for SDR content).
    pub value: f64,
    /// Short metric identifier (`"cvvdp"`).
    pub metric_name: &'static str,
    /// Implementation version tag.
    pub metric_version: &'static str,
}

trait CvvdpInner: Send {
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score>;
}

impl<R> CvvdpInner for Cvvdp<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        let jod = Cvvdp::score(self, ref_rgb, dis_rgb)?;
        Ok(Score {
            value: jod,
            metric_name: "cvvdp",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }
}

/// Opaque ColorVideoVDP scorer.
pub struct CvvdpOpaque {
    inner: Box<dyn CvvdpInner + Send>,
    width: u32,
    height: u32,
    #[allow(dead_code)]
    backend: Backend,
}

impl CvvdpOpaque {
    /// Construct an opaque cvvdp scorer for `width × height` images
    /// using the standard 4K viewing geometry (see
    /// `params::DisplayGeometry::STANDARD_4K`).
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: CvvdpParams,
    ) -> Result<Self> {
        let inner: Box<dyn CvvdpInner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::cuda::CudaRuntime>::new(
                    client, width, height, params,
                )?)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::wgpu::WgpuRuntime>::new(
                    client, width, height, params,
                )?)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::cpu::CpuRuntime>::new(
                    client, width, height, params,
                )?)
            }
        };
        Ok(Self {
            inner,
            width,
            height,
            backend,
        })
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Score one reference / distorted pair (packed sRGB RGB8).
    pub fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb)
    }

    /// Score from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
        let ref_buf = to_srgb_rgb8(&r, self.width, self.height)?;
        let dis_buf = to_srgb_rgb8(&d, self.width, self.height)?;
        self.inner.compute_srgb_u8(&ref_buf, &dis_buf)
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
