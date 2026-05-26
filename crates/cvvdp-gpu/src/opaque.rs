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
    fn dims(&self) -> (u32, u32);
    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score>;
    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle>;
    fn compute_srgb_u8_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<Score>;
    fn warm_reference_srgb(&mut self, ref_rgb: &[u8]) -> Result<()>;
    fn compute_with_warm_ref_srgb(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score>;
    #[allow(clippy::too_many_arguments)]
    fn compute_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score>;
    fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()>;
    fn compute_with_warm_ref_from_linear_planes(
        &mut self,
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score>;
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

    fn dims(&self) -> (u32, u32) {
        Cvvdp::dimensions(self)
    }

    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        let jod = Cvvdp::compute_handles(self, ref_handle, dis_handle)?;
        Ok(Score {
            value: jod,
            metric_name: "cvvdp",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        Cvvdp::pack_srgb_into_packed_u32_handle(self, srgb)
    }

    fn compute_srgb_u8_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<Score> {
        let jod = Cvvdp::score_with_diffmap(self, ref_rgb, dis_rgb, diffmap_out)?;
        Ok(Score {
            value: f64::from(jod),
            metric_name: "cvvdp",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn warm_reference_srgb(&mut self, ref_rgb: &[u8]) -> Result<()> {
        Cvvdp::warm_reference(self, ref_rgb)
    }

    fn compute_with_warm_ref_srgb(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        // For the sRGB-byte warm-ref scalar path the existing public
        // method takes the construction-time PPD; we read it back via
        // `Cvvdp::geometry_ppd_for_warm_ref` (a tiny accessor we add
        // alongside the diffmap API so opaque doesn't reach into a
        // private field). The diffmap variant wraps it internally.
        let jod = match diffmap_out {
            Some(out) => Cvvdp::score_with_warm_ref_diffmap(self, dis_rgb, out)?,
            None => {
                let ppd = Cvvdp::geometry_ppd_for_warm_ref(self);
                Cvvdp::compute_dkl_jod_with_warm_ref(self, dis_rgb, ppd)?
            }
        };
        Ok(Score {
            value: f64::from(jod),
            metric_name: "cvvdp",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn compute_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        let jod = match diffmap_out {
            Some(out) => Cvvdp::score_from_linear_planes_with_diffmap(
                self, ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, out,
            )?,
            None => {
                Cvvdp::score_from_linear_planes(self, ref_r, ref_g, ref_b, dis_r, dis_g, dis_b)?
            }
        };
        Ok(Score {
            value: f64::from(jod),
            metric_name: "cvvdp",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        Cvvdp::warm_reference_from_linear_planes(self, ref_r, ref_g, ref_b)
    }

    fn compute_with_warm_ref_from_linear_planes(
        &mut self,
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        let jod = match diffmap_out {
            Some(out) => Cvvdp::score_from_linear_planes_with_warm_ref_diffmap(
                self, dis_r, dis_g, dis_b, out,
            )?,
            None => Cvvdp::score_from_linear_planes_with_warm_ref(self, dis_r, dis_g, dis_b)?,
        };
        Ok(Score {
            value: f64::from(jod),
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
    /// `params::DisplayGeometry::STANDARD_4K`). Equivalent to
    /// `new_with_memory_mode(.., MemoryMode::Auto)`.
    pub fn new(backend: Backend, width: u32, height: u32, params: CvvdpParams) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque cvvdp scorer with an explicit
    /// [`MemoryMode`](crate::MemoryMode). cvvdp-gpu has no Strip
    /// implementation — `MemoryMode::Strip` / `Tile` return
    /// [`crate::Error::ModeUnsupported`].
    ///
    /// Equivalent to
    /// `new_with_geometry_and_memory_mode(.., DisplayGeometry::STANDARD_4K, mode)`.
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: CvvdpParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        Self::new_with_geometry_and_memory_mode(
            backend,
            width,
            height,
            params,
            crate::params::DisplayGeometry::STANDARD_4K,
            mode,
        )
    }

    /// Construct an opaque cvvdp scorer with a non-default
    /// [`DisplayGeometry`](crate::params::DisplayGeometry). Equivalent
    /// to `new_with_geometry_and_memory_mode(.., geometry, MemoryMode::Auto)`.
    ///
    /// Use this constructor (instead of [`Self::new`]) when the
    /// scoring context isn't STANDARD_4K — e.g. a phone-class viewing
    /// geometry (≈340 PPD on iPhone 14 Pro at 0.30 m) or a TV-class
    /// geometry (≈57 PPD on a 65″ panel at 3 m). PPD shifts the
    /// spatial frequencies the castleCSF kernels are queried with,
    /// which materially changes JOD scores — especially in the
    /// finest pyramid bands.
    ///
    /// See [`crate::pipeline::Cvvdp::new_with_geometry`] for the
    /// underlying typed API surface this forwards to.
    pub fn new_with_geometry(
        backend: Backend,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
    ) -> Result<Self> {
        Self::new_with_geometry_and_memory_mode(
            backend,
            width,
            height,
            params,
            geometry,
            crate::MemoryMode::Auto,
        )
    }

    /// [`MemoryMode`](crate::MemoryMode) + geometry variant of
    /// [`Self::new_with_geometry`]. Mirrors
    /// [`Self::new_with_memory_mode`]'s memory-mode semantics
    /// (`Strip` / `Tile` → [`crate::Error::ModeUnsupported`]) and
    /// accepts a custom viewing [`DisplayGeometry`](crate::params::DisplayGeometry)
    /// for the underlying [`Cvvdp::new_with_geometry`] dispatch.
    pub fn new_with_geometry_and_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: CvvdpParams,
        geometry: crate::params::DisplayGeometry,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        // Resolve the mode host-side to surface ModeUnsupported /
        // TooBigForFull before the backend allocation runs.
        let cap = crate::memory_mode::vram_cap_bytes();
        match mode {
            crate::MemoryMode::Strip { .. } => {
                return Err(crate::Error::ModeUnsupported("Strip"));
            }
            crate::MemoryMode::Tile { .. } => return Err(crate::Error::ModeUnsupported("Tile")),
            crate::MemoryMode::Full => {}
            crate::MemoryMode::Auto => {
                let _ = crate::memory_mode::resolve_auto(width, height, cap)?;
            }
        }
        let inner: Box<dyn CvvdpInner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::cuda::CudaRuntime>::new_with_geometry(
                    client, width, height, params, geometry,
                )?)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::wgpu::WgpuRuntime>::new_with_geometry(
                    client, width, height, params, geometry,
                )?)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Cvvdp::<cubecl::cpu::CpuRuntime>::new_with_geometry(
                    client, width, height, params, geometry,
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
        // Stored width/height and inner.dims() are equivalent — the
        // inner is constructed with the same w/h passed to Self::new.
        // Prefer the inner dispatch so the trait method isn't dead
        // code (for future inner types that compute dims dynamically).
        self.inner.dims()
    }

    /// Score one reference / distorted pair (packed sRGB RGB8).
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb)
    }

    /// Score from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let ref_buf = to_srgb_rgb8(&r, self.width, self.height)?;
        let dis_buf = to_srgb_rgb8(&d, self.width, self.height)?;
        self.inner.compute_srgb_u8(&ref_buf, &dis_buf)
    }

    /// Score against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. See the typed
    /// [`Cvvdp::compute_handles`](crate::pipeline::Cvvdp::compute_handles)
    /// for the layout contract.
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
        self.inner.pack_srgb(srgb)
    }

    /// Score one (reference, distorted) sRGB pair AND fill a per-pixel
    /// diffmap. On return, `diffmap_out.len() == width * height` and
    /// values are non-negative f32 row-major.
    ///
    /// See [`crate::kernels::diffmap`] module docs for the recipe.
    pub fn compute_srgb_u8_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<Score> {
        self.inner
            .compute_srgb_u8_with_diffmap(ref_rgb, dis_rgb, diffmap_out)
    }

    /// Warm the REF side for repeated `compute_with_warm_ref_*` calls.
    /// Subsequent scores against the cached REF skip the REF half of
    /// the pipeline. See [`crate::pipeline::Cvvdp::warm_reference`].
    pub fn warm_reference_srgb(&mut self, ref_rgb: &[u8]) -> Result<()> {
        self.inner.warm_reference_srgb(ref_rgb)
    }

    /// Score a DIST candidate against the warm REF state. Pass
    /// `Some(&mut Vec<f32>)` to also fill a per-pixel diffmap.
    pub fn compute_with_warm_ref_srgb(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        self.inner.compute_with_warm_ref_srgb(dis_rgb, diffmap_out)
    }

    /// Score from three planar `W × H` linear-RGB f32 buffers (unit-
    /// scaled sRGB linear-light). Skips the host-side sRGB pack +
    /// LUT conversion. Pass `Some(&mut Vec<f32>)` to also fill a
    /// per-pixel diffmap. Mirrors butteraugli-gpu's W44-PHASE3-B4
    /// `compute_with_reference_from_linear_planes` pattern.
    #[allow(clippy::too_many_arguments)] // 6 planar slices + diffmap option — natural shape for the linear-planes API.
    pub fn compute_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        self.inner
            .compute_from_linear_planes(ref_r, ref_g, ref_b, dis_r, dis_g, dis_b, diffmap_out)
    }

    /// Warm the REF side from three planar linear-RGB f32 buffers.
    /// See [`crate::pipeline::Cvvdp::warm_reference_from_linear_planes`].
    pub fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        self.inner
            .warm_reference_from_linear_planes(ref_r, ref_g, ref_b)
    }

    /// Score a DIST candidate (linear-RGB f32 planes) against the warm
    /// REF state. Pass `Some(&mut Vec<f32>)` to also fill a per-pixel
    /// diffmap.
    pub fn compute_with_warm_ref_from_linear_planes(
        &mut self,
        dis_r: &[f32],
        dis_g: &[f32],
        dis_b: &[f32],
        diffmap_out: Option<&mut Vec<f32>>,
    ) -> Result<Score> {
        self.inner
            .compute_with_warm_ref_from_linear_planes(dis_r, dis_g, dis_b, diffmap_out)
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
    use zenpixels_convert::{convert_row, ConvertPlan};
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
