//! Uniform opaque API for `iwssim-gpu`.
//!
//! See `dssim-gpu/src/opaque.rs` for the full design rationale.

#[cfg(feature = "pixels")]
use crate::Error;
use crate::pipeline::Iwssim;
use crate::{IwssimConfig, IwssimStrategy, NUM_SCALES, Result};

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// The backend selector and uniform score type are shared verbatim
/// across all six `*-gpu` metric crates — see [`zenmetrics_gpu_core`].
/// Re-exported here so `crate::Backend` / `crate::opaque::Score` keep
/// resolving.
pub use zenmetrics_gpu_core::{Backend, Score};

/// Configuration for [`IwssimOpaque`].
///
/// `allow_small` controls the sub-176-px (sub-[`MIN_NATIVE_DIM`])
/// behaviour: when true, small inputs are **tiled** up to the pyramid
/// floor and scored ([`IwssimStrategy::Tile`] — the empirically best
/// small-image strategy on the 980-pair CID22 validation,
/// `benchmarks/iwssim_smallimg/`), so the scorer returns a finite score
/// down to 1×1. When false the constructor returns
/// `Err(InvalidImageSize)` for sub-176 inputs.
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct IwssimParams {
    /// Forward to the small-image strategy: `true` → tile sub-176 inputs
    /// up to the floor and score; `false` → reject them. Defaults to
    /// `true` via [`IwssimParams::DEFAULT`] so the umbrella/CLI path
    /// scores every size (the typed `IwssimConfig` default stays Reject
    /// for crates.io back-compat).
    pub allow_small: bool,
}

impl Default for IwssimParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl IwssimParams {
    /// Default parameter bundle — `allow_small = true`: sub-176 inputs
    /// are tiled up to the pyramid floor and scored (so the metric has
    /// no minimum-size error). Changed from `false` so the opaque /
    /// umbrella path is size-robust by default; the typed crates.io
    /// `IwssimConfig::default()` is unaffected (still Reject).
    pub const DEFAULT: Self = Self { allow_small: true };

    /// Construct with `allow_small` set explicitly.
    pub const fn allow_small(allow: bool) -> Self {
        Self { allow_small: allow }
    }
}

trait IwssimInner: Send {
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score>;
    /// Score one gray-f32 pair (`width × height` samples, 0..255 scale —
    /// the same range the sRGB path's BT.601 conversion produces).
    fn compute_gray_f32(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<Score>;
    fn dims(&self) -> (u32, u32);
    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score>;
    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle>;
    /// Cache the reference image (Phase 2A). Dispatches to the
    /// stripped or whole-image typed pipeline based on strip mode.
    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()>;
    /// Score a candidate against the cached reference (Phase 2A).
    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score>;
    /// Drop the cached reference state.
    fn clear_reference(&mut self);
    /// Whether a reference has been cached and is ready to score against.
    fn has_reference(&self) -> bool;
}

impl<R> IwssimInner for Iwssim<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        let r = Iwssim::compute_rgb(self, ref_rgb, dis_rgb)?;
        Ok(Score {
            value: r.score,
            metric_name: "iwssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn compute_gray_f32(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<Score> {
        let r = if Iwssim::is_strip_mode(self) {
            Iwssim::compute_gray_stripped(self, ref_gray, dis_gray)?
        } else {
            Iwssim::compute_gray(self, ref_gray, dis_gray)?
        };
        Ok(Score {
            value: r.score,
            metric_name: "iwssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn dims(&self) -> (u32, u32) {
        Iwssim::dimensions(self)
    }

    #[cfg(feature = "cubecl-types")]
    fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        let r = Iwssim::compute_handles(self, ref_handle, dis_handle)?;
        Ok(Score {
            value: r.score,
            metric_name: "iwssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    #[cfg(feature = "cubecl-types")]
    fn pack_srgb(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
        Iwssim::pack_srgb_into_packed_u32_handle(self, srgb)
    }

    fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        if Iwssim::is_strip_mode(self) {
            Iwssim::set_rgb_reference_stripped(self, ref_rgb)
        } else {
            // Full mode: convert sRGB-u8 → gray-f32 host-side (BT.601
            // rounded, matches the on-device `rgb_u32_to_gray_kernel`
            // used by the Strip-mode set_rgb_reference_stripped path).
            let ref_gray = crate::pipeline::rgb_u8_to_gray_bt601(ref_rgb);
            Iwssim::set_reference(self, &ref_gray)
        }
    }

    fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        let result = if Iwssim::is_strip_mode(self) {
            Iwssim::compute_rgb_with_reference_stripped(self, dis_rgb)?
        } else {
            let dis_gray = crate::pipeline::rgb_u8_to_gray_bt601(dis_rgb);
            Iwssim::compute_with_reference(self, &dis_gray)?
        };
        Ok(Score {
            value: result.score,
            metric_name: "iwssim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    }

    fn clear_reference(&mut self) {
        if Iwssim::is_strip_mode(self) {
            Iwssim::clear_reference_stripped(self);
        } else {
            Iwssim::clear_reference(self);
        }
    }

    fn has_reference(&self) -> bool {
        if Iwssim::is_strip_mode(self) {
            Iwssim::has_cached_reference_stripped(self)
        } else {
            Iwssim::has_reference(self)
        }
    }
}

/// Opaque IW-SSIM scorer.
pub struct IwssimOpaque {
    inner: Box<dyn IwssimInner + Send>,
    #[allow(dead_code)]
    backend: Backend,
}

impl IwssimOpaque {
    /// Construct an opaque IW-SSIM scorer for `width × height` images.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidImageSize`] when `min(width, height) <
    /// 176` and `params.allow_small == false`. When `allow_small` is
    /// true, the pipeline is built at the padded dimensions
    /// `(max(width, 176), max(height, 176))` with the **tile** small-
    /// image strategy (changed from reflect-pad in this revision per
    /// `benchmarks/iwssim_smallimg/`; see
    /// [`IwssimConfig::allow_small`] for the back-compat contract).
    pub fn new(backend: Backend, width: u32, height: u32, params: IwssimParams) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque IW-SSIM scorer with an explicit
    /// [`MemoryMode`](crate::MemoryMode). iwssim-gpu is **NOT
    /// strip-preferred** — see
    /// [`Iwssim::new_with_memory_mode`](crate::pipeline::Iwssim::new_with_memory_mode).
    /// Auto picks Full whenever it fits the VRAM cap.
    ///
    /// Note: small-image adaptive padding (`params.allow_small`) is
    /// honored only on `MemoryMode::Full` / Auto-resolved-to-Full.
    /// Strip mode requires `min(w, h) ≥ MIN_NATIVE_DIM`; small
    /// images requested as Strip return
    /// [`crate::Error::InvalidImageSize`].
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: IwssimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
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
        let cfg = IwssimConfig {
            strategy: if params.allow_small {
                IwssimStrategy::Tile
            } else {
                IwssimStrategy::Reject
            },
        };
        let inner: Box<dyn IwssimInner + Send> = match (backend, resolved) {
            #[cfg(feature = "cuda")]
            (Backend::Cuda, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::cuda::CudaRuntime>::with_config(
                    client, width, height, cfg,
                )?)
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::cuda::CudaRuntime>::new_strip(
                    client, width, height, h_body,
                )?)
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::wgpu::WgpuRuntime>::with_config(
                    client, width, height, cfg,
                )?)
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::wgpu::WgpuRuntime>::new_strip(
                    client, width, height, h_body,
                )?)
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, crate::ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::cpu::CpuRuntime>::with_config(
                    client, width, height, cfg,
                )?)
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, crate::ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Iwssim::<cubecl::cpu::CpuRuntime>::new_strip(
                    client, width, height, h_body,
                )?)
            }
            #[allow(unreachable_patterns)]
            _ => return Err(crate::Error::ModeUnsupported("no-backend-enabled")),
        };
        Ok(Self { inner, backend })
    }

    /// Build an [`IwssimOpaque`] from a caller-supplied cubecl client
    /// (which may be bound to an explicit stream). Internal plumbing for
    /// [`crate::session::new_opaque_on_stream`]. Mirrors
    /// [`Self::new_with_memory_mode`]'s host-side mode resolution +
    /// Full(with_config)/Strip selection, on the supplied generic client.
    #[cfg(feature = "cubecl-types")]
    pub(crate) fn build_from_client<R: cubecl::Runtime>(
        client: cubecl::prelude::ComputeClient<R>,
        backend: Backend,
        width: u32,
        height: u32,
        params: IwssimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self>
    where
        Iwssim<R>: Send + 'static,
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
        let cfg = IwssimConfig {
            strategy: if params.allow_small {
                IwssimStrategy::Tile
            } else {
                IwssimStrategy::Reject
            },
        };
        let inner: Box<dyn IwssimInner + Send> = match resolved {
            crate::ResolvedMode::Full => {
                Box::new(Iwssim::<R>::with_config(client, width, height, cfg)?)
            }
            crate::ResolvedMode::Strip { h_body } => {
                Box::new(Iwssim::<R>::new_strip(client, width, height, h_body)?)
            }
        };
        Ok(Self { inner, backend })
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Number of pyramid scales (constant `5`).
    pub fn n_scales(&self) -> usize {
        NUM_SCALES
    }

    /// Score one reference / distorted pair, both packed sRGB
    /// `width × height × 3`.
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        self.inner.compute_srgb_u8(ref_rgb, dis_rgb)
    }

    /// Score one gray-f32 pair (`width × height` samples each, 0..255
    /// scale — the range the sRGB path's BT.601 conversion produces).
    /// This is the ingress for **float PU(luma)** HDR feeding: PU21
    /// quantized to u8 costs IW-SSIM ~0.18 SROCC on UPIQ HDR vs feeding
    /// the same values as floats (`benchmarks/pu_integrated_upiq_2026-06-09.md`,
    /// imazen/zenmetrics#25). Works in both whole-image and strip modes.
    pub fn compute_gray_f32(&mut self, ref_gray: &[f32], dis_gray: &[f32]) -> Result<Score> {
        self.inner.compute_gray_f32(ref_gray, dis_gray)
    }

    /// Score from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_srgb_u8(&ref_buf, &dis_buf)
    }

    /// Score against pre-uploaded packed-u32 device handles —
    /// upload-once Phase 4 entry point. See per-crate typed
    /// `Iwssim::compute_handles` for the layout contract.
    #[cfg(feature = "cubecl-types")]
    pub fn compute_handles(
        &mut self,
        ref_handle: &cubecl::server::Handle,
        dis_handle: &cubecl::server::Handle,
    ) -> Result<Score> {
        self.inner.compute_handles(ref_handle, dis_handle)
    }

    /// Cache the reference image's IW-SSIM state on device. Subsequent
    /// [`Self::compute_with_reference_srgb_u8`] calls skip the
    /// ref-side pyramid build + per-scale C_u eigendecomposition.
    ///
    /// Dispatches to `set_rgb_reference_stripped` when constructed in
    /// [`crate::MemoryMode::Strip`], else converts to gray host-side
    /// (BT.601 rounded) and calls the whole-image `set_reference`.
    ///
    /// # Errors
    ///
    /// - [`crate::Error::DimensionMismatch`] when
    ///   `ref_rgb.len() != width * height * 3`.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        self.inner.set_reference_srgb_u8(ref_rgb)
    }

    /// Score a distorted candidate against the cached reference set by
    /// [`Self::set_reference_srgb_u8`]. Returns
    /// [`crate::Error::NoCachedReference`] if no reference has been
    /// cached.
    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        self.inner.compute_with_reference_srgb_u8(dis_rgb)
    }

    /// Drop cached reference state. Subsequent
    /// [`Self::compute_with_reference_srgb_u8`] calls return
    /// [`crate::Error::NoCachedReference`] until
    /// [`Self::set_reference_srgb_u8`] is called again.
    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    /// `true` if a reference has been cached and
    /// [`Self::compute_with_reference_srgb_u8`] can be called.
    pub fn has_reference(&self) -> bool {
        self.inner.has_reference()
    }

    /// Pack a `width × height × 3` sRGB-u8 buffer into the packed-u32
    /// device handle layout that [`Self::compute_handles`] expects.
    #[cfg(feature = "cubecl-types")]
    pub fn pack_srgb_into_packed_u32_handle(&self, srgb: &[u8]) -> Result<cubecl::server::Handle> {
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
    zenmetrics_gpu_core::convert_to_srgb_rgb8(s, target).map_err(|_| Error::DimensionMismatch {
        expected: (expected_w as usize) * (expected_h as usize) * 3,
        got: (s.width() as usize) * (s.rows() as usize) * 3,
    })
}
