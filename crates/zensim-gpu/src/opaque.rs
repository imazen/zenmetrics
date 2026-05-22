//! Uniform opaque API for `zensim-gpu`.
//!
//! Note: zensim is a *feature extractor*, not a single-scalar metric
//! like dssim or ssim2. The opaque shim therefore exposes BOTH
//! shapes:
//!
//! - `compute_features_srgb_u8` / `compute_features_pixels` — return
//!   the regime-appropriate feature vector directly (228 / 300 / 372
//!   floats depending on [`ZensimParams::regime`]). This is the
//!   "natural" zensim API; the score lookup needs trained weights
//!   that don't live in this crate.
//! - `compute_srgb_u8` / `compute_pixels` — uniform with the other
//!   metric crates' opaque API. Apply
//!   [`crate::score_from_features`] to the **first 228 slots** of the
//!   feature vector using the caller-provided weights in
//!   [`ZensimParams::weights`]. If weights are `None`, returns
//!   `Score { value: f64::NAN, .. }` so callers notice they forgot
//!   to wire weights (no silent zero-score shipping).
//!
//! See `dssim-gpu/src/opaque.rs` for the full Phase 2 design.

use crate::pipeline::Zensim;
#[cfg(feature = "pixels")]
use crate::Error;
use crate::{
    Result, TOTAL_FEATURES, ZensimFeatureRegime, score_from_features,
    weights::WEIGHTS_PREVIEW_V0_2,
};

#[cfg(feature = "pixels")]
use zenpixels::PixelSlice;

/// Selects the GPU/CPU backend the opaque shim dispatches to.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend.
    #[cfg(feature = "cuda")]
    Cuda,
    /// WGPU backend.
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU reference backend.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The numeric score. For zensim, this is the result of
    /// `score_from_features(features, weights)` when
    /// [`ZensimParams::weights`] is `Some`; otherwise `f64::NAN`.
    pub value: f64,
    /// Short metric identifier (`"zensim"`).
    pub metric_name: &'static str,
    /// Implementation version tag.
    pub metric_version: &'static str,
}

/// Configuration for [`ZensimOpaque`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct ZensimParams {
    /// Optional 228-element trained weights for converting the
    /// feature vector into a scalar score. Pass
    /// `zensim::profile::WEIGHTS_PREVIEW_V0_2` (or a similar profile
    /// from the `zensim` crate) to land scores comparable to CPU
    /// zensim's `PreviewV0_2` profile. `None` => uniform
    /// `compute_*` methods return `Score { value: NaN, .. }`.
    ///
    /// Weights are 228 entries regardless of [`Self::regime`] — the
    /// scoring inner product runs over the basic block only.
    pub weights: Option<Box<[f64; TOTAL_FEATURES]>>,
    /// Feature regime: which slot map the underlying `Zensim` pipeline
    /// computes. Default is [`ZensimFeatureRegime::Basic`] (228) for
    /// backwards compatibility. Set to [`ZensimFeatureRegime::Extended`]
    /// (300) or [`ZensimFeatureRegime::WithIw`] (372) to also fill the
    /// masked / IW blocks — needed for picker training data and the
    /// v26+ sweep schema.
    pub regime: ZensimFeatureRegime,
}

impl Default for ZensimParams {
    fn default() -> Self {
        Self::new()
    }
}

impl ZensimParams {
    /// Default parameter bundle (no weights — uniform `compute_*`
    /// methods return NaN, [`ZensimFeatureRegime::Basic`] regime).
    /// Use [`Self::with_weights`] to wire a custom trained profile,
    /// or [`Self::default_weights`] / [`Self::with_canonical_v0_2`]
    /// to get the canonical `WEIGHTS_PREVIEW_V0_2` baked in.
    pub fn new() -> Self {
        Self {
            weights: None,
            regime: ZensimFeatureRegime::Basic,
        }
    }

    /// Bundle the canonical 228-element default weights
    /// ([`crate::WEIGHTS_PREVIEW_V0_2`]) — same constants the CPU
    /// `zensim` crate ships as the stable basic-regime default.
    /// Returns finite scores from [`ZensimOpaque::compute_srgb_u8`]
    /// / [`ZensimOpaque::compute_pixels`] without further wiring.
    ///
    /// This is what the umbrella's `MetricParams::default_for(Zensim)`
    /// returns so the metric is usable out of the box.
    pub fn default_weights() -> Self {
        Self::with_canonical_v0_2()
    }

    /// Explicit version-tagged variant of [`Self::default_weights`] —
    /// loads [`crate::WEIGHTS_PREVIEW_V0_2`]. Reach for this when the
    /// version tag matters to the caller (e.g., audit logs, sweep
    /// metadata).
    pub fn with_canonical_v0_2() -> Self {
        Self {
            weights: Some(Box::new(WEIGHTS_PREVIEW_V0_2)),
            regime: ZensimFeatureRegime::Basic,
        }
    }

    /// Attach a trained weight vector.
    pub fn with_weights(mut self, weights: [f64; TOTAL_FEATURES]) -> Self {
        self.weights = Some(Box::new(weights));
        self
    }

    /// Set the feature regime: [`ZensimFeatureRegime::Basic`] (228, the
    /// default), [`ZensimFeatureRegime::Extended`] (300, adds 72 masked
    /// features), or [`ZensimFeatureRegime::WithIw`] (372, adds 72
    /// information-weighted features on top of Extended).
    ///
    /// **Memory cost on Extended / WithIw**: ~600 MB at 12 MP for the
    /// per-scale persist planes. See
    /// [`crate::Zensim::new_with_regime_budget`] for the budget gate.
    pub fn with_regime(mut self, regime: ZensimFeatureRegime) -> Self {
        self.regime = regime;
        self
    }
}

trait ZensimInner: Send {
    fn compute_features(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]>;
    /// Regime-appropriate feature vector — length matches
    /// `regime.total_features()` (228 / 300 / 372).
    fn compute_features_vec(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>>;
    /// Set + upload + pre-build the reference's XYB pyramid.
    /// Subsequent [`Self::compute_with_reference_vec`] calls skip the
    /// ref upload + ref-pyramid construction entirely. Critical for
    /// sweep workloads with ~80 distortions per reference.
    fn set_reference(&mut self, ref_rgb: &[u8]) -> Result<()>;
    /// Compute features against the cached reference. Returns
    /// [`crate::Error::NoCachedReference`] if [`Self::set_reference`]
    /// was never called.
    fn compute_with_reference_vec(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>>;
    fn dims(&self) -> (u32, u32);
}

impl<R> ZensimInner for Zensim<R>
where
    R: cubecl::Runtime,
    Self: Send,
{
    fn compute_features(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        Zensim::compute_features(self, ref_rgb, dis_rgb)
    }

    fn compute_features_vec(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>> {
        Zensim::compute_features_vec(self, ref_rgb, dis_rgb)
    }

    fn set_reference(&mut self, ref_rgb: &[u8]) -> Result<()> {
        Zensim::set_reference(self, ref_rgb)
    }

    fn compute_with_reference_vec(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>> {
        Zensim::compute_with_reference_vec(self, dis_rgb)
    }

    fn dims(&self) -> (u32, u32) {
        Zensim::dimensions(self)
    }
}

/// Opaque zensim scorer / feature extractor.
pub struct ZensimOpaque {
    inner: Box<dyn ZensimInner + Send>,
    params: ZensimParams,
    #[allow(dead_code)]
    backend: Backend,
}

impl ZensimOpaque {
    /// Construct an opaque zensim instance. Reads
    /// [`ZensimParams::regime`] to choose the pipeline regime —
    /// Basic (228, default) / Extended (300) / WithIw (372).
    /// Equivalent to `new_with_memory_mode(.., MemoryMode::Auto)`.
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: ZensimParams,
    ) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque zensim instance with an explicit
    /// [`MemoryMode`](crate::MemoryMode). zensim-gpu has no Strip
    /// implementation — `MemoryMode::Strip` / `Tile` return
    /// [`crate::Error::ModeUnsupported`].
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: ZensimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        // Route through the typed `new_with_memory_mode` to surface
        // Mode errors before regime allocation.
        let cap = crate::memory_mode::vram_cap_bytes();
        match mode {
            crate::MemoryMode::Strip { .. } => {
                return Err(crate::Error::ModeUnsupported("Strip"));
            }
            crate::MemoryMode::Tile { .. } => return Err(crate::Error::ModeUnsupported("Tile")),
            crate::MemoryMode::Full => {} // pass through
            crate::MemoryMode::Auto => {
                // Validate via resolve_auto so a too-tight cap surfaces
                // TooBigForFull before regime allocation runs.
                let _ = crate::memory_mode::resolve_auto(width, height, cap)?;
            }
        }
        let regime = params.regime;
        let inner: Box<dyn ZensimInner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cuda::CudaRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::wgpu::WgpuRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cpu::CpuRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
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

    /// Configured feature regime (set at construction time via
    /// [`ZensimParams::with_regime`]).
    pub fn regime(&self) -> ZensimFeatureRegime {
        self.params.regime
    }

    /// Compute the **first 228** features for one pair from packed
    /// sRGB regardless of regime. Truncates the Extended / WithIw
    /// output to the basic block — same behaviour as the legacy
    /// pre-regime API.
    pub fn compute_features_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        self.inner.compute_features(ref_rgb, dis_rgb)
    }

    /// Compute the **first 228** features from [`PixelSlice`] inputs.
    /// Truncates the Extended / WithIw output to the basic block.
    #[cfg(feature = "pixels")]
    pub fn compute_features_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<[f64; TOTAL_FEATURES]> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_features(&ref_buf, &dis_buf)
    }

    /// Compute the regime-appropriate feature vector for one pair from
    /// packed sRGB. Length matches [`Self::regime`]:
    ///
    /// - 228 floats on [`ZensimFeatureRegime::Basic`]
    /// - 300 floats on [`ZensimFeatureRegime::Extended`]
    /// - 372 floats on [`ZensimFeatureRegime::WithIw`]
    ///
    /// Use this entry point when the caller needs the full extended /
    /// IW feature block (picker training, v26+ sweep schema). The
    /// fixed-length [`Self::compute_features_srgb_u8`] is kept for
    /// backwards compatibility but only returns the basic block.
    pub fn compute_features_vec_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>> {
        self.inner.compute_features_vec(ref_rgb, dis_rgb)
    }

    /// Compute the regime-appropriate feature vector from
    /// [`PixelSlice`] inputs. See
    /// [`Self::compute_features_vec_srgb_u8`] for the regime → length
    /// table.
    #[cfg(feature = "pixels")]
    pub fn compute_features_vec_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Vec<f64>> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_features_vec(&ref_buf, &dis_buf)
    }

    /// Upload + pyramid-build the reference image ONCE, then call
    /// [`Self::compute_with_reference_srgb_u8`] for each distortion.
    /// Critical for sweep workloads: a single reference with N
    /// distortions saves N-1 ref uploads (~1 MB each at 1 MP) and
    /// N-1 ref-pyramid kernel launches.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        self.inner.set_reference(ref_rgb)
    }

    /// Compute features against the cached reference. Returns
    /// `Vec<f64>` of length `params.regime.total_features()` (228 /
    /// 300 / 372). Returns [`crate::Error::NoCachedReference`] if
    /// [`Self::set_reference_srgb_u8`] was never called.
    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>> {
        self.inner.compute_with_reference_vec(dis_rgb)
    }

    /// Set + upload + pre-build the reference's XYB pyramid from
    /// [`PixelSlice`] input. Companion to
    /// [`Self::compute_with_reference_pixels`].
    #[cfg(feature = "pixels")]
    pub fn set_reference_pixels(&mut self, r: PixelSlice<'_>) -> Result<()> {
        let (w, h) = self.inner.dims();
        let ref_buf = to_srgb_rgb8(&r, w, h)?;
        self.inner.set_reference(&ref_buf)
    }

    /// Compute features against the cached reference from a
    /// [`PixelSlice`] distortion. See
    /// [`Self::compute_with_reference_srgb_u8`] for semantics.
    #[cfg(feature = "pixels")]
    pub fn compute_with_reference_pixels(&mut self, d: PixelSlice<'_>) -> Result<Vec<f64>> {
        let (w, h) = self.inner.dims();
        let dis_buf = to_srgb_rgb8(&d, w, h)?;
        self.inner.compute_with_reference_vec(&dis_buf)
    }

    /// Compute the uniform [`Score`] from packed sRGB. Returns
    /// `Score { value: NAN, .. }` if no weights are wired in
    /// [`ZensimParams::weights`].
    pub fn compute_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Score> {
        let features = self.inner.compute_features(ref_rgb, dis_rgb)?;
        Ok(self.score_from(features))
    }

    /// Compute the uniform [`Score`] from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<Score> {
        let features = self.compute_features_pixels(r, d)?;
        Ok(self.score_from(features))
    }

    fn score_from(&self, features: [f64; TOTAL_FEATURES]) -> Score {
        let value = match &self.params.weights {
            Some(w) => score_from_features(&features, w.as_ref()),
            None => f64::NAN,
        };
        Score {
            value,
            metric_name: "zensim",
            metric_version: env!("CARGO_PKG_VERSION"),
        }
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
