//! Uniform opaque API for `zensim-gpu`.
//!
//! Note: zensim is a *feature extractor*, not a single-scalar metric
//! like dssim or ssim2. The opaque shim therefore exposes BOTH
//! shapes:
//!
//! - `compute_features_srgb_u8` / `compute_features_pixels` — return
//!   the 228-element feature vector directly. This is the "natural"
//!   zensim API; the score lookup needs trained weights that don't
//!   live in this crate.
//! - `compute_srgb_u8` / `compute_pixels` — uniform with the other
//!   metric crates' opaque API. Apply
//!   [`crate::score_from_features`] to the feature vector using the
//!   caller-provided weights in [`ZensimParams::weights`]. If weights
//!   are `None`, returns `Score { value: f64::NAN, .. }` so callers
//!   notice they forgot to wire weights (no silent zero-score
//!   shipping).
//!
//! See `dssim-gpu/src/opaque.rs` for the full Phase 2 design.

use crate::pipeline::Zensim;
#[cfg(feature = "pixels")]
use crate::Error;
use crate::{Result, TOTAL_FEATURES, score_from_features, weights::WEIGHTS_PREVIEW_V0_2};

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
///
/// `acumen_mode_a`: when `Some`, the pipeline runs the castleCSF
/// LUT lookup per-image at the given viewing condition and applies
/// per-(scale, channel) weights to the HF band-energy features
/// before they reach the trained MLP. Used for Gate A in
/// `imazen/zensim#40`. Default `None` → legacy V_22-shipped path
/// is byte-identical to pre-acumen behavior.
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct ZensimParams {
    /// Optional 228-element trained weights for converting the
    /// feature vector into a scalar score. Pass
    /// `zensim::profile::WEIGHTS_PREVIEW_V0_2` (or a similar profile
    /// from the `zensim` crate) to land scores comparable to CPU
    /// zensim's `PreviewV0_2` profile. `None` => uniform
    /// `compute_*` methods return `Score { value: NaN, .. }`.
    pub weights: Option<Box<[f64; TOTAL_FEATURES]>>,

    /// Acumen Mode A: when set, the pipeline computes per-image
    /// castleCSF weights at this viewing condition and applies them
    /// based on [`Self::acumen_arch`]. Default `None` → no weighting,
    /// V_22-shipped path bit-stable. See `zensim::acumen` and tracking
    /// issue `imazen/zensim#40` Gate A.
    pub acumen_mode_a: Option<zensim::acumen::viewing::ViewingCondition>,

    /// How acumen weights are applied. Only consulted when
    /// `acumen_mode_a` is `Some`. Default = HfPost (the original
    /// Mode A wiring — multiply HF band-energy slots 10/11/12 per
    /// scale-channel by per-(scale, channel) castleCSF weight).
    ///
    /// Alternatives investigated for Gate A architectural ablation:
    /// - `WideModulation`: scale ALL 19 features per scale-channel
    ///   by the per-band weight (broader application).
    /// - `AuxFeatures`: leave features unchanged; expose the 12
    ///   per-(channel, scale) weights as a separate getter for the
    ///   caller to append as auxiliary feature columns. The MLP
    ///   then learns to use CSF weights as CONTEXT instead of as
    ///   a multiplicative modulation.
    pub acumen_arch: AcumenArch,

    /// Feature regime to emit. Default `Basic` (228 features, the
    /// V_22-shipped output shape). `Extended` adds 72 masked
    /// features (300 total); `WithIw` adds 72 IW-pool features
    /// on top (372 total, matches CPU `Zensim::compute_372col`).
    pub regime: crate::ZensimFeatureRegime,
}

/// Acumen Mode A application strategy. See [`ZensimParams::acumen_arch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AcumenArch {
    /// Multiply basic feature slots 10/11/12 per (scale, channel)
    /// by per-(scale, channel) castleCSF weight. The original Mode
    /// A wiring; preserved as the default for backward compat.
    #[default]
    HfPost,
    /// Multiply all 19 basic+peaks features per (scale, channel)
    /// by per-(scale, channel) castleCSF weight. Broader
    /// application — every band-relative statistic gets weighted.
    WideModulation,
    /// Leave features unchanged. Caller fetches
    /// [`crate::ZensimOpaque::acumen_band_weights_flat`] separately
    /// and appends to its training/inference vector as auxiliary
    /// features. The MLP learns to USE the CSF weights as input
    /// signal rather than have them applied as a multiplier.
    AuxFeatures,
    /// Mode B (per-pixel L_adapt). Pre-multiply the input RGB by
    /// a per-pixel achromatic CSF weight derived from each pixel's
    /// local adapted luminance (Gaussian-blurred luma). Introduces
    /// spatial CSF variation that Mode A's per-image-mean L can
    /// never capture. Approximation of the full castleCSF Mode B —
    /// applies the same scalar weight uniformly across the 3 color
    /// channels and 4 pyramid scales at each pixel. The full
    /// per-band-per-pixel version requires kernel modification and
    /// is deferred until/unless this approximation shows signal.
    ///
    /// Implementation: caller of `compute_features` is responsible
    /// for pre-multiplying the image bytes before this codepath
    /// runs. The kernel itself is unmodified; the `compute_features`
    /// entry point sees pre-weighted RGB.
    ModeB,
    /// Full per-band per-pixel Mode B. Computes 12 weight maps
    /// (3 channels × 4 pyramid scales), each per-pixel from the
    /// channel's CSF response at the pixel's local-adapted
    /// luminance and the scale's spatial frequency ρ_s. Applies
    /// element-wise to the pyramid's src/dst arrays via a small
    /// multiply kernel BEFORE the existing fused features kernel
    /// runs. This is the architecturally faithful castleCSF Mode B.
    ModeBPerBand,
}

// Manual Debug impl because `Option<ViewingCondition>` doesn't
// participate in the derive's auto layout when the field type
// comes from a path-dep crate that may not impl Debug uniformly.
impl core::fmt::Debug for ZensimParams {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ZensimParams")
            .field("weights", &self.weights.as_ref().map(|_| "<228 f64>"))
            .field("acumen_mode_a", &self.acumen_mode_a)
            .finish()
    }
}

impl ZensimParams {
    /// Default parameter bundle (no weights — uniform `compute_*`
    /// methods return NaN). Use [`Self::with_weights`] to wire a
    /// custom trained profile, or [`Self::default_weights`] /
    /// [`Self::with_canonical_v0_2`] to get the canonical
    /// `WEIGHTS_PREVIEW_V0_2` baked in.
    pub fn new() -> Self {
        Self {
            weights: None,
            acumen_mode_a: None,
            acumen_arch: AcumenArch::default(),
            regime: crate::ZensimFeatureRegime::Basic,
        }
    }

    /// Set the feature regime emitted. Default `Basic` (228 cols).
    /// For production recipe parity with the CPU 372-col extractor,
    /// pass [`crate::ZensimFeatureRegime::WithIw`].
    pub fn with_regime(mut self, regime: crate::ZensimFeatureRegime) -> Self {
        self.regime = regime;
        self
    }

    /// Set the acumen application strategy. Only meaningful when
    /// [`Self::acumen_mode_a`] is `Some`. Default `HfPost` matches
    /// the original Mode A wiring; pass `WideModulation` or
    /// `AuxFeatures` for the architectural ablation variants.
    pub fn with_acumen_arch(mut self, arch: AcumenArch) -> Self {
        self.acumen_arch = arch;
        self
    }

    /// Enable acumen Mode A at the given viewing condition. Per-image
    /// castleCSF weights are computed at feature-extract time and
    /// applied to HF band-energy features. Used for Gate A training
    /// and inference paths in `imazen/zensim#40`.
    pub fn with_acumen_mode_a(
        mut self,
        viewing: zensim::acumen::viewing::ViewingCondition,
    ) -> Self {
        self.acumen_mode_a = Some(viewing);
        self
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
            acumen_mode_a: None,
            acumen_arch: AcumenArch::default(),
            regime: crate::ZensimFeatureRegime::Basic,
        }
    }

    /// Attach a trained weight vector.
    pub fn with_weights(mut self, weights: [f64; TOTAL_FEATURES]) -> Self {
        self.weights = Some(Box::new(weights));
        self
    }
}

trait ZensimInner: Send {
    fn compute_features(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]>;
    /// Regime-aware variant: returns a `Vec<f64>` of length
    /// `regime.total_features()` (228 / 300 / 372). Lets callers
    /// who configured `with_regime(WithIw)` see all 372 features
    /// instead of being silently truncated to 228.
    fn compute_features_vec_inner(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>>;
    /// Set + upload + pre-build the reference's XYB pyramid.
    /// Subsequent `compute_with_reference_vec_inner` calls skip the
    /// ref upload + ref-pyramid construction entirely. Critical
    /// optimization for sweep workloads with ~80 distortions per ref.
    fn set_reference_inner(&mut self, ref_rgb: &[u8]) -> Result<()>;
    /// Compute features against the cached reference. Must call
    /// `set_reference_inner` first or get [`Error::NoCachedReference`].
    fn compute_with_reference_vec_inner(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>>;
    fn dims(&self) -> (u32, u32);
    fn set_acumen_viewing(
        &mut self,
        viewing: Option<zensim::acumen::viewing::ViewingCondition>,
    );
    fn set_acumen_arch(&mut self, arch: AcumenArch);
    /// Returns the 12 cached per-(channel, scale) castleCSF weights
    /// from the most recent reference, flattened in
    /// `[channel * SCALES + scale]` order. `None` if no reference
    /// has been set or acumen is disabled.
    fn acumen_band_weights_flat(&self) -> Option<[f32; 12]>;
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

    fn compute_features_vec_inner(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>> {
        Zensim::compute_features_vec(self, ref_rgb, dis_rgb)
    }

    fn set_reference_inner(&mut self, ref_rgb: &[u8]) -> Result<()> {
        Zensim::set_reference(self, ref_rgb)
    }

    fn compute_with_reference_vec_inner(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>> {
        Zensim::compute_with_reference_vec(self, dis_rgb)
    }

    fn dims(&self) -> (u32, u32) {
        Zensim::dimensions(self)
    }

    fn set_acumen_viewing(
        &mut self,
        viewing: Option<zensim::acumen::viewing::ViewingCondition>,
    ) {
        Zensim::set_acumen_viewing(self, viewing);
    }

    fn set_acumen_arch(&mut self, arch: AcumenArch) {
        Zensim::set_acumen_arch(self, arch);
    }

    fn acumen_band_weights_flat(&self) -> Option<[f32; 12]> {
        Zensim::acumen_band_weights_flat(self)
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
    /// Construct an opaque zensim instance.
    pub fn new(
        backend: Backend,
        width: u32,
        height: u32,
        params: ZensimParams,
    ) -> Result<Self> {
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
        let mut shim = Self {
            inner,
            params,
            backend,
        };
        // Push acumen viewing + arch through to the inner pipeline
        // so subsequent `set_reference` calls compute per-image
        // castleCSF band weights and Phase 4 dispatches correctly.
        let viewing = shim.params.acumen_mode_a;
        let arch = shim.params.acumen_arch;
        shim.inner.set_acumen_viewing(viewing);
        shim.inner.set_acumen_arch(arch);
        Ok(shim)
    }

    /// Expose the cached per-(channel, scale) castleCSF weights.
    /// Returns `None` if no reference has been set or acumen is
    /// disabled. Used by `AuxFeatures` arch callers to append the
    /// 12 weights as additional feature columns.
    pub fn acumen_band_weights_flat(&self) -> Option<[f32; 12]> {
        self.inner.acumen_band_weights_flat()
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Compute the 228-feature vector for one pair from packed sRGB.
    /// Truncates to 228 even when `params.regime` is Extended / WithIw;
    /// use [`Self::compute_features_vec`] for the regime-aware variant.
    pub fn compute_features_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        self.inner.compute_features(ref_rgb, dis_rgb)
    }

    /// Regime-aware feature vector: returns `Vec<f64>` of length
    /// `params.regime.total_features()` (228 / 300 / 372). Use this
    /// when `params.regime` is `Extended` or `WithIw`.
    pub fn compute_features_vec(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<Vec<f64>> {
        self.inner.compute_features_vec_inner(ref_rgb, dis_rgb)
    }

    /// Upload + pyramid-build the reference image ONCE, then call
    /// [`Self::compute_with_reference_vec`] for each distortion.
    /// Critical for sweep workloads: a single reference with N
    /// distortions saves N-1 ref uploads (~1 MB each at 1 MP) and
    /// N-1 ref-pyramid kernel launches.
    pub fn set_reference(&mut self, ref_rgb: &[u8]) -> Result<()> {
        self.inner.set_reference_inner(ref_rgb)
    }

    /// Compute features against the cached reference. Returns
    /// `Vec<f64>` of length `params.regime.total_features()`.
    /// Returns [`Error::NoCachedReference`] if `set_reference` was
    /// never called.
    pub fn compute_with_reference_vec(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>> {
        self.inner.compute_with_reference_vec_inner(dis_rgb)
    }

    /// Compute the 228-feature vector from [`PixelSlice`] inputs.
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
