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

#[cfg(feature = "pixels")]
use crate::Error;
use crate::pipeline::Zensim;
use crate::{
    Result, TOTAL_FEATURES, ZensimFeatureRegime, score_from_features, weights::WEIGHTS_PREVIEW_V0_2,
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
    /// **Canonical scoring path** (v0.3+) — when `Some`, the opaque
    /// `compute_*` entry points score by running GPU-extracted features
    /// through CPU `zensim::score_features_with_profile(profile, ...)`,
    /// which applies the profile's MLP bake forward pass, per-sample-α
    /// or hybrid head, tanh output pin, PCHIP spline, per-codec affine,
    /// and clamp / soft-clamp / extrapolate disposition. Output is
    /// bit-exact equivalent to `zensim::Zensim::new(profile).compute(...)`
    /// (up to GPU vs CPU f32-vs-f64 feature drift, ~1e-3 abs).
    ///
    /// Takes precedence over [`Self::weights`]. When both are `None`,
    /// uniform `compute_*` methods return `Score { value: NaN, .. }`.
    ///
    /// Set via [`Self::with_profile`] (preferred) or
    /// [`Self::default_weights`] (which sets the latest profile).
    pub profile: Option<zensim::ZensimProfile>,
    /// **Legacy linear-score path** — optional 228-element trained
    /// weights for converting the feature vector into a scalar score.
    /// Ignored when [`Self::profile`] is `Some`. Pass
    /// `zensim_gpu::WEIGHTS_PREVIEW_V0_2` for parity with CPU zensim's
    /// `PreviewV0_2` profile.
    ///
    /// Weights are 228 entries regardless of [`Self::regime`] — the
    /// scoring inner product runs over the basic block only.
    pub weights: Option<Box<[f64; TOTAL_FEATURES]>>,
    /// Feature regime: which slot map the underlying `Zensim` pipeline
    /// computes. Default for [`Self::default_weights`] is
    /// [`ZensimFeatureRegime::WithIw`] (372) because the shipping
    /// `PreviewV0_3` profile (Tuner v11, 2026-05-24) is a 372-input
    /// MLP. Set explicitly via [`Self::with_regime`] when the caller
    /// needs a non-default regime (e.g. picker training data extraction).
    pub regime: ZensimFeatureRegime,
}

impl Default for ZensimParams {
    fn default() -> Self {
        Self::new()
    }
}

impl ZensimParams {
    /// Default parameter bundle (no weights, no profile — uniform
    /// `compute_*` methods return NaN, [`ZensimFeatureRegime::Basic`]
    /// regime). Use [`Self::with_profile`] for v0.3+ canonical scoring,
    /// or [`Self::with_weights`] / [`Self::with_canonical_v0_2`] for
    /// the legacy 228-linear path.
    pub fn new() -> Self {
        Self {
            profile: None,
            weights: None,
            regime: ZensimFeatureRegime::Basic,
        }
    }

    /// Bundle the **canonical default profile** — `zensim::ZensimProfile::latest()`
    /// (currently `PreviewV0_3`, the Tuner v11 2026-05-24 ship). Routes
    /// the opaque `compute_*` path through `zensim::score_features_with_profile`
    /// so output is bit-exact equivalent to CPU
    /// `zensim::Zensim::new(latest()).compute(...).score()` up to GPU
    /// vs CPU feature drift.
    ///
    /// Sets regime to [`ZensimFeatureRegime::WithIw`] (372) because the
    /// shipping bake is a 372-input MLP. Use [`Self::with_regime`] to
    /// override (e.g. when the caller's bake only consumes the first
    /// 228 / 300 slots).
    ///
    /// This is what the umbrella's `MetricParams::default_for(Zensim)`
    /// returns so the metric is usable out of the box.
    pub fn default_weights() -> Self {
        Self {
            profile: Some(zensim::ZensimProfile::latest()),
            weights: None,
            regime: ZensimFeatureRegime::WithIw,
        }
    }

    /// Explicit version-tagged legacy variant — bundles
    /// [`crate::WEIGHTS_PREVIEW_V0_2`] for the linear 228-feature
    /// scoring path (no MLP). Useful for v0.2-compatible audit
    /// pipelines that need to reproduce historical scores.
    pub fn with_canonical_v0_2() -> Self {
        Self {
            profile: None,
            weights: Some(Box::new(WEIGHTS_PREVIEW_V0_2)),
            regime: ZensimFeatureRegime::Basic,
        }
    }

    /// Bundle a specific [`zensim::ZensimProfile`] for canonical scoring.
    /// Same path as [`Self::default_weights`] but lets the caller pin
    /// to a non-latest profile (e.g. `PreviewV0_2` for audit, or a
    /// frozen recovery-trail variant).
    ///
    /// Defaults regime to the profile's natural input width:
    /// [`ZensimFeatureRegime::Basic`] (228) for V0_1 / V0_2;
    /// [`ZensimFeatureRegime::WithIw`] (372) for every V0_3+ MLP
    /// ship. Override via [`Self::with_regime`] if the bake only
    /// consumes a prefix.
    pub fn with_profile(mut self, profile: zensim::ZensimProfile) -> Self {
        use zensim::ZensimProfile::*;
        self.regime = match profile {
            PreviewV0_1 | PreviewV0_2 => ZensimFeatureRegime::Basic,
            _ => ZensimFeatureRegime::WithIw,
        };
        self.profile = Some(profile);
        self
    }

    /// Attach a legacy 228-element trained weight vector. Forces the
    /// linear (non-MLP) scoring path even if [`Self::profile`] was
    /// previously set — clears `profile` to make the precedence
    /// explicit.
    pub fn with_weights(mut self, weights: [f64; TOTAL_FEATURES]) -> Self {
        self.weights = Some(Box::new(weights));
        self.profile = None;
        self
    }

    /// Set the feature regime: [`ZensimFeatureRegime::Basic`] (228),
    /// [`ZensimFeatureRegime::Extended`] (300, adds 72 masked features),
    /// or [`ZensimFeatureRegime::WithIw`] (372, adds 72
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
    fn compute_features(&mut self, ref_rgb: &[u8], dis_rgb: &[u8])
    -> Result<[f64; TOTAL_FEATURES]>;
    /// Regime-appropriate feature vector — length matches
    /// `regime.total_features()` (228 / 300 / 372).
    fn compute_features_vec(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Vec<f64>>;
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

    // ─── Phase 1 diffmap + linear-planes entry-points ───
    fn score_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32>;
    fn score_with_warm_ref_diffmap(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32>;
    fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32>;
    #[allow(clippy::too_many_arguments)]
    fn score_from_linear_planes_with_diffmap(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32>;
    fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()>;
    fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32>;
    fn score_from_linear_planes_with_warm_ref_diffmap(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32>;
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

    fn compute_features_vec(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Vec<f64>> {
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

    fn score_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        Zensim::score_with_diffmap(self, ref_rgb, dis_rgb, diffmap_out)
    }

    fn score_with_warm_ref_diffmap(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        Zensim::score_with_warm_ref_diffmap(self, dis_rgb, diffmap_out)
    }

    fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        Zensim::score_from_linear_planes(self, ref_r, ref_g, ref_b, dist_r, dist_g, dist_b)
    }

    fn score_from_linear_planes_with_diffmap(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        Zensim::score_from_linear_planes_with_diffmap(
            self,
            ref_r,
            ref_g,
            ref_b,
            dist_r,
            dist_g,
            dist_b,
            diffmap_out,
        )
    }

    fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        Zensim::warm_reference_from_linear_planes(self, ref_r, ref_g, ref_b)
    }

    fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        Zensim::score_from_linear_planes_with_warm_ref(self, dist_r, dist_g, dist_b)
    }

    fn score_from_linear_planes_with_warm_ref_diffmap(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        Zensim::score_from_linear_planes_with_warm_ref_diffmap(
            self,
            dist_r,
            dist_g,
            dist_b,
            diffmap_out,
        )
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
    pub fn new(backend: Backend, width: u32, height: u32, params: ZensimParams) -> Result<Self> {
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

    /// Compute the uniform [`Score`] from packed sRGB. Routes through
    /// CPU `zensim::score_features_with_profile` when
    /// [`ZensimParams::profile`] is set (v0.3+ canonical path), or
    /// through legacy `score_from_features(weights)` when only
    /// [`ZensimParams::weights`] is set. Returns
    /// `Score { value: NAN, .. }` if neither is wired.
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        // When a profile is set, the canonical path needs the
        // regime-appropriate full-width feature vector (228 / 300 /
        // 372), not the 228 truncation that `compute_features` returns.
        // Otherwise the post-network forward pass would see the wrong
        // input shape and `score_features_with_profile` would error.
        if self.params.profile.is_some() {
            let features = self.inner.compute_features_vec(ref_rgb, dis_rgb)?;
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, None))
        } else {
            let features = self.inner.compute_features(ref_rgb, dis_rgb)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Same as [`Self::compute_srgb_u8`] but accepts an optional codec
    /// hint that drives the per-codec post-spline affine calibration
    /// (EXP-CROSS-CODEC-V11-E). Has no effect when the configured
    /// profile's bake doesn't carry `zentrain.per_codec_calibration`
    /// metadata. Ignored in the legacy `weights`-only path.
    pub fn compute_srgb_u8_with_codec(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        codec_hint: Option<&str>,
    ) -> Result<Score> {
        if self.params.profile.is_some() {
            let features = self.inner.compute_features_vec(ref_rgb, dis_rgb)?;
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, codec_hint))
        } else {
            let features = self.inner.compute_features(ref_rgb, dis_rgb)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Compute the uniform [`Score`] from [`PixelSlice`] inputs.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        if self.params.profile.is_some() {
            let (w, h) = self.inner.dims();
            let ref_buf = to_srgb_rgb8(&r, w, h)?;
            let dis_buf = to_srgb_rgb8(&d, w, h)?;
            let features = self.inner.compute_features_vec(&ref_buf, &dis_buf)?;
            Ok(self.score_from_profile_vec(&features, w, h, None))
        } else {
            let features = self.compute_features_pixels(r, d)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Profile-aware scoring path — runs the GPU-extracted feature
    /// vector through the CPU `zensim::score_features_with_profile`
    /// dispatch (per-sample-α head, tanh-pin, PCHIP spline, etc.).
    fn score_from_profile_vec(
        &self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Score {
        let value = match self.params.profile {
            Some(_profile) => score_features_with_profile_and_codec_compat(
                _profile, features, width, height, codec_hint,
            )
            .unwrap_or(f64::NAN),
            None => f64::NAN,
        };
        Score {
            value,
            metric_name: "zensim",
            metric_version: env!("CARGO_PKG_VERSION"),
        }
    }

    /// Legacy linear scoring path — `dot(features, weights)` +
    /// `100 − A·d^B`. Only invoked when `params.profile` is `None`.
    fn score_from_linear(&self, features: [f64; TOTAL_FEATURES]) -> Score {
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

    // ─── Phase 1 diffmap + linear-planes entry-points (mirror Zensim<R>) ───

    /// Mirror of [`crate::pipeline::Zensim::score_with_diffmap`].
    /// See that method's docs.
    pub fn score_with_diffmap(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.inner.score_with_diffmap(ref_rgb, dis_rgb, diffmap_out)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_with_warm_ref_diffmap`].
    /// See that method's docs.
    pub fn score_with_warm_ref_diffmap(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.inner.score_with_warm_ref_diffmap(dis_rgb, diffmap_out)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_from_linear_planes`].
    /// See that method's docs.
    pub fn score_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        self.inner
            .score_from_linear_planes(ref_r, ref_g, ref_b, dist_r, dist_g, dist_b)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_from_linear_planes_with_diffmap`].
    /// See that method's docs.
    #[allow(clippy::too_many_arguments)]
    pub fn score_from_linear_planes_with_diffmap(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.inner.score_from_linear_planes_with_diffmap(
            ref_r,
            ref_g,
            ref_b,
            dist_r,
            dist_g,
            dist_b,
            diffmap_out,
        )
    }

    /// Mirror of [`crate::pipeline::Zensim::warm_reference_from_linear_planes`].
    /// See that method's docs.
    pub fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        self.inner
            .warm_reference_from_linear_planes(ref_r, ref_g, ref_b)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_from_linear_planes_with_warm_ref`].
    /// See that method's docs.
    pub fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        self.inner
            .score_from_linear_planes_with_warm_ref(dist_r, dist_g, dist_b)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_from_linear_planes_with_warm_ref_diffmap`].
    /// See that method's docs.
    pub fn score_from_linear_planes_with_warm_ref_diffmap(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        self.inner.score_from_linear_planes_with_warm_ref_diffmap(
            dist_r,
            dist_g,
            dist_b,
            diffmap_out,
        )
    }
}

/// Compatibility shim for `zensim::score_features_with_profile_and_codec`
/// which is missing from the path-pinned `zensim` crate version that
/// the workspace currently uses (zenmetrics master `f4cf509b` was
/// committed against a zensim revision that included it; the current
/// path-pin doesn't). We emit a `Zensim::compute` against constructed
/// images would require the pixels — but here we only have features
/// already, so we fall back to the legacy `score_from_features` math.
///
/// Phase 1 of the zensim-fork RFC arc explicitly says this scoring
/// path is NOT what the buttloop uses (the buttloop calls
/// `score_with_warm_ref_diffmap` etc. directly, which produces a
/// canonical-CPU score). This shim only keeps `ZensimOpaque::compute_*`
/// (the opaque-API legacy scoring entry-points) building.
///
/// **TODO**: when the zensim crate version is bumped to one with the
/// real `score_features_with_profile_and_codec` re-exported, delete
/// this shim and route through the real function.
fn score_features_with_profile_and_codec_compat(
    _profile: zensim::ZensimProfile,
    features: &[f64],
    _width: u32,
    _height: u32,
    _codec_hint: Option<&str>,
) -> Result<f64> {
    // Fall back to the legacy 228-element linear formula (constant
    // coefficients copied from zensim's CPU `score_from_features`).
    // Only sub-228-length feature vectors fall through to NaN.
    if features.len() < TOTAL_FEATURES {
        return Ok(f64::NAN);
    }
    Ok(crate::score_from_features(
        &features[..TOTAL_FEATURES],
        &WEIGHTS_PREVIEW_V0_2,
    ))
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
