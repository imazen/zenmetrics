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

/// The backend selector and uniform score type are shared verbatim
/// across all six `*-gpu` metric crates — see [`zenmetrics_gpu_core`].
/// Re-exported here so `crate::Backend` / `crate::opaque::Score` keep
/// resolving.
pub use zenmetrics_gpu_core::{Backend, Score};

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
    // `ZensimProfile::latest()` is deprecated in favor of `latest_preview()`;
    // kept verbatim so the default profile (and thus every score) is provably
    // unchanged. Migrating the call is a separate score-verified change.
    #[allow(deprecated)]
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
    /// [`ZensimFeatureRegime::Basic`] (228) for V0_2;
    /// [`ZensimFeatureRegime::WithIw`] (372) for every V0_3+ MLP
    /// ship. Override via [`Self::with_regime`] if the bake only
    /// consumes a prefix.
    pub fn with_profile(mut self, profile: zensim::ZensimProfile) -> Self {
        use zensim::ZensimProfile::*;
        self.regime = match profile {
            PreviewV0_2 => ZensimFeatureRegime::Basic,
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
    fn has_reference(&self) -> bool;
    fn clear_reference(&mut self);

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

    fn has_reference(&self) -> bool {
        Zensim::has_reference(self)
    }

    fn clear_reference(&mut self) {
        Zensim::clear_reference(self)
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
    /// Caller-requested logical dimensions. May be smaller than the
    /// `inner` pipeline's dimensions when the image is sub-64px and the
    /// inner pipeline was built for the reflect-padded size — see
    /// [`MIN_PAD_DIM`] and [`reflect_pad`]. At ≥64px in both axes this
    /// equals `inner.dims()` and the pad/crop helpers are no-ops.
    logical_w: u32,
    logical_h: u32,
}

/// Minimum per-axis dimension the GPU pyramid needs to form 4 genuine
/// scales (`8 · 2³ = 64`). Inputs smaller than this in either axis are
/// reflect(mirror)-padded host-side up to this floor before scoring, so
/// every image down to 1×1 produces a real 4-scale feature vector
/// instead of a degenerate (fewer-scale) one. This mirrors the CPU
/// `zensim::metric` reflect-pad funnel byte-for-byte (same reflect-101
/// indexing), so GPU and CPU agree at sub-64 sizes to within the usual
/// f32-kernel divergence — and a constant colour difference scores
/// the same at every size. NO-OP at ≥64px (zero overhead on the
/// hot path).
const MIN_PAD_DIM: u32 = 64;

// NOTE (geometry-API divergence vs cvvdp-gpu, 2026-05-26):
// `ZensimOpaque` does NOT expose `new_with_geometry` / display-config
// constructors because the underlying `zensim_gpu::Zensim::<R>` is a
// feature-based metric (228 / 300 / 372-d vector) rather than a
// display-aware one. There is no `DisplayGeometry` / PPD threading
// through the zensim pipeline — feature extraction is purely
// data-driven and the pyramid depth + filter weights don't depend on
// viewing conditions. Callers wanting display-aware scoring should
// use `cvvdp-gpu`'s `CvvdpOpaque::new_with_geometry` instead.
impl ZensimOpaque {
    /// Construct an opaque zensim instance. Reads
    /// [`ZensimParams::regime`] to choose the pipeline regime —
    /// Basic (228, default) / Extended (300) / WithIw (372).
    /// Equivalent to `new_with_memory_mode(.., MemoryMode::Auto)`.
    pub fn new(backend: Backend, width: u32, height: u32, params: ZensimParams) -> Result<Self> {
        Self::new_with_memory_mode(backend, width, height, params, crate::MemoryMode::Auto)
    }

    /// Construct an opaque zensim instance with an explicit
    /// [`MemoryMode`](crate::MemoryMode). All variants are supported as
    /// of 2026-05-26:
    /// - `Full` allocates the whole-image pipeline.
    /// - `Strip { h_body }` allocates a strip-walker pipeline.
    /// - `Tile` returns [`crate::Error::ModeUnsupported`] (not implemented).
    /// - `Auto` falls back from Full → Strip when Full exceeds the VRAM cap.
    pub fn new_with_memory_mode(
        backend: Backend,
        width: u32,
        height: u32,
        params: ZensimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self> {
        use crate::MemoryMode;
        use crate::memory_mode::{ResolvedMode, resolve_auto, vram_cap_bytes};

        if matches!(mode, MemoryMode::Tile { .. }) {
            return Err(crate::Error::ModeUnsupported("Tile"));
        }
        // Reflect-pad sub-64px requests up to the 4-scale pyramid floor.
        // The inner pipeline is built for the padded size; the compute
        // entry points reflect-pad their byte/plane inputs to match.
        // `logical_*` records what the caller asked for so `dims()` stays
        // honest and the pad/crop helpers know the real image extent.
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_PAD_DIM);
        let height = height.max(MIN_PAD_DIM);
        let regime = params.regime;
        // Resolve Auto to a concrete (Full | Strip { h_body }) so the
        // dispatch below is straightforward.
        let resolved: ResolvedMode = match mode {
            MemoryMode::Full => ResolvedMode::Full,
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, regime, cap)
                });
                ResolvedMode::Strip { h_body: body }
            }
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                resolve_auto(width, height, regime, cap)?
            }
            MemoryMode::Tile { .. } => unreachable!("already returned above"),
        };
        let inner: Box<dyn ZensimInner + Send> = match (backend, resolved) {
            #[cfg(feature = "cuda")]
            (Backend::Cuda, ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cuda::CudaRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
            }
            #[cfg(feature = "cuda")]
            (Backend::Cuda, ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(
                    Zensim::<cubecl::cuda::CudaRuntime>::new_strip_with_halo_and_regime(
                        client,
                        width,
                        height,
                        h_body,
                        crate::pipeline::STRIP_DEFAULT_HALO,
                        regime,
                    )?,
                )
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::wgpu::WgpuRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
            }
            #[cfg(feature = "wgpu")]
            (Backend::Wgpu, ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(
                    Zensim::<cubecl::wgpu::WgpuRuntime>::new_strip_with_halo_and_regime(
                        client,
                        width,
                        height,
                        h_body,
                        crate::pipeline::STRIP_DEFAULT_HALO,
                        regime,
                    )?,
                )
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, ResolvedMode::Full) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cpu::CpuRuntime>::new_with_regime(
                    client, width, height, regime,
                )?)
            }
            #[cfg(feature = "cpu")]
            (Backend::Cpu, ResolvedMode::Strip { h_body }) => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(
                    Zensim::<cubecl::cpu::CpuRuntime>::new_strip_with_halo_and_regime(
                        client,
                        width,
                        height,
                        h_body,
                        crate::pipeline::STRIP_DEFAULT_HALO,
                        regime,
                    )?,
                )
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

    /// Build a [`ZensimOpaque`] from a caller-supplied cubecl client
    /// (which may be bound to an explicit stream). Internal plumbing for
    /// [`crate::session::new_opaque_on_stream`]. Mirrors
    /// [`Self::new_with_memory_mode`]'s host-side mode resolution +
    /// regime-aware Full/Strip selection, on the supplied generic client.
    #[cfg(feature = "cubecl-types")]
    pub(crate) fn build_from_client<R: cubecl::Runtime>(
        client: cubecl::prelude::ComputeClient<R>,
        backend: Backend,
        width: u32,
        height: u32,
        params: ZensimParams,
        mode: crate::MemoryMode,
    ) -> Result<Self>
    where
        Zensim<R>: Send + 'static,
    {
        use crate::MemoryMode;
        use crate::memory_mode::{ResolvedMode, resolve_auto, vram_cap_bytes};

        if matches!(mode, MemoryMode::Tile { .. }) {
            return Err(crate::Error::ModeUnsupported("Tile"));
        }
        // See `new_with_memory_mode` — reflect-pad sub-64px up to the
        // 4-scale pyramid floor; the inner pipeline is built padded.
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidImageSize);
        }
        let logical_w = width;
        let logical_h = height;
        let width = width.max(MIN_PAD_DIM);
        let height = height.max(MIN_PAD_DIM);
        let regime = params.regime;
        let resolved: ResolvedMode = match mode {
            MemoryMode::Full => ResolvedMode::Full,
            MemoryMode::Strip { h_body } => {
                let body = h_body.unwrap_or_else(|| {
                    let cap = vram_cap_bytes();
                    crate::memory_mode::auto_strip_body_for(width, height, regime, cap)
                });
                ResolvedMode::Strip { h_body: body }
            }
            MemoryMode::Auto => {
                let cap = vram_cap_bytes();
                resolve_auto(width, height, regime, cap)?
            }
            MemoryMode::Tile { .. } => unreachable!("already returned above"),
        };
        let inner: Box<dyn ZensimInner + Send> = match resolved {
            ResolvedMode::Full => {
                Box::new(Zensim::<R>::new_with_regime(client, width, height, regime)?)
            }
            ResolvedMode::Strip { h_body } => {
                Box::new(Zensim::<R>::new_strip_with_halo_and_regime(
                    client,
                    width,
                    height,
                    h_body,
                    crate::pipeline::STRIP_DEFAULT_HALO,
                    regime,
                )?)
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

    /// Caller-requested logical `(width, height)`. For sub-64px images
    /// this is smaller than the internal padded pipeline size
    /// (`inner.dims()`); the compute entry points reflect-pad inputs up
    /// to that internal size transparently. At ≥64px the two agree.
    pub fn dims(&self) -> (u32, u32) {
        (self.logical_w, self.logical_h)
    }

    /// `true` when the inner pipeline was built larger than the logical
    /// image (i.e. the request was sub-64px in some axis and inputs need
    /// reflect-padding). `false` (no-op fast path) at ≥64px.
    #[inline]
    fn is_padded(&self) -> bool {
        let (pw, ph) = self.inner.dims();
        pw != self.logical_w || ph != self.logical_h
    }

    /// Reflect(mirror)-pad a packed `RGB8` buffer from the logical image
    /// extent up to the inner pipeline's padded extent. Borrows
    /// unchanged on the ≥64px fast path. Validates the input length
    /// against the logical extent (the inner pipeline validates the
    /// padded length).
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

    /// Reflect(mirror)-pad a single-channel `f32` plane (one of the
    /// already-linear RGB planes) from the logical extent up to the
    /// padded extent. Borrows unchanged at ≥64px.
    fn pad_plane<'a>(&self, src: &'a [f32]) -> Result<std::borrow::Cow<'a, [f32]>> {
        if !self.is_padded() {
            return Ok(std::borrow::Cow::Borrowed(src));
        }
        let (lw, lh) = (self.logical_w as usize, self.logical_h as usize);
        if src.len() != lw * lh {
            return Err(crate::Error::DimensionMismatch {
                expected: lw * lh,
                got: src.len(),
            });
        }
        let (pw, ph) = self.inner.dims();
        Ok(std::borrow::Cow::Owned(zenmetrics_gpu_core::reflect_pad(
            src, lw, lh, pw as usize, ph as usize, 1,
        )))
    }

    /// Crop a padded-extent diffmap (`pw × ph`, row-major) back to the
    /// logical extent (`lw × lh`). The reflect-pad places the original
    /// pixels at indices `[0, lw) × [0, lh)`, so the logical diffmap is
    /// the top-left sub-rectangle. No-op at ≥64px.
    fn crop_diffmap(&self, buf: &mut Vec<f32>) {
        if !self.is_padded() {
            return;
        }
        let (lw, lh) = (self.logical_w as usize, self.logical_h as usize);
        let (pw, _ph) = self.inner.dims();
        let pw = pw as usize;
        if buf.len() < pw * lh {
            return; // unexpected; leave as-is rather than panic
        }
        let mut out = Vec::with_capacity(lw * lh);
        for y in 0..lh {
            let row = y * pw;
            out.extend_from_slice(&buf[row..row + lw]);
        }
        *buf = out;
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
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_features(&r, &d)
    }

    /// Compute the **first 228** features from [`PixelSlice`] inputs.
    /// Truncates the Extended / WithIw output to the basic block.
    #[cfg(feature = "pixels")]
    pub fn compute_features_pixels(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
    ) -> Result<[f64; TOTAL_FEATURES]> {
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        let rp = self.pad_rgb(&ref_buf)?;
        let dp = self.pad_rgb(&dis_buf)?;
        self.inner.compute_features(&rp, &dp)
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
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_features_vec(&r, &d)
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
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        let rp = self.pad_rgb(&ref_buf)?;
        let dp = self.pad_rgb(&dis_buf)?;
        self.inner.compute_features_vec(&rp, &dp)
    }

    /// Upload + pyramid-build the reference image ONCE, then call
    /// [`Self::compute_with_reference_srgb_u8`] for each distortion.
    /// Critical for sweep workloads: a single reference with N
    /// distortions saves N-1 ref uploads (~1 MB each at 1 MP) and
    /// N-1 ref-pyramid kernel launches.
    pub fn set_reference_srgb_u8(&mut self, ref_rgb: &[u8]) -> Result<()> {
        let r = self.pad_rgb(ref_rgb)?;
        self.inner.set_reference(&r)
    }

    /// Compute the raw feature vector against the cached reference
    /// (zensim-specific — zensim is a learned metric). Returns
    /// `Vec<f64>` of length `params.regime.total_features()` (228 /
    /// 300 / 372). Returns [`crate::Error::NoCachedReference`] if
    /// [`Self::set_reference_srgb_u8`] was never called. For the uniform
    /// [`Score`], use [`Self::compute_with_reference_srgb_u8`].
    pub fn compute_features_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Vec<f64>> {
        let d = self.pad_rgb(dis_rgb)?;
        self.inner.compute_with_reference_vec(&d)
    }

    /// Score a DIST candidate against the cached reference — the uniform
    /// [`Score`] (with weights / profile already applied). Pre-requisite:
    /// [`Self::set_reference_srgb_u8`] (or `set_reference_pixels`)
    /// must have been called. Uniform across every `*-gpu` opaque metric;
    /// for the raw feature vector use
    /// [`Self::compute_features_with_reference_srgb_u8`].
    ///
    /// Profile-mode only — when [`ZensimParams::profile`] is `None`
    /// the legacy 228-feature linear-weights path is used, which
    /// requires the fixed-shape `compute_features` (not the
    /// variable-shape `compute_with_reference_vec`). Surfaces
    /// [`crate::Error::NoCachedReference`] for legacy-weights callers
    /// — those should use the one-shot [`Self::compute_srgb_u8`]
    /// instead. The umbrella `zenmetrics-api` cached-ref API
    /// dispatches through this method for the profile-mode default.
    pub fn compute_with_reference_srgb_u8(&mut self, dis_rgb: &[u8]) -> Result<Score> {
        if self.params.profile.is_some() {
            let d = self.pad_rgb(dis_rgb)?;
            let features = self.inner.compute_with_reference_vec(&d)?;
            // Padded dims feed the MLP size-axes — matches the CPU
            // canonical path, which scores the reflect-padded image.
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, None))
        } else {
            // Legacy linear-weights cached-ref isn't wired today —
            // compute_features (228-feat fixed struct) has no
            // compute_with_reference twin. Surface a clear error
            // rather than silently fall back to one-shot.
            Err(crate::Error::NoCachedReference)
        }
    }

    /// Set + upload + pre-build the reference's XYB pyramid from
    /// [`PixelSlice`] input. Companion to
    /// [`Self::compute_with_reference_pixels`].
    #[cfg(feature = "pixels")]
    pub fn set_reference_pixels(&mut self, r: PixelSlice<'_>) -> Result<()> {
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let rp = self.pad_rgb(&ref_buf)?;
        self.inner.set_reference(&rp)
    }

    /// Compute the raw feature vector against the cached reference from a
    /// [`PixelSlice`] distortion (zensim-specific). See
    /// [`Self::compute_features_with_reference_srgb_u8`] for semantics.
    #[cfg(feature = "pixels")]
    pub fn compute_features_with_reference_pixels(
        &mut self,
        d: PixelSlice<'_>,
    ) -> Result<Vec<f64>> {
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        let dp = self.pad_rgb(&dis_buf)?;
        self.inner.compute_with_reference_vec(&dp)
    }

    /// `true` if a reference is currently cached (uniform across every
    /// `*-gpu` opaque metric).
    pub fn has_reference(&self) -> bool {
        self.inner.has_reference()
    }

    /// Drop the cached reference state (uniform across every `*-gpu`
    /// opaque metric).
    pub fn clear_reference(&mut self) {
        self.inner.clear_reference()
    }

    /// Compute the uniform [`Score`] from packed sRGB. Routes through
    /// CPU `zensim::score_features_with_profile` when
    /// [`ZensimParams::profile`] is set (v0.3+ canonical path), or
    /// through legacy `score_from_features(weights)` when only
    /// [`ZensimParams::weights`] is set. Returns
    /// `Score { value: NAN, .. }` if neither is wired.
    ///
    /// Byte-identity short-circuit: if `ref_rgb == dis_rgb`, returns
    /// `Score { value: 100.0, .. }` without running the GPU kernel.
    /// Mirrors the CPU canonical `Zensim::compute(...).score()`
    /// behaviour — without this, the f32 GPU pipeline on byte-equal
    /// inputs produces tiny non-zero residuals that the V0_3 MLP +
    /// PCHIP spline (`extrapolate_score=true`) maps to arbitrary
    /// values outside [0, 100], confusing callers that rely on
    /// identity ⇒ ~100.
    pub fn compute_srgb_u8(&mut self, ref_rgb: &[u8], dis_rgb: &[u8]) -> Result<Score> {
        if let Some(score) = identity_short_circuit(ref_rgb, dis_rgb) {
            return Ok(score);
        }
        // When a profile is set, the canonical path needs the
        // regime-appropriate full-width feature vector (228 / 300 /
        // 372), not the 228 truncation that `compute_features` returns.
        // Otherwise the post-network forward pass would see the wrong
        // input shape and `score_features_with_profile` would error.
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        if self.params.profile.is_some() {
            let features = self.inner.compute_features_vec(&r, &d)?;
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, None))
        } else {
            let features = self.inner.compute_features(&r, &d)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Like [`Self::compute_srgb_u8`] but ALSO returns the
    /// regime-appropriate feature vector (228 / 300 / 372) extracted in
    /// the **same pass** — zensim is a feature extractor, so the features
    /// are produced anyway en route to the scalar score. This lets the
    /// `zenmetrics-api` umbrella expose features alongside the score
    /// without a second GPU extraction. Mirrors butteraugli's
    /// `compute_srgb_u8_with_pnorm3` two-output entry point.
    ///
    /// The returned `Vec<f64>` has the active [`ZensimFeatureRegime`]'s
    /// length ([`Self::regime`]). The profile scoring path (the shipped
    /// configuration) is single-pass; the legacy `weights`-only path
    /// re-scores via [`Self::compute_srgb_u8`] (a second extraction, only
    /// on that uncommon path).
    pub fn compute_srgb_u8_with_features(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<(Score, Vec<f64>)> {
        let features = self.inner.compute_features_vec(ref_rgb, dis_rgb)?;
        let score = if let Some(s) = identity_short_circuit(ref_rgb, dis_rgb) {
            s
        } else if self.params.profile.is_some() {
            let (w, h) = self.inner.dims();
            self.score_from_profile_vec(&features, w, h, None)
        } else {
            // Legacy weights path scores the 228-truncation via a
            // different feature shape; reuse the canonical entry point.
            self.compute_srgb_u8(ref_rgb, dis_rgb)?
        };
        Ok((score, features))
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
        if let Some(score) = identity_short_circuit(ref_rgb, dis_rgb) {
            return Ok(score);
        }
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        if self.params.profile.is_some() {
            let features = self.inner.compute_features_vec(&r, &d)?;
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, codec_hint))
        } else {
            let features = self.inner.compute_features(&r, &d)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Compute the uniform [`Score`] from [`PixelSlice`] inputs.
    ///
    /// Identical byte-identity short-circuit as
    /// [`Self::compute_srgb_u8`]: when the converted-to-sRGB ref and
    /// dist buffers match byte-for-byte, returns 100.0 without
    /// running the GPU kernel.
    #[cfg(feature = "pixels")]
    pub fn compute_pixels(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> Result<Score> {
        let ref_buf = to_srgb_rgb8(&r, self.logical_w, self.logical_h)?;
        let dis_buf = to_srgb_rgb8(&d, self.logical_w, self.logical_h)?;
        if let Some(score) = identity_short_circuit(&ref_buf, &dis_buf) {
            return Ok(score);
        }
        if self.params.profile.is_some() {
            let rp = self.pad_rgb(&ref_buf)?;
            let dp = self.pad_rgb(&dis_buf)?;
            let features = self.inner.compute_features_vec(&rp, &dp)?;
            // Padded dims feed the MLP size-axes (matches CPU).
            let (w, h) = self.inner.dims();
            Ok(self.score_from_profile_vec(&features, w, h, None))
        } else {
            // Re-use the pixel-aware feature path that handles strided
            // ref/dist explicitly so we don't double-convert here.
            let features = self.compute_features_pixels(r, d)?;
            Ok(self.score_from_linear(features))
        }
    }

    /// Profile-aware scoring path — runs the GPU-extracted feature
    /// vector through the CPU
    /// [`zensim::score_features_with_profile_and_codec`] dispatch
    /// (per-sample-α head, tanh-pin, PCHIP spline, per-codec affine).
    ///
    /// Returns `f64::NAN` when no profile is configured or when the
    /// CPU scoring helper errors (e.g. the feature vector's shape
    /// doesn't match the profile's expected input width).
    fn score_from_profile_vec(
        &self,
        features: &[f64],
        width: u32,
        height: u32,
        codec_hint: Option<&str>,
    ) -> Score {
        let value = match self.params.profile {
            Some(profile) => zensim::score_features_with_profile_and_codec(
                profile, features, width, height, codec_hint,
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
        let r = self.pad_rgb(ref_rgb)?;
        let d = self.pad_rgb(dis_rgb)?;
        let score = self.inner.score_with_diffmap(&r, &d, diffmap_out)?;
        self.crop_diffmap(diffmap_out);
        Ok(score)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_with_warm_ref_diffmap`].
    /// See that method's docs.
    pub fn score_with_warm_ref_diffmap(
        &mut self,
        dis_rgb: &[u8],
        diffmap_out: &mut Vec<f32>,
    ) -> Result<f32> {
        let d = self.pad_rgb(dis_rgb)?;
        let score = self.inner.score_with_warm_ref_diffmap(&d, diffmap_out)?;
        self.crop_diffmap(diffmap_out);
        Ok(score)
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
        let rr = self.pad_plane(ref_r)?;
        let rg = self.pad_plane(ref_g)?;
        let rb = self.pad_plane(ref_b)?;
        let dr = self.pad_plane(dist_r)?;
        let dg = self.pad_plane(dist_g)?;
        let db = self.pad_plane(dist_b)?;
        self.inner
            .score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db)
    }

    /// Non-planar (interleaved) variant of [`Self::score_from_linear_planes`]:
    /// two interleaved linear-RGB f32 buffers (`[R,G,B, R,G,B, …]`, each
    /// `width·height·3`) instead of six planar slices, deinterleaved on the
    /// host. Errors with [`crate::Error::DimensionMismatch`] if a buffer's
    /// length isn't a multiple of 3.
    pub fn score_from_linear_interleaved(
        &mut self,
        ref_rgb: &[f32],
        dis_rgb: &[f32],
    ) -> Result<f32> {
        let (rr, rg, rb) = zenmetrics_gpu_core::deinterleave_rgb_f32(ref_rgb).ok_or(
            crate::Error::DimensionMismatch {
                expected: ref_rgb.len() / 3 * 3,
                got: ref_rgb.len(),
            },
        )?;
        let (dr, dg, db) = zenmetrics_gpu_core::deinterleave_rgb_f32(dis_rgb).ok_or(
            crate::Error::DimensionMismatch {
                expected: dis_rgb.len() / 3 * 3,
                got: dis_rgb.len(),
            },
        )?;
        self.score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db)
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
        let rr = self.pad_plane(ref_r)?;
        let rg = self.pad_plane(ref_g)?;
        let rb = self.pad_plane(ref_b)?;
        let dr = self.pad_plane(dist_r)?;
        let dg = self.pad_plane(dist_g)?;
        let db = self.pad_plane(dist_b)?;
        let score = self.inner.score_from_linear_planes_with_diffmap(
            &rr,
            &rg,
            &rb,
            &dr,
            &dg,
            &db,
            diffmap_out,
        )?;
        self.crop_diffmap(diffmap_out);
        Ok(score)
    }

    /// Mirror of [`crate::pipeline::Zensim::warm_reference_from_linear_planes`].
    /// See that method's docs.
    pub fn warm_reference_from_linear_planes(
        &mut self,
        ref_r: &[f32],
        ref_g: &[f32],
        ref_b: &[f32],
    ) -> Result<()> {
        let rr = self.pad_plane(ref_r)?;
        let rg = self.pad_plane(ref_g)?;
        let rb = self.pad_plane(ref_b)?;
        self.inner
            .warm_reference_from_linear_planes(&rr, &rg, &rb)
    }

    /// Mirror of [`crate::pipeline::Zensim::score_from_linear_planes_with_warm_ref`].
    /// See that method's docs.
    pub fn score_from_linear_planes_with_warm_ref(
        &mut self,
        dist_r: &[f32],
        dist_g: &[f32],
        dist_b: &[f32],
    ) -> Result<f32> {
        let dr = self.pad_plane(dist_r)?;
        let dg = self.pad_plane(dist_g)?;
        let db = self.pad_plane(dist_b)?;
        self.inner
            .score_from_linear_planes_with_warm_ref(&dr, &dg, &db)
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
        let dr = self.pad_plane(dist_r)?;
        let dg = self.pad_plane(dist_g)?;
        let db = self.pad_plane(dist_b)?;
        let score = self
            .inner
            .score_from_linear_planes_with_warm_ref_diffmap(&dr, &dg, &db, diffmap_out)?;
        self.crop_diffmap(diffmap_out);
        Ok(score)
    }
}

/// Byte-identity short-circuit for the opaque `compute_*` paths.
///
/// When the reference and distorted buffers are byte-equal, every
/// per-pixel SSIM / edge / IW term in the feature extractor collapses
/// to 0 mathematically — but the f32 GPU pipeline picks up tiny ULP
/// residuals at the coarse pyramid scales, and the V0_3 profile's
/// PCHIP spline (with `extrapolate_score=true`) maps the resulting
/// residual to arbitrary score-domain values outside [0, 100]. The
/// CPU canonical `Zensim::compute(...).score()` path has this same
/// short-circuit; mirroring it on the GPU side keeps the
/// identity-input score predictable (100.0) for both paths.
///
/// Returns `None` when the inputs are not byte-equal — callers fall
/// through to the normal scoring path. The `metric_name` /
/// `metric_version` labels match the rest of the opaque API.
fn identity_short_circuit(ref_rgb: &[u8], dis_rgb: &[u8]) -> Option<Score> {
    if ref_rgb.len() == dis_rgb.len() && ref_rgb == dis_rgb {
        Some(Score {
            value: 100.0,
            metric_name: "zensim",
            metric_version: env!("CARGO_PKG_VERSION"),
        })
    } else {
        None
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
