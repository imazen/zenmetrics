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
    /// to the HF band-energy features (basic slots 10-12 per scale
    /// per channel). See `zensim::acumen` and tracking issue
    /// `imazen/zensim#40` Gate A. Default `None` → no weighting,
    /// V_22-shipped path bit-stable.
    pub acumen_mode_a: Option<zensim::acumen::viewing::ViewingCondition>,
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
        }
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
    fn dims(&self) -> (u32, u32);
    fn set_acumen_viewing(
        &mut self,
        viewing: Option<zensim::acumen::viewing::ViewingCondition>,
    );
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

    fn dims(&self) -> (u32, u32) {
        Zensim::dimensions(self)
    }

    fn set_acumen_viewing(
        &mut self,
        viewing: Option<zensim::acumen::viewing::ViewingCondition>,
    ) {
        Zensim::set_acumen_viewing(self, viewing);
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
        let inner: Box<dyn ZensimInner + Send> = match backend {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                use cubecl::Runtime;
                let client = cubecl::cuda::CudaRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cuda::CudaRuntime>::new(client, width, height)?)
            }
            #[cfg(feature = "wgpu")]
            Backend::Wgpu => {
                use cubecl::Runtime;
                let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::wgpu::WgpuRuntime>::new(client, width, height)?)
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu => {
                use cubecl::Runtime;
                let client = cubecl::cpu::CpuRuntime::client(&Default::default());
                Box::new(Zensim::<cubecl::cpu::CpuRuntime>::new(client, width, height)?)
            }
        };
        let mut shim = Self {
            inner,
            params,
            backend,
        };
        // Push acumen viewing through to the inner pipeline so that
        // subsequent `set_reference` calls compute per-image
        // castleCSF band weights. When `params.acumen_mode_a` is
        // None this is a no-op.
        let viewing = shim.params.acumen_mode_a;
        shim.inner.set_acumen_viewing(viewing);
        Ok(shim)
    }

    /// Configured `(width, height)`.
    pub fn dims(&self) -> (u32, u32) {
        self.inner.dims()
    }

    /// Compute the 228-feature vector for one pair from packed sRGB.
    pub fn compute_features_srgb_u8(
        &mut self,
        ref_rgb: &[u8],
        dis_rgb: &[u8],
    ) -> Result<[f64; TOTAL_FEATURES]> {
        self.inner.compute_features(ref_rgb, dis_rgb)
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
