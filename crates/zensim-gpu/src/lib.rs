//! Multi-vendor GPU implementation of the zensim perceptual similarity
//! feature extractor.
//!
//! Built on [CubeCL](https://github.com/tracel-ai/cubecl) — the same
//! `#[cube]` Rust kernel source dispatches across:
//! - **CUDA** (NVIDIA) via the cubecl CUDA runtime
//! - **WGPU** (cross-platform) via Vulkan/Metal/DX12/WebGPU
//! - **HIP** (AMD ROCm) when the `hip` feature is enabled
//! - **CPU** (build-only) — `cubecl-cpu` 0.10 doesn't yet implement
//!   `Atomic<f32>::fetch_add` and our reduction relies on it; use the
//!   published `zensim` crate as the CPU reference instead.
//!
//! Algorithmic parity target is the published `zensim` v0.2.8 crate
//! with `ZensimProfile::latest()` (= `WEIGHTS_PREVIEW_V0_2`, 228
//! features = 4 scales × 3 channels × 19 features). At the pyramid
//! level this also matches `crates/zensim-cuda/`, which uses the same
//! SIMD-padded layout, the same `cbrtf_fast` Halley iterations, the
//! same fused H-blur (mu1/mu2/sigma_sq/sigma12) and fused V-blur +
//! per-pixel feature kernels.
//!
//! The CUDA crate stays in the workspace; this one extends reach to
//! AMD / Intel / Apple / WebGPU without giving anything up on NVIDIA.
//!
//! ## Single-image usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use zensim_gpu::Zensim;
//!
//! let client = WgpuRuntime::client(&Default::default());
//! let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//! let dist_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//!
//! let features = z.compute_features(&ref_srgb, &dist_srgb)?;
//! // Apply trained weights from `zensim::profile::WEIGHTS_PREVIEW_V0_2`
//! // (228 entries) and convert to a 0-100 score.
//! # Ok::<(), zensim_gpu::Error>(())
//! ```
//!
//! ## Cached-reference usage
//!
//! ```no_run
//! use cubecl::Runtime;
//! use cubecl::wgpu::WgpuRuntime;
//! use zensim_gpu::Zensim;
//!
//! # fn candidates() -> Vec<Vec<u8>> { vec![] }
//! let client = WgpuRuntime::client(&Default::default());
//! let mut z = Zensim::<WgpuRuntime>::new(client, 512, 512)?;
//!
//! let ref_srgb: Vec<u8> = vec![0; 512 * 512 * 3];
//! z.set_reference(&ref_srgb)?;
//! for candidate in candidates() {
//!     let features = z.compute_with_reference(&candidate)?;
//!     // ... apply weights ...
//! }
//! # Ok::<(), zensim_gpu::Error>(())
//! ```
//!
//! ## Status
//!
//! Initial port from `zensim-cuda`. See `PORT_STATUS.md`.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::doc_lazy_continuation)]

// `kernels` is reached by-path cross-crate (cvvdp-gpu/src/kernels/color.rs
// shares the scalar reference) — not a supported per-crate API.
#[doc(hidden)]
pub mod kernels;
pub mod memory_mode;
pub(crate) mod opaque;
// `pipeline` is reached by-path from this crate's own `strip_memory_demo`
// example — `#[doc(hidden)]`, like `session`. Not a supported per-crate API.
#[doc(hidden)]
pub mod pipeline;
// Stream-bound session plumbing for `zenmetrics_api::MetricSession`
// (issue #17). `#[doc(hidden)]`, gated `cubecl-types`. Not a supported
// per-crate API.
#[cfg(feature = "cubecl-types")]
#[doc(hidden)]
pub mod session;
pub(crate) mod weights;

pub use memory_mode::{
    MemoryMode, ResolvedMode, ScoreResourceEstimate, estimate_gpu_memory_bytes,
    estimate_score_resources, estimate_score_time_ms, estimate_strip_gpu_memory_bytes,
    vram_cap_bytes,
};

// Re-export the canonical default-weights array so callers can wire
// custom params without rebuilding it themselves.
pub use weights::WEIGHTS_PREVIEW_V0_2;

// Uniform opaque API (Phase 2). See `opaque.rs`.
pub use opaque::{Backend, Score, ZensimOpaque, ZensimParams};

// Typed-generic API (gated behind `cubecl-types`).
#[cfg(feature = "cubecl-types")]
pub use pipeline::Zensim;

// `STRIP_ALIGN` is asserted against by zensim-gpu's own `memory_mode`
// integration test; re-exported `#[doc(hidden)]` so the test keeps a
// handle without `pipeline` being a public module path.
#[doc(hidden)]
pub use pipeline::STRIP_ALIGN;

/// Number of pyramid scales — matches CPU zensim's `WEIGHTS_PREVIEW_V0_2`.
pub const SCALES: usize = 4;

/// Features per channel per scale = 13 basic (mean/L2/L4) + 6 peak
/// (max/L8 pooled) = 19.
pub const FEATURES_PER_CHANNEL: usize = 19;

/// Features per channel per scale, basic ("scored") block — matches
/// CPU `FEATURES_PER_CHANNEL_BASIC`.
pub const FEATURES_PER_CHANNEL_BASIC: usize = 13;

/// Peak-block features per channel per scale.
pub const FEATURES_PER_CHANNEL_PEAKS: usize = 6;

/// Masked (extended) features per channel per scale — matches CPU
/// `streaming::ScaleStats::masked_*` (`masked_ssim_mean/4th/2nd`,
/// `masked_art_4th`, `masked_det_4th`, `masked_mse` = 6).
pub const FEATURES_PER_CHANNEL_MASKED: usize = 6;

/// Information-content-weighted features per channel per scale —
/// matches CPU `FEATURES_PER_CHANNEL_IW`.
pub const FEATURES_PER_CHANNEL_IW: usize = 6;

/// Features per scale (= 19 × 3 channels = 57).
pub const FEATURES_PER_SCALE: usize = FEATURES_PER_CHANNEL * 3;

/// Total features = 4 scales × 57 = 228 (Basic regime).
pub const TOTAL_FEATURES: usize = FEATURES_PER_SCALE * SCALES;

/// Total features = 228 + 4 × 3 × 6 = 300 (Extended regime).
pub const TOTAL_FEATURES_EXTENDED: usize =
    TOTAL_FEATURES + SCALES * 3 * FEATURES_PER_CHANNEL_MASKED;

/// Total features = 300 + 4 × 3 × 6 = 372 (WithIw regime).
pub const TOTAL_FEATURES_WITH_IW: usize =
    TOTAL_FEATURES_EXTENDED + SCALES * 3 * FEATURES_PER_CHANNEL_IW;

/// GPU feature-output regime — mirrors the CPU `Zensim::compute*` /
/// `ZensimConfig` flag combinations.
///
/// - `Basic` (228) — basic + peak features, scored by
///   `WEIGHTS_PREVIEW_V0_2`. Default; bit-for-bit identical to the
///   pre-372 `Zensim::compute_features` output.
/// - `Extended` (300) — adds the masked block (`228..300`). Matches
///   CPU `Zensim::compute_extended_features`.
/// - `WithIw` (372) — adds the IW block (`300..372`). Matches CPU
///   `ZensimConfig { extended_features: true, compute_iw_features: true }`,
///   the input width consumed by `PreviewV0_5` (V_22-IW v2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ZensimFeatureRegime {
    /// 228 features.
    #[default]
    Basic,
    /// 300 features.
    Extended,
    /// 372 features.
    WithIw,
}

impl ZensimFeatureRegime {
    /// Total feature-vector length emitted for this regime.
    pub const fn total_features(self) -> usize {
        match self {
            ZensimFeatureRegime::Basic => TOTAL_FEATURES,
            ZensimFeatureRegime::Extended => TOTAL_FEATURES_EXTENDED,
            ZensimFeatureRegime::WithIw => TOTAL_FEATURES_WITH_IW,
        }
    }

    /// True if the masked-feature pass must run.
    pub const fn needs_masked(self) -> bool {
        matches!(
            self,
            ZensimFeatureRegime::Extended | ZensimFeatureRegime::WithIw
        )
    }

    /// True if the IW pass must run.
    pub const fn needs_iw(self) -> bool {
        matches!(self, ZensimFeatureRegime::WithIw)
    }

    /// True if any extended (masked OR IW) pass must run. When this is
    /// false the original 228-only fast path is taken.
    pub const fn needs_extended_kernel(self) -> bool {
        self.needs_masked() || self.needs_iw()
    }
}

/// Blur radius for the V0.1 / V0.2 profiles (`PROFILE_PREVIEW_V0_*`'s
/// `blur_radius`). Diameter = 11. Bumping this changes the HF ratios
/// nonlinearly; the trained weights only pair with this value.
pub const BLUR_RADIUS: u32 = 5;

/// SIMD-alignment padding applied by CPU zensim at scale 0. Every plane
/// is widened to the next multiple of 16, with an extra 16 cols added
/// when the aligned width ≥ 512 and `aligned/16` is even (cache-set
/// avoidance on x86). The mirror-pad fill matches CPU exactly.
///
/// Must match `zensim::blur::simd_padded_width`.
pub fn simd_padded_width(width: usize) -> usize {
    let aligned = (width + 15) & !15;
    if aligned >= 512 && (aligned / 16).is_multiple_of(2) {
        aligned + 16
    } else {
        aligned
    }
}

/// Apply `100 - 18·d^0.7` to a weighted feature distance, matching CPU
/// zensim's `score_from_features`. Pass the trained weights from
/// `zensim::profile::WEIGHTS_PREVIEW_V0_2`.
pub fn score_from_features(features: &[f64], weights: &[f64]) -> f64 {
    assert_eq!(features.len(), weights.len());
    let raw: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(&f, &w)| w * f)
        .sum();
    let n_scales = features.len() / FEATURES_PER_SCALE;
    let raw_per_scale = raw / n_scales.max(1) as f64;
    if raw_per_scale <= 0.0 {
        100.0
    } else {
        100.0 - 18.0 * raw_per_scale.powf(0.7)
    }
}

#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image is too small for the configured pyramid (logical w/h < 8 at scale 0).
    InvalidImageSize,
    /// The Extended / WithIw regime requires more device memory than the
    /// caller's `max_extended_plane_bytes` budget allows. Returned by
    /// [`Zensim::new_with_regime`] when the per-scale persist-plane
    /// budget exceeds the cap.
    ExtendedPlaneBudgetExceeded {
        needed_bytes: usize,
        max_bytes: usize,
    },
    /// The requested [`MemoryMode`](crate::MemoryMode) variant isn't
    /// implemented yet. As of task #75 (Mode-E strip) only `Tile`
    /// remains unsupported; `Strip` and `Auto` are implemented.
    ModeUnsupported(&'static str),
    /// [`MemoryMode::Auto`](crate::MemoryMode) couldn't fit the image
    /// into the VRAM cap — even the per-strip working set exceeds the
    /// cap. (Strip is implemented since task #75; this fires only when
    /// the smallest viable strip still doesn't fit.)
    TooBigForFull { needed: usize, cap: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::NoCachedReference => write!(f, "no cached reference; call set_reference first"),
            Error::InvalidImageSize => write!(f, "image must be at least 8×8 pixels"),
            Error::ExtendedPlaneBudgetExceeded {
                needed_bytes,
                max_bytes,
            } => write!(
                f,
                "extended-regime persist planes need {needed_bytes} bytes, \
                 over budget {max_bytes} (cap configured via \
                 Zensim::new_with_regime_budget)"
            ),
            Error::ModeUnsupported(variant) => write!(
                f,
                "MemoryMode::{variant} is not yet implemented in zensim-gpu"
            ),
            Error::TooBigForFull { needed, cap } => write!(
                f,
                "Auto could not place image in {cap} byte cap; needs at least {needed} bytes \
                 (even the smallest strip working set exceeds the cap — raise \
                 ZENMETRICS_VRAM_CAP_BYTES or use a smaller image)"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
