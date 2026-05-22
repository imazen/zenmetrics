//! Multi-vendor GPU implementation of **IW-SSIM** (Information-Content
//! Weighted SSIM) — Wang & Li, *IEEE TIP* vol. 20 no. 5, May 2011.
//!
//! Faithful port of the authors' reference code:
//! - MATLAB: <https://ece.uwaterloo.ca/~z70wang/research/iwssim/iwssim_iwpsnr.zip>
//! - Python (PyTorch): <https://github.com/Jack-guo-xy/Python-IW-SSIM>
//!
//! Both references produce identical scores; we treat them as one
//! algorithm and parity-test against the Python reference directly.
//!
//! # Algorithm (paper §III-B)
//!
//! 1. Convert RGB → grayscale (BT.601, rounded) on the host (or accept
//!    grayscale floats directly via [`Iwssim::compute_gray`]).
//! 2. Build a **5-level Laplacian pyramid** using pyrtools' `binom5`
//!    filter (`sqrt(2)·[1,4,6,4,1]/16`) with `reflect1` boundary —
//!    bands `L_1..L_4` are real Laplacians, `L_5` is the residual
//!    lowpass.
//! 3. For each scale, compute the 11×11 Gaussian (σ=1.5)
//!    contrast-structure map `cs_j = (2σ_{12} + C₂) / (σ₁² + σ₂² + C₂)`
//!    with `C₂ = (0.03·255)²`. At the coarsest scale also compute the
//!    luminance map `l_5 = (2µ₁µ₂ + C₁) / (µ₁² + µ₂² + C₁)` with
//!    `C₁ = (0.01·255)²`.
//! 4. For scales 1..4, compute the **information-content weight map**
//!    via the GSM model (paper §II): 3×3 box statistics, a parent
//!    band from `imenlarge2`(`L_{j+1}`), a small (9 or 10)×(9 or 10)
//!    covariance eigendecomposition, and per-pixel mutual information.
//! 5. Pool each scale: `wmcs_j = Σ(cs_j · w_j) / Σ(w_j)` for `j<5`
//!    (after cropping `w_j` by `bound1 = 4`), `wmcs_5 = mean(cs_5 · l_5)`.
//! 6. Final score: `Π_{j=1}^{5} |wmcs_j|^{β_j}` with
//!    `β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.
//!
//! # Pipeline boundaries between GPU and CPU
//!
//! - **GPU:** sRGB→gray (optional), pyramid build, per-scale Gaussian /
//!   box statistics, neighborhood gather, per-pixel quadratic form,
//!   `infow`, weighted sums.
//! - **CPU:** the per-scale `(9 or 10)×(9 or 10)` covariance
//!   eigendecomposition + matrix inverse — a one-shot per scale.
//!   Pushing it to GPU would dominate code complexity for no perf gain
//!   (≤ 100 floats of work, dwarfed by the per-pixel kernels).
//!
//! # Status
//!
//! Initial port. See `PORT_STATUS.md`. Parity target: scalar `score`
//! within 1e-4 (relative) of the reference Python on the published
//! `images/Ref.bmp` / `images/Dist.jpg` pair.

#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod eig;
pub mod filters;
pub mod kernels;
pub mod memory_mode;
pub mod opaque;
pub mod pipeline;

pub use memory_mode::{
    MemoryMode, ResolvedMode, estimate_gpu_memory_bytes, estimate_strip_gpu_memory_bytes,
    live_vram_probe_bytes, vram_cap_bytes,
};

// Uniform opaque API (Phase 2). See `opaque.rs`.
pub use opaque::{Backend, IwssimOpaque, IwssimParams, Score};

// Typed-generic API (gated behind `cubecl-types`).
#[cfg(feature = "cubecl-types")]
pub use pipeline::Iwssim;

/// Number of pyramid scales — fixed at 5 by the IW-SSIM paper.
pub const NUM_SCALES: usize = 5;

/// Minimum native pyramid dimension required by the reference algorithm.
///
/// The paper's `iwssim.m` requires `min(W, H) ≥ 11 · 2^(Nsc-1) = 176` so
/// the coarsest scale (`L_5`) still has enough pixels for a valid-mode
/// 11×11 Gaussian. For inputs smaller than this along either axis we
/// either reject (default — bit-exact stock IW-SSIM) or reflect-pad up
/// to `MIN_NATIVE_DIM` on the short axis (`IwssimConfig::allow_small`).
pub const MIN_NATIVE_DIM: u32 = 176;

/// How to handle inputs smaller than [`MIN_NATIVE_DIM`] on either axis.
///
/// Validated empirically on a 980-pair CID22-JPEG corpus (see
/// `benchmarks/iwssim_smallimg/README.md` in the workspace). At dims
/// {64, 96, 128} all three "adapt" strategies stay within ±0.01
/// Spearman ρ and ±0.01 rank-flip rate of the stock 176-px baseline
/// against ssim2_gpu. **Tile is the best of the three by 0.005-0.010 ρ**
/// at every sub-176 dim and is what [`IwssimConfig::adaptive()`] uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IwssimStrategy {
    /// Reject sub-176 inputs at `Iwssim::new` / `Iwssim::with_config`
    /// time with `Err(InvalidImageSize)`. **Default**. Preserves the
    /// historical (and crates.io API) behaviour exactly. Zero overhead
    /// on stock-size inputs; no host-side preprocessing branch.
    #[default]
    Reject,
    /// Repeat (tile) the native content along each axis until both
    /// dimensions reach `MIN_NATIVE_DIM`, then run stock IW-SSIM on the
    /// tiled image. The pyramid sees a periodic signal whose boundary
    /// statistics match the interior; this is the empirically best
    /// strategy on the validation corpus.
    Tile,
    /// Reflect-pad (pyrtools `reflect1` boundary) the native content
    /// out to `MIN_NATIVE_DIM` on the short axis, then run stock
    /// IW-SSIM on the padded image. Was the only adaptive strategy
    /// in iwssim-gpu 0.0.1; kept for callers that need bit-exact-to-
    /// that behaviour. ~0.005-0.010 ρ worse than [`Tile`] on the
    /// validation corpus.
    ReflectPad,
}

/// Pipeline configuration knobs surfaced to callers.
///
/// `Default` is the historical behaviour: reject any input with
/// `min(width, height) < MIN_NATIVE_DIM`. Switch to
/// [`IwssimConfig::adaptive()`] for the tile-based small-image path.
#[derive(Debug, Clone, Copy, Default)]
pub struct IwssimConfig {
    /// Which strategy to use for inputs below `MIN_NATIVE_DIM` on
    /// either axis. See [`IwssimStrategy`].
    pub strategy: IwssimStrategy,
}

impl IwssimConfig {
    /// Use the tile-based adaptive small-image strategy — the
    /// empirically best of the three on the validation corpus.
    /// Equivalent to `IwssimConfig { strategy: IwssimStrategy::Tile }`.
    pub const fn adaptive() -> Self {
        Self { strategy: IwssimStrategy::Tile }
    }

    /// Use the reflect-pad small-image strategy — kept for callers
    /// that depend on the iwssim-gpu 0.0.1 behaviour. [`adaptive()`]
    /// is recommended for new code.
    pub const fn reflect_pad() -> Self {
        Self { strategy: IwssimStrategy::ReflectPad }
    }

    /// Compatibility shim: `allow_small(true) → IwssimStrategy::Tile`,
    /// `allow_small(false) → IwssimStrategy::Reject`. Existing call
    /// sites do not need to change; new code should reach for
    /// [`adaptive()`] or [`reflect_pad()`] instead.
    ///
    /// **Note: the underlying strategy changed from ReflectPad to Tile
    /// in this revision** based on the 2026-05-17 validation
    /// (`benchmarks/iwssim_smallimg/`). Callers that need the exact
    /// 0.0.1 behaviour must explicitly use [`reflect_pad()`].
    pub const fn allow_small(allow: bool) -> Self {
        Self {
            strategy: if allow {
                IwssimStrategy::Tile
            } else {
                IwssimStrategy::Reject
            },
        }
    }
}

/// Implementation-tagged column name for IW-SSIM scores in parquet
/// sidecars. Mirrors the `cvvdp_gpu::CVVDP_COLUMN_NAME` pattern so
/// multiple IW-SSIM implementations (e.g. a reference Python pyrtools
/// port, this GPU port, a hypothetical Burn port) can coexist in the
/// same joined parquet without column-name collisions. Default form:
/// `iwssim_imazen_v<MAJOR>_<MINOR>_<PATCH>` derived from the crate's
/// own `CARGO_PKG_VERSION`. Overridable at build time via the
/// `IWSSIM_IMPL_TAG` env var (e.g. a Burn port can set its own tag).
///
/// Why not just `iwssim`: a future port may differ on numerics by
/// 1e-3 or so without being wrong; a different column name documents
/// that drift instead of pretending two implementations agree. The
/// CLI flag (`--metric iwssim`) stays stable for users.
pub const IWSSIM_COLUMN_NAME: &str = match option_env!("IWSSIM_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "iwssim_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// Result of one IW-SSIM comparison.
#[derive(Debug, Clone, Copy)]
pub struct GpuIwssimResult {
    /// Final IW-SSIM score in `[0, 1]` — 1 = identical, lower = worse.
    pub score: f64,
    /// Per-scale weighted-mean contrast-structure values (paper notation
    /// `wmcs_j`). Useful for diagnostics — never aggregated outside the
    /// final `score`.
    pub per_scale: [f64; NUM_SCALES],
}

/// Errors that the GPU IW-SSIM pipeline can return.
#[derive(Debug, Clone)]
pub enum Error {
    /// Buffer length doesn't match the configured `width × height`.
    DimensionMismatch { expected: usize, got: usize },
    /// `compute_with_reference*` was called without a prior `set_reference`.
    NoCachedReference,
    /// Image too small for a 5-level pyramid + 11×11 valid blur. The
    /// paper's `iwssim.m` requires `min(W,H) >= 11 * 2^(Nsc-1) = 176`.
    InvalidImageSize,
    /// `set_reference` / `compute_with_reference` was called on an
    /// instance constructed via [`Iwssim::new_strip`]. The cached-
    /// reference fast path is only implemented for the whole-image
    /// pipeline; the strip path rebuilds the LP pyramid for both ref
    /// and dis on every `compute_gray_stripped` call. See
    /// `crates/iwssim-gpu/docs/STRIP_PROCESSING.md` § "Cached-reference
    /// path" for the design rationale and the open follow-up work.
    CachedRefNotSupportedInStripMode,
    /// `compute_gray_stripped` was called on an instance constructed
    /// via the whole-image [`Iwssim::new`] / [`Iwssim::with_config`]
    /// constructors. Use [`Iwssim::compute_gray`] for whole-image
    /// scoring, or construct the pipeline via [`Iwssim::new_strip`]
    /// for the strip-processing path.
    NotStripMode,
    /// The requested [`MemoryMode`](crate::MemoryMode) variant isn't
    /// implemented yet (e.g. `Tile {...}`).
    ModeUnsupported(&'static str),
    /// [`MemoryMode::Auto`](crate::MemoryMode) couldn't fit either
    /// Full or Strip into the VRAM cap.
    TooBigForFull { needed: usize, cap: usize },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Error::NoCachedReference => {
                write!(f, "no cached reference; call set_reference first")
            }
            Error::InvalidImageSize => write!(
                f,
                "image too small for 5-level IW-SSIM (min(W,H) must be ≥ 176)"
            ),
            Error::CachedRefNotSupportedInStripMode => write!(
                f,
                "cached-reference path is not implemented for strip mode; \
                 use compute_gray_stripped(ref, dis) instead"
            ),
            Error::NotStripMode => write!(
                f,
                "compute_gray_stripped requires an instance built via Iwssim::new_strip"
            ),
            Error::ModeUnsupported(variant) => write!(
                f,
                "MemoryMode::{variant} is not yet implemented in iwssim-gpu"
            ),
            Error::TooBigForFull { needed, cap } => write!(
                f,
                "Auto could not place image in {cap} byte cap; needs at least {needed} bytes \
                 (set ZENMETRICS_VRAM_CAP_BYTES or use MemoryMode::Strip explicitly)"
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
