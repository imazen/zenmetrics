//! Pure-Rust CPU port of ColorVideoVDP (cvvdp) still-image scoring.
//!
//! Targets pycvvdp v0.5.4 within `≤ 1e-3 JOD` on the canonical
//! goldens. Built for replacing butteraugli in JPEG XL encoder's
//! iterative quantization loop, so the API also yields a per-pixel
//! diffmap whose Minkowski-p norm folds back to the same JOD scalar.
//!
//! Algorithm shape (matches host_scalar in `cvvdp-gpu`):
//!
//! 1. sRGB byte → linear → DKLd65 opponent `(A, RG, VY)`.
//! 2. Weber-contrast pyramid per channel (Burt-Adelson reduce/expand,
//!    5-tap separable Gaussian, ceil-halving, ~7 levels for 1 MP).
//! 3. Per-pixel CSF sensitivity lookup (32×32 castleCSF LUT).
//! 4. Mult-mutual masking with cross-channel pool + phase-uncertainty.
//! 5. Minkowski pool (spatial β=2, band β=4, channel β=4) → JOD.
//!
//! Public API:
//!
//! - [`Cvvdp::new`] constructs a scorer with persistent scratch.
//! - [`Cvvdp::score`] / [`Cvvdp::score_with_diffmap`] one-shot scoring.
//! - [`Cvvdp::warm_reference`] / [`Cvvdp::score_with_warm_ref`] —
//!   per-buttloop iteration reuse.
//! - [`Cvvdp::score_from_linear_planes`] — direct planar `f32` entry
//!   to avoid `sRGB byte → linear` round-trip on encoder paths that
//!   already carry linear `f32` planes (JPEG XL).
//!
//! The diffmap is the per-pixel scalar `D_px` whose Minkowski-p_band
//! norm (with per-channel weights applied) yields the same `Q` that
//! `met2jod` converts to the scalar JOD. Identical inputs → all-zero
//! diffmap. See `diffmap` module + `tests/diffmap_invariants.rs`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;

use alloc::vec::Vec;

// Re-export upstream cvvdp-gpu params + JOD reference version so
// callers can construct a CvvdpParams without an extra dependency
// import. (The CPU port re-uses the same constants — there's exactly
// one canonical set.)
pub use cvvdp_gpu::params::{
    CvvdpParams, DisplayGeometry, DisplayModel, PerfMode,
};
pub use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;

/// Stable column-name identifier for sweep sidecars.
///
/// `cvvdp_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>` — distinct namespace
/// from `cvvdp_imazen_v*` (cvvdp-gpu) and the canonical
/// `cvvdp_pycvvdp_v054`. The two impls produce scores within ≤ 1e-3
/// JOD of each other but the column name distinguishes them in
/// case a future divergence needs traceability.
pub const CVVDP_COLUMN_NAME: &str = match option_env!("CVVDP_CPU_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "cvvdp_cpu_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// Number of color channels in DKL opponent space (achromatic +
/// red-green + violet-yellow).
pub const N_CHANNELS: usize = 3;

mod color;
pub mod diffmap;
mod masking;
mod pipeline;
mod pool;
mod pyramid;
mod scratch;

pub use pipeline::Cvvdp;

/// Failure modes for `Cvvdp::*`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Buffer length doesn't match `width × height × 3`.
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Got length.
        got: usize,
    },
    /// Plane length doesn't match `padded_width × height` (or for
    /// row-tight `width × height`).
    PlaneShapeMismatch {
        /// Expected length.
        expected: usize,
        /// Got length.
        got: usize,
    },
    /// `score_with_warm_ref*` called before `warm_reference`.
    NoWarmReference,
    /// Image too small for the cvvdp pyramid (`min(w, h) < 8` since
    /// the smallest gauss pyramid baseband is 4×4).
    InvalidImageSize {
        /// Width passed.
        width: u32,
        /// Height passed.
        height: u32,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: expected {expected} bytes, got {got}"
            ),
            Error::PlaneShapeMismatch { expected, got } => write!(
                f,
                "plane shape mismatch: expected {expected} f32, got {got}"
            ),
            Error::NoWarmReference => {
                write!(f, "no warm reference; call warm_reference first")
            }
            Error::InvalidImageSize { width, height } => write!(
                f,
                "image too small for cvvdp pyramid: {width}×{height} (need min dim ≥ 8)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// `Result<T, cvvdp_cpu::Error>` — the crate's standard fallible
/// return type.
pub type Result<T> = core::result::Result<T, Error>;

/// Owned per-channel sRGB-source-derived state used by warm-reference
/// + diffmap output APIs.
///
/// `w`, `h`, `planes`, and `display` are retained for future
/// debug-inspection (`Cvvdp::warm_inspect`) — currently unused but
/// pinned so the layout is stable.
#[allow(dead_code)]
pub(crate) struct ReferenceState {
    /// Image dimensions.
    pub w: usize,
    /// Image dimensions.
    pub h: usize,
    /// DKL planes for the reference (A, RG, VY).
    pub planes: [Vec<f32>; 3],
    /// Per-channel weber pyramid bands.
    pub weber: [pyramid::WeberPyramid; 3],
    /// Pre-computed display-derived constants for re-entry.
    pub display: DisplayModel,
}
