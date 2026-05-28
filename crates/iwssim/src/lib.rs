//! Pure-Rust CPU port of **IW-SSIM** (Information-Content Weighted SSIM)
//! — Wang & Li, *IEEE TIP* vol. 20 no. 5, May 2011.
//!
//! Tracks the canonical Python reference at
//! <https://github.com/Jack-guo-xy/Python-IW-SSIM> commit
//! `f9de37cd` (the only commit at the time of port).
//!
//! # Algorithm
//!
//! 1. Convert RGB → BT.601 rounded grayscale on the host (or accept
//!    grayscale floats directly via [`Iwssim::score_gray`]).
//! 2. Build a **5-level Laplacian pyramid** using pyrtools' `binom5`
//!    filter (`sqrt(2)·[1,4,6,4,1]/16`) with `reflect1` boundary.
//!    Bands `L_1..L_4` are real Laplacians, `L_5` is the residual
//!    lowpass.
//! 3. For each scale, compute the 11×11 Gaussian (σ=1.5)
//!    contrast-structure map `cs_j = (2σ_{12} + C₂) / (σ₁² + σ₂² + C₂)`
//!    with `C₂ = (0.03·255)²`. At the coarsest scale also compute the
//!    luminance map `l_5 = (2µ₁µ₂ + C₁) / (µ₁² + µ₂² + C₁)` with
//!    `C₁ = (0.01·255)²`.
//! 4. For scales 1..4, compute the **information-content weight map**
//!    via the GSM model (paper §II): 3×3 box statistics, a parent band
//!    from `imenlarge2`(`L_{j+1}`), an `(N=9 or 10)×(N=9 or 10)`
//!    covariance eigendecomposition, and per-pixel mutual information.
//! 5. Pool each scale: `wmcs_j = Σ(cs_j · w_j) / Σ(w_j)` for `j<5`
//!    (after cropping `w_j` by `bound1 = 4`), `wmcs_5 = mean(cs_5 · l_5)`.
//! 6. Final score: `Π_{j=1}^{5} |wmcs_j|^{β_j}` with
//!    `β = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`.
//!
//! # Public API
//!
//! - [`Iwssim::new`] constructs a scorer for `width × height` images
//!   with default [`IwssimParams`].
//! - [`Iwssim::with_params`] constructs with custom params (parent
//!   band on/off, IW pooling on/off, etc.).
//! - [`Iwssim::score`] / [`Iwssim::score_gray`] one-shot scoring.
//! - [`Iwssim::warm_reference`] / [`Iwssim::score_with_warm_ref`] —
//!   per-batch reference reuse. Skips the ref-side pyramid + per-scale
//!   covariance eigendecomposition.
//!
//! # SIMD coverage
//!
//! magetypes-dispatched (`#[magetypes(_v4x, v4, v3, neon, wasm128)]`):
//!
//! - `binom5` separable blur (pyramid build + expand).
//! - `gaussian_11x11` separable blur (SSIM stats).
//! - `box_3x3` separable blur (IW weight map first-pass stats).
//! - Per-pixel reductions for `cs`, `l`, weighted sums.
//!
//! Scalar fallback covers every kernel; SIMD is an optimization on top.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::excessive_precision)]

extern crate alloc;

use alloc::vec::Vec;

mod eig;
mod filters;
mod params;
mod pipeline;
mod pyramid;
mod ssim;
mod weights;

pub use params::IwssimParams;
pub use pipeline::Iwssim;

/// Number of pyramid scales — fixed at 5 by the IW-SSIM paper.
pub const NUM_SCALES: usize = 5;

/// Minimum native pyramid dimension required by the reference algorithm.
///
/// `iwssim.m` requires `min(W, H) ≥ 11 · 2^(Nsc-1) = 176` so the
/// coarsest scale (`L_5`) still has enough pixels for a valid-mode
/// 11×11 Gaussian. We mirror the GPU port's behavior: reject by default,
/// or tile via the params.
pub const MIN_NATIVE_DIM: u32 = 176;

/// Stable column-name identifier for sweep sidecars.
///
/// `iwssim_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>` — distinct namespace
/// from `iwssim_imazen_v*` (iwssim-gpu) and a hypothetical
/// `iwssim_python_v*` reference column. The two impls produce scores
/// within atomic-tolerance of each other on the GPU path, but the
/// column name disambiguates them in joined parquets if a future
/// divergence needs traceability.
pub const IWSSIM_COLUMN_NAME: &str = match option_env!("IWSSIM_CPU_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "iwssim_cpu_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// One IW-SSIM comparison result.
#[derive(Debug, Clone, Copy)]
pub struct IwssimScore {
    /// Final IW-SSIM score in `[0, 1]` — 1 = identical, lower = worse.
    pub score: f64,
    /// Per-scale weighted-mean contrast-structure values (paper notation
    /// `wmcs_j`). Useful for diagnostics — never aggregated outside the
    /// final `score`.
    pub per_scale: [f64; NUM_SCALES],
}

/// Failure modes for [`Iwssim::*`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Buffer length doesn't match the configured `width × height × 3`
    /// (RGB sRGB-u8 path) or `width × height` (gray-f32 path).
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Got length.
        got: usize,
    },
    /// `score_with_warm_ref` called before `warm_reference`.
    NoWarmReference,
    /// Image too small for a 5-level pyramid + 11×11 valid blur. The
    /// paper requires `min(W, H) >= 11 * 2^(Nsc-1) = 176`. To accept
    /// smaller inputs, set `IwssimParams::allow_small = true`.
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
            Error::NoWarmReference => {
                write!(f, "no warm reference; call warm_reference first")
            }
            Error::InvalidImageSize { width, height } => write!(
                f,
                "image too small for 5-level IW-SSIM: {width}×{height} \
                 (need min dim >= 176, or enable allow_small)"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// `Result<T, iwssim::Error>` — crate-wide fallible return type.
pub type Result<T> = core::result::Result<T, Error>;

/// Convert sRGB-u8 RGB → BT.601 rounded grayscale (single channel f32).
///
/// Matches `utils.rgb2gray` in the Python reference:
/// `gray = round(0.2989·R + 0.5870·G + 0.1140·B)`.
///
/// `rgb` is `width * height * 3` bytes packed `R0,G0,B0, R1,G1,B1, ...`.
/// `out` is `width * height` f32s.
///
/// # Panics
///
/// Panics if `rgb.len() != out.len() * 3`.
#[inline]
pub fn rgb_u8_to_gray_bt601(rgb: &[u8], out: &mut [f32]) {
    assert_eq!(rgb.len(), out.len() * 3, "rgb len mismatch");
    for (px, o) in rgb.chunks_exact(3).zip(out.iter_mut()) {
        let r = px[0] as f32;
        let g = px[1] as f32;
        let b = px[2] as f32;
        // Round-half-to-even matches numpy's `np.round`. The fractional
        // part is at most ~0.99 here so the exact rounding mode rarely
        // matters for parity, but BT.601 uses banker's rounding too.
        let v = 0.2989 * r + 0.5870 * g + 0.1140 * b;
        *o = round_half_to_even(v);
    }
}

/// Allocate + convert; convenience wrapper around [`rgb_u8_to_gray_bt601`].
pub fn rgb_u8_to_gray_bt601_vec(rgb: &[u8]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut out = alloc::vec![0.0f32; n];
    rgb_u8_to_gray_bt601(rgb, &mut out);
    out
}

#[inline]
fn round_half_to_even(v: f32) -> f32 {
    // f32::round_ties_even() is stable since 1.77; we set MSRV high
    // enough above. NB: numpy's `np.round` is banker's rounding for
    // even-half ties — this matches.
    v.round_ties_even()
}
