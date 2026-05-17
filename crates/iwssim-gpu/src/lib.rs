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
pub mod pipeline;

pub use pipeline::Iwssim;

/// Number of pyramid scales — fixed at 5 by the IW-SSIM paper.
pub const NUM_SCALES: usize = 5;

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
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
