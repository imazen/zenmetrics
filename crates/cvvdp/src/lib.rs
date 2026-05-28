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
// CSF + masking constants imported from cvvdp v0.5.4 carry more digits
// than f32 can represent verbatim (LLVM rounds at compile time, so the
// values are mathematically identical to truncated forms). Allow the
// literal forms so we don't drift from the upstream JSON.
#![allow(clippy::excessive_precision)]

extern crate alloc;

// Phase 8c.1-B: params + presets + the JOD reference version live in
// this crate (CPU) as the canonical owner. cvvdp-gpu re-exports them
// to preserve existing `cvvdp_gpu::params::*` callsites.
pub mod params;
pub mod presets;

pub use params::{CvvdpParams, DisplayGeometry, DisplayModel, PerfMode};

/// The pinned [`gfxdisp/ColorVideoVDP`](https://github.com/gfxdisp/ColorVideoVDP)
/// reference version this implementation tracks for parity. Mirrors the
/// const in `cvvdp-gpu` (which re-exports this); the canonical owner is
/// the CPU crate per Phase 8c.1-B.
pub const PYCVVDP_REFERENCE_VERSION: &str = "v0.5.4";

/// Maximum pyramid depth supported by the kernel allocations.
/// Pinned by `cvvdp-gpu/tests/lib_constants.rs::max_levels_cap_at_nine`.
pub const MAX_LEVELS: usize = 9;

/// Smallest logical width/height at which the pyramid keeps building
/// further coarse levels. Pinned by `cvvdp-gpu/tests/lib_constants.rs`.
pub const PYRAMID_MIN_DIM: u32 = 4;

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

// Scalar kernel helpers (moved from cvvdp-gpu in Phase 8c.1-B).
pub mod kernels;

// Composed scalar reference pipeline (canonical "no GPU" impl).
pub mod host_scalar;

mod color;
#[allow(dead_code)]
mod csf;
pub mod diffmap;
mod masking;
mod pipeline;
mod pool;
mod pyramid;
mod scratch;
mod simd_math;
mod simd_pyramid;
pub(crate) mod strip;

pub use pipeline::Cvvdp;

/// TEST-ONLY re-exports of the `pub(crate)` SIMD kernel entry points.
///
/// Enabled only under the `__simd_equiv_test` cargo feature so the
/// brute-force SIMD-vs-scalar equivalence harness in
/// `tests/simd_equivalence.rs` (an external test crate) can drive the
/// kernels directly while keeping them `pub(crate)` for the production
/// API surface. `#[doc(hidden)]` keeps these out of rustdoc. This
/// module carries NO logic — it is a thin visibility shim.
#[cfg(feature = "__simd_equiv_test")]
#[doc(hidden)]
pub mod __simd_equiv_test_api {
    use alloc::vec::Vec;

    /// Kernel constants used by the scalar references in the harness.
    pub use crate::kernels::masking::PU_BLUR_KERNEL_1D;
    pub use crate::kernels::pyramid::GAUSS5;

    // Thin `pub fn` wrappers around the `pub(crate)` kernel entry
    // points. A `pub use` of a `pub(crate)` item is rejected by the
    // compiler (E0364), so we forward through wrappers instead. Each
    // body is a single delegating call — no logic change.

    /// SIMD σ=3 13-tap separable Gaussian blur (interior SIMD + scalar
    /// boundary patches). Mirrors `gaussian_blur_sigma3_simd`.
    #[inline]
    pub fn gaussian_blur_sigma3_simd(
        src: &[f32],
        w: usize,
        h: usize,
        h_pass: &mut Vec<f32>,
        dst: &mut Vec<f32>,
    ) {
        crate::simd_pyramid::gaussian_blur_sigma3_simd(src, w, h, h_pass, dst);
    }

    /// Horizontal pass of the σ=3 13-tap blur in isolation.
    #[inline]
    pub fn pu_blur_horizontal_pass(src: &[f32], w: usize, h: usize, h_pass: &mut [f32]) {
        crate::simd_pyramid::pu_blur_horizontal_pass(src, w, h, h_pass);
    }

    /// Vertical pass of the σ=3 13-tap blur in isolation.
    #[inline]
    pub fn pu_blur_vertical_pass(h_pass: &[f32], w: usize, h: usize, dst: &mut [f32]) {
        crate::simd_pyramid::pu_blur_vertical_pass(h_pass, w, h, dst);
    }

    /// Vertical pass of the pyramid 5-tap reduce.
    #[inline]
    pub fn reduce_vertical_pass(
        src: &[f32],
        sw: usize,
        sh: usize,
        dh: usize,
        vscratch: &mut [f32],
    ) {
        crate::simd_pyramid::reduce_vertical_pass(src, sw, sh, dh, vscratch);
    }

    /// Horizontal pass of the pyramid 5-tap reduce.
    #[inline]
    pub fn reduce_horizontal_pass(
        vscratch: &[f32],
        sw: usize,
        dw: usize,
        dh: usize,
        dst: &mut [f32],
    ) {
        crate::simd_pyramid::reduce_horizontal_pass(vscratch, sw, dw, dh, dst);
    }

    /// Vertical pass of the pyramid 5-tap expand (zero-insert).
    #[inline]
    pub fn expand_vertical_pass(
        src: &[f32],
        sw: usize,
        sh: usize,
        out_h: usize,
        vscratch: &mut [f32],
    ) {
        crate::simd_pyramid::expand_vertical_pass(src, sw, sh, out_h, vscratch);
    }

    /// Horizontal pass of the pyramid 5-tap expand (zero-insert).
    #[inline]
    pub fn expand_horizontal_pass(
        vscratch: &[f32],
        sw: usize,
        out_w: usize,
        out_h: usize,
        dst: &mut [f32],
        z_h_scratch: &mut Vec<f32>,
    ) {
        crate::simd_pyramid::expand_horizontal_pass(vscratch, sw, out_w, out_h, dst, z_h_scratch);
    }

    /// `out[i] = (xs[i] + offset)^p - offset_pow_p` (magetypes
    /// `pow_midp_unchecked` approximation).
    #[inline]
    pub fn safe_pow_with_offset_into(
        xs: &[f32],
        out: &mut [f32],
        offset: f32,
        p: f32,
        offset_pow_p: f32,
    ) {
        crate::simd_math::safe_pow_with_offset_into(xs, out, offset, p, offset_pow_p);
    }

    /// `out[i] = exp(xs[i])` (magetypes `exp_midp_unchecked`).
    #[inline]
    pub fn vexp_into(xs: &[f32], out: &mut [f32]) {
        crate::simd_math::vexp_into(xs, out);
    }

    /// `out[i] = ln(xs[i])` (magetypes `ln_midp_unchecked`, positive inputs).
    #[inline]
    pub fn vlog_into(xs: &[f32], out: &mut [f32]) {
        crate::simd_math::vlog_into(xs, out);
    }

    /// `out[i] = xs[i]^p` (magetypes `pow_midp_unchecked`, positive inputs).
    #[inline]
    pub fn vpow_into(xs: &[f32], out: &mut [f32], p: f32) {
        crate::simd_math::vpow_into(xs, out, p);
    }
}

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

/// `Result<T, cvvdp::Error>` — the crate's standard fallible
/// return type.
pub type Result<T> = core::result::Result<T, Error>;

// `ReferenceState` was removed in Phase 9.YA — the warm reference cache
// now lives directly in `Scratch` (DKL planes via `Scratch::ref_*`,
// per-channel weber pyramid via `Scratch::weber_ref`). The `Cvvdp::warm_active`
// boolean replaces the prior `Cvvdp::warm: Option<ReferenceState>` field.
// This drops 480 MB of per-warm_reference allocation at 40 MP.
