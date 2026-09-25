//! MAD — Most Apparent Distortion (E. C. Larson & D. M. Chandler,
//! *Most apparent distortion: full-reference image quality assessment
//! and the role of strategy*, JEI 19(1), 2010). Pure-Rust CPU port of
//! the authors' MATLAB release (`hi_index.m` + `lo_index.m` +
//! `ical_std.c`/`ical_stat.c`, distributed in the STMAD_2011 package
//! and archived at `Netflix/vmaf`), validated against it under GNU
//! Octave (`validation/`).
//!
//! MAD models two HVS strategies and blends them geometrically:
//! - **HI** (`hi_index`): appearance/detection of visible distortion
//!   — luminance transform `k·x^(2.2/3)`, Mannos–Sakrison CSF weighting
//!   in the DFT domain, blocky local statistics (`ical_std` C-mex:
//!   16×16 windows at stride 4, 8×8 min-pooled reference std), a
//!   contrast-threshold mask, and a 16×16 `imfilter`-'same' local MSE.
//! - **LO** (`lo_index`): near-threshold statistical deviation — a
//!   5-scale × 4-orientation Kovesi log-Gabor bank (minWaveLength 3,
//!   mult 3, sigmaOnf 0.55), per-subband blocky `ical_stat` maps
//!   (std / skew / kurtosis), weighted by `[0.5 0.75 1 5 6]/13.25`
//!   over scales.
//! - Combine (JEI paper): `sig = 1/(1+b1·HI^b2)`,
//!   `MAD = HI^sig · LO^(1−sig)`, `b1 = e^(−2.55/3.35)`,
//!   `b2 = 1/(ln10·3.35)`.
//!
//! MAD is a **distance**: 0 = identical, higher = more distorted,
//! unbounded above. Identical and constant inputs score `0.0`.
//!
//! # Floating-point discipline
//!
//! Plane data is `f32` end-to-end; one-time grids (CSF, log-Gabor)
//! and all reductions (block-stat numerators, the `Σ mp²` norms) are
//! `f64`. Unlike [`vif`], MAD's thresholds (`Ci = −5`, `G = 0.5`) are
//! signal-level — an `f32` pipeline reproduces the reference's mask
//! decisions (verified on the Octave goldens). Per-element
//! expressions are fixed-order `f32` independent of SIMD lane count,
//! so every tier (scalar / v3 / v4 / neon / wasm128) and every
//! `parallel` thread count produces **bit-identical** output. `_dev`
//! exposes the per-tier entries for tier-parity tests.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// `v += ...` requires `AddAssign`, which the magetypes vector types do
// not implement — the accumulation stays `v = v + ...` inside the
// shared macro bodies (same pattern as the sibling metric crates).
#![allow(clippy::assign_op_pattern)]
// Golden constants are transcribed from the reference implementation's
// output verbatim — keeping their full printed precision is intentional.
#![allow(clippy::excessive_precision)]

extern crate alloc;

use alloc::vec::Vec;

pub mod fft;
mod kernel;

#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::mad_core_tier_neon;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::mad_core_tier_v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::mad_core_tier_v4;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::mad_core_tier_wasm128;
    pub use crate::kernel::{Args, mad_core_tier_scalar};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `mad_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `MAD_IMPL_TAG`).
pub const MAD_COLUMN_NAME: &str = match option_env!("MAD_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "mad_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column-name identifier for the HI (visible-distortion) component:
/// `mad_hi_imazen_v<MAJOR>_<MINOR>_<PATCH>`.
pub const MAD_HI_COLUMN_NAME: &str = match option_env!("MAD_HI_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "mad_hi_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column-name identifier for the LO (near-threshold) component:
/// `mad_lo_imazen_v<MAJOR>_<MINOR>_<PATCH>`.
pub const MAD_LO_COLUMN_NAME: &str = match option_env!("MAD_LO_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "mad_lo_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Errors returned by the public scorers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Buffer shorter than `stride·(h−1) + w·3` (rgb8) or `+ w`
    /// (plane).
    BufferLength {
        /// Minimum byte length the caller must supply.
        needed: usize,
        /// Buffer length actually supplied.
        got: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::BufferLength { needed, got } => {
                write!(f, "buffer too short: need {needed} elements, got {got}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// MAD of a pair: the HI and LO strategy indices and the blended
/// score. All are distances — `0` = identical, higher = more
/// distorted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MadScore {
    /// `hi_index` — visible-distortion (appearance/masking) index.
    pub hi: f64,
    /// `lo_index` — near-threshold (statistical) index.
    pub lo: f64,
    /// `HI^sig · LO^(1−sig)` — the published combine.
    pub mad: f64,
}

/// MATLAB `rgb2gray` luma coefficients (`0.2989 0.5870 0.1140`) — the
/// conversion the reference pipeline applies to colour input.
const LUMA_COEF: [f32; 3] = [0.2989, 0.5870, 0.1140];

fn check_plane(buf: &[f32], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w };
    if buf.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: buf.len(),
        });
    }
    Ok(())
}

fn check_rgb(buf: &[u8], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w * 3 };
    if buf.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: buf.len(),
        });
    }
    Ok(())
}

/// MAD of two single-channel `f32` planes (0–255 signal scale),
/// packed row-major with explicit `stride` (≥ `w`). Images with
/// `min(w,h) < 34` leave no interior after the reference's edge kill
/// and score `0/0 = NaN`, as the reference does.
pub fn mad_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<MadScore, Error> {
    check_plane(reference, width, height, stride)?;
    check_plane(distorted, width, height, stride)?;
    let r = gather_plane(reference, width, height, stride);
    let d = gather_plane(distorted, width, height, stride);
    Ok(kernel::mad_score(&Inputs { r, d }, width, height))
}

/// MAD of two interleaved sRGB RGB8 pairs, on the MATLAB `rgb2gray`
/// luma plane (unrounded `0.2989/0.5870/0.1140` weights).
pub fn mad_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<MadScore, Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    Ok(kernel::mad_score(&inputs, width, height))
}

/// Gathered level-0 input planes.
pub(crate) struct Inputs {
    /// Reference plane, packed `w·h`.
    pub r: Vec<f32>,
    /// Distorted plane, packed `w·h`.
    pub d: Vec<f32>,
}

fn gather_plane(buf: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        out.extend_from_slice(&buf[y * stride..y * stride + w]);
    }
    out
}

impl Inputs {
    /// Luma planes from two packed RGB8 buffers (MATLAB `rgb2gray`,
    /// unrounded f32 result on the 0–255 scale).
    pub(crate) fn from_rgb8(
        reference: &[u8],
        distorted: &[u8],
        w: usize,
        h: usize,
        stride: usize,
    ) -> Self {
        let n = w * h;
        let (mut r, mut d) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for y in 0..h {
            let (rs, ds) = (
                &reference[y * stride..y * stride + 3 * w],
                &distorted[y * stride..y * stride + 3 * w],
            );
            for (&[pr, pg, pb], &[dr, dg, db]) in rs
                .as_chunks::<3>()
                .0
                .iter()
                .zip(ds.as_chunks::<3>().0.iter())
            {
                r.push(
                    LUMA_COEF[0] * pr as f32 + LUMA_COEF[1] * pg as f32 + LUMA_COEF[2] * pb as f32,
                );
                d.push(
                    LUMA_COEF[0] * dr as f32 + LUMA_COEF[1] * dg as f32 + LUMA_COEF[2] * db as f32,
                );
            }
        }
        Self { r, d }
    }
}
