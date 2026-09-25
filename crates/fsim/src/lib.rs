//! FSIM / FSIMc — Feature SIMilarity index for image quality.
//!
//! Pure-Rust CPU port of the reference implementation `FR_FSIMc.m`
//! (FSIM index with automatic downsampling, Version 1.0) by Lin Zhang,
//! Lei Zhang, Xuanqin Mou and David Zhang:
//!
//! > L. Zhang, L. Zhang, X. Mou & D. Zhang: *"FSIM: a feature similarity
//! > index for image quality assessment"*. IEEE Transactions on Image
//! > Processing 20(9), 2378–2386, 2011.
//!
//! The reference is licensed for educational/research use and is not
//! redistributed here; the port implements the published algorithm and
//! constants, verified against the reference under GNU Octave (see
//! `validation/`).
//!
//! Pipeline (exactly the reference's):
//! 1. sRGB → YIQ (unrounded `Y = 0.299R+0.587G+0.114B`, the reference's
//!    I and Q coefficient rows); grayscale input uses the plane as-is.
//! 2. Automatic downsampling: `F = max(1, round(min(w,h)/256))`; every
//!    channel is `conv2(·, ones(F,F)/F², 'same')` then decimated at the
//!    odd (1-based) indices — a forward-looking window for even `F`
//!    (the even-kernel `same` anchor), centered for odd `F`.
//! 3. Phase-congruency maps of the two Y planes (Kovesi's
//!    `phasecong2`: 4-scale log-Gabor × 4-orientation filterbank in the
//!    Fourier domain, `minWaveLength=6`, `mult=2`, `sigmaOnf=0.55`,
//!    `dThetaOnSigma=1.2`, Rayleigh noise threshold `k=2`, rescaled
//!    `T/1.7`, `epsilon=1e-4`). The transforms run on this crate's
//!    vendored self-contained FFT (`crate::fft`, from `hdrvdp`).
//! 4. Gradient magnitudes by the reference's fixed 3×3 kernels
//!    (`dx = [3 0 −3; 10 0 −10; 3 0 −3]/16`, `dy` its transpose).
//! 5. `PCSim = (2·PC1·PC2+T1)/(PC1²+PC2²+T1)` (T1=0.85) and
//!    `GSim = (2·G1·G2+T2)/(G1²+G2²+T2)` (T2=160); with
//!    `PCm = max(PC1,PC2)`, **FSIM** = `Σ(GSim·PCSim·PCm)/ΣPCm`.
//! 6. **FSIMc** additionally weights by
//!    `real((ISim·QSim)^0.03)` with `ISim = (2·I1·I2+T3)/(I1²+I2²+T3)`,
//!    `QSim` likewise, `T3 = T4 = 200` — for a negative product the
//!    real part of the complex power, `|z|^λ·cos(πλ)`, is used exactly
//!    as MATLAB's element-wise `.^` produces.
//!
//! Output range is roughly `[0,1]` (higher = more similar); identical
//! images score `1.0`. A constant pair degenerates the phase-congruency
//! maps (`AnAll = 0`) and the reference's `0/0` surfaces as `NaN`; the
//! port propagates IEEE `NaN` the same way.
//!
//! # Floating-point discipline
//!
//! Plane data is `f32` end-to-end (the FFT included — plane
//! quantisation only, never a different algorithm); every output pixel
//! is a fixed-order `f32` expression independent of the SIMD lane
//! count, and every scalar reduction (medians, noise sums, the final
//! pooling) is `f64` in fixed raster order. Every tier (scalar / v3 /
//! v4 / neon / wasm128) and every `parallel` thread count produces
//! **bit-identical** output; the four PC orientations are combined in
//! fixed orientation order. `_dev` exposes the per-tier entries for
//! tier-parity tests and disassembly.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// assign_op_pattern: the shared `*_body!` macros accumulate with
// `acc = acc + term` — `f32xN` has no `AddAssign`, so the suggested
// `+=` would not compile (and `x = x + y` keeps the fixed-order
// accumulation explicit). Same allow as crates/psnrhvs.
#![allow(clippy::assign_op_pattern)]

extern crate alloc;

use alloc::vec::Vec;

mod fft;
mod kernel;

/// Per-tier entry points for tier-parity tests and disassembly. Not API.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::fsim_core_tier_v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::fsim_core_tier_v4;
    pub use crate::kernel::{Planes, fsim_core_tier_scalar};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `fsim_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `FSIM_IMPL_TAG`). The `fsim` CLI metric emits both the FSIM and
/// FSIMc scores of the two sRGB8 images.
pub const FSIM_COLUMN_NAME: &str = match option_env!("FSIM_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "fsim_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **color** variant score (`fsimc` column of the
/// `fsim` CLI metric): `fsimc_imazen_v<MAJOR>_<MINOR>_<PATCH>`
/// (`FSIMC_IMPL_TAG`). FSIMc multiplies the FSIM map by
/// `real((ISim·QSim)^0.03)` on the I/Q chroma planes.
pub const FSIMC_COLUMN_NAME: &str = match option_env!("FSIMC_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "fsimc_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **luma-only** `fsim-y` CLI metric
/// (`fsimy_imazen_v<MAJOR>_<MINOR>_<PATCH>`, `FSIMY_IMPL_TAG`) — the
/// grayscale path on unrounded BT.601 luma, no chroma term.
pub const FSIMY_COLUMN_NAME: &str = match option_env!("FSIMY_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "fsimy_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Errors returned by the FSIM entry points.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Input slice is too short for the declared geometry.
    BufferLength {
        /// Bytes/elements the caller must supply.
        needed: usize,
        /// Bytes/elements actually supplied.
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

/// The two scores an RGB comparison produces. FSIM uses only the luma
/// phase-congruency/gradient maps; FSIMc adds the I/Q chroma weighting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
    /// FSIM — grayscale feature similarity (`~[0,1]`, 1 = identical).
    pub fsim: f64,
    /// FSIMc — FSIM with the I/Q chroma similarity term (same scale).
    pub fsimc: f64,
}

/// FSIM of two strided `f32` planes on the 0..255 scale.
///
/// Returns the reference's `FSIM` output (phase-congruency × gradient
/// similarity weighted by `max(PC1,PC2)`). The reference's automatic
/// downsampling applies: `F = max(1, round(min(w,h)/256))`.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w`.
pub fn fsim_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_len(reference, width, height, stride)?;
    check_len(distorted, width, height, stride)?;
    Ok(kernel::fsim_planes(
        reference, distorted, width, height, stride,
    ))
}

/// FSIM and FSIMc of two packed sRGB8 images (`rgb` triplets, `stride`
/// bytes per row).
///
/// The reference's pipeline: unrounded YIQ conversion on full
/// resolution, per-channel `F`-factor downsampling, phase-congruency on
/// the luma planes, I/Q similarity maps on the decimated chroma.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w·3`.
pub fn fsim_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<Scores, Error> {
    check_len_rgb(reference, width, height, stride)?;
    check_len_rgb(distorted, width, height, stride)?;
    let (fsim, fsimc) = kernel::fsim_rgb8_planes(reference, distorted, width, height, stride);
    Ok(Scores { fsim, fsimc })
}

/// FSIM of the unrounded BT.601 luma of two packed sRGB8 images
/// (`Y = 0.299R + 0.587G + 0.114B`, grayscale path — identical to
/// [`fsim_plane_f32`] on the converted plane).
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w·3`.
pub fn fsim_luma8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_len_rgb(reference, width, height, stride)?;
    check_len_rgb(distorted, width, height, stride)?;
    Ok(kernel::fsim_luma8_planes(
        reference, distorted, width, height, stride,
    ))
}

fn check_len(data: &[f32], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w };
    if w == 0 || h == 0 || data.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: data.len(),
        });
    }
    Ok(())
}

fn check_len_rgb(data: &[u8], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w * 3 };
    if w == 0 || h == 0 || data.len() < needed || stride < w * 3 {
        return Err(Error::BufferLength {
            needed,
            got: data.len(),
        });
    }
    Ok(())
}

/// sRGB → unrounded BT.601 luma (`Y`), the reference's Y row.
pub(crate) const Y_COEF: [f32; 3] = [0.299, 0.587, 0.114];
/// sRGB → I chroma, the reference's I row.
pub(crate) const I_COEF: [f32; 3] = [0.596, -0.274, -0.322];
/// sRGB → Q chroma, the reference's Q row.
pub(crate) const Q_COEF: [f32; 3] = [0.211, -0.523, 0.312];

/// Build a `w`×`h` f32 plane from strided input rows.
pub(crate) fn plane_from_f32(data: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        out.extend_from_slice(&data[y * stride..y * stride + w]);
    }
    out
}

/// Build a `w`×`h` f32 plane from packed sRGB8 rows through `coef`
/// (unrounded — the reference keeps the `double` products).
pub(crate) fn plane_from_rgb8(
    data: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    coef: [f32; 3],
) -> Vec<f32> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        let row = &data[y * stride..y * stride + w * 3];
        for px in row.as_chunks::<3>().0 {
            out.push(coef[0] * px[0] as f32 + coef[1] * px[1] as f32 + coef[2] * px[2] as f32);
        }
    }
    out
}
