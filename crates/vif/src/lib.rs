//! VIFp — pixel-domain Visual Information Fidelity (H. R. Sheikh &
//! A. C. Bovik, *Image information and visual quality*, IEEE TIP
//! 15(2), 2006). Pure-Rust CPU port of the authors' multi-scale
//! scalar-GSM release `vifp_mscale.m`, validated against it under GNU
//! Octave (`validation/`).
//!
//! Pipeline, matching the reference: four scales; at each scale an
//! N-tap Gaussian (`N = 2^(5−scale)+1` → 17, 9, 5, 3, `σ = N/5`,
//! rank-1 separable) produces `filter2 'valid'` mean/variance/
//! covariance statistics; a scalar GSM regression gives `g` and
//! `sv_sq` through the reference's exact masking chain; the
//! information sums accumulate
//! `Σ log10(1 + g²·σ1²/(sv² + σn²))` and `Σ log10(1 + σ1²/σn²)` with
//! `σn² = 2`. Between scales the images are `filter2 'valid'` +
//! `1:2:end` decimated by that scale's own kernel. `vifp = num/den`.
//!
//! The reference has no minimum-size check: a scale whose `'valid'`
//! map is empty contributes zero to both accumulators, and if every
//! scale is empty (`min(w,h) < 17`) the score is `0/0 = NaN` — this
//! port propagates the same `NaN` rather than inventing an error.
//!
//! Output is ≥ 0; identical inputs score `1.0`. VIF can exceed `1.0`
//! on sharpened inputs (more "information" extracted than present —
//! the reference's own behaviour).
//!
//! # Floating-point discipline
//!
//! The pipeline runs in **`f64`** end-to-end (unlike the `f32`
//! sibling crates): the reference's `1e-10` degenerate masks sit four
//! orders above f64's noise floor on a 0–255 signal, while the
//! `E[x²]−μ²` cancellation error in `f32` is ~`1e-3` there — an `f32`
//! statistics plane would defeat every mask and flip the `NaN`
//! degenerate cases. Inputs are still `f32` planes / `u8` pixels
//! (cast on gather); every filter output and map element is a
//! fixed-order `f64` expression independent of SIMD lane count, and
//! the information sums accumulate `f64` in fixed raster order.
//! Every tier (scalar / v3 / v4 / neon / wasm128) and every
//! `parallel` thread count produces **bit-identical** output. `_dev`
//! exposes the per-tier entries for tier-parity tests and
//! disassembly.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// `v += ...` requires `AddAssign`, which the magetypes vector types do
// not implement — the accumulation stays `v = v + ...` inside the
// shared macro bodies (same pattern as the sibling metric crates).
#![allow(clippy::assign_op_pattern)]

extern crate alloc;

use alloc::vec::Vec;

mod kernel;

#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::vifp_core_tier_neon;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::vifp_core_tier_v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::vifp_core_tier_v4;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::vifp_core_tier_wasm128;
    pub use crate::kernel::{Args, vifp_core_tier_scalar};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `vif_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `VIF_IMPL_TAG`).
pub const VIF_COLUMN_NAME: &str = match option_env!("VIF_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "vif_imazen_v",
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

/// VIFp of two single-channel `f32` planes (0–255 signal scale),
/// packed row-major with explicit `stride` (≥ `w`). The values are
/// cast to `f64` on gather — the statistics pipeline is `f64`
/// (see the crate docs). Images with `min(w,h) < 17` score `NaN`, as
/// the reference does.
pub fn vif_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_plane(reference, width, height, stride)?;
    check_plane(distorted, width, height, stride)?;
    let r = gather_plane(reference, width, height, stride);
    let d = gather_plane(distorted, width, height, stride);
    Ok(kernel::vifp_core(&Inputs { r, d }, width, height))
}

/// VIFp of two interleaved sRGB RGB8 pairs, on the MATLAB `rgb2gray`
/// luma plane (unrounded `0.2989/0.5870/0.1140` weights).
pub fn vif_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    Ok(kernel::vifp_core(&inputs, width, height))
}

/// Gathered level-0 input planes (`f64` — see the crate docs).
pub(crate) struct Inputs {
    /// Reference plane, packed `w·h`.
    pub r: Vec<f64>,
    /// Distorted plane, packed `w·h`.
    pub d: Vec<f64>,
}

fn gather_plane(buf: &[f32], w: usize, h: usize, stride: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        out.extend(buf[y * stride..y * stride + w].iter().map(|&v| v as f64));
    }
    out
}

impl Inputs {
    /// Luma planes from two packed RGB8 buffers (MATLAB `rgb2gray`,
    /// unrounded `f64` result on the 0–255 scale).
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
                    LUMA_COEF[0] as f64 * pr as f64
                        + LUMA_COEF[1] as f64 * pg as f64
                        + LUMA_COEF[2] as f64 * pb as f64,
                );
                d.push(
                    LUMA_COEF[0] as f64 * dr as f64
                        + LUMA_COEF[1] as f64 * dg as f64
                        + LUMA_COEF[2] as f64 * db as f64,
                );
            }
        }
        Self { r, d }
    }
}
