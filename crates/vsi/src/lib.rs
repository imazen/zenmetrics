//! Pure-Rust CPU implementation of **VSI** — the visual
//! saliency-induced index of Zhang, Shen & Li, "VSI: A Visual
//! Saliency-Induced Index for Perceptual Image Quality Assessment",
//! IEEE Trans. Image Processing 23(10):4270–4281, 2014.
//!
//! Ported for behavioral fidelity to the authors' `VSI.m` (the copy
//! distributed from the authors' own site, `cslinzhang.github.io/
//! home/VSI/`). The implementation is validated against that reference
//! under GNU Octave — see `validation/` for the generator.
//!
//! Pipeline, matching the reference exactly:
//!
//! 1. `SDSP` visual-saliency map per image: antialiased-bilinear
//!    resize of each RGB channel to 256×256 (a bit-identical port of
//!    octave-image `conv_interp_vec`, MATLAB `imresize`-compatible:
//!    triangle kernel broadened by `1/scale` on shrink, symmetric
//!    whole-point padding, full-kernel weight normalization), sRGB→
//!    CIE-Lab conversion using the reference's **D50** white point
//!    (`Xr=0.9642`, `Zr=0.8251`, `κ=903.3`), single-scale log-Gabor
//!    bandpass (`ω0=0.021`, `σf=1.34`) applied by 2-D FFT, magnitude
//!    across L/A/B, center prior `exp(−d²/145²)` and warm-color prior
//!    `1−exp(−d²/0.001²)`, resize back to the original size, then
//!    `mat2gray` normalization.
//! 2. Opponent channels `L = 0.06R + 0.63G + 0.27B`,
//!    `M = 0.30R + 0.04G − 0.35B`, `N = 0.34R − 0.60G + 0.17B`.
//! 3. Single-shot subsample `F = max(1, round(min(w,h)/256))` —
//!    `conv2(fspecial('average',F),'same')` + decimation on L, M, N and
//!    both saliency maps (`round`, single shot — the same rule
//!    `FR_FSIMc.m` uses — so F=2 already at min-dim 384).
//! 4. Scharr gradient magnitudes on L; similarity maps with constants
//!    `C_VS=1.27`, `C_GM=386`, `C_chrom=130`; quality map
//!    `gradSim^0.4 · VSSim · real((ISim·QSim)^0.02)` pooled against
//!    `weight = max(SM1, SM2)` (MATLAB NaN-propagating max — flat-color
//!    inputs yield a NaN score like the reference).
//!
//! Image-plane math is f32 with fixed-order f64 pooling; the SIMD
//! tiers share one macro body and are bit-identical by construction
//! (elementwise ops only — the resize/Lab/FFT helpers are scalar and
//! shared). The 2-D FFT is a self-contained radix-2/Bluestein
//! implementation vendored from `crates/hdrvdp/src/fft.rs` (see
//! `fft.rs`'s provenance note).

#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

use alloc::vec::Vec;

mod fft;
// Without `_dev` the `pub` items are unreachable; `dev` re-exports them then.
#[cfg_attr(not(feature = "_dev"), allow(unreachable_pub))]
mod kernel;

#[cfg(feature = "_dev")]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::vsi_core_tier_neon_t;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::vsi_core_tier_v3_t;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::vsi_core_tier_v4x_t;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::vsi_core_tier_wasm;
    pub use crate::kernel::{Planes, vsi_core_tier_scalar_t};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `vsi_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time via
/// `VSI_IMPL_TAG`).
pub const VSI_COLUMN_NAME: &str = match option_env!("VSI_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "vsi_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Errors returned by the VSI entry points.
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

impl core::error::Error for Error {}

/// Opponent-channel coefficients from `VSI.m` (ported verbatim — these
/// are the reference's own, not standard YIQ).
const L_COEF: [f32; 3] = [0.06, 0.63, 0.27];
const M_COEF: [f32; 3] = [0.30, 0.04, -0.35];
const N_COEF: [f32; 3] = [0.34, -0.60, 0.17];

/// The decimated planes one input image contributes to the kernel:
/// opponent `L/M/N` plus the raw `R/G/B` channels the saliency stage
/// (`SDSP`) resizes and converts to Lab.
pub(crate) struct Input {
    /// `0.06R + 0.63G + 0.27B`.
    pub l: Vec<f32>,
    /// `0.30R + 0.04G − 0.35B`.
    pub m: Vec<f32>,
    /// `0.34R − 0.60G + 0.17B`.
    pub n: Vec<f32>,
    /// Raw channels, three planes concatenated `[R | G | B]`.
    pub rgb: Vec<f32>,
}

/// VSI of two packed-sRGB8 RGB images. `stride` is in **bytes**.
///
/// Returns the reference's `sim`: ~[0, 1], 1 = identical. Constant or
/// flat-color inputs produce `NaN` — the saliency normalization divides
/// by a zero range, matching the reference's own `0/0` behavior.
pub fn vsi_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_len(reference, width, height, stride)?;
    check_len(distorted, width, height, stride)?;
    let (w, h) = (width, height);
    Ok(kernel::vsi_planes(
        &Input::from_rgb8(reference, w, h, stride),
        &Input::from_rgb8(distorted, w, h, stride),
        w,
        h,
        None,
    ))
}

/// VSI's per-pixel quality map — the reference's `qualitymap` output
/// (`SimMatrixC`: grad-sim^α · VS-sim · real((I-sim·Q-sim)^λ) · weight),
/// full input resolution. Higher = more similar.
///
/// Implements [`zenmetrics_abstractions::Diffmap`].
#[non_exhaustive]
pub struct VsiDiffmap {
    map: alloc::vec::Vec<f32>,
    /// Map width (input width — VSI's map is full-resolution).
    pub width: u32,
    /// Map height (input height).
    pub height: u32,
    /// The VSI score computed alongside the map.
    pub vsi: f64,
}

impl VsiDiffmap {
    /// The packed `width·height` quality map.
    pub fn map(&self) -> &[f32] {
        &self.map
    }
}

impl zenmetrics_abstractions::Diffmap for VsiDiffmap {
    fn pixels(&self) -> zenpixels::PixelCow<'_> {
        zenpixels::PixelCow::Borrowed(zenmetrics_abstractions::gray_f32_slice(
            &self.map,
            self.width,
            self.height,
        ))
    }

    fn polarity(&self) -> zenmetrics_abstractions::MapPolarity {
        zenmetrics_abstractions::MapPolarity::Similarity
    }

    fn score(&self) -> f64 {
        self.vsi
    }
}

/// [`vsi_rgb8`] plus the per-pixel quality map.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w·3`.
pub fn vsi_rgb8_with_diffmap(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, VsiDiffmap), Error> {
    check_len(reference, width, height, stride)?;
    check_len(distorted, width, height, stride)?;
    let (w, h) = (width, height);
    let mut map = alloc::vec![0.0f32; w * h];
    let score = kernel::vsi_planes(
        &Input::from_rgb8(reference, w, h, stride),
        &Input::from_rgb8(distorted, w, h, stride),
        w,
        h,
        Some(&mut map),
    );
    Ok((
        score,
        VsiDiffmap {
            map,
            width: w as u32,
            height: h as u32,
            vsi: score,
        },
    ))
}

impl Input {
    pub(crate) fn from_rgb8(data: &[u8], w: usize, h: usize, stride: usize) -> Self {
        let (l, m, n, rgb) = kernel::rgb8_planes(data, w, h, stride);
        Self { l, m, n, rgb }
    }
}

/// The VSI metric as a generic scorer — accepts `RGB8_SRGB`
/// [`zenpixels::PixelSlice`] inputs.
#[derive(Debug, Default, Clone, Copy)]
pub struct Vsi;

impl zenmetrics_abstractions::Metric for Vsi {
    fn name(&self) -> &'static str {
        "vsi"
    }

    fn score_kind(&self) -> zenmetrics_abstractions::ScoreKind {
        zenmetrics_abstractions::ScoreKind::Similarity
    }

    fn score(
        &mut self,
        r: zenpixels::PixelSlice<'_>,
        d: zenpixels::PixelSlice<'_>,
    ) -> zenmetrics_abstractions::MetricResult<zenmetrics_abstractions::MetricScores> {
        let (r, d) = zenmetrics_abstractions::rgb8_pair(r, d).map_err(|e| {
            alloc::boxed::Box::new(e) as alloc::boxed::Box<dyn core::error::Error + Send + Sync>
        })?;
        let s = vsi_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride).map_err(|e| {
            alloc::boxed::Box::new(e) as alloc::boxed::Box<dyn core::error::Error + Send + Sync>
        })?;
        Ok(zenmetrics_abstractions::MetricScores::single("vsi", s))
    }

    fn score_with_map(
        &mut self,
        r: zenpixels::PixelSlice<'_>,
        d: zenpixels::PixelSlice<'_>,
    ) -> zenmetrics_abstractions::MetricResult<(
        zenmetrics_abstractions::MetricScores,
        alloc::boxed::Box<dyn zenmetrics_abstractions::Diffmap>,
    )> {
        let (r, d) = zenmetrics_abstractions::rgb8_pair(r, d).map_err(|e| {
            alloc::boxed::Box::new(e) as alloc::boxed::Box<dyn core::error::Error + Send + Sync>
        })?;
        let (s, dm) = vsi_rgb8_with_diffmap(r.pixels, d.pixels, r.width, r.height, r.stride)
            .map_err(|e| {
                alloc::boxed::Box::new(e) as alloc::boxed::Box<dyn core::error::Error + Send + Sync>
            })?;
        Ok((
            zenmetrics_abstractions::MetricScores::single("vsi", s),
            alloc::boxed::Box::new(dm),
        ))
    }
}

fn check_len(buf: &[u8], w: usize, h: usize, stride: usize) -> Result<(), Error> {
    let needed = if h == 0 { 0 } else { stride * (h - 1) + w * 3 };
    if buf.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: buf.len(),
        });
    }
    Ok(())
}
