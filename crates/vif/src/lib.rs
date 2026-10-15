//! VIFp — pixel-domain Visual Information Fidelity (H. R. Sheikh &
//! A. C. Bovik, *Image information and visual quality*, IEEE TIP
//! 15(2), 2006). Pure-Rust CPU port of the authors' multi-scale
//! scalar-GSM release `vifp_mscale.m`, validated against it under GNU
//! Octave (`validation/`).
//!
//! The UT Austin copyright notice that the LIVE releases require in all
//! copies is in `LICENSE-live-vif` next to this crate's manifest;
//! the matlabPyrTools (MIT) notice for `vifvec`'s filter taps is in
//! `LICENSE-matlabPyrTools`.
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
//! Every tier (scalar / v3 / v4x / neon / wasm128) and every
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

use zenmetrics_abstractions as za;

use alloc::vec::Vec;

// Without `_dev` the `pub` items are unreachable; `dev` re-exports them then.
#[cfg_attr(not(feature = "_dev"), allow(unreachable_pub))]
mod kernel;
mod vifvec;

#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::vifp_core_tier_neon_t;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::vifp_core_tier_v3_t;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::vifp_core_tier_v4x_t;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::vifp_core_tier_wasm128_t;
    pub use crate::kernel::{Args, vifp_core_tier_scalar_t};
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

/// Stable column-name identifier for the vector-GSM `vifvec` variant —
/// `vifvec_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable via
/// `VIFVEC_IMPL_TAG`). Kept distinct from [`VIF_COLUMN_NAME`]: these
/// are *different metrics* that share the name "VIF" in the
/// literature (see the [`vifvec`] module docs).
pub const VIFVEC_COLUMN_NAME: &str = match option_env!("VIFVEC_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "vifvec_imazen_v",
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
    /// `min(width, height)` below the variant's pyramid floor —
    /// `vifvec` needs ≥ 72 (the reference's `maxPyrHt` cannot build a
    /// 4-level SP5 pyramid below that and hard-errors).
    TooSmall {
        /// Minimum `min(width, height)` the variant accepts.
        needed: usize,
        /// `min(width, height)` actually supplied.
        got: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::BufferLength { needed, got } => {
                write!(f, "buffer too short: need {needed} elements, got {got}")
            }
            Error::TooSmall { needed, got } => {
                write!(f, "image too small: need min(w,h) >= {needed}, got {got}")
            }
        }
    }
}

impl core::error::Error for Error {}

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
    Ok(kernel::vifp_core(&Inputs { r, d }, width, height, None))
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
    Ok(kernel::vifp_core(&inputs, width, height, None))
}

/// [`vif_rgb8`] plus every non-empty scale's per-element information terms.
fn vif_rgb8_terms(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, Vec<kernel::ScaleTerms>), Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    let mut terms = Vec::new();
    let score = kernel::vifp_core(&inputs, width, height, Some(&mut terms));
    Ok((score, terms))
}

/// Information-loss map names, one per VIFp scale.
const VIFP_NAMES: [&str; 4] = ["info_loss_0", "info_loss_1", "info_loss_2", "info_loss_3"];

/// VIFp numerator / denominator map names, one per scale.
const VIFP_NUM: [&str; 4] = ["num_0", "num_1", "num_2", "num_3"];
const VIFP_DEN: [&str; 4] = ["den_0", "den_1", "den_2", "den_3"];

/// A scored pair with its information maps: the losses (Difference)
/// first, then the raw numerator and denominator information
/// (Feature), each on the same grids.
struct VifMaps {
    vif: f64,
    names: Vec<&'static str>,
    maps: Vec<Vec<f32>>,
    geometry: Vec<za::MapGeometry>,
    polarity: Vec<za::MapPolarity>,
}

impl VifMaps {
    fn empty(vif: f64) -> Self {
        Self {
            vif,
            names: Vec::new(),
            maps: Vec::new(),
            geometry: Vec::new(),
            polarity: Vec::new(),
        }
    }

    /// Append `loss`, then all numerators, then all denominators.
    fn push_all(
        &mut self,
        loss: Vec<(&'static str, Vec<f32>, za::MapGeometry)>,
        num: Vec<(&'static str, Vec<f32>, za::MapGeometry)>,
        den: Vec<(&'static str, Vec<f32>, za::MapGeometry)>,
    ) {
        for (group, polarity) in [
            (loss, za::MapPolarity::Difference),
            (num, za::MapPolarity::Feature),
            (den, za::MapPolarity::Feature),
        ] {
            for (name, map, g) in group {
                self.names.push(name);
                self.maps.push(map);
                self.geometry.push(g);
                self.polarity.push(polarity);
            }
        }
    }
}

impl VifMaps {
    /// VIFp scale `s` samples the input at `c_s + 2^s·i` with
    /// `c_s = 8 + 4s`: scale `s > 0` is the previous plane filtered
    /// 'valid' by this scale's own `N = 2^(4−s)+1`-tap window and
    /// decimated `1:2:end`, and the statistics grid is that window's
    /// 'valid' output. Cells are `2^s` blocks around those centres. A value
    /// reads the input within `8 + 4s` pixels of its centre.
    fn vifp(vif: f64, terms: Vec<kernel::ScaleTerms>, w: usize, h: usize) -> Self {
        let (mut loss, mut num, mut den) = (Vec::new(), Vec::new(), Vec::new());
        for t in terms {
            let cell = 1u32 << t.scale;
            let centre = 8 + 4 * t.scale as i64;
            let halo = 8 + 4 * t.scale as u32;
            let g = za::MapGeometry::grid(
                w as u32,
                h as u32,
                centre - i64::from(cell / 2),
                centre - i64::from(cell / 2),
                cell,
                cell,
                t.w as u32,
                t.h as u32,
            )
            .with_halo(halo, halo);
            let narrow = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let l = t
                .den
                .iter()
                .zip(&t.num)
                .map(|(&d, &n)| (d - n) as f32)
                .collect();
            loss.push((VIFP_NAMES[t.scale], l, g));
            num.push((VIFP_NUM[t.scale], narrow(&t.num), g));
            den.push((VIFP_DEN[t.scale], narrow(&t.den), g));
        }
        let mut out = Self::empty(vif);
        out.push_all(loss, num, den);
        out
    }

    /// `vifvec`'s level `L` (1-based) samples the input at `2^(L−1)·k`
    /// (`corrDn` start 0, the inter-level lowpass at stride 2), and each
    /// block is 3×3 subband samples, so cell `i` is block `i + offset` of
    /// the cropped grid: `3·2^(L−1)` pixels starting `2^(L−2)` before its
    /// first sample's position. The S-field and eigenvalues come from the
    /// whole subband's covariance, so support is global.
    fn vifvec(vif: f64, subs: Vec<vifvec::SubTerms>, w: usize, h: usize) -> Self {
        let (mut loss, mut num, mut den) = (Vec::new(), Vec::new(), Vec::new());
        // `SUBS` runs coarse to fine; serve the finest first.
        for t in subs.into_iter().rev() {
            let step = 1u32 << (t.level - 1);
            let cell = 3 * step;
            let origin = i64::from(cell) * t.offset as i64 - i64::from(step / 2);
            let g = za::MapGeometry::grid(
                w as u32, h as u32, origin, origin, cell, cell, t.w as u32, t.h as u32,
            )
            .with_support(za::Support::Global);
            let k = (t.level - 1) * 2 + usize::from(t.band == 4);
            let narrow = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
            let l = t
                .den
                .iter()
                .zip(&t.num)
                .map(|(&d, &n)| (d - n) as f32)
                .collect();
            loss.push((VIFVEC_NAMES[k], l, g));
            num.push((VIFVEC_NUM[k], narrow(&t.num), g));
            den.push((VIFVEC_DEN[k], narrow(&t.den), g));
        }
        let mut out = Self::empty(vif);
        out.push_all(loss, num, den);
        out
    }

    fn view(&self, i: usize) -> Option<za::MapView<'_>> {
        let g = self.geometry[i];
        let px = za::gray_f32_slice(&self.maps[i], g.cells_x, g.cells_y).ok()?;
        za::MapView::new(self.names[i], px, self.polarity[i], g).ok()
    }
}

/// `vifvec` numerator / denominator map names, in [`VIFVEC_NAMES`] order.
const VIFVEC_NUM: [&str; 8] = [
    "num_l1_b1",
    "num_l1_b4",
    "num_l2_b1",
    "num_l2_b4",
    "num_l3_b1",
    "num_l3_b4",
    "num_l4_b1",
    "num_l4_b4",
];
const VIFVEC_DEN: [&str; 8] = [
    "den_l1_b1",
    "den_l1_b4",
    "den_l2_b1",
    "den_l2_b4",
    "den_l3_b1",
    "den_l3_b4",
    "den_l4_b1",
    "den_l4_b4",
];

/// `vifvec` map names by `(level, band)`, finest level first.
const VIFVEC_NAMES: [&str; 8] = [
    "info_loss_l1_b1",
    "info_loss_l1_b4",
    "info_loss_l2_b1",
    "info_loss_l2_b4",
    "info_loss_l3_b1",
    "info_loss_l3_b4",
    "info_loss_l4_b1",
    "info_loss_l4_b4",
];

impl za::Evaluation for VifMaps {
    fn score(&self) -> f64 {
        self.vif
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(core::iter::once((
            alloc::borrow::Cow::Borrowed("vif"),
            self.vif,
        )))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..self.maps.len()).filter_map(|i| self.view(i)).collect()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// `vifvec` (vector-GSM steerable-pyramid VIF — the original release)
/// of two single-channel `f32` planes (0–255 signal scale), packed
/// row-major with explicit `stride` (≥ `w`). This is **not** VIFp —
/// see the [`vifvec`] module docs. `min(w,h) < 72` returns
/// [`Error::TooSmall`] — the reference's `maxPyrHt` hard-errors when a
/// 4-level SP5 pyramid does not fit (`buildSpyr` bound on the 9-tap
/// `lofilt`), so no score exists below it.
pub fn vifvec_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_vifvec_size(width, height)?;
    check_plane(reference, width, height, stride)?;
    check_plane(distorted, width, height, stride)?;
    let r = gather_plane(reference, width, height, stride);
    let d = gather_plane(distorted, width, height, stride);
    Ok(vifvec::vifvec_core(&r, &d, width, height))
}

/// `vifvec` of two interleaved sRGB RGB8 pairs. Ingress follows the
/// reference pipeline the JPEG AIC-4 `VIF` column was computed with:
/// u8-rounded `round(0.299·R + 0.587·G + 0.114·B)` grayscale (the
/// classic `rgb2gray` result), **not** the unrounded `0.2989` luma
/// [`vif_rgb8`] uses.
pub fn vifvec_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_vifvec_size(width, height)?;
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8_rounded(reference, distorted, width, height, stride);
    Ok(vifvec::vifvec_core(&inputs.r, &inputs.d, width, height))
}

/// [`vifvec_rgb8`] plus every selected subband's per-block terms.
fn vifvec_rgb8_terms(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, Vec<vifvec::SubTerms>), Error> {
    check_vifvec_size(width, height)?;
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8_rounded(reference, distorted, width, height, stride);
    let mut subs = Vec::new();
    let score = vifvec::vifvec_core_impl(&inputs.r, &inputs.d, width, height, Some(&mut subs));
    Ok((score, subs))
}

/// Gathered level-0 input planes (`f64` — see the crate docs).
pub(crate) struct Inputs {
    /// Reference plane, packed `w·h`.
    pub r: Vec<f64>,
    /// Distorted plane, packed `w·h`.
    pub d: Vec<f64>,
}

/// `vifvec`'s size floor: `buildSpyr(im, 4, 'sp5Filters')` requires
/// `maxPyrHt(size, 9) >= 4`, i.e. `min(w,h) >= 72`. Smaller inputs are
/// a reference error condition, not a NaN.
fn check_vifvec_size(w: usize, h: usize) -> Result<(), Error> {
    let got = w.min(h);
    if got < 72 {
        return Err(Error::TooSmall { needed: 72, got });
    }
    Ok(())
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

    /// Luma planes via the `vifvec` reference's ingress: u8-**rounded**
    /// `round(0.299·R + 0.587·G + 0.114·B)` (the classic `rgb2gray`
    /// convention), cast to `f64`.
    pub(crate) fn from_rgb8_rounded(
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
                    (0.299 * pr as f64 + 0.587 * pg as f64 + 0.114 * pb as f64).round() as u8
                        as f64,
                );
                d.push(
                    (0.299 * dr as f64 + 0.587 * dg as f64 + 0.114 * db as f64).round() as u8
                        as f64,
                );
            }
        }
        Self { r, d }
    }
}

/// The pixel-domain VIF metric as a generic scorer — packed `RGB8_SRGB`.
/// The vector-GSM variant is [`Vifvec`].
///
/// Maps, on request: `info_loss_0` (primary) … `info_loss_3`, one per
/// scale with a non-empty statistics grid. Each value is
/// `log10(1 + σ₁²/σₙ²) − log10(1 + g²σ₁²/(σᵥ² + σₙ²))`: the information
/// the reference channel carries at that position minus what survives the
/// distortion channel. `VIF = Σnum/Σden`, so the maps summed over all
/// scales, divided by `Σden`, give `1 − VIF`. They are per-position
/// contributions, not a local VIF ratio: `0` = no loss, higher = more
/// lost, negative where the distortion adds information (VIF > 1)
/// ([`za::MapPolarity::Difference`]). After them, `num_0` … and `den_0` …
/// carry the two information terms themselves on the same grids
/// ([`za::MapPolarity::Feature`]: `den` depends on the reference alone).
#[derive(Debug, Default, Clone, Copy)]
pub struct Vif;

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::BufferLength { .. } => za::ErrorKind::InvalidInput,
            Error::TooSmall { .. } => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of the generic-surface scorer(s) in this crate.
const SCORE_REVISION: u32 = 1;

impl za::Metric for Vif {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("vif", za::implementation!(SCORE_REVISION))
            .with_reference("vifp_mscale.m (Sheikh & Bovik 2006)")
            .with_config("luma", "rgb2gray")
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("vif", za::ScoreKind::Similarity)];
        alloc::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        if let Some(req) = opts.map {
            let (vif, terms) = vif_rgb8_terms(r.pixels, d.pixels, r.width, r.height, r.stride)
                .metric_err_with("vif", Error::kind)?;
            opts.checkpoint()?;
            let eval = VifMaps::vifp(vif, terms, r.width, r.height);
            return za::apply_map_request(alloc::boxed::Box::new(eval), req);
        }
        let s = vif_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("vif", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single("vif", s)))
    }
}

/// The vector-GSM VIF variant (`vifvec`) as a generic scorer.
///
/// Maps, on request: one per pooled subband, finest first —
/// `info_loss_l1_b1` (primary), `info_loss_l1_b4`, … `info_loss_l4_b4`
/// (pyramid level, MATLAB band). Each cell is one 3×3 block of the
/// cropped block grid the score pools, valued
/// `Σ_λ [log2(1 + s·λ/σₙ²) − log2(1 + g²·s·λ/(σᵥ² + σₙ²))]`: the bits the
/// reference channel carries in that block minus those the distorted
/// channel keeps. Summed over all blocks and subbands and divided by the
/// total reference information they give `1 − VIF`. Per-block
/// contributions, not a local VIF ([`za::MapPolarity::Difference`]),
/// with global support (the S-field and eigenvalues use whole-subband
/// statistics). After them, `num_l1_b1` … and `den_l1_b1` … carry the two
/// information terms per block ([`za::MapPolarity::Feature`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct Vifvec;

impl za::Metric for Vifvec {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("vif", za::implementation!(SCORE_REVISION))
            .with_variant("vec")
            .with_reference("vifvec.m (Sheikh & Bovik 2006), vector GSM on a steerable pyramid")
            .with_config("luma", "rgb2gray")
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("vif", za::ScoreKind::Similarity)];
        alloc::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<alloc::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        if let Some(req) = opts.map {
            let (vif, subs) = vifvec_rgb8_terms(r.pixels, d.pixels, r.width, r.height, r.stride)
                .metric_err_with("vif", Error::kind)?;
            opts.checkpoint()?;
            let eval = VifMaps::vifvec(vif, subs, r.width, r.height);
            return za::apply_map_request(alloc::boxed::Box::new(eval), req);
        }
        let s = vifvec_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("vif", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single("vif", s)))
    }
}
