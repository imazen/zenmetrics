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
//! so every tier (scalar / v3 / v4x / neon / wasm128) and every
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

use zenmetrics_abstractions as za;

use alloc::vec::Vec;

pub mod fft;
// Without `_dev` the `pub` items are unreachable; `dev` re-exports them then.
#[cfg_attr(not(feature = "_dev"), allow(unreachable_pub))]
mod kernel;

#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::mad_core_tier_neon_t;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::mad_core_tier_v3_t;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::mad_core_tier_v4x_t;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::mad_core_tier_wasm128_t;
    pub use crate::kernel::{Args, mad_core_tier_scalar_t};
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

impl core::error::Error for Error {}

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

/// The mad-iqa metric as a generic scorer — accepts `RGB8_SRGB`
/// [`zenpixels::PixelSlice`] inputs.
///
/// Maps, on request, are the reference's two strategy maps over the
/// interior both indices pool, `[16, w−17) × [16, h−17)`:
///
/// - `hi` (primary): `hi_index`'s visible-distortion map, the
///   contrast-threshold mask times the 16×16 local MSE, one value per
///   pixel; `HI = 10·RMS(hi)`.
/// - `lo`: `lo_index`'s statistical-difference map at its native 4×4
///   tile resolution, `Σ_scales w_s·(|Δσ| + 2|Δskew| + |Δkurt|)` over the
///   log-Gabor bank; each tile's value covers the 4×4 pixels at its tile
///   origin as the reference writes it, and `LO` is the RMS over interior
///   pixels. The last tile row/column reaches up to 3 pixels past the
///   interior.
///
/// Both are [`za::MapPolarity::Difference`] (`0` = identical) with global
/// support (DFT-domain CSF and log-Gabor filters). They are the strategy
/// components, not a per-pixel MAD: `MAD = HI^σ · LO^(1−σ)` combines the
/// two indices. Below 34 pixels there is no interior and no map.
#[derive(Debug, Default, Clone, Copy)]
pub struct Mad;

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::BufferLength { .. } => za::ErrorKind::InvalidInput,
        }
    }
}

/// Output revision of the generic-surface scorer(s) in this crate.
const SCORE_REVISION: u32 = 1;

impl za::Metric for Mad {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("mad", za::implementation!(SCORE_REVISION))
            .with_reference("STMAD_2011 hi_index.m + lo_index.m (Larson & Chandler 2010)")
            .with_config("luma", "rgb2gray")
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[
            za::ScoreSpec::new("mad", za::ScoreKind::Difference),
            za::ScoreSpec::new("hi", za::ScoreKind::Difference),
            za::ScoreSpec::new("lo", za::ScoreKind::Difference),
        ];
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
            let (s, raw) = mad_rgb8_maps(r.pixels, d.pixels, r.width, r.height, r.stride)
                .metric_err_with("mad", Error::kind)?;
            opts.checkpoint()?;
            let eval = MadMaps::new(s, raw, r.width, r.height);
            return za::apply_map_request(alloc::boxed::Box::new(eval), req);
        }
        let s = mad_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("mad", Error::kind)?;
        Ok(alloc::boxed::Box::new(
            za::ScalarEval::single("mad", s.mad)
                .with("hi", s.hi)
                .with("lo", s.lo),
        ))
    }
}

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
    Ok(kernel::mad_score(&Inputs { r, d }, width, height, None))
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
    Ok(kernel::mad_score(&inputs, width, height, None))
}

/// [`mad_rgb8`] plus the two strategy maps the indices are computed from.
fn mad_rgb8_maps(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(MadScore, kernel::MapsOut), Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    let mut maps = kernel::MapsOut::default();
    let score = kernel::mad_score(&inputs, width, height, Some(&mut maps));
    Ok((score, maps))
}

/// The pooled interior: `[16, n−17) × [16, m−17)`, `None` below 34.
fn interior(w: usize, h: usize) -> Option<(usize, usize)> {
    (w >= 34 && h >= 34).then(|| (w - 33, h - 33))
}

/// A scored pair with its interior strategy maps.
struct MadMaps {
    score: MadScore,
    /// `(w−33) × (h−33)` interior of `hi_index`'s map.
    hi: Vec<f32>,
    /// The tiles of `lo_index`'s map that start inside the interior.
    lo: Vec<f32>,
    geometry: [za::MapGeometry; 2],
}

impl MadMaps {
    fn new(score: MadScore, raw: kernel::MapsOut, w: usize, h: usize) -> Self {
        let Some((iw, ih)) = interior(w, h) else {
            return Self {
                score,
                hi: Vec::new(),
                lo: Vec::new(),
                geometry: [za::MapGeometry::identity(w as u32, h as u32); 2],
            };
        };
        let mut hi = Vec::with_capacity(iw * ih);
        for r in 16..16 + ih {
            hi.extend_from_slice(&raw.hi[r * w + 16..r * w + 16 + iw]);
        }
        // Tiles start at pixel (4·jt, 4·it); those inside the interior run
        // from tile 4 to the one holding its last row/column.
        let (tx, ty) = ((w - 18) / 4 - 3, (h - 18) / 4 - 3);
        let nt = raw.tiles.1;
        let mut lo = Vec::with_capacity(tx * ty);
        for jt in 4..4 + ty {
            lo.extend_from_slice(&raw.lo[jt * nt + 4..jt * nt + 4 + tx]);
        }
        let (w32, h32) = (w as u32, h as u32);
        Self {
            score,
            hi,
            lo,
            geometry: [
                za::MapGeometry::grid(w32, h32, 16, 16, 1, 1, iw as u32, ih as u32)
                    .with_support(za::Support::Global),
                za::MapGeometry::grid(w32, h32, 16, 16, 4, 4, tx as u32, ty as u32)
                    .with_support(za::Support::Global),
            ],
        }
    }

    fn view(&self, i: usize) -> Option<za::MapView<'_>> {
        let (name, data) = [("hi", &self.hi), ("lo", &self.lo)][i];
        let g = self.geometry[i];
        if data.is_empty() {
            return None;
        }
        let px = za::gray_f32_slice(data, g.cells_x, g.cells_y).ok()?;
        za::MapView::new(name, px, za::MapPolarity::Difference, g).ok()
    }
}

impl za::Evaluation for MadMaps {
    fn score(&self) -> f64 {
        self.score.mad
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(
            [
                (alloc::borrow::Cow::Borrowed("mad"), self.score.mad),
                (alloc::borrow::Cow::Borrowed("hi"), self.score.hi),
                (alloc::borrow::Cow::Borrowed("lo"), self.score.lo),
            ]
            .into_iter(),
        )
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view(0)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        (0..2).filter_map(|i| self.view(i)).collect()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
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
