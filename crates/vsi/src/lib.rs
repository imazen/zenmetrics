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
use zenmetrics_abstractions as za;

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
    /// A fixed factor exceeds the shorter input dimension.
    DownscaleFactor {
        /// Requested fixed factor.
        factor: u32,
        /// Input width.
        width: usize,
        /// Input height.
        height: usize,
    },
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
            Error::DownscaleFactor {
                factor,
                width,
                height,
            } => write!(
                f,
                "downscale factor {factor} exceeds image dimensions {width}x{height}"
            ),
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

/// VSI's quality map — the reference's `qualitymap` output (`SimMatrixC`:
/// grad-sim^α · VS-sim · real((I-sim·Q-sim)^λ)), at the decimated
/// resolution the reference computes it: `⌈w/F⌉ × ⌈h/F⌉`,
/// `F = max(1, round(min(w,h)/256))`. Higher = more similar.
#[non_exhaustive]
pub struct VsiDiffmap {
    map: alloc::vec::Vec<f32>,
    /// Map width, `⌈w/F⌉`.
    pub width: u32,
    /// Map height, `⌈h/F⌉`.
    pub height: u32,
    /// The VSI score computed alongside the map.
    pub vsi: f64,
    input: (u32, u32),
    factor: u32,
}

impl VsiDiffmap {
    /// The packed `width·height` quality map.
    pub fn map(&self) -> &[f32] {
        &self.map
    }
}

impl VsiDiffmap {
    /// The decimation factor `F`.
    pub fn factor(&self) -> u32 {
        self.factor
    }

    /// Sample `k` averages input `[k·F + s + 1 − F, k·F + s]`, `s = ⌊F/2⌋`
    /// (`conv2(fspecial('average',F),'same')` then every `F`th sample).
    /// Support is global: the saliency maps come from an FFT log-Gabor
    /// filter and a whole-image `mat2gray` normalization.
    pub fn geometry(&self) -> za::MapGeometry {
        let f = self.factor;
        let origin = i64::from(f / 2) + 1 - i64::from(f);
        za::MapGeometry::grid(
            self.input.0,
            self.input.1,
            origin,
            origin,
            f,
            f,
            self.width,
            self.height,
        )
        .with_support(za::Support::Global)
    }
}

impl za::Evaluation for VsiDiffmap {
    fn score(&self) -> f64 {
        self.vsi
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(core::iter::once((
            alloc::borrow::Cow::Borrowed("vsi"),
            self.vsi,
        )))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        let px = za::gray_f32_slice(&self.map, self.width, self.height).ok()?;
        za::MapView::new("quality", px, za::MapPolarity::Similarity, self.geometry()).ok()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
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
    let f = kernel::decimation_factor(w, h);
    let (mw, mh) = (w.div_ceil(f), h.div_ceil(f));
    let mut map = alloc::vec![0.0f32; mw * mh];
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
            width: mw as u32,
            height: mh as u32,
            vsi: score,
            input: (w as u32, h as u32),
            factor: f as u32,
        },
    ))
}

impl Input {
    pub(crate) fn from_rgb8(data: &[u8], w: usize, h: usize, stride: usize) -> Self {
        let (l, m, n, rgb) = kernel::rgb8_planes(data, w, h, stride);
        Self { l, m, n, rgb }
    }
}

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::BufferLength { .. } | Error::DownscaleFactor { .. } => {
                za::ErrorKind::InvalidInput
            }
        }
    }
}

/// Output revision of [`Vsi`] on the generic surface.
const SCORE_REVISION: u32 = 1;

/// The VSI metric as a generic scorer — packed `RGB8_SRGB`.
#[derive(Debug, Default, Clone, Copy)]
pub struct Vsi {
    downscale: Downscale,
}

/// Pre-scoring opponent/saliency box-filter decimation policy.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Downscale {
    /// Reference max(1, round(min(width, height) / 256)) decimation.
    #[default]
    Auto,
    /// Explicit nonzero factor, with the existing reference filter.
    Factor(core::num::NonZeroU32),
}
impl Downscale {
    fn validate(self, width: usize, height: usize) -> Result<(), Error> {
        match self {
            Self::Factor(n) if n.get() as usize > width.min(height) => {
                Err(Error::DownscaleFactor {
                    factor: n.get(),
                    width,
                    height,
                })
            }
            _ => Ok(()),
        }
    }

    fn factor(self, w: usize, h: usize) -> usize {
        match self {
            Self::Auto => kernel::decimation_factor(w, h),
            Self::Factor(f) => f.get() as usize,
        }
    }
}
impl Vsi {
    /// Construct a scorer with a fixed or automatic decimation policy.
    pub const fn with_downscale(downscale: Downscale) -> Self {
        Self { downscale }
    }
    /// Score strided RGB8 images with this scorer's decimation policy.
    ///
    /// # Errors
    /// Returns [`Error::BufferLength`] for invalid buffer geometry, or
    /// [`Error::DownscaleFactor`] when a fixed factor exceeds the shorter dimension.
    pub fn score_rgb8(
        &self,
        r: &[u8],
        d: &[u8],
        w: usize,
        h: usize,
        stride: usize,
    ) -> Result<f64, Error> {
        check_len(r, w, h, stride)?;
        check_len(d, w, h, stride)?;
        self.downscale.validate(w, h)?;
        Ok(kernel::vsi_planes_with_downscale(
            &Input::from_rgb8(r, w, h, stride),
            &Input::from_rgb8(d, w, h, stride),
            w,
            h,
            None,
            self.downscale,
        ))
    }
    fn diffmap_rgb8(
        &self,
        r: &[u8],
        d: &[u8],
        w: usize,
        h: usize,
        stride: usize,
    ) -> Result<VsiDiffmap, Error> {
        check_len(r, w, h, stride)?;
        check_len(d, w, h, stride)?;
        self.downscale.validate(w, h)?;
        let f = self.downscale.factor(w, h);
        let (mw, mh) = (w.div_ceil(f), h.div_ceil(f));
        let mut map = alloc::vec![0.0;mw*mh];
        let vsi = kernel::vsi_planes_with_downscale(
            &Input::from_rgb8(r, w, h, stride),
            &Input::from_rgb8(d, w, h, stride),
            w,
            h,
            Some(&mut map),
            self.downscale,
        );
        Ok(VsiDiffmap {
            map,
            width: mw as u32,
            height: mh as u32,
            vsi,
            input: (w as u32, h as u32),
            factor: f as u32,
        })
    }
}

impl za::Metric for Vsi {
    fn id(&self) -> za::MetricId {
        let id = za::MetricId::new("vsi", za::implementation!(SCORE_REVISION))
            .with_reference("VSI.m (Zhang, Shen & Li 2014)");
        match self.downscale {
            Downscale::Auto => id,
            Downscale::Factor(n) => id.with_config("downscale", n.get()),
        }
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new("vsi", za::ScoreKind::Similarity)];
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
            let dm = self
                .diffmap_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
                .metric_err_with("vsi", Error::kind)?;
            return za::apply_map_request(alloc::boxed::Box::new(dm), req);
        }
        let s = self
            .score_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("vsi", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single("vsi", s)))
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
