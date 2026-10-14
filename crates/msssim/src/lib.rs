//! MS-SSIM — Multi-Scale Structural Similarity (Wang, Simoncelli &
//! Bovik, *Multi-scale structural similarity for image quality
//! assessment*, IEEE Asilomar Conf. on Signals, Systems and Computers,
//! Nov. 2003). Pure-Rust CPU port of the authors' reference
//! `msssim.m` + `ssim_index_new.m`, validated against them under GNU
//! Octave (`validation/`).
//!
//! Pipeline, matching the reference: at each of `level` scales,
//! `ssim_index_new` computes 11×11 Gaussian-weighted mean/variance/
//! covariance statistics over the `'valid'` correlation region and
//! means the SSIM and contrast–structure maps; between scales both
//! images are downsampled by `imfilter(ones(2)/4,'symmetric','same')`
//! + `1:2:end` (forward 2×2 box, whole-point symmetric border). The
//!
//! The score is `prod(mcs[1..L−1]^w[1..L−1]) · mssim[L]^w[L]` with the
//! canonical weights `[0.0448 0.2856 0.3001 0.2363 0.1333]`,
//! `C1 = (0.01·255)²`, `C2 = (0.03·255)²` — plane values are on the
//! 0–255 scale.
//!
//! The reference hard-errors (`-Inf`) when `min(w,h) < 11·2^(level−1)`;
//! this port instead auto-selects `level = min(5, floor(log2(min/11))+1)`
//! and applies the first `level` canonical weights — exactly what
//! calling the reference with `(level=L, weight=w(1:L))` computes
//! (the goldens pin this at multiple levels). Images with `min < 11`
//! return [`Error::TooSmall`].
//!
//! Output range is roughly `[0,1]` (higher = more similar); identical
//! images score `1.0`, including constant ones (unlike FSIM/VSI the
//! denominators never degenerate — `C1, C2 > 0`).
//!
//! # Floating-point discipline
//!
//! Plane data is `f32` end-to-end; every output pixel of a filter or
//! map stage is a fixed-order `f32` expression independent of the SIMD
//! lane count, and every scalar reduction (the two map means, the
//! weighted-product combine) is `f64` in fixed raster order. Every
//! tier (scalar / v3 / v4x / neon / wasm128) and every `parallel`
//! thread count produces **bit-identical** output. `_dev` exposes the
//! per-tier entries for tier-parity tests and disassembly.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// excessive_precision: the Gaussian taps and weights are transcribed at
// f64 precision so the literals reproduce the reference's constants
// bit-for-bit before the f32 cast.
#![allow(clippy::excessive_precision)]

extern crate alloc;

use alloc::vec::Vec;
use zenmetrics_abstractions as za;

// Without `_dev` the `pub` items are unreachable; `dev` re-exports them then.
#[cfg_attr(not(feature = "_dev"), allow(unreachable_pub))]
mod kernel;
mod libvmaf;

pub use libvmaf::{float_ms_ssim, float_ssim};
#[doc(hidden)]
pub use libvmaf::{float_ms_ssim_scales, float_ssim_lcs};

#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    //! Development hooks — not API. Exposes the per-tier score entry
    //! points so tier parity can be tested and the kernels
    //! disassembled.
    #[cfg(target_arch = "aarch64")]
    pub use crate::kernel::msssim_core_tier_neon_t;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::msssim_core_tier_v3_t;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::msssim_core_tier_v4x_t;
    #[cfg(target_arch = "wasm32")]
    pub use crate::kernel::msssim_core_tier_wasm;
    pub use crate::kernel::{Args, msssim_core_tier_scalar_t};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `msssim_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `MSSSIM_IMPL_TAG`).
pub const MSSSIM_COLUMN_NAME: &str = match option_env!("MSSSIM_IMPL_TAG") {
    Some(tag) => tag,
    None => concat!(
        "msssim_imazen_v",
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
    /// `min(w,h) < 11` — the reference returns `-Inf` here (the 11×11
    /// `'valid'` window cannot be formed at any level).
    TooSmall,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::BufferLength { needed, got } => {
                write!(f, "buffer too short: need {needed} elements, got {got}")
            }
            Error::TooSmall => write!(f, "image too small: min(w,h) must be ≥ 11"),
        }
    }
}

impl core::error::Error for Error {}

/// MATLAB `rgb2gray` luma coefficients (`0.2989 0.5870 0.1140`) — the
/// conversion Wang's pipeline applies to colour input.
const LUMA_COEF: [f32; 3] = [0.2989, 0.5870, 0.1140];

/// `min(5, floor(log2(min(w,h)/11)) + 1)` — the largest level count
/// for which the reference's `min_dim/2^(L−1) ≥ 11` check passes.
pub(crate) fn auto_level(w: usize, h: usize) -> Result<usize, Error> {
    let m = w.min(h);
    if m < 11 {
        return Err(Error::TooSmall);
    }
    // floor(log2(m/11)): number of times m can be halved past 11.
    let mut l = 1usize;
    while m / (1 << l) >= 11 {
        l += 1;
    }
    Ok(l.min(5))
}

/// MS-SSIM's per-pixel map — the canonical SSIM map at the finest
/// scale (the `ssimmap` MATLAB `ssim` returns), `(w−10)×(h−10)` after
/// the level-0 valid Gaussian window. Higher = more similar.
///
/// A [`zenmetrics_abstractions::Evaluation`].
#[non_exhaustive]
pub struct MsssimDiffmap {
    map: alloc::vec::Vec<f32>,
    /// Map width — `width−10` (valid convolution interior).
    pub width: u32,
    /// Map height — `height−10`.
    pub height: u32,
    /// The MS-SSIM score computed alongside the map.
    pub msssim: f64,
    /// Level count the score used.
    pub level: u32,
    input: (u32, u32),
}

impl MsssimDiffmap {
    /// The packed `width·height` SSIM map.
    pub fn map(&self) -> &[f32] {
        &self.map
    }
}

impl MsssimDiffmap {
    /// Cell `(i, j)` is the 11×11 window centered on input pixel
    /// `(i + 5, j + 5)` (the level-0 `'valid'` correlation), so the
    /// border 5 pixels own no cell.
    pub fn geometry(&self) -> za::MapGeometry {
        za::MapGeometry::grid(
            self.input.0,
            self.input.1,
            5,
            5,
            1,
            1,
            self.width,
            self.height,
        )
        .with_halo(5, 5)
    }
}

impl za::Evaluation for MsssimDiffmap {
    fn score(&self) -> f64 {
        self.msssim
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(core::iter::once((
            alloc::borrow::Cow::Borrowed("msssim"),
            self.msssim,
        )))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        let px = za::gray_f32_slice(&self.map, self.width, self.height).ok()?;
        za::MapView::new("ssim", px, za::MapPolarity::Similarity, self.geometry()).ok()
    }

    fn extras(&self) -> za::Extras<'_> {
        alloc::boxed::Box::new(core::iter::once((
            alloc::borrow::Cow::Borrowed("level"),
            alloc::format!("{}", self.level).into(),
        )))
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// [`msssim_plane_f32`] plus the level-0 SSIM map.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w`.
pub fn msssim_plane_f32_with_diffmap(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, MsssimDiffmap), Error> {
    check_plane(reference, width, height, stride)?;
    check_plane(distorted, width, height, stride)?;
    let level = auto_level(width, height)?;
    let r = gather_plane(reference, width, height, stride);
    let d = gather_plane(distorted, width, height, stride);
    let mw = width - 10;
    let mh = height - 10;
    let mut map = alloc::vec![0.0f32; mw * mh];
    let score = kernel::msssim_core(&Inputs { r, d }, width, height, level, Some(&mut map));
    Ok((
        score,
        MsssimDiffmap {
            map,
            width: mw as u32,
            height: mh as u32,
            msssim: score,
            level: level as u32,
            input: (width as u32, height as u32),
        },
    ))
}

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::BufferLength { .. } => za::ErrorKind::InvalidInput,
            Error::TooSmall => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of [`Msssim`] on the generic surface.
const SCORE_REVISION: u32 = 1;

/// The MS-SSIM metric as a generic scorer — packed `RGB8_SRGB`, scored on
/// MATLAB `rgb2gray` luma with auto-selected levels (reported in extras).
#[derive(Debug, Default, Clone, Copy)]
pub struct Msssim;

impl za::Metric for Msssim {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("msssim", za::implementation!(SCORE_REVISION))
            .with_reference("msssim.m + ssim_index_new.m (Wang, Simoncelli & Bovik 2003)")
            .with_config("luma", "rgb2gray")
            .with_config("levels", "auto")
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] =
            &[za::ScoreSpec::new("msssim", za::ScoreKind::Similarity)];
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
            let (_, dm) = msssim_rgb8_with_diffmap(r.pixels, d.pixels, r.width, r.height, r.stride)
                .metric_err_with("msssim", Error::kind)?;
            return za::apply_map_request(alloc::boxed::Box::new(dm), req);
        }
        let s = msssim_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("msssim", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single("msssim", s)))
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

/// MS-SSIM of two single-channel `f32` planes (0–255 signal scale),
/// packed row-major with explicit `stride` (≥ `w`).
///
/// `level` is auto-selected (see crate docs); the returned score
/// equals `msssim(r, d, K, win, L, w(1:L))` of the reference.
pub fn msssim_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_plane(reference, width, height, stride)?;
    check_plane(distorted, width, height, stride)?;
    let level = auto_level(width, height)?;
    let r = gather_plane(reference, width, height, stride);
    let d = gather_plane(distorted, width, height, stride);
    Ok(kernel::msssim_core(
        &Inputs { r, d },
        width,
        height,
        level,
        None,
    ))
}

/// MS-SSIM of two interleaved sRGB RGB8 pairs, on the MATLAB
/// `rgb2gray` luma plane (unrounded `0.2989/0.5870/0.1140` weights).
pub fn msssim_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let level = auto_level(width, height)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    Ok(kernel::msssim_core(&inputs, width, height, level, None))
}

/// [`msssim_rgb8`] plus the level-0 SSIM map.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w·3`.
pub fn msssim_rgb8_with_diffmap(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, MsssimDiffmap), Error> {
    check_rgb(reference, width, height, stride)?;
    check_rgb(distorted, width, height, stride)?;
    let level = auto_level(width, height)?;
    let inputs = Inputs::from_rgb8(reference, distorted, width, height, stride);
    let mw = width - 10;
    let mh = height - 10;
    let mut map = alloc::vec![0.0f32; mw * mh];
    let score = kernel::msssim_core(&inputs, width, height, level, Some(&mut map));
    Ok((
        score,
        MsssimDiffmap {
            map,
            width: mw as u32,
            height: mh as u32,
            msssim: score,
            level: level as u32,
            input: (width as u32, height as u32),
        },
    ))
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

#[archmage::attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
fn rgb_luma_inner(
    token: Token,
    reference: &[u8],
    distorted: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    r: &mut [f32],
    d: &mut [f32],
) {
    let cr = f32x8::splat_t(token, LUMA_COEF[0]);
    let cg = f32x8::splat_t(token, LUMA_COEF[1]);
    let cb = f32x8::splat_t(token, LUMA_COEF[2]);
    let convert = |pixels: &[u8; 24]| {
        let red = f32x8::from_array_t(token, core::array::from_fn(|i| pixels[i * 3] as f32));
        let green = f32x8::from_array_t(token, core::array::from_fn(|i| pixels[i * 3 + 1] as f32));
        let blue = f32x8::from_array_t(token, core::array::from_fn(|i| pixels[i * 3 + 2] as f32));
        (cr * red + cg * green + cb * blue).to_array()
    };
    for y in 0..h {
        let rs = &reference[y * stride..y * stride + 3 * w];
        let ds = &distorted[y * stride..y * stride + 3 * w];
        let rout = &mut r[y * w..(y + 1) * w];
        let dout = &mut d[y * w..(y + 1) * w];
        let mut x = 0;
        for (rp, dp) in rs.as_chunks::<24>().0.iter().zip(ds.as_chunks::<24>().0) {
            rout[x..x + 8].copy_from_slice(&convert(rp));
            dout[x..x + 8].copy_from_slice(&convert(dp));
            x += 8;
        }
        for i in x..w {
            rout[i] = LUMA_COEF[0] * rs[i * 3] as f32
                + LUMA_COEF[1] * rs[i * 3 + 1] as f32
                + LUMA_COEF[2] * rs[i * 3 + 2] as f32;
            dout[i] = LUMA_COEF[0] * ds[i * 3] as f32
                + LUMA_COEF[1] * ds[i * 3 + 1] as f32
                + LUMA_COEF[2] * ds[i * 3 + 2] as f32;
        }
    }
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
        let (mut r, mut d) = (alloc::vec![0.0f32; n], alloc::vec![0.0f32; n]);
        archmage::attuned!(
            rgb_luma_inner(reference, distorted, w, h, stride, &mut r, &mut d),
            [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar]
        );
        Self { r, d }
    }
}
