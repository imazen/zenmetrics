//! HaarPSI — a Haar wavelet-based perceptual similarity index.
//!
//! Pure-Rust CPU port of the reference implementation `HaarPSI.m` by
//! Rafael Reisenhofer (MIT-licensed, `rgcda/haarpsi` — a copy lives in
//! `validation/`):
//!
//! > R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand: *A Haar
//! > Wavelet-Based Perceptual Similarity Index for Image Quality
//! > Assessment*. Signal Processing: Image Communication 61, 33–43, 2018.
//!
//! Pipeline (exactly the reference's):
//! 1. Optional 2× subsample (`preprocess_with_subsampling`, reference
//!    default on): decimating 2×2 box mean, `conv2(ones(2,2)/4,'same')`
//!    at the odd indices — a forward-looking window with zero padding.
//! 2. For color input, RGB → YIQ (unrounded float luma/I/Q, the
//!    reference's coefficient set).
//! 3. 3-scale Haar decomposition of the luma plane — the scale-k filter
//!    is `2^-k·ones(2^k,2^k)` with the top half rows negated; the
//!    transposed filter gives the second orientation. `conv2 'same'`
//!    semantics throughout (zero padding, even-kernel anchor).
//! 4. Per orientation: local similarity = mean over scales 1–2 of
//!    `(2·|a||b|+C)/(|a|²+|b|²+C)` on coefficient magnitudes, weight =
//!    `max(|a₃|,|b₃|)` of the coarsest scale; `C = 30`.
//! 5. For color input a third similarity map from 2×2-boxed I/Q
//!    magnitudes, weighted by the mean of the two luma weights.
//! 6. `score = LogInv(Σ σ(LS,α)·W / ΣW, α)²` with `α = 4.2`,
//!    `σ(x,α) = 1/(1+e^{-αx})`, `LogInv(m,α) = log(m/(1−m))/α`.
//!
//! Output range is `[0,1]` (higher = more similar); identical images
//! score `1.0`. A fully zero pair has no weight anywhere — the
//! reference divides `0/0` and returns `NaN`; so do we (IEEE
//! propagation, no special-casing).
//!
//! # Floating-point discipline
//!
//! All conv/similarity planes are f32; every output pixel is a
//! fixed-order f32 expression independent of the SIMD lane count, and
//! the f64 pooling keeps gmsd's fixed 8-lane accumulator grouping, so
//! every tier (scalar / v3 / v4x / neon / wasm128) is **bit-identical**.
//! `_dev` exposes the per-tier entries for tier-parity tests and
//! disassembly.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
// assign_op_pattern: the shared `*_body!` macros accumulate with
// `acc = acc + term` — `f32xN` has no `AddAssign`, so the suggested
// `+=` would not compile (and `x = x + y` keeps the fixed-order
// accumulation explicit). Same allow as crates/psnrhvs.
#![allow(clippy::assign_op_pattern)]

extern crate alloc;

use zenmetrics_abstractions as za;

// Without `_dev` the `pub` items are unreachable; `dev` re-exports them then.
#[cfg_attr(not(feature = "_dev"), allow(unreachable_pub))]
mod kernel;

/// Per-tier entry points for tier-parity tests and disassembly. Not API.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::haarpsi_core_tier_v3_t;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::haarpsi_core_tier_v4x_t;
    pub use crate::kernel::{Plane, haarpsi_core_tier_scalar_t};
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `haarpsi_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `HAARPSI_IMPL_TAG`). The `haarpsi` CLI metric emits the YIQ
/// (color) HaarPSI score of the two sRGB8 images.
pub const HAARPSI_COLUMN_NAME: &str = match option_env!("HAARPSI_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "haarpsi_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **luma-only** variant (`haarpsi-y` CLI metric):
/// `haarpsiy_imazen_v<MAJOR>_<MINOR>_<PATCH>` (`HAARPSIY_IMPL_TAG`).
/// This is the grayscale HaarPSI of the unrounded BT.601 luma plane.
pub const HAARPSIY_COLUMN_NAME: &str = match option_env!("HAARPSIY_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "haarpsiy_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Errors returned by the HaarPSI entry points.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Input slice is too short for the declared geometry.
    BufferLength {
        /// Bytes/elements the caller must supply.
        needed: usize,
        /// Bytes/elements actually supplied.
        got: usize,
    },
    /// Empty image (`width == 0 || height == 0`).
    Empty,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BufferLength { needed, got } => {
                write!(f, "buffer too short: need {needed}, got {got}")
            }
            Self::Empty => write!(f, "empty image"),
        }
    }
}

impl core::error::Error for Error {}

/// HaarPSI of two strided f32 planes on the 0..255 scale, with the
/// reference's default 2× subsample preprocessing.
///
/// This is the reference's grayscale path (`HaarPSI(imgRef, imgDist)`
/// on single-channel input).
pub fn haarpsi_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    haarpsi_plane_f32_opts(reference, distorted, width, height, stride, true)
}

/// [`haarpsi_plane_f32`] with an explicit
/// `preprocess_with_subsampling` flag (the reference's optional third
/// argument; `false` omits the 2× subsample step).
pub fn haarpsi_plane_f32_opts(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    preprocess_subsample: bool,
) -> Result<f64, Error> {
    check_plane(reference, distorted, width, height, stride)?;
    let pr = kernel::plane_from_f32(reference, width, height, stride, preprocess_subsample);
    let pd = kernel::plane_from_f32(distorted, width, height, stride, preprocess_subsample);
    Ok(kernel::haarpsi_core(&pr, &pd, None, None))
}

/// HaarPSI of two interleaved sRGB RGB8 pairs — the reference's color
/// path: unrounded BT.601-family YIQ transform, luma Haar similarities
/// plus the I/Q color similarity map.
pub fn haarpsi_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    haarpsi_rgb8_opts(reference, distorted, width, height, stride, true)
}

/// [`haarpsi_rgb8`] with an explicit `preprocess_with_subsampling` flag.
pub fn haarpsi_rgb8_opts(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    preprocess_subsample: bool,
) -> Result<f64, Error> {
    check_rgb(reference, distorted, width, height, stride)?;
    // The reference converts RGB→YIQ per pixel (f64) and then subsamples
    // each channel; both steps are linear, so the fused "box-mean then
    // dot" order used here differs only in the last ulps.
    let yr = kernel::plane_from_rgb8(
        reference,
        width,
        height,
        stride,
        Y_COEF,
        preprocess_subsample,
    );
    let yd = kernel::plane_from_rgb8(
        distorted,
        width,
        height,
        stride,
        Y_COEF,
        preprocess_subsample,
    );
    let ir = kernel::plane_from_rgb8(
        reference,
        width,
        height,
        stride,
        I_COEF,
        preprocess_subsample,
    );
    let id = kernel::plane_from_rgb8(
        distorted,
        width,
        height,
        stride,
        I_COEF,
        preprocess_subsample,
    );
    let qr = kernel::plane_from_rgb8(
        reference,
        width,
        height,
        stride,
        Q_COEF,
        preprocess_subsample,
    );
    let qd = kernel::plane_from_rgb8(
        distorted,
        width,
        height,
        stride,
        Q_COEF,
        preprocess_subsample,
    );
    Ok(kernel::haarpsi_core(
        &yr,
        &yd,
        Some((&ir, &id, &qr, &qd)),
        None,
    ))
}

/// [`haarpsi_rgb8`] plus the per-pixel similarity map (color path —
/// the chroma similarity is folded into the map the way the reference's
/// `sum(...,3)` pooling does).
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w·3`.
pub fn haarpsi_rgb8_with_diffmap(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, HaarpsiDiffmap), Error> {
    check_rgb(reference, distorted, width, height, stride)?;
    let mk = |data: &[u8], coef| kernel::plane_from_rgb8(data, width, height, stride, coef, true);
    let yr = mk(reference, Y_COEF);
    let yd = mk(distorted, Y_COEF);
    let ir = mk(reference, I_COEF);
    let id = mk(distorted, I_COEF);
    let qr = mk(reference, Q_COEF);
    let qd = mk(distorted, Q_COEF);
    let mut map = alloc::vec![0.0f32; yr.w * yr.h];
    let score = kernel::haarpsi_core(&yr, &yd, Some((&ir, &id, &qr, &qd)), Some(&mut map));
    Ok((
        score,
        HaarpsiDiffmap {
            map,
            width: yr.w as u32,
            height: yr.h as u32,
            haarpsi: score,
            input: (width as u32, height as u32),
            subsampled: true,
        },
    ))
}

/// HaarPSI of two interleaved sRGB RGB8 pairs on the **unrounded
/// BT.601 luma** plane (the `haarpsi-y` CLI metric): `Y` is computed in
/// f64, stored f32, then scored through the grayscale path. Not the
/// reference's color metric — a luma-only convenience matching
/// `psnrhvs-y`'s convention.
pub fn haarpsi_luma8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<f64, Error> {
    haarpsi_luma8_opts(reference, distorted, width, height, stride, true)
}

/// [`haarpsi_luma8`] with an explicit `preprocess_with_subsampling` flag.
pub fn haarpsi_luma8_opts(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    preprocess_subsample: bool,
) -> Result<f64, Error> {
    check_rgb(reference, distorted, width, height, stride)?;
    let pr = kernel::plane_from_rgb8(
        reference,
        width,
        height,
        stride,
        Y_COEF,
        preprocess_subsample,
    );
    let pd = kernel::plane_from_rgb8(
        distorted,
        width,
        height,
        stride,
        Y_COEF,
        preprocess_subsample,
    );
    Ok(kernel::haarpsi_core(&pr, &pd, None, None))
}

/// The reference's YIQ coefficient sets (applied to unrounded channel
/// means; `HaarPSI.m` lines `imgRefY = 0.299*R + 0.587*G + 0.114*B`,
/// `imgRefI = 0.596*R − 0.274*G − 0.322*B`,
/// `imgRefQ = 0.211*R − 0.523*G + 0.312*B`).
pub(crate) const Y_COEF: [f64; 3] = [0.299, 0.587, 0.114];
pub(crate) const I_COEF: [f64; 3] = [0.596, -0.274, -0.322];
pub(crate) const Q_COEF: [f64; 3] = [0.211, -0.523, 0.312];

/// HaarPSI's per-pixel similarity map — the reference's
/// `similarity_map` output (the weighted mean of the orientation and
/// chroma similarity planes, at the preprocessed resolution — `w×h`
/// after the default 2× subsample). Higher = more similar.
///
/// A [`zenmetrics_abstractions::Evaluation`].
#[non_exhaustive]
pub struct HaarpsiDiffmap {
    map: alloc::vec::Vec<f32>,
    /// Map width — the preprocessed plane width.
    pub width: u32,
    /// Map height — the preprocessed plane height.
    pub height: u32,
    /// The HaarPSI score computed alongside the map.
    pub haarpsi: f64,
    input: (u32, u32),
    subsampled: bool,
}

impl HaarpsiDiffmap {
    /// The packed `width·height` similarity map.
    pub fn map(&self) -> &[f32] {
        &self.map
    }
}

impl HaarpsiDiffmap {
    /// With the 2× preprocessing subsample, cell `k` owns input pixels
    /// `2k, 2k+1` (the forward-looking 2×2 box); otherwise one pixel. The
    /// coarsest Haar filter (8 taps, `conv2 'same'`) reaches 4 samples
    /// each way.
    pub fn geometry(&self) -> za::MapGeometry {
        let (c, halo) = if self.subsampled { (2, 8) } else { (1, 4) };
        za::MapGeometry::grid(
            self.input.0,
            self.input.1,
            0,
            0,
            c,
            c,
            self.width,
            self.height,
        )
        .with_halo(halo, halo)
    }
}

impl za::Evaluation for HaarpsiDiffmap {
    fn score(&self) -> f64 {
        self.haarpsi
    }

    fn scores(&self) -> za::Scores<'_> {
        alloc::boxed::Box::new(core::iter::once((
            alloc::borrow::Cow::Borrowed("haarpsi"),
            self.haarpsi,
        )))
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        let px = za::gray_f32_slice(&self.map, self.width, self.height).ok()?;
        za::MapView::new(
            "similarity",
            px,
            za::MapPolarity::Similarity,
            self.geometry(),
        )
        .ok()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// [`haarpsi_plane_f32`] plus the per-pixel similarity map.
///
/// # Errors
/// [`Error::BufferLength`] when `data` is shorter than
/// `stride·(h−1) + w`.
pub fn haarpsi_plane_f32_with_diffmap(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(f64, HaarpsiDiffmap), Error> {
    check_plane(reference, distorted, width, height, stride)?;
    let pr = kernel::plane_from_f32(reference, width, height, stride, true);
    let pd = kernel::plane_from_f32(distorted, width, height, stride, true);
    let mut map = alloc::vec![0.0f32; pr.w * pr.h];
    let score = kernel::haarpsi_core(&pr, &pd, None, Some(&mut map));
    Ok((
        score,
        HaarpsiDiffmap {
            map,
            width: pr.w as u32,
            height: pr.h as u32,
            haarpsi: score,
            input: (width as u32, height as u32),
            subsampled: true,
        },
    ))
}

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::BufferLength { .. } => za::ErrorKind::InvalidInput,
            Error::Empty => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of [`Haarpsi`] on the generic surface.
const SCORE_REVISION: u32 = 1;

/// The HaarPSI metric as a generic scorer — packed `RGB8_SRGB` through the
/// reference's color path with its default 2× preprocessing subsample.
#[derive(Debug, Default, Clone, Copy)]
pub struct Haarpsi;

impl za::Metric for Haarpsi {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("haarpsi", za::implementation!(SCORE_REVISION))
            .with_reference("HaarPSI.m (Reisenhofer et al. 2018), color path")
            .with_config("preprocess_subsample", true)
    }

    fn outputs(&self) -> alloc::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] =
            &[za::ScoreSpec::new("haarpsi", za::ScoreKind::Similarity)];
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
            let (_, dm) =
                haarpsi_rgb8_with_diffmap(r.pixels, d.pixels, r.width, r.height, r.stride)
                    .metric_err_with("haarpsi", Error::kind)?;
            return za::apply_map_request(alloc::boxed::Box::new(dm), req);
        }
        let s = haarpsi_rgb8(r.pixels, d.pixels, r.width, r.height, r.stride)
            .metric_err_with("haarpsi", Error::kind)?;
        Ok(alloc::boxed::Box::new(za::ScalarEval::single("haarpsi", s)))
    }
}

fn check_plane(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::Empty);
    }
    if stride < width {
        return Err(Error::BufferLength {
            needed: width,
            got: stride,
        });
    }
    let needed = stride * (height - 1) + width;
    if reference.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: reference.len(),
        });
    }
    if distorted.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: distorted.len(),
        });
    }
    Ok(())
}

fn check_rgb(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), Error> {
    if width == 0 || height == 0 {
        return Err(Error::Empty);
    }
    if stride < 3 * width {
        return Err(Error::BufferLength {
            needed: 3 * width,
            got: stride,
        });
    }
    let needed = stride * (height - 1) + 3 * width;
    if reference.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: reference.len(),
        });
    }
    if distorted.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: distorted.len(),
        });
    }
    Ok(())
}
