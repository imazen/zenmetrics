//! Pure-Rust CPU port of **GMSD** (Gradient Magnitude Similarity Deviation) —
//! Xue, Zhang, Mou & Bovik, *IEEE TIP* 23(2), 2014.
//!
//! Tracks the C reference **libgmsd** (<https://github.com/clunietp/libgmsd>,
//! MIT, commit `de646c9a`), which follows the authors' `GMSD.m`. The notice
//! travels with the ported code in `src/kernel.rs` and `LICENSE-libgmsd`.
//!
//! # Algorithm
//!
//! 1. 2×2 average, keep every second sample (half resolution).
//! 2. Prewitt gradient magnitude (`[1 0 −1]/3` family, zero-padded) on both
//!    images.
//! 3. Per-sample similarity `GMS = (2·m_r·m_d + c) / (m_r² + m_d² + c)`,
//!    `c = 170` on the 0..255 scale.
//! 4. **Deviation pooling**: the score is the sample standard deviation of
//!    the GMS map (0 = identical, larger = worse).
//!
//! The GMS map is bit-identical to libgmsd's; the score agrees to f64
//! rounding (it accumulates `1 − GMS` in f64 in one pass instead of
//! libgmsd's two-pass mean/variance). `benchmarks/gmsd_parity_2026-09-22.md`
//! has the measured record.
//!
//! # Differences from the reference, all deliberate
//!
//! - **Odd dimensions**: the half-resolution grid is `⌊w/2⌋ × ⌊h/2⌋` (the size
//!   libgmsd documents); the trailing odd row/column is dropped. libgmsd's own
//!   `downsample_2x2` writes one row/column past that allocation on odd input,
//!   and `GMSD.m` keeps a half-zero-padded extra sample instead.
//! - Input is any strided f32 plane on the 0..255 scale; no `(w, h)` limit
//!   beyond a 2-sample map.
//!
//! # SIMD
//!
//! One `#[arcane]` entry per tier (`v4` AVX-512 behind crate feature
//! `avx512`, `v3` AVX2, `neon`, `wasm128`, `scalar`), with the per-row
//! helpers as `#[rite]` variants of the same tier inlined into it.
//! `_dev` exposes the per-tier entries for tier-parity tests.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]

extern crate alloc;
// The test harness supplies std while the library's no_std arithmetic stays
// selected. Reference calculations in tests use std's floating-point methods.
#[cfg(all(test, not(feature = "std")))]
extern crate std;

use alloc::vec::Vec;

mod kernel;
#[macro_use]
mod chroma_gradient;
mod mdsi;
mod ms_gmsd;

/// Stable column-name identifier for sweep sidecars:
/// `gmsd_cpu_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time via
/// `GMSD_CPU_IMPL_TAG`).
pub const GMSD_COLUMN_NAME: &str = match option_env!("GMSD_CPU_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "gmsd_cpu_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH"),
    ),
};

/// A read-only grayscale f32 plane on the 0..255 scale.
///
/// Sample `(x, y)` is `data[y * stride + x]`. `stride >= width`; the last row
/// needs only `width` samples.
#[derive(Debug, Clone, Copy)]
pub struct GrayImage<'a> {
    data: &'a [f32],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'a> GrayImage<'a> {
    /// Wrap a strided plane. Fails if `stride < width` or `data` is too
    /// short for `height` rows.
    pub fn new(data: &'a [f32], width: usize, height: usize, stride: usize) -> Result<Self> {
        if stride < width {
            return Err(Error::StrideTooSmall { width, stride });
        }
        let need = if height == 0 {
            0
        } else {
            (height - 1) * stride + width
        };
        if data.len() < need {
            return Err(Error::BufferTooSmall {
                expected: need,
                got: data.len(),
            });
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
        })
    }

    /// Wrap a tightly packed plane (`stride == width`).
    pub fn packed(data: &'a [f32], width: usize, height: usize) -> Result<Self> {
        Self::new(data, width, height, width)
    }

    /// Width in samples.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in rows.
    pub fn height(&self) -> usize {
        self.height
    }
}

/// One GMSD comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GmsdScore {
    /// GMSD: standard deviation (n − 1 normalisation, as `std2` and libgmsd)
    /// of the GMS map. 0 = identical; larger = worse.
    pub gmsd: f64,
    /// Mean of the GMS map (the paper's GMSM). 1 = identical.
    pub mean_gms: f64,
}

/// Failure modes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// `stride < width`.
    StrideTooSmall {
        /// Width passed.
        width: usize,
        /// Stride passed.
        stride: usize,
    },
    /// The buffer is shorter than `(height − 1) · stride + width` (or, for the
    /// map / RGB entry points, than their stated size).
    BufferTooSmall {
        /// Required length.
        expected: usize,
        /// Length passed.
        got: usize,
    },
    /// Reference and distorted dimensions differ.
    DimensionMismatch {
        /// Reference `(width, height)`.
        reference: (usize, usize),
        /// Distorted `(width, height)`.
        distorted: (usize, usize),
    },
    /// The half-resolution map would hold fewer than 2 samples, so its
    /// standard deviation is undefined. Needs `(w/2)·(h/2) >= 2`.
    TooSmall {
        /// Width passed.
        width: usize,
        /// Height passed.
        height: usize,
    },
    /// zenpixels-convert could not bring the input to sRGB RGB8.
    #[cfg(feature = "pixels")]
    Convert(alloc::string::String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::StrideTooSmall { width, stride } => {
                write!(f, "stride {stride} is smaller than width {width}")
            }
            Error::BufferTooSmall { expected, got } => {
                write!(f, "buffer too small: need {expected} elements, got {got}")
            }
            Error::DimensionMismatch {
                reference,
                distorted,
            } => write!(
                f,
                "dimension mismatch: reference {}x{}, distorted {}x{}",
                reference.0, reference.1, distorted.0, distorted.1
            ),
            Error::TooSmall { width, height } => write!(
                f,
                "image too small for GMSD: {width}x{height} (the half-resolution map needs >= 2 samples)"
            ),
            #[cfg(feature = "pixels")]
            Error::Convert(m) => write!(f, "pixel conversion failed: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// `Result<T, gmsd::Error>`.
pub type Result<T> = core::result::Result<T, Error>;

/// Dimensions of the GMS map for a `width × height` input:
/// `(width / 2, height / 2)`.
pub fn map_dims(width: usize, height: usize) -> (usize, usize) {
    (width / 2, height / 2)
}

/// GMSD of `distorted` against `reference`.
pub fn gmsd(reference: GrayImage<'_>, distorted: GrayImage<'_>) -> Result<GmsdScore> {
    run(reference, distorted, None)
}

/// GMSD plus the GMS map, written tightly packed into `map`
/// (`map_dims(w, h)`; `map.len()` must be at least `(w/2)·(h/2)`).
pub fn gmsd_with_map(
    reference: GrayImage<'_>,
    distorted: GrayImage<'_>,
    map: &mut [f32],
) -> Result<GmsdScore> {
    run(reference, distorted, Some(map))
}

/// sRGB8 → gray exactly as libgmsd's command-line tool (and MATLAB's
/// `rgb2gray` on uint8, to within its coefficient rounding):
/// `round(0.299·R + 0.587·G + 0.114·B)` in f64, half away from zero.
///
/// `rgb` holds `height` rows of `width` RGB triplets, rows `stride_bytes`
/// apart; `out` receives `width · height` packed samples.
pub fn rgb8_to_gray(
    rgb: &[u8],
    width: usize,
    height: usize,
    stride_bytes: usize,
    out: &mut [f32],
) -> Result<()> {
    check_rgb8(rgb, width, height, stride_bytes)?;
    if out.len() < width * height {
        return Err(Error::BufferTooSmall {
            expected: width * height,
            got: out.len(),
        });
    }
    // SIMD rows (`kernel::gray_row_body!`), bit-identical to the scalar
    // `kernel::gray_px` on every tier.
    kernel::rgb8_to_gray_plane(rgb, width, height, stride_bytes, out);
    Ok(())
}

fn check_rgb8(rgb: &[u8], width: usize, height: usize, stride_bytes: usize) -> Result<()> {
    let row_bytes = width * 3;
    if stride_bytes < row_bytes {
        return Err(Error::StrideTooSmall {
            width: row_bytes,
            stride: stride_bytes,
        });
    }
    let need = if height == 0 {
        0
    } else {
        (height - 1) * stride_bytes + row_bytes
    };
    if rgb.len() < need {
        return Err(Error::BufferTooSmall {
            expected: need,
            got: rgb.len(),
        });
    }
    Ok(())
}

/// GMSD of two sRGB8 images through [`rgb8_to_gray`].
pub fn gmsd_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride_bytes: usize,
) -> Result<GmsdScore> {
    check_rgb8(reference, width, height, stride_bytes)?;
    check_rgb8(distorted, width, height, stride_bytes)?;
    // Converted to gray row by row inside each band: no full-size gray
    // planes, and the conversion runs in the band's tier and thread.
    run_sources(
        kernel::Source::Rgb8 {
            data: reference,
            stride: stride_bytes,
        },
        kernel::Source::Rgb8 {
            data: distorted,
            stride: stride_bytes,
        },
        width,
        height,
        None,
    )
}

/// Mean Deviation Similarity Index, default summation model (Nafchi et al., 2016).
///
/// Both inputs contain gamma-encoded sRGB RGB8 triplets, with `height` rows
/// of `width` pixels, `stride_bytes` bytes apart (padding is ignored).
/// Scores are distances: smaller is better. Implemented from the paper: the
/// size-dependent zero-padded box filter, L/H/M transform, fusion and
/// complex-root mean absolute deviation pooling are evaluated in f64 (see
/// `docs/MDSI_CHOICES.md` for the readings taken where the paper is silent).
/// No ICC, alpha, orientation or transfer conversion is performed here.
///
/// Unlike GMSD's half-resolution rule, MDSI preserves odd edge samples.
/// Nonempty images, including a single pixel, are supported.
pub fn mdsi_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride_bytes: usize,
) -> Result<f64> {
    mdsi::run(reference, distorted, width, height, stride_bytes)
}

/// Paper-derived four-scale MS-GMSD distance, using gamma-encoded RGB8.
///
/// Rows have `stride_bytes` bytes and `width` interleaved RGB pixels. No
/// colour management, alpha or orientation processing is performed. This
/// variant uses unrounded YIQ, zero-padded Prewitt, c=170, and replicated
/// odd-edge 2x2 downsampling. It is not verified against author software;
/// the numerical conventions and independent oracle are recorded in the
/// gmsd-chroma benchmark preregistration.
pub fn ms_gmsd_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride_bytes: usize,
) -> Result<f64> {
    ms_gmsd::run(reference, distorted, width, height, stride_bytes).map(|s| s.0)
}

/// Paper-derived colour MS-GMSDc distance, on the inputs of [`ms_gmsd_rgb8`].
///
/// Adds joint I/Q RMSE at scale 3 with the published logistic fusion.
/// The same explicitly qualified numerical conventions apply; this is
/// agreement with a paper transcription, not author-software parity.
pub fn ms_gmsdc_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride_bytes: usize,
) -> Result<f64> {
    ms_gmsd::run(reference, distorted, width, height, stride_bytes).map(|s| s.1)
}

/// GMSD of two zenpixels images of the same size.
///
/// Each row is converted to sRGB RGB8 by zenpixels-convert (any depth, layout,
/// alpha or transfer it supports), then to gray with [`rgb8_to_gray`] — the
/// same 8-bit luma the reference implementation scores. Strided input is read
/// row by row; no full-size RGB copy is made.
#[cfg(feature = "pixels")]
pub fn gmsd_pixels(
    reference: &zenpixels::PixelSlice<'_>,
    distorted: &zenpixels::PixelSlice<'_>,
) -> Result<GmsdScore> {
    let (w, h) = (reference.width() as usize, reference.rows() as usize);
    let (dw, dh) = (distorted.width() as usize, distorted.rows() as usize);
    if (w, h) != (dw, dh) {
        return Err(Error::DimensionMismatch {
            reference: (w, h),
            distorted: (dw, dh),
        });
    }
    let r = pixels_to_gray(reference)?;
    let d = pixels_to_gray(distorted)?;
    gmsd(GrayImage::packed(&r, w, h)?, GrayImage::packed(&d, w, h)?)
}

/// Row-wise zenpixels → sRGB RGB8 → BT.601 gray (packed `w · h`).
#[cfg(feature = "pixels")]
pub fn pixels_to_gray(s: &zenpixels::PixelSlice<'_>) -> Result<Vec<f32>> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let target = zenpixels::PixelDescriptor::RGB8_SRGB;
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let mut out = alloc::vec![0.0f32; w * h];
    if s.descriptor() == target {
        for y in 0..h {
            rgb8_to_gray(s.row(y as u32), w, 1, w * 3, &mut out[y * w..(y + 1) * w])?;
        }
        return Ok(out);
    }
    let plan = ConvertPlan::new(s.descriptor(), target)
        .map_err(|e| Error::Convert(alloc::format!("{e}")))?;
    let mut rgb = alloc::vec![0u8; w * 3];
    for y in 0..h {
        convert_row(&plan, s.row(y as u32), &mut rgb, s.width());
        rgb8_to_gray(&rgb, w, 1, w * 3, &mut out[y * w..(y + 1) * w])?;
    }
    Ok(out)
}

/// Rows per band. Fixed, so the per-row work is identical at any thread
/// count; the reduction is per row anyway.
const BAND_ROWS: usize = 64;

fn run(
    reference: GrayImage<'_>,
    distorted: GrayImage<'_>,
    map: Option<&mut [f32]>,
) -> Result<GmsdScore> {
    if (reference.width, reference.height) != (distorted.width, distorted.height) {
        return Err(Error::DimensionMismatch {
            reference: (reference.width, reference.height),
            distorted: (distorted.width, distorted.height),
        });
    }
    fn plane<'a>(g: &GrayImage<'a>) -> kernel::Source<'a> {
        kernel::Source::Gray(kernel::Plane {
            data: g.data,
            stride: g.stride,
        })
    }
    run_sources(
        plane(&reference),
        plane(&distorted),
        reference.width,
        reference.height,
        map,
    )
}

fn run_sources(
    r: kernel::Source<'_>,
    d: kernel::Source<'_>,
    width: usize,
    height: usize,
    map: Option<&mut [f32]>,
) -> Result<GmsdScore> {
    let (w2, h2) = map_dims(width, height);
    let n = w2 * h2;
    if n < 2 {
        return Err(Error::TooSmall { width, height });
    }
    let mut sums: Vec<(f64, f64)> = alloc::vec![(0.0, 0.0); h2];
    let band_of = |i: usize| kernel::Band {
        reference: r,
        distorted: d,
        w2,
        h2,
        y0: i * BAND_ROWS,
        y1: ((i + 1) * BAND_ROWS).min(h2),
    };

    match map {
        Some(m) => {
            if m.len() < n {
                return Err(Error::BufferTooSmall {
                    expected: n,
                    got: m.len(),
                });
            }
            let m = &mut m[..n];
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                m.par_chunks_mut(BAND_ROWS * w2)
                    .zip(sums.par_chunks_mut(BAND_ROWS))
                    .enumerate()
                    .for_each(|(i, (m, s))| kernel::gmsd_band(&band_of(i), m, w2, s));
            }
            #[cfg(not(feature = "parallel"))]
            for (i, (m, s)) in m
                .chunks_mut(BAND_ROWS * w2)
                .zip(sums.chunks_mut(BAND_ROWS))
                .enumerate()
            {
                kernel::gmsd_band(&band_of(i), m, w2, s);
            }
        }
        None => {
            // No map requested: each band writes one reusable scratch row.
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                sums.par_chunks_mut(BAND_ROWS)
                    .enumerate()
                    .for_each(|(i, s)| {
                        let mut row = alloc::vec![0.0f32; w2];
                        kernel::gmsd_band(&band_of(i), &mut row, 0, s)
                    });
            }
            #[cfg(not(feature = "parallel"))]
            {
                let mut row = alloc::vec![0.0f32; w2];
                for (i, s) in sums.chunks_mut(BAND_ROWS).enumerate() {
                    kernel::gmsd_band(&band_of(i), &mut row, 0, s);
                }
            }
        }
    }

    Ok(pool(&sums, n))
}

/// Deviation pooling from per-row `(Σe, Σe²)`, `e = 1 − GMS`, summed in row
/// order so the result does not depend on banding or thread count.
fn pool(sums: &[(f64, f64)], n: usize) -> GmsdScore {
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &(a, b) in sums {
        s1 += a;
        s2 += b;
    }
    let nf = n as f64;
    let mean_e = s1 / nf;
    let var = (s2 - s1 * mean_e) / (nf - 1.0);
    let var = if var > 0.0 { var } else { 0.0 };
    GmsdScore {
        gmsd: sqrt_f64(var),
        mean_gms: 1.0 - mean_e,
    }
}

#[inline(always)]
fn sqrt_f32(v: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        v.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(v)
    }
}

#[inline(always)]
fn sqrt_f64(v: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        v.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        // Newton on f64 from an f32 seed; exact to f64 rounding after 3 steps.
        if v == 0.0 {
            return 0.0;
        }
        let mut x = sqrt_f32(v as f32) as f64;
        for _ in 0..3 {
            x = 0.5 * (x + v / x);
        }
        x
    }
}

/// Per-tier entry points for tier-parity tests and disassembly. Not API.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    pub use crate::kernel::gmsd_band_scalar;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::gmsd_band_v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::gmsd_band_v4;
    pub use crate::kernel::{Band, Plane, Source};
}

#[cfg(test)]
mod tests;
