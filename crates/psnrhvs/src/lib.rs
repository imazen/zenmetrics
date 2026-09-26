//! Pure-Rust CPU port of **PSNR-HVS** and **PSNR-HVS-M** —
//! K. Egiazarian, J. Astola, N. Ponomarenko, V. Lukin, F. Battisti,
//! M. Carli, *New full-reference quality metrics based on HVS* (VPQM-06)
//! and N. Ponomarenko, F. Silvestri, K. Egiazarian, M. Carli, J. Astola,
//! V. Lukin, *On between-coefficient contrast masking of DCT basis
//! functions* (VPQM-07).
//!
//! # Algorithm
//!
//! The image is tiled into non-overlapping 8×8 blocks (the reference's
//! default `wstep = 8`; partial strips at the right/bottom edge are
//! dropped, as the reference does). Per block:
//!
//! 1. Orthonormal 2D DCT-II of both blocks (MATLAB `dct2` convention).
//! 2. **Masking threshold** (`maskeff`): the non-DC coefficient energy
//!    `e = Σ zdct²·MaskCof` scaled by the ratio of the four 4×4-quadrant
//!    variances to the whole-block variance,
//!    `m = sqrt(e·pop)/32`, and the block mask is `max(m_ref, m_dist)`.
//! 3. Per coefficient `u = |A_dct − B_dct|`: the **PSNR-HVS** energy
//!    accumulates `(u·CSFCof)²`; the **PSNR-HVS-M** energy additionally
//!    subtracts the masking threshold `mask/MaskCof(k,l)` on non-DC
//!    coefficients (clamped at 0). DC is never masked.
//! 4. `S/Num` is the mean per-coefficient energy; the score is
//!    `10·log10(255²/S)`, or `100000` when `S = 0` (visually
//!    indistinguishable / identical).
//!
//! Per-block energies are summed in f32 in a fixed coefficient order and
//! accumulated into the global totals in f64 in raster order, so every
//! SIMD tier and every thread count produces bit-identical output. The
//! reference computes in f64 throughout; the port's f32 blocks hold the
//! score within ~1e-5 relative of the f64 reference — far below the
//! metric's own precision — while the f64 global accumulation keeps the
//! sum exact. `tests.rs` pins the port against reference-implementation
//! goldens.
//!
//! # SIMD
//!
//! One `#[arcane]` entry per tier (`v4` AVX-512 behind crate feature
//! `avx512`, `v3` AVX2, `neon`, `wasm128`, `scalar`), processing
//! `LANES` blocks per batch in a coefficient-major layout. All lanes are
//! independent and every reduction has a fixed order, so all widths are
//! bit-identical. `_dev` exposes the per-tier entries for tier-parity
//! tests.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::too_many_arguments)]
// excessive_precision: the DCT/CSF/masking tables are transcribed from
// the reference at full precision; they round to the same f32 values
// (same allow as crates/cvvdp, crates/iwssim).
#![allow(clippy::excessive_precision)]
// assign_op_pattern: the shared `band_body!` macro accumulates with
// `acc = acc + term` — `f32xN` has no `AddAssign`, so the suggested
// `+=` would not compile (and `x = x + y` keeps the fixed-order
// accumulation explicit).
#![allow(clippy::assign_op_pattern)]

extern crate alloc;

pub mod daala;
mod kernel;

/// Per-tier entry points for tier-parity tests and disassembly. Not API.
#[cfg(feature = "_dev")]
#[doc(hidden)]
pub mod dev {
    pub use crate::kernel::psnrhvs_band_scalar;
    #[cfg(target_arch = "x86_64")]
    pub use crate::kernel::psnrhvs_band_v3;
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    pub use crate::kernel::psnrhvs_band_v4;
}

#[cfg(test)]
mod tests;

/// Stable column-name identifier for sweep sidecars:
/// `psnrhvs_imazen_v<MAJOR>_<MINOR>_<PATCH>` (overridable at build time
/// via `PSNRHVS_IMPL_TAG`). The `psnrhvs` CLI metric emits the
/// per-channel mean of PSNR-HVS over R, G, B.
pub const PSNRHVS_COLUMN_NAME: &str = match option_env!("PSNRHVS_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "psnrhvs_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **masked** variant (`psnrhvs-m` CLI metric):
/// `psnrhvsm_imazen_v<MAJOR>_<MINOR>_<PATCH>` (`PSNRHVSM_IMPL_TAG`).
pub const PSNRHVSM_COLUMN_NAME: &str = match option_env!("PSNRHVSM_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "psnrhvsm_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **luma-only** variant (`psnrhvs-y` CLI metric):
/// `psnrhvsy_imazen_v<MAJOR>_<MINOR>_<PATCH>` (`PSNRHVSY_IMPL_TAG`).
pub const PSNRHVSY_COLUMN_NAME: &str = match option_env!("PSNRHVSY_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "psnrhvsy_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Column name for the **masked luma** variant (the `psnrhvs-m` score of
/// the `psnrhvs-y` luma pair): `psnrhvsym_imazen_v<MAJOR>_<MINOR>_<PATCH>`
/// (`PSNRHVSYM_IMPL_TAG`).
pub const PSNRHVSYM_COLUMN_NAME: &str = match option_env!("PSNRHVSYM_IMPL_TAG") {
    Some(t) => t,
    None => concat!(
        "psnrhvsym_imazen_v",
        env!("CARGO_PKG_VERSION_MAJOR"),
        "_",
        env!("CARGO_PKG_VERSION_MINOR"),
        "_",
        env!("CARGO_PKG_VERSION_PATCH")
    ),
};

/// Both PSNR-HVS scores of one 0..255-scale plane pair, computed in a
/// single pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaneScore {
    /// CSF-weighted PSNR (Egiazarian et al. 2006): higher is better,
    /// `100000.0` when the images are identical under the metric.
    pub psnr_hvs: f64,
    /// CSF + contrast-masking weighted PSNR (Ponomarenko et al. 2007).
    /// `>= psnr_hvs` on the same pair.
    pub psnr_hvs_m: f64,
    /// Number of complete 8×8 blocks scored.
    pub blocks: usize,
}

/// Per-channel PSNR-HVS scores of an interleaved sRGB8 pair plus their
/// arithmetic mean — the multi-plane convention used by codec evaluation
/// harnesses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RgbScore {
    /// Red channel plane score.
    pub red: PlaneScore,
    /// Green channel plane score.
    pub green: PlaneScore,
    /// Blue channel plane score.
    pub blue: PlaneScore,
    /// Mean of `red/green/blue` `psnr_hvs`.
    pub psnr_hvs: f64,
    /// Mean of `red/green/blue` `psnr_hvs_m`.
    pub psnr_hvs_m: f64,
}

/// Input errors.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// Image smaller than one 8×8 block on either axis — the metric has
    /// no blocks to score (the reference leaves its outputs unset here).
    TooSmall {
        /// Image width in pixels.
        width: usize,
        /// Image height in pixels.
        height: usize,
    },
    /// `stride` is smaller than `width` — rows would overlap.
    InvalidStride {
        /// Supplied stride.
        stride: usize,
        /// Image width.
        width: usize,
    },
    /// `step` was zero — the block grid would be empty/invalid.
    InvalidStep,
    /// A plane slice is shorter than `stride·(height−1) + width`.
    BufferLength {
        /// Required minimum length.
        needed: usize,
        /// Supplied length.
        got: usize,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Error::TooSmall { width, height } => write!(
                f,
                "psnrhvs needs at least one complete 8x8 block; got {width}x{height}"
            ),
            Error::InvalidStride { stride, width } => write!(
                f,
                "psnrhvs stride {stride} is smaller than the {width}-pixel row"
            ),
            Error::InvalidStep => write!(f, "psnrhvs block step must be at least 1"),
            Error::BufferLength { needed, got } => {
                write!(
                    f,
                    "psnrhvs needs a plane of at least {needed} samples; got {got}"
                )
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

fn check_inputs(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<(), Error> {
    if width < 8 || height < 8 {
        return Err(Error::TooSmall { width, height });
    }
    if stride < width {
        return Err(Error::InvalidStride { stride, width });
    }
    let needed = stride * (height - 1) + width;
    if reference.len() < needed || distorted.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: reference.len().min(distorted.len()),
        });
    }
    Ok(())
}

fn finalize(s1: f64, s2: f64, blocks: usize) -> PlaneScore {
    let num = (64 * blocks) as f64;
    let m1 = s1 / num;
    let m2 = s2 / num;
    let c = 255.0f64 * 255.0;
    // `libm::log10`, not the platform libm — bit-identical across the
    // fleet (same precedent as zenfleet-core's `ln`).
    PlaneScore {
        psnr_hvs_m: if m1 == 0.0 {
            100000.0
        } else {
            10.0 * libm::log10(c / m1)
        },
        psnr_hvs: if m2 == 0.0 {
            100000.0
        } else {
            10.0 * libm::log10(c / m2)
        },
        blocks,
    }
}

/// PSNR-HVS / PSNR-HVS-M of two strided f32 planes on the 0..255 scale,
/// block step 8 (the reference's default `wstep`). Returns
/// [`Error::TooSmall`] below 8×8.
pub fn psnrhvs_plane_f32(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<PlaneScore, Error> {
    psnrhvs_plane_f32_step(reference, distorted, width, height, stride, 8)
}

/// [`psnrhvs_plane_f32`] with an explicit block step (the reference's
/// `wstep`; `step < 8` overlaps blocks). `step` must be >= 1.
pub fn psnrhvs_plane_f32_step(
    reference: &[f32],
    distorted: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    step: usize,
) -> Result<PlaneScore, Error> {
    if step == 0 {
        return Err(Error::InvalidStep);
    }
    check_inputs(reference, distorted, width, height, stride)?;
    let nb_x = 1 + (width - 8) / step;
    let nb_y = 1 + (height - 8) / step;
    let nb = kernel::n_bands(nb_y);
    let sz = nb_y.div_ceil(nb);
    let parts = kernel::collect_bands(nb, |b| {
        let lo = b * sz;
        let hi = (lo + sz).min(nb_y);
        if lo >= hi {
            return (0.0, 0.0);
        }
        kernel::psnrhvs_band(reference, distorted, stride, lo, hi, nb_x, step)
    });
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for (a, b) in parts {
        s1 += a;
        s2 += b;
    }
    Ok(finalize(s1, s2, nb_x * nb_y))
}

/// PSNR-HVS / PSNR-HVS-M of two interleaved sRGB RGB8 pairs — each
/// channel scored as its own 0..255 plane, plus the per-channel mean
/// (the multi-plane convention codec harnesses report).
pub fn psnrhvs_rgb8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<RgbScore, Error> {
    if width < 8 || height < 8 {
        return Err(Error::TooSmall { width, height });
    }
    let needed = stride * (height - 1) + 3 * width;
    if stride < 3 * width || reference.len() < needed || distorted.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: reference.len().min(distorted.len()),
        });
    }
    let mut planes_r = [
        alloc::vec![0.0f32; width * height],
        alloc::vec![0.0f32; width * height],
        alloc::vec![0.0f32; width * height],
    ];
    let mut planes_d = [
        alloc::vec![0.0f32; width * height],
        alloc::vec![0.0f32; width * height],
        alloc::vec![0.0f32; width * height],
    ];
    for y in 0..height {
        let rr = &reference[y * stride..y * stride + 3 * width];
        let dd = &distorted[y * stride..y * stride + 3 * width];
        for x in 0..width {
            for c in 0..3 {
                planes_r[c][y * width + x] = rr[3 * x + c] as f32;
                planes_d[c][y * width + x] = dd[3 * x + c] as f32;
            }
        }
    }
    let red = psnrhvs_plane_f32(&planes_r[0], &planes_d[0], width, height, width)?;
    let green = psnrhvs_plane_f32(&planes_r[1], &planes_d[1], width, height, width)?;
    let blue = psnrhvs_plane_f32(&planes_r[2], &planes_d[2], width, height, width)?;
    Ok(RgbScore {
        psnr_hvs: (red.psnr_hvs + green.psnr_hvs + blue.psnr_hvs) / 3.0,
        psnr_hvs_m: (red.psnr_hvs_m + green.psnr_hvs_m + blue.psnr_hvs_m) / 3.0,
        red,
        green,
        blue,
    })
}

/// PSNR-HVS / PSNR-HVS-M of two interleaved sRGB RGB8 pairs on BT.601
/// luma — the same `round(0.299·R + 0.587·G + 0.114·B)` luma `gmsd` scores,
/// so `psnrhvs-y` numbers are directly comparable to codec-harness
/// PSNR-HVS-Y columns.
pub fn psnrhvs_luma8(
    reference: &[u8],
    distorted: &[u8],
    width: usize,
    height: usize,
    stride: usize,
) -> Result<PlaneScore, Error> {
    if width < 8 || height < 8 {
        return Err(Error::TooSmall { width, height });
    }
    let needed = stride * (height - 1) + 3 * width;
    if stride < 3 * width || reference.len() < needed || distorted.len() < needed {
        return Err(Error::BufferLength {
            needed,
            got: reference.len().min(distorted.len()),
        });
    }
    let mut yr = alloc::vec![0.0f32; width * height];
    let mut yd = alloc::vec![0.0f32; width * height];
    for y in 0..height {
        let rr = &reference[y * stride..y * stride + 3 * width];
        let dd = &distorted[y * stride..y * stride + 3 * width];
        for x in 0..width {
            yr[y * width + x] = gray_px(rr[3 * x], rr[3 * x + 1], rr[3 * x + 2]);
            yd[y * width + x] = gray_px(dd[3 * x], dd[3 * x + 1], dd[3 * x + 2]);
        }
    }
    psnrhvs_plane_f32(&yr, &yd, width, height, width)
}

/// sRGB8 → BT.601 luma of one pixel, identical to `gmsd`'s `gray_px`:
/// `((0.299·R + 0.587·G) + 0.114·B)` in f64, then `floor(v + 0.5)`.
#[inline(always)]
fn gray_px(r: u8, g: u8, b: u8) -> f32 {
    let v = 0.299 * r as f64 + 0.587 * g as f64 + 0.114 * b as f64;
    (v + 0.5) as i64 as f32
}
