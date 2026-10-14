//! Shared metric abstractions — the object-surface every metric crate
//! implements so consumers (`zenmetrics-api`, the fleet diffmap executor,
//! the browser viewer, batch scorers) talk to *a* metric, not N crates.
//!
//! Two levels:
//!
//! - [`Metric`] — the generic scoring surface. Inputs are zenpixels
//!   [`PixelSlice`]s: layout, depth and transfer live in the slice's
//!   [`PixelDescriptor`], so one signature covers sRGB8, f32 planes and
//!   PQ content without per-metric entry-point sprawl. Implementations
//!   accept the descriptors they natively understand and refuse the
//!   rest — conversion policy stays at the caller.
//! - [`Diffmap`] — the canonical per-pixel map, where the metric has one
//!   (upstream `FSIMmap`, VSI `qualitymap`, SSIM `ssim_map`, HaarPSI
//!   similarity map, HDR-VDP `P_map`, GMSD `gms`, cvvdp/butteraugli
//!   diffmaps). Object-safe: `Box<dyn Diffmap>` is the owned form.
//!
//! Every metric implements these on **its own structs** — the metric-
//! specific extras (butteraugli's norm variants, HDR-VDP's `p_det`/
//! `q_mos`, FSIM's `fsimc`) ride alongside the trait surface, and the
//! trait is the escape hatch for heterogeneous callers.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::iter::Iterator;
use zenpixels::{PixelCow, PixelSlice};

/// Why a bool can't carry map semantics: "higher is better" is one of
/// three conventions in-tree, and the *meaning* differs too — a
/// similarity map, a masked-error map and a detection-probability map
/// all answer "where" but normalize differently.
///
/// `#[non_exhaustive]`: new metrics may add conventions (e.g.
/// `ConfidenceInterval`, `SnrDb`) — match arms must keep a `_` arm.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MapPolarity {
    /// Higher = *more similar* — FSIM `FSIMmap`, HaarPSI similarity map,
    /// SSIM/MS-SSIM `ssim_map`, GMSD `gms`, VSI `qualitymap`.
    /// Bounded `[0, 1]`-ish; `1` = identical.
    Similarity,
    /// Higher = *more error / more visible difference* — butteraugli's
    /// diffmap, cvvdp's masked-error map. `0` = identical.
    Difference,
    /// Higher = *probability the difference is detected* — HDR-VDP's
    /// `P_map`. Bounded `[0, 1]`; `0.5` is the detection threshold.
    DetectionProbability,
}

/// What a metric's headline [`Metric::score`] number means — direction
/// AND scale, since "0 bad" vs "10 perfect" vs "unbounded" are all in
/// the tree.
///
/// `#[non_exhaustive]`: new metrics may add conventions — match arms
/// must keep a `_` arm.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreKind {
    /// `[0,1]` similarity — `1.0` = identical, lower = worse.
    /// (SSIM family, FSIM, HaarPSI, VSI, IW-SSIM, NLPD.)
    Similarity,
    /// `0` = identical, higher = worse. (MAD, PSNR-HVS's difference
    /// magnitude, GMSD's std, butteraugli's norm.)
    Difference,
    /// JOD scale — `10` = perfect, lower = worse. (cvvdp, HDR-VDP-3's
    /// `Q_JOD`.)
    Jod,
    /// `100` = perfect, lower = worse (HDR-VDP-2's `Q` correlate).
    Quality100,
    /// Higher = better, unbounded (PSNR, PSNR-HVS in dB).
    Decibels,
}

/// One scoring call's scalar outputs — headline plus every named scalar
/// the metric computes, so `Box<dyn Metric>` callers don't lose
/// `fsimc`/`hi`/`lo`/`q_mos`-style secondary channels.
///
/// `named` contains the headline value under its canonical key (`"fsim"`,
/// `"q_jod"`, …) plus the rest. `#[non_exhaustive]` — fields may be
/// added; construct via [`MetricScores::single`] then [`Self::push`].
#[non_exhaustive]
#[derive(Debug)]
pub struct MetricScores {
    /// The headline score in the units [`Metric::score_kind`] declares.
    pub headline: f64,
    /// `(name, value)` pairs — headline first, then secondaries.
    pub named: Vec<(Cow<'static, str>, f64)>,
}

impl MetricScores {
    /// A single-scalar result: `named = [(name, headline)]`.
    pub fn single(name: &'static str, headline: f64) -> Self {
        Self {
            headline,
            named: alloc::vec![(Cow::Borrowed(name), headline)],
        }
    }

    /// Append a secondary scalar.
    pub fn push(&mut self, name: &'static str, value: f64) {
        self.named.push((Cow::Borrowed(name), value));
    }
}

/// Error for the generic surface — leaf errors are boxed through this
/// so `Box<dyn Metric>` doesn't need a concrete `Error` associated type.
pub type MetricResult<T> = Result<T, Box<dyn core::error::Error + Send + Sync + 'static>>;

/// The generic metric surface — implementors are per-crate scorer
/// structs, and `Box<dyn Metric>` is the heterogeneous-caller form.
///
/// Inputs are `PixelSlice`s — the descriptor is the contract: a metric
/// declares (in its own docs) which descriptors it accepts natively and
/// returns `Err` on the rest. Conversion is the caller's concern —
/// zenpixels-convert handles descriptor adaptation.
///
/// Object-safe, `Send`/`Sync`-neutral (scorers may hold scratch).
pub trait Metric {
    /// Stable metric id — matches fleet/job names (`"fsim"`, `"cvvdp"`,
    /// `"hdrvdp"`).
    fn name(&self) -> &'static str;

    /// What [`Self::score`]'s headline means — direction + scale.
    fn score_kind(&self) -> ScoreKind;

    /// Score a pair. `r`/`d` must cover the same image region; dims and
    /// supported descriptors are the implementation's contract.
    fn score(&mut self, r: PixelSlice<'_>, d: PixelSlice<'_>) -> MetricResult<MetricScores>;

    /// Score a pair and produce the canonical per-pixel map when the
    /// metric publishes one. Default: `Err(NoMap)`.
    fn score_with_map(
        &mut self,
        _r: PixelSlice<'_>,
        _d: PixelSlice<'_>,
    ) -> MetricResult<(MetricScores, Box<dyn Diffmap>)> {
        Err(Box::new(NoMap))
    }
}

/// `Metric::score_with_map` on a metric that publishes no map.
#[derive(Debug, Clone, Copy)]
pub struct NoMap;

impl core::fmt::Display for NoMap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("metric publishes no per-pixel map")
    }
}
impl core::error::Error for NoMap {}

/// A metric's per-pixel map — "where is the difference/similarity".
///
/// Object-safe: score functions return `Box<dyn Diffmap>` (owned) when
/// callers need polymorphism, or a concrete `impl Diffmap` when they
/// want the metric's own fields too.
///
/// ## Extending the trait
///
/// New methods always ship with a default impl — adding a method is
/// additive for implementors.
pub trait Diffmap {
    /// The map as `Gray<f32>` pixels, row-major
    /// (`pixels().as_slice().descriptor()` is
    /// `PixelDescriptor::GRAYF32_LINEAR`). Width, height and stride live
    /// in the slice — the map's resolution may legitimately differ from
    /// the input image (e.g. GMSD is `w/2 × h/2`); check
    /// `pixels().width()`/`rows()` instead of assuming.
    ///
    /// `PixelCow::Borrowed` means the view aliases scorer-owned scratch —
    /// cheap, but tied to the impl's borrow. `Owned` is standalone.
    ///
    /// The erased (`()`) representation avoids a `rgb`-crate dep in leaf
    /// metrics — consumers read `&[f32]` rows via `as_strided_bytes()` +
    /// `bytemuck`, or `try_typed::<rgb::Gray<f32>>()` when they have it.
    fn pixels(&self) -> PixelCow<'_, ()>;

    /// The map's polarity+semantics — callers folding maps into scores
    /// or visualizing them MUST consult this (a similarity map and an
    /// error map are inverse under display).
    fn polarity(&self) -> MapPolarity;

    /// The scalar score computed alongside the map (lets consumers check
    /// map-vs-score coherence without re-running the metric).
    fn score(&self) -> f64;

    /// Metric-specific metadata as `(key, value)` pairs with stable
    /// `'static` keys — e.g. butteraugli's `pnorm`/`max_norm`, HDR-VDP's
    /// `p_det`/`jod`, per-scale labels. Default: empty.
    fn extras(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, Cow<'static, str>)> + '_> {
        Box::new(::core::iter::empty())
    }
}

/// View a packed `w × h` `f32` plane as a `GRAYF32_LINEAR` pixel slice —
/// the shape every leaf metric hands to its `Diffmap::pixels()`.
///
/// # Panics
/// If `data.len() != w·h`.
#[must_use]
pub fn gray_f32_slice(data: &[f32], w: u32, h: u32) -> zenpixels::PixelSlice<'_> {
    assert_eq!(
        data.len(),
        w as usize * h as usize,
        "diffmap slice len != w*h"
    );
    let bytes: &[u8] = bytemuck::cast_slice(data);
    zenpixels::PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * core::mem::size_of::<f32>(),
        zenpixels::PixelDescriptor::GRAYF32_LINEAR,
    )
    .expect("packed f32 slice always validates as GRAYF32_LINEAR")
}

/// Move a `Vec<f32>` into `Vec<u8>` for `PixelBuffer::from_vec` —
/// `bytemuck::cast_vec` when the capacity aligns (f32 → u8 always does:
/// both alignments divide the f32 allocs), else a by-value copy.
#[must_use]
pub fn bytemuck_vec(pixels: alloc::vec::Vec<f32>) -> alloc::vec::Vec<u8> {
    match bytemuck::allocation::try_cast_vec::<f32, u8>(pixels) {
        ::core::result::Result::Ok(v) => v,
        ::core::result::Result::Err((_, v)) => bytemuck::cast_slice(&v).to_vec(),
    }
}

/// A `PixelSlice` input the metric can't take — the descriptor declares
/// a layout/depth/transfer the implementation doesn't handle natively.
#[derive(Debug, Clone)]
pub struct UnsupportedInput {
    /// The offending descriptor.
    pub descriptor: zenpixels::PixelDescriptor,
}

impl core::fmt::Display for UnsupportedInput {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unsupported input descriptor {:?}", self.descriptor)
    }
}
impl core::error::Error for UnsupportedInput {}

/// A borrowed strided sRGB8 plane set — what [`rgb8_pair`] returns.
#[derive(Clone, Copy)]
pub struct Rgb8In<'a> {
    /// `RGBRGB…` bytes, rows `stride` apart.
    pub pixels: &'a [u8],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Row stride in **bytes** (≥ `width·3`).
    pub stride: usize,
}

/// Borrow two [`PixelSlice`]s as strided packed sRGB8 — the shared input
/// adapter for the `RGB8_SRGB`-native metrics (fsim, haarpsi, vsi,
/// msssim, gmsd).
///
/// # Errors
/// [`UnsupportedInput`] when either slice isn't `RGB8_SRGB`; the pair
/// must agree on dimensions.
pub fn rgb8_pair<'a>(
    r: PixelSlice<'a>,
    d: PixelSlice<'a>,
) -> Result<(Rgb8In<'a>, Rgb8In<'a>), UnsupportedInput> {
    let take = |s: PixelSlice<'a>| -> Result<Rgb8In<'a>, UnsupportedInput> {
        if s.descriptor() != zenpixels::PixelDescriptor::RGB8_SRGB {
            return Err(UnsupportedInput {
                descriptor: s.descriptor(),
            });
        }
        Ok(Rgb8In {
            pixels: s.as_strided_bytes(),
            width: s.width() as usize,
            height: s.rows() as usize,
            stride: s.stride(),
        })
    };
    Ok((take(r)?, take(d)?))
}
