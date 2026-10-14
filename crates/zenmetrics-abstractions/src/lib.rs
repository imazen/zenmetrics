//! Shared metric abstractions — the object surface every metric crate
//! implements so consumers (`zenmetrics-api`, the fleet diffmap executor,
//! the browser viewer, batch scorers) talk to *a* metric, not N crates.
//!
//! ## The shape
//!
//! [`Metric`] has one entry point: [`Metric::evaluate`] takes two
//! zenpixels [`PixelSlice`]s (layout/depth/transfer ride in the
//! [`PixelDescriptor`]) and returns `Box<dyn Evaluation>` — the metric's
//! owned result object. Three output shapes fold into it:
//!
//! - **score-only** — `Evaluation::map()` returns `None`
//!   (mad-iqa, iwssim, nlpd — no canonical upstream map);
//! - **score + canonical map** — `map()` returns the per-pixel plane
//!   (`FSIMmap`, VSI `qualitymap`, `ssim_map`, HDR-VDP `P_map`, GMSD
//!   `gms`, cvvdp/butteraugli diffmaps), `map_polarity()` says which way
//!   is better;
//! - **map-as-artifact** — the headline score is *one fold* over the
//!   map and the caller can ask for others: `map_score("pnorm:12")`,
//!   `"mean"`, `"std"`, `"max"` are default-implemented for any map;
//!   metrics override for their own fold grammars.
//!
//! `extras()` is the open-ended metadata channel (butteraugli's norm
//! variants, HDR-VDP's `p_det`/`q_mos`), and `as_any()` downcasts to the
//! concrete result type when a caller needs the metric's real API.
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::vec::Vec;
/// Re-export of the pixel crate the trait signatures are written in —
/// implementors don't need their own `zenpixels` dep; write
/// `zenmetrics_abstractions::zenpixels::PixelSlice`.
pub use zenpixels;
use zenpixels::{PixelCow, PixelSlice};

// ---------------------------------------------------------------------------
// Semantics enums.
// ---------------------------------------------------------------------------

/// What a per-pixel map's values mean — direction AND semantics, since
/// "higher is better" is one of three conventions in-tree and the maps
/// normalize differently.
///
/// `#[non_exhaustive]`: new metrics may add conventions — match arms
/// must keep a `_` arm.
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

/// What a metric's headline [`Evaluation::score`] number means —
/// direction + scale, since "0 bad" vs "10 perfect" vs "unbounded" are
/// all in the tree.
///
/// `#[non_exhaustive]`: new metrics may add conventions — match arms
/// must keep a `_` arm.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScoreKind {
    /// `[0,1]` similarity — `1.0` = identical, lower = worse.
    /// (SSIM family, FSIM, HaarPSI, VSI, IW-SSIM, NLPD.)
    Similarity,
    /// `0` = identical, higher = worse. (MAD, GMSD's std, butteraugli's
    /// norm.)
    Difference,
    /// JOD scale — `10` = perfect, lower = worse. (cvvdp, HDR-VDP-3's
    /// `Q_JOD`.)
    Jod,
    /// `100` = perfect, lower = worse (HDR-VDP-2's `Q` correlate).
    Quality100,
    /// Higher = better, unbounded dB (PSNR, PSNR-HVS).
    Decibels,
}

// ---------------------------------------------------------------------------
// The result object.
// ---------------------------------------------------------------------------

/// One comparison's full output — the metric's owned result object,
/// returned as `Box<dyn Evaluation>` from [`Metric::evaluate`].
///
/// ## Extending the trait
///
/// Every method past [`Self::score`] has a default impl — adding a
/// method is additive for implementors.
pub trait Evaluation {
    /// The headline score in the units [`Metric::score_kind`] declares.
    fn score(&self) -> f64;

    /// Every named scalar this comparison produced — headline first
    /// under its canonical key (`"fsimc"`, `"jod"`, `"q"`), then
    /// secondaries (`"fsim"`, `"hi"`/`"lo"`/`"mad"`, `pnorm`/`max_norm`,
    /// …). String-keyed so heterogeneous consumers can log/persist
    /// without knowing the metric — typed accessors stay on the
    /// concrete type (reachable via [`Self::as_any`]).
    ///
    /// Default: a single `("score", self.score())` pair.
    fn scores(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, f64)> + '_> {
        Box::new(core::iter::once((Cow::Borrowed("score"), self.score())))
    }

    /// The canonical per-pixel map as `Gray<f32>` pixels, row-major —
    /// `None` when the metric produces no map. Width/height/stride live
    /// in the slice; the map's resolution may legitimately differ from
    /// the input image (GMSD is `w/2 × h/2`, MS-SSIM's level-0 map is the
    /// `(w−10) × (h−10)` valid-conv interior) — read
    /// `pixels().width()`/`rows()`, never assume.
    ///
    /// `PixelCow::Borrowed` means the view aliases result-owned scratch;
    /// `Owned` is standalone.
    fn map(&self) -> Option<PixelCow<'_, ()>> {
        None
    }

    /// The map's polarity+semantics — `None` iff [`Self::map`] is `None`.
    /// Callers folding maps into scores or visualizing them MUST consult
    /// this (a similarity map and an error map are inverse under display).
    fn map_polarity(&self) -> Option<MapPolarity> {
        None
    }

    /// A named fold over the map — "another score, computed on demand".
    ///
    /// The shared default handles `"mean"`, `"max"`, `"std"`, and
    /// `"pnorm:<p>"` for any map — that's the butteraugli-norm shape
    /// (`pnorm:12` is butteraugli's canonical combine) and it works for
    /// every error/similarity map in-tree. Metrics may override to add
    /// their own grammar or reject specs they don't define.
    fn map_score(&self, spec: &str) -> Option<f64> {
        map_fold(self.map()?, spec)
    }

    /// Metric-specific metadata as `(key, value)` pairs with stable
    /// `'static` keys — HDR-VDP's `p_det`/`jod`/`visible_fraction`,
    /// per-scale diagnostics, decode warnings. Default: empty.
    fn extras(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, Cow<'static, str>)> + '_> {
        Box::new(core::iter::empty())
    }

    /// Downcast to the metric's concrete result type when a caller needs
    /// its typed surface (butteraugli's `norm_p`, HDR-VDP's `c_map`, …).
    fn as_any(&self) -> &dyn core::any::Any;
}

/// A folded scalar `Evaluation` for score-only metrics — the concrete
/// `Box<dyn Evaluation>` when there's no map and no richer result.
#[derive(Debug)]
pub struct ScalarEval {
    /// Headline + secondary `(name, value)` pairs, headline first.
    pub named: Vec<(Cow<'static, str>, f64)>,
}

impl ScalarEval {
    /// A single-named-score evaluation.
    pub fn single(name: &'static str, headline: f64) -> Self {
        Self {
            named: alloc::vec![(Cow::Borrowed(name), headline)],
        }
    }

    /// Append a secondary scalar — owned names allowed.
    pub fn push(&mut self, name: impl Into<Cow<'static, str>>, value: f64) {
        self.named.push((name.into(), value));
    }
}

impl Evaluation for ScalarEval {
    fn score(&self) -> f64 {
        self.named.first().map_or(f64::NAN, |(_, v)| *v)
    }

    fn scores(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, f64)> + '_> {
        Box::new(self.named.iter().cloned())
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// The metric surface.
// ---------------------------------------------------------------------------

/// What [`Metric::evaluate`] should compute — `#[non_exhaustive]` so new
/// optional outputs (uncertainty maps, per-channel results) add fields
/// without breaking callers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default)]
pub struct EvalOpts {
    /// Produce the canonical per-pixel map. Costs one extra
    /// width×height·4-byte allocation and the map-write work — free for
    /// metrics whose map falls out of pooling anyway, real extra work
    /// for cvvdp (masked-error accumulation). Default `false`.
    pub map: bool,
}

impl EvalOpts {
    /// Score only.
    pub const SCORE_ONLY: Self = Self { map: false };
    /// Score + canonical map.
    pub const WITH_MAP: Self = Self { map: true };
}

/// Error for the generic surface — leaf errors are boxed through this
/// so `Box<dyn Metric>` doesn't need a concrete `Error` associated type.
pub type MetricResult<T> = Result<T, Box<dyn core::error::Error + Send + Sync + 'static>>;

/// The generic metric surface — implementors are per-crate scorer
/// structs, `Box<dyn Metric>` is the heterogeneous-caller form.
///
/// Object-safe; `Send`/`Sync`-neutral (scorers may hold scratch).
pub trait Metric {
    /// Stable metric id — matches fleet/job names (`"fsim"`, `"cvvdp"`,
    /// `"hdrvdp"`).
    fn name(&self) -> &'static str;

    /// What [`Evaluation::score`]'s headline means — direction + scale.
    fn score_kind(&self) -> ScoreKind;

    /// Whether [`Self::evaluate`] accepts input slices with this
    /// descriptor. Metrics declare their native inputs; conversion
    /// (zenpixels-convert) is the caller's concern.
    fn accepts(&self, descriptor: zenpixels::PixelDescriptor) -> bool;

    /// Score a pair. `r`/`d` must cover the same image region; the
    /// implementation documents which descriptors it accepts (see
    /// [`Self::accepts`]) and what `opts` it honors.
    fn evaluate(
        &mut self,
        r: PixelSlice<'_>,
        d: PixelSlice<'_>,
        opts: EvalOpts,
    ) -> MetricResult<Box<dyn Evaluation>>;
}

/// Returned when a map-requesting caller (`EvalOpts::map` / a
/// `with_diffmap` entry) hits a metric that publishes no map.
/// — a typed error for API surfaces that promise a map; on the generic
/// surface `Evaluation::map()` returning `None` is the non-error form.
#[derive(Debug, Clone, Copy)]
pub struct NoMap;

impl core::fmt::Display for NoMap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("metric publishes no per-pixel map")
    }
}
impl core::error::Error for NoMap {}

// ---------------------------------------------------------------------------
// Input adapters.
// ---------------------------------------------------------------------------

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
/// msssim, gmsd, cvvdp).
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
    let (a, b) = (take(r)?, take(d)?);
    if (a.width, a.height) != (b.width, b.height) {
        return Err(UnsupportedInput {
            descriptor: zenpixels::PixelDescriptor::RGB8_SRGB,
        });
    }
    Ok((a, b))
}

// ---------------------------------------------------------------------------
// Map helpers.
// ---------------------------------------------------------------------------

/// Named folds over a `GRAYF32_LINEAR` map — the shared default grammar
/// behind [`Evaluation::map_score`].
///
/// - `"mean"` — arithmetic mean
/// - `"max"` / `"min"` — extremes
/// - `"std"` — population standard deviation (GMSD's headline fold)
/// - `"pnorm:<p>"` — `(Σ|v|^p / n)^(1/p)` — butteraugli's norm family
///
/// Returns `None` on an empty map or an unrecognized spec.
#[must_use]
pub fn map_fold(map: PixelCow<'_, ()>, spec: &str) -> Option<f64> {
    let vals = map_f32s(&map)?;
    if vals.is_empty() {
        return None;
    }
    match spec {
        "mean" => Some(vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64),
        "max" => vals.iter().copied().reduce(f32::max).map(|v| v as f64),
        "min" => vals.iter().copied().reduce(f32::min).map(|v| v as f64),
        "std" => {
            let n = vals.len() as f64;
            let mean = vals.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = vals
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            Some(libm::sqrt(var))
        }
        _ => spec
            .strip_prefix("pnorm:")
            .and_then(|p| p.parse::<f64>().ok())
            .map(|p| {
                let n = vals.len() as f64;
                let sum: f64 = vals.iter().map(|&v| libm::pow((v as f64).abs(), p)).sum();
                (sum / n) /*keep*/
            }),
    }
}

/// Read a `GRAYF32_LINEAR` map as flat `&[f32]` — `None` when the slice
/// isn't tightly packed (strided maps need `try_typed`/`as_imgref` at
/// the consumer).
pub fn map_f32s<'a>(map: &'a PixelCow<'a, ()>) -> Option<&'a [f32]> {
    let bytes = map.as_slice().as_contiguous_bytes()?;
    if bytes.len() % core::mem::size_of::<f32>() != 0 {
        return None;
    }
    Some(bytemuck::cast_slice(bytes))
}

/// View a packed `w × h` `f32` plane as a `GRAYF32_LINEAR` pixel slice —
/// the shape every leaf metric hands to its `Evaluation::map()`.
///
/// # Panics
/// If `data.len() != w·h`.
#[must_use]
pub fn gray_f32_slice(data: &[f32], w: u32, h: u32) -> zenpixels::PixelSlice<'_> {
    assert_eq!(data.len(), w as usize * h as usize, "map slice len != w*h");
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

/// Repack a strided rgb8 view to packed `RGBRGB…` — a no-op `Cow`
/// borrow when the stride is already `width·3`, a copy otherwise.
/// For metrics whose native entry takes packed input only.
#[must_use]
pub fn contiguous_rgb8<'a>(r: &'a Rgb8In<'a>) -> Cow<'a, [u8]> {
    let n = r.width * r.height * 3;
    if r.stride == r.width * 3 {
        Cow::Borrowed(&r.pixels[..n])
    } else {
        let mut out = Vec::with_capacity(n);
        for row in r.pixels.chunks_exact(r.stride).take(r.height) {
            out.extend_from_slice(&row[..r.width * 3]);
        }
        Cow::Owned(out)
    }
}
