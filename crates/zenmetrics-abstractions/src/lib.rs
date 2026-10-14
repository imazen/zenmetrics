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
//!   map and the caller can ask for others: `map_score(MapFold::PNorm { p: 12.0 })`,
//!   `Mean`/`Std`/`Max`/`PNorm{p}`/`FractionAbove{t}` are default-implemented for any map;
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
/// f32↔u8 casts without a per-leaf bytemuck dep.
pub use bytemuck;
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
// Map folds — the typed replacement for a `&str` spec.
// ---------------------------------------------------------------------------

/// A fold over a per-pixel map — typed, `#[non_exhaustive]` (new fold
/// kinds are additive), with a `Custom` escape hatch for metric-defined
/// grammars.
///
/// This replaces any `"pnorm:12"` string grammar: [`Evaluation::map_score`]
/// takes `MapFold`, and the shared [`map_fold`] helper implements the
/// common cases over any map so every metric gets butteraugli-style
/// "arbitrary norms over the diffmap" for free.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapFold<'a> {
    /// Arithmetic mean — `Σv/n`.
    Mean,
    /// Population standard deviation — GMSD's headline fold
    /// (`gmsd == MapFold::Std` over the `gms` map).
    Std,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
    /// p-norm `(Σ|v|^p / n)^(1/p)` — the butteraugli norm family
    /// (`PNorm { p: 12.0 }` is butteraugli's canonical combine).
    PNorm {
        /// The norm exponent.
        p: f64,
    },
    /// Fraction of pixels `≥ threshold` — HDR-VDP's `visible_fraction`
    /// is `FractionAbove { threshold: 0.5 }` over `P_map`.
    FractionAbove {
        /// The inclusion threshold.
        threshold: f64,
    },
    /// Metric-defined fold — the escape hatch for grammars outside the
    /// shared set (e.g. HDR-VDP's `"c_max"` alias).
    Custom(&'a str),
}

/// What kind of per-pixel map [`EvalOpts`] asks for — `#[non_exhaustive]`
/// so new request shapes (region-of-interest, per-channel) add
/// non-breaking variants.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MapRequest {
    /// The metric's canonical resolution — the map its upstream
    /// reference publishes (GMSD's `w/2 × h/2`, MS-SSIM's `(w−10) ×
    /// (h−10)` interior, FSIM's decimated grid are all `Native`).
    #[default]
    Native,
    /// ~`factor`× coarser than `Native` — the zensim-binned shape:
    /// box-mean cell averages for thumbnail maps and partition-aligned
    /// consumers. Implementors may compute at low res directly; the
    /// default is [`map_subsample_f32`].
    Subsampled {
        /// Linear downsampling factor (`4` → ~`w/4 × h/4`).
        factor: u32,
    },
}

/// What [`Metric::evaluate`] should compute — `#[non_exhaustive]` so new
/// optional outputs (uncertainty maps, per-channel results) add fields
/// without breaking callers.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvalOpts {
    /// Map request: `None` = score only, `Some(_)` = produce a map.
    /// Default `None`.
    pub map: Option<MapRequest>,
}

impl EvalOpts {
    /// Score only.
    pub const SCORE_ONLY: Self = Self { map: None };
    /// Score + canonical-resolution map.
    pub const WITH_MAP: Self = Self {
        map: Some(MapRequest::Native),
    };
    /// Score + a `factor`-coarsened map.
    pub const fn map_subsampled(factor: u32) -> Self {
        Self {
            map: Some(MapRequest::Subsampled { factor }),
        }
    }
}

/// The shared typed viewing-condition carrier for
/// [`Metric::viewing_conditions`] — covers the common core the
/// viewing-conditioned metrics (cvvdp's display geometry, hdrvdp's
/// `pixels_per_degree`/`surround`, butteraugli's `intensity_target`)
/// actually share. Richer settings (surround models, observer age, EOTF
/// curves) stay metric-specific and ride [`Metric::config`].
///
/// `#[non_exhaustive]` — fields may be added (ambient lux, observer age);
/// construct with `..` spread or the builder.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewingConditions {
    /// Pixels per visual degree — cvvdp's geometry-derived ppd, hdrvdp's
    /// `pixels_per_degree`.
    pub ppd: f32,
    /// Display peak luminance in cd/m² — `None` when the metric assumes
    /// the sRGB nominal (~80) or doesn't model display light.
    pub display_peak_nits: Option<f32>,
}

impl ViewingConditions {
    /// Constructor — `ppd` + optional display peak (nits).
    pub fn new(ppd: f32, display_peak_nits: Option<f32>) -> Self {
        Self {
            ppd,
            display_peak_nits,
        }
    }

    /// Builder — set the display peak.
    #[must_use]
    pub fn with_display_peak_nits(mut self, nits: f32) -> Self {
        self.display_peak_nits = Some(nits);
        self
    }
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
    /// The shared default handles [`MapFold`]`::Mean`/`Max`/`Min`/`Std`,
    /// `PNorm { p }` (the butteraugli norm family — `p: 12` is the
    /// canonical combine), and `FractionAbove { threshold }` for any
    /// map. Metrics may override to add their own `Custom` grammar or
    /// richer folds (per-channel, region-weighted).
    fn map_score(&self, fold: MapFold<'_>) -> Option<f64> {
        map_fold(self.map()?, fold)
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

    /// The scorer's variant label within the family — `"vifvec"`,
    /// `"v3"`, `"hdr"`… Default: [`Self::name`].
    fn variant(&self) -> &'static str {
        self.name()
    }

    /// Construction-time knobs echoed for observability — display model,
    /// geometry, model variant, formula revision, preprocess flags.
    /// Metric-defined keys; a logging surface, not a typed contract.
    /// Default: empty.
    fn config(&self) -> Box<dyn Iterator<Item = (Cow<'static, str>, Cow<'static, str>)> + '_> {
        Box::new(core::iter::empty())
    }

    /// Typed viewing conditions when the metric is viewing-conditioned
    /// (cvvdp, hdrvdp, cambi, butteraugli) — `None` for unconditioned
    /// metrics. This is *inspection*, not mutation: viewing conditions
    /// are a construction-time decision (scorer geometry is baked in).
    fn viewing_conditions(&self) -> Option<ViewingConditions> {
        None
    }

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

/// Typed folds over a `GRAYF32_LINEAR` map — the shared default behind
/// [`Evaluation::map_score`].
#[must_use]
pub fn map_fold(map: PixelCow<'_, ()>, fold: MapFold<'_>) -> Option<f64> {
    // Tight maps borrow directly; strided maps repack once so the fold
    // never silently drops a valid map.
    let owned: Vec<f32>;
    let vals: &[f32] = match map_f32s(&map) {
        Some(v) => v,
        None => {
            let sl = map.as_slice();
            let (w, h) = (sl.width() as usize, sl.rows() as usize);
            owned = (0..h)
                .flat_map(|y| {
                    let row: &[f32] = bytemuck::cast_slice(
                        &sl.as_strided_bytes()[y * sl.stride()..y * sl.stride() + w * 4],
                    );
                    row.to_vec()
                })
                .collect();
            &owned
        }
    };
    if vals.is_empty() {
        return None;
    }
    let n = vals.len() as f64;
    match fold {
        MapFold::Mean => Some(vals.iter().map(|&v| v as f64).sum::<f64>() / n),
        MapFold::Max => vals.iter().copied().reduce(f32::max).map(f64::from),
        MapFold::Min => vals.iter().copied().reduce(f32::min).map(f64::from),
        MapFold::Std => {
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
        MapFold::PNorm { p } => {
            let sum: f64 = vals.iter().map(|&v| libm::pow((v as f64).abs(), p)).sum();
            Some(libm::pow(sum / n, p.recip()))
        }
        MapFold::FractionAbove { threshold } => {
            let t = threshold as f32;
            Some(vals.iter().filter(|&&v| v >= t).count() as f64 / n)
        }
        MapFold::Custom(_) => None,
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

/// Box-mean downsample of a packed `w × h` f32 map by an integer factor —
/// the default [`MapRequest::Subsampled`] implementation (the zensim
/// "binned density" shape: each output cell is its `f × f` block's mean,
/// edge cells clipped to the remainder). Returns `(pixels, w_out, h_out)`.
///
/// `factor` clamps to `max(1, …)`; `factor == 1` returns the input's
/// dims with a copied buffer (callers usually early-out).
#[must_use]
pub fn map_subsample_f32(map: &[f32], w: u32, h: u32, factor: u32) -> (Vec<f32>, u32, u32) {
    let f = factor.max(1) as usize;
    let (w, h) = (w as usize, h as usize);
    debug_assert_eq!(map.len(), w * h);
    let (wo, ho) = (w.div_ceil(f), h.div_ceil(f));
    let mut out = alloc::vec![0.0f32; wo * ho];
    for oy in 0..ho {
        let y0 = oy * f;
        let y1 = (y0 + f).min(h);
        for ox in 0..wo {
            let x0 = ox * f;
            let x1 = (x0 + f).min(w);
            let mut sum = 0.0f64;
            let mut cnt = 0u32;
            for y in y0..y1 {
                for &v in &map[y * w + x0..y * w + x1] {
                    sum += v as f64;
                    cnt += 1;
                }
            }
            out[oy * wo + ox] = (sum / cnt as f64) as f32;
        }
    }
    (out, wo as u32, ho as u32)
}
