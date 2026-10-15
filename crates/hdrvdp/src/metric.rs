//! The end-to-end metric: an image pair in, `res.Q` and a visibility map out.
//!
//! Ported from `hdrvdp.m` (MATLAB `hdrvdp-2.2.x`).
//!
//! Copyright (c) 2011, Rafal Mantiuk <mantiuk@gmail.com>
//!
//! Permission to use, copy, modify, and/or distribute this software for any
//! purpose with or without fee is hereby granted, provided that the above
//! copyright notice and this permission notice appear in all copies.
//!
//! THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
//! WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
//! MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
//! ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
//! WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
//! ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
//! OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//!
//! ## Order of operations that matters
//!
//! The **reference is processed first**, and two things it computes are then
//! reused for the test image rather than recomputed:
//!
//! * the surround luminance, when it is derived from the image
//!   (`Params::surround_l = None` → the reference's geometric mean), and
//! * the base band's FFT pad value.
//!
//! Letting the test image pick its own values for either makes the two
//! decompositions differ along the border for reasons that have nothing to do
//! with the distortion — upstream changed the surround handling in 2.1.3 for
//! exactly this reason ("avoids false detection at the image border").

use crate::bands;
use crate::display::{ColorEncoding, looks_relative, to_nits};
use crate::masking::{self, diff_mask};
use crate::params::{FpMode, Params};
use crate::pathway::{mtf_filter_for, surround_per_channel, visual_pathway_with_filter};
use crate::photoreceptor::Photoreceptor;
use crate::pool::{Visibility, quality_correlate, quality_mos, visibility};
use crate::spectral::{emission_spectra, lmsr_matrix};
use crate::{Error, Result};

/// Everything HDR-VDP-2 predicts for one image pair.
#[derive(Debug, Clone)]
pub struct HdrVdpResult {
    /// Per-pixel probability of detecting the difference, `[0, 1]`, row-major.
    pub p_map: Vec<f32>,
    /// The largest value in [`Self::p_map`] — "is this difference visible
    /// anywhere?"
    pub p_det: f64,
    /// Per-pixel difference magnitude in normalised detection units
    /// (1 = at threshold), row-major.
    pub c_map: Vec<f32>,
    /// The largest value in [`Self::c_map`].
    pub c_max: f64,
    /// Official `res.Q` — the quality correlate HDR-VDP-2.2.2 publishes.
    /// **100 = identical**, lower = worse; may go negative for heavy
    /// distortion. This is `100 − Q` where `Q` is upstream's accumulated
    /// `Σ (log(msre+ε) − log ε) · w_f` over all pyramid planes.
    pub q: f64,
    /// The `Q_MOS` logistic HDR-VDP **removed in 2.2.1** as unreliable
    /// (see upstream `ChangeLog.txt`), computed exactly as upstream's
    /// commented-out line does: `100/(1+exp(q₁·(Q_raw+q₂)))` on the *raw*
    /// accumulator `Q_raw = 100 − q`. Nearly saturated — ≈4.4 on identical
    /// pairs, ≈0 on anything visibly distorted — which is precisely why
    /// upstream removed it. Kept for research parity only; prefer [`Self::q`].
    pub q_mos: f64,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// True when the input looked like relative rather than absolute values
    /// for a colour encoding that expects cd/m² — upstream's
    /// `hdrvdp:lowvals` warning, returned instead of printed. When this is
    /// set, every number above was computed at the wrong adaptation level.
    pub input_looks_relative: bool,
}

use zenmetrics_abstractions as za;

impl HdrVdpResult {
    /// One cell per pixel; support is global (the multi-scale
    /// decomposition is FFT-based and the surround defaults to the
    /// reference's geometric mean).
    pub fn geometry(&self) -> za::MapGeometry {
        za::MapGeometry::identity(self.width as u32, self.height as u32)
            .with_support(za::Support::Global)
    }

    fn view<'s>(
        &'s self,
        name: &'static str,
        map: &'s [f32],
        polarity: za::MapPolarity,
    ) -> Option<za::MapView<'s>> {
        let px = za::gray_f32_slice(map, self.width as u32, self.height as u32).ok()?;
        za::MapView::new(name, px, polarity, self.geometry()).ok()
    }
}

/// The direct-call result is an [`za::Evaluation`] with both maps always
/// present: `p_det` (detection probability, primary) and `c` (normalized
/// detection units, 1 = threshold).
impl za::Evaluation for HdrVdpResult {
    fn score(&self) -> f64 {
        self.q
    }

    fn scores(&self) -> za::Scores<'_> {
        std::boxed::Box::new(
            [
                (std::borrow::Cow::Borrowed("q"), self.q),
                (std::borrow::Cow::Borrowed("q_mos"), self.q_mos),
                (std::borrow::Cow::Borrowed("p_det"), self.p_det),
                (std::borrow::Cow::Borrowed("c_max"), self.c_max),
                (
                    std::borrow::Cow::Borrowed("visible_fraction"),
                    self.visible_fraction(),
                ),
            ]
            .into_iter(),
        )
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        self.view("p_det", &self.p_map, za::MapPolarity::DetectionProbability)
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        [
            self.view("p_det", &self.p_map, za::MapPolarity::DetectionProbability),
            self.view("c", &self.c_map, za::MapPolarity::Difference),
        ]
        .into_iter()
        .flatten()
        .collect()
    }

    /// `Custom("c_max")` / `Custom("visible_fraction")` alias the scalar
    /// fields; everything else folds `p_map`
    /// (`FractionAbove { threshold: 0.5 }` is the visible fraction).
    fn map_score(&self, fold: za::MapFold<'_>) -> za::MetricResult<f64> {
        match fold {
            za::MapFold::Custom("c_max") => Ok(self.c_max),
            za::MapFold::Custom("visible_fraction") => Ok(self.visible_fraction()),
            other => za::fold_values(&self.p_map, other),
        }
    }

    fn extras(&self) -> za::Extras<'_> {
        std::boxed::Box::new(core::iter::once((
            std::borrow::Cow::Borrowed("input_looks_relative"),
            std::borrow::Cow::Owned(std::format!("{}", self.input_looks_relative)),
        )))
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

impl HdrVdpResult {
    /// Fraction of pixels whose difference is above the detection threshold
    /// (`P ≥ 0.5`, equivalently `C ≥ 1`).
    #[must_use]
    pub fn visible_fraction(&self) -> f64 {
        if self.p_map.is_empty() {
            return 0.0;
        }
        let n = self.p_map.iter().filter(|p| **p >= 0.5).count();
        n as f64 / self.p_map.len() as f64
    }
}

/// Score a pair of images.
///
/// * `test`, `reference` — interleaved pixels, `width · height ·
///   encoding.channels()` values each, in whatever units `encoding` implies.
///   For the absolute encodings ([`ColorEncoding::Luminance`],
///   [`ColorEncoding::RgbBt709`], [`ColorEncoding::Xyz`]) that means **cd/m²**.
/// * `par` — model parameters; `par.pix_per_deg` must be finite (see
///   [`crate::pix_per_deg`]).
///
/// # Errors
/// [`Error::SizeMismatch`] if the two images differ in size,
/// [`Error::ChannelMismatch`] if a buffer length does not match the encoding,
/// [`Error::ImpossibleValues`] for non-finite luminance after the display
/// model, and [`Error::InvalidResolution`] if `pix_per_deg` is not a positive
/// finite number.
pub fn hdrvdp(
    test: &[f64],
    reference: &[f64],
    width: usize,
    height: usize,
    encoding: ColorEncoding,
    par: &Params,
) -> Result<HdrVdpResult> {
    let p = prepare(test, reference, width, height, encoding, par)?;
    let m = masking::run(&p.bands_test, &p.bands_ref, &p.l_adapt, &p.diff_mask, par);
    let Visibility {
        p_map,
        p_det,
        c_map,
        c_max,
        ..
    } = visibility(&m.d_bands, par);

    // Upstream: `res.Q = 100 − Q`; the removed `res.Q_MOS` was the logistic
    // applied to the raw `Q` accumulator, not to `res.Q`.
    let q_raw = quality_correlate(&m.quality_terms);
    Ok(HdrVdpResult {
        p_map,
        p_det,
        c_map,
        c_max,
        q: 100.0 - q_raw,
        q_mos: quality_mos(q_raw, par),
        width,
        height,
        input_looks_relative: p.input_looks_relative,
    })
}

/// Score a pair of images, returning only the official `res.Q` correlate.
///
/// Identical to [`hdrvdp`] except that the visibility reconstruction
/// (`P_map`/`P_det`/`C_max`) is never computed — `res.Q` depends only on the
/// per-plane quality terms, so sweep-style callers that only need the scalar
/// score skip the inverse pyramid and spatial pooling entirely.
///
/// # Errors
/// Same conditions as [`hdrvdp`].
pub fn score(
    test: &[f64],
    reference: &[f64],
    width: usize,
    height: usize,
    encoding: ColorEncoding,
    par: &Params,
) -> Result<f64> {
    let p = prepare(test, reference, width, height, encoding, par)?;
    let terms = masking::run_terms(&p.bands_test, &p.bands_ref, &p.l_adapt, &p.diff_mask, par);
    Ok(100.0 - quality_correlate(&terms))
}

// Everything `hdrvdp()` and [`score`] share: display model → visual pathway
// → band decomposition → adapting maps. Returns the two band pyramids plus
// the `l_adapt`/`diff_mask` planes masking consumes.
struct Prepared {
    bands_test: crate::bands::BandPyramid,
    bands_ref: crate::bands::BandPyramid,
    l_adapt: Vec<f32>,
    diff_mask: Vec<f32>,
    input_looks_relative: bool,
}

fn prepare(
    test: &[f64],
    reference: &[f64],
    width: usize,
    height: usize,
    encoding: ColorEncoding,
    par: &Params,
) -> Result<Prepared> {
    if !(par.pix_per_deg.is_finite() && par.pix_per_deg > 0.0) {
        return Err(Error::InvalidResolution(par.pix_per_deg));
    }
    if test.len() != reference.len() {
        return Err(Error::SizeMismatch {
            reference: (width, height),
            distorted: (width, height),
        });
    }
    let channels = encoding.channels();

    let ref_nits = to_nits(reference, width, height, encoding)?;
    let test_nits = to_nits(test, width, height, encoding)?;

    // Surround: one value per channel, derived from the REFERENCE and shared.
    let surround = surround_per_channel(&ref_nits, channels, par.surround_l);

    let lmsr = lmsr_matrix(&emission_spectra(encoding.spectra(), channels));
    let pn = Photoreceptor::new(par);

    // The optical MTF filter depends only on geometry + parameters: build it
    // once and share it across the pair (bit-identical to per-image builds —
    // the filter is a deterministic function of (w, h, par)).
    let mtf_filter = mtf_filter_for(width, height, par);
    // The two pathways are independent — and so are the pyramid builds —
    // so under `parallel` they run as pairs. The base-band CSF filter stays
    // sequential because the test image must reuse the reference's pad value.
    let (path_ref, path_test) = crate::par::join2(
        || {
            visual_pathway_with_filter(
                &ref_nits,
                width,
                height,
                &pn,
                &lmsr,
                &surround,
                mtf_filter.as_deref(),
            )
        },
        || {
            visual_pathway_with_filter(
                &test_nits,
                width,
                height,
                &pn,
                &lmsr,
                &surround,
                mtf_filter.as_deref(),
            )
        },
    );
    drop(mtf_filter);

    let relaxed = par.mode == FpMode::Relaxed;
    let (mut bands_ref, mut bands_test) = crate::par::join2(
        || bands::decompose_build(&path_ref, relaxed),
        || bands::decompose_build(&path_test, relaxed),
    );
    // Reference first, so its base-band pad value can be reused.
    let pad = bands::finalize_baseband(&mut bands_ref, &path_ref, par, None);
    bands::finalize_baseband(&mut bands_test, &path_test, par, Some(pad));

    let l_adapt: Vec<f32> = path_ref
        .l_adapt
        .iter()
        .zip(&path_test.l_adapt)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dm = diff_mask(&test_nits, &ref_nits, channels);

    Ok(Prepared {
        bands_test,
        bands_ref,
        l_adapt,
        diff_mask: dm,
        input_looks_relative: looks_relative(&ref_nits, channels, encoding),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A textured HDR reference in cd/m²: a mid-grey field with structure,
    /// plus a bright highlight well above SDR range.
    fn hdr_reference(w: usize, h: usize) -> Vec<f64> {
        let pi = core::f64::consts::PI;
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let base =
                    80.0 * (1.0 + 0.4 * (2.0 * pi * x / 17.0).sin() * (2.0 * pi * y / 23.0).cos());
                // A 3000 cd/m² specular blob — the HDR part.
                let (cx, cy) = (w as f64 * 0.7, h as f64 * 0.3);
                let r2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (0.02 * (w * h) as f64);
                base + 3000.0 * (-r2).exp()
            })
            .collect()
    }

    fn par() -> Params {
        Params::new(30.0)
    }

    /// `FpMode::Relaxed` is the shipping default — it must land within
    /// noise of `Strict`, not silently diverge. Bound measured on this
    /// pair: |Δq| ≈ 4e-6, |Δp_det| ≈ 1e-7 — keep generous headroom for
    /// other inputs while still catching a wiring mistake (e.g. relaxed
    /// accidentally running a different filter count).
    #[test]
    fn relaxed_mode_stays_within_tolerance_of_strict() {
        let (w, h) = (96usize, 72usize);
        let reference = hdr_reference(w, h);
        let pi = core::f64::consts::PI;
        let test: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                v * (1.0 + 0.05 * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin())
            })
            .collect();
        let strict = hdrvdp(
            &test,
            &reference,
            w,
            h,
            ColorEncoding::Luminance,
            &par().with_strict(),
        )
        .unwrap();
        let relaxed = hdrvdp(&test, &reference, w, h, ColorEncoding::Luminance, &par()).unwrap();
        assert!(
            (strict.q - relaxed.q).abs() < 0.05,
            "Q: strict {} vs relaxed {}",
            strict.q,
            relaxed.q
        );
        assert!(
            (strict.p_det - relaxed.p_det).abs() < 0.01,
            "P_det: strict {} vs relaxed {}",
            strict.p_det,
            relaxed.p_det
        );
        assert_eq!(
            strict.p_map.len(),
            relaxed.p_map.len(),
            "mode must not change the visibility-map shape"
        );
    }

    #[test]
    fn an_identical_pair_is_invisible_and_scores_full_marks() {
        let (w, h) = (96usize, 72usize);
        let im = hdr_reference(w, h);
        let r = hdrvdp(&im, &im, w, h, ColorEncoding::Luminance, &par()).unwrap();
        assert_eq!((r.width, r.height), (w, h));
        assert!(r.p_det < 1e-9, "P_det = {} on an identical pair", r.p_det);
        assert!(r.c_max < 1e-9, "C_max = {}", r.c_max);
        assert_eq!(r.visible_fraction(), 0.0);
        // Every band's msre is 0, so every quality term is
        // (log(0+ε) − log ε)·w_f = 0 → res.Q = 100 exactly.
        assert_eq!(r.q, 100.0, "res.Q = {} on an identical pair", r.q);
        // `q_mos` is the logistic upstream removed in 2.2.1: on Q_raw = 0 it
        // lands at 100/(1+exp(3.455·0.8886)) ≈ 4.4 — the "relatively low for
        // identical images" behaviour the 2.1.2 ChangeLog describes and part
        // of why upstream dropped it.
        assert!(
            (4.0..5.0).contains(&r.q_mos),
            "removed-Q_MOS = {} on an identical pair",
            r.q_mos
        );
        assert!(!r.input_looks_relative);
    }

    #[test]
    fn quality_falls_monotonically_along_a_distortion_ladder() {
        // The property any usable quality metric must have, and the one the
        // UPIQ validation will measure the strength of.
        let (w, h) = (96usize, 72usize);
        let reference = hdr_reference(w, h);
        let pi = core::f64::consts::PI;
        let mut last_q = f64::INFINITY;
        let mut last_pdet = f64::NEG_INFINITY;
        for step in 0..5 {
            let amp = 0.002 * 3f64.powi(step);
            let test: Vec<f64> = reference
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    v * (1.0 + amp * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin())
                })
                .collect();
            let r = hdrvdp(&test, &reference, w, h, ColorEncoding::Luminance, &par()).unwrap();
            assert!(
                r.q < last_q,
                "res.Q should fall as distortion grows: {} then {} at amp {amp}",
                last_q,
                r.q
            );
            assert!(
                r.p_det > last_pdet,
                "P_det should rise as distortion grows: {} then {} at amp {amp}",
                last_pdet,
                r.p_det
            );
            assert!(r.q.is_finite() && r.q <= 100.0);
            assert!((0.0..=100.0).contains(&r.q_mos));
            assert!((0.0..=1.0).contains(&r.p_det));
            last_q = r.q;
            last_pdet = r.p_det;
        }
        // The strongest rung must actually be visible somewhere.
        assert!(
            last_pdet > 0.5,
            "the top of the ladder is not visible: {last_pdet}"
        );
    }

    #[test]
    fn the_visibility_map_localises_the_distortion() {
        // A difference confined to the left half must not light up the right
        // half. This is the property that makes P_map a *map* rather than a
        // scalar smeared over the image.
        let (w, h) = (96usize, 72usize);
        let reference = hdr_reference(w, h);
        let test: Vec<f64> = reference
            .iter()
            .enumerate()
            .map(|(i, v)| if i % w < w / 3 { v * 1.15 } else { *v })
            .collect();
        let r = hdrvdp(&test, &reference, w, h, ColorEncoding::Luminance, &par()).unwrap();
        let mean_over = |xs: std::ops::Range<usize>| -> f64 {
            let mut acc = 0.0;
            let mut n = 0usize;
            for y in 0..h {
                for x in xs.clone() {
                    acc += f64::from(r.p_map[y * w + x]);
                    n += 1;
                }
            }
            acc / n as f64
        };
        let left = mean_over(0..w / 4);
        let right = mean_over(3 * w / 4..w);
        assert!(
            left > 5.0 * right.max(1e-6),
            "distorted region ({left}) should dominate the untouched one ({right})"
        );
    }

    #[test]
    fn adaptation_level_changes_the_verdict() {
        // Same relative distortion, two absolute light levels: HDR-VDP-2 is
        // luminance-aware, so a difference presented in near-darkness must be
        // less visible than the same difference at photopic levels. A metric
        // that ignores absolute luminance cannot tell these apart, which is
        // the entire reason this port exists.
        let (w, h) = (96usize, 72usize);
        let pi = core::f64::consts::PI;
        let pdet_at = |scale: f64| -> f64 {
            let reference: Vec<f64> = hdr_reference(w, h).iter().map(|v| v * scale).collect();
            let test: Vec<f64> = reference
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let (x, y) = ((i % w) as f64, (i / w) as f64);
                    v * (1.0 + 0.02 * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin())
                })
                .collect();
            hdrvdp(&test, &reference, w, h, ColorEncoding::Luminance, &par())
                .unwrap()
                .p_det
        };
        let dark = pdet_at(0.0005); // ~0.04 cd/m², scotopic
        let bright = pdet_at(1.0); // ~80 cd/m², photopic
        assert!(
            bright > dark,
            "the same relative distortion should be more visible at photopic levels: \
             dark {dark} vs bright {bright}"
        );
    }

    #[test]
    fn relative_input_is_flagged_rather_than_silently_mis_scored() {
        let (w, h) = (48usize, 48usize);
        let im: Vec<f64> = (0..w * h)
            .map(|i| 0.2 + 0.5 * ((i % 7) as f64 / 7.0))
            .collect();
        let r = hdrvdp(&im, &im, w, h, ColorEncoding::Luminance, &par()).unwrap();
        assert!(
            r.input_looks_relative,
            "a [0,1] image passed as absolute luminance must be flagged"
        );
    }

    #[test]
    fn errors_are_returned_not_panicked() {
        let (w, h) = (16usize, 16usize);
        let im = vec![50.0; w * h];
        // Wrong buffer length for the encoding.
        let e = hdrvdp(&im, &im, w, h, ColorEncoding::SrgbDisplay, &par()).unwrap_err();
        assert!(matches!(e, Error::ChannelMismatch { .. }), "{e:?}");
        // Mismatched sizes.
        let e = hdrvdp(&im[..10], &im, w, h, ColorEncoding::Luminance, &par()).unwrap_err();
        assert!(matches!(e, Error::SizeMismatch { .. }), "{e:?}");
        // Unset resolution — `Params::default()` leaves it NaN on purpose.
        let e = hdrvdp(&im, &im, w, h, ColorEncoding::Luminance, &Params::default()).unwrap_err();
        assert!(matches!(e, Error::InvalidResolution(_)), "{e:?}");
    }

    #[test]
    fn srgb_display_encoding_scores_an_sdr_pair() {
        // The metric is not HDR-only: an sRGB pair goes through the 99/1 cd/m²
        // display model and scores on the same scale.
        let (w, h) = (64usize, 64usize);
        let pi = core::f64::consts::PI;
        let reference: Vec<f64> = (0..w * h * 3)
            .map(|i| {
                let px = i / 3;
                let (x, y) = ((px % w) as f64, (px / w) as f64);
                0.5 + 0.25 * (2.0 * pi * x / 11.0).sin() * (2.0 * pi * y / 13.0).cos()
            })
            .collect();
        let test: Vec<f64> = reference.iter().map(|v| (v + 0.02).min(1.0)).collect();
        let r = hdrvdp(&test, &reference, w, h, ColorEncoding::SrgbDisplay, &par()).unwrap();
        assert!(r.q_mos.is_finite() && (0.0..=100.0).contains(&r.q_mos));
        assert!(r.p_det > 0.0);
        assert!(
            !r.input_looks_relative,
            "code values must not trip the flag"
        );
    }
}

// ---------------------------------------------------------------------------
// Generic Metric surface.
// ---------------------------------------------------------------------------

impl Error {
    fn kind(&self) -> za::ErrorKind {
        match self {
            Error::ChannelMismatch { .. } | Error::ImpossibleValues(_) => {
                za::ErrorKind::InvalidInput
            }
            Error::DimensionOverflow => za::ErrorKind::Resource,
            Error::SizeMismatch { .. } => za::ErrorKind::DimensionMismatch,
            Error::InvalidResolution(_) | Error::MissingEmission | Error::SpectralData(_) => {
                za::ErrorKind::InvalidConfig
            }
            Error::ImageTooSmall { .. } => za::ErrorKind::TooSmall,
        }
    }
}

/// Output revision of [`HdrVdp`] on the generic surface.
const SCORE_REVISION: u32 = 1;

/// The HDR-VDP-2 metric as a generic scorer. Viewing conditions, encoding
/// and emission are construction-time, so [`HdrVdp::new`] takes them.
///
/// Input is `f32` (widened exactly to `f64`) whose descriptor must say
/// what `encoding` means — no relabeling:
///
/// | encoding | layout | transfer | primaries |
/// |---|---|---|---|
/// | `Luminance` (cd/m²) | gray | linear | any |
/// | `LumaDisplay` | gray | gamma 2.2 | any |
/// | `SrgbDisplay` | RGB | sRGB | BT.709 |
/// | `RgbBt709` (cd/m²) | RGB | linear | BT.709 |
/// | `Xyz` | — | — | — (no zenpixels descriptor; call [`hdrvdp`]) |
#[derive(Debug, Clone)]
pub struct HdrVdp {
    /// Display/observer/viewing params.
    pub par: Params,
    /// Color encoding the scorer expects.
    pub encoding: ColorEncoding,
}

impl HdrVdp {
    /// Constructor — all configuration is explicit, matching `hdrvdp()`.
    #[must_use]
    pub fn new(par: Params, encoding: ColorEncoding) -> Self {
        Self { par, encoding }
    }

    fn expects(
        &self,
    ) -> Option<(
        zenpixels::ChannelLayout,
        zenpixels::TransferFunction,
        Option<zenpixels::ColorPrimaries>,
    )> {
        use zenpixels::{ChannelLayout as L, ColorPrimaries as P, TransferFunction as T};
        Some(match self.encoding {
            ColorEncoding::Luminance => (L::Gray, T::Linear, None),
            ColorEncoding::LumaDisplay => (L::Gray, T::Gamma22, None),
            ColorEncoding::SrgbDisplay => (L::Rgb, T::Srgb, Some(P::Bt709)),
            ColorEncoding::RgbBt709 => (L::Rgb, T::Linear, Some(P::Bt709)),
            ColorEncoding::Xyz => return None,
        })
    }
}

/// Widen a strided f32 slice to packed f64 without reinterpreting casts.
fn widen_f32(s: &zenpixels::PixelSlice<'_>) -> Vec<f64> {
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let ch = s.descriptor().layout().channels();
    let bytes = s.as_strided_bytes();
    let mut out = Vec::with_capacity(w * h * ch);
    for y in 0..h {
        let row = &bytes[y * s.stride()..y * s.stride() + w * ch * 4];
        out.extend(
            row.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_ne_bytes(*c))),
        );
    }
    out
}

impl za::Metric for HdrVdp {
    fn id(&self) -> za::MetricId {
        za::MetricId::new("hdrvdp", za::implementation!(SCORE_REVISION))
            .with_variant("v2")
            .with_reference("hdrvdp-2.2.2 (Mantiuk et al.)")
            .with_config("encoding", std::format!("{:?}", self.encoding))
            .with_config("pix_per_deg", self.par.pix_per_deg)
            .with_config("mode", std::format!("{:?}", self.par.mode))
            .with_config("sensitivity_correction", self.par.sensitivity_correction)
            .with_config("surround_l", std::format!("{:?}", self.par.surround_l))
            .with_config("params", std::format!("{:?}", self.par))
    }

    fn outputs(&self) -> std::borrow::Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[
            za::ScoreSpec::new("q", za::ScoreKind::Quality100),
            // Upstream's removed logistic: ~4.4 on identical pairs, so it has
            // no "perfect" value on any quality scale.
            za::ScoreSpec::new("q_mos", za::ScoreKind::Feature),
            za::ScoreSpec::new("p_det", za::ScoreKind::Probability),
            za::ScoreSpec::new("c_max", za::ScoreKind::Difference),
            za::ScoreSpec::new("visible_fraction", za::ScoreKind::Probability),
        ];
        std::borrow::Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        let (Some(d), Some((layout, transfer, primaries))) = (format.packed(), self.expects())
        else {
            return false;
        };
        d.channel_type() == zenpixels::ChannelType::F32
            && d.layout() == layout
            && d.alpha().is_none()
            && d.transfer() == transfer
            && primaries.is_none_or(|p| d.primaries == p)
            && d.signal_range == zenpixels::SignalRange::Full
    }

    fn viewing_conditions(&self) -> Option<za::ViewingConditions> {
        Some(za::ViewingConditions::new(
            self.par.pix_per_deg as f32,
            None,
        ))
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<std::boxed::Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        for img in [&r, &d] {
            if !za::Metric::accepts(self, &img.format()) {
                return za::fail(
                    za::unsupported(
                        &img.format(),
                        "the f32 layout/transfer/primaries the encoding declares",
                    )
                    .with_metric("hdrvdp"),
                );
            }
        }
        za::check_same_size(&r, &d)?;
        let (Some(rs), Some(ds)) = (r.as_packed(), d.as_packed()) else {
            return za::fail(za::MetricError::new(za::ErrorKind::Internal).with_metric("hdrvdp"));
        };
        let (w, h) = (rs.width() as usize, rs.rows() as usize);
        let res = hdrvdp(
            &widen_f32(ds),
            &widen_f32(rs),
            w,
            h,
            self.encoding,
            &self.par,
        )
        .metric_err_with("hdrvdp", Error::kind)?;
        let eval = std::boxed::Box::new(HdrVdpEval {
            res,
            maps: opts.map.is_some(),
        });
        match opts.map {
            None => Ok(eval),
            Some(req) => za::apply_map_request(eval, req),
        }
    }
}

/// The generic surface's evaluation: [`HdrVdpResult`] with its maps
/// exposed only when requested. `as_any` downcasts to the
/// [`HdrVdpResult`].
struct HdrVdpEval {
    res: HdrVdpResult,
    maps: bool,
}

impl za::Evaluation for HdrVdpEval {
    fn score(&self) -> f64 {
        za::Evaluation::score(&self.res)
    }

    fn scores(&self) -> za::Scores<'_> {
        za::Evaluation::scores(&self.res)
    }

    fn map(&self) -> Option<za::MapView<'_>> {
        if self.maps {
            za::Evaluation::map(&self.res)
        } else {
            None
        }
    }

    fn maps(&self) -> Vec<za::MapView<'_>> {
        if self.maps {
            za::Evaluation::maps(&self.res)
        } else {
            Vec::new()
        }
    }

    fn map_score(&self, fold: za::MapFold<'_>) -> za::MetricResult<f64> {
        za::Evaluation::map_score(&self.res, fold)
    }

    fn extras(&self) -> za::Extras<'_> {
        za::Evaluation::extras(&self.res)
    }

    fn as_any(&self) -> &dyn core::any::Any {
        &self.res
    }
}

#[cfg(test)]
mod metric_trait_tests {
    use super::*;
    use za::Metric as _;

    fn gray(px: &[f32], w: usize, h: usize, d: zenpixels::PixelDescriptor) -> za::Image<'_> {
        zenpixels::PixelSlice::new(za::bytemuck::cast_slice(px), w as u32, h as u32, w * 4, d)
            .unwrap()
            .into()
    }

    fn pair(w: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
        let r: Vec<f32> = (0..w * h).map(|i| 60.0 + ((i * 37) % 97) as f32).collect();
        let mut d = r.clone();
        for v in d.iter_mut().step_by(5) {
            *v *= 0.7;
        }
        (r, d)
    }

    /// `HdrVdp` through the generic surface: identity, viewing conditions,
    /// and the contract (maps only on request, warm == cold, stop).
    #[test]
    fn hdrvdp_metric_surface_conforms() {
        let mut m = HdrVdp::new(Params::new(30.0), ColorEncoding::Luminance);
        assert_eq!(m.id().variant(), Some("v2"));
        assert!((m.viewing_conditions().unwrap().ppd - 30.0).abs() < 1e-6);
        let (r, d) = pair(32, 32);
        let pd = zenpixels::PixelDescriptor::GRAYF32_LINEAR;
        za::conformance::assert_conforms(&mut m, &gray(&r, 32, 32, pd), &gray(&d, 32, 32, pd));
        let ev = m
            .evaluate(
                gray(&r, 32, 32, pd),
                gray(&d, 32, 32, pd),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        assert_eq!(ev.maps().len(), 2);
        assert!(ev.as_any().is::<HdrVdpResult>());
    }

    /// Wrong transfer or channel count is refused, never relabeled.
    #[test]
    fn hdrvdp_refuses_mislabeled_input() {
        let mut m = HdrVdp::new(Params::new(30.0), ColorEncoding::Luminance);
        let (r, d) = pair(16, 16);
        let srgb_gray = zenpixels::PixelDescriptor::GRAYF32_LINEAR
            .with_transfer(zenpixels::TransferFunction::Srgb);
        let err = m
            .evaluate(
                gray(&r, 16, 16, srgb_gray),
                gray(&d, 16, 16, srgb_gray),
                za::EvalOpts::SCORE_ONLY,
            )
            .err()
            .unwrap();
        assert_eq!(err.error().kind(), za::ErrorKind::UnsupportedInput);
        let rgb = HdrVdp::new(Params::new(30.0), ColorEncoding::RgbBt709);
        assert!(!rgb.accepts(&za::InputFormat::Packed(
            zenpixels::PixelDescriptor::GRAYF32_LINEAR
        )));
        assert!(rgb.accepts(&za::InputFormat::Packed(
            zenpixels::PixelDescriptor::RGBF32_LINEAR
        )));
        let xyz = HdrVdp::new(Params::new(30.0), ColorEncoding::Xyz);
        assert!(!xyz.accepts(&za::InputFormat::Packed(
            zenpixels::PixelDescriptor::RGBF32_LINEAR
        )));
    }
}
