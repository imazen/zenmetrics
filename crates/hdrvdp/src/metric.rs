//! The end-to-end metric: an image pair in, `Q_MOS` and a visibility map out.
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

use crate::bands::decompose;
use crate::display::{ColorEncoding, looks_relative, to_nits};
use crate::masking::{self, diff_mask};
use crate::params::Params;
use crate::pathway::{mtf_filter_for, surround_per_channel, visual_pathway_with_filter};
use crate::photoreceptor::Photoreceptor;
use crate::pool::{Visibility, quality_correlate, quality_mos, visibility};
use crate::spectral::{emission_spectra, lmsr_matrix};
use crate::{Error, Result};

/// Everything HDR-VDP-2 predicts for one image pair.
#[derive(Debug, Clone)]
pub struct HdrVdpResult {
    /// Per-pixel probability of detecting the difference, `[0, 1]`, row-major.
    pub p_map: Vec<f64>,
    /// The largest value in [`Self::p_map`] — "is this difference visible
    /// anywhere?"
    pub p_det: f64,
    /// Per-pixel difference magnitude in normalised detection units
    /// (1 = at threshold), row-major.
    pub c_map: Vec<f64>,
    /// The largest value in [`Self::c_map`].
    pub c_max: f64,
    /// The raw quality correlate. Negative; rises toward 0 as distortion grows.
    pub q: f64,
    /// The mean-opinion-score correlate on 0–100, **100 = best**.
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
    let path_ref = visual_pathway_with_filter(
        &ref_nits,
        width,
        height,
        &pn,
        &lmsr,
        &surround,
        mtf_filter.as_deref(),
    );
    let path_test = visual_pathway_with_filter(
        &test_nits,
        width,
        height,
        &pn,
        &lmsr,
        &surround,
        mtf_filter.as_deref(),
    );
    drop(mtf_filter);

    // Reference first, so its base-band pad value can be reused.
    let (bands_ref, pad) = decompose(&path_ref, par, None);
    let (bands_test, _) = decompose(&path_test, par, Some(pad));

    let l_adapt: Vec<f64> = path_ref
        .l_adapt
        .iter()
        .zip(&path_test.l_adapt)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let dm = diff_mask(&test_nits, &ref_nits, channels);

    let m = masking::run(&bands_test, &bands_ref, &l_adapt, &dm, par);
    let Visibility {
        p_map,
        p_det,
        c_map,
        c_max,
        ..
    } = visibility(&m.d_bands, par);

    let q = quality_correlate(&m.quality_terms);
    Ok(HdrVdpResult {
        p_map,
        p_det,
        c_map,
        c_max,
        q,
        q_mos: quality_mos(q, par),
        width,
        height,
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

    #[test]
    fn an_identical_pair_is_invisible_and_scores_full_marks() {
        let (w, h) = (96usize, 72usize);
        let im = hdr_reference(w, h);
        let r = hdrvdp(&im, &im, w, h, ColorEncoding::Luminance, &par()).unwrap();
        assert_eq!((r.width, r.height), (w, h));
        assert!(r.p_det < 1e-9, "P_det = {} on an identical pair", r.p_det);
        assert!(r.c_max < 1e-9, "C_max = {}", r.c_max);
        assert_eq!(r.visible_fraction(), 0.0);
        // Every band's msre is 0, so every quality term is log(0 + 1e-12)·w/n.
        // That epsilon is finite, so `Q_MOS` for an identical pair lands just
        // *short* of 100 — a known upstream property, not a port defect: the
        // 2.1.2 ChangeLog says the updated epsilons "prevent NaN due to log of
        // 0, but also cause Q_MOS to be relatively low for two identical
        // images". Measured here: 99.9998.
        assert!(r.q < 0.0, "Q = {}", r.q);
        assert!(
            r.q_mos > 99.99 && r.q_mos < 100.0,
            "Q_MOS = {} on an identical pair",
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
        let mut last_mos = f64::INFINITY;
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
                r.q_mos < last_mos,
                "Q_MOS should fall as distortion grows: {} then {} at amp {amp}",
                last_mos,
                r.q_mos
            );
            assert!(
                r.p_det > last_pdet,
                "P_det should rise as distortion grows: {} then {} at amp {amp}",
                last_pdet,
                r.p_det
            );
            assert!((0.0..=100.0).contains(&r.q_mos));
            assert!((0.0..=1.0).contains(&r.p_det));
            last_mos = r.q_mos;
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
                    acc += r.p_map[y * w + x];
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
