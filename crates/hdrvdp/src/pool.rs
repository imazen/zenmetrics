//! Visibility pooling: the per-band difference signal → a probability-of-
//! detection map, and the quality correlate `Q` → `Q_MOS`.
//!
//! Ported from the tail of `hdrvdp.m` (MATLAB `hdrvdp-2.2.x`); the `Q_MOS`
//! logistic and its two constants are the 2.2 recalibration (Narwaria et al.,
//! JEI 24(1), 2015).
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
//! ## Two outputs, two scales
//!
//! * **Visibility** — `C_map` is the reconstructed difference magnitude in
//!   *normalised detection units*, where 1 means "at the detection threshold".
//!   `P_map = 1 − exp(log(0.5)·C)` turns that into a probability, so `C = 1`
//!   gives exactly 0.5 — a coin flip, which is what "threshold" means.
//!   `P_det` is the worst pixel.
//! * **Quality** — `Q` is a weighted sum of per-band log energies, so it is
//!   negative and rises toward 0 as distortion grows. `Q_MOS` maps it onto
//!   0–100 with **100 = best**, via a logistic that is *decreasing* in `Q`.

use crate::bands::BandPyramid;
use crate::params::Params;
use crate::spyr::{Band, ORIENTATIONS, SteerablePyramid, reconstruct};

/// The visibility half of an HDR-VDP-2 result.
#[derive(Debug, Clone)]
pub struct Visibility {
    /// Per-pixel probability that a human notices the difference, `[0, 1]`.
    pub p_map: Vec<f64>,
    /// The largest value in [`Self::p_map`].
    pub p_det: f64,
    /// Per-pixel difference magnitude in normalised detection units
    /// (1 = at threshold).
    pub c_map: Vec<f64>,
    /// The largest value in [`Self::c_map`].
    pub c_max: f64,
    /// Map width.
    pub width: usize,
    /// Map height.
    pub height: usize,
}

/// Rebuild a [`SteerablePyramid`] from a band list, so it can be run back
/// through [`reconstruct`].
///
/// # Panics
/// If the band list does not have the `[1, 4×H, 1]` shape.
#[must_use]
pub fn to_steerable(bp: &BandPyramid) -> SteerablePyramid {
    assert!(
        bp.count() >= 2,
        "band list needs a high-pass and a low-pass"
    );
    let high_pass = bp.bands[0][0].clone();
    let low_pass = bp.bands[bp.count() - 1][0].clone();
    let levels: Vec<[Band; ORIENTATIONS]> = bp.bands[1..bp.count() - 1]
        .iter()
        .map(|planes| {
            let v: Vec<Band> = planes.clone();
            v.try_into()
                .unwrap_or_else(|_| panic!("oriented band must carry {ORIENTATIONS} planes"))
        })
        .collect();
    SteerablePyramid {
        high_pass,
        levels,
        low_pass,
        width: bp.width,
        height: bp.height,
    }
}

/// Collapse the per-band difference signal into the visibility maps.
#[must_use]
pub fn visibility(d_bands: &BandPyramid, par: &Params) -> Visibility {
    let pyr = to_steerable(d_bands);
    let rec = reconstruct(&pyr);
    let mut s_map: Vec<f64> = rec.data.iter().map(|v| v.abs()).collect();

    if par.do_spatial_pooling {
        // Upstream: `S_map = sum(S_map)/(max(S_map)+eps) · S_map`.
        //
        // Worth understanding before reading a `C_max`: this scales the whole
        // map by `sum/max`, so the pooled **maximum becomes the un-pooled
        // sum**. `C_max` after pooling is therefore a *total difference mass*,
        // not a peak — which is why it routinely runs into the hundreds while
        // "1" nominally means "at threshold", and why `P_det` saturates to 1
        // for any distortion of real extent. `P_map`'s spatial *shape* is
        // unchanged (it is a uniform rescale), so the map still localises.
        let sum: f64 = s_map.iter().sum();
        let max = s_map.iter().fold(0.0f64, |a, b| a.max(*b));
        let k = sum / (max + 1e-12);
        for v in &mut s_map {
            *v *= k;
        }
    }

    let ln_half = 0.5f64.ln();
    let p_map: Vec<f64> = s_map.iter().map(|c| 1.0 - (ln_half * c).exp()).collect();
    let p_det = p_map.iter().fold(0.0f64, |a, b| a.max(*b));
    let c_max = s_map.iter().fold(0.0f64, |a, b| a.max(*b));

    Visibility {
        p_map,
        p_det,
        c_map: s_map,
        c_max,
        width: rec.width,
        height: rec.height,
    }
}

/// The raw quality correlate `Q` — the sum of the per-plane terms the masking
/// loop produced. Negative, rising toward 0 as distortion grows.
#[must_use]
pub fn quality_correlate(terms: &[f64]) -> f64 {
    terms.iter().sum()
}

/// Map `Q` onto the 0–100 mean-opinion-score scale, **100 = best**.
///
/// `Q_MOS = 100 / (1 + exp(q₁·(Q + q₂)))` with the 2.2 constants
/// `q₁ = 3.455`, `q₂ = 0.8886`. The logistic is decreasing in `Q`, so the
/// rising `Q` of a worse image maps to a falling score.
#[must_use]
#[inline]
pub fn quality_mos(q: f64, par: &Params) -> f64 {
    100.0 / (1.0 + (par.quality_logistic_q1 * (q + par.quality_logistic_q2)).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_is_a_half_exactly_at_threshold() {
        // The defining property of the psychometric mapping: C = 1 is the
        // detection threshold, so P must be exactly 0.5 there.
        let p = |c: f64| 1.0 - (0.5f64.ln() * c).exp();
        assert!((p(1.0) - 0.5).abs() < 1e-15);
        assert_eq!(p(0.0), 0.0);
        assert!(p(10.0) > 0.999);
        // Monotone and bounded.
        let mut prev = -1.0;
        for i in 0..=200 {
            let v = p(i as f64 / 20.0);
            assert!((0.0..=1.0).contains(&v));
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn q_mos_is_decreasing_in_q_and_spans_the_scale() {
        let par = Params::new(30.0);
        // A very negative Q (an all-but-identical pair) sits near 100; a Q
        // approaching 0 (a badly distorted pair) sits near 0.
        assert!(
            quality_mos(-20.0, &par) > 99.99,
            "{}",
            quality_mos(-20.0, &par)
        );
        assert!(quality_mos(5.0, &par) < 0.01, "{}", quality_mos(5.0, &par));
        let mut prev = f64::INFINITY;
        for i in -100..=100 {
            let q = i as f64 / 10.0;
            let m = quality_mos(q, &par);
            assert!((0.0..=100.0).contains(&m), "Q_MOS out of range: {m}");
            assert!(m < prev, "Q_MOS must fall as Q rises");
            prev = m;
        }
        // The logistic's midpoint is at Q = −q2.
        let mid = quality_mos(-par.quality_logistic_q2, &par);
        assert!((mid - 50.0).abs() < 1e-9, "midpoint {mid}");
    }

    #[test]
    fn quality_correlate_is_just_the_sum() {
        assert_eq!(quality_correlate(&[]), 0.0);
        assert!((quality_correlate(&[-1.0, -2.5, 0.5]) + 3.0).abs() < 1e-15);
    }

    #[test]
    fn all_zero_difference_bands_give_no_visibility() {
        // A pyramid of zeros must reconstruct to zero, so P_det = 0 — no
        // spurious detection from the pooling or the reconstruction.
        let (w, h) = (64usize, 48usize);
        let par = Params::new(30.0);
        let im = vec![0.0; w * h];
        let pyr = crate::spyr::build(&im, w, h, None);
        let bp = BandPyramid {
            bands: core::iter::once(vec![pyr.high_pass.clone()])
                .chain(pyr.levels.iter().map(|l| l.to_vec()))
                .chain(core::iter::once(vec![pyr.low_pass.clone()]))
                .collect(),
            width: w,
            height: h,
        };
        let v = visibility(&bp, &par);
        assert_eq!((v.width, v.height), (w, h));
        assert!(v.p_det < 1e-12, "P_det = {}", v.p_det);
        assert!(v.c_max < 1e-12, "C_max = {}", v.c_max);
        assert!(v.p_map.iter().all(|p| p.abs() < 1e-12));
    }

    #[test]
    fn to_steerable_round_trips_the_band_list() {
        let (w, h) = (64usize, 48usize);
        let im: Vec<f64> = (0..w * h).map(|i| (i as f64 * 0.21).sin()).collect();
        let pyr = crate::spyr::build(&im, w, h, None);
        let bp = BandPyramid {
            bands: core::iter::once(vec![pyr.high_pass.clone()])
                .chain(pyr.levels.iter().map(|l| l.to_vec()))
                .chain(core::iter::once(vec![pyr.low_pass.clone()]))
                .collect(),
            width: w,
            height: h,
        };
        let back = to_steerable(&bp);
        assert_eq!(back.height_levels(), pyr.height_levels());
        assert_eq!(back.high_pass.data, pyr.high_pass.data);
        assert_eq!(back.low_pass.data, pyr.low_pass.data);
        for (a, b) in back.levels.iter().zip(&pyr.levels) {
            for (x, y) in a.iter().zip(b) {
                assert_eq!(x.data, y.data);
            }
        }
    }

    #[test]
    fn spatial_pooling_turns_the_peak_into_a_total() {
        // `S_map ·= sum/max` scales the map so that its NEW maximum equals the
        // OLD sum. That is easy to misread as a normalisation, so pin it: with
        // pooling on, `C_max` is the total un-pooled difference mass; with it
        // off, `C_max` is the honest per-pixel peak. Anyone reading a `C_max`
        // of 400 and concluding "400× threshold" is reading the wrong quantity.
        let (w, h) = (64usize, 64usize);
        let mut par = Params::new(30.0);
        let zero = crate::spyr::build(&vec![0.0; w * h], w, h, None);
        let mut bp = BandPyramid {
            bands: core::iter::once(vec![zero.high_pass.clone()])
                .chain(zero.levels.iter().map(|l| l.to_vec()))
                .chain(core::iter::once(vec![zero.low_pass.clone()]))
                .collect(),
            width: w,
            height: h,
        };
        let last = bp.count() - 1;
        let n = bp.bands[last][0].data.len();
        bp.bands[last][0].data[n / 2] = 1.0;

        par.do_spatial_pooling = false;
        let raw = visibility(&bp, &par);
        let raw_sum: f64 = raw.c_map.iter().sum();
        let raw_max = raw.c_max;

        par.do_spatial_pooling = true;
        let pooled = visibility(&bp, &par);
        assert!(
            (pooled.c_max - raw_sum).abs() < 1e-9 * raw_sum,
            "pooled C_max {} should equal the un-pooled sum {raw_sum}",
            pooled.c_max
        );
        assert!(
            raw_sum > raw_max,
            "the fixture must actually spread over more than one pixel"
        );
        // The map's SHAPE is untouched — pooling is a uniform rescale, so the
        // argmax does not move and P_map still localises.
        let arg = |v: &[f64]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        assert_eq!(arg(&raw.c_map), arg(&pooled.c_map));
    }
}
