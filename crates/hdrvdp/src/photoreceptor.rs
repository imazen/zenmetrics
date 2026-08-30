//! Photoreceptor non-linearity: the luminance → JND-space response.
//!
//! Ported from `create_pn_jnd` / `build_jndspace_from_S` in
//! `hdrvdp_visual_pathway.m` of the MATLAB `hdrvdp-2.2.x` release.
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
//! ## What this is
//!
//! The response is defined implicitly by the sensitivity: one JND step is a
//! contrast of `1/S(L)`, i.e. a luminance step of `ΔL = L/S(L)`. Integrating
//! `1/ΔL` over luminance gives a scale on which **equal distance = equal
//! discriminability**, which is what the rest of the metric operates in.
//!
//! Upstream integrates in the log domain, so the integrand picks up the
//! Jacobian of `L = 10^l`:
//!
//! ```text
//!   jnd(l) = ∫ (1/ΔL(L)) · dL = ∫ (S(L)/L) · L·ln(10) dl = ∫ S(10^l)·ln(10) dl
//! ```
//!
//! Cones and rods get separate curves, and the two pathways are DC-removed
//! separately before being summed into the achromatic response.

use crate::interp::{clamp, cumtrapz, interp1_linear, point_op};
use crate::params::Params;

/// Number of samples in the JND lookup table (upstream: `logspace(-5,5,2048)`).
const LUT_N: usize = 2048;
/// `log10` of the lowest tabulated luminance.
const LUT_LOG_MIN: f64 = -5.0;
/// `log10` of the highest tabulated luminance.
const LUT_LOG_MAX: f64 = 5.0;

/// Cone and rod luminance → JND-space lookup tables.
#[derive(Debug, Clone)]
pub struct Photoreceptor {
    /// `log10` luminance grid, uniform over `[-5, 5]`.
    log_lum: Vec<f64>,
    /// Cone response in JND units, already scaled by `sensitivity_correction`.
    jnd_cone: Vec<f64>,
    /// Rod response in JND units, already scaled by `sensitivity_correction`.
    jnd_rod: Vec<f64>,
    /// `10^log_lum[0]`, memoised: [`Self::cone`] / [`Self::rod`] clamp against
    /// the table range per *pixel*, and recomputing `powf` there was a
    /// measurable slice of the whole metric. Same expression, computed once.
    lum_min: f64,
    /// `10^log_lum[last]`, memoised likewise.
    lum_max: f64,
}

impl Photoreceptor {
    /// Build the tables for the given parameters.
    #[must_use]
    pub fn new(par: &Params) -> Self {
        // Luminance grid, log-spaced over 10 decades.
        let log_lum: Vec<f64> = (0..LUT_N)
            .map(|i| LUT_LOG_MIN + (LUT_LOG_MAX - LUT_LOG_MIN) * (i as f64) / ((LUT_N - 1) as f64))
            .collect();
        let lum: Vec<f64> = log_lum.iter().map(|l| 10f64.powf(*l)).collect();

        // Joint (rod+cone) and rod-only sensitivity over the grid.
        let s_a: Vec<f64> = lum
            .iter()
            .map(|&l| crate::csf::joint_rod_cone_sens(l, par))
            .collect();
        let rod_scale = 10f64.powf(par.rod_sensitivity);
        let s_r: Vec<f64> = lum
            .iter()
            .map(|&l| crate::csf::rod_sens(l, par) * rod_scale)
            .collect();

        // Cone sensitivity: the joint curve minus the rod contribution, floored
        // at 1e-3, shifted one octave up in luminance and halved — upstream
        // resamples `max(s_A - s_R, 1e-3)` at `min(2·L, L_max)`, linearly on the
        // *linear* luminance grid. (Both the octave shift and the linear-grid
        // resampling are reproduced deliberately; they are part of the fit.)
        let cone_src: Vec<f64> = s_a
            .iter()
            .zip(&s_r)
            .map(|(a, r)| (a - r).max(1e-3))
            .collect();
        let l_max = lum[LUT_N - 1];
        let s_c: Vec<f64> = lum
            .iter()
            .map(|&l| 0.5 * interp1_linear(&lum, &cone_src, (l * 2.0).min(l_max)))
            .collect();

        let sc = par.sensitivity_correction;
        let jnd_cone: Vec<f64> = build_jnd_space(&log_lum, &s_c)
            .into_iter()
            .map(|v| v * sc)
            .collect();
        let jnd_rod: Vec<f64> = build_jnd_space(&log_lum, &s_r)
            .into_iter()
            .map(|v| v * sc)
            .collect();

        let lum_min = 10f64.powf(log_lum[0]);
        let lum_max = 10f64.powf(log_lum[log_lum.len() - 1]);
        Self {
            log_lum,
            jnd_cone,
            jnd_rod,
            lum_min,
            lum_max,
        }
    }

    /// Lowest tabulated luminance in cd/m² (`1e-5`).
    #[must_use]
    pub fn lum_min(&self) -> f64 {
        self.lum_min
    }

    /// Highest tabulated luminance in cd/m² (`1e5`).
    #[must_use]
    pub fn lum_max(&self) -> f64 {
        self.lum_max
    }

    fn step(&self) -> f64 {
        self.log_lum[1] - self.log_lum[0]
    }

    /// Cone response, in JND units, for a linear luminance in cd/m².
    ///
    /// The input is clamped to the tabulated range before the lookup, matching
    /// upstream.
    #[must_use]
    #[inline]
    pub fn cone(&self, lum: f64) -> f64 {
        let l = clamp(lum, self.lum_min(), self.lum_max()).log10();
        point_op(&self.jnd_cone, self.log_lum[0], self.step(), l)
    }

    /// Rod response, in JND units, for a linear luminance in cd/m².
    #[must_use]
    #[inline]
    pub fn rod(&self, lum: f64) -> f64 {
        let l = clamp(lum, self.lum_min(), self.lum_max()).log10();
        point_op(&self.jnd_rod, self.log_lum[0], self.step(), l)
    }

    /// The raw cone table (JND units) and its `log10`-luminance grid.
    #[must_use]
    pub fn cone_table(&self) -> (&[f64], &[f64]) {
        (&self.log_lum, &self.jnd_cone)
    }

    /// The raw rod table (JND units) and its `log10`-luminance grid.
    #[must_use]
    pub fn rod_table(&self) -> (&[f64], &[f64]) {
        (&self.log_lum, &self.jnd_rod)
    }
}

/// `jnd(l) = ∫ S(10^l) · ln(10) dl`, cumulative from the first sample.
fn build_jnd_space(log_lum: &[f64], s: &[f64]) -> Vec<f64> {
    debug_assert_eq!(log_lum.len(), s.len());
    let ln10 = core::f64::consts::LN_10;
    let d: Vec<f64> = s.iter().map(|v| v * ln10).collect();
    cumtrapz(log_lum, &d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn par() -> Params {
        Params::new(30.0)
    }

    #[test]
    fn tables_are_monotone_and_start_at_zero() {
        let pn = Photoreceptor::new(&par());
        let (_, cone) = pn.cone_table();
        let (_, rod) = pn.rod_table();
        assert_eq!(cone[0], 0.0);
        assert_eq!(rod[0], 0.0);
        for t in [cone, rod] {
            for w in t.windows(2) {
                assert!(w[1] >= w[0], "JND space must be non-decreasing");
            }
        }
        // Cones keep accumulating JNDs across the whole range; rods saturate
        // (their sensitivity vanishes far above the ~1.17 cd/m² peak), so the
        // rod curve must flatten while the cone curve does not.
        let n = cone.len();
        let cone_top_decade = cone[n - 1] - cone[n - 1 - n / 10];
        let rod_top_decade = rod[n - 1] - rod[n - 1 - n / 10];
        assert!(
            cone_top_decade > 1.0,
            "cone response should still be growing at 1e5 cd/m²: {cone_top_decade}"
        );
        assert!(
            rod_top_decade < 1e-6,
            "rod response should be saturated at 1e5 cd/m²: {rod_top_decade}"
        );
    }

    #[test]
    fn one_jnd_step_is_one_unit_of_response() {
        // The defining property: at adaptation L, a luminance increment of
        // L/S(L) (one detection threshold) must move the response by ≈ 1 unit
        // once the `sensitivity_correction` scaling is divided back out.
        //
        // `sensitivity_correction` rescales the whole space, and the cone
        // curve integrates s_C (the octave-shifted half of the joint c.v.i.),
        // so this is checked against the same s_C, not against s_A.
        let p = par();
        let pn = Photoreceptor::new(&p);
        for &l in &[1.0, 10.0, 100.0, 1000.0] {
            let s_a = crate::csf::joint_rod_cone_sens(l * 2.0, &p);
            let s_r = crate::csf::rod_sens(l * 2.0, &p);
            let s_c = 0.5 * (s_a - s_r).max(1e-3);
            let thr = l / s_c;
            let d = (pn.cone(l + thr) - pn.cone(l)) / p.sensitivity_correction;
            assert!(
                (d - 1.0).abs() < 0.05,
                "one threshold at {l} cd/m² moved the cone response by {d}, want ≈ 1"
            );
        }
    }

    #[test]
    fn lookup_clamps_outside_the_table() {
        let pn = Photoreceptor::new(&par());
        assert_eq!(pn.cone(1e-30), pn.cone(pn.lum_min()));
        assert_eq!(pn.cone(1e30), pn.cone(pn.lum_max()));
        assert_eq!(pn.rod(0.0), pn.rod(pn.lum_min()));
        // NaN must not produce NaN downstream — it clamps to the low end.
        assert!(pn.cone(f64::NAN).is_finite());
    }

    #[test]
    fn sensitivity_correction_scales_the_space_linearly() {
        let a = Photoreceptor::new(&Params::new(30.0));
        let b = Photoreceptor::new(&Params::new(30.0).with_peak_sensitivity(2.7));
        let ratio = 10f64.powf(2.7) / 10f64.powf(2.4);
        for &l in &[0.01, 1.0, 100.0] {
            let got = b.cone(l) / a.cone(l);
            assert!((got - ratio).abs() < 1e-9, "{got} != {ratio} at {l}");
        }
    }

    #[test]
    fn rod_response_dominates_in_scotopic_and_cones_in_photopic() {
        // Sanity on the two pathways: below ~0.01 cd/m² the rod curve must
        // accumulate faster than the cone curve, and above ~100 cd/m² the
        // reverse. This is the whole point of carrying both.
        let pn = Photoreceptor::new(&par());
        let d_rod_scotopic = pn.rod(1e-2) - pn.rod(1e-3);
        let d_cone_scotopic = pn.cone(1e-2) - pn.cone(1e-3);
        assert!(
            d_rod_scotopic > d_cone_scotopic,
            "rods should out-accumulate cones in scotopic: {d_rod_scotopic} vs {d_cone_scotopic}"
        );
        let d_rod_photopic = pn.rod(1e3) - pn.rod(1e2);
        let d_cone_photopic = pn.cone(1e3) - pn.cone(1e2);
        assert!(
            d_cone_photopic > d_rod_photopic,
            "cones should out-accumulate rods in photopic: {d_cone_photopic} vs {d_rod_photopic}"
        );
    }
}
