//! Optical MTF, neural CSF, and the rod / cone luminance-sensitivity curves.
//!
//! Ported from `hdrvdp_mtf.m`, `hdrvdp_ncsf.m`, `hdrvdp_joint_rod_cone_sens.m`
//! and `hdrvdp_rod_sens.m` of the MATLAB `hdrvdp-2.2.x` release.
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

use crate::interp::{clamp, interp1_linear};
use crate::params::Params;

/// Custom-fit modulation transfer function of the eye's optics:
/// `MTF(ρ) = Σ_{k=1..4} a_k · exp(−b_k · ρ)`, with `ρ` in cycles per degree.
///
/// Because `Σ a_k = 1`, `mtf(0) = 1` — the optics do not attenuate DC.
#[must_use]
#[inline]
pub fn mtf(rho: f64, par: &Params) -> f64 {
    let mut m = 0.0;
    for k in 0..4 {
        m += par.mtf_params_a[k] * (-par.mtf_params_b[k] * rho).exp();
    }
    m
}

/// Neural contrast sensitivity function at spatial frequency `rho`
/// (cycles/degree) and adapting luminance `lum` (cd/m²).
///
/// This is *neural only*: it excludes the optical component ([`mtf`]) and the
/// luminance-dependent component ([`joint_rod_cone_sens`]). The full CSF is the
/// product of the three. The peaks are deliberately **not** normalised to 1 —
/// that slack absorbs small variations in the contrast-versus-intensity
/// function.
///
/// Shape:
/// `S = p₄ · (1 − exp(−(ρ/7)²))^(p₃/2) / sqrt(1 + (p₁·ρ)^p₂)`,
/// with `p₁..p₄` linearly interpolated in `log₁₀` adapting luminance over the
/// six calibration luminances.
#[must_use]
pub fn ncsf(rho: f64, lum: f64, par: &Params) -> f64 {
    let lum_lut: Vec<f64> = par.csf_lums.iter().map(|l| l.log10()).collect();
    let log_lum = clamp(lum.log10(), lum_lut[0], lum_lut[lum_lut.len() - 1]);

    let mut p = [0.0f64; 4];
    for (k, pk) in p.iter_mut().enumerate() {
        let col: Vec<f64> = par.csf_params.iter().map(|row| row[k + 1]).collect();
        *pk = interp1_linear(&lum_lut, &col, log_lum);
    }
    ncsf_from_coeffs(rho, &p)
}

/// [`ncsf`] with the four luminance-interpolated coefficients already resolved.
///
/// Used by the band loop, which resolves the coefficients once per adapting
/// luminance and then evaluates over a whole frequency plane.
#[must_use]
#[inline]
pub fn ncsf_from_coeffs(rho: f64, p: &[f64; 4]) -> f64 {
    // Low-frequency attenuation; → 0 as ρ → 0, so DC has no neural sensitivity.
    let b = 1.0 - (-(rho / 7.0).powi(2)).exp();
    let a = 1.0 + (p[0] * rho).powf(p[1]);
    p[3] * b.powf(p[2] / 2.0) / a.sqrt()
}

/// Resolve the four nCSF coefficients at adapting luminance `lum` (cd/m²).
#[must_use]
pub fn ncsf_coeffs(lum: f64, par: &Params) -> [f64; 4] {
    let lum_lut: Vec<f64> = par.csf_lums.iter().map(|l| l.log10()).collect();
    let log_lum = clamp(lum.log10(), lum_lut[0], lum_lut[lum_lut.len() - 1]);
    let mut p = [0.0f64; 4];
    for (k, pk) in p.iter_mut().enumerate() {
        let col: Vec<f64> = par.csf_params.iter().map(|row| row[k + 1]).collect();
        *pk = interp1_linear(&lum_lut, &col, log_lum);
    }
    p
}

/// Joint rod+cone sensitivity as a function of adapting luminance (cd/m²).
///
/// `S(L) = p₅ · ((p₆/L)^p₇ + 1)^(−p₈)` — the contrast-versus-intensity
/// (c.v.i.) curve, saturating at `p₅` for high luminance and dropping off
/// below `p₆`.
#[must_use]
#[inline]
pub fn joint_rod_cone_sens(la: f64, par: &Params) -> f64 {
    let [peak, cvi_sens_drop, cvi_trans_slope, cvi_low_slope] = par.csf_sa;
    peak * ((cvi_sens_drop / la).powf(cvi_trans_slope) + 1.0).powf(-cvi_low_slope)
}

/// Rod-only sensitivity as a function of adapting luminance (cd/m²).
///
/// An asymmetric log-Gaussian-like bump peaking at `csf_sr_par[0]` cd/m²,
/// with different widths/exponents below and above the peak, scaled by
/// `10^csf_sr_par[5]`.
#[must_use]
#[inline]
pub fn rod_sens(la: f64, par: &Params) -> f64 {
    let [peak_l, low_s, low_exp, high_s, high_exp, rod_sens_scale] = par.csf_sr_par;
    let t = (la / peak_l).log10().abs();
    let s = if la > peak_l {
        (-t.powf(high_exp) / high_s).exp()
    } else {
        (-t.powf(low_exp) / low_s).exp()
    };
    s * 10f64.powf(rod_sens_scale)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn par() -> Params {
        Params::new(30.0)
    }

    #[test]
    fn mtf_is_unity_at_dc_and_decays_monotonically() {
        let p = par();
        assert!((mtf(0.0, &p) - 1.0).abs() < 1e-12);
        let mut prev = mtf(0.0, &p);
        for i in 1..=200 {
            let rho = i as f64 * 0.5; // 0.5 .. 100 cpd
            let m = mtf(rho, &p);
            assert!(m < prev, "MTF not decreasing at ρ = {rho}: {m} >= {prev}");
            assert!(
                (0.0..=1.0).contains(&m),
                "MTF out of range at ρ = {rho}: {m}"
            );
            prev = m;
        }
        // The three fast-decaying terms (b = 0.37, 37, 360) are spent well
        // before 60 cpd; what survives is a₁·exp(−0.028·60) = 0.4248·0.1864.
        // So the optics still pass ~8% at 60 cpd — small, but far from zero,
        // which is what keeps the fovea's finest gratings visible at all.
        let m60 = mtf(60.0, &p);
        assert!(
            (m60 - 0.0792).abs() < 5e-4,
            "MTF(60 cpd) = {m60}, expected ≈ 0.0792"
        );
    }

    #[test]
    fn ncsf_is_zero_at_dc_and_band_pass() {
        let p = par();
        for &lum in &[0.002, 0.2, 20.0, 150.0] {
            assert!(
                ncsf(0.0, lum, &p).abs() < 1e-12,
                "nCSF(0, {lum}) should vanish"
            );
            // Band-pass: sample 0.1..30 cpd and check the peak is interior.
            let n = 300;
            let vals: Vec<f64> = (0..n)
                .map(|i| {
                    let rho = 0.1 + i as f64 * (30.0 - 0.1) / (n - 1) as f64;
                    ncsf(rho, lum, &p)
                })
                .collect();
            let (argmax, &peak) = vals
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert!(
                argmax > 0 && argmax < n - 1,
                "nCSF at {lum} cd/m² is not band-pass (peak at edge {argmax})"
            );
            assert!(peak > 0.0);
        }
    }

    #[test]
    fn ncsf_peak_frequency_falls_with_luminance() {
        // Classic CSF behaviour: as adaptation drops, the peak moves to lower
        // spatial frequency AND peak sensitivity drops. Reproducing this is a
        // real check on the parameter table and the interpolation.
        let p = par();
        let peak_at = |lum: f64| -> (f64, f64) {
            let n = 2000;
            let mut best = (0.0, f64::NEG_INFINITY);
            for i in 0..n {
                let rho = 0.05 + i as f64 * (40.0 - 0.05) / (n - 1) as f64;
                let s = ncsf(rho, lum, &p);
                if s > best.1 {
                    best = (rho, s);
                }
            }
            best
        };
        let (rho_hi, s_hi) = peak_at(150.0);
        let (rho_mid, s_mid) = peak_at(2.0);
        let (rho_lo, s_lo) = peak_at(0.002);
        assert!(
            rho_hi > rho_mid && rho_mid > rho_lo,
            "peak frequency should fall with luminance: {rho_hi} / {rho_mid} / {rho_lo}"
        );
        assert!(
            s_hi > s_mid && s_mid > s_lo,
            "peak sensitivity should fall with luminance: {s_hi} / {s_mid} / {s_lo}"
        );
    }

    #[test]
    fn ncsf_clamps_outside_the_calibrated_luminance_range() {
        let p = par();
        // Below 0.002 and above 150 cd/m² the coefficients clamp, so the
        // returned sensitivity must equal the endpoint's.
        assert_eq!(ncsf(4.0, 1e-9, &p), ncsf(4.0, 0.002, &p));
        assert_eq!(ncsf(4.0, 1e9, &p), ncsf(4.0, 150.0, &p));
    }

    #[test]
    fn joint_rod_cone_sens_saturates_at_the_fitted_peak() {
        let p = par();
        // S → csf_sa[0] = 30.162 as L → ∞, monotonically increasing.
        let hi = joint_rod_cone_sens(1e8, &p);
        assert!((hi - p.csf_sa[0]).abs() / p.csf_sa[0] < 1e-3, "{hi}");
        let mut prev = 0.0;
        for i in -50..=50 {
            let la = 10f64.powf(i as f64 / 10.0);
            let s = joint_rod_cone_sens(la, &p);
            assert!(s > prev, "c.v.i. not increasing at {la}");
            assert!(s <= p.csf_sa[0]);
            prev = s;
        }
    }

    #[test]
    fn rod_sens_peaks_at_the_fitted_luminance() {
        let p = par();
        let peak_l = p.csf_sr_par[0]; // 1.1732 cd/m²
        let s_peak = rod_sens(peak_l, &p);
        // The peak value is exactly the scale factor 10^rod_sens.
        assert!(
            (s_peak - 10f64.powf(p.csf_sr_par[5])).abs() < 1e-12,
            "{s_peak}"
        );
        for i in -60..=60 {
            if i == 0 {
                continue;
            }
            let la = peak_l * 10f64.powf(i as f64 / 10.0);
            assert!(
                rod_sens(la, &p) < s_peak,
                "rod sensitivity at {la} exceeds the peak"
            );
        }
        // Asymmetric: the high side (exp 2.9899, s 0.5547) falls off much
        // faster than the low side (exp 1.2167, s 1.1478).
        assert!(rod_sens(peak_l * 100.0, &p) < rod_sens(peak_l / 100.0, &p));
    }

    #[test]
    fn ncsf_from_coeffs_matches_the_full_path() {
        let p = par();
        for &lum in &[0.002, 0.05, 3.0, 150.0, 5000.0] {
            let c = ncsf_coeffs(lum, &p);
            for &rho in &[0.5, 2.0, 8.0, 32.0] {
                let a = ncsf(rho, lum, &p);
                let b = ncsf_from_coeffs(rho, &c);
                assert!((a - b).abs() < 1e-15, "{a} != {b} at ρ={rho}, L={lum}");
            }
        }
    }
}
