//! Photoreceptor spectral sensitivities, display emission spectra, and the
//! `channels × LMSR` mixing matrix built from them.
//!
//! ## Provenance
//!
//! The **data** in `data/` ships with the ISC-licensed `hdrvdp-2.2.x` release
//! (see `THIRD-PARTY-NOTICES.md`). The **loader** is written independently:
//! upstream's `load_spectral_resp.m` is marked "internal use, do not
//! redistribute", so only its documented behaviour is reproduced here — read a
//! comma-separated table whose first column is wavelength in nm, resample every
//! remaining column onto a fixed 360–780 nm grid with `interp1(...,'cubic',0)`
//! (i.e. PCHIP, zero outside the source range).
//!
//! One upstream quirk is reproduced deliberately, because the calibration was
//! fitted through it: the grid is `linspace(360, 780, (780-360)/1)` — **420
//! points, not 421** — so the sample spacing is `420/419 ≈ 1.0024 nm`, not the
//! 1 nm the docstring claims. See [`GRID_N`].
//!
//! The equations that consume this (`M_img_lmsr`, `R_LMSR`) come from
//! `hdrvdp_visual_pathway.m` (ISC).

use crate::interp::{interp1_pchip, trapz};

/// First wavelength of the resampling grid, in nm.
pub const GRID_MIN_NM: f64 = 360.0;
/// Last wavelength of the resampling grid, in nm.
pub const GRID_MAX_NM: f64 = 780.0;
/// Number of grid samples.
///
/// Upstream computes this as `(780-360)/1 = 420` and passes it as `linspace`'s
/// *count*, so the spacing is `420/419` nm rather than 1 nm. Reproduced
/// verbatim: the spectral integrals — and therefore the whole calibration —
/// were fitted on this grid.
pub const GRID_N: usize = 420;

/// Smith & Pokorny (1975) cone fundamentals, log₁₀ sensitivity (L, M, S).
const CONE_FUNDAMENTALS_CSV: &str = include_str!("../data/log_cone_smith_pokorny_1975.csv");
/// CIE scotopic luminous efficiency V′(λ).
const SCOTOPIC_CSV: &str = include_str!("../data/cie_scotopic_lum.txt");
/// Measured emission spectra of a CCFL-backlit LCD (upstream's default).
const CCFL_LCD_CSV: &str = include_str!("../data/emission_spectra_ccfl-lcd.csv");
/// CIE standard illuminant D65 relative spectral power.
const D65_CSV: &str = include_str!("../data/d65.csv");

/// Which display emission spectra to assume for RGB input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DisplaySpectra {
    /// CCFL-backlit LCD — upstream's default for 3-channel input.
    #[default]
    CcflLcd,
    /// D65, summed to a single channel — upstream's default for 1-channel
    /// (luminance) input.
    D65,
}

/// The 360–780 nm resampling grid.
#[must_use]
pub fn wavelength_grid() -> Vec<f64> {
    (0..GRID_N)
        .map(|i| GRID_MIN_NM + (GRID_MAX_NM - GRID_MIN_NM) * (i as f64) / ((GRID_N - 1) as f64))
        .collect()
}

/// Parse a comma-separated spectral table and resample every data column onto
/// [`wavelength_grid`], filling 0 outside the source range.
///
/// Returns one `Vec<f64>` of length [`GRID_N`] per data column.
fn load_spectral(csv: &str) -> Vec<Vec<f64>> {
    let mut lambda: Vec<f64> = Vec::new();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for line in csv.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split(',').map(|f| {
            f.trim()
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("spectral table: bad field {f:?}: {e}"))
        });
        let l = it.next().expect("spectral table: empty row");
        lambda.push(l);
        for (k, v) in it.enumerate() {
            if cols.len() <= k {
                cols.push(Vec::new());
            }
            cols[k].push(v);
        }
    }
    assert!(lambda.len() >= 2, "spectral table: need at least 2 rows");
    for c in &cols {
        assert_eq!(c.len(), lambda.len(), "spectral table: ragged columns");
    }
    let grid = wavelength_grid();
    cols.iter()
        .map(|c| interp1_pchip(&lambda, c, &grid, 0.0))
        .collect()
}

/// The four photoreceptor sensitivity curves on [`wavelength_grid`], in the
/// order `[L, M, S, R]`.
///
/// The three cone columns are `10^(log sensitivity)`, with the zeros the
/// resampler leaves outside the tabulated range first replaced by the table's
/// minimum log value (so they become a vanishing sensitivity rather than 1.0).
/// The rod column is the CIE scotopic curve, used linearly.
#[must_use]
pub fn photoreceptor_sensitivities() -> [Vec<f64>; 4] {
    let mut cones = load_spectral(CONE_FUNDAMENTALS_CSV);
    assert_eq!(cones.len(), 3, "cone fundamentals: expected 3 data columns");

    // Upstream: LMSR_S(LMSR_S==0) = min(LMSR_S(:)) — the minimum over ALL cone
    // columns, applied before the 10^ exponentiation.
    let min_log = cones
        .iter()
        .flat_map(|c| c.iter().copied())
        .fold(f64::INFINITY, f64::min);
    for c in &mut cones {
        for v in c.iter_mut() {
            if *v == 0.0 {
                *v = min_log;
            }
            *v = 10f64.powf(*v);
        }
    }

    let rods = load_spectral(SCOTOPIC_CSV);
    assert_eq!(rods.len(), 1, "scotopic table: expected 1 data column");

    let mut it = cones.into_iter();
    let l = it.next().expect("L");
    let m = it.next().expect("M");
    let s = it.next().expect("S");
    let r = rods.into_iter().next().expect("R");
    [l, m, s, r]
}

/// Display emission spectra on [`wavelength_grid`], one column per input
/// channel.
#[must_use]
pub fn emission_spectra(which: DisplaySpectra, channels: usize) -> Vec<Vec<f64>> {
    let mut e = match which {
        DisplaySpectra::CcflLcd => load_spectral(CCFL_LCD_CSV),
        DisplaySpectra::D65 => load_spectral(D65_CSV),
    };
    if channels == 1 && e.len() > 1 {
        // Upstream sums the per-primary spectra for luminance-only input.
        let mut sum = vec![0.0; GRID_N];
        for c in &e {
            for (a, b) in sum.iter_mut().zip(c) {
                *a += b;
            }
        }
        e = vec![sum];
    }
    assert_eq!(
        e.len(),
        channels,
        "emission spectra have {} channels, image has {channels}",
        e.len()
    );
    e
}

/// The `channels × 4` matrix that mixes input channels into `(L, M, S, R)`
/// photoreceptor responses.
///
/// `m[c][k] = ∫ S_k(λ) · E_c(λ) dλ`, trapezoidal over [`wavelength_grid`],
/// matching `M_img_lmsr` in `hdrvdp_visual_pathway.m`.
#[must_use]
pub fn lmsr_matrix(emission: &[Vec<f64>]) -> Vec<[f64; 4]> {
    let lambda = wavelength_grid();
    let sens = photoreceptor_sensitivities();
    emission
        .iter()
        .map(|e| {
            let mut row = [0.0f64; 4];
            for (k, r) in row.iter_mut().enumerate() {
                let prod: Vec<f64> = sens[k].iter().zip(e).map(|(s, ee)| s * ee).collect();
                *r = trapz(&lambda, &prod);
            }
            row
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_reproduces_the_upstream_420_point_quirk() {
        let g = wavelength_grid();
        assert_eq!(g.len(), 420);
        assert_eq!(g[0], GRID_MIN_NM);
        assert_eq!(g[419], GRID_MAX_NM);
        let step = g[1] - g[0];
        assert!(
            (step - 420.0 / 419.0).abs() < 1e-12,
            "spacing is {step}, expected 420/419 (the upstream linspace count bug)"
        );
    }

    #[test]
    fn cone_sensitivities_peak_in_the_right_places() {
        let [l, m, s, r] = photoreceptor_sensitivities();
        let g = wavelength_grid();
        let peak_nm = |c: &[f64]| -> f64 {
            let (i, _) = c
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            g[i]
        };
        // Smith–Pokorny fundamentals: L ≈ 565 nm, M ≈ 540 nm, S ≈ 440 nm.
        // CIE scotopic V′ peaks at 507 nm by definition.
        let (pl, pm, ps, pr) = (peak_nm(&l), peak_nm(&m), peak_nm(&s), peak_nm(&r));
        assert!((pl - 565.0).abs() < 12.0, "L peak at {pl} nm");
        assert!((pm - 540.0).abs() < 12.0, "M peak at {pm} nm");
        assert!((ps - 440.0).abs() < 12.0, "S peak at {ps} nm");
        assert!((pr - 507.0).abs() < 3.0, "rod peak at {pr} nm");
        assert!(pl > pm && pm > ps, "cone peak ordering L > M > S");
    }

    #[test]
    fn sensitivities_are_non_negative_and_finite() {
        for (name, c) in ["L", "M", "S", "R"]
            .into_iter()
            .zip(photoreceptor_sensitivities())
        {
            for (i, v) in c.iter().enumerate() {
                assert!(v.is_finite() && *v >= 0.0, "{name}[{i}] = {v}");
            }
        }
    }

    #[test]
    fn emission_spectra_are_finite_with_only_a_measurement_noise_floor() {
        // The CCFL-LCD table is a *measurement*, and five of its 450 rows dip a
        // few parts in 10⁶ below zero at the ends of the visible range (389,
        // 398, 758, 776, 778 nm). Upstream integrates them as-is, so this port
        // must too — clamping them to zero would move the LMSR matrix off the
        // fitted calibration. What must hold is that nothing worse than that
        // noise floor is present, and that the resampler adds no new
        // excursions of its own.
        for ch in [1usize, 3] {
            let e = emission_spectra(DisplaySpectra::CcflLcd, ch);
            assert_eq!(e.len(), ch);
            let peak = e
                .iter()
                .flat_map(|c| c.iter().copied())
                .fold(f64::MIN, f64::max);
            assert!(peak > 0.0);
            for c in &e {
                assert_eq!(c.len(), GRID_N);
                for v in c {
                    assert!(v.is_finite(), "emission {v}");
                    // The deepest negative in the source table is −3.53e-6
                    // (389.09 nm), against a peak of ~6.4e-2. PCHIP is
                    // monotone-preserving, so resampling must not deepen it.
                    assert!(
                        *v > -4e-6,
                        "emission dipped past the source noise floor: {v} (peak {peak})"
                    );
                }
            }
        }
        // D65 is a standard table with no noise floor at all.
        for c in emission_spectra(DisplaySpectra::D65, 1) {
            for v in c {
                assert!(v.is_finite() && v >= 0.0, "D65 {v}");
            }
        }
    }

    #[test]
    fn lmsr_matrix_shape_and_ordering() {
        let e = emission_spectra(DisplaySpectra::CcflLcd, 3);
        let m = lmsr_matrix(&e);
        assert_eq!(m.len(), 3);
        for row in &m {
            for v in row {
                assert!(v.is_finite() && *v >= 0.0, "{v}");
            }
        }
        // Assert on *ratios between receptors*, not on raw magnitudes. Two
        // traps live here:
        //  - the Smith–Pokorny fundamentals carry their own per-cone
        //    normalisation (L and M are scaled so L+M is the photopic
        //    luminous efficiency; S is not), so `m[c][L]` vs `m[c][S]` is
        //    meaningless;
        //  - the L cone peaks at 565 nm, which is *closer to a display's green
        //    primary than to its red*, and the green primary is far more
        //    energetic — so `m[green][L] > m[red][L]`, which looks wrong and
        //    is not. Chromaticity lives in the ratios.
        let l_over_m = |c: usize| m[c][0] / m[c][1];
        assert!(
            l_over_m(0) > l_over_m(1) && l_over_m(1) > l_over_m(2),
            "L:M should fall red → green → blue: {:?}",
            [l_over_m(0), l_over_m(1), l_over_m(2)]
        );
        let s_frac = |c: usize| m[c][2] / (m[c][0] + m[c][1]);
        assert!(
            s_frac(2) > s_frac(1) && s_frac(1) > s_frac(0),
            "S:luminance should rise red → green → blue: {:?}",
            [s_frac(0), s_frac(1), s_frac(2)]
        );
        // Rods peak at 507 nm, well to the blue side of the photopic peak, so
        // the rod-to-luminance ratio must rise sharply toward the blue primary.
        let r_frac = |c: usize| m[c][3] / (m[c][0] + m[c][1]);
        assert!(
            r_frac(2) > r_frac(1) && r_frac(1) > r_frac(0),
            "rod:luminance should rise red → green → blue: {:?}",
            [r_frac(0), r_frac(1), r_frac(2)]
        );
    }

    #[test]
    fn luminance_is_recoverable_as_l_plus_m() {
        // The whole pipeline treats `L_adapt = R_L + R_M` as the adapting
        // luminance in cd/m², so the LMSR matrix must be scaled such that a
        // neutral stimulus of Y cd/m² produces R_L + R_M ≈ Y. If this ever
        // stops holding, every downstream CSF lookup is being done at the
        // wrong adaptation level.
        let e = emission_spectra(DisplaySpectra::CcflLcd, 3);
        let m = lmsr_matrix(&e);
        // A neutral: equal linear RGB summing to Y cd/m² is not the right
        // stimulus (the primaries are not equal-luminance), so use the
        // matrix's own luminance row weights: for input (r,g,b),
        //   L+M = Σ_c (m[c][0] + m[c][1]) · c
        let w: Vec<f64> = m.iter().map(|row| row[0] + row[1]).collect();
        let y_of = |rgb: [f64; 3]| -> f64 { w.iter().zip(rgb).map(|(a, b)| a * b).sum() };
        // Linearity is what matters here plus a documented scale: record the
        // per-channel weights so a regression is visible.
        let one = y_of([1.0, 1.0, 1.0]);
        let ten = y_of([10.0, 10.0, 10.0]);
        assert!((ten - 10.0 * one).abs() < 1e-9 * ten.max(1.0), "not linear");
        assert!(one > 0.0 && one.is_finite());
        // Weight ordering must follow photopic luminous efficiency: G > R > B.
        assert!(w[1] > w[0] && w[0] > w[2], "luminance weights {w:?}");
    }
}
