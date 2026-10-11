//! zenstats' evaluation modules against values computed independently with
//! scipy, scikit-learn and statsmodels (`scripts/independent_reference.py`,
//! synthetic data; regenerate with the command in that script).

mod independent_reference_data;
use independent_reference_data as r;

use zenstats::jnd::{MapFamily, Shape, excess_error_sd, fit_map, out_of_fold};
use zenstats::pairs::{HumanNoise, PairInput, expected_agreement};
use zenstats::panel::Orientation;

#[test]
fn knee_power_fit_is_at_least_as_good_as_scipy_curve_fit() {
    let m = fit_map(MapFamily::KneePower(Shape::Quality), &r::X, &r::Y).unwrap();
    let sse: f64 = r::X
        .iter()
        .zip(&r::Y)
        .map(|(&x, &y)| (m.predict(x) - y).powi(2))
        .sum();
    assert!(
        sse <= r::KNEE_SSE * (1.0 + 1e-6),
        "zenstats {sse} vs scipy {}",
        r::KNEE_SSE
    );
    let p = m.params();
    for (got, want) in p.iter().zip(r::KNEE_PARAMS) {
        assert!(
            ((got - want) / want).abs() < 1e-3,
            "params {p:?} vs scipy {:?}",
            r::KNEE_PARAMS
        );
    }
}

#[test]
fn isotonic_matches_scikit_learn_including_interpolation_and_clipping() {
    let m = fit_map(MapFamily::Isotonic(Shape::Quality), &r::X, &r::Y).unwrap();
    for (&x, &want) in r::ISO_PROBE.iter().zip(&r::ISO_AT_PROBE) {
        let got = m.predict(x);
        assert!((got - want).abs() < 1e-12, "x={x}: {got} vs sklearn {want}");
    }
}

#[test]
fn leave_one_source_out_linear_matches_numpy_polyfit() {
    let oof = out_of_fold(MapFamily::Linear, &r::X, &r::Y, &r::SOURCE).unwrap();
    for (i, (&got, &want)) in oof.iter().zip(&r::OOF_LINEAR).enumerate() {
        assert!((got - want).abs() < 1e-9, "row {i}: {got} vs numpy {want}");
    }
}

#[test]
fn excess_error_matches_scipy_likelihood_maximum() {
    let e = excess_error_sd(&r::OOF_LINEAR, &r::Y, &r::SIGMA).unwrap();
    assert!((e.moment - r::TAU_MOMENT).abs() < 1e-12, "{e:?}");
    assert!(
        (e.ml - r::TAU_ML).abs() < 1e-6,
        "{e:?} vs scipy {}",
        r::TAU_ML
    );
}

#[test]
fn holm_matches_statsmodels() {
    let got = zenstats::multiple::holm(&r::HOLM_P);
    for (g, w) in got.iter().zip(r::HOLM_ADJ) {
        assert!((g - w).abs() < 1e-15, "{got:?} vs {:?}", r::HOLM_ADJ);
    }
}

#[test]
fn expected_observer_agreement_matches_brute_force_scipy() {
    let input = PairInput {
        predicted: &r::X,
        target: &r::Y,
        source: &r::SOURCE,
        codec: &r::CODEC,
        weights: None,
        orientation: Orientation::LowerIsBetter,
    };
    let got = expected_agreement(&input, HumanNoise::JND_75).unwrap();
    let classes = [
        got.same_source_same_codec,
        got.same_source_cross_codec,
        got.cross_source,
    ];
    for (k, c) in classes.iter().enumerate() {
        let (agree, ceil) = (
            r::EXPECTED_AGREEMENT[2 * k],
            r::EXPECTED_AGREEMENT[2 * k + 1],
        );
        // scipy's norm.cdf is exact to double precision; zenstats' erfc fit has
        // relative error below 1.2e-7
        assert!(
            (c.agreement - agree).abs() < 2e-7,
            "class {k}: {} vs {agree}",
            c.agreement
        );
        assert!(
            (c.ceiling - ceil).abs() < 2e-7,
            "class {k}: {} vs {ceil}",
            c.ceiling
        );
    }
}
