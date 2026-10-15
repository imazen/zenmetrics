//! Independent scipy ppf/sf fixtures include all nine Holm tail levels.
#[path = "data/student_reference.rs"]
mod student_reference_data;
use zenmetrics_stats::{student_t_quantile, student_t_upper_tail};

#[test]
fn quantiles_and_upper_tails_match_scipy_at_holm_levels() {
    for &(df, p, q, sf) in student_reference_data::QUANTILES {
        let got = student_t_quantile(p, df);
        assert!(
            (got / q - 1.0).abs() < 1e-10,
            "ppf({p}, {df}): {got} != {q}"
        );
        let got = student_t_upper_tail(q, df);
        assert!(
            (got / sf - 1.0).abs() < 1e-10,
            "sf({q}, {df}): {got} != {sf}"
        );
    }
    for &(df, x, sf) in student_reference_data::TAILS {
        let got = student_t_upper_tail(x, df);
        assert!(
            (got / sf - 1.0).abs() < 1e-10,
            "sf({x}, {df}): {got} != {sf}"
        );
    }
}

#[test]
fn nonfinite_and_degenerate_arguments_are_explicit() {
    for df in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
        assert!(student_t_quantile(0.95, df).is_nan());
        assert!(student_t_upper_tail(1.0, df).is_nan());
    }
    for p in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -1.0,
        0.0,
        1.0,
        2.0,
    ] {
        assert!(student_t_quantile(p, 9.0).is_nan());
    }
    assert!(student_t_upper_tail(f64::NAN, 9.0).is_nan());
    assert_eq!(student_t_upper_tail(f64::INFINITY, 9.0), 0.0);
    assert_eq!(student_t_upper_tail(f64::NEG_INFINITY, 9.0), 1.0);
    assert_eq!(student_t_upper_tail(0.0, 9.0), 0.5);
    assert_eq!(student_t_upper_tail(-0.0, 9.0), 0.5);
    assert!(student_t_quantile(0.5, 9.0).abs() < 1e-12);
    // Direct survival calculation keeps tails that 1-CDF would round to zero.
    assert!(student_t_upper_tail(100.0, 9.0) > 0.0);
}
