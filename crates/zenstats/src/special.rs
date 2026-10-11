//! Special functions with relative (not absolute) accuracy in the tails.

/// Complementary error function, Chebyshev fit with fractional error below
/// 1.2e-7 everywhere (Press et al., *Numerical Recipes*, `erfcc`).
pub(crate) fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398
                                + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let ans = t * poly.exp();
    if x >= 0.0 { ans } else { 2.0 - ans }
}

/// Standard normal CDF Pr(Z ≤ z) with small relative error in the lower tail.
/// For the upper tail call it with `-z` rather than computing `1 - norm_cdf(z)`.
pub(crate) fn norm_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    0.5 * erfc(-z / core::f64::consts::SQRT_2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_values_with_relative_accuracy() {
        // Reference values: Φ(z) from the exact erfc (Abramowitz & Stegun tables / mpmath).
        let cases = [
            (0.0, 0.5),
            (1.0, 0.8413447460685429),
            (-1.0, 0.15865525393145707),
            (0.6744897501960817, 0.75),
            (-3.0, 0.0013498980316300946),
            (-6.0, 9.865876450376946e-10),
            (-8.0, 6.220960574271785e-16),
        ];
        for (z, want) in cases {
            let got = norm_cdf(z);
            assert!(((got - want) / want).abs() < 2e-7, "z={z}: {got} vs {want}");
        }
    }
}
