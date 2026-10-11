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

/// ln Φ(z), accurate in the far lower tail (asymptotic series below −30,
/// where Φ itself approaches the smallest normal `f64`).
pub(crate) fn ln_norm_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z > -30.0 {
        return norm_cdf(z).ln();
    }
    // Φ(z) = φ(z)/(−z) · (1 − 1/z² + 3/z⁴ − …) for z → −∞
    let z2 = z * z;
    -0.5 * z2 - (-z).ln() - 0.5 * (2.0 * core::f64::consts::PI).ln()
        + (1.0 - 1.0 / z2 + 3.0 / (z2 * z2)).ln()
}

/// The crate's one percentile rule: Hyndman & Fan type 7 (linear
/// interpolation between order statistics at `h = (n − 1)·p`), numpy's
/// default and `bake_verdict`'s. `sorted` must be sorted and non-empty.
pub(crate) fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = (h.floor() as usize).min(n - 1);
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// ln Γ(x) for x > 0, Lanczos approximation (g = 7, 9 terms; relative error
/// below 1e-15 over the range used here).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: Γ(x)Γ(1−x) = π / sin(πx)
        let pi = core::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + 7.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * core::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized incomplete beta I_x(a, b), continued fraction by the modified
/// Lentz method (Press et al., *Numerical Recipes*, `betai`/`betacf`).
fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln();
    let cf = |a: f64, b: f64, x: f64| -> f64 {
        const TINY: f64 = 1e-300;
        let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
        let mut c = 1.0;
        let mut d = 1.0 - qab * x / qap;
        if d.abs() < TINY {
            d = TINY;
        }
        d = 1.0 / d;
        let mut h = d;
        for m in 1..=1000 {
            let m = m as f64;
            let m2 = 2.0 * m;
            let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
            d = 1.0 + aa * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + aa / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            h *= d * c;
            let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
            d = 1.0 + aa * d;
            if d.abs() < TINY {
                d = TINY;
            }
            c = 1.0 + aa / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        h
    };
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * cf(b, a, 1.0 - x) / b
    }
}

/// Student t CDF with `df > 0` degrees of freedom. Near `t = 0` it uses
/// `I_y(1/2, df/2)` with `y = t²/(df + t²)` computed directly, because
/// `df/(df + t²)` rounds to 1 there and the CDF would go flat.
fn student_t_cdf(t: f64, df: f64) -> f64 {
    let t2 = t * t;
    let tail = if t2 < df {
        0.5 - 0.5 * inc_beta(0.5, 0.5 * df, t2 / (df + t2))
    } else {
        0.5 * inc_beta(0.5 * df, 0.5, df / (df + t2))
    };
    if t >= 0.0 { 1.0 - tail } else { tail }
}

/// Student t quantile: the `t` with `Pr(T ≤ t) = p`, for `0 < p < 1` and
/// `df > 0`, by bracketing and bisection on [`student_t_cdf`]. NaN outside
/// that domain.
pub(crate) fn student_t_quantile(p: f64, df: f64) -> f64 {
    if !(p > 0.0 && p < 1.0 && df > 0.0 && df.is_finite()) {
        return f64::NAN;
    }
    let (mut lo, mut hi) = (-1.0, 1.0);
    while student_t_cdf(lo, df) > p {
        lo *= 2.0;
        if lo < -1e300 {
            return f64::NEG_INFINITY;
        }
    }
    while student_t_cdf(hi, df) < p {
        hi *= 2.0;
        if hi > 1e300 {
            return f64::INFINITY;
        }
    }
    for _ in 0..300 {
        let mid = 0.5 * (lo + hi);
        if mid == lo || mid == hi {
            break;
        }
        if student_t_cdf(mid, df) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
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

    #[test]
    fn ln_norm_cdf_is_continuous_at_the_switch_and_finite_far_out() {
        let below = ln_norm_cdf(-30.000001);
        let above = ln_norm_cdf(-29.999999);
        assert!((below - above).abs() < 1e-3, "{below} {above}");
        // ln Φ(−40) from the exact erfc: −804.608442013754
        assert!((ln_norm_cdf(-40.0) + 804.608442013754).abs() < 1e-6);
        assert!(ln_norm_cdf(-1e6).is_finite());
    }

    /// scipy 1.18.1 `stats.t.ppf(p, df)`.
    #[test]
    fn student_t_quantile_matches_scipy() {
        let cases = [
            (0.975, 1.0, 12.706204736174694),
            (0.975, 2.0, 4.302652729749462),
            (0.975, 3.0, 3.1824463052837078),
            (0.975, 5.0, 2.5705818356363146),
            (0.975, 11.0, 2.200985160091639),
            (0.975, 30.0, 2.0422724563012378),
            (0.975, 69.0, 1.9949454151072374),
            (0.95, 4.0, 2.1318467863266495),
            (0.995, 7.0, 3.4994832973504924),
            (0.6, 2.5, 0.2814595127485477),
            (0.025, 9.0, -2.2621571627982053),
            (0.9995, 1.0, 636.6192487687897),
        ];
        for (p, df, want) in cases {
            let got = student_t_quantile(p, df);
            assert!(
                ((got - want) / want).abs() < 1e-10,
                "p={p} df={df}: {got} vs scipy {want}"
            );
        }
        assert!(student_t_quantile(0.0, 3.0).is_nan());
        assert!(student_t_quantile(0.5, 0.0).is_nan());
        let m = student_t_quantile(0.5, 4.0);
        assert!(m.abs() < 1e-12, "median {m:e}");
    }

    /// numpy.percentile (default linear) on the same vector.
    #[test]
    fn quantile7_matches_numpy_percentile() {
        let mut x = [
            0.31, -1.2, 4.4, 0.05, 2.2, -0.7, 1.9, 3.3, 0.0, -2.5, 1.1, 0.6,
        ];
        x.sort_by(f64::total_cmp);
        assert!((quantile7(&x, 0.025) - -2.1425).abs() < 1e-12);
        assert!((quantile7(&x, 0.975) - 4.0975).abs() < 1e-12);
        assert!((quantile7(&x, 0.5) - 0.455).abs() < 1e-12);
    }
}
