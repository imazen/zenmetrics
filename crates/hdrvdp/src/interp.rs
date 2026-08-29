//! Interpolation and quadrature helpers.
//!
//! These reproduce the *semantics* the reference relies on — MATLAB's
//! `interp1` (linear, and `'cubic'` which is PCHIP), `cumtrapz`, and
//! `matlabPyrTools`' `pointOp` LUT lookup — written from their documented
//! behaviour rather than transcribed from any source.

/// Clamp `x` to `[lo, hi]`. `NaN` maps to `lo` (matching how the reference's
/// `clamp` behaves on the clamped-log paths, where a NaN would otherwise
/// poison the LUT index).
#[must_use]
#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x.is_nan() || x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Linear interpolation of `(xs, ys)` at `x`, clamping to the end values
/// outside the sample range.
///
/// `xs` must be strictly increasing and the same length as `ys` (≥ 2).
#[must_use]
pub fn interp1_linear(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    debug_assert_eq!(xs.len(), ys.len());
    debug_assert!(xs.len() >= 2);
    if x <= xs[0] {
        return ys[0];
    }
    let n = xs.len();
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // Binary search for the interval containing x.
    let i = match xs.binary_search_by(|probe| probe.partial_cmp(&x).expect("finite grid")) {
        Ok(i) => return ys[i],
        Err(i) => i - 1,
    };
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

/// Look `x` up in a LUT sampled on a uniform grid starting at `x0` with step
/// `dx`, linearly interpolating and clamping at both ends.
///
/// This is the `pointOp(img, lut, x0, dx, 0)` shape the reference uses to map
/// log-luminance to the photoreceptor JND space.
#[must_use]
#[inline]
pub fn point_op(lut: &[f64], x0: f64, dx: f64, x: f64) -> f64 {
    debug_assert!(lut.len() >= 2);
    debug_assert!(dx > 0.0);
    let pos = (x - x0) / dx;
    // NaN must land on the low end rather than propagate, so the guard is
    // written as "not definitely above zero" via `partial_cmp`.
    if !matches!(pos.partial_cmp(&0.0), Some(core::cmp::Ordering::Greater)) {
        return lut[0];
    }
    let last = lut.len() - 1;
    if pos >= last as f64 {
        return lut[last];
    }
    let i = pos as usize;
    let t = pos - i as f64;
    lut[i] + t * (lut[i + 1] - lut[i])
}

/// Cumulative trapezoidal integral of `y` over `x`, as MATLAB's `cumtrapz`:
/// the result has the same length as the input and starts at 0.
#[must_use]
pub fn cumtrapz(x: &[f64], y: &[f64]) -> Vec<f64> {
    debug_assert_eq!(x.len(), y.len());
    let mut out = vec![0.0; x.len()];
    let mut acc = 0.0;
    for i in 1..x.len() {
        acc += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
        out[i] = acc;
    }
    out
}

/// Trapezoidal integral of `y` over `x` (MATLAB's `trapz`).
#[must_use]
pub fn trapz(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let mut acc = 0.0;
    for i in 1..x.len() {
        acc += 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1]);
    }
    acc
}

/// Shape-preserving piecewise cubic Hermite interpolation (PCHIP) of
/// `(xs, ys)` at each point of `query`, returning `fill` outside `[xs[0],
/// xs[n-1]]`.
///
/// This is what MATLAB's `interp1(..., 'cubic', fill)` does — `'cubic'` is an
/// alias for `'pchip'`. Slopes follow the Fritsch–Carlson construction
/// (harmonic mean of neighbouring secants, zeroed at local extrema), which is
/// what makes it monotone-preserving: resampling a monotone sensitivity curve
/// cannot introduce ringing, and resampling a non-negative emission spectrum
/// cannot introduce negative lobes. A natural cubic spline would do both.
///
/// `xs` must be strictly increasing with at least 2 points.
#[must_use]
pub fn interp1_pchip(xs: &[f64], ys: &[f64], query: &[f64], fill: f64) -> Vec<f64> {
    assert_eq!(xs.len(), ys.len(), "pchip: x and y length mismatch");
    assert!(xs.len() >= 2, "pchip: need at least 2 samples");
    let n = xs.len();

    // Secant slopes.
    let mut h = vec![0.0; n - 1];
    let mut delta = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = xs[i + 1] - xs[i];
        debug_assert!(h[i] > 0.0, "pchip: x must be strictly increasing");
        delta[i] = (ys[i + 1] - ys[i]) / h[i];
    }

    // Fritsch–Carlson derivatives.
    let mut d = vec![0.0; n];
    if n == 2 {
        d[0] = delta[0];
        d[1] = delta[0];
    } else {
        for i in 1..n - 1 {
            if delta[i - 1] * delta[i] > 0.0 {
                let w1 = 2.0 * h[i] + h[i - 1];
                let w2 = h[i] + 2.0 * h[i - 1];
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            } else {
                d[i] = 0.0;
            }
        }
        d[0] = pchip_end_slope(h[0], h[1], delta[0], delta[1]);
        d[n - 1] = pchip_end_slope(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
    }

    query
        .iter()
        .map(|&x| {
            if x < xs[0] || x > xs[n - 1] {
                return fill;
            }
            let i = match xs.binary_search_by(|p| p.partial_cmp(&x).expect("finite grid")) {
                Ok(i) => return ys[i],
                Err(i) => i - 1,
            };
            // Hermite basis on [xs[i], xs[i+1]].
            let s = x - xs[i];
            let hi = h[i];
            let c = (3.0 * delta[i] - 2.0 * d[i] - d[i + 1]) / hi;
            let b = (d[i] - 2.0 * delta[i] + d[i + 1]) / (hi * hi);
            ys[i] + s * (d[i] + s * (c + s * b))
        })
        .collect()
}

/// One-sided three-point slope estimate at an interval end, shape-limited the
/// way PCHIP requires (never overshooting the adjacent secant, and zeroed if it
/// would point the wrong way).
fn pchip_end_slope(h0: f64, h1: f64, d0: f64, d1: f64) -> f64 {
    let mut d = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if d * d0 <= 0.0 {
        d = 0.0;
    } else if (d0 * d1 <= 0.0) && (d.abs() > (3.0 * d0).abs()) {
        d = 3.0 * d0;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_interp_clamps_and_hits_nodes() {
        let xs = [0.0, 1.0, 2.0];
        let ys = [0.0, 10.0, 30.0];
        assert_eq!(interp1_linear(&xs, &ys, -5.0), 0.0);
        assert_eq!(interp1_linear(&xs, &ys, 5.0), 30.0);
        assert_eq!(interp1_linear(&xs, &ys, 1.0), 10.0);
        assert!((interp1_linear(&xs, &ys, 0.5) - 5.0).abs() < 1e-15);
        assert!((interp1_linear(&xs, &ys, 1.25) - 15.0).abs() < 1e-15);
    }

    #[test]
    fn point_op_matches_uniform_linear_interp() {
        let lut = [0.0, 1.0, 4.0, 9.0];
        // grid: x0 = 2.0, dx = 0.5 → nodes at 2.0, 2.5, 3.0, 3.5
        assert_eq!(point_op(&lut, 2.0, 0.5, 1.0), 0.0);
        assert_eq!(point_op(&lut, 2.0, 0.5, 9.0), 9.0);
        assert!((point_op(&lut, 2.0, 0.5, 2.25) - 0.5).abs() < 1e-15);
        assert!((point_op(&lut, 2.0, 0.5, 3.25) - 6.5).abs() < 1e-15);
        // exactly on the last node, not past it
        assert!((point_op(&lut, 2.0, 0.5, 3.5) - 9.0).abs() < 1e-15);
    }

    #[test]
    fn cumtrapz_of_constant_is_linear() {
        let x: Vec<f64> = (0..5).map(|i| i as f64).collect();
        let y = vec![2.0; 5];
        let c = cumtrapz(&x, &y);
        assert_eq!(c, vec![0.0, 2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn trapz_of_a_line_is_exact() {
        // ∫₀³ x dx = 4.5, and the trapezoid rule is exact for affine functions.
        let x: Vec<f64> = (0..=3).map(|i| i as f64).collect();
        let y = x.clone();
        assert!((trapz(&x, &y) - 4.5).abs() < 1e-15);
    }

    #[test]
    fn pchip_is_exact_on_nodes_and_fills_outside() {
        let xs = [0.0, 1.0, 2.0, 3.0];
        let ys = [0.0, 1.0, 8.0, 27.0];
        let q = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let got = interp1_pchip(&xs, &ys, &q, -99.0);
        assert_eq!(got[0], -99.0);
        assert_eq!(got[5], -99.0);
        assert!((got[1] - 0.0).abs() < 1e-12);
        assert!((got[2] - 1.0).abs() < 1e-12);
        assert!((got[3] - 8.0).abs() < 1e-12);
        assert!((got[4] - 27.0).abs() < 1e-12);
    }

    #[test]
    fn pchip_preserves_monotonicity_and_non_negativity() {
        // A step-like monotone sequence: a natural cubic spline overshoots
        // here (goes below 0 / above 1); PCHIP must not. This is exactly the
        // property we rely on when resampling emission spectra, which must
        // stay non-negative.
        let xs: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let ys = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let q: Vec<f64> = (0..=700).map(|i| i as f64 / 100.0).collect();
        let got = interp1_pchip(&xs, &ys, &q, f64::NAN);
        let mut prev = f64::NEG_INFINITY;
        for (x, v) in q.iter().zip(&got) {
            assert!(
                (-1e-12..=1.0 + 1e-12).contains(v),
                "pchip overshot at x={x}: {v}"
            );
            assert!(*v >= prev - 1e-12, "pchip lost monotonicity at x={x}");
            prev = *v;
        }
    }
}
