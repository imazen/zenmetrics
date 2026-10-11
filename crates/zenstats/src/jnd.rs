//! Accuracy on an absolute scale (JND or any interval target), with the
//! mapping from metric units declared and cross-validated.
//!
//! Most metrics do not output the target's units, so absolute accuracy is
//! the accuracy of *metric + mapping*. Which mapping family is fitted can
//! change an error criterion more than the metric does, so this module makes
//! the family an explicit, declared choice ([`MapFamily`]) and provides
//! out-of-fold predictions grouped by source ([`out_of_fold`]): each source's
//! rows are predicted by a map fitted on the other sources, so no map is ever
//! scored on the rows it was fitted to (Varma & Simon 2006,
//! doi:10.1186/1471-2105-7-91; Roberts et al. 2017, doi:10.1111/ecog.02881).
//! Statistics should be computed on the pooled out-of-fold predictions, with
//! a source-clustered interval from [`crate::resample`]; per-fold intervals
//! under-cover (Bates, Hastie & Tibshirani 2024, doi:10.1080/01621459.2023.2197686).
//!
//! Error criteria:
//!
//! * [`rmse`]: root mean squared error in target units.
//! * [`excess_error_sd`]: τ̂, the standard deviation of the prediction error
//!   left after removing the target's own measurement noise, from the model
//!   rᵢ ~ N(0, σᵢ² + τ²) (moment and maximum-likelihood forms; DerSimonian &
//!   Laird 1986, doi:10.1016/0197-2456(86)90046-2). Unlike an error divided
//!   by σᵢ, it does not grow as a study adds observers and σᵢ shrinks, and it
//!   does not let the few stimuli with the smallest σ dominate.
//! * [`rmse_star`]: ITU-T P.1401 §7.7 epsilon-insensitive RMSE, which only
//!   counts error beyond each stimulus' 95 % confidence half-width.
//!
//! The normalised error `(pred − target) / σ` (`crate::z_rmse_per_sample`)
//! remains useful as a goodness-of-fit diagnostic; it is weighted by 1/σ²,
//! so on data whose σ shrinks toward zero distortion it mostly scores the
//! near-threshold stimuli.

use crate::panel::{fit_logistic4, logistic_eval};

/// Which way the metric's raw score runs relative to the target distortion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// Higher score = better quality = smaller target distortion (SSIMULACRA2, PSNR).
    Quality,
    /// Higher score = more distortion (Butteraugli, DSSIM, a JND estimate).
    Distance,
}

/// A mapping family from metric score to target units. Declare it before
/// looking at the evaluation data (ITU-T P.1401 §7.3.3 requires it stated).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapFamily {
    /// `y = a + b·x` by least squares (P.1401 Appendix II first-order map).
    Linear,
    /// 4-parameter logistic, the same fit as [`crate::rescale_logistic`].
    Logistic4,
    /// Power law with a knee, the form the JPEG AIC common test conditions use
    /// to put metrics on the JND scale: `y = a·max(0, b − x)^c` for
    /// [`Shape::Quality`] and `y = a·max(0, x − b)^c` for [`Shape::Distance`],
    /// with `a ≥ 0`, `c > 0`.
    KneePower(Shape),
    /// Monotone least squares (pool-adjacent-violators), decreasing for
    /// [`Shape::Quality`] and increasing for [`Shape::Distance`]; predictions
    /// between fitted points are linearly interpolated and clamped at the
    /// ends. The best monotone shape, so in-sample it flatters every metric;
    /// use it out of fold.
    Isotonic(Shape),
}

impl MapFamily {
    /// Free parameters, for P.1401's `N − d` denominator. Isotonic regression
    /// has no fixed count; this returns the number of fitted blocks when
    /// called on a [`FittedMap`] (see [`FittedMap::n_params`]).
    fn fixed_params(self) -> Option<usize> {
        match self {
            MapFamily::Linear => Some(2),
            MapFamily::Logistic4 => Some(4),
            MapFamily::KneePower(_) => Some(3),
            MapFamily::Isotonic(_) => None,
        }
    }
}

/// Why a fit or criterion could not be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MapError {
    /// Slices differ in length; names the offender.
    Length(&'static str),
    /// A value is non-finite (or a σ / half-width is negative).
    NonFinite(&'static str),
    /// Too few rows for the family.
    TooFew { rows: usize, need: usize },
    /// Fewer than two groups for out-of-fold prediction.
    TooFewGroups(usize),
    /// The optimiser did not produce a finite fit.
    FitFailed,
    /// An out-of-fold fit failed for the fold holding this group label.
    FoldFailed(u32),
}

impl core::fmt::Display for MapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length(w) => write!(f, "`{w}` has a different length"),
            Self::NonFinite(w) => write!(f, "`{w}` holds a non-finite or negative value"),
            Self::TooFew { rows, need } => write!(f, "{rows} rows; the family needs {need}"),
            Self::TooFewGroups(k) => write!(f, "{k} group(s); out-of-fold needs at least 2"),
            Self::FitFailed => write!(f, "the mapping fit did not converge to a finite solution"),
            Self::FoldFailed(g) => write!(f, "the fit with group {g} held out failed"),
        }
    }
}

impl std::error::Error for MapError {}

/// A fitted map.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedMap {
    family: MapFamily,
    params: Vec<f64>,
    /// Isotonic knots: strictly increasing x and fitted y.
    knots: Vec<(f64, f64)>,
    blocks: usize,
}

impl FittedMap {
    /// The family this map was fitted from.
    pub fn family(&self) -> MapFamily {
        self.family
    }

    /// Fitted parameters: `[a, b]` (linear), the four logistic parameters,
    /// `[a, b, c]` (knee power), empty for isotonic.
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Parameter count for P.1401's `N − d`; for isotonic, the number of
    /// constant blocks the fit produced.
    pub fn n_params(&self) -> usize {
        self.family.fixed_params().unwrap_or(self.blocks)
    }

    /// Map one metric score to target units.
    pub fn predict(&self, x: f64) -> f64 {
        match self.family {
            MapFamily::Linear => self.params[0] + self.params[1] * x,
            MapFamily::Logistic4 => {
                let b = [
                    self.params[0],
                    self.params[1],
                    self.params[2],
                    self.params[3],
                ];
                logistic_eval(&b, x)
            }
            MapFamily::KneePower(shape) => {
                knee_eval(shape, self.params[0], self.params[1], self.params[2], x)
            }
            MapFamily::Isotonic(_) => interp(&self.knots, x),
        }
    }

    /// Map many scores.
    pub fn predict_all(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| self.predict(v)).collect()
    }
}

fn check_xy(x: &[f64], y: &[f64]) -> Result<(), MapError> {
    if x.len() != y.len() {
        return Err(MapError::Length("y"));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(MapError::NonFinite("x"));
    }
    if !y.iter().all(|v| v.is_finite()) {
        return Err(MapError::NonFinite("y"));
    }
    Ok(())
}

/// Fit `family` to scores `x` and targets `y` by least squares.
pub fn fit_map(family: MapFamily, x: &[f64], y: &[f64]) -> Result<FittedMap, MapError> {
    check_xy(x, y)?;
    let need = family.fixed_params().unwrap_or(2);
    if x.len() < need.max(2) {
        return Err(MapError::TooFew {
            rows: x.len(),
            need,
        });
    }
    let (params, knots, blocks) = match family {
        MapFamily::Linear => (fit_linear(x, y).to_vec(), Vec::new(), 0),
        MapFamily::Logistic4 => {
            let b = fit_logistic4(x, y).ok_or(MapError::FitFailed)?;
            (b.to_vec(), Vec::new(), 0)
        }
        MapFamily::KneePower(shape) => (fit_knee(shape, x, y)?.to_vec(), Vec::new(), 0),
        MapFamily::Isotonic(shape) => {
            let (knots, blocks) = fit_isotonic(shape, x, y);
            (Vec::new(), knots, blocks)
        }
    };
    let map = FittedMap {
        family,
        params,
        knots,
        blocks,
    };
    if x.iter().any(|&v| !map.predict(v).is_finite()) {
        return Err(MapError::FitFailed);
    }
    Ok(map)
}

/// Leave-one-group-out predictions: each row is predicted by a map fitted on
/// every row whose group differs. `groups` is usually the source image.
pub fn out_of_fold(
    family: MapFamily,
    x: &[f64],
    y: &[f64],
    groups: &[u32],
) -> Result<Vec<f64>, MapError> {
    check_xy(x, y)?;
    if groups.len() != x.len() {
        return Err(MapError::Length("groups"));
    }
    let mut labels: Vec<u32> = groups.to_vec();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 {
        return Err(MapError::TooFewGroups(labels.len()));
    }
    let mut out = vec![f64::NAN; x.len()];
    for &g in &labels {
        let (mut fx, mut fy) = (Vec::new(), Vec::new());
        for i in 0..x.len() {
            if groups[i] != g {
                fx.push(x[i]);
                fy.push(y[i]);
            }
        }
        let map = fit_map(family, &fx, &fy).map_err(|_| MapError::FoldFailed(g))?;
        for i in 0..x.len() {
            if groups[i] == g {
                out[i] = map.predict(x[i]);
            }
        }
    }
    Ok(out)
}

/// Root mean squared error.
pub fn rmse(predicted: &[f64], target: &[f64]) -> Result<f64, MapError> {
    check_xy(predicted, target)?;
    if predicted.is_empty() {
        return Err(MapError::TooFew { rows: 0, need: 1 });
    }
    let s: f64 = predicted
        .iter()
        .zip(target)
        .map(|(p, t)| (p - t) * (p - t))
        .sum();
    Ok((s / predicted.len() as f64).sqrt())
}

/// ITU-T P.1401 §7.7 epsilon-insensitive RMSE:
/// `sqrt( Σ max(0, |pᵢ − tᵢ| − ci95ᵢ)² / (N − d) )`, where `ci95ᵢ` is the
/// 95 % confidence half-width of target i and `d` the mapping's parameter
/// count (0 for a metric scored on its own output).
pub fn rmse_star(
    predicted: &[f64],
    target: &[f64],
    ci95_half: &[f64],
    d: usize,
) -> Result<f64, MapError> {
    check_xy(predicted, target)?;
    if ci95_half.len() != target.len() {
        return Err(MapError::Length("ci95_half"));
    }
    if !ci95_half.iter().all(|c| c.is_finite() && *c >= 0.0) {
        return Err(MapError::NonFinite("ci95_half"));
    }
    let n = predicted.len();
    if n <= d {
        return Err(MapError::TooFew {
            rows: n,
            need: d + 1,
        });
    }
    let s: f64 = (0..n)
        .map(|i| {
            let e = ((predicted[i] - target[i]).abs() - ci95_half[i]).max(0.0);
            e * e
        })
        .sum();
    Ok((s / (n - d) as f64).sqrt())
}

/// τ̂ in both estimators.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExcessError {
    /// `sqrt(max(0, mean r² − mean σ²))`.
    pub moment: f64,
    /// Maximum-likelihood τ under rᵢ ~ N(0, σᵢ² + τ²).
    pub ml: f64,
}

/// τ̂: prediction-error standard deviation beyond the target's measurement
/// noise. `sigma` is each target's measurement standard deviation (for a
/// symmetric 95 % interval, half-width / 1.96).
///
/// The model also absorbs misfit of the target reconstruction that is common
/// to every metric, so τ̂ is an upper bound on the metric's own error.
pub fn excess_error_sd(
    predicted: &[f64],
    target: &[f64],
    sigma: &[f64],
) -> Result<ExcessError, MapError> {
    check_xy(predicted, target)?;
    if sigma.len() != target.len() {
        return Err(MapError::Length("sigma"));
    }
    if !sigma.iter().all(|s| s.is_finite() && *s >= 0.0) {
        return Err(MapError::NonFinite("sigma"));
    }
    let n = predicted.len();
    if n == 0 {
        return Err(MapError::TooFew { rows: 0, need: 1 });
    }
    let r2: Vec<f64> = predicted
        .iter()
        .zip(target)
        .map(|(p, t)| (p - t) * (p - t))
        .collect();
    let s2: Vec<f64> = sigma.iter().map(|s| s * s).collect();
    let mean_r2 = r2.iter().sum::<f64>() / n as f64;
    let mean_s2 = s2.iter().sum::<f64>() / n as f64;
    let moment = (mean_r2 - mean_s2).max(0.0).sqrt();

    // d/dτ² of the log-likelihood, up to a factor −½:
    // g(v) = Σ [ rᵢ²/(σᵢ²+v)² − 1/(σᵢ²+v) ]; the ML v solves g(v) = 0.
    let g = |v: f64| -> f64 {
        r2.iter()
            .zip(&s2)
            .map(|(r, s)| {
                let d = s + v;
                if d > 0.0 { r / (d * d) - 1.0 / d } else { 0.0 }
            })
            .sum()
    };
    // With a zero σ and a zero residual the likelihood is unbounded at v = 0;
    // fall back to the boundary like any other non-positive slope there.
    let g0 = g(f64::MIN_POSITIVE);
    let ml = if !g0.is_finite() || g0 <= 0.0 {
        if g0.is_finite() { 0.0 } else { moment }
    } else {
        let max_r2 = r2.iter().cloned().fold(0.0, f64::max);
        let mut lo = 0.0f64;
        let mut hi = max_r2.max(mean_r2).max(1e-300) * 2.0;
        while g(hi) > 0.0 {
            hi *= 2.0;
        }
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if g(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (0.5 * (lo + hi)).sqrt()
    };
    Ok(ExcessError { moment, ml })
}

// ---------------------------------------------------------------- fits

fn fit_linear(x: &[f64], y: &[f64]) -> [f64; 2] {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
    }
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    [my - b * mx, b]
}

fn knee_eval(shape: Shape, a: f64, b: f64, c: f64, x: f64) -> f64 {
    let u = match shape {
        Shape::Quality => b - x,
        Shape::Distance => x - b,
    };
    if u <= 0.0 { 0.0 } else { a * u.powf(c) }
}

/// Profile out `a` for fixed (b, c): least-squares `a = Σ y·u / Σ u²`, clamped at 0.
fn knee_profile(shape: Shape, b: f64, c: f64, x: &[f64], y: &[f64]) -> (f64, f64) {
    let (mut yu, mut uu) = (0.0, 0.0);
    let us: Vec<f64> = x
        .iter()
        .map(|&xi| knee_eval(shape, 1.0, b, c, xi))
        .collect();
    for (u, t) in us.iter().zip(y) {
        yu += u * t;
        uu += u * u;
    }
    let a = if uu > 0.0 { (yu / uu).max(0.0) } else { 0.0 };
    let sse = us.iter().zip(y).map(|(u, t)| (a * u - t).powi(2)).sum();
    (a, sse)
}

fn fit_knee(shape: Shape, x: &[f64], y: &[f64]) -> Result<[f64; 3], MapError> {
    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (xmax - xmin).max(f64::EPSILON);
    // the knee sits past the data's best end; start at several distances from it
    let edge = match shape {
        Shape::Quality => xmax,
        Shape::Distance => xmin,
    };
    let dir = match shape {
        Shape::Quality => 1.0,
        Shape::Distance => -1.0,
    };
    let cost = |p: &[f64; 2]| -> f64 {
        let (_, sse) = knee_profile(shape, p[0], p[1].exp(), x, y);
        if sse.is_finite() { sse } else { f64::INFINITY }
    };
    let mut best: Option<([f64; 2], f64)> = None;
    for off in [0.0, 0.02, 0.1, 0.3, 1.0] {
        for c0 in [0.5f64, 1.0, 2.0] {
            let start = [edge + dir * off * range, c0.ln()];
            let step = [0.1 * range, 0.3];
            let (p, f) = nelder_mead_2d(&cost, start, step);
            if f.is_finite() && best.is_none_or(|(_, bf)| f < bf) {
                best = Some((p, f));
            }
        }
    }
    let (p, _) = best.ok_or(MapError::FitFailed)?;
    let c = p[1].exp();
    let (a, _) = knee_profile(shape, p[0], c, x, y);
    if !(a.is_finite() && p[0].is_finite() && c.is_finite()) {
        return Err(MapError::FitFailed);
    }
    Ok([a, p[0], c])
}

/// Nelder–Mead in two dimensions with the standard coefficients.
fn nelder_mead_2d<F: Fn(&[f64; 2]) -> f64>(
    f: &F,
    start: [f64; 2],
    step: [f64; 2],
) -> ([f64; 2], f64) {
    let mut s = [
        start,
        [start[0] + step[0], start[1]],
        [start[0], start[1] + step[1]],
    ];
    let mut fs = [f(&s[0]), f(&s[1]), f(&s[2])];
    for _ in 0..4000 {
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| fs[a].total_cmp(&fs[b]));
        s = [s[order[0]], s[order[1]], s[order[2]]];
        fs = [fs[order[0]], fs[order[1]], fs[order[2]]];
        let spread = (fs[2] - fs[0]).abs();
        if spread <= 1e-14 * (fs[0].abs() + 1e-300) || spread == 0.0 {
            let size = (s[2][0] - s[0][0]).abs() + (s[2][1] - s[0][1]).abs();
            if size < 1e-12 * (1.0 + s[0][0].abs()) {
                break;
            }
        }
        let c = [(s[0][0] + s[1][0]) / 2.0, (s[0][1] + s[1][1]) / 2.0];
        let lerp = |t: f64| [c[0] + t * (s[2][0] - c[0]), c[1] + t * (s[2][1] - c[1])];
        let r = lerp(-1.0);
        let fr = f(&r);
        if fr < fs[0] {
            let e = lerp(-2.0);
            let fe = f(&e);
            if fe < fr {
                s[2] = e;
                fs[2] = fe;
            } else {
                s[2] = r;
                fs[2] = fr;
            }
        } else if fr < fs[1] {
            s[2] = r;
            fs[2] = fr;
        } else {
            let k = if fr < fs[2] { lerp(-0.5) } else { lerp(0.5) };
            let fk = f(&k);
            if fk < fs[2].min(fr) {
                s[2] = k;
                fs[2] = fk;
            } else {
                for i in 1..3 {
                    s[i] = [(s[0][0] + s[i][0]) / 2.0, (s[0][1] + s[i][1]) / 2.0];
                    fs[i] = f(&s[i]);
                }
            }
        }
    }
    let mut bi = 0;
    for i in 1..3 {
        if fs[i] < fs[bi] {
            bi = i;
        }
    }
    (s[bi], fs[bi])
}

/// Weighted pool-adjacent-violators on x-sorted data with x ties averaged
/// first. Returns knots (unique x, fitted y) and the number of blocks.
fn fit_isotonic(shape: Shape, x: &[f64], y: &[f64]) -> (Vec<(f64, f64)>, usize) {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    // unique-x points with weight = count
    let mut pts: Vec<(f64, f64, f64)> = Vec::new(); // (x, mean y, weight)
    for &i in &idx {
        match pts.last_mut() {
            Some(last) if last.0 == x[i] => {
                last.1 = (last.1 * last.2 + y[i]) / (last.2 + 1.0);
                last.2 += 1.0;
            }
            _ => pts.push((x[i], y[i], 1.0)),
        }
    }
    let sign = match shape {
        Shape::Distance => 1.0,
        Shape::Quality => -1.0,
    };
    // PAVA for a non-decreasing fit of sign·y
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new(); // (mean, weight, n points)
    for p in &pts {
        blocks.push((sign * p.1, p.2, 1));
        while blocks.len() > 1 {
            let k = blocks.len();
            if blocks[k - 2].0 > blocks[k - 1].0 {
                let (m2, w2, n2) = blocks.pop().unwrap();
                let (m1, w1, n1) = blocks.pop().unwrap();
                blocks.push(((m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, n1 + n2));
            } else {
                break;
            }
        }
    }
    let mut knots = Vec::with_capacity(pts.len());
    let mut k = 0;
    for (m, _, n) in &blocks {
        for _ in 0..*n {
            knots.push((pts[k].0, sign * m));
            k += 1;
        }
    }
    (knots, blocks.len())
}

fn interp(knots: &[(f64, f64)], x: f64) -> f64 {
    if knots.is_empty() {
        return f64::NAN;
    }
    if x <= knots[0].0 {
        return knots[0].1;
    }
    let last = knots[knots.len() - 1];
    if x >= last.0 {
        return last.1;
    }
    let j = knots.partition_point(|k| k.0 <= x);
    let (x0, y0) = knots[j - 1];
    let (x1, y1) = knots[j];
    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knee_power_recovers_its_own_curve() {
        // AIC-style: JND = 0.09·max(0, 93 − x)^0.94 for a quality score x
        let x: Vec<f64> = (0..60).map(|i| 40.0 + i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| knee_eval(Shape::Quality, 0.09, 93.0, 0.94, v))
            .collect();
        let m = fit_map(MapFamily::KneePower(Shape::Quality), &x, &y).unwrap();
        let p = m.params();
        assert!(
            (p[0] - 0.09).abs() < 1e-4 && (p[1] - 93.0).abs() < 1e-2 && (p[2] - 0.94).abs() < 1e-3,
            "{p:?}"
        );
        for (&xi, &yi) in x.iter().zip(&y) {
            assert!((m.predict(xi) - yi).abs() < 1e-4);
        }
    }

    #[test]
    fn knee_power_distance_shape() {
        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| knee_eval(Shape::Distance, 1.5, 0.3, 1.2, v))
            .collect();
        let m = fit_map(MapFamily::KneePower(Shape::Distance), &x, &y).unwrap();
        for (&xi, &yi) in x.iter().zip(&y) {
            assert!((m.predict(xi) - yi).abs() < 1e-3, "{:?}", m.params());
        }
    }

    #[test]
    fn isotonic_is_monotone_and_least_squares_on_a_violation() {
        // decreasing (Quality) target with one violation pooled
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [4.0, 2.0, 3.0, 1.0];
        let m = fit_map(MapFamily::Isotonic(Shape::Quality), &x, &y).unwrap();
        let fit = m.predict_all(&x);
        assert_eq!(fit, vec![4.0, 2.5, 2.5, 1.0]);
        assert_eq!(m.n_params(), 3);
        assert_eq!(m.predict(2.5), 2.5);
        assert_eq!(m.predict(0.0), 4.0);
        assert_eq!(m.predict(9.0), 1.0);
    }

    #[test]
    fn out_of_fold_never_sees_its_own_group() {
        // group 7's targets are wildly off; an in-sample linear fit would bend
        // toward them, an out-of-fold one cannot.
        let x = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 1.5];
        let y = [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 100.0];
        let g = [1u32, 1, 1, 1, 2, 2, 2, 2, 7];
        let oof = out_of_fold(MapFamily::Linear, &x, &y, &g).unwrap();
        assert!((oof[8] - 1.5).abs() < 1e-12, "{}", oof[8]);
        assert_eq!(
            out_of_fold(MapFamily::Linear, &x, &y, &[3u32; 9]),
            Err(MapError::TooFewGroups(1))
        );
    }

    #[test]
    fn logistic_matches_rescale_logistic() {
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .map(|&v| 3.0 / (1.0 + (-(v - 20.0) / 4.0).exp()) + 0.01 * (v % 3.0))
            .collect();
        let m = fit_map(MapFamily::Logistic4, &x, &y).unwrap();
        let r = crate::rescale_logistic(&x, &y);
        for (a, b) in m.predict_all(&x).iter().zip(&r) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn excess_error_is_zero_when_residuals_are_pure_measurement_noise() {
        // residuals exactly ±σ: mean r² = mean σ² → τ̂ = 0 by both estimators
        let t: Vec<f64> = (0..100).map(|i| i as f64 * 0.03).collect();
        let sigma: Vec<f64> = (0..100).map(|i| 0.01 + 0.002 * i as f64).collect();
        let p: Vec<f64> = (0..100)
            .map(|i| t[i] + if i % 2 == 0 { sigma[i] } else { -sigma[i] })
            .collect();
        let e = excess_error_sd(&p, &t, &sigma).unwrap();
        assert!(e.moment < 1e-9, "{e:?}");
        assert!(e.ml < 1e-6, "{e:?}");
    }

    #[test]
    fn excess_error_recovers_a_constant_extra_error() {
        // residuals ±sqrt(σ² + τ²) with τ = 0.25 → both estimators give 0.25
        let sigma: Vec<f64> = (0..200).map(|i| 0.01 + 0.0005 * i as f64).collect();
        let t = vec![1.0; 200];
        let tau = 0.25;
        let p: Vec<f64> = (0..200)
            .map(|i| {
                1.0 + (sigma[i] * sigma[i] + tau * tau).sqrt() * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        let e = excess_error_sd(&p, &t, &sigma).unwrap();
        assert!((e.moment - tau).abs() < 1e-9, "{e:?}");
        assert!((e.ml - tau).abs() < 1e-6, "{e:?}");
    }

    /// Negative control: a normalised error grows as σ shrinks for the same
    /// metric error; τ̂ does not.
    #[test]
    fn excess_error_is_stable_where_normalised_error_explodes() {
        let t = vec![0.0; 100];
        let p: Vec<f64> = (0..100)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        let wide = vec![0.2; 100];
        let narrow = vec![0.01; 100];
        let z_wide = crate::z_rmse_per_sample(&p, &t, &wide);
        let z_narrow = crate::z_rmse_per_sample(&p, &t, &narrow);
        assert!(z_narrow > 10.0 * z_wide);
        let tw = excess_error_sd(&p, &t, &wide).unwrap().moment;
        let tn = excess_error_sd(&p, &t, &narrow).unwrap().moment;
        assert!(tw < tn && tn <= 0.25 + 1e-12 && tw > 0.14, "{tw} {tn}");
    }

    #[test]
    fn rmse_star_counts_only_error_beyond_the_interval() {
        let p = [1.0, 2.0, 3.5];
        let t = [1.1, 2.0, 3.0];
        let ci = [0.2, 0.0, 0.3];
        // excess: 0, 0, 0.2 → sqrt(0.04 / (3 − 1))
        let got = rmse_star(&p, &t, &ci, 1).unwrap();
        assert!((got - (0.04f64 / 2.0).sqrt()).abs() < 1e-12);
        assert!((rmse(&p, &t).unwrap() - ((0.01 + 0.0 + 0.25) / 3.0f64).sqrt()).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_input() {
        assert_eq!(
            fit_map(MapFamily::Linear, &[1.0], &[1.0]),
            Err(MapError::TooFew { rows: 1, need: 2 })
        );
        assert_eq!(
            fit_map(MapFamily::Linear, &[1.0, f64::NAN], &[1.0, 2.0]),
            Err(MapError::NonFinite("x"))
        );
        assert_eq!(
            excess_error_sd(&[1.0], &[1.0], &[-1.0]),
            Err(MapError::NonFinite("sigma"))
        );
    }
}
