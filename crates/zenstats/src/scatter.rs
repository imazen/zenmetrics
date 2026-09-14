//! Geometric diagnostics of paired raw predictions and reference targets.
//!
//! These engineering diagnostics complement the Mohammadi panel. They do not
//! fit a served calibration or choose pass thresholds. All rows must be finite.
//! Quantile mapping averages the target order statistics occupied by a tied
//! prediction group, so input order cannot manufacture a clean scatter.

/// Complete-population scatter diagnostics. Absent normalized quantities mean
/// zero spread; inspect absolute residuals and saturation in that case.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ScatterDiagnostics {
    /// Tie-invariant mapped predictions, in original row order, for plotting.
    pub mapped_prediction: Vec<f64>,
    /// Number of pairs, without dropping or subsampling rows.
    pub n: usize,
    /// Shape residual robust scale, 1.4826 times its median absolute deviation.
    pub shape_mad: f64,
    /// Reference p99 minus p1.
    pub target_span: f64,
    /// Fraction of shape residuals outside the diagonal's ±4 robust-scale band.
    pub outside_envelope: f64,
    /// Absolute shape residual p99 divided by reference span.
    pub shape_p99: Option<f64>,
    /// Largest absolute shape residual divided by reference span.
    pub shape_max: Option<f64>,
    /// Fraction of 20 reference-range bins holding at least 0.5% of pairs.
    pub target_coverage: f64,
    /// Largest reference-bin share.
    pub target_clump: f64,
    /// Coverage using the mapped prediction, on the same reference-range bins.
    pub prediction_coverage: f64,
    /// Largest mapped-prediction bin share.
    pub prediction_clump: f64,
    /// Fraction at the exact raw prediction minimum (not proof of a hard clamp).
    pub floor_mass: f64,
    /// Fraction at the exact raw prediction maximum.
    pub ceiling_mass: f64,
    /// Raw prediction minimum.
    pub prediction_min: f64,
    /// Raw prediction maximum.
    pub prediction_max: f64,
    /// Robust scale of raw OLS(prediction ~ target) residuals.
    pub raw_mad: f64,
    /// Absolute raw OLS residual p99, in served score units.
    pub raw_residual_p99: f64,
    /// Largest absolute raw OLS residual, in served score units.
    pub raw_residual_max: f64,
    /// Raw residual p99 divided by raw robust scale (historical chart-z).
    pub raw_chart_p99: Option<f64>,
    /// Largest absolute raw residual divided by raw robust scale.
    pub raw_chart_max: Option<f64>,
    /// Raw OLS slope; zero for a constant target.
    pub raw_slope: f64,
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    let x = p * (sorted.len() - 1) as f64;
    let lo = x.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] * (1.0 - (x - lo as f64)) + sorted[hi] * (x - lo as f64)
}

fn residual_stats(residual: &[f64]) -> (f64, f64, f64) {
    let mut sorted = residual.to_vec();
    sorted.sort_by(f64::total_cmp);
    let center = quantile(&sorted, 0.5);
    let mut deviations: Vec<_> = residual.iter().map(|r| (r - center).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    let mad = 1.4826 * quantile(&deviations, 0.5);
    let mut abs: Vec<_> = residual.iter().map(|r| r.abs()).collect();
    abs.sort_by(f64::total_cmp);
    (mad, quantile(&abs, 0.99), abs[abs.len() - 1])
}

fn occupancy(values: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let mut bins = [0usize; 20];
    for &v in values {
        let bin = if hi > lo {
            (((v - lo) / (hi - lo)) * 20.0) as usize
        } else {
            10
        };
        bins[bin.min(19)] += 1;
    }
    let threshold = values.len().div_ceil(200);
    (
        bins.iter().filter(|&&n| n >= threshold).count() as f64 / 20.0,
        *bins.iter().max().unwrap() as f64 / values.len() as f64,
    )
}

fn ratio(numerator: f64, denominator: f64) -> Option<f64> {
    (denominator > 0.0)
        .then(|| numerator / denominator)
        .filter(|v| v.is_finite())
}

/// Diagnose scatter without modifying served predictions or calibrating gates.
///
/// Requires at least 50 finite aligned pairs. Polarity is caller-declared: no
/// automatic sign flip hides a globally inverted quality model. Constants and
/// tied predictions remain represented and visibly saturated.
pub fn diagnose(predicted: &[f64], target: &[f64]) -> Result<ScatterDiagnostics, &'static str> {
    let n = predicted.len();
    if n != target.len() || n < 50 {
        return Err("scatter requires at least 50 aligned pairs");
    }
    if predicted.iter().chain(target).any(|v| !v.is_finite()) {
        return Err("nonfinite scatter input");
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by(|&a, &b| predicted[a].total_cmp(&predicted[b]));
    let mut targets = target.to_vec();
    targets.sort_by(f64::total_cmp);
    let mut mapped = vec![0.0; n];
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && predicted[order[start]] == predicted[order[end]] {
            end += 1;
        }
        let mean = targets[start..end]
            .iter()
            .map(|v| v / (end - start) as f64)
            .sum::<f64>();
        for &i in &order[start..end] {
            mapped[i] = mean;
        }
        start = end;
    }
    let residual: Vec<_> = mapped.iter().zip(target).map(|(a, b)| a - b).collect();
    let (shape_mad, shape_p99, shape_max) = residual_stats(&residual);
    let target_span = quantile(&targets, 0.99) - quantile(&targets, 0.01);
    let (target_coverage, target_clump) = occupancy(target, targets[0], targets[n - 1]);
    let (prediction_coverage, prediction_clump) = occupancy(&mapped, targets[0], targets[n - 1]);
    // Centered OLS avoids the unstable sum(x*x)-sum(x)^2 formulation.
    let xmean = target[0]
        + target
            .iter()
            .map(|v| (v - target[0]) / n as f64)
            .sum::<f64>();
    let ymean = predicted[0]
        + predicted
            .iter()
            .map(|v| (v - predicted[0]) / n as f64)
            .sum::<f64>();
    let variance = target.iter().map(|x| (x - xmean).powi(2)).sum::<f64>();
    let cov = target
        .iter()
        .zip(predicted)
        .map(|(x, y)| (x - xmean) * (y - ymean))
        .sum::<f64>();
    let slope = if variance > 0.0 { cov / variance } else { 0.0 };
    let raw: Vec<_> = target
        .iter()
        .zip(predicted)
        .map(|(x, y)| (y - ymean) - slope * (x - xmean))
        .collect();
    if residual.iter().chain(&raw).any(|v| !v.is_finite())
        || !target_span.is_finite()
        || !slope.is_finite()
    {
        return Err("scatter arithmetic overflow");
    }
    let (raw_mad, raw_residual_p99, raw_residual_max) = residual_stats(&raw);
    let prediction_min = predicted[order[0]];
    let prediction_max = predicted[order[n - 1]];
    Ok(ScatterDiagnostics {
        mapped_prediction: mapped,
        n,
        shape_mad,
        target_span,
        outside_envelope: residual
            .iter()
            .filter(|r| r.abs() > 4.0 * shape_mad)
            .count() as f64
            / n as f64,
        shape_p99: ratio(shape_p99, target_span),
        shape_max: ratio(shape_max, target_span),
        target_coverage,
        target_clump,
        prediction_coverage,
        prediction_clump,
        floor_mass: predicted.iter().filter(|&&v| v == prediction_min).count() as f64 / n as f64,
        ceiling_mass: predicted.iter().filter(|&&v| v == prediction_max).count() as f64 / n as f64,
        prediction_min,
        prediction_max,
        raw_mad,
        raw_residual_p99,
        raw_residual_max,
        raw_chart_p99: ratio(raw_residual_p99, raw_mad),
        raw_chart_max: ratio(raw_residual_max, raw_mad),
        raw_slope: slope,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tied_predictions_do_not_invent_order_or_hide_clumping() {
        let target: Vec<_> = (0..100).map(|v| v as f64).collect();
        let constant = vec![20.0; 100];
        let a = diagnose(&constant, &target).unwrap();
        let mut reversed = target.clone();
        reversed.reverse();
        let b = diagnose(&constant, &reversed).unwrap();
        assert_eq!(a.shape_p99, b.shape_p99);
        assert!(a.shape_p99.unwrap() > 0.49);
        assert_eq!(a.prediction_clump, 1.0);
        assert_eq!(a.floor_mass, 1.0);
        assert_eq!(a.ceiling_mass, 1.0);
        assert!(a.raw_chart_max.is_none());
        assert_eq!(a.raw_residual_max, 0.0);
    }
    #[test]
    fn monotone_shaping_preserves_geometry_but_raw_tail_exposes_blowup() {
        let target: Vec<_> = (0..1000).map(|v| v as f64).collect();
        let mut pred = target.clone();
        pred[999] = 1e6;
        let a = diagnose(&pred, &target).unwrap();
        assert_eq!(a.shape_max, Some(0.0));
        assert!(a.raw_chart_max.unwrap() > 100.0);
        assert!(a.raw_residual_max > 900_000.0);
        let inverted: Vec<_> = target.iter().map(|v| -v).collect();
        assert!(diagnose(&inverted, &target).unwrap().shape_max.unwrap() > 1.0);
    }
    #[test]
    fn perfect_constant_and_invalid_inputs_are_explicit() {
        let target: Vec<_> = (0..100).map(|v| v as f64).collect();
        let a = diagnose(&target, &target).unwrap();
        assert_eq!(a.shape_max, Some(0.0));
        assert_eq!(a.outside_envelope, 0.0);
        let c = diagnose(&vec![1.0; 100], &vec![2.0; 100]).unwrap();
        assert!(c.shape_p99.is_none());
        assert!(diagnose(&target[..99], &target).is_err());
        let mut bad = target.clone();
        bad[5] = f64::NAN;
        assert!(diagnose(&bad, &target).is_err());
    }
}
