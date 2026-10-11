//! Multiple-comparison adjustment.

/// Holm step-down adjusted p-values (Holm 1979, *Scand. J. Statist.* 6(2)).
///
/// Controls the family-wise error rate for any dependence between the tests.
/// Returns one adjusted value per input, in input order. NaN inputs stay NaN
/// and are excluded from the family size.
pub fn holm(p: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = idx.len();
    let mut out = vec![f64::NAN; p.len()];
    let mut running = 0.0f64;
    for (rank, &i) in idx.iter().enumerate() {
        let adj = ((m - rank) as f64 * p[i]).min(1.0);
        running = running.max(adj);
        out[i] = running;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_hand_computed_holm() {
        // sorted p: 0.01, 0.02, 0.03, 0.5 with m = 4 → 0.04, 0.06, 0.06 (monotone), 0.5
        let got = holm(&[0.03, 0.01, 0.5, 0.02]);
        let want = [0.06, 0.04, 0.5, 0.06];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-15, "{got:?}");
        }
    }

    #[test]
    fn nan_is_kept_and_not_counted() {
        let got = holm(&[0.02, f64::NAN, 0.04]);
        assert!(got[1].is_nan());
        assert!((got[0] - 0.04).abs() < 1e-15);
        assert!((got[2] - 0.04).abs() < 1e-15);
    }
}
