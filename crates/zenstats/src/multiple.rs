//! Multiple-comparison adjustment.

/// Holm step-down adjusted p-values (Holm 1979, *Scand. J. Statist.* 6(2)).
///
/// Controls the family-wise error rate for any dependence between the tests,
/// like the Bonferroni correction ITU-T P.1401 §7.6.5 names, and rejects at
/// least everything Bonferroni rejects.
/// Returns one adjusted value per input, in input order. NaN inputs stay NaN
/// and are excluded from the family size. A registered test that has no result
/// but must still count toward the family (an incomplete entry that can never
/// pass) should be passed as `1.0`, not NaN.
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

/// Holm step-down rejections at level `alpha`: sort ascending, reject p₍ₖ₎
/// while p₍ₖ₎ ≤ α / (m − k + 1), stop at the first miss. Identical to
/// `holm(p)[i] <= alpha`, and to the rule zensim's confirmatory reads
/// registered (`scripts/rev4_featpot/v2_confirm_read.py::holm`, where an
/// incomplete entry enters as p = 1 and counts in m). NaN never rejects and
/// does not count in m.
pub fn holm_reject(p: &[f64], alpha: f64) -> Vec<bool> {
    let mut idx: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    idx.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let m = idx.len();
    let mut reject = vec![false; p.len()];
    for (rank, &i) in idx.iter().enumerate() {
        if p[i] > alpha / (m - rank) as f64 {
            break;
        }
        reject[i] = true;
    }
    reject
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

    /// The registered cases of zensim's `test_v2c.py::test_holm_step_down`.
    #[test]
    fn rejections_match_the_registered_v2c_cases() {
        let cases: [(&[f64], &[bool]); 5] = [
            (
                &[0.001, 0.03, 0.04, 0.5, 0.6, 0.7],
                &[true, false, false, false, false, false],
            ),
            (&[0.03, 0.5, 0.6, 0.7, 0.8, 0.9], &[false; 6]),
            (&[0.008, 0.012, 0.9], &[true, true, false]),
            (&[0.02, 0.5], &[true, false]),
            (&[0.03, 0.5], &[false, false]),
        ];
        for (p, want) in cases {
            assert_eq!(holm_reject(p, 0.05), want.to_vec(), "{p:?}");
            // same decisions from the adjusted p-values
            let adj: Vec<bool> = holm(p).iter().map(|a| *a <= 0.05).collect();
            assert_eq!(adj, want.to_vec(), "{p:?}");
        }
        // an incomplete entry passed as 1.0 counts in m
        assert_eq!(holm_reject(&[0.02, 1.0], 0.05), vec![true, false]);
        assert_eq!(holm_reject(&[0.03, 1.0], 0.05), vec![false, false]);
        assert_eq!(holm_reject(&[0.03, f64::NAN], 0.05), vec![true, false]);
    }
}
