//! Forced-choice (2AFC / triplet) agreement between a metric and human
//! choices, scored without reducing the responses to a per-stimulus scalar.
//!
//! Moved here from zensim's `zensim-validate/src/pairwise.rs` so every metric
//! in zenmetrics is scored by the same code; the arithmetic is unchanged.
//! Imazen, the copyright holder, relicensed it from AGPL-3.0-or-later OR
//! commercial to this crate's MIT OR Apache-2.0 with the move (2026-10-11).
//!
//! # Why this statistic exists
//!
//! Every rank statistic in [`crate::panel`] consumes `(predicted, target)`
//! with a per-stimulus SCALAR target: a MOS, a DMOS, a reconstructed JND.
//! Triplet-comparison corpora (JPEG AIC-3 boosted/plain triplets, the JPEG
//! AI SDR25 study) do not have one: their raw data is N forced choices of the
//! form "which of these two is more different from the reference".
//! Reconstructing a scalar first (ordered-probit / Thurstone) collapses the
//! response count and injects the reconstruction's own model into the
//! comparison. Scoring the choices directly keeps the power and adds no model.
//!
//! # Definitions
//!
//! A row is one (group, s_left, s_right, choice, weight):
//!
//! * `group`: the comparison the responses belong to (a triplet question).
//!   Responses within a group are NOT independent. For intervals, resample
//!   source images (see [`crate::resample`]); a group-level bootstrap is
//!   available through [`agreement_by_group_index`].
//! * `s_left` / `s_right`: metric **quality** scores (higher = better), both
//!   against the same reference. The metric calls the LEFT image more
//!   distorted exactly when `s_left < s_right`.
//! * `choice`: the side the human called MORE distorted.
//! * `weight`: how many responses this row stands for. One row per
//!   (group, choice) with a count is exactly equivalent to expanding it into
//!   unit-weight rows (`weights_match_expansion`).
//!
//! Per row, agreement is `1` when the metric and the human name the same side,
//! `0` when they disagree, and `0.5` when the metric ties.
//!
//! * [`PairwiseStats::acc_response`]: weighted mean agreement over responses.
//! * [`PairwiseStats::ceiling_response`]: `Σ_g max(w_left, w_right) / Σ_g
//!   (w_left + w_right)`, the accuracy of an oracle that always names the
//!   group's majority side. A metric cannot exceed it, and it is usually well
//!   below 1 because near-threshold comparisons split the observers.
//! * [`PairwiseStats::acc_norm`]: `(acc − 0.5) / (ceiling − 0.5)`, the share of
//!   achievable signal captured: 0 = chance, 1 = ceiling.
//! * [`PairwiseStats::acc_group_majority`]: unweighted agreement with the
//!   majority side over groups that have one.
//!
//! [`log_likelihood`] scores the same rows as probabilities once the metric
//! is mapped to the target scale: a proper scoring rule that keeps magnitudes
//! and weights each comparison by how much the observers actually disagreed.
//!
//! # Preparing raw responses
//!
//! Score the same response set the target scale was reconstructed from:
//!
//! * **Screen workers and questions as the reconstruction did.** Crowd data
//!   carries inattentive workers; on JPEG AIC plain-triplet data an unscreened
//!   response set gives the human JND scale itself a fitted observer slope
//!   near 0.3 instead of 0.6745, and log-likelihood then ranks metrics by how
//!   flat their predictions are. Drop trap questions and any question that
//!   shows the same image on both sides (every metric ties there).
//! * **Keep "not sure" answers** as [`Choice::NotSure`]. They count half for
//!   each side, the convention of the JPEG AIC reconstruction (Testolina et
//!   al. 2024, arXiv:2410.09501), of Men et al. 2021 (boosted triplet
//!   comparisons, IEEE Access, doi:10.1109/ACCESS.2021.3118295, R = 1/2) and
//!   of Hanhart et al. 2016 for ties. Dropping them scores a different response
//!   set from the one the scale was built on.
//!
//!   What the split changes: a not-sure response adds half its weight to the
//!   metric's agreement whichever side the metric picks, and half to the
//!   majority ceiling, so `acc_norm` and `acc_group_majority` are unchanged
//!   (the normalisation cancels it; tested). `acc_response`,
//!   `ceiling_response`, `ceiling_loo`, `observer_agreement`, `tie_rate`,
//!   `n_responses` and every log-likelihood figure do change. Label figures of
//!   those kinds computed with not-sure answers dropped as "not-sure dropped".
//! * **Splitting does not replace screening.** On unscreened JPEG AIC-3
//!   plain-triplet responses the human scale scores below chance in
//!   log-likelihood with not-sure split. Apply the dataset's own screening;
//!   for AIC-3 that keeps batches with at least 70 % correct on their 10 trap
//!   questions.
//! * **Check the unit:** [`fit_observer_slope`] on the human scale values
//!   themselves should come out near the unit's own slope (0.6745 for a JND
//!   defined at 75 %). Report it beside every log-likelihood; a slope far from
//!   it means the responses and the unit disagree.
//!
//! # Ceilings
//!
//! The in-sample ceilings (`ceiling_response`, `LogLikelihood::ceiling_ll`)
//! fit each group's own responses, so they are upper bounds, optimistic by an
//! amount that shrinks with responses per group. `ceiling_loo` (leave one
//! response out) and `ceiling_ll_corrected` (first-order bias correction) are
//! the cross-fitted counterparts; compare datasets with different response
//! counts on those. `observer_agreement` is the expected agreement of two
//! different observers, p² + (1 − p)² per group, the human ceiling LPIPS
//! reports (Zhang et al. 2018, arXiv:1801.03924); `ceiling_response` is the
//! oracle max(p, 1 − p) that paper calls the theoretical maximum.

use core::fmt;

/// Which side the human named as MORE distorted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Choice {
    Left,
    Right,
    /// The observer answered "not sure": counted half for each side.
    NotSure,
}

impl Choice {
    /// Parse the response token used by the JPEG-AIC response CSVs.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "left" | "l" => Some(Choice::Left),
            "right" | "r" => Some(Choice::Right),
            "notsure" | "not sure" | "not_sure" | "not-sure" | "unsure" => Some(Choice::NotSure),
            _ => None,
        }
    }
}

/// One weighted forced-choice observation.
#[derive(Clone, Copy, Debug)]
pub struct PairwiseRow {
    /// Index into the caller's group table (dense, 0-based).
    pub group: usize,
    /// Metric QUALITY of the left image (higher = better).
    pub s_left: f64,
    /// Metric QUALITY of the right image (higher = better).
    pub s_right: f64,
    /// The side the human called more distorted.
    pub choice: Choice,
    /// Number of responses this row stands for (must be finite and >= 0).
    pub weight: f64,
}

/// Result of [`agreement`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairwiseStats {
    /// Distinct groups contributing at least one response.
    pub n_groups: usize,
    /// Σ weight — the number of human responses scored.
    pub n_responses: f64,
    /// Weighted mean agreement (ties = 0.5).
    pub acc_response: f64,
    /// Weighted fraction of responses where the metric tied.
    pub tie_rate: f64,
    /// Majority-oracle accuracy on the same responses, in sample: an upper
    /// bound on any metric's `acc_response`.
    pub ceiling_response: f64,
    /// Leave-one-response-out majority accuracy: each response is predicted
    /// by the majority of the other responses in its group (exact for whole
    /// response counts). The cross-fitted ceiling.
    pub ceiling_loo: f64,
    /// Expected agreement of two different observers on the same question,
    /// `(l(l−1) + r(r−1)) / (n(n−1))` per group, response-weighted over groups
    /// with at least two responses.
    pub observer_agreement: f64,
    /// `(acc_response − 0.5) / (ceiling_response − 0.5)`; NaN if the ceiling
    /// is at chance.
    pub acc_norm: f64,
    /// `(acc_response − 0.5) / (ceiling_loo − 0.5)`; NaN if that ceiling is at
    /// or below chance.
    pub acc_norm_loo: f64,
    /// Mean agreement with the majority side over groups that have one.
    pub acc_group_majority: f64,
    /// How many groups have a strict majority.
    pub n_groups_majority: usize,
}

impl fmt::Display for PairwiseStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "acc={:.6} ceiling={:.6} loo={:.6} observers={:.6} norm={:.6} tie={:.6} groups={} maj={}/{} n={:.0}",
            self.acc_response,
            self.ceiling_response,
            self.ceiling_loo,
            self.observer_agreement,
            self.acc_norm,
            self.tie_rate,
            self.n_groups,
            self.acc_group_majority,
            self.n_groups_majority,
            self.n_responses
        )
    }
}

/// The row's response weight on (left, right); "not sure" splits it.
#[inline]
fn sides(r: &PairwiseRow) -> (f64, f64) {
    match r.choice {
        Choice::Left => (r.weight, 0.0),
        Choice::Right => (0.0, r.weight),
        Choice::NotSure => (0.5 * r.weight, 0.5 * r.weight),
    }
}

/// Weighted agreement of one row: Σ over its sides of weight × (1 / 0.5 / 0).
#[inline]
fn row_agreement_sum(r: &PairwiseRow) -> f64 {
    let (wl, wr) = sides(r);
    if r.s_left == r.s_right {
        0.5 * (wl + wr)
    } else if r.s_left < r.s_right {
        wl
    } else {
        wr
    }
}

/// Compute the statistic over every row.
///
/// `n_groups_total` is the size of the caller's group table; groups with no
/// rows simply do not contribute.
pub fn agreement(rows: &[PairwiseRow], n_groups_total: usize) -> PairwiseStats {
    agreement_impl(rows.iter().copied(), n_groups_total)
}

/// Compute the statistic over a RESAMPLE of groups.
///
/// `group_rows[g]` holds the rows of group `g` (build it once with
/// [`index_rows_by_group`]); `picked` is a multiset of group indices, so a
/// group drawn twice contributes twice. This is the cluster-bootstrap
/// primitive — the caller owns the RNG.
pub fn agreement_by_group_index(
    group_rows: &[Vec<PairwiseRow>],
    picked: &[usize],
) -> PairwiseStats {
    // Re-key each drawn group to a fresh index so a group drawn twice counts
    // as two independent groups (that is what a cluster bootstrap means).
    let mut rows: Vec<PairwiseRow> = Vec::new();
    for (fresh, &g) in picked.iter().enumerate() {
        for r in &group_rows[g] {
            let mut r = *r;
            r.group = fresh;
            rows.push(r);
        }
    }
    agreement(&rows, picked.len())
}

/// Bucket rows by group index. `n_groups_total` sets the outer length.
pub fn index_rows_by_group(rows: &[PairwiseRow], n_groups_total: usize) -> Vec<Vec<PairwiseRow>> {
    let mut out = vec![Vec::new(); n_groups_total];
    for r in rows {
        out[r.group].push(*r);
    }
    out
}

fn agreement_impl<I: Iterator<Item = PairwiseRow>>(
    rows: I,
    n_groups_total: usize,
) -> PairwiseStats {
    let mut w_sum = 0.0f64;
    let mut agree_sum = 0.0f64;
    let mut tie_sum = 0.0f64;
    // per group: (w_left, w_right, agree_left_side_score, agree_right_side_score)
    let mut gl = vec![0.0f64; n_groups_total];
    let mut gr = vec![0.0f64; n_groups_total];
    // agreement the metric would score against a "left is worse" verdict
    let mut g_metric_left = vec![f64::NAN; n_groups_total];
    for r in rows {
        debug_assert!(r.weight.is_finite() && r.weight >= 0.0);
        w_sum += r.weight;
        agree_sum += row_agreement_sum(&r);
        if r.s_left == r.s_right {
            tie_sum += r.weight;
        }
        let (wl, wr) = sides(&r);
        gl[r.group] += wl;
        gr[r.group] += wr;
        // The metric's verdict is a property of the group, not of the row.
        let m = if r.s_left == r.s_right {
            0.5
        } else if r.s_left < r.s_right {
            1.0
        } else {
            0.0
        };
        g_metric_left[r.group] = m;
    }
    let mut ceil_num = 0.0f64;
    let mut ceil_den = 0.0f64;
    let mut loo_num = 0.0f64;
    let mut obs_num = 0.0f64;
    let mut obs_den = 0.0f64;
    let mut maj_sum = 0.0f64;
    let mut maj_n = 0usize;
    let mut n_groups = 0usize;
    for g in 0..n_groups_total {
        let (l, r) = (gl[g], gr[g]);
        if l + r <= 0.0 {
            continue;
        }
        n_groups += 1;
        ceil_num += if l > r { l } else { r };
        ceil_den += l + r;
        // a left response is predicted by the other responses: (l − 1, r)
        let loo = |mine: f64, other: f64| -> f64 {
            if mine - 1.0 > other {
                1.0
            } else if mine - 1.0 == other {
                0.5
            } else {
                0.0
            }
        };
        loo_num += l * loo(l, r) + r * loo(r, l);
        let n = l + r;
        if n >= 2.0 {
            obs_num += n * (l * (l - 1.0) + r * (r - 1.0)) / (n * (n - 1.0));
            obs_den += n;
        }
        if l != r {
            maj_n += 1;
            let m = g_metric_left[g];
            // m == 1.0 means "metric says LEFT is worse"
            maj_sum += if m == 0.5 {
                0.5
            } else if (m == 1.0) == (l > r) {
                1.0
            } else {
                0.0
            };
        }
    }
    let acc = if w_sum > 0.0 {
        agree_sum / w_sum
    } else {
        f64::NAN
    };
    let ceiling = if ceil_den > 0.0 {
        ceil_num / ceil_den
    } else {
        f64::NAN
    };
    let acc_norm = if ceiling > 0.5 {
        (acc - 0.5) / (ceiling - 0.5)
    } else {
        f64::NAN
    };
    let ceiling_loo = if ceil_den > 0.0 {
        loo_num / ceil_den
    } else {
        f64::NAN
    };
    let acc_norm_loo = if ceiling_loo > 0.5 {
        (acc - 0.5) / (ceiling_loo - 0.5)
    } else {
        f64::NAN
    };
    PairwiseStats {
        n_groups,
        n_responses: w_sum,
        acc_response: acc,
        tie_rate: if w_sum > 0.0 {
            tie_sum / w_sum
        } else {
            f64::NAN
        },
        ceiling_response: ceiling,
        ceiling_loo,
        observer_agreement: if obs_den > 0.0 {
            obs_num / obs_den
        } else {
            f64::NAN
        },
        acc_norm,
        acc_norm_loo,
        acc_group_majority: if maj_n > 0 {
            maj_sum / maj_n as f64
        } else {
            f64::NAN
        },
        n_groups_majority: maj_n,
    }
}

/// Result of [`log_likelihood`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogLikelihood {
    /// Σ weight.
    pub n_responses: f64,
    /// Mean log-likelihood per response of the metric's predicted choice
    /// probabilities (natural log, ≤ 0).
    pub mean_ll: f64,
    /// The same for each group's own observed choice proportion, in sample:
    /// an upper bound.
    pub ceiling_ll: f64,
    /// `ceiling_ll − G / (2N)` for G groups and N responses: the first-order
    /// (χ²) bias correction of the in-sample saturated likelihood.
    pub ceiling_ll_corrected: f64,
    /// `ln 0.5`, a predictor that always says 50 %.
    pub chance_ll: f64,
    /// `(mean_ll − chance_ll) / (ceiling_ll − chance_ll)`: 0 = chance,
    /// 1 = in-sample ceiling, negative = worse than always saying 50 %.
    pub normalized: f64,
    /// The same against `ceiling_ll_corrected`.
    pub normalized_corrected: f64,
}

/// Why a forced-choice score could not be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ForcedChoiceError {
    /// The observer scale is not positive and finite.
    BadScale,
    /// A weight or score is not finite, or a weight is negative.
    NonFinite,
    /// A row's group index is not below `n_groups_total`.
    GroupOutOfRange,
}

impl fmt::Display for ForcedChoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadScale => write!(f, "observer scale must be positive and finite"),
            Self::NonFinite => write!(
                f,
                "a score or weight is non-finite, or a weight is negative"
            ),
            Self::GroupOutOfRange => write!(f, "a row's group index is out of range"),
        }
    }
}

impl std::error::Error for ForcedChoiceError {}

fn check_rows(rows: &[PairwiseRow], n_groups_total: usize) -> Result<(), ForcedChoiceError> {
    for r in rows {
        if !(r.weight.is_finite()
            && r.weight >= 0.0
            && r.s_left.is_finite()
            && r.s_right.is_finite())
        {
            return Err(ForcedChoiceError::NonFinite);
        }
        if r.group >= n_groups_total {
            return Err(ForcedChoiceError::GroupOutOfRange);
        }
    }
    Ok(())
}

/// Held-out log-likelihood of the human choices under a Thurstone Case V
/// observer: P(left judged more distorted) = Φ((s_right − s_left) / scale).
///
/// `s_left` / `s_right` must already be on the target's **interval scale**,
/// quality-oriented (for a JND target, `s = −predicted JND`, with the JND
/// map frozen before these responses are scored). `scale` is the observer
/// scale in those units; for the JPEG AIC JND unit (1 JND = 75 % correct)
/// it is `1 / Φ⁻¹(0.75) ≈ 1.4826`, so P = Φ(0.6745 · ΔJND). See the module
/// docs for screening and the "not sure" convention.
///
/// Probabilities are clamped to [1e-12, 1 − 1e-12] so one confident miss
/// cannot make the score −∞. Groups with no responses contribute nothing.
pub fn log_likelihood(
    rows: &[PairwiseRow],
    n_groups_total: usize,
    scale: f64,
) -> Result<LogLikelihood, ForcedChoiceError> {
    const EPS: f64 = 1e-12;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(ForcedChoiceError::BadScale);
    }
    check_rows(rows, n_groups_total)?;
    let chance_ll = -core::f64::consts::LN_2;
    let mut w_sum = 0.0f64;
    let mut ll_sum = 0.0f64;
    let mut gl = vec![0.0f64; n_groups_total];
    let mut gr = vec![0.0f64; n_groups_total];
    for r in rows {
        if r.weight == 0.0 {
            continue;
        }
        let z = (r.s_right - r.s_left) / scale;
        let p_left = crate::special::norm_cdf(z).clamp(EPS, 1.0 - EPS);
        let p_right = crate::special::norm_cdf(-z).clamp(EPS, 1.0 - EPS);
        let (wl, wr) = sides(r);
        ll_sum += wl * p_left.ln() + wr * p_right.ln();
        w_sum += r.weight;
        gl[r.group] += wl;
        gr[r.group] += wr;
    }
    let mut ceil_sum = 0.0f64;
    let mut groups = 0usize;
    for g in 0..n_groups_total {
        let (l, rr) = (gl[g], gr[g]);
        let n = l + rr;
        if n <= 0.0 {
            continue;
        }
        groups += 1;
        if l > 0.0 {
            ceil_sum += l * (l / n).ln();
        }
        if rr > 0.0 {
            ceil_sum += rr * (rr / n).ln();
        }
    }
    let (mean_ll, ceiling_ll, ceiling_ll_corrected) = if w_sum > 0.0 {
        let c = ceil_sum / w_sum;
        (ll_sum / w_sum, c, c - groups as f64 / (2.0 * w_sum))
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };
    let norm = |ceil: f64| {
        if ceil > chance_ll {
            (mean_ll - chance_ll) / (ceil - chance_ll)
        } else {
            f64::NAN
        }
    };
    Ok(LogLikelihood {
        n_responses: w_sum,
        mean_ll,
        ceiling_ll,
        ceiling_ll_corrected,
        chance_ll,
        normalized: norm(ceiling_ll),
        normalized_corrected: norm(ceiling_ll_corrected),
    })
}

/// Maximum-likelihood observer slope k for P(left more distorted) =
/// Φ(k · (s_right − s_left)), fitted on the responses themselves.
///
/// Call it with the *human* scale values as `s` to check a response set
/// against its unit: for a JND defined at 75 % correct, k should be near
/// Φ⁻¹(0.75) ≈ 0.6745. The log-likelihood is concave in k, so the maximum is
/// unique; it is searched on [−1000, 1000]. Returns `None` when every score
/// difference is zero or the maximum sits on the search edge (the responses
/// separate perfectly).
pub fn fit_observer_slope(rows: &[PairwiseRow]) -> Option<f64> {
    if rows
        .iter()
        .all(|r| r.s_left == r.s_right || r.weight == 0.0)
    {
        return None;
    }
    let ll = |k: f64| -> f64 {
        rows.iter()
            .map(|r| {
                let z = k * (r.s_right - r.s_left);
                let (wl, wr) = sides(r);
                // exact tails keep the likelihood strictly concave, so the
                // golden-section search cannot stall on a clamped plateau
                let mut v = 0.0;
                if wl > 0.0 {
                    v += wl * crate::special::ln_norm_cdf(z);
                }
                if wr > 0.0 {
                    v += wr * crate::special::ln_norm_cdf(-z);
                }
                v
            })
            .sum()
    };
    let (mut a, mut b) = (-1000.0f64, 1000.0f64);
    let g = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (ll(c), ll(d));
    for _ in 0..200 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - g * (b - a);
            fc = ll(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + g * (b - a);
            fd = ll(d);
        }
    }
    let k = 0.5 * (a + b);
    if k.abs() > 999.0 { None } else { Some(k) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(g: usize, l: f64, r: f64, c: Choice, w: f64) -> PairwiseRow {
        PairwiseRow {
            group: g,
            s_left: l,
            s_right: r,
            choice: c,
            weight: w,
        }
    }

    /// Independent brute-force reference: expand weights into unit rows and
    /// recompute every field the slow, obvious way. Used as the parity
    /// oracle for the real implementation.
    fn reference(rows: &[PairwiseRow], n_groups_total: usize) -> PairwiseStats {
        let mut flat: Vec<PairwiseRow> = Vec::new();
        for r in rows {
            let n = r.weight.round() as usize;
            for _ in 0..n {
                flat.push(PairwiseRow { weight: 1.0, ..*r });
            }
        }
        let mut agree = 0.0;
        let mut ties = 0.0;
        for r in &flat {
            let m_left = r.s_left < r.s_right;
            let tie = r.s_left == r.s_right;
            if tie {
                ties += 1.0;
                agree += 0.5;
            } else if m_left == (r.choice == Choice::Left) {
                agree += 1.0;
            }
        }
        let n = flat.len() as f64;
        let mut ceil_num = 0.0;
        let mut ceil_den = 0.0;
        let mut maj = Vec::new();
        let mut n_groups = 0;
        for g in 0..n_groups_total {
            let l = flat
                .iter()
                .filter(|r| r.group == g && r.choice == Choice::Left)
                .count() as f64;
            let rr = flat
                .iter()
                .filter(|r| r.group == g && r.choice == Choice::Right)
                .count() as f64;
            if l + rr == 0.0 {
                continue;
            }
            n_groups += 1;
            ceil_num += l.max(rr);
            ceil_den += l + rr;
            if l != rr {
                let any = flat.iter().find(|r| r.group == g).unwrap();
                let v = if any.s_left == any.s_right {
                    0.5
                } else if (any.s_left < any.s_right) == (l > rr) {
                    1.0
                } else {
                    0.0
                };
                maj.push(v);
            }
        }
        let acc = agree / n;
        let ceiling = ceil_num / ceil_den;
        // leave-one-out: predict each unit response by the others in its group
        let mut loo = 0.0;
        for (x, r) in flat.iter().enumerate() {
            let (mut l, mut rr) = (0.0, 0.0);
            for (y, o) in flat.iter().enumerate() {
                if y != x && o.group == r.group {
                    if o.choice == Choice::Left {
                        l += 1.0;
                    } else {
                        rr += 1.0;
                    }
                }
            }
            let mine_left = r.choice == Choice::Left;
            let (mine, other) = if mine_left { (l, rr) } else { (rr, l) };
            loo += if mine > other {
                1.0
            } else if mine == other {
                0.5
            } else {
                0.0
            };
        }
        let ceiling_loo = loo / n;
        // two different observers: count agreeing ordered pairs per group
        let (mut obs_num, mut obs_den) = (0.0, 0.0);
        for g in 0..n_groups_total {
            let members: Vec<&PairwiseRow> = flat.iter().filter(|r| r.group == g).collect();
            let m = members.len() as f64;
            if m < 2.0 {
                continue;
            }
            let mut agree_pairs = 0.0;
            for (x, a) in members.iter().enumerate() {
                for (y, b) in members.iter().enumerate() {
                    if x != y && a.choice == b.choice {
                        agree_pairs += 1.0;
                    }
                }
            }
            obs_num += m * agree_pairs / (m * (m - 1.0));
            obs_den += m;
        }
        PairwiseStats {
            n_groups,
            n_responses: n,
            acc_response: acc,
            tie_rate: ties / n,
            ceiling_response: ceiling,
            ceiling_loo,
            observer_agreement: if obs_den > 0.0 {
                obs_num / obs_den
            } else {
                f64::NAN
            },
            acc_norm: (acc - 0.5) / (ceiling - 0.5),
            acc_norm_loo: if ceiling_loo > 0.5 {
                (acc - 0.5) / (ceiling_loo - 0.5)
            } else {
                f64::NAN
            },
            acc_group_majority: maj.iter().sum::<f64>() / maj.len() as f64,
            n_groups_majority: maj.len(),
        }
    }

    fn close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < 1e-12 || (a.is_nan() && b.is_nan()),
            "{a} vs {b}"
        );
    }

    fn assert_parity(rows: &[PairwiseRow], n: usize) {
        let got = agreement(rows, n);
        let want = reference(rows, n);
        assert_eq!(got.n_groups, want.n_groups);
        assert_eq!(got.n_groups_majority, want.n_groups_majority);
        close(got.n_responses, want.n_responses);
        close(got.acc_response, want.acc_response);
        close(got.tie_rate, want.tie_rate);
        close(got.ceiling_response, want.ceiling_response);
        close(got.acc_norm, want.acc_norm);
        close(got.acc_group_majority, want.acc_group_majority);
        close(got.ceiling_loo, want.ceiling_loo);
        close(got.observer_agreement, want.observer_agreement);
        close(got.acc_norm_loo, want.acc_norm_loo);
    }

    #[test]
    fn perfect_metric_scores_one() {
        // group 0: humans say LEFT worse; metric scores left lower.
        let rows = [row(0, 60.0, 90.0, Choice::Left, 7.0)];
        let s = agreement(&rows, 1);
        close(s.acc_response, 1.0);
        close(s.ceiling_response, 1.0);
        close(s.acc_norm, 1.0);
        close(s.acc_group_majority, 1.0);
        assert_parity(&rows, 1);
    }

    #[test]
    fn inverted_metric_scores_zero() {
        let rows = [row(0, 90.0, 60.0, Choice::Left, 5.0)];
        let s = agreement(&rows, 1);
        close(s.acc_response, 0.0);
        close(s.acc_norm, -1.0);
        close(s.acc_group_majority, 0.0);
        assert_parity(&rows, 1);
    }

    #[test]
    fn metric_tie_scores_chance_and_is_reported() {
        let rows = [row(0, 77.0, 77.0, Choice::Left, 4.0)];
        let s = agreement(&rows, 1);
        close(s.acc_response, 0.5);
        close(s.tie_rate, 1.0);
        close(s.acc_group_majority, 0.5);
        assert_parity(&rows, 1);
    }

    #[test]
    fn ceiling_is_the_majority_oracle_not_one() {
        // 6 say LEFT, 4 say RIGHT: no predictor can beat 0.6 here.
        let rows = [
            row(0, 60.0, 90.0, Choice::Left, 6.0),
            row(0, 60.0, 90.0, Choice::Right, 4.0),
        ];
        let s = agreement(&rows, 1);
        close(s.ceiling_response, 0.6);
        close(s.acc_response, 0.6); // metric agrees with the 6
        close(s.acc_norm, 1.0); // ...which is the ceiling
        close(s.acc_group_majority, 1.0);
        assert_parity(&rows, 1);
    }

    #[test]
    fn split_group_has_no_majority_and_is_excluded_from_that_column() {
        let rows = [
            row(0, 60.0, 90.0, Choice::Left, 5.0),
            row(0, 60.0, 90.0, Choice::Right, 5.0),
            row(1, 60.0, 90.0, Choice::Left, 3.0),
        ];
        let s = agreement(&rows, 2);
        assert_eq!(s.n_groups, 2);
        assert_eq!(s.n_groups_majority, 1);
        close(s.acc_group_majority, 1.0);
        assert_parity(&rows, 2);
    }

    #[test]
    fn weights_match_expansion() {
        // The load-bearing equivalence: one weighted row == N unit rows.
        let weighted = [
            row(0, 60.0, 90.0, Choice::Left, 6.0),
            row(0, 60.0, 90.0, Choice::Right, 4.0),
            row(1, 95.0, 91.0, Choice::Left, 2.0),
            row(1, 95.0, 91.0, Choice::Right, 9.0),
            row(2, 70.0, 70.0, Choice::Right, 3.0),
        ];
        let mut expanded: Vec<PairwiseRow> = Vec::new();
        for r in weighted {
            for _ in 0..(r.weight as usize) {
                expanded.push(PairwiseRow { weight: 1.0, ..r });
            }
        }
        let a = agreement(&weighted, 3);
        let b = agreement(&expanded, 3);
        close(a.acc_response, b.acc_response);
        close(a.ceiling_response, b.ceiling_response);
        close(a.acc_norm, b.acc_norm);
        close(a.tie_rate, b.tie_rate);
        close(a.acc_group_majority, b.acc_group_majority);
        assert_eq!(a.n_groups, b.n_groups);
        assert_parity(&weighted, 3);
        assert_parity(&expanded, 3);
    }

    #[test]
    fn resample_of_all_groups_once_reproduces_the_full_statistic() {
        let rows = [
            row(0, 60.0, 90.0, Choice::Left, 6.0),
            row(0, 60.0, 90.0, Choice::Right, 4.0),
            row(1, 95.0, 91.0, Choice::Left, 2.0),
            row(1, 95.0, 91.0, Choice::Right, 9.0),
            row(2, 70.0, 70.0, Choice::Right, 3.0),
        ];
        let idx = index_rows_by_group(&rows, 3);
        let full = agreement(&rows, 3);
        let same = agreement_by_group_index(&idx, &[0, 1, 2]);
        close(full.acc_response, same.acc_response);
        close(full.ceiling_response, same.ceiling_response);
        close(full.acc_group_majority, same.acc_group_majority);
        assert_eq!(full.n_groups, same.n_groups);
    }

    #[test]
    fn duplicate_draw_counts_a_group_twice() {
        let rows = [
            row(0, 60.0, 90.0, Choice::Left, 1.0),
            row(1, 90.0, 60.0, Choice::Left, 1.0),
        ];
        let idx = index_rows_by_group(&rows, 2);
        // Drawing group 0 (metric right) twice and group 1 (metric wrong) once
        // must read 2/3, not 1/2.
        let s = agreement_by_group_index(&idx, &[0, 0, 1]);
        close(s.acc_response, 2.0 / 3.0);
        assert_eq!(s.n_groups, 3);
    }

    #[test]
    fn randomized_parity_against_brute_force() {
        // Deterministic LCG — no rand dependency, reproducible failure.
        let mut st: u64 = 0x2026_0901_0000_0001;
        let mut next = move || {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };
        for _case in 0..200 {
            let n_groups = 1 + (next() % 6) as usize;
            let mut rows = Vec::new();
            for g in 0..n_groups {
                // Some groups deliberately get NO rows.
                if next() % 7 == 0 {
                    continue;
                }
                // Force ties sometimes.
                let l = (next() % 100) as f64;
                let r = if next() % 5 == 0 {
                    l
                } else {
                    (next() % 100) as f64
                };
                let wl = (next() % 9) as f64;
                let wr = (next() % 9) as f64;
                if wl > 0.0 {
                    rows.push(row(g, l, r, Choice::Left, wl));
                }
                if wr > 0.0 {
                    rows.push(row(g, l, r, Choice::Right, wr));
                }
            }
            if rows.is_empty() {
                continue;
            }
            assert_parity(&rows, n_groups);
        }
    }

    const JND_SCALE: f64 = 1.0 / 0.674_489_750_196_081_7;

    #[test]
    fn log_likelihood_hits_the_ceiling_when_the_metric_predicts_the_observed_split() {
        // 75 % say LEFT worse; a metric 1 JND apart predicts exactly 75 %.
        let rows = [
            row(0, -1.0, 0.0, Choice::Left, 75.0),
            row(0, -1.0, 0.0, Choice::Right, 25.0),
        ];
        let ll = log_likelihood(&rows, 1, JND_SCALE).unwrap();
        assert!((ll.mean_ll - ll.ceiling_ll).abs() < 1e-6, "{ll:?}");
        assert!((ll.normalized - 1.0).abs() < 1e-5, "{ll:?}");
        let want = 0.75f64 * 0.75f64.ln() + 0.25 * 0.25f64.ln();
        assert!((ll.ceiling_ll - want).abs() < 1e-12);
    }

    #[test]
    fn log_likelihood_chance_and_overconfidence() {
        let rows = [
            row(0, 0.0, 0.0, Choice::Left, 6.0),
            row(0, 0.0, 0.0, Choice::Right, 4.0),
        ];
        // a metric tie predicts 50 %: exactly chance
        let ll = log_likelihood(&rows, 1, JND_SCALE).unwrap();
        assert!((ll.mean_ll - ll.chance_ll).abs() < 1e-7);
        assert!(ll.normalized.abs() < 1e-5);
        // a confident wrong call (5 JND the other way) is far below chance
        let wrong = [
            row(0, 0.0, -5.0, Choice::Left, 9.0),
            row(0, 0.0, -5.0, Choice::Right, 1.0),
        ];
        let ll = log_likelihood(&wrong, 1, JND_SCALE).unwrap();
        assert!(ll.normalized < -1.0, "{ll:?}");
    }

    #[test]
    fn log_likelihood_weights_match_expansion() {
        let weighted = [
            row(0, -0.3, 0.2, Choice::Left, 3.0),
            row(0, -0.3, 0.2, Choice::Right, 2.0),
            row(1, 1.0, -0.5, Choice::Right, 4.0),
        ];
        let mut expanded = Vec::new();
        for r in weighted {
            for _ in 0..(r.weight as usize) {
                expanded.push(PairwiseRow { weight: 1.0, ..r });
            }
        }
        let a = log_likelihood(&weighted, 2, JND_SCALE).unwrap();
        let b = log_likelihood(&expanded, 2, JND_SCALE).unwrap();
        close(a.mean_ll, b.mean_ll);
        close(a.ceiling_ll, b.ceiling_ll);
        close(a.normalized, b.normalized);
    }

    #[test]
    fn not_sure_counts_half_for_each_side() {
        let with_ns = [
            row(0, 60.0, 90.0, Choice::Left, 5.0),
            row(0, 60.0, 90.0, Choice::NotSure, 4.0),
            row(1, -0.4, 0.3, Choice::NotSure, 2.0),
            row(1, -0.4, 0.3, Choice::Right, 1.0),
        ];
        let split = [
            row(0, 60.0, 90.0, Choice::Left, 5.0),
            row(0, 60.0, 90.0, Choice::Left, 2.0),
            row(0, 60.0, 90.0, Choice::Right, 2.0),
            row(1, -0.4, 0.3, Choice::Left, 1.0),
            row(1, -0.4, 0.3, Choice::Right, 1.0),
            row(1, -0.4, 0.3, Choice::Right, 1.0),
        ];
        let (a, b) = (agreement(&with_ns, 2), agreement(&split, 2));
        close(a.acc_response, b.acc_response);
        close(a.ceiling_response, b.ceiling_response);
        close(a.ceiling_loo, b.ceiling_loo);
        close(a.observer_agreement, b.observer_agreement);
        close(a.n_responses, b.n_responses);
        let (la, lb) = (
            log_likelihood(&with_ns, 2, JND_SCALE).unwrap(),
            log_likelihood(&split, 2, JND_SCALE).unwrap(),
        );
        close(la.mean_ll, lb.mean_ll);
        close(la.ceiling_ll, lb.ceiling_ll);
        assert_eq!(Choice::parse("notSure"), Some(Choice::NotSure));
        assert_eq!(Choice::parse(" not sure "), Some(Choice::NotSure));
    }

    #[test]
    fn cross_fitted_ceilings_sit_below_the_in_sample_ones() {
        // 6 vs 4: in-sample oracle 0.6; leaving one out, a LEFT response sees
        // 5 vs 4 (right), a RIGHT response sees 6 vs 3 (wrong): 0.6 as well,
        // but a 3 vs 2 group drops from 0.6 to 0.3 (left: 2 vs 2 tie, right: 3 vs 1).
        let rows = [
            row(0, 60.0, 90.0, Choice::Left, 3.0),
            row(0, 60.0, 90.0, Choice::Right, 2.0),
        ];
        let s = agreement(&rows, 1);
        close(s.ceiling_response, 0.6);
        close(s.ceiling_loo, (3.0 * 0.5 + 2.0 * 0.0) / 5.0);
        close(s.observer_agreement, (3.0 * 2.0 + 2.0 * 1.0) / (5.0 * 4.0));
        let ll = log_likelihood(&rows, 1, JND_SCALE).unwrap();
        assert!(ll.ceiling_ll_corrected < ll.ceiling_ll);
        close(ll.ceiling_ll - ll.ceiling_ll_corrected, 1.0 / (2.0 * 5.0));
    }

    #[test]
    fn bad_scale_and_rows_are_refused() {
        let rows = [row(0, 0.0, 1.0, Choice::Left, 1.0)];
        assert_eq!(
            log_likelihood(&rows, 1, 0.0),
            Err(ForcedChoiceError::BadScale)
        );
        assert_eq!(
            log_likelihood(&rows, 1, -1.0),
            Err(ForcedChoiceError::BadScale)
        );
        assert_eq!(
            log_likelihood(&rows, 0, 1.0),
            Err(ForcedChoiceError::GroupOutOfRange)
        );
        let bad = [row(0, f64::NAN, 1.0, Choice::Left, 1.0)];
        assert_eq!(
            log_likelihood(&bad, 1, 1.0),
            Err(ForcedChoiceError::NonFinite)
        );
    }

    #[test]
    fn observer_slope_recovers_the_generating_slope() {
        // expected response shares exactly Φ(k·Δ) for k = 0.6745 → the MLE is k
        let k = 0.674_489_750_196_081_7;
        let mut rows = Vec::new();
        for (g, d) in [0.2f64, 0.5, 1.0, 1.5, 2.5, -0.7].iter().enumerate() {
            let p = crate::special::norm_cdf(k * d);
            rows.push(row(g, 0.0, *d, Choice::Left, 1000.0 * p));
            rows.push(row(g, 0.0, *d, Choice::Right, 1000.0 * (1.0 - p)));
        }
        let got = fit_observer_slope(&rows).unwrap();
        assert!((got - k).abs() < 1e-5, "{got}");
        // perfectly separated responses have no finite maximum
        let sep = [row(0, 0.0, 1.0, Choice::Left, 5.0)];
        assert_eq!(fit_observer_slope(&sep), None);
    }

    /// Splitting not-sure answers half/half leaves `acc_norm` and
    /// `acc_group_majority` unchanged and moves the raw figures.
    #[test]
    fn not_sure_split_leaves_acc_norm_and_group_majority_unchanged() {
        let base = [
            row(0, 60.0, 90.0, Choice::Left, 6.0),
            row(0, 60.0, 90.0, Choice::Right, 2.0),
            row(1, 95.0, 91.0, Choice::Left, 3.0),
            row(1, 95.0, 91.0, Choice::Right, 5.0),
            row(2, 70.0, 70.0, Choice::Right, 4.0),
            row(2, 70.0, 70.0, Choice::Left, 1.0),
            row(3, 10.0, 40.0, Choice::Right, 3.0),
            row(3, 10.0, 40.0, Choice::Left, 2.0),
        ];
        let mut with_ns = base.to_vec();
        with_ns.push(row(0, 60.0, 90.0, Choice::NotSure, 3.0));
        with_ns.push(row(1, 95.0, 91.0, Choice::NotSure, 1.0));
        with_ns.push(row(2, 70.0, 70.0, Choice::NotSure, 2.0));
        with_ns.push(row(3, 10.0, 40.0, Choice::NotSure, 5.0));
        let (a, b) = (agreement(&base, 4), agreement(&with_ns, 4));
        close(a.acc_norm, b.acc_norm);
        close(a.acc_group_majority, b.acc_group_majority);
        assert!((a.acc_response - b.acc_response).abs() > 1e-6);
        assert!((a.ceiling_response - b.ceiling_response).abs() > 1e-6);
    }
}
