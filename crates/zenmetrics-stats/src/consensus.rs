//! Consensus ranking of items (metrics, models) across several evaluation criteria.
//!
//! Criteria such as SROCC, PLCC after a declared map, RMSE, PWRC or an outlier
//! ratio measure different aspects of agreement with the subjective data, and
//! none is universally best. When one overall order is wanted, each criterion
//! ranks the items ([`ranks_best_first`], 1 = best, tied scores share the
//! average rank) and an item's consensus rank is the **median** of its
//! per-criterion ranks. Items are ordered by that median, ties broken by the
//! **mean** rank; items equal on both share the better rank. The median limits
//! the influence of one outlying criterion, while agreement between criteria
//! carries through.
//!
//! This is a descriptive summary, not a test: it says nothing about whether
//! two adjacent items differ. Use paired cluster-bootstrap differences with
//! [`crate::multiple::holm`] for that (see `docs/EVALUATION_METHODS.md`).
//! Correlated criteria (PLCC and Z-RMSE after the same map, say) count
//! separately, so declare the criterion list with the result.

use crate::panel::ranks;

/// Which direction of a criterion is better.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Better {
    /// Larger values are better (SROCC, PLCC, PWRC).
    Higher,
    /// Smaller values are better (RMSE, outlier ratio, Z-RMSE).
    Lower,
}

/// Why a consensus could not be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConsensusError {
    /// No criteria or no items.
    Empty,
    /// Criterion `index` ranks a different number of items than criterion 0.
    Ragged { index: usize },
    /// A score or rank is not finite (criterion, item).
    NonFinite { criterion: usize, item: usize },
    /// A score passed to [`ranks_best_first`] is not finite (item).
    NonFiniteScore { item: usize },
}

impl core::fmt::Display for ConsensusError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "consensus needs at least one criterion and one item"),
            Self::Ragged { index } => {
                write!(f, "criterion {index} ranks a different number of items")
            }
            Self::NonFinite { criterion, item } => {
                write!(f, "criterion {criterion}, item {item} is not finite")
            }
            Self::NonFiniteScore { item } => write!(f, "score {item} is not finite"),
        }
    }
}

impl std::error::Error for ConsensusError {}

/// Best-first ranks of `scores`: 1 for the best item, average ranks for ties
/// (the [`ranks`] convention, shifted to start at 1). Refuses a score that is
/// not finite ([`ConsensusError::NonFiniteScore`]); drop or impute it
/// explicitly before ranking.
pub fn ranks_best_first(scores: &[f64], better: Better) -> Result<Vec<f64>, ConsensusError> {
    if let Some(item) = scores.iter().position(|v| !v.is_finite()) {
        return Err(ConsensusError::NonFiniteScore { item });
    }
    let oriented: Vec<f64> = match better {
        Better::Higher => scores.iter().map(|v| -v).collect(),
        Better::Lower => scores.to_vec(),
    };
    Ok(ranks(&oriented).into_iter().map(|r| r + 1.0).collect())
}

/// Consensus of per-criterion ranks.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Consensus {
    /// Median of each item's per-criterion ranks (mean of the two middle
    /// ranks for an even number of criteria).
    pub median_rank: Vec<f64>,
    /// Mean of each item's per-criterion ranks (the tie-break).
    pub mean_rank: Vec<f64>,
    /// Final consensus rank, 1 = best. Items equal on both median and mean
    /// share the better (smaller) rank.
    pub rank: Vec<usize>,
}

/// Consensus ranking from `ranks[criterion][item]` (1 = best within each
/// criterion, typically from [`ranks_best_first`]).
pub fn consensus(ranks: &[Vec<f64>]) -> Result<Consensus, ConsensusError> {
    let n_items = ranks.first().map(Vec::len).unwrap_or(0);
    if n_items == 0 {
        return Err(ConsensusError::Empty);
    }
    for (c, r) in ranks.iter().enumerate() {
        if r.len() != n_items {
            return Err(ConsensusError::Ragged { index: c });
        }
        if let Some(item) = r.iter().position(|v| !v.is_finite()) {
            return Err(ConsensusError::NonFinite { criterion: c, item });
        }
    }
    let k = ranks.len();
    let mut median_rank = Vec::with_capacity(n_items);
    let mut mean_rank = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let mut col: Vec<f64> = ranks.iter().map(|r| r[i]).collect();
        col.sort_by(f64::total_cmp);
        median_rank.push(if k % 2 == 1 {
            col[k / 2]
        } else {
            (col[k / 2 - 1] + col[k / 2]) / 2.0
        });
        mean_rank.push(col.iter().sum::<f64>() / k as f64);
    }
    let mut order: Vec<usize> = (0..n_items).collect();
    order.sort_by(|&a, &b| {
        median_rank[a]
            .total_cmp(&median_rank[b])
            .then(mean_rank[a].total_cmp(&mean_rank[b]))
            .then(a.cmp(&b))
    });
    let mut rank = vec![0usize; n_items];
    for (pos, &i) in order.iter().enumerate() {
        rank[i] = if pos > 0 {
            let prev = order[pos - 1];
            if median_rank[prev] == median_rank[i] && mean_rank[prev] == mean_rank[i] {
                rank[prev]
            } else {
                pos + 1
            }
        } else {
            1
        };
    }
    Ok(Consensus {
        median_rank,
        mean_rank,
        rank,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Worked example: three criteria ranking seven metrics. Medians tie for
    /// the second and third metrics (3 and 3); their mean ranks (7/3 and 8/3)
    /// break the tie, giving consensus ranks 1..7 in input order.
    #[test]
    fn median_then_mean_on_a_worked_example() {
        let m1 = vec![2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 7.0];
        let m2 = vec![1.0, 3.0, 4.0, 2.0, 5.0, 7.0, 6.0];
        let m3 = vec![2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let c = consensus(&[m1, m2, m3]).unwrap();
        assert_eq!(c.median_rank, vec![2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(c.rank, vec![1, 2, 3, 4, 5, 6, 7]);
        assert!((c.mean_rank[1] - 7.0 / 3.0).abs() < 1e-15);
        assert!((c.mean_rank[2] - 8.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn best_first_ranks_follow_orientation_and_average_ties() {
        let s = [0.9, 0.7, 0.9, 0.5];
        assert_eq!(
            ranks_best_first(&s, Better::Higher).unwrap(),
            vec![1.5, 3.0, 1.5, 4.0]
        );
        assert_eq!(
            ranks_best_first(&s, Better::Lower).unwrap(),
            vec![3.5, 2.0, 3.5, 1.0]
        );
        assert_eq!(
            ranks_best_first(&[0.1, f64::NAN], Better::Higher),
            Err(ConsensusError::NonFiniteScore { item: 1 })
        );
    }

    /// Even criterion count: the median is the mean of the two middle ranks;
    /// items equal on median and mean share the better rank, and the next
    /// item skips accordingly (competition ranking).
    #[test]
    fn even_criteria_and_full_ties() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 1.0, 3.0];
        let c = consensus(&[a, b]).unwrap();
        assert_eq!(c.median_rank, vec![1.5, 1.5, 3.0]);
        assert_eq!(c.rank, vec![1, 1, 3]);
    }

    #[test]
    fn refuses_empty_ragged_and_nonfinite() {
        assert_eq!(consensus(&[]), Err(ConsensusError::Empty));
        assert_eq!(consensus(&[vec![]]), Err(ConsensusError::Empty));
        assert_eq!(
            consensus(&[vec![1.0, 2.0], vec![1.0]]),
            Err(ConsensusError::Ragged { index: 1 })
        );
        assert_eq!(
            consensus(&[vec![1.0, f64::INFINITY]]),
            Err(ConsensusError::NonFinite {
                criterion: 0,
                item: 1
            })
        );
    }
}
