//! Bootstrap intervals that resample the units a study actually sampled.
//!
//! An IQA dataset is a sample of source images and of codecs, not of
//! independent rows: every distorted image of one source shares its content,
//! and every image from one codec shares its artefacts. Resampling rows treats
//! a 10-source dataset as if it had hundreds of independent observations and
//! gives intervals that are too narrow for any question about new images or
//! new codecs. [`Design`] names the sampled units:
//!
//! * [`Design::Rows`]: the classic row bootstrap. Kept for comparison and for
//!   data whose rows really are independent.
//! * [`Design::OneWay`]: one clustering factor (usually the source image).
//!   Clusters are drawn with replacement and carry all their rows.
//! * [`Design::TwoWay`]: two crossed factors (source × codec), the pigeonhole
//!   bootstrap of Owen (2007, *Ann. Appl. Stat.* 1(2), doi:10.1214/07-AOAS122).
//!   Each factor's levels are drawn independently with replacement, and a row
//!   is weighted by the product of its two levels' multiplicities. Owen shows
//!   the method is mildly conservative: for a statistic with variance
//!   `(ν_A σ_A² + ν_B σ_B² + σ_E²)/N` the bootstrap estimates
//!   `((ν_A+2) σ_A² + (ν_B+2) σ_B² + 3 σ_E²)/N`, so with few levels it
//!   inflates the source and codec terms (×1.33 and ×1.4 for 6 sources and 5
//!   codecs) and triples the interaction term. A codec comparison driven by
//!   source × codec interaction gets the widest inflation.
//!
//! A codec that is applied to only some sources (an "extended" codec nested
//! in its source) should be given a label per (source, codec) so it is not
//! treated as crossed with every source.
//!
//! With few levels a cluster interval under-covers. In `tests/coverage.rs`
//! (12 sources, a skewed per-source pair statistic, 1000 datasets) the 95 %
//! percentile interval covered 0.884 and a Student interval
//! `estimate ± t(G−1) · se · √(G/(G−1))` from [`Interval::se`] covered 0.911,
//! the same as the exact t interval over the 12 per-source values (0.910).
//! Prefer the Student form with few sources, and with five or so treat any
//! interval as descriptive.
//!
//! # Copies keep their identity
//!
//! The statistic receives a [`Draw`]. A level drawn twice is two *copies*, and
//! statistics that group or pair rows must treat the copies as distinct:
//!
//! * [`Draw::row_weights`] is right for statistics that are sums over rows
//!   (means, error sums).
//! * [`Draw::pair_weight`] is right for statistics over pairs of rows: pairs
//!   inside one copy of a source count once per copy (`m_s`), not `m_s²`.
//! * [`Draw::expand`] materialises the resampled rows with copy-aware group
//!   labels, for statistics such as `per_group_srocc` that group rows or
//!   refit a model (out-of-fold maps inside the bootstrap).

use crate::panel::Xoshiro256ss;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Which units the bootstrap resamples. Labels are arbitrary `u32` values,
/// one per row; they need not be dense.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Design<'a> {
    /// Independent rows.
    Rows,
    /// One clustering factor, e.g. the source image.
    OneWay(&'a [u32]),
    /// Two crossed factors, e.g. source image × codec (pigeonhole bootstrap).
    TwoWay(&'a [u32], &'a [u32]),
}

/// Bootstrap settings.
#[derive(Clone, Copy, Debug)]
pub struct BootstrapConfig {
    /// Number of resamples.
    pub resamples: usize,
    /// RNG seed; the same seed, design and statistic give the same intervals.
    pub seed: u64,
    /// Two-sided coverage of the percentile interval, e.g. 0.95.
    pub level: f64,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            resamples: 2000,
            seed: 0,
            level: 0.95,
        }
    }
}

/// One statistic's estimate and percentile interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Interval {
    /// The statistic on the full data (every multiplicity 1).
    pub estimate: f64,
    /// Lower percentile bound (linear interpolation between order statistics,
    /// Hyndman & Fan type 7, at probability (1 − level)/2).
    pub lo: f64,
    /// Upper percentile bound (same definition, at (1 + level)/2).
    pub hi: f64,
    /// Standard deviation of the finite bootstrap draws (the bootstrap
    /// standard error).
    pub se: f64,
    /// Resamples on which the statistic was finite. Non-finite resamples are
    /// dropped, never imputed.
    pub n_finite: usize,
    /// Two-sided bootstrap p-value for "the statistic is zero":
    /// `2 · min((1 + #{θ* ≤ 0}) / (B + 1), (1 + #{θ* ≥ 0}) / (B + 1))`, capped
    /// at 1, with B the finite resamples. Never exactly 0; the smallest value is
    /// `2 / (B + 1)`. Meaningful for paired differences.
    pub p_zero: f64,
}

/// Why a bootstrap could not run.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum BootstrapError {
    /// No rows.
    Empty,
    /// A label slice's length differs from the number of rows.
    LabelLength {
        factor: usize,
        len: usize,
        rows: usize,
    },
    /// A factor has fewer than two levels, so it cannot be resampled.
    TooFewLevels { factor: usize, levels: usize },
    /// `resamples` is zero or `level` is outside (0, 1).
    BadConfig,
    /// The statistic returned a different number of values on a resample
    /// than on the full data.
    StatisticLength { expected: usize, got: usize },
}

impl core::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => write!(f, "no rows to resample"),
            Self::LabelLength { factor, len, rows } => {
                write!(f, "factor {factor} has {len} labels for {rows} rows")
            }
            Self::TooFewLevels { factor, levels } => {
                write!(
                    f,
                    "factor {factor} has {levels} level(s); at least 2 are needed"
                )
            }
            Self::BadConfig => write!(f, "resamples must be > 0 and level in (0, 1)"),
            Self::StatisticLength { expected, got } => {
                write!(f, "statistic returned {got} values, expected {expected}")
            }
        }
    }
}

impl std::error::Error for BootstrapError {}

/// Dense level index per row, and the number of levels.
fn densify(labels: &[u32]) -> (Vec<usize>, usize) {
    let mut sorted: Vec<u32> = labels.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let idx = labels
        .iter()
        .map(|l| sorted.binary_search(l).expect("label present"))
        .collect();
    (idx, sorted.len())
}

/// A design resolved to dense level indices.
#[derive(Clone, Debug)]
struct Dense {
    n_rows: usize,
    /// `Design::Rows`: the single factor is the row itself.
    rows_design: bool,
    /// Per factor: dense level per row and the number of levels.
    factors: Vec<(Vec<usize>, usize)>,
}

fn dense_design(n_rows: usize, design: &Design<'_>) -> Result<Dense, BootstrapError> {
    let check = |factor: usize, labels: &[u32]| -> Result<(Vec<usize>, usize), BootstrapError> {
        if labels.len() != n_rows {
            return Err(BootstrapError::LabelLength {
                factor,
                len: labels.len(),
                rows: n_rows,
            });
        }
        let (idx, k) = densify(labels);
        if k < 2 {
            return Err(BootstrapError::TooFewLevels { factor, levels: k });
        }
        Ok((idx, k))
    };
    let factors = match design {
        Design::Rows => {
            if n_rows < 2 {
                return Err(BootstrapError::TooFewLevels {
                    factor: 0,
                    levels: n_rows,
                });
            }
            vec![((0..n_rows).collect(), n_rows)]
        }
        Design::OneWay(a) => vec![check(0, a)?],
        Design::TwoWay(a, b) => vec![check(0, a)?, check(1, b)?],
    };
    Ok(Dense {
        n_rows,
        rows_design: matches!(design, Design::Rows),
        factors,
    })
}

/// One bootstrap resample: how many copies of each level of each factor were
/// drawn. See the module docs for which accessor a statistic should use.
///
/// For [`Design::Rows`] the single factor is the row itself.
#[derive(Clone, Debug)]
pub struct Draw<'a> {
    dense: &'a Dense,
    /// Per factor: multiplicity of each level.
    mult: Vec<Vec<f64>>,
    rows: Vec<f64>,
}

impl<'a> Draw<'a> {
    fn new(dense: &'a Dense, mult: Vec<Vec<f64>>) -> Self {
        let rows = (0..dense.n_rows)
            .map(|i| {
                dense
                    .factors
                    .iter()
                    .zip(&mult)
                    .map(|((idx, _), m)| m[idx[i]])
                    .product()
            })
            .collect();
        Self { dense, mult, rows }
    }

    /// Number of rows in the original data.
    pub fn n_rows(&self) -> usize {
        self.dense.n_rows
    }

    /// Copies of each row in this resample (the product of its levels'
    /// multiplicities). Right for statistics that sum over rows.
    pub fn row_weights(&self) -> &[f64] {
        &self.rows
    }

    /// Whether this draw comes from [`Design::Rows`].
    pub fn is_row_design(&self) -> bool {
        self.dense.rows_design
    }

    /// Dense level of row `i` for `factor`.
    pub(crate) fn level(&self, factor: usize, i: usize) -> usize {
        self.dense.factors[factor].0[i]
    }

    /// Multiplicity of a dense level of `factor` in this draw.
    pub(crate) fn multiplicity(&self, factor: usize, level: usize) -> f64 {
        self.mult[factor][level]
    }

    /// Number of factors resampled (1 for `Rows` and `OneWay`, 2 for `TwoWay`).
    pub fn n_factors(&self) -> usize {
        self.dense.factors.len()
    }

    /// Whether rows `i` and `j` share their level of `factor`.
    pub fn same_level(&self, factor: usize, i: usize, j: usize) -> bool {
        let idx = &self.dense.factors[factor].0;
        idx[i] == idx[j]
    }

    /// Weight of the pair (i, j), i ≠ j, in the resampled data with copies
    /// kept distinct.
    ///
    /// * Rows from different sources (factor 0 levels differ) pair across all
    ///   their copies: `wᵢ·wⱼ`.
    /// * Rows of the same source pair only inside one copy of it: `m_s`, times,
    ///   under `TwoWay`, `m_c` for the same codec (inside one codec copy) or
    ///   `m_c·m_c'` for different codecs.
    ///
    /// Pairs between two *different copies of the same source*, and between
    /// different copies of one codec inside one source copy, are dropped: in a
    /// real new dataset two distinct sources are never the same image, so such
    /// pairs would be an artefact of resampling. With every multiplicity 1 the
    /// weight is 1. For [`Design::Rows`] it is `wᵢ·wⱼ`.
    pub fn pair_weight(&self, i: usize, j: usize) -> f64 {
        let levels = |r: usize| -> [usize; 2] {
            let mut l = [0usize; 2];
            for (f, (idx, _)) in self.dense.factors.iter().enumerate().take(2) {
                l[f] = idx[r];
            }
            l
        };
        hierarchical_pair_weight(
            self.dense.rows_design,
            self.dense.factors.len(),
            levels(i),
            levels(j),
            |f, l| self.mult[f][l],
        )
    }

    /// Materialise the resampled rows. Returns, per output row, the original
    /// row index and one copy-aware group label per factor: rows from
    /// different copies of the same level get different labels. Pass a
    /// factor's labels as the groups of `per_group_srocc`, or as the folds of
    /// `jnd::out_of_fold`, so a twice-drawn source counts as two sources.
    pub fn expand(&self) -> Expanded {
        let nf = self.dense.factors.len();
        let mut rows = Vec::new();
        let mut groups: Vec<Vec<u64>> = vec![Vec::new(); nf];
        for i in 0..self.dense.n_rows {
            let levels: Vec<usize> = self.dense.factors.iter().map(|(idx, _)| idx[i]).collect();
            let counts: Vec<usize> = levels
                .iter()
                .zip(&self.mult)
                .map(|(&l, m)| m[l] as usize)
                .collect();
            if counts.contains(&0) {
                continue;
            }
            // every combination of one copy per factor is one output row
            let mut copy = vec![0usize; nf];
            loop {
                rows.push(i);
                for f in 0..nf {
                    groups[f].push(((levels[f] as u64) << 32) | copy[f] as u64);
                }
                let mut f = 0;
                while f < nf {
                    copy[f] += 1;
                    if copy[f] < counts[f] {
                        break;
                    }
                    copy[f] = 0;
                    f += 1;
                }
                if f == nf {
                    break;
                }
            }
        }
        Expanded { rows, groups }
    }
}

/// Pair weight from the two rows' design levels; see [`Draw::pair_weight`].
pub(crate) fn hierarchical_pair_weight(
    rows_design: bool,
    n_factors: usize,
    li: [usize; 2],
    lj: [usize; 2],
    m: impl Fn(usize, usize) -> f64,
) -> f64 {
    let row_w = |l: [usize; 2]| -> f64 { (0..n_factors).map(|f| m(f, l[f])).product() };
    if rows_design || li[0] != lj[0] {
        return row_w(li) * row_w(lj);
    }
    let mut w = m(0, li[0]);
    if n_factors > 1 {
        w *= if li[1] == lj[1] {
            m(1, li[1])
        } else {
            m(1, li[1]) * m(1, lj[1])
        };
    }
    w
}

/// A design resolved to dense levels, for crate-internal precomputation.
#[derive(Clone, Debug)]
pub(crate) struct DesignLevels(Dense);

impl DesignLevels {
    pub(crate) fn new(n_rows: usize, design: &Design<'_>) -> Result<Self, BootstrapError> {
        dense_design(n_rows, design).map(Self)
    }

    /// The draw with every multiplicity 1.
    pub(crate) fn unit_draw(&self) -> Draw<'_> {
        let ones = self.0.factors.iter().map(|(_, k)| vec![1.0; *k]).collect();
        Draw::new(&self.0, ones)
    }
}

/// Rows of a resample with copy-aware labels, from [`Draw::expand`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expanded {
    /// Original row index of each resampled row.
    pub rows: Vec<usize>,
    /// Per factor, a copy-aware level label for each resampled row.
    pub groups: Vec<Vec<u64>>,
}

impl Expanded {
    /// A factor's copy-aware labels squeezed into `u32` (dense, in order of
    /// first appearance), for APIs that take `u32` groups.
    pub fn groups_u32(&self, factor: usize) -> Vec<u32> {
        let mut seen: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        self.groups[factor]
            .iter()
            .map(|g| {
                let next = seen.len() as u32;
                *seen.entry(*g).or_insert(next)
            })
            .collect()
    }
}

fn multiplicities(rng: &mut Xoshiro256ss, k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k];
    for _ in 0..k {
        m[rng.next_usize_below(k)] += 1.0;
    }
    m
}

/// Hyndman & Fan type 7 quantile of sorted data.
fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// Percentile bootstrap of a vector-valued statistic under `design`.
///
/// `stat` receives a [`Draw`] (every multiplicity 1 for the point estimate)
/// and returns any number of values; every call must return the same number.
/// Resample `k` is seeded from `(cfg.seed, k)`, so results do not depend on
/// thread scheduling.
pub fn cluster_bootstrap<F>(
    n_rows: usize,
    design: &Design<'_>,
    cfg: &BootstrapConfig,
    stat: F,
) -> Result<Vec<Interval>, BootstrapError>
where
    F: Fn(&Draw<'_>) -> Vec<f64> + Sync,
{
    if n_rows == 0 {
        return Err(BootstrapError::Empty);
    }
    if cfg.resamples == 0 || !(cfg.level > 0.0 && cfg.level < 1.0) {
        return Err(BootstrapError::BadConfig);
    }
    let d = dense_design(n_rows, design)?;
    let ones: Vec<Vec<f64>> = d.factors.iter().map(|(_, k)| vec![1.0; *k]).collect();
    let estimate = stat(&Draw::new(&d, ones));
    let k = estimate.len();

    let one = |r: usize| -> Vec<f64> {
        let mut rng = Xoshiro256ss::new(
            cfg.seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(r as u64),
        );
        let mult = d
            .factors
            .iter()
            .map(|(_, k)| multiplicities(&mut rng, *k))
            .collect();
        stat(&Draw::new(&d, mult))
    };
    #[cfg(feature = "parallel")]
    let draws: Vec<Vec<f64>> = (0..cfg.resamples).into_par_iter().map(one).collect();
    #[cfg(not(feature = "parallel"))]
    let draws: Vec<Vec<f64>> = (0..cfg.resamples).map(one).collect();

    if let Some(bad) = draws.iter().find(|v| v.len() != k) {
        return Err(BootstrapError::StatisticLength {
            expected: k,
            got: bad.len(),
        });
    }
    let alpha = (1.0 - cfg.level) / 2.0;
    Ok((0..k)
        .map(|j| {
            let mut col: Vec<f64> = draws
                .iter()
                .map(|v| v[j])
                .filter(|x| x.is_finite())
                .collect();
            col.sort_by(f64::total_cmp);
            let n = col.len();
            if n == 0 {
                return Interval {
                    estimate: estimate[j],
                    lo: f64::NAN,
                    hi: f64::NAN,
                    se: f64::NAN,
                    n_finite: 0,
                    p_zero: f64::NAN,
                };
            }
            let le = col.iter().filter(|&&x| x <= 0.0).count() as f64;
            let ge = col.iter().filter(|&&x| x >= 0.0).count() as f64;
            let b1 = n as f64 + 1.0;
            let mean = col.iter().sum::<f64>() / n as f64;
            let se = if n > 1 {
                (col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1) as f64).sqrt()
            } else {
                f64::NAN
            };
            Interval {
                estimate: estimate[j],
                lo: quantile7(&col, alpha),
                hi: quantile7(&col, 1.0 - alpha),
                se,
                n_finite: n,
                p_zero: (2.0 * ((1.0 + le) / b1).min((1.0 + ge) / b1)).min(1.0),
            }
        })
        .collect())
}

/// Weighted mean, a convenience statistic for [`Draw::row_weights`].
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    let (mut s, mut w) = (0.0, 0.0);
    for (&v, &wi) in values.iter().zip(weights) {
        if wi > 0.0 {
            s += v * wi;
            w += wi;
        }
    }
    if w > 0.0 { s / w } else { f64::NAN }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(resamples: usize) -> BootstrapConfig {
        BootstrapConfig {
            resamples,
            seed: 7,
            level: 0.95,
        }
    }

    fn draw_with<'a>(d: &'a Dense, mult: Vec<Vec<f64>>) -> Draw<'a> {
        Draw::new(d, mult)
    }

    #[test]
    fn one_way_weights_move_whole_clusters() {
        let labels = [10u32, 10, 10, 20, 20, 30];
        let d = dense_design(6, &Design::OneWay(&labels)).unwrap();
        let mut rng = Xoshiro256ss::new(1);
        for _ in 0..200 {
            let m = vec![multiplicities(&mut rng, 3)];
            let draw = draw_with(&d, m);
            let w = draw.row_weights();
            assert_eq!(w[0], w[1]);
            assert_eq!(w[1], w[2]);
            assert_eq!(w[3], w[4]);
            assert_eq!(w[0] + w[3] + w[5], 3.0);
        }
    }

    #[test]
    fn two_way_row_weight_is_product_of_level_multiplicities() {
        let src = [0u32, 0, 1, 1];
        let codec = [5u32, 6, 5, 6];
        let d = dense_design(4, &Design::TwoWay(&src, &codec)).unwrap();
        let draw = draw_with(&d, vec![vec![2.0, 0.0], vec![1.0, 1.0]]);
        assert_eq!(draw.row_weights(), &[2.0, 2.0, 0.0, 0.0]);
        let draw = draw_with(&d, vec![vec![1.0, 1.0], vec![0.0, 2.0]]);
        assert_eq!(draw.row_weights(), &[0.0, 2.0, 0.0, 2.0]);
    }

    #[test]
    fn pair_weight_counts_pairs_inside_one_copy() {
        // source 0 has rows 0,1 (codecs 5,6); source 1 has rows 2,3
        let src = [0u32, 0, 1, 1];
        let codec = [5u32, 6, 5, 6];
        let one = dense_design(4, &Design::OneWay(&src)).unwrap();
        let draw = draw_with(&one, vec![vec![2.0, 1.0]]);
        // within source 0, drawn twice: one pair per copy → 2, not 4
        assert_eq!(draw.pair_weight(0, 1), 2.0);
        // across sources: 2 copies × 1 copy
        assert_eq!(draw.pair_weight(0, 2), 2.0);
        let two = dense_design(4, &Design::TwoWay(&src, &codec)).unwrap();
        let draw = draw_with(&two, vec![vec![2.0, 1.0], vec![3.0, 1.0]]);
        // same source, different codec: m_s · m_c · m_c'
        assert_eq!(draw.pair_weight(0, 1), 2.0 * 3.0 * 1.0);
        // different source, same codec: every copy pairs with every copy
        assert_eq!(draw.pair_weight(0, 2), (2.0 * 3.0) * (1.0 * 3.0));
    }

    #[test]
    fn expand_gives_each_copy_its_own_label() {
        let src = [7u32, 7, 9];
        let d = dense_design(3, &Design::OneWay(&src)).unwrap();
        let draw = draw_with(&d, vec![vec![2.0, 0.0]]);
        let e = draw.expand();
        assert_eq!(e.rows, vec![0, 0, 1, 1]);
        // rows 0 and 1 of copy 0 share a label; copy 1 has another
        let g = e.groups_u32(0);
        assert_eq!(g, vec![0, 1, 0, 1]);
        let total: f64 = draw.row_weights().iter().sum();
        assert_eq!(e.rows.len() as f64, total);
    }

    /// Negative control for the whole reason this module exists: with a
    /// strong source effect, the row bootstrap's interval for a mean is far
    /// narrower than the source-clustered one.
    #[test]
    fn clustered_interval_is_wider_than_row_interval_under_a_source_effect() {
        let mut values = Vec::new();
        let mut src = Vec::new();
        for s in 0..8u32 {
            let offset = (s as f64 - 3.5) * 1.0;
            for k in 0..40 {
                values.push(offset + 0.01 * (k as f64 - 20.0));
                src.push(s);
            }
        }
        let stat = |d: &Draw<'_>| vec![weighted_mean(&values, d.row_weights())];
        let rows = cluster_bootstrap(values.len(), &Design::Rows, &cfg(1000), stat).unwrap()[0];
        let clus =
            cluster_bootstrap(values.len(), &Design::OneWay(&src), &cfg(1000), stat).unwrap()[0];
        let wr = rows.hi - rows.lo;
        let wc = clus.hi - clus.lo;
        assert!(wc > 4.0 * wr, "row {wr} vs clustered {wc}");
        assert!((rows.estimate - clus.estimate).abs() < 1e-15);
    }

    #[test]
    fn deterministic_for_a_seed_and_p_zero_is_never_zero() {
        let values: Vec<f64> = (0..60).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
        let src: Vec<u32> = (0..60).map(|i| i / 6).collect();
        let stat = |d: &Draw<'_>| vec![weighted_mean(&values, d.row_weights())];
        let a = cluster_bootstrap(60, &Design::OneWay(&src), &cfg(500), stat).unwrap();
        let b = cluster_bootstrap(60, &Design::OneWay(&src), &cfg(500), stat).unwrap();
        assert_eq!(a, b);
        // every draw is > 0, so the p-value is its floor 2/(B+1), not 0
        assert!((a[0].p_zero - 2.0 / 501.0).abs() < 1e-15, "{:?}", a[0]);
        assert!(a[0].lo > 0.0);
    }

    #[test]
    fn type7_quantile() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile7(&s, 0.0), 1.0);
        assert_eq!(quantile7(&s, 1.0), 5.0);
        assert_eq!(quantile7(&s, 0.5), 3.0);
        assert!((quantile7(&s, 0.1) - 1.4).abs() < 1e-12);
    }

    #[test]
    fn refuses_bad_shapes() {
        let stat = |_: &Draw<'_>| vec![0.0];
        assert_eq!(
            cluster_bootstrap(3, &Design::OneWay(&[1, 1]), &cfg(10), stat),
            Err(BootstrapError::LabelLength {
                factor: 0,
                len: 2,
                rows: 3
            })
        );
        assert_eq!(
            cluster_bootstrap(3, &Design::OneWay(&[4, 4, 4]), &cfg(10), stat),
            Err(BootstrapError::TooFewLevels {
                factor: 0,
                levels: 1
            })
        );
        assert_eq!(
            cluster_bootstrap(0, &Design::Rows, &cfg(10), stat),
            Err(BootstrapError::Empty)
        );
        let varying = |d: &Draw<'_>| {
            vec![
                0.0;
                if d.row_weights().iter().all(|&x| x == 1.0) {
                    1
                } else {
                    2
                }
            ]
        };
        assert_eq!(
            cluster_bootstrap(4, &Design::Rows, &cfg(10), varying),
            Err(BootstrapError::StatisticLength {
                expected: 1,
                got: 2
            })
        );
    }
}
