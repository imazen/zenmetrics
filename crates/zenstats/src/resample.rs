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
//!   is weighted by the product of its two levels' multiplicities.
//!
//! A codec that is applied to only some sources (an "extended" codec nested
//! in its source) should be given a label per (source, codec) so it is not
//! treated as crossed with every source.
//!
//! With very few levels (five sources, say) a cluster interval is descriptive:
//! the bootstrap distribution of a statistic over five units is coarse.
//!
//! The statistic sees **row weights** (non-negative integers stored as `f64`),
//! not an index list, because pigeonhole weights are products. Statistics that
//! need explicit rows can call [`weights_to_indices`].

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
    /// The statistic on the full data (every weight 1).
    pub estimate: f64,
    /// Lower percentile bound.
    pub lo: f64,
    /// Upper percentile bound.
    pub hi: f64,
    /// Resamples on which the statistic was finite. Non-finite resamples are
    /// dropped, never imputed.
    pub n_finite: usize,
    /// Two-sided percentile p-value for "the statistic is zero":
    /// `2 · min(P*(θ ≤ 0), P*(θ ≥ 0))`, capped at 1. Meaningful for paired
    /// differences; with B resamples the smallest value is about 2/B.
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

enum Dense {
    Rows(usize),
    One(Vec<usize>, usize),
    Two(Vec<usize>, usize, Vec<usize>, usize),
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
    match design {
        Design::Rows => {
            if n_rows < 2 {
                return Err(BootstrapError::TooFewLevels {
                    factor: 0,
                    levels: n_rows,
                });
            }
            Ok(Dense::Rows(n_rows))
        }
        Design::OneWay(a) => {
            let (ia, ka) = check(0, a)?;
            Ok(Dense::One(ia, ka))
        }
        Design::TwoWay(a, b) => {
            let (ia, ka) = check(0, a)?;
            let (ib, kb) = check(1, b)?;
            Ok(Dense::Two(ia, ka, ib, kb))
        }
    }
}

fn multiplicities(rng: &mut Xoshiro256ss, k: usize) -> Vec<f64> {
    let mut m = vec![0.0; k];
    for _ in 0..k {
        m[rng.next_usize_below(k)] += 1.0;
    }
    m
}

/// Row weights for one resample.
fn draw_weights(d: &Dense, rng: &mut Xoshiro256ss) -> Vec<f64> {
    match d {
        Dense::Rows(n) => multiplicities(rng, *n),
        Dense::One(ia, ka) => {
            let m = multiplicities(rng, *ka);
            ia.iter().map(|&g| m[g]).collect()
        }
        Dense::Two(ia, ka, ib, kb) => {
            let ma = multiplicities(rng, *ka);
            let mb = multiplicities(rng, *kb);
            ia.iter().zip(ib).map(|(&a, &b)| ma[a] * mb[b]).collect()
        }
    }
}

/// Expand integer row weights into a row-index list (row `i` repeated
/// `weights[i]` times), for statistics that need explicit rows.
pub fn weights_to_indices(weights: &[f64]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, &w) in weights.iter().enumerate() {
        let k = w.max(0.0).round() as usize;
        out.extend(core::iter::repeat_n(i, k));
    }
    out
}

/// Percentile bootstrap of a vector-valued statistic under `design`.
///
/// `stat` receives one weight per row (all ones for the point estimate) and
/// returns any number of values; every call must return the same number.
/// Resample `k` is seeded from `(cfg.seed, k)`, so results do not depend on
/// thread scheduling.
pub fn cluster_bootstrap<F>(
    n_rows: usize,
    design: &Design<'_>,
    cfg: &BootstrapConfig,
    stat: F,
) -> Result<Vec<Interval>, BootstrapError>
where
    F: Fn(&[f64]) -> Vec<f64> + Sync,
{
    if n_rows == 0 {
        return Err(BootstrapError::Empty);
    }
    if cfg.resamples == 0 || !(cfg.level > 0.0 && cfg.level < 1.0) {
        return Err(BootstrapError::BadConfig);
    }
    let d = dense_design(n_rows, design)?;
    let estimate = stat(&vec![1.0; n_rows]);
    let k = estimate.len();

    let one = |r: usize| -> Vec<f64> {
        let mut rng = Xoshiro256ss::new(
            cfg.seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(r as u64),
        );
        stat(&draw_weights(&d, &mut rng))
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
                    n_finite: 0,
                    p_zero: f64::NAN,
                };
            }
            let q = |p: f64| col[((p * n as f64).floor() as usize).min(n - 1)];
            let le = col.iter().filter(|&&x| x <= 0.0).count() as f64 / n as f64;
            let ge = col.iter().filter(|&&x| x >= 0.0).count() as f64 / n as f64;
            Interval {
                estimate: estimate[j],
                lo: q(alpha),
                hi: q(1.0 - alpha),
                n_finite: n,
                p_zero: (2.0 * le.min(ge)).min(1.0),
            }
        })
        .collect())
}

/// Weighted mean, a convenience statistic for weights from [`cluster_bootstrap`].
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

    #[test]
    fn one_way_weights_move_whole_clusters() {
        let labels = [10u32, 10, 10, 20, 20, 30];
        let d = dense_design(6, &Design::OneWay(&labels)).unwrap();
        let mut rng = Xoshiro256ss::new(1);
        for _ in 0..200 {
            let w = draw_weights(&d, &mut rng);
            assert_eq!(w[0], w[1]);
            assert_eq!(w[1], w[2]);
            assert_eq!(w[3], w[4]);
            // three clusters drawn, so the cluster multiplicities sum to 3
            assert_eq!(w[0] + w[3] + w[5], 3.0);
        }
    }

    #[test]
    fn two_way_weight_is_product_of_level_multiplicities() {
        let src = [0u32, 0, 1, 1];
        let codec = [5u32, 6, 5, 6];
        let d = dense_design(4, &Design::TwoWay(&src, &codec)).unwrap();
        let mut rng = Xoshiro256ss::new(3);
        for _ in 0..200 {
            let w = draw_weights(&d, &mut rng);
            // rows (0,5),(0,6),(1,5),(1,6): w00*w11 == w01*w10 for a rank-one product
            assert_eq!(w[0] * w[3], w[1] * w[2]);
            let sa = w[0] + w[1] + w[2] + w[3];
            assert!(sa == 0.0 || sa == 4.0 || sa == 2.0, "{w:?}");
        }
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
        let stat = |w: &[f64]| vec![weighted_mean(&values, w)];
        let rows = cluster_bootstrap(values.len(), &Design::Rows, &cfg(1000), stat).unwrap()[0];
        let clus =
            cluster_bootstrap(values.len(), &Design::OneWay(&src), &cfg(1000), stat).unwrap()[0];
        let wr = rows.hi - rows.lo;
        let wc = clus.hi - clus.lo;
        assert!(wc > 4.0 * wr, "row {wr} vs clustered {wc}");
        assert!((rows.estimate - clus.estimate).abs() < 1e-15);
    }

    #[test]
    fn deterministic_for_a_seed_and_p_zero_flags_a_clear_difference() {
        let values: Vec<f64> = (0..60).map(|i| 1.0 + 0.1 * (i % 7) as f64).collect();
        let src: Vec<u32> = (0..60).map(|i| i / 6).collect();
        let stat = |w: &[f64]| vec![weighted_mean(&values, w)];
        let a = cluster_bootstrap(60, &Design::OneWay(&src), &cfg(500), stat).unwrap();
        let b = cluster_bootstrap(60, &Design::OneWay(&src), &cfg(500), stat).unwrap();
        assert_eq!(a, b);
        assert!(a[0].p_zero < 0.01, "{:?}", a[0]);
        assert!(a[0].lo > 0.0);
    }

    #[test]
    fn refuses_bad_shapes() {
        let stat = |_: &[f64]| vec![0.0];
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
        let varying = |w: &[f64]| vec![0.0; if w.iter().all(|&x| x == 1.0) { 1 } else { 2 }];
        assert_eq!(
            cluster_bootstrap(4, &Design::Rows, &cfg(10), varying),
            Err(BootstrapError::StatisticLength {
                expected: 1,
                got: 2
            })
        );
    }

    #[test]
    fn weights_to_indices_expands_counts() {
        assert_eq!(weights_to_indices(&[2.0, 0.0, 1.0]), vec![0, 0, 2]);
    }
}
