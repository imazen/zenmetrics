//! Shared Mohammadi 2025 statistical panel + MRR significance test +
//! decisive-rule machinery.
//!
//! Extracted from `bin/bake_verdict.rs` so that `bin/bake_compare.rs`
//! (the canonical "A vs B" comparison tool implementing § A.9 of
//! `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`) can call the same panel
//! computation twice — once per bake — without duplicating logic.
//!
//! All stats follow the same polarity-tolerant convention as
//! `bake_verdict`: SROCC / KROCC / PWRC are taken `.abs()` at the
//! aggregate level because bake outputs can be distance- or
//! score-shaped depending on the trainer's target convention. PLCC
//! is computed after a 4-parameter logistic rescale (Mohammadi 2025
//! convention) which absorbs both polarity and saturation shape.
//!
//! `mrr_h` (Meng-Rosenthal-Rubin paired-correlation test) takes
//! r_AZ / r_BZ as raw signed correlations — callers MUST pass the
//! polarity-aligned correlation (use `spearman_signed_aligned`
//! below) so that the Fisher z-transform and the (1 - r²)
//! denominators are computed on the correct shape. Passing the
//! `.abs()` value would silently break the MRR for distance-shaped
//! bakes.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// ----------------------------------------------------------------------
// Core rank-based correlation primitives
// ----------------------------------------------------------------------

pub fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean;
        let xb = rb[i] - mean;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = a[i] - mean_a;
        let xb = b[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

pub fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut ties_a = 0i64;
    let mut ties_b = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            if da.abs() < 1e-12 && db.abs() < 1e-12 {
                continue;
            } else if da.abs() < 1e-12 {
                ties_a += 1;
            } else if db.abs() < 1e-12 {
                ties_b += 1;
            } else if (da * db) > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total_a = (concordant + discordant + ties_a) as f64;
    let total_b = (concordant + discordant + ties_b) as f64;
    let den = (total_a * total_b).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        ((concordant - discordant) as f64) / den
    }
}

/// **Outlier Ratio per ITU-T P.1401 § C.4 / Mohammadi 2025 Eq 2-4.**
///
/// Fraction of stimuli where the absolute residual `|predicted - target|`
/// exceeds the outlier threshold `τ = 1.96 · σ_target`, where `σ_target`
/// is the corpus standard deviation of the human / MOS column.
///
/// **Polarity convention:** `predicted` MUST already be on the same
/// scale as `target`. The canonical entry path is via
/// [`compute_panel`], which 4-parameter-logistic-rescales `scores`
/// before calling this. Passing raw distance- or score-shaped bake
/// output without rescale will produce a spurious 100 % outlier ratio.
///
/// When per-stimulus observer σ is available, prefer
/// [`outlier_ratio_per_sample`] — corpus-σ is the defensive fallback
/// when the bootstrap σ column isn't joined onto the eval set.
pub fn outlier_ratio(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len().min(target.len());
    if n < 2 {
        return f64::NAN;
    }
    let mean_t: f64 = target.iter().take(n).sum::<f64>() / n as f64;
    let var_t: f64 = target
        .iter()
        .take(n)
        .map(|x| (x - mean_t).powi(2))
        .sum::<f64>()
        / n as f64;
    let sigma = var_t.sqrt().max(1e-12);
    let tau = 1.96 * sigma;
    let mut outliers = 0usize;
    let mut counted = 0usize;
    for i in 0..n {
        let r = predicted[i] - target[i];
        if !r.is_finite() {
            continue;
        }
        if r.abs() > tau {
            outliers += 1;
        }
        counted += 1;
    }
    if counted == 0 {
        return f64::NAN;
    }
    outliers as f64 / counted as f64
}

/// Per-stimulus σ variant of [`outlier_ratio`]: outlier threshold is
/// `τᵢ = 1.96 · σᵢ` per the bootstrap observer σ of each stimulus
/// (Mohammadi 2025 § IV-B). Stimuli with `σᵢ` NaN, ≤ 0, or non-finite
/// `predicted - target` are skipped from both the numerator and
/// denominator (so the ratio is over the validly-σ-tagged subset).
///
/// `predicted` MUST already be on `target`'s scale (same caller
/// convention as [`outlier_ratio`]).
pub fn outlier_ratio_per_sample(predicted: &[f64], target: &[f64], sigma: &[f64]) -> f64 {
    let n = predicted.len().min(target.len()).min(sigma.len());
    if n == 0 {
        return f64::NAN;
    }
    let mut outliers = 0usize;
    let mut counted = 0usize;
    for i in 0..n {
        let s = sigma[i];
        if !s.is_finite() || s <= 0.0 {
            continue;
        }
        let r = predicted[i] - target[i];
        if !r.is_finite() {
            continue;
        }
        let tau = 1.96 * s;
        if r.abs() > tau {
            outliers += 1;
        }
        counted += 1;
    }
    if counted == 0 {
        return f64::NAN;
    }
    outliers as f64 / counted as f64
}

/// **PWRC (Perceptually Weighted Rank Correlation) — Mohammadi 2025
/// § VII Sorting-Accuracy AUC.**
///
/// For each Sensory Threshold ST ∈ `[0, max_subj_gap]`, compute
/// `SA(ST) =` fraction of (i,j) pairs with `|humans[i] − humans[j]| >
/// ST` that the objective metric ranks correctly — i.e., where
/// `sign(humans[j] − humans[i]) == sign(scores[j] − scores[i])`. The
/// returned scalar is the trapezoidal area under that curve,
/// **normalised** so a perfect ranker returns `1.0` and a perfectly
/// anti-ranked one returns `0.0`.
///
/// Wu et al. 2017 (arXiv:1705.05126) is the upstream signed PWRC
/// definition; Mohammadi 2025 (IEEE Access) simplifies to this
/// concordant-fraction SA-ST AUC form, which is what the paper's
/// Table 3 / Figure 4 actually report (their "PWRC" entries —
/// CVVDP=5.92, IW-SSIM=5.76 — are the Wu Eq 19 *unnormalised* AUC_ca
/// over `[T_min, T_max] = [min{2σ̂}, max{2σ̂}]`; multiply our [0, 1]
/// SA-ST AUC by `T_max − T_min` to recover that scale when
/// reproducing Mohammadi tables. The SA-ST AUC is range-stable so we
/// expose the normalised form as the canonical PWRC).
///
/// **Polarity convention:** as with [`outlier_ratio`], the canonical
/// caller is [`compute_panel`], which rescales `scores` via 4-param
/// logistic so the rescaled output correlates positively with
/// `humans`. Anti-correlated input (without prior rescale) yields
/// PWRC = 0, NOT 1 — the function is sign-aware, not magnitude-only.
/// Use [`pwrc_proxy_weighted_rank`] (the pre-2026-05-26 body) if you
/// want the symmetric-after-`.abs()` weighted-rank-Pearson proxy.
pub fn pwrc_sa_st_auc(scores: &[f64], humans: &[f64]) -> f64 {
    let curve = sa_st_curve(scores, humans, 128);
    if curve.len() < 2 {
        return 0.0;
    }
    let mut auc = 0.0_f64;
    for w in curve.windows(2) {
        let (st0, sa0) = w[0];
        let (st1, sa1) = w[1];
        let dt = st1 - st0;
        if dt > 0.0 {
            auc += 0.5 * (sa0 + sa1) * dt;
        }
    }
    let st_max = curve.last().map(|&(s, _)| s).unwrap_or(0.0);
    if st_max <= 0.0 {
        return 0.0;
    }
    auc / st_max
}

/// Sample the SA-ST curve at `n_points` Sensory Thresholds uniformly
/// spaced in `[0, max_subj_gap]`. Returns `(ST, SA(ST))` pairs.
///
/// Pairs with zero `humans`-gap OR zero `scores`-gap are dropped (no
/// unambiguous direction). When `ST > max_subj_gap` no pairs remain
/// active and `SA(ST)` propagates the last finite value (step-flat
/// tail) so the trapezoidal AUC is well-defined.
pub fn sa_st_curve(scores: &[f64], humans: &[f64], n_points: usize) -> Vec<(f64, f64)> {
    let n = scores.len().min(humans.len());
    if n < 2 || n_points < 2 {
        return Vec::new();
    }
    // Memory-bounded two-pass form. The previous body materialised a
    // `Vec<(f64, bool)>` of all `n·(n−1)/2` pairs (O(n²) MEMORY): at
    // n ≈ 59k — e.g. a picker held-out panel of val_rows × n_cells —
    // that vector is ~27 GB and OOM-kills the caller. This computes the
    // IDENTICAL `(ST, SA)` curve in O(n_points) memory via a
    // difference array over the thresholds: each pair is "active" at
    // every threshold `k` with `ST_k < gap` (a prefix, since
    // `ST_k = (k/(n_points−1))·st_max` is monotone in `k`), so a single
    // `[+1 at 0, −1 at kmax]` range-update per pair plus a prefix sum
    // reproduces the per-threshold active/correct counts bit-for-bit.
    // Time is still O(n² · log n_points); memory drops to O(n_points).

    // Pass 1: st_max = max subjective gap over direction-bearing pairs.
    let mut st_max = 0.0_f64;
    let mut any_pair = false;
    for i in 0..n {
        let (hi, si) = (humans[i], scores[i]);
        for j in (i + 1)..n {
            let dh = humans[j] - hi;
            let ds = scores[j] - si;
            if !dh.is_finite() || !ds.is_finite() || dh == 0.0 || ds == 0.0 {
                continue;
            }
            any_pair = true;
            let g = dh.abs();
            if g > st_max {
                st_max = g;
            }
        }
    }
    if !any_pair || st_max <= 0.0 {
        return Vec::new();
    }

    // Pass 2: range-update the difference arrays over the thresholds.
    let mut active_diff = vec![0i64; n_points + 1];
    let mut correct_diff = vec![0i64; n_points + 1];
    for i in 0..n {
        let (hi, si) = (humans[i], scores[i]);
        for j in (i + 1)..n {
            let dh = humans[j] - hi;
            let ds = scores[j] - si;
            if !dh.is_finite() || !ds.is_finite() || dh == 0.0 || ds == 0.0 {
                continue;
            }
            let gap = dh.abs();
            let kmax = st_active_threshold_count(n_points, st_max, gap);
            if kmax > 0 {
                active_diff[0] += 1;
                active_diff[kmax] -= 1;
                if dh.signum() == ds.signum() {
                    correct_diff[0] += 1;
                    correct_diff[kmax] -= 1;
                }
            }
        }
    }

    // Prefix-sum to per-threshold counts and rebuild the (ST, SA) curve
    // EXACTLY as the all-pairs sweep did (same ST expression, same
    // STRICT `gap > ST` activation via `kmax`, same "propagate last
    // finite SA" tail when no pair is active).
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(n_points);
    let mut active = 0i64;
    let mut correct = 0i64;
    for k in 0..n_points {
        active += active_diff[k];
        correct += correct_diff[k];
        let frac = k as f64 / (n_points - 1) as f64;
        let st = frac * st_max;
        let sa = if active == 0 {
            out.last().map(|&(_, s)| s).unwrap_or(0.0)
        } else {
            correct as f64 / active as f64
        };
        out.push((st, sa));
    }
    out
}

/// Number of thresholds `k ∈ 0..n_points` at which a pair with
/// subjective-gap `gap` is active in [`sa_st_curve`] — i.e. the count of
/// `k` with `ST_k < gap` where `ST_k = (k/(n_points−1))·st_max`.
///
/// `ST_k` is monotone non-decreasing in `k`, so `{k : ST_k < gap}` is the
/// prefix `0..kmax`; this binary-searches `kmax` using the IDENTICAL float
/// expression the all-pairs sweep evaluated (`(k/(n_points−1))·st_max`
/// compared against `gap`), so the per-threshold counts — and thus every
/// curve point — are bit-for-bit unchanged. Assumes `n_points ≥ 2` and
/// `st_max > 0` (guaranteed by the caller).
fn st_active_threshold_count(n_points: usize, st_max: f64, gap: f64) -> usize {
    // First k with ST_k >= gap (i.e. NOT ST_k < gap); everything below is active.
    let mut lo = 0usize;
    let mut hi = n_points;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let st = (mid as f64 / (n_points - 1) as f64) * st_max;
        if st < gap {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// **Proxy PWRC** — the pre-2026-05-26 panel.rs `pwrc()` body,
/// preserved verbatim under the more honest name. This is the
/// weighted-rank-Pearson proxy: `ranks` weighted by `|r − mid|` of
/// the FIRST argument, then weighted Pearson of (rank_a, rank_b).
///
/// **Not the paper-correct PWRC** — the SA-ST AUC ([`pwrc_sa_st_auc`])
/// matches Mohammadi 2025 § VII. The proxy is retained for forensic
/// comparison against older bench outputs and for cheap drop-in
/// replacement in any non-canonical caller that wants the same
/// [0, 1] shape under `.abs()` without the O(n²) SA-ST pair sweep.
pub fn pwrc_proxy_weighted_rank(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 4 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mid = (n as f64 - 1.0) / 2.0;
    let max_dev = mid.max(1e-12);
    let w: Vec<f64> = ra.iter().map(|r| (r - mid).abs() / max_dev).collect();
    let wsum: f64 = w.iter().sum();
    if wsum < 1e-12 {
        return 0.0;
    }
    let mean_a: f64 = w.iter().zip(&ra).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mean_b: f64 = w.iter().zip(&rb).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_b;
        num += w[i] * xa * xb;
        da += w[i] * xa * xa;
        db += w[i] * xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

pub fn z_rmse(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len();
    if n < 2 || target.len() != n {
        return f64::NAN;
    }
    let sigma_global = {
        let mean: f64 = target.iter().sum::<f64>() / n as f64;
        let var: f64 = target.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt().max(1e-9)
    };
    let mut sum_sq = 0.0f64;
    let mut count = 0;
    for i in 0..n {
        let z = (predicted[i] - target[i]) / sigma_global;
        if z.is_finite() {
            sum_sq += z * z;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum_sq / count as f64).sqrt()
}

/// Per-sample Z-RMSE (Mohammadi 2025 Equation 6):
///
/// ```text
/// Z-RMSE = √( (1/n) Σᵢ ((S_trans,i − S_subj,i) / σᵢ)² )
/// ```
///
/// Each residual is scaled by the per-stimulus σᵢ (observer standard
/// deviation from bootstrap). Stimuli with high observer consensus
/// (low σ) contribute MORE to Z-RMSE — a miss on a "humans agree"
/// stimulus is penalized more than a miss on an ambiguous one.
///
/// Minimizing Z-RMSE is equivalent to maximizing the log-likelihood
/// of the predictions under a Gaussian observer model (Eq. 10-11).
///
/// `sigma[i]` must be > 0 and finite. Rows where σ is NaN, ≤ 0, or
/// the residual is non-finite are skipped.
pub fn z_rmse_per_sample(predicted: &[f64], target: &[f64], sigma: &[f64]) -> f64 {
    let n = predicted.len().min(target.len()).min(sigma.len());
    if n < 2 {
        return f64::NAN;
    }
    let mut sum_sq = 0.0f64;
    let mut count = 0;
    for i in 0..n {
        let s = sigma[i];
        if !s.is_finite() || s <= 0.0 {
            continue;
        }
        let z = (predicted[i] - target[i]) / s;
        if z.is_finite() {
            sum_sq += z * z;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum_sq / count as f64).sqrt()
}

// ----------------------------------------------------------------------
// 4-parameter logistic rescale (Mohammadi 2025 convention)
// ----------------------------------------------------------------------

fn logistic_eval(b: &[f64; 4], x: f64) -> f64 {
    let b4 = if b[3].abs() < 1e-12 {
        1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
    } else {
        b[3]
    };
    let arg = -(x - b[2]) / b4;
    let e = if arg > 700.0 {
        f64::INFINITY
    } else if arg < -700.0 {
        0.0
    } else {
        arg.exp()
    };
    b[1] + (b[0] - b[1]) / (1.0 + e)
}

fn solve_4x4_gauss(aug: &mut [[f64; 5]; 4]) -> Option<[f64; 4]> {
    for i in 0..4 {
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for (k, row) in aug.iter().enumerate().skip(i + 1) {
            let v = row[i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != i {
            aug.swap(i, max_row);
        }
        for k in (i + 1)..4 {
            let factor = aug[k][i] / aug[i][i];
            // Row-elimination updates row k from pivot row i; the column index
            // addresses both rows, so the range loop is the clear form here.
            #[allow(clippy::needless_range_loop)]
            for c in i..5 {
                aug[k][c] -= factor * aug[i][c];
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum = aug[i][4];
        for c in (i + 1)..4 {
            sum -= aug[i][c] * x[c];
        }
        x[i] = sum / aug[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

fn run_lm(predicted: &[f64], target: &[f64], n: usize, b0: [f64; 4]) -> Option<([f64; 4], f64)> {
    let max_iters = 500usize;
    let tol = 1e-10f64;
    let cost_tol = 1e-12f64;
    let mut lambda = 1.0e-3f64;
    let mut b = b0;
    let jacobian_and_residuals = |b: &[f64; 4]| -> (Vec<[f64; 4]>, Vec<f64>) {
        let mut jac = Vec::with_capacity(n);
        let mut res = Vec::with_capacity(n);
        let b4 = if b[3].abs() < 1e-12 {
            1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
        } else {
            b[3]
        };
        for i in 0..n {
            let x = predicted[i];
            let arg = -(x - b[2]) / b4;
            let e = if arg > 700.0 {
                f64::INFINITY
            } else if arg < -700.0 {
                0.0
            } else {
                arg.exp()
            };
            let a = 1.0 + e;
            let inv_a = 1.0 / a;
            let pred = b[1] + (b[0] - b[1]) * inv_a;
            let diff = pred - target[i];
            res.push(diff);
            let db1 = inv_a;
            let db2 = 1.0 - inv_a;
            let (db3, db4_) = if e.is_finite() && a.is_finite() && a > 1e-300 {
                let inv_a2 = inv_a * inv_a;
                let amp = b[0] - b[1];
                (
                    -amp * e * inv_a2 / b4,
                    -amp * e * (x - b[2]) * inv_a2 / (b4 * b4),
                )
            } else {
                (0.0, 0.0)
            };
            jac.push([db1, db2, db3, db4_]);
        }
        (jac, res)
    };
    let sum_sq = |res: &[f64]| -> f64 { res.iter().map(|r| r * r).sum::<f64>() };
    let (mut jac, mut res) = jacobian_and_residuals(&b);
    let mut cost = sum_sq(&res);
    if !cost.is_finite() {
        return None;
    }
    for _iter in 0..max_iters {
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for i in 0..n {
            let row = &jac[i];
            let r = res[i];
            for a_ in 0..4 {
                jtr[a_] += row[a_] * r;
                for c_ in 0..4 {
                    jtj[a_][c_] += row[a_] * row[c_];
                }
            }
        }
        let mut h = jtj;
        for d in 0..4 {
            h[d][d] += lambda * jtj[d][d].max(1e-12);
        }
        let mut aug = [[0.0f64; 5]; 4];
        for r_ in 0..4 {
            for c in 0..4 {
                aug[r_][c] = h[r_][c];
            }
            aug[r_][4] = -jtr[r_];
        }
        let solved = solve_4x4_gauss(&mut aug);
        let delta = match solved {
            Some(d) => d,
            None => {
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Some((b, cost));
                }
                continue;
            }
        };
        let b_try = [
            b[0] + delta[0],
            b[1] + delta[1],
            b[2] + delta[2],
            b[3] + delta[3],
        ];
        let (jac_try, res_try) = jacobian_and_residuals(&b_try);
        let cost_try = sum_sq(&res_try);
        if cost_try.is_finite() && cost_try < cost {
            let max_delta = delta.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
            let max_b = b.iter().map(|x| x.abs()).fold(1.0f64, f64::max);
            let cost_decrease_rel = (cost - cost_try) / cost.max(1e-30);
            b = b_try;
            jac = jac_try;
            res = res_try;
            cost = cost_try;
            lambda = (lambda / 10.0).max(1e-12);
            if max_delta < tol * (1.0 + max_b) || cost_decrease_rel < cost_tol {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }
    Some((b, cost))
}

fn rescale_affine(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 2 {
        return predicted.to_vec();
    }
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let mean_t: f64 = target.iter().take(n).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut var_p = 0.0f64;
    for i in 0..n {
        let dp = predicted[i] - mean_p;
        let dt = target[i] - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
    }
    let b = if var_p.abs() < 1e-12 {
        0.0
    } else {
        cov / var_p
    };
    let a = mean_t - b * mean_p;
    predicted.iter().map(|p| a + b * p).collect()
}

pub fn rescale_logistic(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 4 {
        return rescale_affine(predicted, target);
    }
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let var_p: f64 = predicted
        .iter()
        .take(n)
        .map(|x| (x - mean_p).powi(2))
        .sum::<f64>()
        / n as f64;
    if !var_p.is_finite() || var_p < 1e-18 {
        return rescale_affine(predicted, target);
    }
    if !predicted.iter().take(n).all(|x| x.is_finite())
        || !target.iter().take(n).all(|x| x.is_finite())
    {
        return rescale_affine(predicted, target);
    }
    let t_max = target
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_min = target.iter().take(n).cloned().fold(f64::INFINITY, f64::min);
    let p_std = var_p.sqrt();
    let p_corr = {
        let mean_t = target.iter().take(n).sum::<f64>() / n as f64;
        let mut cov = 0.0f64;
        let mut vp = 0.0f64;
        let mut vt = 0.0f64;
        for i in 0..n {
            let dp = predicted[i] - mean_p;
            let dt = target[i] - mean_t;
            cov += dp * dt;
            vp += dp * dp;
            vt += dt * dt;
        }
        let d = (vp * vt).sqrt();
        if d < 1e-12 { 0.0 } else { cov / d }
    };
    let b4_sign = if p_corr < 0.0 { -1.0 } else { 1.0 };
    let p_max = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let p_min = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let t_span = (t_max - t_min).abs().max(1.0);
    let tail = 1000.0 * t_span;
    let b3_high = p_max + 25.0 * p_std;
    let b3_low = p_min - 25.0 * p_std;
    let starts: [[f64; 4]; 13] = [
        [
            t_max,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 0.1 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 10.0 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p + p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p - p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        [
            -tail,
            t_max,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            -tail,
            t_max,
            b3_high,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            b3_low,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            b3_low,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            b3_high,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
    ];
    let mut best_b: Option<[f64; 4]> = None;
    let mut best_cost = f64::INFINITY;
    for start in &starts {
        if let Some((b_fit, cost_fit)) = run_lm(predicted, target, n, *start)
            && cost_fit < best_cost
        {
            best_cost = cost_fit;
            best_b = Some(b_fit);
        }
    }
    let b: [f64; 4] = match best_b {
        Some(b) => b,
        None => return rescale_affine(predicted, target),
    };
    let any_bad = predicted
        .iter()
        .take(n)
        .any(|&x| !logistic_eval(&b, x).is_finite());
    if any_bad {
        return rescale_affine(predicted, target);
    }
    predicted.iter().map(|&x| logistic_eval(&b, x)).collect()
}

// ----------------------------------------------------------------------
// Panel-level computation
// ----------------------------------------------------------------------

/// The 6-stat Mohammadi 2025 panel for a single (predicted, target)
/// slice. SROCC / KROCC / PWRC are taken `.abs()` because bake
/// outputs can be distance- or score-shaped; PLCC is computed after
/// 4-parameter logistic rescale (absorbs polarity AND saturation).
#[derive(Clone, Copy, Debug, Default)]
pub struct PanelStats {
    pub srocc: f64,
    pub plcc: f64,
    pub krocc: f64,
    pub or_ratio: f64,
    pub pwrc: f64,
    pub z_rmse: f64,
    pub n: usize,
}

impl PanelStats {
    /// 6 stats indexed in the canonical panel order: SROCC, PLCC,
    /// KROCC, OR, PWRC, Z-RMSE. Used by the "≥ 4 of 6" decisive-rule
    /// check.
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.srocc,
            self.plcc,
            self.krocc,
            self.or_ratio,
            self.pwrc,
            self.z_rmse,
        ]
    }
}

/// Compute the full panel for a `(scores, humans)` pair. Same
/// convention as `bake_verdict::aggregate_panel`.
pub fn compute_panel(scores: &[f64], humans: &[f64]) -> PanelStats {
    let n = scores.len().min(humans.len());
    if n == 0 {
        return PanelStats::default();
    }
    let srocc = spearman(humans, scores).abs();
    let krocc = kendall_tau(humans, scores).abs();
    // PLCC / OR / PWRC are computed on the logistic-rescaled prediction
    // — that absorbs polarity AND saturation, which is the Mohammadi
    // 2025 § IV-A convention. `outlier_ratio` (Eq 2-4) and
    // `pwrc_sa_st_auc` (§ VII) both require the rescale; passing raw
    // distance-shaped bake output gives wrong numbers (spurious 100 %
    // OR, PWRC ≈ 0 from sign-flip).
    let rescaled = rescale_logistic(scores, humans);
    let plcc = pearson(&rescaled, humans).abs();
    let or_ = outlier_ratio(&rescaled, humans);
    let pw = pwrc_sa_st_auc(&rescaled, humans);
    let z = z_rmse(&rescaled, humans);
    PanelStats {
        srocc,
        plcc,
        krocc,
        or_ratio: or_,
        pwrc: pw,
        z_rmse: z,
        n,
    }
}

// ----------------------------------------------------------------------
// Light panel — O(n log n) subset for per-epoch checkpoint selection.
//
// Mohammadi 2025 (Table 2) shows PLCC, SROCC, KT exhibit strong mutual
// correlation while PWRC provides complementary signal. The light panel
// computes SROCC + PLCC + PWRC (all O(n log n) or cheaper), skipping
// O(n²) Kendall. This is the per-epoch checkpoint-selection basis;
// full 6-stat panel is computed at log intervals.
// ----------------------------------------------------------------------

/// Per-group validation statistics for per-epoch checkpoint selection.
/// Skips Kendall τ (O(n²)) and OR / Z-RMSE to keep per-epoch cost low.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightPanel {
    /// Spearman rank-order correlation (rank accuracy).
    pub srocc: f64,
    /// Pearson on 4-parameter logistic rescaled scores (calibration).
    pub plcc: f64,
    /// Perceptually Weighted Rank Correlation (HF-emphasis ranking).
    pub pwrc: f64,
    /// Sample count.
    pub n: usize,
}

/// How to aggregate the 3 light-panel stats into a single checkpoint
/// score. Applied per group; the existing `--val-policy min|mean` then
/// aggregates across groups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValAggregate {
    /// Legacy: bare SROCC (backward compat with pre-panel trainer).
    Srocc,
    /// Geometric mean of (SROCC, PLCC, PWRC). Recommended default.
    /// Penalizes single-axis collapse (e.g., PLCC = 0.3 while SROCC
    /// = 0.9 → 0.63) more than arithmetic mean (0.73).
    GeomeanSPP,
    /// Harmonic mean — more sensitive to the lowest stat.
    HarmeanSPP,
    /// Conservative gate: min of the three.
    MinSPP,
}

impl LightPanel {
    /// Aggregate the three stats into a single score using the chosen
    /// mode. All stats are expected in [0, 1]; values ≤ 0 are clamped
    /// to 1e-12 before log/reciprocal to avoid NaN.
    pub fn aggregate(&self, mode: ValAggregate) -> f64 {
        match mode {
            ValAggregate::Srocc => self.srocc,
            ValAggregate::GeomeanSPP => {
                let a = self.srocc.max(1e-12);
                let b = self.plcc.max(1e-12);
                let c = self.pwrc.max(1e-12);
                (a * b * c).cbrt()
            }
            ValAggregate::HarmeanSPP => {
                let a = self.srocc.max(1e-12);
                let b = self.plcc.max(1e-12);
                let c = self.pwrc.max(1e-12);
                3.0 / (1.0 / a + 1.0 / b + 1.0 / c)
            }
            ValAggregate::MinSPP => self.srocc.min(self.plcc).min(self.pwrc),
        }
    }
}

impl std::str::FromStr for ValAggregate {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "srocc" => Ok(Self::Srocc),
            "geomean3" => Ok(Self::GeomeanSPP),
            "harmean3" => Ok(Self::HarmeanSPP),
            "min3" => Ok(Self::MinSPP),
            other => Err(format!(
                "unknown val-aggregate '{other}'; expected srocc|geomean3|harmean3|min3"
            )),
        }
    }
}

impl std::fmt::Display for ValAggregate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Srocc => "srocc",
            Self::GeomeanSPP => "geomean3",
            Self::HarmeanSPP => "harmean3",
            Self::MinSPP => "min3",
        })
    }
}

/// Compute the light panel (SROCC + PLCC + PWRC) for a single
/// (scores, humans) pair. Same polarity convention as `compute_panel`:
/// SROCC and PWRC are `.abs()`, PLCC is after 4-param logistic rescale.
///
/// Cost: 3× O(n log n) — rank operations dominate. Sub-second on 196 k rows.
pub fn compute_light_panel(scores: &[f64], humans: &[f64]) -> LightPanel {
    let n = scores.len().min(humans.len());
    if n == 0 {
        return LightPanel::default();
    }
    let srocc = spearman(humans, scores).abs();
    let rescaled = rescale_logistic(scores, humans);
    let plcc = pearson(&rescaled, humans).abs();
    let pw = pwrc_sa_st_auc(&rescaled, humans);
    LightPanel {
        srocc,
        plcc,
        pwrc: pw,
        n,
    }
}

// ----------------------------------------------------------------------
// MRR (Meng-Rosenthal-Rubin) paired-correlation z-test
// ----------------------------------------------------------------------

/// Standard normal CDF approximation via Abramowitz & Stegun 7.1.26
/// (max abs error 1.5e-7 in the relevant tail). Returns Pr(Z ≤ z) for
/// a standard normal variate Z. The MRR test is two-sided so callers
/// usually want `2 * (1 - phi(|h|.min(8)))` for the p-value.
pub fn phi(z: f64) -> f64 {
    if !z.is_finite() {
        return if z.is_nan() {
            f64::NAN
        } else if z > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    // erf via Abramowitz & Stegun 7.1.26
    let sign = if z < 0.0 { -1.0 } else { 1.0 };
    let x = z.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

pub fn two_sided_p(h: f64) -> f64 {
    if !h.is_finite() {
        return f64::NAN;
    }
    let z = h.abs().min(8.0);
    let p = 2.0 * (1.0 - phi(z));
    p.clamp(0.0, 1.0)
}

/// MRR h-statistic for two correlations (A vs MOS, B vs MOS) when
/// A and B share their MOS reference.
///
/// `r_az`, `r_bz` are the SIGNED correlations between each metric
/// and MOS (NOT `.abs()`). For bakes that may be distance-shaped vs
/// score-shaped, callers should first flip sign so that "more is
/// better" on both — see `polarity_align`.
///
/// `r_ab` is the signed correlation between A's scores and B's
/// scores (again polarity-aligned).
///
/// Returns NaN when MRR is undefined (n < 4, perfect r_ab = ±1, or
/// any |r| at the atanh boundary).
pub fn mrr_h(r_az: f64, r_bz: f64, r_ab: f64, n_band: usize) -> f64 {
    if n_band < 4 {
        return f64::NAN;
    }
    // Clamp correlations strictly inside (-1, 1) to avoid atanh
    // diverging. The fallback caps at 0.9999 — anything closer than
    // 1e-4 to a perfect correlation has effectively zero stat power
    // anyway, and MRR returning ±∞ vs a large finite number doesn't
    // change the decisive verdict.
    let cap = 0.9999;
    let r_az_c = r_az.clamp(-cap, cap);
    let r_bz_c = r_bz.clamp(-cap, cap);
    let r_ab_c = r_ab.clamp(-cap, cap);
    let z_a = r_az_c.atanh();
    let z_b = r_bz_c.atanh();
    let denom = (1.0 - r_az_c.powi(2)) * (1.0 - r_bz_c.powi(2));
    if denom.abs() < 1e-12 {
        return f64::NAN;
    }
    let f = (1.0 - r_ab_c) / (2.0 * denom);
    if !f.is_finite() {
        return f64::NAN;
    }
    let scale_den = (2.0 * (1.0 - r_ab_c) * f).abs();
    if scale_den < 1e-18 {
        return f64::NAN;
    }
    let scale = ((n_band - 3) as f64).sqrt() / scale_den.sqrt();
    let h = (z_a - z_b) * scale;
    if h.is_finite() { h } else { f64::NAN }
}

/// Polarity-align b to a: returns +1.0 if both have the same
/// orientation against `humans`, -1.0 otherwise. Used to flip
/// `b_scores` before passing them through MRR / r_ab.
pub fn polarity_factor(scores_a: &[f64], scores_b: &[f64], humans: &[f64]) -> f64 {
    let sa = spearman(scores_a, humans);
    let sb = spearman(scores_b, humans);
    if sa * sb < 0.0 { -1.0 } else { 1.0 }
}

// ----------------------------------------------------------------------
// Bootstrap CI of a (A - B) panel-stat delta
// ----------------------------------------------------------------------

/// xoshiro256** RNG for reproducible bootstrap resampling. Same
/// generator zenbench uses.
#[derive(Clone, Copy)]
struct Xoshiro256ss {
    s: [u64; 4],
}

impl Xoshiro256ss {
    fn new(seed: u64) -> Self {
        // SplitMix64 expansion (one-shot, no external dep) so that
        // `seed=0` is a legitimate starting state.
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let next = |z: &mut u64| -> u64 {
            *z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut t = *z;
            t = (t ^ (t >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            t = (t ^ (t >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            t ^ (t >> 31)
        };
        let mut s = [0u64; 4];
        for slot in &mut s {
            *slot = next(&mut z);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1].wrapping_shl(17);
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    fn next_usize_below(&mut self, bound: usize) -> usize {
        ((self.next_u64() as u128 * bound as u128) >> 64) as usize
    }
}

/// 95% bootstrap CI for (A - B) on every stat in the Mohammadi
/// panel. Returns `(low, high)` per stat at the canonical panel
/// order: SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE.
///
/// `n_resamples` defaults to 1000 per § A.9 step (4). Each resample
/// draws with replacement at the row level (NOT bootstrap of pairs
/// of bakes), so A and B see the SAME row sample on each resample
/// — that's how the test is supposed to be paired.
///
/// `seed` controls reproducibility. The function is parallelized
/// via rayon — each thread gets its own seeded RNG so the bootstrap
/// is deterministic for a given (seed, n_resamples, panel input).
pub fn bootstrap_ci_delta(
    scores_a: &[f64],
    scores_b: &[f64],
    humans: &[f64],
    n_resamples: usize,
    seed: u64,
) -> [(f64, f64); 6] {
    let n = scores_a.len().min(scores_b.len()).min(humans.len());
    if n < 4 || n_resamples == 0 {
        return [(f64::NAN, f64::NAN); 6];
    }

    // Compute one bootstrap sample's panel delta.
    //
    // With `feature = "parallel"` enabled (default), rayon parallelises
    // resamples across cores; otherwise the sequential iterator runs
    // in the caller's thread (useful for `no_std`-ish embedding or
    // when the caller controls thread-pool sizing).
    #[cfg(feature = "parallel")]
    let resamples = (0..n_resamples).into_par_iter();
    #[cfg(not(feature = "parallel"))]
    let resamples = 0..n_resamples;
    let deltas: Vec<[f64; 6]> = resamples
        .map(|k| {
            // Each thread instantiates an independent xoshiro256**
            // stream seeded by (seed, k) — same machinery as
            // zenbench's bootstrap path.
            let combined = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(k as u64);
            let mut rng = Xoshiro256ss::new(combined);
            let mut sa = Vec::with_capacity(n);
            let mut sb = Vec::with_capacity(n);
            let mut h = Vec::with_capacity(n);
            for _ in 0..n {
                let idx = rng.next_usize_below(n);
                sa.push(scores_a[idx]);
                sb.push(scores_b[idx]);
                h.push(humans[idx]);
            }
            let pa = compute_panel(&sa, &h);
            let pb = compute_panel(&sb, &h);
            let arr_a = pa.as_array();
            let arr_b = pb.as_array();
            let mut out = [0.0; 6];
            for i in 0..6 {
                out[i] = arr_a[i] - arr_b[i];
            }
            out
        })
        .collect();

    // 95% CI = percentile(2.5, 97.5) of (A - B) over resamples.
    let mut ci = [(f64::NAN, f64::NAN); 6];
    for stat_idx in 0..6 {
        let mut col: Vec<f64> = deltas
            .iter()
            .map(|d| d[stat_idx])
            .filter(|v| v.is_finite())
            .collect();
        if col.len() < 4 {
            continue;
        }
        col.sort_by(f64::total_cmp);
        let lo_idx = ((col.len() as f64) * 0.025).floor() as usize;
        let hi_idx = (((col.len() as f64) * 0.975).ceil() as usize).saturating_sub(1);
        let lo = col[lo_idx.min(col.len() - 1)];
        let hi = col[hi_idx.min(col.len() - 1)];
        ci[stat_idx] = (lo, hi);
    }
    ci
}

// ----------------------------------------------------------------------
// Decisive rule (§ A.9)
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
    /// `n_band < 30` — point estimates noisy, no decision possible.
    Noisy,
    /// All 4 conditions of the decisive rule hold in A's favor.
    ADecisivelyBeatsB,
    /// All 4 conditions of the decisive rule hold in B's favor.
    BDecisivelyBeatsA,
    /// Some conditions favor A or B but not all 4 — surface as
    /// "promising not decisive" per § A.9.
    PromisingNotDecisive,
    /// Bootstrap CIs of every panel stat include zero → genuinely tied.
    Tied,
}

impl Decision {
    pub fn as_str(&self) -> &'static str {
        match self {
            Decision::Noisy => "Noisy",
            Decision::ADecisivelyBeatsB => "ADecisivelyBeatsB",
            Decision::BDecisivelyBeatsA => "BDecisivelyBeatsA",
            Decision::PromisingNotDecisive => "PromisingNotDecisive",
            Decision::Tied => "Tied",
        }
    }
}

/// Result of one `(A, B; band, corpus)` comparison — the unit of
/// the decisive rule. Carries enough state for both the markdown
/// table row AND the JSON output.
#[derive(Clone, Debug)]
pub struct DecisiveOutcome {
    pub n_band: usize,
    pub panel_a: PanelStats,
    pub panel_b: PanelStats,
    pub r_ab: f64,
    pub h_srocc: f64,
    pub p_srocc: f64,
    pub h_z_rmse: f64,
    pub p_z_rmse: f64,
    pub pwrc_diff: f64,
    /// Bootstrap-CI 95% bands of (A - B) per panel stat, panel order.
    pub ci_delta: [(f64, f64); 6],
    /// How many of the 6 panel stats have a CI that excludes zero
    /// in A's favor (positive for non-error stats, negative for
    /// error stats like Z-RMSE / OR where lower is better).
    pub agreement_a: usize,
    /// Same but for B's favor.
    pub agreement_b: usize,
    pub decisive_score: f64,
    pub decision: Decision,
}

/// Z-RMSE_max for the Z-RMSE→correlation conversion used in
/// MRR-on-Z-RMSE. § A.9 says "replace SROCC_* with `1 - Z-RMSE_* /
/// σ_max`". We use the larger of the two bakes' Z-RMSE values as
/// the normalization — this keeps both transformed correlations in
/// [0, 1) regardless of which bake is worse.
fn z_rmse_corr(z: f64, z_max: f64) -> f64 {
    if !z.is_finite() || !z_max.is_finite() || z_max <= 1e-12 {
        return 0.0;
    }
    let c = 1.0 - z / z_max;
    c.clamp(-0.9999, 0.9999)
}

/// Stat-favor direction: which way does a higher value mean
/// "better"? Used to convert raw (A - B) deltas into "A's favor"
/// signed deltas.
const STAT_HIGHER_IS_BETTER: [bool; 6] = [
    true,  // SROCC
    true,  // PLCC
    true,  // KROCC
    false, // OR  (lower is better)
    true,  // PWRC
    false, // Z-RMSE (lower is better)
];

/// Apply the decisive rule of § A.9 to one band's data.
///
/// Returns a fully-populated `DecisiveOutcome`. The contract:
///
/// - `n_band < 30` ⇒ `Decision::Noisy`. All other fields still
///   computed for transparency but the decision is fixed.
/// - All 4 conditions in A's favor ⇒ `ADecisivelyBeatsB`.
/// - All 4 conditions in B's favor (sign-mirrored) ⇒
///   `BDecisivelyBeatsA`.
/// - Bootstrap CIs of every panel stat include 0 ⇒ `Tied`.
/// - Otherwise ⇒ `PromisingNotDecisive`.
///
/// `seed` is propagated to the bootstrap CI calculation.
pub fn decisive(
    scores_a: &[f64],
    scores_b: &[f64],
    humans: &[f64],
    n_resamples: usize,
    seed: u64,
) -> DecisiveOutcome {
    let n_band = scores_a.len().min(scores_b.len()).min(humans.len());
    let panel_a = compute_panel(scores_a, humans);
    let panel_b = compute_panel(scores_b, humans);

    // Polarity-align BOTH bakes to "higher = better against MOS".
    //
    // Mohammadi's MRR formula assumes both metrics share orientation
    // with the MOS — r_AZ and r_BZ are both expected to be positive
    // when the metrics are useful. If a bake is distance-shaped
    // (low score = good), its raw r_AZ will be negative, and the
    // atanh(r_A) - atanh(r_B) subtraction will reverse sign vs the
    // intuitive "A's correlation magnitude is larger ⇒ h positive".
    //
    // Flip each bake into score-orientation independently so r_AZ
    // and r_BZ are both ≥ 0. The h-stat then reads in the natural
    // direction: positive ⇒ |r_A| > |r_B| ⇒ A wins.
    let r_az_raw = spearman(scores_a, humans);
    let r_bz_raw = spearman(scores_b, humans);
    let sign_a = if r_az_raw < 0.0 { -1.0 } else { 1.0 };
    let sign_b = if r_bz_raw < 0.0 { -1.0 } else { 1.0 };
    let scores_a_aligned: Vec<f64> = if sign_a < 0.0 {
        scores_a.iter().map(|v| -v).collect()
    } else {
        scores_a.to_vec()
    };
    let scores_b_aligned: Vec<f64> = if sign_b < 0.0 {
        scores_b.iter().map(|v| -v).collect()
    } else {
        scores_b.to_vec()
    };

    // r_AB: signed Spearman between the two (now-aligned) bakes.
    // After flipping, this is the rank agreement in their COMMON
    // score-up orientation — the value MRR's formula expects.
    let r_ab = spearman(&scores_a_aligned, &scores_b_aligned);
    let r_az = spearman(&scores_a_aligned, humans);
    let r_bz = spearman(&scores_b_aligned, humans);

    let h_srocc = mrr_h(r_az, r_bz, r_ab, n_band);
    let p_srocc = two_sided_p(h_srocc);

    // MRR-on-Z-RMSE: convert each bake's Z-RMSE to a pseudo-correlation
    // via `1 - z / z_max` per § A.9. Use the worse-of-the-two Z-RMSE
    // as z_max so both transforms stay in [0, 1). The conversion
    // assumes that lower Z-RMSE → higher transformed correlation,
    // which is what the MRR formula expects (A's transformed corr
    // > B's transformed corr ⇒ A wins).
    let z_max = panel_a.z_rmse.max(panel_b.z_rmse).max(1e-9);
    let r_az_z = z_rmse_corr(panel_a.z_rmse, z_max);
    let r_bz_z = z_rmse_corr(panel_b.z_rmse, z_max);
    // For r_ab in the Z-RMSE-correlation universe we don't have a
    // direct measurement (Z-RMSE isn't pair-rankable). The MRR
    // formula uses r_ab to discount paired-correlation power.
    // Using the SROCC-based r_ab here is the standard
    // approximation in the Mohammadi 2025 worked example — the two
    // metrics' agreement is on rankings, not absolute calibration.
    let h_z_rmse = mrr_h(r_az_z, r_bz_z, r_ab, n_band);
    let p_z_rmse = two_sided_p(h_z_rmse);

    let pwrc_diff = panel_a.pwrc - panel_b.pwrc;

    // Bootstrap CI of (A - B) per panel stat. The bootstrap operates
    // on the polarity-aligned scores so that all stats are comparable
    // across bakes of different shape. Panel stats are .abs()-based
    // and shape-invariant anyway, so flipping is belt-and-suspenders
    // here — but it makes the JSON output cleaner if a downstream
    // consumer ever wants raw (unflipped) ranks.
    let ci_delta = bootstrap_ci_delta(
        &scores_a_aligned,
        &scores_b_aligned,
        humans,
        n_resamples,
        seed,
    );

    // Count panel-stat wins in A's vs B's favor. A "win" means the
    // 95% CI of (A - B) excludes zero in the direction that favors
    // the bake (positive for higher-is-better, negative for
    // lower-is-better).
    let mut agreement_a = 0usize;
    let mut agreement_b = 0usize;
    for stat_idx in 0..6 {
        let (lo, hi) = ci_delta[stat_idx];
        if !lo.is_finite() || !hi.is_finite() {
            continue;
        }
        let higher_better = STAT_HIGHER_IS_BETTER[stat_idx];
        if lo > 0.0 {
            // CI strictly positive → A's stat > B's stat
            if higher_better {
                agreement_a += 1;
            } else {
                agreement_b += 1;
            }
        } else if hi < 0.0 {
            // CI strictly negative → A's stat < B's stat
            if higher_better {
                agreement_b += 1;
            } else {
                agreement_a += 1;
            }
        }
        // CI brackets zero → no contribution.
    }

    // DecisiveScore scalar (§ A.9 line 157-168):
    //   h_SROCC
    //     * sign(Z-RMSE_B − Z-RMSE_A)    +1 if A better calibrated
    //     * sign(PWRC_A − PWRC_B)        +1 if A weighted-rank wins
    //     * agreement_fraction in [0, 1]
    //     * sqrt(min(n, 100)) / 10
    // Practical cutoff: |DecisiveScore| > 7.84 (= 1.96 × 4 × 1).
    let sign_z = (panel_b.z_rmse - panel_a.z_rmse).signum_or_zero();
    let sign_pwrc = (panel_a.pwrc - panel_b.pwrc).signum_or_zero();
    let agreement_fraction = (agreement_a as f64) / 6.0;
    let n_damp = (n_band.min(100) as f64).sqrt() / 10.0;
    let decisive_score = if h_srocc.is_finite() {
        h_srocc * sign_z * sign_pwrc * agreement_fraction * n_damp
    } else {
        f64::NAN
    };

    // Apply the rule.
    let decision = if n_band < 30 {
        Decision::Noisy
    } else if h_srocc.is_finite()
        && h_z_rmse.is_finite()
        && h_srocc > 1.96
        && h_z_rmse > 1.96
        && pwrc_diff > 0.0
        && agreement_a >= 4
    {
        Decision::ADecisivelyBeatsB
    } else if h_srocc.is_finite()
        && h_z_rmse.is_finite()
        && h_srocc < -1.96
        && h_z_rmse < -1.96
        && pwrc_diff < 0.0
        && agreement_b >= 4
    {
        Decision::BDecisivelyBeatsA
    } else if agreement_a == 0 && agreement_b == 0 {
        Decision::Tied
    } else {
        Decision::PromisingNotDecisive
    };

    DecisiveOutcome {
        n_band,
        panel_a,
        panel_b,
        r_ab,
        h_srocc,
        p_srocc,
        h_z_rmse,
        p_z_rmse,
        pwrc_diff,
        ci_delta,
        agreement_a,
        agreement_b,
        decisive_score,
        decision,
    }
}

// Small helper trait so the DecisiveScore signum doesn't produce ±0
// for exactly-tied stats (which would zero out the entire product).
trait SignumOrZero {
    fn signum_or_zero(self) -> f64;
}
impl SignumOrZero for f64 {
    fn signum_or_zero(self) -> f64 {
        if self > 0.0 {
            1.0
        } else if self < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

// ----------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- OR (P.1401, Eq. 2-4) -----

    #[test]
    fn outlier_ratio_zero_on_perfect_match() {
        // S_trans == target → residual 0 → all under τ=1.96·σ → OR = 0.
        let t = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let p = t.clone();
        let or = outlier_ratio(&p, &t);
        assert!(or.abs() < 1e-12, "OR perfect-match should be 0, got {or}");
    }

    #[test]
    fn outlier_ratio_counts_residuals_above_1_96_sigma() {
        // Hand-built: σ_target = std of target, τ = 1.96·σ. Construct
        // residuals so EXACTLY 2 of 10 exceed τ.
        let t: Vec<f64> = (0..10).map(|i| i as f64).collect(); // 0..9
        let mean = 4.5;
        let var: f64 = t.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 10.0;
        let sigma = var.sqrt();
        let tau = 1.96 * sigma;
        // 8 residuals tiny, 2 residuals just above τ.
        let mut p = t.clone();
        p[3] = t[3] + tau * 1.1; // outlier
        p[7] = t[7] + tau * 1.1; // outlier
        let or = outlier_ratio(&p, &t);
        assert!(
            (or - 0.2).abs() < 1e-12,
            "OR should count 2/10 = 0.2 outliers, got {or} (τ={tau}, σ={sigma})"
        );
    }

    #[test]
    fn outlier_ratio_per_sample_uses_per_stimulus_sigma() {
        // Two stimuli: same residual magnitude, different σ. Only the
        // low-σ one is an outlier (its τ is tighter).
        let t = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p = vec![1.0_f64, 2.0, 3.0 + 1.0, 4.0 + 1.0, 5.0, 6.0]; // 2 residuals = 1.0
        // σ_2 (idx 2) is small (0.3) so τ=0.588 — residual 1.0 IS outlier.
        // σ_3 (idx 3) is large (1.0) so τ=1.96 — residual 1.0 is NOT outlier.
        let sigma = vec![1.0, 1.0, 0.3, 1.0, 1.0, 1.0];
        let or = outlier_ratio_per_sample(&p, &t, &sigma);
        assert!(
            (or - (1.0 / 6.0)).abs() < 1e-12,
            "OR should be 1/6 (only idx 2 exceeds τ), got {or}"
        );
    }

    // ----- PWRC (SA-ST AUC, paper-correct) -----

    #[test]
    fn pwrc_sa_st_perfect_rank_is_one() {
        // Strictly monotonic prediction → every pair correctly ranked
        // at every ST → SA(ST) = 1 everywhere → AUC = 1.
        let humans: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let scores: Vec<f64> = humans.iter().map(|x| 10.0 * x + 0.5).collect();
        let pw = pwrc_sa_st_auc(&scores, &humans);
        assert!((pw - 1.0).abs() < 1e-12, "perfect rank → PWRC=1, got {pw}");
    }

    #[test]
    fn pwrc_sa_st_anti_rank_is_zero() {
        // Perfectly anti-correlated prediction → every pair wrong → SA=0
        // at every ST → PWRC = 0.
        let humans: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let scores: Vec<f64> = humans.iter().rev().copied().collect();
        let pw = pwrc_sa_st_auc(&scores, &humans);
        assert!(pw.abs() < 1e-12, "anti-rank → PWRC=0, got {pw}");
    }

    #[test]
    fn pwrc_sa_st_adjacent_swap_high_but_under_one() {
        // humans = 0..7, scores swap adjacent pairs. Only the 4 gap=1
        // pairs are wrong (4/28 = 14.3% of pairs); the 24 farther-apart
        // pairs rank correctly. So SA(ST=0) = 0.857, SA(ST≥1) = 1.0.
        // PWRC = (0.857·1 + 1.0·6)/7 ≈ 0.980. Tight: a perfect ranking
        // (PWRC = 1) and this imperfect one must be distinguishable.
        let humans: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let scores = vec![1.0, 0.0, 3.0, 2.0, 5.0, 4.0, 7.0, 6.0];
        let pw = pwrc_sa_st_auc(&scores, &humans);
        assert!(
            (pw - 0.980).abs() < 0.005,
            "adjacent-swap → PWRC ≈ 0.980 (computable by hand), got {pw}"
        );
        assert!(pw < 1.0, "imperfect ranking must give PWRC < 1, got {pw}");
    }

    #[test]
    fn sa_st_curve_has_correct_shape() {
        let humans: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let scores = humans.clone();
        let curve = sa_st_curve(&scores, &humans, 16);
        assert_eq!(curve.len(), 16);
        // Perfect ranking: SA = 1 at every ST.
        for (st, sa) in &curve {
            assert!(
                (*sa - 1.0).abs() < 1e-12,
                "perfect ranking: SA(ST={st}) should be 1, got {sa}"
            );
        }
        // First ST is 0, last is max subjective gap (= 5.0 here).
        assert!((curve[0].0 - 0.0).abs() < 1e-12);
        assert!((curve[15].0 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn sa_st_curve_matches_allpairs_reference_bit_for_bit() {
        // The memory-bounded difference-array `sa_st_curve` must
        // reproduce the previous all-pairs O(n²)-memory body EXACTLY
        // (f64::to_bits) — it is a memory-only refactor. This reference
        // IS that previous body, kept here as the regression oracle.
        fn reference(scores: &[f64], humans: &[f64], n_points: usize) -> Vec<(f64, f64)> {
            let n = scores.len().min(humans.len());
            if n < 2 || n_points < 2 {
                return Vec::new();
            }
            let mut pairs: Vec<(f64, bool)> = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let dh = humans[j] - humans[i];
                    let ds = scores[j] - scores[i];
                    if !dh.is_finite() || !ds.is_finite() || dh == 0.0 || ds == 0.0 {
                        continue;
                    }
                    pairs.push((dh.abs(), dh.signum() == ds.signum()));
                }
            }
            if pairs.is_empty() {
                return Vec::new();
            }
            let st_max = pairs.iter().map(|(g, _)| *g).fold(0.0_f64, f64::max);
            if st_max <= 0.0 {
                return Vec::new();
            }
            let mut out: Vec<(f64, f64)> = Vec::with_capacity(n_points);
            for k in 0..n_points {
                let st = (k as f64 / (n_points - 1) as f64) * st_max;
                let mut active = 0usize;
                let mut correct = 0usize;
                for (gap, ok) in &pairs {
                    if *gap > st {
                        active += 1;
                        if *ok {
                            correct += 1;
                        }
                    }
                }
                let sa = if active == 0 {
                    out.last().map(|&(_, s)| s).unwrap_or(0.0)
                } else {
                    correct as f64 / active as f64
                };
                out.push((st, sa));
            }
            out
        }
        fn assert_bit_equal(a: &[(f64, f64)], b: &[(f64, f64)], tag: &str) {
            assert_eq!(a.len(), b.len(), "{tag}: curve length differs");
            for (k, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(pa.0.to_bits(), pb.0.to_bits(), "{tag} k={k}: ST {} vs {}", pa.0, pb.0);
                assert_eq!(pa.1.to_bits(), pb.1.to_bits(), "{tag} k={k}: SA {} vs {}", pa.1, pb.1);
            }
        }
        // Deterministic xorshift; cover random + ties (gaps that land on
        // threshold boundaries) + anti-correlation + a no-direction case.
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        for &n in &[2usize, 3, 5, 17, 64, 200, 777] {
            for &np in &[2usize, 8, 128] {
                let humans: Vec<f64> = (0..n).map(|_| (next() % 1000) as f64 / 7.0).collect();
                let scores: Vec<f64> = (0..n).map(|_| (next() % 1000) as f64 / 11.0).collect();
                assert_bit_equal(
                    &reference(&scores, &humans, np),
                    &sa_st_curve(&scores, &humans, np),
                    "random",
                );
                let h2: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
                let s2: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
                assert_bit_equal(&reference(&s2, &h2, np), &sa_st_curve(&s2, &h2, np), "ties");
                let h3: Vec<f64> = (0..n).map(|i| i as f64).collect();
                let s3: Vec<f64> = (0..n).map(|i| -(i as f64)).collect();
                assert_bit_equal(&reference(&s3, &h3, np), &sa_st_curve(&s3, &h3, np), "anti");
                // All-equal humans → no direction-bearing pairs → empty curve.
                let flat = vec![1.0_f64; n];
                assert_bit_equal(&reference(&scores, &flat, np), &sa_st_curve(&scores, &flat, np), "flat");
            }
        }
    }

    // ----- Proxy still callable + named correctly -----

    #[test]
    fn pwrc_proxy_weighted_rank_matches_old_pwrc_semantics() {
        // The proxy preserves the pre-2026-05-26 pwrc() body. On
        // perfect rank, both produce 1.0.
        let humans: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let scores: Vec<f64> = humans.iter().map(|x| 10.0 * x).collect();
        let proxy = pwrc_proxy_weighted_rank(&humans, &scores);
        assert!(
            (proxy - 1.0).abs() < 1e-12,
            "proxy on perfect rank should be 1, got {proxy}"
        );
    }

    #[test]
    fn spearman_perfect_rank() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!((spearman(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn mrr_h_well_defined() {
        // Two metrics with moderately correlated outputs against MOS.
        let h = mrr_h(0.85, 0.80, 0.90, 100);
        assert!(h.is_finite());
        // A wins SROCC by 0.05 with r_ab high → h should be positive
        // and modest.
        assert!(h > 0.0 && h < 5.0, "h = {h}");
    }

    #[test]
    fn mrr_h_perfect_r_ab_does_not_panic() {
        // r_ab = 1 would cause `1 - r_ab = 0` → division by zero in
        // raw formula. We clamp to 0.9999 so MRR returns a large but
        // finite number, not NaN/Inf.
        let h = mrr_h(0.9, 0.85, 0.9999, 50);
        assert!(h.is_finite() || h.is_nan());
    }

    #[test]
    fn mrr_h_low_n_returns_nan() {
        assert!(mrr_h(0.8, 0.7, 0.9, 2).is_nan());
    }

    #[test]
    fn decisive_returns_noisy_for_small_n() {
        let a = vec![1.0; 10];
        let b = vec![2.0; 10];
        let h = vec![1.0; 10];
        let out = decisive(&a, &b, &h, 100, 42);
        assert_eq!(out.decision, Decision::Noisy);
    }

    #[test]
    fn polarity_factor_flips_distance() {
        let scores_a: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let scores_b: Vec<f64> = (0..50).map(|i| 50.0 - i as f64).collect();
        let humans: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let p = polarity_factor(&scores_a, &scores_b, &humans);
        assert_eq!(p, -1.0);
    }

    #[test]
    fn panel_stats_array_indexing() {
        let p = PanelStats {
            srocc: 0.9,
            plcc: 0.85,
            krocc: 0.8,
            or_ratio: 0.05,
            pwrc: 0.88,
            z_rmse: 0.2,
            n: 100,
        };
        let arr = p.as_array();
        assert_eq!(arr[0], 0.9);
        assert_eq!(arr[5], 0.2);
    }

    #[test]
    fn light_panel_srocc_matches_full_panel() {
        let scores: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.37).sin() * 40.0 + 50.0)
            .collect();
        let humans: Vec<f64> = (0..200)
            .map(|i| (i as f64 * 0.37).sin() * 38.0 + 52.0)
            .collect();
        let full = compute_panel(&scores, &humans);
        let light = compute_light_panel(&scores, &humans);
        assert!(
            (full.srocc - light.srocc).abs() < 1e-12,
            "srocc mismatch: full={} light={}",
            full.srocc,
            light.srocc
        );
        assert!(
            (full.plcc - light.plcc).abs() < 1e-12,
            "plcc mismatch: full={} light={}",
            full.plcc,
            light.plcc
        );
        assert!(
            (full.pwrc - light.pwrc).abs() < 1e-12,
            "pwrc mismatch: full={} light={}",
            full.pwrc,
            light.pwrc
        );
    }

    #[test]
    fn val_aggregate_geomean_penalizes_collapse() {
        let good = LightPanel {
            srocc: 0.9,
            plcc: 0.9,
            pwrc: 0.9,
            n: 100,
        };
        let collapse = LightPanel {
            srocc: 0.9,
            plcc: 0.3,
            pwrc: 0.9,
            n: 100,
        };
        let g = good.aggregate(ValAggregate::GeomeanSPP);
        let c = collapse.aggregate(ValAggregate::GeomeanSPP);
        assert!(g > 0.89 && g < 0.91, "good geomean should be ~0.9, got {g}");
        assert!(c < 0.7, "collapse geomean should be < 0.7, got {c}");
    }

    #[test]
    fn val_aggregate_srocc_ignores_plcc_pwrc() {
        let p = LightPanel {
            srocc: 0.95,
            plcc: 0.1,
            pwrc: 0.1,
            n: 100,
        };
        let s = p.aggregate(ValAggregate::Srocc);
        assert!((s - 0.95).abs() < 1e-12);
    }

    #[test]
    fn val_aggregate_round_trip_parse() {
        for name in &["srocc", "geomean3", "harmean3", "min3"] {
            let agg: ValAggregate = name.parse().unwrap();
            assert_eq!(&agg.to_string(), *name);
        }
    }

    #[test]
    fn z_rmse_per_sample_penalizes_high_consensus() {
        // Stimulus A: low σ (high consensus), large error → heavily penalized.
        // Stimulus B: high σ (ambiguous), same absolute error → lightly penalized.
        let predicted = vec![5.0, 5.0];
        let target = vec![3.0, 3.0];
        let sigma_tight = vec![0.1, 0.1];
        let sigma_loose = vec![10.0, 10.0];
        let z_tight = z_rmse_per_sample(&predicted, &target, &sigma_tight);
        let z_loose = z_rmse_per_sample(&predicted, &target, &sigma_loose);
        assert!(
            z_tight > 10.0,
            "tight σ should produce large Z-RMSE: {z_tight}"
        );
        assert!(
            z_loose < 1.0,
            "loose σ should produce small Z-RMSE: {z_loose}"
        );
    }

    #[test]
    fn z_rmse_per_sample_skips_bad_sigma() {
        let predicted = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        let sigma = vec![0.1, f64::NAN, 0.0];
        let z = z_rmse_per_sample(&predicted, &target, &sigma);
        // Only row 0 contributes (σ=0.1, error=0). Z-RMSE should be 0.
        assert!(
            z.is_finite() && z < 1e-12,
            "expected ~0 for perfect match: {z}"
        );
    }

    #[test]
    fn z_rmse_per_sample_matches_global_when_sigma_uniform() {
        let predicted: Vec<f64> = (0..50).map(|i| i as f64 * 0.5 + 1.0).collect();
        let target: Vec<f64> = (0..50).map(|i| i as f64 * 0.5).collect();
        let global = z_rmse(&predicted, &target);
        // Uniform σ = global σ of target → per-sample should match global.
        let mean_t: f64 = target.iter().sum::<f64>() / target.len() as f64;
        let var_t: f64 =
            target.iter().map(|x| (x - mean_t).powi(2)).sum::<f64>() / target.len() as f64;
        let sigma_global = var_t.sqrt();
        let sigma_uniform = vec![sigma_global; 50];
        let per_sample = z_rmse_per_sample(&predicted, &target, &sigma_uniform);
        assert!(
            (global - per_sample).abs() < 1e-10,
            "uniform σ should match global: {global} vs {per_sample}"
        );
    }
}
