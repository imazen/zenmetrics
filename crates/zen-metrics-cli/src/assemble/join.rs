#![forbid(unsafe_code)]

//! The four corpus-join safety guarantees, ported 1:1 from
//! `zensim/scripts/canonical_corpus/join_safety.py`.
//!
//! Each function here defends a specific corruption mode root-caused in
//! `zensim/benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md`. The Python
//! module made the two modes "structurally impossible to reintroduce" at
//! runtime; this Rust port keeps the same runtime guards AND adds the
//! compile-time [`super::key::PairKey`] defense on top.
//!
//! | Fn | Defends | Python analogue |
//! |---|---|---|
//! | [`safe_join`] | Mode B ref-misjoin + silent metric averaging | `safe_metric_join` |
//! | [`attach_positional`] | misaligned positional attach | `attach_metric_positional` |
//! | [`assert_not_constant_per_ref`] | Mode B (post-hoc detector) | `assert_metric_not_constant_per_ref` |
//! | [`assert_no_leaked_columns`] | Mode A mock leak + human-copy leak | `assert_no_leaked_metric_columns` |

use std::collections::HashMap;

use super::key::PairKey;
use super::table::{AssembleError, Column, Table};

/// How an unmatched target row is treated by [`safe_join`].
///
/// `#[allow(dead_code)]`: [`safe_join`] + this enum are the public per-pair
/// join primitives for the canonical-corpus migration and are exercised by
/// `tests/assemble_join_safety.rs` (a separate crate the lib build can't see).
/// The per-codec assembler joins whole feature tables via the internal
/// `join_generic` path, so the lib itself doesn't call `safe_join` — but it is
/// load-bearing public API and the typed corruption fix lives here.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinHow {
    /// Keep every target row; unmatched rows get `NaN` for the metric column
    /// (`pandas` `how="left"`).
    Left,
    /// Keep only target rows that matched a metric row (`how="inner"`).
    Inner,
}

/// Default metric-name prefixes whose bit-identity to `human_score` signals a
/// target leak (Mode A). Mirrors `join_safety`'s `metric_prefixes`. `mix_*`
/// is deliberately ABSENT — for konjnd-dense / LARGE the anchor `human_score`
/// legitimately IS the active mix target, so a `mix_*` column equaling it is
/// correct, not a leak.
pub const LEAK_METRIC_PREFIXES: &[&str] = &["iwssim", "ssim2", "cvvdp", "butter", "dssim"];

/// The human-anchor column name. A raw metric must never equal it.
pub const HUMAN_SCORE_COL: &str = "human_score";

/// Join `metric[metric_col]` onto `target` keyed on the full [`PairKey`].
///
/// This is the Rust port of `join_safety.safe_metric_join`, and the core
/// corruption fix. file:join.rs is the canonical join entry point.
///
/// Guarantees (each a hard error, never a silent collapse / average):
/// 1. **No ref-only collapse.** Both tables must carry all four per-pair key
///    columns. Enforced statically by [`PairKey`] (no ref-only constructor)
///    AND at runtime by [`PairKey::require_columns`].
/// 2. **No silent averaging.** If the metric side has duplicate rows for any
///    key, that's a hard error — never `groupby().mean()` (which is exactly
///    how Mode B destroyed the signal).
/// 3. The metric column is attached by exact key match; unmatched target
///    rows get `NaN` (`Left`) or are dropped (`Inner`).
///
/// `metric` must carry exactly the [`PairKey`] columns plus `metric_col`
/// (extra columns on `metric` are ignored).
#[allow(dead_code)] // public corpus-join API; see `JoinHow` note above.
pub fn safe_join(
    target: &Table,
    metric: &Table,
    metric_col: &str,
    how: JoinHow,
) -> Result<Table, AssembleError> {
    // (1) Both sides must carry the full per-pair key. PairKey has no
    //     ref-only constructor, so this is the runtime backstop for the
    //     dynamic-column case (a parquet read at runtime whose schema we
    //     can't check at compile time).
    PairKey::require_columns(&format!("safe_join({metric_col:?}) target"), target)?;
    PairKey::require_columns(&format!("safe_join({metric_col:?}) metric"), metric)?;

    let metric_values = metric.column(metric_col).ok_or_else(|| {
        AssembleError::Schema(format!("safe_join: metric table lacks column {metric_col:?}"))
    })?;

    // (2) Build the metric-side index, REFUSING duplicate keys. Averaging
    //     duplicates is exactly the Mode-B broadcast — we error instead.
    let mut index: HashMap<PairKey, usize> = HashMap::with_capacity(metric.num_rows());
    let mut dupes = 0usize;
    for i in 0..metric.num_rows() {
        let k = PairKey::for_row(metric, i);
        if index.insert(k, i).is_some() {
            dupes += 1;
        }
    }
    if dupes > 0 {
        return Err(AssembleError::JoinSafety(format!(
            "safe_join({metric_col:?}): metric side has {dupes} row(s) NOT unique on the \
             per-pair key {cols:?} — averaging them would destroy per-pair signal \
             (the ssim2_gpu ref-misjoin failure mode). De-duplicate the metric source first.",
            cols = PairKey::COLUMNS
        )));
    }

    // (3) Materialise the joined metric column row-by-row over the target.
    let mut matched_rows: Vec<usize> = Vec::with_capacity(target.num_rows());
    let mut joined_vals: Vec<f64> = Vec::with_capacity(target.num_rows());
    for i in 0..target.num_rows() {
        let k = PairKey::for_row(target, i);
        match index.get(&k) {
            Some(&mi) => {
                matched_rows.push(i);
                joined_vals.push(metric_values.f64_at(mi));
            }
            None => {
                if how == JoinHow::Left {
                    matched_rows.push(i);
                    joined_vals.push(f64::NAN);
                }
                // Inner: drop the unmatched target row.
            }
        }
    }

    let mut out = target.take_rows(&matched_rows)?;
    out.set_column(metric_col, Column::F64(joined_vals))?;
    Ok(out)
}

/// Attach a per-pair metric POSITIONALLY (row order == target row order).
///
/// Rust port of `join_safety.attach_metric_positional`. Use ONLY when the
/// target genuinely cannot carry a per-pair key (the KADID/TID features
/// tables emit only `ref_basename`, and the metric was computed in the SAME
/// row order as the `dmos.csv` / `mos_with_names.txt` that produced the
/// features). This is the *correct* attachment for KADID/TID — the one the
/// `fix_kadid_tid_apply_scores.py` repair used after the Mode-B misjoin.
///
/// Errors on a length mismatch so a misalignment cannot slip through
/// silently (a silent truncation is how a positional attach would
/// re-introduce a subtler version of the broadcast).
#[allow(dead_code)] // public corpus-join API; see `JoinHow` note above.
pub fn attach_positional(
    target: &Table,
    values: &[f64],
    metric_col: &str,
) -> Result<Table, AssembleError> {
    if values.len() != target.num_rows() {
        return Err(AssembleError::JoinSafety(format!(
            "attach_positional({metric_col:?}): {} metric values vs {} target rows — \
             positional alignment requires an EXACT row-count match.",
            values.len(),
            target.num_rows()
        )));
    }
    let mut out = target.clone();
    out.set_column(metric_col, Column::F64(values.to_vec()))?;
    Ok(out)
}

/// Raise if `metric_col` is constant within every `ref_col` group — the
/// Mode-B signature, detected post-hoc.
///
/// Rust port of `join_safety.assert_metric_not_constant_per_ref`. The
/// `min_group_size` gate (default 1.5) suppresses the false positive on
/// per-pair score sidecars where each ref key is unique (mean group size
/// near one). There "one value per ref" is trivially true and NOT a misjoin.
/// This is the exact false-positive the root-cause doc section 5 calls out
/// for `ssim2_imazen`.
pub fn assert_not_constant_per_ref(
    label: &str,
    ref_col: &str,
    metric_col: &str,
    table: &Table,
) -> Result<(), AssembleError> {
    assert_not_constant_per_ref_tuned(label, ref_col, metric_col, table, 5, 1.5)
}

/// As [`assert_not_constant_per_ref`] with explicit `min_groups` /
/// `min_group_size` thresholds (exposed for tests).
pub fn assert_not_constant_per_ref_tuned(
    label: &str,
    ref_col: &str,
    metric_col: &str,
    table: &Table,
    min_groups: usize,
    min_group_size: f64,
) -> Result<(), AssembleError> {
    let refs = table
        .column(ref_col)
        .ok_or_else(|| AssembleError::Schema(format!("no ref column {ref_col:?}")))?;
    let vals = table
        .column(metric_col)
        .ok_or_else(|| AssembleError::Schema(format!("no metric column {metric_col:?}")))?;

    // Per-ref set of rounded values (collapse duplicates) + raw group sizes.
    let mut by_ref: HashMap<String, std::collections::HashSet<i64>> = HashMap::new();
    let mut sizes: HashMap<String, usize> = HashMap::new();
    for i in 0..table.num_rows() {
        let v = vals.f64_at(i);
        if !v.is_finite() {
            continue;
        }
        let r = refs.key_at(i);
        // round to 4 decimals, like the Python `round(float(v), 4)`.
        let quantised = (v * 10_000.0).round() as i64;
        by_ref.entry(r.clone()).or_default().insert(quantised);
        *sizes.entry(r).or_insert(0) += 1;
    }
    if by_ref.len() < min_groups {
        return Ok(());
    }
    let total: usize = sizes.values().sum();
    let mean_sz = if sizes.is_empty() {
        0.0
    } else {
        total as f64 / sizes.len() as f64
    };
    if mean_sz <= min_group_size {
        return Ok(()); // per-pair sidecar; test N/A (false-positive gate)
    }
    let n_const = by_ref.values().filter(|s| s.len() == 1).count();
    if n_const == by_ref.len() {
        return Err(AssembleError::JoinSafety(format!(
            "{label} {metric_col:?} is constant within every reference group \
             ({} refs, each 1 unique value, mean group size {mean_sz:.1}) — joined on \
             {ref_col:?} only (ref-vs-ref broadcast). Recompute on the correct \
             (ref, dist) pairs (use safe_join or attach_positional).",
            by_ref.len()
        )));
    }
    Ok(())
}

/// Reject mock columns and `human_score`-identical raw-metric columns.
///
/// Rust port of `join_safety.assert_no_leaked_metric_columns`. Raises on:
///
/// 1. **Any column whose name contains `mock`** (case-insensitive) — Mode A.
///    A mock metric column must never enter a training/canonical corpus; the
///    iwssim leak survived three corpus generations purely because the
///    "mock" qualifier lived only in a filename that got renamed away.
/// 2. **Any RAW-metric column (by [`LEAK_METRIC_PREFIXES`]) BIT-IDENTICAL to
///    `human_score`** — Mode A's target leak. A raw metric is an independent
///    measurement and must never equal the human anchor it predicts.
///
/// Deliberately NOT flagged (legitimate by design — matches the Python):
/// - `mix_*` columns equal to `human_score` (the anchor IS the active mix
///   target for konjnd-dense / LARGE). `mix` is absent from the prefixes.
/// - A perfect *correlation* without bit-identity. safesyn's `human_score`
///   is a linear rescale of `ssim2_gpu` (`/100`), so correlation is 1.0 by
///   design. **Only value bit-identity is the leak signature.**
pub fn assert_no_leaked_columns(label: &str, table: &Table) -> Result<(), AssembleError> {
    assert_no_leaked_columns_with_prefixes(label, table, LEAK_METRIC_PREFIXES)
}

/// As [`assert_no_leaked_columns`] with caller-supplied metric prefixes.
pub fn assert_no_leaked_columns_with_prefixes(
    label: &str,
    table: &Table,
    metric_prefixes: &[&str],
) -> Result<(), AssembleError> {
    // (1) mock columns — forbidden outright.
    for name in table.column_names() {
        if name.to_lowercase().contains("mock") {
            return Err(AssembleError::JoinSafety(format!(
                "{label} carries a MOCK column {name:?}. Mock metric columns must never enter \
                 a training/canonical corpus (the iwssim leak survived three corpus generations \
                 because the 'mock' qualifier lived only in a filename). Drop it, or rename the \
                 validation-only signal so it cannot be mistaken for a real metric and exclude it \
                 from the canonical schema."
            )));
        }
    }

    // (2) human_score-identical raw-metric columns.
    let Some(hs_col) = table.column(HUMAN_SCORE_COL) else {
        return Ok(());
    };
    let hs: Vec<f64> = (0..table.num_rows()).map(|i| hs_col.f64_at(i)).collect();
    let hs_finite = hs.iter().filter(|x| x.is_finite()).count();
    // Match the Python's `< 100` short-circuit: too few finite anchor rows to
    // make a confident bit-identity call.
    if hs_finite < 100 {
        return Ok(());
    }

    for name in table.column_names() {
        if name == HUMAN_SCORE_COL {
            continue;
        }
        let ln = name.to_lowercase();
        let is_metric = metric_prefixes
            .iter()
            .any(|p| ln.starts_with(p) || ln.contains(p));
        if !is_metric {
            continue;
        }
        let col = table.column(name).expect("name from column_names");
        // Compare on finite-in-both rows; require ≥ 100 to decide.
        let mut n_compared = 0usize;
        let mut n_identical = 0usize;
        for (i, &b) in hs.iter().enumerate() {
            let a = col.f64_at(i);
            if a.is_finite() && b.is_finite() {
                n_compared += 1;
                if (a - b).abs() <= 1e-9 {
                    n_identical += 1;
                }
            }
        }
        if n_compared < 100 {
            continue;
        }
        let ident = n_identical as f64 / n_compared as f64;
        if ident > 0.995 {
            return Err(AssembleError::JoinSafety(format!(
                "{label} metric column {name:?} is a bit-identical copy of {HUMAN_SCORE_COL:?} \
                 ({:.1}% identical) — target leak (Mode A). A metric column may never equal the \
                 human anchor it is meant to predict.",
                ident * 100.0
            )));
        }
    }
    Ok(())
}
