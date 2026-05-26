#![forbid(unsafe_code)]

//! A minimal, column-oriented in-memory table used by the corpus assembler.
//!
//! # Why a bespoke table and not DuckDB / Polars?
//!
//! The Python builders this module replaces (`build_per_codec_training.py`,
//! the `canonical_corpus` builders) leaned on DuckDB's `union_by_name` to
//! widen dtypes across per-chunk parquet files and on pandas `merge` for the
//! joins. Both are heavy native deps. The task brief is explicit: **do not
//! add DuckDB**, reuse the arrow/parquet stack already vendored by the
//! `sweep` feature. So we do all join + integrity logic over a small,
//! self-describing [`Table`] built on top of `arrow`/`parquet` readers.
//!
//! The table is deliberately simple:
//! - Every column is one of a handful of logical types ([`Column`]).
//! - The only "widening" we perform is the one DuckDB's `union_by_name`
//!   bought us: a column that is numeric in some files and string/null in
//!   others is coerced to `Float64` (with non-parseable cells becoming
//!   `NaN`). That single rule covers the documented failure mode — "a
//!   per-cell score occasionally went null / empty string on metric-fail
//!   rows" — without importing a query engine.
//!
//! The table is row-addressable for joins via the typed [`super::key::PairKey`]
//! and column-addressable for the integrity checks. It is NOT a general
//! dataframe — it has exactly the operations the assembler needs and nothing
//! more, which keeps the corruption-prevention surface small and auditable.

use std::collections::BTreeMap;
use std::fmt;

/// One logical column. Strings and numbers are the only two physical
/// representations we need; everything in the corpus parquets is one or the
/// other (identity-tuple keys are strings/ints, features + scores are floats).
///
/// `Float64` carries `NaN` for null / non-parseable cells — this is the
/// widening rule that replaces DuckDB's `union_by_name`. `Str` carries
/// `Option<String>` so a genuinely-null string stays distinct from `""`.
#[derive(Clone, Debug)]
pub enum Column {
    /// UTF-8 string column. `None` is a SQL null.
    Str(Vec<Option<String>>),
    /// 64-bit float column. `NaN` doubles as the null sentinel — the
    /// assembler never treats `NaN` as a real metric value (the integrity
    /// checks filter non-finite before comparing).
    F64(Vec<f64>),
    /// 64-bit signed integer column (used for `q`). Kept distinct from
    /// `F64` so the round-tripped parquet preserves the integer schema the
    /// downstream trainer expects on the identity tuple.
    I64(Vec<i64>),
}

impl Column {
    /// Row count of this column.
    pub fn len(&self) -> usize {
        match self {
            Column::Str(v) => v.len(),
            Column::F64(v) => v.len(),
            Column::I64(v) => v.len(),
        }
    }

    /// `#[allow(dead_code)]`: clippy pairs this with `len`; kept for the
    /// public Column API even though the assembler always queries `len`.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read row `i` as a string for join-key construction. Numbers are
    /// rendered canonically (integers without a decimal point, floats via
    /// `{}`) so that an `I64` `q` and a string `"50"` key compare equal —
    /// the score sidecars store `q` as `Int64`, while a features table may
    /// carry it as text.
    pub fn key_at(&self, i: usize) -> String {
        match self {
            Column::Str(v) => v[i].clone().unwrap_or_default(),
            Column::I64(v) => v[i].to_string(),
            Column::F64(v) => {
                let f = v[i];
                if f.fract() == 0.0 && f.is_finite() {
                    format!("{}", f as i64)
                } else {
                    format!("{f}")
                }
            }
        }
    }

    /// Materialise row `i` as an `f64` for integrity comparisons. String
    /// cells return `NaN` (they are never a metric value).
    pub fn f64_at(&self, i: usize) -> f64 {
        match self {
            Column::F64(v) => v[i],
            Column::I64(v) => v[i] as f64,
            Column::Str(v) => v[i]
                .as_deref()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(f64::NAN),
        }
    }

    /// Append the cell at `src` index `i` of `other` onto `self`. Both
    /// columns must share the same variant; mismatches are a programming
    /// error (the caller coerces variants before concatenation).
    fn push_from(&mut self, other: &Column, i: usize) -> Result<(), AssembleError> {
        match (self, other) {
            (Column::Str(dst), Column::Str(src)) => dst.push(src[i].clone()),
            (Column::F64(dst), Column::F64(src)) => dst.push(src[i]),
            (Column::I64(dst), Column::I64(src)) => dst.push(src[i]),
            // The only legal cross-variant push is widening to F64 (the
            // union_by_name rule); the caller coerces first, so reaching
            // here is a bug.
            (dst, src) => {
                return Err(AssembleError::Schema(format!(
                    "internal: cannot push {} cell onto {} column",
                    src.variant_name(),
                    dst.variant_name()
                )));
            }
        }
        Ok(())
    }

    fn variant_name(&self) -> &'static str {
        match self {
            Column::Str(_) => "Str",
            Column::F64(_) => "F64",
            Column::I64(_) => "I64",
        }
    }

    /// Coerce this column to `Float64`, parsing strings and widening ints.
    /// This is the single widening rule (see module docs).
    fn into_f64(self) -> Column {
        match self {
            Column::F64(v) => Column::F64(v),
            Column::I64(v) => Column::F64(v.into_iter().map(|x| x as f64).collect()),
            Column::Str(v) => Column::F64(
                v.into_iter()
                    .map(|s| s.and_then(|t| t.parse::<f64>().ok()).unwrap_or(f64::NAN))
                    .collect(),
            ),
        }
    }
}

/// Errors surfaced by the assembler. [`AssembleError::JoinSafety`] carries the
/// integrity-violation messages that are the whole point of this module — they
/// mirror `join_safety.py`'s `JoinSafetyError`.
#[derive(Debug)]
pub enum AssembleError {
    /// A schema / shape problem (missing column, length mismatch, dtype clash).
    Schema(String),
    /// A join-safety / integrity violation — the Rust analogue of
    /// `join_safety.JoinSafetyError`. These are the corruption-prevention
    /// errors: ref-only collapse, duplicate metric keys, leaked columns,
    /// positional length mismatch, constant-per-ref broadcast.
    JoinSafety(String),
    /// An I/O / parquet error.
    Io(String),
}

impl fmt::Display for AssembleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssembleError::Schema(m) => write!(f, "schema error: {m}"),
            AssembleError::JoinSafety(m) => write!(f, "DATA-INTEGRITY: {m}"),
            AssembleError::Io(m) => write!(f, "io error: {m}"),
        }
    }
}

impl std::error::Error for AssembleError {}

/// A column-oriented table with named columns kept in insertion order.
///
/// Insertion order matters: the round-tripped parquet preserves the column
/// ordering so the canonical schema (`ref_basename, human_score, …, f0..f371`)
/// stays stable across builds, exactly as the Python builders documented.
#[derive(Clone, Debug, Default)]
pub struct Table {
    /// Column names in insertion order.
    names: Vec<String>,
    /// Parallel to `names`.
    columns: Vec<Column>,
    /// Row count (all columns share this). 0 for an empty table.
    n_rows: usize,
}

impl Table {
    /// Build a table from `(name, column)` pairs, validating equal lengths.
    pub fn from_columns(cols: Vec<(String, Column)>) -> Result<Self, AssembleError> {
        let n_rows = cols.first().map(|(_, c)| c.len()).unwrap_or(0);
        for (name, c) in &cols {
            if c.len() != n_rows {
                return Err(AssembleError::Schema(format!(
                    "column {name:?} has {} rows, expected {n_rows}",
                    c.len()
                )));
            }
        }
        let (names, columns): (Vec<_>, Vec<_>) = cols.into_iter().unzip();
        Ok(Self {
            names,
            columns,
            n_rows,
        })
    }

    pub fn num_rows(&self) -> usize {
        self.n_rows
    }

    pub fn num_columns(&self) -> usize {
        self.names.len()
    }

    pub fn column_names(&self) -> &[String] {
        &self.names
    }

    pub fn has_column(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    pub fn column(&self, name: &str) -> Option<&Column> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| &self.columns[i])
    }

    /// Insert (or replace) a column. Length must match the existing row
    /// count unless the table is empty.
    pub fn set_column(&mut self, name: &str, col: Column) -> Result<(), AssembleError> {
        if self.n_rows == 0 && self.names.is_empty() {
            self.n_rows = col.len();
        } else if col.len() != self.n_rows {
            return Err(AssembleError::Schema(format!(
                "set_column {name:?}: {} rows vs table {} rows",
                col.len(),
                self.n_rows
            )));
        }
        if let Some(i) = self.names.iter().position(|n| n == name) {
            self.columns[i] = col;
        } else {
            self.names.push(name.to_string());
            self.columns.push(col);
        }
        Ok(())
    }

    /// Rename every column matching `pred` by prefixing it with `prefix`,
    /// leaving names in `keep` untouched. This is the Rust port of
    /// `rename_feat_columns` — it namespaces colliding `feat_<N>` columns
    /// (`zsm_feat_<N>` / `src_feat_<N>`) so a join doesn't clobber them.
    pub fn prefix_columns_where<F>(&mut self, prefix: &str, keep: &[&str], pred: F)
    where
        F: Fn(&str) -> bool,
    {
        for name in &mut self.names {
            if keep.contains(&name.as_str()) {
                continue;
            }
            if pred(name) {
                *name = format!("{prefix}{name}");
            }
        }
    }

    /// Select a subset of rows by index, preserving column order. Used after
    /// a join to materialise matched rows, and to split per-codec.
    pub fn take_rows(&self, indices: &[usize]) -> Result<Table, AssembleError> {
        let mut out_cols: Vec<(String, Column)> = Vec::with_capacity(self.names.len());
        for (name, col) in self.names.iter().zip(self.columns.iter()) {
            let new_col = match col {
                Column::Str(v) => Column::Str(indices.iter().map(|&i| v[i].clone()).collect()),
                Column::F64(v) => Column::F64(indices.iter().map(|&i| v[i]).collect()),
                Column::I64(v) => Column::I64(indices.iter().map(|&i| v[i]).collect()),
            };
            out_cols.push((name.clone(), new_col));
        }
        Table::from_columns(out_cols)
    }

    /// Distinct values of a string column (used to enumerate codecs).
    pub fn distinct_str(&self, name: &str) -> Result<Vec<String>, AssembleError> {
        let col = self
            .column(name)
            .ok_or_else(|| AssembleError::Schema(format!("no column {name:?}")))?;
        let Column::Str(v) = col else {
            return Err(AssembleError::Schema(format!(
                "distinct_str: column {name:?} is not a string column"
            )));
        };
        let mut seen: BTreeMap<String, ()> = BTreeMap::new();
        for cell in v.iter().flatten() {
            seen.insert(cell.clone(), ());
        }
        Ok(seen.into_keys().collect())
    }

    /// Concatenate `others` onto `self` by column name (the `union_by_name`
    /// replacement). Columns present in only some tables are filled with
    /// nulls for the rows where they are absent. When the same column name
    /// has different physical variants across tables, ALL instances are
    /// widened to `Float64` — this is exactly the dtype-promotion DuckDB's
    /// `union_by_name=true` performed for partial / failed cells.
    pub fn union_by_name(tables: Vec<Table>) -> Result<Table, AssembleError> {
        if tables.is_empty() {
            return Ok(Table::default());
        }
        // 1. Collect the union of column names in first-seen order.
        let mut all_names: Vec<String> = Vec::new();
        for t in &tables {
            for n in &t.names {
                if !all_names.contains(n) {
                    all_names.push(n.clone());
                }
            }
        }

        // 2. Decide each column's target variant. If every table that has
        //    the column agrees on a non-float variant, keep it; otherwise
        //    widen to F64. A column absent everywhere can't happen (it came
        //    from the union).
        let mut target_variant: BTreeMap<String, &'static str> = BTreeMap::new();
        for name in &all_names {
            let mut variant: Option<&'static str> = None;
            let mut widen = false;
            for t in &tables {
                if let Some(c) = t.column(name) {
                    let v = c.variant_name();
                    match variant {
                        None => variant = Some(v),
                        Some(prev) if prev == v => {}
                        Some(_) => widen = true,
                    }
                }
            }
            target_variant.insert(
                name.clone(),
                if widen { "F64" } else { variant.unwrap_or("F64") },
            );
        }

        // 3. Build each output column by appending coerced cells.
        let mut out_cols: Vec<(String, Column)> = Vec::with_capacity(all_names.len());
        for name in &all_names {
            let want = target_variant[name];
            let mut acc: Column = match want {
                "Str" => Column::Str(Vec::new()),
                "I64" => Column::I64(Vec::new()),
                _ => Column::F64(Vec::new()),
            };
            for t in &tables {
                match t.column(name) {
                    Some(c) => {
                        // Coerce the source column to the target variant if
                        // needed (only F64 widening is legal).
                        let coerced: Column = if c.variant_name() == want {
                            c.clone()
                        } else if want == "F64" {
                            c.clone().into_f64()
                        } else {
                            return Err(AssembleError::Schema(format!(
                                "union_by_name: column {name:?} has incompatible variant {} \
                                 (target {want}) — only F64 widening is supported",
                                c.variant_name()
                            )));
                        };
                        for i in 0..t.n_rows {
                            acc.push_from(&coerced, i)?;
                        }
                    }
                    None => {
                        // Column absent in this table → null-fill its rows.
                        match &mut acc {
                            Column::Str(d) => {
                                for _ in 0..t.n_rows {
                                    d.push(None);
                                }
                            }
                            Column::F64(d) => {
                                for _ in 0..t.n_rows {
                                    d.push(f64::NAN);
                                }
                            }
                            Column::I64(_) => {
                                // An I64 column missing in some table can't
                                // be null-filled (no I64 null sentinel) — the
                                // variant decision above would have widened it
                                // to F64 if it were absent anywhere AND
                                // present-as-int elsewhere only when all
                                // present agreed; to be safe, treat absence as
                                // a widen trigger.
                                return Err(AssembleError::Schema(format!(
                                    "union_by_name: integer column {name:?} is absent in one \
                                     input; promote it to F64 upstream so nulls are representable"
                                )));
                            }
                        }
                    }
                }
            }
            out_cols.push((name.clone(), acc));
        }
        Table::from_columns(out_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_widens_mismatched_variants_to_f64() {
        // Table A: score as I64; Table B: same column as Str (a metric-fail
        // row that went to empty string). union_by_name must widen to F64.
        let a = Table::from_columns(vec![
            ("k".into(), Column::Str(vec![Some("x".into())])),
            ("score".into(), Column::I64(vec![5])),
        ])
        .unwrap();
        let b = Table::from_columns(vec![
            ("k".into(), Column::Str(vec![Some("y".into())])),
            ("score".into(), Column::Str(vec![Some("".into())])),
        ])
        .unwrap();
        let u = Table::union_by_name(vec![a, b]).unwrap();
        assert_eq!(u.num_rows(), 2);
        let s = u.column("score").unwrap();
        assert!(matches!(s, Column::F64(_)));
        assert_eq!(s.f64_at(0), 5.0);
        assert!(s.f64_at(1).is_nan()); // empty string → NaN
    }

    #[test]
    fn union_null_fills_absent_columns() {
        let a = Table::from_columns(vec![
            ("k".into(), Column::Str(vec![Some("x".into())])),
            ("only_a".into(), Column::F64(vec![1.0])),
        ])
        .unwrap();
        let b = Table::from_columns(vec![("k".into(), Column::Str(vec![Some("y".into())]))]).unwrap();
        let u = Table::union_by_name(vec![a, b]).unwrap();
        assert_eq!(u.num_rows(), 2);
        assert!(u.column("only_a").unwrap().f64_at(1).is_nan());
    }

    #[test]
    fn prefix_columns_namespaces_feat_only() {
        let mut t = Table::from_columns(vec![
            ("image_path".into(), Column::Str(vec![Some("a".into())])),
            ("feat_0".into(), Column::F64(vec![1.0])),
            ("zensim_score".into(), Column::F64(vec![99.0])),
        ])
        .unwrap();
        t.prefix_columns_where("zsm_", &["image_path", "zensim_score"], |n| {
            n.starts_with("feat_")
        });
        assert!(t.has_column("zsm_feat_0"));
        assert!(t.has_column("image_path"));
        assert!(t.has_column("zensim_score"));
        assert!(!t.has_column("feat_0"));
    }
}
