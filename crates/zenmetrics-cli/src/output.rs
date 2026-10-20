#![forbid(unsafe_code)]

//! Output formatting for `score` and `compare` subcommands.

use clap::ValueEnum;
use std::io::{self, Write};

use crate::metrics::MetricKind;

/// CLI output format.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Plain text: `metric=<name> score=<value>`.
    Plain,
    /// JSON object on a single line.
    Json,
    /// Two-row TSV with headers on the first row.
    Tsv,
}

/// Render the score(s) returned by a single [`crate::metrics::run_metric`]
/// invocation. `scores` is the `(column_name, value)` list — one entry for
/// most metrics, two for butteraugli (max + 3-norm).
///
/// TSV layout matches the `batch` / `sweep` convention: one column per
/// score with the column name as the header. Plain and JSON layouts emit
/// every column too — JSON nests the values under `"scores"` keyed by
/// column name so callers can read individual fields without parsing the
/// metric out.
pub fn print_score(format: OutputFormat, kind: MetricKind, scores: &[(&'static str, f64)]) {
    match format {
        OutputFormat::Plain => {
            print!("metric={}", kind.name());
            for (col, value) in scores {
                print!(" {col}={value}");
            }
            println!();
        }
        OutputFormat::Json => {
            let mut score_map = serde_json::Map::with_capacity(scores.len());
            for (col, value) in scores {
                score_map.insert((*col).to_string(), serde_json::Value::from(*value));
            }
            let v = serde_json::json!({
                "metric": kind.name(),
                "scores": serde_json::Value::Object(score_map),
            });
            println!("{v}");
        }
        OutputFormat::Tsv => {
            // Two-row TSV: columns are the score names; the metric name is
            // not a column anymore because a single metric can emit several
            // columns (butteraugli emits both max + pnorm3).
            let header: Vec<&str> = scores.iter().map(|(col, _)| *col).collect();
            println!("{}", header.join("\t"));
            let row: Vec<String> = scores.iter().map(|(_, v)| format!("{v}")).collect();
            println!("{}", row.join("\t"));
        }
    }
}

/// One (reference, variant) row of a `compare` run, with one score per metric
/// in `metrics_order`. `Err(reason)` slots are recorded as `null` / `NaN` /
/// `ERROR: <reason>` depending on the output format.
///
/// Each metric may contribute multiple columns (butteraugli emits two), so
/// `scores[i]` is itself a `Vec<(column_name, value)>` parallel to
/// `metrics_order[i].columns_for(display)`. When a metric fails the whole entry
/// is `Err` — we don't track per-column success because the metric backend
/// is what fails, not individual aggregations.
pub struct CompareRow {
    pub reference: String,
    pub variant: String,
    /// Parallel to `metrics_order` — one entry per metric, in the same
    /// order. Each entry is the metric's full `(column_name, value)` list.
    pub scores: Vec<Result<Vec<(&'static str, f64)>, String>>,
}

/// Render the full result set of a `compare` invocation to `w`.
/// `display` is the cvvdp display selection; cvvdp columns name it
/// ([`MetricKind::columns_for`]).
pub fn render_compare(
    w: &mut dyn Write,
    format: OutputFormat,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    match format {
        OutputFormat::Plain => render_compare_plain(w, metrics_order, rows, display),
        OutputFormat::Json => render_compare_json(w, metrics_order, rows, display),
        OutputFormat::Tsv => render_compare_tsv(w, metrics_order, rows, display),
    }
}

fn render_compare_plain(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    // Width of the longest column name, for alignment of the score column.
    // We use column_names rather than metric names so butteraugli's two
    // columns line up.
    let name_width = metrics_order
        .iter()
        .flat_map(|m| m.columns_for(display).iter().copied())
        .map(str::len)
        .max()
        .unwrap_or(0);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            writeln!(w)?;
        }
        writeln!(w, "{} vs {}:", row.reference, row.variant)?;
        for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
            match score {
                Ok(values) => {
                    for (col, v) in values {
                        writeln!(w, "  {col:<name_width$}  {v}")?;
                    }
                }
                Err(reason) => {
                    // Print an error line per column the metric would have
                    // produced, so callers diff'ing TSV-shaped output still
                    // see one entry per column they expected.
                    for col in m.columns_for(display) {
                        writeln!(w, "  {col:<name_width$}  ERROR: {reason}")?;
                    }
                }
            }
        }
    }
    Ok(())
}

fn render_compare_json(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    let metric_names: Vec<&str> = metrics_order.iter().map(|m| m.name()).collect();
    let results: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| {
            // `scores` is keyed by column name, not metric name. For
            // single-column metrics that's identical to the previous
            // behaviour; for butteraugli it lets callers pull
            // `scores.butteraugli_max` and `scores.butteraugli_pnorm3`
            // independently. Failed metrics produce nulls for every
            // column the metric would have emitted.
            let total_cols: usize = metrics_order
                .iter()
                .map(|m| m.columns_for(display).len())
                .sum();
            let mut scores = serde_json::Map::with_capacity(total_cols);
            for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
                match score {
                    Ok(values) => {
                        for (col, v) in values {
                            scores.insert((*col).to_string(), serde_json::Value::from(*v));
                        }
                    }
                    Err(_) => {
                        for col in m.columns_for(display) {
                            scores.insert((*col).to_string(), serde_json::Value::Null);
                        }
                    }
                }
            }
            // Field order: reference, variant, scores (matches the
            // `compare --help` documented schema). `preserve_order` on
            // serde_json keeps this insertion order in the rendered JSON.
            let mut row_obj = serde_json::Map::with_capacity(3);
            row_obj.insert("reference".into(), row.reference.clone().into());
            row_obj.insert("variant".into(), row.variant.clone().into());
            row_obj.insert("scores".into(), serde_json::Value::Object(scores));
            serde_json::Value::Object(row_obj)
        })
        .collect();
    let doc = serde_json::json!({
        "metrics": metric_names,
        "results": results,
    });
    serde_json::to_writer_pretty(&mut *w, &doc)?;
    writeln!(w)?;
    Ok(())
}

fn render_compare_tsv(
    w: &mut dyn Write,
    metrics_order: &[MetricKind],
    rows: &[CompareRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    // Header: reference, variant, then one column per (metric × emitted
    // column). Single-metric butteraugli expands to two columns; everything
    // else is one. Column names use the friendly underscore form (matches
    // the sweep TSV `score_<column>` convention without the `score_`
    // prefix).
    write!(w, "reference\tvariant")?;
    for m in metrics_order {
        for col in m.columns_for(display) {
            write!(w, "\t{col}")?;
        }
    }
    writeln!(w)?;
    for row in rows {
        write!(w, "{}\t{}", row.reference, row.variant)?;
        for (m, score) in metrics_order.iter().zip(row.scores.iter()) {
            match score {
                Ok(values) => {
                    for (_, v) in values {
                        write!(w, "\t{v}")?;
                    }
                }
                // NaN signals a failed cell in TSV mode — distinguishable
                // from any real metric value because metrics never produce
                // NaN on success (we explicitly reject non-finite scores).
                // Emit one NaN per column the metric would have produced.
                Err(_) => {
                    for _ in m.columns_for(display) {
                        write!(w, "\tNaN")?;
                    }
                }
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

/// One (reference, distorted) row of a folder scoring run.
pub struct FolderScoreRow {
    pub reference: String,
    pub distorted: String,
    pub scores: Result<Vec<(&'static str, f64)>, String>,
}

/// Render the results of a folder scoring run to stdout according to `format`.
pub fn print_folder_scores(
    format: OutputFormat,
    kind: MetricKind,
    rows: &[FolderScoreRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) {
    let mut stdout = io::stdout().lock();
    let _ = render_folder_scores(&mut stdout, format, kind, rows, display);
}

/// Render the results of a folder scoring run to `w`.
pub fn render_folder_scores(
    w: &mut dyn Write,
    format: OutputFormat,
    kind: MetricKind,
    rows: &[FolderScoreRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    match format {
        OutputFormat::Plain => render_folder_scores_plain(w, kind, rows),
        OutputFormat::Json => render_folder_scores_json(w, kind, rows),
        OutputFormat::Tsv => render_folder_scores_tsv(w, kind, rows, display),
    }
}

fn render_folder_scores_plain(
    w: &mut dyn Write,
    kind: MetricKind,
    rows: &[FolderScoreRow],
) -> io::Result<()> {
    let mut col_stats: std::collections::BTreeMap<&'static str, Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut failed_count = 0usize;

    for row in rows {
        write!(w, "metric={}", kind.name())?;
        write!(w, " ref={} dist={}", row.reference, row.distorted)?;
        match &row.scores {
            Ok(scores) => {
                for (col, v) in scores {
                    write!(w, " {col}={v}")?;
                    col_stats.entry(*col).or_default().push(*v);
                }
            }
            Err(e) => {
                write!(w, " ERROR: {e}")?;
                failed_count += 1;
            }
        }
        writeln!(w)?;
    }

    writeln!(w, "--- Summary ---")?;
    writeln!(w, "Total pairs: {}", rows.len())?;
    writeln!(w, "Successful: {}", rows.len() - failed_count)?;
    if failed_count > 0 {
        writeln!(w, "Failed: {}", failed_count)?;
    }
    for (col, vals) in &col_stats {
        if !vals.is_empty() {
            let sum: f64 = vals.iter().sum();
            let mean = sum / vals.len() as f64;
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            writeln!(
                w,
                "{col}: count={} mean={mean} min={min} max={max}",
                vals.len()
            )?;
        }
    }
    Ok(())
}

fn render_folder_scores_tsv(
    w: &mut dyn Write,
    kind: MetricKind,
    rows: &[FolderScoreRow],
    display: Option<&crate::metrics::CvvdpDisplay>,
) -> io::Result<()> {
    let cols = kind.columns_for(display);
    write!(w, "reference\tdistorted")?;
    for col in cols {
        write!(w, "\t{col}")?;
    }
    writeln!(w)?;

    for row in rows {
        write!(w, "{}\t{}", row.reference, row.distorted)?;
        match &row.scores {
            Ok(scores) => {
                for (_, v) in scores {
                    write!(w, "\t{v}")?;
                }
            }
            Err(_) => {
                for _ in cols {
                    write!(w, "\tNaN")?;
                }
            }
        }
        writeln!(w)?;
    }
    Ok(())
}

fn render_folder_scores_json(
    w: &mut dyn Write,
    kind: MetricKind,
    rows: &[FolderScoreRow],
) -> io::Result<()> {
    let mut col_stats: std::collections::BTreeMap<&'static str, Vec<f64>> =
        std::collections::BTreeMap::new();
    let mut results = Vec::with_capacity(rows.len());
    let mut failed_count = 0usize;

    for row in rows {
        match &row.scores {
            Ok(scores) => {
                let mut score_map = serde_json::Map::with_capacity(scores.len());
                for (col, v) in scores {
                    score_map.insert((*col).to_string(), serde_json::Value::from(*v));
                    col_stats.entry(*col).or_default().push(*v);
                }
                results.push(serde_json::json!({
                    "reference": row.reference,
                    "distorted": row.distorted,
                    "scores": score_map,
                }));
            }
            Err(e) => {
                failed_count += 1;
                results.push(serde_json::json!({
                    "reference": row.reference,
                    "distorted": row.distorted,
                    "error": e,
                }));
            }
        }
    }

    let mut summary_map = serde_json::Map::new();
    for (col, vals) in &col_stats {
        if !vals.is_empty() {
            let sum: f64 = vals.iter().sum();
            let mean = sum / vals.len() as f64;
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            summary_map.insert(
                (*col).to_string(),
                serde_json::json!({
                    "count": vals.len(),
                    "mean": mean,
                    "min": min,
                    "max": max,
                }),
            );
        }
    }

    let doc = serde_json::json!({
        "metric": kind.name(),
        "total_pairs": rows.len(),
        "successful_pairs": rows.len() - failed_count,
        "failed_pairs": failed_count,
        "summary": summary_map,
        "results": results,
    });
    serde_json::to_writer_pretty(&mut *w, &doc)?;
    writeln!(w)?;
    Ok(())
}

#[cfg(test)]
mod precision_tests {
    use super::*;

    #[test]
    fn compare_and_folder_tsv_preserve_f64_bits() {
        for value in [0.123_456_789_012_345_66, 1e-100, 1e100, f64::MIN_POSITIVE] {
            let row = CompareRow {
                reference: "ref.png".into(),
                variant: "dist.png".into(),
                scores: vec![Ok(vec![("psnr", value)])],
            };
            let mut out = Vec::new();
            render_compare(
                &mut out,
                OutputFormat::Tsv,
                &[MetricKind::Psnr],
                &[row],
                None,
            )
            .unwrap();
            let text = String::from_utf8(out).unwrap();
            let emitted = text
                .lines()
                .nth(1)
                .unwrap()
                .split('\t')
                .next_back()
                .unwrap();
            assert_eq!(emitted.parse::<f64>().unwrap().to_bits(), value.to_bits());
        }
    }
}
