//! Per-chunk zenanalyze source-image features.
//!
//! Computes [`zenanalyze::analyze_features_rgb8`] features for each
//! UNIQUE source image in a chunk. Output: one parquet per chunk
//! at `s3://zentrain/<run>/source_features/<chunk_id>.parquet`,
//! with one row per source. Schema:
//!
//! ```text
//! chunk_id: Utf8
//! run_id: Utf8
//! image_basename: Utf8
//! image_path: Utf8
//! width: UInt32
//! height: UInt32
//! feat_<id>: Float32   (one column per feature in FeatureSet::SUPPORTED)
//! ```
//!
//! ## Rationale (user, 2026-05-19)
//!
//! > zenanalyze is rust and produces raw values that absolutely
//! > could be stored and are absolutely needed for original
//! > undistorted sources, and only for distortions if we want to
//! > explore adding those vars into zensim which is unlikely.
//!
//! So: source images get their features computed and stored as a
//! sidecar; distorted variants do NOT. The omni sidecar (per-cell
//! metric scores) stays unchanged; this is an orthogonal sidecar.
//!
//! ## Why a wide-form schema
//!
//! N=102 feature columns is uncomfortably wide for some readers,
//! but it lets a downstream join `omni[image_path] × features[image_basename]`
//! resolve in one pass. The alternative — long-form `(image, feat_id,
//! value)` triples — would force a pivot on every consumer.
//!
//! ## Idempotency
//!
//! The Rust worker computes features for each unique basename
//! seen in the chunk's input parquet. If the same basename
//! appears across N chunks, we redo the computation N times.
//! That's wasteful but cheap: per-image cost is <100ms for typical
//! sizes. A future dedupe pass could move features into a
//! global parquet keyed by basename — out of scope for now.

#![cfg(feature = "source-features")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tracing::warn;
use zenanalyze::feature::{AnalysisQuery, FeatureSet};

/// One row's worth of computed features for one source image.
pub struct SourceFeaturesRow {
    pub image_basename: String,
    pub image_path: String,
    pub width: u32,
    pub height: u32,
    /// (feature_id, value) pairs. Sorted by feature_id ascending so
    /// the parquet column ordering is deterministic.
    pub features: Vec<(u16, f32)>,
}

/// Compute features for one PNG / JPEG / WebP / etc on local disk.
/// Returns the row to be appended to the chunk's source_features
/// parquet.
pub fn compute_for_source(local_path: &Path, basename: &str) -> Result<SourceFeaturesRow> {
    let img = image::open(local_path)
        .with_context(|| format!("decode image {}", local_path.display()))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let rgb_bytes = rgb.as_raw();

    // Use FeatureSet::SUPPORTED for "every feature in this build".
    // The crate doc explicitly recommends this for production callers
    // who want the full feature snapshot.
    let supported = FeatureSet::SUPPORTED;
    let query = AnalysisQuery::new(supported);

    let results = zenanalyze::try_analyze_features_rgb8(rgb_bytes, w, h, &query)
        .map_err(|e| anyhow::anyhow!("zenanalyze: {e:?}"))?;

    let mut features: Vec<(u16, f32)> = Vec::with_capacity(128);
    for feat in supported.iter() {
        let Some(value) = results.get(feat) else {
            continue;
        };
        features.push((feat.id(), value.to_f32()));
    }
    features.sort_by_key(|(id, _)| *id);

    Ok(SourceFeaturesRow {
        image_basename: basename.to_string(),
        image_path: local_path.to_string_lossy().into_owned(),
        width: w,
        height: h,
        features,
    })
}

/// Build the source_features parquet for a chunk and write it to
/// `output_path`. `rows` is one row per unique source image. The
/// schema is derived from the FIRST row's feature ids; we assume
/// FeatureSet::SUPPORTED is constant within one build, so every
/// row has the same columns.
pub fn write_parquet(
    rows: &[SourceFeaturesRow],
    chunk_id: &str,
    run_id: &str,
    output_path: &Path,
) -> Result<usize> {
    if rows.is_empty() {
        anyhow::bail!("no rows to write");
    }
    let n = rows.len();
    let first = &rows[0];
    let feat_ids: Vec<u16> = first.features.iter().map(|(id, _)| *id).collect();

    let mut fields: Vec<Field> = vec![
        Field::new("chunk_id", DataType::Utf8, false),
        Field::new("run_id", DataType::Utf8, false),
        Field::new("image_basename", DataType::Utf8, false),
        Field::new("image_path", DataType::Utf8, false),
        Field::new("width", DataType::UInt32, false),
        Field::new("height", DataType::UInt32, false),
    ];
    for id in &feat_ids {
        fields.push(Field::new(
            format!("feat_{id}"),
            DataType::Float32,
            true, // nullable in case a row dropped this feature
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    let chunk_col: StringArray = (0..n).map(|_| Some(chunk_id)).collect();
    let run_col: StringArray = (0..n).map(|_| Some(run_id)).collect();
    let basename_col: StringArray = rows.iter().map(|r| Some(r.image_basename.as_str())).collect();
    let path_col: StringArray = rows.iter().map(|r| Some(r.image_path.as_str())).collect();
    let width_col: UInt32Array = rows.iter().map(|r| r.width).collect();
    let height_col: UInt32Array = rows.iter().map(|r| r.height).collect();

    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(chunk_col),
        Arc::new(run_col),
        Arc::new(basename_col),
        Arc::new(path_col),
        Arc::new(width_col),
        Arc::new(height_col),
    ];
    for (col_idx, id) in feat_ids.iter().enumerate() {
        let values: Vec<Option<f32>> = rows
            .iter()
            .map(|r| {
                r.features
                    .get(col_idx)
                    .filter(|(fid, _)| fid == id)
                    .map(|(_, v)| *v)
                    .or_else(|| {
                        // Defensive: re-search if order misaligned.
                        r.features.iter().find(|(fid, _)| *fid == *id).map(|(_, v)| *v)
                    })
            })
            .collect();
        cols.push(Arc::new(Float32Array::from(values)));
    }

    let batch = RecordBatch::try_new(schema.clone(), cols).context("build record batch")?;
    let file = std::fs::File::create(output_path)
        .with_context(|| format!("create {}", output_path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut wtr = ArrowWriter::try_new(file, schema, Some(props)).context("arrow writer")?;
    wtr.write(&batch).context("write batch")?;
    wtr.close().context("close writer")?;
    Ok(n)
}

/// Compute + write end-to-end. The dispatcher calls this once per
/// chunk after step 2 (sync sources) completes. Failures are logged
/// but non-fatal: source features are nice-to-have, not load-bearing.
pub async fn compute_and_write(
    sources_dir: &Path,
    basenames: &[String],
    output_path: &Path,
    chunk_id: &str,
    run_id: &str,
) -> Result<usize> {
    let mut rows = Vec::with_capacity(basenames.len());
    for b in basenames {
        let local = sources_dir.join(b);
        let bname = b.clone();
        let local_owned = local.clone();
        let result = tokio::task::spawn_blocking(move || compute_for_source(&local_owned, &bname))
            .await
            .map_err(|e| anyhow::anyhow!("task panic: {e}"))?;
        match result {
            Ok(row) => rows.push(row),
            Err(e) => {
                warn!(basename = %b, error = %e, "skip source: features failed");
            }
        }
    }
    if rows.is_empty() {
        anyhow::bail!("no source-features rows produced");
    }
    write_parquet(&rows, chunk_id, run_id, output_path)
}
