//! Feature-backfill mode — compute zensim 300-feature vectors for
//! cells that ALREADY have an omni sidecar + encoded variant on R2,
//! without re-encoding.
//!
//! The user mandate (2026-05-19):
//!
//! > we need to plumb it through and backfill only features with the
//! > encoded images already on r2, duplicating no work
//!
//! Pipeline per chunk:
//!
//! 1. Idempotency: skip if `s3://zentrain/<run>/zensim_features/<chunk>.parquet`
//!    already exists.
//! 2. Claim (same token-race as the omni mode).
//! 3. Download the chunk's omni sidecar from R2.
//! 4. Read it via arrow-rs; extract `(image_path, codec, q,
//!    knob_tuple_json, encoded_filename)` tuples.
//! 5. Download every unique source PNG (from the chunk's `source_dir_r2`)
//!    AND every encoded variant (from
//!    `s3://zentrain/<run>/encoded/<chunk>/<filename>`).
//! 6. For each tuple, decode ref + dist to `Rgb8Image`, call
//!    `zen_metrics_cli::metrics::run_zensim_with_features` to get
//!    `(score, [f64; 300])`.
//! 7. Write a wide-form parquet keyed by the identity tuple +
//!    `zensim_score:Float32 + feat_0..feat_299:Float32`. Same schema
//!    as the inline pipeline's per-chunk feature parquet — these
//!    interoperate.
//! 8. Upload + cleanup.
//!
//! Single cubecl init is irrelevant here (CPU zensim doesn't need
//! the GPU). What matters is that decode + score are cheap (~80ms
//! per pair on a typical small image), so 365 chunks × 200 cells
//! / 4 boxes ~~ ~30 min wall time. Cost: trivial.

#![cfg(feature = "inline-sweep")]

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow_array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow_array::cast::AsArray as _;
use arrow_array::types::Int64Type;
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::Deserialize;
use tracing::{debug, info, warn};
use zen_metrics_cli::decode::decode_image_to_rgb8;
use zen_metrics_cli::metrics::run_zensim_with_features;

use super::WorkerArgs;
use super::r2::R2Client;

/// Subset of a chunk record that we need to compute features.
#[derive(Debug, Deserialize)]
struct ChunkRecord {
    chunk_id: String,
    source_dir_r2: String,
    run_id: Option<String>,
}

/// One row of the input omni sidecar — the (image, codec, q, knobs)
/// identity tuple plus the `encoded_filename` that lets us locate
/// the existing distorted variant on R2.
#[derive(Debug, Clone)]
struct OmniRow {
    image_path: String,
    codec: String,
    q: u32,
    knob_tuple_json: String,
    encoded_filename: String,
}

/// One row of feature output: identity tuple + zensim_score + 300 feats.
struct FeatureRow {
    image_path: String,
    codec: String,
    q: u32,
    knob_tuple_json: String,
    zensim_score: f32,
    features: Vec<f32>,
}

/// Number of zensim features. Matches
/// `zen_metrics_cli::sweep::feature_writer::NUM_FEATURES`.
const NUM_FEATURES: usize = 300;

/// Top-level: read chunk JSON, do the feature backfill, upload.
pub async fn backfill_features_for_chunk(
    args: &WorkerArgs,
    r2: &R2Client,
    line: &str,
) -> Result<()> {
    let rec: ChunkRecord = serde_json::from_str(line).context("parse chunk JSON")?;
    let run_id = rec
        .run_id
        .clone()
        .unwrap_or_else(|| args.run_id.clone());

    let out_uri = format!(
        "s3://zentrain/{run_id}/zensim_features/{}.parquet",
        rec.chunk_id
    );

    // Idempotency.
    if r2.exists(&out_uri).await {
        info!(chunk_id = %rec.chunk_id, "skip: feature sidecar already in R2");
        return Ok(());
    }

    // Locate the omni sidecar — same scheme the inline mode uses.
    let omni_uri = format!(
        "s3://zentrain/{run_id}/omni/{}.parquet",
        rec.chunk_id
    );
    if !r2.exists(&omni_uri).await {
        warn!(
            chunk_id = %rec.chunk_id,
            omni_uri = %omni_uri,
            "skip: no omni sidecar to backfill from"
        );
        return Ok(());
    }

    let scratch = args.workdir.join(format!("features-{}", rec.chunk_id));
    let sources_dir = scratch.join("sources");
    let encoded_dir = scratch.join("encoded");
    for d in [&sources_dir, &encoded_dir] {
        tokio::fs::create_dir_all(d).await
            .with_context(|| format!("mkdir {}", d.display()))?;
    }

    info!(chunk_id = %rec.chunk_id, "downloading omni sidecar");
    let omni_local = scratch.join("omni.parquet");
    r2.download(&omni_uri, &omni_local)
        .await
        .context("download omni sidecar")?;

    let omni_path_for_blocking = omni_local.clone();
    let rows_result: std::result::Result<Result<Vec<OmniRow>>, tokio::task::JoinError> =
        tokio::task::spawn_blocking(move || read_omni_rows(&omni_path_for_blocking)).await;
    let rows: Vec<OmniRow> = match rows_result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            warn!(chunk_id = %rec.chunk_id, error = %e, "skip: omni read failed");
            if std::env::var_os("KEEP_WORK").is_none() {
                let _ = tokio::fs::remove_dir_all(&scratch).await;
            }
            return Ok(());
        }
        Err(e) => {
            warn!(
                chunk_id = %rec.chunk_id,
                error = %e,
                "skip: omni read task panicked"
            );
            if std::env::var_os("KEEP_WORK").is_none() {
                let _ = tokio::fs::remove_dir_all(&scratch).await;
            }
            return Ok(());
        }
    };
    info!(chunk_id = %rec.chunk_id, n_rows = rows.len(), "omni rows parsed");

    // Build the unique-basename + unique-encoded-filename sets so we
    // download each blob exactly once.
    let mut unique_basenames: std::collections::BTreeSet<String> = Default::default();
    let mut unique_encoded: std::collections::BTreeSet<String> = Default::default();
    for r in &rows {
        let basename = r.image_path.rsplit('/').next().unwrap_or(&r.image_path);
        unique_basenames.insert(basename.to_string());
        if !r.encoded_filename.is_empty() {
            unique_encoded.insert(r.encoded_filename.clone());
        }
    }
    info!(
        chunk_id = %rec.chunk_id,
        n_sources = unique_basenames.len(),
        n_encoded = unique_encoded.len(),
        "syncing inputs"
    );

    sync_files(
        r2,
        &rec.source_dir_r2,
        &unique_basenames.iter().cloned().collect::<Vec<_>>(),
        &sources_dir,
    )
    .await
    .context("sync sources")?;
    let encoded_prefix = format!("s3://zentrain/{run_id}/encoded/{}", rec.chunk_id);
    sync_files(
        r2,
        &encoded_prefix,
        &unique_encoded.iter().cloned().collect::<Vec<_>>(),
        &encoded_dir,
    )
    .await
    .context("sync encoded variants")?;

    info!(chunk_id = %rec.chunk_id, "scoring + extracting features");
    let sources_dir_for_blocking = sources_dir.clone();
    let encoded_dir_for_blocking = encoded_dir.clone();
    let rows_for_blocking = rows.clone();
    let feature_rows: Vec<FeatureRow> = tokio::task::spawn_blocking(move || {
        compute_features_for_rows(
            &rows_for_blocking,
            &sources_dir_for_blocking,
            &encoded_dir_for_blocking,
        )
    })
    .await
    .map_err(|e| anyhow!("feature compute task panicked: {e}"))??;

    if feature_rows.is_empty() {
        warn!(chunk_id = %rec.chunk_id, "no usable rows after feature extraction");
        if std::env::var_os("KEEP_WORK").is_none() {
            let _ = tokio::fs::remove_dir_all(&scratch).await;
        }
        return Ok(());
    }
    info!(
        chunk_id = %rec.chunk_id,
        n_out = feature_rows.len(),
        "features computed"
    );

    let out_local = scratch.join("features.parquet");
    let chunk_id_owned = rec.chunk_id.clone();
    let out_local_for_blocking = out_local.clone();
    tokio::task::spawn_blocking(move || {
        write_feature_parquet(&feature_rows, &out_local_for_blocking, &chunk_id_owned)
    })
    .await
    .map_err(|e| anyhow!("write task panicked: {e}"))??;

    r2.upload(&out_local, &out_uri).await
        .context("upload feature parquet")?;
    info!(chunk_id = %rec.chunk_id, uri = %out_uri, "feature parquet uploaded");

    if std::env::var_os("KEEP_WORK").is_none() {
        let _ = tokio::fs::remove_dir_all(&scratch).await;
    }
    Ok(())
}

/// Read the omni sidecar's `(image_path, codec, q, knob_tuple_json,
/// encoded_filename)` columns into a Vec<OmniRow>. Strict on schema:
/// missing required columns is an error.
fn read_omni_rows(path: &Path) -> Result<Vec<OmniRow>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .context("parquet reader builder")?;
    let schema = builder.schema().clone();
    for col in ["image_path", "codec", "q", "knob_tuple_json", "encoded_filename"] {
        if schema.index_of(col).is_err() {
            return Err(anyhow!("omni sidecar missing column `{col}`"));
        }
    }
    // Early-out: if encoded_filename is the all-null arrow type, the
    // chunk has no encoded variants saved and feature backfill is
    // impossible without re-encoding. Fail loud so the dispatcher
    // logs + skips.
    let enc_idx = schema.index_of("encoded_filename")?;
    if matches!(schema.field(enc_idx).data_type(), arrow_schema::DataType::Null) {
        return Err(anyhow!(
            "encoded_filename column is Null type — no encoded variants on R2 to score"
        ));
    }
    let reader = builder.build().context("build reader")?;

    let mut rows: Vec<OmniRow> = Vec::new();
    for batch in reader {
        let batch = batch.context("read parquet batch")?;
        let image_path = batch
            .column(batch.schema().index_of("image_path")?)
            .as_string::<i32>();
        let codec = batch
            .column(batch.schema().index_of("codec")?)
            .as_string::<i32>();
        let q_col = batch.column(batch.schema().index_of("q")?);
        let knob = batch
            .column(batch.schema().index_of("knob_tuple_json")?)
            .as_string::<i32>();
        let enc = batch
            .column(batch.schema().index_of("encoded_filename")?)
            .as_string::<i32>();
        let n = batch.num_rows();
        // q can be Int64 (from chunks.jsonl) or UInt32 (from feature
        // writer) — handle both.
        for i in 0..n {
            let q = if let Some(arr) = q_col.as_any().downcast_ref::<Int64Array>() {
                arr.value(i) as u32
            } else if let Some(arr) = q_col.as_any().downcast_ref::<UInt32Array>() {
                arr.value(i)
            } else if let Some(_arr) = q_col.as_primitive_opt::<Int64Type>() {
                q_col.as_primitive::<Int64Type>().value(i) as u32
            } else {
                continue; // unknown q dtype — skip row
            };
            rows.push(OmniRow {
                image_path: image_path.value(i).to_string(),
                codec: codec.value(i).to_string(),
                q,
                knob_tuple_json: knob.value(i).to_string(),
                encoded_filename: enc.value(i).to_string(),
            });
        }
    }
    Ok(rows)
}

fn compute_features_for_rows(
    rows: &[OmniRow],
    sources_dir: &Path,
    encoded_dir: &Path,
) -> Result<Vec<FeatureRow>> {
    let mut out = Vec::with_capacity(rows.len());
    // Cache decoded reference images by basename so we decode each
    // source once. `Rgb8Image` doesn't implement Clone, so wrap in
    // Arc and pass &*Arc<Rgb8Image> as &Rgb8Image.
    let mut ref_cache: std::collections::HashMap<String, std::sync::Arc<zen_metrics_cli::decode::Rgb8Image>> =
        std::collections::HashMap::new();
    let mut n_skip_missing = 0usize;
    let mut n_skip_decode = 0usize;
    let mut n_skip_score = 0usize;
    for row in rows {
        if row.encoded_filename.is_empty() {
            n_skip_missing += 1;
            continue;
        }
        let basename = row.image_path.rsplit('/').next().unwrap_or(&row.image_path);
        let dist_path = encoded_dir.join(&row.encoded_filename);
        if !dist_path.exists() {
            n_skip_missing += 1;
            continue;
        }
        let reference: std::sync::Arc<zen_metrics_cli::decode::Rgb8Image> =
            if let Some(r) = ref_cache.get(basename) {
                r.clone()
            } else {
                let p = sources_dir.join(basename);
                match decode_image_to_rgb8(&p) {
                    Ok(r) => {
                        let arc = std::sync::Arc::new(r);
                        ref_cache.insert(basename.to_string(), arc.clone());
                        arc
                    }
                    Err(e) => {
                        debug!(basename, error = %e, "decode ref failed; skip");
                        n_skip_decode += 1;
                        continue;
                    }
                }
            };
        let distorted = match decode_image_to_rgb8(&dist_path) {
            Ok(d) => d,
            Err(e) => {
                debug!(file = %dist_path.display(), error = %e, "decode dist failed; skip");
                n_skip_decode += 1;
                continue;
            }
        };
        match run_zensim_with_features(&reference, &distorted) {
            Ok((score, feats)) => {
                if feats.len() != NUM_FEATURES {
                    warn!(
                        got = feats.len(),
                        want = NUM_FEATURES,
                        "feature count mismatch — skip"
                    );
                    n_skip_score += 1;
                    continue;
                }
                out.push(FeatureRow {
                    image_path: row.image_path.clone(),
                    codec: row.codec.clone(),
                    q: row.q,
                    knob_tuple_json: row.knob_tuple_json.clone(),
                    zensim_score: score as f32,
                    features: feats.into_iter().map(|x| x as f32).collect(),
                });
            }
            Err(e) => {
                debug!(error = %e, "zensim feature compute failed; skip");
                n_skip_score += 1;
            }
        }
    }
    info!(
        n_in = rows.len(),
        n_out = out.len(),
        n_skip_missing,
        n_skip_decode,
        n_skip_score,
        "feature compute summary"
    );
    Ok(out)
}

fn write_feature_parquet(rows: &[FeatureRow], path: &Path, _chunk_id: &str) -> Result<()> {
    let mut fields: Vec<Field> = vec![
        Field::new("image_path", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::UInt32, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new("zensim_score", DataType::Float32, false),
    ];
    for i in 0..NUM_FEATURES {
        fields.push(Field::new(format!("feat_{i}"), DataType::Float32, false));
    }
    let schema = Arc::new(Schema::new(fields));

    let n = rows.len();
    let ip: StringArray = rows.iter().map(|r| Some(r.image_path.as_str())).collect();
    let cd: StringArray = rows.iter().map(|r| Some(r.codec.as_str())).collect();
    let q: UInt32Array = rows.iter().map(|r| r.q).collect();
    let kt: StringArray = rows.iter().map(|r| Some(r.knob_tuple_json.as_str())).collect();
    let zs: Float32Array = rows.iter().map(|r| r.zensim_score).collect();
    let mut cols: Vec<ArrayRef> = vec![
        Arc::new(ip),
        Arc::new(cd),
        Arc::new(q),
        Arc::new(kt),
        Arc::new(zs),
    ];
    for i in 0..NUM_FEATURES {
        let vals: Float32Array = rows.iter().map(|r| r.features[i]).collect();
        cols.push(Arc::new(vals));
    }
    let batch = RecordBatch::try_new(schema.clone(), cols).context("build record batch")?;

    let file = std::fs::File::create(path)
        .with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut wtr = ArrowWriter::try_new(file, schema, Some(props)).context("arrow writer")?;
    wtr.write(&batch).context("write batch")?;
    wtr.close().context("close writer")?;
    let _ = n; // suppress unused
    Ok(())
}

async fn sync_files(
    r2: &R2Client,
    r2_prefix: &str,
    basenames: &[String],
    local_dir: &Path,
) -> Result<()> {
    if basenames.is_empty() {
        return Ok(());
    }
    let mut run_lines = String::new();
    for b in basenames {
        let src = format!("{r2_prefix}/{b}");
        let dst = local_dir.join(b);
        run_lines.push_str(&format!("cp {} {}\n", src, dst.to_string_lossy()));
    }
    let run_file = local_dir.join("_dl.run");
    tokio::fs::write(&run_file, run_lines).await?;
    let out = tokio::process::Command::new(&r2.bin)
        .arg("--endpoint-url")
        .arg(&r2.endpoint)
        .arg("--profile")
        .arg(&r2.profile)
        .arg("run")
        .arg(&run_file)
        .kill_on_drop(true)
        .output()
        .await
        .context("spawn s5cmd run")?;
    if !out.status.success() {
        return Err(anyhow!(
            "s5cmd run failed: {} stderr: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}
