//! Source-features-only mode — compute zenanalyze features for the
//! UNIQUE source images referenced by a chunk's omni sidecar, without
//! touching any encoded variant.
//!
//! Output: a parquet at
//! `s3://zentrain/<run>/source_features/<chunk>.parquet`, one row per
//! unique source image:
//!
//! ```text
//! chunk_id     : Utf8
//! run_id       : Utf8
//! image_basename : Utf8
//! image_path   : Utf8
//! width        : UInt32
//! height       : UInt32
//! feat_<id>    : Float32   one column per zenanalyze AnalysisFeature
//! ```
//!
//! Why a separate mode (not as part of omni or feature-backfill):
//!
//! - The OMNI runs that produced the existing v15rc + multi-codec
//!   data either lacked the `source-features` Cargo feature (v22)
//!   or only ran on a subset (v23 reencode-pass). 2587 of the 2933
//!   chunks have no source_features sidecar today.
//! - feature-backfill (zensim) needs the encoded variants; source-
//!   feature backfill only needs the source PNG. They have
//!   different input dependencies; keeping them separate makes the
//!   dispatch + claim logic clearer.
//! - No re-encoding, no zensim, no GPU touch. The work is ~30ms
//!   per unique source on a modern CPU. Bandwidth-bound on R2
//!   source PNG downloads.
//!
//! Per-source cache: identical to `feature_backfill::compute_features_for_rows`,
//! we cache decoded sources by basename so re-occurrences across
//! rows decode once. (But typical chunks have 1-3 unique sources
//! so the cache rarely matters.)

#![cfg(feature = "source-features")]

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use tracing::{info, warn};

use super::WorkerArgs;
use super::r2::R2Client;
use super::source_features;

#[derive(Debug, Deserialize)]
struct ChunkRecord {
    chunk_id: String,
    source_dir_r2: String,
    image_basenames: Vec<String>,
    run_id: Option<String>,
}

/// Top-level: read chunk JSON, sync sources, compute features, upload.
pub async fn backfill_source_features_for_chunk(
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
        "s3://zentrain/{run_id}/source_features/{}.parquet",
        rec.chunk_id
    );
    if r2.exists(&out_uri).await {
        info!(chunk_id = %rec.chunk_id, "skip: source_features sidecar already in R2");
        return Ok(());
    }

    let scratch = args.workdir.join(format!("sf-{}", rec.chunk_id));
    let sources_dir = scratch.join("sources");
    tokio::fs::create_dir_all(&sources_dir).await
        .with_context(|| format!("mkdir {}", sources_dir.display()))?;

    info!(
        chunk_id = %rec.chunk_id,
        n_sources = rec.image_basenames.len(),
        "syncing source PNGs"
    );
    sync_files(r2, &rec.source_dir_r2, &rec.image_basenames, &sources_dir).await
        .context("sync sources")?;

    let out_local = scratch.join("source_features.parquet");
    let chunk_id_owned = rec.chunk_id.clone();
    let run_id_owned = run_id.clone();
    let basenames_owned = rec.image_basenames.clone();
    let sources_dir_owned = sources_dir.clone();
    let out_local_owned = out_local.clone();
    let result = source_features::compute_and_write(
        &sources_dir_owned,
        &basenames_owned,
        &out_local_owned,
        &chunk_id_owned,
        &run_id_owned,
    )
    .await;

    match result {
        Ok(n) => {
            info!(chunk_id = %rec.chunk_id, n_sources = n, "source_features computed");
            r2.upload(&out_local, &out_uri).await.context("upload sidecar")?;
            info!(chunk_id = %rec.chunk_id, uri = %out_uri, "source_features uploaded");
        }
        Err(e) => {
            warn!(chunk_id = %rec.chunk_id, error = %e, "source_features failed");
        }
    }

    if std::env::var_os("KEEP_WORK").is_none() {
        let _ = tokio::fs::remove_dir_all(&scratch).await;
    }
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
