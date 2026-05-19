//! End-to-end in-process chunk pipeline.
//!
//! Replaces the bash `omni_backfill_chunk_worker.sh` script with a
//! single Rust function. One claim → one [`process_chunk_inline`]
//! call → one sidecar in R2. All steps run within the same process,
//! sharing cubecl's device cache.
//!
//! Pipeline:
//!
//! 1. **Parse** the chunk JSON (extracts input_parquet_r2,
//!    row_range, source_dir_r2, image_basenames, out_sidecar_omni,
//!    out_encoded_prefix).
//! 2. **Stage scratch** at `<workdir>/<chunk_id>/{sources,sweeps,encoded}`.
//! 3. **Download** the input parquet to scratch.
//! 4. **Sync sources** — for each basename in image_basenames,
//!    `s5cmd cp <source_dir_r2>/<basename> sources/<basename>`.
//!    Parallel via s5cmd's `run` batch mode.
//! 5. **Group** rows from input parquet by (codec, knob_tuple_json)
//!    via [`chunk_input::read_and_group`].
//! 6. **Per group** — symlink the needed source basenames into
//!    `<gid>/sources/`, then call [`sweep_runner::run_group_inline`]
//!    with output TSV `sweeps/g<gid>.tsv`. Within the same worker
//!    process, cubecl init pays ONCE total — every subsequent
//!    group call reuses the cached device.
//! 7. **Concat** the per-group TSVs into one parquet sidecar via
//!    [`chunk_output::concat_groups_to_parquet`].
//! 8. **Upload** the sidecar to `out_sidecar_omni`. Optionally
//!    upload encoded variants under `out_encoded_prefix`.
//! 9. **Cleanup** scratch (unless KEEP_WORK=1).

#![cfg(feature = "inline-sweep")]

use anyhow::{Context, Result, anyhow};
use clap::ValueEnum;
use serde::Deserialize;
use tracing::{info, warn};
use zen_metrics_cli::metrics::{GpuRuntime, MetricKind};
use zen_metrics_cli::sweep::CodecKind;

use super::WorkerArgs;
use super::chunk_input::read_and_group;
use super::chunk_output::concat_groups_to_parquet;
use super::r2::R2Client;
use super::sweep_runner::{InlineGroupSpec, knob_tuple_to_grid_json, run_group_inline};

/// Full chunk record. Mirrors the bash worker's `jq` extractions.
#[derive(Debug, Deserialize)]
struct ChunkRecord {
    chunk_id: String,
    input_parquet: String,
    input_parquet_r2: String,
    row_range: [usize; 2],
    source_dir_r2: String,
    image_basenames: Vec<String>,
    run_id: Option<String>,
    out_sidecar_omni: Option<String>,
    out_encoded_prefix: Option<String>,
}

/// Top-level entry point. The caller (chunk.rs::process_chunk) has
/// already won the claim race for this chunk.
pub async fn process_chunk_inline(
    args: &WorkerArgs,
    r2: &R2Client,
    line: &str,
) -> Result<()> {
    let rec: ChunkRecord = serde_json::from_str(line).context("parse chunk JSON")?;
    let run_id = rec
        .run_id
        .clone()
        .unwrap_or_else(|| args.run_id.clone());

    let scratch = args.workdir.join(&rec.chunk_id);
    let sources = scratch.join("sources");
    let sweeps = scratch.join("sweeps");
    let encoded = scratch.join("encoded");
    for dir in [&sources, &sweeps, &encoded] {
        tokio::fs::create_dir_all(dir).await
            .with_context(|| format!("mkdir {}", dir.display()))?;
    }

    let out_sidecar = rec.out_sidecar_omni.clone().unwrap_or_else(|| {
        format!("s3://zentrain/{run_id}/omni/{}.parquet", rec.chunk_id)
    });
    let out_encoded_prefix = rec
        .out_encoded_prefix
        .clone()
        .unwrap_or_else(|| format!("s3://zentrain/{run_id}/encoded/{}/", rec.chunk_id));

    info!(chunk_id = %rec.chunk_id, "step 1/5: download input parquet");
    let input_parquet = scratch.join(&rec.input_parquet);
    r2.download(&rec.input_parquet_r2, &input_parquet)
        .await
        .context("download input parquet")?;

    info!(
        chunk_id = %rec.chunk_id,
        n_basenames = rec.image_basenames.len(),
        "step 2/5: sync sources"
    );
    sync_sources(r2, &rec.source_dir_r2, &rec.image_basenames, &sources).await?;

    // Source-image features (zenanalyze) — feature-gated. Computed
    // here, after sources are on disk, in parallel-friendly tokio
    // tasks. Failures are non-fatal: the omni sidecar still ships
    // even if source_features doesn't.
    #[cfg(feature = "source-features")]
    {
        let sf_local = scratch.join(format!("{}.source_features.parquet", rec.chunk_id));
        let sf_uri = format!(
            "s3://zentrain/{run_id}/source_features/{}.parquet",
            rec.chunk_id
        );
        // Skip if the sidecar already exists in R2 (idempotency).
        if r2.exists(&sf_uri).await {
            info!(chunk_id = %rec.chunk_id, "skip: source_features sidecar already in R2");
        } else {
            match super::source_features::compute_and_write(
                &sources,
                &rec.image_basenames,
                &sf_local,
                &rec.chunk_id,
                &run_id,
            )
            .await
            {
                Ok(n) => {
                    info!(chunk_id = %rec.chunk_id, n_sources = n, "source_features built");
                    if let Err(e) = r2.upload(&sf_local, &sf_uri).await {
                        warn!(chunk_id = %rec.chunk_id, error = %e, "source_features upload failed");
                    } else {
                        info!(chunk_id = %rec.chunk_id, uri = %sf_uri, "source_features uploaded");
                    }
                }
                Err(e) => {
                    warn!(chunk_id = %rec.chunk_id, error = %e, "source_features skipped");
                }
            }
        }
    }

    info!(chunk_id = %rec.chunk_id, "step 3/5: group by (codec, knob_tuple_json)");
    let groups = {
        let p = input_parquet.clone();
        let rs = rec.row_range[0];
        let re_ = rec.row_range[1];
        tokio::task::spawn_blocking(move || read_and_group(&p, rs, re_))
            .await
            .context("group task panicked")??
    };
    info!(
        chunk_id = %rec.chunk_id,
        n_groups = groups.len(),
        "groups built"
    );

    info!(
        chunk_id = %rec.chunk_id,
        n_groups = groups.len(),
        "step 4/5: run sweep per group (in-process, shared cubecl)"
    );
    let metrics = parse_metrics_env_or_default();
    // If CPU `zensim` is in the metric set, also write the 372-feature
    // extended vector to a parquet sidecar at
    // s3://zentrain/<run>/zensim_features/<chunk>.parquet. Joins back
    // to the omni sidecar by `(image_path, codec, q, knob_tuple_json)`.
    // Skipped silently if metrics is GPU-only (e.g. zensim-gpu).
    let want_features = metrics.contains(&MetricKind::Zensim);
    let feature_out_path = if want_features {
        Some(scratch.join(format!("{}.zensim_features.parquet", rec.chunk_id)))
    } else {
        None
    };
    let feature_out_r2 = if want_features {
        Some(format!(
            "s3://zentrain/{run_id}/zensim_features/{}.parquet",
            rec.chunk_id
        ))
    } else {
        None
    };

    let mut groups_ok: usize = 0;
    let mut groups_fail: usize = 0;
    for (gid, group) in groups.iter().enumerate() {
        let gid_str = format!("{gid}");
        let group_sources = scratch.join(format!("g{gid_str}/sources"));
        tokio::fs::create_dir_all(&group_sources).await
            .with_context(|| format!("mkdir {}", group_sources.display()))?;
        for b in &group.image_basenames {
            let src = sources.join(b);
            let dst = group_sources.join(b);
            // symlink (hardlink falls back if the FS doesn't allow
            // symlinks across mount boundaries).
            let _ = tokio::fs::symlink(&src, &dst).await;
        }

        let q_grid_str = group
            .q_values
            .iter()
            .map(|q| q.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let knob_grid_json = if group.knob_tuple_json == "{}"
            || group.knob_tuple_json.is_empty()
        {
            String::new() // empty knob grid -> zen-metrics defaults
        } else {
            knob_tuple_to_grid_json(&group.knob_tuple_json)?
        };

        let codec = match CodecKind::from_str(&group.codec, true) {
            Ok(c) => c,
            Err(e) => {
                warn!(
                    chunk_id = %rec.chunk_id,
                    codec = %group.codec,
                    error = ?e,
                    "skip group: unknown codec"
                );
                groups_fail += 1;
                continue;
            }
        };
        let spec = InlineGroupSpec {
            codec,
            sources_dir: group_sources,
            q_grid: q_grid_str,
            knob_grid_json,
            metrics: metrics.clone(),
            gpu_runtime: GpuRuntime::Cuda,
            output_tsv: sweeps.join(format!("g{gid_str}.tsv")),
            // Per-group feature parquet — one per group; the final
            // chunk-level upload concatenates them. zen-metrics-cli's
            // `run_sweep` writes features inline when this is set
            // AND the metric list contains CPU zensim.
            feature_output: feature_out_path
                .as_ref()
                .map(|_| sweeps.join(format!("g{gid_str}.features.parquet"))),
            encoded_out_dir: Some(encoded.clone()),
            jobs: 0,
        };

        let span_chunk_id = rec.chunk_id.clone();
        let result = tokio::task::spawn_blocking(move || run_group_inline(spec))
            .await
            .map_err(|e| anyhow!("group {gid_str} task panicked: {e}"))?;
        match result {
            Ok(()) => {
                groups_ok += 1;
                info!(chunk_id = %span_chunk_id, gid = gid, "group ok");
            }
            Err(e) => {
                groups_fail += 1;
                warn!(chunk_id = %span_chunk_id, gid = gid, error = %e, "group failed");
                // Don't bail the chunk — drop the partial TSV and
                // keep going. The bash worker behaves the same.
                let f = sweeps.join(format!("g{gid_str}.tsv"));
                let _ = tokio::fs::remove_file(&f).await;
            }
        }
    }
    info!(
        chunk_id = %rec.chunk_id,
        groups_ok, groups_fail,
        "step 4/5 done"
    );
    if groups_ok == 0 {
        anyhow::bail!("no group produced output; abandoning chunk");
    }

    info!(chunk_id = %rec.chunk_id, "step 5/5: concat → parquet → upload");
    let sidecar_local = scratch.join(format!("{}.omni.parquet", rec.chunk_id));
    let chunk_id_owned = rec.chunk_id.clone();
    let run_id_owned = run_id.clone();
    let encoded_prefix_owned = if encoded_dir_has_files(&encoded).await {
        Some(out_encoded_prefix.clone())
    } else {
        None
    };
    let sweeps_clone = sweeps.clone();
    let sidecar_local_clone = sidecar_local.clone();
    let n_rows = tokio::task::spawn_blocking(move || {
        concat_groups_to_parquet(
            &sweeps_clone,
            &sidecar_local_clone,
            &chunk_id_owned,
            &run_id_owned,
            encoded_prefix_owned.as_deref(),
        )
    })
    .await
    .map_err(|e| anyhow!("concat task panicked: {e}"))??;
    info!(chunk_id = %rec.chunk_id, n_rows, "sidecar built");

    r2.upload(&sidecar_local, &out_sidecar)
        .await
        .context("upload sidecar")?;
    info!(chunk_id = %rec.chunk_id, sidecar = %out_sidecar, "sidecar uploaded");

    // Optional zensim feature parquet — concat per-group feature
    // sidecars + upload. Schema produced by zen-metrics-cli's
    // `feature_writer.rs`: `image_path codec q knob_tuple_json f0..f<N>`
    // joinable to the omni sidecar on the identity tuple.
    if let (Some(out_local), Some(out_uri)) = (&feature_out_path, &feature_out_r2) {
        let sweeps_clone = sweeps.clone();
        let out_local_clone = out_local.clone();
        let concat_result = tokio::task::spawn_blocking(move || {
            concat_feature_parquets(&sweeps_clone, &out_local_clone)
        })
        .await
        .map_err(|e| anyhow!("feature concat task panicked: {e}"))?;
        match concat_result {
            Ok(0) => {
                info!(chunk_id = %rec.chunk_id, "no zensim feature rows produced");
            }
            Ok(n) => {
                if let Err(e) = r2.upload(out_local, out_uri).await {
                    warn!(chunk_id = %rec.chunk_id, error = %e, "feature parquet upload failed");
                } else {
                    info!(
                        chunk_id = %rec.chunk_id,
                        n_rows = n,
                        feature_sidecar = %out_uri,
                        "zensim features uploaded"
                    );
                }
            }
            Err(e) => {
                warn!(chunk_id = %rec.chunk_id, error = %e, "feature concat failed");
            }
        }
    }

    // Encoded variants — only if any files actually got written.
    if encoded_dir_has_files(&encoded).await {
        upload_encoded_variants(r2, &encoded, &out_encoded_prefix).await?;
        info!(chunk_id = %rec.chunk_id, "encoded variants uploaded");
    } else {
        info!(chunk_id = %rec.chunk_id, "no encoded variants to upload");
    }

    // Cleanup unless KEEP_WORK=1 in env.
    if std::env::var_os("KEEP_WORK").is_none() {
        let _ = tokio::fs::remove_dir_all(&scratch).await;
    }
    Ok(())
}

/// Parse the METRICS env (comma-list) into the typed enum. The
/// production omni default is the six GPU metrics; we keep that
/// alignment so the Rust worker is a drop-in for the bash one.
fn parse_metrics_env_or_default() -> Vec<MetricKind> {
    let raw = std::env::var("METRICS").unwrap_or_else(|_| {
        "zensim-gpu,ssim2-gpu,butteraugli-gpu,cvvdp,dssim-gpu,iwssim-gpu".to_string()
    });
    let mut out = Vec::new();
    for name in raw.split(',') {
        let n = name.trim();
        if n.is_empty() {
            continue;
        }
        match MetricKind::from_str(n, true) {
            Ok(m) => out.push(m),
            Err(e) => warn!(metric = %n, error = ?e, "skip unknown metric"),
        }
    }
    if out.is_empty() {
        warn!("no metrics parsed; defaulting to cvvdp");
        out.push(MetricKind::Cvvdp);
    }
    out
}

async fn sync_sources(
    r2: &R2Client,
    source_dir_r2: &str,
    basenames: &[String],
    local_sources: &std::path::Path,
) -> Result<()> {
    // s5cmd's `run` mode reads a list of `cp src dst` commands and
    // executes them in parallel. We build the run file then exec.
    let mut run_lines = String::new();
    for b in basenames {
        let src = format!("{source_dir_r2}/{b}");
        let dst = local_sources.join(b);
        run_lines.push_str(&format!(
            "cp {} {}\n",
            src,
            dst.to_string_lossy()
        ));
    }
    let run_file = local_sources.join("_dl.run");
    tokio::fs::write(&run_file, run_lines).await
        .with_context(|| format!("write run file {}", run_file.display()))?;

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

async fn encoded_dir_has_files(dir: &std::path::Path) -> bool {
    let Ok(mut rd) = tokio::fs::read_dir(dir).await else {
        return false;
    };
    while let Ok(Some(entry)) = rd.next_entry().await {
        if entry.path().is_file() {
            return true;
        }
    }
    false
}

async fn upload_encoded_variants(
    r2: &R2Client,
    local_dir: &std::path::Path,
    r2_prefix: &str,
) -> Result<()> {
    let out = tokio::process::Command::new(&r2.bin)
        .arg("--endpoint-url")
        .arg(&r2.endpoint)
        .arg("--profile")
        .arg(&r2.profile)
        .arg("cp")
        .arg("--concurrency")
        .arg("8")
        // s5cmd's wildcard expansion needs the trailing slash.
        .arg(format!("{}/*", local_dir.to_string_lossy()))
        .arg(r2_prefix)
        .kill_on_drop(true)
        .output()
        .await
        .context("spawn s5cmd cp encoded")?;
    if !out.status.success() {
        // Non-fatal: log and continue. The sidecar is already up.
        warn!(
            status = %out.status,
            stderr = %String::from_utf8_lossy(&out.stderr),
            "encoded upload failed (non-fatal)"
        );
    }
    Ok(())
}

/// Concatenate per-group zensim feature parquets into one chunk-level
/// parquet. Each group's `g<gid>.features.parquet` has the schema
/// emitted by `zen_metrics_cli::sweep::feature_writer` —
/// `image_path:Utf8, codec:Utf8, q:UInt32, knob_tuple_json:Utf8,
/// zensim_score:Float32, feat_0..feat_299:Float32`. We read all
/// of them with arrow-rs's parquet reader, concat into one batch,
/// and write a single zstd parquet at `output_path`.
///
/// Returns the row count written. Returns Ok(0) if no per-group
/// feature parquets exist (e.g. CPU zensim wasn't in the metric
/// set, or all groups failed before writing any features).
#[cfg(feature = "inline-sweep")]
fn concat_feature_parquets(sweep_dir: &std::path::Path, output_path: &std::path::Path) -> Result<usize> {
    use arrow::compute::concat_batches;
    use arrow_array::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::basic::Compression;
    use parquet::file::properties::WriterProperties;
    use std::sync::Arc;

    let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(sweep_dir)
        .with_context(|| format!("read sweep dir {}", sweep_dir.display()))?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            let name = p.file_name()?.to_string_lossy().into_owned();
            // Per-group feature parquets are named "g<gid>.features.parquet".
            if name.starts_with('g') && name.ends_with(".features.parquet") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    if files.is_empty() {
        return Ok(0);
    }

    let mut batches: Vec<RecordBatch> = Vec::new();
    let mut schema_arc: Option<Arc<arrow_schema::Schema>> = None;
    for f in &files {
        let file = std::fs::File::open(f)
            .with_context(|| format!("open {}", f.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("init reader {}", f.display()))?;
        if schema_arc.is_none() {
            schema_arc = Some(builder.schema().clone());
        }
        let reader = builder.build()
            .with_context(|| format!("build reader {}", f.display()))?;
        for batch in reader {
            batches.push(batch.with_context(|| format!("read batch from {}", f.display()))?);
        }
    }
    if batches.is_empty() {
        return Ok(0);
    }
    let schema = schema_arc.expect("schema set when batches present");
    let merged = concat_batches(&schema, &batches).context("concat feature batches")?;
    let n_rows = merged.num_rows();

    let out_file = std::fs::File::create(output_path)
        .with_context(|| format!("create {}", output_path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut wtr = ArrowWriter::try_new(out_file, schema, Some(props))
        .context("arrow writer for features")?;
    wtr.write(&merged).context("write feature batch")?;
    wtr.close().context("close feature writer")?;
    Ok(n_rows)
}
