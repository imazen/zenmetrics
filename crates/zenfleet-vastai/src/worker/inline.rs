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
use zenmetrics_cli::metrics::{GpuRuntime, MetricKind, ZensimFeatureRegime};
use zenmetrics_cli::sweep::CodecKind;

use super::WorkerArgs;
use super::chunk_input::read_and_group;
use super::chunk_output::{WorkerColumns, concat_groups_to_parquet_with_worker};
use super::r2::R2Client;
use super::sweep_runner::{InlineGroupSpec, knob_tuple_to_grid_json, run_group_inline};

fn unix_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build a [`WorkerColumns`] from env + the captured timing.
/// `chunk_start_unix` is the unix-ts when the worker began processing
/// this chunk; `chunk_end_unix` is the moment we serialize the sidecar.
fn build_worker_columns(start: i64, end: i64) -> WorkerColumns {
    // Prefer Salad-injected machine_id; fall back to hostname or
    // worker id env.
    let machine_id = std::env::var("SALAD_MACHINE_ID")
        .or_else(|_| std::env::var("HOSTNAME"))
        .or_else(|_| std::env::var("WORKER_ID"))
        .unwrap_or_default();
    let gpu_class = std::env::var("ZEN_BOOT_GPU_CLASS").unwrap_or_default();
    let warmup_seconds = std::env::var("ZEN_BOOT_WARMUP_SECONDS")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok());
    WorkerColumns {
        machine_id,
        gpu_class,
        chunk_claim_unix: Some(start), // claim and start coincide in this worker's flow
        chunk_start_unix: Some(start),
        chunk_end_unix: Some(end),
        warmup_seconds,
    }
}

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
    /// HDR sweep mode (v27 schema addition, 2026-06-12; absent = false,
    /// so every v26 chunks.jsonl deserialises unchanged). When set, the
    /// chunk's groups run the HDR pipeline — PQ-PNG refs to nits, the
    /// HDR codec round-trip (zenjxl today), HdrScorer feedings — and
    /// every TSV/omni row carries the `hdr_mode` column. Builds without
    /// zenmetrics-cli's `hdr` feature fail such chunks loudly.
    #[serde(default)]
    hdr: bool,
}

/// Top-level entry point. The caller (chunk.rs::process_chunk) has
/// already won the claim race for this chunk.
///
/// On any failure, best-effort uploads the error string + context to a
/// durable R2 sidecar at `s3://<out-bucket>/<run_id>/errors/<chunk_id>.txt`
/// (see [`best_effort_error_sidecar`]) so fleet failures are diagnosable
/// without a logging provider — the container can die immediately after
/// this returns and the error survives in R2. The capture is non-fatal:
/// the ORIGINAL error is always returned regardless of upload success.
pub async fn process_chunk_inline(args: &WorkerArgs, r2: &R2Client, line: &str) -> Result<()> {
    let rec: ChunkRecord = serde_json::from_str(line).context("parse chunk JSON")?;
    let run_id = rec.run_id.clone().unwrap_or_else(|| args.run_id.clone());

    match run_chunk_inline_impl(args, r2, &rec, &run_id).await {
        Ok(()) => Ok(()),
        Err(e) => {
            // Durable, best-effort error capture. Never mask the real error.
            best_effort_error_sidecar(args, r2, &rec, &run_id, &format!("{e:#}")).await;
            Err(e)
        }
    }
}

/// Build the R2 URI for a chunk's durable error sidecar.
///
/// Derives the path from the chunk's `out_sidecar_omni` (an
/// `s3://<bucket>/<prefix>/omni/<chunk>.parquet` URI) by swapping the
/// trailing `omni/<chunk>.parquet` segment for `errors/<chunk>.txt`.
/// This preserves whatever scoped-cred prefix the launcher used for the
/// omni sidecar (e.g. `runs/<sweep_id>/`), so the error upload lands
/// under the SAME scoped prefix the worker has write access to.
///
/// Historical bug (fixed 2026-05-28): the prior implementation hard-coded
/// `s3://<bucket>/<run_id>/errors/...` which DROPPED any `runs/` (or other)
/// prefix the launcher had baked into `out_sidecar_omni`. With a scoped
/// cred restricted to `runs/<sweep_id>/`, every error-sidecar upload 403'd
/// silently — masking the underlying chunk failure across all of Salad
/// Runs 1-6b 2026-05-28. The path-derivation form below stays correctly
/// inside the scope as long as the launcher's omni-sidecar path does.
///
/// Falls back to the production `zentrain` bucket when the chunk record
/// omits the sidecar.
fn error_sidecar_uri(rec: &ChunkRecord, run_id: &str) -> String {
    // Path-derivation form: take `s3://<bucket>/<prefix>/omni/<chunk>.parquet`
    // and rewrite the trailing `omni/<chunk>.parquet` to `errors/<chunk>.txt`.
    // Preserves <prefix> verbatim so a scoped-cred prefix like `runs/<sweep>/`
    // still covers the error upload.
    if let Some(omni) = rec.out_sidecar_omni.as_deref()
        && let Some(rest) = omni.strip_prefix("s3://")
    {
        // Find the LAST `/omni/` segment so we don't accidentally match an
        // earlier path component named `omni`. Fall back to the bucket-only
        // form if the omni path doesn't have an `/omni/` segment we
        // recognise.
        if let Some(idx) = rest.rfind("/omni/") {
            let prefix = &rest[..idx]; // `<bucket>/<prefix>` (no trailing slash)
            return format!("s3://{prefix}/errors/{}.txt", rec.chunk_id);
        }
    }
    // Fallback: extract bucket from out_sidecar_omni (or default to zentrain),
    // emit the legacy `<bucket>/<run_id>/errors/` form. This path is OUTSIDE
    // any `runs/` scoped prefix the launcher may have set — callers relying
    // on this branch need to either set out_sidecar_omni with the full
    // `runs/<sweep>/omni/<chunk>.parquet` shape OR widen the scoped cred to
    // cover the bucket-root-level `<run_id>/` prefix.
    let bucket = rec
        .out_sidecar_omni
        .as_deref()
        .and_then(|uri| uri.strip_prefix("s3://"))
        .and_then(|rest| rest.split('/').next())
        .filter(|b| !b.is_empty())
        .unwrap_or("zentrain");
    format!("s3://{bucket}/{run_id}/errors/{}.txt", rec.chunk_id)
}

/// Best-effort durable error capture. Writes a small text file with the
/// chunk_id, run_id, hostname, and the full anyhow error chain, then
/// uploads it to R2 via the worker's existing client + scoped cred. All
/// failures are swallowed (logged at `warn`) — this MUST NOT mask the
/// original chunk error.
async fn best_effort_error_sidecar(
    args: &WorkerArgs,
    r2: &R2Client,
    rec: &ChunkRecord,
    run_id: &str,
    error: &str,
) {
    let uri = error_sidecar_uri(rec, run_id);
    // Identify the node: HOSTNAME (Linux container default) → SALAD_MACHINE_ID
    // → WORKER_ID → "unknown". Salad injects SALAD_MACHINE_ID / SALAD_CONTAINER_GROUP_ID.
    let hostname = std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("SALAD_MACHINE_ID"))
        .or_else(|_| std::env::var("WORKER_ID"))
        .unwrap_or_else(|_| "unknown".to_string());
    let salad_machine = std::env::var("SALAD_MACHINE_ID").unwrap_or_default();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "chunk_id: {}\nrun_id: {}\nhostname: {}\nsalad_machine_id: {}\nunix_ts: {}\n\
         input_parquet_r2: {}\nsource_dir_r2: {}\n\nerror:\n{}\n",
        rec.chunk_id,
        run_id,
        hostname,
        salad_machine,
        ts,
        rec.input_parquet_r2,
        rec.source_dir_r2,
        error,
    );
    // Write to a temp file under the scratch workdir (already on a writable
    // mount) then upload. If the scratch dir doesn't exist, fall back to the
    // workdir root.
    let scratch = args.workdir.join(&rec.chunk_id);
    let local = if tokio::fs::metadata(&scratch).await.is_ok() {
        scratch.join("_error.txt")
    } else {
        args.workdir.join(format!("{}._error.txt", rec.chunk_id))
    };
    if let Err(e) = tokio::fs::write(&local, body.as_bytes()).await {
        warn!(chunk_id = %rec.chunk_id, error = %e, "error-sidecar local write failed (non-fatal)");
        return;
    }
    match r2.upload(&local, &uri).await {
        Ok(()) => info!(chunk_id = %rec.chunk_id, uri = %uri, "durable error sidecar uploaded"),
        Err(e) => {
            warn!(chunk_id = %rec.chunk_id, uri = %uri, error = %e, "error-sidecar R2 upload failed (non-fatal)")
        }
    }
}

/// The chunk pipeline body. Separated from [`process_chunk_inline`] so the
/// latter can wrap it with durable error capture without threading the
/// capture through every `?` early-return.
async fn run_chunk_inline_impl(
    args: &WorkerArgs,
    r2: &R2Client,
    rec: &ChunkRecord,
    run_id: &str,
) -> Result<()> {
    let run_id = run_id.to_string();
    let chunk_start_unix = unix_now();
    let scratch = args.workdir.join(&rec.chunk_id);
    let sources = scratch.join("sources");
    let sweeps = scratch.join("sweeps");
    let encoded = scratch.join("encoded");
    for dir in [&sources, &sweeps, &encoded] {
        tokio::fs::create_dir_all(dir)
            .await
            .with_context(|| format!("mkdir {}", dir.display()))?;
    }

    let out_sidecar = rec
        .out_sidecar_omni
        .clone()
        .unwrap_or_else(|| format!("s3://zentrain/{run_id}/omni/{}.parquet", rec.chunk_id));
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
    let feature_regime = parse_feature_regime_env_or_default();
    // If CPU `zensim` OR GPU `zensim-gpu` is in the metric set, also
    // write the regime-appropriate feature vector to a parquet sidecar
    // at s3://zentrain/<run>/zensim_features/<chunk>.parquet. Joins
    // back to the omni sidecar by
    // `(image_path, codec, q, knob_tuple_json)`. CPU emits 300 floats;
    // GPU honours `feature_regime` (default WithIw = 372).
    let want_features =
        metrics.contains(&MetricKind::Zensim) || metrics.contains(&MetricKind::ZensimGpu);
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
        tokio::fs::create_dir_all(&group_sources)
            .await
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
        let knob_grid_json = if group.knob_tuple_json == "{}" || group.knob_tuple_json.is_empty() {
            String::new() // empty knob grid -> zenmetrics defaults
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
            // The parquet-driven backfill flow re-encodes concrete
            // per-cell knob tuples; plan-driven cells are for fresh
            // sweeps (CLI --plan or a jobspec that sets these).
            plan: None,
            plan_budget: None,
            plan_compute_limit: None,
            plan_max_deviations: None,
            metrics: metrics.clone(),
            gpu_runtime: GpuRuntime::Cuda,
            output_tsv: sweeps.join(format!("g{gid_str}.tsv")),
            // Per-group feature parquet — one per group; the final
            // chunk-level upload concatenates them. zenmetrics-cli's
            // `run_sweep` writes features inline when this is set
            // AND the metric list contains CPU zensim OR ZensimGpu.
            feature_output: feature_out_path
                .as_ref()
                .map(|_| sweeps.join(format!("g{gid_str}.features.parquet"))),
            feature_regime,
            encoded_out_dir: Some(encoded.clone()),
            jobs: parse_jobs_env_or_default(),
            hdr: rec.hdr,
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
    let worker_cols = build_worker_columns(chunk_start_unix, unix_now());
    let n_rows = tokio::task::spawn_blocking(move || {
        concat_groups_to_parquet_with_worker(
            &sweeps_clone,
            &sidecar_local_clone,
            &chunk_id_owned,
            &run_id_owned,
            encoded_prefix_owned.as_deref(),
            Some(&worker_cols),
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
    // sidecars + upload. Schema produced by zenmetrics-cli's
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

/// Parse the `JOBS` env var into a rayon worker count. `0` (default)
/// = num_cpus auto-detect. Set `JOBS=1` to disable cell-level
/// parallelism, which is the only way to keep concurrent cubecl
/// allocations from saturating the pool on a memory-constrained card
/// (12 GB RTX 3060 with v26's 372-feature regime needs JOBS=1 to fit).
fn parse_jobs_env_or_default() -> usize {
    match std::env::var("JOBS") {
        Ok(s) => s.trim().parse::<usize>().unwrap_or_else(|_| {
            warn!(value = %s, "JOBS env not parseable as usize; defaulting to 0 (auto)");
            0
        }),
        Err(_) => 0,
    }
}

/// Parse the `ZENSIM_FEATURES_REGIME` env var (`basic` / `extended` /
/// `with_iw`) into the typed enum. Defaults to
/// [`ZensimFeatureRegime::WithIw`] (372 features) — the v26+ default.
fn parse_feature_regime_env_or_default() -> ZensimFeatureRegime {
    let raw = match std::env::var("ZENSIM_FEATURES_REGIME") {
        Ok(s) => s,
        Err(_) => return ZensimFeatureRegime::WithIw,
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "basic" => ZensimFeatureRegime::Basic,
        "extended" => ZensimFeatureRegime::Extended,
        "with-iw" | "with_iw" | "withiw" | "iw" => ZensimFeatureRegime::WithIw,
        other => {
            warn!(
                value = %other,
                "unknown ZENSIM_FEATURES_REGIME; falling back to with-iw (372)"
            );
            ZensimFeatureRegime::WithIw
        }
    }
}

/// Parse the METRICS env (comma-list) into the typed enum. The
/// production omni default is the five GPU metrics that ship six
/// score columns total (butteraugli emits both max and pnorm3).
///
/// iwssim-gpu is intentionally NOT in the default set: it has a
/// 176-pixel minimum dimension which causes hard failures on every
/// gif / small wikimedia image in the corpus AND its cubecl pool
/// footprint contributes ~16% of pool pressure per cell. Operators
/// who want iwssim coverage on a 24 GB+ box must pass METRICS
/// explicitly including `iwssim-gpu`.
fn parse_metrics_env_or_default() -> Vec<MetricKind> {
    let raw = std::env::var("METRICS")
        .unwrap_or_else(|_| "zensim-gpu,ssim2-gpu,butteraugli-gpu,cvvdp,dssim-gpu".to_string());
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
        run_lines.push_str(&format!("cp {} {}\n", src, dst.to_string_lossy()));
    }
    let run_file = local_sources.join("_dl.run");
    tokio::fs::write(&run_file, run_lines)
        .await
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
/// emitted by `zenmetrics_cli::sweep::feature_writer` —
/// `image_path:Utf8, codec:Utf8, q:UInt32, knob_tuple_json:Utf8,
/// zensim_score:Float32, feat_0..feat_299:Float32`. We read all
/// of them with arrow-rs's parquet reader, concat into one batch,
/// and write a single zstd parquet at `output_path`.
///
/// Returns the row count written. Returns Ok(0) if no per-group
/// feature parquets exist (e.g. CPU zensim wasn't in the metric
/// set, or all groups failed before writing any features).
#[cfg(feature = "inline-sweep")]
fn concat_feature_parquets(
    sweep_dir: &std::path::Path,
    output_path: &std::path::Path,
) -> Result<usize> {
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
        let file = std::fs::File::open(f).with_context(|| format!("open {}", f.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .with_context(|| format!("init reader {}", f.display()))?;
        if schema_arc.is_none() {
            schema_arc = Some(builder.schema().clone());
        }
        let reader = builder
            .build()
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
    let mut wtr =
        ArrowWriter::try_new(out_file, schema, Some(props)).context("arrow writer for features")?;
    wtr.write(&merged).context("write feature batch")?;
    wtr.close().context("close feature writer")?;
    Ok(n_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec_with_sidecar(sidecar: Option<&str>) -> ChunkRecord {
        let sidecar_json = match sidecar {
            Some(s) => format!(r#","out_sidecar_omni":"{s}""#),
            None => String::new(),
        };
        let line = format!(
            r#"{{"chunk_id":"salad-smoke-001","input_parquet":"in.parquet",
               "input_parquet_r2":"s3://b/in.parquet","row_range":[0,1],
               "source_dir_r2":"s3://b/src","image_basenames":["a.png"]{sidecar_json}}}"#
        );
        serde_json::from_str(&line).expect("parse test chunk record")
    }

    #[test]
    fn error_sidecar_uri_derives_path_from_out_sidecar() {
        // Legacy layout (no `runs/` prefix): omni path is
        // `<bucket>/<run_id>/omni/<chunk>.parquet`; error becomes
        // `<bucket>/<run_id>/errors/<chunk>.txt` — same prefix.
        let rec = rec_with_sidecar(Some(
            "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/omni/salad-smoke-001.parquet",
        ));
        assert_eq!(
            error_sidecar_uri(&rec, "salad-smoke-2026-05-27"),
            "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/errors/salad-smoke-001.txt"
        );
    }

    #[test]
    fn error_sidecar_uri_preserves_runs_prefix_for_scoped_cred() {
        // Current launcher layout: omni path is
        // `<bucket>/runs/<sweep_id>/omni/<chunk>.parquet`. The error
        // sidecar MUST land at `<bucket>/runs/<sweep_id>/errors/<chunk>.txt`
        // — preserving the `runs/<sweep_id>/` prefix so a scoped-cred
        // restricted to `runs/<sweep_id>/` still permits the upload.
        // This is the bug exposed by Salad Runs 1-6b 2026-05-28: the
        // prior URI form dropped the `runs/` prefix and every error
        // upload 403'd, masking the chunk failures entirely.
        // rec_with_sidecar hard-codes chunk_id = "salad-smoke-001"; the
        // out_sidecar_omni path uses a different stem for clarity but the
        // assertion's chunk_id MUST match the helper's hard-coded value.
        let rec = rec_with_sidecar(Some(
            "s3://zen-tuning-ephemeral/runs/n1-v5-1779959731/omni/scaleup-000.parquet",
        ));
        // run_id is the LAUNCHER's sweep_id (no `runs/` prefix). The URI
        // must still anchor under `runs/n1-v5-1779959731/`.
        assert_eq!(
            error_sidecar_uri(&rec, "n1-v5-1779959731"),
            "s3://zen-tuning-ephemeral/runs/n1-v5-1779959731/errors/salad-smoke-001.txt"
        );
    }

    #[test]
    fn error_sidecar_uri_falls_back_to_zentrain_without_sidecar() {
        let rec = rec_with_sidecar(None);
        assert_eq!(
            error_sidecar_uri(&rec, "run-x"),
            "s3://zentrain/run-x/errors/salad-smoke-001.txt"
        );
    }

    #[test]
    fn error_sidecar_uri_handles_nested_prefix() {
        // Defensive: deeper prefix like `runs/<sweep>/<sub>/omni/...` —
        // confirm we strip the LAST `/omni/<chunk>.parquet`, not an
        // earlier one.
        let rec = rec_with_sidecar(Some("s3://b/runs/x/omni-subdir/omni/c.parquet"));
        assert_eq!(
            error_sidecar_uri(&rec, "x"),
            "s3://b/runs/x/omni-subdir/errors/salad-smoke-001.txt"
        );
    }
}
