//! Per-chunk processor.
//!
//! ## Phase A: subprocess to bash worker
//!
//! For now this just calls `omni_backfill_chunk_worker.sh --chunk-json <line>`
//! and propagates the exit status. The bash script already handles:
//!
//! 1. Downloading the input parquet + source PNGs.
//! 2. Grouping cells by `(codec, knob_tuple_json)`.
//! 3. Running `zen-metrics sweep` per group.
//! 4. Combining per-group TSVs into one parquet sidecar.
//! 5. Uploading the sidecar to R2.
//!
//! Keeping it as a subprocess for phase A means we get the Rust
//! dispatcher's reliability + adaptive concurrency without touching
//! the chunk-level logic that's already producing correct output.
//!
//! ## Phase B: in-process via `zen_metrics_cli::sweep::run_sweep`
//!
//! Phase B will:
//!
//! 1. Parse the chunk JSON in Rust (already what
//!    [`parse_chunk_json`] below does for the claim step).
//! 2. Download input parquet + sources via [`super::r2::R2Client`].
//! 3. Build a `SweepConfig` and call `run_sweep` directly, sharing
//!    a single persistent cubecl device across chunks.
//! 4. Stream rows directly into an Arrow `RecordBatch` and write
//!    parquet via `parquet::arrow::ArrowWriter`.
//!
//! Phase B's killer feature is *one cubecl init per process* instead
//! of one per group. The 4-5 min/chunk wall time is dominated by
//! cubecl init overhead × 30 groups; phase B should drop it ~2x.

use std::process::Stdio;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use tokio::process::Command;
use tracing::{info, warn};

use super::claim::{ClaimConfig, ClaimOutcome, try_claim};
use super::r2::R2Client;
use super::WorkerArgs;

/// Subset of the chunks.jsonl record we need at the dispatcher
/// level. The bash worker re-parses the full record itself so we
/// just need the fields used for claiming + idempotency here.
#[derive(Debug, Deserialize)]
pub struct ChunkRecord {
    pub chunk_id: String,
    /// Optional explicit sidecar URL. If absent, we synthesize
    /// `s3://zentrain/<run>/omni/<chunk>.parquet`.
    #[serde(default)]
    pub out_sidecar_omni: Option<String>,
}

pub fn parse_chunk_json(line: &str) -> Result<ChunkRecord> {
    serde_json::from_str(line).context("parse chunk JSON")
}

/// Process one chunk: claim it, shell out to the bash worker, log.
/// Never returns an error that should kill the dispatcher — chunk
/// failures are logged + counted; the next chunk proceeds.
pub async fn process_chunk(
    args: &WorkerArgs,
    worker_id: &str,
    r2: &R2Client,
    line: &str,
) -> Result<()> {
    let rec = parse_chunk_json(line)?;
    let sidecar_uri = rec
        .out_sidecar_omni
        .clone()
        .unwrap_or_else(|| {
            format!(
                "s3://zentrain/{}/omni/{}.parquet",
                args.run_id, rec.chunk_id
            )
        });
    let claim_uri = format!(
        "s3://coefficient/claims/{}/{}.claim",
        args.run_id, rec.chunk_id
    );

    // Claim phase.
    if !args.skip_claims {
        let cfg = ClaimConfig::default();
        let outcome =
            try_claim(r2, worker_id, &rec.chunk_id, &sidecar_uri, &claim_uri, &cfg).await?;
        match outcome {
            ClaimOutcome::Acquired { token } => {
                info!(chunk_id = %rec.chunk_id, %token, "claimed");
            }
            ClaimOutcome::AlreadyDone => {
                info!(chunk_id = %rec.chunk_id, "skip: sidecar present");
                return Ok(());
            }
            ClaimOutcome::HeldByPeer => {
                info!(chunk_id = %rec.chunk_id, "skip: peer holds claim");
                return Ok(());
            }
            ClaimOutcome::LostRace => {
                info!(chunk_id = %rec.chunk_id, "skip: lost claim race");
                return Ok(());
            }
            ClaimOutcome::Errored => {
                warn!(chunk_id = %rec.chunk_id, "skip: claim errored");
                return Ok(());
            }
        }
    } else {
        info!(chunk_id = %rec.chunk_id, "SKIP_CLAIMS: processing without claim");
    }

    // Execute phase. Mode-dispatch:
    //   - `feature-backfill` reads existing omni sidecar + encoded
    //     variants and computes zensim features without re-encoding.
    //   - `omni` (default) runs the full encode+score+upload pipeline.
    let started = Instant::now();
    #[cfg(feature = "inline-sweep")]
    {
        if args.mode == "feature-backfill" {
            match super::backfill_features_for_chunk(args, r2, line).await {
                Ok(()) => {
                    info!(
                        chunk_id = %rec.chunk_id,
                        elapsed_sec = started.elapsed().as_secs_f32(),
                        "done (feature-backfill)"
                    );
                    return Ok(());
                }
                Err(e) => {
                    warn!(
                        chunk_id = %rec.chunk_id,
                        error = %e,
                        "feature-backfill failed"
                    );
                    return Err(e);
                }
            }
        }
        #[cfg(feature = "source-features")]
        if args.mode == "source-features" {
            match super::backfill_source_features_for_chunk(args, r2, line).await {
                Ok(()) => {
                    info!(
                        chunk_id = %rec.chunk_id,
                        elapsed_sec = started.elapsed().as_secs_f32(),
                        "done (source-features)"
                    );
                    return Ok(());
                }
                Err(e) => {
                    warn!(
                        chunk_id = %rec.chunk_id,
                        error = %e,
                        "source-features failed"
                    );
                    return Err(e);
                }
            }
        }
        match super::inline::process_chunk_inline(args, r2, line).await {
            Ok(()) => {
                info!(
                    chunk_id = %rec.chunk_id,
                    elapsed_sec = started.elapsed().as_secs_f32(),
                    "done (inline)"
                );
                return Ok(());
            }
            Err(e) => {
                warn!(
                    chunk_id = %rec.chunk_id,
                    error = %e,
                    "inline pipeline failed; falling back to bash subprocess"
                );
                // Fall through to bash subprocess as a safety net.
            }
        }
    }
    let mut cmd = Command::new(&args.chunk_worker_bin);
    cmd.arg("--chunk-json").arg(line);
    // Pass through env vars the bash worker reads. We don't override
    // anything the operator set in /proc/1/environ — child inherits.
    cmd.kill_on_drop(true);
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let out = cmd
        .output()
        .await
        .with_context(|| format!("spawn chunk worker for {}", rec.chunk_id))?;

    let elapsed = started.elapsed();
    if out.status.success() {
        info!(
            chunk_id = %rec.chunk_id,
            elapsed_sec = elapsed.as_secs_f32(),
            "done"
        );
        // Stream worker stdout to our stdout for parity with bash
        // onstart's `sed "s/^/  /"` decoration.
        if !out.stdout.is_empty() {
            for line in String::from_utf8_lossy(&out.stdout).lines() {
                println!("  [{}] {}", rec.chunk_id, line);
            }
        }
        Ok(())
    } else {
        // Non-zero exit. Log full stderr so the operator can diagnose.
        // We do NOT delete the claim — leaving it lets a peer notice
        // the claim is stale (>600s) and retry.
        for line in String::from_utf8_lossy(&out.stderr).lines() {
            warn!(chunk_id = %rec.chunk_id, worker_stderr = line);
        }
        Err(anyhow!(
            "chunk {} failed (exit {})",
            rec.chunk_id,
            out.status
        ))
    }
}
