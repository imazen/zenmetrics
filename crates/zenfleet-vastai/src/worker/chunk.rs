//! Per-chunk processor.
//!
//! ## Phase A: subprocess to bash worker
//!
//! For now this just calls `omni_backfill_chunk_worker.sh --chunk-json <line>`
//! and propagates the exit status. The bash script already handles:
//!
//! 1. Downloading the input parquet + source PNGs.
//! 2. Grouping cells by `(codec, knob_tuple_json)`.
//! 3. Running `zenmetrics sweep` per group.
//! 4. Combining per-group TSVs into one parquet sidecar.
//! 5. Uploading the sidecar to R2.
//!
//! Keeping it as a subprocess for phase A means we get the Rust
//! dispatcher's reliability + adaptive concurrency without touching
//! the chunk-level logic that's already producing correct output.
//!
//! ## Phase B: in-process via `zenmetrics_cli::sweep::run_sweep`
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

use std::time::Instant;
// The bash-subprocess execute path (and the `Stdio` / `Command` / `anyhow!`
// items it alone uses) only compiles when `inline-sweep` is OFF — the inline
// Rust pipeline is the sole execute path otherwise, so these would be unused.
#[cfg(not(feature = "inline-sweep"))]
use std::process::Stdio;

#[cfg(not(feature = "inline-sweep"))]
use anyhow::anyhow;
use anyhow::{Context, Result};
use serde::Deserialize;
#[cfg(not(feature = "inline-sweep"))]
use tokio::process::Command;
use tracing::{info, warn};

use super::WorkerArgs;
use super::claim::{ClaimConfig, ClaimOutcome, try_claim};
use super::r2::R2Client;

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

/// Process one chunk: claim it, run the execute path, log. With
/// `inline-sweep` (the default + production build) the execute path is the
/// in-process Rust pipeline ([`super::inline::process_chunk_inline`]); a
/// failure there fails honestly (its durable error sidecar is already in R2).
/// Without `inline-sweep` the execute path is the bash subprocess
/// ([`run_chunk_via_bash`]). The caller in [`super`] logs + counts the
/// returned error without killing the dispatcher; the next chunk proceeds.
pub async fn process_chunk(
    args: &WorkerArgs,
    worker_id: &str,
    r2: &R2Client,
    line: &str,
) -> Result<()> {
    let rec = parse_chunk_json(line)?;
    let sidecar_uri = rec.out_sidecar_omni.clone().unwrap_or_else(|| {
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

    // With `inline-sweep` (default + production), the in-process Rust pipeline
    // is the sole execute path. Each mode returns its own result; an inline
    // failure FAILS HONESTLY (see the omni `Err` arm below) — there is no bash
    // fallback. The bash subprocess is gated out of this build entirely.
    #[cfg(feature = "inline-sweep")]
    {
        if args.mode == "feature-backfill" {
            return match super::backfill_features_for_chunk(args, r2, line).await {
                Ok(()) => {
                    info!(
                        chunk_id = %rec.chunk_id,
                        elapsed_sec = started.elapsed().as_secs_f32(),
                        "done (feature-backfill)"
                    );
                    Ok(())
                }
                Err(e) => {
                    warn!(
                        chunk_id = %rec.chunk_id,
                        error = %e,
                        "feature-backfill failed"
                    );
                    Err(e)
                }
            };
        }
        #[cfg(feature = "source-features")]
        if args.mode == "source-features" {
            return match super::backfill_source_features_for_chunk(args, r2, line).await {
                Ok(()) => {
                    info!(
                        chunk_id = %rec.chunk_id,
                        elapsed_sec = started.elapsed().as_secs_f32(),
                        "done (source-features)"
                    );
                    Ok(())
                }
                Err(e) => {
                    warn!(
                        chunk_id = %rec.chunk_id,
                        error = %e,
                        "source-features failed"
                    );
                    Err(e)
                }
            };
        }
        return match super::inline::process_chunk_inline(args, r2, line).await {
            Ok(()) => {
                info!(
                    chunk_id = %rec.chunk_id,
                    elapsed_sec = started.elapsed().as_secs_f32(),
                    "done (inline)"
                );
                Ok(())
            }
            Err(e) => {
                // FAIL HONESTLY — do NOT fall back to the bash subprocess.
                // `process_chunk_inline` has ALREADY uploaded a durable
                // Failed→R2 marker for this chunk; the claim/atomic-sidecar
                // discipline lets a peer retry the (still-not-marked-done)
                // chunk later. Re-running it through the deprecated bash
                // worker would be a DIVERGENT code path (W44 incident: it
                // discards encoded bytes, keeping only `len() as u32`). A bash
                // "success" after an inline failure would then leave a
                // Failed→R2 error marker AND a contradictory omni sidecar
                // produced by the wrong pipeline. Match the `feature-backfill`
                // / `source-features` arms above and surface the real error.
                warn!(
                    chunk_id = %rec.chunk_id,
                    error = %e,
                    "inline pipeline failed"
                );
                Err(e)
            }
        };
    }

    // The bash subprocess is the execute path ONLY when `inline-sweep` is NOT
    // compiled in — there it is the sole pipeline by design.
    #[cfg(not(feature = "inline-sweep"))]
    run_chunk_via_bash(args, &rec, line, started).await
}

/// Phase A bash-subprocess execute path: shells out to
/// `omni_backfill_chunk_worker.sh --chunk-json <line>` and propagates the exit
/// status. This is the SOLE execute path when the crate is built WITHOUT the
/// `inline-sweep` feature; the inline Rust pipeline supersedes it in the
/// default (and production) build.
#[cfg(not(feature = "inline-sweep"))]
async fn run_chunk_via_bash(
    args: &WorkerArgs,
    rec: &ChunkRecord,
    line: &str,
    started: Instant,
) -> Result<()> {
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
