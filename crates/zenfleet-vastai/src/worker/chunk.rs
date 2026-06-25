//! Per-chunk processor.
//!
//! Production scoring runs **in-process** via
//! `zenmetrics_cli::sweep::run_sweep` — the `inline-sweep` feature, which is on
//! in every real build. For each chunk we parse the JSON, download the input
//! parquet + sources via [`super::r2::R2Client`], build a `SweepConfig`, call
//! `run_sweep` sharing ONE persistent cubecl device across chunks, and stream
//! rows into an Arrow `RecordBatch` written as parquet. One cubecl init per
//! process (not per group) roughly halves per-chunk wall time vs the old
//! per-group init.
//!
//! The earlier Phase-A design subprocessed to a bash
//! `omni_backfill_chunk_worker.sh`; that script and the subprocess path were
//! **removed 2026-06-25**. A build WITHOUT `inline-sweep` (e.g. the `vastai-min`
//! compile-check) has no chunk processor and [`process_chunk`] fails loudly at
//! runtime.

use std::time::Instant;

use anyhow::{Context, Result};
use serde::Deserialize;
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

/// Process one chunk: claim it, run the execute path, log. The execute path is
/// the in-process Rust pipeline ([`super::inline::process_chunk_inline`]); a
/// failure there fails honestly (its durable error sidecar is already in R2).
/// (Without the `inline-sweep` feature there is no processor — the chunk bails
/// loudly; see the module doc.) The caller in [`super`] logs + counts the
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

    // Without `inline-sweep` there is no chunk processor: the bash subprocess
    // fallback (`omni_backfill_chunk_worker.sh`) was removed 2026-06-25. Such a
    // build (e.g. the `vastai-min` compile-check) must never actually run a
    // chunk — fail loudly if it does, rather than silently no-op.
    #[cfg(not(feature = "inline-sweep"))]
    {
        let _ = started;
        anyhow::bail!(
            "chunk {} cannot be processed: this worker was built WITHOUT the \
             `inline-sweep` feature, and the bash chunk-worker fallback was \
             removed 2026-06-25 — rebuild with `--features inline-sweep`",
            rec.chunk_id
        )
    }
}
