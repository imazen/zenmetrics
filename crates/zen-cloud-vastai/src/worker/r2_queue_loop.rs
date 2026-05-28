//! R2-queue polling loop — used by backends without a managed sidecar
//! queue (Hetzner being the first consumer; future bare-metal / VPS
//! backends can reuse the same loop).
//!
//! Polls `s3://<bucket>/runs/<sweep_id>/queue/<chunk_id>.json` on a
//! loop, downloads one queue entry, runs it through the shared inline
//! pipeline ([`super::inline::process_chunk_inline`]), and deletes
//! the queue entry on success.
//!
//! ## Idempotency
//!
//! Two workers may LIST the queue prefix in the same window and both
//! pick the same alphabetic-first entry. That's the expected race —
//! the omni-sidecar dedup pattern we validated on vast.ai iter2/3
//! reconciles it: the SECOND worker that runs the same chunk produces
//! a duplicate sidecar; the fleet_summary stitch keeps the oldest
//! `worker_chunk_start_unix` row per chunk_id. The queue delete is
//! idempotent (a 404 is treated as success).
//!
//! ## Termination
//!
//! The loop runs until:
//! - LIST returns empty AND a configurable idle window has elapsed
//!   (default 5 min — workers cooperate with the launcher's
//!   speculative re-dispatch / TTL window without dying mid-sweep).
//! - The process receives SIGTERM (cooperative shutdown).
//! - The per-process chunk cap is hit (`MAX_CHUNKS_PER_PROCESS`).

#![cfg(feature = "inline-sweep")]

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use tracing::{info, warn};

use super::WorkerArgs;
use super::inline::process_chunk_inline;
use super::r2::R2Client;

/// Configuration for the R2-queue polling loop.
#[derive(Debug, Clone)]
pub struct R2QueueLoopConfig {
    /// R2 bucket the queue lives in.
    pub bucket: String,
    /// Prefix WITH trailing slash — e.g. `runs/<sweep_id>/queue/`.
    pub prefix: String,
    /// Per-process chunk cap. 0 = no cap.
    pub max_chunks_per_process: usize,
    /// Empty-queue sleep interval (seconds).
    pub poll_secs: u64,
    /// After this many seconds of consecutive empty LISTs, exit
    /// cleanly so docker can recycle the container.
    pub idle_exit_secs: u64,
}

impl Default for R2QueueLoopConfig {
    fn default() -> Self {
        Self {
            bucket: String::new(),
            prefix: String::new(),
            max_chunks_per_process: 0,
            poll_secs: 10,
            idle_exit_secs: 300,
        }
    }
}

impl R2QueueLoopConfig {
    /// Build from the standard worker env vars.
    pub fn from_env() -> Result<Self> {
        let bucket = std::env::var("BUCKET")
            .context("BUCKET env not set (R2 bucket for the queue)")?;
        let prefix = std::env::var("CHUNKS_QUEUE_PREFIX").context(
            "CHUNKS_QUEUE_PREFIX env not set (expected `runs/<sweep_id>/queue/`)",
        )?;
        let max_chunks = std::env::var("MAX_CHUNKS_PER_PROCESS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let poll_secs = std::env::var("R2_QUEUE_POLL_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let idle_exit_secs = std::env::var("R2_QUEUE_IDLE_EXIT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        Ok(Self {
            bucket,
            prefix,
            max_chunks_per_process: max_chunks,
            poll_secs,
            idle_exit_secs,
        })
    }
}

/// Run the R2-queue polling loop.
///
/// Returns when the queue has been empty for `cfg.idle_exit_secs`
/// OR `cfg.max_chunks_per_process` chunks have been dispatched.
pub async fn run_r2_queue_loop(
    args: &WorkerArgs,
    r2: &R2Client,
    cfg: &R2QueueLoopConfig,
) -> Result<u32> {
    info!(
        bucket = %cfg.bucket,
        prefix = %cfg.prefix,
        poll_secs = cfg.poll_secs,
        idle_exit_secs = cfg.idle_exit_secs,
        max_chunks = cfg.max_chunks_per_process,
        "r2-queue loop starting"
    );
    let mut processed: u32 = 0;
    let mut last_nonempty = Instant::now();
    let poll_uri_prefix = format!("s3://{}/{}", cfg.bucket, cfg.prefix);

    loop {
        if cfg.max_chunks_per_process > 0
            && (processed as usize) >= cfg.max_chunks_per_process
        {
            info!(processed, "max_chunks_per_process reached; exiting loop");
            break;
        }

        let keys = match r2.ls_keys(&poll_uri_prefix).await {
            Ok(k) => k,
            Err(e) => {
                warn!(error = %e, "ls_keys failed; retrying after poll interval");
                tokio::time::sleep(Duration::from_secs(cfg.poll_secs)).await;
                continue;
            }
        };
        // Filter to `*.json` entries only.
        let mut queue_files: Vec<String> =
            keys.into_iter().filter(|k| k.ends_with(".json")).collect();
        queue_files.sort();

        if queue_files.is_empty() {
            let idle = last_nonempty.elapsed().as_secs();
            if idle >= cfg.idle_exit_secs {
                info!(idle_secs = idle, "queue idle past threshold; exiting loop");
                break;
            }
            tokio::time::sleep(Duration::from_secs(cfg.poll_secs)).await;
            continue;
        }
        last_nonempty = Instant::now();

        // Take the alphabetic-first one. Race: another worker may pick
        // the same. Sidecar idempotency reconciles dupes downstream.
        let key = &queue_files[0];
        let uri = format!("s3://{}/{}", cfg.bucket, key);
        info!(uri = %uri, "picking up queue entry");

        let body = match r2.cat(&uri).await {
            // R2Client::cat returns Vec<u8>; empty = not present / 403
            // / something already deleted (peer worker won the race).
            b if b.is_empty() => {
                tokio::time::sleep(Duration::from_secs(cfg.poll_secs)).await;
                continue;
            }
            b => b,
        };
        let body_str = match std::str::from_utf8(&body) {
            Ok(s) => s.to_string(),
            Err(e) => {
                warn!(uri = %uri, error = %e, "queue entry isn't valid UTF-8; deleting + skipping");
                let _ = r2.rm(&uri).await;
                continue;
            }
        };

        // Run the standard inline pipeline.
        match process_chunk_inline(args, r2, &body_str).await {
            Ok(()) => {
                info!(uri = %uri, "chunk processed; deleting queue entry");
                if let Err(e) = r2.rm(&uri).await {
                    warn!(uri = %uri, error = %e, "queue entry delete FAILED (chunk already done, harmless)");
                }
                processed += 1;
            }
            Err(e) => {
                // Don't delete the queue entry on failure — let TTL or
                // speculative re-dispatch surface it. The error sidecar
                // has already been written by process_chunk_inline.
                warn!(uri = %uri, error = %format!("{e:#}"), "chunk failed; leaving queue entry for re-dispatch");
                // Brief backoff before another LIST so we don't spin on
                // a persistently-failing chunk.
                tokio::time::sleep(Duration::from_secs(cfg.poll_secs)).await;
            }
        }
    }
    Ok(processed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_are_sane() {
        let cfg = R2QueueLoopConfig::default();
        assert!(cfg.poll_secs > 0);
        assert!(cfg.idle_exit_secs >= 60);
    }
}
