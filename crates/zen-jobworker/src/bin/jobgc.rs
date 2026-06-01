#![forbid(unsafe_code)]
//! `zen-jobgc` — safe garbage collection over the blob index + ledger (goal G).
//!
//! Reachability mark-sweep: referenced blobs (output of Done ledger rows) are always kept; unreferenced
//! cheap-regenerable blobs are an LRU-capped cache (evicted oldest-first over the cap, lossless rebuild);
//! unreferenced **irreplaceable** blobs are never auto-deleted — surfaced for a human pin/archive call.
//! Each delete writes a tombstone first. **Dry-run by default**; pass `--execute` to actually delete.
//!
//!   zen-jobgc --blob-index s3://b/jobsys/dashboard-seed/blob_index.parquet \
//!     --ledger s3://b/jobsys/dashboard-seed/ledger.parquet \
//!     --blobs-r2 s3://b/blobs --tombstones-r2 s3://b/jobsys/tombstones \
//!     --r2-endpoint https://<acct>.r2.cloudflarestorage.com --cheap-cap-bytes 1000000 [--execute]

use std::collections::HashSet;

use clap::Parser;
use zen_job_core::{JobStatus, LedgerView, Sha256Hex};
use zen_jobworker::{GcExecCfg, gc_execute};

#[derive(Parser)]
#[command(
    name = "zen-jobgc",
    about = "Safe reachability GC: LRU-cap cheap blobs, refuse irreplaceable, tombstone every delete"
)]
struct Cli {
    /// Blob index (Parquet) — local path or s3:// URI.
    #[arg(long = "blob-index")]
    blob_index: String,
    /// Ledger sidecar(s) — local or s3:// — to compute the referenced set (Done rows' output_sha).
    #[arg(long = "ledger")]
    ledger: Vec<String>,
    /// R2 endpoint for any s3:// URI (+ AWS_* creds in env).
    #[arg(long = "r2-endpoint")]
    r2_endpoint: Option<String>,
    /// Base URI of the content-addressed blobs, e.g. s3://bucket/blobs.
    #[arg(long = "blobs-r2")]
    blobs_r2: String,
    /// Base URI for tombstones (one JSON per deleted blob). Optional but recommended.
    #[arg(long = "tombstones-r2")]
    tombstones_r2: Option<String>,
    /// Keep at most this many bytes of cheap-regenerable cache; evict the LRU tail above it.
    #[arg(long = "cheap-cap-bytes", default_value_t = 0)]
    cheap_cap_bytes: u64,
    /// Explicitly pinned shas (always kept), repeatable.
    #[arg(long = "pin")]
    pin: Vec<String>,
    /// Unix secs for tombstones (0 = system clock).
    #[arg(long, default_value_t = 0)]
    now: u64,
    /// Actually delete. Without it, this is a dry-run preview.
    #[arg(long)]
    execute: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Cli::parse();
    let ep = c.r2_endpoint.as_deref();
    let now = if c.now != 0 {
        c.now
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    };

    // Referenced = output blobs of Done rows (latest-wins view).
    let mut view = LedgerView::new();
    for l in &c.ledger {
        for r in zen_ledger::read_ledger_uri(l, ep)? {
            view.apply(r);
        }
    }
    let referenced: HashSet<Sha256Hex> = view
        .rows()
        .filter(|r| r.status == JobStatus::Done)
        .filter_map(|r| r.output_sha.clone())
        .collect();
    let roots: HashSet<Sha256Hex> = c
        .pin
        .iter()
        .filter_map(|s| Sha256Hex::parse(s.clone()).ok())
        .collect();

    let index = zen_ledger::read_blob_index_uri(&c.blob_index, ep)?;
    let endpoint = ep.unwrap_or("");
    let cfg = GcExecCfg {
        endpoint,
        blobs_base_uri: &c.blobs_r2,
        tombstones_base_uri: c.tombstones_r2.as_deref(),
        cheap_cap_bytes: c.cheap_cap_bytes,
        now,
        execute: c.execute,
    };
    let report = gc_execute(&index, &referenced, &roots, &cfg);

    eprintln!(
        "zen-jobgc: {} ({} index rows, {} referenced)",
        if c.execute { "EXECUTED" } else { "DRY-RUN" },
        index.len(),
        referenced.len()
    );
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.refused.is_empty() {
        eprintln!(
            "zen-jobgc: REFUSED {} unreferenced irreplaceable blob(s) — surfaced, not deleted (pin or archive).",
            report.refused.len()
        );
    }
    Ok(())
}
