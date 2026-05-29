#![forbid(unsafe_code)]
//! Runnable worker: one `zen-jobworker` pass. Loads a job manifest + existing ledger, reconciles the
//! gap, executes each job via `--exec` (stdin-JSON → stdout-bytes), content-addresses outputs, and
//! writes the new ledger rows. Loop it (cron / `/loop` / a shell `while`) for a long queue.
//!
//! Example:
//!   zen-jobworker --manifest jobs.json --ledger-in run/ledger.parquet \
//!     --ledger-out run/pass.parquet --blobs run/blobs --exec ./my-encoder

use std::path::PathBuf;

use clap::Parser;
use zen_jobworker::{run, WorkerConfig};

#[derive(Parser)]
#[command(name = "zen-jobworker", about = "Execute the reconciler's gap: run jobs, content-address outputs, write ledger rows")]
struct Cli {
    /// JSON file: array of DesiredJob (the desired artifacts).
    #[arg(long)]
    manifest: PathBuf,
    /// Existing ledger sidecar(s) to fold into the latest-wins view (repeatable).
    #[arg(long = "ledger-in")]
    ledger_in: Vec<PathBuf>,
    /// Output ledger sidecar for this pass's rows.
    #[arg(long = "ledger-out")]
    ledger_out: PathBuf,
    /// Content-addressed blob directory.
    #[arg(long)]
    blobs: PathBuf,
    /// Executor program (reads a job JSON on stdin, emits output bytes on stdout, exit 0 = success).
    #[arg(long)]
    exec: String,
    #[arg(long, default_value = "local-1")]
    worker: String,
    #[arg(long, default_value = "local")]
    provider: String,
    /// Override the timestamp (unix secs). 0 = use the system clock.
    #[arg(long, default_value_t = 0)]
    now: u64,
    #[arg(long = "max-attempts", default_value_t = 3)]
    max_attempts: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let c = Cli::parse();
    let now = if c.now != 0 {
        c.now
    } else {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    };
    let cfg = WorkerConfig {
        manifest: c.manifest,
        ledger_in: c.ledger_in,
        ledger_out: c.ledger_out,
        blobs: c.blobs,
        exec: c.exec,
        worker: c.worker,
        provider: c.provider,
        now,
        max_attempts: c.max_attempts,
    };
    let out = run(&cfg)?;
    eprintln!(
        "zen-jobworker: done={} failed={} poisoned={} rows={}",
        out.done, out.failed, out.poisoned, out.rows.len()
    );
    Ok(())
}
