#![forbid(unsafe_code)]
//! Runnable worker: one `zenfleet-worker` pass. Loads a job manifest + existing ledger, reconciles the
//! gap, executes each job via `--exec` (stdin-JSON → stdout-bytes), content-addresses outputs, and
//! writes the new ledger rows. Loop it (cron / `/loop` / a shell `while`) for a long queue.
//!
//! Example:
//!   zenfleet-worker --manifest jobs.json --ledger-in run/ledger.parquet \
//!     --ledger-out run/pass.parquet --blobs run/blobs --exec ./my-encoder

use std::path::PathBuf;

use clap::Parser;
use zenfleet_worker::{WorkerConfig, run};

#[derive(Parser)]
#[command(
    name = "zenfleet-worker",
    about = "Execute the reconciler's gap: run jobs, content-address outputs, write ledger rows"
)]
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
    /// Content-addressed blob directory (used when --blobs-r2-bucket is not given).
    #[arg(long, default_value = "./blobs")]
    blobs: PathBuf,
    /// Write blobs to R2 instead: the S3 bucket name (requires --r2-endpoint).
    #[arg(long = "blobs-r2-bucket")]
    blobs_r2_bucket: Option<String>,
    /// Key prefix under the R2 bucket.
    #[arg(long = "blobs-r2-prefix", default_value = "blobs")]
    blobs_r2_prefix: String,
    /// R2 S3 endpoint, e.g. https://<account>.r2.cloudflarestorage.com (requires AWS_* creds in env).
    #[arg(long = "r2-endpoint")]
    r2_endpoint: Option<String>,
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
    /// Claim each gap job in R2 before executing (concurrent-safe fleet). Requires --blobs-r2-bucket.
    #[arg(long = "claims-r2-bucket")]
    claims_r2_bucket: Option<String>,
    #[arg(long = "claims-prefix", default_value = "claims")]
    claims_prefix: String,
    /// Claim is stealable once this old (presumed-dead worker) — dead-worker reclaim.
    #[arg(long = "claim-ttl-secs", default_value_t = 600)]
    claim_ttl_secs: u64,
    /// Speculative-execution threshold (goal E): co-run a *live* straggler whose claim is older than
    /// this (but younger than the TTL). Bounds the long tail; off by default.
    #[arg(long = "spec-threshold-secs")]
    spec_threshold_secs: Option<u64>,
    /// R2 key of a RunControl object ({"paused":bool,"drain":bool}); when paused/draining this pass
    /// claims no new work (goal C). Requires --blobs-r2-bucket + --r2-endpoint.
    #[arg(long = "control-r2-key")]
    control_r2_key: Option<String>,
    /// Resource class(es) this worker's hardware serves (goal H capability routing), repeatable:
    /// cpu_light / cpu_heavy / cpu_arm / gpu / high_ram. Omit = serve everything. A job runs only if
    /// its kind's class is served (e.g. a gpu box pulls only metric/diffmap jobs).
    #[arg(long = "capability")]
    capability: Vec<String>,
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
    let r2 = match (c.blobs_r2_bucket, c.r2_endpoint) {
        (Some(bucket), Some(endpoint)) => Some(zenfleet_worker::R2Target {
            endpoint,
            bucket,
            prefix: c.blobs_r2_prefix,
        }),
        (None, None) => None,
        _ => return Err("--blobs-r2-bucket and --r2-endpoint must be given together".into()),
    };
    let claims = c.claims_r2_bucket.map(|bucket| zenfleet_worker::ClaimCfg {
        bucket,
        prefix: c.claims_prefix,
        ttl_secs: c.claim_ttl_secs,
        spec_threshold_secs: c.spec_threshold_secs,
    });
    if claims.is_some() && r2.is_none() {
        return Err("--claims-r2-bucket requires --blobs-r2-bucket + --r2-endpoint".into());
    }
    if c.control_r2_key.is_some() && r2.is_none() {
        return Err("--control-r2-key requires --blobs-r2-bucket + --r2-endpoint".into());
    }
    let mut served = Vec::new();
    for cap in &c.capability {
        match zenfleet_core::ResourceClass::parse(cap) {
            Some(rc) => served.push(rc),
            None => return Err(format!("--capability '{cap}' is not a resource class (cpu_light/cpu_heavy/cpu_arm/gpu/high_ram)").into()),
        }
    }
    let cfg = WorkerConfig {
        manifest: c.manifest,
        ledger_in: c.ledger_in,
        ledger_out: c.ledger_out,
        blobs: c.blobs,
        r2,
        claims,
        control_key: c.control_r2_key,
        exec: c.exec,
        worker: c.worker,
        provider: c.provider,
        now,
        max_attempts: c.max_attempts,
        served,
    };
    let out = run(&cfg)?;
    eprintln!(
        "zenfleet-worker: done={} failed={} poisoned={} skipped={} rows={}",
        out.done,
        out.failed,
        out.poisoned,
        out.skipped,
        out.rows.len()
    );
    Ok(())
}
