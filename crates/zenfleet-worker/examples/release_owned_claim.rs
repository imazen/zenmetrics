#![forbid(unsafe_code)]
//! Release one quiescent, stopped worker's lease through the existing native owner guard.
//! Does not modify the ledger, retry policy, manifests or execution budgets.

use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(long, default_value = "zentrain")]
    bucket: String,
    #[arg(long)]
    run: String,
    /// Bare job SHA or chunk-<SHA>; never an arbitrary object path.
    #[arg(long)]
    key: String,
    /// Must equal ZEN_WORKER, the unique identity of the stopped allocation.
    #[arg(long)]
    expected_worker: String,
    /// Caller has freshly inspected that allocation and confirmed it stopped.
    #[arg(long)]
    confirm_stopped: bool,
    /// All claimers for this run are stopped/paused; release is GET-then-DELETE.
    #[arg(long)]
    all_workers_quiesced: bool,
}

fn validate(args: &Args, worker: &str) -> Result<(), &'static str> {
    if !args.confirm_stopped || !args.all_workers_quiesced {
        return Err("requires a stopped allocation and quiescent run");
    }
    if worker.is_empty()
        || worker.trim() != worker
        || worker.split_whitespace().count() != 1
        || worker != args.expected_worker
    {
        return Err("ZEN_WORKER must match the unique stopped allocation identity");
    }
    if args.run.is_empty()
        || !args
            .run
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err("run must be one namespace token");
    }
    let hash = args.key.strip_prefix("chunk-").unwrap_or(&args.key);
    if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("key must be a bare job SHA or chunk-SHA");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let worker = std::env::var("ZEN_WORKER")?;
    validate(&args, &worker)?;
    let endpoint = std::env::var("ZEN_R2_ENDPOINT").or_else(|_| std::env::var("EP"))?;
    if endpoint.is_empty() {
        return Err("configured store endpoint is empty".into());
    }
    let prefix = format!("jobs/{}/claims", args.run);
    if !zenfleet_worker::release_claim_r2_key(&endpoint, &args.bucket, &prefix, &args.key) {
        return Err("release refused or store operation failed; no ledger row changed".into());
    }
    println!("released {prefix}/{} owned by {worker}", args.key);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> Args {
        Args {
            bucket: "zentrain".into(),
            run: "engineering-run".into(),
            key: format!("chunk-{}", "a".repeat(64)),
            expected_worker: "engineering-worker-unique".into(),
            confirm_stopped: true,
            all_workers_quiesced: true,
        }
    }

    #[test]
    fn refuses_missing_attestations_and_different_identity() {
        let mut a = args();
        assert!(validate(&a, &a.expected_worker).is_ok());
        assert!(validate(&a, "another-worker").is_err());
        assert!(validate(&a, "").is_err());
        a.confirm_stopped = false;
        assert!(validate(&a, &a.expected_worker).is_err());
        a.confirm_stopped = true;
        a.all_workers_quiesced = false;
        assert!(validate(&a, &a.expected_worker).is_err());
    }

    #[test]
    fn refuses_namespace_and_key_injection() {
        let mut a = args();
        for run in ["", "../another-run", "run/claims", "run token"] {
            a.run = run.into();
            assert!(validate(&a, &a.expected_worker).is_err());
        }
        a.run = "engineering-run".into();
        for key in ["", "chunk-short", "../claim", "spec/claim"] {
            a.key = key.into();
            assert!(validate(&a, &a.expected_worker).is_err());
        }
        a.key = "a".repeat(64);
        assert!(validate(&a, &a.expected_worker).is_ok());
    }
}
