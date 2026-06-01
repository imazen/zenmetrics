#![forbid(unsafe_code)]
//! `zen-jobctl` — the agent/human enqueue + discovery CLI.
//!
//!   zen-jobctl declare --spec spec.json --out manifest.json
//!   zen-jobctl catalog --manifest manifest.json --ledger run/ledger.parquet
//!   zen-jobctl gap     --manifest manifest.json --ledger run/ledger.parquet --out gap.json
//!
//! `gap`'s output feeds straight into `zen-jobworker --manifest`. Re-running after a sweep yields an
//! empty gap — the no-duplicate-work guarantee, end to end.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use zen_job_core::{DesiredJob, LedgerView, RetryPolicy};
use zen_jobctl::{DeclareSpec, coverage, declare, gap};

#[derive(Parser)]
#[command(
    name = "zen-jobctl",
    about = "Declare desired jobs and query coverage/gap from the ledger"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Expand a spec.json into a DesiredJob manifest (goal A: declare).
    Declare {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Print coverage (done/poison/gap per codec×metric) for a manifest vs the ledger (goal I).
    Catalog {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long = "ledger")]
        ledger: Vec<PathBuf>,
        /// R2 endpoint, if any --ledger path is an s3:// URI (needs AWS_* creds in env).
        #[arg(long = "r2-endpoint")]
        r2_endpoint: Option<String>,
    },
    /// Write the not-yet-done subset (the gap) of a manifest, given the ledger.
    Gap {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long = "ledger")]
        ledger: Vec<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        /// R2 endpoint, if any --ledger path is an s3:// URI.
        #[arg(long = "r2-endpoint")]
        r2_endpoint: Option<String>,
    },
}

fn load_view(
    paths: &[PathBuf],
    endpoint: Option<&str>,
) -> Result<LedgerView, Box<dyn std::error::Error>> {
    let mut v = LedgerView::new();
    for p in paths {
        let uri = p.to_string_lossy();
        for r in zen_ledger::read_ledger_uri(uri.as_ref(), endpoint)? {
            v.apply(r);
        }
    }
    Ok(v)
}

fn read_manifest(p: &PathBuf) -> Result<Vec<DesiredJob>, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Declare { spec, out } => {
            let s: DeclareSpec = serde_json::from_slice(&std::fs::read(&spec)?)?;
            let jobs = declare(&s)?;
            std::fs::write(&out, serde_json::to_vec_pretty(&jobs)?)?;
            eprintln!("declared {} jobs -> {}", jobs.len(), out.display());
        }
        Cmd::Catalog {
            manifest,
            ledger,
            r2_endpoint,
        } => {
            let jobs = read_manifest(&manifest)?;
            let view = load_view(&ledger, r2_endpoint.as_deref())?;
            println!("{}", serde_json::to_string_pretty(&coverage(&jobs, &view))?);
        }
        Cmd::Gap {
            manifest,
            ledger,
            out,
            r2_endpoint,
        } => {
            let jobs = read_manifest(&manifest)?;
            let view = load_view(&ledger, r2_endpoint.as_deref())?;
            let g = gap(&jobs, &view, RetryPolicy::default());
            std::fs::write(&out, serde_json::to_vec_pretty(&g)?)?;
            eprintln!(
                "gap: {} of {} jobs remain -> {}",
                g.len(),
                jobs.len(),
                out.display()
            );
        }
    }
    Ok(())
}
