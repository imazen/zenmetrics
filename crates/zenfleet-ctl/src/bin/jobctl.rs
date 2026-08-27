#![forbid(unsafe_code)]
//! `zenfleet-ctl` — the agent/human enqueue + discovery CLI.
//!
//!   zenfleet-ctl declare --spec spec.json --out manifest.json
//!   zenfleet-ctl catalog --manifest manifest.json --ledger run/ledger.parquet
//!   zenfleet-ctl gap     --manifest manifest.json --ledger run/ledger.parquet --out gap.json
//!   zenfleet-ctl report  --runlist jobs/_pool/runlist.tsv [--auto-pause]
//!   zenfleet-ctl compact --run <run> [--upload]
//!
//! `report` (migrated 2026-08-27 from scripts/jobsys/pool_reconcile_report.py) prints
//! per-run declared/done/failed-only/gap accounting from the LIVE ledgers — tolerant of
//! in-flight (footerless) sidecars — plus the TOTAL + VERDICT lines the fleet sentinel
//! greps; `--auto-pause` sets control.json paused:true on every gap<=0 run (the 37.8x
//! rescore-tax self-heal). `compact` (migrated from scripts/jobsys/compact_ledgers.py)
//! builds the failure-carrying snapshot — done rows first-wins PLUS the newest failed
//! row per job with no done row (anti-wedge invariant 4) — and with `--upload` writes it
//! to the ONE key both worker modes read: s3://<bucket>/jobs/<run>/ledger_snapshot.parquet
//! (the misplaced-key footgun of 2026-08-27, fixed at the owner). Endpoint resolution:
//! --endpoint, else ZEN_S3_ENDPOINT, else ZEN_R2_ENDPOINT (creds = AWS_* in env, i.e.
//! `set -a; . scripts/lib/s3env.sh`).
//!
//! `gap`'s output feeds straight into `zenfleet-worker --manifest`. Re-running after a sweep yields an
//! empty gap — the no-duplicate-work guarantee, end to end.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use zenfleet_core::{DesiredJob, LedgerView, RetryPolicy};

fn shellexpand_home(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Ok(h) = std::env::var("HOME") {
            return format!("{h}/{rest}");
        }
    }
    p.to_string()
}
use zenfleet_ctl::{
    DeclareSpec, coverage, declare, declare_diffmaps, declare_encodes, gap, parse_emit_cells,
};

#[derive(Parser)]
#[command(
    name = "zenfleet-ctl",
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
    /// Declare encode jobs from a plan's `--emit-cells` JSONL (the plan-cell path; goal A).
    /// Each line is an EncodeDeclareItem (image_path/codec/q/knob_tuple_json{cell,fp,plan}/source_sha).
    DeclareEncodes {
        #[arg(long)]
        cells: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Declare per-pixel DIFFMAP jobs from the same spec.json shape as `declare`
    /// (items x metrics; metrics must be map owners: butteraugli, cvvdp). The
    /// HDR-corpus B2 wave. `--hdr` computes maps at the HDR display peak.
    DeclareDiffmaps {
        #[arg(long)]
        spec: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        hdr: bool,
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
    /// Print `idx<TAB>job_id<TAB>image_path` for every manifest cell, in
    /// manifest order — the canonical way to join a manifest to ledger rows
    /// (provenance/attribution tooling; e.g. the bf944 tier-matched declare).
    Ids {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Per-run reconcile accounting over LIVE s3 ledgers: declared vs distinct
    /// done/failed-only/gap, tolerant per-file reads, TOTAL + VERDICT lines.
    Report {
        /// Runlist TSV (first column = run names); local path or s3:// URI.
        #[arg(long)]
        runlist: Option<String>,
        /// Explicit run name(s) instead of a runlist (repeatable).
        #[arg(long = "run")]
        runs: Vec<String>,
        #[arg(long, default_value = "zentrain")]
        bucket: String,
        /// S3 endpoint; default ZEN_S3_ENDPOINT, else ZEN_R2_ENDPOINT.
        #[arg(long)]
        endpoint: Option<String>,
        /// Pause (control.json paused:true) every run whose gap<=0.
        #[arg(long)]
        auto_pause: bool,
    },
    /// Build a run's failure-carrying ledger snapshot (done first-wins + newest
    /// failed per no-done job — invariant 4); `--upload` writes it to the run-root
    /// key both worker modes read.
    Compact {
        #[arg(long)]
        run: String,
        #[arg(long, default_value = "zentrain")]
        bucket: String,
        #[arg(long)]
        endpoint: Option<String>,
        /// Local output dir (writes snap_<run>.parquet there).
        #[arg(long, default_value = "~/tmp/zen-snaps")]
        out_dir: String,
        /// Also upload to s3://<bucket>/jobs/<run>/ledger_snapshot.parquet.
        #[arg(long)]
        upload: bool,
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
        for r in zenfleet_ledger::read_ledger_uri(uri.as_ref(), endpoint)? {
            v.apply(r);
        }
    }
    Ok(v)
}

fn read_manifest(p: &PathBuf) -> Result<Vec<DesiredJob>, Box<dyn std::error::Error>> {
    Ok(serde_json::from_slice(&std::fs::read(p)?)?)
}

fn resolve_endpoint(cli: Option<String>) -> Result<String, Box<dyn std::error::Error>> {
    cli.or_else(|| std::env::var("ZEN_S3_ENDPOINT").ok())
        .or_else(|| std::env::var("ZEN_R2_ENDPOINT").ok())
        .ok_or_else(|| "no endpoint: pass --endpoint or set ZEN_S3_ENDPOINT (source scripts/lib/s3env.sh)".into())
}

/// Declared cell count from jobs/<run>/manifest.json.gz (else plain .json).
/// Gunzip via the `gunzip` binary (a baked fleet tool; same contract as the
/// worker entrypoint) so no decompression codepath is duplicated here.
fn declared_count(ep: &str, bucket: &str, run: &str) -> Result<usize, String> {
    use std::io::Write as _;
    let gz = zenfleet_ledger::read_bytes_uri(
        &format!("s3://{bucket}/jobs/{run}/manifest.json.gz"),
        Some(ep),
    );
    let plain: Vec<u8> = match gz {
        Ok(bytes) if bytes.starts_with(&[0x1f, 0x8b]) => {
            let mut c = std::process::Command::new("gunzip")
                .arg("-c")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()
                .map_err(|e| format!("spawn gunzip: {e}"))?;
            // Writer THREAD: gunzip fills its stdout pipe while we are still
            // feeding stdin (5 MB in, ~97 MB out) — a same-thread write_all
            // then wait_with_output deadlocks on the full pipe (measured:
            // both parity runs hung here). Write from a thread, drop to close.
            let mut stdin = c.stdin.take().unwrap();
            let writer = std::thread::spawn(move || {
                let _ = stdin.write_all(&bytes);
            });
            let out = c.wait_with_output().map_err(|e| e.to_string())?;
            let _ = writer.join();
            if !out.status.success() {
                return Err("gunzip failed on manifest.json.gz".into());
            }
            out.stdout
        }
        Ok(bytes) => bytes, // transport already inflated it (the Python's fallback)
        Err(_) => zenfleet_ledger::read_bytes_uri(
            &format!("s3://{bucket}/jobs/{run}/manifest.json"),
            Some(ep),
        )
        .map_err(|e| format!("manifest: {e:?}"))?,
    };
    let v: serde_json::Value =
        serde_json::from_slice(&plain).map_err(|e| format!("manifest parse: {e}"))?;
    v.as_array().map(|a| a.len()).ok_or_else(|| "manifest is not an array".into())
}

/// Tolerant read of every sidecar under jobs/<run>/ledger/ — an in-flight
/// (footerless) chunk is skipped and counted, never fatal (the 2026-08-26
/// SeaweedFS lesson the Python carried).
fn read_run_ledgers(ep: &str, bucket: &str, run: &str) -> (Vec<zenfleet_core::LedgerRow>, usize) {
    // ONE bulk s5cmd wildcard download per run (a per-file spawn costs a process +
    // round-trip each; at ~130 sidecars x N runs that timed out the first parity
    // run), then tolerant LOCAL reads — an in-flight/footerless chunk is skipped
    // and counted, never fatal (the 2026-08-26 SeaweedFS lesson).
    let tmp = std::env::temp_dir().join(format!(
        "jobctl_ledgers_{}_{}",
        std::process::id(),
        run.replace('/', "_")
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    if std::fs::create_dir_all(&tmp).is_err() {
        return (Vec::new(), 0);
    }
    let status = std::process::Command::new("s5cmd")
        .args([
            "--endpoint-url",
            ep,
            "cp",
            &format!("s3://{bucket}/jobs/{run}/ledger/*"),
            &format!("{}/", tmp.display()),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
    let mut rows = Vec::new();
    let mut skipped = 0usize;
    if status.map(|s| s.success()).unwrap_or(false) {
        if let Ok(rd) = std::fs::read_dir(&tmp) {
            for e in rd.flatten() {
                match zenfleet_ledger::read_ledger(&e.path()) {
                    Ok(mut r) => rows.append(&mut r),
                    Err(_) => skipped += 1,
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    (rows, skipped)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().cmd {
        Cmd::Declare { spec, out } => {
            let s: DeclareSpec = serde_json::from_slice(&std::fs::read(&spec)?)?;
            let jobs = declare(&s)?;
            std::fs::write(&out, serde_json::to_vec_pretty(&jobs)?)?;
            eprintln!("declared {} jobs -> {}", jobs.len(), out.display());
        }
        Cmd::DeclareDiffmaps { spec, out, hdr } => {
            let s: DeclareSpec = serde_json::from_slice(&std::fs::read(&spec)?)?;
            let jobs = declare_diffmaps(&s, hdr)?;
            std::fs::write(&out, serde_json::to_vec_pretty(&jobs)?)?;
            eprintln!(
                "declared {} diffmap jobs (hdr={hdr}) -> {}",
                jobs.len(),
                out.display()
            );
        }
        Cmd::DeclareEncodes { cells, out } => {
            let items = parse_emit_cells(&std::fs::read_to_string(&cells)?)?;
            let jobs = declare_encodes(&items)?;
            std::fs::write(&out, serde_json::to_vec_pretty(&jobs)?)?;
            eprintln!(
                "declared {} encode jobs from {} cells -> {}",
                jobs.len(),
                items.len(),
                out.display()
            );
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
        Cmd::Ids { manifest } => {
            let jobs = read_manifest(&manifest)?;
            let mut out = String::with_capacity(jobs.len() * 100);
            for (i, j) in jobs.iter().enumerate() {
                use std::fmt::Write as _;
                writeln!(out, "{i}\t{}\t{}", j.job_id().as_str(), j.cell.image_path)?;
            }
            print!("{out}");
            eprintln!("{} ids", jobs.len());
        }
        Cmd::Report {
            runlist,
            runs,
            bucket,
            endpoint,
            auto_pause,
        } => {
            let ep = resolve_endpoint(endpoint)?;
            let t0 = std::time::Instant::now();
            let mut run_names: Vec<String> = runs;
            if let Some(rl) = runlist {
                let bytes = if rl.starts_with("s3://") {
                    zenfleet_ledger::read_bytes_uri(&rl, Some(&ep)).map_err(|e| format!("runlist: {e:?}"))?
                } else if rl.starts_with("jobs/") {
                    zenfleet_ledger::read_bytes_uri(&format!("s3://{bucket}/{rl}"), Some(&ep))
                        .map_err(|e| format!("runlist: {e:?}"))?
                } else {
                    std::fs::read(&rl)?
                };
                run_names.extend(
                    String::from_utf8_lossy(&bytes)
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .map(|l| l.split('\t').next().unwrap_or(l).to_string()),
                );
            }
            if run_names.is_empty() {
                return Err("report: no runs (pass --runlist or --run)".into());
            }
            let (mut td, mut tdn, mut tf, mut tg, mut tr) = (0usize, 0usize, 0usize, 0i64, 0usize);
            let mut bad: Vec<String> = Vec::new();
            let mut complete: Vec<(String, usize, usize)> = Vec::new();
            run_names.sort();
            // Fan out over runs (8 wide, matching the Python's ThreadPoolExecutor);
            // outputs are collected per run and printed in order, never interleaved.
            type RunOut = (String, Option<zenfleet_ctl::RunAccounting>);
            let mut results: Vec<Option<RunOut>> = (0..run_names.len()).map(|_| None).collect();
            let next = std::sync::atomic::AtomicUsize::new(0);
            let res_mx = std::sync::Mutex::new(&mut results);
            std::thread::scope(|sc| {
                for _ in 0..8.min(run_names.len()) {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= run_names.len() {
                            break;
                        }
                        let run = &run_names[i];
                        let out: RunOut = match declared_count(&ep, &bucket, run) {
                            Err(e) => (format!("{run}: ERROR {e}"), None),
                            Ok(declared) => {
                                let (rows, skipped) = read_run_ledgers(&ep, &bucket, run);
                                let acc = zenfleet_ctl::account_rows(declared, &rows);
                                let mut text = String::new();
                                if skipped > 0 {
                                    text.push_str(&format!(
                                        "  [{run}] WARNING: skipped {skipped} unreadable/in-flight ledger chunk(s)\n"
                                    ));
                                }
                                let flag = if acc.gap == 0 {
                                    String::new()
                                } else {
                                    format!("  <-- GAP {}", acc.gap)
                                };
                                text.push_str(&format!(
                                    "{run}: declared={} done={} failed-only={} raw_rows={}{flag}",
                                    acc.declared, acc.distinct_done, acc.failed_only, acc.raw_rows
                                ));
                                (text, Some(acc))
                            }
                        };
                        res_mx.lock().unwrap()[i] = Some(out);
                    });
                }
            });
            for (i, r) in results.into_iter().enumerate() {
                let (text, acc) = r.expect("every run produced a result");
                println!("{text}");
                match acc {
                    None => bad.push(run_names[i].clone()),
                    Some(acc) => {
                        td += acc.declared;
                        tdn += acc.distinct_done;
                        tf += acc.failed_only;
                        tg += acc.gap.max(0);
                        tr += acc.raw_rows;
                        if acc.gap <= 0 {
                            complete.push((run_names[i].clone(), acc.raw_rows, acc.distinct_done));
                        }
                    }
                }
            }
            println!(
                "\nTOTAL declared={td} distinct_done={tdn} failed-only={tf} gap={tg} raw_ledger_rows={tr} (rescore tax {:.2}x) errors={} ({:.0}s)",
                tr as f64 / tdn.max(1) as f64,
                bad.len(),
                t0.elapsed().as_secs_f64()
            );
            println!(
                "VERDICT: {}",
                if tg == 0 && bad.is_empty() { "COMPLETE — every run gap==0" } else { "NOT COMPLETE" }
            );
            if auto_pause && !complete.is_empty() {
                println!(
                    "\n--auto-pause: {} run(s) at gap<=0 — pausing so no worker re-scores them",
                    complete.len()
                );
                for (run, raw, done) in &complete {
                    let ckey = format!("s3://{bucket}/jobs/{run}/control.json");
                    let mut cur: serde_json::Value = zenfleet_ledger::read_bytes_uri(&ckey, Some(&ep))
                        .ok()
                        .and_then(|b| serde_json::from_slice(&b).ok())
                        .unwrap_or_else(|| serde_json::json!({}));
                    if cur.get("paused").and_then(|p| p.as_bool()) == Some(true) {
                        println!("  {run}: already paused (rescore tax {:.1}x)", *raw as f64 / (*done).max(1) as f64);
                        continue;
                    }
                    cur["paused"] = serde_json::Value::Bool(true);
                    match zenfleet_ledger::write_bytes_uri(&ckey, cur.to_string().as_bytes(), Some(&ep)) {
                        Ok(()) => println!(
                            "  AUTO-PAUSED {run}: control.json paused:true (was re-scoring {done} done cells at {:.1}x tax)",
                            *raw as f64 / (*done).max(1) as f64
                        ),
                        Err(e) => println!("  {run}: FAILED to pause ({e:?})"),
                    }
                }
            } else if auto_pause {
                println!("\n--auto-pause: no run at gap<=0; nothing to pause");
            }
        }
        Cmd::Compact {
            run,
            bucket,
            endpoint,
            out_dir,
            upload,
        } => {
            let ep = resolve_endpoint(endpoint)?;
            let t0 = std::time::Instant::now();
            let (rows, skipped) = read_run_ledgers(&ep, &bucket, &run);
            if skipped > 0 {
                println!("  [{run}] WARNING: skipped {skipped} unreadable/in-flight ledger chunk(s)");
            }
            let raw = rows.len();
            let (snap, n_done, n_failed) = zenfleet_ctl::snapshot_rows(rows);
            println!("  snapshot: {n_done} done + {n_failed} newest-failed rows");
            let dir = shellexpand_home(&out_dir);
            std::fs::create_dir_all(&dir)?;
            let out = format!("{dir}/snap_{run}.parquet");
            zenfleet_ledger::write_ledger(std::path::Path::new(&out), &snap)?;
            println!(
                "run={run} read_rows={raw} distinct_done={n_done} wrote={out} ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
            if upload {
                let key = format!("s3://{bucket}/jobs/{run}/ledger_snapshot.parquet");
                zenfleet_ledger::write_ledger_uri(&key, &snap, Some(&ep))?;
                println!("uploaded {key}");
            }
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
