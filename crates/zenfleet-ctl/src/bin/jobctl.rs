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
    /// Fast pool-progress readout from per-run ledger_snapshot.parquet FOOTERS
    /// (num_rows only — seconds, not a live-ledger scan). Reflects the last
    /// snapshot refresh; runs with no snapshot count 0.
    Progress {
        /// Runlist TSV(s) (first column = run names); s3:// URI, bucket-relative
        /// jobs/... key, or local path. Repeatable.
        #[arg(long = "runlist", required = true)]
        runlists: Vec<String>,
        /// Denominator for the percentage line.
        #[arg(long, default_value_t = 490173)]
        total: usize,
        /// Keep only run names with this prefix (the bf-pool convention).
        #[arg(long)]
        filter_prefix: Option<String>,
        #[arg(long, default_value = "zentrain")]
        bucket: String,
        #[arg(long)]
        endpoint: Option<String>,
    },
    /// The Encode→ScoreFile bridge: reduce an encode run's ledger latest-wins
    /// (LedgerView, the owner semantics), keep DONE rows, emit
    /// <out>.parquet + <out>.tsv pairs (ref/dist URIs + full cell identity).
    Pairs {
        /// Ledger dir: s3:// prefix or local directory of sidecar parquets.
        #[arg(long)]
        ledger: String,
        /// Prefix for ref URIs (joined with each row's image_path).
        #[arg(long)]
        refs_prefix: String,
        /// Prefix for blob URIs (joined with each row's output_sha).
        #[arg(long)]
        blobs_prefix: String,
        /// Output basename (writes <out>.parquet and <out>.tsv).
        #[arg(long)]
        out: String,
        #[arg(long)]
        endpoint: Option<String>,
    },
    /// Declare ScoreFile (CHUNKed per ref) or Diffmap (per variant x metric)
    /// jobs from pairs parquet(s) — direct-object inputs the worker resolves
    /// via ZEN_ENCODES_PREFIX. Emits manifest.json(.gz) + control.json to the
    /// run (or --manifest-out locally for multi-part declares). Unlike the
    /// Python this replaced, jobs carry invariant-5 `requires` tokens.
    DeclareScorefiles {
        /// pairs parquet path(s). Repeatable.
        #[arg(long = "pairs", required = true)]
        pairs: Vec<String>,
        #[arg(long)]
        run: Option<String>,
        #[arg(long, default_value = "zentrain")]
        bucket: String,
        #[arg(long)]
        endpoint: Option<String>,
        /// Comma-separated metric list.
        #[arg(long, default_value = "zensim-gpu")]
        metrics: String,
        #[arg(long, default_value_t = 12)]
        chunk: usize,
        /// Cell codec label when the pairs carry no identity columns.
        #[arg(long, default_value = "zenjpeg")]
        cell_codec: String,
        /// Group by full ref URI + use dist_path URIs (else basenames of
        /// image_path/dist_member).
        #[arg(long)]
        full_uri: bool,
        #[arg(long)]
        hdr: bool,
        #[arg(long)]
        hdr_transfer: Option<String>,
        /// "score_file" (default) or "diffmap".
        #[arg(long, default_value = "score_file")]
        kind: String,
        /// Flat per-job hint: peak_mem GB / threads.
        #[arg(long)]
        hint_mem_gb: Option<f64>,
        #[arg(long)]
        hint_threads: Option<u32>,
        /// Pixel-derived hint (scaleWxH token in the ref name): MiB per MP + base MiB.
        #[arg(long, default_value_t = 0.0)]
        hint_mib_per_mp: f64,
        #[arg(long, default_value_t = 0.0)]
        hint_base_mib: f64,
        /// Write the manifest locally and SKIP uploads (multi-part declares).
        #[arg(long)]
        manifest_out: Option<PathBuf>,
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
    // Precedence honors the canonical resolver (scripts/lib/s3env.sh): its CONTRACT
    // output is `EP` — the endpoint every s5cmd/aws call uses for the SELECTED store.
    // Preferring the ambient ZEN_S3_ENDPOINT here broke ZEN_STORE=r2 (LAN endpoint
    // paired with R2 creds -> 403 InvalidAccessKeyId, measured 2026-08-27): that var
    // is the LAN env-file's raw input, not the resolver's decision.
    cli.or_else(|| std::env::var("EP").ok())
        .or_else(|| std::env::var("ZEN_S3_ENDPOINT").ok())
        .or_else(|| std::env::var("ZEN_R2_ENDPOINT").ok())
        .ok_or_else(|| "no endpoint: pass --endpoint or source scripts/lib/s3env.sh (exports EP)".into())
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
    read_ledger_prefix(ep, &format!("s3://{bucket}/jobs/{run}/ledger/"))
}

/// Bulk-download + tolerantly read every parquet under an s3 ledger prefix
/// (or read a local dir directly).
fn read_ledger_prefix(ep: &str, prefix: &str) -> (Vec<zenfleet_core::LedgerRow>, usize) {
    // ONE bulk s5cmd wildcard download per run (a per-file spawn costs a process +
    // round-trip each; at ~130 sidecars x N runs that timed out the first parity
    // run), then tolerant LOCAL reads — an in-flight/footerless chunk is skipped
    // and counted, never fatal (the 2026-08-26 SeaweedFS lesson).
    if !prefix.starts_with("s3://") {
        let mut rows = Vec::new();
        let mut skipped = 0usize;
        if let Ok(rd) = std::fs::read_dir(prefix) {
            for e in rd.flatten() {
                if e.path().extension().is_some_and(|x| x == "parquet") {
                    match zenfleet_ledger::read_ledger(&e.path()) {
                        Ok(mut r) => rows.append(&mut r),
                        Err(_) => skipped += 1,
                    }
                }
            }
        }
        return (rows, skipped);
    }
    let tmp = std::env::temp_dir().join(format!(
        "jobctl_ledgers_{}_{}",
        std::process::id(),
        prefix.replace(['/', ':'], "_")
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
            &format!("{}*", prefix),
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
        Cmd::Progress {
            runlists,
            total,
            filter_prefix,
            bucket,
            endpoint,
        } => {
            let ep = resolve_endpoint(endpoint)?;
            let mut runs: Vec<String> = Vec::new();
            for rl in &runlists {
                let bytes = if rl.starts_with("s3://") {
                    zenfleet_ledger::read_bytes_uri(rl, Some(&ep)).map_err(|e| format!("runlist {rl}: {e:?}"))
                } else if rl.starts_with("jobs/") {
                    zenfleet_ledger::read_bytes_uri(&format!("s3://{bucket}/{rl}"), Some(&ep))
                        .map_err(|e| format!("runlist {rl}: {e:?}"))
                } else {
                    std::fs::read(rl).map_err(|e| format!("runlist {rl}: {e}"))
                };
                match bytes {
                    Ok(b) => runs.extend(
                        String::from_utf8_lossy(&b)
                            .lines()
                            .filter(|l| !l.trim().is_empty())
                            .map(|l| l.split('\t').next().unwrap_or(l).to_string())
                            .filter(|r| filter_prefix.as_deref().is_none_or(|p| r.starts_with(p))),
                    ),
                    Err(e) => println!("WARN: runlist {rl} unreadable: {e}"),
                }
            }
            let next = std::sync::atomic::AtomicUsize::new(0);
            let counts: Vec<std::sync::Mutex<Option<usize>>> =
                (0..runs.len()).map(|_| std::sync::Mutex::new(None)).collect();
            std::thread::scope(|sc| {
                for _ in 0..16.min(runs.len().max(1)) {
                    sc.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= runs.len() {
                            break;
                        }
                        let tmp = std::env::temp_dir().join(format!(
                            "jobctl_snap_{}_{i}.parquet",
                            std::process::id()
                        ));
                        let uri = format!("s3://{bucket}/jobs/{}/ledger_snapshot.parquet", runs[i]);
                        let n = if std::process::Command::new("s5cmd")
                            .args(["--endpoint-url", &ep, "cp", &uri, &tmp.to_string_lossy()])
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .status()
                            .map(|st| st.success())
                            .unwrap_or(false)
                        {
                            zenfleet_ledger::parquet_num_rows(&tmp).unwrap_or(0)
                        } else {
                            0
                        };
                        let _ = std::fs::remove_file(&tmp);
                        *counts[i].lock().unwrap() = Some(n);
                    });
                }
            });
            let res: Vec<usize> = counts.iter().map(|m| m.lock().unwrap().unwrap_or(0)).collect();
            let sum: usize = res.iter().sum();
            let missing: Vec<&String> =
                runs.iter().zip(&res).filter(|(_, n)| **n == 0).map(|(r, _)| r).collect();
            println!(
                "distinct_done={sum} / {total} = {:.2}%  (from {}/{} snapshots)",
                100.0 * sum as f64 / total.max(1) as f64,
                runs.len() - missing.len(),
                runs.len()
            );
            if !missing.is_empty() {
                println!(
                    "no-snapshot runs (~0 done): {}",
                    missing.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
                );
            }
        }
        Cmd::Pairs {
            ledger,
            refs_prefix,
            blobs_prefix,
            out,
            endpoint,
        } => {
            let ep = resolve_endpoint(endpoint)?;
            let refs_prefix = refs_prefix.trim_end_matches('/');
            let blobs_prefix = blobs_prefix.trim_end_matches('/');
            let (rows, skipped) = read_ledger_prefix(&ep, &ledger);
            if skipped > 0 {
                println!("WARNING: skipped {skipped} unreadable ledger chunk(s)");
            }
            let nfiles = rows.len();
            // Latest-wins via the OWNER (LedgerView) — the Python approximated this
            // with a ts>= scan; on exact-ts ties LedgerView's status-rank tie-break
            // is the canonical answer.
            let mut view = LedgerView::new();
            for r in rows {
                view.apply(r);
            }
            let done: Vec<&zenfleet_core::LedgerRow> = view
                .rows()
                .filter(|r| r.status == zenfleet_core::JobStatus::Done)
                .collect();
            let skipped_jobs = view.rows().count() - done.len();
            use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
            use std::sync::Arc;
            let s_col = |f: &dyn Fn(&zenfleet_core::LedgerRow) -> String| -> ArrayRef {
                Arc::new(StringArray::from_iter_values(done.iter().map(|r| f(r))))
            };
            let sha = |r: &zenfleet_core::LedgerRow| {
                r.output_sha.as_ref().map(|s| s.as_str().to_string()).unwrap_or_default()
            };
            let batch = RecordBatch::try_from_iter(vec![
                ("ref_path", s_col(&|r| format!("{refs_prefix}/{}", r.cell.image_path))),
                ("dist_path", s_col(&|r| format!("{blobs_prefix}/{}", sha(r)))),
                ("image_path", s_col(&|r| r.cell.image_path.clone())),
                ("codec", s_col(&|r| r.cell.codec.clone())),
                (
                    "q",
                    Arc::new(Int64Array::from_iter_values(done.iter().map(|r| r.cell.q))) as ArrayRef,
                ),
                ("knob_tuple_json", s_col(&|r| r.cell.knob_tuple_json.clone())),
                ("encode_sha", s_col(&|r| sha(r))),
                ("worker", s_col(&|r| r.worker.clone())),
                ("provider", s_col(&|r| r.provider.clone())),
            ])?;
            use parquet::arrow::ArrowWriter;
            use parquet::basic::{Compression, ZstdLevel};
            use parquet::file::properties::WriterProperties;
            let f = std::fs::File::create(format!("{out}.parquet"))?;
            let props = WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
                .build();
            let mut w = ArrowWriter::try_new(f, batch.schema(), Some(props))?;
            w.write(&batch)?;
            w.close()?;
            let mut tsv = String::from(
                "ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\tencode_sha\tworker\tprovider\n",
            );
            for r in &done {
                use std::fmt::Write as _;
                let _ = writeln!(
                    tsv,
                    "{refs_prefix}/{}\t{blobs_prefix}/{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    r.cell.image_path,
                    sha(r),
                    r.cell.image_path,
                    r.cell.codec,
                    r.cell.q,
                    r.cell.knob_tuple_json,
                    sha(r),
                    r.worker,
                    r.provider
                );
            }
            std::fs::write(format!("{out}.tsv"), tsv)?;
            println!(
                "pairs: {} DONE cells ({skipped_jobs} non-done job_ids skipped) from {nfiles} ledger rows -> {out}.parquet / .tsv",
                done.len()
            );
        }
        Cmd::DeclareScorefiles {
            pairs,
            run,
            bucket,
            endpoint,
            metrics,
            chunk,
            cell_codec,
            full_uri,
            hdr,
            hdr_transfer,
            kind,
            hint_mem_gb,
            hint_threads,
            hint_mib_per_mp,
            hint_base_mib,
            manifest_out,
        } => {
            let metrics: Vec<String> = metrics.split(',').filter(|m| !m.is_empty()).map(str::to_string).collect();
            let diffmap = match kind.as_str() {
                "score_file" => false,
                "diffmap" => true,
                other => return Err(format!("--kind must be score_file|diffmap (got {other})").into()),
            };
            let flat_hint = match (hint_mem_gb, hint_threads) {
                (None, None) => None,
                (m, t) => Some(zenfleet_core::ResourceHint {
                    peak_mem_bytes: (m.unwrap_or(0.5) * (1u64 << 30) as f64) as u64,
                    threads: t.unwrap_or(1),
                    vram_bytes: None,
                }),
            };
            let mut rows: Vec<zenfleet_ctl::PairRow> = Vec::new();
            use arrow_array::Array as _; // is_null on typed arrays
            for pp in &pairs {
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
                let f = std::fs::File::open(pp)?;
                let rdr = ParquetRecordBatchReaderBuilder::try_new(f)?.build()?;
                for batch in rdr {
                    let batch = batch?;
                    let get = |name: &str| -> Option<&arrow_array::StringArray> {
                        batch
                            .column_by_name(name)
                            .and_then(|c| c.as_any().downcast_ref::<arrow_array::StringArray>())
                    };
                    let (ipc, memc) = if full_uri { ("ref_path", "dist_path") } else { ("image_path", "dist_member") };
                    let (Some(ip), Some(dm)) = (get(ipc), get(memc)) else {
                        return Err(format!("{pp}: missing {ipc}/{memc} columns").into());
                    };
                    let codec = get("codec");
                    let qcol = batch
                        .column_by_name("q")
                        .and_then(|c| c.as_any().downcast_ref::<arrow_array::Int64Array>());
                    let knobs = get("knob_tuple_json");
                    for i in 0..batch.num_rows() {
                        if ip.is_null(i) || dm.is_null(i) {
                            continue;
                        }
                        let (ipv, dmv) = (ip.value(i), dm.value(i));
                        if ipv.is_empty() || dmv.is_empty() {
                            continue;
                        }
                        let base = |s: &str| s.rsplit('/').next().unwrap_or(s).to_string();
                        let identity = match (codec, qcol, knobs) {
                            (Some(c), Some(q), Some(k)) if !c.is_null(i) => {
                                Some((c.value(i).to_string(), q.value(i), k.value(i).to_string()))
                            }
                            _ => None,
                        };
                        rows.push(zenfleet_ctl::PairRow {
                            ref_key: if full_uri { ipv.to_string() } else { base(ipv) },
                            member: if full_uri { dmv.to_string() } else { base(dmv) },
                            identity,
                        });
                    }
                }
            }
            // Pixel-derived hint (scaleWxH in the ref name) overrides the flat hint per job's ref.
            let pixel_hint = |ref_key: &str| -> Option<zenfleet_core::ResourceHint> {
                if hint_mib_per_mp <= 0.0 && hint_base_mib <= 0.0 {
                    return None;
                }
                let mp = ref_key.split("scale").nth(1).and_then(|rest| {
                    let (w, rest) = rest.split_once('x')?;
                    let h: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                    Some(w.parse::<f64>().ok()? * h.parse::<f64>().ok()? / 1e6)
                });
                Some(zenfleet_core::ResourceHint {
                    peak_mem_bytes: ((hint_base_mib + mp.unwrap_or(0.0) * hint_mib_per_mp) * 1048576.0) as u64,
                    threads: 1,
                    vram_bytes: None,
                })
            };
            let mut jobs = zenfleet_ctl::declare_scorefile_jobs(
                &rows, &metrics, chunk, &cell_codec, hdr, hdr_transfer.as_deref(), diffmap, flat_hint.clone(),
            );
            for j in &mut jobs {
                if let Some(h) = pixel_hint(&j.cell.image_path) {
                    j.hint = Some(h);
                } else if j.hint.is_none() {
                    j.hint = flat_hint.clone();
                }
            }
            let manifest = serde_json::to_vec(&jobs)?;
            if let Some(mo) = manifest_out {
                std::fs::write(&mo, &manifest)?;
                println!(
                    "wrote {} jobs for {} pairs rows -> {} (no upload)",
                    jobs.len(),
                    rows.len(),
                    mo.display()
                );
                return Ok(());
            }
            let run = run.ok_or("--run required unless --manifest-out")?;
            let ep = resolve_endpoint(endpoint)?;
            // gzip via the baked `gzip` binary (writer-thread pattern — see declared_count).
            use std::io::Write as _;
            let mut c = std::process::Command::new("gzip")
                .arg("-c")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .spawn()?;
            let mut stdin = c.stdin.take().unwrap();
            let mbytes = manifest.clone();
            let wtr = std::thread::spawn(move || {
                let _ = stdin.write_all(&mbytes);
            });
            let gz = c.wait_with_output()?;
            let _ = wtr.join();
            if !gz.status.success() {
                return Err("gzip failed".into());
            }
            zenfleet_ledger::write_bytes_uri(&format!("s3://{bucket}/jobs/{run}/manifest.json"), &manifest, Some(&ep))?;
            zenfleet_ledger::write_bytes_uri(&format!("s3://{bucket}/jobs/{run}/manifest.json.gz"), &gz.stdout, Some(&ep))?;
            // control.json: create ONLY IF ABSENT — unconditionally writing
            // {"paused":false} would silently UNPAUSE an existing run on a
            // re-declare (a live hazard the Python this replaced also had).
            let ckey = format!("s3://{bucket}/jobs/{run}/control.json");
            if zenfleet_ledger::read_bytes_uri(&ckey, Some(&ep)).is_err() {
                zenfleet_ledger::write_bytes_uri(&ckey, b"{\"paused\":false}", Some(&ep))?;
            } else {
                println!("control.json exists — left untouched (pause state preserved)");
            }
            println!(
                "declared {} jobs for {} pairs rows -> s3://{bucket}/jobs/{run}/ (direct-object; requires stamped)",
                jobs.len(),
                rows.len()
            );
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
