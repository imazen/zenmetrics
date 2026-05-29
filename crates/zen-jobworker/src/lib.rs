#![forbid(unsafe_code)]
//! # zen-jobworker
//!
//! The bridge from the reconciler's *gap* to real execution (goal A: declare → execute). For each
//! gap job: run a handler → **content-address its output to a blob store** (goal G) → emit a
//! [`LedgerRow`] (Done/Failed/Poison). It also emits the POISON rows the reconciler decided
//! (goal F — doomed work stops, recorded). The ledger is the source of truth, so a second pass over
//! the updated ledger does nothing (goal E — converges).
//!
//! Handlers are plain closures: `Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError>`. The production
//! handler shells out to the encoder/scorer (`zen-metrics`); tests use a stub. [`BlobStore`] is
//! content-addressed local FS today; an R2 impl drops in behind the trait. Pure enough to test the
//! whole loop end-to-end with a temp dir.

use std::collections::HashMap;
use std::io::{self, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use zen_job_core::{
    reconcile, sha256, DesiredJob, ErrorClass, JobId, JobStatus, LedgerRow, LedgerView, RetryPolicy,
    Sha256Hex,
};

/// A classified execution failure — becomes a FAILED ledger row carrying this `error_class`, which
/// the reconciler then treats as transient (retry) or deterministic (poison).
#[derive(Debug, Clone)]
pub struct HandlerError {
    pub class: ErrorClass,
    pub msg: String,
}

impl HandlerError {
    pub fn new(class: ErrorClass, msg: impl Into<String>) -> Self {
        Self { class, msg: msg.into() }
    }
}

/// Content-addressed blob storage. Local FS today; an R2 impl drops in behind this trait.
pub trait BlobStore {
    /// Store `bytes`, returning their content address. Identical bytes dedup to one object.
    fn put(&self, bytes: &[u8]) -> io::Result<Sha256Hex>;
    fn exists(&self, sha: &Sha256Hex) -> bool;
}

/// `blobs/<sha256>` on the local filesystem (the `zen-cloud-local` dev mode).
pub struct LocalBlobStore {
    root: PathBuf,
}

impl LocalBlobStore {
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    fn path(&self, sha: &Sha256Hex) -> PathBuf {
        self.root.join(sha.as_str())
    }
}

impl BlobStore for LocalBlobStore {
    fn put(&self, bytes: &[u8]) -> io::Result<Sha256Hex> {
        let sha = sha256(bytes);
        let p = self.path(&sha);
        if !p.exists() {
            std::fs::write(&p, bytes)?; // content-addressed → identical bytes are written once
        }
        Ok(sha)
    }

    fn exists(&self, sha: &Sha256Hex) -> bool {
        self.path(sha).exists()
    }
}

/// Identity/time context for the rows a worker emits (who ran it, on what provider, when).
#[derive(Clone, Copy, Debug)]
pub struct WorkerCtx<'a> {
    pub worker: &'a str,
    pub provider: &'a str,
    /// Unix seconds — injected, no clock in this layer.
    pub now: u64,
}

/// Result of executing one gap pass.
pub struct ExecOutcome {
    /// Rows to append to the ledger (Done / Failed / Poison).
    pub rows: Vec<LedgerRow>,
    pub done: usize,
    pub failed: usize,
    pub poisoned: usize,
}

/// Execute the reconciler's gap. For each job to enqueue: run `handler`, content-address its output
/// via `store`, emit a row. Emit POISON rows the reconciler decided. `now` (unix secs) is injected —
/// no clock here. Returns rows for the caller to persist via `zen_ledger::write_ledger`.
pub fn execute_gap<H, B>(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    handler: H,
    store: &B,
    ctx: WorkerCtx<'_>,
) -> ExecOutcome
where
    H: Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError>,
    B: BlobStore,
{
    let plan = reconcile(desired, view, policy);
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    let mut out = ExecOutcome { rows: Vec::new(), done: 0, failed: 0, poisoned: 0 };

    let make = |d: &DesiredJob,
                status: JobStatus,
                output_sha: Option<Sha256Hex>,
                error_class: Option<ErrorClass>|
     -> LedgerRow {
        LedgerRow {
            job_id: d.job_id(),
            kind: d.kind.clone(),
            cell: d.cell.clone(),
            output_sha,
            status,
            error_class,
            attempts: view.get(&d.job_id()).map(|r| r.attempts + 1).unwrap_or(1),
            ts: ctx.now,
            worker: ctx.worker.to_string(),
            provider: ctx.provider.to_string(),
        }
    };

    for id in &plan.enqueue {
        let Some(d) = by_id.get(id) else { continue };
        match handler(d) {
            Ok(bytes) => match store.put(&bytes) {
                Ok(sha) => {
                    out.rows.push(make(d, JobStatus::Done, Some(sha), None));
                    out.done += 1;
                }
                Err(_) => {
                    // the encode/score succeeded but persistence failed → transient, retry next pass
                    out.rows.push(make(d, JobStatus::Failed, None, Some(ErrorClass::UploadFail)));
                    out.failed += 1;
                }
            },
            Err(he) => {
                out.rows.push(make(d, JobStatus::Failed, None, Some(he.class)));
                out.failed += 1;
            }
        }
    }

    for id in &plan.poison {
        if let Some(d) = by_id.get(id) {
            let prev_err = view.get(id).and_then(|r| r.error_class);
            out.rows.push(make(d, JobStatus::Poison, None, prev_err));
            out.poisoned += 1;
        }
    }

    out
}

/// Production handler: shell out to an executor `program`. The job descriptor is written as JSON to
/// the program's stdin; its stdout is the output bytes (which get content-addressed). Exit 0 =
/// success; spawn failure → transient `WorkerLost`; non-zero exit → `EncoderPanic` (deterministic).
/// Any executor honoring this stdin-JSON → stdout-bytes contract plugs in (e.g. a future
/// `zen-metrics jobexec` subcommand).
pub fn exec_command(program: &str, job: &DesiredJob) -> Result<Vec<u8>, HandlerError> {
    let job_json = serde_json::to_vec(job)
        .map_err(|e| HandlerError::new(ErrorClass::Unknown, format!("serialize job: {e}")))?;
    let mut child = Command::new(program)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| HandlerError::new(ErrorClass::WorkerLost, format!("spawn {program}: {e}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&job_json)
            .map_err(|e| HandlerError::new(ErrorClass::WorkerLost, format!("write stdin: {e}")))?;
        // stdin dropped here → EOF to the child
    }
    let output = child
        .wait_with_output()
        .map_err(|e| HandlerError::new(ErrorClass::WorkerLost, format!("wait {program}: {e}")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        let code = output.status.code().unwrap_or(-1);
        Err(HandlerError::new(
            ErrorClass::EncoderPanic,
            format!("{program} exited {code}: {}", String::from_utf8_lossy(&output.stderr)),
        ))
    }
}

/// Configuration for one worker pass (the runnable `zen-jobworker` binary parses CLI args into this).
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// JSON file: an array of `DesiredJob`.
    pub manifest: PathBuf,
    /// Existing ledger sidecars to fold into the latest-wins view (the "actual" state).
    pub ledger_in: Vec<PathBuf>,
    /// Where this pass's new rows are written.
    pub ledger_out: PathBuf,
    /// Content-addressed blob dir (local stand-in for `blobs/<sha>` on R2).
    pub blobs: PathBuf,
    /// Executor program (stdin-JSON → stdout-bytes contract).
    pub exec: String,
    pub worker: String,
    pub provider: String,
    pub now: u64,
    pub max_attempts: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerRunError {
    #[error("io {0}")]
    Io(String),
    #[error("manifest {0}")]
    Manifest(String),
    #[error("ledger {0}")]
    Ledger(String),
}

/// One worker pass: load the manifest + existing ledger → reconcile the gap → execute each job via
/// `exec` → content-address outputs → write the resulting rows. Returns the outcome. Deterministic
/// given `cfg.now` (the binary supplies the wall clock; the library stays clock-free + testable).
pub fn run(cfg: &WorkerConfig) -> Result<ExecOutcome, WorkerRunError> {
    let bytes = std::fs::read(&cfg.manifest)
        .map_err(|e| WorkerRunError::Io(format!("read manifest {}: {e}", cfg.manifest.display())))?;
    let desired: Vec<DesiredJob> =
        serde_json::from_slice(&bytes).map_err(|e| WorkerRunError::Manifest(e.to_string()))?;

    let mut view = LedgerView::new();
    for p in &cfg.ledger_in {
        for row in zen_ledger::read_ledger(p).map_err(|e| WorkerRunError::Ledger(e.to_string()))? {
            view.apply(row);
        }
    }

    let store = LocalBlobStore::new(cfg.blobs.clone()).map_err(|e| WorkerRunError::Io(e.to_string()))?;
    let out = execute_gap(
        &desired,
        &view,
        RetryPolicy { max_attempts: cfg.max_attempts },
        |job| exec_command(&cfg.exec, job),
        &store,
        WorkerCtx { worker: &cfg.worker, provider: &cfg.provider, now: cfg.now },
    );
    zen_ledger::write_ledger(&cfg.ledger_out, &out.rows)
        .map_err(|e| WorkerRunError::Ledger(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use zen_job_core::{CellId, JobKind};

    static N: AtomicU64 = AtomicU64::new(0);
    fn tmp() -> PathBuf {
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("zenjobworker_{}_{}", std::process::id(), n))
    }
    fn desired(metric: &str, enc: &[u8]) -> DesiredJob {
        DesiredJob {
            kind: JobKind::Metric { metric: metric.into() },
            inputs: vec![sha256(enc)],
            cell: CellId { image_path: "x".into(), codec: "zenjpeg".into(), q: 80, knob_tuple_json: "{}".into() },
        }
    }

    #[test]
    fn executes_gap_writes_blobs_and_rows() {
        let store = LocalBlobStore::new(tmp()).unwrap();
        let d = vec![desired("cvvdp", b"a"), desired("ssim2", b"a")];
        let out = execute_gap(
            &d,
            &LedgerView::new(),
            RetryPolicy::default(),
            |job| Ok(format!("score:{}", job.job_id().as_str()).into_bytes()),
            &store,
            WorkerCtx { worker: "w1", provider: "local", now: 100 },
        );
        assert_eq!(out.done, 2);
        assert_eq!(out.rows.len(), 2);
        for r in &out.rows {
            assert_eq!(r.status, JobStatus::Done);
            let sha = r.output_sha.clone().unwrap();
            assert!(store.exists(&sha), "output blob is written content-addressed");
        }
    }

    #[test]
    fn converges_on_second_pass() {
        let store = LocalBlobStore::new(tmp()).unwrap();
        let d = vec![desired("cvvdp", b"a")];
        let out1 = execute_gap(
            &d,
            &LedgerView::new(),
            RetryPolicy::default(),
            |job| Ok(job.job_id().as_str().as_bytes().to_vec()),
            &store,
            WorkerCtx { worker: "w1", provider: "local", now: 100 },
        );
        let view = LedgerView::from_rows(out1.rows);
        let out2 = execute_gap(
            &d,
            &view,
            RetryPolicy::default(),
            |_| panic!("handler must NOT run for an already-done job"),
            &store,
            WorkerCtx { worker: "w1", provider: "local", now: 200 },
        );
        assert_eq!(out2.done, 0);
        assert!(out2.rows.is_empty(), "converged — nothing left in the gap");
    }

    #[test]
    fn failure_is_classified_and_writes_no_blob() {
        let store = LocalBlobStore::new(tmp()).unwrap();
        let d = vec![desired("cvvdp", b"a")];
        let out = execute_gap(
            &d,
            &LedgerView::new(),
            RetryPolicy::default(),
            |_| Err(HandlerError::new(ErrorClass::DecodeError, "bad input")),
            &store,
            WorkerCtx { worker: "w1", provider: "local", now: 100 },
        );
        assert_eq!(out.failed, 1);
        assert_eq!(out.rows[0].status, JobStatus::Failed);
        assert_eq!(out.rows[0].error_class, Some(ErrorClass::DecodeError));
        assert!(out.rows[0].output_sha.is_none());
    }

    #[test]
    fn over_cap_transient_becomes_poison() {
        let store = LocalBlobStore::new(tmp()).unwrap();
        let d = vec![desired("cvvdp", b"a")];
        let view = LedgerView::from_rows([LedgerRow {
            job_id: d[0].job_id(),
            kind: d[0].kind.clone(),
            cell: d[0].cell.clone(),
            output_sha: None,
            status: JobStatus::Failed,
            error_class: Some(ErrorClass::Timeout),
            attempts: 3,
            ts: 1,
            worker: "w".into(),
            provider: "local".into(),
        }]);
        let out = execute_gap(
            &d,
            &view,
            RetryPolicy { max_attempts: 3 },
            |_| Ok(vec![1, 2, 3]),
            &store,
            WorkerCtx { worker: "w1", provider: "local", now: 200 },
        );
        assert_eq!(out.poisoned, 1);
        assert_eq!(out.done, 0, "poisoned job is not executed");
        assert_eq!(out.rows[0].status, JobStatus::Poison);
    }

    #[test]
    fn exec_command_captures_stdout() {
        let d = desired("cvvdp", b"a");
        // `cat` echoes the job JSON it receives on stdin → that's the (content-addressable) output
        let out = exec_command("cat", &d).unwrap();
        assert_eq!(out, serde_json::to_vec(&d).unwrap());
    }

    #[test]
    fn exec_command_missing_program_is_transient() {
        let d = desired("cvvdp", b"a");
        let err = exec_command("zzz-no-such-program-12345", &d).unwrap_err();
        assert_eq!(err.class, ErrorClass::WorkerLost, "infra failure → retryable, not poison");
    }

    #[test]
    fn run_pass_is_end_to_end_and_converges() {
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("jobs.json");
        let d = vec![desired("cvvdp", b"a"), desired("ssim2", b"b")];
        std::fs::write(&manifest, serde_json::to_vec(&d).unwrap()).unwrap();
        let cfg = WorkerConfig {
            manifest,
            ledger_in: vec![],
            ledger_out: dir.join("out.parquet"),
            blobs: dir.join("blobs"),
            exec: "cat".into(),
            worker: "w1".into(),
            provider: "local".into(),
            now: 100,
            max_attempts: 3,
        };
        let out = run(&cfg).unwrap();
        assert_eq!(out.done, 2);
        let rows = zen_ledger::read_ledger(&cfg.ledger_out).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.status == JobStatus::Done && r.output_sha.is_some()));

        // second pass folds in the just-written ledger → gap empty → executor never invoked
        let cfg2 = WorkerConfig {
            ledger_in: vec![cfg.ledger_out.clone()],
            ledger_out: dir.join("out2.parquet"),
            exec: "false".into(), // would fail if called; it must not be
            ..cfg.clone()
        };
        let out2 = run(&cfg2).unwrap();
        assert_eq!(out2.done, 0, "all jobs already DONE → converged, nothing re-run");
        assert!(out2.rows.is_empty());
    }
}
