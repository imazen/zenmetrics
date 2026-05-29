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
use std::io;
use std::path::PathBuf;

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
}
