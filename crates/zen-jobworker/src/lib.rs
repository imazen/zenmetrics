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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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

/// An R2 (or any S3-compatible) target. Blobs land at `s3://{bucket}/{prefix}/{sha}`.
#[derive(Debug, Clone)]
pub struct R2Target {
    pub endpoint: String,
    pub bucket: String,
    pub prefix: String,
}

/// Content-addressed blob store over R2 via the `s5cmd` CLI. Requires `s5cmd` on PATH and
/// `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment at runtime (the launcher maps the
/// `R2_*` creds to `AWS_*`). The `blobs/<sha>` layout was verified live against R2.
pub struct R2BlobStore {
    target: R2Target,
}

impl R2BlobStore {
    pub fn new(endpoint: String, bucket: String, prefix: String) -> Self {
        Self { target: R2Target { endpoint, bucket, prefix } }
    }

    /// The full `s3://` key for a content hash.
    pub fn key(&self, sha: &Sha256Hex) -> String {
        format!("s3://{}/{}/{}", self.target.bucket, self.target.prefix.trim_matches('/'), sha)
    }

    fn s5cmd(&self) -> Command {
        let mut c = Command::new("s5cmd");
        c.arg("--endpoint-url").arg(&self.target.endpoint);
        c
    }
}

impl BlobStore for R2BlobStore {
    fn put(&self, bytes: &[u8]) -> io::Result<Sha256Hex> {
        let sha = sha256(bytes);
        if self.exists(&sha) {
            return Ok(sha); // content-addressed dedup — already in R2
        }
        let tmp = std::env::temp_dir().join(format!("zenblob_{}_{}", std::process::id(), sha.as_str()));
        std::fs::write(&tmp, bytes)?;
        let status = self
            .s5cmd()
            .arg("cp")
            .arg(&tmp)
            .arg(self.key(&sha))
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status();
        let _ = std::fs::remove_file(&tmp);
        match status {
            Ok(s) if s.success() => Ok(sha),
            Ok(s) => Err(io::Error::other(format!("s5cmd cp exited {:?}", s.code()))),
            Err(e) => Err(e),
        }
    }

    fn exists(&self, sha: &Sha256Hex) -> bool {
        self.s5cmd()
            .arg("ls")
            .arg(self.key(sha))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// R2 claim config: claims live at `s3://{bucket}/{prefix}/{job_id}`, created via conditional write.
/// The endpoint comes from the R2 blob target.
#[derive(Debug, Clone)]
pub struct ClaimCfg {
    pub bucket: String,
    pub prefix: String,
    /// A claim older than this (and not yet a terminal ledger row) is presumed dead and stealable.
    pub ttl_secs: u64,
}

/// Atomically claim a job via R2 conditional write (`If-None-Match: *`). Returns true iff THIS worker
/// won (object created); false if it already existed (another worker owns it) or on error. R2 admits
/// exactly one create per key, so concurrent workers can't both win — no double execution.
pub fn try_claim_r2(endpoint: &str, bucket: &str, prefix: &str, job_id: &JobId) -> bool {
    let key = format!("{}/{}", prefix.trim_matches('/'), job_id.as_str());
    Command::new("aws")
        .arg("--endpoint-url")
        .arg(endpoint)
        .arg("s3api")
        .arg("put-object")
        .arg("--bucket")
        .arg(bucket)
        .arg("--key")
        .arg(&key)
        .arg("--if-none-match")
        .arg("*")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Release (delete) a claim so the job requeues immediately — used on spot preemption (goal F:
/// "spot reclaim is a non-event") instead of waiting out the claim TTL. Best-effort: a failed delete
/// just falls back to the slower TTL-based stale-reclaim (goal E), so correctness never depends on it.
pub fn release_claim_r2(endpoint: &str, bucket: &str, prefix: &str, job_id: &JobId) -> bool {
    let key = format!("{}/{}", prefix.trim_matches('/'), job_id.as_str());
    aws_s3api(endpoint)
        .arg("delete-object")
        .arg("--bucket")
        .arg(bucket)
        .arg("--key")
        .arg(&key)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Install the spot-preemption handler (goal F): on SIGTERM/SIGINT, release the in-flight claim (if
/// any) so the job requeues immediately, then exit. Runs on a dedicated signal-hook thread (safe to
/// spawn `aws`). No-op if signal registration fails (falls back to TTL reclaim, goal E).
fn spawn_spot_reclaim(inflight: Arc<Mutex<Option<JobId>>>, endpoint: &str, bucket: &str, prefix: &str) {
    let (endpoint, bucket, prefix) = (endpoint.to_string(), bucket.to_string(), prefix.to_string());
    let Ok(mut signals) =
        signal_hook::iterator::Signals::new([signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT])
    else {
        return;
    };
    std::thread::spawn(move || {
        if signals.forever().next().is_some() {
            if let Some(id) = inflight.lock().ok().and_then(|g| g.clone()) {
                let released = release_claim_r2(&endpoint, &bucket, &prefix, &id);
                eprintln!(
                    "zen-jobworker: spot preemption — {} claim {} for fast requeue",
                    if released { "released" } else { "could not release" },
                    id.as_str()
                );
            } else {
                eprintln!("zen-jobworker: spot preemption — no in-flight claim to release");
            }
            std::process::exit(130);
        }
    });
}

static CLAIM_TMP_N: AtomicU64 = AtomicU64::new(0);

fn aws_s3api(endpoint: &str) -> Command {
    let mut c = Command::new("aws");
    c.arg("--endpoint-url").arg(endpoint).arg("s3api");
    c
}

/// Pure staleness check: a claim is stealable once its age reaches the TTL.
fn claim_is_stale(now: u64, claim_ts: u64, ttl_secs: u64) -> bool {
    now.saturating_sub(claim_ts) >= ttl_secs
}

/// Read a claim's `(etag, ts)` — `ts` is the first whitespace token of the body. None on any error.
fn read_claim(endpoint: &str, bucket: &str, key: &str) -> Option<(String, u64)> {
    let n = CLAIM_TMP_N.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!("zenclaim_rd_{}_{}", std::process::id(), n));
    let res = aws_s3api(endpoint)
        .arg("get-object")
        .arg("--bucket").arg(bucket)
        .arg("--key").arg(key)
        .arg(&out)
        .arg("--query").arg("ETag")
        .arg("--output").arg("text")
        .stderr(Stdio::null())
        .output();
    let etag = match res {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => {
            let _ = std::fs::remove_file(&out);
            return None;
        }
    };
    let body = std::fs::read_to_string(&out).ok();
    let _ = std::fs::remove_file(&out);
    let ts = body?.split_whitespace().next()?.parse::<u64>().ok()?;
    Some((etag, ts))
}

/// Claim a job, **stealing a stale claim** if the prior owner is presumed dead (claim age ≥ `ttl_secs`).
/// Steal is itself a CAS (`If-Match` on the claim's ETag), so two reclaimers can't both win. Returns
/// true iff this worker now owns the claim. This is the dead-worker reclaim (goal E).
pub fn claim_or_steal_r2(
    endpoint: &str,
    bucket: &str,
    prefix: &str,
    job_id: &JobId,
    now: u64,
    ttl_secs: u64,
    owner: &str,
) -> bool {
    let key = format!("{}/{}", prefix.trim_matches('/'), job_id.as_str());
    let n = CLAIM_TMP_N.fetch_add(1, Ordering::Relaxed);
    let body = std::env::temp_dir().join(format!("zenclaim_bd_{}_{}", std::process::id(), n));
    if std::fs::write(&body, format!("{now} {owner}")).is_err() {
        return false;
    }
    // 1. fresh claim (create-if-absent)
    let fresh = aws_s3api(endpoint)
        .arg("put-object").arg("--bucket").arg(bucket).arg("--key").arg(&key)
        .arg("--body").arg(&body).arg("--if-none-match").arg("*")
        .stdout(Stdio::null()).stderr(Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false);
    if fresh {
        let _ = std::fs::remove_file(&body);
        return true;
    }
    // 2. exists — steal only if stale, via If-Match CAS on the current ETag
    let won = match read_claim(endpoint, bucket, &key) {
        Some((etag, prev_ts)) if claim_is_stale(now, prev_ts, ttl_secs) => aws_s3api(endpoint)
            .arg("put-object").arg("--bucket").arg(bucket).arg("--key").arg(&key)
            .arg("--body").arg(&body).arg("--if-match").arg(&etag)
            .stdout(Stdio::null()).stderr(Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false),
        _ => false,
    };
    let _ = std::fs::remove_file(&body);
    won
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
    /// Gap jobs another worker claimed first (concurrent-safety; not executed here).
    pub skipped: usize,
}

/// Execute the reconciler's gap (single worker — no concurrent claiming). Thin wrapper over
/// [`execute_gap_claimed`] with an always-win claim.
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
    execute_gap_claimed(desired, view, policy, handler, store, |_| true, ctx)
}

/// Execute the gap with a per-job `claim` predicate — a job runs only if `claim(job_id)` is true.
/// With an R2 conditional-write claim, concurrent workers win disjoint subsets → no double execution.
/// Emit POISON rows the reconciler decided; `now` is injected; failures are rows.
#[allow(clippy::too_many_arguments)]
pub fn execute_gap_claimed<H, B, C>(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    handler: H,
    store: &B,
    claim: C,
    ctx: WorkerCtx<'_>,
) -> ExecOutcome
where
    H: Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError>,
    B: BlobStore,
    C: Fn(&JobId) -> bool,
{
    let plan = reconcile(desired, view, policy);
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    let mut out = ExecOutcome { rows: Vec::new(), done: 0, failed: 0, poisoned: 0, skipped: 0 };

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
        if !claim(id) {
            out.skipped += 1; // another worker claimed this job first
            continue;
        }
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
    /// Content-addressed blob dir (used when `r2` is None).
    pub blobs: PathBuf,
    /// If set, write content-addressed blobs to R2 instead of the local dir.
    pub r2: Option<R2Target>,
    /// If set (requires `r2`), claim each gap job via R2 conditional write before executing it —
    /// concurrent-safe fleet claiming (no two workers run the same job).
    pub claims: Option<ClaimCfg>,
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

    // Ledger paths may be local or s3:// — the R2 endpoint (if any) comes from the blob target.
    let endpoint = cfg.r2.as_ref().map(|t| t.endpoint.as_str());
    let mut view = LedgerView::new();
    for p in &cfg.ledger_in {
        let uri = p.to_string_lossy();
        for row in zen_ledger::read_ledger_uri(uri.as_ref(), endpoint)
            .map_err(|e| WorkerRunError::Ledger(e.to_string()))?
        {
            view.apply(row);
        }
    }

    let policy = RetryPolicy { max_attempts: cfg.max_attempts };
    let ctx = WorkerCtx { worker: &cfg.worker, provider: &cfg.provider, now: cfg.now };
    // Pick the blob store: R2 if configured, else local FS. execute_gap is generic over the store,
    // so each arm monomorphizes against the concrete type.
    let out = match &cfg.r2 {
        Some(t) => {
            let store = R2BlobStore::new(t.endpoint.clone(), t.bucket.clone(), t.prefix.clone());
            match &cfg.claims {
                Some(cc) => {
                    // Spot-reclaim (goal F): track the in-flight claim; on SIGTERM/SIGINT (spot
                    // preemption) release it so the job requeues immediately instead of waiting out
                    // the TTL. The signal runs on a dedicated signal-hook thread, so spawning `aws`
                    // to delete the claim is safe (not an async-signal handler). Best-effort — if the
                    // release misses, TTL stale-reclaim (goal E) still requeues it.
                    let inflight: Arc<Mutex<Option<JobId>>> = Arc::new(Mutex::new(None));
                    spawn_spot_reclaim(inflight.clone(), &t.endpoint, &cc.bucket, &cc.prefix);
                    execute_gap_claimed(
                        &desired,
                        &view,
                        policy,
                        |job| exec_command(&cfg.exec, job),
                        &store,
                        |id| {
                            let won = claim_or_steal_r2(
                                &t.endpoint, &cc.bucket, &cc.prefix, id, cfg.now, cc.ttl_secs, &cfg.worker,
                            );
                            if won {
                                if let Ok(mut g) = inflight.lock() {
                                    *g = Some(id.clone());
                                }
                            }
                            won
                        },
                        ctx,
                    )
                }
                None => execute_gap(&desired, &view, policy, |job| exec_command(&cfg.exec, job), &store, ctx),
            }
        }
        None => {
            let store = LocalBlobStore::new(cfg.blobs.clone())
                .map_err(|e| WorkerRunError::Io(e.to_string()))?;
            execute_gap(&desired, &view, policy, |job| exec_command(&cfg.exec, job), &store, ctx)
        }
    };
    let out_uri = cfg.ledger_out.to_string_lossy();
    zen_ledger::write_ledger_uri(out_uri.as_ref(), &out.rows, endpoint)
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
            r2: None,
            claims: None,
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

    #[test]
    fn r2_key_derivation() {
        let s = R2BlobStore::new("https://acct.r2.cloudflarestorage.com".into(), "zen-tuning-ephemeral".into(), "blobs".into());
        let sha = sha256(b"hi");
        assert_eq!(s.key(&sha), format!("s3://zen-tuning-ephemeral/blobs/{sha}"));
        // leading/trailing slashes in the prefix don't double up
        let s2 = R2BlobStore::new("e".into(), "b".into(), "/blobs/".into());
        assert_eq!(s2.key(&sha), format!("s3://b/blobs/{sha}"));
    }

    #[test]
    fn claim_staleness_check() {
        assert!(claim_is_stale(1000, 0, 10), "ancient claim is stealable");
        assert!(claim_is_stale(1000, 990, 10), "exactly ttl old is stealable");
        assert!(!claim_is_stale(1000, 995, 10), "fresh claim (5s < 10s ttl) is NOT stealable");
        assert!(!claim_is_stale(5, 0, 10), "clock skew / before ttl elapsed: not stealable");
    }
}
