#![forbid(unsafe_code)]
//! # zenfleet-worker
//!
//! The bridge from the reconciler's *gap* to real execution (goal A: declare → execute). For each
//! gap job: run a handler → **content-address its output to a blob store** (goal G) → emit a
//! [`LedgerRow`] (Done/Failed/Poison). It also emits the POISON rows the reconciler decided
//! (goal F — doomed work stops, recorded). The ledger is the source of truth, so a second pass over
//! the updated ledger does nothing (goal E — converges).
//!
//! Handlers are plain closures: `Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError>`. The production
//! handler shells out to the encoder/scorer (`zenmetrics`); tests use a stub. [`BlobStore`] is
//! content-addressed local FS today; an R2 impl drops in behind the trait. Pure enough to test the
//! whole loop end-to-end with a temp dir.

use std::collections::{HashMap, HashSet};
use std::io::{self, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use zenfleet_core::{
    BlobIndexEntry, BoxBudget, DesiredJob, ErrorClass, InFlight, JobCost, JobId, JobStatus,
    LedgerRow, LedgerView, Regenerability, ResourceClass, ResourceHint, RetryPolicy, RunControl,
    Sha256Hex, Tombstone, gc_plan, lru_cap_evict, reconcile, sha256, worker_serves,
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
        Self {
            class,
            msg: msg.into(),
        }
    }
}

/// Content-addressed blob storage. Local FS today; an R2 impl drops in behind this trait.
pub trait BlobStore {
    /// Store `bytes`, returning their content address. Identical bytes dedup to one object.
    fn put(&self, bytes: &[u8]) -> io::Result<Sha256Hex>;
    fn exists(&self, sha: &Sha256Hex) -> bool;
}

/// `blobs/<sha256>` on the local filesystem (the `zenfleet-local` dev mode).
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
        Self {
            target: R2Target {
                endpoint,
                bucket,
                prefix,
            },
        }
    }

    /// The full `s3://` key for a content hash.
    pub fn key(&self, sha: &Sha256Hex) -> String {
        format!(
            "s3://{}/{}/{}",
            self.target.bucket,
            self.target.prefix.trim_matches('/'),
            sha
        )
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
        let tmp =
            std::env::temp_dir().join(format!("zenblob_{}_{}", std::process::id(), sha.as_str()));
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
    /// Speculative-execution threshold (goal E): a *live* primary claim older than this (but younger
    /// than `ttl_secs`) is a straggler — a second worker may co-run it speculatively to bound the long
    /// tail. The ledger's latest-wins on `job_id` makes the loser a harmless duplicate. `None` = off.
    pub spec_threshold_secs: Option<u64>,
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
fn spawn_spot_reclaim(
    inflight: Arc<Mutex<Option<JobId>>>,
    endpoint: &str,
    bucket: &str,
    prefix: &str,
) {
    let (endpoint, bucket, prefix) = (endpoint.to_string(), bucket.to_string(), prefix.to_string());
    let Ok(mut signals) = signal_hook::iterator::Signals::new([
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGINT,
    ]) else {
        return;
    };
    std::thread::spawn(move || {
        if signals.forever().next().is_some() {
            if let Some(id) = inflight.lock().ok().and_then(|g| g.clone()) {
                let released = release_claim_r2(&endpoint, &bucket, &prefix, &id);
                eprintln!(
                    "zenfleet-worker: spot preemption — {} claim {} for fast requeue",
                    if released {
                        "released"
                    } else {
                        "could not release"
                    },
                    id.as_str()
                );
            } else {
                eprintln!("zenfleet-worker: spot preemption — no in-flight claim to release");
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

/// Read the run-control object (goal C: pause/drain). Absent or unparseable → `RUNNING` — fail-open,
/// so a missing/garbled control object can never wedge the fleet.
pub fn fetch_control_r2(endpoint: &str, bucket: &str, key: &str) -> RunControl {
    let n = CLAIM_TMP_N.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!("zenctl_{}_{}", std::process::id(), n));
    let ok = aws_s3api(endpoint)
        .arg("get-object")
        .arg("--bucket")
        .arg(bucket)
        .arg("--key")
        .arg(key)
        .arg(&out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let ctl = if ok {
        std::fs::read(&out)
            .ok()
            .and_then(|b| serde_json::from_slice::<RunControl>(&b).ok())
            .unwrap_or_default()
    } else {
        RunControl::default()
    };
    let _ = std::fs::remove_file(&out);
    ctl
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
        .arg("--bucket")
        .arg(bucket)
        .arg("--key")
        .arg(key)
        .arg(&out)
        .arg("--query")
        .arg("ETag")
        .arg("--output")
        .arg("text")
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
#[allow(clippy::too_many_arguments)]
pub fn claim_or_steal_r2(
    endpoint: &str,
    bucket: &str,
    prefix: &str,
    job_id: &JobId,
    now: u64,
    ttl_secs: u64,
    spec_threshold_secs: Option<u64>,
    owner: &str,
) -> bool {
    claim_or_steal_r2_key(
        endpoint,
        bucket,
        prefix,
        job_id.as_str(),
        now,
        ttl_secs,
        spec_threshold_secs,
        owner,
    )
}

/// The string-keyed core of [`claim_or_steal_r2`]: claim/steal an arbitrary `id` (the claim object is
/// `{prefix}/{id}`), so the same exactly-once R2-lease mechanism covers both per-cell claims (id =
/// `JobId`) and the chunked path's coarse per-chunk claims (id = [`chunk_id`], a `chunk-…` key that
/// never collides with a bare-sha cell claim). Same CAS semantics as [`claim_or_steal_r2`].
// All eight are irreducible CAS inputs (endpoint/bucket/prefix/id/now/ttl/spec-threshold/owner);
// same rationale as the `#[allow]` on `execute_gap_claimed` below.
#[allow(clippy::too_many_arguments)]
pub fn claim_or_steal_r2_key(
    endpoint: &str,
    bucket: &str,
    prefix: &str,
    id: &str,
    now: u64,
    ttl_secs: u64,
    spec_threshold_secs: Option<u64>,
    owner: &str,
) -> bool {
    let key = format!("{}/{}", prefix.trim_matches('/'), id);
    let n = CLAIM_TMP_N.fetch_add(1, Ordering::Relaxed);
    let body = std::env::temp_dir().join(format!("zenclaim_bd_{}_{}", std::process::id(), n));
    if std::fs::write(&body, format!("{now} {owner}")).is_err() {
        return false;
    }
    // 1. fresh claim (create-if-absent)
    let fresh = aws_s3api(endpoint)
        .arg("put-object")
        .arg("--bucket")
        .arg(bucket)
        .arg("--key")
        .arg(&key)
        .arg("--body")
        .arg(&body)
        .arg("--if-none-match")
        .arg("*")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if fresh {
        let _ = std::fs::remove_file(&body);
        return true;
    }
    // 2. exists — steal only if stale, via If-Match CAS on the current ETag
    let won = match read_claim(endpoint, bucket, &key) {
        Some((etag, prev_ts)) if claim_is_stale(now, prev_ts, ttl_secs) => aws_s3api(endpoint)
            .arg("put-object")
            .arg("--bucket")
            .arg(bucket)
            .arg("--key")
            .arg(&key)
            .arg("--body")
            .arg(&body)
            .arg("--if-match")
            .arg(&etag)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false),
        // 3. live but a straggler (age in [spec_threshold, ttl)) → speculate: take a *separate*
        //    spec claim (create-if-absent, so at most one speculator) and co-run it. The ledger's
        //    latest-wins on job_id makes the loser a harmless duplicate.
        Some((_, prev_ts)) => match spec_threshold_secs {
            Some(spec) if now.saturating_sub(prev_ts) >= spec => {
                let spec_key = format!("{}/spec/{}", prefix.trim_matches('/'), id);
                aws_s3api(endpoint)
                    .arg("put-object")
                    .arg("--bucket")
                    .arg(bucket)
                    .arg("--key")
                    .arg(&spec_key)
                    .arg("--body")
                    .arg(&body)
                    .arg("--if-none-match")
                    .arg("*")
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false)
            }
            _ => false,
        },
        None => false,
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
    let mut plan = reconcile(desired, view, policy);
    // Shuffle the gap per worker so concurrent workers don't all iterate from job 0 in the same order
    // and collide on (wasting an aws claim-attempt skipping) the same already-claimed prefix — without
    // this, a late-joining box burns ~1s/job skipping thousands of jobs the early boxes already claimed
    // before it reaches free work (observed 2026-06-24: 24 boxes idle at GPU 0% behind the prefix). A
    // deterministic hash(job_id, worker) order spreads each worker across the gap so it hits free jobs
    // immediately. Deterministic (no RNG) so a re-run of the same worker is reproducible.
    plan.enqueue.sort_by_cached_key(|id| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(id, &mut h);
        std::hash::Hash::hash(ctx.worker, &mut h);
        std::hash::Hasher::finish(&h)
    });
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    let mut out = ExecOutcome {
        rows: Vec::new(),
        done: 0,
        failed: 0,
        poisoned: 0,
        skipped: 0,
    };

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
                    out.rows.push(make(
                        d,
                        JobStatus::Failed,
                        None,
                        Some(ErrorClass::UploadFail),
                    ));
                    out.failed += 1;
                }
            },
            Err(he) => {
                out.rows
                    .push(make(d, JobStatus::Failed, None, Some(he.class)));
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

/// Stable id for a chunk = `chunk-` + SHA-256 over its members' content-addressed job-ids (in chunk
/// order). Deterministic across workers — every worker forms identical chunks from the same
/// manifest-ordered gap — so the per-chunk R2 claim is **exclusive** (two workers never form
/// overlapping chunks). The `chunk-` prefix namespaces it away from bare-sha per-cell claim keys.
pub fn chunk_id(job_ids: &[JobId]) -> String {
    let mut buf = String::new();
    for id in job_ids {
        buf.push_str(id.as_str());
        buf.push('\n');
    }
    format!("chunk-{}", sha256(buf.as_bytes()).as_str())
}

/// The per-box knobs the chunked claim path runs under (see [`execute_gap_chunked`]).
#[derive(Clone, Copy, Debug)]
pub struct ChunkParams {
    /// RAM + core admission envelope (≈ 0.75 × physical RAM, usable cores). Caps in-chunk concurrency.
    pub budget: BoxBudget,
    /// Target wall-time per chunk in seconds (the `ZEN_CHUNK_WALL_SEC` opt-in; the user's "~5 min").
    pub chunk_wall_sec: f64,
    /// Footprint assumed for a gap job carrying no [`ResourceHint`] (declare couldn't estimate it).
    pub fallback_hint: ResourceHint,
}

/// Chunked, resource-bounded gap execution — the DEFAULT path (`ZEN_CHUNK_WALL_SEC > 0`, and it
/// defaults to 300s), with [`execute_gap_claimed`] the serial `ZEN_CHUNK_WALL_SEC=0` opt-out. Two
/// differences, nothing else:
///
///  - **Claim granularity is a chunk, not a cell.** The gap (from [`reconcile`]) is packed by
///    [`BoxBudget::pack_chunks_lpt`] into units each estimated at ≈ `chunk_wall_sec` on this box, and
///    `claim_chunk(chunk_id)` takes ONE R2 lease per chunk. This kills the per-cell claim round-trip
///    that idled boxes behind the gap prefix (one `aws put-object` per sub-second cell). The packer is
///    longest-processing-time-first, so the heaviest cells land in the earliest chunks and no box
///    finishes the light work then idles on a heavy tail (validated in `zenfleet-sim`). Chunk
///    boundaries are still deterministic (LPT sorts stably on `(−cost, index)` — a pure function of the
///    manifest-order gap), so a chunk claim is exclusive; workers iterate chunk *indices* in a
///    per-worker order so they don't all contend on chunk 0.
///
///  - **In-chunk concurrency is bounded by [`BoxBudget::can_admit`].** A won chunk's cells run
///    concurrently as **fresh processes** (`handler` is the one-shot [`exec_command`], so the
///    `modes_full` per-cell memory bound holds — see the crate Known Bugs) with Σpeak_mem ≤
///    `budget.ram_budget_bytes` and Σthreads ≤ `budget.cores`. Set the RAM budget to ~75% of
///    physical RAM and peak stays under it; cores are never oversubscribed (no cache thrash).
///
/// **Idempotence + crash recovery are identical to the per-cell path.** Chunks are formed FROM the
/// reconciler's gap, so a cell already Done in `view` is never in a chunk — the existing per-cell
/// done-check still gates every cell. `flush(chunk_id, rows)` is called the moment a chunk finishes
/// (a durable per-chunk ledger sidecar), so a crash only loses the in-flight chunk: the next pass
/// re-derives the gap from the persisted rows and a re-claimed chunk runs only the still-missing
/// cells. Content-addressed blob puts make any re-run cell a no-op — no cell is lost or harmfully
/// double-run.
///
/// Spot preemption: a chunk claim simply ages out (TTL stale-reclaim, goal E) and another box takes
/// it — chunk 2 does not fast-release a chunk claim on SIGTERM (the per-cell path's nicety); that is
/// a follow-up. Correctness is unaffected.
#[allow(clippy::too_many_arguments)]
pub fn execute_gap_chunked<H, B, CC, F>(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    handler: H,
    store: &B,
    claim_chunk: CC,
    params: ChunkParams,
    mut flush: F,
    ctx: WorkerCtx<'_>,
) -> ExecOutcome
where
    H: Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError> + Sync,
    B: BlobStore + Sync,
    CC: Fn(&str) -> bool,
    F: FnMut(&str, &[LedgerRow]),
{
    let plan = reconcile(desired, view, policy);
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    // Gap DesiredJobs in deterministic manifest order (reconcile preserves `desired` order). NOT
    // shuffled — identical across workers so chunk boundaries (and thus claims) are exclusive.
    let gap: Vec<&DesiredJob> = plan
        .enqueue
        .iter()
        .filter_map(|id| by_id.get(id).copied())
        .collect();

    // Size the chunks: per-cell (cost_sec, peak_mem, threads), with the safe fallback for cells that
    // carried no declare-time hint.
    let costs: Vec<JobCost> = gap
        .iter()
        .map(|d| {
            let h = d.hint.unwrap_or(params.fallback_hint);
            JobCost {
                cost_sec: d.kind.estimate_cost_sec(h.peak_mem_bytes),
                peak_mem_bytes: h.peak_mem_bytes,
                threads: h.threads.max(1),
            }
        })
        .collect();
    let chunks = params.budget.pack_chunks_lpt(&costs, params.chunk_wall_sec);
    let chunk_ids: Vec<String> = chunks
        .iter()
        .map(|members| {
            let ids: Vec<JobId> = members.iter().map(|&m| gap[m].job_id()).collect();
            chunk_id(&ids)
        })
        .collect();

    // Per-worker iteration order over chunk indices (deterministic hash(chunk_id, worker)) so
    // late-joining boxes don't all start at chunk 0 — same rationale as the gap shuffle in
    // execute_gap_claimed, but over coarse chunks.
    let mut order: Vec<usize> = (0..chunks.len()).collect();
    order.sort_by_cached_key(|&ci| {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&chunk_ids[ci], &mut h);
        std::hash::Hash::hash(ctx.worker, &mut h);
        std::hash::Hasher::finish(&h)
    });

    let mut out = ExecOutcome {
        rows: Vec::new(),
        done: 0,
        failed: 0,
        poisoned: 0,
        skipped: 0,
    };
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

    for &ci in &order {
        let members = &chunks[ci];
        let cid = &chunk_ids[ci];
        if !claim_chunk(cid) {
            out.skipped += members.len(); // another worker owns this chunk
            continue;
        }
        // Won the chunk — run its cells concurrently under the budget (fresh processes), then turn
        // each (post-persist) result into a ledger row.
        let results = run_chunk_concurrent(members, &gap, &params, &handler, store);
        let mut chunk_rows: Vec<LedgerRow> = Vec::with_capacity(results.len());
        for (gi, res) in results {
            let d = gap[gi];
            match res {
                Ok(sha) => {
                    chunk_rows.push(make(d, JobStatus::Done, Some(sha), None));
                    out.done += 1;
                }
                Err(class) => {
                    chunk_rows.push(make(d, JobStatus::Failed, None, Some(class)));
                    out.failed += 1;
                }
            }
        }
        // Durable per-chunk write BEFORE the next claim: a crash now only re-does later chunks.
        flush(cid, &chunk_rows);
        out.rows.append(&mut chunk_rows);
    }

    // POISON rows the reconciler decided — identical to execute_gap_claimed; persisted in their own
    // sidecar so the "doomed work stops, recorded" signal (goals B/F) survives a crash too.
    let mut poison_rows: Vec<LedgerRow> = Vec::new();
    for id in &plan.poison {
        if let Some(d) = by_id.get(id) {
            let prev_err = view.get(id).and_then(|r| r.error_class);
            poison_rows.push(make(d, JobStatus::Poison, None, prev_err));
            out.poisoned += 1;
        }
    }
    if !poison_rows.is_empty() {
        flush("poison", &poison_rows);
        out.rows.append(&mut poison_rows);
    }
    out
}

/// Run a claimed chunk's cells concurrently as fresh processes, admitting under
/// [`BoxBudget::can_admit`] so Σpeak_mem ≤ the RAM budget and Σthreads ≤ cores at all times. Returns
/// each cell's outcome keyed by its index into `gap`: `Ok(sha)` once `handler` produced bytes AND
/// `store.put` persisted them (→ a Done row), `Err(class)` otherwise (→ a Failed row). Persisting
/// inside the worker thread overlaps a cell's upload with peers' encode/score.
///
/// Concurrency = a fixed pool of ≤ `min(chunk_len, cores)` scoped threads sharing a cursor +
/// running-footprint [`InFlight`]; a thread admits the cell at the cursor when it fits, else waits on
/// the condvar for a completion to free room. `can_admit` always admits when nothing is running, so a
/// single over-budget cell still runs (alone) — no deadlock, and the cursor advances in order.
fn run_chunk_concurrent<H, B>(
    members: &[usize],
    gap: &[&DesiredJob],
    params: &ChunkParams,
    handler: &H,
    store: &B,
) -> Vec<(usize, Result<Sha256Hex, ErrorClass>)>
where
    H: Fn(&DesiredJob) -> Result<Vec<u8>, HandlerError> + Sync,
    B: BlobStore + Sync,
{
    struct Shared {
        cursor: usize,
        running: InFlight,
        results: Vec<(usize, Result<Sha256Hex, ErrorClass>)>,
    }
    let shared = Mutex::new(Shared {
        cursor: 0,
        running: InFlight::default(),
        results: Vec::with_capacity(members.len()),
    });
    let cv = Condvar::new();
    let fallback = params.fallback_hint;
    let budget = params.budget;

    // Never more concurrent cells than cores (each uses ≥1 thread) or cells in the chunk; ≥1.
    let n_threads = (budget.cores.max(1) as usize).min(members.len()).max(1);

    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| {
                loop {
                    // Acquire the next admissible cell (admission-gated), or stop when none remain.
                    let (gi, mem, thr) = {
                        let mut g = shared.lock().unwrap_or_else(|p| p.into_inner());
                        loop {
                            if g.cursor >= members.len() {
                                return; // every cell started — this thread is done
                            }
                            let gi = members[g.cursor];
                            let h = gap[gi].hint.unwrap_or(fallback);
                            let (mem, thr) = (h.peak_mem_bytes, h.threads.max(1));
                            if budget.can_admit(&g.running, mem, thr) {
                                g.running.add(mem, thr);
                                g.cursor += 1;
                                break (gi, mem, thr);
                            }
                            // running full → wait for an in-flight cell to finish and free room.
                            g = cv.wait(g).unwrap_or_else(|p| p.into_inner());
                        }
                    };
                    // Encode/score (fresh process) + persist — OUTSIDE the lock so peers run too.
                    let res = handler(gap[gi]).and_then(|bytes| {
                        store.put(&bytes).map_err(|e| {
                            HandlerError::new(ErrorClass::UploadFail, format!("put: {e}"))
                        })
                    });
                    let mapped = res.map_err(|he| he.class);
                    {
                        let mut g = shared.lock().unwrap_or_else(|p| p.into_inner());
                        g.running.remove(mem, thr);
                        g.results.push((gi, mapped));
                    }
                    cv.notify_all(); // a slot freed → wake a waiter to admit the next cell
                }
            });
        }
    });

    shared
        .into_inner()
        .unwrap_or_else(|p| p.into_inner())
        .results
}

/// Production handler: shell out to an executor `program`. The job descriptor is written as JSON to
/// the program's stdin; its stdout is the output bytes (which get content-addressed). Exit 0 =
/// success; spawn failure → transient `WorkerLost`; non-zero exit → `EncoderPanic` (deterministic).
/// Any executor honoring this stdin-JSON → stdout-bytes contract plugs in (e.g. a future
/// `zenmetrics jobexec` subcommand).
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
            format!(
                "{program} exited {code}: {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ))
    }
}

/// A long-lived `program --serve` child for the persistent executor. One per worker PROCESS; since the
/// fleet runs one (long) pass per process, this child stays warm across all of a pass's jobs, so CUDA
/// init + GPU kernel compilation are paid ONCE rather than per job.
struct PersistentExec {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

static PERSISTENT: Mutex<Option<PersistentExec>> = Mutex::new(None);

fn persistent_io_lost(e: io::Error) -> HandlerError {
    HandlerError::new(ErrorClass::WorkerLost, format!("persistent exec io: {e}"))
}

fn spawn_serve(program: &str) -> Result<PersistentExec, HandlerError> {
    let mut child = Command::new(program)
        .arg("--serve")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // stderr inherited → the child's logs land in the worker's stderr (the fleet box log).
        .spawn()
        .map_err(|e| {
            HandlerError::new(
                ErrorClass::WorkerLost,
                format!("spawn {program} --serve: {e}"),
            )
        })?;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| HandlerError::new(ErrorClass::WorkerLost, "no child stdin"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| HandlerError::new(ErrorClass::WorkerLost, "no child stdout"))?;
    Ok(PersistentExec {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

impl PersistentExec {
    /// Send one length-framed job and read its length-framed response. `Ok` = output bytes (status 0);
    /// `Err(EncoderPanic)` = the child framed a per-job error/panic but is still alive; `Err(WorkerLost)`
    /// = an I/O error (the child is presumed dead → the caller drops it so the next job respawns one).
    fn run_job(&mut self, job_json: &[u8]) -> Result<Vec<u8>, HandlerError> {
        let len = u32::try_from(job_json.len())
            .map_err(|_| HandlerError::new(ErrorClass::Unknown, "job json too large"))?;
        self.stdin
            .write_all(&len.to_le_bytes())
            .map_err(persistent_io_lost)?;
        self.stdin.write_all(job_json).map_err(persistent_io_lost)?;
        self.stdin.flush().map_err(persistent_io_lost)?;

        let mut status = [0u8; 1];
        self.stdout
            .read_exact(&mut status)
            .map_err(persistent_io_lost)?;
        let mut lenb = [0u8; 4];
        self.stdout
            .read_exact(&mut lenb)
            .map_err(persistent_io_lost)?;
        let plen = u32::from_le_bytes(lenb) as usize;
        let mut payload = vec![0u8; plen];
        self.stdout
            .read_exact(&mut payload)
            .map_err(persistent_io_lost)?;
        if status[0] == 0 {
            Ok(payload)
        } else {
            // The child framed an error/panic for THIS job and stayed alive → deterministic failure.
            Err(HandlerError::new(
                ErrorClass::EncoderPanic,
                String::from_utf8_lossy(&payload).into_owned(),
            ))
        }
    }
}

/// Persistent variant of [`exec_command`]: keep ONE warm `program --serve` child for this worker
/// process and stream length-framed jobs to it, so CUDA init + kernel compilation are paid once rather
/// than per job (the fix for ~20s/job cold-process overhead on GPU metric fleets). On child death the
/// global handle is dropped and the next call respawns; a per-job error/panic (child still alive) is a
/// deterministic failure that does NOT kill the warm child.
pub fn exec_command_persistent(program: &str, job: &DesiredJob) -> Result<Vec<u8>, HandlerError> {
    let job_json = serde_json::to_vec(job)
        .map_err(|e| HandlerError::new(ErrorClass::Unknown, format!("serialize job: {e}")))?;
    let mut guard = PERSISTENT.lock().unwrap_or_else(|p| p.into_inner());
    if guard.is_none() {
        *guard = Some(spawn_serve(program)?);
    }
    let res = guard.as_mut().expect("just set above").run_job(&job_json);
    if let Err(e) = &res
        && matches!(e.class, ErrorClass::WorkerLost)
    {
        // Child presumed dead → drop it so the next job respawns a fresh warm child.
        if let Some(mut pe) = guard.take() {
            let _ = pe.child.kill();
            let _ = pe.child.wait();
        }
    }
    res
}

/// Choose the executor handler: the warm persistent child (one `--serve` process reused across jobs)
/// when `persistent`, else the original one-process-per-job [`exec_command`]. Persistence is opt-in
/// (via `ZEN_PERSISTENT_EXEC`) so non-GPU/basement tiers keep the simple one-shot path.
fn dispatch_exec(
    persistent: bool,
    program: &str,
    job: &DesiredJob,
) -> Result<Vec<u8>, HandlerError> {
    if persistent {
        exec_command_persistent(program, job)
    } else {
        exec_command(program, job)
    }
}

/// Configuration for one worker pass (the runnable `zenfleet-worker` binary parses CLI args into this).
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
    /// If set (requires `r2`), the R2 key of a `RunControl` object checked before pulling work —
    /// when paused/draining this pass claims nothing (goal C: pause/resume/drain).
    pub control_key: Option<String>,
    /// Executor program (stdin-JSON → stdout-bytes contract).
    pub exec: String,
    pub worker: String,
    pub provider: String,
    pub now: u64,
    pub max_attempts: u32,
    /// Resource classes this worker serves (goal H capability-routing). Empty = serve everything.
    /// A job is only claimed/run if its `JobKind::profile().class` is in this set.
    pub served: Vec<ResourceClass>,
    /// **Opt-in ~5-minute chunked claiming** (from env `ZEN_CHUNK_WALL_SEC`; the binary parses it).
    /// `0.0` (default/unset) = **disabled** → byte-identical to the per-cell claim path. When `> 0`,
    /// `run()` packs the gap into chunks each ≈ this many seconds on this box and claims/executes a
    /// chunk at a time under a `BoxBudget(0.75 × RAM, cores)` admission cap (see
    /// [`execute_gap_chunked`]). Activate only after a real-box smoke run.
    pub chunk_wall_sec: f64,
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

// ──────────────────────────── garbage collection (goal G) ────────────────────────────

/// Delete an R2 object via `s5cmd rm`.
fn s5cmd_rm(endpoint: &str, uri: &str) -> Result<(), String> {
    let st = Command::new("s5cmd")
        .arg("--endpoint-url")
        .arg(endpoint)
        .arg("rm")
        .arg(uri)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("s5cmd spawn: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("s5cmd rm {uri} exit {:?}", st.code()))
    }
}

/// Verify a Tower-mirror copy is present and byte-identical (goal G: "Tower-mirror-verify before any
/// non-regenerable delete"). `mirror_dir/<sha>` is read and its hash compared to `sha`.
pub fn verify_mirror(sha: &Sha256Hex, mirror_dir: &Path) -> bool {
    match std::fs::read(mirror_dir.join(sha.as_str())) {
        Ok(bytes) => sha256(&bytes).as_str() == sha.as_str(),
        Err(_) => false,
    }
}

/// GC execution config. Blobs live at `{blobs_base_uri}/<sha>`; tombstones (if set) at
/// `{tombstones_base_uri}/<sha>`. `execute=false` is a dry-run (decide + report, delete nothing).
pub struct GcExecCfg<'a> {
    pub endpoint: &'a str,
    pub blobs_base_uri: &'a str,
    pub tombstones_base_uri: Option<&'a str>,
    pub cheap_cap_bytes: u64,
    pub now: u64,
    pub execute: bool,
}

/// Outcome of a GC pass.
#[derive(Debug, Default, serde::Serialize)]
pub struct GcReport {
    pub kept: usize,
    /// Cheap-regenerable blobs evicted (or, in dry-run, that *would* be evicted) by the LRU cap.
    pub lru_evicted: Vec<String>,
    /// Unreferenced irreplaceable blobs — NEVER auto-deleted; surfaced for a human pin/archive call.
    pub refused: Vec<String>,
    pub freed_bytes: u64,
    pub tombstones_written: usize,
    pub errors: Vec<String>,
}

/// Execute the safe-eviction half of GC (goal G): evict the unreferenced cheap-regenerable LRU tail
/// over `cheap_cap_bytes` (lossless — rebuildable), writing a tombstone before each delete; and
/// *refuse* to touch unreferenced irreplaceable blobs (surface them instead). Referenced/pinned blobs
/// are never considered. Expensive-regenerable is left for an explicit under-pressure pass. Pure
/// decision via [`gc_plan`]/[`lru_cap_evict`]; this only performs the R2 deletes + tombstones.
pub fn gc_execute(
    index: &[BlobIndexEntry],
    referenced: &HashSet<Sha256Hex>,
    roots: &HashSet<Sha256Hex>,
    cfg: &GcExecCfg<'_>,
) -> GcReport {
    let plan = gc_plan(index, referenced, roots);
    let lru = lru_cap_evict(index, referenced, roots, cfg.cheap_cap_bytes);
    let size_of: HashMap<&Sha256Hex, u64> = index.iter().map(|e| (&e.sha, e.size)).collect();
    let mut report = GcReport {
        kept: plan.keep.len(),
        refused: plan.refuse_surface.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    };
    let base = cfg.blobs_base_uri.trim_end_matches('/');
    for sha in &lru {
        let size = size_of.get(sha).copied().unwrap_or(0);
        if cfg.execute {
            // Tombstone first (cheap-regenerable is mirror_verified=true: a cache miss is a lossless
            // recompute, so no Tower copy is required). Then delete the blob.
            if let Some(tb) = cfg.tombstones_base_uri {
                let t = Tombstone {
                    sha: sha.clone(),
                    size,
                    regenerability: Regenerability::CheapRegenerable,
                    reason: "lru_evict".to_string(),
                    deleted_at: cfg.now,
                    mirror_verified: true,
                };
                let uri = format!("{}/{}", tb.trim_end_matches('/'), sha.as_str());
                if zenfleet_ledger::write_bytes_uri(
                    &uri,
                    &serde_json::to_vec(&t).unwrap_or_default(),
                    Some(cfg.endpoint),
                )
                .is_ok()
                {
                    report.tombstones_written += 1;
                }
            }
            match s5cmd_rm(cfg.endpoint, &format!("{base}/{}", sha.as_str())) {
                Ok(()) => {
                    report.lru_evicted.push(sha.to_string());
                    report.freed_bytes += size;
                }
                Err(e) => report.errors.push(e),
            }
        } else {
            // dry-run: report what *would* be freed.
            report.lru_evicted.push(sha.to_string());
            report.freed_bytes += size;
        }
    }
    report
}

/// Parse `MemTotal:` (in kB) out of a `/proc/meminfo` body → bytes. `None` if the field is absent or
/// unparseable (e.g. a non-Linux host). Split out so the parse is unit-testable without the file.
fn parse_meminfo_total(meminfo: &str) -> Option<u64> {
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// Total physical RAM in bytes from `/proc/meminfo`. `None` if unreadable.
fn read_meminfo_total_bytes() -> Option<u64> {
    parse_meminfo_total(&std::fs::read_to_string("/proc/meminfo").ok()?)
}

/// This box's admission budget for the chunked path: **RAM budget = 75 % of physical RAM** (leaves
/// headroom for the OS, page cache, GPU readback, and the estimate's slop — see [`BoxBudget`]) and
/// **cores = usable parallelism** (`available_parallelism` honors cgroup/cpuset affinity, which the
/// fleet onstart pins; RAM is bounded separately by `can_admit`, so we do NOT also shrink cores by
/// RAM the way a blind N-per-core launcher would). Conservative fallbacks (2 GiB / 1 core) if either
/// probe fails — never panics.
fn host_box_budget() -> BoxBudget {
    let total_ram = read_meminfo_total_bytes().unwrap_or(2 << 30); // 2 GiB if /proc/meminfo unreadable
    let ram_budget = (((total_ram as f64) * 0.75) as u64).max(1);
    let cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1);
    BoxBudget::new(ram_budget, cores)
}

/// Per-chunk wall-time target (seconds) when the box is left to its default: the
/// resource-aware concurrent chunked path packs ≈5-minute claim units.
pub const DEFAULT_CHUNK_WALL_SEC: f64 = 300.0;

/// Resolve the chunk wall-time target from the `ZEN_CHUNK_WALL_SEC` env value.
///
/// The concurrent, resource-aware chunked path is the **default** — unset or an
/// unparseable value yields [`DEFAULT_CHUNK_WALL_SEC`]. Serial per-cell execution
/// is **opt-in**: set `ZEN_CHUNK_WALL_SEC=0` (any explicit ≤0 value) to get it.
/// A positive value sets a custom chunk target.
pub fn resolve_chunk_wall_sec(env: Option<&str>) -> f64 {
    match env.map(|v| v.trim().parse::<f64>()) {
        // Unset or garbage → the concurrent default (never accidentally serial).
        None | Some(Err(_)) => DEFAULT_CHUNK_WALL_SEC,
        // Explicit value: 0 (or negative) opts into the serial per-cell path.
        Some(Ok(n)) => n.max(0.0),
    }
}

/// Derive a per-chunk ledger sidecar URI from the pass's `ledger_out` by inserting `chunk-<id8>`
/// before the extension: `…/pass.parquet` → `…/pass.chunk-ab12cd34.parquet`. Per-chunk durable
/// writes make a completed chunk's progress survive a crash; the next pass folds the sidecar into the
/// view and reconcile skips the now-Done cells (crash recovery at chunk granularity). Pure string op,
/// so it works for both local paths and `s3://…` URIs.
fn chunk_ledger_uri(ledger_out: &str, chunk_id: &str) -> String {
    let tag = chunk_id.strip_prefix("chunk-").unwrap_or(chunk_id);
    let tag8: String = tag.chars().take(8).collect();
    match ledger_out.rsplit_once('.') {
        // only treat the trailing dot as an extension if it's in the filename, not a dir name
        Some((stem, ext)) if !ext.contains('/') => format!("{stem}.chunk-{tag8}.{ext}"),
        _ => format!("{ledger_out}.chunk-{tag8}.parquet"),
    }
}

/// The DEFAULT chunked claim path (`cfg.chunk_wall_sec > 0`; 300s unless `ZEN_CHUNK_WALL_SEC=0` opts
/// into serial), split out of [`run`]. Packs the gap into ≈`chunk_wall_sec` work-stealing chunks and runs a
/// chunk at a time under a `BoxBudget(0.75 × RAM, cores)` admission cap, executing each cell as a
/// fresh process. Persists a durable per-chunk ledger sidecar via the `flush` callback, so unlike the
/// per-cell path it does NOT write a single end-of-pass sidecar — the chunk sidecars ARE the output.
fn run_chunked(
    cfg: &WorkerConfig,
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    ctx: WorkerCtx<'_>,
    endpoint: Option<&str>,
) -> Result<ExecOutcome, WorkerRunError> {
    let params = ChunkParams {
        budget: host_box_budget(),
        chunk_wall_sec: cfg.chunk_wall_sec,
        // No declare-time hint → assume a modest 512 MB / 1-thread footprint (admission stays safe).
        fallback_hint: ResourceHint {
            peak_mem_bytes: 512 << 20,
            threads: 1,
        },
    };
    eprintln!(
        "zenfleet-worker: resource-aware concurrent mode (LPT + can_admit, chunk target {:.0}s) \
         — budget {:.1} GiB / {} cores. Set ZEN_CHUNK_WALL_SEC=0 for the serial per-cell path.",
        params.chunk_wall_sec,
        params.budget.ram_budget_bytes as f64 / (1u64 << 30) as f64,
        params.budget.cores
    );
    // Fresh process per cell (NOT the persistent warm child) — keeps the modes_full per-cell memory
    // bound and lets cells run truly concurrently under the budget.
    let handler = |job: &DesiredJob| exec_command(&cfg.exec, job);
    let ledger_out_uri = cfg.ledger_out.to_string_lossy().into_owned();
    let mut flush = |chunk_id: &str, rows: &[LedgerRow]| {
        if rows.is_empty() {
            return;
        }
        let uri = chunk_ledger_uri(&ledger_out_uri, chunk_id);
        if let Err(e) = zenfleet_ledger::write_ledger_uri(&uri, rows, endpoint) {
            eprintln!("zenfleet-worker: chunk {chunk_id} ledger write to {uri} failed: {e}");
        }
    };
    let out = match (&cfg.r2, &cfg.claims) {
        (Some(t), Some(cc)) => {
            let store = R2BlobStore::new(t.endpoint.clone(), t.bucket.clone(), t.prefix.clone());
            execute_gap_chunked(
                desired,
                view,
                policy,
                handler,
                &store,
                |cid| {
                    // One R2 lease per chunk (spec-execution off for chunks; TTL reclaim covers it).
                    claim_or_steal_r2_key(
                        &t.endpoint,
                        &cc.bucket,
                        &cc.prefix,
                        cid,
                        cfg.now,
                        cc.ttl_secs,
                        None,
                        &cfg.worker,
                    )
                },
                params,
                &mut flush,
                ctx,
            )
        }
        (Some(t), None) => {
            // R2 blobs but single-worker (no concurrent claiming) → win every chunk.
            let store = R2BlobStore::new(t.endpoint.clone(), t.bucket.clone(), t.prefix.clone());
            execute_gap_chunked(
                desired,
                view,
                policy,
                handler,
                &store,
                |_| true,
                params,
                &mut flush,
                ctx,
            )
        }
        (None, _) => {
            let store = LocalBlobStore::new(cfg.blobs.clone())
                .map_err(|e| WorkerRunError::Io(e.to_string()))?;
            execute_gap_chunked(
                desired,
                view,
                policy,
                handler,
                &store,
                |_| true,
                params,
                &mut flush,
                ctx,
            )
        }
    };
    Ok(out)
}

/// One worker pass: load the manifest + existing ledger → reconcile the gap → execute each job via
/// `exec` → content-address outputs → write the resulting rows. Returns the outcome. Deterministic
/// given `cfg.now` (the binary supplies the wall clock; the library stays clock-free + testable).
pub fn run(cfg: &WorkerConfig) -> Result<ExecOutcome, WorkerRunError> {
    let bytes = std::fs::read(&cfg.manifest).map_err(|e| {
        WorkerRunError::Io(format!("read manifest {}: {e}", cfg.manifest.display()))
    })?;
    let mut desired: Vec<DesiredJob> =
        serde_json::from_slice(&bytes).map_err(|e| WorkerRunError::Manifest(e.to_string()))?;
    // Capability routing (goal H): drop jobs this worker's hardware doesn't serve, so an ARM/CPU/GPU
    // box pulls only its class off the shared queue. Empty `served` = general worker (keep all).
    if !cfg.served.is_empty() {
        desired.retain(|d| worker_serves(&cfg.served, &d.kind));
    }

    // Ledger paths may be local or s3:// — the R2 endpoint (if any) comes from the blob target.
    let endpoint = cfg.r2.as_ref().map(|t| t.endpoint.as_str());
    let mut view = LedgerView::new();
    for p in &cfg.ledger_in {
        let uri = p.to_string_lossy();
        for row in zenfleet_ledger::read_ledger_uri(uri.as_ref(), endpoint)
            .map_err(|e| WorkerRunError::Ledger(e.to_string()))?
        {
            view.apply(row);
        }
    }

    // Run control (goal C): if the run is paused/draining, pull no new work this pass. Fail-open —
    // an absent control object reads as RUNNING. The ledger is untouched, so resuming continues
    // exactly where it left off ("without losing state").
    if let (Some(t), Some(key)) = (&cfg.r2, &cfg.control_key) {
        let ctl = fetch_control_r2(&t.endpoint, &t.bucket, key);
        if ctl.claims_blocked() {
            eprintln!(
                "zenfleet-worker: run control = {} — pulling no new work this pass",
                if ctl.paused { "PAUSED" } else { "DRAINING" }
            );
            return Ok(ExecOutcome {
                rows: Vec::new(),
                done: 0,
                failed: 0,
                poisoned: 0,
                skipped: 0,
            });
        }
    }

    let policy = RetryPolicy {
        max_attempts: cfg.max_attempts,
    };
    let ctx = WorkerCtx {
        worker: &cfg.worker,
        provider: &cfg.provider,
        now: cfg.now,
    };

    // DEFAULT: the resource-aware concurrent chunked path — LPT-packed ≈5-min chunks run with
    // `can_admit`-bounded concurrency, so the box saturates its cores/GPU within its RAM envelope
    // instead of executing one cell at a time. Serial per-cell execution is now OPT-IN via
    // `ZEN_CHUNK_WALL_SEC=0` (`chunk_wall_sec == 0.0`) — kept only as an escape hatch for debugging or
    // memory-pathological single-cell runs. The chunked path persists a durable sidecar per chunk, so
    // it returns here without the single end-of-pass ledger write at the bottom.
    if cfg.chunk_wall_sec > 0.0 {
        return run_chunked(cfg, &desired, &view, policy, ctx, endpoint);
    }

    // Pick the blob store: R2 if configured, else local FS. execute_gap is generic over the store,
    // so each arm monomorphizes against the concrete type.
    // Persistent warm executor (one `--serve` child reused across this pass's jobs) when enabled —
    // amortizes GPU init + kernel compilation; opt-in so non-GPU/basement tiers keep one-shot exec.
    let persistent = std::env::var("ZEN_PERSISTENT_EXEC")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
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
                        |job| dispatch_exec(persistent, &cfg.exec, job),
                        &store,
                        |id| {
                            let won = claim_or_steal_r2(
                                &t.endpoint,
                                &cc.bucket,
                                &cc.prefix,
                                id,
                                cfg.now,
                                cc.ttl_secs,
                                cc.spec_threshold_secs,
                                &cfg.worker,
                            );
                            if won && let Ok(mut g) = inflight.lock() {
                                *g = Some(id.clone());
                            }
                            won
                        },
                        ctx,
                    )
                }
                None => execute_gap(
                    &desired,
                    &view,
                    policy,
                    |job| dispatch_exec(persistent, &cfg.exec, job),
                    &store,
                    ctx,
                ),
            }
        }
        None => {
            let store = LocalBlobStore::new(cfg.blobs.clone())
                .map_err(|e| WorkerRunError::Io(e.to_string()))?;
            execute_gap(
                &desired,
                &view,
                policy,
                |job| dispatch_exec(persistent, &cfg.exec, job),
                &store,
                ctx,
            )
        }
    };
    let out_uri = cfg.ledger_out.to_string_lossy();
    zenfleet_ledger::write_ledger_uri(out_uri.as_ref(), &out.rows, endpoint)
        .map_err(|e| WorkerRunError::Ledger(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use zenfleet_core::{CellId, JobKind};

    static N: AtomicU64 = AtomicU64::new(0);
    fn tmp() -> PathBuf {
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("zenjobworker_{}_{}", std::process::id(), n))
    }
    fn desired(metric: &str, enc: &[u8]) -> DesiredJob {
        DesiredJob {
            kind: JobKind::Metric {
                metric: metric.into(),
            },
            inputs: vec![sha256(enc)],
            cell: CellId {
                image_path: "x".into(),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            hint: None,
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
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 100,
            },
        );
        assert_eq!(out.done, 2);
        assert_eq!(out.rows.len(), 2);
        for r in &out.rows {
            assert_eq!(r.status, JobStatus::Done);
            let sha = r.output_sha.clone().unwrap();
            assert!(
                store.exists(&sha),
                "output blob is written content-addressed"
            );
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
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 100,
            },
        );
        let view = LedgerView::from_rows(out1.rows);
        let out2 = execute_gap(
            &d,
            &view,
            RetryPolicy::default(),
            |_| panic!("handler must NOT run for an already-done job"),
            &store,
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 200,
            },
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
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 100,
            },
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
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 200,
            },
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
        assert_eq!(
            err.class,
            ErrorClass::WorkerLost,
            "infra failure → retryable, not poison"
        );
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
            control_key: None,
            served: vec![],
            exec: "cat".into(),
            worker: "w1".into(),
            provider: "local".into(),
            now: 100,
            max_attempts: 3,
            chunk_wall_sec: 0.0, // per-cell path (default-off)
        };
        let out = run(&cfg).unwrap();
        assert_eq!(out.done, 2);
        let rows = zenfleet_ledger::read_ledger(&cfg.ledger_out).unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter()
                .all(|r| r.status == JobStatus::Done && r.output_sha.is_some())
        );

        // second pass folds in the just-written ledger → gap empty → executor never invoked
        let cfg2 = WorkerConfig {
            ledger_in: vec![cfg.ledger_out.clone()],
            ledger_out: dir.join("out2.parquet"),
            exec: "false".into(), // would fail if called; it must not be
            ..cfg.clone()
        };
        let out2 = run(&cfg2).unwrap();
        assert_eq!(
            out2.done, 0,
            "all jobs already DONE → converged, nothing re-run"
        );
        assert!(out2.rows.is_empty());
    }

    #[test]
    fn r2_key_derivation() {
        let s = R2BlobStore::new(
            "https://acct.r2.cloudflarestorage.com".into(),
            "zen-tuning-ephemeral".into(),
            "blobs".into(),
        );
        let sha = sha256(b"hi");
        assert_eq!(
            s.key(&sha),
            format!("s3://zen-tuning-ephemeral/blobs/{sha}")
        );
        // leading/trailing slashes in the prefix don't double up
        let s2 = R2BlobStore::new("e".into(), "b".into(), "/blobs/".into());
        assert_eq!(s2.key(&sha), format!("s3://b/blobs/{sha}"));
    }

    #[test]
    fn concurrent_chunked_path_is_the_default_serial_is_opt_in() {
        // Unset or garbage → the resource-aware concurrent chunked path.
        assert_eq!(resolve_chunk_wall_sec(None), DEFAULT_CHUNK_WALL_SEC);
        assert_eq!(resolve_chunk_wall_sec(Some("  ")), DEFAULT_CHUNK_WALL_SEC);
        assert_eq!(resolve_chunk_wall_sec(Some("not-a-number")), DEFAULT_CHUNK_WALL_SEC);
        // A positive value sets a custom chunk target.
        assert_eq!(resolve_chunk_wall_sec(Some("120")), 120.0);
        // Explicit 0 (or negative) is the ONLY way to get the serial per-cell path.
        assert_eq!(
            resolve_chunk_wall_sec(Some("0")),
            0.0,
            "serial is opt-in via an explicit ZEN_CHUNK_WALL_SEC=0"
        );
        assert_eq!(resolve_chunk_wall_sec(Some("-5")), 0.0);
        // The dispatch (run) selects chunked iff chunk_wall_sec > 0.0, so the
        // default (300) is chunked and only an explicit 0 falls through to serial.
        assert!(DEFAULT_CHUNK_WALL_SEC > 0.0);
    }

    #[test]
    fn claim_staleness_check() {
        assert!(claim_is_stale(1000, 0, 10), "ancient claim is stealable");
        assert!(
            claim_is_stale(1000, 990, 10),
            "exactly ttl old is stealable"
        );
        assert!(
            !claim_is_stale(1000, 995, 10),
            "fresh claim (5s < 10s ttl) is NOT stealable"
        );
        assert!(
            !claim_is_stale(5, 0, 10),
            "clock skew / before ttl elapsed: not stealable"
        );
    }

    // ──────────────────── chunked claim path (ZEN_CHUNK_WALL_SEC) ────────────────────

    /// A distinct, cheap encode cell (4 MB / 1 thread → packs many per chunk). Distinct `inputs`
    /// give each a distinct content-addressed `JobId`.
    fn cheap_cell(i: u8) -> DesiredJob {
        DesiredJob {
            kind: JobKind::Encode {
                codec: "zenjpeg".into(),
                q: 80,
                knobs: "{}".into(),
            },
            inputs: vec![sha256(&[i])],
            cell: CellId {
                image_path: format!("img{i}.png"),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            hint: Some(ResourceHint {
                peak_mem_bytes: 4 << 20,
                threads: 1,
            }),
        }
    }

    fn test_params() -> ChunkParams {
        ChunkParams {
            budget: BoxBudget::new(24 << 30, 2), // 24 GiB, 2 cores
            chunk_wall_sec: 2.0,
            fallback_hint: ResourceHint {
                peak_mem_bytes: 512 << 20,
                threads: 1,
            },
        }
    }

    #[test]
    fn chunk_id_is_stable_and_distinct() {
        let a = sha256(b"a");
        let b = sha256(b"b");
        let mk = |s| {
            JobId::of(
                &JobKind::Metric {
                    metric: "ssim2".into(),
                },
                std::slice::from_ref(s),
            )
        };
        let (j1, j2) = (mk(&a), mk(&b));
        // Deterministic in membership + order.
        assert_eq!(
            chunk_id(&[j1.clone(), j2.clone()]),
            chunk_id(&[j1.clone(), j2.clone()])
        );
        // Different membership → different id; and namespaced away from bare-sha cell claims.
        assert_ne!(
            chunk_id(std::slice::from_ref(&j1)),
            chunk_id(&[j1.clone(), j2])
        );
        assert!(chunk_id(std::slice::from_ref(&j1)).starts_with("chunk-"));
    }

    #[test]
    fn parse_meminfo_total_reads_memtotal_kb() {
        let s = "MemTotal:       65792840 kB\nMemFree:          123 kB\n";
        assert_eq!(parse_meminfo_total(s), Some(65792840u64 * 1024));
        assert_eq!(parse_meminfo_total("no memtotal here\n"), None);
    }

    #[test]
    fn chunk_ledger_uri_inserts_tag_before_extension() {
        assert_eq!(
            chunk_ledger_uri("run/pass.parquet", "chunk-ab12cd34ef"),
            "run/pass.chunk-ab12cd34.parquet"
        );
        assert_eq!(
            chunk_ledger_uri("s3://b/run/pass.parquet", "chunk-deadbeef00"),
            "s3://b/run/pass.chunk-deadbeef.parquet"
        );
        // a dot only in a directory name (not the filename) → append a sidecar name.
        assert_eq!(
            chunk_ledger_uri("s3://b/run.v2/pass", "chunk-feed0000"),
            "s3://b/run.v2/pass.chunk-feed0000.parquet"
        );
    }

    #[test]
    fn host_box_budget_is_sane() {
        // Reads /proc/meminfo + available_parallelism on this host; both must yield ≥1, never panic.
        let b = host_box_budget();
        assert!(b.cores >= 1);
        assert!(b.ram_budget_bytes >= 1);
    }

    #[test]
    fn chunked_path_packs_runs_and_is_idempotent() {
        // 12 cheap cells, 2.0s chunks on a 2-core budget → packs several cells per chunk (FEWER
        // claims than cells), runs every cell exactly once, and a second pass over the resulting
        // ledger re-runs nothing — the chunked counterpart of converges_on_second_pass.
        let store = LocalBlobStore::new(tmp()).unwrap();
        let cells: Vec<DesiredJob> = (0..12u8).map(cheap_cell).collect();
        let ran: Arc<Mutex<HashSet<JobId>>> = Arc::new(Mutex::new(HashSet::new()));
        let mut chunk_rows_seen: Vec<usize> = Vec::new();

        let out1 = {
            let ran = ran.clone();
            execute_gap_chunked(
                &cells,
                &LedgerView::new(),
                RetryPolicy::default(),
                move |job| {
                    ran.lock().unwrap().insert(job.job_id());
                    Ok(format!("enc:{}", job.job_id().as_str()).into_bytes())
                },
                &store,
                |_| true, // single worker wins every chunk
                test_params(),
                |_cid, rows| chunk_rows_seen.push(rows.len()),
                WorkerCtx {
                    worker: "w1",
                    provider: "local",
                    now: 100,
                },
            )
        };
        assert_eq!(out1.done, 12, "every gap cell completes once");
        assert_eq!(
            ran.lock().unwrap().len(),
            12,
            "handler ran each cell exactly once"
        );
        assert_eq!(out1.rows.len(), 12);
        // Chunking produced fewer claim units than cells, and the flushed rows cover the whole gap.
        assert!(
            chunk_rows_seen.len() < 12,
            "chunking must produce fewer claim units than cells (got {} chunks)",
            chunk_rows_seen.len()
        );
        assert_eq!(chunk_rows_seen.iter().sum::<usize>(), 12);
        assert!(
            chunk_rows_seen.iter().any(|&n| n >= 2),
            "at least one chunk packed ≥2 cells"
        );

        // Pass 2 over the just-written ledger → gap empty → handler MUST NOT run (idempotent).
        let view = LedgerView::from_rows(out1.rows);
        let out2 = execute_gap_chunked(
            &cells,
            &view,
            RetryPolicy::default(),
            |_| panic!("handler must NOT run for an already-done cell"),
            &store,
            |_| true,
            test_params(),
            |_cid, _rows| panic!("nothing to flush on a converged pass"),
            WorkerCtx {
                worker: "w1",
                provider: "local",
                now: 200,
            },
        );
        assert_eq!(out2.done, 0);
        assert!(out2.rows.is_empty(), "converged — nothing left in the gap");
    }

    #[test]
    fn chunked_re_claim_after_crash_skips_already_done_cells() {
        // Simulate a crash mid-pass: only SOME cells' rows reached the ledger. A re-claimed chunk
        // must run ONLY the still-missing cells (the per-cell done-check still gates inside a chunk),
        // never re-running the persisted ones — "a re-claimed chunk skips already-completed cells".
        let store = LocalBlobStore::new(tmp()).unwrap();
        let cells: Vec<DesiredJob> = (0..8u8).map(cheap_cell).collect();

        // Pre-seed the view with Done rows for the even-indexed cells (as if a prior pass persisted
        // those chunk sidecars before crashing).
        let done_rows: Vec<LedgerRow> = cells
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, d)| LedgerRow {
                job_id: d.job_id(),
                kind: d.kind.clone(),
                cell: d.cell.clone(),
                output_sha: Some(sha256(b"prior")),
                status: JobStatus::Done,
                error_class: None,
                attempts: 1,
                ts: 50,
                worker: "crashed".into(),
                provider: "local".into(),
            })
            .collect();
        let view = LedgerView::from_rows(done_rows);
        let expected_missing: HashSet<JobId> = cells
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, d)| d.job_id())
            .collect();

        let ran: Arc<Mutex<HashSet<JobId>>> = Arc::new(Mutex::new(HashSet::new()));
        let out = {
            let ran = ran.clone();
            execute_gap_chunked(
                &cells,
                &view,
                RetryPolicy::default(),
                move |job| {
                    ran.lock().unwrap().insert(job.job_id());
                    Ok(b"enc".to_vec())
                },
                &store,
                |_| true,
                test_params(),
                |_cid, _rows| {},
                WorkerCtx {
                    worker: "w1",
                    provider: "local",
                    now: 300,
                },
            )
        };
        assert_eq!(out.done, 4, "only the 4 still-missing cells run");
        assert_eq!(
            *ran.lock().unwrap(),
            expected_missing,
            "exactly the not-yet-Done cells run; the persisted ones are skipped"
        );
    }

    #[test]
    fn run_pass_chunked_is_end_to_end_and_converges() {
        // Full run() chunked path with local blobs: pack → run cells as FRESH `cat` processes →
        // write a DURABLE per-chunk ledger sidecar each. A second (also chunked) pass that folds
        // those sidecars in re-runs nothing (exec="false" would Fail-row if any cell ran) — proves
        // per-chunk persistence + crash-recovery skip end to end.
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let manifest = dir.join("jobs.json");
        let cells: Vec<DesiredJob> = (0..6u8).map(cheap_cell).collect();
        std::fs::write(&manifest, serde_json::to_vec(&cells).unwrap()).unwrap();
        let cfg = WorkerConfig {
            manifest,
            ledger_in: vec![],
            ledger_out: dir.join("p1.parquet"),
            blobs: dir.join("blobs"),
            r2: None,
            claims: None,
            control_key: None,
            served: vec![],
            exec: "cat".into(),
            worker: "w1".into(),
            provider: "local".into(),
            now: 100,
            max_attempts: 3,
            chunk_wall_sec: 4.0, // CHUNKED ON
        };
        let out = run(&cfg).unwrap();
        assert_eq!(out.done, 6, "all cells encoded via fresh `cat` processes");

        // The chunked pass wrote per-chunk sidecars (NOT a single p1.parquet). Collect them.
        let sidecars: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("p1.chunk-") && n.ends_with(".parquet"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(
            !sidecars.is_empty(),
            "chunked path writes ≥1 durable per-chunk sidecar"
        );

        // Pass 2 folds the sidecars in → gap empty → exec must never run (would fail with `false`).
        let cfg2 = WorkerConfig {
            ledger_in: sidecars,
            ledger_out: dir.join("p2.parquet"),
            exec: "false".into(),
            ..cfg.clone()
        };
        let out2 = run(&cfg2).unwrap();
        assert_eq!(
            out2.done, 0,
            "all cells already Done via chunk sidecars → converged"
        );
        assert!(out2.rows.is_empty());
    }
}
