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

mod s3io;

use zenfleet_core::epoch::{self, ClaimMode, EpochShardCfg, Handicaps, Roster, ShardDecision};
use zenfleet_core::{
    BlobIndexEntry, BoxBudget, DesiredJob, ErrorClass, InFlight, JobCost, JobId, JobKind,
    JobStatus, LedgerRow, LedgerView, Regenerability, ResourceClass, ResourceHint, RetryPolicy,
    RunControl, Sha256Hex, Tombstone, gc_plan, lru_cap_evict, reconcile, sha256, worker_serves,
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

/// Content-addressed blob store over R2. Every operation is in-process via
/// [`crate::s3io`] (`object_store`) — the `s5cmd`/`aws` CLI spawns this type
/// used to shell out to are gone, so `s5cmd` is no longer required on PATH.
/// Still needs `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in the environment
/// at runtime (the launcher maps the `R2_*` creds to `AWS_*`). The
/// `blobs/<sha>` layout was verified live against R2.
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
}

impl R2BlobStore {
    fn obj_key(&self, sha: &Sha256Hex) -> String {
        format!("{}/{}", self.target.prefix.trim_matches('/'), sha.as_str())
    }
}

impl BlobStore for R2BlobStore {
    fn put(&self, bytes: &[u8]) -> io::Result<Sha256Hex> {
        let sha = sha256(bytes);
        // In-process conditional PUT: one round-trip, no `s5cmd cp`/`ls` spawn, no temp
        // file. `AlreadyExists` = content-addressed dedup hit (both are success).
        crate::s3io::put_create(
            &self.target.endpoint,
            &self.target.bucket,
            &self.obj_key(&sha),
            bytes,
        )
        .map_err(io::Error::other)?;
        Ok(sha)
    }

    fn exists(&self, sha: &Sha256Hex) -> bool {
        crate::s3io::head_exists(
            &self.target.endpoint,
            &self.target.bucket,
            &self.obj_key(sha),
        )
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
    // In-process conditional PUT (If-None-Match:*). Won iff WE created it. Replaces the
    // per-claim `aws s3api put-object` spawn (aws-cli's ~1.5s Python startup, at claim
    // rate, was the box's top CPU cost). Any error → not won (safe: no double-execute).
    crate::s3io::put_create(endpoint, bucket, &key, b"").unwrap_or(false)
}

/// Release (delete) a claim so the job requeues immediately — used on spot preemption (goal F:
/// "spot reclaim is a non-event") instead of waiting out the claim TTL. Best-effort: a failed delete
/// just falls back to the slower TTL-based stale-reclaim (goal E), so correctness never depends on it.
pub fn release_claim_r2(endpoint: &str, bucket: &str, prefix: &str, job_id: &JobId) -> bool {
    let key = format!("{}/{}", prefix.trim_matches('/'), job_id.as_str());
    crate::s3io::delete(endpoint, bucket, &key).is_ok() // in-proc DELETE (was `aws delete-object`)
}

/// Install the spot-preemption handler (goal F): on SIGTERM/SIGINT, release the in-flight claim (if
/// any) so the job requeues immediately, then exit. Runs on a dedicated signal-hook thread (safe to
/// spawn `aws`). No-op if signal registration fails (falls back to TTL reclaim, goal E).
#[cfg(unix)]
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

/// Non-unix (Windows) build: no POSIX signals, so there is nothing to install. The worker relies on
/// TTL-based claim reclaim (goal E) instead of fast spot-release. The idle-only Windows tier is stopped
/// by Task Scheduler rather than signalled, so there is no in-flight claim to fast-release here anyway.
#[cfg(not(unix))]
fn spawn_spot_reclaim(
    _inflight: Arc<Mutex<Option<JobId>>>,
    _endpoint: &str,
    _bucket: &str,
    _prefix: &str,
) {
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

// ─────────────────── epoch-sharded claiming (claim mode `epoch_sharded`) ───────────────────
//
// The lease path pays one R2 round-trip per claim ATTEMPT and N workers attempt most claims;
// when the reconcile view is stale, dedup degrades to those leases alone (the avifgen campaign's
// measured 3.6× re-work). Epoch sharding computes ownership instead of racing for it: wall-clock
// epochs + rendezvous hashing over the alive roster (`zenfleet_core::epoch` — the pure math),
// with leases kept only where ownership just moved (the boundary seam) and in the straggler-tail
// steal. See `docs/RUNNING_JOBS.md` for adoption.

/// The tiny shared-storage surface epoch-sharded claiming needs beyond blobs + ledger:
/// heartbeat writes, roster listings, and the per-cell lease used at ownership seams.
/// R2-backed in production; directory-backed for local runs and tests.
pub trait EpochBoard {
    /// Record "`worker` was alive in `epoch`" (idempotent overwrite). Best-effort: a failed
    /// beat only risks dropping out of the NEXT epoch's roster, which self-heals.
    fn heartbeat(&self, epoch: u64, worker: &str) -> bool;
    /// Workers that heartbeated during `epoch`. Errors → empty (callers fail open to lease mode).
    fn roster_members(&self, epoch: u64) -> Vec<String>;
    /// Per-cell lease claim/steal — same TTL semantics as the lease claim path. True = this
    /// worker may run the cell.
    fn claim_cell(&self, id: &str, now: u64, ttl_secs: u64, owner: &str) -> bool;
}

/// R2-backed [`EpochBoard`]: heartbeats live at `{prefix}/hb/{epoch}/{worker}` in the claims
/// bucket (the `hb/` segment can never collide with bare-sha cell claims or `chunk-…` keys), and
/// the seam lease IS the ordinary claim object — epoch and lease workers see each other's leases.
pub struct R2EpochBoard {
    pub endpoint: String,
    pub bucket: String,
    pub prefix: String,
}

impl EpochBoard for R2EpochBoard {
    fn heartbeat(&self, epoch: u64, worker: &str) -> bool {
        let key = format!("{}/hb/{}/{}", self.prefix.trim_matches('/'), epoch, worker);
        match crate::s3io::put(&self.endpoint, &self.bucket, &key, b"1") {
            Ok(()) => true,
            Err(e) => {
                eprintln!("zenfleet-worker: heartbeat {key} failed: {e}");
                false
            }
        }
    }

    fn roster_members(&self, epoch: u64) -> Vec<String> {
        let prefix = format!("{}/hb/{}", self.prefix.trim_matches('/'), epoch);
        match crate::s3io::list_basenames(&self.endpoint, &self.bucket, &prefix) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "zenfleet-worker: roster list {prefix} failed ({e}) — empty roster, lease-mode this epoch"
                );
                Vec::new()
            }
        }
    }

    fn claim_cell(&self, id: &str, now: u64, ttl_secs: u64, owner: &str) -> bool {
        claim_or_steal_r2_key(
            &self.endpoint,
            &self.bucket,
            &self.prefix,
            id,
            now,
            ttl_secs,
            None,
            owner,
        )
    }
}

/// Directory-backed [`EpochBoard`] for local runs and tests. The stale-claim steal here is
/// read-then-overwrite (not CAS) — fine for processes on one box, NOT a distributed substitute
/// for the R2 board.
pub struct LocalEpochBoard {
    root: PathBuf,
}

impl LocalEpochBoard {
    pub fn new(root: impl Into<PathBuf>) -> io::Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }
}

impl EpochBoard for LocalEpochBoard {
    fn heartbeat(&self, epoch: u64, worker: &str) -> bool {
        let dir = self.root.join("hb").join(epoch.to_string());
        std::fs::create_dir_all(&dir)
            .and_then(|_| std::fs::write(dir.join(worker), b"1"))
            .is_ok()
    }

    fn roster_members(&self, epoch: u64) -> Vec<String> {
        match std::fs::read_dir(self.root.join("hb").join(epoch.to_string())) {
            Ok(rd) => rd
                .filter_map(|e| e.ok()?.file_name().into_string().ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    fn claim_cell(&self, id: &str, now: u64, ttl_secs: u64, owner: &str) -> bool {
        let dir = self.root.join("claims");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join(id);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(mut f) => f.write_all(format!("{now} {owner}").as_bytes()).is_ok(),
            Err(_) => {
                // Exists — steal only once stale (dead holder), same TTL rule as R2.
                let ts = std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| s.split_whitespace().next().map(str::to_string))
                    .and_then(|t| t.parse::<u64>().ok());
                match ts {
                    Some(t) if claim_is_stale(now, t, ttl_secs) => {
                        std::fs::write(&path, format!("{now} {owner}")).is_ok()
                    }
                    _ => false,
                }
            }
        }
    }
}

/// Sanitize a worker id into its roster/heartbeat key (also the HRW hash input). Every worker
/// must apply the identical mapping or rosters and shards diverge — so it lives here, once.
pub fn worker_key(worker: &str) -> String {
    worker
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// This worker's slice of one epoch, split by [`ShardDecision`] (enqueue vs poison tracked so
/// only real work spends leases; another owner's poison rows are left to that owner).
struct EpochParts {
    fast: Vec<DesiredJob>,
    guarded_enq: Vec<DesiredJob>,
    guarded_poison: Vec<DesiredJob>,
    others_enq: Vec<DesiredJob>,
}

/// The committed speed-handicap registry (`fleet/handicaps.toml`), baked into the binary so it
/// travels with every image automatically. See the file's header for the measurement basis and
/// the re-measure → edit → commit update procedure.
const EMBEDDED_HANDICAPS_TOML: &str = include_str!("../../../fleet/handicaps.toml");

#[derive(serde::Deserialize, Default)]
struct HandicapRegistryFile {
    #[serde(default)]
    workers: std::collections::BTreeMap<String, zenfleet_core::epoch::WorkerHandicap>,
}

fn parse_handicaps_toml(text: &str) -> Result<Handicaps, String> {
    toml::from_str::<HandicapRegistryFile>(text)
        .map(|f| Handicaps(f.workers))
        .map_err(|e| e.to_string())
}

/// The embedded registry, parsed once. Loud + all-1.0 on a parse error at runtime — a broken
/// registry must never stop a fleet — but `handicap_registry_parses_and_is_sane` makes the same
/// breakage fail CI first.
fn embedded_handicaps() -> &'static Handicaps {
    static H: std::sync::OnceLock<Handicaps> = std::sync::OnceLock::new();
    H.get_or_init(|| match parse_handicaps_toml(EMBEDDED_HANDICAPS_TOML) {
        Ok(h) => h,
        Err(e) => {
            eprintln!(
                "zenfleet-worker: EMBEDDED fleet/handicaps.toml failed to parse ({e}) — \
                 sharding with uniform weights (fix the registry; its unit test gates this)"
            );
            Handicaps::default()
        }
    })
}

/// Resolve the effective speed handicaps. Precedence (same convergence story as the claim
/// mode): campaign `RunControl.worker_weights` — a WHOLESALE replacement, next-pass fleet-wide
/// — else the committed registry baked into this binary, else uniform 1.0.
pub fn resolve_handicaps(ctl: &RunControl) -> Handicaps {
    match &ctl.worker_weights {
        Some(m) => Handicaps(m.clone()),
        None => embedded_handicaps().clone(),
    }
}

fn partition_epoch(
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    roster: &Roster,
    prev: Option<&Roster>,
    me: &str,
    handicaps: &Handicaps,
) -> EpochParts {
    let plan = reconcile(desired, view, policy);
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    let mut p = EpochParts {
        fast: Vec::new(),
        guarded_enq: Vec::new(),
        guarded_poison: Vec::new(),
        others_enq: Vec::new(),
    };
    let tagged = plan
        .enqueue
        .iter()
        .map(|i| (i, false))
        .chain(plan.poison.iter().map(|i| (i, true)));
    for (id, is_poison) in tagged {
        let Some(&d) = by_id.get(id) else { continue };
        // Weighted ownership, keyed by this cell's workload mode (encode-by-type / cpu-metric /
        // gpu-metric) — mixed-mode manifests shard each cell under its own weight column.
        let mode = epoch::shard_mode(&d.kind);
        let weight_of = |w: &str| handicaps.weight(w, &mode);
        match epoch::shard_decision_weighted(roster, prev, me, id.as_str(), &weight_of) {
            ShardDecision::OwnedFast => p.fast.push(d.clone()),
            ShardDecision::OwnedGuarded if is_poison => p.guarded_poison.push(d.clone()),
            ShardDecision::OwnedGuarded => p.guarded_enq.push(d.clone()),
            ShardDecision::Other if !is_poison => {
                // Steal candidacy also requires a nonzero weight for the cell's mode: weight 0
                // is the role-specialization exclusion, and it covers the tail-steal path too.
                if handicaps.weight(me, &mode) > 0.0 {
                    p.others_enq.push(d.clone());
                }
            }
            ShardDecision::Other => {}
        }
    }
    p
}

fn merge_outcome(a: &mut ExecOutcome, mut b: ExecOutcome) {
    a.rows.append(&mut b.rows);
    a.done += b.done;
    a.failed += b.failed;
    a.poisoned += b.poisoned;
    a.skipped += b.skipped;
}

/// How many steal targets to lease per batch in the straggler tail (bounds the claim burst and
/// keeps the between-batch epoch check responsive).
const STEAL_BATCH: usize = 64;

/// One epoch-sharded pass (see [`run`] for entry): heartbeat → roster → partition → execute the
/// owned shard lease-free → lease-guard the seam cells → optionally steal the straggler tail.
/// `now_fn`/`sleep_fn` are injected so tests drive virtual time; production passes the system
/// clock. Falls back to the lease path for this pass when the roster doesn't include this worker
/// yet (bootstrap epoch, fresh join, or a failed roster listing) — fail-open to productive work.
#[allow(clippy::too_many_arguments)]
fn run_epoch_sharded(
    cfg: &WorkerConfig,
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    ctx: WorkerCtx<'_>,
    endpoint: Option<&str>,
    es: &EpochShardCfg,
    board: &dyn EpochBoard,
    claim_ttl: u64,
    handicaps: &Handicaps,
    now_fn: &dyn Fn() -> u64,
    sleep_fn: &dyn Fn(u64),
) -> Result<ExecOutcome, WorkerRunError> {
    let len = es.epoch_len_secs.max(1);
    let e_now = epoch::epoch_index(cfg.now, len);
    let me = worker_key(&cfg.worker);
    board.heartbeat(e_now, &me);
    let roster = match e_now.checked_sub(1) {
        Some(prev_e) => Roster::new(e_now, board.roster_members(prev_e)),
        None => Roster::new(e_now, Vec::new()),
    };
    let prev = e_now
        .checked_sub(2)
        .map(|pp| Roster::new(e_now - 1, board.roster_members(pp)));
    if roster.is_empty() || !roster.contains(&me) {
        eprintln!(
            "zenfleet-worker: epoch-sharded epoch {e_now}: not in the roster yet ({} members) — \
             lease-mode this epoch (bootstrap/join; sharded from the next boundary)",
            roster.len()
        );
        return run_chunked(cfg, desired, view, policy, ctx, endpoint);
    }
    let parts = partition_epoch(
        desired,
        view,
        policy,
        &roster,
        prev.as_ref(),
        &me,
        handicaps,
    );
    eprintln!(
        "zenfleet-worker: epoch-sharded epoch {e_now} (len {len}s, boundary in {}s): roster {} \
         workers ({} handicap rows), shard fast={} guarded={} steal-pool={}",
        epoch::secs_to_next_boundary(cfg.now, len),
        roster.len(),
        handicaps.0.len(),
        parts.fast.len(),
        parts.guarded_enq.len() + parts.guarded_poison.len(),
        parts.others_enq.len()
    );
    match &cfg.r2 {
        Some(t) => {
            let store = R2BlobStore::new(t.endpoint.clone(), t.bucket.clone(), t.prefix.clone());
            Ok(epoch_exec(
                cfg, view, policy, ctx, endpoint, es, board, claim_ttl, now_fn, sleep_fn, &store,
                parts, e_now, &me,
            ))
        }
        None => {
            let store = LocalBlobStore::new(cfg.blobs.clone())
                .map_err(|e| WorkerRunError::Io(e.to_string()))?;
            Ok(epoch_exec(
                cfg, view, policy, ctx, endpoint, es, board, claim_ttl, now_fn, sleep_fn, &store,
                parts, e_now, &me,
            ))
        }
    }
}

/// The store-generic body of [`run_epoch_sharded`]. Three phases, all through the ordinary
/// chunked executor (LPT packing + `can_admit` bounds + durable per-chunk sidecars unchanged):
///
/// 1. **Fast shard** — cells owned now AND under the previous roster: NO lease. The chunk-claim
///    callback instead refreshes the heartbeat and checks the epoch is still current, so a pass
///    stops claiming new chunks at the boundary (in-flight cells finish; overrun ≤ one chunk).
/// 2. **Seam cells** — ownership just moved here: per-cell lease first (fences a divergent
///    roster view or a straggling prior owner's *next* epoch), then the same executor.
/// 3. **Straggler tail** (`tail_steal`) — shard exhausted with epoch time left: wait (still
///    heartbeating) for the tail window, then work OTHER workers' remaining cells in
///    deterministic per-worker order, lease-guarded. The wait matters: returning `done=0`
///    passes here would trip the entrypoint's drain-exit while peers still own live work.
#[allow(clippy::too_many_arguments)]
fn epoch_exec<B: BlobStore + Sync>(
    cfg: &WorkerConfig,
    view: &LedgerView,
    policy: RetryPolicy,
    ctx: WorkerCtx<'_>,
    endpoint: Option<&str>,
    es: &EpochShardCfg,
    board: &dyn EpochBoard,
    claim_ttl: u64,
    now_fn: &dyn Fn() -> u64,
    sleep_fn: &dyn Fn(u64),
    store: &B,
    parts: EpochParts,
    e_now: u64,
    me: &str,
) -> ExecOutcome {
    let len = es.epoch_len_secs.max(1);
    let params = default_chunk_params(cfg.chunk_wall_sec);
    let handler = |job: &DesiredJob| exec_command(&cfg.exec, job);
    let ledger_out_uri = cfg.ledger_out.to_string_lossy().into_owned();
    let mut flush = |chunk_id: &str, rows: &[LedgerRow]| {
        flush_chunk_rows(&ledger_out_uri, endpoint, chunk_id, rows);
    };
    let last_beat = std::cell::Cell::new(cfg.now);
    let heartbeat_maybe = || {
        let t = now_fn();
        if t.saturating_sub(last_beat.get()) >= es.heartbeat_interval_secs.max(1) {
            board.heartbeat(epoch::epoch_index(t, len), me);
            last_beat.set(t);
        }
    };
    let epoch_open = || epoch::epoch_index(now_fn(), len) == e_now;
    let gate = |_cid: &str| {
        heartbeat_maybe();
        epoch_open()
    };
    let EpochParts {
        fast,
        guarded_enq,
        guarded_poison,
        others_enq,
    } = parts;

    let mut out = ExecOutcome {
        rows: Vec::new(),
        done: 0,
        failed: 0,
        poisoned: 0,
        skipped: 0,
    };

    // Phase 1 — the lease-free fast shard (zero claim traffic; this is the whole point).
    if !fast.is_empty() {
        let o = execute_gap_chunked(
            &fast, view, policy, handler, store, gate, params, &mut flush, ctx,
        );
        merge_outcome(&mut out, o);
    }

    // Phase 2 — seam cells: lease each, run the won ones. Poison rows spend no lease.
    if (!guarded_enq.is_empty() || !guarded_poison.is_empty()) && epoch_open() {
        let mut won = guarded_poison;
        for d in guarded_enq {
            if board.claim_cell(d.job_id().as_str(), cfg.now, claim_ttl, me) {
                won.push(d);
            } else {
                out.skipped += 1;
            }
        }
        if !won.is_empty() {
            let o = execute_gap_chunked(
                &won, view, policy, handler, store, gate, params, &mut flush, ctx,
            );
            merge_outcome(&mut out, o);
        }
    }

    // Phase 3 — straggler-tail steal (endgame): shard exhausted, others' cells remain.
    if es.tail_steal && !others_enq.is_empty() && epoch_open() {
        while epoch_open() && !epoch::in_tail(now_fn(), len) {
            heartbeat_maybe();
            let step = epoch::secs_to_tail(now_fn(), len)
                .clamp(1, es.heartbeat_interval_secs.clamp(1, 30));
            sleep_fn(step);
        }
        if epoch_open() {
            let ids: Vec<JobId> = others_enq.iter().map(|d| d.job_id()).collect();
            let keys: Vec<&str> = ids.iter().map(|i| i.as_str()).collect();
            let order = epoch::steal_order(me, &keys);
            eprintln!(
                "zenfleet-worker: epoch-sharded epoch {e_now}: shard exhausted — tail-stealing \
                 from {} remaining cells (lease-guarded)",
                keys.len()
            );
            for batch in order.chunks(STEAL_BATCH) {
                if !epoch_open() {
                    break;
                }
                let mut won: Vec<DesiredJob> = Vec::new();
                for &i in batch {
                    if board.claim_cell(keys[i], cfg.now, claim_ttl, me) {
                        won.push(others_enq[i].clone());
                    } else {
                        out.skipped += 1;
                    }
                }
                if won.is_empty() {
                    continue;
                }
                let o = execute_gap_chunked(
                    &won, view, policy, handler, store, gate, params, &mut flush, ctx,
                );
                merge_outcome(&mut out, o);
            }
        }
    }
    out
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
    let ct = std::time::Instant::now();
    let ctimed = std::env::var("ZEN_TIME_PASS").ok().as_deref() == Some("1");
    macro_rules! cmark {
        ($p:expr) => {
            if ctimed {
                eprintln!(
                    "zenfleet-worker[time] {:<16} {:?} ({} jobs)",
                    $p,
                    ct.elapsed(),
                    desired.len()
                );
            }
        };
    }
    let plan = reconcile(desired, view, policy);
    cmark!("reconcile");
    let by_id: HashMap<JobId, &DesiredJob> = desired.iter().map(|d| (d.job_id(), d)).collect();
    cmark!("by_id-hashmap");
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
    cmark!("job-costs");
    let chunks = params.budget.pack_chunks_lpt(&costs, params.chunk_wall_sec);
    cmark!("pack-chunks");
    let chunk_ids: Vec<String> = chunks
        .iter()
        .map(|members| {
            let ids: Vec<JobId> = members.iter().map(|&m| gap[m].job_id()).collect();
            chunk_id(&ids)
        })
        .collect();
    cmark!("chunk-ids");

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
        // Classification priority: (1) the executor's own stderr classification — an explicit
        // `ZEN_ERROR_CLASS: <class>` marker line, or a well-known raw failure string (CUDA OOM /
        // ENOSPC) — then (2) the exit-shape heuristic below. This is the one-shot half of the
        // error-class fidelity fix (#45): before it, a CUDA_ERROR_OUT_OF_MEMORY panic (avifgen
        // sf-gpu storm, 10,291 jobs) and ENOSPC (hdrgrid) exited nonzero → `encoder_panic` →
        // poisoned as deterministic, though both are transient.
        let stderr = String::from_utf8_lossy(&output.stderr);
        // A signal-killed child (no exit code) is a lost worker / OOM under memory pressure — TRANSIENT,
        // so the reconciler requeues it instead of poisoning it. Only a real non-zero EXIT code is a
        // deterministic EncoderPanic. (Before: every failure was EncoderPanic, so an OOM-killed cell was
        // poisoned and its 720 features lost forever — a latent trap raised by ZEN_CORE_OVERSUBSCRIBE>1.)
        let class =
            class_from_stderr(&stderr).unwrap_or_else(|| classify_child_failure(&output.status));
        let detail = match output.status.code() {
            Some(code) => format!("{program} exited {code}"),
            None => format!("{program} killed by signal"),
        };
        Err(HandlerError::new(class, format!("{detail}: {stderr}")))
    }
}

/// Extract the executor's own failure classification from its captured stderr.
///
/// Two channels, in priority order:
/// 1. An explicit `ZEN_ERROR_CLASS: <snake_case>` marker line printed by a class-aware executor
///    (`zenmetrics jobexec`) just before a nonzero exit. Last occurrence wins; an unknown class
///    token is IGNORED (strict parse — a garbled marker must not upgrade a deterministic failure
///    to transient).
/// 2. Well-known raw failure strings for class-unaware executors and panics that unwound past
///    classification: CUDA VRAM exhaustion → [`ErrorClass::Oom`]; ENOSPC → [`ErrorClass::DiskFull`].
///    Deliberately conservative — exact, unambiguous markers only.
fn class_from_stderr(stderr: &str) -> Option<ErrorClass> {
    const MARKER: &str = "ZEN_ERROR_CLASS:";
    if let Some(idx) = stderr.rfind(MARKER) {
        let token = stderr[idx + MARKER.len()..]
            .lines()
            .next()
            .unwrap_or("")
            .trim();
        if let Some(c) = ErrorClass::parse_strict(token) {
            return Some(c);
        }
    }
    if stderr.contains("CUDA_ERROR_OUT_OF_MEMORY") || stderr.contains("OutOfMemory") {
        return Some(ErrorClass::Oom);
    }
    if stderr.contains("No space left on device") || stderr.contains("ENOSPC") {
        return Some(ErrorClass::DiskFull);
    }
    None
}

/// Classify a failed child's exit into an [`ErrorClass`]. A **signal-killed** child (`code() == None`)
/// is a lost worker or an OOM-kill under memory pressure → transient, so the reconciler retries it on a
/// less-loaded box instead of poisoning it (permanent loss of that cell's features). A real non-zero
/// **exit code** is a deterministic failure (bad bytes / encoder panic) → `EncoderPanic` → poison after
/// the retry cap. `SIGKILL`(9) is the kernel OOM-killer's signal; other signals (SIGSEGV/SIGABRT/SIGBUS)
/// are usually allocator/load-induced and still worth a retry — a genuinely deterministic crash just
/// poisons after `max_attempts` anyway, so classifying it transient costs at most a few retries.
fn classify_child_failure(status: &std::process::ExitStatus) -> ErrorClass {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = status.signal() {
            return if sig == 9 {
                ErrorClass::Oom
            } else {
                ErrorClass::WorkerLost
            };
        }
    }
    let _ = status; // referenced only on unix (the signal check above)
    ErrorClass::EncoderPanic
}

/// A long-lived `program --serve` child for the persistent executor. Children live in a
/// [`WarmExecPool`] (one per worker PROCESS); since the fleet runs one (long) pass per process,
/// each child stays warm across many of the pass's jobs, so CUDA init + GPU kernel compilation are
/// paid ONCE per child rather than per job.
struct PersistentExec {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

/// The process-global warm pool [`exec_command_persistent`] draws from (one worker pass = one
/// process = one executor program, so a single global is the pass-lifetime pool).
static PERSISTENT: Mutex<Option<Arc<WarmExecPool>>> = Mutex::new(None);

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
        match status[0] {
            0 => Ok(payload),
            // Status 3 = CLASSIFIED error frame (`{"class":"…","detail":"…"}`) from a class-aware
            // executor — the serve-mode half of the error-class fidelity fix (#45). An unknown
            // class string (newer executor) degrades to the transient `Unknown`, mirroring the
            // forgiving ledger read; a malformed payload falls back to `EncoderPanic` + raw text.
            3 => {
                #[derive(serde::Deserialize)]
                struct ClassifiedError {
                    class: String,
                    #[serde(default)]
                    detail: String,
                }
                match serde_json::from_slice::<ClassifiedError>(&payload) {
                    Ok(c) => Err(HandlerError::new(
                        ErrorClass::parse_lossy(&c.class),
                        c.detail,
                    )),
                    Err(_) => Err(HandlerError::new(
                        ErrorClass::EncoderPanic,
                        String::from_utf8_lossy(&payload).into_owned(),
                    )),
                }
            }
            // Status 1 (job error) / 2 (caught panic) from a legacy executor, or anything else:
            // the child framed a failure for THIS job and stayed alive → deterministic failure.
            // Raw-marker scan still rescues the two known transient shapes (CUDA OOM / ENOSPC)
            // when a legacy frame carries them in its message text.
            _ => {
                let text = String::from_utf8_lossy(&payload).into_owned();
                let class = class_from_stderr(&text).unwrap_or(ErrorClass::EncoderPanic);
                Err(HandlerError::new(class, text))
            }
        }
    }
}

/// Recycle policy for warm executor children. Long-lived GPU processes accumulate RSS (allocator
/// high-water, decoder scratch, driver caches) — a child past either bound is killed after its
/// current job and the next job spawns a fresh one, re-paying CUDA init + JIT ONCE per recycle
/// instead of per job.
#[derive(Debug, Clone, Copy)]
pub struct WarmPoolCfg {
    /// Kill a child whose `/proc/<pid>/status` VmRSS exceeds this after a job. `0` disables.
    pub rss_max_bytes: u64,
    /// Kill a child after this many jobs. `0` disables.
    pub max_jobs_per_child: u64,
}

impl WarmPoolCfg {
    /// Defaults: 8 GiB RSS watermark (`ZEN_PERSISTENT_RSS_MAX_GB`), 10,000 jobs
    /// (`ZEN_PERSISTENT_MAX_JOBS`). The watermark also bounds the admission blind spot: idle warm
    /// children's RSS is NOT counted by [`BoxBudget::can_admit`] (only admitted jobs' hints are),
    /// so worst-case unaccounted memory ≈ watermark × idle children.
    pub fn from_env() -> Self {
        let gb = std::env::var("ZEN_PERSISTENT_RSS_MAX_GB")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(8.0);
        let max_jobs = std::env::var("ZEN_PERSISTENT_MAX_JOBS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(10_000);
        WarmPoolCfg {
            rss_max_bytes: (gb.max(0.0) * (1u64 << 30) as f64) as u64,
            max_jobs_per_child: max_jobs,
        }
    }
}

/// VmRSS of `pid` in bytes from `/proc/<pid>/status` (Linux; `None` elsewhere or on parse failure —
/// the RSS watermark simply doesn't fire there, `max_jobs_per_child` still bounds recycling).
fn rss_bytes_of_pid(pid: u32) -> Option<u64> {
    let s = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

struct PooledChild {
    pe: PersistentExec,
    jobs_run: u64,
}

/// A pool of warm `program --serve` children for this worker process — the chunked-path fix for
/// GPU duty cycle (#45): the chunked claim path admits cells CONCURRENTLY (`can_admit`), so a
/// single warm child would serialize the pass; instead each admitted cell checks a child out
/// (spawning one if none is idle), streams its length-framed job, and returns the child warm.
/// Pool size is therefore bounded by peak admitted concurrency (≤2 on the hinted GPU queues) —
/// never configured, never grows past it.
///
/// Crash isolation is per-JOB: a child that dies mid-job fails THAT job (`WorkerLost`, transient)
/// and is not returned; the next checkout spawns fresh. A framed per-job error keeps the child
/// warm. [`WarmPoolCfg`] recycles children at an RSS watermark / job count so leaks stay bounded.
pub struct WarmExecPool {
    program: String,
    cfg: WarmPoolCfg,
    idle: Mutex<Vec<PooledChild>>,
}

impl WarmExecPool {
    pub fn new(program: impl Into<String>, cfg: WarmPoolCfg) -> Self {
        WarmExecPool {
            program: program.into(),
            cfg,
            idle: Mutex::new(Vec::new()),
        }
    }

    fn kill_child(mut c: PooledChild) {
        let _ = c.pe.child.kill();
        let _ = c.pe.child.wait();
    }

    /// Run one job on a warm child (checkout → framed round-trip → recycle-or-checkin).
    pub fn run_job(&self, job: &DesiredJob) -> Result<Vec<u8>, HandlerError> {
        let job_json = serde_json::to_vec(job)
            .map_err(|e| HandlerError::new(ErrorClass::Unknown, format!("serialize job: {e}")))?;
        let mut child = {
            let popped = self.idle.lock().unwrap_or_else(|p| p.into_inner()).pop();
            match popped {
                Some(c) => c,
                None => PooledChild {
                    pe: spawn_serve(&self.program)?,
                    jobs_run: 0,
                },
            }
        };
        let res = child.pe.run_job(&job_json);
        match &res {
            // Framing I/O failed → the child is presumed dead. Do NOT return it; the job's
            // FAILED row is transient (`WorkerLost`) and the next checkout spawns fresh.
            Err(e) if matches!(e.class, ErrorClass::WorkerLost) => Self::kill_child(child),
            // Success or a framed per-job failure: the child is alive and warm. Recycle it if
            // it crossed the RSS watermark or job cap, else return it to the pool.
            _ => {
                child.jobs_run += 1;
                let over_jobs = self.cfg.max_jobs_per_child > 0
                    && child.jobs_run >= self.cfg.max_jobs_per_child;
                let over_rss = self.cfg.rss_max_bytes > 0
                    && rss_bytes_of_pid(child.pe.child.id())
                        .is_some_and(|rss| rss > self.cfg.rss_max_bytes);
                if over_jobs || over_rss {
                    eprintln!(
                        "zenfleet-worker: recycling warm executor child (pid {}, jobs {}, {}) — \
                         next job spawns fresh",
                        child.pe.child.id(),
                        child.jobs_run,
                        if over_rss {
                            "RSS over watermark"
                        } else {
                            "job cap reached"
                        }
                    );
                    Self::kill_child(child);
                } else {
                    self.idle
                        .lock()
                        .unwrap_or_else(|p| p.into_inner())
                        .push(child);
                }
            }
        }
        res
    }
}

impl Drop for WarmExecPool {
    fn drop(&mut self) {
        // Kill-and-reap (hang-proof): serve children hold no un-flushed state — outputs stream
        // per job — so SIGKILL at pass end loses nothing.
        let children = std::mem::take(&mut *self.idle.lock().unwrap_or_else(|p| p.into_inner()));
        for c in children {
            Self::kill_child(c);
        }
    }
}

/// Persistent variant of [`exec_command`]: run the job on a warm `program --serve` child from the
/// process-global [`WarmExecPool`], so CUDA init + kernel compilation are paid once per child rather
/// than per job (the fix for ~20s/job cold-process overhead on GPU metric fleets). Under concurrent
/// callers the pool holds one child per concurrent job; a dead child fails only its own job
/// (transient) and is respawned on the next call.
pub fn exec_command_persistent(program: &str, job: &DesiredJob) -> Result<Vec<u8>, HandlerError> {
    let pool = {
        let mut guard = PERSISTENT.lock().unwrap_or_else(|p| p.into_inner());
        guard
            .get_or_insert_with(|| Arc::new(WarmExecPool::new(program, WarmPoolCfg::from_env())))
            .clone()
    };
    if pool.program != program {
        // Defensive: one worker pass has one executor; a mismatched program falls back to the
        // one-shot path rather than running the wrong warm binary.
        return exec_command(program, job);
    }
    pool.run_job(job)
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

/// The manifest kind-tag of a job (`snake_case`, matching the wire `{"kind": {"kind": …}}`).
fn kind_name(kind: &JobKind) -> &'static str {
    match kind {
        JobKind::Encode { .. } => "encode",
        JobKind::Metric { .. } => "metric",
        JobKind::ScoreFile { .. } => "score_file",
        JobKind::Feature { .. } => "feature",
        JobKind::Diffmap { .. } => "diffmap",
        JobKind::Resample { .. } => "resample",
        JobKind::Bake { .. } => "bake",
    }
}

/// Which job kinds run on WARM children under the chunked path (`ZEN_PERSISTENT_KINDS`, csv;
/// `all` = every kind). Default: the decode+score kinds (`score_file`, `diffmap`, `feature`) —
/// the GPU-metric shapes where per-job CUDA init + JIT dominates. `encode`/`metric` stay on
/// fresh processes by default: both re-encode, and the jxl `modes_full` per-cell RSS ramp
/// (13–24 GB within one process — see zenmetrics CLAUDE.md Known Bugs) is exactly the shape the
/// fresh-process design bounds. Opting them in is allowed (the RSS watermark then bounds the
/// ramp) but is a deliberate, per-fleet decision.
fn warm_kinds_from_env() -> Vec<String> {
    match std::env::var("ZEN_PERSISTENT_KINDS") {
        Ok(v) => v
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect(),
        Err(_) => vec![
            "score_file".to_string(),
            "diffmap".to_string(),
            "feature".to_string(),
        ],
    }
}

fn kind_is_warm_eligible(kind: &JobKind, warm_kinds: &[String]) -> bool {
    let name = kind_name(kind);
    warm_kinds.iter().any(|k| k == "all" || k == name)
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
    /// Epoch-sharded claiming (`--claim-mode epoch-sharded`). `None` (the default) = lease
    /// claiming, behavior identical to before this field existed. The run's control object can
    /// override the mode fleet-wide (see [`RunControl::claim_mode`] and [`resolve_epoch_cfg`]).
    pub epoch_shard: Option<EpochShardCfg>,
}

/// Resolve the effective epoch-shard config from the campaign control object + this worker's own
/// config. The control object wins when it names a mode (that is how a whole fleet converges on
/// one mode — mixed modes on one run re-introduce the duplicate-work tax); its `epoch_len_secs` /
/// `heartbeat_interval_secs` override the worker's numbers when present. `None` = lease.
pub fn resolve_epoch_cfg(
    ctl: &RunControl,
    worker_cfg: Option<EpochShardCfg>,
) -> Option<EpochShardCfg> {
    match ctl.claim_mode {
        Some(ClaimMode::Lease) => None,
        Some(ClaimMode::EpochSharded) => {
            let mut es = worker_cfg.unwrap_or_default();
            if let Some(l) = ctl.epoch_len_secs {
                es.epoch_len_secs = l;
            }
            if let Some(h) = ctl.heartbeat_interval_secs {
                es.heartbeat_interval_secs = h;
            }
            Some(es)
        }
        None => worker_cfg,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerRunError {
    #[error("io {0}")]
    Io(String),
    #[error("manifest {0}")]
    Manifest(String),
    #[error("ledger {0}")]
    Ledger(String),
    #[error("config {0}")]
    Config(String),
}

// ──────────────────────────── garbage collection (goal G) ────────────────────────────

/// Delete an R2 object via `s5cmd rm`.
fn s5cmd_rm(endpoint: &str, uri: &str) -> Result<(), String> {
    // In-proc DELETE (was an `s5cmd rm` spawn). `uri` = s3://<bucket>/<key>.
    let rest = uri
        .strip_prefix("s3://")
        .ok_or_else(|| format!("not an s3:// uri: {uri}"))?;
    let (bucket, key) = rest
        .split_once('/')
        .ok_or_else(|| format!("s3 uri has no key: {uri}"))?;
    crate::s3io::delete(endpoint, bucket, key)
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

/// Total physical RAM in bytes, cross-platform: `ZEN_RAM_BYTES` env override first,
/// then `/proc/meminfo` (Linux), then `sysctl -n hw.memsize` (macOS — no procfs
/// there; without this a darwin worker fell to the 2 GiB floor and its admission
/// budget rejected every cell, cycling done=0 — found on lilith-mac 2026-07-27).
fn total_ram_bytes() -> Option<u64> {
    if let Ok(v) = std::env::var("ZEN_RAM_BYTES")
        && let Ok(n) = v.trim().parse::<u64>()
        && n > 0
    {
        return Some(n);
    }
    if let Some(n) = read_meminfo_total_bytes() {
        return Some(n);
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

/// This box's admission budget for the chunked path: **RAM budget = 75 % of physical RAM** (leaves
/// headroom for the OS, page cache, GPU readback, and the estimate's slop — see [`BoxBudget`]) and
/// **cores = usable parallelism** (`available_parallelism` honors cgroup/cpuset affinity, which the
/// fleet onstart pins; RAM is bounded separately by `can_admit`, so we do NOT also shrink cores by
/// RAM the way a blind N-per-core launcher would). Conservative fallbacks (2 GiB / 1 core) if either
/// probe fails — never panics.
fn host_box_budget() -> BoxBudget {
    let total_ram = total_ram_bytes().unwrap_or(2 << 30); // 2 GiB only if every probe fails
    let ram_budget = (((total_ram as f64) * 0.75) as u64).max(1);
    let phys_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(1)
        .max(1);
    // I/O-bound cells (e.g. R2-fetch-dominated feature extraction: each cell fetches its variants and
    // spends most of its wall time blocked on the network, not the CPU) leave cores idle — a box scoring
    // at load ~2/8 is bottlenecked on fetch latency, not compute. `ZEN_CORE_OVERSUBSCRIBE` (a float ≥ 1,
    // default 1.0 = no change) multiplies the admit budget's core term so more cells run concurrently and
    // overlap their fetches. RAM is bounded SEPARATELY by `can_admit` (Σpeak_mem ≤ 0.75×RAM), so
    // oversubscribing threads can never OOM the box — it only admits more concurrent I/O waits. Leave it
    // at 1.0 for CPU-bound tiers (encode/GPU); raise to ~3 for fetch-dominated feature backfills.
    let oversub = std::env::var("ZEN_CORE_OVERSUBSCRIBE")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|f| f.is_finite() && *f >= 1.0)
        .unwrap_or(1.0);
    let cores = (((phys_cores as f64) * oversub).round() as u32).max(1);
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
/// The chunk envelope both the lease path and the epoch-sharded path run under: this box's
/// budget, the caller's chunk wall target, and the no-hint fallback footprint (a modest
/// 512 MB / 1 thread, so admission stays safe).
fn default_chunk_params(chunk_wall_sec: f64) -> ChunkParams {
    ChunkParams {
        budget: host_box_budget(),
        chunk_wall_sec,
        fallback_hint: ResourceHint {
            peak_mem_bytes: 512 << 20,
            threads: 1,
        },
    }
}

/// Durable per-chunk sidecar write (shared by the lease and epoch-sharded paths). Failure is
/// loud but non-fatal: the rows also ride the pass outcome, and the next pass re-derives any
/// truly lost work from the gap.
fn flush_chunk_rows(
    ledger_out_uri: &str,
    endpoint: Option<&str>,
    chunk_id: &str,
    rows: &[LedgerRow],
) {
    if rows.is_empty() {
        return;
    }
    let uri = chunk_ledger_uri(ledger_out_uri, chunk_id);
    if let Err(e) = zenfleet_ledger::write_ledger_uri(&uri, rows, endpoint) {
        eprintln!("zenfleet-worker: chunk {chunk_id} ledger write to {uri} failed: {e}");
    }
}

fn run_chunked(
    cfg: &WorkerConfig,
    desired: &[DesiredJob],
    view: &LedgerView,
    policy: RetryPolicy,
    ctx: WorkerCtx<'_>,
    endpoint: Option<&str>,
) -> Result<ExecOutcome, WorkerRunError> {
    let params = default_chunk_params(cfg.chunk_wall_sec);
    eprintln!(
        "zenfleet-worker: resource-aware concurrent mode (LPT + can_admit, chunk target {:.0}s) \
         — budget {:.1} GiB / {} cores. Set ZEN_CHUNK_WALL_SEC=0 for the serial per-cell path.",
        params.chunk_wall_sec,
        params.budget.ram_budget_bytes as f64 / (1u64 << 30) as f64,
        params.budget.cores
    );
    // Executor dispatch (#45, the warm-process fix): with `ZEN_PERSISTENT_EXEC=1`, warm-eligible
    // kinds (default: the decode+score kinds — see [`warm_kinds_from_env`]) run on POOLED warm
    // `--serve` children, so CUDA init + CubeCL JIT are paid once per child instead of per job
    // (pre-fix dmon on the avifgen sf-gpu wave: node-2 sm 85→0→49→0, host-stall dominated).
    // Everything else — and everything, when the env is unset — keeps the fresh-process path,
    // which bounds the jxl modes_full per-cell RSS ramp by construction. Cells still run truly
    // concurrently under the budget either way: the pool hands each concurrent job its own child.
    let persistent = std::env::var("ZEN_PERSISTENT_EXEC")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let warm_kinds = warm_kinds_from_env();
    if persistent {
        eprintln!(
            "zenfleet-worker: warm persistent executor ENABLED for kinds [{}] (ZEN_PERSISTENT_EXEC; \
             recycle: {:?})",
            warm_kinds.join(","),
            WarmPoolCfg::from_env()
        );
    }
    let handler = |job: &DesiredJob| {
        if persistent && kind_is_warm_eligible(&job.kind, &warm_kinds) {
            exec_command_persistent(&cfg.exec, job)
        } else {
            exec_command(&cfg.exec, job)
        }
    };
    let ledger_out_uri = cfg.ledger_out.to_string_lossy().into_owned();
    let mut flush = |chunk_id: &str, rows: &[LedgerRow]| {
        flush_chunk_rows(&ledger_out_uri, endpoint, chunk_id, rows);
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
    // Phase timing (ZEN_TIME_PASS=1) — isolates where a pass spends its time so a slow
    // pass can be diagnosed LOCALLY (run this binary against the real R2 run + a nop
    // exec) instead of SSH-guessing on a fleet box.
    let timed = std::env::var("ZEN_TIME_PASS").ok().as_deref() == Some("1");
    let t = std::time::Instant::now();
    macro_rules! mark {
        ($p:expr) => {
            if timed {
                eprintln!("zenfleet-worker[time] {:<16} {:?}", $p, t.elapsed());
            }
        };
    }
    let bytes = std::fs::read(&cfg.manifest).map_err(|e| {
        WorkerRunError::Io(format!("read manifest {}: {e}", cfg.manifest.display()))
    })?;
    mark!("read-manifest");
    let mut desired: Vec<DesiredJob> =
        serde_json::from_slice(&bytes).map_err(|e| WorkerRunError::Manifest(e.to_string()))?;
    mark!("parse-manifest");
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
    mark!("read-ledger");

    // Run control (goal C): if the run is paused/draining, pull no new work this pass. Fail-open —
    // an absent control object reads as RUNNING. The ledger is untouched, so resuming continues
    // exactly where it left off ("without losing state").
    let ctl = match (&cfg.r2, &cfg.control_key) {
        (Some(t), Some(key)) => fetch_control_r2(&t.endpoint, &t.bucket, key),
        _ => RunControl::RUNNING,
    };
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

    let policy = RetryPolicy {
        max_attempts: cfg.max_attempts,
    };
    let ctx = WorkerCtx {
        worker: &cfg.worker,
        provider: &cfg.provider,
        now: cfg.now,
    };

    // Epoch-sharded claiming (opt-in; the campaign control object may set/override the mode).
    // Rides the chunked executor, so it requires the chunked path.
    if let Some(es) = resolve_epoch_cfg(&ctl, cfg.epoch_shard) {
        if cfg.chunk_wall_sec <= 0.0 {
            return Err(WorkerRunError::Config(
                "epoch-sharded claiming requires the chunked path (keep ZEN_CHUNK_WALL_SEC > 0)"
                    .into(),
            ));
        }
        let (board, ttl): (Box<dyn EpochBoard>, u64) = match (&cfg.r2, &cfg.claims) {
            (Some(t), Some(cc)) => (
                Box::new(R2EpochBoard {
                    endpoint: t.endpoint.clone(),
                    bucket: cc.bucket.clone(),
                    prefix: cc.prefix.clone(),
                }),
                cc.ttl_secs,
            ),
            (Some(_), None) => {
                return Err(WorkerRunError::Config(
                    "epoch-sharded claiming on R2 requires --claims-r2-bucket (the heartbeat + seam-lease namespace)"
                        .into(),
                ));
            }
            (None, _) => (
                Box::new(
                    LocalEpochBoard::new(cfg.blobs.join("_epoch"))
                        .map_err(|e| WorkerRunError::Io(e.to_string()))?,
                ),
                cfg.claims.as_ref().map(|c| c.ttl_secs).unwrap_or(600),
            ),
        };
        let now_fn = || {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        };
        let sleep_fn = |s: u64| std::thread::sleep(std::time::Duration::from_secs(s));
        // Registered speed handicaps: control-object override > committed registry > 1.0.
        let handicaps = resolve_handicaps(&ctl);
        return run_epoch_sharded(
            cfg,
            &desired,
            &view,
            policy,
            ctx,
            endpoint,
            &es,
            board.as_ref(),
            ttl,
            &handicaps,
            &now_fn,
            &sleep_fn,
        );
    }

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

    #[test]
    #[cfg(unix)]
    fn signal_killed_cells_are_transient_not_poisoned() {
        use std::os::unix::process::ExitStatusExt;
        // SIGKILL(9) — kernel OOM-killer → Oom (transient → retried, NOT poisoned/lost)
        let killed = std::process::ExitStatus::from_raw(9);
        assert_eq!(classify_child_failure(&killed), ErrorClass::Oom);
        assert!(classify_child_failure(&killed).is_transient());
        // other signal (SIGSEGV=11) → WorkerLost (transient)
        let segv = std::process::ExitStatus::from_raw(11);
        assert_eq!(classify_child_failure(&segv), ErrorClass::WorkerLost);
        // a real non-zero exit code (1) → EncoderPanic (deterministic → poison after the cap)
        let exited = std::process::ExitStatus::from_raw(1 << 8);
        assert_eq!(classify_child_failure(&exited), ErrorClass::EncoderPanic);
        assert!(!classify_child_failure(&exited).is_transient());
    }

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
    fn class_from_stderr_marker_and_raw_scan() {
        // Explicit marker wins, last occurrence, trimmed.
        assert_eq!(
            class_from_stderr("blah\nZEN_ERROR_CLASS: oom\nmore"),
            Some(ErrorClass::Oom)
        );
        assert_eq!(
            class_from_stderr("ZEN_ERROR_CLASS: decode_error\nZEN_ERROR_CLASS: source_fetch"),
            Some(ErrorClass::SourceFetch)
        );
        // Garbled/unknown marker token is IGNORED (must not upgrade to transient)…
        assert_eq!(class_from_stderr("ZEN_ERROR_CLASS: banana"), None);
        // …but a raw CUDA marker elsewhere still classifies.
        assert_eq!(
            class_from_stderr(
                "ZEN_ERROR_CLASS: banana\nDriverError(CUDA_ERROR_OUT_OF_MEMORY, \"oom\")"
            ),
            Some(ErrorClass::Oom)
        );
        // Raw markers for class-unaware executors.
        assert_eq!(
            class_from_stderr("thread panicked: DriverError(CUDA_ERROR_OUT_OF_MEMORY)"),
            Some(ErrorClass::Oom)
        );
        assert_eq!(
            class_from_stderr("write /scratch/x: No space left on device (os error 28)"),
            Some(ErrorClass::DiskFull)
        );
        assert_eq!(class_from_stderr("ordinary encoder panic"), None);
    }

    /// One-shot executor prints the explicit class marker on stderr + exits nonzero →
    /// the FAILED row carries the real class, not `encoder_panic` (#45, the avifgen
    /// OOM-storm / hdrgrid-ENOSPC mislabel fix, one-shot half).
    #[test]
    #[cfg(unix)]
    fn exec_command_reads_stderr_class_marker() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fail_disk_full.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\ncat >/dev/null\necho 'scratch: No space left on device' >&2\necho 'ZEN_ERROR_CLASS: disk_full' >&2\nexit 7\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let d = desired("cvvdp", b"a");
        let err = exec_command(script.to_str().unwrap(), &d).unwrap_err();
        assert_eq!(err.class, ErrorClass::DiskFull, "msg: {}", err.msg);
        assert!(err.class.is_transient());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A class-UNAWARE executor whose panic text carries the CUDA OOM string is still
    /// classified transient by the raw-marker scan (panics unwind past classification).
    #[test]
    #[cfg(unix)]
    fn exec_command_raw_cuda_oom_scan() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp();
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fail_cuda_oom.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\ncat >/dev/null\necho 'thread panicked: DriverError(CUDA_ERROR_OUT_OF_MEMORY, \"out of memory\")' >&2\nexit 101\n",
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let d = desired("cvvdp", b"a");
        let err = exec_command(script.to_str().unwrap(), &d).unwrap_err();
        assert_eq!(err.class, ErrorClass::Oom, "msg: {}", err.msg);
        std::fs::remove_dir_all(&dir).ok();
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
            epoch_shard: None,
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
        assert_eq!(
            resolve_chunk_wall_sec(Some("not-a-number")),
            DEFAULT_CHUNK_WALL_SEC
        );
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
        const {
            assert!(DEFAULT_CHUNK_WALL_SEC > 0.0);
        }
    }

    #[test]
    fn warm_kind_gating() {
        // Default warm set = the decode+score kinds; encode/metric stay fresh-process.
        let defaults = ["score_file", "diffmap", "feature"].map(String::from);
        let score = JobKind::ScoreFile {
            metrics: vec!["ssim2".into()],
            hdr: false,
            hdr_transfer: None,
        };
        let feat = JobKind::Feature {
            regime: "944".into(),
        };
        let diff = JobKind::Diffmap {
            metric: "ssim2".into(),
            hdr: false,
        };
        let enc = JobKind::Encode {
            codec: "zenjxl".into(),
            q: 80,
            knobs: "{}".into(),
            hdr: false,
        };
        let met = JobKind::Metric {
            metric: "cvvdp".into(),
        };
        assert!(kind_is_warm_eligible(&score, &defaults));
        assert!(kind_is_warm_eligible(&feat, &defaults));
        assert!(kind_is_warm_eligible(&diff, &defaults));
        assert!(
            !kind_is_warm_eligible(&enc, &defaults),
            "encode keeps the fresh-process memory bound by default"
        );
        assert!(!kind_is_warm_eligible(&met, &defaults));
        // 'all' opts everything in; an explicit list opts kinds in by name.
        let all = ["all".to_string()];
        assert!(kind_is_warm_eligible(&enc, &all));
        let with_metric = ["metric".to_string()];
        assert!(kind_is_warm_eligible(&met, &with_metric));
        assert!(!kind_is_warm_eligible(&score, &with_metric));
        // kind_name covers every variant the manifest can carry.
        assert_eq!(kind_name(&score), "score_file");
        assert_eq!(kind_name(&enc), "encode");
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
                hdr: false,
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
            epoch_shard: None,
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

    // ───────────────────── epoch-sharded claiming (claim mode `epoch_sharded`) ─────────────────────

    use std::cell::Cell as StdCell;
    use std::rc::Rc;

    /// 24 distinct metric cells (distinct input shas → distinct JobIds).
    fn epoch_cells() -> Vec<DesiredJob> {
        (0u8..24).map(|i| desired("ssim2", &[i])).collect()
    }

    fn epoch_worker_cfg(root: &Path, worker: &str, now: u64, tail_steal: bool) -> WorkerConfig {
        WorkerConfig {
            manifest: root.join("unused-manifest.json"),
            ledger_in: vec![],
            ledger_out: root.join(format!("pass-{worker}.parquet")),
            blobs: root.join("blobs"),
            r2: None,
            claims: None,
            control_key: None,
            exec: "cat".into(),
            worker: worker.into(),
            provider: "local".into(),
            now,
            max_attempts: 3,
            served: vec![],
            chunk_wall_sec: 2.0,
            epoch_shard: Some(EpochShardCfg {
                epoch_len_secs: 600,
                heartbeat_interval_secs: 120,
                tail_steal,
            }),
        }
    }

    /// Virtual clock: `now_fn` reads it, `sleep_fn` advances it (no real sleeping in tests).
    fn fake_time(start: u64) -> (Rc<StdCell<u64>>, impl Fn() -> u64, impl Fn(u64)) {
        let c = Rc::new(StdCell::new(start));
        let n = c.clone();
        let s = c.clone();
        (c, move || n.get(), move |secs| s.set(s.get() + secs))
    }

    fn run_epoch_pass(
        cfg: &WorkerConfig,
        desired: &[DesiredJob],
        view: &LedgerView,
        board: &dyn EpochBoard,
        now_fn: &dyn Fn() -> u64,
        sleep_fn: &dyn Fn(u64),
    ) -> ExecOutcome {
        run_epoch_pass_weighted(
            cfg,
            desired,
            view,
            board,
            &Handicaps::default(),
            now_fn,
            sleep_fn,
        )
    }

    fn run_epoch_pass_weighted(
        cfg: &WorkerConfig,
        desired: &[DesiredJob],
        view: &LedgerView,
        board: &dyn EpochBoard,
        handicaps: &Handicaps,
        now_fn: &dyn Fn() -> u64,
        sleep_fn: &dyn Fn(u64),
    ) -> ExecOutcome {
        run_epoch_sharded(
            cfg,
            desired,
            view,
            RetryPolicy::default(),
            WorkerCtx {
                worker: &cfg.worker,
                provider: &cfg.provider,
                now: cfg.now,
            },
            None,
            cfg.epoch_shard.as_ref().unwrap(),
            board,
            600,
            handicaps,
            now_fn,
            sleep_fn,
        )
        .unwrap()
    }

    fn row_ids(out: &ExecOutcome) -> HashSet<String> {
        out.rows
            .iter()
            .map(|r| r.job_id.as_str().to_string())
            .collect()
    }

    #[test]
    fn epoch_two_workers_split_disjointly_and_take_no_leases() {
        let root = tmp();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let board = LocalEpochBoard::new(root.join("blobs").join("_epoch")).unwrap();
        // Stable roster {w1,w2} for epoch 10 (beats in epoch 9) AND for epoch 9 (beats in 8).
        for e in [8, 9] {
            for w in ["w1", "w2"] {
                assert!(board.heartbeat(e, w));
            }
        }
        let cells = epoch_cells();
        let now = 600 * 10 + 5;
        let now_fn = move || now; // static clock — no boundary crossing
        let sleep_fn = |_s: u64| {};
        let view = LedgerView::new(); // both workers see the same (empty) snapshot
        let out1 = run_epoch_pass(
            &epoch_worker_cfg(&root, "w1", now, false),
            &cells,
            &view,
            &board,
            &now_fn,
            &sleep_fn,
        );
        let out2 = run_epoch_pass(
            &epoch_worker_cfg(&root, "w2", now, false),
            &cells,
            &view,
            &board,
            &now_fn,
            &sleep_fn,
        );
        // Disjoint + exhaustive: every cell executed exactly once across the fleet.
        assert_eq!(out1.done + out2.done, cells.len());
        assert_eq!(out1.skipped + out2.skipped, 0);
        let (ids1, ids2) = (row_ids(&out1), row_ids(&out2));
        assert!(ids1.is_disjoint(&ids2), "shards must not overlap");
        assert_eq!(ids1.len() + ids2.len(), cells.len());
        // Each worker executed exactly the cells the pure core assigns it.
        let roster = Roster::new(10, vec!["w1".into(), "w2".into()]);
        for c in &cells {
            let id = c.job_id();
            let owner = epoch::hrw_owner(&roster, id.as_str()).unwrap();
            let ran_by_w1 = ids1.contains(id.as_str());
            assert_eq!(ran_by_w1, owner == "w1", "cell must run on its HRW owner");
        }
        // THE point of the mode: the stable-roster fast path acquired zero leases.
        let claims_dir = root.join("blobs").join("_epoch").join("claims");
        let n_claims = std::fs::read_dir(&claims_dir)
            .map(|d| d.count())
            .unwrap_or(0);
        assert_eq!(n_claims, 0, "steady state must take no claim leases");
    }

    #[test]
    fn epoch_churn_lease_guards_exactly_the_moved_cells() {
        let root = tmp();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let board = LocalEpochBoard::new(root.join("blobs").join("_epoch")).unwrap();
        // w3 was alive in epoch 8 (so it's in epoch 9's roster) but died: no beat in epoch 9.
        for w in ["w1", "w2", "w3"] {
            assert!(board.heartbeat(8, w));
        }
        for w in ["w1", "w2"] {
            assert!(board.heartbeat(9, w));
        }
        let cells = epoch_cells();
        let now = 600 * 10 + 5;
        let now_fn = move || now;
        let sleep_fn = |_s: u64| {};
        let view = LedgerView::new();
        let out1 = run_epoch_pass(
            &epoch_worker_cfg(&root, "w1", now, false),
            &cells,
            &view,
            &board,
            &now_fn,
            &sleep_fn,
        );
        let out2 = run_epoch_pass(
            &epoch_worker_cfg(&root, "w2", now, false),
            &cells,
            &view,
            &board,
            &now_fn,
            &sleep_fn,
        );
        // Takeover is complete and still disjoint (w3's cells landed on exactly one survivor).
        assert_eq!(out1.done + out2.done, cells.len());
        assert!(row_ids(&out1).is_disjoint(&row_ids(&out2)));
        // Leases were spent on exactly the cells that MOVED (previous owner = the dead w3).
        let prev = Roster::new(9, vec!["w1".into(), "w2".into(), "w3".into()]);
        let moved: HashSet<String> = cells
            .iter()
            .map(|c| c.job_id())
            .filter(|id| epoch::hrw_owner(&prev, id.as_str()) == Some("w3"))
            .map(|id| id.as_str().to_string())
            .collect();
        assert!(!moved.is_empty(), "w3 must have owned something");
        let claims_dir = root.join("blobs").join("_epoch").join("claims");
        let claimed: HashSet<String> = std::fs::read_dir(&claims_dir)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        assert_eq!(claimed, moved, "leases cover the seam cells, nothing else");
    }

    #[test]
    fn epoch_tail_steal_takes_over_a_dead_workers_cells_lease_guarded() {
        let root = tmp();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let board = LocalEpochBoard::new(root.join("blobs").join("_epoch")).unwrap();
        // Stable roster {w1,w2} — but w2 is dead THIS epoch (roster won't reflect that until the
        // next boundary). w1 exhausts its shard, waits for the tail window, then steals.
        for e in [8, 9] {
            for w in ["w1", "w2"] {
                assert!(board.heartbeat(e, w));
            }
        }
        let cells = epoch_cells();
        let start = 600 * 10 + 5;
        let (_clock, now_fn, sleep_fn) = fake_time(start);
        let view = LedgerView::new();
        let out = run_epoch_pass(
            &epoch_worker_cfg(&root, "w1", start, true),
            &cells,
            &view,
            &board,
            &now_fn,
            &sleep_fn,
        );
        // Everything got done in ONE pass — own shard lease-free, w2's via lease-guarded steal.
        assert_eq!(out.done, cells.len());
        assert_eq!(row_ids(&out).len(), cells.len(), "no duplicate rows");
        let roster = Roster::new(10, vec!["w1".into(), "w2".into()]);
        let w2_cells: HashSet<String> = cells
            .iter()
            .map(|c| c.job_id())
            .filter(|id| epoch::hrw_owner(&roster, id.as_str()) == Some("w2"))
            .map(|id| id.as_str().to_string())
            .collect();
        assert!(!w2_cells.is_empty(), "w2 must have owned something");
        let claims_dir = root.join("blobs").join("_epoch").join("claims");
        let claimed: HashSet<String> = std::fs::read_dir(&claims_dir)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .collect();
        assert_eq!(
            claimed, w2_cells,
            "steals are lease-guarded; own shard is not"
        );
        // The wait-to-tail kept heartbeating (so the worker stays in the NEXT roster).
        assert!(board.roster_members(10).contains(&"w1".to_string()));
    }

    #[test]
    fn epoch_bootstrap_falls_back_to_lease_mode_and_joins_next_epoch() {
        let root = tmp();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let board = LocalEpochBoard::new(root.join("blobs").join("_epoch")).unwrap();
        // Nobody heartbeated yet — the roster is empty, so this pass runs the lease path.
        let cells = epoch_cells();
        let now = 600 * 10 + 5;
        let now_fn = move || now;
        let sleep_fn = |_s: u64| {};
        let out = run_epoch_pass(
            &epoch_worker_cfg(&root, "w1", now, true),
            &cells,
            &LedgerView::new(),
            &board,
            &now_fn,
            &sleep_fn,
        );
        assert_eq!(out.done, cells.len(), "bootstrap epoch still does the work");
        // The pass heartbeated first, so the NEXT epoch's roster includes this worker.
        let next = Roster::new(11, board.roster_members(10));
        assert!(next.contains("w1"));
    }

    #[test]
    fn resolve_epoch_cfg_control_object_wins() {
        let mine = Some(EpochShardCfg {
            epoch_len_secs: 300,
            heartbeat_interval_secs: 60,
            tail_steal: false,
        });
        // No control opinion → the worker's own config decides.
        assert_eq!(resolve_epoch_cfg(&RunControl::RUNNING, None), None);
        assert_eq!(resolve_epoch_cfg(&RunControl::RUNNING, mine), mine);
        // Control forces lease → epoch config is ignored.
        let mut ctl = RunControl::RUNNING;
        ctl.claim_mode = Some(ClaimMode::Lease);
        assert_eq!(resolve_epoch_cfg(&ctl, mine), None);
        // Control forces epoch-sharded → defaults when the worker had none, overrides applied.
        ctl.claim_mode = Some(ClaimMode::EpochSharded);
        assert_eq!(
            resolve_epoch_cfg(&ctl, None),
            Some(EpochShardCfg::default())
        );
        ctl.epoch_len_secs = Some(1200);
        let got = resolve_epoch_cfg(&ctl, mine).unwrap();
        assert_eq!(got.epoch_len_secs, 1200, "control override wins");
        assert_eq!(
            got.heartbeat_interval_secs, 60,
            "worker value kept where control is silent"
        );
        assert!(!got.tail_steal, "tail_steal stays worker-controlled");
    }

    #[test]
    fn epoch_mode_run_rejects_incoherent_configs() {
        let root = tmp();
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("m.json"), "[]").unwrap();
        let mut cfg = epoch_worker_cfg(&root, "w1", 6005, false);
        cfg.manifest = root.join("m.json");
        // Epoch mode on the serial per-cell path is refused (it rides the chunked executor).
        cfg.chunk_wall_sec = 0.0;
        assert!(matches!(run(&cfg), Err(WorkerRunError::Config(_))));
        // Epoch mode on R2 without a claims namespace is refused (no heartbeat/lease home).
        cfg.chunk_wall_sec = 2.0;
        cfg.r2 = Some(R2Target {
            endpoint: "http://127.0.0.1:1".into(),
            bucket: "b".into(),
            prefix: "blobs".into(),
        });
        cfg.claims = None;
        assert!(matches!(run(&cfg), Err(WorkerRunError::Config(_))));
    }

    #[test]
    fn handicap_registry_parses_and_is_sane() {
        // CI gate for the COMMITTED fleet/handicaps.toml: a registry that fails here would
        // otherwise fail loud-but-soft (uniform weights) on every box at once.
        let h = parse_handicaps_toml(EMBEDDED_HANDICAPS_TOML)
            .expect("committed fleet/handicaps.toml must parse");
        assert!(!h.is_empty(), "registry must carry the seeded rows");
        for (worker, row) in &h.0 {
            assert_eq!(
                *worker,
                worker_key(worker),
                "keys must be sanitized worker ids"
            );
            for (ty, w) in &row.encode {
                assert!(
                    w.is_finite() && (0.0..=100.0).contains(w),
                    "{worker}.encode.{ty} = {w} out of sane range"
                );
            }
            assert!(
                row.encode.contains_key("default"),
                "{worker} needs an encode.default"
            );
            for (name, w) in [("metric", row.metric), ("gpu_metric", row.gpu_metric)] {
                assert!(
                    w.is_finite() && (0.0..=100.0).contains(&w),
                    "{worker}.{name} = {w} out of sane range"
                );
            }
        }
        // The measured seed rows from the avifgen attribution audit (0f9fa073) are present.
        let mode = zenfleet_core::ShardMode::Encode { codec: "zenavif" };
        assert_eq!(h.weight("lianli", &mode), 1.0);
        assert_eq!(h.weight("node-3", &mode), 0.19);
        // Unmeasured types fall to default (no cross-type extrapolation).
        let jxl = zenfleet_core::ShardMode::Encode { codec: "zenjxl" };
        assert_eq!(h.weight("node-3", &jxl), 1.0);
        // Role exclusions: no-GPU boxes are excluded from gpu_metric sharding.
        assert_eq!(h.weight("tower", &zenfleet_core::ShardMode::GpuMetric), 0.0);
        assert_eq!(
            h.weight("node-2", &zenfleet_core::ShardMode::GpuMetric),
            1.0
        );
    }

    #[test]
    fn resolve_handicaps_precedence_control_overrides_registry() {
        // No control opinion → the committed registry (embedded) applies.
        let from_registry = resolve_handicaps(&RunControl::RUNNING);
        assert!(!from_registry.is_empty());
        // A control override REPLACES the registry wholesale (absent workers → 1.0).
        let mut ctl = RunControl::RUNNING;
        let mut m = std::collections::BTreeMap::new();
        m.insert(
            "only-worker".to_string(),
            zenfleet_core::WorkerHandicap {
                metric: 0.5,
                ..Default::default()
            },
        );
        ctl.worker_weights = Some(m);
        let h = resolve_handicaps(&ctl);
        assert_eq!(
            h.weight("only-worker", &zenfleet_core::ShardMode::Metric),
            0.5
        );
        assert_eq!(
            h.weight("lianli", &zenfleet_core::ShardMode::Metric),
            1.0,
            "override is wholesale — registry rows do not merge through"
        );
    }

    #[test]
    fn epoch_zero_weight_worker_owns_nothing_and_never_steals() {
        // Role specialization end-to-end: w2's weight for this manifest's mode is 0, so it must
        // execute nothing — no owned shard, no seam claims, and NO tail-steal (even with time
        // left in the epoch) — while w1 covers the whole set.
        let root = tmp();
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        let board = LocalEpochBoard::new(root.join("blobs").join("_epoch")).unwrap();
        for e in [8, 9] {
            for w in ["w1", "w2"] {
                assert!(board.heartbeat(e, w));
            }
        }
        let cells = epoch_cells(); // JobKind::Metric{"ssim2"} → ShardMode::Metric
        let mut h = Handicaps::default();
        h.0.insert(
            "w2".into(),
            zenfleet_core::WorkerHandicap {
                metric: 0.0,
                ..Default::default()
            },
        );
        let start = 600 * 10 + 5;
        let (_clock, now_fn, sleep_fn) = fake_time(start);
        let view = LedgerView::new();
        let out2 = run_epoch_pass_weighted(
            &epoch_worker_cfg(&root, "w2", start, true),
            &cells,
            &view,
            &board,
            &h,
            &now_fn,
            &sleep_fn,
        );
        assert_eq!(out2.done, 0, "excluded worker executes nothing");
        assert_eq!(out2.rows.len(), 0);
        let (_c1, now1, sleep1) = fake_time(start);
        let out1 = run_epoch_pass_weighted(
            &epoch_worker_cfg(&root, "w1", start, true),
            &cells,
            &view,
            &board,
            &h,
            &now1,
            &sleep1,
        );
        assert_eq!(
            out1.done,
            cells.len(),
            "the weighted peer absorbs the whole set"
        );
        let claims_dir = root.join("blobs").join("_epoch").join("claims");
        let n_claims = std::fs::read_dir(&claims_dir)
            .map(|d| d.count())
            .unwrap_or(0);
        assert_eq!(
            n_claims, 0,
            "sole-owner fast path takes no leases; excluded worker none"
        );
    }

    #[test]
    fn local_epoch_board_heartbeat_roster_and_claim_semantics() {
        let board = LocalEpochBoard::new(tmp()).unwrap();
        assert!(board.roster_members(4).is_empty());
        assert!(board.heartbeat(4, "wa"));
        assert!(board.heartbeat(4, "wb"));
        assert!(
            board.heartbeat(4, "wa"),
            "re-beat is an idempotent overwrite"
        );
        let mut got = board.roster_members(4);
        got.sort();
        assert_eq!(got, vec!["wa".to_string(), "wb".to_string()]);
        assert!(
            board.roster_members(5).is_empty(),
            "epochs are separate buckets"
        );
        // Fresh claim wins once; a live claim is not stealable; a stale one is.
        assert!(board.claim_cell("cellX", 1000, 600, "wa"));
        assert!(
            !board.claim_cell("cellX", 1100, 600, "wb"),
            "live claim holds"
        );
        assert!(
            board.claim_cell("cellX", 1600, 600, "wb"),
            "stale claim steals"
        );
    }

    #[test]
    fn worker_key_sanitizes_consistently() {
        assert_eq!(worker_key("node-2"), "node-2");
        assert_eq!(worker_key("vast/12 34"), "vast_12_34");
        assert_eq!(worker_key("a.b_c-9"), "a.b_c-9");
    }
}
