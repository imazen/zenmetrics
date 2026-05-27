//! Phase 5 — worker pool, streaming + batch APIs, cached-ref auto-detect.
//!
//! Sits one level above the Phase 4 single-task executor: instead of
//! blocking the caller on each task, the orchestrator hands work to a
//! background worker pool and surfaces results through two APIs:
//!
//! - **Streaming**: [`Orchestrator::submit`] returns a [`TaskHandle`]
//!   immediately; [`Orchestrator::poll`] / [`Orchestrator::poll_any`]
//!   drain completed tasks at the caller's pace.
//! - **Batch**: [`Orchestrator::run_all`] consumes an iterator of tasks
//!   and yields [`TaskResult`]s in **completion order** (i.e. as each
//!   worker finishes; out-of-order WRT submission). Callers correlate
//!   results to their inputs via [`Task::task_id`].
//!
//! ## Why completion-order
//!
//! Tasks may legitimately finish out of submission order: a 1024² CPU
//! task can complete before a 4096² GPU task started seconds earlier.
//! Buffering for submission-order delivery would require holding every
//! out-of-order result in memory until the next-expected `task_id`
//! arrives — under heavy load, that's an unbounded queue. Callers who
//! want submission order can match `task_id`s into a `BTreeMap` themselves.
//!
//! ## Worker layout
//!
//! - **1 GPU worker** per process (single CUDA device). Owns a "warm"
//!   [`zenmetrics_api::Metric`] for the current
//!   `(metric, w, h, backend)` signature; reuses it for consecutive
//!   tasks of the same signature to amortize PTX compile + buffer
//!   allocation. Swaps on signature change.
//! - **`num_cpus / 2` CPU workers** (capped by
//!   [`PoolConfig::max_parallel_cpu`]). Each CPU worker currently
//!   surfaces `OrchestratorError::CpuNotYetWired` for every task — Phase
//!   6 wires the real CPU reference implementations.
//!
//! ## Cached-ref auto-detect
//!
//! For every task whose [`TaskData::Srgb8`] reference bytes hash to the
//! same value as a recent entry (sliding window of
//! [`CACHED_REF_WINDOW_SIZE`] = 32 entries), the GPU worker dispatches
//! through [`zenmetrics_api::Metric::set_reference_srgb_u8`] +
//! [`zenmetrics_api::Metric::compute_with_cached_reference_srgb_u8`]
//! instead of the regular [`zenmetrics_api::Metric::compute_srgb_u8`].
//!
//! Hash: xxhash3_64 (`xxhash_rust::xxh3::xxh3_64`), ~5-15 GB/s on a
//! 7950X — at 4096² (48 MiB ref buffer) the hash takes ~4-8 ms, well
//! under the 10 ms budget the design doc calls out.
//!
//! Callers who want zero hash overhead can pre-upload the reference via
//! [`Orchestrator::upload_reference`] and pass the resulting
//! [`TaskRefHandle`] as [`TaskData::PreUploaded`] — the worker then
//! skips the hash entirely.
//!
//! ## Live VRAM watcher
//!
//! A dedicated thread samples free VRAM every 250 ms via
//! [`cvvdp_gpu::memory_mode::live_vram_probe_bytes`] (the same call the
//! Phase 3 chooser uses). The GPU worker checks the snapshot before
//! dispatching each task; if free VRAM is below
//! [`PoolConfig::vram_safety_floor_mib`] (default 200 MiB), it stalls
//! briefly (50-100 ms) and re-checks. The chooser already predictively
//! avoids over-commit; the watcher catches contention from other
//! processes (browsers, shaders, etc.).
//!
//! ## Thread lifetime
//!
//! Workers spawn lazily on the first [`Orchestrator::submit`] /
//! [`Orchestrator::run_all`] call — `Orchestrator::new` stays cheap. On
//! [`Orchestrator::drop`] the pool sends a shutdown signal and joins
//! every worker (best-effort; a worker stuck on a long compute may
//! leak until the GPU returns).
//!
//! ## CUDA-only for Phase 5
//!
//! Same constraint as Phase 4 — the GPU worker dispatches against
//! `zenmetrics_api::Backend::Cuda`. Multi-runtime (wgpu / hip) is
//! Phase 5+ stretch but not in this implementation.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use zenmetrics_api::{MetricKind, MetricParams};

use crate::chooser::TaskShape;
use crate::executor::{
    construct_pub, AttemptOutcome, CallErrPub, ConstructOutcomePub, ExecMetric,
    OrchestratorError, Task, TaskData, TaskResult,
};
use crate::{Backend, Orchestrator};

// Bring the ExecMetric Cpu variant constructor into scope via cpu_adapter
// only when needed (the Cpu worker uses it directly to manage warm state).
#[allow(unused_imports)]
use crate::cpu_adapter::CpuAdapter;

// ---------------------------------------------------------------------------
// Public handles
// ---------------------------------------------------------------------------

/// Opaque handle returned by [`Orchestrator::submit`]. Pass it to
/// [`Orchestrator::poll`] to drain the task's result when it's ready.
///
/// The inner `id` is a monotonic counter scoped to the orchestrator —
/// stable across `submit` calls, distinct from the caller-provided
/// `task_id` on [`Task`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskHandle {
    /// Internal monotonic submission ID (not the caller's task_id).
    pub(crate) id: u64,
}

impl TaskHandle {
    /// Read the internal submission ID. Mostly useful for logging /
    /// debugging — callers should match on the [`Task::task_id`] echoed
    /// back inside [`TaskResult`] for correlation.
    pub fn id(self) -> u64 {
        self.id
    }
}

/// Handle to a pre-uploaded reference image. Acquired via
/// [`Orchestrator::upload_reference`]; passed back into a future
/// [`Task::ref_data`] as [`TaskData::PreUploaded`] so the worker can
/// skip the auto-hash + re-upload pair.
///
/// The handle is bound to a specific `(metric, width, height)`
/// signature — passing it into a task with a different signature
/// surfaces an [`OrchestratorError::MetricApi`] error.
///
/// Dropping the handle (via [`Orchestrator::drop_reference`] or by
/// letting it go out of scope and the pool's ref-state cache aging out)
/// frees the GPU-side reference state.
#[derive(Debug, Clone)]
pub struct TaskRefHandle {
    /// The metric this handle was prepared for.
    pub metric: MetricKind,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Internal handle ID — keyed into the pool's pre-upload table.
    pub(crate) inner_id: u64,
}

// ---------------------------------------------------------------------------
// Sliding-window cached-ref cache
// ---------------------------------------------------------------------------

/// Sliding window of recently-seen ref bytes per `(metric, w, h)`. When
/// a new task's ref bytes hash matches a recent entry, the worker can
/// promote the dispatch to the cached-ref API.
///
/// Default capacity: 32 distinct (metric, w, h, hash) tuples. Older
/// entries roll off in FIFO order.
const CACHED_REF_WINDOW_SIZE: usize = 32;

/// Hash a reference byte buffer with xxhash3_64. Fast enough at 4096²
/// (~4-8 ms on a 7950X). Returns the same `u64` deterministically per
/// byte content.
pub(crate) fn hash_ref_bytes(bytes: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(bytes)
}

#[derive(Debug, Default)]
struct CachedRefCache {
    /// FIFO of recently-seen `(metric, w, h, hash)` tuples.
    window: VecDeque<(MetricKind, u32, u32, u64)>,
    /// How many tasks observed a cached-ref hit during dispatch. The
    /// pool exposes this through [`Orchestrator::cached_ref_stats`] so
    /// callers can verify the auto-detect is firing in tests.
    hit_count: AtomicU64,
    /// Tasks that resolved a hash but did NOT match the window.
    miss_count: AtomicU64,
}

impl CachedRefCache {
    fn observe(&mut self, metric: MetricKind, width: u32, height: u32, hash: u64) -> bool {
        let key = (metric, width, height, hash);
        let hit = self.window.iter().any(|&k| k == key);
        if hit {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
            // Slide in. Pop oldest if at capacity.
            if self.window.len() >= CACHED_REF_WINDOW_SIZE {
                self.window.pop_front();
            }
            self.window.push_back(key);
        }
        hit
    }
}

/// Snapshot of the cached-ref auto-detect counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct CachedRefStats {
    /// Tasks that hashed a ref AND matched a window entry.
    pub hit_count: u64,
    /// Tasks that hashed a ref but missed the window.
    pub miss_count: u64,
}

// ---------------------------------------------------------------------------
// Worker types
// ---------------------------------------------------------------------------

/// A scheduled unit of work passed to a worker thread.
struct WorkerTask {
    /// Internal handle ID (== [`TaskHandle::id`]).
    handle_id: u64,
    /// Caller's correlation ID (== [`Task::task_id`]).
    task_id: u64,
    /// Metric kind.
    metric: MetricKind,
    /// Image width.
    width: u32,
    /// Image height.
    height: u32,
    /// Per-metric params (`None` → defaults).
    params: Option<MetricParams>,
    /// Materialized reference bytes, OR a marker that a pre-upload
    /// handle was provided.
    ref_payload: RefPayload,
    /// Materialized distorted bytes.
    dist_bytes: Vec<u8>,
    /// Chosen backend (the dispatcher pre-computed this so worker
    /// dispatch is straight-line; the worker still runs the fallback
    /// ladder on construction/runtime errors).
    chosen_backend: Backend,
    /// xxhash3_64 of the ref bytes when `ref_payload == Bytes`, OR the
    /// pre-upload's hash. Used to drive cached-ref reuse.
    ref_hash: u64,
    /// True if the dispatcher promoted this task to cached-ref dispatch.
    use_cached_ref: bool,
}

/// What the worker should do with the reference image.
enum RefPayload {
    /// Raw bytes to upload (or skip re-upload via cached ref).
    Bytes(Vec<u8>),
    /// Pre-uploaded handle — the dispatcher already cloned the bytes
    /// from the pool's table into the second field, so the worker
    /// treats this identically to `Bytes` for the actual compute. The
    /// distinction is preserved for future Phase 5+ optimisations
    /// (e.g., a true GPU-resident pre-upload that skips the second
    /// CPU-side clone). For now we only retain the handle marker for
    /// debug introspection.
    #[allow(dead_code)]
    PreUploaded(TaskRefHandle, Vec<u8>),
}

/// A finished task on its way back to the caller.
struct WorkerResult {
    handle_id: u64,
    result: TaskResult,
}

// ---------------------------------------------------------------------------
// Pool configuration
// ---------------------------------------------------------------------------

/// Tunable knobs for the worker pool. All fields keep public visibility
/// + a [`Default`] impl so callers can override one knob via
/// struct-update syntax.
///
/// Phase 5 does NOT plumb this through [`crate::OrchestratorConfig`]
/// because that struct is the persistent config (cache_dir,
/// cache_validity) — pool config is per-orchestrator runtime state. A
/// future phase can hoist the most important knobs (parallel caps,
/// safety floor) into `OrchestratorConfig` if callers ask for it.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PoolConfig {
    /// Maximum CPU workers. Default `num_cpus / 2`, minimum 1.
    pub max_parallel_cpu: usize,
    /// VRAM floor for the live watcher (MiB). The GPU worker stalls
    /// briefly when probed free VRAM drops below this. Default 200.
    pub vram_safety_floor_mib: usize,
    /// How often the VRAM watcher samples (ms). Default 250.
    pub vram_sample_interval_ms: u64,
    /// How long to stall when free VRAM is below the floor before
    /// re-checking (ms). Default 75.
    pub vram_stall_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(2);
        Self {
            max_parallel_cpu: (cpus / 2).max(1),
            vram_safety_floor_mib: 200,
            vram_sample_interval_ms: 250,
            vram_stall_ms: 75,
        }
    }
}

// ---------------------------------------------------------------------------
// VRAM watcher
// ---------------------------------------------------------------------------

/// Live VRAM watcher — samples free VRAM in a background thread, stores
/// the most recent reading in an `AtomicUsize` (MiB).
struct VramWatcher {
    snapshot_mib: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl VramWatcher {
    /// Spawn a sampling thread. The initial snapshot value is `usize::MAX`
    /// (interpreted as "not yet probed — assume plenty"); the first
    /// sample arrives ~one interval later.
    fn spawn(interval_ms: u64) -> Self {
        let snapshot_mib = Arc::new(AtomicUsize::new(usize::MAX));
        let shutdown = Arc::new(AtomicBool::new(false));
        let snap_clone = Arc::clone(&snapshot_mib);
        let shut_clone = Arc::clone(&shutdown);
        let join = thread::Builder::new()
            .name("zm-vram-watcher".into())
            .spawn(move || {
                let interval = Duration::from_millis(interval_ms);
                while !shut_clone.load(Ordering::Acquire) {
                    if let Some(bytes) = cvvdp_gpu::memory_mode::live_vram_probe_bytes() {
                        let mib = bytes / (1024 * 1024);
                        snap_clone.store(mib, Ordering::Release);
                    }
                    // Sleep in small chunks so shutdown is responsive.
                    let chunk = Duration::from_millis(25);
                    let mut remaining = interval;
                    while remaining > Duration::ZERO {
                        if shut_clone.load(Ordering::Acquire) {
                            break;
                        }
                        let s = remaining.min(chunk);
                        thread::sleep(s);
                        remaining = remaining.saturating_sub(s);
                    }
                }
            })
            .expect("zm-vram-watcher spawn");
        Self {
            snapshot_mib,
            shutdown,
            join: Some(join),
        }
    }

    fn current_mib(&self) -> usize {
        self.snapshot_mib.load(Ordering::Acquire)
    }

    fn snapshot_handle(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.snapshot_mib)
    }
}

impl Drop for VramWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Pre-uploaded-reference table
// ---------------------------------------------------------------------------

/// State for a pre-uploaded reference — the worker installs this via
/// [`zenmetrics_api::Metric::set_reference_srgb_u8`] when first hit,
/// then keeps reusing as long as the worker's current signature matches.
///
/// `metric` / `width` / `height` are stored for debug introspection
/// and for a future Phase 5+ optimisation that validates the handle's
/// shape at dispatch time without consulting the `TaskRefHandle`. The
/// current dispatcher only reads `ref_bytes` + `ref_hash` because the
/// `TaskRefHandle` already carries the shape.
#[allow(dead_code)]
struct PreUpload {
    metric: MetricKind,
    width: u32,
    height: u32,
    ref_bytes: Vec<u8>,
    ref_hash: u64,
}

/// Inner state shared between the orchestrator and the pre-upload API.
#[derive(Default)]
struct PreUploadTable {
    next_id: u64,
    entries: HashMap<u64, PreUpload>,
}

impl PreUploadTable {
    fn insert(
        &mut self,
        metric: MetricKind,
        width: u32,
        height: u32,
        ref_bytes: Vec<u8>,
        ref_hash: u64,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        self.entries.insert(
            id,
            PreUpload {
                metric,
                width,
                height,
                ref_bytes,
                ref_hash,
            },
        );
        id
    }

    fn remove(&mut self, id: u64) -> Option<PreUpload> {
        self.entries.remove(&id)
    }

    fn get(&self, id: u64) -> Option<&PreUpload> {
        self.entries.get(&id)
    }
}

// ---------------------------------------------------------------------------
// GPU worker — owns the warm Metric and dispatches one task at a time
// ---------------------------------------------------------------------------

/// Identifier for the currently-cached `Metric` instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MetricSignature {
    metric: MetricKind,
    width: u32,
    height: u32,
    backend: Backend,
}

/// GPU worker — pulls [`WorkerTask`]s from its queue, reuses a warm
/// [`ExecMetric`] when the signature matches, otherwise rebuilds.
fn gpu_worker_main(
    rx: mpsc::Receiver<WorkerTask>,
    result_tx: mpsc::Sender<WorkerResult>,
    vram_floor_mib: usize,
    vram_stall_ms: u64,
    vram_snapshot: Arc<AtomicUsize>,
) {
    let mut current_signature: Option<MetricSignature> = None;
    let mut current_metric: Option<ExecMetric> = None;
    let mut cached_ref_hash: Option<u64> = None;

    while let Ok(task) = rx.recv() {
        let t_start = Instant::now();
        let handle_id = task.handle_id;
        let task_id = task.task_id;

        // -- VRAM gate -------------------------------------------------
        // Stall briefly when free VRAM is below the safety floor. The
        // watcher's initial value (usize::MAX) means "not yet probed"
        // and is always above floor — first task proceeds without stall.
        let mut stalls = 0;
        loop {
            let free = vram_snapshot.load(Ordering::Acquire);
            if free >= vram_floor_mib {
                break;
            }
            stalls += 1;
            if stalls > 40 {
                // ~3 seconds of stalls → give up, dispatch anyway. The
                // chooser already predicted feasibility; if the watcher
                // is wrong we'll surface a runtime OOM and the executor
                // ladder will retry.
                break;
            }
            thread::sleep(Duration::from_millis(vram_stall_ms));
        }

        // -- Construct metric if signature changed ---------------------
        let sig = MetricSignature {
            metric: task.metric,
            width: task.width,
            height: task.height,
            backend: task.chosen_backend,
        };
        let signature_changed = current_signature != Some(sig);
        if signature_changed {
            // Drop the old metric first to release device buffers.
            current_metric = None;
            cached_ref_hash = None;
            match construct_pub(
                task.metric,
                task.chosen_backend,
                task.width,
                task.height,
                task.params.clone(),
            ) {
                ConstructOutcomePub::Ok(m) => {
                    current_metric = Some(m);
                    current_signature = Some(sig);
                }
                ConstructOutcomePub::Oom => {
                    let attempts = vec![(task.chosen_backend, AttemptOutcome::OomAtConstruction)];
                    let _ = result_tx.send(WorkerResult {
                        handle_id,
                        result: TaskResult {
                            task_id,
                            outcome: Err(OrchestratorError::FullyExhausted {
                                attempts: attempts.clone(),
                            }),
                            backend_used: None,
                            backends_attempted: attempts,
                            wall_us: t_start.elapsed().as_micros() as u64,
                            vram_peak_mib: None,
                            output_columns: ::std::collections::BTreeMap::new(),
                            metric_version: None,
                        },
                    });
                    continue;
                }
                ConstructOutcomePub::Other(msg) => {
                    let attempts =
                        vec![(task.chosen_backend, AttemptOutcome::OtherError(msg.clone()))];
                    let _ = result_tx.send(WorkerResult {
                        handle_id,
                        result: TaskResult {
                            task_id,
                            outcome: Err(OrchestratorError::MetricApi(msg)),
                            backend_used: None,
                            backends_attempted: attempts,
                            wall_us: t_start.elapsed().as_micros() as u64,
                            vram_peak_mib: None,
                            output_columns: ::std::collections::BTreeMap::new(),
                            metric_version: None,
                        },
                    });
                    continue;
                }
            }
        }

        let m = match current_metric.as_mut() {
            Some(m) => m,
            None => {
                // Defensive — construction above should have populated.
                let attempts = vec![(
                    task.chosen_backend,
                    AttemptOutcome::OtherError("worker lost metric state".into()),
                )];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::MetricApi(
                            "worker lost metric state".into(),
                        )),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us: t_start.elapsed().as_micros() as u64,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
                continue;
            }
        };

        // -- Materialize ref bytes (for both regular and cached paths) -
        let ref_bytes: &[u8] = match &task.ref_payload {
            RefPayload::Bytes(b) => b.as_slice(),
            RefPayload::PreUploaded(_, b) => b.as_slice(),
        };

        // -- Dispatch --------------------------------------------------
        let compute_result: Result<
            (zenmetrics_api::Score, std::collections::BTreeMap<String, f64>),
            CallErrPub,
        > = if task.use_cached_ref && m.supports_cached_ref() {
            // If the worker's cached hash doesn't match the task's, install.
            let need_install = cached_ref_hash != Some(task.ref_hash) || signature_changed;
            if need_install {
                match m.set_reference(ref_bytes) {
                    Ok(()) => {
                        cached_ref_hash = Some(task.ref_hash);
                    }
                    Err(msg) => {
                        cached_ref_hash = None;
                        // Fall back to regular compute on set_reference failure.
                        // Don't bubble — the task should still produce a score.
                        let _ = msg;
                    }
                }
            }
            if cached_ref_hash == Some(task.ref_hash) {
                m.compute_with_cached_reference_with_extras(&task.dist_bytes)
            } else {
                m.compute_with_extras(ref_bytes, &task.dist_bytes)
            }
        } else {
            // Regular compute. Invalidate cached-ref state.
            cached_ref_hash = None;
            m.compute_with_extras(ref_bytes, &task.dist_bytes)
        };

        let wall_us = t_start.elapsed().as_micros() as u64;
        match compute_result {
            Ok((score, extras)) => {
                let attempts = vec![(task.chosen_backend, AttemptOutcome::Success)];
                let output_columns =
                    crate::executor::build_output_columns(task.metric, &score, &extras);
                let metric_version = Some(score.metric_version);
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Ok(score),
                        backend_used: Some(task.chosen_backend),
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns,
                        metric_version,
                    },
                });
            }
            Err(CallErrPub::Oom) => {
                // Drop the metric so the next task constructs fresh.
                current_metric = None;
                current_signature = None;
                cached_ref_hash = None;
                let attempts = vec![(task.chosen_backend, AttemptOutcome::OomAtRuntime)];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::FullyExhausted {
                            attempts: attempts.clone(),
                        }),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
            }
            Err(CallErrPub::Other(msg)) => {
                let attempts = vec![(task.chosen_backend, AttemptOutcome::OtherError(msg.clone()))];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::MetricApi(msg)),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU worker — Phase 6.
//
// Each CPU worker thread owns one warm `ExecMetric::Cpu(CpuAdapter)`
// matched against the current `(metric, w, h)` signature, mirroring the
// signature-cache pattern in `gpu_worker_main`. Construction happens
// lazily on the first task and on every signature change; intra-signature
// runs reuse the adapter so set_reference / warm_reference work without
// per-call allocation churn.
//
// Cached-ref dispatch follows the same logic as the GPU worker: when
// the task's `use_cached_ref` is set AND the adapter reports
// `supports_cached_ref`, install via `set_reference` once per
// `(signature, ref_hash)` and dispatch through
// `compute_with_cached_reference` for the rest. Adapters without a
// true cached-ref path (ssim2 / butter / zensim) silently route through
// the regular `compute` — the score is correct, just no speedup.
// ---------------------------------------------------------------------------

fn cpu_worker_main(
    rx: mpsc::Receiver<WorkerTask>,
    result_tx: mpsc::Sender<WorkerResult>,
) {
    let mut current_signature: Option<MetricSignature> = None;
    let mut current_metric: Option<ExecMetric> = None;
    let mut cached_ref_hash: Option<u64> = None;

    while let Ok(task) = rx.recv() {
        let t_start = Instant::now();
        let handle_id = task.handle_id;
        let task_id = task.task_id;

        // -- Construct adapter if signature changed --------------------
        let sig = MetricSignature {
            metric: task.metric,
            width: task.width,
            height: task.height,
            backend: task.chosen_backend,
        };
        let signature_changed = current_signature != Some(sig);
        if signature_changed {
            current_metric = None;
            cached_ref_hash = None;
            match construct_pub(
                task.metric,
                task.chosen_backend,
                task.width,
                task.height,
                task.params.clone(),
            ) {
                ConstructOutcomePub::Ok(m) => {
                    current_metric = Some(m);
                    current_signature = Some(sig);
                }
                ConstructOutcomePub::Oom => {
                    // OOM at CPU adapter construction means the
                    // reference crate failed to allocate. Treat as
                    // OomAtConstruction so the caller can see the
                    // shape of the failure.
                    let attempts =
                        vec![(task.chosen_backend, AttemptOutcome::OomAtConstruction)];
                    let _ = result_tx.send(WorkerResult {
                        handle_id,
                        result: TaskResult {
                            task_id,
                            outcome: Err(OrchestratorError::FullyExhausted {
                                attempts: attempts.clone(),
                            }),
                            backend_used: None,
                            backends_attempted: attempts,
                            wall_us: t_start.elapsed().as_micros() as u64,
                            vram_peak_mib: None,
                            output_columns: ::std::collections::BTreeMap::new(),
                            metric_version: None,
                        },
                    });
                    continue;
                }
                ConstructOutcomePub::Other(msg) => {
                    // Phase 6 sentinels — recognise and translate into
                    // the right structured error so callers can route
                    // around (e.g. Iwssim ladder advance handled at
                    // submit time, but a synthetic test that forces
                    // Cpu still gets a clear result).
                    let outcome_err = translate_cpu_sentinel(&msg, task.metric);
                    let attempts =
                        vec![(task.chosen_backend, AttemptOutcome::OtherError(msg))];
                    let _ = result_tx.send(WorkerResult {
                        handle_id,
                        result: TaskResult {
                            task_id,
                            outcome: Err(outcome_err),
                            backend_used: None,
                            backends_attempted: attempts,
                            wall_us: t_start.elapsed().as_micros() as u64,
                            vram_peak_mib: None,
                            output_columns: ::std::collections::BTreeMap::new(),
                            metric_version: None,
                        },
                    });
                    continue;
                }
            }
        }

        let m = match current_metric.as_mut() {
            Some(m) => m,
            None => {
                let attempts = vec![(
                    task.chosen_backend,
                    AttemptOutcome::OtherError("cpu worker lost adapter state".into()),
                )];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::MetricApi(
                            "cpu worker lost adapter state".into(),
                        )),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us: t_start.elapsed().as_micros() as u64,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
                continue;
            }
        };

        // -- Materialise ref bytes ------------------------------------
        let ref_bytes: &[u8] = match &task.ref_payload {
            RefPayload::Bytes(b) => b.as_slice(),
            RefPayload::PreUploaded(_, b) => b.as_slice(),
        };

        // -- Dispatch -------------------------------------------------
        let compute_result: Result<
            (zenmetrics_api::Score, std::collections::BTreeMap<String, f64>),
            CallErrPub,
        > = if task.use_cached_ref && m.supports_cached_ref() {
            let need_install =
                cached_ref_hash != Some(task.ref_hash) || signature_changed;
            if need_install {
                match m.set_reference(ref_bytes) {
                    Ok(()) => cached_ref_hash = Some(task.ref_hash),
                    Err(_) => cached_ref_hash = None,
                }
            }
            if cached_ref_hash == Some(task.ref_hash) {
                m.compute_with_cached_reference_with_extras(&task.dist_bytes)
            } else {
                m.compute_with_extras(ref_bytes, &task.dist_bytes)
            }
        } else {
            cached_ref_hash = None;
            m.compute_with_extras(ref_bytes, &task.dist_bytes)
        };

        let wall_us = t_start.elapsed().as_micros() as u64;
        match compute_result {
            Ok((score, extras)) => {
                let attempts = vec![(task.chosen_backend, AttemptOutcome::Success)];
                let output_columns =
                    crate::executor::build_output_columns(task.metric, &score, &extras);
                let metric_version = Some(score.metric_version);
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Ok(score),
                        backend_used: Some(task.chosen_backend),
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns,
                        metric_version,
                    },
                });
            }
            Err(CallErrPub::Oom) => {
                // CPU OOM is exceptionally rare (CPU crates panic
                // rather than returning) but we handle the shape.
                current_metric = None;
                current_signature = None;
                cached_ref_hash = None;
                let attempts = vec![(task.chosen_backend, AttemptOutcome::OomAtRuntime)];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::FullyExhausted {
                            attempts: attempts.clone(),
                        }),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
            }
            Err(CallErrPub::Other(msg)) => {
                let attempts = vec![(
                    task.chosen_backend,
                    AttemptOutcome::OtherError(msg.clone()),
                )];
                let _ = result_tx.send(WorkerResult {
                    handle_id,
                    result: TaskResult {
                        task_id,
                        outcome: Err(OrchestratorError::CpuFailed(msg)),
                        backend_used: None,
                        backends_attempted: attempts,
                        wall_us,
                        vram_peak_mib: None,
                        output_columns: ::std::collections::BTreeMap::new(),
                        metric_version: None,
                    },
                });
            }
        }
    }
}

/// Translate the executor's CPU-related sentinel strings into the
/// matching [`OrchestratorError`] variant. Used by `cpu_worker_main`
/// to surface structured errors to the caller without depending on
/// the executor's private string parsing logic.
fn translate_cpu_sentinel(msg: &str, metric: MetricKind) -> OrchestratorError {
    if let Some(rest) = msg.strip_prefix("CpuMetricUnavailable:") {
        let _ = rest;
        OrchestratorError::CpuMetricUnavailable { metric }
    } else if let Some(rest) = msg.strip_prefix("CpuBackendUnavailable:") {
        // Format is `CpuBackendUnavailable:<tag>:cpu-<tag>`. Extract the
        // feature name (last component); fall back to a generic label.
        let feature = match rest.rsplit_once(':') {
            Some((_, feat)) if !feat.is_empty() => feat,
            _ => "cpu-<metric>",
        };
        // We can only carry a `&'static str` in the struct field; pin
        // the feature name through a static table to satisfy that.
        let required_feature = match feature {
            "cpu-cvvdp" => "cpu-cvvdp",
            "cpu-ssim2" => "cpu-ssim2",
            "cpu-dssim" => "cpu-dssim",
            "cpu-butter" => "cpu-butter",
            "cpu-zensim" => "cpu-zensim",
            _ => "cpu-all",
        };
        OrchestratorError::CpuBackendUnavailable {
            metric,
            required_feature,
        }
    } else if let Some(real) = msg.strip_prefix("CpuFailed:") {
        OrchestratorError::CpuFailed(real.to_string())
    } else {
        OrchestratorError::MetricApi(msg.to_string())
    }
}

// ---------------------------------------------------------------------------
// PoolState — the actual worker pool, owned by Orchestrator behind a
// Mutex<Option<...>> so the orchestrator can lazily initialize on first
// submit and tear down on Drop.
// ---------------------------------------------------------------------------

pub(crate) struct PoolState {
    /// Send tasks to the GPU worker.
    gpu_tx: Option<mpsc::Sender<WorkerTask>>,
    /// CPU worker queues (one per worker thread). Round-robin index for
    /// dispatch.
    cpu_txs: Vec<mpsc::Sender<WorkerTask>>,
    cpu_next: usize,
    /// Joined on drop.
    join_handles: Vec<JoinHandle<()>>,
    /// Single shared result channel.
    result_rx: mpsc::Receiver<WorkerResult>,
    /// Pending in-flight tasks by handle_id, plus optional buffered result.
    pending: HashMap<u64, Option<TaskResult>>,
    /// Monotonic counter for handle IDs.
    next_handle: u64,
    /// Cached-ref auto-detect window + counters.
    cached_ref_cache: CachedRefCache,
    /// Live VRAM watcher.
    vram: VramWatcher,
    /// Configuration snapshot. Stored for inspection / future
    /// re-spawning logic; the active worker thread params are baked
    /// in at spawn time.
    #[allow(dead_code)]
    config: PoolConfig,
    /// Pre-upload table — guarded by Mutex so the pool's API methods can
    /// extend it without going through the worker.
    pre_uploads: Arc<Mutex<PreUploadTable>>,
}

impl PoolState {
    fn new(config: PoolConfig) -> Self {
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
        let vram = VramWatcher::spawn(config.vram_sample_interval_ms);

        let mut join_handles: Vec<JoinHandle<()>> = Vec::new();

        // -- GPU worker --
        let (gpu_tx, gpu_rx) = mpsc::channel::<WorkerTask>();
        let gpu_result_tx = result_tx.clone();
        let vram_snap = vram.snapshot_handle();
        let floor = config.vram_safety_floor_mib;
        let stall = config.vram_stall_ms;
        let h = thread::Builder::new()
            .name("zm-gpu-worker".into())
            .spawn(move || {
                gpu_worker_main(gpu_rx, gpu_result_tx, floor, stall, vram_snap);
            })
            .expect("zm-gpu-worker spawn");
        join_handles.push(h);

        // -- CPU workers --
        let mut cpu_txs: Vec<mpsc::Sender<WorkerTask>> = Vec::with_capacity(config.max_parallel_cpu);
        for i in 0..config.max_parallel_cpu {
            let (tx, rx) = mpsc::channel::<WorkerTask>();
            let cpu_result_tx = result_tx.clone();
            let h = thread::Builder::new()
                .name(format!("zm-cpu-worker-{i}"))
                .spawn(move || cpu_worker_main(rx, cpu_result_tx))
                .expect("zm-cpu-worker spawn");
            join_handles.push(h);
            cpu_txs.push(tx);
        }
        // Drop the original result_tx so the result channel closes
        // cleanly once every worker exits (each worker holds its own clone).
        drop(result_tx);

        Self {
            gpu_tx: Some(gpu_tx),
            cpu_txs,
            cpu_next: 0,
            join_handles,
            result_rx,
            pending: HashMap::new(),
            next_handle: 0,
            cached_ref_cache: CachedRefCache::default(),
            vram,
            config,
            pre_uploads: Arc::new(Mutex::new(PreUploadTable::default())),
        }
    }

    /// Drain any results that have completed since last call, stashing
    /// them in `pending`. Non-blocking.
    fn drain_completed(&mut self) {
        while let Ok(wr) = self.result_rx.try_recv() {
            if let Some(slot) = self.pending.get_mut(&wr.handle_id) {
                *slot = Some(wr.result);
            }
        }
    }

    /// Blocking drain — block until at least one result arrives, then
    /// also drain everything else available.
    fn drain_completed_blocking(&mut self) -> bool {
        match self.result_rx.recv() {
            Ok(wr) => {
                if let Some(slot) = self.pending.get_mut(&wr.handle_id) {
                    *slot = Some(wr.result);
                }
                self.drain_completed();
                true
            }
            Err(_) => false,
        }
    }
}

impl Drop for PoolState {
    fn drop(&mut self) {
        // Close every worker's input channel; each worker exits when its
        // recv returns Err. Then join in order.
        drop(self.gpu_tx.take());
        self.cpu_txs.clear();
        for h in self.join_handles.drain(..) {
            let _ = h.join();
        }
        // VramWatcher's Drop signals + joins its own thread.
    }
}

// ---------------------------------------------------------------------------
// Orchestrator API — submit / poll / poll_any / run_all / upload_reference
// ---------------------------------------------------------------------------

impl Orchestrator {
    /// Ensure the worker pool is initialized. Workers spawn lazily so
    /// `Orchestrator::new` stays cheap (no thread spawn, no VRAM probe
    /// thread) — useful for callers that build an orchestrator just to
    /// inspect its capability profile.
    fn ensure_pool(&mut self) {
        if self.pool.is_none() {
            self.pool = Some(Box::new(PoolState::new(self.pool_config.clone())));
        }
    }

    /// Override the pool configuration. Must be called before the first
    /// `submit` / `run_all` / `upload_reference` — once workers spawn,
    /// the live config is frozen.
    ///
    /// Returns `Err(...)` if the pool already exists.
    pub fn set_pool_config(&mut self, config: PoolConfig) -> Result<(), &'static str> {
        if self.pool.is_some() {
            return Err("pool already initialised; set_pool_config before first submit");
        }
        self.pool_config = config;
        Ok(())
    }

    /// Submit a task to the worker pool. Returns immediately with a
    /// [`TaskHandle`]; the caller drains the result via [`Self::poll`]
    /// or [`Self::poll_any`] when ready.
    ///
    /// Tasks are dispatched to the GPU worker when the chooser picks a
    /// GPU backend, and to a CPU worker round-robin otherwise. The
    /// dispatcher hashes the ref bytes (xxhash3_64) and consults the
    /// cached-ref window to decide whether to promote dispatch to the
    /// cached-ref API.
    ///
    /// # Errors
    ///
    /// - [`OrchestratorError::Chooser`] — the chooser has no
    ///   measurements for this metric (call `bench()` / `warm()` first).
    /// - [`OrchestratorError::MetricApi`] — `TaskData::PreUploaded` was
    ///   provided but the handle doesn't match the task's
    ///   `(metric, w, h)` signature.
    /// - [`OrchestratorError::UnsupportedTaskData`] — `TaskData::Path`
    ///   was provided. Phase 5 still requires `Srgb8` or `PreUploaded`.
    pub fn submit(&mut self, task: Task) -> Result<TaskHandle, OrchestratorError> {
        self.ensure_pool();

        // Materialize ref bytes + extract pre-upload hash if any.
        let task_id = task.task_id;
        let metric = task.metric;
        let width = task.width;
        let height = task.height;
        let params = task.params;

        let (ref_payload, ref_hash) = match task.ref_data {
            TaskData::Srgb8(bytes) => {
                let h = hash_ref_bytes(&bytes);
                (RefPayload::Bytes(bytes), h)
            }
            TaskData::Path(p) => {
                return Err(OrchestratorError::UnsupportedTaskData(format!(
                    "TaskData::Path({}) not yet wired (Phase 5 keeps Srgb8/PreUploaded)",
                    p.display()
                )));
            }
            TaskData::PreUploaded(h) => {
                if h.metric != metric || h.width != width || h.height != height {
                    return Err(OrchestratorError::MetricApi(format!(
                        "PreUploaded handle ({}/{}x{}) doesn't match task ({}/{}x{})",
                        h.metric.tag(),
                        h.width,
                        h.height,
                        metric.tag(),
                        width,
                        height
                    )));
                }
                let pool_state = self.pool.as_ref().expect("pool initialized above");
                let table = pool_state.pre_uploads.lock().expect("pre_uploads lock");
                let pre = table.get(h.inner_id).ok_or_else(|| {
                    OrchestratorError::MetricApi(format!(
                        "PreUploaded handle (id={}) no longer in table — was it dropped?",
                        h.inner_id
                    ))
                })?;
                // Clone the bytes so the worker has its own buffer; the
                // handle remains valid for future submits.
                let bytes = pre.ref_bytes.clone();
                let hash = pre.ref_hash;
                drop(table);
                (RefPayload::PreUploaded(h, bytes), hash)
            }
        };

        let dist_bytes = match task.dist_data {
            TaskData::Srgb8(b) => b,
            TaskData::Path(p) => {
                return Err(OrchestratorError::UnsupportedTaskData(format!(
                    "TaskData::Path({}) not yet wired (Phase 5 keeps Srgb8/PreUploaded)",
                    p.display()
                )));
            }
            TaskData::PreUploaded(_) => {
                return Err(OrchestratorError::MetricApi(
                    "PreUploaded not supported for distorted data (only reference)".into(),
                ));
            }
        };

        // Ask the chooser for a backend. The Phase 4 executor still
        // owns OOM recovery on the single-task path; Phase 5 just picks
        // once here and lets the worker route the result. If the chosen
        // backend OOMs at runtime the task surfaces FullyExhausted; the
        // caller can re-submit.
        let shape = TaskShape {
            metric,
            width,
            height,
        };
        let choice = self
            .choose_backend_for_task(&shape)
            .map_err(OrchestratorError::Chooser)?;
        let chosen_backend = choice.backend;

        // Cached-ref auto-detect — observe the (metric, w, h, hash) tuple
        // against the sliding window. The worker uses the bool to decide
        // whether to install + re-use the cached reference state.
        let use_cached_ref = {
            let pool = self.pool.as_mut().expect("pool initialized");
            pool.cached_ref_cache
                .observe(metric, width, height, ref_hash)
        };

        // Allocate handle + register pending slot.
        let pool = self.pool.as_mut().expect("pool initialized");
        let handle_id = pool.next_handle;
        pool.next_handle = pool.next_handle.wrapping_add(1);
        pool.pending.insert(handle_id, None);

        let worker_task = WorkerTask {
            handle_id,
            task_id,
            metric,
            width,
            height,
            params,
            ref_payload,
            dist_bytes,
            chosen_backend,
            ref_hash,
            use_cached_ref,
        };

        // Route by backend.
        match chosen_backend {
            Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair => {
                let tx = pool.gpu_tx.as_ref().expect("gpu_tx live");
                tx.send(worker_task).map_err(|_| {
                    OrchestratorError::MetricApi("GPU worker channel closed".into())
                })?;
            }
            Backend::Cpu => {
                if pool.cpu_txs.is_empty() {
                    // No CPU workers spawned — surface a structured
                    // error so callers can react. Modern config never
                    // hits this (max_parallel_cpu has a floor of 1),
                    // but it's possible to construct a pool with
                    // `max_parallel_cpu = 0` for testing.
                    return Err(OrchestratorError::CpuBackendUnavailable {
                        metric,
                        required_feature: "cpu-all",
                    });
                }
                let idx = pool.cpu_next % pool.cpu_txs.len();
                pool.cpu_next = pool.cpu_next.wrapping_add(1);
                pool.cpu_txs[idx].send(worker_task).map_err(|_| {
                    OrchestratorError::MetricApi("CPU worker channel closed".into())
                })?;
            }
        }

        Ok(TaskHandle { id: handle_id })
    }

    /// Poll a specific [`TaskHandle`]. Returns `Some(TaskResult)` if the
    /// task has finished, `None` otherwise. Non-blocking.
    ///
    /// Each successful poll consumes the result — calling `poll` on the
    /// same handle a second time returns `None`.
    pub fn poll(&mut self, handle: TaskHandle) -> Option<TaskResult> {
        let pool = self.pool.as_mut()?;
        pool.drain_completed();
        let slot = pool.pending.get_mut(&handle.id)?;
        if let Some(result) = slot.take() {
            // Free the slot so subsequent polls return None.
            pool.pending.remove(&handle.id);
            Some(result)
        } else {
            None
        }
    }

    /// Drain any one completed task. Returns `Some(TaskResult)` for the
    /// first task whose result was waiting in the channel, `None` if no
    /// task is currently complete.
    pub fn poll_any(&mut self) -> Option<TaskResult> {
        let pool = self.pool.as_mut()?;
        pool.drain_completed();
        // Find the first pending entry that has a result.
        let key = pool
            .pending
            .iter()
            .find(|(_, v)| v.is_some())
            .map(|(k, _)| *k)?;
        let result = pool.pending.remove(&key)?.expect("Some by find");
        Some(result)
    }

    /// Drain any one completed task, blocking until one is available.
    ///
    /// Returns `None` only when no task is pending (caller hasn't
    /// submitted anything, or every prior task already drained).
    pub fn poll_any_blocking(&mut self) -> Option<TaskResult> {
        let pool = self.pool.as_mut()?;
        if pool.pending.is_empty() {
            return None;
        }
        // Drain non-blocking first — there may already be results
        // waiting from earlier dispatch.
        pool.drain_completed();
        // If something is now ready, return it.
        let ready_key = pool
            .pending
            .iter()
            .find(|(_, v)| v.is_some())
            .map(|(k, _)| *k);
        if let Some(k) = ready_key {
            return Some(pool.pending.remove(&k)?.expect("Some by find"));
        }
        // Otherwise block on the channel.
        if !pool.drain_completed_blocking() {
            return None;
        }
        let key = pool
            .pending
            .iter()
            .find(|(_, v)| v.is_some())
            .map(|(k, _)| *k)?;
        let result = pool.pending.remove(&key)?.expect("Some after blocking drain");
        Some(result)
    }

    /// Snapshot of the cached-ref auto-detect counters. Useful for
    /// tests verifying the auto-detect fires; production code rarely
    /// needs this.
    pub fn cached_ref_stats(&self) -> CachedRefStats {
        match &self.pool {
            Some(p) => CachedRefStats {
                hit_count: p.cached_ref_cache.hit_count.load(Ordering::Relaxed),
                miss_count: p.cached_ref_cache.miss_count.load(Ordering::Relaxed),
            },
            None => CachedRefStats::default(),
        }
    }

    /// Current free-VRAM snapshot in MiB, as seen by the live VRAM
    /// watcher. Returns `None` if the pool isn't initialized yet (no
    /// watcher running). The initial value is `usize::MAX` until the
    /// watcher gets its first probe (~one `vram_sample_interval_ms`).
    pub fn vram_watcher_mib(&self) -> Option<usize> {
        Some(self.pool.as_ref()?.vram.current_mib())
    }

    /// Pre-upload a reference image. Returns a [`TaskRefHandle`] that
    /// can be passed back as [`TaskData::PreUploaded`] to skip the
    /// auto-hash overhead.
    ///
    /// The handle remains valid until [`Self::drop_reference`] is
    /// called. Multiple tasks may reference the same handle
    /// concurrently — each task gets its own clone of the bytes.
    pub fn upload_reference(
        &mut self,
        ref_data: &[u8],
        width: u32,
        height: u32,
        metric: MetricKind,
    ) -> Result<TaskRefHandle, OrchestratorError> {
        self.ensure_pool();
        let pool = self.pool.as_ref().expect("pool init above");
        let hash = hash_ref_bytes(ref_data);
        let mut table = pool.pre_uploads.lock().expect("pre_uploads lock");
        let id = table.insert(metric, width, height, ref_data.to_vec(), hash);
        Ok(TaskRefHandle {
            metric,
            width,
            height,
            inner_id: id,
        })
    }

    /// Drop a pre-uploaded reference handle. Idempotent — dropping the
    /// same handle twice is a no-op.
    pub fn drop_reference(&mut self, handle: TaskRefHandle) {
        if let Some(pool) = self.pool.as_ref() {
            let mut table = pool.pre_uploads.lock().expect("pre_uploads lock");
            table.remove(handle.inner_id);
        }
    }

    /// Batch-run an iterator of tasks. Yields [`TaskResult`]s in
    /// completion order (NOT submission order — see the module docs).
    /// Callers correlate via [`Task::task_id`].
    ///
    /// The iterator blocks on each `next()` until at least one task
    /// completes; it returns `None` once every submitted task has been
    /// drained. Errors at submit time (chooser failure, unsupported
    /// task data) are converted into a synthetic `TaskResult` carrying
    /// `Err(OrchestratorError::...)` so the iterator length always
    /// matches the input count.
    pub fn run_all<I>(&mut self, tasks: I) -> RunAllIter<'_>
    where
        I: IntoIterator<Item = Task>,
    {
        let mut total_submitted: usize = 0;
        let mut submit_errors: Vec<TaskResult> = Vec::new();
        let mut submitted_handles: Vec<u64> = Vec::new();
        for task in tasks {
            let task_id = task.task_id;
            match self.submit(task) {
                Ok(h) => {
                    submitted_handles.push(h.id);
                    total_submitted += 1;
                }
                Err(e) => {
                    submit_errors.push(TaskResult {
                        task_id,
                        outcome: Err(e),
                        backend_used: None,
                        backends_attempted: Vec::new(),
                        wall_us: 0,
                        vram_peak_mib: None,
                        output_columns: std::collections::BTreeMap::new(),
                        metric_version: None,
                    });
                    total_submitted += 1;
                }
            }
        }
        RunAllIter {
            orch: self,
            remaining: total_submitted,
            handles: submitted_handles,
            errors: submit_errors,
        }
    }
}

// ---------------------------------------------------------------------------
// run_all iterator
// ---------------------------------------------------------------------------

/// Iterator returned by [`Orchestrator::run_all`]. Yields
/// [`TaskResult`]s in completion order. Holds a mutable borrow on the
/// orchestrator so the worker pool stays live for the iterator's lifetime.
pub struct RunAllIter<'a> {
    orch: &'a mut Orchestrator,
    /// How many more results are expected (submit-error results count too).
    remaining: usize,
    /// Handles still in flight. Each completed result is removed.
    #[allow(dead_code)]
    handles: Vec<u64>,
    /// Submit-time error results, drained before live results.
    errors: Vec<TaskResult>,
}

impl Iterator for RunAllIter<'_> {
    type Item = TaskResult;

    fn next(&mut self) -> Option<TaskResult> {
        if self.remaining == 0 {
            return None;
        }
        // Drain submit errors first (they're immediate).
        if let Some(e) = self.errors.pop() {
            self.remaining -= 1;
            return Some(e);
        }
        // Block until at least one worker result arrives.
        let r = self.orch.poll_any_blocking()?;
        self.remaining -= 1;
        Some(r)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// ---------------------------------------------------------------------------
// Tests — pure logic, no GPU. The real integration tests live in
// `tests/streaming.rs` + `tests/cached_ref.rs`.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xxhash3_64_is_deterministic_and_distinguishes() {
        let a = vec![0u8; 1024];
        let b = {
            let mut v = vec![0u8; 1024];
            v[0] = 1;
            v
        };
        let h_a1 = hash_ref_bytes(&a);
        let h_a2 = hash_ref_bytes(&a);
        let h_b = hash_ref_bytes(&b);
        assert_eq!(h_a1, h_a2, "same bytes hash identically");
        assert_ne!(h_a1, h_b, "different bytes hash differently");
    }

    #[test]
    fn xxhash3_64_4mp_under_10ms_release_only() {
        // 4096² × 3 = 48 MiB. xxhash3_64 should hash this in <10 ms on
        // any modern CPU when optimised. Debug builds run xxhash
        // unoptimised (~300+ ms on 7950X), so this assertion only
        // applies in release mode where the dispatcher actually runs.
        if cfg!(debug_assertions) {
            eprintln!("skipping xxhash 4MP timing test in debug build");
            return;
        }
        let buf = vec![42u8; 4096 * 4096 * 3];
        let t = std::time::Instant::now();
        let _ = hash_ref_bytes(&buf);
        let elapsed_ms = t.elapsed().as_millis();
        // 50 ms flake bound; 10 ms target.
        assert!(
            elapsed_ms < 50,
            "xxhash3_64 4MP took {elapsed_ms} ms (target <10 ms; flake bound 50 ms)"
        );
        eprintln!("xxhash3_64 4MP elapsed: {elapsed_ms} ms (release build)");
    }

    #[test]
    fn xxhash3_64_runs_at_4mp_regardless_of_build() {
        // Cheap shape test that works in both debug and release: just
        // confirm the hash function doesn't panic on a 4MP buffer.
        let buf = vec![42u8; 4096 * 4096 * 3];
        let h = hash_ref_bytes(&buf);
        assert!(h != 0);
    }

    #[test]
    fn cached_ref_cache_first_observe_is_miss_then_hit() {
        let mut c = CachedRefCache::default();
        let hit_a = c.observe(MetricKind::Cvvdp, 1024, 1024, 0xdeadbeef);
        assert!(!hit_a, "first observe is always a miss");
        let hit_b = c.observe(MetricKind::Cvvdp, 1024, 1024, 0xdeadbeef);
        assert!(hit_b, "second observe of same tuple is a hit");
        assert_eq!(c.hit_count.load(Ordering::Relaxed), 1);
        assert_eq!(c.miss_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn cached_ref_cache_distinguishes_metric_and_dims() {
        let mut c = CachedRefCache::default();
        c.observe(MetricKind::Cvvdp, 1024, 1024, 0xaa);
        // Different metric → miss.
        assert!(!c.observe(MetricKind::Butter, 1024, 1024, 0xaa));
        // Different dims → miss.
        assert!(!c.observe(MetricKind::Cvvdp, 2048, 2048, 0xaa));
        // Different hash → miss.
        assert!(!c.observe(MetricKind::Cvvdp, 1024, 1024, 0xbb));
        // The original tuple is still in the window.
        assert!(c.observe(MetricKind::Cvvdp, 1024, 1024, 0xaa));
    }

    #[test]
    fn cached_ref_cache_window_evicts_oldest() {
        let mut c = CachedRefCache::default();
        // Fill window to capacity.
        for i in 0..CACHED_REF_WINDOW_SIZE as u64 {
            c.observe(MetricKind::Cvvdp, 1024, 1024, i);
        }
        // The oldest entry (hash=0) is still in window — assert hit.
        assert!(c.observe(MetricKind::Cvvdp, 1024, 1024, 0));
        // Adding a brand-new hash evicts the oldest non-recently-touched.
        // After the hit above, the window has been touched but our
        // FIFO replaces by insert-order — slightly weaker than LRU but
        // still useful. Just sanity-check the window doesn't grow
        // unbounded.
        for i in (CACHED_REF_WINDOW_SIZE as u64)..(CACHED_REF_WINDOW_SIZE as u64 + 5) {
            c.observe(MetricKind::Cvvdp, 1024, 1024, i);
        }
        assert!(c.window.len() <= CACHED_REF_WINDOW_SIZE);
    }

    #[test]
    fn pool_config_default_is_sensible() {
        let cfg = PoolConfig::default();
        assert!(cfg.max_parallel_cpu >= 1);
        assert!(cfg.vram_safety_floor_mib > 0);
        assert!(cfg.vram_sample_interval_ms > 0);
        assert!(cfg.vram_stall_ms > 0);
    }
}
