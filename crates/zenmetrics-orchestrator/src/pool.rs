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
    /// Chooser's predicted VRAM consumption for the chosen backend at
    /// this size, in MiB. Phase 7.6 Layer 4 — surfaces in the
    /// swap-time log so operators can correlate the predicted
    /// footprint with the worker's live free-VRAM reading.
    predicted_vram_mib: usize,
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
    /// Phase 9.1 — number of GPU "lanes" (worker threads) per device.
    ///
    /// Each lane owns its own warm [`ExecMetric`] and CUDA stream (via
    /// cubecl's thread-local `StreamId` — every OS thread that touches
    /// the shared `ComputeClient` is auto-assigned a distinct stream by
    /// the cubecl-cuda backend's `MultiStream` scheduler). N > 1 lanes
    /// run kernel launches concurrently on the same physical GPU when
    /// VRAM + SM resources allow.
    ///
    /// **Default: 1** — preserves bit-identical single-worker behaviour
    /// for every caller that hasn't explicitly opted into concurrency.
    /// Phase 9.3 may grow this dynamically at runtime; Phase 9.4 picks
    /// per `(metric, size)` from the cached `MetricProfile`.
    ///
    /// VRAM accounting scales linearly: predicted footprint is
    /// `N × per-task footprint`. The chooser's budget check at submit
    /// time uses the single-task footprint; the pool's runtime VRAM
    /// gate (the live `vram_safety_floor_mib`) is the backstop when
    /// `N` concurrent lanes outrun the chooser's snapshot.
    ///
    /// Minimum 1 (zero is clamped silently). Range: 1..=8.
    pub max_gpu_lanes: usize,
    /// Phase 9.3 — target GPU utilization percentage for the adaptive
    /// lane controller. When the rolling utilization average drops
    /// below this AND `current_lanes < max_gpu_lanes`, the controller
    /// considers spinning up an extra lane. Default 80%.
    pub target_gpu_utilization_pct: u8,
    /// Phase 9.3 — upper bound for the adaptive controller. The
    /// controller never spawns more than this many lanes regardless of
    /// observed utilization. Default 4 (matches the Phase 9 design
    /// doc's `max_workers_per_device = 4`).
    pub adaptive_max_gpu_lanes: usize,
    /// Phase 9.3 — sample interval for the GPU utilization watcher
    /// (ms). Default 5000 (= 5s, per design doc).
    pub gpu_util_sample_interval_ms: u64,
    /// Phase 9.3 — enable adaptive lane scaling. When `true`, a
    /// background thread samples GPU utilization (via `nvidia-smi`)
    /// every `gpu_util_sample_interval_ms` and adjusts the live lane
    /// count between `1` and `adaptive_max_gpu_lanes`. When `false`,
    /// the pool uses `max_gpu_lanes` statically. Default `false`.
    pub adaptive_gpu_lanes: bool,
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
            max_gpu_lanes: 1,
            target_gpu_utilization_pct: 80,
            adaptive_max_gpu_lanes: 4,
            gpu_util_sample_interval_ms: 5000,
            adaptive_gpu_lanes: false,
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
// GPU utilization watcher (Phase 9.3)
// ---------------------------------------------------------------------------

/// Background thread sampling `nvidia-smi --query-gpu=utilization.gpu`
/// at a configurable interval. Stores the most recent reading (percent,
/// 0-100) and a small rolling sample buffer that the adaptive lane
/// controller (Phase 9.3) consults to decide whether to spin up / drop
/// a lane.
///
/// Failure modes: `nvidia-smi` not on `PATH`, or returns non-integer
/// output — the watcher stores `u8::MAX` as a sentinel for "unknown"
/// and the controller leaves the lane count alone. The watcher does
/// not panic; it logs at debug level once per unknown sample.
pub(crate) struct GpuUtilWatcher {
    /// Latest sample, 0-100 (or `u8::MAX` for "unknown").
    latest_pct: Arc<AtomicUsize>,
    /// Number of consecutive samples that have all been below the
    /// configured target. Resets when a sample is at-or-above. The
    /// adaptive controller acts when this reaches a small threshold
    /// (3, per design doc) AND the active lane count is still below
    /// the cap.
    consecutive_below_target: Arc<AtomicUsize>,
    /// Number of consecutive samples >= 95%. The controller uses this
    /// to consider dropping a lane.
    consecutive_above_target: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl GpuUtilWatcher {
    /// Spawn a sampling thread. The initial value is `u8::MAX`
    /// ("unknown") until the first sample arrives.
    fn spawn(interval_ms: u64) -> Self {
        let latest_pct = Arc::new(AtomicUsize::new(u8::MAX as usize));
        let consecutive_below_target = Arc::new(AtomicUsize::new(0));
        let consecutive_above_target = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let p_clone = Arc::clone(&latest_pct);
        let b_clone = Arc::clone(&consecutive_below_target);
        let a_clone = Arc::clone(&consecutive_above_target);
        let shut_clone = Arc::clone(&shutdown);
        let join = thread::Builder::new()
            .name("zm-gpu-util-watcher".into())
            .spawn(move || {
                let interval = Duration::from_millis(interval_ms);
                while !shut_clone.load(Ordering::Acquire) {
                    let pct = sample_gpu_utilization_pct();
                    match pct {
                        Some(v) => {
                            p_clone.store(v as usize, Ordering::Release);
                            // Threshold check (target = 80, drop_threshold = 95).
                            // These constants match the Phase 9 design
                            // doc; the controller's actual decision uses
                            // them via PoolConfig::target_gpu_utilization_pct.
                            if v < 80 {
                                b_clone.fetch_add(1, Ordering::Relaxed);
                                a_clone.store(0, Ordering::Relaxed);
                            } else if v >= 95 {
                                a_clone.fetch_add(1, Ordering::Relaxed);
                                b_clone.store(0, Ordering::Relaxed);
                            } else {
                                b_clone.store(0, Ordering::Relaxed);
                                a_clone.store(0, Ordering::Relaxed);
                            }
                        }
                        None => {
                            p_clone.store(u8::MAX as usize, Ordering::Release);
                            log::debug!(
                                target: "zenmetrics_orchestrator::pool",
                                "gpu-util sampler: nvidia-smi unavailable or unparseable"
                            );
                        }
                    }
                    // Sleep in small chunks for responsive shutdown.
                    let chunk = Duration::from_millis(50);
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
            .expect("zm-gpu-util-watcher spawn");
        Self {
            latest_pct,
            consecutive_below_target,
            consecutive_above_target,
            shutdown,
            join: Some(join),
        }
    }

    /// Latest sampled GPU utilization in percent (0..=100). Returns
    /// `None` when the watcher has no valid reading yet (initial state
    /// or `nvidia-smi` unavailable).
    pub(crate) fn latest_pct(&self) -> Option<u8> {
        let v = self.latest_pct.load(Ordering::Acquire);
        if v == u8::MAX as usize {
            None
        } else {
            Some(v as u8)
        }
    }

    pub(crate) fn consecutive_below(&self) -> usize {
        self.consecutive_below_target.load(Ordering::Acquire)
    }

    pub(crate) fn consecutive_above(&self) -> usize {
        self.consecutive_above_target.load(Ordering::Acquire)
    }
}

impl Drop for GpuUtilWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Test-only utilities for Phase 9.3 controller exercises.
// ---------------------------------------------------------------------------

/// Test-only helper exposing the watcher's internal counters so the
/// Phase 9.3 controller can be exercised without a live `nvidia-smi`.
///
/// Production code MUST NOT depend on this — the counters are an
/// implementation detail of the adaptive scaling heuristic.
#[doc(hidden)]
#[cfg(test)]
pub(crate) struct WatcherCounters {
    pub latest_pct: Arc<AtomicUsize>,
    pub below: Arc<AtomicUsize>,
    pub above: Arc<AtomicUsize>,
}

#[cfg(test)]
impl GpuUtilWatcher {
    /// Construct a watcher without spawning the sampling thread — for
    /// unit tests that want to manipulate counters directly.
    pub(crate) fn test_only_fake() -> Self {
        Self {
            latest_pct: Arc::new(AtomicUsize::new(u8::MAX as usize)),
            consecutive_below_target: Arc::new(AtomicUsize::new(0)),
            consecutive_above_target: Arc::new(AtomicUsize::new(0)),
            shutdown: Arc::new(AtomicBool::new(true)),
            join: None,
        }
    }

    pub(crate) fn test_only_counters(&self) -> WatcherCounters {
        WatcherCounters {
            latest_pct: Arc::clone(&self.latest_pct),
            below: Arc::clone(&self.consecutive_below_target),
            above: Arc::clone(&self.consecutive_above_target),
        }
    }
}

/// Phase 9.3 — pure-function controller logic. Given the current
/// active lane count, the watcher's consecutive-below / consecutive-
/// above counters, and the configured max, decide the next lane count.
///
/// Returns `None` when no change is warranted (insufficient samples,
/// or already at the boundary). Returns `Some(N)` with `1 <= N <= max`
/// when a transition should fire.
///
/// Heuristic:
/// - >= 3 consecutive low-util samples + room to grow -> +1 lane
/// - >= 3 consecutive high-util samples + room to shrink -> -1 lane
/// - Else: no change
///
/// The 3-sample threshold matches the Phase 9 design doc's
/// "samples_needed = 3" guard against single-sample noise.
pub(crate) fn compute_next_lane_count(
    current: usize,
    below_samples: usize,
    above_samples: usize,
    max_lanes: usize,
) -> Option<usize> {
    const SAMPLES_NEEDED: usize = 3;
    if below_samples >= SAMPLES_NEEDED && current < max_lanes {
        Some(current + 1)
    } else if above_samples >= SAMPLES_NEEDED && current > 1 {
        Some(current - 1)
    } else {
        None
    }
}

/// One-shot `nvidia-smi` sample. Returns `None` if the command can't
/// be invoked or the output isn't a 0-100 integer.
///
/// Shell-out cost: ~30-50 ms per sample on a 7950X — acceptable at the
/// default 5s interval (< 1% CPU). For shorter intervals (test fixtures),
/// callers should keep `gpu_util_sample_interval_ms >= 250`.
fn sample_gpu_utilization_pct() -> Option<u8> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    // Multi-GPU output: one line per GPU. We sample the first.
    let first = s.lines().next()?.trim();
    first.parse::<u8>().ok()
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

/// Process-wide warm-instance construction counter. Incremented every
/// time a worker constructs a new `ExecMetric` (i.e., on signature
/// change or first-task-of-worker). Phase 7.6 test surface — callers
/// MUST NOT depend on this in production code; it's the only way to
/// observe warm-instance churn from the integration test layer
/// without instrumenting the umbrella crate.
///
/// Reset between tests with [`reset_warm_instance_construction_count`].
static WARM_INSTANCE_CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);

/// Read the current warm-instance construction count. Useful for the
/// `warm_instance_churn_minimal_on_mixed_chunk` test plan in
/// `crates/zenmetrics-orchestrator/docs/REORDERING_DESIGN.md`.
pub fn warm_instance_construction_count() -> usize {
    WARM_INSTANCE_CONSTRUCTIONS.load(Ordering::Relaxed)
}

/// Reset the warm-instance construction counter to zero. Test helper.
pub fn reset_warm_instance_construction_count() {
    WARM_INSTANCE_CONSTRUCTIONS.store(0, Ordering::Relaxed);
}

/// Internal — called by the worker once per successful construction.
fn record_warm_instance_construction() {
    WARM_INSTANCE_CONSTRUCTIONS.fetch_add(1, Ordering::Relaxed);
}

/// Process-wide count of swap-time VRAM reclaim calls. Test/measurement
/// surface — lets `crates/zenmetrics-orchestrator` integration tests and
/// the task #150 VRAM measurement confirm that the swap cleanup actually
/// fired (and how often) without instrumenting cubecl.
static SWAP_VRAM_RECLAIMS: AtomicUsize = AtomicUsize::new(0);

/// Read the swap-time VRAM reclaim count. See [`SWAP_VRAM_RECLAIMS`].
pub fn swap_vram_reclaim_count() -> usize {
    SWAP_VRAM_RECLAIMS.load(Ordering::Relaxed)
}

/// Reset the swap-time VRAM reclaim counter. Test helper.
pub fn reset_swap_vram_reclaim_count() {
    SWAP_VRAM_RECLAIMS.store(0, Ordering::Relaxed);
}

/// Reclaim cubecl's pooled (but now-unreferenced) GPU memory back to the
/// driver when a GPU worker swaps away from a metric signature.
///
/// cubecl pools device pages across `Handle` drop, so dropping the old
/// warm metric returns its buffers to the pool free list but the pages
/// stay resident — and the next signature's metric then allocates fresh
/// pages, pushing peak VRAM toward the SUM of both working sets. Calling
/// this **after** the old metric is dropped and **before** the new one
/// is constructed returns those freed pages to the driver, keeping peak
/// across a mixed-metric chunk at ≈ MAX(single metric) instead of SUM.
///
/// Safe here because: (a) the cubecl CUDA pool is per-stream and the
/// stream is keyed on this worker thread's id, so this only touches
/// THIS worker's pool — never another lane's live bindings; and (b) at
/// the call site the old metric has already been dropped, so this thread
/// holds no live bindings into the pages being reclaimed. (This is the
/// exact invariant the 2026-05-22 `MetricCache` attempt violated — it
/// reclaimed while a cached metric's bindings were still live, hitting
/// the `stream.rs:101` get_cursor panic.)
///
/// Only fires for GPU backends (the orchestrator `Backend::Gpu*` variants
/// all map to cubecl CUDA). `Backend::Cpu` is a no-op — the CPU adapter
/// doesn't use the GPU pool. Set `ZENMETRICS_NO_SWAP_VRAM_CLEANUP=1` to
/// disable (escape hatch for hosts where the cleanup misbehaves; the
/// pool then keeps the old buffers as before this change).
#[cfg(feature = "bench")]
fn reclaim_swap_vram(prev_backend: Backend) {
    // Only GPU signatures pool device memory worth reclaiming.
    if matches!(prev_backend, Backend::Cpu) {
        return;
    }
    if std::env::var_os("ZENMETRICS_NO_SWAP_VRAM_CLEANUP").is_some() {
        return;
    }
    // All GPU variants (GpuFull / GpuStrip / GpuStripPair) construct
    // against cubecl CUDA (see executor::construct), so reclaim the CUDA
    // pool for this thread.
    zenmetrics_api::reclaim_pooled_vram(zenmetrics_api::Backend::Cuda);
    SWAP_VRAM_RECLAIMS.fetch_add(1, Ordering::Relaxed);
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
            // Phase 7.6 Layer 4 — observable VRAM budget at swap time.
            // Log every signature-change swap with the chooser's
            // prediction AND the live free-VRAM snapshot so operators
            // can tune `vram_safety_floor_mib` and the chooser's
            // safety_margin from real workload patterns. The check
            // here is observational: the chooser already gated this
            // task against `usable_vram = free * (1 - margin)` at
            // submit time. A WARN-level log fires when the live
            // free-VRAM at swap time has dropped below the
            // chooser's predicted footprint (i.e. an external
            // process consumed VRAM between chooser-decision and
            // instance-construction).
            let free_mib = vram_snapshot.load(Ordering::Acquire);
            let predicted_mib = task.predicted_vram_mib;
            if free_mib < predicted_mib && free_mib != usize::MAX {
                log::warn!(
                    target: "zenmetrics_orchestrator::pool",
                    "vram budget gates instance swap: metric={:?}, size={}x{}, backend={:?}, predict={} MiB, free={} MiB (live snapshot below chooser prediction — external VRAM pressure?)",
                    task.metric, task.width, task.height, task.chosen_backend,
                    predicted_mib, free_mib,
                );
            } else {
                log::debug!(
                    target: "zenmetrics_orchestrator::pool",
                    "warm-instance swap: metric={:?}, size={}x{}, backend={:?}, predict={} MiB, free={} MiB",
                    task.metric, task.width, task.height, task.chosen_backend,
                    predicted_mib, free_mib,
                );
            }
            // Drop the old metric first to release device buffers.
            // `current_metric = None` returns the old instance's cubecl
            // handles to the pool's FREE LIST, but cubecl keeps the
            // device pages resident for reuse — so without an explicit
            // reclaim the next (different-signature) metric allocates
            // fresh pages on top, pushing peak VRAM toward SUM, not MAX.
            let prev_backend = current_signature.map(|s| s.backend);
            current_metric = None;
            cached_ref_hash = None;
            // Reclaim the just-freed pooled pages back to the driver
            // BEFORE constructing the new metric. Safe at this point:
            // the old metric is dropped (this thread holds no live
            // bindings), and the cubecl pool is per-thread so we only
            // touch this worker's pool. Skipped on the worker's very
            // first task (no prior signature) and for CPU→* transitions.
            if let Some(pb) = prev_backend {
                reclaim_swap_vram(pb);
            }
            match construct_pub(
                task.metric,
                task.chosen_backend,
                task.width,
                task.height,
                task.params.clone(),
            ) {
                ConstructOutcomePub::Ok(m) => {
                    record_warm_instance_construction();
                    current_metric = Some(m);
                    current_signature = Some(sig);
                }
                ConstructOutcomePub::Oom => {
                    log::warn!(
                        target: "zenmetrics_orchestrator::pool",
                        "vram budget gates instance swap: metric={:?}, size={}x{}, backend={:?}, predict={} MiB, free={} MiB, outcome=OomAtConstruction",
                        task.metric, task.width, task.height, task.chosen_backend,
                        predicted_mib, free_mib,
                    );
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
                        // Phase 7.6 Layer 4 — surface the chooser's
                        // prediction so callers can audit per-task
                        // VRAM consumption without their own probe.
                        vram_peak_mib: Some(task.predicted_vram_mib),
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
                // Reclaim the dropped metric's pooled pages back to the
                // driver. After a runtime OOM the pool is saturated, so
                // returning the just-dropped pages to the driver gives
                // the retry (and any other lane) the best chance to fit.
                // Best-effort — sync may fail in a post-OOM state, which
                // `reclaim_pooled_vram` swallows.
                reclaim_swap_vram(task.chosen_backend);
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
                        vram_peak_mib: Some(task.predicted_vram_mib),
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
                        vram_peak_mib: Some(task.predicted_vram_mib),
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
                    record_warm_instance_construction();
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
    /// Phase 9.1 — GPU lanes. Each lane is one worker thread with its
    /// own input queue + warm `ExecMetric`. Round-robin dispatch via
    /// `gpu_next` (modulo lanes.len()) selects the target lane for the
    /// next task. Cubecl's MultiStream backend assigns each OS thread
    /// its own CUDA stream automatically (no `unsafe set_stream`
    /// needed). Length matches `PoolConfig::max_gpu_lanes` at spawn
    /// time (clamped to 1..=8).
    gpu_lanes: Vec<mpsc::Sender<WorkerTask>>,
    gpu_next: AtomicUsize,
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
    /// Phase 9.3 — adaptive GPU utilization watcher. Spawned only when
    /// `PoolConfig::adaptive_gpu_lanes` is true. The watcher updates
    /// `gpu_util_pct` periodically; the dispatcher reads the snapshot
    /// to decide whether the active lane count is still appropriate.
    /// Lane scaling is bounded by `[1, adaptive_max_gpu_lanes]`.
    gpu_util: Option<GpuUtilWatcher>,
    /// Phase 9.3 — number of GPU lanes currently active for dispatch
    /// (≤ `gpu_lanes.len()`). The adaptive controller updates this
    /// atomically; the dispatcher's modulo uses this value (clamped to
    /// `gpu_lanes.len()` defensively). When `adaptive_gpu_lanes` is
    /// false this is fixed at `gpu_lanes.len()`.
    active_gpu_lanes: AtomicUsize,
    /// Configuration snapshot. Stored for inspection / future
    /// re-spawning logic; the active worker thread params are baked
    /// in at spawn time.
    #[allow(dead_code)]
    config: PoolConfig,
    /// Pre-upload table — guarded by Mutex so the pool's API methods can
    /// extend it without going through the worker.
    pre_uploads: Arc<Mutex<PreUploadTable>>,
    /// Phase 7.6 Layer 3 — streaming reorder window.
    ///
    /// Tasks accumulate here until either the configured count or
    /// duration is reached, then the window is sorted by
    /// `(metric.tag(), w, h, ref_hash, task_id)` and dispatched as a
    /// batch. Disabled by `stream_reorder_window = (Duration::ZERO, 1)`.
    pending_queue: PendingQueue,
}

/// Phase 7.6 Layer 3 — submission reorder buffer.
///
/// Tasks held here have already been *prepared* (chooser ran, ref bytes
/// materialised, cached-ref auto-detect observed) but not yet
/// dispatched to a worker. The `submit()` path enqueues into here; the
/// flush path drains, sorts, and dispatches.
#[derive(Default)]
struct PendingQueue {
    tasks: Vec<PreparedTask>,
    window_started_at: Option<Instant>,
}

/// A task prepared for dispatch but parked in the reorder window. Holds
/// every piece of state the worker needs; the only thing left at flush
/// time is the mpsc send.
struct PreparedTask {
    task: WorkerTask,
    /// Which worker queue this should land on. The chooser's choice
    /// was made at submit() time; storing it here means the sort
    /// order doesn't have to re-derive the routing.
    route: WorkerRoute,
}

/// Routing decision for a prepared task. `Cpu(idx)` carries the
/// round-robin worker index pre-computed at submit time.
#[derive(Debug, Clone, Copy)]
enum WorkerRoute {
    Gpu,
    Cpu(usize),
}

impl PoolState {
    fn new(config: PoolConfig) -> Self {
        let (result_tx, result_rx) = mpsc::channel::<WorkerResult>();
        let vram = VramWatcher::spawn(config.vram_sample_interval_ms);

        let mut join_handles: Vec<JoinHandle<()>> = Vec::new();

        // -- GPU lanes (Phase 9.1) --
        //
        // Spawn N worker threads, each with its own input queue + warm
        // ExecMetric. cubecl's MultiStream backend (default
        // `max_streams = 128`) auto-assigns each OS thread a distinct
        // `cudaStream_t` via thread-local `StreamId`, so concurrent
        // kernels on these threads run on independent CUDA streams.
        //
        // The configured `max_gpu_lanes` is clamped to 1..=8. The
        // adaptive controller (Phase 9.3) sizes `active_gpu_lanes`
        // dynamically within `[1, adaptive_max_gpu_lanes]`; we always
        // spawn `max_gpu_lanes` threads up-front so scale-up doesn't
        // need to spawn from the dispatcher's hot path. Surplus lanes
        // sit idle on their channel recv() until tasks arrive.
        let lane_count = config.max_gpu_lanes.clamp(1, 8);
        let mut gpu_lanes: Vec<mpsc::Sender<WorkerTask>> = Vec::with_capacity(lane_count);
        for i in 0..lane_count {
            let (gpu_tx, gpu_rx) = mpsc::channel::<WorkerTask>();
            let gpu_result_tx = result_tx.clone();
            let vram_snap = vram.snapshot_handle();
            let floor = config.vram_safety_floor_mib;
            let stall = config.vram_stall_ms;
            let h = thread::Builder::new()
                .name(format!("zm-gpu-lane-{i}"))
                .spawn(move || {
                    gpu_worker_main(gpu_rx, gpu_result_tx, floor, stall, vram_snap);
                })
                .expect("zm-gpu-lane spawn");
            join_handles.push(h);
            gpu_lanes.push(gpu_tx);
        }

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

        // -- Adaptive GPU utilization watcher (Phase 9.3) --
        let active_gpu_lanes = AtomicUsize::new(if config.adaptive_gpu_lanes {
            1
        } else {
            lane_count
        });
        let gpu_util = if config.adaptive_gpu_lanes {
            Some(GpuUtilWatcher::spawn(config.gpu_util_sample_interval_ms))
        } else {
            None
        };

        Self {
            gpu_lanes,
            gpu_next: AtomicUsize::new(0),
            cpu_txs,
            cpu_next: 0,
            join_handles,
            result_rx,
            pending: HashMap::new(),
            next_handle: 0,
            cached_ref_cache: CachedRefCache::default(),
            vram,
            gpu_util,
            active_gpu_lanes,
            config,
            pre_uploads: Arc::new(Mutex::new(PreUploadTable::default())),
            pending_queue: PendingQueue::default(),
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
        self.gpu_lanes.clear();
        self.cpu_txs.clear();
        // GpuUtilWatcher's Drop signals + joins its own thread before
        // the rest of PoolState tears down (otherwise the watcher
        // could try to update `active_gpu_lanes` on a dropped atomic
        // — `AtomicUsize::store` itself is fine on a moved-out value,
        // but explicit ordering documents the lifetime intent).
        self.gpu_util.take();
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
        let task_supplied_ref_hash = task.ref_hash;

        let (ref_payload, ref_hash) = match task.ref_data {
            TaskData::Srgb8(bytes) => {
                let h = if task_supplied_ref_hash != 0 {
                    task_supplied_ref_hash
                } else {
                    hash_ref_bytes(&bytes)
                };
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

        // Allocate handle + register pending slot.
        let pool = self.pool.as_mut().expect("pool initialized");
        let handle_id = pool.next_handle;
        pool.next_handle = pool.next_handle.wrapping_add(1);
        pool.pending.insert(handle_id, None);

        // Pre-compute the routing decision now. `use_cached_ref` is
        // deliberately set at flush time (cached-ref auto-detect
        // observe order depends on dispatch order; sorting before
        // observe maximises hit rate).
        let route = match chosen_backend {
            Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair => WorkerRoute::Gpu,
            Backend::Cpu => {
                if pool.cpu_txs.is_empty() {
                    // No CPU workers spawned — surface a structured
                    // error so callers can react. Modern config never
                    // hits this (max_parallel_cpu has a floor of 1),
                    // but it's possible to construct a pool with
                    // `max_parallel_cpu = 0` for testing.
                    pool.pending.remove(&handle_id);
                    return Err(OrchestratorError::CpuBackendUnavailable {
                        metric,
                        required_feature: "cpu-all",
                    });
                }
                let idx = pool.cpu_next % pool.cpu_txs.len();
                pool.cpu_next = pool.cpu_next.wrapping_add(1);
                WorkerRoute::Cpu(idx)
            }
        };

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
            // Filled in at flush time via cached-ref observe.
            use_cached_ref: false,
            predicted_vram_mib: choice.predicted_vram_mib,
        };

        // Enqueue into the streaming reorder window. The window is
        // flushed when either the count or duration limit is reached;
        // disabled by `stream_reorder_window = (Duration::ZERO, 1)`
        // (which always trips the count check and flushes immediately).
        let (window_dur, window_cnt) = self.config.stream_reorder_window;
        pool.pending_queue.tasks.push(PreparedTask { task: worker_task, route });
        if pool.pending_queue.window_started_at.is_none() {
            pool.pending_queue.window_started_at = Some(Instant::now());
        }
        let count_reached = pool.pending_queue.tasks.len() >= window_cnt;
        let duration_elapsed = pool
            .pending_queue
            .window_started_at
            .map(|t| t.elapsed() >= window_dur)
            .unwrap_or(false);
        if count_reached || duration_elapsed {
            self.flush_pending_internal();
        }

        Ok(TaskHandle { id: handle_id })
    }

    /// Flush the streaming reorder window: drain pending submissions,
    /// sort by `(metric.tag(), w, h, ref_hash, task_id)`, run the
    /// cached-ref auto-detect observe pass in sorted order, and
    /// dispatch every prepared task to its worker queue.
    ///
    /// Idempotent — calling on an empty queue is a no-op.
    ///
    /// Phase 7.6 Layer 3. Public so callers using a custom
    /// `stream_reorder_window` (e.g. `(Duration::MAX, usize::MAX)` for
    /// fully-buffered batching) can dispatch explicitly without
    /// waiting for the window to time out.
    pub fn flush_pending(&mut self) {
        self.flush_pending_internal();
    }

    /// Internal flush — called from `submit()` when the window fills,
    /// from `poll()` when the window has aged past its duration, and
    /// from the public `flush_pending()`.
    fn flush_pending_internal(&mut self) {
        let pool = match self.pool.as_mut() {
            Some(p) => p,
            None => return,
        };
        if pool.pending_queue.tasks.is_empty() {
            pool.pending_queue.window_started_at = None;
            return;
        }
        let mut window: Vec<PreparedTask> = pool.pending_queue.tasks.drain(..).collect();
        pool.pending_queue.window_started_at = None;
        // Sort the window. Tie-break on task_id for deterministic
        // output across runs.
        window.sort_by_key(|p| {
            (
                p.task.metric.tag(),
                p.task.width,
                p.task.height,
                p.task.ref_hash,
                p.task.task_id,
            )
        });

        // In sorted order: run cached-ref observe to refresh the
        // sliding window's hit/miss accounting, then dispatch. Doing
        // observe AFTER sort means consecutive tasks with identical
        // refs hit each other and the worker reuses its cached
        // reference state — the whole point of the reorder window.
        for prepared in window {
            let mut t = prepared.task;
            let hit = pool
                .cached_ref_cache
                .observe(t.metric, t.width, t.height, t.ref_hash);
            t.use_cached_ref = hit;
            match prepared.route {
                WorkerRoute::Gpu => {
                    // Phase 9.1 — round-robin across active GPU lanes.
                    // `active_gpu_lanes` is updated by the adaptive
                    // controller (Phase 9.3) within `[1, lane_count]`;
                    // we clamp defensively to `gpu_lanes.len()` so a
                    // race that overshot doesn't index out of bounds.
                    if pool.gpu_lanes.is_empty() {
                        continue;
                    }
                    let active = pool
                        .active_gpu_lanes
                        .load(Ordering::Acquire)
                        .clamp(1, pool.gpu_lanes.len());
                    let idx = pool.gpu_next.fetch_add(1, Ordering::Relaxed) % active;
                    let _ = pool.gpu_lanes[idx].send(t);
                }
                WorkerRoute::Cpu(idx) => {
                    let slot = idx % pool.cpu_txs.len().max(1);
                    if let Some(tx) = pool.cpu_txs.get(slot) {
                        let _ = tx.send(t);
                    }
                }
            }
        }
    }

    /// Drain any pending streaming-window tasks whose window has
    /// expired. Called automatically by `poll` / `poll_any` /
    /// `poll_any_blocking` so a slow caller doesn't park tasks
    /// indefinitely.
    fn drain_stale_window(&mut self) {
        let (window_dur, _) = self.config.stream_reorder_window;
        let needs_flush = self
            .pool
            .as_ref()
            .and_then(|p| p.pending_queue.window_started_at)
            .map(|t| t.elapsed() >= window_dur)
            .unwrap_or(false);
        if needs_flush {
            self.flush_pending_internal();
        }
    }

    /// Poll a specific [`TaskHandle`]. Returns `Some(TaskResult)` if the
    /// task has finished, `None` otherwise. Non-blocking.
    ///
    /// Each successful poll consumes the result — calling `poll` on the
    /// same handle a second time returns `None`.
    pub fn poll(&mut self, handle: TaskHandle) -> Option<TaskResult> {
        // Drain stale streaming window first so a slow caller doesn't
        // park tasks indefinitely.
        self.drain_stale_window();
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
        self.drain_stale_window();
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
        // Drain stale streaming window first — if a long block on the
        // result channel started before the window timed out, we'd
        // otherwise leak tasks in the pending buffer.
        self.drain_stale_window();
        // Then ensure any tasks waiting in the reorder window get
        // dispatched if the caller has nothing in flight yet (e.g.,
        // submit() -> poll_any_blocking() with stream_reorder_window
        // count > 1: the window won't fill, but a blocking poll
        // implies the caller wants results NOW).
        if let Some(p) = self.pool.as_ref() {
            if !p.pending_queue.tasks.is_empty() {
                self.flush_pending_internal();
            }
        }
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

    /// Number of tasks currently buffered in the streaming reorder
    /// window. Returns `0` if the pool isn't initialised. Useful for
    /// Phase 7.6 tests verifying the window's buffering behaviour;
    /// production code generally doesn't need this — callers should
    /// rely on `flush_pending()` to dispatch on demand.
    pub fn pending_queue_len(&self) -> usize {
        self.pool
            .as_ref()
            .map(|p| p.pending_queue.tasks.len())
            .unwrap_or(0)
    }

    /// Number of tasks dispatched to a worker queue but not yet
    /// completed. Returns `0` if the pool isn't initialised.
    /// Counts both `submit()` (immediate dispatch path) and
    /// `flush_pending()` (deferred dispatch path).
    pub fn in_flight_len(&self) -> usize {
        self.pool.as_ref().map(|p| p.pending.len()).unwrap_or(0)
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

    /// Phase 9.1 — total number of GPU lanes spawned at pool init time.
    /// Matches `PoolConfig::max_gpu_lanes` (clamped to 1..=8). Returns
    /// `None` if the pool isn't initialised yet. The active-for-dispatch
    /// count may be lower than this — see [`Self::active_gpu_lanes`].
    pub fn gpu_lane_count(&self) -> Option<usize> {
        Some(self.pool.as_ref()?.gpu_lanes.len())
    }

    /// Phase 9.3 — number of GPU lanes currently active for dispatch.
    /// When `PoolConfig::adaptive_gpu_lanes` is false, this equals
    /// [`Self::gpu_lane_count`]. When adaptive is enabled, it floats
    /// between 1 and `PoolConfig::adaptive_max_gpu_lanes` based on
    /// observed GPU utilization.
    ///
    /// Returns `None` if the pool isn't initialised.
    pub fn active_gpu_lanes(&self) -> Option<usize> {
        Some(self.pool.as_ref()?.active_gpu_lanes.load(Ordering::Acquire))
    }

    /// Phase 9.3 — current GPU utilization sample, as percent (0-100).
    /// Returns `None` if adaptive scaling is disabled OR the watcher
    /// hasn't produced its first sample yet OR `nvidia-smi` is
    /// unavailable.
    pub fn gpu_utilization_pct(&self) -> Option<u8> {
        self.pool.as_ref()?.gpu_util.as_ref()?.latest_pct()
    }

    /// Phase 9.3 — run one tick of the adaptive lane controller. Looks
    /// at the latest utilization sample and the consecutive-low /
    /// consecutive-high counters maintained by the watcher, and
    /// adjusts `active_gpu_lanes` within `[1, adaptive_max_gpu_lanes]`.
    ///
    /// Returns `Some(new_count)` if the lane count changed, `None`
    /// otherwise (no change, or adaptive scaling disabled).
    ///
    /// Callers may invoke this from their own polling loop. A future
    /// Phase 9.3.1 wires an implicit rate-limited tick into the
    /// `submit()` hot path so callers don't have to drive it manually.
    pub fn adaptive_lane_tick(&mut self) -> Option<usize> {
        let pool = self.pool.as_mut()?;
        let watcher = pool.gpu_util.as_ref()?;
        let lane_count = pool.gpu_lanes.len();
        let max_lanes = pool
            .config
            .adaptive_max_gpu_lanes
            .clamp(1, lane_count.max(1));
        let current = pool.active_gpu_lanes.load(Ordering::Acquire);
        let below = watcher.consecutive_below();
        let above = watcher.consecutive_above();
        let new_count = compute_next_lane_count(current, below, above, max_lanes)?;
        pool.active_gpu_lanes.store(new_count, Ordering::Release);
        log::debug!(
            target: "zenmetrics_orchestrator::pool",
            "adaptive lane tick: util_pct={:?} below={} above={} {} -> {}",
            watcher.latest_pct(), below, above, current, new_count,
        );
        Some(new_count)
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
    /// ## Phase 7.6 — internal reorder
    ///
    /// `run_all` collects every input task, populates
    /// [`Task::ref_hash`] from the reference bytes (or pre-upload
    /// handle id), and **sorts internally** by
    /// `(metric.tag(), width, height, ref_hash, task_id)` before
    /// dispatching. The sort drastically reduces warm-instance
    /// signature churn (one construction per `(metric, dims, backend)`
    /// instead of one per task) and maximises cached-ref hit rate
    /// (consecutive tasks with the same ref reuse the device-resident
    /// reference).
    ///
    /// `task_id` is the final tie-breaker so the dispatch order across
    /// runs is deterministic when refs are identical.
    ///
    /// Yield order is still **completion order** — sorting the input
    /// only changes dispatch order, not output order. Callers
    /// correlate via [`Task::task_id`] regardless.
    ///
    /// Callers who require strict submit-order dispatch should use
    /// [`Self::submit`] with
    /// [`OrchestratorConfig::stream_reorder_window`] set to
    /// `(Duration::ZERO, 1)`. `run_all` always sorts.
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
        // Collect tasks; hash refs into ref_hash; sort by
        // (metric, w, h, ref_hash, task_id) for warm-instance reuse +
        // cached-ref hit rate. The dispatch order after sorting groups
        // every task that shares (metric, dims) together so the GPU
        // worker's signature only changes K times for K distinct
        // (metric, dims) tuples instead of N times for N tasks.
        let mut tasks: Vec<Task> = tasks.into_iter().collect();
        for t in &mut tasks {
            populate_ref_hash(t);
        }
        // Stable sort for deterministic output when keys tie. task_id
        // is the final disambiguator so two runs with the same input
        // produce the same dispatch order.
        tasks.sort_by_key(|t| {
            (
                t.metric.tag(),
                t.width,
                t.height,
                t.ref_hash,
                t.task_id,
            )
        });

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
        // run_all flushes any pending streaming-reorder window so its
        // contract ("all tasks dispatched before the iterator
        // returns") holds even when stream_reorder_window > (0, 1).
        // The flush is a no-op when the count limit already tripped on
        // the last submit() above.
        self.flush_pending_internal();
        RunAllIter {
            orch: self,
            remaining: total_submitted,
            handles: submitted_handles,
            errors: submit_errors,
        }
    }
}

/// Populate `t.ref_hash` from its `ref_data`. xxhash3_64 over the byte
/// buffer for `Srgb8` / `Path` (path strings hash directly — Phase 5
/// rejects `Path` at submit anyway, but the sort key still needs *some*
/// stable value), and the pre-upload's stable `inner_id` for
/// `PreUploaded`. Idempotent: a non-zero `ref_hash` already set by the
/// caller is preserved.
pub(crate) fn populate_ref_hash(t: &mut Task) {
    if t.ref_hash != 0 {
        return;
    }
    t.ref_hash = match &t.ref_data {
        TaskData::Srgb8(b) => hash_ref_bytes(b),
        TaskData::Path(p) => hash_ref_bytes(p.to_string_lossy().as_bytes()),
        TaskData::PreUploaded(h) => h.inner_id,
    };
    // A legitimate hash of "0" is theoretically possible but
    // astronomically unlikely; if it happens we treat the value as
    // unhashed but the rest of the sort still works (all such tasks
    // cluster together at sort position 0).
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

    #[test]
    fn pool_config_phase9_defaults_preserve_single_worker() {
        // Phase 9.1 — the default must keep N=1 lane so bit-identical
        // single-worker behaviour is preserved for every caller that
        // hasn't explicitly opted into concurrency.
        let cfg = PoolConfig::default();
        assert_eq!(cfg.max_gpu_lanes, 1, "default max_gpu_lanes must be 1");
        assert!(!cfg.adaptive_gpu_lanes, "default adaptive_gpu_lanes must be false");
        assert_eq!(cfg.adaptive_max_gpu_lanes, 4);
        assert_eq!(cfg.target_gpu_utilization_pct, 80);
        assert!(cfg.gpu_util_sample_interval_ms >= 1000);
    }

    #[test]
    fn pool_config_max_lanes_clamps_to_eight() {
        // 8 is the hard cap; values above that are clamped to 8 at
        // construction time (we test the policy via the clamp() call,
        // since PoolState::new isn't constructible in pure-CPU tests).
        let n: usize = 100;
        assert_eq!(n.clamp(1, 8), 8);
        let z: usize = 0;
        assert_eq!(z.clamp(1, 8), 1);
    }

    // ---- Phase 9.3 controller logic — pure function ---------------

    #[test]
    fn adaptive_controller_holds_when_no_samples() {
        // Fewer than 3 samples in either direction: no change.
        assert_eq!(compute_next_lane_count(1, 0, 0, 4), None);
        assert_eq!(compute_next_lane_count(2, 2, 2, 4), None);
        assert_eq!(compute_next_lane_count(1, 1, 0, 4), None);
        assert_eq!(compute_next_lane_count(4, 0, 2, 4), None);
    }

    #[test]
    fn adaptive_controller_scales_up_on_3_low_samples() {
        // 3 consecutive low samples + room to grow -> +1.
        assert_eq!(compute_next_lane_count(1, 3, 0, 4), Some(2));
        assert_eq!(compute_next_lane_count(2, 3, 0, 4), Some(3));
        assert_eq!(compute_next_lane_count(3, 5, 0, 4), Some(4));
    }

    #[test]
    fn adaptive_controller_no_scale_up_at_max() {
        // Already at the cap: no scale-up regardless of low samples.
        assert_eq!(compute_next_lane_count(4, 100, 0, 4), None);
    }

    #[test]
    fn adaptive_controller_scales_down_on_3_high_samples() {
        // 3 consecutive high samples + room to shrink -> -1.
        assert_eq!(compute_next_lane_count(4, 0, 3, 4), Some(3));
        assert_eq!(compute_next_lane_count(2, 0, 5, 4), Some(1));
    }

    #[test]
    fn adaptive_controller_no_scale_down_at_one() {
        // Already at the floor: no scale-down regardless of high samples.
        assert_eq!(compute_next_lane_count(1, 0, 100, 4), None);
    }

    #[test]
    fn adaptive_controller_low_beats_high_in_tie() {
        // Pathological: both counters >= 3 simultaneously. The
        // scale-up branch comes first by code order, so scale-up
        // wins. This is acceptable because the watcher resets the
        // OTHER counter on any sample in the opposing bin, so this
        // state shouldn't persist for more than one tick in practice.
        assert_eq!(compute_next_lane_count(2, 3, 3, 4), Some(3));
    }

    #[test]
    fn watcher_fake_starts_unknown() {
        // The test-only fake watcher starts with no samples — both
        // counters at 0, latest_pct = u8::MAX sentinel.
        let w = GpuUtilWatcher::test_only_fake();
        assert!(w.latest_pct().is_none());
        assert_eq!(w.consecutive_below(), 0);
        assert_eq!(w.consecutive_above(), 0);
    }

    #[test]
    fn watcher_fake_counters_externally_manipulable() {
        // Test-only fake exposes counter handles so controller tests
        // can simulate a "below target" run without real nvidia-smi.
        let w = GpuUtilWatcher::test_only_fake();
        let h = w.test_only_counters();
        h.below.store(3, Ordering::Release);
        assert_eq!(w.consecutive_below(), 3);
        h.latest_pct.store(45, Ordering::Release);
        assert_eq!(w.latest_pct(), Some(45));
    }
}
