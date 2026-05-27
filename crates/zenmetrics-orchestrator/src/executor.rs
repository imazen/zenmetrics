//! Phase 4 — single-task executor with OOM recovery.
//!
//! Sits one level above the Phase 3 chooser: given a [`Task`], it asks
//! the chooser for a primary backend, constructs the metric via the
//! umbrella `zenmetrics-api`, runs the score, and recovers from OOM
//! errors by walking a fallback ladder
//! (`GpuFull → GpuStrip → GpuStripPair → Cpu`) and re-consulting the
//! chooser after each rejection.
//!
//! Each OOM observation:
//!
//! 1. Records `(backend, size_pixels)` in
//!    [`crate::MetricProfile::cells_failed_oom`].
//! 2. Persists the capability cache to disk **immediately** so the
//!    learning survives a process death mid-task.
//! 3. Re-runs [`crate::Orchestrator::choose_backend`] — the new entry in
//!    `cells_failed_oom` now causes the previously-failing backend to be
//!    rejected as [`crate::RejectReason::KnownOomCell`].
//!
//! Non-OOM errors do **not** trigger fallback. They surface immediately
//! as `Err(MetricApi(...))` so caller-visible mistakes (dim mismatch,
//! backend-not-enabled, etc.) aren't silently retried against backends
//! that also won't work.
//!
//! ## CUDA-only for Phase 4
//!
//! Like the Phase 2 bench, this executor always asks the umbrella for
//! [`zenmetrics_api::Backend::Cuda`]. Multi-runtime dispatch (wgpu / hip)
//! is Phase 5. The orchestrator's `cuda` feature gates this module —
//! callers who build without `cuda` get a clear `cargo` error rather
//! than a runtime "backend not enabled" surprise.
//!
//! ## What's deliberately *not* here
//!
//! - No worker pool, no concurrency. [`crate::Orchestrator::run_single`]
//!   blocks on the calling thread. Phase 5 layers a pool on top.
//! - No cached-reference auto-detect. Phase 5.
//! - No CPU backend execution. Phase 6 wires the per-crate CPU
//!   references; until then every `Backend::Cpu` attempt returns
//!   [`AttemptOutcome::OtherError`] with a `CpuNotYetWired` message and
//!   the ladder advances.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::path::PathBuf;
use std::time::Instant;

use zenmetrics_api::{
    Backend as ApiBackend, Error as ApiError, MemoryMode, Metric, MetricKind, MetricParams, Score,
};

use crate::chooser::{BackendChoice, ChooserError, RejectReason};
use crate::{save_profile, Backend, Orchestrator};

// ---------------------------------------------------------------------------
// Task / TaskData
// ---------------------------------------------------------------------------

/// One scoring job submitted to the executor.
///
/// Per-task `task_id` is the caller's opaque correlation handle — the
/// executor echoes it back on the returned [`TaskResult`] so the caller
/// can match up async-style batch responses against the original
/// submissions in Phase 5. For [`Orchestrator::run_single`] it's just
/// passed through.
#[derive(Debug, Clone)]
pub struct Task {
    /// Caller-chosen correlation identifier. Echoed back unchanged.
    pub task_id: u64,
    /// Reference image bytes (packed sRGB R,G,B,…) or a filesystem path
    /// to load on first use.
    pub ref_data: TaskData,
    /// Distorted image bytes or path.
    pub dist_data: TaskData,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Which metric to compute.
    pub metric: MetricKind,
    /// Per-metric parameters. `None` → [`MetricParams::default_for(metric)`].
    pub params: Option<MetricParams>,
}

/// Source for a single image buffer. The executor materializes this to
/// a `Vec<u8>` once and reuses across every backend attempt for the same
/// task — re-reading from disk on each fallback would be wasteful.
///
/// `PreUploaded` is Phase 5 territory; intentionally absent here.
#[derive(Debug, Clone)]
pub enum TaskData {
    /// Already-loaded packed sRGB `R,G,B,…` bytes (length `width * height * 3`).
    Srgb8(Vec<u8>),
    /// Path to a PNG/JPEG/etc on disk. Loaded on first use via the
    /// `image` crate decoder chain (when Phase 5 wires it). Phase 4
    /// surfaces an `UnsupportedTaskData` error for `Path` because the
    /// loader integration isn't wired yet — pass `Srgb8` directly.
    Path(PathBuf),
}

// ---------------------------------------------------------------------------
// TaskResult / AttemptOutcome
// ---------------------------------------------------------------------------

/// Result of running one [`Task`] through the fallback ladder.
///
/// `outcome` is `Ok(Score)` on first-successful-backend (the common
/// case) or `Err(...)` when no backend in the ladder survived. The
/// `backends_attempted` list captures every backend the ladder tried,
/// including those rejected by the chooser before construction was even
/// attempted — so callers can post-hoc audit the decision tree.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Echoed [`Task::task_id`].
    pub task_id: u64,
    /// Final outcome — either a successful score or the structured
    /// error from the last attempt (and a summary of what was tried).
    pub outcome: std::result::Result<Score, OrchestratorError>,
    /// The backend that actually produced `outcome` when it's `Ok`.
    /// `None` if every attempt failed.
    pub backend_used: Option<Backend>,
    /// Every backend the ladder attempted, in chronological order, with
    /// each attempt's outcome. The successful attempt (if any) is the
    /// last entry with [`AttemptOutcome::Success`].
    pub backends_attempted: Vec<(Backend, AttemptOutcome)>,
    /// Wall time from `run_single` entry to return, in microseconds.
    pub wall_us: u64,
    /// Predictive VRAM ceiling the chooser used for the chosen backend.
    /// `None` if no attempt completed (so no prediction was logged).
    pub vram_peak_mib: Option<usize>,
}

/// Per-attempt outcome inside the fallback ladder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// Backend constructed AND `compute_srgb_u8` returned `Ok(score)`.
    Success,
    /// Construction (`Metric::new_with_memory_mode`) returned a
    /// recognised OOM error (the per-crate `TooBigForFull` family).
    OomAtConstruction,
    /// Construction succeeded but `compute_srgb_u8` failed with a
    /// runtime OOM (typically a cubecl `cudaErrorMemoryAllocation`
    /// bubbling up through the umbrella's `Error::Metric { message }`).
    OomAtRuntime,
    /// Construction or compute failed with a non-OOM error. The string
    /// is the umbrella's `Display` rendering for debugging.
    OtherError(String),
}

// ---------------------------------------------------------------------------
// Extended OrchestratorError variants for Phase 4
// ---------------------------------------------------------------------------

/// Phase 4-specific error variants returned in [`TaskResult::outcome`].
///
/// Distinct from [`crate::OrchestratorError`] (which covers cache I/O,
/// detection failures, etc.) — Phase 4's errors live closer to the
/// per-task surface and don't need to thread back through the
/// orchestrator-construction flow.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum OrchestratorError {
    /// The fallback ladder ran out of backends. Inspect the
    /// `backends_attempted` field on the surrounding [`TaskResult`] for
    /// which backends were tried and why each failed.
    FullyExhausted {
        /// One entry per attempt, in chronological order.
        attempts: Vec<(Backend, AttemptOutcome)>,
    },
    /// The chooser refused to pick a backend (e.g. metric not in the
    /// capability cache, or every candidate rejected by the safety
    /// margin).
    Chooser(ChooserError),
    /// The umbrella metric API surfaced a non-OOM, non-recoverable
    /// error (dimension mismatch, params variant mismatch, etc.).
    /// Wrapped as a string because [`zenmetrics_api::Error`] is
    /// `#[non_exhaustive]` and not `Clone` — the executor only ever
    /// re-emits the rendered message.
    MetricApi(String),
    /// The task carries a [`TaskData`] variant the executor doesn't
    /// know how to materialize yet. Phase 4 wires only `Srgb8`; `Path`
    /// surfaces this until Phase 5 adds the loader.
    UnsupportedTaskData(String),
    /// CPU backend was selected by the ladder, but Phase 4 doesn't yet
    /// wire any CPU executor. Always treated as a fallback-eligible
    /// failure (not a hard error) — the ladder advances to the next
    /// backend on this.
    CpuNotYetWired,
}

impl std::fmt::Display for OrchestratorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorError::FullyExhausted { attempts } => write!(
                f,
                "all {} backend(s) failed — task fully exhausted",
                attempts.len()
            ),
            OrchestratorError::Chooser(e) => write!(f, "chooser: {e}"),
            OrchestratorError::MetricApi(msg) => write!(f, "metric api: {msg}"),
            OrchestratorError::UnsupportedTaskData(msg) => {
                write!(f, "unsupported task data: {msg}")
            }
            OrchestratorError::CpuNotYetWired => {
                write!(f, "CPU backend selected but not yet wired (Phase 6)")
            }
        }
    }
}

impl std::error::Error for OrchestratorError {}

// ---------------------------------------------------------------------------
// Internal: metric wrapper covering Umbrella + cvvdp StripPair
// ---------------------------------------------------------------------------

/// Wrapper so the executor body doesn't care whether the configured
/// backend goes through the umbrella `Metric` (Full / Strip) or the
/// direct cvvdp `CvvdpOpaque` (StripPair).
///
/// Mirrors the shape of `bench::BenchMetric` — both modules share the
/// same per-backend construction matrix; the executor adds OOM-recovery
/// state on top.
enum ExecMetric {
    Umbrella(Box<Metric>),
    CvvdpStripPair(Box<zenmetrics_api::cvvdp::CvvdpOpaque>),
}

impl ExecMetric {
    fn compute(&mut self, r: &[u8], d: &[u8]) -> Result<Score, CallErr> {
        match self {
            ExecMetric::Umbrella(m) => m
                .compute_srgb_u8(r, d)
                .map_err(|e| classify_call_err(&e.to_string())),
            ExecMetric::CvvdpStripPair(c) => c
                .compute_srgb_u8(r, d)
                .map(convert_cvvdp_score)
                .map_err(|e| classify_call_err(&e.to_string())),
        }
    }
}

/// Convert a cvvdp_gpu native `Score` into the umbrella `Score` shape.
/// Mirrors `metric.rs::convert_score` in zenmetrics-api but inlined
/// because we hold a `CvvdpOpaque` directly, not through the umbrella.
fn convert_cvvdp_score(s: zenmetrics_api::cvvdp::Score) -> Score {
    Score {
        value: s.value,
        metric_name: s.metric_name,
        metric_version: s.metric_version,
    }
}

/// Per-attempt construction outcome.
enum ConstructOutcome {
    Ok(ExecMetric),
    Oom,
    Other(String),
}

/// Per-attempt compute outcome.
enum CallErr {
    Oom,
    Other(String),
}

/// Classify a `compute_srgb_u8` error string into OOM vs other. Mirrors
/// the same heuristic the bench uses (`bench.rs::classify_call_err`) —
/// kept in sync deliberately; if the bench's OOM patterns expand, this
/// one should too.
fn classify_call_err(msg: &str) -> CallErr {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("oom")
        || lowered.contains("out of memory")
        || lowered.contains("toobigforfull")
        || lowered.contains("cuda_error_out_of_memory")
        || lowered.contains("cudaerrormemoryallocation")
    {
        CallErr::Oom
    } else {
        CallErr::Other(msg.to_string())
    }
}

/// Classify a constructor error (umbrella `Error`) into OOM vs other.
/// The umbrella wraps the per-crate `TooBigForFull` as
/// `Error::Metric { message: "...TooBigForFull..." }`, so this is a
/// string-match. Same heuristic as `bench.rs::classify_construct_err`.
fn classify_construct_err(e: ApiError) -> ConstructOutcome {
    let msg = e.to_string();
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("toobigforfull")
        || lowered.contains("out of memory")
        || lowered.contains("oom")
    {
        ConstructOutcome::Oom
    } else {
        ConstructOutcome::Other(msg)
    }
}

/// Same as above but for the direct cvvdp constructor (returns
/// `cvvdp_gpu::Error`, which has a dedicated `TooBigForFull` variant).
fn classify_cvvdp_construct_err(e: zenmetrics_api::cvvdp::Error) -> ConstructOutcome {
    // Pattern-match on the structured variant first for a clean signal,
    // then fall through to the string heuristic in case future cubecl
    // OOMs surface as InvalidImageSize (the catch-all variant for GPU
    // readback / dispatch errors).
    if matches!(&e, zenmetrics_api::cvvdp::Error::TooBigForFull { .. }) {
        return ConstructOutcome::Oom;
    }
    let msg = e.to_string();
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("toobigforfull")
        || lowered.contains("out of memory")
        || lowered.contains("oom")
    {
        ConstructOutcome::Oom
    } else {
        ConstructOutcome::Other(msg)
    }
}

// ---------------------------------------------------------------------------
// Construction for each backend
// ---------------------------------------------------------------------------

fn construct(
    kind: MetricKind,
    backend: Backend,
    width: u32,
    height: u32,
    params: Option<MetricParams>,
) -> ConstructOutcome {
    match backend {
        Backend::GpuFull => construct_via_umbrella(kind, width, height, params, MemoryMode::Full),
        Backend::GpuStrip => {
            construct_via_umbrella(kind, width, height, params, MemoryMode::Strip { h_body: None })
        }
        Backend::GpuStripPair => {
            // StripPair is cvvdp-specific. Other metrics fall through to
            // "Other" so the ladder advances; the chooser shouldn't pick
            // StripPair for non-cvvdp metrics, but defend in depth.
            if kind != MetricKind::Cvvdp {
                return ConstructOutcome::Other(format!(
                    "GpuStripPair not supported by metric '{}'",
                    kind.tag()
                ));
            }
            construct_cvvdp_strip_pair(width, height, params)
        }
        Backend::Cpu => ConstructOutcome::Other("CpuNotYetWired".to_string()),
    }
}

fn construct_via_umbrella(
    kind: MetricKind,
    width: u32,
    height: u32,
    params: Option<MetricParams>,
    mode: MemoryMode,
) -> ConstructOutcome {
    let params = match params {
        Some(p) => p,
        None => match MetricParams::try_default_for(kind) {
            Ok(p) => p,
            Err(e) => return ConstructOutcome::Other(e.to_string()),
        },
    };
    match Metric::new_with_memory_mode(kind, ApiBackend::Cuda, width, height, params, mode) {
        Ok(m) => ConstructOutcome::Ok(ExecMetric::Umbrella(Box::new(m))),
        Err(e) => classify_construct_err(e),
    }
}

fn construct_cvvdp_strip_pair(
    width: u32,
    height: u32,
    params: Option<MetricParams>,
) -> ConstructOutcome {
    use zenmetrics_api::cvvdp::{CvvdpOpaque, CvvdpParams, MemoryMode as CvvdpMode};
    // Extract the cvvdp params if the caller supplied them; otherwise
    // default. We MUST NOT panic on a variant mismatch — surface it as
    // an Other error so the ladder advances cleanly.
    let p: CvvdpParams = match params {
        Some(MetricParams::Cvvdp(p)) => p,
        Some(_) => {
            return ConstructOutcome::Other(
                "MetricParams variant != Cvvdp for cvvdp StripPair construction".to_string(),
            );
        }
        None => CvvdpParams::default(),
    };
    let mode = CvvdpMode::StripPair {
        h_body: Some(256),
    };
    match CvvdpOpaque::new_with_memory_mode(
        zenmetrics_api::cvvdp::Backend::Cuda,
        width,
        height,
        p,
        mode,
    ) {
        Ok(c) => ConstructOutcome::Ok(ExecMetric::CvvdpStripPair(Box::new(c))),
        Err(e) => classify_cvvdp_construct_err(e),
    }
}

// ---------------------------------------------------------------------------
// Materialize TaskData → bytes
// ---------------------------------------------------------------------------

/// Materialize `data` into a packed sRGB `Vec<u8>`. Phase 4 wires only
/// `Srgb8`; `Path` surfaces a clear "Phase 5" error.
fn materialize(data: TaskData) -> Result<Vec<u8>, OrchestratorError> {
    match data {
        TaskData::Srgb8(b) => Ok(b),
        TaskData::Path(p) => Err(OrchestratorError::UnsupportedTaskData(format!(
            "TaskData::Path({}) not yet wired (Phase 5)",
            p.display()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Orchestrator::run_single — the Phase 4 entry point
// ---------------------------------------------------------------------------

impl Orchestrator {
    /// Run a single task end-to-end. Synchronous; blocks until done or
    /// every backend in the fallback ladder is exhausted.
    ///
    /// ## Ladder
    ///
    /// 1. Materialize `ref_data` + `dist_data` into byte buffers (one
    ///    load per task, reused across every attempt — no repeated
    ///    disk I/O on fallback).
    /// 2. Ask the chooser for the best backend given the current
    ///    capability cache + live free VRAM (the chooser's own probe).
    /// 3. Construct the metric. On constructor OOM:
    ///    - Append `(backend, size_pixels)` to
    ///      [`crate::MetricProfile::cells_failed_oom`].
    ///    - **Persist the cache to disk immediately** so a process
    ///      crash mid-task doesn't lose the learning.
    ///    - Re-ask the chooser; the new OOM entry rejects this backend
    ///      as [`crate::RejectReason::KnownOomCell`].
    /// 4. Run `compute_srgb_u8`. On runtime OOM, same treatment as
    ///    constructor OOM. On non-OOM error, surface immediately as
    ///    `Err(MetricApi(...))` — no fallback for caller-visible bugs.
    /// 5. On success, return [`TaskResult`] with `outcome = Ok(score)`
    ///    and `backend_used = Some(backend)`.
    /// 6. If the chooser returns
    ///    [`ChooserError::NoFeasibleBackend`] at any iteration, return
    ///    `Err(FullyExhausted)`.
    ///
    /// ## OOM detection
    ///
    /// - Construction: `Error::Metric { message }` whose message
    ///   contains `TooBigForFull` / `out of memory` / `OOM`.
    /// - Runtime: same heuristic on the `compute_srgb_u8` error
    ///   string. cubecl's CUDA runtime surfaces OOM as a backend error
    ///   whose `Display` contains `cudaErrorMemoryAllocation`
    ///   (verified against the v0.7 cubecl-cuda backend).
    ///
    /// **Known limitation**: a non-CUDA cubecl backend may surface
    /// runtime OOM under a different error string. Phase 5 widens the
    /// pattern list as we encounter them.
    pub fn run_single(&mut self, task: Task) -> TaskResult {
        let t_start = Instant::now();
        let task_id = task.task_id;
        let metric = task.metric;
        let width = task.width;
        let height = task.height;
        let pixels = (width as u64) * (height as u64);

        // Materialize buffers up front — every backend attempt reuses
        // the same bytes. Errors here are unrecoverable.
        let ref_bytes = match materialize(task.ref_data) {
            Ok(b) => b,
            Err(e) => return finalize_err(task_id, e, t_start, Vec::new()),
        };
        let dist_bytes = match materialize(task.dist_data) {
            Ok(b) => b,
            Err(e) => return finalize_err(task_id, e, t_start, Vec::new()),
        };

        let params = task.params;
        let mut attempts: Vec<(Backend, AttemptOutcome)> = Vec::new();
        // Cap iterations defensively — at most 4 backend variants so 5
        // iterations is impossible. Guard against an infinite loop if a
        // future chooser change forgets to reject Cpu / a previously-
        // OOMed backend.
        let mut last_choice_vram_mib: Option<usize> = None;

        for _iteration in 0..5 {
            // Re-ask the chooser each iteration — the previous attempt's
            // OOM observation may have updated cells_failed_oom.
            let choice = match self.choose_backend_for_task(&crate::chooser::TaskShape {
                metric,
                width,
                height,
            }) {
                Ok(c) => c,
                Err(e) => {
                    // Convert NoFeasibleBackend into FullyExhausted when
                    // we already attempted at least one backend; UnknownMetric
                    // / first-iteration NoFeasibleBackend remain Chooser errors.
                    let err = match (&e, attempts.is_empty()) {
                        (ChooserError::NoFeasibleBackend { .. }, false) => {
                            OrchestratorError::FullyExhausted {
                                attempts: attempts.clone(),
                            }
                        }
                        _ => OrchestratorError::Chooser(e),
                    };
                    return finalize_err(task_id, err, t_start, attempts);
                }
            };
            let backend = choice.backend;
            last_choice_vram_mib = Some(choice.predicted_vram_mib);

            // Construct.
            match construct(metric, backend, width, height, params.clone()) {
                ConstructOutcome::Ok(mut em) => {
                    // Try compute.
                    match em.compute(&ref_bytes, &dist_bytes) {
                        Ok(score) => {
                            attempts.push((backend, AttemptOutcome::Success));
                            return TaskResult {
                                task_id,
                                outcome: Ok(score),
                                backend_used: Some(backend),
                                backends_attempted: attempts,
                                wall_us: elapsed_us(t_start),
                                vram_peak_mib: last_choice_vram_mib,
                            };
                        }
                        Err(CallErr::Oom) => {
                            attempts.push((backend, AttemptOutcome::OomAtRuntime));
                            // Drop the metric instance to release any
                            // device buffers it still holds before
                            // attempting the next backend.
                            drop(em);
                            self.record_oom_and_persist(metric, backend, pixels);
                            continue;
                        }
                        Err(CallErr::Other(msg)) => {
                            attempts
                                .push((backend, AttemptOutcome::OtherError(msg.clone())));
                            // Non-OOM compute error → surface immediately.
                            return finalize_err(
                                task_id,
                                OrchestratorError::MetricApi(msg),
                                t_start,
                                attempts,
                            );
                        }
                    }
                }
                ConstructOutcome::Oom => {
                    attempts.push((backend, AttemptOutcome::OomAtConstruction));
                    self.record_oom_and_persist(metric, backend, pixels);
                    continue;
                }
                ConstructOutcome::Other(msg) => {
                    // CpuNotYetWired is the recoverable "advance the
                    // ladder" case. Anything else is a hard error.
                    if msg == "CpuNotYetWired" {
                        attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
                        // CPU isn't in cells_failed_oom — the chooser
                        // already rejects Cpu via CpuNotYetWired. But if
                        // we get here, something forced us to try Cpu
                        // anyway (e.g. test scenario). Mark and continue;
                        // the chooser will reject the same backend next
                        // iteration so the loop exits via NoFeasibleBackend.
                        self.record_oom_and_persist(metric, backend, pixels);
                        continue;
                    }
                    attempts
                        .push((backend, AttemptOutcome::OtherError(msg.clone())));
                    return finalize_err(
                        task_id,
                        OrchestratorError::MetricApi(msg),
                        t_start,
                        attempts,
                    );
                }
            }
        }

        // Iteration cap — shouldn't be reachable in production. Surface
        // as FullyExhausted for visibility.
        TaskResult {
            task_id,
            outcome: Err(OrchestratorError::FullyExhausted {
                attempts: attempts.clone(),
            }),
            backend_used: None,
            backends_attempted: attempts,
            wall_us: elapsed_us(t_start),
            vram_peak_mib: last_choice_vram_mib,
        }
    }

    /// Append `(backend, pixels)` to the metric's
    /// [`MetricProfile::cells_failed_oom`] list and persist the cache
    /// to disk. A save failure is logged via `eprintln!` and swallowed
    /// — the in-memory state is still updated so the next iteration
    /// of the ladder sees the new OOM cell. Persistence is best-effort:
    /// losing it across a crash means the next process starts the
    /// ladder fresh, but the current run still completes correctly.
    fn record_oom_and_persist(&mut self, metric: MetricKind, backend: Backend, pixels: u64) {
        let tag = metric.tag().to_string();
        // Ensure the metric has an entry (Phase 2 populates this in the
        // common path, but a partial cache or a synthetic test setup
        // may not have one).
        let entry = self.capability_mut().metrics.entry(tag).or_default();
        // Avoid duplicate entries — `cells_failed_oom` is monotonic.
        let already = entry
            .cells_failed_oom
            .iter()
            .any(|&(b, px)| b == backend && px == pixels);
        if !already {
            entry.cells_failed_oom.push((backend, pixels));
        }
        // Persist to disk so a process death mid-task doesn't drop the
        // learning. Log + continue on failure.
        let path = self.cache_path();
        if let Err(e) = save_profile(&path, self.capability()) {
            eprintln!(
                "zenmetrics-orchestrator: failed to persist OOM-cache update at {}: {}",
                path.display(),
                e
            );
        }
    }

    /// Mutable accessor for [`Self::capability`]. Internal — kept out
    /// of the public surface so callers don't accidentally invalidate
    /// the `machine_hash` ↔ cache-file invariant.
    fn capability_mut(&mut self) -> &mut crate::CapabilityProfile {
        &mut self.capability
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn elapsed_us(start: Instant) -> u64 {
    start.elapsed().as_micros() as u64
}

fn finalize_err(
    task_id: u64,
    err: OrchestratorError,
    t_start: Instant,
    attempts: Vec<(Backend, AttemptOutcome)>,
) -> TaskResult {
    TaskResult {
        task_id,
        outcome: Err(err),
        backend_used: None,
        backends_attempted: attempts,
        wall_us: elapsed_us(t_start),
        vram_peak_mib: None,
    }
}

// Keep `RejectReason` in scope so future executor-side branching on
// rejection types compiles cleanly; Phase 4 only uses it indirectly via
// the chooser's `BackendChoice` but Phase 5 will want to surface
// per-candidate reasons up to the caller.
#[allow(dead_code)]
fn _force_use_reject_reason(_r: RejectReason) {}

// Keep `BackendChoice` re-exported in scope for downstream tests; the
// chooser already re-exports it from the crate root.
#[allow(dead_code)]
fn _force_use_backend_choice(_c: BackendChoice) {}
