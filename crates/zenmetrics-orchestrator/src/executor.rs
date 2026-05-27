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
//!
//! ## CPU backend wiring (Phase 6)
//!
//! As of Phase 6 the executor constructs a real
//! [`crate::cpu_adapter::CpuAdapter`] when the ladder picks
//! [`Backend::Cpu`] — the previous `CpuNotYetWired` shim was removed.
//! Per-metric mapping + feature flags live in
//! `crates/zenmetrics-orchestrator/docs/CPU_BACKENDS.md`. Metrics
//! without a clean CPU reference (Iwssim) surface
//! [`OrchestratorError::CpuMetricUnavailable`] and the ladder advances
//! to the next candidate; the chooser already filters them out at
//! decision time so this only fires when a synthetic test forces the
//! Cpu branch.

#![cfg(all(feature = "bench", feature = "cuda"))]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use zenmetrics_api::{
    Backend as ApiBackend, Error as ApiError, MemoryMode, Metric, MetricKind, MetricParams, Score,
};

use crate::chooser::{BackendChoice, ChooserError, RejectReason};
use crate::cpu_adapter::{CpuAdapter, CpuAdapterError};
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
///
/// `ref_hash` is the Phase 7.6 sort key for reorder batching. Callers
/// can leave it at `0` (the default) — the orchestrator populates it
/// from `ref_data` before sorting in [`Orchestrator::run_all`] /
/// [`Orchestrator::submit`]. For `TaskData::PreUploaded`, the hash is
/// the pre-upload's stable `ref_id` so identical handles cluster.
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
    /// Phase 7.6 sort key. Set to `0` (the default) at construction; the
    /// orchestrator populates this with `xxhash3_64(ref_bytes)` (or the
    /// pre-upload's stable id) before reorder batching. Callers MAY set
    /// it themselves to skip the auto-hash, but the orchestrator will
    /// overwrite a zero value either way. See
    /// `crates/zenmetrics-orchestrator/docs/REORDERING_DESIGN.md`.
    pub ref_hash: u64,
}

/// Source for a single image buffer. The executor materializes this to
/// a `Vec<u8>` once and reuses across every backend attempt for the same
/// task — re-reading from disk on each fallback would be wasteful.
///
/// `PreUploaded` is Phase 5: callers who want zero auto-hash overhead
/// can pre-upload a reference via
/// [`crate::Orchestrator::upload_reference`] and pass the resulting
/// [`crate::TaskRefHandle`] as `TaskData::PreUploaded`. The handle is
/// only valid as a *reference* — passing it as the distorted side
/// returns an error at submit time.
#[derive(Debug, Clone)]
pub enum TaskData {
    /// Already-loaded packed sRGB `R,G,B,…` bytes (length `width * height * 3`).
    Srgb8(Vec<u8>),
    /// Path to a PNG/JPEG/etc on disk. Loaded on first use via the
    /// `image` crate decoder chain (when Phase 5 wires it). Phase 4
    /// surfaces an `UnsupportedTaskData` error for `Path` because the
    /// loader integration isn't wired yet — pass `Srgb8` directly.
    Path(PathBuf),
    /// Pre-uploaded reference handle. The worker pool skips the
    /// xxhash3_64 ref-bytes hash entirely when the task arrives with
    /// `PreUploaded` instead of `Srgb8`. The handle must match the
    /// task's `(metric, width, height)` signature — submit returns
    /// [`OrchestratorError::MetricApi`] otherwise.
    PreUploaded(crate::pool::TaskRefHandle),
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
    /// Per-task output columns the caller should emit to a parquet /
    /// TSV sidecar. Phase 7.5 surface so multi-column metrics (butter
    /// = `butteraugli_max_gpu` + `butteraugli_pnorm3_gpu`; cvvdp =
    /// `cvvdp_imazen_v<VERSION>`) can flow through the orchestrator
    /// without losing the column names the legacy direct-dispatch
    /// path emits.
    ///
    /// Empty `BTreeMap` when `outcome` is `Err` (no columns to write).
    /// When `outcome` is `Ok`, contains at minimum
    /// `{ metric.tag(): score.value }`; metric-specific extras
    /// (butter pnorm_3, cvvdp versioned column) appear alongside.
    ///
    /// The map is sorted by key so callers iterating it produce
    /// deterministic column order in their output files.
    pub output_columns: BTreeMap<String, f64>,
    /// Echoed `metric_version` from the successful backend's
    /// [`Score::metric_version`]. `None` when no attempt completed.
    /// Useful for cvvdp's per-version column tagging without forcing
    /// callers to re-parse it out of the `output_columns` keys.
    pub metric_version: Option<&'static str>,
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
    ///
    /// **Phase 6**: this variant is no longer produced under normal
    /// operation — CPU backends are wired. Kept for backwards
    /// compatibility (Phase 5 callers that match on it still compile).
    /// New Phase 6 errors use [`Self::CpuMetricUnavailable`] /
    /// [`Self::CpuBackendUnavailable`].
    CpuNotYetWired,
    /// The selected metric has no CPU reference implementation in this
    /// release. Currently this means Iwssim (no clean upstream port);
    /// see `docs/CPU_BACKENDS.md`. Recoverable — the ladder advances.
    CpuMetricUnavailable {
        /// The metric whose CPU adapter could not be constructed.
        metric: MetricKind,
    },
    /// The build was compiled without the `cpu-<metric>` feature
    /// needed for this metric's CPU reference. Distinct from
    /// `CpuMetricUnavailable` (which is a permanent upstream gap) —
    /// callers can rebuild with the missing feature. Recoverable.
    CpuBackendUnavailable {
        /// The metric whose CPU adapter feature is disabled.
        metric: MetricKind,
        /// Which feature flag would enable it.
        required_feature: &'static str,
    },
    /// The CPU adapter constructed but the underlying CPU reference
    /// crate failed at runtime (allocation, validation, internal
    /// error). Non-recoverable: ladder advances but the next CPU
    /// attempt at the same size will likely fail the same way.
    CpuFailed(String),
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
            OrchestratorError::CpuMetricUnavailable { metric } => write!(
                f,
                "metric '{}' has no CPU reference implementation",
                metric.tag()
            ),
            OrchestratorError::CpuBackendUnavailable {
                metric,
                required_feature,
            } => write!(
                f,
                "CPU backend for '{}' disabled (rebuild with --features {required_feature})",
                metric.tag()
            ),
            OrchestratorError::CpuFailed(msg) => write!(f, "CPU backend failed: {msg}"),
        }
    }
}

impl std::error::Error for OrchestratorError {}

// ---------------------------------------------------------------------------
// Internal: metric wrapper covering Umbrella + cvvdp StripPair
// ---------------------------------------------------------------------------

/// Wrapper so the executor body doesn't care whether the configured
/// backend goes through the umbrella `Metric` (Full / Strip), the
/// direct cvvdp `CvvdpOpaque` (StripPair), or the per-metric CPU
/// reference (Phase 6).
///
/// Mirrors the shape of `bench::BenchMetric` — both modules share the
/// same per-backend construction matrix; the executor adds OOM-recovery
/// state on top.
///
/// `pub(crate)` so the Phase 5 worker pool (`pool.rs`) can hold a warm
/// instance and dispatch through [`Self::compute`] /
/// [`Self::set_reference`] / [`Self::compute_with_cached_reference`].
pub(crate) enum ExecMetric {
    Umbrella(Box<Metric>),
    CvvdpStripPair(Box<zenmetrics_api::cvvdp::CvvdpOpaque>),
    /// Phase 6: CPU reference implementation. The inner [`CpuAdapter`]
    /// dispatches to the right per-metric crate at runtime; backends
    /// without a matching feature flag never reach this branch
    /// (construction routes them straight to
    /// [`ConstructOutcome::Other`] / `CpuBackendUnavailable`).
    Cpu(Box<CpuAdapter>),
}

impl ExecMetric {
    /// Phase 4 entrypoint. Returns the private `CallErr` so the
    /// `run_single` ladder can match on it without the `pub(crate)`
    /// shim's wrapper variants. Phase 5 calls
    /// [`Self::compute`] (the `pub(crate)` shim added below) instead.
    ///
    /// Phase 7.5: the inner machinery is `compute_phase4_with_extras`,
    /// which threads metric-specific extras (butter `pnorm_3`,
    /// cvvdp's versioned column tag) up to the caller. This wrapper
    /// keeps the legacy "just the score" call site working.
    ///
    /// `#[allow(dead_code)]` because Phase 7.5 moved every direct
    /// caller to `compute_phase4_with_extras`; the wrapper is kept
    /// for any in-tree consumer that only needs the primary score.
    #[allow(dead_code)]
    fn compute_phase4(&mut self, r: &[u8], d: &[u8]) -> Result<Score, CallErr> {
        self.compute_phase4_with_extras(r, d).map(|(s, _)| s)
    }

    /// Phase 7.5 entrypoint. Returns `(Score, extras)` where `extras`
    /// carries metric-specific output columns beyond the primary
    /// score. The contract:
    ///
    /// - Default (every metric except butter and cvvdp): `extras` is
    ///   empty. The caller defaults to `{ metric.tag(): score.value }`.
    /// - Butter (umbrella path): `extras` contains
    ///   `butteraugli_pnorm3_gpu` -> `pnorm_3` from the same fused
    ///   reduction kernel. The opaque path's `compute_srgb_u8` used
    ///   to drop this column; the new
    ///   `ButteraugliOpaque::compute_srgb_u8_with_pnorm3` API restores
    ///   it without re-running the kernel. The primary score column
    ///   (`butteraugli_max_gpu`) is still surfaced via the standard
    ///   `Score.value` -> `metric.tag()` mapping at the caller.
    /// - Cvvdp: `extras` is empty; the cvvdp versioned column
    ///   (`cvvdp_imazen_v<VERSION>`) is computed from
    ///   `Score::metric_version` at the caller, which is the only
    ///   place that knows the build-time `CVVDP_IMPL_TAG` override.
    fn compute_phase4_with_extras(
        &mut self,
        r: &[u8],
        d: &[u8],
    ) -> Result<(Score, BTreeMap<String, f64>), CallErr> {
        match self {
            ExecMetric::Umbrella(m) => {
                // Butter via the umbrella: peek at the inner Butter
                // variant and route through the pnorm3-returning API
                // so we don't lose the second column. Every other
                // umbrella metric stays on the regular path.
                #[cfg(feature = "bench")]
                if let Metric::Butter(opaque) = m.as_mut() {
                    return match opaque.compute_srgb_u8_with_pnorm3(r, d) {
                        Ok((s, pnorm3)) => {
                            let score = Score {
                                value: s.value,
                                metric_name: s.metric_name,
                                metric_version: s.metric_version,
                            };
                            let mut extras = BTreeMap::new();
                            // Match the legacy cache.rs column name —
                            // see `crates/zen-metrics-cli/src/metrics/cache.rs`
                            // which emits `butteraugli_pnorm3_gpu`
                            // alongside `butteraugli_max_gpu`.
                            extras.insert(
                                "butteraugli_pnorm3_gpu".to_string(),
                                pnorm3,
                            );
                            Ok((score, extras))
                        }
                        Err(e) => Err(classify_call_err(&e.to_string())),
                    };
                }
                m.compute_srgb_u8(r, d)
                    .map(|s| (s, BTreeMap::new()))
                    .map_err(|e| classify_call_err(&e.to_string()))
            }
            ExecMetric::CvvdpStripPair(c) => c
                .compute_srgb_u8(r, d)
                .map(|s| (convert_cvvdp_score(s), BTreeMap::new()))
                .map_err(|e| classify_call_err(&e.to_string())),
            ExecMetric::Cpu(adapter) => {
                // CPU adapters never OOM in the GPU sense; an allocation
                // failure inside ssimulacra2/etc surfaces as a panic
                // (which we can't catch) or an Err here that we treat
                // as a hard "Other" — the ladder ends. The chooser
                // ensures CPU is the last attempt anyway.
                //
                // CPU adapters never produce extras today; if a future
                // cpu-butter adapter starts returning pnorm_3 too,
                // wire the extras BTreeMap through here.
                match adapter.compute(r, d) {
                    Ok(s) => Ok((s, BTreeMap::new())),
                    Err(e) => Err(CallErr::Other(e.to_string())),
                }
            }
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

/// Phase 8a: detect runtime-libcuda-missing error patterns. cubecl-cuda
/// lazily dlopens `libcuda.so.1` on the first kernel launch; if the
/// library isn't on the system (CPU-only host with `--features cuda`
/// build, snap-docker without nvidia-container-toolkit, etc.) the
/// returned error contains tokens like `cuInit`,
/// `CUDA_ERROR_NOT_INITIALIZED`, `libcuda.so`, or `DriverError`.
///
/// When the executor sees one of these patterns it marks the
/// orchestrator's capability profile as `gpu.present = false` so all
/// subsequent backend choices skip GPU candidates with
/// [`RejectReason::NoGpuPresent`] instead of churning through the same
/// dlopen failure. The downgrade is persisted to disk so a process
/// crash mid-task doesn't lose the learning.
pub(crate) fn is_no_cuda_driver(msg: &str) -> bool {
    let lowered = msg.to_ascii_lowercase();
    // Match the most informative tokens first. Each one is a clear
    // signal that the runtime driver is missing, not that this
    // particular kernel ran out of memory.
    lowered.contains("libcuda.so")
        || lowered.contains("cuinit")
        || lowered.contains("cuda_error_not_initialized")
        || lowered.contains("cuda_error_no_device")
        || lowered.contains("cuda_error_operating_system")
        || lowered.contains("error_library_not_found")
        || lowered.contains("nvml")
        // cubecl-cuda's top-level wrapper. Covers the cases where the
        // inner cuInit error is wrapped: `DriverError(...)`.
        || lowered.contains("drivererror")
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

/// Phase 7.7.1: construct the metric for a given backend.
///
/// The `after_oom` parameter controls memory-mode selection for GPU
/// backends:
///
/// - `after_oom == false` (first attempt for this task): pass
///   [`MemoryMode::Auto`] to the per-crate constructor so the metric's
///   own `resolve_auto` policy owns the choice. This is the legacy
///   `MetricCache::new()` behaviour — butter's strip-preferred resolver,
///   ssim2's full-by-default resolver, etc. Matching it is the whole
///   reason Phase 7.7's parity sweep gates the orchestrator default.
/// - `after_oom == true` (OOM ladder retry): force the explicit mode
///   matching the chosen backend ([`Backend::GpuFull`] →
///   [`MemoryMode::Full`], [`Backend::GpuStrip`] →
///   [`MemoryMode::Strip { h_body: None }`]). This prevents the
///   per-crate Auto from re-picking the mode that just OOM'd —
///   `cells_failed_oom` is the chooser's signal, not the per-crate
///   resolver's, so without forcing the mode we'd churn between Full
///   and Strip on every ladder step.
///
/// `Backend::GpuStripPair` is cvvdp-only and always uses cvvdp's
/// explicit `StripPair` mode (no Auto equivalent). `Backend::Cpu` is
/// unaffected by `after_oom`.
fn construct(
    kind: MetricKind,
    backend: Backend,
    width: u32,
    height: u32,
    params: Option<MetricParams>,
    after_oom: bool,
) -> ConstructOutcome {
    match backend {
        Backend::GpuFull => {
            let mode = if after_oom {
                MemoryMode::Full
            } else {
                MemoryMode::Auto
            };
            construct_via_umbrella(kind, width, height, params, mode)
        }
        Backend::GpuStrip => {
            let mode = if after_oom {
                MemoryMode::Strip { h_body: None }
            } else {
                MemoryMode::Auto
            };
            construct_via_umbrella(kind, width, height, params, mode)
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
        Backend::Cpu => construct_cpu(kind, width, height, params),
    }
}

/// Phase 6: build a CPU adapter for the requested metric. Routes the
/// adapter's structured errors into the executor's ConstructOutcome
/// shape so the ladder can advance / fail cleanly.
fn construct_cpu(
    kind: MetricKind,
    width: u32,
    height: u32,
    params: Option<MetricParams>,
) -> ConstructOutcome {
    let params = match params {
        Some(p) => p,
        None => match MetricParams::try_default_for(kind) {
            Ok(p) => p,
            Err(e) => return ConstructOutcome::Other(e.to_string()),
        },
    };
    match CpuAdapter::new(kind, width, height, &params) {
        Ok(adapter) => ConstructOutcome::Ok(ExecMetric::Cpu(Box::new(adapter))),
        Err(CpuAdapterError::FeatureNotEnabled(k)) => {
            // Format a sentinel that the executor recognises so it can
            // surface a structured `CpuBackendUnavailable` rather than a
            // generic MetricApi.
            ConstructOutcome::Other(format!(
                "CpuBackendUnavailable:{}:cpu-{}",
                k.tag(),
                k.tag()
            ))
        }
        Err(CpuAdapterError::Unavailable(k)) => {
            ConstructOutcome::Other(format!("CpuMetricUnavailable:{}", k.tag()))
        }
        Err(CpuAdapterError::Failed(msg)) => {
            ConstructOutcome::Other(format!("CpuFailed:{msg}"))
        }
        Err(CpuAdapterError::InvalidInputSize { expected, got }) => {
            ConstructOutcome::Other(format!(
                "CpuFailed:invalid input size (expected {expected}, got {got})"
            ))
        }
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

/// Materialize `data` into a packed sRGB `Vec<u8>`. Phase 4's
/// `run_single` wires only `Srgb8`; `Path` and `PreUploaded` surface
/// a clear "Phase 5 — use submit/run_all" error. Phase 5's worker
/// pool resolves `PreUploaded` against its own state table before
/// dispatching the worker, so this fall-through is `run_single`-only.
fn materialize(data: TaskData) -> Result<Vec<u8>, OrchestratorError> {
    match data {
        TaskData::Srgb8(b) => Ok(b),
        TaskData::Path(p) => Err(OrchestratorError::UnsupportedTaskData(format!(
            "TaskData::Path({}) not yet wired (Phase 5)",
            p.display()
        ))),
        TaskData::PreUploaded(h) => Err(OrchestratorError::UnsupportedTaskData(format!(
            "TaskData::PreUploaded(id={}) is for submit/run_all only; run_single uses Srgb8",
            h.inner_id
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
        // Phase 7.7.1: track whether a prior attempt OOM'd so the
        // per-iteration `construct` knows whether to pass MemoryMode::Auto
        // (first attempt, let per-crate own policy) or force the explicit
        // mode matching the chosen backend (post-OOM, chooser already
        // ruled out the bigger mode via cells_failed_oom).
        let mut after_oom = false;

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
            match construct(metric, backend, width, height, params.clone(), after_oom) {
                ConstructOutcome::Ok(mut em) => {
                    // Try compute. Phase 7.5 routes through the extras
                    // path so butter's pnorm_3 column survives all the
                    // way back to TaskResult.
                    match em.compute_phase4_with_extras(&ref_bytes, &dist_bytes) {
                        Ok((score, extras)) => {
                            attempts.push((backend, AttemptOutcome::Success));
                            let output_columns =
                                build_output_columns(metric, &score, &extras);
                            return TaskResult {
                                task_id,
                                outcome: Ok(score),
                                backend_used: Some(backend),
                                backends_attempted: attempts,
                                wall_us: elapsed_us(t_start),
                                vram_peak_mib: last_choice_vram_mib,
                                output_columns,
                                metric_version: Some(score.metric_version),
                            };
                        }
                        Err(CallErr::Oom) => {
                            attempts.push((backend, AttemptOutcome::OomAtRuntime));
                            // Drop the metric instance to release any
                            // device buffers it still holds before
                            // attempting the next backend.
                            drop(em);
                            self.record_oom_and_persist(metric, backend, pixels);
                            after_oom = true;
                            continue;
                        }
                        Err(CallErr::Other(msg)) => {
                            attempts
                                .push((backend, AttemptOutcome::OtherError(msg.clone())));
                            // Phase 8a: a runtime cuInit / libcuda
                            // failure can also surface here (e.g.,
                            // construct succeeded by trapping a soft
                            // error but the first kernel launch
                            // explodes). Same treatment as the
                            // construct-time path — downgrade + advance.
                            if matches!(
                                backend,
                                Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair
                            ) && is_no_cuda_driver(&msg)
                            {
                                drop(em);
                                self.downgrade_to_no_gpu_and_persist();
                                continue;
                            }
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
                    after_oom = true;
                    continue;
                }
                ConstructOutcome::Other(msg) => {
                    // Phase 8a: libcuda dlopen failure at runtime —
                    // mark gpu.present = false in the capability cache
                    // and persist so the next iteration's chooser
                    // rejects every GPU backend with NoGpuPresent.
                    // This handles the "build with --features cuda on
                    // a host where libcuda.so.1 is missing" case where
                    // detect_gpu() succeeded (nvidia-smi worked) but
                    // the cubecl-cuda runtime can't actually launch a
                    // kernel. Treat it as a recoverable per-task event
                    // for THIS attempt — record and advance ladder.
                    if matches!(
                        backend,
                        Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair
                    ) && is_no_cuda_driver(&msg)
                    {
                        attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
                        self.downgrade_to_no_gpu_and_persist();
                        // Stay in the loop; the next iteration's
                        // chooser sees gpu.present = false and picks
                        // Cpu (or returns NoFeasibleBackend if no
                        // CPU candidate is available).
                        continue;
                    }
                    // Phase 6 sentinel: CpuMetricUnavailable means the
                    // metric has no CPU reference (Iwssim). Advance the
                    // ladder so a different backend can be picked.
                    if let Some(_tag) = msg.strip_prefix("CpuMetricUnavailable:") {
                        attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
                        // Mark the cell as failed so the chooser doesn't
                        // pick this backend again at the same size.
                        self.record_oom_and_persist(metric, backend, pixels);
                        continue;
                    }
                    // Phase 6 sentinel: CpuBackendUnavailable means the
                    // build doesn't include the feature for this metric.
                    // Same recovery as Unavailable — advance ladder.
                    if msg.starts_with("CpuBackendUnavailable:") {
                        attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
                        self.record_oom_and_persist(metric, backend, pixels);
                        continue;
                    }
                    // Phase 6 sentinel: CpuFailed is a real CPU runtime
                    // error. Surface as MetricApi — the operator
                    // probably needs to investigate.
                    if let Some(real) = msg.strip_prefix("CpuFailed:") {
                        attempts.push((
                            backend,
                            AttemptOutcome::OtherError(real.to_string()),
                        ));
                        return finalize_err(
                            task_id,
                            OrchestratorError::CpuFailed(real.to_string()),
                            t_start,
                            attempts,
                        );
                    }
                    // Pre-Phase-6 legacy: CpuNotYetWired sentinel from
                    // older build paths. Kept for completeness; modern
                    // construct() never emits this. Same recovery.
                    if msg == "CpuNotYetWired" {
                        attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
                        self.record_oom_and_persist(metric, backend, pixels);
                        continue;
                    }
                    attempts.push((backend, AttemptOutcome::OtherError(msg.clone())));
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
            output_columns: BTreeMap::new(),
            metric_version: None,
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

    /// Phase 8a: mark the in-memory capability profile as no-GPU and
    /// persist to disk. Called by [`Self::run_single`] when a GPU
    /// backend attempt surfaced a libcuda-missing error
    /// ([`is_no_cuda_driver`]). After this fires:
    ///
    /// 1. The chooser's `gpu.present == false` fast-path rejects every
    ///    GPU backend with [`RejectReason::NoGpuPresent`].
    /// 2. The OOM ladder continues with Cpu the only feasible
    ///    candidate (assuming the metric has a CPU reference enabled).
    /// 3. The downgrade survives a process crash — the next process
    ///    sees `gpu.present == false` from the cache and skips
    ///    nvidia-smi entirely.
    ///
    /// Note: we deliberately do NOT touch `gpu.model` /
    /// `gpu.driver_version` so the `machine_hash` remains stable.
    /// Only `present` and `total_vram_mib` flip; the hash key inputs
    /// are unchanged. This keeps the cache filename stable across the
    /// downgrade.
    fn downgrade_to_no_gpu_and_persist(&mut self) {
        let cap = self.capability_mut();
        cap.gpu.present = false;
        cap.gpu.total_vram_mib = 0;
        let path = self.cache_path();
        if let Err(e) = save_profile(&path, self.capability()) {
            eprintln!(
                "zenmetrics-orchestrator: failed to persist no-GPU downgrade at {}: {}",
                path.display(),
                e
            );
        }
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
        output_columns: BTreeMap::new(),
        metric_version: None,
    }
}

/// Build the per-task output column map from a successful score +
/// any metric-specific extras returned by [`ExecMetric::compute_phase4_with_extras`].
///
/// **The keys emitted here are the canonical / GPU-variant column
/// names** — i.e. the same shape the legacy `MetricCache` produces
/// when its CLI input is the `*Gpu` variant (or `Cvvdp` / `Iwssim`,
/// which have only one variant). The orchestrator's chooser may pick
/// CPU at runtime via OOM fallback, but the column name carries the
/// CLI-requested-variant intent. CLI consumers using the orchestrator
/// for a CPU-only variant (e.g. `--metric ssim2` not `ssim2-gpu`)
/// re-key these strings before writing to a sweep TSV.
///
/// - butter -> `butteraugli_max_gpu` + extras carry
///   `butteraugli_pnorm3_gpu`.
/// - cvvdp -> versioned `CVVDP_COLUMN_NAME` (e.g.
///   `cvvdp_imazen_v0_0_1`). CVVDP has no CPU/GPU split in the CLI
///   variant — same column name on both paths.
/// - iwssim -> versioned `IWSSIM_COLUMN_NAME`. Iwssim has no CPU
///   reference at all; column name is always the versioned tag.
/// - ssim2 / dssim / zensim -> `<tag>_gpu` (matching legacy GPU
///   MetricCache output).
///
/// `pub(crate)` so the worker pool can call it too; the pool's
/// streaming path needs the same column mapping.
pub(crate) fn build_output_columns(
    metric: MetricKind,
    score: &Score,
    extras: &BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    // Primary column. Matches the legacy `MetricCache::run_metric_cached`
    // output keys exactly so a downstream parquet writer that joins
    // the orchestrator + legacy paths sees byte-identical column
    // names.
    match metric {
        MetricKind::Butter => {
            out.insert("butteraugli_max_gpu".to_string(), score.value);
        }
        MetricKind::Cvvdp => {
            #[cfg(feature = "bench")]
            {
                out.insert(
                    zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME.to_string(),
                    score.value,
                );
            }
            #[cfg(not(feature = "bench"))]
            {
                out.insert("cvvdp".to_string(), score.value);
            }
        }
        MetricKind::Iwssim => {
            #[cfg(feature = "bench")]
            {
                out.insert(
                    zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME.to_string(),
                    score.value,
                );
            }
            #[cfg(not(feature = "bench"))]
            {
                out.insert("iwssim".to_string(), score.value);
            }
        }
        MetricKind::Ssim2 => {
            out.insert("ssim2_gpu".to_string(), score.value);
        }
        MetricKind::Dssim => {
            out.insert("dssim_gpu".to_string(), score.value);
        }
        MetricKind::Zensim => {
            out.insert("zensim_gpu".to_string(), score.value);
        }
    }
    // Merge metric-specific extras. Extras keys take precedence on
    // collision so a future per-metric extra can override the primary
    // if needed.
    for (k, v) in extras {
        out.insert(k.clone(), *v);
    }
    out
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

// ---------------------------------------------------------------------------
// Phase 5: pub(crate) shims for pool.rs
//
// The pool's GPU worker reuses the same per-backend construction matrix
// + the same OOM-classification heuristic as `run_single`. We expose
// them as `pub(crate)` shims rather than making the internal types
// `pub` directly — keeps the public API minimal.
// ---------------------------------------------------------------------------

/// Wrapper exposing [`ExecMetric`] outside this module for the worker
/// pool. Mirrors the variants exactly; the worker doesn't care which
/// underlying type it's calling, only that `compute` and the
/// cached-ref variants work.
pub(crate) enum ConstructOutcomePub {
    Ok(ExecMetric),
    Oom,
    Other(String),
}

/// pool-facing classification for [`compute`] call errors.
pub(crate) enum CallErrPub {
    Oom,
    Other(String),
}

impl ExecMetric {
    /// Try to score `(ref_bytes, dist_bytes)`. The umbrella's
    /// `compute_srgb_u8` is the regular path; cvvdp StripPair routes
    /// through the direct crate. OOM vs other errors are classified
    /// via the shared string heuristic.
    ///
    /// **Phase 7.5**: this single-score shim is kept for backwards
    /// compatibility but no in-tree caller uses it any more — the
    /// worker pool routes through [`Self::compute_with_extras`] so
    /// multi-column metrics (butter `pnorm_3`) survive end-to-end.
    /// Keeping it `pub(crate)` + `#[allow(dead_code)]` lets future
    /// consumers that only need the primary score pick it up
    /// without re-deriving the OOM-classification glue.
    #[allow(dead_code)]
    pub(crate) fn compute(&mut self, r: &[u8], d: &[u8]) -> Result<Score, CallErrPub> {
        match self.compute_phase4(r, d) {
            Ok(s) => Ok(s),
            Err(CallErr::Oom) => Err(CallErrPub::Oom),
            Err(CallErr::Other(msg)) => Err(CallErrPub::Other(msg)),
        }
    }

    /// Phase 7.5 sibling of [`Self::compute`]. Returns the score plus
    /// metric-specific output columns (butter `pnorm_3`, etc.) so the
    /// worker pool can populate `TaskResult.output_columns` for
    /// multi-column metrics without re-running the kernel.
    pub(crate) fn compute_with_extras(
        &mut self,
        r: &[u8],
        d: &[u8],
    ) -> Result<(Score, BTreeMap<String, f64>), CallErrPub> {
        match self.compute_phase4_with_extras(r, d) {
            Ok((s, extras)) => Ok((s, extras)),
            Err(CallErr::Oom) => Err(CallErrPub::Oom),
            Err(CallErr::Other(msg)) => Err(CallErrPub::Other(msg)),
        }
    }

    /// Cached-ref sibling of [`Self::compute_with_extras`]. Threads
    /// metric-specific extras through the cached-reference fast path
    /// (set_reference + compute_with_cached_reference). Butter's
    /// `pnorm_3` aggregate is only produced by `compute_*` calls that
    /// run the full reduction kernel; the umbrella's cached-ref API
    /// re-runs the reduction so pnorm_3 IS produced — we route the
    /// Butter variant through a fresh `compute_srgb_u8_with_pnorm3`
    /// after the reference is installed (skipping the host upload
    /// of the reference but still re-running the dist-side pipeline).
    pub(crate) fn compute_with_cached_reference_with_extras(
        &mut self,
        d: &[u8],
    ) -> Result<(Score, BTreeMap<String, f64>), CallErrPub> {
        // Butter via the umbrella has cached-ref support but the
        // current opaque API drops pnorm_3 from the cached path. For
        // now route butter cached-ref through the regular compute by
        // surfacing extras as empty — the worker pool will see a
        // single-column score and the parquet writer can still emit
        // `butteraugli_max_gpu`. Phase 7.5+ work: add a
        // `compute_with_cached_reference_with_pnorm3` to butter's
        // opaque trait. Until then, the cached-ref path produces the
        // primary column but not the pnorm_3 extra — same shape as
        // every other metric.
        match self.compute_with_cached_reference(d) {
            Ok(s) => Ok((s, BTreeMap::new())),
            Err(e) => Err(e),
        }
    }

    /// True when this backend supports cached-ref dispatch. cvvdp
    /// StripPair uses a one-shot strip walker that doesn't expose
    /// a separate set_reference / compute_with_cached_reference pair —
    /// the pool still calls regular `compute` for it. The umbrella
    /// metrics that DO expose cached-ref (cvvdp Full, butter, ssim2,
    /// dssim, iwssim, zensim) report true here so the worker pool
    /// promotes the dispatch.
    ///
    /// CPU adapters expose `supports_cached_ref` per-metric; the
    /// caller delegates so a cvvdp fallback still benefits from
    /// `warm_reference`, while ssim2 / butter / zensim CPU fall back
    /// to regular compute.
    pub(crate) fn supports_cached_ref(&self) -> bool {
        match self {
            ExecMetric::Umbrella(_) => true,
            ExecMetric::CvvdpStripPair(_) => false,
            ExecMetric::Cpu(adapter) => adapter.supports_cached_ref(),
        }
    }

    /// Install the reference state. Returns `Err(msg)` if the
    /// underlying metric crate's cached-ref API isn't wired or
    /// failed at dispatch.
    pub(crate) fn set_reference(&mut self, r: &[u8]) -> Result<(), String> {
        match self {
            ExecMetric::Umbrella(m) => m
                .set_reference_srgb_u8(r)
                .map_err(|e| e.to_string()),
            ExecMetric::CvvdpStripPair(_) => {
                Err("cvvdp StripPair has no separate set_reference path".into())
            }
            ExecMetric::Cpu(adapter) => adapter.set_reference(r).map_err(|e| e.to_string()),
        }
    }

    /// Score a distorted candidate against the previously-cached
    /// reference. Pre-requisite: [`Self::set_reference`] succeeded.
    pub(crate) fn compute_with_cached_reference(
        &mut self,
        d: &[u8],
    ) -> Result<Score, CallErrPub> {
        match self {
            ExecMetric::Umbrella(m) => m
                .compute_with_cached_reference_srgb_u8(d)
                .map_err(|e| {
                    let msg = e.to_string();
                    match classify_call_err(&msg) {
                        CallErr::Oom => CallErrPub::Oom,
                        CallErr::Other(s) => CallErrPub::Other(s),
                    }
                }),
            ExecMetric::CvvdpStripPair(_) => Err(CallErrPub::Other(
                "cvvdp StripPair has no cached-reference path".into(),
            )),
            ExecMetric::Cpu(adapter) => adapter
                .compute_with_cached_reference(d)
                .map_err(|e| CallErrPub::Other(e.to_string())),
        }
    }
}

/// pool-facing entry to `construct`. Same dispatch as the
/// `run_single` ladder; pool.rs uses this to route the per-task
/// `(metric, backend, w, h, params)` tuple into the right per-crate
/// constructor.
///
/// Phase 7.7.1: this entry is *always* a first-attempt construction
/// from the pool's perspective — the pool maintains warm instances per
/// `(metric, w, h, backend)` signature and does not yet have an
/// in-place OOM-ladder reconstruction. So we pass `after_oom = false`
/// to get the legacy-compatible `MemoryMode::Auto` path. Future work
/// can plumb an explicit `after_oom` here when the pool gets its own
/// ladder.
pub(crate) fn construct_pub(
    kind: MetricKind,
    backend: Backend,
    width: u32,
    height: u32,
    params: Option<MetricParams>,
) -> ConstructOutcomePub {
    match construct(kind, backend, width, height, params, false) {
        ConstructOutcome::Ok(em) => ConstructOutcomePub::Ok(em),
        ConstructOutcome::Oom => ConstructOutcomePub::Oom,
        ConstructOutcome::Other(msg) => ConstructOutcomePub::Other(msg),
    }
}

/// Pool-facing OOM classifier (re-exported for symmetry with
/// `construct_pub`). Currently the pool doesn't call this directly —
/// it relies on `ExecMetric::compute` to surface the classification —
/// but keeping it crate-visible matches the bench's symmetry.
#[allow(dead_code)]
pub(crate) fn classify_call_err_pub(msg: &str) -> CallErrPub {
    match classify_call_err(msg) {
        CallErr::Oom => CallErrPub::Oom,
        CallErr::Other(s) => CallErrPub::Other(s),
    }
}

#[allow(dead_code)]
pub(crate) fn classify_construct_err_pub(e: ApiError) -> ConstructOutcomePub {
    match classify_construct_err(e) {
        ConstructOutcome::Ok(m) => ConstructOutcomePub::Ok(m),
        ConstructOutcome::Oom => ConstructOutcomePub::Oom,
        ConstructOutcome::Other(s) => ConstructOutcomePub::Other(s),
    }
}

#[allow(dead_code)]
pub(crate) fn classify_cvvdp_construct_err_pub(
    e: zenmetrics_api::cvvdp::Error,
) -> ConstructOutcomePub {
    match classify_cvvdp_construct_err(e) {
        ConstructOutcome::Ok(m) => ConstructOutcomePub::Ok(m),
        ConstructOutcome::Oom => ConstructOutcomePub::Oom,
        ConstructOutcome::Other(s) => ConstructOutcomePub::Other(s),
    }
}

// ---------------------------------------------------------------------------
// Phase 7.5 tests — build_output_columns column-name shape
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zenmetrics_api::Score;

    fn mk_score(name: &'static str, version: &'static str, v: f64) -> Score {
        Score {
            value: v,
            metric_name: name,
            metric_version: version,
        }
    }

    #[test]
    fn build_output_columns_ssim2_emits_gpu_suffix() {
        let s = mk_score("ssim2", "0.0.1", 95.5);
        let cols = build_output_columns(MetricKind::Ssim2, &s, &BTreeMap::new());
        assert_eq!(cols.len(), 1);
        assert_eq!(cols.get("ssim2_gpu"), Some(&95.5));
    }

    #[test]
    fn build_output_columns_dssim_emits_gpu_suffix() {
        let s = mk_score("dssim", "0.0.1", 0.05);
        let cols = build_output_columns(MetricKind::Dssim, &s, &BTreeMap::new());
        assert_eq!(cols.get("dssim_gpu"), Some(&0.05));
    }

    #[test]
    fn build_output_columns_zensim_emits_gpu_suffix() {
        let s = mk_score("zensim", "0.0.1", 82.4);
        let cols = build_output_columns(MetricKind::Zensim, &s, &BTreeMap::new());
        assert_eq!(cols.get("zensim_gpu"), Some(&82.4));
    }

    #[test]
    fn build_output_columns_butter_merges_extras() {
        let s = mk_score("butter", "0.0.1", 1.2);
        let mut extras = BTreeMap::new();
        extras.insert("butteraugli_pnorm3_gpu".to_string(), 2.4_f64);
        let cols = build_output_columns(MetricKind::Butter, &s, &extras);
        assert_eq!(cols.len(), 2);
        assert_eq!(cols.get("butteraugli_max_gpu"), Some(&1.2));
        assert_eq!(cols.get("butteraugli_pnorm3_gpu"), Some(&2.4));
    }

    #[test]
    fn build_output_columns_cvvdp_uses_versioned_column() {
        let s = mk_score("cvvdp", "0.0.1", 9.5);
        let cols = build_output_columns(MetricKind::Cvvdp, &s, &BTreeMap::new());
        // The exact key depends on CVVDP_COLUMN_NAME (which depends
        // on the build-time CVVDP_IMPL_TAG env var). Default is
        // `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`.
        assert!(
            cols.keys().any(|k| k.starts_with("cvvdp_")),
            "expected cvvdp_ prefix in {:?}",
            cols.keys().collect::<Vec<_>>()
        );
        // Value lookup is independent of the key prefix.
        let (_, v) = cols.iter().next().unwrap();
        assert_eq!(*v, 9.5);
    }

    #[test]
    fn build_output_columns_iwssim_uses_versioned_column() {
        let s = mk_score("iwssim", "0.0.1", 0.987);
        let cols = build_output_columns(MetricKind::Iwssim, &s, &BTreeMap::new());
        assert!(
            cols.keys().any(|k| k.starts_with("iwssim_")),
            "expected iwssim_ prefix in {:?}",
            cols.keys().collect::<Vec<_>>()
        );
    }
}
