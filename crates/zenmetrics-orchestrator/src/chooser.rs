//! Phase 3 backend chooser — a pure decision function over the
//! capability cache populated in Phase 2.
//!
//! Given a `(metric, width, height)` shape and a snapshot of live
//! free VRAM, [`Orchestrator::choose_backend`] interpolates the
//! cached `ns_per_px` + `vram_mib` measurements to predict the cost
//! of each candidate backend, rejects backends that would OOM with
//! safety margin, and returns the fastest survivor. No execution,
//! no GPU queries, no allocations beyond the diagnostic `considered`
//! Vec. Runs in well under 100 µs once the cache is loaded.
//!
//! Phase 4's executor will call this directly. Phase 3 ships the
//! decision logic only — `Backend::Cpu` is always rejected as
//! `CpuNotYetWired` until Phase 6 supplies a CPU runtime.
//!
//! ## Phase 6 update
//!
//! `Backend::Cpu` is no longer universally rejected. The chooser
//! evaluates CPU as a real candidate when the metric has a CPU
//! reference. Phase 8g (2026-05-27) landed the iwssim CPU port from
//! Python-IW-SSIM, so all six metrics now expose a CPU backend; the
//! historical `CpuMetricUnavailable` rejection is unreachable for
//! ordinary callers. The reason variant is retained for forwards
//! compatibility with future CPU-only-blocking metrics. See
//! `docs/CPU_BACKENDS.md`.
//! CPU candidates report `vram_mib = 0` since they consume RAM, not
//! VRAM.

#![cfg(feature = "bench")]

use std::time::SystemTime;

use zenmetrics_api::MetricKind;

use crate::{Backend, MetricProfile, Orchestrator};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Static tuning knobs for the chooser. All fields keep public
/// visibility + a `Default` impl so callers can override one knob via
/// struct-update syntax without going through a builder.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ChooserConfig {
    /// Fraction of free VRAM that must remain unused as a safety
    /// margin (driver overhead, browser tabs, compositor, etc.). A
    /// candidate is rejected when `predicted_vram_mib >
    /// vram_free_mib * (1.0 - vram_safety_margin)`. Default `0.15`
    /// (so up to 85 % of free VRAM is usable).
    pub vram_safety_margin: f32,

    /// Multiplier applied to predicted `ns_per_px` for sizes
    /// strictly above the highest measured point. Compensates for
    /// the fact that extrapolation in log-pixel space is optimistic
    /// (real per-pixel cost trends up with size due to LLC pressure,
    /// memory-bandwidth contention, etc.). Default `1.20`.
    pub extrapolation_pessimism: f32,

    /// When two surviving candidates predict the same `ns_per_px`
    /// (rare in practice but easy to construct in tests), break the
    /// tie by preferring backends earlier in this array. Default:
    /// `[GpuFull, GpuStrip, GpuStripPair, Cpu]`.
    pub tie_break_order: [Backend; 4],
}

impl Default for ChooserConfig {
    fn default() -> Self {
        Self {
            vram_safety_margin: 0.15,
            extrapolation_pessimism: 1.20,
            tie_break_order: [
                Backend::GpuFull,
                Backend::GpuStrip,
                Backend::GpuStripPair,
                Backend::Cpu,
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Public output types
// ---------------------------------------------------------------------------

/// A backend the chooser evaluated, plus its status. Always populated
/// for every candidate (even when rejected) so operators can answer
/// "why did it pick X?" with one method call.
#[derive(Debug, Clone, PartialEq)]
pub struct ConsideredCandidate {
    pub backend: Backend,
    pub status: CandidateStatus,
}

/// Result of evaluating one backend candidate.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    /// This backend was a feasible choice. Two surviving backends
    /// both carry `Selected`; the chooser picks one of them as
    /// [`BackendChoice::backend`] using `tie_break_order`.
    Selected {
        ns_per_px: f64,
        vram_mib: usize,
    },
    /// This backend was rejected. Predicted numbers are recorded
    /// when meaningful so the operator can compare against survivors.
    Rejected {
        reason: RejectReason,
        predicted_ns_per_px: Option<f64>,
        predicted_vram_mib: Option<usize>,
    },
}

/// Why a backend was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// The metric never supports this backend (e.g., GpuStripPair on
    /// non-cvvdp metrics).
    UnsupportedByMetric,
    /// Predicted VRAM exceeds `vram_free * (1 - safety_margin)`.
    PredictedOomWithMargin,
    /// `(backend, size_pixels)` is present in
    /// [`MetricProfile::cells_failed_oom`]. The match is on the
    /// nearest measured size — see the design doc.
    KnownOomCell,
    /// Phase 3 never selects [`Backend::Cpu`]; Phase 6 wires this.
    ///
    /// **Phase 6**: this variant is no longer produced by the chooser
    /// — CPU is a real candidate. Kept in the enum for backwards
    /// compatibility (Phase 5 callers that match on it still compile).
    CpuNotYetWired,
    /// The selected metric has no CPU reference implementation in this
    /// release (Iwssim). Phase 6.
    CpuMetricUnavailable,
    /// Cache has no measurement for this backend at any size — e.g.
    /// Phase 2 OOMed at its smallest measured size.
    NoMeasuredData,
    /// Phase 7.7.1: log-linear extrapolation produced a non-positive
    /// `predicted_ns_per_px` for this backend at this size. Happens
    /// when only 2 measured points exist for a monotonically
    /// decreasing per-pixel cost (faster per-px at larger sizes) and
    /// the requested size is far above the largest measured point —
    /// the line goes below zero. A negative prediction would falsely
    /// rank this backend as the fastest candidate; reject instead so
    /// the chooser falls back to a backend with a real prediction.
    /// Operators see this when their bench cache is sparse for a
    /// given backend; re-running `bench` at the requested size cures
    /// it.
    NonPositivePrediction,
    /// Phase 8a: the cached `CapabilityProfile.gpu.present == false`
    /// so no GPU backend is feasible. Every `Backend::Gpu*` candidate
    /// is rejected with this reason on hosts with no NVIDIA driver,
    /// or when `ZENMETRICS_FORCE_NO_GPU=1` is set, or after the
    /// executor caught a runtime libcuda-missing error and downgraded
    /// the profile in place. Only [`Backend::Cpu`] candidates can
    /// survive when this fires.
    NoGpuPresent,
}

/// Final decision returned by [`Orchestrator::choose_backend`].
#[derive(Debug, Clone, PartialEq)]
pub struct BackendChoice {
    /// The chosen backend.
    pub backend: Backend,
    /// Interpolated `ns_per_px` at the requested size.
    pub predicted_ns_per_px: f64,
    /// Interpolated VRAM peak at the requested size.
    pub predicted_vram_mib: usize,
    /// Safety-margin headroom (in MiB) over the prediction —
    /// `vram_free_mib * (1 - safety_margin) - predicted_vram_mib`.
    /// Always non-negative for surviving candidates.
    pub safety_margin_mib: usize,
    /// Every backend the chooser evaluated, in evaluation order
    /// (GpuFull, GpuStrip, GpuStripPair, Cpu). Includes Rejected.
    pub considered: Vec<ConsideredCandidate>,
}

/// Failure modes of [`Orchestrator::choose_backend`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ChooserError {
    /// No backend survived rejection. The `considered` list captures
    /// why each candidate failed.
    NoFeasibleBackend {
        considered: Vec<ConsideredCandidate>,
    },
    /// The requested `MetricKind` has no entry in
    /// [`crate::CapabilityProfile::metrics`]. Caller can call
    /// [`Orchestrator::bench`] or [`Orchestrator::warm`] to populate.
    UnknownMetric(MetricKind),
}

impl std::fmt::Display for ChooserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChooserError::NoFeasibleBackend { considered } => {
                write!(
                    f,
                    "no feasible backend (considered {} candidates)",
                    considered.len()
                )
            }
            ChooserError::UnknownMetric(k) => {
                write!(f, "no measurements for metric '{}'", k.tag())
            }
        }
    }
}

impl std::error::Error for ChooserError {}

// ---------------------------------------------------------------------------
// Task shape (Phase 3-only — Phase 4+ extends this with ref/dist data)
// ---------------------------------------------------------------------------

/// Minimal task descriptor for the chooser. Phase 4 expands this with
/// reference + distorted bytes; Phase 3 only needs the
/// `(metric, dims)` shape because the chooser is a pure function over
/// the capability cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskShape {
    pub metric: MetricKind,
    pub width: u32,
    pub height: u32,
}

// ---------------------------------------------------------------------------
// Backend-support matrix per MetricKind
// ---------------------------------------------------------------------------

/// All backends the chooser will evaluate, in stable order. Rejected
/// candidates still appear in `considered` so the operator sees the
/// full decision surface.
const ALL_BACKENDS: [Backend; 4] = [
    Backend::GpuFull,
    Backend::GpuStrip,
    Backend::GpuStripPair,
    Backend::Cpu,
];

/// Which backends a given metric supports. Matches the
/// `backends_for_kind` table in `bench.rs` plus `Cpu` (Phase 6 wires
/// per-metric CPU references — Phase 8g landed the Iwssim port so
/// every metric in this release surfaces a CPU backend).
///
/// `pub(crate)` so `executor::record_oom_and_persist` (Phase 8i Fix B)
/// can prune fossilized OOM entries whose backend is no longer in this
/// list (e.g. legacy `(gpu_strip, *)` entries for cvvdp written by a
/// pre-orchestrator binary).
pub(crate) fn supported_backends(metric: MetricKind) -> &'static [Backend] {
    match metric {
        // cvvdp uniquely supports StripPair via its single-pass
        // one-shot pipeline. CPU reference: cvvdp (in-tree).
        MetricKind::Cvvdp => &[Backend::GpuFull, Backend::GpuStripPair, Backend::Cpu],
        // zensim has a fused full-image kernel only. CPU reference: zensim.
        MetricKind::Zensim => &[Backend::GpuFull, Backend::Cpu],
        // Butter / Ssim2 / Dssim / Iwssim each have a CPU reference crate.
        MetricKind::Butter
        | MetricKind::Ssim2
        | MetricKind::Dssim
        | MetricKind::Iwssim => {
            &[Backend::GpuFull, Backend::GpuStrip, Backend::Cpu]
        }
    }
}

/// Whether the build was compiled with the `cpu-<metric>` feature
/// required to actually run this metric on CPU. The chooser uses this
/// to surface `CpuMetricUnavailable` for metrics whose CPU feature is
/// disabled at compile time, rather than picking CPU and having the
/// executor crash at construct time.
fn cpu_feature_enabled_for(metric: MetricKind) -> bool {
    match metric {
        MetricKind::Cvvdp => cfg!(feature = "cpu-cvvdp"),
        MetricKind::Ssim2 => cfg!(feature = "cpu-ssim2"),
        MetricKind::Dssim => cfg!(feature = "cpu-dssim"),
        MetricKind::Butter => cfg!(feature = "cpu-butter"),
        MetricKind::Zensim => cfg!(feature = "cpu-zensim"),
        MetricKind::Iwssim => cfg!(feature = "cpu-iwssim"),
    }
}

// ---------------------------------------------------------------------------
// Log-pixel interpolation
// ---------------------------------------------------------------------------

/// Interpolate a per-backend `f64` measurement from the sparse cache
/// at the requested pixel count. Returns `None` when the cache has
/// no entry for this backend at any measured size.
///
/// Semantics:
///
/// - Exact match on a measured size: use directly.
/// - Between two measured sizes: log-pixel linear interpolation.
/// - Below the smallest measured size: clamp to that smallest value
///   (fixed-cost overhead dominates at tiny sizes — don't shrink
///   optimistically).
/// - Above the largest measured size: log-pixel linear extrapolation
///   from the top two measured points, then multiply by
///   `extrapolation_pessimism`.
fn interpolate_ns_per_px(
    profile: &MetricProfile,
    backend: Backend,
    pixels: u64,
    extrapolation_pessimism: f32,
) -> Option<f64> {
    // Collect (size, value) pairs for this backend, in ascending size order.
    // The bench cache stores BTreeMap<u64, BackendBench>, so a single
    // pass over `ns_per_px_at` yields ascending sizes for free.
    let mut points: Vec<(u64, f64)> = Vec::new();
    for (&size_px, bench) in &profile.ns_per_px_at {
        if let Some(v) = bench.get(backend) {
            points.push((size_px, v));
        }
    }
    interpolate_from_points(&points, pixels, extrapolation_pessimism)
}

/// VRAM analogue of [`interpolate_ns_per_px`]. Stored as `usize`; the
/// interpolation is done in `f64` then rounded.
fn interpolate_vram_mib(
    profile: &MetricProfile,
    backend: Backend,
    pixels: u64,
    extrapolation_pessimism: f32,
) -> Option<usize> {
    let mut points: Vec<(u64, f64)> = Vec::new();
    for (&size_px, vram) in &profile.vram_mib_at {
        if let Some(v) = vram.get(backend) {
            points.push((size_px, v as f64));
        }
    }
    interpolate_from_points(&points, pixels, extrapolation_pessimism).map(|v| {
        // Round half-up; bench cells already integer-MiB so this is
        // mostly cosmetic for interpolated points.
        if v < 0.0 {
            0
        } else {
            (v + 0.5) as usize
        }
    })
}

/// Shared core for the two interpolators above.
fn interpolate_from_points(
    points: &[(u64, f64)],
    pixels: u64,
    extrapolation_pessimism: f32,
) -> Option<f64> {
    if points.is_empty() {
        return None;
    }
    if points.len() == 1 {
        // Only one measured point — return it directly. Extrapolating
        // from a single point would be a guess; better to surface the
        // exact measurement and let the operator notice the sparse
        // cache.
        return Some(points[0].1);
    }
    // Exact match short-circuit.
    if let Some((_, v)) = points.iter().find(|(p, _)| *p == pixels) {
        return Some(*v);
    }
    let smallest = points[0];
    let largest = *points.last().unwrap();

    if pixels < smallest.0 {
        // Below the cache — clamp to smallest. Fixed overhead is a
        // big part of tiny-image cost; optimistic shrink would
        // under-budget.
        return Some(smallest.1);
    }
    if pixels > largest.0 {
        // Above the cache — extrapolate from the top two points in
        // log-pixel space, then apply the pessimism multiplier.
        let (p_lo, v_lo) = points[points.len() - 2];
        let (p_hi, v_hi) = largest;
        let v = log_pixel_interpolate(p_lo, v_lo, p_hi, v_hi, pixels);
        return Some(v * extrapolation_pessimism as f64);
    }
    // Between two measured points — find the bracket and interpolate.
    let mut lo = points[0];
    let mut hi = points[1];
    for w in points.windows(2) {
        if pixels >= w[0].0 && pixels <= w[1].0 {
            lo = w[0];
            hi = w[1];
            break;
        }
    }
    Some(log_pixel_interpolate(lo.0, lo.1, hi.0, hi.1, pixels))
}

/// Linear interpolation in `log2(pixels)` space. Both `p_lo` and
/// `p_hi` are assumed positive (the cache keys are `width * height`
/// — never zero for a real bench).
fn log_pixel_interpolate(p_lo: u64, v_lo: f64, p_hi: u64, v_hi: f64, pixels: u64) -> f64 {
    if p_lo == p_hi {
        return v_lo;
    }
    let log_lo = (p_lo as f64).log2();
    let log_hi = (p_hi as f64).log2();
    let log_p = (pixels as f64).log2();
    let t = (log_p - log_lo) / (log_hi - log_lo);
    v_lo * (1.0 - t) + v_hi * t
}

// ---------------------------------------------------------------------------
// OOM-cell match helper
// ---------------------------------------------------------------------------

/// True when `(backend, pixels)` is in the OOM list at the nearest
/// measured size. We snap `pixels` to the closest size that appears
/// in the OOM list for this backend; if the request is within ±1 step
/// of an OOMed cell we treat it as a hard hit. "Step" here is one
/// element in `ns_per_px_at.keys()`.
///
/// Rationale: the chooser doesn't know which size was actually OOMed
/// at bench time (the cache stores `size_pixels` exactly), so we
/// match on the nearest measured size in the OOM log. A request at
/// 3000² (9 MP) where the OOM log says `(GpuFull, 4096²)` should
/// still flag — the user is interpolating into the OOM zone.
fn known_oom_cell(profile: &MetricProfile, backend: Backend, pixels: u64) -> bool {
    let measured_sizes: Vec<u64> = profile.ns_per_px_at.keys().copied().collect();
    if measured_sizes.is_empty() {
        // No measurements at all — check exact-pixel match against
        // the OOM list as a fallback.
        return profile
            .cells_failed_oom
            .iter()
            .any(|(b, px)| *b == backend && *px == pixels);
    }
    // Snap the request to the nearest measured size.
    let snapped = nearest(&measured_sizes, pixels);
    // Also, any OOMed cell at a size ≤ the request implies the
    // request will also OOM (more pixels → more memory).
    for (b, px) in &profile.cells_failed_oom {
        if *b != backend {
            continue;
        }
        // Exact match against requested size.
        if *px == pixels {
            return true;
        }
        // Snapped-size match (the chooser is asked about a size that
        // would interpolate at the OOMed cell).
        if *px == snapped {
            return true;
        }
        // OOMed cell is smaller than the request — bigger is worse,
        // EXCEPT when the cache contains a *positive* measurement at
        // a size >= the OOMed size for this backend. A successful
        // later measurement contradicts the cascade hypothesis: the
        // OOM is stale (transient pressure during bench, fossilized
        // from a prior binary version, or recorded after positive
        // bench data via record_oom_and_persist). The chooser should
        // believe the measurement, not the cascade.
        //
        // Phase 8i (2026-05-27): see
        // `docs/CVVDP_CHOOSER_REGRESSION_INVESTIGATION.md` for the
        // failure mode this defeats — a single fossilized 256² OOM
        // for cvvdp/GpuFull was locking out every cvvdp request at
        // any size >= 256² for the cache file's lifetime even though
        // the bench had recorded positive measurements at 1024² and
        // 4096² for that same backend.
        if *px < pixels {
            let has_positive_measurement = profile
                .ns_per_px_at
                .iter()
                .any(|(size, bench)| *size >= *px && bench.get(backend).is_some());
            if has_positive_measurement {
                // Stale OOM — positive measurement at size >= *px
                // proves the cascade hypothesis wrong. Ignore.
                continue;
            }
            return true;
        }
    }
    false
}

fn nearest(sorted_sizes: &[u64], pixels: u64) -> u64 {
    // sorted_sizes is ascending (BTreeMap iteration order). Linear
    // scan is fine — typical cache has ≤ 5 sizes per metric.
    let mut best = sorted_sizes[0];
    let mut best_d = abs_diff(best, pixels);
    for &p in &sorted_sizes[1..] {
        let d = abs_diff(p, pixels);
        if d < best_d {
            best = p;
            best_d = d;
        }
    }
    best
}

fn abs_diff(a: u64, b: u64) -> u64 {
    a.abs_diff(b)
}

// ---------------------------------------------------------------------------
// The chooser itself
// ---------------------------------------------------------------------------

impl Orchestrator {
    /// Choose the best backend for a given `(metric, dims)` using the
    /// cached `MetricProfile` and the supplied `vram_free_mib`
    /// snapshot.
    ///
    /// **Pure** — does not query the GPU or system state. Test suites
    /// pass a synthetic `vram_free_mib`; production callers should
    /// pass the result of [`Self::choose_backend_for_task`] instead,
    /// which threads a live probe through.
    ///
    /// Returns `Err(ChooserError::UnknownMetric)` if the metric has
    /// no cached profile (call `bench()` / `warm()` first), or
    /// `Err(ChooserError::NoFeasibleBackend)` if every candidate was
    /// rejected (e.g., requested size exceeds every backend's VRAM
    /// budget).
    pub fn choose_backend(
        &self,
        metric: MetricKind,
        width: u32,
        height: u32,
        vram_free_mib: usize,
    ) -> Result<BackendChoice, ChooserError> {
        self.choose_backend_with_config(
            metric,
            width,
            height,
            vram_free_mib,
            &ChooserConfig::default(),
        )
    }

    /// Same as [`Self::choose_backend`] with a caller-provided
    /// [`ChooserConfig`]. Useful for tests + datacenter callers who
    /// want a larger VRAM safety floor or a different tie-break order.
    pub fn choose_backend_with_config(
        &self,
        metric: MetricKind,
        width: u32,
        height: u32,
        vram_free_mib: usize,
        config: &ChooserConfig,
    ) -> Result<BackendChoice, ChooserError> {
        let profile = self
            .capability()
            .metrics
            .get(metric.tag())
            .ok_or(ChooserError::UnknownMetric(metric))?;

        let pixels = (width as u64) * (height as u64);
        let usable_vram_mib =
            ((vram_free_mib as f64) * (1.0 - config.vram_safety_margin as f64)).floor() as usize;

        let supported = supported_backends(metric);
        let cpu_feature_on = cpu_feature_enabled_for(metric);
        let mut considered: Vec<ConsideredCandidate> = Vec::with_capacity(ALL_BACKENDS.len());

        // Phase 8a fast-path: when the capability profile says no GPU
        // is present, every GPU backend is rejected up-front with
        // `NoGpuPresent` and only the Cpu candidate goes through the
        // regular evaluator. This skips the cache lookups for GPU
        // cells (which are absent anyway because Phase 2's bench
        // short-circuits) and surfaces a clearer "no GPU" reason than
        // `NoMeasuredData` to operators reading the `considered` list.
        let gpu_absent = !self.capability().gpu.present;

        for &backend in &ALL_BACKENDS {
            let status = if gpu_absent
                && matches!(
                    backend,
                    Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair
                )
            {
                CandidateStatus::Rejected {
                    reason: RejectReason::NoGpuPresent,
                    predicted_ns_per_px: None,
                    predicted_vram_mib: None,
                }
            } else {
                evaluate_candidate(
                    profile,
                    metric,
                    backend,
                    pixels,
                    supported,
                    cpu_feature_on,
                    usable_vram_mib,
                    config,
                )
            };
            considered.push(ConsideredCandidate { backend, status });
        }

        // Pick lowest ns_per_px among Selected candidates.
        let mut best_idx: Option<usize> = None;
        let mut best_ns: f64 = f64::INFINITY;
        for (i, c) in considered.iter().enumerate() {
            if let CandidateStatus::Selected { ns_per_px, .. } = c.status {
                if ns_per_px < best_ns - f64::EPSILON {
                    best_ns = ns_per_px;
                    best_idx = Some(i);
                } else if (ns_per_px - best_ns).abs() <= f64::EPSILON {
                    // Tie — break using the configured priority list.
                    // The earlier element of `tie_break_order` wins.
                    if let Some(prev) = best_idx {
                        let prev_rank = tie_rank(&config.tie_break_order, considered[prev].backend);
                        let cur_rank = tie_rank(&config.tie_break_order, c.backend);
                        if cur_rank < prev_rank {
                            best_idx = Some(i);
                        }
                    } else {
                        best_idx = Some(i);
                    }
                }
            }
        }

        let chosen_idx = best_idx.ok_or(ChooserError::NoFeasibleBackend {
            considered: considered.clone(),
        })?;
        let chosen = &considered[chosen_idx];
        let (ns, mib) = match chosen.status {
            CandidateStatus::Selected { ns_per_px, vram_mib } => (ns_per_px, vram_mib),
            // Unreachable — `best_idx` only points at Selected.
            _ => unreachable!(),
        };
        let safety_margin_mib = usable_vram_mib.saturating_sub(mib);

        Ok(BackendChoice {
            backend: chosen.backend,
            predicted_ns_per_px: ns,
            predicted_vram_mib: mib,
            safety_margin_mib,
            considered,
        })
    }

    /// Convenience: probe live free VRAM (via the cvvdp-gpu nvidia-smi
    /// helper) and call [`Self::choose_backend`].
    ///
    /// If the live probe fails (no GPU / nvidia-smi missing), falls
    /// back to the cached `total_vram_mib` as a best-effort upper
    /// bound. Callers that need stricter behavior should call
    /// `choose_backend` directly with their own probe.
    pub fn choose_backend_for_task(
        &self,
        task: &TaskShape,
    ) -> Result<BackendChoice, ChooserError> {
        let vram_free_mib = probe_free_vram_mib()
            .unwrap_or(self.capability().gpu.total_vram_mib);
        self.choose_backend(task.metric, task.width, task.height, vram_free_mib)
    }
}

/// Per-candidate evaluation: looks at support, OOM history, cache
/// availability, and VRAM budget. Always populates the diagnostic
/// fields where they're meaningful (e.g., a `PredictedOomWithMargin`
/// rejection still carries the predicted ns/px so an operator can see
/// "GpuFull would have been 2.7 ns/px but didn't fit").
#[allow(clippy::too_many_arguments)]
fn evaluate_candidate(
    profile: &MetricProfile,
    metric: MetricKind,
    backend: Backend,
    pixels: u64,
    supported: &[Backend],
    cpu_feature_on: bool,
    usable_vram_mib: usize,
    config: &ChooserConfig,
) -> CandidateStatus {
    // Phase 6: CPU is a real candidate. Per-metric availability is
    // resolved via `supported` (set by the per-metric matrix) and the
    // build-time `cpu-<metric>` feature flag.
    if !supported.contains(&backend) {
        // CPU specifically: when the metric supports CPU upstream but
        // the build is missing the feature flag, surface
        // `CpuMetricUnavailable` for operator clarity. Otherwise the
        // generic "unsupported by metric" rejection.
        if backend == Backend::Cpu {
            return CandidateStatus::Rejected {
                reason: RejectReason::CpuMetricUnavailable,
                predicted_ns_per_px: None,
                predicted_vram_mib: None,
            };
        }
        return CandidateStatus::Rejected {
            reason: RejectReason::UnsupportedByMetric,
            predicted_ns_per_px: None,
            predicted_vram_mib: None,
        };
    }
    if backend == Backend::Cpu {
        // The metric supports CPU upstream but the build may not
        // include the `cpu-<metric>` feature. Reject early so the
        // executor doesn't have to handle the construction failure.
        let _ = metric;
        if !cpu_feature_on {
            return CandidateStatus::Rejected {
                reason: RejectReason::CpuMetricUnavailable,
                predicted_ns_per_px: None,
                predicted_vram_mib: None,
            };
        }
        // Test-mode override + production safety: even with the
        // feature on, if a previous run learned that the CPU adapter
        // at this size was bad (e.g. allocation failure surfaced as
        // OOM), the OOM-cell log lets us skip it. Same rule as the
        // GPU candidates so the OOM ladder works symmetrically.
        if known_oom_cell(profile, backend, pixels) {
            return CandidateStatus::Rejected {
                reason: RejectReason::KnownOomCell,
                predicted_ns_per_px: Some(0.0),
                predicted_vram_mib: Some(0),
            };
        }
        // CPU doesn't allocate VRAM; the per-cell `vram_mib` is always
        // 0. ns/px comes from the bench cache (Phase 2 CPU cells); if
        // the cache lacks CPU data, fall back to a conservative
        // estimate so the OOM ladder can still route here when GPU has
        // failed.
        let ns = interpolate_ns_per_px(profile, backend, pixels, config.extrapolation_pessimism)
            .unwrap_or_else(|| {
                // Heuristic fallback: 200 ns/px is roughly cvvdp
                // single-thread at 1024² on a 7950X. Conservative
                // (some CPU metrics are faster, butteraugli is slower)
                // but adequate as a last-resort ranking signal.
                200.0
            });
        // Phase 7.7.1: reject non-positive predictions. Log-linear
        // extrapolation from 2 monotonically decreasing CPU points
        // produces negative ns/px values at sizes well above the
        // cached range; ranking by `min(ns_per_px)` then picks the
        // negative value as "fastest," which is structurally wrong
        // (negative time cannot be faster than positive time). The
        // operator should re-bench at the requested size; until they
        // do, reject this candidate so a GPU backend with a real
        // measurement wins instead.
        if ns <= 0.0 {
            return CandidateStatus::Rejected {
                reason: RejectReason::NonPositivePrediction,
                predicted_ns_per_px: Some(ns),
                predicted_vram_mib: Some(0),
            };
        }
        return CandidateStatus::Selected {
            ns_per_px: ns,
            vram_mib: 0,
        };
    }
    let ns =
        match interpolate_ns_per_px(profile, backend, pixels, config.extrapolation_pessimism) {
            Some(v) => v,
            None => {
                return CandidateStatus::Rejected {
                    reason: RejectReason::NoMeasuredData,
                    predicted_ns_per_px: None,
                    predicted_vram_mib: None,
                };
            }
        };
    // Phase 7.7.1: see same comment on the CPU branch above. Symmetric
    // rejection here so a GPU backend with a runaway negative
    // extrapolation can't out-rank a backend with a real measurement.
    if ns <= 0.0 {
        return CandidateStatus::Rejected {
            reason: RejectReason::NonPositivePrediction,
            predicted_ns_per_px: Some(ns),
            predicted_vram_mib: None,
        };
    }
    let mib = match interpolate_vram_mib(profile, backend, pixels, config.extrapolation_pessimism) {
        Some(v) => v,
        None => {
            return CandidateStatus::Rejected {
                reason: RejectReason::NoMeasuredData,
                predicted_ns_per_px: Some(ns),
                predicted_vram_mib: None,
            };
        }
    };
    if known_oom_cell(profile, backend, pixels) {
        return CandidateStatus::Rejected {
            reason: RejectReason::KnownOomCell,
            predicted_ns_per_px: Some(ns),
            predicted_vram_mib: Some(mib),
        };
    }
    if mib > usable_vram_mib {
        return CandidateStatus::Rejected {
            reason: RejectReason::PredictedOomWithMargin,
            predicted_ns_per_px: Some(ns),
            predicted_vram_mib: Some(mib),
        };
    }
    CandidateStatus::Selected {
        ns_per_px: ns,
        vram_mib: mib,
    }
}

fn tie_rank(order: &[Backend; 4], b: Backend) -> usize {
    for (i, &x) in order.iter().enumerate() {
        if x == b {
            return i;
        }
    }
    // Unreachable when `order` is a permutation of all four variants
    // (the only sane configuration). Default-low rank otherwise.
    usize::MAX
}

/// Best-effort live VRAM probe. Tries cvvdp-gpu's
/// `live_vram_probe_bytes()` first (it caches one nvidia-smi call
/// per process), falling back to a direct nvidia-smi spawn.
///
/// Returns `None` if neither path produces a number (CI without a
/// GPU, snap-docker, WSL without nvidia-smi, etc.).
fn probe_free_vram_mib() -> Option<usize> {
    if let Some(bytes) = cvvdp_gpu::memory_mode::live_vram_probe_bytes() {
        return Some(bytes / (1024 * 1024));
    }
    None
}

// Suppress unused warnings until Phase 4 wires this up. The
// SystemTime use is reserved for a "stale measurement" rejection
// reason that Phase 4 will add.
#[allow(dead_code)]
fn _suppress_unused() -> Option<SystemTime> {
    None
}
