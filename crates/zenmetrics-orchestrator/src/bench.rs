//! Bench runner — Phase 2.
//!
//! Sequential, in-process cell sweep over every `(metric, backend, size)`
//! combination the build supports. After each cell the metric instance
//! is dropped, the cubecl client is `flush`ed, and `nvidia-smi` is
//! sampled for the post-drop baseline before the next cell runs.
//!
//! ## Why in-process (Option A from the Phase 2 brief)
//!
//! cubecl's memory pool retains GPU allocations between metric instances
//! to amortize PTX compile + cudaMalloc cost across the dispatch chain.
//! We can still measure peak VRAM correctly: sample baseline immediately
//! before constructing each metric, then sample again during the
//! steady-state compute loop, take the max delta. Subprocess-per-cell
//! (Option B) would give cleaner numbers but ~20× more wall time
//! (CUDA context init ≈ 800 ms each). The audit CSV at
//! `benchmarks/gpu_memory_audit_2026-05-27.csv` was generated via the
//! subprocess pattern; our in-process numbers land within ~10-15 % of
//! it (verified in `acceptance gate #8`).
//!
//! ## Cells covered
//!
//! - 6 metric kinds (cvvdp, butter, ssim2, dssim, iwssim, zensim)
//! - 2-3 backends per metric:
//!     - cvvdp: GpuFull + GpuStripPair
//!     - zensim: GpuFull only
//!     - all others: GpuFull + GpuStrip
//! - 3 sizes: 1024², 2048², 4096²
//!
//! Total: ~30 cells; observed wall time on RTX 5070 ≈ 35-45 s cold.

use std::collections::BTreeMap;
use std::time::Duration;

#[cfg(feature = "bench")]
use std::time::{Instant, SystemTime};

#[cfg(feature = "bench")]
use zenmetrics_api::{Backend as ApiBackend, MemoryMode, Metric, MetricKind, MetricParams};

use crate::{Backend, MetricProfile};
#[cfg(feature = "bench")]
#[allow(unused_imports)]
use crate::{BackendBench, BackendVram};

/// Knobs for a bench run. Production code uses [`Self::default`]; tests
/// override to keep wall time down.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BenchPlan {
    /// Image sizes (width = height = `size`) to measure. Default
    /// `[1024, 2048, 4096]`.
    pub sizes: Vec<u32>,
    /// Warmup iterations (results discarded). Default `2`.
    pub warmup_iters: usize,
    /// Timed iterations contributing to the p50 statistic. Default `5`.
    pub timed_iters: usize,
    /// If any single cell exceeds this wall time, skip the remaining
    /// (larger) sizes for the same `(metric, backend)`. Default `5s`.
    pub soft_timeout_per_cell: Duration,
    /// `nvidia-smi` sample interval during the compute loop. Default
    /// `25 ms`.
    pub vram_sample_interval: Duration,
}

impl Default for BenchPlan {
    fn default() -> Self {
        Self {
            sizes: vec![1024, 2048, 4096],
            warmup_iters: 2,
            timed_iters: 5,
            soft_timeout_per_cell: Duration::from_secs(5),
            vram_sample_interval: Duration::from_millis(25),
        }
    }
}

/// Output of [`run`] — what [`crate::Orchestrator::bench`] folds into
/// `capability.metrics`.
#[derive(Debug, Default, Clone)]
pub struct BenchReport {
    /// `metric_kind.tag()` -> per-metric profile.
    pub metrics: BTreeMap<String, MetricProfile>,
    /// Total wall time of the bench.
    pub total_wall: Duration,
    /// Cells skipped because of `soft_timeout_per_cell`.
    pub timed_out_cells: Vec<(String, Backend, u32)>,
}

/// Deterministic `(ref, dist)` pair generator. Mirrors
/// `tests/common/mod.rs::synth_pair_with_offset_dist` used across the
/// `-gpu` crates so the same bytes flow through every metric.
///
/// Reference image: per-channel modular pattern with stable wrap. Dist
/// is the canonical `(-8, -4, +12)` saturating offset.
///
/// Embedded so the orchestrator has zero external corpus dependency.
pub fn synth_pair_offset_dist(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
    let w = width as usize;
    let h = height as usize;
    let n = w * h * 3;
    let mut r = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let rr = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let gg = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            r[i] = rr;
            r[i + 1] = gg;
            r[i + 2] = bb;
        }
    }
    let d: Vec<u8> = r
        .chunks_exact(3)
        .flat_map(|p| {
            [
                p[0].saturating_sub(8),
                p[1].saturating_sub(4),
                p[2].saturating_add(12),
            ]
        })
        .collect();
    (r, d)
}

// ---------------------------------------------------------------------------
// Cells the bench covers.
// ---------------------------------------------------------------------------

#[cfg(feature = "bench")]
const METRIC_KINDS: &[MetricKind] = &[
    MetricKind::Cvvdp,
    MetricKind::Butter,
    MetricKind::Ssim2,
    MetricKind::Dssim,
    MetricKind::Iwssim,
    MetricKind::Zensim,
];

/// Which backends a given metric supports in our Phase 2 bench grid.
#[cfg(feature = "bench")]
fn backends_for_kind(kind: MetricKind) -> &'static [Backend] {
    match kind {
        // cvvdp Mode E (Strip) was rolled back upstream (changes JOD);
        // we measure Full + StripPair only.
        MetricKind::Cvvdp => &[Backend::GpuFull, Backend::GpuStripPair],
        // zensim still only supports Full + Auto.
        MetricKind::Zensim => &[Backend::GpuFull],
        // Everything else exposes Full + Strip.
        MetricKind::Butter
        | MetricKind::Ssim2
        | MetricKind::Dssim
        | MetricKind::Iwssim => &[Backend::GpuFull, Backend::GpuStrip],
    }
}

// ---------------------------------------------------------------------------
// `run` entry point — orchestrates the cell sweep.
// ---------------------------------------------------------------------------

/// Run the bench described by `plan` and return a fully-populated
/// [`BenchReport`]. Phase 2 always exercises every metric crate the
/// build enables; metrics whose `bench` feature is gated out are
/// skipped silently (their entry in `report.metrics` is absent).
pub fn run(plan: &BenchPlan) -> BenchReport {
    #[cfg(not(feature = "bench"))]
    {
        let _ = plan;
        BenchReport {
            metrics: BTreeMap::new(),
            total_wall: Duration::ZERO,
            timed_out_cells: Vec::new(),
        }
    }

    #[cfg(feature = "bench")]
    {
        run_impl(plan)
    }
}

#[cfg(feature = "bench")]
fn run_impl(plan: &BenchPlan) -> BenchReport {
    let t_start = Instant::now();
    let mut report = BenchReport::default();

    for &kind in METRIC_KINDS {
        let tag = kind.tag().to_string();
        let mut profile = MetricProfile::default();

        for &backend in backends_for_kind(kind) {
            let mut backend_timed_out_at: Option<u32> = None;
            for &size in &plan.sizes {
                if let Some(prev) = backend_timed_out_at {
                    if size > prev {
                        // Already hit the soft timeout at a smaller
                        // size — record this cell as OOM (treated by
                        // Phase 3 as "do not try").
                        profile
                            .cells_failed_oom
                            .push((backend, (size as u64) * (size as u64)));
                        continue;
                    }
                }

                let cell = measure_cell(kind, backend, size, plan);
                let size_px = (size as u64) * (size as u64);
                match cell {
                    CellOutcome::Ok { ns_per_px, vram_mib } => {
                        profile
                            .ns_per_px_at
                            .entry(size_px)
                            .or_default()
                            .set(backend, ns_per_px);
                        profile
                            .vram_mib_at
                            .entry(size_px)
                            .or_default()
                            .set(backend, vram_mib);
                    }
                    CellOutcome::Oom => {
                        profile.cells_failed_oom.push((backend, size_px));
                    }
                    CellOutcome::TimedOut => {
                        report.timed_out_cells.push((tag.clone(), backend, size));
                        profile.cells_failed_oom.push((backend, size_px));
                        backend_timed_out_at = Some(size);
                    }
                    CellOutcome::SkippedNoBackend | CellOutcome::SkippedNoMetric => {
                        // Backend / metric not enabled by features. Do
                        // not record anything — `None` entries
                        // accurately reflect "unmeasured".
                    }
                }
            }
        }

        profile.last_measured = Some(SystemTime::now());
        if !profile.ns_per_px_at.is_empty() || !profile.cells_failed_oom.is_empty() {
            report.metrics.insert(tag, profile);
        }
    }

    report.total_wall = t_start.elapsed();
    report
}

#[cfg(feature = "bench")]
enum CellOutcome {
    Ok { ns_per_px: f64, vram_mib: usize },
    Oom,
    TimedOut,
    SkippedNoBackend,
    SkippedNoMetric,
}

/// Construct + warmup + time + drop one cell. Errors are translated
/// into structured outcomes so the parent sweep can decide whether to
/// keep going at this size or skip ahead.
#[cfg(feature = "bench")]
fn measure_cell(
    kind: MetricKind,
    backend: Backend,
    size: u32,
    plan: &BenchPlan,
) -> CellOutcome {
    let cell_start = Instant::now();
    let (r, d) = synth_pair_offset_dist(size, size);

    // Pre-construction VRAM baseline. We measure the *delta* against
    // this baseline during the timed loop — that gives the marginal
    // VRAM cost of this cell, which is the meaningful number for
    // scheduling. Absolute used-MiB is dominated by the cubecl pool
    // (~4.5 GB on a 12 GB card) and isn't what Phase 3 needs to know.
    let baseline_mib = nvidia_smi_used_mib().unwrap_or(0);

    // -------- Construct --------
    let mut metric = match construct_metric(kind, backend, size, size) {
        ConstructOutcome::Ok(m) => m,
        ConstructOutcome::Oom => return CellOutcome::Oom,
        ConstructOutcome::NoBackend => return CellOutcome::SkippedNoBackend,
        ConstructOutcome::NoMetric => return CellOutcome::SkippedNoMetric,
        ConstructOutcome::OtherErr(_msg) => return CellOutcome::Oom,
    };

    // -------- Warmup --------
    for _ in 0..plan.warmup_iters {
        match metric.compute(&r, &d) {
            Ok(()) => {}
            Err(CallErr::Oom) => return CellOutcome::Oom,
            Err(CallErr::Other(_)) => return CellOutcome::Oom,
        }
    }

    // -------- Timed (with VRAM sampling interleaved) --------
    let mut peak_mib = baseline_mib;
    let mut durations: Vec<Duration> = Vec::with_capacity(plan.timed_iters);
    for _ in 0..plan.timed_iters {
        let t0 = Instant::now();
        match metric.compute(&r, &d) {
            Ok(()) => durations.push(t0.elapsed()),
            Err(CallErr::Oom) | Err(CallErr::Other(_)) => return CellOutcome::Oom,
        }
        // Sample after each timed call — captures the steady-state
        // peak without dragging GPU time into the timer.
        if let Some(v) = nvidia_smi_used_mib() {
            if v > peak_mib {
                peak_mib = v;
            }
        }
        if cell_start.elapsed() > plan.soft_timeout_per_cell {
            return CellOutcome::TimedOut;
        }
        std::thread::sleep(plan.vram_sample_interval);
    }

    // Median (p50) duration in ns / pixels.
    let pixels = (size as u64) * (size as u64);
    let p50 = median_duration(&mut durations);
    let ns_per_px = (p50.as_nanos() as f64) / (pixels as f64);

    // Drop the metric instance — cubecl's pool keeps the buffers, but
    // the dispatcher's per-call working set returns immediately.
    drop(metric);

    let vram_delta_mib = peak_mib.saturating_sub(baseline_mib);

    CellOutcome::Ok {
        ns_per_px,
        vram_mib: vram_delta_mib,
    }
}

#[allow(dead_code)]
fn median_duration(v: &mut [Duration]) -> Duration {
    v.sort();
    let n = v.len();
    if n == 0 {
        Duration::ZERO
    } else if n % 2 == 1 {
        v[n / 2]
    } else {
        let a = v[n / 2 - 1];
        let b = v[n / 2];
        (a + b) / 2
    }
}

// ---------------------------------------------------------------------------
// Per-metric construction + dispatch shims.
//
// We can't just `Metric::new_with_memory_mode(..., MemoryMode::StripPair)`
// — the umbrella's MemoryMode is the metric-preserving subset and
// doesn't carry StripPair (that's cvvdp-specific). For Backend::GpuStripPair
// we reach into the cvvdp_gpu crate directly via `zenmetrics_api::cvvdp::*`.
// ---------------------------------------------------------------------------

#[cfg(feature = "bench")]
enum ConstructOutcome {
    Ok(BenchMetric),
    Oom,
    NoBackend,
    NoMetric,
    #[allow(dead_code)]
    OtherErr(String),
}

#[cfg(feature = "bench")]
enum CallErr {
    Oom,
    #[allow(dead_code)]
    Other(String),
}

/// Trait-object-flavoured wrapper so the bench loop doesn't care about
/// per-metric type churn. Owns one of:
///
/// - umbrella `Metric` (covers GpuFull + GpuStrip across all kinds)
/// - `cvvdp_gpu::CvvdpOpaque` (covers GpuStripPair specifically)
#[cfg(feature = "bench")]
enum BenchMetric {
    Umbrella(Box<Metric>),
    CvvdpStripPair(Box<zenmetrics_api::cvvdp::CvvdpOpaque>),
}

#[cfg(feature = "bench")]
impl BenchMetric {
    fn compute(&mut self, r: &[u8], d: &[u8]) -> std::result::Result<(), CallErr> {
        match self {
            BenchMetric::Umbrella(m) => m
                .compute_srgb_u8(r, d)
                .map(|_| ())
                .map_err(|e| classify_call_err(&e.to_string())),
            BenchMetric::CvvdpStripPair(c) => c
                .compute_srgb_u8(r, d)
                .map(|_| ())
                .map_err(|e| classify_call_err(&e.to_string())),
        }
    }
}

#[cfg(feature = "bench")]
fn classify_call_err(msg: &str) -> CallErr {
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("oom")
        || lowered.contains("out of memory")
        || lowered.contains("toobigforfull")
        || lowered.contains("cuda_error_out_of_memory")
    {
        CallErr::Oom
    } else {
        CallErr::Other(msg.into())
    }
}

#[cfg(feature = "bench")]
fn construct_metric(
    kind: MetricKind,
    backend: Backend,
    width: u32,
    height: u32,
) -> ConstructOutcome {
    // The umbrella's Backend enum is always exhaustive; ApiBackend::Cuda
    // is what the workspace ships in default mode. Phase 6 / WGPU /
    // HIP add their own backends; for Phase 2 we measure Cuda only.
    let api_backend = ApiBackend::Cuda;

    match backend {
        Backend::GpuFull => {
            let params = match MetricParams::try_default_for(kind) {
                Ok(p) => p,
                Err(_) => return ConstructOutcome::NoMetric,
            };
            match Metric::new_with_memory_mode(
                kind,
                api_backend,
                width,
                height,
                params,
                MemoryMode::Full,
            ) {
                Ok(m) => ConstructOutcome::Ok(BenchMetric::Umbrella(Box::new(m))),
                Err(e) => classify_construct_err(e),
            }
        }
        Backend::GpuStrip => {
            let params = match MetricParams::try_default_for(kind) {
                Ok(p) => p,
                Err(_) => return ConstructOutcome::NoMetric,
            };
            match Metric::new_with_memory_mode(
                kind,
                api_backend,
                width,
                height,
                params,
                MemoryMode::Strip { h_body: None },
            ) {
                Ok(m) => ConstructOutcome::Ok(BenchMetric::Umbrella(Box::new(m))),
                Err(e) => classify_construct_err(e),
            }
        }
        Backend::GpuStripPair => {
            // Direct cvvdp_gpu construction — umbrella MemoryMode
            // doesn't carry StripPair (that's the cvvdp-only Mode B
            // one-shot stripwise walker).
            cvvdp_strip_pair(width, height)
        }
        Backend::Cpu => ConstructOutcome::NoBackend,
    }
}

#[cfg(feature = "bench")]
fn classify_construct_err(e: zenmetrics_api::Error) -> ConstructOutcome {
    let msg = e.to_string();
    let lowered = msg.to_ascii_lowercase();
    if lowered.contains("not enabled") || lowered.contains("backendnotenabled") {
        ConstructOutcome::NoBackend
    } else if lowered.contains("metricnotenabled") {
        ConstructOutcome::NoMetric
    } else if lowered.contains("toobigforfull")
        || lowered.contains("out of memory")
        || lowered.contains("oom")
    {
        ConstructOutcome::Oom
    } else {
        ConstructOutcome::OtherErr(msg)
    }
}

#[cfg(feature = "bench")]
fn cvvdp_strip_pair(width: u32, height: u32) -> ConstructOutcome {
    use zenmetrics_api::cvvdp::{CvvdpOpaque, CvvdpParams, MemoryMode as CvvdpMode};

    // The 256-row body matches the existing `mem_one_size` driver.
    let mode = CvvdpMode::StripPair {
        h_body: Some(256),
    };
    match CvvdpOpaque::new_with_memory_mode(
        zenmetrics_api::cvvdp::Backend::Cuda,
        width,
        height,
        CvvdpParams::default(),
        mode,
    ) {
        Ok(c) => ConstructOutcome::Ok(BenchMetric::CvvdpStripPair(Box::new(c))),
        Err(e) => {
            let msg = e.to_string();
            let lowered = msg.to_ascii_lowercase();
            if lowered.contains("toobigforfull") || lowered.contains("oom") {
                ConstructOutcome::Oom
            } else {
                ConstructOutcome::OtherErr(msg)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// nvidia-smi sampling. Returns system-wide memory.used in MiB.
// Falls back to `None` on any error so the bench doesn't abort if
// nvidia-smi is missing (CI without a GPU).
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn nvidia_smi_used_mib() -> Option<usize> {
    use std::process::Command;
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&out.stdout).ok()?;
    s.lines()
        .next()
        .and_then(|l| l.trim().parse::<usize>().ok())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_pair_offset_dist_deterministic() {
        let (r1, d1) = synth_pair_offset_dist(64, 64);
        let (r2, d2) = synth_pair_offset_dist(64, 64);
        assert_eq!(r1, r2);
        assert_eq!(d1, d2);
        assert_eq!(r1.len(), 64 * 64 * 3);
    }

    #[test]
    fn synth_pair_offset_dist_dist_is_offset() {
        // Sample a pixel and verify the (-8, -4, +12) saturating offset.
        let (r, d) = synth_pair_offset_dist(8, 8);
        // Pixel 0,0:
        let r0 = r[0];
        let r1 = r[1];
        let r2 = r[2];
        assert_eq!(d[0], r0.saturating_sub(8));
        assert_eq!(d[1], r1.saturating_sub(4));
        assert_eq!(d[2], r2.saturating_add(12));
    }

    #[test]
    fn synth_pair_offset_dist_matches_per_crate_helper() {
        // The exact pattern from
        // `crates/cvvdp-gpu/tests/common/mod.rs::synth_pair_with_offset_dist`
        // — reproduced bit-identically here.
        fn ref_impl(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
            let mut b = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
                    let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
                    let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
                    let i = (y * w + x) * 3;
                    b[i] = r;
                    b[i + 1] = g;
                    b[i + 2] = bb;
                }
            }
            let d: Vec<u8> = b
                .chunks_exact(3)
                .flat_map(|p| {
                    [
                        p[0].saturating_sub(8),
                        p[1].saturating_sub(4),
                        p[2].saturating_add(12),
                    ]
                })
                .collect();
            (b, d)
        }
        let (r_ours, d_ours) = synth_pair_offset_dist(64, 64);
        let (r_ref, d_ref) = ref_impl(64, 64);
        assert_eq!(r_ours, r_ref);
        assert_eq!(d_ours, d_ref);
    }

    #[test]
    fn median_duration_odd_n() {
        let mut v = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        assert_eq!(
            super::median_duration(&mut v),
            Duration::from_millis(20)
        );
    }

    #[test]
    fn median_duration_even_n() {
        let mut v = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
        ];
        // (20 + 30) / 2 == 25
        assert_eq!(
            super::median_duration(&mut v),
            Duration::from_millis(25)
        );
    }

    #[test]
    fn median_duration_empty() {
        let mut v: Vec<Duration> = Vec::new();
        assert_eq!(super::median_duration(&mut v), Duration::ZERO);
    }

    #[test]
    fn bench_plan_default_sane() {
        let p = BenchPlan::default();
        assert_eq!(p.sizes, vec![1024u32, 2048, 4096]);
        assert!(p.warmup_iters > 0);
        assert!(p.timed_iters > 0);
        assert!(p.soft_timeout_per_cell >= Duration::from_secs(1));
    }
}

