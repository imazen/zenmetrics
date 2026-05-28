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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "bench")]
use std::sync::Arc;
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
    /// `10 ms`.
    pub vram_sample_interval: Duration,
    /// Optional `bench_worker` example binary path. When set, the
    /// bench runner spawns this binary as a subprocess per cell so
    /// each cell sees a fresh cubecl pool — accurate VRAM at the
    /// cost of ~700 ms CUDA-init per cell. When `None`, the bench
    /// runs in-process (default). The orchestrator's actual
    /// deployment schedules all metrics in one process, so the
    /// in-process numbers are usually MORE representative; use
    /// subprocess mode when audit-style isolated VRAM numbers are
    /// required (Phase 3 chooser calibration, comparison vs the
    /// `benchmarks/gpu_memory_audit_*.csv` reference table).
    pub worker_binary: Option<std::path::PathBuf>,
    /// Hold time the subprocess worker sleeps post-READY so the
    /// parent can sample `nvidia-smi memory.used` during a quiescent
    /// window. Default 400 ms — matches `audit_gpu_metrics.py`.
    pub subprocess_hold: Duration,
    /// Phase 8a: when `false`, skip every GPU cell and only run CPU
    /// cells (for metrics whose `cpu-<metric>` feature is enabled).
    /// Set by [`crate::Orchestrator::bench_with_plan`] from the
    /// detected `gpu.present` flag — callers driving `bench::run`
    /// directly typically leave the default (`true`) and let the
    /// per-cell constructor fail loudly if the host has no GPU.
    pub gpu_present: bool,
}

impl Default for BenchPlan {
    fn default() -> Self {
        Self {
            sizes: vec![1024, 2048, 4096],
            warmup_iters: 2,
            // 3 iters keeps p50 stable while staying inside the
            // < 60 s total budget on subprocess mode. In-process mode
            // is fast enough that 5 iters comfortably fit, but the
            // shared default trades a small statistical loss for a
            // big wall-time win.
            timed_iters: 3,
            soft_timeout_per_cell: Duration::from_secs(5),
            // 10 ms — fast enough to catch the steady-state peak even
            // on small-image cells where total GPU time per call is
            // ~5 ms × 5 iters = 25 ms. 25 ms sampling missed most
            // small-cell peaks.
            vram_sample_interval: Duration::from_millis(10),
            worker_binary: None,
            // 250 ms — shorter than audit's 400 ms but the actual peak
            // is visible within the first ~50 ms after the worker's
            // READY line (CUDA driver decommits pool pages lazily on
            // exit, not immediately on `drop`). Trim to fit budget.
            subprocess_hold: Duration::from_millis(250),
            // Default-true preserves Phase 1-7 callers that drive the
            // bench directly from a real-GPU host.
            // [`crate::Orchestrator::bench_with_plan`] overrides this
            // from the detected `gpu.present` flag so a CPU-only host
            // populates only CPU cells.
            gpu_present: true,
        }
    }
}

/// Convenience: locate the `bench_worker` example binary that ships
/// alongside the calling binary. Looks at `std::env::current_exe()`,
/// resolves its parent directory, then searches for
/// `examples/bench_worker` (or platform-equivalent). Returns `None` if
/// no such binary is found — typical for tests or non-release builds.
///
/// Use this from the calling binary (e.g., the `print_capability`
/// example) to plumb a [`BenchPlan::worker_binary`] without hard-
/// coding paths.
pub fn locate_bench_worker() -> Option<std::path::PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    // Cargo's `examples/` build path: `target/<profile>/examples/<name>`.
    // If we're already in `target/<profile>/examples`, the worker is a
    // sibling. If we're elsewhere (e.g. `cargo run -p` from a workspace
    // root), search `target/release/examples` directly.
    let candidates = [
        dir.join("bench_worker"),
        dir.join("bench_worker.exe"),
        dir.join("examples/bench_worker"),
        dir.join("../examples/bench_worker"),
    ];
    for c in &candidates {
        if c.is_file() {
            return Some(c.clone());
        }
    }
    None
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
///
/// **Phase 6 update**: the grid is now per-metric × {GPU backends} ∪
/// {Cpu} when the corresponding `cpu-<metric>` feature is enabled. The
/// CPU cell extends the warm() budget by ~30s total across all metrics
/// (CPU references at 1024² + 2048² — 4096² is skipped to stay within
/// the < 60s combined budget; the chooser interpolates the missing
/// 4096² value from the 2048² point).
#[cfg(feature = "bench")]
fn backends_for_kind(kind: MetricKind) -> &'static [Backend] {
    match kind {
        // cvvdp Mode E (Strip) was rolled back upstream (changes JOD);
        // we measure Full + StripPair only.
        MetricKind::Cvvdp => {
            if cfg!(feature = "cpu-cvvdp") {
                &[Backend::GpuFull, Backend::GpuStripPair, Backend::Cpu]
            } else {
                &[Backend::GpuFull, Backend::GpuStripPair]
            }
        }
        // zensim still only supports Full + Auto.
        MetricKind::Zensim => {
            if cfg!(feature = "cpu-zensim") {
                &[Backend::GpuFull, Backend::Cpu]
            } else {
                &[Backend::GpuFull]
            }
        }
        MetricKind::Butter => {
            if cfg!(feature = "cpu-butter") {
                &[Backend::GpuFull, Backend::GpuStrip, Backend::Cpu]
            } else {
                &[Backend::GpuFull, Backend::GpuStrip]
            }
        }
        MetricKind::Ssim2 => {
            if cfg!(feature = "cpu-ssim2") {
                &[Backend::GpuFull, Backend::GpuStrip, Backend::Cpu]
            } else {
                &[Backend::GpuFull, Backend::GpuStrip]
            }
        }
        MetricKind::Dssim => {
            if cfg!(feature = "cpu-dssim") {
                &[Backend::GpuFull, Backend::GpuStrip, Backend::Cpu]
            } else {
                &[Backend::GpuFull, Backend::GpuStrip]
            }
        }
        // Phase 8g: iwssim now ships an in-tree CPU port (pure Rust
        // port of Python-IW-SSIM with magetypes SIMD).
        MetricKind::Iwssim => {
            if cfg!(feature = "cpu-iwssim") {
                &[Backend::GpuFull, Backend::GpuStrip, Backend::Cpu]
            } else {
                &[Backend::GpuFull, Backend::GpuStrip]
            }
        }
    }
}

/// CPU-only bench sizes. Smaller than the GPU grid because a CPU 4096²
/// run on butteraugli costs several seconds — the warm() budget is
/// the dominant constraint. The chooser extrapolates from these two
/// points; 4096² CPU is rare in production anyway (GPU is preferred).
#[cfg(feature = "bench")]
const CPU_BENCH_SIZES: &[u32] = &[512, 1024];

/// CPU bench warmup iterations (CPU paths don't have the same PTX-
/// compile-on-first-call jitter; one warmup is enough). Keeps the
/// budget tight.
#[cfg(feature = "bench")]
const CPU_BENCH_WARMUP: usize = 1;

/// CPU bench timed iterations. At 1024² with cvvdp (~80 ms / call)
/// 2 iters costs ~160 ms; at 2048² it's ~700 ms — within budget.
#[cfg(feature = "bench")]
const CPU_BENCH_TIMED: usize = 2;

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
            // Phase 8a: skip GPU cells when the host has no GPU. The
            // orchestrator populates `plan.gpu_present` from its
            // detected `CapabilityProfile.gpu.present`; direct callers
            // of `bench::run` default to `true`. Without this guard,
            // the per-cell constructor would attempt to dlopen
            // libcuda.so.1 and panic (cubecl-cuda assumes the runtime
            // is reachable when the `cuda` feature is enabled).
            if !plan.gpu_present
                && matches!(
                    backend,
                    Backend::GpuFull | Backend::GpuStrip | Backend::GpuStripPair
                )
            {
                continue;
            }
            // Iterate sizes descending — largest first. This way the
            // cubecl pool grows to its max early; subsequent smaller-
            // size cells slot into existing free blocks instead of
            // triggering page allocations that might trip into OOM
            // mid-bench. Important for in-process bench accuracy where
            // the pool is shared across metric instances.
            //
            // Phase 6: CPU cells use a tighter grid (CPU_BENCH_SIZES)
            // to keep the warm() budget under 60s. Per-metric override.
            let mut sizes_desc: Vec<u32> = if backend == Backend::Cpu {
                CPU_BENCH_SIZES.to_vec()
            } else {
                plan.sizes.clone()
            };
            sizes_desc.sort();
            sizes_desc.reverse();
            // With descending iteration, a soft-timeout at the largest
            // size means smaller sizes might still finish under the
            // budget — keep trying them. (The reverse — timeout at
            // small size aborting larger — would never apply since
            // we visit large first.)
            for &size in &sizes_desc {
                let cell = if let Some(ref worker) = plan.worker_binary {
                    measure_cell_subprocess(kind, backend, size, plan, worker)
                } else {
                    measure_cell(kind, backend, size, plan)
                };
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

/// Tag string for the worker `WORKER_METRIC` env var.
#[cfg(feature = "bench")]
fn metric_kind_tag(kind: MetricKind) -> &'static str {
    match kind {
        MetricKind::Cvvdp => "cvvdp",
        MetricKind::Butter => "butter",
        MetricKind::Ssim2 => "ssim2",
        MetricKind::Dssim => "dssim",
        MetricKind::Iwssim => "iwssim",
        MetricKind::Zensim => "zensim",
    }
}

/// Subprocess-per-cell variant of [`measure_cell`]. Spawns
/// `plan.worker_binary` with the cell's params in env vars, samples
/// `nvidia-smi memory.used` before launch + during the child's
/// post-READY hold window, parses `READY ns_per_px=<n> warm_ms=<n>`
/// from stdout.
///
/// Matches the proven pattern in
/// `scripts/memory_audit/audit_gpu_metrics.py` so VRAM numbers land
/// within ~10-15 % of the
/// `benchmarks/gpu_memory_audit_2026-05-27.csv` reference table.
#[cfg(feature = "bench")]
fn measure_cell_subprocess(
    kind: MetricKind,
    backend: Backend,
    size: u32,
    plan: &BenchPlan,
    worker: &std::path::Path,
) -> CellOutcome {
    // Pre-launch baseline. Wait briefly to let any prior cell's pool
    // settle (mirrors audit_gpu_metrics.py).
    std::thread::sleep(Duration::from_millis(300));
    let baseline_mib = nvidia_smi_used_mib().unwrap_or(0);

    let mut cmd = std::process::Command::new(worker);
    cmd.env("WORKER_METRIC", metric_kind_tag(kind))
        .env("WORKER_BACKEND", backend.tag())
        .env("WORKER_W", size.to_string())
        .env("WORKER_H", size.to_string())
        .env("WORKER_WARMUP", plan.warmup_iters.to_string())
        .env("WORKER_TIMED", plan.timed_iters.to_string())
        .env("WORKER_HOLD_MS", plan.subprocess_hold.as_millis().to_string())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return CellOutcome::Oom,
    };

    let cell_start = Instant::now();

    // Read READY line line-by-line. Time out if it never arrives.
    let stdout = child.stdout.take().expect("piped stdout");
    let reader = std::io::BufReader::new(stdout);
    use std::io::BufRead;
    let mut ready_line: Option<String> = None;
    let timeout_deadline = cell_start
        .checked_add(plan.soft_timeout_per_cell * 4)
        .unwrap_or(Instant::now() + Duration::from_secs(20));
    for line in reader.lines().map_while(Result::ok) {
        if Instant::now() > timeout_deadline {
            break;
        }
        if line.starts_with("READY ") {
            ready_line = Some(line);
            break;
        }
    }

    let Some(ready) = ready_line else {
        // Child failed or hung. Drain + reap.
        let _ = child.kill();
        let _ = child.wait();
        return CellOutcome::Oom;
    };

    // Parse ns_per_px= value.
    let ns_per_px = ready
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix("ns_per_px="))
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(f64::NAN);

    // Sample nvidia-smi during the hold window. Slightly less than
    // subprocess_hold to leave headroom for the child's exit syscall.
    let mut peak = baseline_mib;
    let sample_until = Instant::now() + plan.subprocess_hold - Duration::from_millis(50);
    while Instant::now() < sample_until {
        if let Some(v) = nvidia_smi_used_mib()
            && v > peak
        {
            peak = v;
        }
        std::thread::sleep(plan.vram_sample_interval);
    }

    let _ = child.wait();
    let vram_delta_mib = peak.saturating_sub(baseline_mib);

    if ns_per_px.is_nan() || ns_per_px <= 0.0 {
        return CellOutcome::Oom;
    }
    CellOutcome::Ok {
        ns_per_px,
        vram_mib: vram_delta_mib,
    }
}

/// Construct + warmup + time + drop one cell. Errors are translated
/// into structured outcomes so the parent sweep can decide whether to
/// keep going at this size or skip ahead.
///
/// VRAM sampling runs on a background thread that polls
/// `nvidia-smi --query-gpu=memory.used` at `plan.vram_sample_interval`
/// throughout the construct + warmup + timed phases. The parent thread
/// reads the atomic-tracked peak after the timed loop completes and
/// signals the sampler to exit. Sampling-during-compute is the only
/// way to catch the peak: `compute_srgb_u8` is host-side synchronous
/// (waits for the GPU to finish), so by the time it returns cubecl
/// has already released per-call scratch buffers back into the pool —
/// the peak is invisible to post-call sampling.
#[cfg(feature = "bench")]
fn measure_cell(
    kind: MetricKind,
    backend: Backend,
    size: u32,
    plan: &BenchPlan,
) -> CellOutcome {
    let cell_start = Instant::now();
    let (r, d) = synth_pair_offset_dist(size, size);

    // Phase 6: CPU paths don't touch the GPU, so the nvidia-smi sampler
    // / VRAM bookkeeping is a no-op. We still spawn the sampler because
    // its result is harmless (RAM growth isn't visible to nvidia-smi)
    // but use a tighter warmup/timed budget per the CPU constants.
    let (warmup_iters, timed_iters) = if backend == Backend::Cpu {
        (CPU_BENCH_WARMUP, CPU_BENCH_TIMED)
    } else {
        (plan.warmup_iters, plan.timed_iters)
    };

    // Pre-construction VRAM baseline. We measure the *delta* against
    // this baseline during the timed loop — that gives the marginal
    // VRAM cost of this cell, which is the meaningful number for
    // scheduling. Absolute used-MiB is dominated by the cubecl pool
    // (which retains allocations across metric instances within one
    // process) and isn't what Phase 3 needs to know.
    let baseline_mib = nvidia_smi_used_mib().unwrap_or(0);

    // -------- Background VRAM sampler --------
    let peak_mib = Arc::new(AtomicUsize::new(baseline_mib));
    let stop = Arc::new(AtomicBool::new(false));
    let sampler_peak = Arc::clone(&peak_mib);
    let sampler_stop = Arc::clone(&stop);
    let sample_interval = plan.vram_sample_interval;
    let sampler = std::thread::spawn(move || {
        while !sampler_stop.load(Ordering::Relaxed) {
            if let Some(v) = nvidia_smi_used_mib() {
                let prev = sampler_peak.load(Ordering::Relaxed);
                if v > prev {
                    sampler_peak.store(v, Ordering::Relaxed);
                }
            }
            std::thread::sleep(sample_interval);
        }
    });

    let finalize = |outcome: CellOutcome,
                    stop: Arc<AtomicBool>,
                    sampler: std::thread::JoinHandle<()>,
                    peak: Arc<AtomicUsize>,
                    baseline: usize|
     -> CellOutcome {
        stop.store(true, Ordering::Relaxed);
        let _ = sampler.join();
        match outcome {
            CellOutcome::Ok { ns_per_px, .. } => {
                let p = peak.load(Ordering::Relaxed);
                CellOutcome::Ok {
                    ns_per_px,
                    vram_mib: p.saturating_sub(baseline),
                }
            }
            other => other,
        }
    };

    // -------- Construct --------
    let mut metric = match construct_metric(kind, backend, size, size) {
        ConstructOutcome::Ok(m) => m,
        ConstructOutcome::Oom => {
            return finalize(CellOutcome::Oom, stop, sampler, peak_mib, baseline_mib);
        }
        ConstructOutcome::NoBackend => {
            return finalize(
                CellOutcome::SkippedNoBackend,
                stop,
                sampler,
                peak_mib,
                baseline_mib,
            );
        }
        ConstructOutcome::NoMetric => {
            return finalize(
                CellOutcome::SkippedNoMetric,
                stop,
                sampler,
                peak_mib,
                baseline_mib,
            );
        }
        ConstructOutcome::OtherErr(_msg) => {
            return finalize(CellOutcome::Oom, stop, sampler, peak_mib, baseline_mib);
        }
    };

    // -------- Warmup --------
    for _ in 0..warmup_iters {
        if let Err(_e) = metric.compute(&r, &d) {
            return finalize(CellOutcome::Oom, stop, sampler, peak_mib, baseline_mib);
        }
        if cell_start.elapsed() > plan.soft_timeout_per_cell {
            return finalize(CellOutcome::TimedOut, stop, sampler, peak_mib, baseline_mib);
        }
    }

    // -------- Timed --------
    let mut durations: Vec<Duration> = Vec::with_capacity(timed_iters);
    for _ in 0..timed_iters {
        let t0 = Instant::now();
        match metric.compute(&r, &d) {
            Ok(()) => durations.push(t0.elapsed()),
            Err(_) => {
                return finalize(CellOutcome::Oom, stop, sampler, peak_mib, baseline_mib);
            }
        }
        if cell_start.elapsed() > plan.soft_timeout_per_cell {
            return finalize(CellOutcome::TimedOut, stop, sampler, peak_mib, baseline_mib);
        }
    }

    // Median (p50) duration in ns / pixels.
    let pixels = (size as u64) * (size as u64);
    let p50 = median_duration(&mut durations);
    let ns_per_px = (p50.as_nanos() as f64) / (pixels as f64);

    // Drop the metric instance BEFORE stopping the sampler — gives the
    // sampler one last chance to catch any post-drop pool shrink (rare
    // but happens on big allocs).
    drop(metric);

    finalize(
        CellOutcome::Ok {
            ns_per_px,
            vram_mib: 0, // overwritten in finalize from atomic peak.
        },
        stop,
        sampler,
        peak_mib,
        baseline_mib,
    )
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
/// - Phase 6: a CPU adapter for CPU cells.
#[cfg(feature = "bench")]
enum BenchMetric {
    Umbrella(Box<Metric>),
    CvvdpStripPair(Box<zenmetrics_api::cvvdp::CvvdpOpaque>),
    Cpu(Box<crate::cpu_adapter::CpuAdapter>),
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
            BenchMetric::Cpu(adapter) => adapter
                .compute(r, d)
                .map(|_| ())
                .map_err(|e| CallErr::Other(e.to_string())),
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
        Backend::Cpu => {
            // Phase 6: CPU bench cell. Build the adapter via the same
            // factory the executor uses so production cells = bench
            // cells, byte-identical dispatch.
            let params = match MetricParams::try_default_for(kind) {
                Ok(p) => p,
                Err(_) => return ConstructOutcome::NoMetric,
            };
            match crate::cpu_adapter::CpuAdapter::new(kind, width, height, &params) {
                Ok(a) => ConstructOutcome::Ok(BenchMetric::Cpu(Box::new(a))),
                Err(e) => {
                    use crate::cpu_adapter::CpuAdapterError as E;
                    match e {
                        E::FeatureNotEnabled(_) | E::Unavailable(_) => {
                            ConstructOutcome::NoBackend
                        }
                        E::Failed(msg) => ConstructOutcome::OtherErr(msg),
                        E::InvalidInputSize { expected, got } => {
                            ConstructOutcome::OtherErr(format!(
                                "invalid input size (expected {expected}, got {got})"
                            ))
                        }
                    }
                }
            }
        }
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

