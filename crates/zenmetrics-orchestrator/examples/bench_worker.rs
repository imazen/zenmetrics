//! Single-cell bench worker. Reads `WORKER_METRIC`, `WORKER_BACKEND`,
//! `WORKER_W`, `WORKER_H`, `WORKER_WARMUP`, `WORKER_TIMED` from the
//! environment, runs ONE bench cell, prints
//!
//!     READY ns_per_px=<f64> warm_ms=<f64>
//!
//! to stdout, then sleeps for `WORKER_HOLD_MS` (default 400 ms) so the
//! parent process can sample `nvidia-smi memory.used` during a
//! quiescent post-compute window.
//!
//! The parent — `zenmetrics-orchestrator::bench::run_subprocess` —
//! samples nvidia-smi:
//!   1. Before launching the child (baseline).
//!   2. Polling-loop after the READY line arrives, capturing peak.
//!   3. Reports `peak - baseline` as the cell's VRAM cost.
//!
//! This matches the proven pattern in
//! `scripts/memory_audit/audit_gpu_metrics.py` so the orchestrator's
//! per-cell VRAM numbers land in the same ballpark as the
//! `benchmarks/gpu_memory_audit_2026-05-27.csv` reference table.
//!
//! No CUDA-feature gates on the worker itself — the orchestrator's
//! `bench` feature unconditionally enables `cuda` on the metric crates
//! it pulls in, so this worker compiles whenever the orchestrator
//! crate's `bench` feature is on.

#![cfg(feature = "bench")]

use std::io::Write;
use std::time::{Duration, Instant};

use zenmetrics_api::{Backend as ApiBackend, MemoryMode, Metric, MetricKind, MetricParams};
use zenmetrics_orchestrator::synth_pair_offset_dist;

const DEFAULT_HOLD_MS: u64 = 400;
const DEFAULT_WARMUP: usize = 2;
const DEFAULT_TIMED: usize = 5;

fn env_str(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

fn env_u32(key: &str, default: u32) -> u32 {
    env_str(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    env_str(key).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let metric_tag = env_str("WORKER_METRIC").unwrap_or_else(|| "cvvdp".into());
    let backend_tag = env_str("WORKER_BACKEND").unwrap_or_else(|| "gpu_full".into());
    let w = env_u32("WORKER_W", 1024);
    let h = env_u32("WORKER_H", 1024);
    let warmup = env_usize("WORKER_WARMUP", DEFAULT_WARMUP);
    let timed = env_usize("WORKER_TIMED", DEFAULT_TIMED);
    let hold_ms = env_u32("WORKER_HOLD_MS", DEFAULT_HOLD_MS as u32) as u64;

    let kind = match metric_tag.as_str() {
        "cvvdp" => MetricKind::Cvvdp,
        "butter" => MetricKind::Butter,
        "ssim2" => MetricKind::Ssim2,
        "dssim" => MetricKind::Dssim,
        "iwssim" => MetricKind::Iwssim,
        "zensim" => MetricKind::Zensim,
        other => {
            eprintln!("ERROR unknown WORKER_METRIC: {other}");
            std::process::exit(2);
        }
    };

    let (r, d) = synth_pair_offset_dist(w, h);

    // Construct + warm + time.
    let t_construct = Instant::now();
    let mut metric = match backend_tag.as_str() {
        "gpu_full" => construct_umbrella(kind, MemoryMode::Full, w, h),
        "gpu_strip" => construct_umbrella(kind, MemoryMode::Strip { h_body: None }, w, h),
        "gpu_strip_pair" => construct_cvvdp_strip_pair(w, h),
        other => {
            eprintln!("ERROR unknown WORKER_BACKEND: {other}");
            std::process::exit(2);
        }
    }
    .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Warmup.
    for _ in 0..warmup {
        metric.compute(&r, &d)?;
    }

    // Timed.
    let mut durations: Vec<Duration> = Vec::with_capacity(timed);
    for _ in 0..timed {
        let t0 = Instant::now();
        metric.compute(&r, &d)?;
        durations.push(t0.elapsed());
    }

    let total_construct = t_construct.elapsed();
    let pixels = (w as u64) * (h as u64);
    durations.sort();
    let p50 = durations[durations.len() / 2];
    let ns_per_px = (p50.as_nanos() as f64) / (pixels as f64);

    // Print result. Parent reads this READY line.
    println!(
        "READY ns_per_px={ns_per_px:.6} warm_ms={:.3}",
        total_construct.as_secs_f64() * 1e3,
    );
    std::io::stdout().flush().ok();

    // Hold the GPU resident so the parent can sample nvidia-smi.
    std::thread::sleep(Duration::from_millis(hold_ms));

    // Explicit drop before exit — pool released back to driver here.
    drop(metric);
    Ok(())
}

enum BenchMetric {
    Umbrella(Box<Metric>),
    CvvdpStripPair(Box<zenmetrics_api::cvvdp::CvvdpOpaque>),
}

impl BenchMetric {
    fn compute(&mut self, r: &[u8], d: &[u8]) -> Result<(), zenmetrics_api::Error> {
        match self {
            BenchMetric::Umbrella(m) => m.compute_srgb_u8(r, d).map(|_| ()),
            BenchMetric::CvvdpStripPair(c) => {
                c.compute_srgb_u8(r, d)
                    .map(|_| ())
                    .map_err(|e| zenmetrics_api::Error::Metric {
                        kind: "cvvdp",
                        message: e.to_string(),
                    })
            }
        }
    }
}

fn construct_umbrella(
    kind: MetricKind,
    mode: MemoryMode,
    w: u32,
    h: u32,
) -> Result<BenchMetric, zenmetrics_api::Error> {
    let params = MetricParams::try_default_for(kind)?;
    let m = Metric::new_with_memory_mode(kind, ApiBackend::Cuda, w, h, params, mode)?;
    Ok(BenchMetric::Umbrella(Box::new(m)))
}

fn construct_cvvdp_strip_pair(w: u32, h: u32) -> Result<BenchMetric, zenmetrics_api::Error> {
    use zenmetrics_api::cvvdp::{CvvdpOpaque, CvvdpParams, MemoryMode as CvvdpMode};
    let mode = CvvdpMode::StripPair { h_body: Some(256) };
    let c = CvvdpOpaque::new_with_memory_mode(
        zenmetrics_api::cvvdp::Backend::Cuda,
        w,
        h,
        CvvdpParams::default(),
        mode,
    )
    .map_err(|e| zenmetrics_api::Error::Metric {
        kind: "cvvdp",
        message: e.to_string(),
    })?;
    Ok(BenchMetric::CvvdpStripPair(Box::new(c)))
}
