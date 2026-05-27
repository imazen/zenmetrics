//! End-to-end smoke test for the Phase 4 `Orchestrator::run_single`.
//!
//! Runs against a real CUDA device. Builds the orchestrator (detect +
//! cache), runs `warm()` if the capability cache is empty or stale
//! (Phase 2 bench, ~35–45 s on RTX 5070), then dispatches one cvvdp
//! task at the requested image size (default 4096²) and prints the
//! result with the backend the executor picked.
//!
//! Build with:
//!
//! ```bash
//! cargo run --release --features cuda \
//!     -p zenmetrics-orchestrator --example run_single
//! ```
//!
//! Optional env knobs:
//!
//! - `ZM_TASK_SIZE` (default `4096`): image side length (square images).
//! - `ZM_TASK_METRIC` (default `cvvdp`): one of `cvvdp`, `butter`,
//!   `ssim2`, `dssim`, `iwssim`, `zensim`.
//! - `ZM_SKIP_WARM` (default unset): set to `1` to skip the bench
//!   warmup (useful when the cache is already populated).
//!
//! ## Force-low-VRAM fallback test
//!
//! Set `ZM_FORCE_OOM_FULL=1` to pre-populate the OOM log for `(GpuFull,
//! width*height)` before running. The chooser then picks the next
//! survivor (StripPair for cvvdp, Strip for others). The output should
//! show `backend_used: Some(GpuStripPair)` (or `GpuStrip`).

use std::env;
use std::time::Instant;

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    save_profile, synth_pair_offset_dist, Backend, Orchestrator, OrchestratorConfig, Task, TaskData,
};

fn parse_metric(s: &str) -> Option<MetricKind> {
    Some(match s {
        "cvvdp" => MetricKind::Cvvdp,
        "butter" => MetricKind::Butter,
        "ssim2" => MetricKind::Ssim2,
        "dssim" => MetricKind::Dssim,
        "iwssim" => MetricKind::Iwssim,
        "zensim" => MetricKind::Zensim,
        _ => return None,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size: u32 = env::var("ZM_TASK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let metric_str = env::var("ZM_TASK_METRIC").unwrap_or_else(|_| "cvvdp".into());
    let metric = parse_metric(&metric_str).ok_or_else(|| {
        format!("unknown metric '{metric_str}' (try cvvdp/butter/ssim2/dssim/iwssim/zensim)")
    })?;
    let skip_warm = env::var("ZM_SKIP_WARM").as_deref() == Ok("1");
    let force_oom_full = env::var("ZM_FORCE_OOM_FULL").as_deref() == Ok("1");
    // Phase 6: poison every GPU backend so the OOM ladder lands on CPU.
    // Useful for the brief's acceptance gate
    // ("ZM_FORCE_OOM_FULL=1 cvvdp@4096² → CPU fallback success").
    // ZM_FORCE_OOM_FULL only poisons GpuFull (preserves GpuStripPair as
    // a fallback for cvvdp). ZM_FORCE_OOM_ALL poisons everything GPU.
    let force_oom_all = env::var("ZM_FORCE_OOM_ALL").as_deref() == Ok("1");

    // Phase 6: report which CPU backends are baked in. The OOM
    // fallback ladder lands here when every GPU candidate fails.
    let cpu_backends_enabled: Vec<&'static str> = {
        let mut v = Vec::new();
        if cfg!(feature = "cpu-cvvdp") {
            v.push("cvvdp");
        }
        if cfg!(feature = "cpu-ssim2") {
            v.push("ssim2");
        }
        if cfg!(feature = "cpu-dssim") {
            v.push("dssim");
        }
        if cfg!(feature = "cpu-butter") {
            v.push("butter");
        }
        if cfg!(feature = "cpu-zensim") {
            v.push("zensim");
        }
        v
    };

    println!(
        "zenmetrics-orchestrator Phase 4 (+ Phase 6 CPU adapters) — run_single example\n\
         size:   {size}×{size} ({} MP)\n\
         metric: {} ({})\n\
         force-oom-full: {}\n\
         cpu_backends_enabled: {:?}\n",
        ((size as u64) * (size as u64)) as f64 / 1_000_000.0,
        metric.tag(),
        metric_str,
        force_oom_full,
        cpu_backends_enabled,
    );

    // Build the orchestrator with the default capability cache dir
    // (~/.cache/zenmetrics/). This populates `gpu` + `cpu` snapshots
    // and loads any prior bench results.
    let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
    println!(
        "machine: {} / {} / {} cores / {} MiB RAM",
        orch.capability().gpu.model,
        orch.capability().cpu.brand,
        orch.capability().cpu.logical_cores,
        orch.capability().cpu.ram_mib,
    );
    println!("cache:   {}", orch.cache_path().display());
    println!(
        "metrics: {} cached profiles ({:?})\n",
        orch.capability().metrics.len(),
        orch.capability()
            .metrics
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
    );

    // Warm the cache if needed — Phase 2 bench fills `metrics.*` so the
    // Phase 3 chooser has data to interpolate over.
    if !skip_warm {
        let t = Instant::now();
        let ran = orch.warm()?;
        if ran {
            println!("warm: bench ran in {:.2?}", t.elapsed());
        } else {
            println!("warm: cache hit, no bench needed");
        }
    } else {
        println!("warm: skipped (ZM_SKIP_WARM=1)");
    }

    // Optionally pre-poison the OOM log so the chooser falls back away
    // from GpuFull. Demonstrates the executor's recovery path without
    // needing a real VRAM-pressure scenario.
    if force_oom_full || force_oom_all {
        // Need to access mutable capability — go through a dance via
        // a new orchestrator constructed from the modified capability.
        let mut cap = orch.capability().clone();
        let entry = cap.metrics.entry(metric.tag().into()).or_default();
        let pixels = (size as u64) * (size as u64);
        let backends_to_poison: &[Backend] = if force_oom_all {
            // Force CPU fallback by poisoning every GPU candidate.
            &[Backend::GpuFull, Backend::GpuStrip, Backend::GpuStripPair]
        } else {
            // Original ZM_FORCE_OOM_FULL behavior — poison only Full.
            &[Backend::GpuFull]
        };
        for &backend in backends_to_poison {
            if !entry
                .cells_failed_oom
                .iter()
                .any(|&(b, p)| b == backend && p == pixels)
            {
                entry.cells_failed_oom.push((backend, pixels));
                println!(
                    "force-oom: injected ({}, {pixels}) into cells_failed_oom for '{}'",
                    backend.tag(),
                    metric.tag(),
                );
            }
        }
        // Persist + rebuild the orchestrator so the modified profile is
        // what `run_single` sees through `choose_backend_for_task`.
        let path = orch.cache_path();
        save_profile(&path, &cap)?;
        orch = Orchestrator::from_capability(OrchestratorConfig::default(), cap);
    }

    // Build a synthetic task.
    let (r, d) = synth_pair_offset_dist(size, size);
    let task = Task {
        task_id: 1,
        ref_data: TaskData::Srgb8(r),
        dist_data: TaskData::Srgb8(d),
        width: size,
        height: size,
        metric,
        params: None,
    };

    // Run + report.
    let result = orch.run_single(task);
    println!("\n--- TaskResult ---");
    println!("task_id:            {}", result.task_id);
    println!(
        "outcome:            {}",
        match &result.outcome {
            Ok(s) => format!("Ok({:.6})", s.value),
            Err(e) => format!("Err({})", e),
        }
    );
    if let Ok(score) = &result.outcome {
        println!("metric_name:        {}", score.metric_name);
        println!("metric_version:     {}", score.metric_version);
    }
    println!(
        "backend_used:       {:?}",
        result.backend_used.map(|b| b.tag())
    );
    println!("backends_attempted: {} attempt(s)", result.backends_attempted.len());
    for (i, (b, o)) in result.backends_attempted.iter().enumerate() {
        println!("  {i}. {} -> {:?}", b.tag(), o);
    }
    println!("wall_us:            {} ({} ms)", result.wall_us, result.wall_us / 1000);
    println!("vram_peak_mib:      {:?}", result.vram_peak_mib);

    Ok(())
}
