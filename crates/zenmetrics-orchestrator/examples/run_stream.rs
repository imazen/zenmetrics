//! Phase 5 streaming example. Submits 100 tasks then drains via
//! `poll_any` in a loop, printing progress every 10 completions.
//!
//! ```bash
//! cargo run --release --features cuda \
//!     -p zenmetrics-orchestrator --example run_stream
//! ```
//!
//! Env knobs:
//! - `ZM_TASK_METRIC` (default `cvvdp`)
//! - `ZM_STREAM_COUNT` (default 100)
//! - `ZM_TASK_SIZE` (default 512)
//! - `ZM_SKIP_WARM` (default unset)

use std::env;
use std::time::{Duration, Instant};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    synth_pair_offset_dist, Orchestrator, OrchestratorConfig, Task, TaskData,
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
    let metric_str = env::var("ZM_TASK_METRIC").unwrap_or_else(|_| "cvvdp".into());
    let metric = parse_metric(&metric_str).ok_or_else(|| {
        format!("unknown metric '{metric_str}' (try cvvdp/butter/ssim2/dssim/iwssim/zensim)")
    })?;
    let total: usize = env::var("ZM_STREAM_COUNT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);
    let size: u32 = env::var("ZM_TASK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let skip_warm = env::var("ZM_SKIP_WARM").as_deref() == Ok("1");

    println!(
        "zenmetrics-orchestrator Phase 5 — run_stream example\n\
         metric: {} ({})\n\
         size:   {size}×{size}\n\
         count:  {total}\n",
        metric.tag(),
        metric_str
    );

    let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
    println!(
        "machine: {} / {}",
        orch.capability().gpu.model,
        orch.capability().cpu.brand
    );

    if !skip_warm {
        let t = Instant::now();
        let ran = orch.warm()?;
        if ran {
            println!("warm: bench ran in {:.2?}\n", t.elapsed());
        } else {
            println!("warm: cache hit, no bench needed\n");
        }
    } else {
        println!("warm: skipped\n");
    }

    let (r, d) = synth_pair_offset_dist(size, size);

    // Submit all tasks up front; drain as they finish.
    let t_submit = Instant::now();
    let mut submitted = 0;
    for i in 0..total {
        let task = Task {
            task_id: 6000 + i as u64,
            ref_data: TaskData::Srgb8(r.clone()),
            dist_data: TaskData::Srgb8(d.clone()),
            width: size,
            height: size,
            metric,
            params: None,
        };
        orch.submit(task)?;
        submitted += 1;
    }
    let submit_ms = t_submit.elapsed().as_millis();
    println!("submitted {submitted} tasks in {submit_ms} ms\n");

    let t_drain = Instant::now();
    let mut completed = 0usize;
    let mut ok = 0usize;
    let mut err = 0usize;
    while completed < total {
        if let Some(result) = orch.poll_any_blocking() {
            completed += 1;
            match result.outcome {
                Ok(_) => ok += 1,
                Err(_) => err += 1,
            }
            if completed % 10 == 0 || completed == total {
                let dt = t_drain.elapsed().as_secs_f64();
                println!(
                    "progress: {completed}/{total} ({} ok, {} err) in {:.2}s — {:.1} tasks/s",
                    ok,
                    err,
                    dt,
                    completed as f64 / dt
                );
            }
        } else {
            // No pending work — break out of the loop. Defensive guard
            // for the case where the pool returns None unexpectedly.
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    let stats = orch.cached_ref_stats();
    println!(
        "\ncached-ref auto-detect: hits={}, misses={}",
        stats.hit_count, stats.miss_count
    );
    let vram = orch.vram_watcher_mib();
    if let Some(mib) = vram {
        if mib != usize::MAX {
            println!("vram_watcher_mib snapshot: {mib} MiB free");
        }
    }

    Ok(())
}
