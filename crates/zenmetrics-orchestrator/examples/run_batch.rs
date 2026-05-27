//! Phase 5 batch example. Drives [`Orchestrator::run_all`] with a mix of
//! sizes and prints a summary table.
//!
//! ```bash
//! cargo run --release --features cuda \
//!     -p zenmetrics-orchestrator --example run_batch
//! ```
//!
//! Env knobs:
//! - `ZM_TASK_METRIC` (default `cvvdp`): metric to run.
//! - `ZM_BATCH_SIZE` (default 50): how many tasks to run.
//! - `ZM_SKIP_WARM` (default unset): skip the bench warmup.

use std::env;
use std::time::Instant;

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
    let batch: usize = env::var("ZM_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let skip_warm = env::var("ZM_SKIP_WARM").as_deref() == Ok("1");

    println!(
        "zenmetrics-orchestrator Phase 5 — run_batch example\n\
         metric: {} ({})\n\
         batch:  {batch} tasks\n",
        metric.tag(),
        metric_str,
    );

    let mut orch = Orchestrator::new(OrchestratorConfig::default())?;
    println!(
        "machine: {} / {} / {} cores",
        orch.capability().gpu.model,
        orch.capability().cpu.brand,
        orch.capability().cpu.logical_cores,
    );
    println!("cache:   {}", orch.cache_path().display());
    println!(
        "metrics: {} cached profiles\n",
        orch.capability().metrics.len()
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
        println!("warm: skipped (ZM_SKIP_WARM=1)\n");
    }

    // Mix of sizes for realism: cycle 256/512/1024/2048. Same ref
    // across tasks so cached-ref auto-detect kicks in.
    let sizes: [u32; 4] = [256, 512, 1024, 2048];
    let mut tasks: Vec<Task> = Vec::with_capacity(batch);
    let mut ref_cache: std::collections::HashMap<u32, (Vec<u8>, Vec<u8>)> =
        std::collections::HashMap::new();
    for i in 0..batch {
        let size = sizes[i % sizes.len()];
        let (r, d) = ref_cache
            .entry(size)
            .or_insert_with(|| synth_pair_offset_dist(size, size))
            .clone();
        tasks.push(Task {
            task_id: 5000 + i as u64,
            ref_data: TaskData::Srgb8(r),
            dist_data: TaskData::Srgb8(d),
            width: size,
            height: size,
            metric,
            params: None,
        });
    }

    let t = Instant::now();
    let results: Vec<_> = orch.run_all(tasks).collect();
    let total_wall = t.elapsed();

    println!("--- Results: {} tasks in {:.2?} ---", results.len(), total_wall);
    println!(
        "task_id    size      backend    wall_us     score"
    );
    println!(
        "---------- ---- ------------- -----------  ----------"
    );
    let mut ok = 0;
    for r in &results {
        let size_label = "?";
        let backend = r
            .backend_used
            .map(|b| b.tag())
            .unwrap_or("(none)");
        match &r.outcome {
            Ok(score) => {
                ok += 1;
                println!(
                    "{:<10} {:<4} {:<13} {:>10}  {:>10.6}",
                    r.task_id, size_label, backend, r.wall_us, score.value
                );
            }
            Err(e) => {
                println!(
                    "{:<10} {:<4} {:<13} {:>10}  ERR {e}",
                    r.task_id, size_label, backend, r.wall_us
                );
            }
        }
    }
    println!("\n{ok}/{} tasks ok", results.len());
    let stats = orch.cached_ref_stats();
    println!(
        "cached_ref auto-detect: hits={}, misses={}",
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
