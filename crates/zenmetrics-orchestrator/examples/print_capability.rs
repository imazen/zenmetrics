//! Print the detected capability profile and (when the `bench` feature
//! is enabled) run the Phase 2 quick-bench to populate the metrics table.
//!
//! Run twice in a row: the first run benches + writes the cache, the
//! second run cache-hits (you'll see the same `detected_at` and same
//! per-metric `last_measured` stamps).
//!
//! Usage:
//!
//!     # Detection-only (Phase 1 surface):
//!     cargo run --release -p zenmetrics-orchestrator --example print_capability
//!
//!     # With the Phase 2 bench runner:
//!     cargo run --release -p zenmetrics-orchestrator --example print_capability \
//!         --features bench

use std::time::{Duration, Instant};

use zenmetrics_orchestrator::{
    detect_wsl2_host_ram_mib_hint, Backend, Orchestrator, OrchestratorConfig,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the default cache_dir (~/.cache/zenmetrics/) so the example
    // round-trips through real user storage. Mutate after `default()`
    // because OrchestratorConfig is `#[non_exhaustive]` from outside
    // the crate.
    let mut cfg = OrchestratorConfig::default();
    cfg.cache_validity = Duration::from_secs(7 * 24 * 60 * 60);

    println!("cache dir: {}", cfg.cache_dir.display());

    let mut orch = Orchestrator::new(cfg)?;
    {
        let cap = orch.capability();

        println!("cache file: {}", orch.cache_path().display());
        println!();
        println!("machine_hash:    {}", cap.machine_hash);
        println!("short_hash:      {}", cap.short_hash());
        println!(
            "detected_at:     {} (UNIX seconds)",
            cap.detected_at
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
        );
        println!(
            "last_validated:  {} (UNIX seconds)",
            cap.last_validated
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs()
        );
        println!();
        println!("[gpu]");
        println!("  present:            {}", cap.gpu.present);
        println!("  model:              {}", cap.gpu.model);
        println!("  total_vram_mib:     {}", cap.gpu.total_vram_mib);
        println!("  driver_version:     {}", cap.gpu.driver_version);
        println!("  cuda_runtime:       {:?}", cap.gpu.cuda_runtime);
        println!("  compute_capability: {:?}", cap.gpu.compute_capability);
        println!();
        println!("[cpu]");
        println!("  brand:          {}", cap.cpu.brand);
        println!("  logical_cores:  {}", cap.cpu.logical_cores);
        println!("  features:       {:?}", cap.cpu.features);
        println!("  ram_mib:        {}", cap.cpu.ram_mib);
        if let Some(host) = detect_wsl2_host_ram_mib_hint() {
            // WSL2 detected. host == 0 means we know we're under WSL2
            // but didn't probe the Windows host RAM (best-effort hint).
            if host == 0 {
                println!(
                    "  (WSL2 detected — ram_mib reports Linux-kernel-visible total; .wslconfig:memory= caps this)"
                );
            } else {
                println!("  wsl_host_ram_mib_hint: {host}");
            }
        }
    }

    // -------- Phase 2 bench (only when the `bench` feature is on) --------
    let bench_t0 = Instant::now();
    let ran = orch.warm()?;
    let bench_wall = bench_t0.elapsed();
    println!();
    if ran {
        println!(
            "[bench] populated metrics in {:.2} s (cold cache)",
            bench_wall.as_secs_f64()
        );
    } else {
        println!(
            "[bench] cache-hit (no bench needed) in {:.3} s",
            bench_wall.as_secs_f64()
        );
    }

    println!();
    println!("metrics:");
    if orch.capability().metrics.is_empty() {
        println!("  (no metrics measured — build with --features bench to populate)");
    } else {
        // Pretty-print in a stable order — alphabetical metric tag.
        for (tag, profile) in &orch.capability().metrics {
            println!("  {tag}:");
            // Identify which backends this metric measured at ANY size,
            // so the row layout stays stable per metric.
            let mut backends_seen: Vec<Backend> = Vec::new();
            for size_px in profile
                .ns_per_px_at
                .keys()
                .chain(profile.vram_mib_at.keys())
                .copied()
                .collect::<std::collections::BTreeSet<u64>>()
            {
                if let Some(bench) = profile.ns_per_px_at.get(&size_px) {
                    for b in [
                        Backend::GpuFull,
                        Backend::GpuStrip,
                        Backend::GpuStripPair,
                        Backend::Cpu,
                    ] {
                        if bench.get(b).is_some() && !backends_seen.contains(&b) {
                            backends_seen.push(b);
                        }
                    }
                }
            }

            for backend in &backends_seen {
                print!("    {:<14}:", backend.tag());
                let sizes: Vec<u64> = profile
                    .ns_per_px_at
                    .keys()
                    .copied()
                    .collect::<std::collections::BTreeSet<u64>>()
                    .into_iter()
                    .collect();
                let mut printed_any = false;
                for size_px in sizes {
                    let side = ((size_px as f64).sqrt() as u32).max(1);
                    let bench = profile.ns_per_px_at.get(&size_px);
                    let vram = profile.vram_mib_at.get(&size_px);
                    let ns = bench.and_then(|b| b.get(*backend));
                    let mib = vram.and_then(|v| v.get(*backend));
                    match (ns, mib) {
                        (Some(ns), Some(mib)) => {
                            if printed_any {
                                print!("  | ");
                            } else {
                                print!(" ");
                            }
                            print!("{side}²= {ns:>5.2} ns/px / {mib:>4} MiB");
                            printed_any = true;
                        }
                        (Some(ns), None) => {
                            if printed_any {
                                print!("  | ");
                            } else {
                                print!(" ");
                            }
                            print!("{side}²= {ns:>5.2} ns/px /    ? MiB");
                            printed_any = true;
                        }
                        _ => {}
                    }
                }
                if !printed_any {
                    print!(" (no successful cells)");
                }
                println!();
            }
            if !profile.cells_failed_oom.is_empty() {
                let oom_summary: Vec<String> = profile
                    .cells_failed_oom
                    .iter()
                    .map(|(b, px)| {
                        let side = ((*px as f64).sqrt() as u32).max(1);
                        format!("{}@{side}²", b.tag())
                    })
                    .collect();
                println!("    failed_oom: {}", oom_summary.join(", "));
            }
            if let Some(t) = profile.last_measured {
                println!(
                    "    last_measured: {} (UNIX seconds)",
                    t.duration_since(std::time::UNIX_EPOCH)?.as_secs()
                );
            }
        }
    }

    Ok(())
}
