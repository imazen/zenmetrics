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
//!
//!     # Phase 3: ask the chooser what it would pick for a given task:
//!     cargo run --release -p zenmetrics-orchestrator --example print_capability \
//!         --features bench -- --task-size 4096x4096 --metric cvvdp

use std::time::{Duration, Instant};

use zenmetrics_orchestrator::{
    detect_wsl2_host_ram_mib_hint, locate_bench_worker, Backend, BenchPlan, Orchestrator,
    OrchestratorConfig,
};

#[cfg(feature = "bench")]
use zenmetrics_orchestrator::{CandidateStatus, RejectReason, TaskShape};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tiny ad-hoc CLI parser — `--task-size WxH` + `--metric NAME`
    // are optional. We don't want to pull `clap` into the example
    // just for two flags. Unknown flags are surfaced loudly.
    let mut args = std::env::args().skip(1);
    let mut task_size: Option<(u32, u32)> = None;
    let mut task_metric: Option<String> = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--task-size" => {
                let v = args
                    .next()
                    .ok_or_else(|| "--task-size requires a WxH argument".to_string())?;
                task_size = Some(parse_size(&v)?);
            }
            "--metric" => {
                let v = args
                    .next()
                    .ok_or_else(|| "--metric requires a NAME argument".to_string())?;
                task_metric = Some(v);
            }
            "-h" | "--help" => {
                print_help();
                return Ok(());
            }
            other => {
                return Err(format!("unknown flag: {other} (try --help)").into());
            }
        }
    }
    // Both flags are required for a chooser invocation, or both omitted.
    if task_size.is_some() ^ task_metric.is_some() {
        return Err(
            "--task-size and --metric must be used together (or both omitted)".into(),
        );
    }

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
    // Bench mode:
    //
    // - Default (in-process, <60 s): the orchestrator schedules
    //   metrics in one process at deployment, so the cumulative-pool
    //   numbers in-process measurement produces ARE the orchestrator's
    //   actual operating reality. VRAM under-counts vs the
    //   subprocess-isolated audit by ~50-99 % on cells where the
    //   cubecl pool already holds enough free pages.
    //
    // - Subprocess (~100 s on RTX 5070, opt-in via
    //   ZENMETRICS_BENCH_SUBPROCESS=1): each cell runs in a fresh
    //   process so cubecl's pool starts empty. Matches the audit
    //   CSV within ~2-15 %. Use for Phase 3 chooser calibration or
    //   when comparing against `benchmarks/gpu_memory_audit_*.csv`.
    let want_subprocess = std::env::var_os("ZENMETRICS_BENCH_SUBPROCESS")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false);
    let worker = if want_subprocess { locate_bench_worker() } else { None };
    let mut plan = BenchPlan::default();
    plan.worker_binary = worker.clone();
    let ran = if orch.capability().metrics.is_empty() {
        orch.bench_with_plan(plan)?;
        true
    } else {
        false
    };
    let bench_wall = bench_t0.elapsed();
    if let Some(ref w) = worker {
        eprintln!("[bench] subprocess mode: worker = {}", w.display());
    } else if want_subprocess {
        eprintln!("[bench] subprocess requested but bench_worker not found; using in-process");
    } else {
        eprintln!("[bench] in-process mode (set ZENMETRICS_BENCH_SUBPROCESS=1 for subprocess)");
    }
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

    // -------- Phase 3 chooser demo (only when --task-size + --metric) --------
    #[cfg(feature = "bench")]
    if let (Some((w, h)), Some(metric_name)) = (task_size, task_metric) {
        println!();
        let metric = parse_metric_kind(&metric_name)?;
        let task = TaskShape {
            metric,
            width: w,
            height: h,
        };
        // Run the chooser. `choose_backend_for_task` threads a live
        // nvidia-smi VRAM probe through; on no-GPU hosts it falls
        // back to capability.gpu.total_vram_mib.
        match orch.choose_backend_for_task(&task) {
            Ok(choice) => {
                println!(
                    "chosen backend for {} at {w}x{h}:",
                    metric.tag()
                );
                println!(
                    "  selected: {} (predicted {:.2} ns/px, {} MiB, margin {} MiB)",
                    choice.backend.tag(),
                    choice.predicted_ns_per_px,
                    choice.predicted_vram_mib,
                    choice.safety_margin_mib,
                );
                println!("  considered:");
                for cand in &choice.considered {
                    print!("    {:<14}: ", cand.backend.tag());
                    match &cand.status {
                        CandidateStatus::Selected {
                            ns_per_px,
                            vram_mib,
                        } => {
                            let marker = if cand.backend == choice.backend {
                                "selected"
                            } else {
                                "selectable"
                            };
                            println!(
                                "{marker} ({ns_per_px:.2} ns/px, {vram_mib} MiB)"
                            );
                        }
                        CandidateStatus::Rejected {
                            reason,
                            predicted_ns_per_px,
                            predicted_vram_mib,
                        } => {
                            let ns_str = predicted_ns_per_px
                                .map(|v| format!("{v:.2} ns/px"))
                                .unwrap_or_else(|| "ns ?".into());
                            let mib_str = predicted_vram_mib
                                .map(|m| format!("{m} MiB"))
                                .unwrap_or_else(|| "MiB ?".into());
                            println!(
                                "rejected ({}) — would have been {ns_str}, {mib_str}",
                                reject_reason_tag(*reason)
                            );
                        }
                    }
                }
            }
            Err(e) => {
                println!("chooser error: {e}");
            }
        }
    }
    #[cfg(not(feature = "bench"))]
    {
        // Unused-variable suppression when --features bench is off but
        // the user still passes --task-size / --metric.
        let _ = (task_size, task_metric);
    }

    Ok(())
}

fn parse_size(s: &str) -> Result<(u32, u32), Box<dyn std::error::Error>> {
    // Accept WxH or W,H. Both width and height must be > 0.
    let lower = s.to_ascii_lowercase();
    let parts: Vec<&str> = if lower.contains('x') {
        lower.split('x').collect()
    } else {
        lower.split(',').collect()
    };
    if parts.len() != 2 {
        return Err(format!("--task-size expects WxH (got '{s}')").into());
    }
    let w: u32 = parts[0].trim().parse().map_err(|e| format!("width: {e}"))?;
    let h: u32 = parts[1].trim().parse().map_err(|e| format!("height: {e}"))?;
    if w == 0 || h == 0 {
        return Err("--task-size: width and height must be > 0".into());
    }
    Ok((w, h))
}

#[cfg(feature = "bench")]
fn parse_metric_kind(s: &str) -> Result<zenmetrics_api::MetricKind, Box<dyn std::error::Error>> {
    match s.to_ascii_lowercase().as_str() {
        "cvvdp" => Ok(zenmetrics_api::MetricKind::Cvvdp),
        "butter" | "butteraugli" => Ok(zenmetrics_api::MetricKind::Butter),
        "ssim2" | "ssimulacra2" => Ok(zenmetrics_api::MetricKind::Ssim2),
        "dssim" => Ok(zenmetrics_api::MetricKind::Dssim),
        "iwssim" => Ok(zenmetrics_api::MetricKind::Iwssim),
        "zensim" => Ok(zenmetrics_api::MetricKind::Zensim),
        other => Err(format!(
            "--metric: unknown kind '{other}' (try cvvdp|butter|ssim2|dssim|iwssim|zensim)"
        )
        .into()),
    }
}

#[cfg(feature = "bench")]
fn reject_reason_tag(r: RejectReason) -> &'static str {
    match r {
        RejectReason::UnsupportedByMetric => "UnsupportedByMetric",
        RejectReason::PredictedOomWithMargin => "PredictedOomWithMargin",
        RejectReason::KnownOomCell => "KnownOomCell",
        RejectReason::CpuNotYetWired => "CpuNotYetWired",
        RejectReason::CpuMetricUnavailable => "CpuMetricUnavailable",
        RejectReason::NoMeasuredData => "NoMeasuredData",
        RejectReason::NonPositivePrediction => "NonPositivePrediction",
        RejectReason::NoGpuPresent => "NoGpuPresent",
    }
}

fn print_help() {
    println!(
        "print_capability — print detected capability profile and (with --features bench)\n\
         run the Phase 2 quick-bench. Optional Phase 3 chooser demo when both\n\
         --task-size and --metric are supplied.\n\
         \n\
         Usage:\n\
           print_capability                                  # detection only\n\
           print_capability --task-size 4096x4096 --metric cvvdp   # ask chooser\n\
         \n\
         Flags:\n\
           --task-size WxH       width and height in pixels\n\
           --metric NAME         cvvdp | butter | ssim2 | dssim | iwssim | zensim\n\
           --help / -h           this message\n"
    );
}
