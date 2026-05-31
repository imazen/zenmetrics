//! multiwarm_bench — MEASURE the task #155 multi-warm session pool vs the
//! single-warm path on an interleaved-reference workload.
//!
//! The workload: R distinct references, D distortions each, presented
//! round-robin (ref0, ref1, …, refR-1, ref0, …). A single-warm cache
//! holds exactly ONE reference, so every ref switch re-runs
//! `set_reference` (the per-metric reference precompute). The multi-warm
//! pool keeps up to R references warm, so each reference is precomputed
//! ONCE and every later occurrence is a warm hit. The win is largest for
//! cvvdp (heaviest `set_reference`) and smallest/none for ssim2 (light).
//!
//! For each (size, metric) cell we run the SAME workload twice in two
//! COLD child subprocesses — multi-warm ON and OFF — so the cubecl pool
//! starts fresh each time and the wall/peak are directly comparable. The
//! child runs `--reps` repetitions and reports the MEDIAN wall plus the
//! `set_reference` call count (multi-warm) / cached-ref miss count
//! (single-warm proxy). The driver polls `nvidia-smi memory.used` to
//! capture peak VRAM per cell.
//!
//! Sizes 256² / 1024² / 4096² (tiny+medium+large per the size-sweep
//! rule). Honest-stop: this REPORTS the measured deltas; it does not
//! claim a uniform win. cvvdp at 4096² is where the unlock should be
//! largest; ssim2 at 256² is where it should be ~0.
//!
//! Run (writes a committed TSV):
//! ```sh
//! cargo run -p zenmetrics-orchestrator --release --features cuda \
//!   --example multiwarm_bench -- --reps 5 \
//!   --tsv benchmarks/multiwarm_session_pool_2026-05-30.tsv
//! ```

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::env;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

use zenmetrics_api::MetricKind;
use zenmetrics_orchestrator::{
    cache_file_path, compute_machine_hash, multiwarm_stats, reset_multiwarm_stats, save_profile,
    synth_pair_offset_dist, Backend, BackendBench, BackendVram, CapabilityProfile, CpuCapability,
    GpuCapability, MetricProfile, Orchestrator, OrchestratorConfig, PoolConfig, Task, TaskData,
};

// --- synthetic profile so the chooser deterministically picks GpuFull --
fn fake_gpu() -> GpuCapability {
    GpuCapability {
        present: true,
        model: "NVIDIA GeForce RTX 5070".into(),
        total_vram_mib: 12288,
        driver_version: "596.21".into(),
        cuda_runtime: Some("13.2.1".into()),
        compute_capability: Some("8.9".into()),
    }
}
fn fake_cpu() -> CpuCapability {
    CpuCapability {
        brand: "AMD Ryzen 9 7950X".into(),
        logical_cores: 32,
        features: vec!["avx2".into()],
        ram_mib: 131072,
    }
}
fn bench_row(rows: &[(Backend, f64)]) -> BackendBench {
    let mut b = BackendBench::default();
    for &(bk, ns) in rows {
        b.set(bk, ns);
    }
    b
}
fn vram_row(rows: &[(Backend, usize)]) -> BackendVram {
    let mut v = BackendVram::default();
    for &(bk, mib) in rows {
        v.set(bk, mib);
    }
    v
}

/// GpuFull-cheapest profile for `metric` across the bench sizes, so the
/// chooser always picks GpuFull (the session-pool path).
fn metric_profile(_metric: MetricKind) -> MetricProfile {
    let mut m = MetricProfile::default();
    for size in [256u64 * 256, 1024 * 1024, 4096 * 4096] {
        m.ns_per_px_at
            .insert(size, bench_row(&[(Backend::GpuFull, 5.0)]));
        // Keep predicted VRAM well under 12 GiB so the chooser admits it.
        let mib = ((size / (1024 * 1024)).max(1) as usize) * 200;
        m.vram_mib_at
            .insert(size, vram_row(&[(Backend::GpuFull, mib.min(6000))]));
    }
    m.last_measured = Some(SystemTime::now());
    m
}

fn build_orch(metric: MetricKind, multiwarm: bool) -> (Orchestrator, tempfile::TempDir) {
    let tmpdir = tempfile::tempdir().unwrap();
    let gpu = fake_gpu();
    let cpu = fake_cpu();
    let machine_hash = compute_machine_hash(&gpu, &cpu);
    let now = SystemTime::now();
    let mut map: BTreeMap<String, MetricProfile> = BTreeMap::new();
    map.insert(metric.tag().to_string(), metric_profile(metric));
    let profile = CapabilityProfile {
        machine_hash,
        detected_at: now,
        last_validated: now,
        gpu,
        cpu,
        metrics: map,
    };
    let mut cfg = OrchestratorConfig::default();
    cfg.cache_dir = tmpdir.path().to_path_buf();
    cfg.cache_validity = Duration::from_secs(120);
    // FIFO window (no reorder) — the interleaved arrival order is
    // PRESERVED so a single-warm cache thrashes set_reference on every
    // ref switch, while the multi-warm pool reuses each warm reference.
    // (run_all's internal sort would group by reference and neutralize
    // the interleaving for BOTH paths — that's a different, already-
    // shipped optimization; this bench isolates the multi-warm unlock on
    // the streaming submit() path where sort does NOT apply.)
    cfg.stream_reorder_window = (Duration::ZERO, 1);
    let path = cache_file_path(&cfg.cache_dir, &profile.machine_hash);
    save_profile(&path, &profile).unwrap();
    let mut orch = Orchestrator::from_capability(cfg, profile);
    let mut pc = PoolConfig::default();
    pc.multiwarm_session_pool = multiwarm;
    pc.multiwarm_budget_mib = 9000; // ample so all R refs stay warm at 256²/1024²
    pc.multiwarm_max_entries = 16;
    orch.set_pool_config(pc).expect("set_pool_config");
    (orch, tmpdir)
}

/// R distinct refs + D distinct dists, round-robin so consecutive tasks
/// switch reference. R refs, D dists each → R*D tasks per rep.
fn interleaved(size: u32, r_refs: usize, d_each: usize) -> Vec<Task> {
    let (base_ref, base_dist) = synth_pair_offset_dist(size, size);
    let refs: Vec<Vec<u8>> = (0..r_refs)
        .map(|ri| {
            let mut r = base_ref.clone();
            for (i, b) in r.iter_mut().enumerate() {
                *b = b.wrapping_add(((ri * 37 + i * 3) & 0x0f) as u8);
            }
            r
        })
        .collect();
    let dists: Vec<Vec<u8>> = (0..d_each)
        .map(|di| {
            let mut d = base_dist.clone();
            for (i, b) in d.iter_mut().enumerate() {
                *b = b.wrapping_add(((di * 53 + i) & 0x07) as u8);
            }
            d
        })
        .collect();
    let mut tasks = Vec::with_capacity(r_refs * d_each);
    let mut tid = 0u64;
    for di in 0..d_each {
        for ri in 0..r_refs {
            tasks.push(Task {
                task_id: tid,
                ref_data: TaskData::Srgb8(refs[ri].clone()),
                dist_data: TaskData::Srgb8(dists[di].clone()),
                width: size,
                height: size,
                metric: MetricKind::Cvvdp, // overwritten below
                params: None,
                ref_hash: 0,
            });
            tid += 1;
        }
    }
    tasks
}

// Default interleaving grid (overridable via --refs / --dists). 6 refs ×
// 8 dists = 48 tasks/rep. For large sizes (4096²) shrink to keep the
// wall-clock per cell bounded.
static R_REFS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(6);
static D_EACH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(8);

fn r_refs() -> usize {
    R_REFS.load(Ordering::Relaxed)
}
fn d_each() -> usize {
    D_EACH.load(Ordering::Relaxed)
}

// ===================================================================
// Child — runs `reps` interleaved workloads, prints median wall +
// set_reference count + cached-ref miss count.
// ===================================================================

fn run_child(size: u32, metric: MetricKind, multiwarm: bool, reps: usize) {
    // One warmup rep (PTX compile + first device alloc) is discarded.
    let mut walls_ms: Vec<f64> = Vec::with_capacity(reps);
    let mut total_set_ref: u64 = 0;
    let mut total_cached_miss: u64 = 0;
    let mut ok_total = 0usize;
    let mut task_total = 0usize;

    for rep in 0..(reps + 1) {
        let (mut orch, _tmp) = build_orch(metric, multiwarm);
        reset_multiwarm_stats();
        let cr_before = orch.cached_ref_stats();
        let mut tasks = interleaved(size, r_refs(), d_each());
        for t in &mut tasks {
            t.metric = metric;
        }
        let n = tasks.len();
        // Drive via streaming submit() (FIFO window) so the interleaved
        // arrival order is preserved — NOT run_all (which sorts).
        let t0 = Instant::now();
        for t in tasks {
            let _ = orch.submit(t);
        }
        let mut ok = 0usize;
        let mut drained = 0usize;
        while drained < n {
            match orch.poll_any_blocking() {
                Some(r) => {
                    if r.outcome.is_ok() {
                        ok += 1;
                    }
                    drained += 1;
                }
                None => break,
            }
        }
        let wall = t0.elapsed();
        let cr_after = orch.cached_ref_stats();
        if rep == 0 {
            // warmup — discard timing, keep nothing.
            continue;
        }
        walls_ms.push(wall.as_secs_f64() * 1e3);
        total_set_ref += multiwarm_stats().set_reference_calls;
        // cached-ref miss count is the single-warm proxy for "how many
        // reference installs the single-warm path ran" (each miss → a
        // set_reference on the worker).
        total_cached_miss += cr_after.miss_count.saturating_sub(cr_before.miss_count);
        ok_total += ok;
        task_total += n;
        drop(orch); // tear down lane (reclaims) before next rep
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if walls_ms.is_empty() {
        0.0
    } else {
        walls_ms[walls_ms.len() / 2]
    };
    let set_ref_per_rep = total_set_ref as f64 / reps.max(1) as f64;
    let cached_miss_per_rep = total_cached_miss as f64 / reps.max(1) as f64;
    println!(
        "CHILD_DONE median_ms={median:.3} set_ref_per_rep={set_ref_per_rep:.2} \
         cached_miss_per_rep={cached_miss_per_rep:.2} ok={ok_total} tasks={task_total}"
    );
}

// ===================================================================
// Driver
// ===================================================================

#[derive(Clone)]
struct Cell {
    size: u32,
    metric: MetricKind,
    multiwarm: bool,
    median_ms: f64,
    set_ref_per_rep: f64,
    cached_miss_per_rep: f64,
    peak_delta_mib: i64,
    ok: usize,
    tasks: usize,
}

fn nvidia_used_mib() -> Option<u64> {
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
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()?
        .trim()
        .parse()
        .ok()
}

fn run_cell(self_exe: &std::path::Path, size: u32, metric: MetricKind, mw: bool, reps: usize) -> Cell {
    thread::sleep(Duration::from_millis(400));
    let baseline = nvidia_used_mib().unwrap_or(0);
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(baseline));
    let s2 = stop.clone();
    let p2 = peak.clone();
    let poller = thread::spawn(move || {
        while !s2.load(Ordering::Relaxed) {
            if let Some(m) = nvidia_used_mib() {
                p2.fetch_max(m, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(40));
        }
    });

    let out = Command::new(self_exe)
        .arg("--child")
        .arg(size.to_string())
        .arg(metric.tag())
        .arg(if mw { "mw" } else { "sw" })
        .arg(reps.to_string())
        // Forward the interleaving grid so the child (a fresh process,
        // statics reset to defaults) uses the same refs/dists.
        .env("MW_REFS", r_refs().to_string())
        .env("MW_DISTS", d_each().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .expect("spawn child");
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut median_ms = 0.0;
    let mut set_ref = 0.0;
    let mut cached_miss = 0.0;
    let mut ok = 0;
    let mut tasks = 0;
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("CHILD_DONE ") {
            for kv in rest.split_whitespace() {
                let (k, v) = kv.split_once('=').unwrap_or(("", ""));
                match k {
                    "median_ms" => median_ms = v.parse().unwrap_or(0.0),
                    "set_ref_per_rep" => set_ref = v.parse().unwrap_or(0.0),
                    "cached_miss_per_rep" => cached_miss = v.parse().unwrap_or(0.0),
                    "ok" => ok = v.parse().unwrap_or(0),
                    "tasks" => tasks = v.parse().unwrap_or(0),
                    _ => {}
                }
            }
        }
    }
    let peak_mib = peak.load(Ordering::Relaxed);
    Cell {
        size,
        metric,
        multiwarm: mw,
        median_ms,
        set_ref_per_rep: set_ref,
        cached_miss_per_rep: cached_miss,
        peak_delta_mib: peak_mib as i64 - baseline as i64,
        ok,
        tasks,
    }
}

/// Read MW_REFS / MW_DISTS env vars into the grid statics (set by the
/// driver before spawning a child, or by the user before the driver).
fn apply_grid_env() {
    if let Ok(v) = env::var("MW_REFS") {
        if let Ok(n) = v.parse::<usize>() {
            R_REFS.store(n.max(1), Ordering::Relaxed);
        }
    }
    if let Ok(v) = env::var("MW_DISTS") {
        if let Ok(n) = v.parse::<usize>() {
            D_EACH.store(n.max(1), Ordering::Relaxed);
        }
    }
}

fn main() {
    apply_grid_env();
    let args: Vec<String> = env::args().collect();
    if args.len() >= 6 && args[1] == "--child" {
        let size: u32 = args[2].parse().expect("size");
        let metric = match args[3].as_str() {
            "cvvdp" => MetricKind::Cvvdp,
            "ssim2" => MetricKind::Ssim2,
            "butter" => MetricKind::Butter,
            other => panic!("unknown metric {other}"),
        };
        let mw = args[4] == "mw";
        let reps: usize = args[5].parse().expect("reps");
        run_child(size, metric, mw, reps);
        return;
    }

    let mut reps = 5usize;
    let mut tsv: Option<String> = None;
    let mut sizes: Vec<u32> = vec![256, 1024, 4096];
    let mut metrics: Vec<MetricKind> = vec![MetricKind::Cvvdp, MetricKind::Ssim2];
    let mut iter = args.iter().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--reps" => reps = iter.next().and_then(|v| v.parse().ok()).unwrap_or(5),
            "--tsv" => tsv = iter.next().cloned(),
            "--sizes" => {
                if let Some(v) = iter.next() {
                    sizes = v.split(',').filter_map(|s| s.parse().ok()).collect();
                }
            }
            "--metrics" => {
                if let Some(v) = iter.next() {
                    metrics = v
                        .split(',')
                        .filter_map(|s| match s {
                            "cvvdp" => Some(MetricKind::Cvvdp),
                            "ssim2" => Some(MetricKind::Ssim2),
                            "butter" => Some(MetricKind::Butter),
                            _ => None,
                        })
                        .collect();
                }
            }
            "--refs" => {
                if let Some(n) = iter.next().and_then(|v| v.parse::<usize>().ok()) {
                    R_REFS.store(n.max(1), Ordering::Relaxed);
                }
            }
            "--dists" => {
                if let Some(n) = iter.next().and_then(|v| v.parse::<usize>().ok()) {
                    D_EACH.store(n.max(1), Ordering::Relaxed);
                }
            }
            _ => {}
        }
    }

    let self_exe = env::current_exe().expect("current exe");
    let commit = env::var("GIT_COMMIT").unwrap_or_else(|_| "unknown".into());
    let host = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".into());

    println!(
        "# multiwarm_bench — interleaved {} refs x {} dists ({} tasks/rep), reps={reps} (FIFO submit, no sort)",
        r_refs(),
        d_each(),
        r_refs() * d_each()
    );
    println!("# host={host} commit={commit} gpu={:?} MiB used now", nvidia_used_mib());
    println!("# sizes={sizes:?} metrics={:?}", metrics.iter().map(|m| m.tag()).collect::<Vec<_>>());
    println!("# NOTE: single 'win_miss' col = cached-ref WINDOW misses (NOT the worker's actual set_reference installs, which the single-warm worker re-runs on every ref switch); wall time is the load-bearing signal.");

    let mut cells: Vec<Cell> = Vec::new();
    for &metric in &metrics {
        for &size in &sizes {
            let sw = run_cell(&self_exe, size, metric, false, reps);
            thread::sleep(Duration::from_millis(500));
            let mw = run_cell(&self_exe, size, metric, true, reps);
            thread::sleep(Duration::from_millis(500));
            let speedup = if mw.median_ms > 0.0 {
                sw.median_ms / mw.median_ms
            } else {
                0.0
            };
            println!(
                "  {:>6} {:<6}  single: {:>9.2} ms (win_miss~{:.1})  multi: {:>9.2} ms (set_ref~{:.1})  \
                 speedup={:.3}x  peak sw/mw={}/{} MiB  ok sw/mw={}/{} of {}",
                format!("{size}^2"),
                metric.tag(),
                sw.median_ms,
                sw.cached_miss_per_rep,
                mw.median_ms,
                mw.set_ref_per_rep,
                speedup,
                sw.peak_delta_mib,
                mw.peak_delta_mib,
                sw.ok,
                mw.ok,
                sw.tasks,
            );
            cells.push(sw);
            cells.push(mw);
        }
    }

    if let Some(path) = tsv {
        use std::io::Write;
        let mut f = std::fs::File::create(&path).expect("create tsv");
        writeln!(
            f,
            "# multiwarm_session_pool bench — commit={commit} host={host} \
             grid: R_REFS={} D_EACH={} tasks_per_rep={} reps={reps} \
             sizes={sizes:?} metrics={:?} budget_mib=9000 max_entries=16 \
             drive=FIFO-submit(no-sort) warmup_reps=1 wall=median",
            r_refs(),
            d_each(),
            r_refs() * d_each(),
            metrics.iter().map(|m| m.tag()).collect::<Vec<_>>()
        )
        .unwrap();
        writeln!(
            f,
            "mode\tmetric\twidth\theight\tmedian_ms\tset_ref_per_rep\tcached_miss_per_rep\tpeak_delta_mib\tok\ttasks_per_rep"
        )
        .unwrap();
        for c in &cells {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{:.3}\t{:.2}\t{:.2}\t{}\t{}\t{}",
                if c.multiwarm { "multiwarm" } else { "single" },
                c.metric.tag(),
                c.size,
                c.size,
                c.median_ms,
                c.set_ref_per_rep,
                c.cached_miss_per_rep,
                c.peak_delta_mib,
                c.ok,
                c.tasks,
            )
            .unwrap();
        }
        println!("\n# TSV at {path}");
    }
}
