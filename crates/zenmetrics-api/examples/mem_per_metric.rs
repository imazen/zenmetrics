//! mem_per_metric — measure per-metric peak GPU + host memory at
//! varying image sizes for the cached-ref sweep workload shape.
//!
//! ## Why subprocess-per-cell
//!
//! `cubecl` pools device memory inside one process: a Vec<f32> dropped
//! after `compute()` returns to the pool, not to the driver. nvidia-smi
//! `memory.used` deltas inside one process therefore *under-report*
//! every metric after the first (the first one inflates the pool, the
//! rest reuse). The honest per-metric peak comes from launching one
//! subprocess per (metric, size, regime) cell so cubecl starts cold
//! every time.
//!
//! This binary has two modes:
//!
//! - **Driver** (default): launches itself recursively as a child for
//!   each cell, polling `nvidia-smi --query-gpu=memory.used` in a
//!   ~100 ms loop while the child runs, captures the peak delta, and
//!   writes CSV.
//! - **Child** (with `--child <metric> <regime> <w> <h>` args):
//!   constructs the metric, runs the cached-ref workload, prints a
//!   summary line with HOST RSS + the score, then exits.
//!
//! ## Per-cell workload
//!
//!   1. Construct the metric in `MemoryMode::Full` via the umbrella
//!      `Metric::new_with_memory_mode`.
//!   2. Run ONE `compute_srgb_u8(ref, dist)` — "one-shot" footprint.
//!   3. `set_reference_srgb_u8(ref)` — caches ref-side state.
//!   4. `compute_with_cached_reference_srgb_u8(dist)` twice — steady
//!      per-dist working set.
//!
//! The driver records the *peak* `nvidia-smi memory.used` delta
//! observed over the child's lifetime, alongside the static
//! `estimate_gpu_memory_bytes` prediction.
//!
//! Run with:
//!
//! ```sh
//! cargo run -p zenmetrics-api --release \
//!   --features cuda,all-metrics,cubecl-types,pixels \
//!   --example mem_per_metric -- --out /tmp/mem.csv
//! ```

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};

#[cfg(feature = "zensim")]
use zenmetrics_api::zensim::ZensimFeatureRegime;

const GRID: &[(u32, u32)] = &[
    (64, 64),
    (256, 256),
    (1024, 1024),
    (2048, 2048),
    (3000, 3000),
    (4096, 4096),
    (6000, 4000),
    (8192, 8192),
];

fn nvidia_smi_used_mib() -> Option<u64> {
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
    let s = String::from_utf8_lossy(&out.stdout);
    s.trim().lines().next()?.trim().parse::<u64>().ok()
}

fn proc_self_vmrss_kib() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let v: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(v);
        }
    }
    None
}

fn make_image(seed: u64, w: u32, h: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let n = (w as usize) * (h as usize) * 3;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state & 0xFF) as u8);
    }
    out
}

fn static_estimate_full_bytes(kind: MetricKind, w: u32, h: u32, _regime: &str) -> usize {
    use zenmetrics_api as api;
    match kind {
        #[cfg(feature = "butter")]
        MetricKind::Butter => api::butter::estimate_gpu_memory_bytes(w, h),
        #[cfg(feature = "ssim2")]
        MetricKind::Ssim2 => api::ssim2::estimate_gpu_memory_bytes(w, h),
        #[cfg(feature = "dssim")]
        MetricKind::Dssim => api::dssim::estimate_gpu_memory_bytes(w, h),
        #[cfg(feature = "iwssim")]
        MetricKind::Iwssim => api::iwssim::estimate_gpu_memory_bytes(w, h),
        #[cfg(feature = "cvvdp")]
        MetricKind::Cvvdp => api::cvvdp::estimate_gpu_memory_bytes_usize(w, h),
        #[cfg(feature = "zensim")]
        MetricKind::Zensim => {
            let r = match _regime {
                "basic" => api::zensim::ZensimFeatureRegime::Basic,
                "extended" => api::zensim::ZensimFeatureRegime::Extended,
                "withiw" => api::zensim::ZensimFeatureRegime::WithIw,
                _ => api::zensim::ZensimFeatureRegime::WithIw,
            };
            api::zensim::estimate_gpu_memory_bytes(w, h, r)
        }
        #[allow(unreachable_patterns)]
        _ => 0,
    }
}

fn metric_kind_from_tag(s: &str) -> Option<MetricKind> {
    match s {
        "butter" => Some(MetricKind::Butter),
        "ssim2" => Some(MetricKind::Ssim2),
        "dssim" => Some(MetricKind::Dssim),
        "iwssim" => Some(MetricKind::Iwssim),
        "cvvdp" => Some(MetricKind::Cvvdp),
        "zensim" => Some(MetricKind::Zensim),
        _ => None,
    }
}

// ===================================================================
// Child mode — actually constructs the metric + runs the workload.
// ===================================================================

fn run_child(metric_tag: &str, regime: &str, w: u32, h: u32) {
    let kind = match metric_kind_from_tag(metric_tag) {
        Some(k) => k,
        None => {
            println!("CHILD_ERR unknown metric {metric_tag}");
            std::process::exit(2);
        }
    };

    // Build params (with zensim regime when relevant).
    let params: MetricParams = {
        let base = match MetricParams::try_default_for(kind) {
            Ok(p) => p,
            Err(e) => {
                println!("CHILD_ERR params:{e}");
                std::process::exit(2);
            }
        };
        #[cfg(feature = "zensim")]
        {
            if kind == MetricKind::Zensim {
                if let MetricParams::Zensim(p) = base {
                    let r = match regime {
                        "basic" => ZensimFeatureRegime::Basic,
                        "extended" => ZensimFeatureRegime::Extended,
                        "withiw" => ZensimFeatureRegime::WithIw,
                        _ => ZensimFeatureRegime::WithIw,
                    };
                    MetricParams::Zensim(p.with_regime(r))
                } else {
                    base
                }
            } else {
                base
            }
        }
        #[cfg(not(feature = "zensim"))]
        {
            let _ = regime;
            base
        }
    };

    let host_rss_kib_baseline = proc_self_vmrss_kib().unwrap_or(0);
    let ref_buf = make_image(0xA5A5, w, h);
    let dist_buf = make_image(0x5A5A, w, h);
    let host_rss_kib_after_alloc = proc_self_vmrss_kib().unwrap_or(0);

    // Print baseline NOW so the driver can scrape it.
    let init_smi = nvidia_smi_used_mib().unwrap_or(0);
    println!("CHILD_BASELINE smi_mib={init_smi}");
    // Tell driver we're starting metric construction.
    println!("CHILD_PHASE construct");

    let t_ctor = Instant::now();
    let metric_res = Metric::new_with_memory_mode(
        kind,
        Backend::Cuda,
        w,
        h,
        params,
        MemoryMode::Full,
    );
    let ctor_ms = t_ctor.elapsed().as_millis();
    let mut metric = match metric_res {
        Ok(m) => m,
        Err(e) => {
            println!("CHILD_ERR construct:{e}");
            std::process::exit(3);
        }
    };

    println!("CHILD_PHASE first_compute ctor_ms={ctor_ms}");
    let t_compute = Instant::now();
    let score_res = metric.compute_srgb_u8(&ref_buf, &dist_buf);
    let compute_ms = t_compute.elapsed().as_millis();
    let score_val = match score_res {
        Ok(s) => s.value,
        Err(e) => {
            println!("CHILD_ERR compute:{e}");
            std::process::exit(4);
        }
    };
    println!("CHILD_PHASE post_first_compute compute_ms={compute_ms} score={score_val:.4}");

    println!("CHILD_PHASE set_reference");
    let set_ref_res = metric.set_reference_srgb_u8(&ref_buf);
    let cached_ok = set_ref_res.is_ok();
    println!("CHILD_PHASE post_set_reference cached_ok={cached_ok}");

    if cached_ok {
        println!("CHILD_PHASE cached_dist_1");
        let _ = metric.compute_with_cached_reference_srgb_u8(&dist_buf);
        println!("CHILD_PHASE cached_dist_2");
        let dist2 = make_image(0xDEAD_BEEF, w, h);
        let _ = metric.compute_with_cached_reference_srgb_u8(&dist2);
    }

    let host_rss_kib_final = proc_self_vmrss_kib().unwrap_or(0);
    let host_metric_kib = host_rss_kib_final.saturating_sub(host_rss_kib_after_alloc);
    let host_total_kib = host_rss_kib_final.saturating_sub(host_rss_kib_baseline);

    println!(
        "CHILD_DONE score={score_val:.6} ctor_ms={ctor_ms} compute_ms={compute_ms} \
         cached_ref_supported={cached_ok} host_metric_mb={:.2} host_total_mb={:.2}",
        host_metric_kib as f64 / 1024.0,
        host_total_kib as f64 / 1024.0
    );
}

// ===================================================================
// Driver mode — spawns one child per cell and polls nvidia-smi.
// ===================================================================

#[derive(Clone, Debug)]
struct CellResult {
    metric: String,
    regime: String,
    w: u32,
    h: u32,
    mp: f64,
    estimated_gpu_mb: f64,
    baseline_gpu_mib: u64,
    peak_gpu_mib: u64,
    peak_delta_gpu_mb: f64,
    host_metric_mb: f64,
    score: f64,
    cached_ref_supported: bool,
    ctor_ms: u64,
    compute_ms: u64,
    duration_s: f64,
    status: String,
}

fn run_cell_subprocess(
    self_exe: &std::path::Path,
    kind: MetricKind,
    regime: &str,
    w: u32,
    h: u32,
) -> CellResult {
    let estimated = static_estimate_full_bytes(kind, w, h, regime);
    let estimated_mb = if estimated == usize::MAX {
        f64::NAN
    } else {
        estimated as f64 / (1024.0 * 1024.0)
    };

    let t_start = Instant::now();

    // Capture pre-child GPU baseline.
    thread::sleep(Duration::from_millis(300));
    let baseline_smi = nvidia_smi_used_mib().unwrap_or(0);

    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(baseline_smi));
    let stop2 = stop.clone();
    let peak2 = peak.clone();
    let poller = thread::spawn(move || {
        while !stop2.load(Ordering::Relaxed) {
            if let Some(m) = nvidia_smi_used_mib() {
                let prev = peak2.load(Ordering::Relaxed);
                if m > prev {
                    peak2.store(m, Ordering::Relaxed);
                }
            }
            thread::sleep(Duration::from_millis(80));
        }
    });

    let mut cmd = Command::new(self_exe);
    cmd.arg("--child")
        .arg(kind.tag())
        .arg(regime)
        .arg(w.to_string())
        .arg(h.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            stop.store(true, Ordering::Relaxed);
            let _ = poller.join();
            return CellResult {
                metric: kind.tag().to_string(),
                regime: regime.to_string(),
                w,
                h,
                mp: (w as f64 * h as f64) / 1_000_000.0,
                estimated_gpu_mb: estimated_mb,
                baseline_gpu_mib: baseline_smi,
                peak_gpu_mib: baseline_smi,
                peak_delta_gpu_mb: 0.0,
                host_metric_mb: 0.0,
                score: f64::NAN,
                cached_ref_supported: false,
                ctor_ms: 0,
                compute_ms: 0,
                duration_s: t_start.elapsed().as_secs_f64(),
                status: format!("spawn_err:{e}"),
            };
        }
    };

    // Parse the child's stdout line-by-line for the final summary.
    let mut ctor_ms = 0u64;
    let mut compute_ms = 0u64;
    let mut score = f64::NAN;
    let mut cached_ref_supported = false;
    let mut host_metric_mb = f64::NAN;
    let mut error_line = String::new();
    let stdout = child.stdout.take().expect("stdout pipe");
    let reader = BufReader::new(stdout);
    for line in reader.lines().flatten() {
        if line.starts_with("CHILD_DONE ") {
            for kv in line[10..].split_whitespace() {
                if let Some(v) = kv.strip_prefix("score=") {
                    score = v.parse().unwrap_or(f64::NAN);
                } else if let Some(v) = kv.strip_prefix("ctor_ms=") {
                    ctor_ms = v.parse().unwrap_or(0);
                } else if let Some(v) = kv.strip_prefix("compute_ms=") {
                    compute_ms = v.parse().unwrap_or(0);
                } else if let Some(v) = kv.strip_prefix("cached_ref_supported=") {
                    cached_ref_supported = v == "true";
                } else if let Some(v) = kv.strip_prefix("host_metric_mb=") {
                    host_metric_mb = v.parse().unwrap_or(f64::NAN);
                }
            }
        } else if line.starts_with("CHILD_ERR ") {
            error_line = line[10..].to_string();
        }
    }
    let status_exit = child.wait().ok();

    // Stop the poller and capture peak.
    stop.store(true, Ordering::Relaxed);
    let _ = poller.join();
    let peak_smi = peak.load(Ordering::Relaxed);
    let peak_delta_mb = peak_smi as f64 - baseline_smi as f64;

    // Read any stderr for context if the child failed.
    let mut stderr_capture = String::new();
    if let Some(mut sterr) = child.stderr.take() {
        let _ = std::io::Read::read_to_string(&mut sterr, &mut stderr_capture);
    }

    let status = if !error_line.is_empty() {
        format!("err:{error_line}")
    } else if let Some(s) = status_exit {
        if s.success() {
            "ok".to_string()
        } else {
            format!("exit:{s:?}")
        }
    } else {
        "wait_err".to_string()
    };

    let _ = stderr_capture;

    CellResult {
        metric: kind.tag().to_string(),
        regime: regime.to_string(),
        w,
        h,
        mp: (w as f64 * h as f64) / 1_000_000.0,
        estimated_gpu_mb: estimated_mb,
        baseline_gpu_mib: baseline_smi,
        peak_gpu_mib: peak_smi,
        peak_delta_gpu_mb: peak_delta_mb,
        host_metric_mb,
        score,
        cached_ref_supported,
        ctor_ms,
        compute_ms,
        duration_s: t_start.elapsed().as_secs_f64(),
        status,
    }
}

fn append_csv_row(out_path: &str, r: &CellResult) {
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .open(out_path)
        .expect("append csv");
    writeln!(
        f,
        "{},full,{},{},{},{:.3},{:.2},{},{},{:.2},{:.2},{:.4},{},{},{},{:.2},{}",
        r.metric,
        r.regime,
        r.w,
        r.h,
        r.mp,
        r.estimated_gpu_mb,
        r.baseline_gpu_mib,
        r.peak_gpu_mib,
        r.peak_delta_gpu_mb,
        r.host_metric_mb,
        r.score,
        if r.cached_ref_supported { "yes" } else { "no" },
        r.ctor_ms,
        r.compute_ms,
        r.duration_s,
        r.status.replace(',', ";"),
    )
    .ok();
}

fn driver(out_path: &str, only_metric: Option<&str>, max_side: Option<u32>) {
    let self_exe = env::current_exe().expect("current exe");
    println!("# mem_per_metric driver — out={out_path}");
    println!("# GPU0 baseline: {:?} MiB", nvidia_smi_used_mib());
    println!("# child binary: {}", self_exe.display());

    // Build kind list.
    let mut kinds: Vec<(MetricKind, &'static str)> = Vec::new();
    #[cfg(feature = "butter")]
    kinds.push((MetricKind::Butter, "-"));
    #[cfg(feature = "ssim2")]
    kinds.push((MetricKind::Ssim2, "-"));
    #[cfg(feature = "dssim")]
    kinds.push((MetricKind::Dssim, "-"));
    #[cfg(feature = "iwssim")]
    kinds.push((MetricKind::Iwssim, "-"));
    #[cfg(feature = "cvvdp")]
    kinds.push((MetricKind::Cvvdp, "-"));
    #[cfg(feature = "zensim")]
    {
        kinds.push((MetricKind::Zensim, "basic"));
        kinds.push((MetricKind::Zensim, "extended"));
        kinds.push((MetricKind::Zensim, "withiw"));
    }

    if let Some(only) = only_metric {
        kinds.retain(|(k, _)| k.tag() == only);
    }

    // Open CSV with header.
    {
        let mut f = File::create(out_path).expect("create csv");
        writeln!(
            f,
            "metric,mode,regime,width,height,mp,estimated_gpu_mb,baseline_gpu_mib,peak_gpu_mib,peak_delta_gpu_mb,host_metric_mb,score,cached_ref_supported,ctor_ms,compute_ms,duration_s,status"
        )
        .unwrap();
    }

    for (kind, regime) in kinds {
        for &(w, h) in GRID {
            if let Some(limit) = max_side {
                if w.max(h) > limit {
                    continue;
                }
            }
            print!("{}/{} {}x{}: ", kind.tag(), regime, w, h);
            std::io::stdout().flush().ok();
            let r = run_cell_subprocess(&self_exe, kind, regime, w, h);
            println!(
                "est={:.0} MB peak_delta={:.0} MB host={:.1} MB score={:.4} status={} ({:.1}s ctor={}ms compute={}ms)",
                r.estimated_gpu_mb,
                r.peak_delta_gpu_mb,
                r.host_metric_mb,
                r.score,
                r.status,
                r.duration_s,
                r.ctor_ms,
                r.compute_ms,
            );
            append_csv_row(out_path, &r);
            // Brief settle between cells.
            thread::sleep(Duration::from_millis(400));
        }
    }
    println!("# done — CSV at {out_path}");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // Child mode: --child <tag> <regime> <w> <h>
    if args.len() >= 6 && args[1] == "--child" {
        let tag = &args[2];
        let regime = &args[3];
        let w: u32 = args[4].parse().expect("w");
        let h: u32 = args[5].parse().expect("h");
        run_child(tag, regime, w, h);
        return;
    }

    // Driver mode.
    let mut out_path = "/tmp/mem_per_metric.csv".to_string();
    let mut only_metric: Option<String> = None;
    let mut max_side: Option<u32> = None;
    let mut iter = args.iter().enumerate().skip(1);
    while let Some((_, a)) = iter.next() {
        match a.as_str() {
            "--out" => {
                if let Some((_, v)) = iter.next() {
                    out_path = v.clone();
                }
            }
            "--only" => {
                if let Some((_, v)) = iter.next() {
                    only_metric = Some(v.clone());
                }
            }
            "--max-side" => {
                if let Some((_, v)) = iter.next() {
                    max_side = v.parse().ok();
                }
            }
            _ => {}
        }
    }
    driver(&out_path, only_metric.as_deref(), max_side);
}
