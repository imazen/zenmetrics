// Cosmetic doc-list lints (column-aligned mode tables in the crate docs); allowed per
// the CI clippy policy in .github/workflows/ci.yml.
#![allow(clippy::doc_overindented_list_items, clippy::doc_lazy_continuation)]

//! 2026-09-24 — wall-time + thread-scaling + heaptrack harness for the NEW
//! metrics added under `cpu-metrics`:
//!
//!   nlpd       (native, crates/nlpd — magetypes SIMD partial)
//!   psnr       (native, metrics/classical.rs)
//!   psnr-y     (native, metrics/classical.rs)
//!   ssim       (native, metrics/classical.rs — scalar f64 Gaussian 11-tap)
//!   yuv420     (the VMAF adapter's RGB→YUV420 + file-write half;
//!               `metrics/vmaf.rs::write_yuv420`)
//!
//! All metric cells go through the REAL `zenmetrics_cli::metrics::run_metric`
//! dispatch (yuv420 calls `vmaf::write_yuv420` directly — the conversion is
//! the in-process cost).
//!
//! Modes per metric:
//!   `lat`   — serial single-score per iter. mean_* = per-pair cost.
//!   `par{T}`— T concurrent scores per iter on a dedicated rayon pool of T
//!             workers (T ∈ {1,4,8}); mean_* is normalized PER PAIR
//!             (group mean / T), so rows are directly comparable and the
//!             par-vs-lat delta is the thread-scaling efficiency. Run the
//!             binary under `taskset -c 0-7` so par8 gets 8 physical cores.
//!
//! Usage:
//!   new-metrics-wall <size_label> <out_tsv> [metric_filter]
//!   new-metrics-wall heap <size_label> [metric_filter]   # fixed serial reps
//!   size_label ∈ { 512 1024 4K 8K }
//!   metric_filter ∈ { nlpd psnr psnr-y ssim yuv420 }
//!
//! TSV columns (same shape as cpu_wall so the join tooling still parses):
//!   size_label  metric  mode  cold_or_warm  w  h  mean_ns  mean_ms  n_rounds  score
//! For `par{T}` rows, mean_ns/mean_ms are per-PAIR (iter wall / T), not per
//! iter — `n_rounds` still counts iters.
//!
//! Built release, NO `-C target-cpu=native` (runtime SIMD dispatch is
//! what users get — per CLAUDE.md).

use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::Write;
use std::time::Duration;

use rayon::prelude::*;
use zenbench::prelude::*;

use zenmetrics_cli::decode::Rgb8Image;
use zenmetrics_cli::metrics::{GpuRuntime, MetricKind, run_metric};

// ---------------------------------------------------------------------------
// Synthetic inputs — identical pattern to cpu_wall.rs / the heaptrack driver
// so wall and memory measurements use the same input shape.
// ---------------------------------------------------------------------------
fn synth_pair(width: u32, height: u32) -> (Vec<u8>, Vec<u8>) {
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
        .as_chunks::<3>()
        .0
        .iter()
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

fn size_dims(label: &str) -> Option<(u32, u32)> {
    match label {
        "512" => Some((512, 512)),
        "1024" => Some((1024, 1024)),
        // UHD raster sizes, not square — "4k"/"8k" in the request means the
        // video-raster convention (3840×2160 / 7680×4320).
        "4K" => Some((3840, 2160)),
        "8K" => Some((7680, 4320)),
        _ => None,
    }
}

const METRICS: &[(&str, MetricKind)] = &[
    ("nlpd", MetricKind::Nlpd),
    ("psnr", MetricKind::Psnr),
    ("psnr-y", MetricKind::PsnrY),
    ("ssim", MetricKind::Ssim),
];

fn mk_image(pixels: Vec<u8>, w: u32, h: u32) -> Rgb8Image {
    Rgb8Image {
        pixels,
        width: w,
        height: h,
    }
}

fn score_once(metric: &str, r: &Rgb8Image, d: &Rgb8Image, lane: usize) -> f64 {
    if metric == "yuv420" {
        // Conversion + plane-write half of the libvmaf adapter. Each lane
        // writes its own tmpfs path so parallel cells don't serialize on
        // one inode; production writes into a per-call tempdir.
        let path = std::env::temp_dir().join(format!("zenmetrics_yuv420_{lane}.yuv"));
        zenmetrics_cli::metrics::vmaf::write_yuv420(&path, r).unwrap();
        let _ = std::fs::remove_file(&path);
        return f64::from(r.pixels[0]);
    }
    let kind = METRICS
        .iter()
        .find(|(n, _)| *n == metric)
        .map(|(_, k)| *k)
        .unwrap_or_else(|| panic!("unknown metric {metric}"));
    let cols = run_metric(kind, r, d, GpuRuntime::Cpu).unwrap();
    cols[0].1
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 2 && args[1] == "heap" {
        heap_mode(&args);
        return;
    }
    if args.len() < 3 || args.len() > 4 {
        eprintln!(
            "usage: new-metrics-wall <size_label> <out_tsv> [metric_filter]\n  \
             or:  new-metrics-wall heap <size_label> [metric_filter]\n  \
             size_label: 512 1024 4K 8K\n  \
             metric_filter (optional): nlpd psnr psnr-y ssim yuv420"
        );
        std::process::exit(64);
    }
    let label = args[1].clone();
    let out_tsv = args[2].clone();
    let metric_filter: Option<String> = args.get(3).cloned();
    let want = |m: &str| metric_filter.as_deref().map(|f| f == m).unwrap_or(true);
    let (w, h) = match size_dims(&label) {
        Some(d) => d,
        None => {
            eprintln!("bad size label: {label}");
            std::process::exit(64);
        }
    };

    let (rv, dv) = synth_pair(w, h);
    let r: &'static Rgb8Image = Box::leak(Box::new(mk_image(rv, w, h)));
    let d: &'static Rgb8Image = Box::leak(Box::new(mk_image(dv, w, h)));

    // Dedicated pools for the par{T} cells — built ONCE so pool spin-up is
    // not inside the timed iter. Leaked to satisfy g.bench's 'static bound;
    // a bench binary's pools living to exit is fine. Under `taskset -c 0-7`
    // these land on distinct physical cores.
    let pool1: &'static rayon::ThreadPool = Box::leak(Box::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap(),
    ));
    let pool4: &'static rayon::ThreadPool = Box::leak(Box::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap(),
    ));
    let pool8: &'static rayon::ThreadPool = Box::leak(Box::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap(),
    ));

    let (group_wall, per_cell_max_time, min_rounds) = match label.as_str() {
        "512" => (Duration::from_secs(600), Duration::from_secs(15), 8usize),
        "1024" => (Duration::from_secs(900), Duration::from_secs(30), 6),
        "4K" => (Duration::from_secs(1500), Duration::from_secs(90), 3),
        "8K" => (Duration::from_secs(3600), Duration::from_secs(300), 2),
        _ => (Duration::from_secs(600), Duration::from_secs(15), 8),
    };

    let mut metric_names: Vec<&'static str> = METRICS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| want(n))
        .collect();
    if want("yuv420") {
        metric_names.push("yuv420");
    }

    let mut scores: Vec<(String, f64)> = Vec::new();

    let build = |suite: &mut zenbench::prelude::Suite| {
        suite.group(format!("new_metrics_wall_{label}"), |g| {
            g.config().max_wall_time(group_wall);
            g.config().max_time(per_cell_max_time);
            g.config().min_rounds(min_rounds);

            for &mname in &metric_names {
                let mname: &'static str = mname;
                let (r, d) = (r, d);
                g.bench(format!("{mname}__lat"), {
                    move |b| b.iter(|| black_box(score_once(mname, r, d, 0)))
                });
                for (tname, pool, t) in [
                    ("par1", pool1, 1usize),
                    ("par4", pool4, 4),
                    ("par8", pool8, 8),
                ] {
                    g.bench(format!("{mname}__{tname}"), {
                        move |b| {
                            b.iter(|| {
                                pool.install(|| {
                                    (0..t)
                                        .into_par_iter()
                                        .map(|lane| black_box(score_once(mname, r, d, lane)))
                                        .sum::<f64>()
                                })
                            })
                        }
                    });
                }
            }
        });
    };

    // Same gate convention as cpu_wall.rs: the zenbench resource gate does a
    // full sysinfo process scan per round — set CPU_WALL_NO_GATE=1 when the
    // caller guarantees a quiet machine.
    let no_gate = std::env::var("CPU_WALL_NO_GATE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let result = if no_gate {
        eprintln!("[new-metrics-wall] CPU_WALL_NO_GATE=1 — resource gate DISABLED");
        zenbench::run_gated(zenbench::GateConfig::disabled(), build)
    } else {
        zenbench::run(build)
    };

    // Score sentinels for provenance (one serial call per metric).
    for mname in &metric_names {
        let s = score_once(mname, r, d, 0);
        eprintln!("sentinel {label}/{mname}: {s}");
        scores.push((mname.to_string(), s));
    }

    write_tsv(&out_tsv, &label, w, h, &result, &scores);
}

fn lookup_score(scores: &[(String, f64)], key: &str) -> String {
    scores
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| format!("{v}"))
        .unwrap_or_else(|| "-".to_string())
}

fn write_tsv(
    out_tsv: &str,
    label: &str,
    w: u32,
    h: u32,
    result: &SuiteResult,
    scores: &[(String, f64)],
) {
    let need_header = !std::path::Path::new(out_tsv).exists();
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_tsv)
        .expect("open out tsv");
    if need_header {
        writeln!(
            f,
            "size_label\tmetric\tmode\tcold_or_warm\tw\th\tmean_ns\tmean_ms\tn_rounds\tscore"
        )
        .unwrap();
    }

    for comp in &result.comparisons {
        for bm in &comp.benchmarks {
            // names: "<metric>__lat" / "<metric>__par{1,4,8}"
            let (metric, mode) = bm.name.split_once("__").unwrap_or((&bm.name, "?"));
            let t: f64 = mode
                .strip_prefix("par")
                .and_then(|tt| tt.parse().ok())
                .unwrap_or(1.0);
            let per_pair_ns = bm.summary.mean / t;
            let mean_ms = per_pair_ns / 1.0e6;
            let score = lookup_score(scores, metric);
            writeln!(
                f,
                "{label}\t{metric}\t{mode}\tna\t{w}\t{h}\t{per_pair_ns:.1}\t{mean_ms:.4}\t{}\t{score}",
                comp.completed_rounds
            )
            .unwrap();
        }
    }
    eprintln!("wrote wall rows for size {label} to {out_tsv}");
}

/// Serial fixed-rep pass for heaptrack/callgrind wrapping — no zenbench, no
/// rayon, so the allocation profile is attributable purely to the metric.
fn heap_mode(args: &[String]) {
    let label = args.get(2).map(|s| s.as_str()).unwrap_or("1024");
    let metric_filter: Option<String> = args.get(3).cloned();
    let want = |m: &str| metric_filter.as_deref().map(|f| f == m).unwrap_or(true);
    let (w, h) = size_dims(label).unwrap_or_else(|| {
        eprintln!("bad size label: {label}");
        std::process::exit(64)
    });
    let (rv, dv) = synth_pair(w, h);
    let r = mk_image(rv, w, h);
    let d = mk_image(dv, w, h);
    let mut order: Vec<&str> = METRICS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| want(n))
        .collect();
    if want("yuv420") {
        order.push("yuv420");
    }
    for m in order {
        // Two reps: first = cold (alloc growth), second = steady-state.
        for rep in 0..2 {
            let t0 = std::time::Instant::now();
            let s = score_once(m, &r, &d, 0);
            eprintln!(
                "heap {label}/{m} rep{rep}: {:.3} ms score={}",
                t0.elapsed().as_secs_f64() * 1e3,
                s
            );
        }
    }
}
