//! Strip-mode vs whole-image throughput + GPU peak-memory bench.
//!
//! Drives both `Butteraugli::new + compute` (whole-image) and
//! `Butteraugli::new_strip + compute_strip` at 1MP, 4MP, and 12MP on
//! a single CUDA client, reports paired-compare timings + MP/s, and
//! tallies the analytical GPU peak allocation per mode. Writes a CSV
//! into `benchmarks/butter_strip_vs_whole_<date>.csv` when run with
//! the default (no flag) — that file is committed alongside the
//! commit landing the strip path so future sessions can compare.
//!
//! Heaptrack support: pass `--whole-only-12mp` (or `--strip-only-12mp`)
//! to run one mode at 12 MP without zenbench, so an external
//! `heaptrack` wrapper captures a clean process trace of just that
//! path's host allocations. Stdout reports a one-line summary
//! (`MODE size score pnorm3`).
//!
//! Run:
//!     cargo run --release -p butteraugli-gpu --features cuda,cubecl-types \
//!         --example bench_strip_vs_whole_cuda
//!
//!     # Heaptrack: host RSS at 12 MP, whole-image path.
//!     heaptrack --output /tmp/heaptrack_butter_whole.zst \
//!         target/release/examples/bench_strip_vs_whole_cuda --whole-only-12mp
//!
//!     # Heaptrack: host RSS at 12 MP, strip path.
//!     heaptrack --output /tmp/heaptrack_butter_strip.zst \
//!         target/release/examples/bench_strip_vs_whole_cuda --strip-only-12mp

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use zenbench::Throughput;
use zenbench::black_box;

/// 1 MP, 4 MP, 12 MP. 24 MP is intentionally skipped — that's the
/// size whole-image doesn't fit on a 12 GB consumer GPU (it's the
/// whole point of strip mode). The strip path at 24 MP is exercised
/// by the in-tree tests via the smaller 2048² grid + the analytical
/// GPU-allocation tally below.
const SIZES: &[(u32, u32, &str)] = &[
    (1024, 1024, "1MP_1024x1024"),
    (2000, 2000, "4MP_2000x2000"),
    (4000, 3000, "12MP_4000x3000"),
];

/// Body row count for the strip walker. 256 keeps the strip slab at
/// `W × (256 + 2·80) × 50 × 4 B = 50 × W × 416 × 4 B`. For 12 MP that's
/// ~320 MB instead of 2.4 GB — the strip path's reason to exist.
const STRIP_BODY_H: u32 = 256;

/// Halo rows (matches the strip walker's HALO_ROWS constant). Hard-
/// coded here to keep the analytical tally self-contained; if the
/// crate value changes, this constant needs an update.
const STRIP_HALO_H: u32 = 80;

/// Plane count per `Butteraugli<R>` instance. Derived from
/// `pipeline.rs::new`: 2 packed-u32 src buffers + 48 f32 planes
/// (`lin_a/lin_b ×3 = 6`, `blur_a/blur_b ×3 = 6`,
/// `freq_a/freq_b ×4 ×3 = 24`, `block_diff_dc/ac ×3 = 6`,
/// `mask + mask_scratch + cached_blurred_a + diffmap_buf + temp1 + temp2 = 6`).
/// All 50 buffers are `n × 4 bytes` regardless of mode.
const PLANES_PER_INSTANCE: usize = 50;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w as usize) * (h as usize) * 3);
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as f32 / 32.0).sin() * 50.0 + 128.0) as u8;
            let sy = ((y as f32 / 24.0).cos() * 40.0 + 128.0) as u8;
            let hf = (((x ^ y).wrapping_mul(seed.max(1)) ^ seed) & 0x3f) as u8;
            out.push(sx.wrapping_add(hf));
            out.push(sy.wrapping_add(hf));
            out.push(sx.wrapping_add(sy).wrapping_add(hf >> 1));
        }
    }
    out
}

/// Analytical GPU-peak allocation in bytes for one
/// `Butteraugli<R>` instance at the given working geometry.
/// Returns (bytes, human-readable).
fn gpu_alloc_bytes(width: u32, working_h: u32) -> (u64, String) {
    let n = (width as u64) * (working_h as u64);
    let bytes = n * 4 * (PLANES_PER_INSTANCE as u64);
    let human = if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{:.0} MB", bytes as f64 / (1024.0 * 1024.0))
    };
    (bytes, human)
}

fn iso_date() -> String {
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Plain YYYY-MM-DD via chrono-less arithmetic — calendar-correct
    // is fine since we only use it as a filename suffix.
    let days = now / 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Howard Hinnant civil_from_days. Reproduced because we don't want
/// to pull chrono just for a CSV filename.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y } as i32;
    (y, m, d)
}

fn print_alloc_table() {
    println!();
    println!("GPU peak allocation per Butteraugli<R> instance:");
    println!(
        "  {:>14}  {:>16}  {:>14}  {:>14}  {:>7}",
        "size", "n (pixels)", "whole", "strip(body=256)", "ratio"
    );
    let extra_sizes: &[(u32, u32, &str)] = &[
        (1024, 1024, "1MP"),
        (2000, 2000, "4MP"),
        (4000, 3000, "12MP"),
        (6144, 4096, "24MP"),
    ];
    for &(w, h, label) in extra_sizes {
        let (whole_b, whole_s) = gpu_alloc_bytes(w, h);
        let strip_working_h = STRIP_BODY_H + 2 * STRIP_HALO_H;
        let (strip_b, strip_s) = gpu_alloc_bytes(w, strip_working_h);
        let ratio = (whole_b as f64) / (strip_b as f64);
        println!(
            "  {:>14}  {:>16}  {:>14}  {:>14}  {:>5.2}x",
            format!("{label} {w}x{h}"),
            (w as u64) * (h as u64),
            whole_s,
            strip_s,
            ratio,
        );
    }
    println!();
}

fn run_whole_only_12mp() {
    let (w, h, _) = (4000_u32, 3000_u32, "12MP");
    let r = make_image(w, h, 0);
    let d = make_image(w, h, 7);
    let client = Backend::client(&Default::default());
    let mut b = Butteraugli::<Backend>::new(client, w, h);
    // Warm one launch so kernel compilation isn't in the heaptrack trace.
    let _ = b.compute(&r, &d).expect("whole warmup");
    let res = b.compute(&r, &d).expect("whole compute");
    println!(
        "WHOLE 12MP {}x{} score={:.6} pnorm3={:.6}",
        w, h, res.score, res.pnorm_3
    );
}

fn run_strip_only_12mp() {
    let (w, h) = (4000_u32, 3000_u32);
    let r = make_image(w, h, 0);
    let d = make_image(w, h, 7);
    let client = Backend::client(&Default::default());
    let mut b = Butteraugli::<Backend>::new_strip(client, w, h, STRIP_BODY_H);
    let _ = b.compute_strip(&r, &d).expect("strip warmup");
    let res = b.compute_strip(&r, &d).expect("strip compute");
    println!(
        "STRIP 12MP {}x{} score={:.6} pnorm3={:.6}",
        w, h, res.score, res.pnorm_3
    );
}

fn run_zenbench_suite() {
    use std::sync::{Arc, Mutex};
    use zenbench::SuiteResult;

    let result: SuiteResult = zenbench::run(|suite| {
        for &(w, h, label) in SIZES {
            // Pre-build instances + inputs OUTSIDE the timed closure.
            // The whole-image instance shares the client with the strip
            // instance: cubecl runtime keeps one buffer pool per client,
            // so allocating both in sequence does NOT double-tax the GPU
            // (when the whole-image instance is dropped first, its
            // allocations recycle). Order matters for peak — we hand the
            // bench harness fully-built instances so per-iter doesn't
            // pay any alloc cost.
            //
            // Wrapped in Arc<Mutex<…>> so the bench closures can each
            // own a clone (zenbench requires `FnMut + Send + 'static`).
            // Locks are uncontended — zenbench runs round-robin
            // single-threaded — so the Mutex is overhead-free in
            // practice.
            let r = Arc::new(make_image(w, h, 0));
            let d = Arc::new(make_image(w, h, 7));
            let client = Backend::client(&Default::default());
            let whole = Arc::new(Mutex::new(Butteraugli::<Backend>::new(
                client.clone(),
                w,
                h,
            )));
            let strip = Arc::new(Mutex::new(Butteraugli::<Backend>::new_strip(
                client,
                w,
                h,
                STRIP_BODY_H,
            )));
            // Warm both — first GPU launch triggers kernel compilation
            // (~100 ms one-off cost). Without these warmups the first
            // sample of each mode would be a wild outlier.
            let _ = whole.lock().unwrap().compute(&r, &d).expect("whole warmup");
            let _ = strip
                .lock()
                .unwrap()
                .compute_strip(&r, &d)
                .expect("strip warmup");

            suite.compare(label, |group| {
                group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
                group.throughput_unit("px");

                let whole_h = whole.clone();
                let r_h = r.clone();
                let d_h = d.clone();
                group.bench("whole", move |b| {
                    b.iter(|| {
                        let res = whole_h
                            .lock()
                            .unwrap()
                            .compute(black_box(&r_h[..]), black_box(&d_h[..]))
                            .expect("whole");
                        black_box(res)
                    })
                });

                let strip_h = strip.clone();
                let r_h = r.clone();
                let d_h = d.clone();
                group.bench("strip", move |b| {
                    b.iter(|| {
                        let res = strip_h
                            .lock()
                            .unwrap()
                            .compute_strip(black_box(&r_h[..]), black_box(&d_h[..]))
                            .expect("strip");
                        black_box(res)
                    })
                });
            });
        }
    });

    // Build CSV from the suite result. Schema (one row per (size, mode)):
    //
    //     size_label, mode, n_pixels, mean_ns, median_ns, mad_ns, min_ns,
    //     max_ns, samples, mpx_per_s, whole_alloc_bytes, strip_alloc_bytes,
    //     ratio
    let date = iso_date();
    let csv_path: PathBuf = ["benchmarks", &format!("butter_strip_vs_whole_{date}.csv")]
        .iter()
        .collect();
    if let Some(parent) = csv_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut csv = String::new();
    csv.push_str(
        "size_label,mode,n_pixels,mean_ns,median_ns,mad_ns,min_ns,max_ns,\
         samples,mpx_per_s,whole_alloc_bytes,strip_alloc_bytes,ratio\n",
    );
    for cmp in &result.comparisons {
        // Map zenbench group name back to (w, h)
        let Some(&(w, h, _label)) = SIZES.iter().find(|(_, _, l)| *l == cmp.group_name) else {
            continue;
        };
        let n_pixels = (w as u64) * (h as u64);
        let (whole_bytes, _) = gpu_alloc_bytes(w, h);
        let (strip_bytes, _) = gpu_alloc_bytes(w, STRIP_BODY_H + 2 * STRIP_HALO_H);
        let ratio = (whole_bytes as f64) / (strip_bytes as f64);
        for bench in &cmp.benchmarks {
            let s = &bench.summary;
            let mpx_per_s = (n_pixels as f64) / (s.mean / 1.0e9);
            csv.push_str(&format!(
                "{},{},{},{:.0},{:.0},{:.0},{:.0},{:.0},{},{:.1},{},{},{:.2}\n",
                cmp.group_name,
                bench.name,
                n_pixels,
                s.mean,
                s.median,
                s.mad,
                s.min,
                s.max,
                s.n,
                mpx_per_s / 1.0e6,
                whole_bytes,
                strip_bytes,
                ratio,
            ));
        }
    }
    if let Err(e) = fs::write(&csv_path, &csv) {
        eprintln!("warning: could not write {csv_path:?}: {e}");
    } else {
        println!("wrote {csv_path:?}");
    }

    // Console table summary — median ns, MP/s, and speedup.
    //
    // Median is the representative-case timing: the mean is dragged
    // up by the first sample's cold-kernel-cache cost (cubecl
    // compiles the PTX module once per kernel-signature, ~100-1000
    // ms on first launch even with explicit warm-up calls if the
    // first call from the timed group sees the warm cache replaced).
    // The median across 4+ rounds is unaffected.
    println!();
    println!("Strip-vs-whole compute() throughput (paired-compare, median):");
    println!(
        "  {:>20}  {:>10}  {:>12}  {:>12}  {:>10}",
        "size", "mode", "median_ms", "MP/s", "speedup"
    );
    for cmp in &result.comparisons {
        let Some(&(w, h, _)) = SIZES.iter().find(|(_, _, l)| *l == cmp.group_name) else {
            continue;
        };
        let n_pixels = (w as u64) * (h as u64);
        let mut medians: [Option<f64>; 2] = [None, None];
        for b in &cmp.benchmarks {
            if b.name == "whole" {
                medians[0] = Some(b.summary.median);
            } else if b.name == "strip" {
                medians[1] = Some(b.summary.median);
            }
        }
        let names = ["whole", "strip"];
        let mut whole_median = None;
        for (i, m) in medians.iter().enumerate() {
            if let Some(median) = m {
                let mpx_per_s = (n_pixels as f64) / (median / 1.0e9) / 1.0e6;
                let speedup = match (i, whole_median) {
                    (0, _) => "1.00x".to_string(),
                    (_, Some(w)) => format!("{:.2}x", w / median),
                    _ => "—".to_string(),
                };
                if i == 0 {
                    whole_median = Some(*median);
                }
                println!(
                    "  {:>20}  {:>10}  {:>9.2} ms  {:>9.1} MP/s  {:>9}",
                    cmp.group_name,
                    names[i],
                    median / 1.0e6,
                    mpx_per_s,
                    speedup,
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--whole-only-12mp") {
        run_whole_only_12mp();
        return;
    }
    if args.iter().any(|a| a == "--strip-only-12mp") {
        run_strip_only_12mp();
        return;
    }
    print_alloc_table();
    run_zenbench_suite();
}
