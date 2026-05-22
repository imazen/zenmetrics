//! Multires-strip vs multires-whole throughput bench.
//!
//! Drives both [`Butteraugli::new_multires + compute`] (whole-image,
//! full-res + half-res sibling) and
//! [`Butteraugli::new_multires_strip + compute_strip`] (strip walker
//! with a synchronized half-res strip sibling) at 4MP, 12MP, and 24MP
//! on a single CUDA client. Reports paired-compare timings + MP/s and
//! the analytical GPU peak allocation per mode. Writes a CSV into
//! `benchmarks/butter_multires_strip_<date>.csv` so future sessions can
//! compare.
//!
//! At 24 MP the whole-image multires path doesn't fit on a 12 GB
//! consumer GPU — the whole-image entry is skipped at that size and
//! only the strip path is benched. The CSV records that with
//! `n_samples=0` rows for the missing mode so the table layout stays
//! consistent.
//!
//! Run:
//!     cargo run --release -p butteraugli-gpu --features cuda,cubecl-types \
//!         --example bench_multires_strip_vs_whole_cuda

use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use zenbench::Throughput;
use zenbench::black_box;

/// 4 MP, 12 MP, 24 MP. 1MP is omitted — the multires path's win shows
/// up once the half-res sibling itself becomes large enough that strip
/// memory locality pays off. The 24 MP whole-image case is skipped:
/// `new_multires` at 24 MP needs ~5 GB host RGB + ~2.4 GB GPU full-res
/// + ~600 MB half-res, which OOMs on a 12 GB consumer GPU. Strip mode
/// stays under 300 MB at 24 MP and is the whole reason the multires
/// strip walker exists.
const SIZES: &[(u32, u32, &str, bool)] = &[
    (2000, 2000, "4MP_2000x2000", true),
    (4000, 3000, "12MP_4000x3000", true),
    (6144, 4096, "24MP_6144x4096", false), // whole-image skipped
];

/// Body row count for the strip walker. 256 keeps the strip slab at
/// `W × (256 + 80) × 50 × 4 B` (the multires walker still uses the
/// same body for the full-res strip; the half-res strip uses body/2).
const STRIP_BODY_H: u32 = 256;

/// Halo rows (matches the strip walker's HALO_ROWS constant).
const STRIP_HALO_H: u32 = 40;

/// Plane count per `Butteraugli<R>` instance — see
/// `bench_strip_vs_whole_cuda.rs` for the derivation. Multires adds a
/// second instance for the half-res sibling, so the peak is roughly
/// `5/4 ×` the single-resolution allocation (half-res is 1/4 the
/// pixel count of full-res).
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

/// Multires GPU-peak allocation: full-res working_h + half-res working_h.
fn gpu_alloc_bytes_multires(width: u32, working_h: u32) -> (u64, String) {
    let full = (width as u64) * (working_h as u64) * 4 * (PLANES_PER_INSTANCE as u64);
    // Half-res sibling: width/2, working_h/2 — quarter of the full-res
    // plane bytes (per the half-res pyramid construction).
    let half_w = (width / 2).max(1) as u64;
    let half_h = (working_h / 2).max(1) as u64;
    let half = half_w * half_h * 4 * (PLANES_PER_INSTANCE as u64);
    let bytes = full + half;
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
    let days = now / 86_400;
    let (y, m, d) = civil_from_days(days as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Howard Hinnant civil_from_days (copied from
/// `bench_strip_vs_whole_cuda.rs` — we don't want a chrono dep just
/// for a CSV filename).
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
    println!("Multires GPU peak allocation per Butteraugli<R> instance (full-res + half-res sibling):");
    println!(
        "  {:>20}  {:>14}  {:>16}  {:>7}",
        "size", "whole", "strip(body=256)", "ratio"
    );
    for &(w, h, label, _whole_ok) in SIZES {
        let (whole_b, whole_s) = gpu_alloc_bytes_multires(w, h);
        let strip_working_h = STRIP_BODY_H + 2 * STRIP_HALO_H;
        let (strip_b, strip_s) = gpu_alloc_bytes_multires(w, strip_working_h);
        let ratio = (whole_b as f64) / (strip_b as f64);
        println!(
            "  {:>20}  {:>14}  {:>16}  {:>5.2}x",
            label, whole_s, strip_s, ratio,
        );
    }
    println!();
}

fn run_zenbench_suite() {
    use std::sync::{Arc, Mutex};
    use zenbench::SuiteResult;

    let result: SuiteResult = zenbench::run(|suite| {
        for &(w, h, label, whole_ok) in SIZES {
            // Pre-build instances + inputs OUTSIDE the timed closure.
            // See bench_strip_vs_whole_cuda.rs for the rationale on
            // pre-built instances + Arc<Mutex<…>> for zenbench's
            // FnMut + Send + 'static requirement.
            let r = Arc::new(make_image(w, h, 0));
            let d = Arc::new(make_image(w, h, 7));
            let client = Backend::client(&Default::default());

            if whole_ok {
                let whole = Arc::new(Mutex::new(
                    Butteraugli::<Backend>::new_multires(client.clone(), w, h),
                ));
                let strip = Arc::new(Mutex::new(
                    Butteraugli::<Backend>::new_multires_strip(client, w, h, STRIP_BODY_H),
                ));
                let _ = whole
                    .lock()
                    .unwrap()
                    .compute(&r, &d)
                    .expect("multires-whole warmup");
                let _ = strip
                    .lock()
                    .unwrap()
                    .compute_strip(&r, &d)
                    .expect("multires-strip warmup");

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
                                .expect("multires-whole");
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
                                .expect("multires-strip");
                            black_box(res)
                        })
                    });
                });
            } else {
                // 24 MP whole-image is skipped — only run strip.
                let strip = Arc::new(Mutex::new(
                    Butteraugli::<Backend>::new_multires_strip(client, w, h, STRIP_BODY_H),
                ));
                let _ = strip
                    .lock()
                    .unwrap()
                    .compute_strip(&r, &d)
                    .expect("multires-strip warmup");

                suite.compare(label, |group| {
                    group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
                    group.throughput_unit("px");

                    let strip_h = strip.clone();
                    let r_h = r.clone();
                    let d_h = d.clone();
                    group.bench("strip", move |b| {
                        b.iter(|| {
                            let res = strip_h
                                .lock()
                                .unwrap()
                                .compute_strip(black_box(&r_h[..]), black_box(&d_h[..]))
                                .expect("multires-strip");
                            black_box(res)
                        })
                    });
                });
            }
        }
    });

    let date = iso_date();
    let csv_path: PathBuf = ["benchmarks", &format!("butter_multires_strip_{date}.csv")]
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
        let Some(&(w, h, _label, _)) = SIZES.iter().find(|(_, _, l, _)| *l == cmp.group_name)
        else {
            continue;
        };
        let n_pixels = (w as u64) * (h as u64);
        let (whole_bytes, _) = gpu_alloc_bytes_multires(w, h);
        let (strip_bytes, _) = gpu_alloc_bytes_multires(w, STRIP_BODY_H + 2 * STRIP_HALO_H);
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

    // Console summary table.
    println!();
    println!("Multires strip-vs-whole compute() throughput (paired-compare, median):");
    println!(
        "  {:>20}  {:>10}  {:>12}  {:>12}  {:>10}",
        "size", "mode", "median_ms", "MP/s", "speedup"
    );
    for cmp in &result.comparisons {
        let Some(&(w, h, _, _)) = SIZES.iter().find(|(_, _, l, _)| *l == cmp.group_name) else {
            continue;
        };
        let n_pixels = (w as u64) * (h as u64);
        let mut whole_median: Option<f64> = None;
        // Pass 1: print whole row first if present so speedup is well-defined.
        for b in &cmp.benchmarks {
            if b.name == "whole" {
                let median = b.summary.median;
                let mpx_per_s = (n_pixels as f64) / (median / 1.0e9) / 1.0e6;
                println!(
                    "  {:>20}  {:>10}  {:>9.2} ms  {:>9.1} MP/s  {:>9}",
                    cmp.group_name, "whole", median / 1.0e6, mpx_per_s, "1.00x",
                );
                whole_median = Some(median);
            }
        }
        for b in &cmp.benchmarks {
            if b.name == "strip" {
                let median = b.summary.median;
                let mpx_per_s = (n_pixels as f64) / (median / 1.0e9) / 1.0e6;
                let speedup = match whole_median {
                    Some(w) => format!("{:.2}x", w / median),
                    None => "—".to_string(),
                };
                println!(
                    "  {:>20}  {:>10}  {:>9.2} ms  {:>9.1} MP/s  {:>9}",
                    cmp.group_name, "strip", median / 1.0e6, mpx_per_s, speedup,
                );
            }
        }
    }
}

fn main() {
    print_alloc_table();
    run_zenbench_suite();
}
