//! Task #158 wall delta: multires Full vs the fixed multires Strip.
//!
//! After #158, the umbrella `MemoryMode::Strip` routes through
//! `new_multires_strip` (it previously used single-res `new_strip`,
//! which is what produced the 8% score divergence). This bench measures
//! the WALL of the corrected Strip mode against Full at the task's exact
//! sizes (256² / 1024² / 4096²) so we can confirm the score-safe Strip
//! is not slower (ideally faster).
//!
//! Two contexts are reported, because the mode_wall sweep
//! (`benchmarks/mode_wall_2026-05-31.md`) showed they invert:
//!
//! - ONE-OFF: construct + compute + drop inside the timed region (what a
//!   `score_pair` caller pays). This is where Strip wins — it never
//!   allocates the full working set.
//! - WARM: pre-built instance reused across iters (the orchestrator's
//!   cached-ref hot loop shape; Full wins here once construction is
//!   amortized).
//!
//! Run:
//!     cargo run --release -p butteraugli-gpu --features cuda,cubecl-types \
//!         --example bench_strip_wall_task158
//!
//! Writes `benchmarks/butter_strip_wall_task158_<date>.csv`.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

/// Minimum interleaved rounds per cell. Task #158 requires n≥10; we ask
/// for 12 with a generous wall cap so even the 6 s/call 4096² one-off
/// Full cell completes its rounds instead of being truncated.
const MIN_ROUNDS: usize = 12;
/// Wall cap per comparison group. 4096² one-off Full is ~6 s/call → 12
/// rounds ≈ 75 s for Full alone; 240 s leaves headroom for both arms.
const MAX_WALL: Duration = Duration::from_secs(240);

use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use zenbench::black_box;
use zenbench::{SuiteResult, Throughput};

/// 256² / 1024² / 4096² — the task #158 size grid. 4096² is the size
/// where Full's whole working set makes one-off Strip win big.
const SIZES: &[(u32, u32, &str)] = &[
    (256, 256, "256x256"),
    (1024, 1024, "1024x1024"),
    (4096, 4096, "4096x4096"),
];

/// Strip body. 256 rows; the multires walker uses body/2 for the
/// half-res strip. For images shorter than body the walker runs a
/// single strip whose body covers the whole image.
const STRIP_BODY_H: u32 = 256;

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

fn iso_date() -> String {
    let now = SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (now / 86_400) as i64 + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y0 = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y0 + 1 } else { y0 } as i32;
    format!("{y:04}-{m:02}-{d:02}")
}

fn main() {
    let result: SuiteResult = zenbench::run(|suite| {
        for &(w, h, label) in SIZES {
            let r = Arc::new(make_image(w, h, 0));
            let d = Arc::new(make_image(w, h, 7));
            let client = Backend::client(&Default::default());

            // ── ONE-OFF: construct + compute + drop inside the timer. ──
            {
                let client_o = client.clone();
                suite.compare(&format!("{label}/oneoff"), |group| {
                    group
                        .config()
                        .min_rounds(MIN_ROUNDS)
                        .max_wall_time(MAX_WALL);
                    group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
                    group.throughput_unit("px");

                    let c = client_o.clone();
                    let r_h = r.clone();
                    let d_h = d.clone();
                    group.bench("full", move |b| {
                        b.iter(|| {
                            let mut m = Butteraugli::<Backend>::new_multires(c.clone(), w, h);
                            let res = m
                                .compute(black_box(&r_h[..]), black_box(&d_h[..]))
                                .expect("oneoff multires-full");
                            black_box(res)
                        })
                    });

                    let c = client_o.clone();
                    let r_h = r.clone();
                    let d_h = d.clone();
                    group.bench("strip", move |b| {
                        b.iter(|| {
                            let mut m = Butteraugli::<Backend>::new_multires_strip(
                                c.clone(),
                                w,
                                h,
                                STRIP_BODY_H,
                            );
                            let res = m
                                .compute_strip(black_box(&r_h[..]), black_box(&d_h[..]))
                                .expect("oneoff multires-strip");
                            black_box(res)
                        })
                    });
                });
            }

            // ── WARM: pre-built instance reused each iter. ──
            {
                let whole = Arc::new(Mutex::new(Butteraugli::<Backend>::new_multires(
                    client.clone(),
                    w,
                    h,
                )));
                let strip = Arc::new(Mutex::new(Butteraugli::<Backend>::new_multires_strip(
                    client,
                    w,
                    h,
                    STRIP_BODY_H,
                )));
                let _ = whole
                    .lock()
                    .unwrap()
                    .compute(&r, &d)
                    .expect("warm full warmup");
                let _ = strip
                    .lock()
                    .unwrap()
                    .compute_strip(&r, &d)
                    .expect("warm strip warmup");

                suite.compare(&format!("{label}/warm"), |group| {
                    group
                        .config()
                        .min_rounds(MIN_ROUNDS)
                        .max_wall_time(MAX_WALL);
                    group.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));
                    group.throughput_unit("px");

                    let wh = whole.clone();
                    let r_h = r.clone();
                    let d_h = d.clone();
                    group.bench("full", move |b| {
                        b.iter(|| {
                            let res = wh
                                .lock()
                                .unwrap()
                                .compute(black_box(&r_h[..]), black_box(&d_h[..]))
                                .expect("warm multires-full");
                            black_box(res)
                        })
                    });

                    let sh = strip.clone();
                    let r_h = r.clone();
                    let d_h = d.clone();
                    group.bench("strip", move |b| {
                        b.iter(|| {
                            let res = sh
                                .lock()
                                .unwrap()
                                .compute_strip(black_box(&r_h[..]), black_box(&d_h[..]))
                                .expect("warm multires-strip");
                            black_box(res)
                        })
                    });
                });
            }
        }
    });

    // Emit CSV + a console summary with the strip/full ratio per cell.
    let date = iso_date();
    let csv_path: PathBuf = [
        "benchmarks",
        &format!("butter_strip_wall_task158_{date}.csv"),
    ]
    .iter()
    .collect();
    if let Some(parent) = csv_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut csv = String::new();
    csv.push_str("cell,mode,n_pixels,median_ns,mean_ns,mad_ns,samples,mpx_per_s\n");
    println!("\n# task#158 wall — multires Full vs fixed multires Strip");
    println!(
        "{:>18}  {:>12}  {:>12}  {:>9}  {:>5}",
        "cell", "full_med_ms", "strip_med_ms", "strip/full", "n"
    );
    for cmp in &result.comparisons {
        let px = cmp_pixels(&cmp.group_name);
        let mut full_med = 0.0f64;
        let mut strip_med = 0.0f64;
        let mut n = 0u64;
        for bench in &cmp.benchmarks {
            let s = &bench.summary;
            let mpx = if s.median > 0.0 {
                (px as f64) / (s.median / 1e9) / 1e6
            } else {
                0.0
            };
            csv.push_str(&format!(
                "{},{},{},{:.0},{:.0},{:.0},{:.0},{:.2}\n",
                cmp.group_name, bench.name, px, s.median, s.mean, s.mad, s.n, mpx,
            ));
            if bench.name == "full" {
                full_med = s.median;
            }
            if bench.name == "strip" {
                strip_med = s.median;
                n = s.n as u64;
            }
        }
        let ratio = if full_med > 0.0 {
            strip_med / full_med
        } else {
            0.0
        };
        println!(
            "{:>18}  {:>12.3}  {:>12.3}  {:>9.3}  {:>5}",
            cmp.group_name,
            full_med / 1e6,
            strip_med / 1e6,
            ratio,
            n
        );
    }
    fs::write(&csv_path, csv).expect("write csv");
    println!("\nwrote {}", csv_path.display());
}

/// Recover the pixel count from a `WxH/ctx` cell label.
fn cmp_pixels(label: &str) -> u64 {
    let dims = label.split('/').next().unwrap_or(label);
    let mut it = dims.split('x');
    let w: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let h: u64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    w * h
}
