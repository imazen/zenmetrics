//! Size-sweep timer for `Cvvdp::compute_dkl_jod` across 4 image
//! sizes. Reports per-phase wall-time + per-pixel cost plus the
//! `α + β · pixels` linear fit so launch-overhead intercept is
//! visible separately from per-pixel work.
//!
//! Sweep buckets follow the global "tiny + small + medium + large"
//! discipline:
//!
//! | bucket | dims        | pixels   |
//! | ----   | ----        | ----     |
//! | tiny   |   64 ×   64 |    4 096 |
//! | small  |  256 ×  256 |   65 536 |
//! | medium | 1024 × 1024 | 1 048 576 |
//! | large  | 4000 × 3000 | 12 000 000 |
//!
//! Run with:
//!     cargo run --release --example time_size_sweep -p cvvdp-gpu --features cuda

#![cfg(feature = "cuda")]

use std::hint::black_box;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

const SIZES: &[(u32, u32, &str)] = &[
    (64, 64, "tiny"),
    (256, 256, "small"),
    (1024, 1024, "medium"),
    (4000, 3000, "large"),
];

const WARMUP_ITERS: usize = 1;
const TIMED_ITERS: usize = 5;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let mut ref_b = vec![0u8; n];
    let mut dis_b = vec![0u8; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let b = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w as usize + x) * 3;
            ref_b[i] = r;
            ref_b[i + 1] = g;
            ref_b[i + 2] = b;
            dis_b[i] = r.saturating_sub(8);
            dis_b[i + 1] = g.saturating_sub(4);
            dis_b[i + 2] = b.saturating_add(12);
        }
    }
    (ref_b, dis_b)
}

#[derive(Clone, Copy)]
struct Row {
    label: &'static str,
    pixels: u64,
    weber_med: Duration,
    d_bands_med: Duration,
    jod_med: Duration,
}

fn median(xs: &mut [Duration]) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

fn measure_one(w: u32, h: u32, label: &'static str) -> Row {
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_b, dis_b) = synth_pair(w, h);

    let client = CudaRuntime::client(&Default::default());
    let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on cuda");

    // Warm-up — compile kernels + first allocations.
    for _ in 0..WARMUP_ITERS {
        let _ = cvvdp.compute_dkl_jod(&ref_b, &dis_b, ppd).expect("warm-up");
    }

    let mut weber_times = Vec::with_capacity(TIMED_ITERS);
    let mut d_bands_times = Vec::with_capacity(TIMED_ITERS);
    let mut jod_times = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        let t = Instant::now();
        let r = cvvdp
            .compute_dkl_weber_pyramid(&ref_b)
            .expect("weber");
        weber_times.push(t.elapsed());
        black_box(r);

        let t = Instant::now();
        let r = cvvdp
            .compute_dkl_d_bands(&ref_b, &dis_b, ppd)
            .expect("d_bands");
        d_bands_times.push(t.elapsed());
        black_box(r);

        let t = Instant::now();
        let r = cvvdp
            .compute_dkl_jod(&ref_b, &dis_b, ppd)
            .expect("jod");
        jod_times.push(t.elapsed());
        black_box(r);
    }

    Row {
        label,
        pixels: (w as u64) * (h as u64),
        weber_med: median(&mut weber_times),
        d_bands_med: median(&mut d_bands_times),
        jod_med: median(&mut jod_times),
    }
}

/// OLS fit `y = α + β · x` over the rows' (pixels, duration_ns).
/// Returns (α_ns, β_ns_per_pixel).
fn fit(rows: &[Row], extract: impl Fn(&Row) -> Duration) -> (f64, f64) {
    let n = rows.len() as f64;
    let xs: Vec<f64> = rows.iter().map(|r| r.pixels as f64).collect();
    let ys: Vec<f64> = rows.iter().map(|r| extract(r).as_nanos() as f64).collect();
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..rows.len() {
        let dx = xs[i] - mean_x;
        num += dx * (ys[i] - mean_y);
        den += dx * dx;
    }
    let beta = num / den;
    let alpha = mean_y - beta * mean_x;
    (alpha, beta)
}

fn main() {
    let mut rows = Vec::with_capacity(SIZES.len());
    for &(w, h, label) in SIZES {
        eprintln!("---- measuring {label} ({w}×{h}) ----");
        let row = measure_one(w, h, label);
        eprintln!(
            "    weber  : {:?}   d_bands: {:?}   jod: {:?}",
            row.weber_med, row.d_bands_med, row.jod_med
        );
        rows.push(row);
    }

    println!("\n=== per-phase medians ({TIMED_ITERS} timed iters) ===\n");
    println!(
        "{:<8}{:>12}  {:>14}  {:>14}  {:>14}",
        "bucket", "pixels", "weber", "d_bands", "jod"
    );
    for r in &rows {
        println!(
            "{:<8}{:>12}  {:>10.3} ms  {:>10.3} ms  {:>10.3} ms",
            r.label,
            r.pixels,
            r.weber_med.as_secs_f64() * 1000.0,
            r.d_bands_med.as_secs_f64() * 1000.0,
            r.jod_med.as_secs_f64() * 1000.0,
        );
    }

    println!("\n=== per-pixel cost (ns/px) ===\n");
    println!(
        "{:<8}{:>12}  {:>10}  {:>10}  {:>10}",
        "bucket", "pixels", "weber", "d_bands", "jod"
    );
    for r in &rows {
        let p = r.pixels as f64;
        println!(
            "{:<8}{:>12}  {:>10.2}  {:>10.2}  {:>10.2}",
            r.label,
            r.pixels,
            r.weber_med.as_nanos() as f64 / p,
            r.d_bands_med.as_nanos() as f64 / p,
            r.jod_med.as_nanos() as f64 / p,
        );
    }

    println!("\n=== linear fit  duration_ns = α + β · pixels ===\n");
    println!(
        "{:<10}  {:>10}  {:>10}",
        "phase", "α (ms)", "β (ns/px)"
    );
    let (a_w, b_w) = fit(&rows, |r| r.weber_med);
    let (a_d, b_d) = fit(&rows, |r| r.d_bands_med);
    let (a_j, b_j) = fit(&rows, |r| r.jod_med);
    println!("{:<10}  {:>10.3}  {:>10.2}", "weber", a_w / 1.0e6, b_w);
    println!("{:<10}  {:>10.3}  {:>10.2}", "d_bands", a_d / 1.0e6, b_d);
    println!("{:<10}  {:>10.3}  {:>10.2}", "jod", a_j / 1.0e6, b_j);

    println!(
        "\nIntercept α (in ms) is the launch-overhead floor — what a\n\
        single call costs regardless of image size. β is the\n\
        per-pixel slope. The tick-91–93 launch-count fusions should\n\
        be visible as a lower α than pre-fuse code."
    );
}
