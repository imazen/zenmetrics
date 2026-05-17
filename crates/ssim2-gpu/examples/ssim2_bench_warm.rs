//! ssim2-specific bench so multi-agent example builds don't overwrite
//! each other (a `bench_t4_warm` exists in every GPU metric crate; the
//! last `cargo build --example bench_t4_warm` writes to the same path).
//! Use this file when running ssim2-gpu performance comparisons.
//!
//! Reports steady-state per-call ms at 12 MP across the full
//! `Ssim2Mode × Ssim2Blur` grid (4 × 2 = 8 cells):
//!
//! - `Ssim2Mode::{Full, Lossless, Fast, Faster}` (Kanetaka et al.
//!   IWAIT 2026 Technique 2: skip-map dispatch).
//! - `Ssim2Blur::{Iir, Fir}` (Kanetaka et al. IWAIT 2026 Technique 1:
//!   separable FIR D=5 blur as an opt-in distinct metric).
//!
//! Two harnesses:
//! - **Warm-ref** (`compute_with_reference`): the encoder rate-distortion
//!   hot path. Reference cached once per instance; measure each
//!   subsequent score call.
//! - **Cold** (`compute`): full single-shot pipeline including
//!   reference-side build.
//!
//! Drop the first iteration (transient kernel-compile / cache settle);
//! report the median of the remaining iterations.

use std::time::{Duration, Instant};
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::{Ssim2, Ssim2Blur, Ssim2Mode};

const W: u32 = 4000;
const H: u32 = 3000;
const ITERS: usize = 10;
const WARMUP: usize = 3;

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) / 2
    } else {
        xs[mid]
    }
}

/// Warm-ref bench: cache reference once, measure compute_with_reference_with_mode.
fn bench_warm(
    blur: Ssim2Blur,
    mode: Ssim2Mode,
    r: &[u8],
    d: &[u8],
) -> (Duration, Duration, f64) {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, W, H)
        .expect("Ssim2::new")
        .with_blur(blur);
    s.set_reference(r).expect("set_reference");

    let mut last_score = 0.0_f64;
    for _ in 0..WARMUP {
        let res = s
            .compute_with_reference_with_mode(mode, d)
            .expect("warmup");
        last_score = res.score;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let res = s
            .compute_with_reference_with_mode(mode, d)
            .expect("compute_with_reference_with_mode");
        samples.push(t.elapsed());
        last_score = res.score;
    }
    samples.sort();
    let min = samples[0];
    (median(samples.clone()), min, last_score)
}

/// Cold bench: full pipeline each call (compute_with_mode).
fn bench_cold(
    blur: Ssim2Blur,
    mode: Ssim2Mode,
    r: &[u8],
    d: &[u8],
) -> (Duration, f64) {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, W, H)
        .expect("Ssim2::new")
        .with_blur(blur);

    let mut last_score = 0.0_f64;
    for _ in 0..WARMUP {
        let res = s.compute_with_mode(mode, r, d).expect("warmup");
        last_score = res.score;
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let res = s.compute_with_mode(mode, r, d).expect("compute_with_mode");
        samples.push(t.elapsed());
        last_score = res.score;
    }
    (median(samples), last_score)
}

fn main() {
    let n = (W as usize) * (H as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();

    let modes = [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ];
    let blurs = [Ssim2Blur::Iir, Ssim2Blur::Fir];

    eprintln!(
        "ssim2 {W}x{H} ({:.1} MP) — warm-ref (compute_with_reference_with_mode):",
        (n as f64 / 3.0) / 1e6
    );
    eprintln!(
        "{:>6}  {:>10}  {:>11}  {:>10}  {:>14}",
        "blur", "mode", "median (ms)", "min (ms)", "score"
    );
    // Capture median per (blur, mode) for the speedup summary table.
    let mut warm_med_ms = [[0.0_f64; 4]; 2];
    let mut warm_scores = [[0.0_f64; 4]; 2];
    for (bi, &blur) in blurs.iter().enumerate() {
        for (mi, &mode) in modes.iter().enumerate() {
            let (med, min, score) = bench_warm(blur, mode, &r, &d);
            let med_ms = med.as_secs_f64() * 1000.0;
            let min_ms = min.as_secs_f64() * 1000.0;
            warm_med_ms[bi][mi] = med_ms;
            warm_scores[bi][mi] = score;
            eprintln!(
                "{:>6?}  {:>10?}  {:>11.3}  {:>10.3}  {:>14.6}",
                blur, mode, med_ms, min_ms, score
            );
        }
    }

    eprintln!();
    eprintln!(
        "ssim2 {W}x{H} ({:.1} MP) — cold (compute_with_mode):",
        (n as f64 / 3.0) / 1e6
    );
    eprintln!("{:>6}  {:>10}  {:>11}  {:>14}", "blur", "mode", "median (ms)", "score");
    let mut cold_med_ms = [[0.0_f64; 4]; 2];
    for (bi, &blur) in blurs.iter().enumerate() {
        for (mi, &mode) in modes.iter().enumerate() {
            let (med, score) = bench_cold(blur, mode, &r, &d);
            let med_ms = med.as_secs_f64() * 1000.0;
            cold_med_ms[bi][mi] = med_ms;
            eprintln!(
                "{:>6?}  {:>10?}  {:>11.3}  {:>14.6}",
                blur, mode, med_ms, score
            );
        }
    }

    eprintln!();
    eprintln!("Speedup summary (warm-ref median, all ratios vs IIR/Full):");
    let base_warm = warm_med_ms[0][0]; // IIR + Full
    let base_cold = cold_med_ms[0][0];
    eprintln!(
        "  {:>6}  {:>10}  {:>10}  {:>10}",
        "blur", "mode", "warm-ref", "cold"
    );
    for bi in 0..2 {
        for mi in 0..4 {
            let warm_ratio = base_warm / warm_med_ms[bi][mi];
            let cold_ratio = base_cold / cold_med_ms[bi][mi];
            eprintln!(
                "  {:>6?}  {:>10?}  {:>9.3}x  {:>9.3}x",
                blurs[bi], modes[mi], warm_ratio, cold_ratio
            );
        }
    }

    eprintln!();
    eprintln!("Score scale (FIR is a DISTINCT metric — different scale by design):");
    for bi in 0..2 {
        for mi in 0..4 {
            eprintln!(
                "  {:>6?} / {:>10?}: {:.6}",
                blurs[bi], modes[mi], warm_scores[bi][mi]
            );
        }
    }
}
