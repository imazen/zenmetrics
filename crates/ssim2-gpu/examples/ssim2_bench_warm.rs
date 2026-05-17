//! ssim2-specific bench so multi-agent example builds don't overwrite
//! each other (a `bench_t4_warm` exists in every GPU metric crate; the
//! last `cargo build --example bench_t4_warm` writes to the same path).
//! Use this file when running ssim2-gpu performance comparisons.
//!
//! Reports timings for BOTH blur paths at 12 MP:
//! - `Ssim2Blur::Iir` — the default Charalampidis recursive Gaussian.
//! - `Ssim2Blur::Fir` — the opt-in 5-tap separable Gaussian per
//!   Kanetaka et al. IWAIT 2026.
//!
//! Per the task's stopping criterion, the FIR must be ≥ 1.2× faster
//! than the IIR at 12 MP for the opt-in to be worth the maintenance
//! burden. The summary at the end prints the speedup ratio.

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::{Ssim2, Ssim2Blur};
use std::time::{Duration, Instant};

const W: u32 = 4000;
const H: u32 = 3000;
const WARMUP: usize = 2;
const ITERS: usize = 10;

fn bench_blur(label: &str, blur: Ssim2Blur, r: &[u8], d: &[u8]) -> (Duration, Duration, f64) {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, W, H)
        .expect("Ssim2::new")
        .with_blur(blur);

    // warm-ref path: cache reference once, compute against the cached
    // ref each iteration.
    s.set_reference(r).expect("set_reference");

    for _ in 0..WARMUP {
        let _ = s.compute_with_reference(d).expect("warmup");
    }

    let mut samples: Vec<Duration> = Vec::with_capacity(ITERS);
    let mut last_score = 0.0_f64;
    for _ in 0..ITERS {
        let t = Instant::now();
        let res = s.compute_with_reference(d).expect("compute_with_reference");
        samples.push(t.elapsed());
        last_score = res.score;
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    let min = samples[0];
    eprintln!("  {label}: median={median:?}  min={min:?}  score={last_score:.6}");
    (median, min, last_score)
}

fn bench_blur_cold(label: &str, blur: Ssim2Blur, r: &[u8], d: &[u8]) -> Duration {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, W, H)
        .expect("Ssim2::new")
        .with_blur(blur);

    for _ in 0..WARMUP {
        let _ = s.compute(r, d).expect("warmup");
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _res = s.compute(r, d).expect("compute");
        samples.push(t.elapsed());
    }
    samples.sort();
    let median = samples[samples.len() / 2];
    eprintln!("  {label}: median={median:?}", );
    median
}

fn main() {
    let n = (W as usize) * (H as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();

    eprintln!("ssim2 {W}×{H} ({:.1} MP) — warm-ref (compute_with_reference):", (n as f64 / 3.0) / 1e6);
    let (iir_med, iir_min, iir_score) = bench_blur("Iir ", Ssim2Blur::Iir, &r, &d);
    let (fir_med, fir_min, fir_score) = bench_blur("Fir ", Ssim2Blur::Fir, &r, &d);

    eprintln!();
    eprintln!("ssim2 {W}×{H} ({:.1} MP) — cold (compute):", (n as f64 / 3.0) / 1e6);
    let iir_cold = bench_blur_cold("Iir ", Ssim2Blur::Iir, &r, &d);
    let fir_cold = bench_blur_cold("Fir ", Ssim2Blur::Fir, &r, &d);

    eprintln!();
    let warm_med_ratio = iir_med.as_secs_f64() / fir_med.as_secs_f64();
    let warm_min_ratio = iir_min.as_secs_f64() / fir_min.as_secs_f64();
    let cold_ratio = iir_cold.as_secs_f64() / fir_cold.as_secs_f64();
    eprintln!("Summary (FIR speedup vs IIR — higher is better, target ≥ 1.20×):");
    eprintln!("  warm-ref median: {warm_med_ratio:.3}×");
    eprintln!("  warm-ref min:    {warm_min_ratio:.3}×");
    eprintln!("  cold median:     {cold_ratio:.3}×");
    eprintln!();
    eprintln!("Score values (different scale by design — see Kanetaka et al. IWAIT 2026):");
    eprintln!("  IIR last sample: {iir_score:.6}");
    eprintln!("  FIR last sample: {fir_score:.6}");
}
