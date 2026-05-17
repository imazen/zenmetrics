//! ssim2-specific bench so multi-agent example builds don't overwrite
//! each other (a `bench_t4_warm` exists in every GPU metric crate; the
//! last `cargo build --example bench_t4_warm` writes to the same path).
//! Use this file when running ssim2-gpu performance comparisons.
//!
//! Reports steady-state per-call ms at 12 MP under each [`Ssim2Mode`].
//! Drop the first iteration (transient kernel-compile / cache settle);
//! report the median of the remaining iterations.

use std::time::{Duration, Instant};
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::{Ssim2, Ssim2Mode};

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

fn bench_mode(s: &mut Ssim2<CudaRuntime>, mode: Ssim2Mode, r: &[u8], d: &[u8]) -> (Duration, f64) {
    // Warmup.
    let mut last_score = 0.0_f64;
    for _ in 0..WARMUP {
        let res = s.compute_with_mode(mode, r, d).expect("warmup");
        last_score = res.score;
    }
    // Measure.
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let res = s.compute_with_mode(mode, r, d).expect("compute");
        samples.push(t.elapsed());
        last_score = res.score;
    }
    (median(samples), last_score)
}

fn main() {
    let w: u32 = 4000;
    let h: u32 = 3000;
    let n = (w as usize) * (h as usize) * 3;
    let r: Vec<u8> = (0..n).map(|i| ((i * 17 + 5) & 0xff) as u8).collect();
    let d: Vec<u8> = (0..n).map(|i| ((i * 23 + 11) & 0xff) as u8).collect();
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("new");

    eprintln!("ssim2 12 MP (4000x3000) steady-state — median of {ITERS} (after {WARMUP} warmup):");
    eprintln!("{:>10}  {:>14}  {:>10}  {:>10}", "mode", "score", "median (ms)", "vs Full");
    let mut full_ms = None;
    for mode in [
        Ssim2Mode::Full,
        Ssim2Mode::Lossless,
        Ssim2Mode::Fast,
        Ssim2Mode::Faster,
    ] {
        let (med, score) = bench_mode(&mut s, mode, &r, &d);
        let med_ms = med.as_secs_f64() * 1000.0;
        let ratio = match full_ms {
            None => {
                full_ms = Some(med_ms);
                "1.000×".to_string()
            }
            Some(fm) => format!("{:.3}×", fm / med_ms),
        };
        eprintln!(
            "{:>10?}  {:>14.6}  {:>10.3}  {:>10}",
            mode, score, med_ms, ratio
        );
    }
}
