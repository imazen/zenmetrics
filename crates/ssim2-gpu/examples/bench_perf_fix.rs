//! Bench harness for the launch-fusion perf fix work tracked in
//! `docs/SSIM2_FIX_ASSESSMENT.md`. Reports first-call wall and
//! steady-state median wall at 1024² / 2048² / 4096² in
//! `Ssim2Mode::Full` (the worst case — no skip-map cells).
//!
//! Use:
//!   cargo run --release -p ssim2-gpu --features cuda \
//!       --example bench_perf_fix
//!
//! Captures the two numbers the perf-fix work cares about:
//! 1. **first-call wall** — includes kernel PTX compilation and
//!    cubecl autotune. Dominated by compile time; comparable
//!    only between commits with the same kernel set.
//! 2. **steady-state wall p50** — 8 iters after a 3-iter warmup,
//!    median of the 8. This is the production "compute_with_reference"
//!    cost per call.

#![cfg(feature = "cuda")]

use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime as Backend;
use ssim2_gpu::{Ssim2, Ssim2Mode};

const SIZES: &[u32] = &[1024, 2048, 4096];
const WARMUP: usize = 3;
const ITERS: usize = 8;

fn synth_srgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize) * 3;
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed);
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as u8);
    }
    v
}

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) / 2
    } else {
        xs[mid]
    }
}

fn main() {
    println!("size  first_call_ms  steady_p50_ms  steady_min_ms  score");
    for &sz in SIZES {
        let w = sz;
        let h = sz;
        let r = synth_srgb(w, h, 42);
        let d = synth_srgb(w, h, 137);
        let client = Backend::client(&Default::default());
        let mut s = Ssim2::<Backend>::new(client, w, h).expect("Ssim2::new");
        s.set_reference(&r).expect("set_reference");

        // First call — includes PTX compile.
        let t = Instant::now();
        let first = s
            .compute_with_reference_with_mode(Ssim2Mode::Full, &d)
            .expect("first");
        let first_dt = t.elapsed();
        let mut last_score = first.score;

        // Warmup: drop transient JIT / autotune effects.
        for _ in 0..WARMUP {
            let res = s
                .compute_with_reference_with_mode(Ssim2Mode::Full, &d)
                .expect("warm");
            last_score = res.score;
        }

        let mut samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t = Instant::now();
            let res = s
                .compute_with_reference_with_mode(Ssim2Mode::Full, &d)
                .expect("iter");
            samples.push(t.elapsed());
            last_score = res.score;
        }
        samples.sort();
        let p50 = median(samples.clone());
        let min = samples[0];
        println!(
            "{:4}  {:13.2}  {:13.3}  {:13.3}  {:.6}",
            sz,
            first_dt.as_secs_f64() * 1e3,
            p50.as_secs_f64() * 1e3,
            min.as_secs_f64() * 1e3,
            last_score
        );
    }
}
