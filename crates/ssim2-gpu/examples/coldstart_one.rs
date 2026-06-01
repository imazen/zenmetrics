//! GPU cold-start measurement driver for ssim2-gpu (task #140).
//!
//! Times the fixed one-shot overhead a fresh process pays before any
//! per-pixel work: CUDA context init (`Backend::client`) + kernel
//! JIT/PTX load + first upload + first compute + readback. Then warm
//! repeats in the same process. The cold timer starts BEFORE
//! `Backend::client()` so context init is captured. `compute` returns
//! the score to host (readback), forcing a GPU sync.
//!
//! Env: WORKER_W / WORKER_H / WORKER_REPS.
//! Output:
//!   READY <score> client_ms=<f> new_ms=<f> first_compute_ms=<f> \
//!         cold_total_ms=<f> warm_median_ms=<f> warm_all_ms=<csv>

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::io::Write;
use std::time::Instant;

use cubecl::Runtime;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use ssim2_gpu::Ssim2;

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

fn parse_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(mut t: Vec<f64>) -> f64 {
    if t.is_empty() {
        return f64::NAN;
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = t.len();
    if n % 2 == 0 {
        (t[n / 2 - 1] + t[n / 2]) / 2.0
    } else {
        t[n / 2]
    }
}

fn main() {
    let w = parse_u32("WORKER_W", 1024);
    let h = parse_u32("WORKER_H", 1024);
    let reps = parse_u32("WORKER_REPS", 10) as usize;

    let r = synth_srgb(w, h, 42);
    let d = synth_srgb(w, h, 137);
    let d2 = synth_srgb(w, h, 9001);

    let t = Instant::now();
    let client = Backend::client(&Default::default());
    let client_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut s = Ssim2::<Backend>::new(client.clone(), w, h).expect("Ssim2::new");
    let new_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let res = s.compute(&r, &d).expect("compute (cold)");
    let first_compute_ms = t.elapsed().as_secs_f64() * 1e3;
    let score = res.score;

    let cold_total_ms = client_ms + new_ms + first_compute_ms;

    let mut warm: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = s.compute(&r, &d2).expect("compute (warm)");
        warm.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let warm_median_ms = median(warm.clone());
    let warm_csv = warm
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "READY {score:.6} client_ms={client_ms:.3} new_ms={new_ms:.3} \
         first_compute_ms={first_compute_ms:.3} cold_total_ms={cold_total_ms:.3} \
         warm_median_ms={warm_median_ms:.3} warm_all_ms={warm_csv}"
    );
    std::io::stdout().flush().expect("flush");
}
