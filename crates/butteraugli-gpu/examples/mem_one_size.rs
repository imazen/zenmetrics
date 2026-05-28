//! GPU memory + timing measurement driver for butteraugli-gpu.
//!
//! Subprocess-per-cell pattern: the parent orchestrator spawns this
//! binary once per (mode, width, height, backend) cell, samples
//! nvidia-smi memory.used in a brief hold window, and reports the
//! delta. The child holds the GPU buffers alive after signaling
//! READY so the parent can sample at quiescent steady state —
//! cubecl's memory pool caches buffers across Drop, so the only
//! way to see a clean baseline is to let the OS reclaim the process.
//!
//! Modes:
//! - `full`           — `Butteraugli::new` + `compute` (cold-ref, one-shot).
//! - `strip`          — `Butteraugli::new_strip` + `compute_strip`
//!                      (cold-ref, one-shot, strip pipeline).
//! - `warm_ref`       — `Butteraugli::new` + `set_reference` +
//!                      `compute_with_reference` (cached-ref hot path).
//! - `warm_ref_strip` — `Butteraugli::new_strip` + `set_reference` +
//!                      `compute_with_reference` (cached strip-mode path,
//!                      Mode E auto-dispatched in the pipeline).
//!
//! Environment:
//! - `WORKER_MODE`    — one of the modes above.
//! - `WORKER_W`       — image width in pixels (default 1024).
//! - `WORKER_H`       — image height in pixels (default 1024).
//! - `WORKER_REPS`    — how many compute calls to run after the first
//!                      warm-up (default 2; total runs = 1 + WORKER_REPS).
//! - `WORKER_BODY`    — strip body height (default 256).
//!
//! Output: one line on stdout
//!   `READY <score> warm_ms=<ms> wall_median_ms=<ms> warm_then_reps_ms=<csv>`
//! after the warm-up + repeats finish. Stays alive for CHILD_HOLD_MS
//! so the parent can poll nvidia-smi. Exit code 0 on success.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use butteraugli_gpu::Butteraugli;

const CHILD_HOLD_MS: u64 = 400;
const DEFAULT_BODY: u32 = 256;

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

fn fmt_ms_csv(t: &[f64]) -> String {
    t.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(",")
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
    let mode = std::env::var("WORKER_MODE").unwrap_or_else(|_| "full".into());
    let w = parse_u32("WORKER_W", 1024);
    let h = parse_u32("WORKER_H", 1024);
    let reps = parse_u32("WORKER_REPS", 2) as usize;
    let body = parse_u32("WORKER_BODY", DEFAULT_BODY);

    let r = synth_srgb(w, h, 42);
    let d = synth_srgb(w, h, 137);
    let d2 = synth_srgb(w, h, 9001);

    let client = Backend::client(&Default::default());

    let mut all_runs: Vec<f64> = Vec::with_capacity(1 + reps);
    let t_warm0 = Instant::now();
    let score: f32 = match mode.as_str() {
        "full" => {
            let mut b = Butteraugli::<Backend>::new(client.clone(), w, h);
            let res = b.compute(&r, &d).expect("compute");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = b.compute(&r, &d2).expect("compute");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "strip" => {
            let mut b = Butteraugli::<Backend>::new_strip(client.clone(), w, h, body);
            let res = b.compute_strip(&r, &d).expect("compute_strip");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = b.compute_strip(&r, &d2).expect("compute_strip");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "warm_ref" => {
            let mut b = Butteraugli::<Backend>::new(client.clone(), w, h);
            b.set_reference(&r).expect("set_reference");
            let res = b
                .compute_with_reference(&d)
                .expect("compute_with_reference");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = b
                    .compute_with_reference(&d2)
                    .expect("compute_with_reference rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "warm_ref_strip" => {
            let mut b = Butteraugli::<Backend>::new_strip(client.clone(), w, h, body);
            b.set_reference(&r).expect("set_reference (strip)");
            let res = b
                .compute_with_reference(&d)
                .expect("compute_with_reference (strip)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = b
                    .compute_with_reference(&d2)
                    .expect("compute_with_reference (strip) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        other => panic!("unknown WORKER_MODE: {other}"),
    };

    let warm_ms = all_runs[0];
    let post_warm: Vec<f64> = all_runs.iter().skip(1).copied().collect();
    let median_ms = if !post_warm.is_empty() {
        median(post_warm)
    } else {
        warm_ms
    };

    println!(
        "READY {score:.6} warm_ms={:.3} wall_median_ms={:.3} warm_then_reps_ms={}",
        warm_ms,
        median_ms,
        fmt_ms_csv(&all_runs)
    );
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
