//! GPU memory + timing measurement driver for iwssim-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.
//!
//! Modes for #57 native-RGB strip coverage:
//! - `full`               — `Iwssim::new` + `compute_with_reference` (gray,
//!                          set_reference + cached-ref). Same shape as the
//!                          pre-existing example (now reproducible).
//! - `strip`              — `Iwssim::new_strip` + `compute_gray_stripped`
//!                          (cold-ref strip, gray).
//! - `warm_ref`           — Full + cached-ref hot path (gray).
//! - `warm_ref_strip`     — Strip + cached-ref (gray, via
//!                          `set_reference_stripped` +
//!                          `compute_with_reference_stripped`).
//! - `rgb_full`           — RGB whole-image + cached-ref (`compute_rgb` is
//!                          one-shot only; cached-RGB hot path is via
//!                          `set_reference_srgb_u8` +
//!                          `compute_with_cached_reference_srgb_u8`).
//! - `rgb_strip`          — `Iwssim::new_strip` + `compute_rgb_stripped`
//!                          (cold-ref native RGB strip — #57 path).
//! - `rgb_warm_ref_strip` — Strip + `set_rgb_reference_stripped` +
//!                          `compute_rgb_with_reference_stripped` (cached
//!                          native-RGB strip — the #57 fast path).

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use iwssim_gpu::Iwssim;

const CHILD_HOLD_MS: u64 = 400;
const DEFAULT_BODY: u32 = 256;

fn synth_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    use std::num::Wrapping;
    let n = (w as usize) * (h as usize);
    let mut v = Vec::with_capacity(n);
    let mut s = Wrapping(seed);
    for _ in 0..n {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xff) as f32);
    }
    v
}

fn synth_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
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
    t.iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(",")
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

    let client = Backend::client(&Default::default());
    let mut all_runs: Vec<f64> = Vec::with_capacity(1 + reps);

    let t_warm0 = Instant::now();
    let score: f64 = match mode.as_str() {
        // Gray paths
        "full" | "warm_ref" => {
            let r = synth_gray(w, h, 42);
            let d = synth_gray(w, h, 137);
            let d2 = synth_gray(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new(client.clone(), w, h).expect("Iwssim::new");
            iw.set_reference(&r).expect("set_reference");
            let res = iw
                .compute_with_reference(&d)
                .expect("compute_with_reference");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw
                    .compute_with_reference(&d2)
                    .expect("compute_with_reference rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "strip" => {
            let r = synth_gray(w, h, 42);
            let d = synth_gray(w, h, 137);
            let d2 = synth_gray(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Iwssim::new_strip");
            let res = iw
                .compute_gray_stripped(&r, &d)
                .expect("compute_gray_stripped");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw
                    .compute_gray_stripped(&r, &d2)
                    .expect("compute_gray_stripped rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "warm_ref_strip" => {
            let r = synth_gray(w, h, 42);
            let d = synth_gray(w, h, 137);
            let d2 = synth_gray(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Iwssim::new_strip");
            iw.set_reference_stripped(&r)
                .expect("set_reference_stripped");
            let res = iw
                .compute_with_reference_stripped(&d)
                .expect("compute_with_reference_stripped");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw
                    .compute_with_reference_stripped(&d2)
                    .expect("compute_with_reference_stripped rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        // RGB native paths
        "rgb_full" => {
            let r = synth_rgb(w, h, 42);
            let d = synth_rgb(w, h, 137);
            let d2 = synth_rgb(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new(client.clone(), w, h).expect("Iwssim::new");
            let res = iw.compute_rgb(&r, &d).expect("compute_rgb");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw.compute_rgb(&r, &d2).expect("compute_rgb rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "rgb_strip" => {
            let r = synth_rgb(w, h, 42);
            let d = synth_rgb(w, h, 137);
            let d2 = synth_rgb(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Iwssim::new_strip");
            let res = iw
                .compute_rgb_stripped(&r, &d)
                .expect("compute_rgb_stripped");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw
                    .compute_rgb_stripped(&r, &d2)
                    .expect("compute_rgb_stripped rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            res.score
        }
        "rgb_warm_ref_strip" => {
            let r = synth_rgb(w, h, 42);
            let d = synth_rgb(w, h, 137);
            let d2 = synth_rgb(w, h, 9001);
            let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Iwssim::new_strip");
            iw.set_rgb_reference_stripped(&r)
                .expect("set_rgb_reference_stripped");
            let res = iw
                .compute_rgb_with_reference_stripped(&d)
                .expect("compute_rgb_with_reference_stripped");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = iw
                    .compute_rgb_with_reference_stripped(&d2)
                    .expect("compute_rgb_with_reference_stripped rep");
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
