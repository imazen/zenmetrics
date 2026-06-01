//! GPU memory + timing measurement driver for zensim-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.
//!
//! `Zensim::compute_features` is the one-shot path; `set_reference` +
//! `compute_with_reference` is the cached-ref hot path used by encoders.
//! `new_strip` enables the Mode E strip walker — `compute_with_reference`
//! auto-dispatches to `compute_with_reference_vec_strip` internally.
//!
//! Modes (#49 strip support, #75 Mode E refinement):
//! - `full`           — `Zensim::new` + `compute_features` (cold-ref).
//! - `warm_ref`       — `Zensim::new` + `set_reference` +
//!                      `compute_with_reference` (cached, full-image).
//! - `strip`          — `Zensim::new_strip` + `compute_features` cold-ref
//!                      one-shot path (still strip pipeline internally).
//! - `warm_ref_strip` — `Zensim::new_strip` + `set_reference` +
//!                      `compute_with_reference` (cached, Mode E strip
//!                      walker dispatched by the pipeline).

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use zensim_gpu::Zensim;

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

    let r = synth_srgb(w, h, 42);
    let d = synth_srgb(w, h, 137);
    let d2 = synth_srgb(w, h, 9001);

    let client = Backend::client(&Default::default());
    let mut all_runs: Vec<f64> = Vec::with_capacity(1 + reps);

    let t_warm0 = Instant::now();
    // Report a single representative score: feat[0] of the first run.
    let score: f64 = match mode.as_str() {
        "full" => {
            let mut z = Zensim::<Backend>::new(client.clone(), w, h).expect("Zensim::new");
            let feat = z.compute_features(&r, &d).expect("compute_features");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = z.compute_features(&r, &d2).expect("compute_features rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            feat[0]
        }
        "warm_ref" => {
            let mut z = Zensim::<Backend>::new(client.clone(), w, h).expect("Zensim::new");
            z.set_reference(&r).expect("set_reference");
            let feat = z
                .compute_with_reference(&d)
                .expect("compute_with_reference");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = z
                    .compute_with_reference(&d2)
                    .expect("compute_with_reference rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            feat[0]
        }
        "strip" => {
            let mut z = Zensim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Zensim::new_strip");
            let feat = z
                .compute_features(&r, &d)
                .expect("compute_features (strip)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = z
                    .compute_features(&r, &d2)
                    .expect("compute_features (strip) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            feat[0]
        }
        "warm_ref_strip" => {
            let mut z = Zensim::<Backend>::new_strip(client.clone(), w, h, body)
                .expect("Zensim::new_strip");
            z.set_reference(&r).expect("set_reference (strip)");
            let feat = z
                .compute_with_reference(&d)
                .expect("compute_with_reference (strip)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = z
                    .compute_with_reference(&d2)
                    .expect("compute_with_reference (strip) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            feat[0]
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
