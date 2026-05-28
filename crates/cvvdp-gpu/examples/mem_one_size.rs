//! GPU memory + timing measurement driver for cvvdp-gpu.
//! See `butteraugli-gpu/examples/mem_one_size.rs` for the protocol.
//!
//! Modes (covers task brief MemoryMode matrix):
//! - `full`           — `Cvvdp::new` (whole-image, cold-ref one-shot
//!                      `compute_dkl_jod`).
//! - `warm_ref`       — `Cvvdp::new` + `warm_reference` +
//!                      `compute_dkl_jod_with_warm_ref` (cached-ref fast
//!                      path; ~1.8× faster per-dist than `full`).
//! - `strip`          — `Cvvdp::new_strip` (Mode E REF cache) + cold-ref
//!                      `compute_dkl_jod` (still strip pipeline).
//! - `warm_ref_strip` — `Cvvdp::new_strip` + `warm_reference` +
//!                      `compute_dkl_jod_with_warm_ref` (Mode E cached
//!                      strip fast path).
//! - `strip_pair`     — `Cvvdp::new_strip_pair` (Mode B one-shot pair
//!                      strip walker; `compute_dkl_jod` cold-ref).
//! - `capped`         — `Cvvdp::new_capped_pyramid` (JOD-shifting capped
//!                      pyramid depth; default levels=5).
//! - `auto`           — `Cvvdp::new_with_memory_mode(MemoryMode::Auto)`
//!                      (Full when it fits the VRAM cap, else Strip).

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use std::io::Write;
use std::time::{Duration, Instant};

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::MemoryMode;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "../tests/common/mod.rs"]
mod common;

use common::Backend;

const CHILD_HOLD_MS: u64 = 400;
const DEFAULT_BODY: u32 = 256;
const DEFAULT_CAPPED_LEVELS: u32 = 5;

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
    let capped_levels = parse_u32("WORKER_CAPPED_LEVELS", DEFAULT_CAPPED_LEVELS);

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (r, d) = common::synth_pair_with_offset_dist(w as usize, h as usize);
    let (_r2, d2) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let client = Backend::client(&Default::default());
    let mut all_runs: Vec<f64> = Vec::with_capacity(1 + reps);
    let t_warm0 = Instant::now();

    let jod: f32 = match mode.as_str() {
        "full" => {
            let mut c = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
                .expect("Cvvdp::new");
            let j = c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c.compute_dkl_jod(&r, &d2, ppd).expect("compute_dkl_jod rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "warm_ref" => {
            let mut c = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
                .expect("Cvvdp::new");
            c.warm_reference(&r).expect("warm_reference");
            let j = c
                .compute_dkl_jod_with_warm_ref(&d, ppd)
                .expect("compute_dkl_jod_with_warm_ref");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod_with_warm_ref(&d2, ppd)
                    .expect("compute_dkl_jod_with_warm_ref rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "strip" => {
            let mut c =
                Cvvdp::<Backend>::new_strip(client.clone(), w, h, body, CvvdpParams::PLACEHOLDER)
                    .expect("Cvvdp::new_strip");
            let j = c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod (strip)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod(&r, &d2, ppd)
                    .expect("compute_dkl_jod (strip) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "warm_ref_strip" => {
            let mut c =
                Cvvdp::<Backend>::new_strip(client.clone(), w, h, body, CvvdpParams::PLACEHOLDER)
                    .expect("Cvvdp::new_strip");
            c.warm_reference(&r).expect("warm_reference (strip)");
            let j = c
                .compute_dkl_jod_with_warm_ref(&d, ppd)
                .expect("compute_dkl_jod_with_warm_ref (strip)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod_with_warm_ref(&d2, ppd)
                    .expect("compute_dkl_jod_with_warm_ref (strip) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "strip_pair" => {
            let mut c = Cvvdp::<Backend>::new_strip_pair(
                client.clone(),
                w,
                h,
                body,
                CvvdpParams::PLACEHOLDER,
            )
            .expect("Cvvdp::new_strip_pair");
            let j = c
                .compute_dkl_jod(&r, &d, ppd)
                .expect("compute_dkl_jod (strip_pair)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod(&r, &d2, ppd)
                    .expect("compute_dkl_jod (strip_pair) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "capped" => {
            let mut c = Cvvdp::<Backend>::new_capped_pyramid(
                client.clone(),
                w,
                h,
                CvvdpParams::PLACEHOLDER,
                capped_levels,
            )
            .expect("Cvvdp::new_capped_pyramid");
            let j = c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod (capped)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod(&r, &d2, ppd)
                    .expect("compute_dkl_jod (capped) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
        }
        "auto" => {
            let mut c = Cvvdp::<Backend>::new_with_memory_mode(
                client.clone(),
                w,
                h,
                CvvdpParams::PLACEHOLDER,
                MemoryMode::Auto,
            )
            .expect("Cvvdp::new_with_memory_mode(Auto)");
            let j = c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod (auto)");
            let warm = t_warm0.elapsed().as_secs_f64() * 1e3;
            all_runs.push(warm);
            for _ in 0..reps {
                let t = Instant::now();
                let _ = c
                    .compute_dkl_jod(&r, &d2, ppd)
                    .expect("compute_dkl_jod (auto) rep");
                all_runs.push(t.elapsed().as_secs_f64() * 1e3);
            }
            j
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
        "READY {jod:.6} warm_ms={:.3} wall_median_ms={:.3} warm_then_reps_ms={}",
        warm_ms,
        median_ms,
        fmt_ms_csv(&all_runs)
    );
    std::io::stdout().flush().expect("flush stdout");

    std::thread::sleep(Duration::from_millis(CHILD_HOLD_MS));
}
