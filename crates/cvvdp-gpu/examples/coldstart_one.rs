//! GPU cold-start measurement driver for cvvdp-gpu (task #140).
//!
//! Measures the fixed one-shot overhead a fresh process pays before any
//! per-pixel work: CUDA context init (`Backend::client`) + kernel
//! JIT/PTX load + first host→device upload + first compute + readback.
//! Then runs warm repeats in the SAME process for steady-state per-call
//! wall. The cold timer starts BEFORE `Backend::client()` so the
//! context-init component is captured.
//!
//! `compute_dkl_jod` returns the JOD scalar to host (a `client.read_one`
//! readback inside the pipeline), forcing a GPU sync — the wall is real.
//!
//! Environment: WORKER_W / WORKER_H / WORKER_REPS.
//! Output:
//!   READY <jod> client_ms=<f> new_ms=<f> first_compute_ms=<f> \
//!         cold_total_ms=<f> warm_median_ms=<f> warm_all_ms=<csv>

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use std::io::Write;
use std::time::Instant;

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "../tests/common/mod.rs"]
mod common;
use common::Backend;

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

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (r, d) = common::synth_pair_with_offset_dist(w as usize, h as usize);
    let (_r2, d2) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let t = Instant::now();
    let client = Backend::client(&Default::default());
    let client_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let mut c =
        Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");
    let new_ms = t.elapsed().as_secs_f64() * 1e3;

    let t = Instant::now();
    let jod = c
        .compute_dkl_jod(&r, &d, ppd)
        .expect("compute_dkl_jod (cold)");
    let first_compute_ms = t.elapsed().as_secs_f64() * 1e3;

    let cold_total_ms = client_ms + new_ms + first_compute_ms;

    let mut warm: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let _ = c
            .compute_dkl_jod(&r, &d2, ppd)
            .expect("compute_dkl_jod (warm)");
        warm.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let warm_median_ms = median(warm.clone());
    let warm_csv = warm
        .iter()
        .map(|v| format!("{v:.3}"))
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "READY {jod:.6} client_ms={client_ms:.3} new_ms={new_ms:.3} \
         first_compute_ms={first_compute_ms:.3} cold_total_ms={cold_total_ms:.3} \
         warm_median_ms={warm_median_ms:.3} warm_all_ms={warm_csv}"
    );
    std::io::stdout().flush().expect("flush");
}
