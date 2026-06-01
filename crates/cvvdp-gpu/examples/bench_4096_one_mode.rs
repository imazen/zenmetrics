//! Timing driver for head-to-head bench vs pycvvdp.
//!
//! Usage:
//!     bench_4096_one_mode <full|strip_pair> <size> <n_iters>
//!
//! Emits one JSON line on stdout for the Python wrapper to parse.

#![cfg(feature = "cuda")]

use std::hint::black_box;
use std::time::Instant;

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "../tests/common/mod.rs"]
mod common;

use common::Backend;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    common::synth_pair_with_offset_dist(w as usize, h as usize)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("full");
    let size: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4096);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_pair(size, size);

    let client = Backend::client(&Default::default());
    let mut cvvdp = match mode {
        "full" => Cvvdp::<Backend>::new(client, size, size, CvvdpParams::PLACEHOLDER)
            .expect("new Cvvdp Full"),
        "strip_pair" => Cvvdp::<Backend>::new_strip_pair(
            client,
            size,
            size,
            256, // h_body
            CvvdpParams::PLACEHOLDER,
        )
        .expect("new Cvvdp StripPair"),
        other => panic!("unknown mode: {other}; want full or strip_pair"),
    };

    // Warm-up — compiles cubecl kernels + first allocations + pool fill.
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up");

    let mut times = Vec::with_capacity(iters);
    let mut last_jod = 0.0f32;
    for _ in 0..iters {
        let t = Instant::now();
        let jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let dt = t.elapsed();
        last_jod = jod;
        times.push(dt.as_secs_f64());
        black_box(jod);
    }

    let secs_json: Vec<String> = times.iter().map(|t| format!("{:.6}", t)).collect();
    println!(
        "{{\"mode\":\"{}\",\"size\":{},\"jod\":{:.6},\"per_iter_seconds\":[{}]}}",
        mode,
        size,
        last_jod,
        secs_json.join(",")
    );
}
