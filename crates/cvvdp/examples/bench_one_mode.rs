//! Timing driver for head-to-head bench vs pycvvdp and cvvdp-gpu.
//!
//! Usage:
//!     bench_one_mode <size> <n_iters>
//!
//! Emits one JSON line on stdout for the Python wrapper to parse.
//! Mirrors `cvvdp-gpu/examples/bench_4096_one_mode.rs` so timings
//! are directly comparable.

use std::hint::black_box;
use std::time::Instant;

use cvvdp::{Cvvdp, CvvdpParams};

fn synth_pair(size: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (size as usize) * (size as usize) * 3;
    let mut rng_state: u64 = 0xC0FFEE;
    let mut ref_bytes = vec![0u8; n];
    for b in ref_bytes.iter_mut() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *b = ((rng_state >> 33) & 0xFF) as u8;
    }
    let w = size as usize;
    let h = size as usize;
    let (dx, dy) = (3usize, 2usize);
    let mut dist_bytes = ref_bytes.clone();
    for y in dy..h {
        for x in dx..w {
            let src = ((y - dy) * w + (x - dx)) * 3;
            let dst = (y * w + x) * 3;
            dist_bytes[dst] = ref_bytes[src];
            dist_bytes[dst + 1] = ref_bytes[src + 1];
            dist_bytes[dst + 2] = ref_bytes[src + 2];
        }
    }
    (ref_bytes, dist_bytes)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let size: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let iters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    let (ref_bytes, dist_bytes) = synth_pair(size);

    let mut cvvdp = Cvvdp::new(size, size, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new");

    // Warm-up (first call pays kernel-cache fills, allocation, etc.).
    let _ = cvvdp.score(&ref_bytes, &dist_bytes).expect("warm-up");

    let mut times = Vec::with_capacity(iters);
    let mut last_jod = 0.0f32;
    for _ in 0..iters {
        let t = Instant::now();
        let jod = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        let dt = t.elapsed();
        last_jod = jod;
        times.push(dt.as_secs_f64());
        black_box(jod);
    }

    let secs_json: Vec<String> = times.iter().map(|t| format!("{:.6}", t)).collect();
    println!(
        "{{\"crate\":\"cvvdp\",\"size\":{},\"jod\":{:.6},\"per_iter_seconds\":[{}]}}",
        size,
        last_jod,
        secs_json.join(",")
    );
}
