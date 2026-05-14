//! Quick one-shot timer for `Cvvdp::compute_dkl_jod` at 12 MP.
//!
//! Sidesteps zenbench's iteration-count calibration which gets
//! pathologically slow when each iter takes seconds — runs a
//! fixed number of iterations and reports the per-call timing.
//!
//! Run with:
//!     cargo run --release --example time_12mp -p cvvdp-gpu --features cuda

#![cfg(feature = "cuda")]

use std::hint::black_box;
use std::time::Instant;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

const W: u32 = 4000;
const H: u32 = 3000;
const ITERS: usize = 5;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let mut ref_b = vec![0u8; n];
    let mut dis_b = vec![0u8; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let b = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w as usize + x) * 3;
            ref_b[i] = r;
            ref_b[i + 1] = g;
            ref_b[i + 2] = b;
            dis_b[i] = r.saturating_sub(8);
            dis_b[i + 1] = g.saturating_sub(4);
            dis_b[i + 2] = b.saturating_add(12);
        }
    }
    (ref_b, dis_b)
}

fn main() {
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_bytes, dist_bytes) = synth_pair(W, H);

    let client = CudaRuntime::client(&Default::default());
    let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, W, H, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on cuda");

    // Warm-up (compiles cubecl kernels + first allocations).
    let t0 = Instant::now();
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up");
    eprintln!("warm-up: {:?}", t0.elapsed());

    let mut times = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        let t = Instant::now();
        let jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let dt = t.elapsed();
        black_box(jod);
        eprintln!("iter {i}: {dt:?}");
        times.push(dt);
    }
    times.sort();
    let median = times[times.len() / 2];
    let total_pixels = (W as u64) * (H as u64);
    let ns_per_px = median.as_nanos() as f64 / total_pixels as f64;
    println!("\n12 MP ({W}×{H}) GPU compute_dkl_jod:");
    println!("  median: {median:?}");
    println!("  per-pixel: {ns_per_px:.1} ns/px");
}
