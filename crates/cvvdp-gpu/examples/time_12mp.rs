//! Quick one-shot timer for `Cvvdp::compute_dkl_jod` at 12 MP.
//!
//! Sidesteps zenbench's iteration-count calibration which gets
//! pathologically slow when each iter takes seconds — runs a
//! fixed number of iterations and reports the per-call timing.
//!
//! Run with:
//!     cargo run --release --example time_12mp -p cvvdp-gpu --features cuda
//!
//! Falls back to wgpu / hip when cuda isn't compiled in:
//!     cargo run --release --example time_12mp -p cvvdp-gpu --no-default-features --features wgpu
//!     cargo run --release --example time_12mp -p cvvdp-gpu --no-default-features --features hip

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use std::hint::black_box;
use std::time::Instant;

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::hip::HipRuntime;

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

    let client = Backend::client(&Default::default());
    let mut cvvdp = Cvvdp::<Backend>::new(client, W, H, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on GPU backend");

    // Warm-up (compiles cubecl kernels + first allocations).
    let t0 = Instant::now();
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up");
    eprintln!("warm-up: {:?}", t0.elapsed());

    // Diagnostic: do two consecutive weber calls on the SAME data
    // to test whether the "weber(dist) is 2× weber(ref)" slowdown
    // observed inside compute_dkl_d_bands is data-specific or
    // purely position-dependent (consecutive-call overhead).
    eprintln!("---- consecutive-weber diagnostic ----");
    for i in 0..3 {
        let t = Instant::now();
        let (w_b, w_l) = cvvdp
            .compute_dkl_weber_pyramid(&ref_bytes)
            .expect("weber call 1");
        let dt1 = t.elapsed();
        black_box((w_b, w_l));

        let t = Instant::now();
        let (w_b, w_l) = cvvdp
            .compute_dkl_weber_pyramid(&ref_bytes)
            .expect("weber call 2");
        let dt2 = t.elapsed();
        black_box((w_b, w_l));

        eprintln!("run {i}: weber#1={dt1:?}  weber#2={dt2:?}");
    }
    eprintln!("---- end diagnostic ----\n");

    let mut jod_times = Vec::with_capacity(ITERS);
    let mut d_bands_times = Vec::with_capacity(ITERS);
    let mut weber_times = Vec::with_capacity(ITERS);
    for i in 0..ITERS {
        // Phase 1: weber pyramid only (one side).
        let t = Instant::now();
        let (w_b, w_l) = cvvdp
            .compute_dkl_weber_pyramid(&ref_bytes)
            .expect("weber");
        let dt_weber = t.elapsed();
        black_box((w_b, w_l));

        // Phase 2: full D-bands (color + 2× weber + CSF + masking + readback).
        let t = Instant::now();
        let d = cvvdp
            .compute_dkl_d_bands(&ref_bytes, &dist_bytes, ppd)
            .expect("d_bands");
        let dt_d = t.elapsed();
        black_box(d);

        // Phase 3: full JOD (D bands + host pool + Minkowski + met2jod).
        let t = Instant::now();
        let jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let dt_jod = t.elapsed();
        black_box(jod);

        eprintln!("iter {i}: weber={dt_weber:?} d_bands={dt_d:?} jod={dt_jod:?}");
        weber_times.push(dt_weber);
        d_bands_times.push(dt_d);
        jod_times.push(dt_jod);
    }
    weber_times.sort();
    d_bands_times.sort();
    jod_times.sort();
    let total_pixels = (W as u64) * (H as u64);
    let mid = ITERS / 2;
    println!("\n12 MP ({W}×{H}) per-phase medians:");
    let w = weber_times[mid];
    let d = d_bands_times[mid];
    let j = jod_times[mid];
    println!(
        "  weber_pyramid (1 side):  {w:?}  → {:.1} ns/px",
        w.as_nanos() as f64 / total_pixels as f64
    );
    println!(
        "  d_bands (full GPU):       {d:?}  → {:.1} ns/px",
        d.as_nanos() as f64 / total_pixels as f64
    );
    println!(
        "  jod (full + host pool):   {j:?}  → {:.1} ns/px",
        j.as_nanos() as f64 / total_pixels as f64
    );
    println!("  jod - d_bands (host pool): {:?}", j.saturating_sub(d));
    println!(
        "  d_bands - 2*weber (CSF+mask+IO): {:?}",
        d.saturating_sub(w * 2)
    );
}
