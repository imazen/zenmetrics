//! Per-stage profiler. Inserts forced syncs between pipeline phases so
//! the kernel time of each phase is attributable. Use this only for
//! perf-tuning — the syncs themselves add a few microseconds per
//! phase, so absolute numbers are slightly inflated.

use std::time::Instant;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::prelude::*;
use iwssim_gpu::Iwssim;

fn make_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity((w * h) as usize);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xFF) as f32);
    }
    v
}

fn main() {
    let w = 1024_u32;
    let h = 1024_u32;
    let ref_gray = make_gray(w, h, 42);
    let dis_gray = make_gray(w, h, 137);
    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new(client.clone(), w, h).unwrap();

    // Warm up.
    for _ in 0..4 {
        let _ = iw.compute_gray(&ref_gray, &dis_gray).unwrap();
    }

    // Total wall.
    client.sync();
    let t = Instant::now();
    for _ in 0..16 {
        let _ = iw.compute_gray(&ref_gray, &dis_gray).unwrap();
    }
    client.sync();
    println!(
        "total mean over 16 iters @ {w}x{h}: {:.3} ms/pair",
        t.elapsed().as_secs_f64() / 16.0 * 1e3
    );
}
