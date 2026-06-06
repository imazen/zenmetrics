//! A/B timing for the f64->f32 GPU-reduction change (zenmetrics#20).
//!
//! Measures total `score_from_linear_planes_with_diffmap` wall time across sizes.
//! The change converts the finals readback from f64 (8 B/elem) to f32 (4 B/elem)
//! plus a host f32->f64 widen, so any "host gpu sync" delta shows up here — and
//! is largest at SMALL sizes, where the per-score GPU compute is least and the
//! fixed readback/sync is the biggest fraction of total.
//!
//! Run on the f32 branch and on master, same machine/backend, and diff the rows.
//!   cargo run --release -p zensim-gpu --no-default-features --features cuda \
//!     --example f64_f32_readback_timing

use cubecl::Runtime;
use std::time::Instant;
use zensim_gpu::Zensim;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn srgb_to_lin(b: u8) -> f32 {
    let v = b as f32 / 255.0;
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// Deterministic LCG image -> 3 linear planes.
fn planes(seed: u32, n: usize) -> [Vec<f32>; 3] {
    let mut s = seed.wrapping_add(1);
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 24) & 0xff) as u8
    };
    let r: Vec<f32> = (0..n).map(|_| srgb_to_lin(next())).collect();
    let g: Vec<f32> = (0..n).map(|_| srgb_to_lin(next())).collect();
    let b: Vec<f32> = (0..n).map(|_| srgb_to_lin(next())).collect();
    [r, g, b]
}

fn main() {
    let backend = std::any::type_name::<Backend>();
    println!("f64/f32 readback timing — backend = {backend}");
    println!("  size        n_iter   avg_ms   (total score incl. finals readback)");

    for &(w, h, iters) in &[
        (64usize, 64usize, 400u32),
        (128, 128, 300),
        (256, 256, 200),
        (512, 512, 100),
        (1024, 1024, 40),
    ] {
        let n = w * h;
        let [rr, rg, rb] = planes(1, n);
        let [dr, dg, db] = planes(2, n);
        let client = Backend::client(&Default::default());
        let mut gpu = match Zensim::<Backend>::new(client, w as u32, h as u32) {
            Ok(z) => z,
            Err(e) => {
                println!("  {w}x{h}: Zensim::new failed: {e:?}");
                continue;
            }
        };
        let mut dm = Vec::new();
        // Warm up (JIT compile + allocations + first sync).
        for _ in 0..10 {
            let _ = gpu.score_from_linear_planes_with_diffmap(&rr, &rg, &rb, &dr, &dg, &db, &mut dm);
        }
        let t = Instant::now();
        for _ in 0..iters {
            let _ = gpu
                .score_from_linear_planes_with_diffmap(&rr, &rg, &rb, &dr, &dg, &db, &mut dm)
                .expect("score");
        }
        let avg_ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  {w:>4}x{h:<4}  {iters:>6}   {avg_ms:8.4}");
    }
}
