//! Wall-clock benchmark for the IW-SSIM GPU pipeline.
//!
//! Build + run:
//! ```bash
//! cargo run --release -p iwssim-gpu --example bench --no-default-features --features cuda
//! ```
//!
//! Output: per image size, mean time per pair, throughput in MP/s,
//! and the score (kept side by side so a perf-regressing change
//! that happens to change scores is immediately visible).

use std::time::Instant;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::prelude::*;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
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

fn bench_size(w: u32, h: u32) {
    let ref_gray = make_gray(w, h, 42);
    let dis_gray = make_gray(w, h, 137);
    let n_warmup = 3;
    let n_measure = 12;

    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new(client, w, h).unwrap();

    for _ in 0..n_warmup {
        let _ = iw.compute_gray(&ref_gray, &dis_gray).unwrap();
    }

    use cubecl::Runtime;
    let client = Backend::client(&Default::default());
    let mut last_score = 0.0_f64;
    let mut min_dt = f64::INFINITY;
    cubecl::future::block_on(client.sync()).expect("client.sync");
    let t = Instant::now();
    for _ in 0..n_measure {
        let t_iter = Instant::now();
        last_score = iw.compute_gray(&ref_gray, &dis_gray).unwrap().score;
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let dt_iter = t_iter.elapsed().as_secs_f64();
        if dt_iter < min_dt {
            min_dt = dt_iter;
        }
    }
    let dt = t.elapsed().as_secs_f64() / n_measure as f64;

    // ----- Cached-reference path -----
    iw.set_reference(&ref_gray).unwrap();
    for _ in 0..n_warmup {
        let _ = iw.compute_with_reference(&dis_gray).unwrap();
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");
    let mut min_dt_cwr = f64::INFINITY;
    let t = Instant::now();
    for _ in 0..n_measure {
        let t_iter = Instant::now();
        let _ = iw.compute_with_reference(&dis_gray).unwrap();
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let dt_iter = t_iter.elapsed().as_secs_f64();
        if dt_iter < min_dt_cwr {
            min_dt_cwr = dt_iter;
        }
    }
    let dt_cwr = t.elapsed().as_secs_f64() / n_measure as f64;

    let mp = (w as f64 * h as f64) / 1e6;
    println!(
        "{:>5}x{:<5}  full {:>6.2} ms (min {:>6.2})  cwr {:>6.2} ms (min {:>6.2})  cwr/full = {:.2}×  {:>6.1} MP/s (cwr)  score={:.6}",
        w,
        h,
        dt * 1e3,
        min_dt * 1e3,
        dt_cwr * 1e3,
        min_dt_cwr * 1e3,
        dt / dt_cwr,
        mp / dt_cwr,
        last_score,
    );
}

fn main() {
    println!("iwssim-gpu bench");
    for (w, h) in [
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ] {
        bench_size(w, h);
    }
}
