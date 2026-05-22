//! CUDA perf gate for the Phase 1 plane-aliasing change in
//! `pipeline.rs` (2026-05-22).
//!
//! Measures `Ssim2::compute` (pair path) and `Ssim2::compute_with_reference`
//! (cached-ref path) at 1MP, 4MP, 12MP. Reports median ms / call after
//! 3 warmup iterations. Output is CSV-friendly so master vs Phase 1
//! runs land in `benchmarks/ssim2_aliasing_perf_<date>.csv`.
//!
//! Build:
//! ```sh
//! cargo build --release -p ssim2-gpu \
//!     --no-default-features --features cuda,fast-reduction,cubecl-types,pixels \
//!     --example bench_pair_vs_cached_cuda
//! ```
//!
//! Run:
//! ```sh
//! cargo run --release -p ssim2-gpu \
//!     --no-default-features --features cuda,fast-reduction,cubecl-types,pixels \
//!     --example bench_pair_vs_cached_cuda
//! ```

use std::time::{Duration, Instant};

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use ssim2_gpu::Ssim2;

const ITERS: usize = 10;
const WARMUP: usize = 3;

/// Sizes per the perf-gate spec (1 MP, 4 MP, 12 MP). The 24 MP target
/// would need ~7.3 GB GPU after Phase 1 and ~10.4 GB before — RTX 5070
/// (12 GB) handles it but it's slow enough to balloon CI time. The
/// 1/4/12 MP triple covers the linear regression cleanly.
const SIZES: &[(u32, u32, &str)] = &[
    (1024, 1024, "1MP"),
    (2048, 2048, "4MP"),
    (4000, 3000, "12MP"),
];

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    let mid = xs.len() / 2;
    if xs.len() % 2 == 0 {
        (xs[mid - 1] + xs[mid]) / 2
    } else {
        xs[mid]
    }
}

fn synthetic_pair(width: usize, height: usize, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 220 / width.max(1)) & 0xff) as u8;
            let g = ((y * 220 / height.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) & 0xff) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 {
                mag as i32
            } else {
                -(mag as i32)
            };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

fn bench_pair(w: u32, h: u32, ref_b: &[u8], dis_b: &[u8]) -> Duration {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("Ssim2::new");

    for _ in 0..WARMUP {
        let _ = s.compute(ref_b, dis_b).expect("warmup");
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = s.compute(ref_b, dis_b).expect("compute");
        samples.push(t.elapsed());
    }
    median(samples)
}

fn bench_cached(w: u32, h: u32, ref_b: &[u8], dis_b: &[u8]) -> Duration {
    let client = CudaRuntime::client(&Default::default());
    let mut s = Ssim2::<CudaRuntime>::new(client, w, h).expect("Ssim2::new");
    s.set_reference(ref_b).expect("set_reference");

    for _ in 0..WARMUP {
        let _ = s.compute_with_reference(dis_b).expect("warmup");
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        let _ = s.compute_with_reference(dis_b).expect("compute_with_reference");
        samples.push(t.elapsed());
    }
    median(samples)
}

fn main() {
    // CSV-friendly header. Tag column lets master vs Phase 1 runs be
    // concatenated into the same file.
    let tag = std::env::var("SSIM2_BENCH_TAG").unwrap_or_else(|_| "?".to_string());

    println!("tag,size,pair_ms,cached_ms");
    for (w, h, label) in SIZES {
        let (a, b) = synthetic_pair(*w as usize, *h as usize, 6);
        let pair = bench_pair(*w, *h, &a, &b);
        let cached = bench_cached(*w, *h, &a, &b);
        println!(
            "{tag},{label},{:.3},{:.3}",
            pair.as_secs_f64() * 1000.0,
            cached.as_secs_f64() * 1000.0
        );
    }
}
