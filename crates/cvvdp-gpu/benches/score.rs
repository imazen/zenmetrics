//! Score-path benchmarks for cvvdp-gpu.
//!
//! Compares the all-host scalar path (`host_scalar::predict_jod_still_3ch`)
//! against the GPU-composed path (`Cvvdp::compute_dkl_jod` on the
//! CUDA backend) on the v1 manifest corpus images (256×256, source
//! PNG + JPEG q=20 distorted). Tracks both the cold path and the
//! warm path (Cvvdp instance reused across iterations).
//!
//! Run with:
//!     cargo bench -p cvvdp-gpu --features cuda
//!
//! `Cvvdp::score` is still pinned to the host path; this bench
//! exposes the GPU-vs-host speedup so future ticks can decide
//! whether the (production-quality at q ≥ 20) GPU path is worth
//! the cumulative f32 drift trade-off documented in the
//! drift-survey integration tests.

#![cfg(feature = "cuda")]

use std::hint::black_box;
use std::path::PathBuf;

use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};
use image::ImageReader;
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

const W: u32 = 256;
const H: u32 = 256;

fn load_rgb_bytes(path: &PathBuf) -> Vec<u8> {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e}"))
        .decode()
        .unwrap_or_else(|e| panic!("decode {path:?}: {e}"))
        .to_rgb8();
    assert_eq!(img.width(), W);
    assert_eq!(img.height(), H);
    img.into_raw()
}

fn bench_score(c: &mut Criterion) {
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png());
    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20));

    let mut g = c.benchmark_group("cvvdp_jod_256x256_q20");
    g.throughput(Throughput::Elements((W * H) as u64));

    // All-host scalar path. This is what Cvvdp::score routes
    // through today and what shadow_jod pins to pycvvdp.
    g.bench_function("host_scalar", |b| {
        b.iter(|| {
            let jod = predict_jod_still_3ch(
                black_box(&ref_bytes),
                black_box(&dist_bytes),
                W as usize,
                H as usize,
                display,
                ppd,
            );
            black_box(jod);
        });
    });

    // GPU-composed path on CUDA. Setup cost (Cvvdp::new, GPU buffer
    // allocs, srgb_lut upload) hoisted outside iter() — the inner
    // measurement is the per-call cost amortized over many runs.
    let client = CudaRuntime::client(&Default::default());
    let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, W, H, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on cuda");
    // Warm up so the first measured iteration doesn't pay for
    // cubecl kernel compilation.
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up");
    g.bench_function("gpu_compute_dkl_jod_cuda", |b| {
        b.iter(|| {
            let jod = cvvdp
                .compute_dkl_jod(black_box(&ref_bytes), black_box(&dist_bytes), ppd)
                .expect("compute_dkl_jod");
            black_box(jod);
        });
    });

    g.finish();
}

criterion_group!(benches, bench_score);
criterion_main!(benches);
