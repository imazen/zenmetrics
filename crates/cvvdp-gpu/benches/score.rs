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

fn bench_at_quality(c: &mut Criterion, q: u32) {
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png());
    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q));

    let group_name = format!("cvvdp_jod_256x256_q{q}");
    let mut g = c.benchmark_group(&group_name);
    g.throughput(Throughput::Elements((W * H) as u64));

    // All-host scalar path. What Cvvdp::score routes through today;
    // shadow_jod pins it to pycvvdp.
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

    // GPU-composed path on CUDA. Setup cost (Cvvdp::new + GPU
    // buffer allocs + srgb_lut upload + first-call cubecl kernel
    // compilation) hoisted via a warm-up call outside iter().
    let client = CudaRuntime::client(&Default::default());
    let mut cvvdp = Cvvdp::<CudaRuntime>::new(client, W, H, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on cuda");
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up jod");
    let _ = cvvdp
        .compute_dkl_d_bands(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up d_bands");

    // Inner GPU work only (color → weber × 2 sides → CSF → masking
    // → D-band read-back). Excludes the host-side lp_norm_mean +
    // 3-stage Minkowski + met2jod. Difference vs gpu_compute_dkl_jod_cuda
    // is the host post-processing cost.
    g.bench_function("gpu_compute_dkl_d_bands_cuda", |b| {
        b.iter(|| {
            let d = cvvdp
                .compute_dkl_d_bands(black_box(&ref_bytes), black_box(&dist_bytes), ppd)
                .expect("compute_dkl_d_bands");
            black_box(d);
        });
    });

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

fn bench_score_q20(c: &mut Criterion) {
    bench_at_quality(c, 20);
}

fn bench_score_q1(c: &mut Criterion) {
    // q=1 is the severe-distortion case where the GPU JOD drifts
    // 0.40 from the host scalar (cumulative f32 noise through the
    // soft clamp + met2jod non-linearity, see drift survey in
    // tests/pipeline_score.rs). Per-call timing should match q=20
    // since the algorithm shape doesn't change with quality —
    // running both is a sanity check that the bench captures the
    // intrinsic cost, not data-dependent fast paths.
    bench_at_quality(c, 1);
}

criterion_group!(benches, bench_score_q20, bench_score_q1);
criterion_main!(benches);
