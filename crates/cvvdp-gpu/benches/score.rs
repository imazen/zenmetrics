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
//! Both paths match pycvvdp v0.5.4 on the v1 manifest at f32
//! precision (≤ 0.005 JOD across q=1..90 since tick 207); the
//! public `Cvvdp::score` API routes through the GPU path as of
//! tick 213. This bench keeps the host-vs-GPU comparison live so
//! perf regressions on either side surface immediately.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use std::hint::black_box;
use std::path::Path;

use cubecl::Runtime;

#[path = "../tests/it/common/mod.rs"]
mod common;

use common::Backend;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

const W_256: u32 = 256;
const H_256: u32 = 256;

fn load_rgb_bytes(path: &Path) -> Vec<u8> {
    common::load_rgb_bytes(path, W_256, H_256)
}

/// Synthetic ref + dist pattern for benches at arbitrary
/// resolution. The actual pixel content doesn't matter for timing
/// (the algorithm has no data-dependent fast paths on GPU), so a
/// deterministic perlin-ish pattern + a perturbation is enough.
fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    common::synth_pair_with_offset_dist(w as usize, h as usize)
}

fn bench_resolution(c: &mut Criterion, w: u32, h: u32, label: &str, include_host: bool) {
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let (ref_bytes, dist_bytes) = synth_pair(w, h);

    let group_name = format!("cvvdp_jod_{label}");
    let mut g = c.benchmark_group(&group_name);
    g.throughput(Throughput::Elements(u64::from(w) * u64::from(h)));

    let client = Backend::client(&Default::default());
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on GPU backend");
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up jod");

    // host_scalar at 12 MP would take seconds per iteration —
    // skip it where it'd make the bench unrunnable. The GPU-only
    // numbers are what matter for the per-resolution scaling
    // question; we already have host_scalar at 256×256 in the
    // q20/q1 groups for absolute calibration.
    if include_host {
        g.bench_function("host_scalar", |b| {
            b.iter(|| {
                let jod = predict_jod_still_3ch(
                    black_box(&ref_bytes),
                    black_box(&dist_bytes),
                    w as usize,
                    h as usize,
                    display,
                    ppd,
                );
                black_box(jod);
            });
        });
    }

    g.bench_function("gpu_compute_dkl_jod", |b| {
        b.iter(|| {
            let jod = cvvdp
                .compute_dkl_jod(black_box(&ref_bytes), black_box(&dist_bytes), ppd)
                .expect("compute_dkl_jod");
            black_box(jod);
        });
    });

    // Warm-ref batch-scoring fast path. warm_reference pre-dispatches
    // the REF weber pyramid once; per-DIST calls skip it. Lib.rs
    // Status quotes ~1.8× per-DIST throughput vs cold at 12 MP — this
    // bench gives the empirical handle so regressions in the warm-
    // state path (ticks 236-240) surface.
    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    let _ = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect("warm-up warm-ref jod");
    g.bench_function("gpu_compute_dkl_jod_with_warm_ref", |b| {
        b.iter(|| {
            let jod = cvvdp
                .compute_dkl_jod_with_warm_ref(black_box(&dist_bytes), ppd)
                .expect("compute_dkl_jod_with_warm_ref");
            black_box(jod);
        });
    });

    g.finish();
}

fn bench_at_quality(c: &mut Criterion, q: u32) {
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png());
    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q));

    let group_name = format!("cvvdp_jod_256x256_q{q}");
    let mut g = c.benchmark_group(&group_name);
    g.throughput(Throughput::Elements(u64::from(W_256) * u64::from(H_256)));

    // All-host scalar path. Cvvdp::score used to route through this
    // before tick 213 switched to the GPU `compute_dkl_jod`; the
    // host path remains a faster-to-debug reference and is what
    // `host_scalar::predict_jod_still_3ch` exposes directly.
    // `shadow_jod` pins it to pycvvdp at ≤ 0.005 JOD.
    g.bench_function("host_scalar", |b| {
        b.iter(|| {
            let jod = predict_jod_still_3ch(
                black_box(&ref_bytes),
                black_box(&dist_bytes),
                W_256 as usize,
                H_256 as usize,
                display,
                ppd,
            );
            black_box(jod);
        });
    });

    // GPU-composed path on CUDA. Setup cost (Cvvdp::new + GPU
    // buffer allocs + srgb_lut upload + first-call cubecl kernel
    // compilation) hoisted via a warm-up call outside iter().
    let client = Backend::client(&Default::default());
    let mut cvvdp = Cvvdp::<Backend>::new(client, W_256, H_256, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on GPU backend");
    let _ = cvvdp
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up jod");
    let _ = cvvdp
        .compute_dkl_d_bands(&ref_bytes, &dist_bytes, ppd)
        .expect("warm-up d_bands");

    // GPU dispatch (color → weber × 2 sides → CSF → masking) plus
    // a per-band host read-back of every D plane (n_levels × 3
    // channels × pixels = ~432 MB at 12 MP, ~70 KB at 256×256).
    // Excludes the spatial pool, Minkowski fold, and met2jod.
    //
    // The delta vs `gpu_compute_dkl_jod` below is **not** just
    // "host post-processing": the JOD path replaces the per-band
    // readback with a single GPU `pool_band_kernel` pass and a
    // ~144-byte partials readback (tick 95+96). So compute_dkl_jod
    // can be *faster* than compute_dkl_d_bands at large sizes
    // — it skips the ~432 MB readback this function pays.
    g.bench_function("gpu_compute_dkl_d_bands", |b| {
        b.iter(|| {
            let d = cvvdp
                .compute_dkl_d_bands(black_box(&ref_bytes), black_box(&dist_bytes), ppd)
                .expect("compute_dkl_d_bands");
            black_box(d);
        });
    });

    g.bench_function("gpu_compute_dkl_jod", |b| {
        b.iter(|| {
            let jod = cvvdp
                .compute_dkl_jod(black_box(&ref_bytes), black_box(&dist_bytes), ppd)
                .expect("compute_dkl_jod");
            black_box(jod);
        });
    });

    // Warm-ref batch-scoring fast path — pair with the corpus
    // `gpu_compute_dkl_jod` bench above. See `bench_resolution` for
    // the rationale.
    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    let _ = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect("warm-up warm-ref jod");
    g.bench_function("gpu_compute_dkl_jod_with_warm_ref", |b| {
        b.iter(|| {
            let jod = cvvdp
                .compute_dkl_jod_with_warm_ref(black_box(&dist_bytes), ppd)
                .expect("compute_dkl_jod_with_warm_ref");
            black_box(jod);
        });
    });

    g.finish();
}

fn bench_1mp(c: &mut Criterion) {
    bench_resolution(c, 1024, 1024, "1mp_1024x1024", true);
}

fn bench_12mp(c: &mut Criterion) {
    // 4000×3000 = 12.0 MP, common DSLR aspect. host_scalar skipped
    // — single-threaded scalar at 12 MP takes seconds per iter.
    bench_resolution(c, 4000, 3000, "12mp_4000x3000", false);
}

fn bench_score_q20(c: &mut Criterion) {
    bench_at_quality(c, 20);
}

fn bench_score_q1(c: &mut Criterion) {
    // q=1 is the severe-distortion case. Pre-tick-204/206 the GPU
    // JOD drifted ~0.4 from pycvvdp at q=1 (cumulative f32 noise
    // through the soft clamp + met2jod non-linearity); the
    // chroma_shift CSF + gausspyr_reduce parity-bug fixes closed
    // that to 0.0000 (tracked by
    // `compute_dkl_jod_on_v1_manifest_corpus`). Per-call timing
    // should match q=20 since the algorithm shape doesn't change
    // with quality — running both is a sanity check that the bench
    // captures the intrinsic cost, not data-dependent fast paths.
    bench_at_quality(c, 1);
}

criterion_group!(
    benches,
    bench_score_q20,
    bench_score_q1,
    bench_1mp,
    bench_12mp
);
criterion_main!(benches);
