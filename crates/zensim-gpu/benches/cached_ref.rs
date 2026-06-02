//! Cached-reference vs. pair-mode benchmark for zensim-gpu.
//!
//! Compares the throughput of `ZensimOpaque::set_reference_srgb_u8` +
//! `compute_features_with_reference_srgb_u8` (cached path) against
//! `compute_features_vec_srgb_u8` (pair path) for a sweep workload:
//! 10 distortions scored against 1 reference. The cached path skips
//! N-1 ref uploads + N-1 ref-pyramid kernel launches.
//!
//! Expected win: roughly the ref-side fraction of total wall time
//! (30-50% on a 1024² grid). Run with:
//!
//!     cargo bench -p zensim-gpu --features "cubecl-types,wgpu" --bench cached_ref
//!     # or
//!     cargo bench -p zensim-gpu --features "cubecl-types,cuda" --bench cached_ref

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use std::hint::black_box;

use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};
use zensim_gpu::{Backend, ZensimOpaque, ZensimParams};

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

const W: u32 = 1024;
const H: u32 = 1024;
const N_DIST: usize = 10;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(31).wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_mul(17).wrapping_add(seed.wrapping_mul(3))) & 0xff) as u8;
            let b = ((x.wrapping_mul(7) ^ y.wrapping_mul(11) ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

/// Pair-mode: N_DIST × (set ref + score) — what the legacy sweep
/// path does when callers don't cache the reference.
fn bench_pair_mode(c: &mut Criterion) {
    let ref_buf = make_image(W, H, 0);
    let dists: Vec<Vec<u8>> = (1..=N_DIST as u32)
        .map(|s| make_image(W, H, s.wrapping_mul(13)))
        .collect();

    let mut group = c.benchmark_group("cached_ref_1024x1024_10dist");
    group.sample_size(10);

    group.bench_function("pair_mode", |b| {
        let mut z = ZensimOpaque::new(BACKEND_E, W, H, ZensimParams::new()).expect("opaque new");
        b.iter(|| {
            let mut acc: f64 = 0.0;
            for d in &dists {
                let v = z
                    .compute_features_vec_srgb_u8(&ref_buf, d)
                    .expect("pair compute");
                acc += v[0];
            }
            black_box(acc);
        });
    });

    group.bench_function("cached_ref", |b| {
        let mut z = ZensimOpaque::new(BACKEND_E, W, H, ZensimParams::new()).expect("opaque new");
        b.iter(|| {
            z.set_reference_srgb_u8(&ref_buf).expect("set_reference");
            let mut acc: f64 = 0.0;
            for d in &dists {
                let v = z
                    .compute_features_with_reference_srgb_u8(d)
                    .expect("compute_with_reference");
                acc += v[0];
            }
            black_box(acc);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_pair_mode);
criterion_main!(benches);
