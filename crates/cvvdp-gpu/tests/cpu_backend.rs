//! CPU-runtime smoke + parity tests.
//!
//! After tick 208 closed the cpu-pool blocker by adding
//! [`Cvvdp::compute_dkl_jod_host_pool`], the cubecl-cpu runtime
//! can produce JOD. This file pins that the cpu-only build:
//!
//! 1. Compiles + initialises a cubecl-cpu runtime
//! 2. Runs the host-pool JOD path without panicking
//! 3. Matches `host_scalar::predict_jod_still_3ch` at f32 precision
//!    (both paths share `lp_norm_mean` + `do_pooling_and_jod_still_3ch`;
//!    only the upstream stages run on different backends).
//!
//! cpu-only build:
//!     cargo test -p cvvdp-gpu --no-default-features --features cpu \
//!         --test cpu_backend
//!
//! All other GPU test files gate themselves out of cpu-only builds
//! (`#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]`),
//! so this file is the only place cpu-backend coverage lives.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

type Backend = cubecl::cpu::CpuRuntime;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w * h * 3) as usize;
    let mut r = vec![0u8; n];
    let mut d = vec![0u8; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let rr = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let b = (((x + y) * 4) % 256) as u8;
            let i = (y * w as usize + x) * 3;
            r[i] = rr;
            r[i + 1] = g;
            r[i + 2] = b;
            d[i] = rr.saturating_sub(8);
            d[i + 1] = g.saturating_sub(4);
            d[i + 2] = b.saturating_add(12);
        }
    }
    (r, d)
}

#[test]
fn compute_dkl_jod_host_pool_runs_on_cpu_backend() {
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);
    let jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool on cpu");

    eprintln!("cpu-backend JOD = {jod:.6}");
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
}

#[test]
fn compute_dkl_jod_host_pool_with_warm_ref_runs_on_cpu_backend() {
    // Tick 212 follow-up: validates the warm-ref host-pool variant
    // on the cpu runtime. Batch CPU scoring against a warmed REF
    // should produce the same JOD as the cold-ref host_pool path.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);
    let cold = cvvdp
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("cold host_pool");

    cvvdp.warm_reference(&ref_b).expect("warm_reference on cpu");
    let warm = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_b, ppd)
        .expect("warm host_pool on cpu");

    let diff = (cold - warm).abs();
    eprintln!("cpu cold host_pool = {cold:.6}, warm host_pool = {warm:.6}, |diff| = {diff:.6}");
    assert!(
        diff < 0.005,
        "cpu warm host_pool {warm:.6} diverges from cold {cold:.6} by {diff:.6}"
    );
}

#[test]
fn compute_dkl_jod_host_pool_matches_host_scalar_on_cpu_backend() {
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);
    let cpu_jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool on cpu");
    let host_jod = predict_jod_still_3ch(
        &ref_b,
        &dist_b,
        w as usize,
        h as usize,
        display,
        ppd,
    );
    let diff = (cpu_jod - host_jod).abs();
    eprintln!(
        "cpu_backend (host_pool) = {cpu_jod:.6}, host_scalar = {host_jod:.6}, |diff| = {diff:.6}"
    );
    assert!(
        diff < 0.005,
        "cpu-backend host_pool diverges from host_scalar by {diff:.6}"
    );
}
