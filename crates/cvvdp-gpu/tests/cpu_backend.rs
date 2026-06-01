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
// pycvvdp golden literals come from the bench script's printed
// output verbatim. The 7-digit decimal documents the source value
// even though LLVM rounds at f32 precision — same pattern as the
// library's `#![allow(clippy::excessive_precision)]`.
#![allow(clippy::excessive_precision)]

use cubecl::Runtime;
use cubecl::prelude::*;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::kernels::pyramid::{
    DOWNSCALE_TILED_BLOCK_DIM, downscale_kernel, downscale_tiled_kernel, gausspyr_reduce_scalar,
};
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

#[path = "common/mod.rs"]
mod common;

type Backend = cubecl::cpu::CpuRuntime;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    // Uses the odd-dim ref (`(x * 8) % 256` etc) — the 73×91
    // pycvvdp golden is computed against this construction in
    // `bench_12mp_cuda.py::synth_pair_odd_dim`.
    common::synth_pair_odd_dim_with_offset_dist(w as usize, h as usize)
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
fn compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs() {
    // End-to-end identity gate: feeding the same buffer in as both
    // reference AND distorted side must produce JOD ≈ 10.0
    // (`met2jod(0) == 10`). The chain — color → weber pyramid →
    // CSF → masking → spatial pool — must compute exact-zero D
    // for every (band, channel), then fold to Q = 0, then map to
    // JOD = 10.
    //
    // The doctest on `Cvvdp::score` exercises the same property
    // but only in `cargo test --doc` runs. This test gates it in
    // the integration suite so a refactor that breaks identity
    // (e.g. a stray `+ eps` in a kernel, a saturation off-by-one
    // in the sRGB→linear LUT, a missing baseband-bypass) trips
    // here even when doctest runs are skipped.
    //
    // Tolerance is 1e-3 JOD — matches the doctest. Identity should
    // give exactly 10.0 in theory, but the eps shift in
    // `pool_band_finalize` produces a tiny non-zero floor for
    // empty input that propagates into the final JOD; the
    // tolerance accommodates that without admitting any
    // meaningful chain drift.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    // Mid-gray (not pure black or white) avoids any sRGB-LUT
    // boundary artifacts that an off-by-one quantization regression
    // could hide.
    let buf = vec![128u8; (w * h * 3) as usize];
    let jod = cvvdp
        .compute_dkl_jod_host_pool(&buf, &buf, ppd)
        .expect("compute_dkl_jod_host_pool on cpu (identity)");

    eprintln!("identity JOD (host_pool, cpu) = {jod:.6}");
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "identity should give JOD ≈ 10, got {jod}",
    );
}

#[test]
fn compute_dkl_jod_host_pool_with_warm_ref_runs_on_cpu_backend() {
    // Tick 212 follow-up: validates the warm-ref host-pool variant
    // on the cpu runtime. Batch CPU scoring against a warmed REF
    // should produce the same JOD as the cold-ref host_pool path.
    //
    // Tick 496: strengthened from 0.005 tolerance to bit-equality.
    // The cpu runtime executes every kernel sequentially (no GPU
    // atomics → no nondeterminism), and the host_pool path bypasses
    // the GPU Atomic<f32>::fetch_add pool kernel entirely
    // (lp_norm_mean is deterministic sequential f32). Both halves
    // of the warm-vs-cold comparison run the same Weber+CSF+masking
    // dispatch on REF and DIST, then fold via the same host pool
    // — output should be bit-identical. Catches a refactor that
    // accidentally introduces nondeterminism on the warm-ref path
    // (e.g. accumulating across calls without resetting a scratch).
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

    eprintln!(
        "cpu cold host_pool = {cold:.6} ({:#010x}), warm host_pool = {warm:.6} ({:#010x})",
        cold.to_bits(),
        warm.to_bits(),
    );
    assert_eq!(
        cold.to_bits(),
        warm.to_bits(),
        "cpu warm host_pool {warm} not bit-identical to cold {cold}: the cpu runtime + \
         sequential host pool path should be deterministic across warm/cold dispatches",
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
    let host_jod = predict_jod_still_3ch(&ref_b, &dist_b, w as usize, h as usize, display, ppd);
    let diff = (cpu_jod - host_jod).abs();
    eprintln!(
        "cpu_backend (host_pool) = {cpu_jod:.6}, host_scalar = {host_jod:.6}, |diff| = {diff:.6}"
    );
    assert!(
        diff < 0.005,
        "cpu-backend host_pool diverges from host_scalar by {diff:.6}"
    );
}

#[test]
fn compute_dkl_jod_host_pool_matches_pycvvdp_at_73x91_odd_on_cpu_backend() {
    // Tick 223: direct cpu-backend vs pycvvdp parity on the 73×91
    // odd-dim fixture. synth_pair() above uses the exact
    // synth_pair_odd_dim construction from
    // scripts/cvvdp_goldens/bench_12mp_cuda.py:152, so the
    // pycvvdp golden 9.390370 applies at 73×91.
    //
    // Previously the cpu-backend was only covered transitively:
    //   - host_pool == host_scalar at f32 noise (3 tests above)
    //   - host_scalar == pycvvdp at 0.005 (shadow_jod_runs_and_is_monotonic_on_corpus,
    //     pipeline_color tests)
    // This pins the cpu-backend JOD path directly against the
    // canonical pycvvdp reference. Also exercises tick 206's
    // gausspyr_reduce parity-bug replication — the 73×91 fixture
    // hits the mixed-parity (6×5 → 3×3, 46×37 → 23×19) reduce
    // levels where the bug-compat fix matters.
    //
    // 73×91 = 6643 px, ~6.5× the existing 32×32 cpu smoke tests;
    // expected runtime ~5-10s on cubecl-cpu.
    // Tick 265 dedup: golden loaded from
    // scripts/cvvdp_goldens/pycvvdp_synth_goldens.json via the
    // common helper (was hardcoded 9.390370, kept in sync by hand).
    //
    // Phase 9zc (2026-05-28): re-enabled. Root cause was a
    // multi-cube SharedMemory + sync_cube isolation bug in
    // zenforks-cubecl-cpu: the runtime generated 3 nested
    // scf::for loops over CubeCount* inside the per-unit MLIR
    // kernel body, but the global sync_cube barrier (counted
    // in cube_dim_size arrivals) lost isolation between cubes.
    // Different units could advance to different cube iterations
    // between syncs — cube k's units could read shared memory
    // written by cube k+1's unit 0. Surfaced on the 73×91
    // odd-dim fixture because the gauss pyramid's
    // `downscale_tiled_kernel` (LDS-tiled) launches 3×3
    // workgroups at this size and uses `SharedMemory` for the
    // cooperative tile load. Fixed in zenforks-cubecl-cpu 0.10.2
    // by emitting an implicit sync_cube at the end of every
    // cube-iteration body.
    let pycvvdp_golden = common::pycvvdp_synth_golden_jod("synth_73x91_odd");
    const TOLERANCE: f32 = 0.005;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);
    let jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool on cpu");
    let diff = (jod - pycvvdp_golden).abs();
    eprintln!(
        "cpu-backend 73×91: jod = {jod:.6}, pycvvdp golden = {pycvvdp_golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < TOLERANCE,
        "cpu-backend JOD {jod:.6} drifts from pycvvdp golden {pycvvdp_golden:.6} by {diff:.6} > {TOLERANCE:.6}"
    );
}

#[test]
fn perf_mode_fast_matches_strict_on_cpu_host_pool() {
    // Tick 327 sibling to
    // `perf_mode_fast_matches_strict_today` (pipeline_score.rs).
    // That test pins the no-op contract on the GPU pool path,
    // where `Atomic<f32>::fetch_add`'s non-deterministic reduce
    // order forces a 1e-4 tolerance instead of bit-equality.
    //
    // The cpu-runtime host-pool path bypasses the GPU atomic pool
    // entirely (`compute_dkl_jod_host_pool` reads D bands back to
    // host then folds via `host_scalar::lp_norm_mean`, which is
    // deterministic sequential f32 arithmetic). So Fast vs Strict
    // here SHOULD produce bit-identical output today, and that's
    // a tighter contract worth pinning.
    //
    // When a real Fast-mode optimization lands the test should be
    // RELAXED (not deleted) to that optimization's documented drift
    // budget on the cpu/host-pool path; the CHANGELOG entry
    // documents the new tolerance.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let (ref_b, dist_b) = synth_pair(w, h);

    let mut strict = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new (strict)");
    let strict_jod = strict
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool (strict)");

    let mut fast = Cvvdp::<Backend>::new(
        client,
        w,
        h,
        CvvdpParams {
            perf_mode: cvvdp_gpu::PerfMode::Fast,
            ..CvvdpParams::PLACEHOLDER
        },
    )
    .expect("Cvvdp::new (fast)");
    let fast_jod = fast
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool (fast)");

    // Deterministic host-pool path. Bit-equality today.
    assert_eq!(
        strict_jod.to_bits(),
        fast_jod.to_bits(),
        "PerfMode::Fast must produce bit-identical output to PerfMode::Strict \
         on the cpu host-pool path (no atomic-add non-determinism here) \
         until a Fast-mode optimization lands (strict={strict_jod}, fast={fast_jod})"
    );

    // Also pin the warm-ref host-pool variant: same code path
    // post-warm, same determinism guarantee.
    strict
        .warm_reference(&ref_b)
        .expect("warm_reference (strict)");
    let strict_warm = strict
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_b, ppd)
        .expect("compute_dkl_jod_host_pool_with_warm_ref (strict)");
    fast.warm_reference(&ref_b).expect("warm_reference (fast)");
    let fast_warm = fast
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_b, ppd)
        .expect("compute_dkl_jod_host_pool_with_warm_ref (fast)");
    assert_eq!(
        strict_warm.to_bits(),
        fast_warm.to_bits(),
        "PerfMode::Fast must produce bit-identical output to PerfMode::Strict \
         on the warm-ref cpu host-pool path (strict={strict_warm}, fast={fast_warm})"
    );
}

#[test]
fn host_pool_flat_vs_flat_yields_max_jod() {
    // Tick 546: third-leg sibling of `flat_vs_flat_yields_max_jod_regardless_of_brightness`
    // (host scalar, tick 542) and `cvvdp_score_flat_vs_flat_yields_max_jod`
    // (GPU score path, tick 545). Pins cvvdp's spatial-contrast contract
    // on the cubecl-cpu host-pool dispatch path: flat black vs flat
    // white returns JOD ≈ 10 because both flat inputs have zero
    // Weber-band energy at every level.
    //
    // The cpu runtime executes every kernel sequentially with no atomic
    // races, so 1e-3 (matching the existing `compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`
    // tolerance) is the right band — tighter than the GPU's 1e-2.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let ref_black: Vec<u8> = vec![0u8; (w * h * 3) as usize];
    let dist_white: Vec<u8> = vec![255u8; (w * h * 3) as usize];
    let jod_bw = cvvdp
        .compute_dkl_jod_host_pool(&ref_black, &dist_white, ppd)
        .expect("host_pool black-vs-white");
    eprintln!("host_pool flat-vs-flat (black vs white): jod = {jod_bw:.4}");
    assert!(
        (jod_bw - 10.0).abs() < 1e-3,
        "host_pool flat-vs-flat should give JOD ≈ 10, got {jod_bw}",
    );

    let ref_gray: Vec<u8> = vec![128u8; (w * h * 3) as usize];
    let dist_gray: Vec<u8> = vec![64u8; (w * h * 3) as usize];
    let jod_gg = cvvdp
        .compute_dkl_jod_host_pool(&ref_gray, &dist_gray, ppd)
        .expect("host_pool gray-vs-gray");
    assert!(
        (jod_gg - 10.0).abs() < 1e-3,
        "host_pool flat 128 vs flat 64 should give JOD ≈ 10, got {jod_gg}",
    );
}

#[test]
fn host_pool_textured_vs_flat_detects_detail_loss() {
    // Tick 546: third-leg sibling of `textured_ref_vs_flat_dist_detects_detail_loss`
    // (host scalar, tick 543) and `cvvdp_score_textured_vs_flat_detects_detail_loss`
    // (GPU, tick 545). Textured ref + flat dist (catastrophic blur)
    // on the cubecl-cpu host-pool path must give JOD ≪ 10.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let n = (w * h * 3) as usize;
    let ref_textured: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let dist_flat: Vec<u8> = vec![128u8; n];
    let jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_textured, &dist_flat, ppd)
        .expect("host_pool textured-vs-flat");
    eprintln!("host_pool textured-ref-vs-flat-dist: jod = {jod:.4}");
    assert!(
        jod.is_finite(),
        "host_pool blur JOD must be finite, got {jod}"
    );
    assert!(
        jod < 9.0,
        "host_pool textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    assert!(
        jod > -10.0,
        "host_pool blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn host_pool_monotonically_decreases_with_noise_amplitude() {
    // Tick 546: third-leg sibling of `jod_monotonically_decreases_with_noise_amplitude`
    // (host scalar, tick 544) and `cvvdp_score_monotonically_decreases_with_noise_amplitude`
    // (GPU, tick 545). Dense alternating-sign noise at amplitudes
    // {2, 8, 32} on the cubecl-cpu host-pool path must give strictly
    // monotonically-decreasing JOD.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let n = (w * h * 3) as usize;
    let ref_: Vec<u8> = (0..n).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
                let sign: i16 = if ((i * 31).wrapping_add(17)) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let delta = sign * amplitude as i16;
                (b as i16 + delta).clamp(0, 255) as u8
            })
            .collect()
    }

    let jod_a2 = cvvdp
        .compute_dkl_jod_host_pool(&ref_, &add_alt_noise(&ref_, 2), ppd)
        .expect("host_pool a=2");
    let jod_a8 = cvvdp
        .compute_dkl_jod_host_pool(&ref_, &add_alt_noise(&ref_, 8), ppd)
        .expect("host_pool a=8");
    let jod_a32 = cvvdp
        .compute_dkl_jod_host_pool(&ref_, &add_alt_noise(&ref_, 32), ppd)
        .expect("host_pool a=32");
    eprintln!("host_pool noise sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}");

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite host_pool JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    assert!(
        jod_a2 > jod_a8,
        "host_pool JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8}",
    );
    assert!(
        jod_a8 > jod_a32,
        "host_pool JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32}",
    );
    assert!(
        jod_a2 < 10.0 - 1e-4,
        "a=2 noise should detectably drop host_pool JOD below 10, got {jod_a2}",
    );
}

#[test]
fn host_pool_warm_ref_flat_vs_flat_yields_max_jod() {
    // Tick 547: sixth-leg sibling of the spatial-contrast contract.
    // cpu warm-ref host_pool dispatch (`warm_reference` +
    // `compute_dkl_jod_host_pool_with_warm_ref`) must give JOD ≈ 10
    // for flat ref + flat dist.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let ref_black: Vec<u8> = vec![0u8; (w * h * 3) as usize];
    let dist_white: Vec<u8> = vec![255u8; (w * h * 3) as usize];
    cvvdp
        .warm_reference(&ref_black)
        .expect("warm_reference black");
    let jod_bw = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_white, ppd)
        .expect("host_pool_warm_ref black-vs-white");
    eprintln!("host_pool warm-ref flat-vs-flat (black vs white): jod = {jod_bw:.4}");
    assert!(
        (jod_bw - 10.0).abs() < 1e-3,
        "host_pool warm-ref flat-vs-flat should give JOD ≈ 10, got {jod_bw}",
    );

    let ref_gray: Vec<u8> = vec![128u8; (w * h * 3) as usize];
    let dist_gray: Vec<u8> = vec![64u8; (w * h * 3) as usize];
    cvvdp
        .warm_reference(&ref_gray)
        .expect("warm_reference gray");
    let jod_gg = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_gray, ppd)
        .expect("host_pool_warm_ref gray-vs-gray");
    assert!(
        (jod_gg - 10.0).abs() < 1e-3,
        "host_pool warm-ref flat 128 vs flat 64 should give JOD ≈ 10, got {jod_gg}",
    );
}

#[test]
fn host_pool_warm_ref_textured_vs_flat_detects_detail_loss() {
    // Tick 547: sixth-leg sibling of the blur-detection pin. cpu
    // warm-ref host_pool dispatch must detect catastrophic blur.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let n = (w * h * 3) as usize;
    let ref_textured: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let dist_flat: Vec<u8> = vec![128u8; n];
    cvvdp
        .warm_reference(&ref_textured)
        .expect("warm_reference textured");
    let jod = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&dist_flat, ppd)
        .expect("host_pool_warm_ref textured-vs-flat");
    eprintln!("host_pool warm-ref textured-ref-vs-flat-dist: jod = {jod:.4}");
    assert!(
        jod.is_finite(),
        "host_pool warm-ref blur JOD must be finite, got {jod}"
    );
    assert!(
        jod < 9.0,
        "host_pool warm-ref textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    assert!(
        jod > -10.0,
        "host_pool warm-ref blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn host_pool_warm_ref_monotonically_decreases_with_noise_amplitude() {
    // Tick 547: sixth-leg sibling of the noise-amplitude monotonicity
    // pin. cpu warm-ref host_pool dispatch must show strict
    // monotonicity across dense alternating-sign noise amplitudes
    // {2, 8, 32}.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let n = (w * h * 3) as usize;
    let ref_: Vec<u8> = (0..n).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
                let sign: i16 = if ((i * 31).wrapping_add(17)) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let delta = sign * amplitude as i16;
                (b as i16 + delta).clamp(0, 255) as u8
            })
            .collect()
    }

    cvvdp.warm_reference(&ref_).expect("warm_reference");
    let jod_a2 = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&add_alt_noise(&ref_, 2), ppd)
        .expect("host_pool_warm_ref a=2");
    let jod_a8 = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&add_alt_noise(&ref_, 8), ppd)
        .expect("host_pool_warm_ref a=8");
    let jod_a32 = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&add_alt_noise(&ref_, 32), ppd)
        .expect("host_pool_warm_ref a=32");
    eprintln!(
        "host_pool warm-ref noise sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}"
    );

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite host_pool warm-ref JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    assert!(
        jod_a2 > jod_a8,
        "host_pool warm-ref JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8}",
    );
    assert!(
        jod_a8 > jod_a32,
        "host_pool warm-ref JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32}",
    );
    assert!(
        jod_a2 < 10.0 - 1e-4,
        "a=2 noise should detectably drop host_pool warm-ref JOD below 10, got {jod_a2}",
    );
}

// ============================================================================
// Diagnostic tests for task #120: cubecl-cpu odd-dim downscale_kernel divergence
// ============================================================================

/// Drive `downscale_kernel` on the cpu runtime directly for the input
/// size and compare against the host scalar reference
/// `gausspyr_reduce_scalar`. Returns the max-abs error.
fn diag_downscale_on_cpu(sw: u32, sh: u32) -> (Vec<f32>, Vec<f32>, f32) {
    let client = Backend::client(&Default::default());
    let nsrc = (sw * sh) as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let ndst = (dw * dh) as usize;

    // Deterministic input: same pattern as the synth_pair_odd_dim
    // generator, but as a single f32 channel so we drive downscale
    // directly. Values stay in a similar scale (0..32) so f32 noise
    // is in the 1e-6 range like the rest of the pipeline.
    let src: Vec<f32> = (0..nsrc)
        .map(|i| {
            let y = i / sw as usize;
            let x = i - y * sw as usize;
            ((x % 8 + y % 8) as f32) * 0.5
        })
        .collect();
    let dst = vec![0.0_f32; ndst];

    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let dst_h = client.create_from_slice(f32::as_bytes(&dst));

    let cube_dim = CubeDim::new_1d(64);
    let cube_count = CubeCount::Static((ndst as u32).div_ceil(64), 1, 1);
    unsafe {
        downscale_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), nsrc),
            ArrayArg::from_raw_parts(dst_h.clone(), ndst),
            sw,
            sh,
            dw,
            dh,
        );
    }
    let out_bytes = client.read_one(dst_h.clone()).expect("read dst");
    let gpu_out: Vec<f32> = f32::from_bytes(&out_bytes).to_vec();

    let mut cpu_out = Vec::new();
    gausspyr_reduce_scalar(&src, sw as usize, sh as usize, &mut cpu_out);
    assert_eq!(cpu_out.len(), gpu_out.len(), "len mismatch sw={sw} sh={sh}");

    let max_err = gpu_out
        .iter()
        .zip(&cpu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);

    (gpu_out, cpu_out, max_err)
}

#[test]
fn diag_downscale_cpu_even_w_even_h() {
    // sw=8 sh=8 — both even. Baseline: should pass on cpu backend.
    let (gpu, cpu, e) = diag_downscale_on_cpu(8, 8);
    eprintln!("diag 8×8: max-abs = {e:.2e}\ngpu={gpu:?}\ncpu={cpu:?}");
    assert!(e < 1e-5, "8×8 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_cpu_odd_w_even_h() {
    // sw=7 sh=8 — sw odd, sh even.
    // pycvvdp picks even-W patch using sh's parity; reflect gives
    // "odd-W" patch. Delta correction applies.
    let (gpu, cpu, e) = diag_downscale_on_cpu(7, 8);
    eprintln!("diag 7×8: max-abs = {e:.4}\ngpu={gpu:?}\ncpu={cpu:?}");
    assert!(e < 1e-5, "7×8 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_cpu_even_w_odd_h() {
    // sw=8 sh=7 — sw even, sh odd.
    // Reflect gives "even-W" patch; pycvvdp picks odd-W. Delta correction applies.
    let (gpu, cpu, e) = diag_downscale_on_cpu(8, 7);
    eprintln!("diag 8×7: max-abs = {e:.4}\ngpu={gpu:?}\ncpu={cpu:?}");
    assert!(e < 1e-5, "8×7 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_cpu_odd_w_odd_h() {
    // sw=7 sh=7 — both odd. Same parity, no delta correction. Should pass.
    let (gpu, cpu, e) = diag_downscale_on_cpu(7, 7);
    eprintln!("diag 7×7: max-abs = {e:.4}\ngpu={gpu:?}\ncpu={cpu:?}");
    assert!(e < 1e-5, "7×7 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_cpu_at_73x91() {
    // The exact size where the 73×91 cvvdp parity test fails.
    let (_gpu, _cpu, e) = diag_downscale_on_cpu(73, 91);
    eprintln!("diag 73×91: max-abs = {e:.6}");
    assert!(e < 1e-5, "73×91 cpu downscale max-abs = {e}");
}

#[test]
fn diag_host_pool_vs_host_scalar_at_73x91_odd() {
    // Direct cpu-backend host_pool vs host_scalar at 73×91, the
    // exact size where the pycvvdp parity test fails. If this also
    // diverges, the bug is in a GPU kernel (cpu-backend codegen).
    // If this matches, the bug is somewhere in the host->JOD chain
    // (less likely given the 32×32 test passes bit-equal).
    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);
    let cpu_jod = cvvdp
        .compute_dkl_jod_host_pool(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod_host_pool on cpu");
    let host_jod = predict_jod_still_3ch(&ref_b, &dist_b, w as usize, h as usize, display, ppd);
    let diff = (cpu_jod - host_jod).abs();
    eprintln!(
        "[diag] 73×91 cpu host_pool = {cpu_jod:.6}, host_scalar = {host_jod:.6}, |diff| = {diff:.6}"
    );
    // At 32×32 this diff is 0.0; at 73×91 we expect a divergence
    // since the pycvvdp parity test diverges by 1.73 JOD. Document
    // here that the bug exists in a GPU stage (since host_scalar
    // computes everything on host).
    if diff > 0.005 {
        eprintln!(
            "[diag] CONFIRMED: cpu-backend host_pool diverges from host_scalar at 73×91 by {diff:.6} JOD."
        );
        eprintln!(
            "[diag] Bug is in cubecl-cpu codegen of one of the GPU kernels in the host_pool dispatch chain."
        );
    }
    // Don't assert — this is diagnostic. Just print.
}

/// Compute per-band per-channel D arrays using fully-scalar reference
/// code. Mirrors the inner loop of `predict_jod_still_3ch` but
/// returns the D arrays instead of pooling. Used by the
/// stage-probe to compare against the GPU-computed D bands.
fn scalar_d_bands(
    ref_srgb: &[u8],
    dist_srgb: &[u8],
    width: usize,
    height: usize,
    display: DisplayModel,
    ppd: f32,
) -> Vec<[Vec<f32>; 3]> {
    use cvvdp::kernels::color::display_byte_to_dkl_scalar;
    use cvvdp::kernels::csf::{CSF_BASEBAND_RHO, CsfChannel, sensitivity_corrected_scalar};
    use cvvdp::kernels::masking::{CH_GAIN, mult_mutual_band};
    use cvvdp::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};

    let ch_gain: [f32; 3] = CH_GAIN;

    let n = width * height;
    let mut ref_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut dis_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) = display_byte_to_dkl_scalar(
            ref_srgb[i * 3],
            ref_srgb[i * 3 + 1],
            ref_srgb[i * 3 + 2],
            display,
        );
        ref_planes[0][i] = a;
        ref_planes[1][i] = rg;
        ref_planes[2][i] = vy;
        let (a, rg, vy) = display_byte_to_dkl_scalar(
            dist_srgb[i * 3],
            dist_srgb[i * 3 + 1],
            dist_srgb[i * 3 + 2],
            display,
        );
        dis_planes[0][i] = a;
        dis_planes[1][i] = rg;
        dis_planes[2][i] = vy;
    }

    let n_levels_query = band_frequencies(ppd, width, height).len();
    let ref_weber = [
        weber_contrast_pyr_dec_scalar(
            &ref_planes[0],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[1],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[2],
            &ref_planes[0],
            width,
            height,
            n_levels_query,
        ),
    ];
    let dis_weber = [
        weber_contrast_pyr_dec_scalar(
            &dis_planes[0],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[1],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[2],
            &dis_planes[0],
            width,
            height,
            n_levels_query,
        ),
    ];
    let n_levels = ref_weber[0].bands.len();
    let freqs = band_frequencies(ppd, width, height);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    let mut d_bands: Vec<[Vec<f32>; 3]> = Vec::with_capacity(n_levels);
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = ref_weber[0].bands[k].w;
        let bh = ref_weber[0].bands[k].h;
        let n_px = bw * bh;
        let rho = if is_baseband {
            CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };
        let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];

        let mut t_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        let mut r_p_per_ch: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
        for i in 0..n_px {
            let log_l = log_l_bkg_band[i];
            let s_a = sensitivity_corrected_scalar(rho, log_l, channels[0]);
            let s_rg = sensitivity_corrected_scalar(rho, log_l, channels[1]);
            let s_vy = sensitivity_corrected_scalar(rho, log_l, channels[2]);
            t_p_per_ch[0][i] = band_mul * dis_weber[0].bands[k].data[i] * s_a * ch_gain[0];
            t_p_per_ch[1][i] = band_mul * dis_weber[1].bands[k].data[i] * s_rg * ch_gain[1];
            t_p_per_ch[2][i] = band_mul * dis_weber[2].bands[k].data[i] * s_vy * ch_gain[2];
            r_p_per_ch[0][i] = band_mul * ref_weber[0].bands[k].data[i] * s_a * ch_gain[0];
            r_p_per_ch[1][i] = band_mul * ref_weber[1].bands[k].data[i] * s_rg * ch_gain[1];
            r_p_per_ch[2][i] = band_mul * ref_weber[2].bands[k].data[i] * s_vy * ch_gain[2];
        }

        let d_per_ch = if is_baseband {
            let mut out: [Vec<f32>; 3] = [vec![0.0; n_px], vec![0.0; n_px], vec![0.0; n_px]];
            for i in 0..n_px {
                let log_l = log_l_bkg_band[i];
                let s_a = sensitivity_corrected_scalar(rho, log_l, channels[0]);
                let s_rg = sensitivity_corrected_scalar(rho, log_l, channels[1]);
                let s_vy = sensitivity_corrected_scalar(rho, log_l, channels[2]);
                let diff_a = dis_weber[0].bands[k].data[i] - ref_weber[0].bands[k].data[i];
                let diff_rg = dis_weber[1].bands[k].data[i] - ref_weber[1].bands[k].data[i];
                let diff_vy = dis_weber[2].bands[k].data[i] - ref_weber[2].bands[k].data[i];
                out[0][i] = diff_a.abs() * s_a;
                out[1][i] = diff_rg.abs() * s_rg;
                out[2][i] = diff_vy.abs() * s_vy;
            }
            out
        } else {
            mult_mutual_band(&t_p_per_ch, &r_p_per_ch, bw, bh)
        };
        d_bands.push(d_per_ch);
    }
    d_bands
}

#[test]
fn diag_d_bands_per_band_at_73x91_odd() {
    // Compare GPU-computed D bands against scalar D bands. This will
    // tell us WHICH band the divergence first appears in, narrowing
    // down the kernel responsible.
    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, dist_b) = synth_pair(w, h);

    let gpu_d = cvvdp
        .compute_dkl_d_bands(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_d_bands on cpu");
    let cpu_d = scalar_d_bands(&ref_b, &dist_b, w as usize, h as usize, display, ppd);

    assert_eq!(gpu_d.len(), cpu_d.len(), "n_levels mismatch");
    eprintln!("[diag] n_levels = {}", gpu_d.len());

    for k in 0..gpu_d.len() {
        for c in 0..3 {
            let g = &gpu_d[k][c];
            let s = &cpu_d[k][c];
            assert_eq!(g.len(), s.len(), "band {k} channel {c} len mismatch");

            let max_err = g
                .iter()
                .zip(s)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let mean_g = g.iter().sum::<f32>() / g.len() as f32;
            let mean_s = s.iter().sum::<f32>() / s.len() as f32;

            // Find argmax of the divergence
            let mut max_i = 0;
            let mut max_d = 0.0_f32;
            for (i, (&a, &b)) in g.iter().zip(s).enumerate() {
                let d = (a - b).abs();
                if d > max_d {
                    max_d = d;
                    max_i = i;
                }
            }
            eprintln!(
                "[diag] band k={k} ch={c}: max_err={max_err:.6} mean_gpu={mean_g:.6} mean_scl={mean_s:.6} argmax_i={max_i} gpu[i]={:.6} scl[i]={:.6}",
                g[max_i], s[max_i],
            );
        }
    }
}

#[test]
fn diag_dkl_planes_at_73x91_odd() {
    // First sanity check: just the sRGB->DKL color stage. If this
    // diverges, the color kernel is the culprit. If it matches,
    // we move further down the chain.
    use cvvdp::kernels::color::display_byte_to_dkl_scalar;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, _dist_b) = synth_pair(w, h);
    let gpu_planes = cvvdp
        .compute_dkl_planes(&ref_b)
        .expect("compute_dkl_planes");
    let n = (w * h) as usize;
    let mut scl_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) =
            display_byte_to_dkl_scalar(ref_b[i * 3], ref_b[i * 3 + 1], ref_b[i * 3 + 2], display);
        scl_planes[0][i] = a;
        scl_planes[1][i] = rg;
        scl_planes[2][i] = vy;
    }
    for c in 0..3 {
        let max_err = gpu_planes[c]
            .iter()
            .zip(&scl_planes[c])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[diag] dkl plane ch={c}: max_err={max_err:.6}");
    }
}

#[test]
fn diag_weber_pyramid_at_73x91_odd() {
    // Compare GPU weber pyramid against scalar weber pyramid.
    // If the weber pyramid is already broken, the cause is in
    // pyramid kernels (downscale + upscale + subtract_weber). If
    // weber is fine, the bug is in CSF / masking.
    use cvvdp::kernels::color::display_byte_to_dkl_scalar;
    use cvvdp::kernels::pyramid::weber_contrast_pyr_dec_scalar;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, _dist_b) = synth_pair(w, h);

    // GPU weber pyramid
    let gpu_weber = cvvdp.compute_dkl_weber_pyramid(&ref_b).expect("gpu weber");

    // Scalar weber pyramid (one per channel, with channel 0 = luminance as backg)
    let n = (w as usize) * (h as usize);
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) =
            display_byte_to_dkl_scalar(ref_b[i * 3], ref_b[i * 3 + 1], ref_b[i * 3 + 2], display);
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }
    let n_levels_query =
        cvvdp::kernels::pyramid::band_frequencies(ppd, w as usize, h as usize).len();
    let scl_weber: [_; 3] = [
        weber_contrast_pyr_dec_scalar(
            &planes[0],
            &planes[0],
            w as usize,
            h as usize,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &planes[1],
            &planes[0],
            w as usize,
            h as usize,
            n_levels_query,
        ),
        weber_contrast_pyr_dec_scalar(
            &planes[2],
            &planes[0],
            w as usize,
            h as usize,
            n_levels_query,
        ),
    ];

    let (gpu_bands, gpu_logl) = gpu_weber;
    eprintln!(
        "[diag] n_levels: gpu={} scl={}",
        gpu_bands.len(),
        scl_weber[0].bands.len()
    );
    let n_levels = gpu_bands.len().min(scl_weber[0].bands.len());
    for k in 0..n_levels {
        for c in 0..3 {
            let g = &gpu_bands[k][c];
            let s = &scl_weber[c].bands[k].data;
            let bw = scl_weber[c].bands[k].w;
            let bh = scl_weber[c].bands[k].h;
            assert_eq!(
                g.len(),
                s.len(),
                "band {k} ch {c} len mismatch (gpu {} vs scl {})",
                g.len(),
                s.len()
            );

            let max_err = g
                .iter()
                .zip(s)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let mean_g = g.iter().sum::<f32>() / g.len() as f32;
            let mean_s = s.iter().sum::<f32>() / s.len() as f32;
            eprintln!(
                "[diag] weber band k={k} ch={c} {}x{}: max_err={max_err:.6} mean_g={mean_g:.6} mean_s={mean_s:.6}",
                bw, bh,
            );
        }
        // also log_l_bkg comparison
        let g_logl = &gpu_logl[k];
        let s_logl = &scl_weber[0].log_l_bkg[k];
        let max_err = g_logl
            .iter()
            .zip(s_logl)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!("[diag] weber band k={k} log_l_bkg: max_err={max_err:.6}");
    }
}

#[test]
fn diag_gauss_pyramid_at_73x91_odd() {
    // Build the Gaussian pyramid (no subtract, no weber) at 73x91 and
    // compare to scalar reference. If gauss is wrong, it's a downscale
    // issue, but we already showed downscale is fine in diag_downscale_*.
    // If gauss is fine, the bug is in upscale/subtract/weber.
    use cvvdp::kernels::color::display_byte_to_dkl_scalar;
    use cvvdp::kernels::pyramid::gausspyr_reduce_scalar;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, _dist_b) = synth_pair(w, h);

    let gpu_gauss = cvvdp.compute_dkl_gauss_pyramid(&ref_b).expect("gpu gauss");

    let n = (w as usize) * (h as usize);
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) =
            display_byte_to_dkl_scalar(ref_b[i * 3], ref_b[i * 3 + 1], ref_b[i * 3 + 2], display);
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }

    // Build scalar Gauss pyramid for each channel
    let n_levels = gpu_gauss.len();
    eprintln!("[diag] gauss n_levels = {n_levels}");
    let mut scl_per_ch: [Vec<(Vec<f32>, usize, usize)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for c in 0..3 {
        let mut cur = planes[c].clone();
        let mut cw = w as usize;
        let mut ch = h as usize;
        scl_per_ch[c].push((cur.clone(), cw, ch));
        for _ in 1..n_levels {
            let mut next = Vec::new();
            let (dw, dh) = gausspyr_reduce_scalar(&cur, cw, ch, &mut next);
            scl_per_ch[c].push((next.clone(), dw, dh));
            cur = next;
            cw = dw;
            ch = dh;
        }
    }

    for k in 0..n_levels {
        for c in 0..3 {
            let g = &gpu_gauss[k][c];
            let (s, sw, sh) = &scl_per_ch[c][k];
            assert_eq!(g.len(), s.len(), "band {k} ch {c} len mismatch");
            let max_err = g
                .iter()
                .zip(s)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let mean_g = g.iter().sum::<f32>() / g.len() as f32;
            let mean_s = s.iter().sum::<f32>() / s.len() as f32;
            eprintln!(
                "[diag] gauss k={k} ch={c} {}x{}: max_err={max_err:.6} mean_g={mean_g:.6} mean_s={mean_s:.6}",
                sw, sh
            );
        }
    }
}

#[test]
fn diag_laplacian_pyramid_at_73x91_odd() {
    // Build the Laplacian pyramid (downscale + upscale + subtract,
    // BUT without weber contrast) at 73x91 and compare to scalar.
    // Pinpoints whether the upscale or subtract are causing the
    // divergence (the weber kernel is upstream of pyramid).
    use cvvdp::kernels::color::display_byte_to_dkl_scalar;
    use cvvdp::kernels::pyramid::laplacian_pyramid_dec_scalar;

    let client = Backend::client(&Default::default());
    let (w, h) = (73u32, 91u32);
    let display = DisplayModel::STANDARD_4K;
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new on cubecl-cpu");

    let (ref_b, _dist_b) = synth_pair(w, h);

    let gpu_lap = cvvdp
        .compute_dkl_laplacian_pyramid(&ref_b)
        .expect("gpu lap");

    let n = (w as usize) * (h as usize);
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) =
            display_byte_to_dkl_scalar(ref_b[i * 3], ref_b[i * 3 + 1], ref_b[i * 3 + 2], display);
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }

    let n_levels_query = cvvdp::kernels::pyramid::band_frequencies(
        DisplayGeometry::STANDARD_4K.pixels_per_degree(),
        w as usize,
        h as usize,
    )
    .len();

    let scl_lap: [_; 3] = [
        laplacian_pyramid_dec_scalar(&planes[0], w as usize, h as usize, n_levels_query),
        laplacian_pyramid_dec_scalar(&planes[1], w as usize, h as usize, n_levels_query),
        laplacian_pyramid_dec_scalar(&planes[2], w as usize, h as usize, n_levels_query),
    ];

    eprintln!(
        "[diag] lap gpu n_levels={} scl n_levels={}",
        gpu_lap.len(),
        scl_lap[0].len()
    );

    let n_levels = gpu_lap.len().min(scl_lap[0].len());
    for k in 0..n_levels {
        for c in 0..3 {
            let g = &gpu_lap[k][c];
            let s = &scl_lap[c][k].data;
            let sw = scl_lap[c][k].w;
            let sh = scl_lap[c][k].h;
            assert_eq!(g.len(), s.len(), "band {k} ch {c} len mismatch");
            let max_err = g
                .iter()
                .zip(s)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let mean_g = g.iter().sum::<f32>() / g.len() as f32;
            let mean_s = s.iter().sum::<f32>() / s.len() as f32;
            eprintln!(
                "[diag] lap k={k} ch={c} {}x{}: max_err={max_err:.6} mean_g={mean_g:.6} mean_s={mean_s:.6}",
                sw, sh
            );
        }
    }
}

/// Drive `downscale_tiled_kernel` on cpu and compare to scalar.
/// This is the LDS-tiled variant used by `compute_dkl_gauss_pyramid`
/// via `_reduce_gauss_pyramid_tiled`.
fn diag_downscale_tiled_on_cpu(sw: u32, sh: u32) -> (Vec<f32>, Vec<f32>, f32) {
    let client = Backend::client(&Default::default());
    let nsrc = (sw * sh) as usize;
    let dw = sw.div_ceil(2);
    let dh = sh.div_ceil(2);
    let ndst = (dw * dh) as usize;

    let src: Vec<f32> = (0..nsrc)
        .map(|i| {
            let y = i / sw as usize;
            let x = i - y * sw as usize;
            ((x % 8 + y % 8) as f32) * 0.5
        })
        .collect();
    let dst = vec![0.0_f32; ndst];

    let src_h = client.create_from_slice(f32::as_bytes(&src));
    let dst_h = client.create_from_slice(f32::as_bytes(&dst));

    let cube_dim = CubeDim::new_2d(DOWNSCALE_TILED_BLOCK_DIM, DOWNSCALE_TILED_BLOCK_DIM);
    let cube_count = CubeCount::Static(
        dw.div_ceil(DOWNSCALE_TILED_BLOCK_DIM),
        dh.div_ceil(DOWNSCALE_TILED_BLOCK_DIM),
        1,
    );
    unsafe {
        downscale_tiled_kernel::launch::<Backend>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(src_h.clone(), nsrc),
            ArrayArg::from_raw_parts(dst_h.clone(), ndst),
            sw,
            sh,
            dw,
            dh,
        );
    }
    let out_bytes = client.read_one(dst_h.clone()).expect("read dst");
    let gpu_out: Vec<f32> = f32::from_bytes(&out_bytes).to_vec();

    let mut cpu_out = Vec::new();
    gausspyr_reduce_scalar(&src, sw as usize, sh as usize, &mut cpu_out);
    assert_eq!(cpu_out.len(), gpu_out.len(), "len mismatch sw={sw} sh={sh}");

    let max_err = gpu_out
        .iter()
        .zip(&cpu_out)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    (gpu_out, cpu_out, max_err)
}

#[test]
fn diag_downscale_tiled_cpu_at_73x91_odd() {
    let (_g, _c, e) = diag_downscale_tiled_on_cpu(73, 91);
    eprintln!("diag tiled 73×91: max-abs = {e:.6}");
    assert!(e < 1e-5, "tiled 73×91 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_tiled_cpu_sweep_sizes() {
    // Sweep across sizes to find the threshold where the tiled
    // kernel starts diverging on cpu backend.
    let sizes = [
        (32u32, 32u32), // 1×1 workgroups for dst 16×16
        (33, 33),
        (34, 34), // mixed-parity reduce starts at non-pow-2
        (40, 40), // dst 20x20: 2x2 workgroups
        (48, 48), // dst 24x24: 2x2 workgroups
        (56, 56),
        (64, 64), // dst 32x32: 2x2 workgroups
        (65, 65), // odd-W odd-H
        (66, 66),
        (72, 72),
        (73, 73),
        (73, 91),
        (74, 92), // both even — multiple workgroups
    ];
    for (w, h) in sizes {
        let (_g, _c, e) = diag_downscale_tiled_on_cpu(w, h);
        let dw = w.div_ceil(2);
        let dh = h.div_ceil(2);
        let n_wg_x = dw.div_ceil(16);
        let n_wg_y = dh.div_ceil(16);
        eprintln!(
            "[diag] tiled {}×{} → {}×{}, wg={}×{}: max_err={:.6}",
            w, h, dw, dh, n_wg_x, n_wg_y, e
        );
    }
}

#[test]
fn diag_downscale_tiled_cpu_at_32x32() {
    // Even-dim sanity check
    let (_g, _c, e) = diag_downscale_tiled_on_cpu(32, 32);
    eprintln!("diag tiled 32×32: max-abs = {e:.6}");
    assert!(e < 1e-5, "tiled 32×32 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_tiled_cpu_at_8x8_even() {
    let (g, c, e) = diag_downscale_tiled_on_cpu(8, 8);
    eprintln!("diag tiled 8×8: max-abs = {e:.6}");
    eprintln!("gpu = {g:?}");
    eprintln!("cpu = {c:?}");
    assert!(e < 1e-5, "tiled 8×8 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_tiled_cpu_at_7x7_odd() {
    let (g, c, e) = diag_downscale_tiled_on_cpu(7, 7);
    eprintln!("diag tiled 7×7: max-abs = {e:.6}");
    eprintln!("gpu = {g:?}");
    eprintln!("cpu = {c:?}");
    assert!(e < 1e-5, "tiled 7×7 cpu downscale max-abs = {e}");
}

#[test]
fn diag_downscale_cpu_at_46x37_mixed_parity() {
    // 46×37 — second-level reduce of 91 → ceil(91/2)=46, and a
    // mixed-parity case (sw even, sh odd). This is the exact level
    // where pycvvdp's bug-compat delta fires in the 73×91 chain.
    let (gpu, cpu, e) = diag_downscale_on_cpu(46, 37);
    eprintln!(
        "diag 46×37: max-abs = {e:.6}\ngpu_last_col={:?}\ncpu_last_col={:?}",
        (0..(46u32.div_ceil(2)) as usize)
            .step_by(46usize.div_ceil(2).saturating_sub(1).max(1))
            .take(3)
            .collect::<Vec<_>>(),
        (0..(46u32.div_ceil(2)) as usize)
            .step_by(46usize.div_ceil(2).saturating_sub(1).max(1))
            .take(3)
            .collect::<Vec<_>>(),
    );
    // Extract right column for easier diff inspection
    let dw = 46u32.div_ceil(2);
    let dh = 37u32.div_ceil(2);
    for dy in 0..dh as usize {
        let i = dy * dw as usize + (dw as usize - 1);
        let g = gpu[i];
        let c = cpu[i];
        eprintln!("  right-col dy={dy}: gpu={g:.6} cpu={c:.6} d={:.6}", g - c);
    }
    assert!(e < 1e-5, "46×37 cpu downscale max-abs = {e}");
}
