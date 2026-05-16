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
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
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
