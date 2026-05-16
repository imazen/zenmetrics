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
    assert!(jod.is_finite(), "host_pool blur JOD must be finite, got {jod}");
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
                let sign: i16 = if ((i * 31).wrapping_add(17)) % 2 == 0 { 1 } else { -1 };
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
    cvvdp.warm_reference(&ref_black).expect("warm_reference black");
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
    cvvdp.warm_reference(&ref_gray).expect("warm_reference gray");
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
    assert!(jod.is_finite(), "host_pool warm-ref blur JOD must be finite, got {jod}");
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
                let sign: i16 = if ((i * 31).wrapping_add(17)) % 2 == 0 { 1 } else { -1 };
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
