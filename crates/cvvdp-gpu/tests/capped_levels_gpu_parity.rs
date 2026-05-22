//! GPU-side capped-pyramid-depth parity tests.
//!
//! Companion to `capped_levels_parity.rs` (host-scalar) — pins the
//! actual GPU `Cvvdp::new_with_geometry_and_cap` constructor against
//! the same fixtures. Runtime requirement: a real GPU backend
//! (CUDA/WGPU/HIP). Skipped at compile time when none is enabled.
//!
//! Run with:
//!
//!     cargo test -p cvvdp-gpu --features cubecl-types \
//!         --test capped_levels_gpu_parity -- --nocapture

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use cvvdp_gpu::Cvvdp;

#[path = "common/mod.rs"]
mod common;

use common::Backend;

const TOLERANCE: f32 = 0.005;

#[test]
fn gpu_cap_8_matches_uncapped_at_12mp_synth() {
    // 12 MP fixture: natural_n_levels = 9. Cap=8 drops one coarse
    // band; per the host-scalar sweep, the JOD shift is
    // 0.000051 — well within the 0.005 gate.
    //
    // This test pins the GPU path's behavior matches the host_scalar
    // path's: both produce the same capped JOD when given the same
    // cap depth. (The GPU path may have its own f32 noise floor on
    // top of the host's, but the cap mechanism — clamping n_levels —
    // is identical between paths.)
    let client = Backend::client(&Default::default());
    let (w, h) = (4000u32, 3000u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let (ref_srgb, dist_srgb) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    // Reference: uncapped GPU JOD.
    let mut cvvdp_full = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp (uncapped)");
    let jod_full = cvvdp_full
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod uncapped");

    // Cap=8 GPU JOD via the new constructor.
    let mut cvvdp_cap = Cvvdp::<Backend>::new_with_geometry_and_cap(
        client,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        geom,
        Some(8),
    )
    .expect("new Cvvdp (cap=8)");
    let jod_cap8 = cvvdp_cap
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod cap=8");

    let pycvvdp_golden = common::pycvvdp_synth_golden_jod("synth_4000x3000");
    let diff_cap_vs_golden = (jod_cap8 - pycvvdp_golden).abs();
    let diff_cap_vs_full = (jod_cap8 - jod_full).abs();

    eprintln!(
        "12mp synth: full={jod_full:.4}, cap=8={jod_cap8:.4}, pycvvdp={pycvvdp_golden:.4}, \
         cap_vs_full={diff_cap_vs_full:.4}, cap_vs_golden={diff_cap_vs_golden:.4}"
    );

    // Gate 1: cap=8 JOD vs pycvvdp golden ≤ 0.005.
    assert!(
        diff_cap_vs_golden < TOLERANCE,
        "12mp cap=8 JOD {jod_cap8:.4} drifts from pycvvdp golden {pycvvdp_golden:.4} by \
         {diff_cap_vs_golden:.4} > {TOLERANCE:.4}"
    );

    // Gate 2: GPU cap=8 vs GPU uncapped should differ by < 0.001
    // (the host_scalar sweep measured the cap shift as 0.000051; the
    // GPU adds its own f32 noise but stays well below 0.005).
    assert!(
        diff_cap_vs_full < 0.001,
        "12mp GPU cap=8 vs uncapped diff {diff_cap_vs_full:.6} exceeded \
         0.001 — the cap-mechanism may not be matching the host_scalar reference"
    );
}

#[test]
fn gpu_cap_8_matches_pycvvdp_at_1024x1024() {
    // Smaller-fixture pin to catch a cap-mechanism regression that
    // 12mp might absorb in its dominant per-band variance. natural_n=9
    // at 1024² standard_4k; cap=8 should drift 0.000074 vs golden.
    let client = Backend::client(&Default::default());
    let (w, h) = (1024u32, 1024u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let (ref_srgb, dist_srgb) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let mut cvvdp = Cvvdp::<Backend>::new_with_geometry_and_cap(
        client,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        geom,
        Some(8),
    )
    .expect("new Cvvdp (cap=8)");
    let jod = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod cap=8");

    let pycvvdp_golden = common::pycvvdp_synth_golden_jod("synth_1024x1024_offset");
    let diff = (jod - pycvvdp_golden).abs();
    eprintln!(
        "1024² cap=8: gpu={jod:.4}, pycvvdp={pycvvdp_golden:.4}, diff={diff:.4}"
    );
    assert!(
        diff < TOLERANCE,
        "1024² cap=8 JOD {jod:.4} drifts from pycvvdp golden {pycvvdp_golden:.4} by \
         {diff:.4} > {TOLERANCE:.4}"
    );
}

#[test]
fn gpu_cap_none_matches_uncapped_jod_at_1024x1024() {
    // cap=None on the new constructor must produce byte-identical JOD
    // to `Cvvdp::new` (the uncapped public entry point). This pins
    // that the cap-handling clause is a no-op when None — without
    // this, a refactor that accidentally re-routes the None branch
    // through some other code path would silently change the
    // production score.
    let client = Backend::client(&Default::default());
    let (w, h) = (1024u32, 1024u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let (ref_srgb, dist_srgb) = common::synth_pair_with_offset_dist(w as usize, h as usize);

    let mut cvvdp_full = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp (full)");
    let jod_full = cvvdp_full
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod full");

    let mut cvvdp_none = Cvvdp::<Backend>::new_with_geometry_and_cap(
        client,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        geom,
        None,
    )
    .expect("new Cvvdp (cap=None)");
    let jod_none = cvvdp_none
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod cap=None");

    let diff = (jod_full - jod_none).abs();
    eprintln!(
        "1024² cap=None vs full: full={jod_full:.6}, none={jod_none:.6}, diff={diff:.6}"
    );
    assert!(
        diff < 1e-6,
        "1024² cap=None GPU JOD {jod_none:.6} differs from uncapped {jod_full:.6} by \
         {diff:.6} — the cap=None branch is not a no-op"
    );
}
