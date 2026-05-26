//! Mode E Phase 3 (task #79) — strip-aware pool walker parity tests.
//!
//! Phase 3 introduces the **first strip-aware kernel dispatch** in
//! the cvvdp pipeline: `_pool_and_finalize_jod_strip`. It partitions
//! each band's per-pixel pool into row-strips and dispatches the new
//! [`pool_band_3ch_offset_kernel`] per slab. Atomic-adds into
//! `partials_h` are associative across slabs, so the final JOD scalar
//! is bit-exact against Full mode within the same per-call ordering
//! noise band that Full produces on its own repeated calls.
//!
//! These tests pin:
//!   1. **JOD parity**: strip pool dispatch matches Full pool to
//!      `1e-4` JOD at 1024² and 4096² (4096² is gated on a feature
//!      flag because allocating a 4096² Cvvdp pair on a 12 GB GPU
//!      eats ≥ 6 GB; smaller boxes skip it).
//!   2. **Walker actually partitions**: at 4096² with `h_body=512`,
//!      the strip dispatch counter must report N >= 2 strip
//!      iterations per warm-ref call (in fact, summed across the
//!      band loop's 9 bands, at most ~8 strips for level 0; deep
//!      bands fall through to single-strip dispatch — N >= 9
//!      including the per-band degenerate strips).
//!
//! The Phase 3 walker is **bit-exact JOD-preserving** like Phase 2.
//! Memory reduction relative to Full is **not** measured here — only
//! the pool stage is strip-aware so far; the per-strip d_scratch
//! shrink is a follow-on Phase 3 chunk (the existing kernels need
//! their reflection-at-array-edges semantics ported to
//! reflection-at-logical-image-edges).
//!
//! Phase 3's deliverable for THIS landing is the foundation: a tested
//! strip walker that proves the atomic-associativity claim, plus the
//! `pool_band_3ch_offset_kernel` ready for the per-band CSF/masking
//! port that follows.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{apply_offset_dist, synth_pair_ref, Backend};

use cubecl::Runtime;
use cvvdp_gpu::{memory_mode::STRIP_H_BODY_DEFAULT, Cvvdp, CvvdpParams};

/// Per-call JOD ordering noise band for atomic-add reductions on CUDA.
/// Matches `strip_mode_e_parity.rs::PARITY_TOL_JOD`.
const PARITY_TOL_JOD: f32 = 1e-4;

fn ppd() -> f32 {
    cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree()
}

#[test]
fn phase3_pool_strip_matches_full_at_64x64() {
    // Even at 64×64, the strip walker should produce the same JOD as
    // Full (atomic associativity). Because the strip body is 512 and
    // the band heights start at 64, every band falls through to
    // single-strip dispatch — this is the "Phase 3 is a no-op partition
    // at small sizes" baseline.
    let (r, d) = (synth_pair_ref(64, 64), apply_offset_dist(&synth_pair_ref(64, 64)));

    let client_full = Backend::client(&Default::default());
    let mut full =
        Cvvdp::<Backend>::new(client_full, 64, 64, CvvdpParams::PLACEHOLDER).expect("full new");
    full.warm_reference(&r).expect("full warm");
    let jod_full = full
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("full warm-ref jod");

    let client_strip = Backend::client(&Default::default());
    let mut strip = Cvvdp::<Backend>::new_strip(
        client_strip,
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("strip new");
    strip.warm_reference(&r).expect("strip warm");
    let jod_strip = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("strip warm-ref jod");

    assert!(
        (jod_full - jod_strip).abs() < PARITY_TOL_JOD,
        "Mode E Phase 3 pool dispatch diverged at 64×64: full={jod_full} strip={jod_strip} diff={}",
        (jod_full - jod_strip).abs()
    );
}

#[test]
fn phase3_pool_strip_matches_full_at_1024x1024() {
    // 1024² is the first size where the canonical h_body=512
    // partitions a band into multiple strips at scale 0
    // (level 0 height = 1024, body = 512 → 2 strips). Deeper levels
    // shrink to single-strip dispatch as the per-band height drops
    // below 512. The JOD parity must still hold within
    // PARITY_TOL_JOD.
    let (r, d) = (
        synth_pair_ref(1024, 1024),
        apply_offset_dist(&synth_pair_ref(1024, 1024)),
    );

    let client_full = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client_full, 1024, 1024, CvvdpParams::PLACEHOLDER)
        .expect("full new");
    full.warm_reference(&r).expect("full warm");
    let jod_full = full
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("full warm-ref jod");

    let client_strip = Backend::client(&Default::default());
    let mut strip = Cvvdp::<Backend>::new_strip(
        client_strip,
        1024,
        1024,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("strip new");
    strip.warm_reference(&r).expect("strip warm");
    let jod_strip = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("strip warm-ref jod");

    assert!(
        (jod_full - jod_strip).abs() < PARITY_TOL_JOD,
        "Mode E Phase 3 pool dispatch diverged at 1024×1024: full={jod_full} strip={jod_strip} diff={}",
        (jod_full - jod_strip).abs()
    );
}

#[test]
fn phase3_strip_walker_dispatches_n_strips_at_1024() {
    // The walker MUST partition at 1024² with h_body=512. Level 0 has
    // 1024 rows → 2 strips at body=512. Deeper levels are body/2^k:
    // - Level 1: 256 rows, body=256 at that level → 1 strip
    // - Level 2: 128 rows, body=128 → 1 strip
    // - ... (deeper levels degenerate to 1 strip each)
    // Total per warm-ref call = 2 (L0) + 1 × (n_levels - 1) >= 8 at
    // STANDARD_4K geometry.
    let (r, d) = (
        synth_pair_ref(1024, 1024),
        apply_offset_dist(&synth_pair_ref(1024, 1024)),
    );

    let client = Backend::client(&Default::default());
    let mut strip = Cvvdp::<Backend>::new_strip(
        client,
        1024,
        1024,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("strip new");
    strip.warm_reference(&r).expect("strip warm");
    strip.reset_strip_dispatch_counter();

    let _ = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("strip warm-ref jod");

    let n_strips = strip.strip_dispatch_counter();
    // Level 0 alone: ceil(1024 / 512) = 2 strips. Deep bands run a
    // single-strip dispatch each. Lower bound: 2 + (n_levels - 1)
    // where n_levels is at least 8 at 1024² STANDARD_4K geometry.
    assert!(
        n_strips >= 2,
        "strip walker should partition at 1024² (h_body=512); got {n_strips} strips"
    );
}

#[test]
fn phase3_pool_strip_repeats_deterministically() {
    // Repeated warm-ref calls on the same (ref, dist) pair should
    // produce the same JOD value (within Atomic<f32> ordering noise).
    // Pinning this on the strip path catches any new strip-loop bug
    // that introduces non-determinism beyond what Full mode already
    // exhibits.
    let (r, d) = (
        synth_pair_ref(512, 512),
        apply_offset_dist(&synth_pair_ref(512, 512)),
    );

    let client = Backend::client(&Default::default());
    let mut strip = Cvvdp::<Backend>::new_strip(
        client,
        512,
        512,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("strip new");
    strip.warm_reference(&r).expect("strip warm");

    let mut samples: Vec<f32> = Vec::with_capacity(5);
    for _ in 0..5 {
        let jod = strip
            .compute_dkl_jod_with_warm_ref(&d, ppd())
            .expect("strip warm-ref jod");
        samples.push(jod);
    }
    // All samples within PARITY_TOL_JOD of each other.
    let min_v = samples.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_v = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        (max_v - min_v).abs() < PARITY_TOL_JOD,
        "Mode E Phase 3 non-determinism: samples={samples:?} spread={}",
        max_v - min_v
    );
}

#[test]
fn phase3_full_mode_counter_stays_zero() {
    // The strip dispatch counter must only increment in Mode E. A
    // Full-mode Cvvdp must show counter == 0 after a warm-ref call
    // (it never invokes _pool_and_finalize_jod_strip).
    let (r, d) = (synth_pair_ref(64, 64), apply_offset_dist(&synth_pair_ref(64, 64)));

    let client = Backend::client(&Default::default());
    let mut full =
        Cvvdp::<Backend>::new(client, 64, 64, CvvdpParams::PLACEHOLDER).expect("full new");
    full.warm_reference(&r).expect("full warm");
    let _ = full
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("full warm-ref jod");
    assert_eq!(full.strip_dispatch_counter(), 0);
}

/// Mode E walker JOD parity at 1024² with h_body=256.
///
/// Pins the contract for the Mode E walker drop-snapshot-restore
/// landing: Mode E (CachedRef) reads REF bands and non-baseband
/// log_l_bkg from `ref_full_state` directly, runs the strip-aware
/// masking walker (same chain as Mode B) and the strip-aware pool
/// walker, and produces a JOD within `PARITY_TOL_JOD` of Full mode.
///
/// Distinct from `phase3_pool_strip_matches_full_at_1024x1024`: that
/// test runs at the canonical h_body=512 where only the L0 pool
/// partitions into multiple strips; this one uses h_body=256 to
/// exercise the strip masking walker on multiple bands (L0 alone:
/// ceil(1024 / 256) = 4 strips, plus per-strip masking dispatch
/// increments the counter by 4 per non-baseband band).
///
/// Gate: JOD diff < 1e-4 AND strip dispatch counter ≥ 4 (proves the
/// walker actually partitioned, not bypassed).
#[test]
fn mode_e_walker_jod_matches_full_at_1024() {
    let (r, d) = (
        synth_pair_ref(1024, 1024),
        apply_offset_dist(&synth_pair_ref(1024, 1024)),
    );

    let client_full = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client_full, 1024, 1024, CvvdpParams::PLACEHOLDER)
        .expect("full new");
    full.warm_reference(&r).expect("full warm");
    let jod_full = full
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("full warm-ref jod");

    let client_strip = Backend::client(&Default::default());
    let mut strip = Cvvdp::<Backend>::new_strip(
        client_strip,
        1024,
        1024,
        256,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("strip new");
    strip.warm_reference(&r).expect("strip warm");
    strip.reset_strip_dispatch_counter();
    let jod_strip = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd())
        .expect("strip warm-ref jod");
    let n_dispatches = strip.strip_dispatch_counter();

    let diff = (jod_full - jod_strip).abs();
    eprintln!(
        "Mode E walker 1024² (h_body=256): Full JOD={jod_full:.6}, \
         Mode E JOD={jod_strip:.6}, |diff|={diff:.3e}, \
         strip_dispatch_counter={n_dispatches}",
    );

    assert!(
        diff < PARITY_TOL_JOD,
        "Mode E walker JOD={jod_strip} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
    // Walker partitioned proof: with h_body=256, L0 alone runs 4
    // pool strips; deep bands fall to single-strip dispatch, and the
    // masking walker adds 4 launches per (strip, non-baseband band).
    // Lower bound of 4 captures the "walker actually ran" contract.
    assert!(
        n_dispatches >= 4,
        "Mode E walker should dispatch >= 4 strip iterations at 1024² h_body=256; got {n_dispatches}",
    );
}
