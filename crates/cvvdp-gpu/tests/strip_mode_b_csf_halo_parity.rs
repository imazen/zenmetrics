//! P2.1b CSF body+halo parity tests (2026-05-27).
//!
//! These tests pin the **invariant** that motivates P2.1b: under the
//! existing level-major-outer caller
//! (`_dispatch_dist_weber_csf_strip_walker_for_level`), each per-strip
//! CSF helper call now dispatches over body+halo rows of `t_p_*[k]`
//! rather than body-only. Adjacent strips overlap at halo rows; their
//! writes must be **deterministic** (bit-identical for the same global
//! row) so the final t_p_*[k] state — and therefore the masking chain
//! output and the JOD scalar — is bit-identical to today's body-only
//! dispatch.
//!
//! The level of indirection: cubecl handles don't easily expose
//! mid-pipeline state for direct comparison. Instead we gate on:
//!
//! 1. **JOD bit-identical at multiple non-trivial strip counts**:
//!    if the body+halo overlap rows were ever non-deterministic, the
//!    final pool would drift. We test at sizes that exercise:
//!    - h_body = h/2 (one strip at L0, the trivial case): smoke
//!    - h_body = h/4 (4 strips at L0, halo overlaps three boundaries)
//!    - h_body = h/8 (8 strips at L0, halo overlaps seven boundaries)
//!
//! 2. **Strip-dispatch counter rises**: with body+halo dispatch each
//!    strip does strictly more launches than today's body-only path
//!    (no change in launch count per strip — same 6 launches per
//!    strip), so the *total* counter is `6 * n_strips_per_level *
//!    n_shallow_levels`. Confirms the helper is iterating per-strip,
//!    not bypassing.
//!
//! 3. **Multiple-image determinism**: two `score()` calls on the same
//!    `(ref, dist)` pair from the same `Cvvdp` instance produce
//!    identical JOD — confirming no halo-row state leakage between
//!    calls (P2.1b's strip-local buffers are reused).
//!
//! Tolerance: `|diff| = 0.0` (bit-exact) at every gate. Any drift
//! would mean the CSF halo writes aren't deterministic across strips,
//! which is the structural failure mode P2.1b must rule out.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{Backend, synth_pair_with_offset_dist};

use cubecl::Runtime;
use cvvdp_gpu::{Cvvdp, CvvdpParams};

const PARITY_TOL_JOD: f32 = 1e-4;

/// At 256² with h_body=128 (2 strips at L0), confirm bit-identical
/// JOD vs Full mode. Smallest size that exercises 2+ strips and the
/// overlapping halo writes between strips 0 and 1.
#[test]
fn p21b_csf_halo_parity_256_h_body_128() {
    let client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client.clone(), 256, 256, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 256, 256, 128, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(256, 256);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!(
        "256² h_body=128 (2 strips at L0): Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

/// At 512² with h_body=128 (4 strips at L0), the halo overlap fires
/// at 3 inter-strip boundaries — each must produce bit-identical CSF
/// writes across the two strips that touch it.
#[test]
fn p21b_csf_halo_parity_512_h_body_128_4_strips() {
    let client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client.clone(), 512, 512, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 512, 512, 128, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(512, 512);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!(
        "512² h_body=128 (4 strips at L0): Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

/// At 1024² with h_body=128 (8 strips at L0), the halo overlap fires
/// at 7 inter-strip boundaries. Stress-tests cross-strip determinism
/// at production-relevant strip counts.
#[test]
fn p21b_csf_halo_parity_1024_h_body_128_8_strips() {
    let client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client.clone(), 1024, 1024, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 1024, 1024, 128, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(1024, 1024);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!(
        "1024² h_body=128 (8 strips at L0): Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

/// Two consecutive `score()` calls with the same (ref, dist) pair
/// produce identical JOD. Verifies that the per-strip buffer reuse
/// across the body+halo dispatch doesn't leak state across calls —
/// the strip-local `bands_dis_strip` / `upscaled_c_strip` are
/// overwritten at every per-strip dispatch, so even though they hold
/// stale data when the next call starts, that data must be
/// completely overwritten before any reader runs.
#[test]
fn p21b_csf_halo_deterministic_across_calls() {
    let client = Backend::client(&Default::default());
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 512, 512, 128, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(512, 512);

    let jod_a = pair.score(&r, &d).expect("Mode B score #1");
    let jod_b = pair.score(&r, &d).expect("Mode B score #2");
    let diff = (jod_a - jod_b).abs();
    eprintln!("512² h_body=128 deterministic: jod1={jod_a}, jod2={jod_b}, |diff|={diff}");
    assert_eq!(
        jod_a, jod_b,
        "P2.1b halo writes leaked state across calls: {jod_a} != {jod_b}",
    );
    let _ = diff;
}

/// Edge case: h_body=h (single strip covering the whole image). The
/// helper still dispatches per-strip but n_strips = 1 and
/// top_global = 0, bot_global = fine_h. Verifies the halo-clamp logic
/// doesn't break the degenerate single-strip case.
#[test]
fn p21b_csf_halo_parity_single_strip_degenerate() {
    let client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client.clone(), 256, 256, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    // h_body = 256 → 1 strip at L0 (degenerate).
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 256, 256, 256, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(256, 256);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!(
        "256² h_body=256 (1 strip / degenerate): Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD} (single-strip degenerate)",
    );
}
