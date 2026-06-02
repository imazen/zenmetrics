//! Mode E (task #79) strip-mode parity tests.
//!
//! Phase 1+2 of Mode E ships a re-introduced
//! [`MemoryMode::Strip { h_body }`] variant where the cached-ref
//! state lives in dedicated [`RefFullState`] buffers (full-image,
//! survives across one-shot dispatches). The dist-side dispatch is
//! still Full-mode for Phase 2 — Phase 3 will shrink it to a
//! per-strip working set. These tests pin the **JOD-preservation
//! contract**: strip-mode `compute_with_warm_ref` must produce JOD
//! values that match Full-mode `compute_with_warm_ref` within the
//! documented Atomic<f32> reduction-order noise band.
//!
//! Tolerance: `1e-4` absolute JOD. cvvdp's
//! `compute_dkl_jod_is_deterministic_across_repeated_calls` test
//! (in `pipeline_score.rs`) shows the per-call drift band sits well
//! below this tolerance on CUDA.
//!
//! These tests skip when no compatible cubecl runtime is enabled.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{Backend, synth_pair_ref, synth_pair_with_offset_dist};

use cubecl::Runtime;
use cvvdp_gpu::{Cvvdp, CvvdpParams, MemoryMode, memory_mode::STRIP_H_BODY_DEFAULT};

const PARITY_TOL_JOD: f32 = 1e-4;

#[test]
fn strip_mode_is_strip_mode_reports_true_after_new_strip() {
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_strip(
        client,
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip");
    assert!(cvvdp.is_strip_mode());
    assert_eq!(cvvdp.strip_h_body(), Some(STRIP_H_BODY_DEFAULT));
    assert!(!cvvdp.has_reference()); // fresh state
}

#[test]
fn full_mode_is_strip_mode_reports_false() {
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new(client, 64, 64, CvvdpParams::PLACEHOLDER).expect("new");
    assert!(!cvvdp.is_strip_mode());
    assert_eq!(cvvdp.strip_h_body(), None);
}

#[test]
fn strip_mode_warm_ref_then_clear_reset_round_trip() {
    let client = Backend::client(&Default::default());
    let mut cvvdp = cvvdp_64x64_strip(&client);
    assert!(!cvvdp.has_reference());

    let r = synth_pair_ref(64, 64);
    cvvdp.warm_reference(&r).expect("warm_reference");
    assert!(cvvdp.has_reference());

    // Re-arming with a different ref overwrites the prior cache.
    let r2: Vec<u8> = r.iter().map(|b| b.wrapping_add(17)).collect();
    cvvdp.warm_reference(&r2).expect("re-warm");
    assert!(cvvdp.has_reference());
}

#[test]
fn mode_e_matches_full_64x64() {
    // The canonical parity contract: at the same (ref, dist) pair,
    // strip-mode and Full-mode should produce the same JOD scalar
    // within the Atomic<f32> reduction-order band.
    let client_full = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client_full, 64, 64, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let client_strip = Backend::client(&Default::default());
    let mut strip = cvvdp_64x64_strip(&client_strip);

    let (r, d) = synth_pair_with_offset_dist(64, 64);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    full.warm_reference(&r).expect("warm_reference full");
    strip.warm_reference(&r).expect("warm_reference strip");

    let jod_full = full
        .compute_dkl_jod_with_warm_ref(&d, ppd)
        .expect("warm-ref full");
    let jod_strip = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd)
        .expect("warm-ref strip");

    let diff = (jod_full - jod_strip).abs();
    assert!(
        diff <= PARITY_TOL_JOD,
        "Mode E parity broken at 64×64: full = {jod_full}, strip = {jod_strip}, |diff| = {diff}"
    );
}

#[test]
fn mode_e_survives_intervening_one_shot_dispatch() {
    // The key Mode E claim: the cached-ref state survives across
    // one-shot scoring calls because it lives in dedicated buffers.
    // In Full mode, an intervening `score` invalidates the warm
    // state (the shared bands_ref scratch gets clobbered); in
    // strip mode the warm state should still be valid afterward.
    let client = Backend::client(&Default::default());
    let mut strip = cvvdp_64x64_strip(&client);
    let (r, d) = synth_pair_with_offset_dist(64, 64);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    strip.warm_reference(&r).expect("warm_reference");
    let jod_before = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd)
        .expect("warm jod before");

    // Intervening one-shot dispatch clobbers the shared scratch.
    let _ = strip.score(&r, &d).expect("one-shot score");

    // In strip mode the cached state should still be valid AND
    // produce the same JOD as before.
    assert!(strip.has_reference());
    let jod_after = strip
        .compute_dkl_jod_with_warm_ref(&d, ppd)
        .expect("warm jod after intervening one-shot");
    let diff = (jod_before - jod_after).abs();
    assert!(
        diff <= PARITY_TOL_JOD,
        "Mode E ref cache did not survive intervening one-shot: before = {jod_before}, after = {jod_after}, |diff| = {diff}"
    );
}

#[test]
fn mode_e_matches_full_n_distortions_64x64() {
    // Mode E parity must hold across multiple distortions against
    // the same warm REF — the bread-and-butter encoder-loop pattern.
    let client_full = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client_full, 64, 64, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let client_strip = Backend::client(&Default::default());
    let mut strip = cvvdp_64x64_strip(&client_strip);

    let r = synth_pair_ref(64, 64);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    full.warm_reference(&r).expect("warm_reference full");
    strip.warm_reference(&r).expect("warm_reference strip");

    for shift in [4u8, 8, 16, 32] {
        let d: Vec<u8> = r.iter().map(|b| b.wrapping_add(shift)).collect();
        let jod_full = full
            .compute_dkl_jod_with_warm_ref(&d, ppd)
            .expect("warm-ref full");
        let jod_strip = strip
            .compute_dkl_jod_with_warm_ref(&d, ppd)
            .expect("warm-ref strip");
        let diff = (jod_full - jod_strip).abs();
        assert!(
            diff <= PARITY_TOL_JOD,
            "Mode E parity broken at shift={shift}: full = {jod_full}, strip = {jod_strip}, |diff| = {diff}"
        );
    }
}

#[test]
fn mode_e_via_new_with_memory_mode_routes_to_strip() {
    // Exercise the unified MemoryMode constructor for strip mode.
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_with_memory_mode(
        client,
        64,
        64,
        CvvdpParams::PLACEHOLDER,
        MemoryMode::Strip { h_body: None },
    )
    .expect("new_with_memory_mode Strip");
    assert!(cvvdp.is_strip_mode());
    assert_eq!(cvvdp.strip_h_body(), Some(STRIP_H_BODY_DEFAULT));
}

#[test]
fn mode_e_strip_h_body_explicit_override() {
    // Phase 8j: `h_body` must be a positive power of two per the
    // `Cvvdp::new_strip` / `new_with_memory_mode` constructor
    // contract (see `memory_mode.rs::MemoryMode::Strip` docs and
    // the `mode_e_rejects_misaligned_h_body` test below). The
    // pre-Phase-8j form of this test passed `Some(768)` (= 3 ×
    // STRIP_ALIGN), which is no longer accepted now that the
    // constructor validates the power-of-two rule directly. 1024
    // is the next valid value above the default (STRIP_H_BODY_DEFAULT
    // = STRIP_ALIGN = 256); using a non-default value still
    // exercises the "explicit override survives round-trip" contract
    // that the original test was pinning.
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_with_memory_mode(
        client,
        64,
        64,
        CvvdpParams::PLACEHOLDER,
        MemoryMode::Strip { h_body: Some(1024) }, // valid power-of-two override above default
    )
    .expect("new_with_memory_mode Strip explicit");
    assert_eq!(cvvdp.strip_h_body(), Some(1024));
}

#[test]
fn mode_e_rejects_misaligned_h_body() {
    let client = Backend::client(&Default::default());
    let r = Cvvdp::<Backend>::new_strip(client, 64, 64, 100, CvvdpParams::PLACEHOLDER);
    assert!(matches!(r, Err(cvvdp_gpu::Error::ModeUnsupported(_))));
}

#[test]
fn mode_e_rejects_zero_h_body() {
    let client = Backend::client(&Default::default());
    let r = Cvvdp::<Backend>::new_strip(client, 64, 64, 0, CvvdpParams::PLACEHOLDER);
    assert!(matches!(r, Err(cvvdp_gpu::Error::ModeUnsupported(_))));
}

#[test]
fn mode_e_compute_with_warm_ref_errors_without_warm_reference() {
    // In strip mode, calling compute_with_warm_ref before
    // warm_reference must surface NoWarmReference. Pre-task-#79
    // Full-mode behavior was equivalent; we pin the same semantics
    // for strip mode.
    let client = Backend::client(&Default::default());
    let mut strip = cvvdp_64x64_strip(&client);
    let d = vec![128u8; 64 * 64 * 3];
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let r = strip.compute_dkl_jod_with_warm_ref(&d, ppd);
    assert!(matches!(r, Err(cvvdp_gpu::Error::NoWarmReference)));
}

fn cvvdp_64x64_strip(client: &cubecl::client::ComputeClient<Backend>) -> Cvvdp<Backend> {
    Cvvdp::<Backend>::new_strip(
        client.clone(),
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("Cvvdp::new_strip 64×64")
}

// Suppress dead_code warning when running test subsets via -- --test-threads=1.
#[allow(dead_code)]
fn _ensure_helpers_used() {
    let _ = synth_pair_ref;
    let _ = synth_pair_with_offset_dist;
}
