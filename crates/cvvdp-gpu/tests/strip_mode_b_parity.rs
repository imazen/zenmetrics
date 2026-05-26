//! Mode B (StripPair) JOD parity tests.
//!
//! Mode B is the one-shot pair stripwise variant: both ref and dist
//! walk in strips together with no full-ref cache. As of this
//! landing the walker still routes through the existing Full
//! pipeline (allocations are at full-image dims while the walker is
//! wired); the JOD value is therefore identical to Full mode by
//! construction. These tests pin that contract so a future commit
//! that shrinks allocations / introduces per-strip dispatch surfaces
//! any JOD drift immediately.
//!
//! Tolerance: `1e-4` absolute JOD — same band as Mode E parity.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{synth_pair_with_offset_dist, Backend};

use cubecl::Runtime;
use cvvdp_gpu::{memory_mode::STRIP_H_BODY_DEFAULT, Cvvdp, CvvdpParams, MemoryMode};

const PARITY_TOL_JOD: f32 = 1e-4;

fn cvvdp_64x64_strip_pair(
    client: &cubecl::prelude::ComputeClient<Backend>,
) -> Cvvdp<Backend> {
    Cvvdp::<Backend>::new_strip_pair(
        client.clone(),
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip_pair")
}

#[test]
fn strip_pair_is_strip_pair_mode_reports_true_after_new() {
    let client = Backend::client(&Default::default());
    let cvvdp = cvvdp_64x64_strip_pair(&client);
    assert!(cvvdp.is_strip_mode());
    assert!(cvvdp.is_strip_pair_mode());
    assert_eq!(cvvdp.strip_h_body(), Some(STRIP_H_BODY_DEFAULT));
}

#[test]
fn strip_mode_e_is_not_strip_pair_mode() {
    // Mode E (CachedRef) reports is_strip_mode = true but
    // is_strip_pair_mode = false. They are disjoint subclasses of
    // strip-mode dispatch.
    let client = Backend::client(&Default::default());
    let cvvdp_e = Cvvdp::<Backend>::new_strip(
        client,
        64,
        64,
        STRIP_H_BODY_DEFAULT,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip");
    assert!(cvvdp_e.is_strip_mode());
    assert!(!cvvdp_e.is_strip_pair_mode());
}

#[test]
fn full_mode_is_neither_strip_mode_nor_strip_pair_mode() {
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new(client, 64, 64, CvvdpParams::PLACEHOLDER).expect("new");
    assert!(!cvvdp.is_strip_mode());
    assert!(!cvvdp.is_strip_pair_mode());
}

#[test]
fn mode_b_score_matches_full_64x64() {
    let client_full = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client_full, 64, 64, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let client_pair = Backend::client(&Default::default());
    let mut pair = cvvdp_64x64_strip_pair(&client_pair);

    let (r, d) = synth_pair_with_offset_dist(64, 64);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

#[test]
fn mode_b_rejects_invalid_h_body() {
    let client = Backend::client(&Default::default());
    // h_body = 0
    let zero =
        Cvvdp::<Backend>::new_strip_pair(client.clone(), 64, 64, 0, CvvdpParams::PLACEHOLDER);
    assert!(zero.is_err());
    // h_body not aligned to STRIP_ALIGN
    let bad =
        Cvvdp::<Backend>::new_strip_pair(client, 64, 64, 100, CvvdpParams::PLACEHOLDER);
    assert!(bad.is_err());
}

#[test]
fn umbrella_memory_mode_strip_pair_constructs_via_new_with_memory_mode() {
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_with_memory_mode(
        client,
        64,
        64,
        CvvdpParams::PLACEHOLDER,
        MemoryMode::StripPair { h_body: None },
    )
    .expect("StripPair via new_with_memory_mode");
    assert!(cvvdp.is_strip_pair_mode());
    assert_eq!(cvvdp.strip_h_body(), Some(STRIP_H_BODY_DEFAULT));
}
