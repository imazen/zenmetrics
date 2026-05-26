//! Mode B (StripPair) walker parity + memory reduction tests.
//!
//! These tests pin the **shipping contract** for Mode B as of the
//! 2026-05-26 Chunk 1 landing:
//!
//! 1. **Memory reduction (estimator-based)**: the estimator
//!    [`estimate_gpu_memory_bytes_strip_pair`] models the per-strip
//!    hybrid K_SPLIT allocator (Chunk 1) and returns substantially
//!    less than the Full-mode estimate at 1024² and 4096². This pins
//!    the design path even though the constructor today still
//!    allocates Full-sized buffers (the walker port + actual
//!    re-allocation is Chunk 2).
//!
//! 2. **JOD parity at 1024²**: until Chunk 2 lands, Mode B `score()`
//!    routes through the existing Full-mode pipeline, so the JOD
//!    value is bit-identical to Full mode (within the Atomic<f32>
//!    pool ordering noise band). This is the SAME contract Round 2
//!    landed (the prior agent's `mode_b_score_matches_full_64x64`).
//!    The 1024² variant added here pins parity at the next size up.
//!
//! 3. **Strip dispatch counter**: in Mode B, `score()` routes the
//!    pool stage through the strip-aware
//!    [`Cvvdp::_pool_and_finalize_jod_strip`] walker (because
//!    `strip_config.is_some()`), so the strip dispatch counter
//!    increments. At 1024² with `h_body = 256`, the L0 pool runs in
//!    4 strips; combined with shallower strips at deeper levels,
//!    the counter should be ≥ 4 after a single `score()` call.
//!
//! When Chunk 2 lands (per-strip pyramid build + band loop), the
//! parity test stays in its current band (1e-4 abs JOD); the
//! memory test will switch from estimator-based to runtime-based.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{synth_pair_with_offset_dist, Backend};

use cubecl::Runtime;
use cvvdp_gpu::pipeline::{
    estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_strip_pair, mode_b_k_split,
};
use cvvdp_gpu::{Cvvdp, CvvdpParams};

const PARITY_TOL_JOD: f32 = 1e-4;

/// At 1024² with h_body=256, the estimator predicts substantially
/// less memory than Full. Pins the design path: even if the runtime
/// constructor hasn't been reshaped yet (Chunk 2), the estimator
/// is the source of truth for "Mode B's footprint when the walker
/// lands".
#[test]
fn mode_b_estimator_reduces_memory_at_1024() {
    let full = estimate_gpu_memory_bytes(1024, 1024).expect("Full estimate at 1024²");
    let pair256 = estimate_gpu_memory_bytes_strip_pair(1024, 1024, 256)
        .expect("StripPair(256) estimate at 1024²");
    let pair512 = estimate_gpu_memory_bytes_strip_pair(1024, 1024, 512)
        .expect("StripPair(512) estimate at 1024²");

    let ratio_256 = pair256 as f64 / full as f64;
    let ratio_512 = pair512 as f64 / full as f64;
    eprintln!(
        "1024² memory: Full={:.1} MB, StripPair(256)={:.1} MB ({:.1}%), \
         StripPair(512)={:.1} MB ({:.1}%)",
        full as f64 / 1e6,
        pair256 as f64 / 1e6,
        ratio_256 * 100.0,
        pair512 as f64 / 1e6,
        ratio_512 * 100.0,
    );

    // Brief's gate: StripPair < 70% of Full at 1024².
    assert!(
        ratio_256 < 0.70,
        "StripPair(256) = {:.1}% of Full, expected < 70%",
        ratio_256 * 100.0,
    );
    // h_body=512 should also pass the gate.
    assert!(
        ratio_512 < 0.70,
        "StripPair(512) = {:.1}% of Full, expected < 70%",
        ratio_512 * 100.0,
    );
}

/// At 4096² the savings should be even more dramatic — deep bands are
/// proportionally tinier so the strip-mode storage dominates the
/// budget. Stretch goal per the brief.
#[test]
fn mode_b_estimator_reduces_memory_at_4096_stretch() {
    let full = estimate_gpu_memory_bytes(4096, 4096).expect("Full estimate at 4096²");
    let pair256 = estimate_gpu_memory_bytes_strip_pair(4096, 4096, 256)
        .expect("StripPair(256) estimate at 4096²");

    let ratio = pair256 as f64 / full as f64;
    eprintln!(
        "4096² memory: Full={:.1} GB, StripPair(256)={:.1} GB ({:.1}%)",
        full as f64 / 1e9,
        pair256 as f64 / 1e9,
        ratio * 100.0,
    );
    // Expect deep memory reduction at 4096²: < 25%.
    assert!(
        ratio < 0.25,
        "StripPair(256) at 4096² = {:.1}% of Full, expected < 25%",
        ratio * 100.0,
    );
}

/// At 1024² with h_body=256, the Mode B walker should produce JOD
/// within 1e-4 abs of Full mode. Today this passes because the
/// walker routes through Full; when Chunk 2 lands the same gate
/// pins the per-strip walker against Full mode.
#[test]
fn mode_b_walker_jod_matches_full_at_1024() {
    let client = Backend::client(&Default::default());
    let mut full =
        Cvvdp::<Backend>::new(client.clone(), 1024, 1024, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new full");
    let mut pair = Cvvdp::<Backend>::new_strip_pair(client, 1024, 1024, 256, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(1024, 1024);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!("1024² JOD: Full={jod_full}, StripPair(256)={jod_pair}, |diff|={diff}");
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

/// At 1024² with h_body=256 the L0 pool dispatch partitions into
/// `ceil(1024 / 256) = 4` strips; the strip dispatch counter
/// increments by one per (level, strip), so after a single
/// `score()` call the counter should be ≥ 4.
#[test]
fn mode_b_walker_dispatches_n_strips_at_1024() {
    let client = Backend::client(&Default::default());
    let mut pair = Cvvdp::<Backend>::new_strip_pair(
        client,
        1024,
        1024,
        256,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("Cvvdp::new_strip_pair");

    pair.reset_strip_dispatch_counter();
    let (r, d) = synth_pair_with_offset_dist(1024, 1024);
    let _jod = pair.score(&r, &d).expect("Mode B score");

    let n_dispatches = pair.strip_dispatch_counter();
    eprintln!(
        "1024² Mode B (h_body=256): {n_dispatches} strip pool dispatches",
    );
    assert!(
        n_dispatches >= 4,
        "Expected ≥ 4 strip dispatches at L0; got {n_dispatches}",
    );
}

/// Estimator hybrid K_SPLIT picks the expected value for representative
/// `(h_body, n_levels)` combos. Pins the design-doc table:
///
/// | h_body | n_levels | K_SPLIT |
/// |-------:|---------:|--------:|
/// | 256    | 9        | 5       |
/// | 512    | 9        | 6       |
/// | 256    | 7        | 5       |
#[test]
fn mode_b_k_split_matches_design_table() {
    assert_eq!(mode_b_k_split(256, 9), 5);
    assert_eq!(mode_b_k_split(512, 9), 6);
    assert_eq!(mode_b_k_split(256, 7), 5);
    // h_body=128 gives K_SPLIT=4 (128>>4 = 8 < 12 threshold).
    assert_eq!(mode_b_k_split(128, 9), 4);
}

/// At 128×128 with h_body=32, the Mode B walker produces JOD within
/// 1e-4 abs of Full mode AND the strip dispatch counter increments
/// to ≥ 4 — proving the walker actually partitioned the work and
/// did NOT bypass to the full-image one-shot dispatch.
///
/// This is the **tiny-end-to-end** test for Mode B: the smallest
/// viable image size where the walker iterates at least 4 strips at
/// L0 (`128 / 32 = 4`). Deeper levels (L1..L5) add more strip
/// iterations on top, so the actual counter value lands well above 4.
///
/// The test combines BOTH gates of the Mode B walker contract:
/// 1. `compute_dkl_jod(ref, dist)` bit-exact (within atomic pool
///    ordering noise) against the Full-mode reference scorer.
/// 2. Strip dispatch counter ≥ 4, distinguishing a real strip
///    walker from a degenerate single-shot bypass.
///
/// Use deterministic noise inputs (the canonical
/// `synth_pair_with_offset_dist`) so the JOD value is reproducible
/// across runs and the parity gate stays meaningful.
#[test]
fn mode_b_walker_jod_matches_full_at_128() {
    let client = Backend::client(&Default::default());
    let mut full = Cvvdp::<Backend>::new(client.clone(), 128, 128, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new full");
    let mut pair = Cvvdp::<Backend>::new_strip_pair(
        client,
        128,
        128,
        32,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("Cvvdp::new_strip_pair");

    pair.reset_strip_dispatch_counter();
    let (r, d) = synth_pair_with_offset_dist(128, 128);

    // Use compute_dkl_jod directly (not score) so we test the JOD
    // value the strip walker computes inside its dispatch chain,
    // not the score()-wrapper indirection. Both gates apply to this
    // path equally.
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let jod_full = full
        .compute_dkl_jod(&r, &d, ppd)
        .expect("Full compute_dkl_jod");
    let jod_pair = pair
        .compute_dkl_jod(&r, &d, ppd)
        .expect("Mode B compute_dkl_jod");
    let diff = (jod_full - jod_pair).abs();
    let n_dispatches = pair.strip_dispatch_counter();
    eprintln!(
        "128² (h_body=32): JOD Full={jod_full:.6}, Mode B={jod_pair:.6}, \
         |diff|={diff:.3e}, strip_dispatch_counter={n_dispatches}",
    );

    // Gate 1: JOD parity within 1e-4 abs.
    assert!(
        diff < PARITY_TOL_JOD,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );

    // Gate 2: walker iterated ≥ 4 strips. At 128×128 with h_body=32,
    // L0 alone produces 128/32 = 4 strips, and deeper levels add
    // more iterations on top, so the counter lands well above 4.
    // The lower bound = 4 captures the "walker partitioned vs.
    // bypassed" contract.
    assert!(
        n_dispatches >= 4,
        "Expected ≥ 4 strip dispatches (4 strips at L0); got {n_dispatches}",
    );
}
