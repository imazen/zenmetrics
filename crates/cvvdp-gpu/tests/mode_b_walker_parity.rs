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

/// Mode B estimate vs the **measured** strip_pair peak at 1024²
/// (task137). The previous version of this test asserted the estimate
/// was `< 65%` of Full — but that bound validated the OLD
/// under-predicting estimator, NOT reality. The committed measured
/// peaks (`benchmarks/gpu_metrics_sweep_2026-05-28.tsv`, cuda) show
/// strip_pair at 1024² = 417 MB vs Full = 385 MB — i.e. **Mode B is
/// NOT a memory win at 1024²** (108% of Full). The corrected estimator
/// must therefore NOT claim a fictional saving here; it must
/// over-predict the measured peak (the safe direction for
/// `resolve_auto`).
#[test]
fn mode_b_estimator_at_1024_over_predicts_measured() {
    // Measured strip_pair (cuda) peak at 1024², from the committed sweep.
    const MEASURED_STRIP_PAIR_1024: f64 = 437_256_192.0;
    let pair256 = estimate_gpu_memory_bytes_strip_pair(1024, 1024, 256)
        .expect("StripPair(256) estimate at 1024²");
    let full = estimate_gpu_memory_bytes(1024, 1024).expect("Full estimate at 1024²");

    let pct_vs_measured = (pair256 as f64 - MEASURED_STRIP_PAIR_1024)
        / MEASURED_STRIP_PAIR_1024
        * 100.0;
    eprintln!(
        "1024² Mode B: est={:.1} MB, measured={:.1} MB ({:+.1}%), Full est={:.1} MB",
        pair256 as f64 / 1e6,
        MEASURED_STRIP_PAIR_1024 / 1e6,
        pct_vs_measured,
        full as f64 / 1e6,
    );

    // SAFE-DIRECTION gate: estimate must be >= measured (never
    // under-budget resolve_auto) and not absurdly high (<= +200% leaves
    // generous headroom; the fixed-context base inflates the small-size
    // estimate but that's the correct conservative bias at 1 MP, which
    // the validation gate exempts from the tight ±20% band).
    assert!(
        pair256 as f64 >= MEASURED_STRIP_PAIR_1024,
        "Mode B estimate {:.1} MB under-predicts the measured peak {:.1} MB \
         (under-prediction is a resolve_auto bug)",
        pair256 as f64 / 1e6,
        MEASURED_STRIP_PAIR_1024 / 1e6,
    );
    assert!(
        pct_vs_measured <= 200.0,
        "Mode B estimate over-predicts by {pct_vs_measured:+.1}% — far above the \
         conservative-but-sane ceiling",
    );
}

/// Mode B estimate vs the **measured** strip_pair peak at 4096²
/// (task137). The previous version asserted `< 25%` of Full, which
/// again validated the under-predicting estimator. Measured reality
/// (cuda): strip_pair at 4096² = 2273 MB vs Full = 3969 MB ≈ **57% of
/// Full** — a real but modest win, NOT the fictional <25%. The
/// corrected estimator must over-predict the measured peak within the
/// ±20% gate (safe direction).
#[test]
fn mode_b_estimator_at_4096_over_predicts_measured() {
    // Measured strip_pair (cuda) peak at 4096², from the committed sweep.
    const MEASURED_STRIP_PAIR_4096: f64 = 2_383_413_248.0;
    let pair256 = estimate_gpu_memory_bytes_strip_pair(4096, 4096, 256)
        .expect("StripPair(256) estimate at 4096²");
    let full = estimate_gpu_memory_bytes(4096, 4096).expect("Full estimate at 4096²");

    let pct_vs_measured = (pair256 as f64 - MEASURED_STRIP_PAIR_4096)
        / MEASURED_STRIP_PAIR_4096
        * 100.0;
    let ratio_vs_full = pair256 as f64 / full as f64;
    eprintln!(
        "4096² Mode B: est={:.2} GB, measured={:.2} GB ({:+.1}%), {:.1}% of Full est",
        pair256 as f64 / 1e9,
        MEASURED_STRIP_PAIR_4096 / 1e9,
        pct_vs_measured,
        ratio_vs_full * 100.0,
    );

    // SAFE-DIRECTION gate within the ±20% validation band: the estimate
    // must over-predict the measured peak but stay within +20%.
    assert!(
        pair256 as f64 >= MEASURED_STRIP_PAIR_4096,
        "Mode B estimate {:.2} GB under-predicts the measured peak {:.2} GB \
         (under-prediction is a resolve_auto bug)",
        pair256 as f64 / 1e9,
        MEASURED_STRIP_PAIR_4096 / 1e9,
    );
    assert!(
        pct_vs_measured <= 20.0,
        "Mode B estimate over-predicts by {pct_vs_measured:+.1}% — exceeds the +20% gate",
    );
}

/// Off-calibration body sizes (task137). The four other estimator
/// tests pin the canonical h_body = 256 (and one degenerate 512); a
/// curve-fit estimator can pass those vacuously while being arbitrary
/// in between. This test exercises body = 128 AND body = 512 at 1024²
/// and 4096² with a LOOSE monotonicity/sanity bound — it does not pin
/// exact bytes (no measured peak for these bodies), but guards that:
///   1. the estimate stays positive and finite,
///   2. a SMALLER body never produces a LARGER estimate at a given size
///      (a smaller strip body can only shrink or hold the working set),
///   3. the estimate never drops below the per-instance context floor.
#[test]
fn mode_b_estimator_off_calibration_bodies_are_sane() {
    const CONTEXT_FLOOR: usize = 256 * 1024 * 1024; // POOL_B_CONTEXT_BASE_BYTES
    for &(w, h) in &[(1024_u32, 1024_u32), (4096, 4096)] {
        let b128 = estimate_gpu_memory_bytes_strip_pair(w, h, 128)
            .unwrap_or_else(|| panic!("body=128 at {w}×{h}"));
        let b256 = estimate_gpu_memory_bytes_strip_pair(w, h, 256)
            .unwrap_or_else(|| panic!("body=256 at {w}×{h}"));
        let b512 = estimate_gpu_memory_bytes_strip_pair(w, h, 512)
            .unwrap_or_else(|| panic!("body=512 at {w}×{h}"));
        eprintln!(
            "{w}² off-cal bodies: 128={:.1} MB, 256={:.1} MB, 512={:.1} MB",
            b128 as f64 / 1e6,
            b256 as f64 / 1e6,
            b512 as f64 / 1e6,
        );
        // Floor: every estimate must clear the context base.
        for (body, est) in [(128, b128), (256, b256), (512, b512)] {
            assert!(
                est > CONTEXT_FLOOR,
                "body={body} at {w}×{h}: estimate {est} below context floor {CONTEXT_FLOOR}",
            );
        }
        // Monotonicity: a larger body is a larger (or equal) strip
        // buffer at every level, so the estimate must be non-decreasing
        // in body.
        assert!(
            b128 <= b256 && b256 <= b512,
            "Mode B estimate not monotonic in h_body at {w}×{h}: \
             128={b128}, 256={b256}, 512={b512}",
        );
    }
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

/// Explicit-h_body alias of the 1024² parity test — pins the
/// canonical (size, h_body) brief from the Mode B walker scope
/// (1024² with h_body = 256) under the name pattern the brief
/// requested. Equivalent to `mode_b_walker_jod_matches_full_at_1024`
/// but kept distinct so the per-(size, h_body) parity contract is
/// explicit in the test inventory.
#[test]
fn mode_b_walker_jod_matches_full_at_1024_h_body_256() {
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
    eprintln!(
        "1024² h_body=256 JOD: Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
    assert!(
        diff < PARITY_TOL_JOD as f64,
        "Mode B JOD={jod_pair} drifts from Full JOD={jod_full} by {diff} > {PARITY_TOL_JOD}",
    );
}

/// 4096² walker parity at h_body=512. Stretch goal per the brief.
///
/// Memory profile: as of this landing the Cvvdp constructor still
/// allocates Full-mode buffers for Mode B (the per-strip reshape is
/// a separate landing). At 4096² that's ~4-5 GB per `Cvvdp` instance,
/// so running TWO instances (Full + Mode B) in the same process
/// approaches the upper end of a 12 GB GPU. The test is marked
/// `#[ignore]` to keep the default `cargo test` run safe on the
/// development workstation; remove the attribute (or run with
/// `cargo test -- --ignored`) on a box with ≥ 16 GB free GPU memory
/// to verify the stretch parity gate.
///
/// To run manually:
/// ```sh
/// cargo test -p cvvdp-gpu --features cuda --test mode_b_walker_parity \
///     mode_b_walker_jod_matches_full_at_4096_h_body_512 -- --ignored
/// ```
///
/// Tolerance is the same 1e-4 abs JOD band as the other parity
/// tests; Atomic<f32> reduction-order noise is the only expected
/// divergence and it stays well under that band.
#[test]
#[ignore = "Allocates two 4096² Cvvdp instances (~10 GB GPU); run with --ignored on a ≥ 16 GB GPU"]
fn mode_b_walker_jod_matches_full_at_4096_h_body_512() {
    let client = Backend::client(&Default::default());
    let mut full =
        Cvvdp::<Backend>::new(client.clone(), 4096, 4096, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new full");
    let mut pair =
        Cvvdp::<Backend>::new_strip_pair(client, 4096, 4096, 512, CvvdpParams::PLACEHOLDER)
            .expect("Cvvdp::new_strip_pair");

    let (r, d) = synth_pair_with_offset_dist(4096, 4096);

    let jod_full = full.score(&r, &d).expect("Full score");
    let jod_pair = pair.score(&r, &d).expect("Mode B score");
    let diff = (jod_full - jod_pair).abs();
    eprintln!(
        "4096² h_body=512 JOD: Full={jod_full}, StripPair={jod_pair}, |diff|={diff}",
    );
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
