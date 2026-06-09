//! Flow invariants on [`do_pooling_and_jod_still_3ch`]. The existing
//! `pool_scalar.rs` tests are 3 pointwise pycvvdp parity checks
//! (`pool_near_perfect_matches_pycvvdp`, `pool_middling_matches_pycvvdp`,
//! `pool_strong_matches_pycvvdp`) plus a panic-on-empty assertion.
//! Those pin specific JOD values for fixed inputs; this file pins the
//! function's *shape*:
//!
//! - Zero input → JOD == 10 (perfect-quality limit).
//! - Output bounded above by 10 for non-negative input.
//! - Increasing any element of `q_per_ch` cannot *increase* JOD
//!   (monotonicity in the "more distortion" direction).
//! - Determinism.
//! - Sensitivity: scaling the entire input by 100× changes JOD
//!   measurably (catches a refactor that accidentally short-circuits
//!   the pool stage to a constant).

use cvvdp_gpu::kernels::pool::do_pooling_and_jod_still_3ch;

#[test]
fn zero_input_yields_jod_ten() {
    // Every q value zero → lp_norm_sum over zeros = 0 → met2jod(0) = 10.
    let q_per_ch = vec![[0.0_f32; 3]; 8];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(
        (jod - 10.0).abs() < 1e-5,
        "all-zero input should give JOD = 10, got {jod}"
    );
}

#[test]
fn output_bounded_above_by_ten() {
    // For any non-negative input, the pool stage produces Q ≥ 0
    // and met2jod is strictly decreasing from 10. JOD must be ≤ 10
    // (tiny float overshoot acceptable but cap at 10 + epsilon).
    let cases: Vec<Vec<[f32; 3]>> = vec![
        vec![[0.0; 3]; 4],
        vec![[0.001; 3]; 8],
        vec![[0.1; 3]; 6],
        vec![[1.0, 0.5, 0.3]; 5],
    ];
    for case in cases {
        let jod = do_pooling_and_jod_still_3ch(&case);
        assert!(
            jod <= 10.0 + 1e-3,
            "JOD = {jod} > 10 for input len={}",
            case.len()
        );
        assert!(jod.is_finite(), "JOD = {jod} non-finite");
    }
}

#[test]
fn monotonic_in_each_input_position() {
    // Increase one element of q_per_ch (more distortion) → JOD
    // should not increase. Test each (level, channel) position.
    let base = vec![[0.1_f32, 0.1, 0.1]; 4];
    let base_jod = do_pooling_and_jod_still_3ch(&base);

    for lvl in 0..base.len() {
        for ch in 0..3 {
            let mut perturbed = base.clone();
            perturbed[lvl][ch] += 0.5; // make it bigger (more distortion)
            let pert_jod = do_pooling_and_jod_still_3ch(&perturbed);
            assert!(
                pert_jod <= base_jod + 1e-5,
                "monotonicity broken at (lvl={lvl}, ch={ch}): \
                 base = {base_jod}, perturbed = {pert_jod}"
            );
        }
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let q_per_ch = vec![[0.5_f32, 0.3, 0.2], [0.4, 0.25, 0.18], [0.3, 0.2, 0.15]];
    let a = do_pooling_and_jod_still_3ch(&q_per_ch);
    let b = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic: {a} vs {b}");
}

#[test]
fn output_responds_to_input_magnitude() {
    // Catches a refactor that accidentally returns a constant
    // independent of input. Test that scaling q_per_ch by 100×
    // changes the JOD by more than 1e-3 (the pycvvdp parity
    // tolerance — large enough to detect a stuck-output bug).
    let small = vec![[0.01_f32, 0.01, 0.01]; 4];
    let big = vec![[1.0_f32, 1.0, 1.0]; 4];
    let jod_small = do_pooling_and_jod_still_3ch(&small);
    let jod_big = do_pooling_and_jod_still_3ch(&big);
    assert!(
        (jod_small - jod_big).abs() > 1e-3,
        "JOD insensitive to 100× scaling: small={jod_small}, big={jod_big}"
    );
    // And the larger input should give the smaller JOD.
    assert!(
        jod_big < jod_small,
        "larger input should give smaller JOD: small={jod_small}, big={jod_big}"
    );
}

#[test]
fn single_level_input_supported() {
    // Edge case: only 1 pyramid level (the baseband). The function
    // documents that n_levels >= 1 is required (panics otherwise).
    // Verify the 1-level path doesn't panic and produces finite JOD.
    let q_per_ch = vec![[0.1_f32, 0.1, 0.1]];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(jod.is_finite(), "single-level JOD = {jod} non-finite");
    assert!(
        jod < 10.0,
        "single-level non-zero input should give JOD < 10, got {jod}"
    );
}

#[test]
fn many_levels_input_supported() {
    // Stress: 12 pyramid levels (more than any realistic image).
    // No panics, finite output.
    let q_per_ch = vec![[0.05_f32, 0.05, 0.05]; 12];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(jod.is_finite(), "12-level JOD = {jod} non-finite");
    assert!(
        (0.0..=10.0 + 1e-3).contains(&jod),
        "12-level JOD = {jod} out of range"
    );
}
