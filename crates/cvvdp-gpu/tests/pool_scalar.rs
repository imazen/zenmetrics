//! Parity test for `kernels::pool::do_pooling_and_jod_still_3ch`
//! against pycvvdp v0.5.4's `cvvdp.do_pooling_and_jods()`.
//!
//! Three Q_per_ch fixtures covering the JOD curve:
//! - near-perfect (~10 JOD)
//! - middling (~9.99 JOD)
//! - strongly distorted (~9.93 JOD)

use cvvdp_gpu::kernels::pool::{do_pooling_and_jod_still_3ch, met2jod};

#[test]
fn pool_near_perfect_matches_pycvvdp() {
    let q_per_ch = vec![[0.01_f32; 3]; 8];
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "near-perfect: got {jod}, expected ~10.0"
    );
}

#[test]
fn pool_middling_matches_pycvvdp() {
    // ch0..2 rows, 8 bands each. Layout: q[k] = [ch0, ch1, ch2].
    let ch = [
        [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.04],
        [0.4, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03],
        [0.3, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..8)
        .map(|k| [ch[0][k], ch[1][k], ch[2][k]])
        .collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.987_316_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "middling: got {jod}, expected {expected}"
    );
}

#[test]
fn pool_strong_matches_pycvvdp() {
    let ch = [
        [2.5, 1.5, 1.0, 0.8, 0.5, 0.4],
        [2.0, 1.2, 0.8, 0.6, 0.4, 0.3],
        [1.5, 0.9, 0.6, 0.5, 0.3, 0.2],
    ];
    let q_per_ch: Vec<[f32; 3]> = (0..6)
        .map(|k| [ch[0][k], ch[1][k], ch[2][k]])
        .collect();
    let jod = do_pooling_and_jod_still_3ch(&q_per_ch);
    let expected = 9.931_840_f32;
    assert!(
        (jod - expected).abs() < 1e-3,
        "strong: got {jod}, expected {expected}"
    );
}

#[test]
fn met2jod_continuous_at_kink() {
    // The piecewise transform is C0 at Q=0.1; verify the two
    // branches agree there to within f32 epsilon.
    let q = 0.1_f32;
    let from_low = met2jod(q);
    let from_high = met2jod(q + 1e-6);
    assert!(
        (from_low - from_high).abs() < 1e-3,
        "discontinuity at Q=0.1: low={from_low}, high={from_high}"
    );
}

#[test]
fn met2jod_clamps_at_origin() {
    // Q=0 should give JOD=10 (no perceptible difference).
    let jod = met2jod(0.0);
    assert!((jod - 10.0).abs() < 1e-6, "met2jod(0) = {jod}, expected 10");
}
