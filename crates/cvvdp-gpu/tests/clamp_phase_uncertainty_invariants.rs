//! Invariant pins on two small masking primitives:
//!
//! - [`clamp_diff_soft`] — soft clamp `D_max * D / (D_max + D)` with
//!   `D_max = 10^D_MAX`. Used by every per-pixel masking decision in
//!   the cvvdp pipeline. No direct unit tests until now (pipeline
//!   parity covers it indirectly).
//! - [`phase_uncertainty_no_blur`] — `m * 10^MASK_C`, the phase-
//!   uncertainty branch for small bands. Also no direct unit tests.
//!
//! `safe_pow` is already covered by `tests/masking_safe_pow.rs`. The
//! constants `D_MAX` and `MASK_C` are bit-pinned in
//! `tests/masking_constants.rs`. This file pins the FUNCTION
//! semantics on top of those constants.

use cvvdp_gpu::kernels::masking::{
    D_MAX, MASK_C, clamp_diff_soft, phase_uncertainty_no_blur,
};

#[test]
fn clamp_diff_soft_zero_input_returns_zero() {
    // f(0) = d_max * 0 / (d_max + 0) = 0. Pin via to_bits() so
    // a future refactor that introduces an additive bias trips
    // here.
    let v = clamp_diff_soft(0.0);
    assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "f(0) must be exactly 0");
}

#[test]
fn clamp_diff_soft_strictly_monotonic_on_positive() {
    // For 0 ≤ d1 < d2, f(d1) < f(d2). The function is f(d) = d_max·d / (d_max+d),
    // derivative wrt d is d_max² / (d_max+d)² > 0 for any d > -d_max.
    let samples: Vec<f32> = (0..=200).map(|i| (i as f32) * 5.0).collect(); // 0..1000
    let mut prev = f32::NEG_INFINITY;
    for &d in &samples {
        let f = clamp_diff_soft(d);
        assert!(
            f.is_finite(),
            "f({d}) = {f} non-finite"
        );
        assert!(
            f > prev,
            "non-monotonic at d={d}: prev={prev}, got={f}"
        );
        prev = f;
    }
}

#[test]
fn clamp_diff_soft_asymptotically_bounded_by_d_max() {
    // f(d) = d_max·d / (d_max + d). The bound `f(d) < d_max` holds
    // for every finite d > 0. Convergence is slow: at d = 100·d_max
    // the gap is ~1%. Pin both halves:
    // (a) `f(d) < d_max` strictly across a huge sweep, AND
    // (b) at d = 1e6 (≈ 2700·d_max) the gap is < 0.1%.
    let d_max = 10.0_f32.powf(D_MAX); // ≈ 366.69

    for &d in &[1e3_f32, 1e4, 1e5, 1e6, 1e7, 1e9] {
        let f = clamp_diff_soft(d);
        assert!(
            f < d_max,
            "f({d}) = {f} should be strictly < d_max = {d_max}"
        );
    }

    // For d ≥ 1e6 we should be within 0.1% of d_max.
    for &d in &[1e6_f32, 1e7, 1e9] {
        let f = clamp_diff_soft(d);
        let gap = (d_max - f) / d_max;
        assert!(
            gap < 1e-3,
            "f({d}) = {f} too far below d_max = {d_max} (gap = {gap})"
        );
    }
}

#[test]
fn clamp_diff_soft_half_saturation_at_d_max() {
    // f(d_max) = d_max * d_max / (d_max + d_max) = d_max / 2.
    // This is the half-saturation point and pins the algebraic form.
    let d_max = 10.0_f32.powf(D_MAX);
    let f = clamp_diff_soft(d_max);
    let expected = d_max / 2.0;
    let rel_err = ((f - expected) / expected).abs();
    assert!(
        rel_err < 1e-5,
        "f(d_max) = {f} but expected d_max/2 = {expected} (rel_err = {rel_err})"
    );
}

#[test]
fn clamp_diff_soft_is_deterministic() {
    // Pure function — same input bit-equal output.
    for d in [0.0_f32, 1.0, 10.0, 100.0, 1e6] {
        let a = clamp_diff_soft(d);
        let b = clamp_diff_soft(d);
        assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic at d={d}");
    }
}

#[test]
fn phase_uncertainty_no_blur_is_pure_scaling() {
    // f(m) = m * 10^MASK_C. The scale factor is constant; verify
    // that f(2m) = 2 * f(m) exactly (modulo f32 rounding).
    let scale = 10.0_f32.powf(MASK_C);
    for &m in &[0.0_f32, 1.0, 5.0, 10.0, 100.0, 1e3, -1.0, -100.0] {
        let f = phase_uncertainty_no_blur(m);
        let expected = m * scale;
        assert_eq!(
            f.to_bits(),
            expected.to_bits(),
            "f({m}) = {f} but m * 10^MASK_C = {expected}"
        );
    }
}

#[test]
fn phase_uncertainty_scale_factor_is_in_known_range() {
    // MASK_C ≈ -0.7955 ⇒ 10^MASK_C ≈ 0.1603. Pin that the function's
    // effective scale lands in [0.15, 0.17] — bounds the value while
    // remaining tolerant to the bit-pinned MASK_C in masking_constants.rs.
    let scale = phase_uncertainty_no_blur(1.0);
    assert!(
        (0.15..=0.17).contains(&scale),
        "phase_uncertainty scale = {scale}, expected in [0.15, 0.17]"
    );
}

#[test]
fn phase_uncertainty_zero_returns_zero() {
    let v = phase_uncertainty_no_blur(0.0);
    assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "f(0) must be exactly 0");
}

#[test]
fn phase_uncertainty_monotonic() {
    // 10^MASK_C > 0, so the function preserves order of its input
    // across the entire real line.
    let mut prev = f32::NEG_INFINITY;
    let mut m = -100.0_f32;
    while m <= 100.0 {
        let f = phase_uncertainty_no_blur(m);
        assert!(f.is_finite() && f > prev, "non-monotonic at m={m}");
        prev = f;
        m += 1.0;
    }
}

#[test]
fn phase_uncertainty_is_deterministic() {
    for m in [0.0_f32, 1.0, -5.0, 1e3, -1e3] {
        let a = phase_uncertainty_no_blur(m);
        let b = phase_uncertainty_no_blur(m);
        assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic at m={m}");
    }
}
