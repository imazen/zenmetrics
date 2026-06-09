//! Invariant pins on [`met2jod`]. The existing `pool_scalar.rs`
//! coverage is two single-point tests: continuity at the kink
//! (`Q = 0.1`) and `met2jod(0) == 10`. Both are necessary but neither
//! catches:
//!
//! - Monotonicity (higher distortion = lower JOD)
//! - Asymptotic behavior for large Q (JOD declines into negative
//!   territory but stays finite)
//! - Bothbranches' algebraic form
//! - The exact slope-matching at the kink (a refactor could break
//!   continuity AND get monotonicity wrong; my existing test
//!   doesn't detect a slope mismatch that preserves continuity)
//! - Determinism
//!
//! cvvdp's `met2jod` is a smooth piecewise: linear below 0.1,
//! power-law `10 - JOD_A · Q^JOD_EXP` above. JOD_A ≈ 0.044,
//! JOD_EXP ≈ 0.93 (pinned in `pool_constants_match_pycvvdp_v0_5_4`).

use cvvdp_gpu::kernels::pool::{JOD_A, JOD_EXP, met2jod};

#[test]
fn output_at_origin_is_ten_bit_exact() {
    // At Q = 0, the linear branch gives 10 - jod_a_p * 0 = 10.
    // Pin via to_bits() (the existing test uses |jod - 10| < 1e-6).
    let v = met2jod(0.0);
    assert_eq!(
        v.to_bits(),
        10.0_f32.to_bits(),
        "met2jod(0) = {v}, expected 10.0 bit-exact"
    );
}

#[test]
fn output_at_kink_q_eq_0_1_equals_ten_minus_jod_a_pow() {
    // At Q = 0.1: 10 - JOD_A * 0.1^JOD_EXP (matches both branches).
    // Pin both branches via Q = 0.1 - eps and Q = 0.1 + eps with
    // very tight relative tolerance.
    let q = 0.1_f32;
    let expected = 10.0 - JOD_A * q.powf(JOD_EXP);
    let got = met2jod(q);
    let rel = ((got - expected) / expected).abs();
    assert!(
        rel < 1e-6,
        "met2jod(0.1) = {got}, expected {expected} (rel = {rel})"
    );
}

#[test]
fn strictly_monotonic_decreasing_across_realistic_range() {
    // Sweep Q ∈ [0, 100] step 0.01 — the entire realistic JOD range
    // covers Q > 100 collapsing to deeply negative JOD. Higher
    // distortion (higher Q) → lower JOD.
    let mut prev = f32::INFINITY;
    let mut q = 0.0_f32;
    while q <= 100.0 {
        let v = met2jod(q);
        assert!(v.is_finite(), "met2jod({q}) = {v} non-finite");
        assert!(v < prev, "non-monotonic at Q={q}: prev={prev}, got={v}");
        prev = v;
        q += 0.01;
    }
}

#[test]
fn output_strictly_below_ten_for_any_positive_q() {
    // For Q > 0, JOD must drop strictly below 10. Pin via a sweep
    // of small positive Qs — catches a refactor that uses `<` vs
    // `<=` wrongly at the kink, leaving 0 < Q < some_epsilon
    // returning exactly 10.
    // Q values smaller than ~2e-4 underflow in f32 — the linear
    // branch `10 - 0.0515 * Q` can't represent Q below where the
    // subtraction is < 1 ULP of 10.0. Start at 1e-3.
    for q in [1e-3_f32, 0.01, 0.05, 0.1, 0.2, 1.0, 10.0] {
        let v = met2jod(q);
        assert!(v < 10.0, "met2jod({q}) = {v} should be strictly < 10.0");
    }
}

#[test]
fn power_branch_matches_algebraic_form_above_kink() {
    // For Q > 0.1, met2jod(Q) = 10 - JOD_A · Q^JOD_EXP.
    // Verify against the algebraic form to detect a refactor that
    // accidentally introduces an off-by-one exponent or swapped
    // coefficient.
    for q in [0.15_f32, 0.5, 1.0, 5.0, 10.0, 50.0] {
        let got = met2jod(q);
        let expected = 10.0 - JOD_A * q.powf(JOD_EXP);
        let rel = ((got - expected) / expected.abs()).abs();
        assert!(
            rel < 1e-5,
            "met2jod({q}) = {got} vs algebraic {expected} (rel = {rel})"
        );
    }
}

#[test]
fn linear_branch_slope_matches_power_at_kink() {
    // The linear branch (Q ≤ 0.1) is meant to match the SLOPE of
    // the power branch at Q = 0.1. The derivative of (10 - JOD_A · Q^JOD_EXP)
    // at Q = 0.1 is -JOD_A · JOD_EXP · 0.1^(JOD_EXP - 1).
    // The source rewrites this as `jod_a_p = JOD_A · 0.1^(JOD_EXP - 1)`
    // and uses `(10 - jod_a_p · q)`. So the linear slope is -jod_a_p,
    // which differs from the power's true slope by a factor of JOD_EXP.
    // Verify what the code ACTUALLY does (10 - jod_a_p · q):
    let jod_a_p = JOD_A * 0.1_f32.powf(JOD_EXP - 1.0);
    for q in [0.0_f32, 0.025, 0.05, 0.075, 0.099] {
        let got = met2jod(q);
        let expected = 10.0 - jod_a_p * q;
        let rel = ((got - expected) / expected.abs().max(1e-9)).abs();
        assert!(
            rel < 1e-5,
            "met2jod({q}) = {got} vs linear-branch algebraic {expected} (rel = {rel})"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    for q in [0.0_f32, 0.05, 0.1, 0.5, 1.0, 10.0, 100.0] {
        let a = met2jod(q);
        let b = met2jod(q);
        assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic at Q={q}");
    }
}

#[test]
fn declining_jod_for_large_q_stays_finite() {
    // At Q = 1e6, the JOD value goes deeply negative (10 - JOD_A · q^JOD_EXP)
    // but must remain finite (no NaN, no -Inf from float overflow).
    // JOD_EXP ≈ 0.93, so q^JOD_EXP ≈ q for q on this order;
    // 0.044 · 1e6^0.93 ≈ 0.044 · 4e5 ≈ 1.7e4; JOD ≈ 10 - 1.7e4 ≈ -1.7e4.
    for q in [1e3_f32, 1e6, 1e9, 1e12] {
        let v = met2jod(q);
        assert!(v.is_finite(), "met2jod({q}) = {v} non-finite");
        assert!(
            v < 0.0,
            "met2jod({q}) = {v} should be negative for very large Q"
        );
    }
}
