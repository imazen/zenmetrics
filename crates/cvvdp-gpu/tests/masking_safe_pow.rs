//! Direct unit tests for `kernels::masking::safe_pow` — cvvdp's
//! `(x + eps)^p - eps^p` formulation used in the masking chain.
//!
//! Note this is NOT the same as `pool::safe_pow_lp` (which is
//! private but uses `|x|.abs() + eps`); the masking variant
//! operates on a non-negative magnitude directly. Both share the
//! `eps = 1e-5` offset and the trailing `- eps^p` correction.
//!
//! Previously `safe_pow` was exercised only transitively through
//! `mult_mutual_pixel` and the composed pipeline parity tests
//! (`pipeline_color.rs`). Same gap-shape as ticks 351 / 383 / 386
//! / 387: a public helper with no direct CPU-only coverage. A
//! refactor that drops the `- eps^p` correction (or adds an
//! `.abs()` that doesn't belong here) would surface in pipeline
//! drift, not at the per-function level.
//!
//! Lives in a dedicated file (not `masking_scalar.rs`) per the
//! tick 401 precedent — the latter has historically been linter-
//! revert-sensitive, so consts pins + direct primitive tests stay
//! standalone.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::kernels::masking::safe_pow;

#[test]
fn safe_pow_at_zero_input_returns_zero() {
    // safe_pow(0, p) = (0 + eps)^p - eps^p = 0 for all p. Pin
    // exact 0 via .to_bits() across p ∈ {1, 2, 2.264 (MASK_P), 4}.
    // A refactor that drops the `- eps^p` tail silently floors
    // every masking term at eps^p (= ~1e-5 at p=1).
    for &p in &[1.0_f32, 2.0, 2.264_355_2, 4.0] {
        let got = safe_pow(0.0, p);
        assert!(
            got.abs() < 1e-6,
            "safe_pow(0, {p}) = {got}, expected 0 (eps^p tail must cancel head)",
        );
    }
}

#[test]
fn safe_pow_at_unity_matches_one_plus_eps_minus_eps_pow_p() {
    // Direct closed-form check at x = 1 for several p:
    //   safe_pow(1, p) = (1 + eps)^p - eps^p
    // For p ≈ 2.264 (cvvdp's MASK_P), this is ≈ 1.0 + 2.264 * eps
    // - eps^2.264, which is essentially 1.0 in f32. Pin so a
    // refactor that drops the additive correction (or replaces it
    // with subtractive) trips here.
    let eps: f32 = 1e-5;
    for &p in &[1.0_f32, 2.0, 2.264_355_2, 4.0] {
        let got = safe_pow(1.0, p);
        let expected = (1.0_f32 + eps).powf(p) - eps.powf(p);
        let abs_err = (got - expected).abs();
        assert!(
            abs_err < 1e-6,
            "safe_pow(1, {p}) = {got}, expected ~{expected} (|err| = {abs_err:.4e})",
        );
    }
}

#[test]
fn safe_pow_monotonic_in_x_for_positive_p() {
    // For p > 0 and x ≥ 0, safe_pow(x, p) is strictly monotonic
    // increasing in x. A refactor that flips the sign of the
    // outer correction (e.g. `+ eps^p` instead of `- eps^p`)
    // preserves the algebra but breaks monotonicity at small x.
    let xs: Vec<f32> = (0..20).map(|i| (i as f32) * 0.1).collect();
    for &p in &[1.0_f32, 2.0, 2.264_355_2, 4.0] {
        let mut prev = safe_pow(xs[0], p);
        for &x in &xs[1..] {
            let got = safe_pow(x, p);
            assert!(
                got > prev,
                "safe_pow not strictly monotonic at p={p}: f({x}) = {got} <= f(prev) = {prev}",
            );
            prev = got;
        }
    }
}

#[test]
fn safe_pow_eps_offset_dominates_only_near_zero() {
    // Sanity check on the eps shift's regime: for x >> eps, the
    // result should be very close to x^p (the `- eps^p` tail is
    // negligible). At x = 10 and p = 2.264, safe_pow ≈ 10^2.264
    // ≈ 183.7; the eps^p tail is ~1e-11 at this scale.
    let eps: f32 = 1e-5;
    let p: f32 = 2.264_355_2;
    let x: f32 = 10.0;
    let got = safe_pow(x, p);
    let naive = x.powf(p);
    let rel = ((got - naive) / naive).abs();
    assert!(
        rel < 1e-3,
        "safe_pow({x}, {p}) = {got}, naive x^p = {naive}, rel = {rel:.4e} (eps tail should be negligible here)",
    );
    // Near-zero regime: safe_pow(eps, p) ≈ (2eps)^p - eps^p
    //   = eps^p * (2^p - 1)
    let got_near_eps = safe_pow(eps, p);
    let expected_near = eps.powf(p) * ((2.0_f32).powf(p) - 1.0);
    let rel_near = ((got_near_eps - expected_near).abs()) / expected_near.abs().max(1e-30);
    assert!(
        rel_near < 1e-3,
        "safe_pow(eps, {p}) = {got_near_eps}, closed-form {expected_near}, rel = {rel_near:.4e}",
    );
}

#[test]
fn safe_pow_returns_finite_at_extreme_x() {
    // Out-of-range inputs (huge x) must still be finite — no
    // overflow to inf. cvvdp's masking inputs can spike to ~10^3
    // at extreme contrasts. Pin so a refactor that introduces a
    // non-finite path (e.g. dividing by `x` somewhere) surfaces
    // here.
    for &x in &[100.0_f32, 1_000.0, 10_000.0] {
        for &p in &[1.0_f32, 2.0, 2.264_355_2, 4.0] {
            let got = safe_pow(x, p);
            assert!(
                got.is_finite(),
                "safe_pow({x}, {p}) = {got}, expected finite",
            );
            assert!(got > 0.0, "safe_pow({x}, {p}) = {got}, expected > 0");
        }
    }
}
