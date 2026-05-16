//! Function-level invariant pins on [`mult_mutual_pixel`]. The
//! existing `masking_scalar.rs::mult_mutual_pixel_matches_pycvvdp_4x4`
//! is a single fixed 4×4 pycvvdp parity check. It catches kernel
//! coefficient drift but not:
//!
//! - Identical inputs (T == R) → output [0, 0, 0]. The natural
//!   contract for a perceptual-difference operator: zero difference
//!   when test == reference.
//! - Argument symmetry: f(T, R) == f(R, T). Both the masking term
//!   (uses `min(|T|, |R|)`) and the diff (uses `|T - R|`) are
//!   symmetric in T and R, so the whole function must be too.
//! - Non-negativity of D[cc]: clamp_diff_soft of non-negative input
//!   never goes negative.
//! - Upper bound D[cc] < d_max = 10^D_MAX (from clamp_diff_soft).
//! - Determinism.
//! - Finite output for finite input across a wide dynamic range.
//!
//! These complement the pycvvdp parity test by pinning the function's
//! shape and contract.

use cvvdp_gpu::kernels::masking::{D_MAX, mult_mutual_pixel};

#[test]
fn identical_inputs_yield_zero_output() {
    // T == R means |T - R| == 0, so safe_pow(0, MASK_P) - eps^MASK_P
    // is small but not exactly zero (safe_pow returns 0 only via the
    // bias-and-subtract scheme: `(0 + eps)^p - eps^p = 0`). Then
    // clamp_diff_soft(0) = 0 exactly.
    for tr in [
        [0.0_f32, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [10.0, -5.0, 3.0],
        [-100.0, 200.0, -50.0],
    ] {
        let d = mult_mutual_pixel(tr, tr);
        for (i, &v) in d.iter().enumerate() {
            // safe_pow(0, p) = (0+eps)^p - eps^p = 0 exactly,
            // so clamp_diff_soft(0 / (1+M)) = clamp_diff_soft(0) = 0.
            assert_eq!(
                v.to_bits(),
                0.0_f32.to_bits(),
                "T == R = {tr:?} should give D[{i}] = 0 bit-exact, got {v}"
            );
        }
    }
}

#[test]
fn symmetric_in_arguments() {
    // f(T, R) == f(R, T) because:
    //   M_mm uses min(|T|, |R|) — symmetric
    //   diff uses |T - R| — symmetric
    // The composition must therefore be symmetric across all (T, R).
    let cases = [
        ([0.5_f32, -0.3, 1.2], [0.1, 0.4, -0.8]),
        ([10.0, -5.0, 7.5], [0.0, 0.0, 0.0]),
        ([-100.0, 50.0, 20.0], [100.0, -50.0, -20.0]),
        ([1e-5, 1e-5, 1e-5], [1.0, 1.0, 1.0]),
    ];
    for (t, r) in cases {
        let d_tr = mult_mutual_pixel(t, r);
        let d_rt = mult_mutual_pixel(r, t);
        for i in 0..3 {
            assert_eq!(
                d_tr[i].to_bits(),
                d_rt[i].to_bits(),
                "asymmetry at D[{i}]: f(T, R) = {} vs f(R, T) = {}",
                d_tr[i],
                d_rt[i]
            );
        }
    }
}

#[test]
fn non_negative_for_all_inputs() {
    // safe_pow on non-negative input is non-negative;
    // `... / (1 + M)` is non-negative (M ≥ 0);
    // clamp_diff_soft preserves sign and is non-negative for non-negative input.
    let cases = [
        ([0.5_f32, -0.3, 1.2], [0.1, 0.4, -0.8]),
        ([10.0, -5.0, 7.5], [-2.0, 1.0, -3.0]),
        ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ([-1e3, 1e3, -1e3], [1e3, -1e3, 1e3]),
    ];
    for (t, r) in cases {
        let d = mult_mutual_pixel(t, r);
        for (i, &v) in d.iter().enumerate() {
            assert!(
                v >= 0.0,
                "D[{i}] = {v} is negative for T={t:?}, R={r:?}"
            );
        }
    }
}

#[test]
fn bounded_by_d_max_asymptote() {
    // clamp_diff_soft caps the per-channel output at d_max = 10^D_MAX
    // (D_MAX ≈ 2.564 ⇒ d_max ≈ 366.69). Synthesize a large-diff input
    // and confirm D[cc] < d_max for every channel.
    let d_max = 10.0_f32.powf(D_MAX);
    let t = [1e6_f32, -1e6, 1e6];
    let r = [-1e6_f32, 1e6, -1e6];
    let d = mult_mutual_pixel(t, r);
    for (i, &v) in d.iter().enumerate() {
        assert!(
            v < d_max,
            "D[{i}] = {v} should be strictly < d_max = {d_max}"
        );
        assert!(
            v.is_finite(),
            "D[{i}] = {v} not finite"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let t = [0.5_f32, -0.3, 1.2];
    let r = [0.1_f32, 0.4, -0.8];
    let a = mult_mutual_pixel(t, r);
    let b = mult_mutual_pixel(t, r);
    for i in 0..3 {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "D[{i}] non-deterministic: {} vs {}",
            a[i],
            b[i]
        );
    }
}

#[test]
fn positive_for_non_identical_inputs() {
    // Any non-trivial (T, R) with T != R AND non-zero diff above
    // f32-rounding-near-zero must produce strictly positive output
    // on at least one channel.
    let cases = [
        ([1.0_f32, 0.0, 0.0], [0.0_f32, 0.0, 0.0]),
        ([5.0_f32, -3.0, 2.0], [4.0_f32, -3.0, 2.0]),
        ([100.0_f32, 0.0, 0.0], [0.0_f32, 100.0, 100.0]),
    ];
    for (t, r) in cases {
        let d = mult_mutual_pixel(t, r);
        let any_positive = d.iter().any(|&v| v > 0.0);
        assert!(
            any_positive,
            "T != R should give some positive D, got D = {d:?} for T={t:?}, R={r:?}"
        );
    }
}

#[test]
fn finite_output_across_wide_dynamic_range() {
    // Sweep extreme inputs: very small, very large, mixed signs.
    let cases = [
        ([0.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([1e-10, 1e-10, 1e-10], [1e-10, 1e-10, 1e-10]),
        ([1e6, 1e6, 1e6], [1e6, 1e6, 1e6]),
        ([1e6, 0.0, 1e6], [0.0, 1e6, 0.0]),
        ([-1e6, 1e6, -1e6], [1e6, -1e6, 1e6]),
    ];
    for (t, r) in cases {
        let d = mult_mutual_pixel(t, r);
        for (i, &v) in d.iter().enumerate() {
            assert!(
                v.is_finite(),
                "D[{i}] = {v} non-finite for T={t:?}, R={r:?}"
            );
        }
    }
}
