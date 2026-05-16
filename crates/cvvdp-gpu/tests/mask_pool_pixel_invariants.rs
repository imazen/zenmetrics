//! Invariant pins on [`mask_pool_pixel`]. The function is a 3×3
//! matrix-vector multiply against the `XCM_3X3` cross-channel-masking
//! matrix. The matrix entries themselves are bit-pinned in
//! `tests/masking_constants.rs`. This file pins the FUNCTION
//! semantics: linearity, zero-input, determinism, channel scaling.
//!
//! No direct unit tests existed previously — `mult_mutual_pixel`
//! covers it indirectly through full-pipeline parity, but a regression
//! in the matrix-multiply implementation (e.g., row-column transposition,
//! a missing accumulator term) wouldn't show up cleanly without
//! function-level invariants.

use cvvdp_gpu::kernels::masking::{XCM_3X3, mask_pool_pixel};

#[test]
fn zero_input_yields_zero_output() {
    let out = mask_pool_pixel([0.0, 0.0, 0.0]);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0.0_f32.to_bits(),
            "out[{i}] = {v}, expected 0 bit-exact"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let term = [1.5_f32, -2.3, 0.7];
    let a = mask_pool_pixel(term);
    let b = mask_pool_pixel(term);
    for i in 0..3 {
        assert_eq!(
            a[i].to_bits(),
            b[i].to_bits(),
            "out[{i}] non-deterministic: {} vs {}",
            a[i],
            b[i]
        );
    }
}

#[test]
fn linearity_alpha_scaling() {
    // Linearity: f(α · v) = α · f(v) for any scalar α. f32 multiply
    // is associative-by-scalar, so this should hold bit-exact when
    // expressed as a single multiply.
    let term = [0.4_f32, 1.2, -0.8];
    let f1 = mask_pool_pixel(term);

    for &alpha in &[2.0_f32, 0.5, -1.0, 3.5, -0.25] {
        let scaled_term = [term[0] * alpha, term[1] * alpha, term[2] * alpha];
        let f_scaled = mask_pool_pixel(scaled_term);
        let f_alpha = [f1[0] * alpha, f1[1] * alpha, f1[2] * alpha];
        for i in 0..3 {
            let rel = ((f_scaled[i] - f_alpha[i]) / (f_alpha[i].abs() + 1e-12)).abs();
            assert!(
                rel < 1e-6,
                "linearity broken at α={alpha} out[{i}]: f(αv) = {} vs α·f(v) = {} (rel = {rel})",
                f_scaled[i],
                f_alpha[i]
            );
        }
    }
}

#[test]
fn additivity_in_input() {
    // f(a + b) = f(a) + f(b). Linear maps are additive; pin within
    // a tight relative tolerance.
    let a = [0.3_f32, 0.7, -0.4];
    let b = [1.1_f32, -0.2, 0.9];
    let sum = [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

    let fa = mask_pool_pixel(a);
    let fb = mask_pool_pixel(b);
    let f_sum = mask_pool_pixel(sum);

    for i in 0..3 {
        let expected = fa[i] + fb[i];
        let rel = ((f_sum[i] - expected) / (expected.abs() + 1e-12)).abs();
        assert!(
            rel < 1e-5,
            "additivity broken at i={i}: f(a+b) = {} vs f(a)+f(b) = {} (rel = {rel})",
            f_sum[i],
            expected
        );
    }
}

#[test]
fn unit_basis_inputs_recover_matrix_columns() {
    // f([1, 0, 0]) selects row 0 of XCM_3X3 → out = XCM_3X3[0]
    // f([0, 1, 0]) → out = XCM_3X3[1]
    // f([0, 0, 1]) → out = XCM_3X3[2]
    // The function uses `XCM_3X3[in][out]` so input ch n picks row n.
    let e0 = mask_pool_pixel([1.0, 0.0, 0.0]);
    let e1 = mask_pool_pixel([0.0, 1.0, 0.0]);
    let e2 = mask_pool_pixel([0.0, 0.0, 1.0]);

    for j in 0..3 {
        assert_eq!(
            e0[j].to_bits(),
            XCM_3X3[0][j].to_bits(),
            "e_0 out[{j}] = {} but XCM_3X3[0][{j}] = {}",
            e0[j],
            XCM_3X3[0][j]
        );
        assert_eq!(
            e1[j].to_bits(),
            XCM_3X3[1][j].to_bits(),
            "e_1 out[{j}] = {} but XCM_3X3[1][{j}] = {}",
            e1[j],
            XCM_3X3[1][j]
        );
        assert_eq!(
            e2[j].to_bits(),
            XCM_3X3[2][j].to_bits(),
            "e_2 out[{j}] = {} but XCM_3X3[2][{j}] = {}",
            e2[j],
            XCM_3X3[2][j]
        );
    }
}

#[test]
fn output_finite_for_finite_input() {
    // The function is a matrix-vector product with finite f32
    // coefficients; any finite input produces finite output. Sweep
    // a wide dynamic range.
    for &term in &[
        [0.0_f32, 0.0, 0.0],
        [1e-10, 1e-10, 1e-10],
        [1.0, 1.0, 1.0],
        [1e6, 1e6, 1e6],
        [-1e6, -1e6, -1e6],
        [1.0, -1.0, 1.0],
    ] {
        let out = mask_pool_pixel(term);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "term={term:?}: out[{i}] = {v} non-finite");
        }
    }
}

#[test]
fn xcm_matrix_first_row_dominates_for_channel_a() {
    // XCM_3X3[0] = [0.877, 0.016, 0.050] — channel A's row is
    // strongly self-dominant. Pin: for input [1, 0, 0], the
    // function's row contribution to output index 0 (A) is the
    // largest. This guards against a refactor that transposes the
    // matrix (would put XCM_3X3[col=0] into position 0 differently).
    let out = mask_pool_pixel([1.0, 0.0, 0.0]);
    assert!(
        out[0] > 0.5,
        "A self-coupling out[0] = {} should be > 0.5 (XCM_3X3[0][0] = 0.877)",
        out[0]
    );
    assert!(
        out[0] > out[1].abs() && out[0] > out[2].abs(),
        "A's row should dominate: out = {out:?}"
    );
}
