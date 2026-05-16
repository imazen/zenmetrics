//! Function-level invariant pins on
//! [`srgb_byte_to_dkl_scalar`]. The existing
//! `color_scalar.rs::matches_pycvvdp_standard_4k` test is a single
//! pointwise-numeric goldens check against pycvvdp. The matrix
//! itself is independently bit-pinned by
//! `srgb_linear_to_dkl_matrix_matches_pycvvdp_v0_5_4`. This file
//! adds function-level semantic checks:
//!
//! - Luminance monotonicity: increasing all three channels together
//!   increases the achromatic output.
//! - Grayscale neutrality: r = g = b zeros the chroma channels (modulo
//!   the matrix's row-sum residual — opponent rows are mean-zero by
//!   construction in DKL).
//! - Display-constant linearity: scaling `y_peak` by k linearly scales
//!   the output (the function is affine in display parameters).
//! - Boundary safety: (0, 0, 0) and (255, 255, 255) don't panic and
//!   produce finite output (no LUT-bounds escape).
//! - Determinism: same input → bit-identical output.
//!
//! Why these matter that the existing pycvvdp parity test misses:
//! the parity test is one point in (r, g, b, display) ℝ⁶ space.
//! These invariants pin global function shape.

use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
use cvvdp_gpu::params::DisplayModel;

fn d() -> DisplayModel {
    DisplayModel::STANDARD_4K
}

#[test]
fn dkl_a_monotonic_in_luminance() {
    // Grayscale ramp through 0..256 step 16. Each step's A must be
    // strictly greater than the previous.
    let dm = d();
    let mut prev_a = f32::NEG_INFINITY;
    for v in (0_u32..256).step_by(16) {
        let v8 = v as u8;
        let (a, _, _) = srgb_byte_to_dkl_scalar(v8, v8, v8, dm.y_peak, dm.y_black, dm.y_refl);
        assert!(a.is_finite(), "A non-finite at v={v}: {a}");
        assert!(
            a > prev_a,
            "A non-monotonic at v={v}: prev={prev_a}, got={a}"
        );
        prev_a = a;
    }
}

#[test]
fn grayscale_input_zeros_chroma_within_matrix_tolerance() {
    // For r = g = b, the chroma rows reduce to L * row_sum. The
    // matrix is pinned (by srgb_linear_to_dkl_row_sign_signature)
    // such that row_sum_RG and row_sum_VY are small relative to
    // the achromatic row sum. Pin the resulting per-pixel chroma
    // to be < 5% of A's magnitude for any neutral byte.
    let dm = d();
    for v in [0_u8, 32, 64, 96, 128, 160, 192, 224, 255] {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(v, v, v, dm.y_peak, dm.y_black, dm.y_refl);
        assert!(a.is_finite() && rg.is_finite() && vy.is_finite());
        // A is always positive for non-black input; for v=0 it's the
        // black + refl scaled by row 0, still > 0.
        let denom = a.abs().max(1e-6);
        assert!(
            rg.abs() / denom < 0.05,
            "v={v}: |RG|/|A| = {} should be < 5% (a={a}, rg={rg})",
            rg.abs() / denom
        );
        assert!(
            vy.abs() / denom < 0.05,
            "v={v}: |VY|/|A| = {} should be < 5% (a={a}, vy={vy})",
            vy.abs() / denom
        );
    }
}

#[test]
fn black_input_produces_minimum_luminance() {
    // (0, 0, 0) → L_rgb = y_black + y_refl (the float-equiv "black
    // ambient"). The A value must be the smallest in any input we
    // could sample.
    let dm = d();
    let (a_black, _, _) = srgb_byte_to_dkl_scalar(0, 0, 0, dm.y_peak, dm.y_black, dm.y_refl);
    let (a_mid, _, _) = srgb_byte_to_dkl_scalar(128, 128, 128, dm.y_peak, dm.y_black, dm.y_refl);
    let (a_white, _, _) = srgb_byte_to_dkl_scalar(255, 255, 255, dm.y_peak, dm.y_black, dm.y_refl);
    assert!(
        a_black < a_mid && a_mid < a_white,
        "Luminance order broken: black={a_black} mid={a_mid} white={a_white}"
    );
    assert!(
        a_black > 0.0,
        "A_black must remain positive (y_black + y_refl > 0)"
    );
}

#[test]
fn linear_in_y_peak() {
    // The function is L_rgb = (y_peak - y_black) * lin + y_black + y_refl
    // and then matrix-multiplied. Fixing (y_black, y_refl) and
    // doubling (y_peak - y_black) should produce ΔA that's (lin × row[0]).
    // That's a linear relationship in y_peak for fixed lin > 0.
    let y_black = 0.2_f32;
    let y_refl = 0.4_f32;
    let r = 128_u8;
    let (a_100, _, _) = srgb_byte_to_dkl_scalar(r, r, r, 100.0, y_black, y_refl);
    let (a_200, _, _) = srgb_byte_to_dkl_scalar(r, r, r, 200.0, y_black, y_refl);
    let (a_400, _, _) = srgb_byte_to_dkl_scalar(r, r, r, 400.0, y_black, y_refl);

    // ΔA(100→200) should be ~half of ΔA(100→400) because (y_peak - y_black)
    // doubled vs quadrupled.
    let d1 = a_200 - a_100;
    let d2 = a_400 - a_100;
    let ratio = d2 / d1;
    assert!(
        (ratio - 3.0).abs() < 1e-3,
        "y_peak linearity broken: d1={d1}, d2={d2}, ratio={ratio} (expected ~3.0)"
    );
}

#[test]
fn boundary_inputs_are_finite_and_dont_panic() {
    // The function indexes SRGB8_TO_LINEAR_LUT[r as usize] with
    // r ∈ {0..=255}. (255, 255, 255) is the last entry; (0, 0, 0)
    // is the first. Neither should panic, and outputs must be
    // finite for every corner pixel.
    let dm = d();
    for (r, g, b) in [
        (0_u8, 0_u8, 0_u8),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 255),
    ] {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(r, g, b, dm.y_peak, dm.y_black, dm.y_refl);
        assert!(
            a.is_finite() && rg.is_finite() && vy.is_finite(),
            "({r}, {g}, {b}) → ({a}, {rg}, {vy}) — non-finite output"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    // Pure function — same args yield bit-identical output.
    let dm = d();
    for (r, g, b) in [(50, 100, 200), (10, 10, 10), (255, 128, 0)] {
        let a1 = srgb_byte_to_dkl_scalar(r, g, b, dm.y_peak, dm.y_black, dm.y_refl);
        let a2 = srgb_byte_to_dkl_scalar(r, g, b, dm.y_peak, dm.y_black, dm.y_refl);
        assert_eq!(
            a1.0.to_bits(),
            a2.0.to_bits(),
            "({r},{g},{b}) A bit-mismatch"
        );
        assert_eq!(
            a1.1.to_bits(),
            a2.1.to_bits(),
            "({r},{g},{b}) RG bit-mismatch"
        );
        assert_eq!(
            a1.2.to_bits(),
            a2.2.to_bits(),
            "({r},{g},{b}) VY bit-mismatch"
        );
    }
}

#[test]
fn pure_red_pushes_rg_positive_relative_to_neutral() {
    // The DKL Rg channel opposes R against (G + B). A pure-red input
    // (255, 0, 0) should give a positive RG; a pure-cyan (0, 255, 255)
    // should give a negative RG. This pins the sign convention of
    // the matrix's row 1 against a swap with row 2 (VY).
    let dm = d();
    let (_, rg_red, _) = srgb_byte_to_dkl_scalar(255, 0, 0, dm.y_peak, dm.y_black, dm.y_refl);
    let (_, rg_cyan, _) = srgb_byte_to_dkl_scalar(0, 255, 255, dm.y_peak, dm.y_black, dm.y_refl);
    assert!(
        rg_red > 0.0,
        "pure red should give positive RG, got {rg_red}"
    );
    assert!(
        rg_cyan < 0.0,
        "pure cyan should give negative RG, got {rg_cyan}"
    );
}

#[test]
fn pure_blue_pushes_vy_positive_relative_to_neutral() {
    // The DKL Vy channel opposes B against (R + G). Pure blue gives
    // positive VY; pure yellow (255, 255, 0) gives negative VY.
    let dm = d();
    let (_, _, vy_blue) = srgb_byte_to_dkl_scalar(0, 0, 255, dm.y_peak, dm.y_black, dm.y_refl);
    let (_, _, vy_yellow) = srgb_byte_to_dkl_scalar(255, 255, 0, dm.y_peak, dm.y_black, dm.y_refl);
    assert!(
        vy_blue > 0.0,
        "pure blue should give positive VY, got {vy_blue}"
    );
    assert!(
        vy_yellow < 0.0,
        "pure yellow should give negative VY, got {vy_yellow}"
    );
}
