//! Invariant pins on [`gaussian_blur_sigma3`]. The function is used
//! by `phase_uncertainty_band`'s large-band branch and indirectly by
//! the GPU pu_blur kernels via `masking_kernel.rs`'s parity tests.
//! No dedicated direct unit tests until now — the existing coverage
//! exercises it only as a CPU reference for GPU parity. This file
//! pins:
//!
//! - Output length matches `w * h`.
//! - Constant input → constant output (DC preservation: `PU_BLUR_KERNEL_1D`
//!   sums to 1, so a uniform value passes through unchanged within f32).
//! - Zero input → zero output (special case).
//! - Reflect-padding boundary doesn't introduce non-finite values.
//! - Non-negative output for non-negative input (kernel is all-positive).
//! - Determinism via `to_bits()`.
//! - Symmetric input (horizontal mirror) produces symmetric output.

use cvvdp_gpu::kernels::masking::gaussian_blur_sigma3;

#[test]
fn output_length_matches_w_times_h() {
    for (w, h) in [(7_usize, 7_usize), (8, 16), (32, 32), (50, 17)] {
        let src = vec![0.0_f32; w * h];
        let out = gaussian_blur_sigma3(&src, w, h);
        assert_eq!(
            out.len(),
            w * h,
            "({w}, {h}): out.len() = {} but w*h = {}",
            out.len(),
            w * h
        );
    }
}

#[test]
fn constant_input_yields_constant_output() {
    // PU_BLUR_KERNEL_1D sums to 1 (DC gain = 1 per axis ⇒ 1×1 = 1
    // for the separable 2D pass). So a uniform input must come
    // back uniform within f32 rounding. Pin via relative tolerance
    // 1e-5 (the kernel sum's deviation from 1.0 is on the order of
    // 1e-7 per coefficient × 13 taps = ~1e-6; squared for 2D pass).
    for value in [1.0_f32, 100.0, -5.0, 1e3] {
        let (w, h) = (16_usize, 16_usize);
        let src = vec![value; w * h];
        let out = gaussian_blur_sigma3(&src, w, h);
        for (i, &v) in out.iter().enumerate() {
            let rel = ((v - value) / value).abs();
            assert!(
                rel < 1e-5,
                "constant-input [{i}] = {v} for value={value} (rel = {rel})"
            );
        }
    }
}

#[test]
fn zero_input_yields_zero_output() {
    let (w, h) = (16_usize, 16_usize);
    let src = vec![0.0_f32; w * h];
    let out = gaussian_blur_sigma3(&src, w, h);
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0.0_f32.to_bits(),
            "zero-input [{i}] = {v}, expected 0.0 bit-exact"
        );
    }
}

#[test]
fn boundary_pixels_remain_finite_under_reflect_padding() {
    // The kernel's 13-tap half-width is 6. For a 7×7 input every
    // pixel touches the reflect-padding logic. Pin no NaN/Inf
    // leaks out of the boundary-index computation.
    let (w, h) = (7_usize, 7_usize);
    let src: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.5).collect();
    let out = gaussian_blur_sigma3(&src, w, h);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "boundary [{i}] = {v} non-finite (reflect padding broke?)"
        );
    }
}

#[test]
fn non_negative_input_yields_non_negative_output() {
    // PU_BLUR_KERNEL_1D entries are all positive (smallest is
    // 1.854e-2 at the tails). Sum of positives × non-negative
    // input cannot go negative.
    let (w, h) = (16_usize, 16_usize);
    let src: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.5).collect();
    let out = gaussian_blur_sigma3(&src, w, h);
    for (i, &v) in out.iter().enumerate() {
        assert!(v >= 0.0, "non-negative-input [{i}] = {v} is negative");
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let (w, h) = (16_usize, 16_usize);
    let src: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let a = gaussian_blur_sigma3(&src, w, h);
    let b = gaussian_blur_sigma3(&src, w, h);
    assert_eq!(a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "[{i}] non-deterministic: {va} vs {vb}"
        );
    }
}

#[test]
fn horizontal_mirror_symmetry_preserved() {
    // The kernel is symmetric (PU_BLUR_KERNEL_1D[t] == [12-t]) AND
    // the reflect-padding boundary is symmetric. So if the input
    // is left-right symmetric (mirror around the vertical axis),
    // the output must be too.
    let (w, h) = (15_usize, 8_usize); // odd width so mirror is well-defined
    let mut src = vec![0.0_f32; w * h];
    // Build a horizontally symmetric pattern: each row mirrors itself.
    for y in 0..h {
        for x in 0..w {
            let d = (x as isize - (w as isize) / 2).unsigned_abs() as f32;
            src[y * w + x] = d; // distance from center column
        }
    }
    let out = gaussian_blur_sigma3(&src, w, h);
    // Verify: out[y * w + x] should equal out[y * w + (w - 1 - x)]
    // for every (x, y) within bit-relevant tolerance.
    for y in 0..h {
        for x in 0..w / 2 {
            let l = out[y * w + x];
            let r = out[y * w + (w - 1 - x)];
            let rel = ((l - r) / l.abs().max(1e-9)).abs();
            assert!(
                rel < 1e-5,
                "asymmetric output at ({x}, {y}): left = {l}, right = {r}"
            );
        }
    }
}

#[test]
fn impulse_input_concentrates_near_impulse_location() {
    // A single-pixel impulse should produce its largest output value
    // AT the impulse location (since the kernel peaks at the center
    // tap, PU_BLUR_KERNEL_1D[6] ≈ 0.137).
    let (w, h) = (16_usize, 16_usize);
    let mut src = vec![0.0_f32; w * h];
    let cx = w / 2;
    let cy = h / 2;
    src[cy * w + cx] = 100.0;
    let out = gaussian_blur_sigma3(&src, w, h);

    let center_val = out[cy * w + cx];
    assert!(center_val > 0.0, "center = {center_val} should be positive");

    // Center must be the max over all 256 cells.
    for y in 0..h {
        for x in 0..w {
            let v = out[y * w + x];
            assert!(
                v <= center_val + 1e-6,
                "out[{x},{y}] = {v} exceeds center {center_val} — impulse spread broken?"
            );
        }
    }
}
