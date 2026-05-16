//! Structural invariant pins on [`mult_mutual_band`] (the band-level
//! 3-channel masking helper). It is the band analogue of
//! `mult_mutual_pixel` and uses `phase_uncertainty_band` internally.
//! The existing coverage in `masking_kernel.rs` is GPU-parity (CPU
//! reference vs GPU output) on fixed inputs; this file pins:
//!
//! - Output shape: 3 channels × `w * h` for the same `(w, h)`.
//! - `T == R` → output identically zero across all 3 channels and all
//!   pixels (the trivial-zero-diff case for a perceptual metric).
//! - Argument symmetry: `f(T, R) == f(R, T)` bit-exact (since both
//!   the `min(|T|, |R|)` masking input and the `|T - R|` diff are
//!   symmetric).
//! - D values are non-negative.
//! - D values bounded by `d_max = 10^D_MAX ≈ 366.69` (clamp_diff_soft).
//! - Determinism.
//! - Output is finite for finite input.

#![allow(clippy::needless_range_loop)]
// Intentional `for cc in 0..3` loops — the per-channel iteration is
// part of the test's readability and the [_; 3] indexing is the
// explicit contract being pinned.

use cvvdp_gpu::kernels::masking::{D_MAX, mult_mutual_band};

fn ramp(w: usize, h: usize, off: f32) -> Vec<f32> {
    (0..w * h).map(|i| (i as f32) * 0.01 + off).collect()
}

#[test]
fn output_shape_matches_input() {
    for (w, h) in [(8_usize, 8_usize), (16, 16), (32, 8)] {
        let t = [ramp(w, h, 0.0), ramp(w, h, 0.5), ramp(w, h, 1.0)];
        let r = [ramp(w, h, 0.1), ramp(w, h, 0.4), ramp(w, h, 0.9)];
        let d = mult_mutual_band(&t, &r, w, h);
        for cc in 0..3 {
            assert_eq!(
                d[cc].len(),
                w * h,
                "({w}, {h}) ch={cc}: d.len() = {} but w*h = {}",
                d[cc].len(),
                w * h
            );
        }
    }
}

#[test]
fn identical_inputs_yield_zero_output() {
    // T == R means |T - R| = 0; safe_pow(0, p) = 0 exactly via
    // `(0 + eps)^p - eps^p`; clamp_diff_soft(0 / anything) = 0.
    let (w, h) = (16_usize, 16_usize);
    let t = [ramp(w, h, 0.0), ramp(w, h, 0.5), ramp(w, h, 1.0)];
    let d = mult_mutual_band(&t, &t, w, h);
    for cc in 0..3 {
        for (i, &v) in d[cc].iter().enumerate() {
            assert_eq!(
                v.to_bits(),
                0.0_f32.to_bits(),
                "T == R ch={cc} [{i}] = {v}, expected 0.0 bit-exact"
            );
        }
    }
}

#[test]
fn symmetric_in_arguments() {
    // f(T, R) == f(R, T) bit-exact — both the masker (min(|T|,|R|))
    // and the diff (|T - R|) are symmetric.
    let (w, h) = (8_usize, 8_usize);
    let t = [ramp(w, h, 0.0), ramp(w, h, 0.3), ramp(w, h, -0.5)];
    let r = [ramp(w, h, 0.4), ramp(w, h, -0.2), ramp(w, h, 0.8)];
    let d_tr = mult_mutual_band(&t, &r, w, h);
    let d_rt = mult_mutual_band(&r, &t, w, h);
    for cc in 0..3 {
        for (i, (&a, &b)) in d_tr[cc].iter().zip(d_rt[cc].iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ch={cc} [{i}]: f(T, R) = {a} vs f(R, T) = {b} (asymmetric?)"
            );
        }
    }
}

#[test]
fn non_negative_output() {
    let (w, h) = (8_usize, 8_usize);
    let t = [ramp(w, h, -1.0), ramp(w, h, 0.3), ramp(w, h, -0.5)];
    let r = [ramp(w, h, 0.4), ramp(w, h, -2.0), ramp(w, h, 0.8)];
    let d = mult_mutual_band(&t, &r, w, h);
    for cc in 0..3 {
        for (i, &v) in d[cc].iter().enumerate() {
            assert!(v >= 0.0, "ch={cc} [{i}] = {v} should be non-negative");
        }
    }
}

#[test]
fn output_bounded_by_d_max() {
    // clamp_diff_soft caps individual D values at d_max ≈ 366.69.
    let d_max = 10.0_f32.powf(D_MAX);
    let (w, h) = (8_usize, 8_usize);

    // Synthesize an extreme contrast — large positive T, large
    // negative R (diff magnitude 2e6) over a small band.
    let t = [vec![1e6_f32; w * h], vec![1e6_f32; w * h], vec![1e6_f32; w * h]];
    let r = [
        vec![-1e6_f32; w * h],
        vec![-1e6_f32; w * h],
        vec![-1e6_f32; w * h],
    ];
    let d = mult_mutual_band(&t, &r, w, h);
    for cc in 0..3 {
        for (i, &v) in d[cc].iter().enumerate() {
            assert!(
                v < d_max,
                "ch={cc} [{i}] = {v} should be strictly < d_max = {d_max}"
            );
            assert!(v.is_finite(), "ch={cc} [{i}] = {v} non-finite");
        }
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let (w, h) = (8_usize, 8_usize);
    let t = [ramp(w, h, 0.0), ramp(w, h, 0.5), ramp(w, h, 1.0)];
    let r = [ramp(w, h, 0.1), ramp(w, h, 0.4), ramp(w, h, 0.9)];
    let a = mult_mutual_band(&t, &r, w, h);
    let b = mult_mutual_band(&t, &r, w, h);
    for cc in 0..3 {
        for (i, (&va, &vb)) in a[cc].iter().zip(b[cc].iter()).enumerate() {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "ch={cc} [{i}] non-deterministic: {va} vs {vb}"
            );
        }
    }
}

#[test]
fn finite_output_for_finite_input() {
    // Mix of small + large + negative inputs.
    let (w, h) = (16_usize, 16_usize);
    let t = [
        ramp(w, h, -1e3),
        ramp(w, h, 0.0),
        ramp(w, h, 1e3),
    ];
    let r = [
        ramp(w, h, 1e3),
        ramp(w, h, -1e3),
        ramp(w, h, 0.0),
    ];
    let d = mult_mutual_band(&t, &r, w, h);
    for cc in 0..3 {
        for (i, &v) in d[cc].iter().enumerate() {
            assert!(v.is_finite(), "ch={cc} [{i}] = {v} non-finite");
        }
    }
}

#[test]
fn small_band_branch_exercised_at_4x4() {
    // 4x4 is below PU_PADSIZE=6 — phase_uncertainty_band takes the
    // small (no-blur) branch internally. Pin that the function
    // still produces valid output without panicking.
    let (w, h) = (4_usize, 4_usize);
    let t = [ramp(w, h, 0.0), ramp(w, h, 0.5), ramp(w, h, 1.0)];
    let r = [ramp(w, h, 0.1), ramp(w, h, 0.4), ramp(w, h, 0.9)];
    let d = mult_mutual_band(&t, &r, w, h);
    for cc in 0..3 {
        assert_eq!(d[cc].len(), 16);
        for &v in &d[cc] {
            assert!(v.is_finite() && v >= 0.0, "ch={cc}: bad value {v}");
        }
    }
}
