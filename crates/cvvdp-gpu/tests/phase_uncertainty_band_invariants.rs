//! Invariant pins on [`phase_uncertainty_band`]. The function has
//! two distinct branches based on the band's spatial dimensions:
//!
//! - **Small-band branch** (`w ≤ PU_PADSIZE` OR `h ≤ PU_PADSIZE`):
//!   pure scaling by `10^MASK_C`. No blur — there's no room for the
//!   σ=3 kernel's 13-tap support.
//! - **Large-band branch** (`w > PU_PADSIZE` AND `h > PU_PADSIZE`):
//!   separable σ=3 Gaussian blur followed by the same scalar.
//!
//! No prior direct unit tests — `mult_mutual_band` calls it but
//! pipeline parity tests don't isolate the branch behavior. A
//! refactor that swaps the branch condition (e.g. `w > 6 || h > 6`
//! instead of `&&`) would silently blur degenerate-strip bands that
//! the σ=3 kernel can't safely cover.

use cvvdp_gpu::kernels::masking::{MASK_C, PU_PADSIZE, phase_uncertainty_band};

// Tick 549: compile-time pin of PU_PADSIZE = 6 — the boundary
// parameter that splits this file's small-band vs large-band tests.
// If a refactor changes PU_PADSIZE, the (6, 6) / (7, 7) hardcoded
// pairs in `branch_boundary_at_pu_padsize` (and "PU_PADSIZE = 6"
// references in the docstrings) would be silently wrong. This
// static assert makes that case fail at compile time. Same pattern
// as ticks 522-524 + 548.
const _: () = assert!(
    PU_PADSIZE == 6,
    "PU_PADSIZE drifted from pycvvdp canonical 6; this test file's hardcoded boundary pairs would be wrong",
);

#[test]
fn small_band_branch_is_pure_scaling() {
    // PU_PADSIZE = 6. For w ≤ 6 OR h ≤ 6 the function MUST skip the
    // blur and just multiply by 10^MASK_C. Output[i] should be
    // bit-equal to `input[i] * scale`.
    let scale = 10.0_f32.powf(MASK_C);
    let cases: Vec<(usize, usize)> = vec![(1, 1), (6, 6), (6, 100), (100, 6), (3, 8), (4, 5)];
    for (w, h) in cases {
        let src: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let out = phase_uncertainty_band(&src, w, h);
        assert_eq!(out.len(), src.len(), "({w}, {h}): length mismatch");
        for (i, (&s, &o)) in src.iter().zip(out.iter()).enumerate() {
            let expected = s * scale;
            assert_eq!(
                o.to_bits(),
                expected.to_bits(),
                "({w}, {h}) [{i}]: got {o}, expected {expected} (bit-mismatch)"
            );
        }
    }
}

#[test]
fn large_band_branch_applies_blur_not_just_scaling() {
    // For w > 6 AND h > 6, the function applies a separable σ=3
    // Gaussian blur before the scalar. Output should NOT bit-equal
    // input × scale (the blur is a spatial average; unless input is
    // already constant, the values diverge from pure scaling).
    let scale = 10.0_f32.powf(MASK_C);
    let (w, h) = (16_usize, 16_usize);

    // Non-constant input (spike at center) — blur diffuses it.
    let mut src = vec![0.0_f32; w * h];
    src[(h / 2) * w + (w / 2)] = 100.0;

    let out = phase_uncertainty_band(&src, w, h);
    assert_eq!(out.len(), src.len());

    let any_diff = src
        .iter()
        .zip(out.iter())
        .any(|(&s, &o)| (s * scale).to_bits() != o.to_bits());
    assert!(
        any_diff,
        "large band ({w}, {h}) produced bit-equal-to-scaling — blur skipped?"
    );
}

#[test]
fn output_length_matches_input_in_both_branches() {
    for (w, h) in [(1_usize, 1_usize), (6, 6), (7, 7), (16, 16), (64, 32)] {
        let src = vec![1.0_f32; w * h];
        let out = phase_uncertainty_band(&src, w, h);
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
fn determinism_across_repeated_calls() {
    // Both branches must be deterministic.
    let small = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2×2 → small branch
    let s_a = phase_uncertainty_band(&small, 2, 2);
    let s_b = phase_uncertainty_band(&small, 2, 2);
    for (i, (&a, &b)) in s_a.iter().zip(s_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "small branch [{i}] non-deterministic"
        );
    }

    let large: Vec<f32> = (0..16 * 16).map(|i| i as f32 * 0.01).collect();
    let l_a = phase_uncertainty_band(&large, 16, 16);
    let l_b = phase_uncertainty_band(&large, 16, 16);
    for (i, (&a, &b)) in l_a.iter().zip(l_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "large branch [{i}] non-deterministic"
        );
    }
}

#[test]
fn empty_input_returns_empty_output() {
    // 0×0 has w == PU_PADSIZE+something? PU_PADSIZE = 6, 0 < 6, so
    // small branch is taken. Pure-scaling over an empty slice yields
    // an empty Vec. No panic, no extra allocation.
    let out = phase_uncertainty_band(&[], 0, 0);
    assert!(out.is_empty());
}

#[test]
fn finite_input_yields_finite_output_in_both_branches() {
    // Small branch (3×3) and large branch (8×8).
    for &(w, h) in &[(3_usize, 3_usize), (8, 8)] {
        let src: Vec<f32> = (0..w * h).map(|i| (i as f32) * 0.1 - 5.0).collect();
        let out = phase_uncertainty_band(&src, w, h);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "({w}, {h}) [{i}] = {v} non-finite");
        }
    }
}

#[test]
fn branch_threshold_pinned_at_pu_padsize() {
    // The branch condition is `w > PU_PADSIZE && h > PU_PADSIZE`.
    // PU_PADSIZE = 6 (pinned in masking_constants.rs). Verify both
    // sides of the boundary:
    // - (6, 6): both ≤, small branch (pure scaling)
    // - (7, 7): both >, large branch (blur applied)
    // - (7, 6): one >, one =, small branch (need BOTH >)
    // - (6, 7): symmetric to above, small branch
    let scale = 10.0_f32.powf(MASK_C);
    assert_eq!(PU_PADSIZE, 6, "test predicated on PU_PADSIZE == 6");

    for &(w, h) in &[(6_usize, 6_usize), (7, 6), (6, 7)] {
        // Each cell index acts as a unique non-zero value so we
        // can detect "any blurring" via inequality with pure scaling.
        let src: Vec<f32> = (0..w * h).map(|i| (i + 1) as f32).collect();
        let out = phase_uncertainty_band(&src, w, h);
        for (i, (&s, &o)) in src.iter().zip(out.iter()).enumerate() {
            let expected = s * scale;
            assert_eq!(
                o.to_bits(),
                expected.to_bits(),
                "({w}, {h}) [{i}]: small-branch bit-mismatch — branch wrongly took blur path?"
            );
        }
    }

    // (7, 7): MUST take the large branch — at least one cell differs
    // from pure scaling because the blur kernel mixes neighbors.
    let src: Vec<f32> = (0..7 * 7).map(|i| (i + 1) as f32).collect();
    let out = phase_uncertainty_band(&src, 7, 7);
    let any_diff = src
        .iter()
        .zip(out.iter())
        .any(|(&s, &o)| (s * scale).to_bits() != o.to_bits());
    assert!(
        any_diff,
        "(7, 7) failed to take large-branch — blur skipped?"
    );
}
