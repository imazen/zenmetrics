//! Structural invariant pins on [`gausspyr_expand_scalar`]. Mirror
//! of `gausspyr_reduce_invariants.rs` but for the expand direction.
//! Existing coverage in `pyramid_scalar.rs::expand_matches_pycvvdp`
//! is a single pycvvdp-parity check on a fixed 4×4 → 8×8 input. This
//! file adds:
//!
//! - Output dimension contract — `dst.len() == out_w * out_h`
//!   regardless of caller-passed `out_w` / `out_h` (within the
//!   `[2*sw - 1, 2*sw]` documented range).
//! - The unusual debug-assert range `out_w ∈ [2*sw - 1, 2*sw]`: both
//!   even and odd target dims must work.
//! - `dst` is fully overwritten (pre-fill with NaN, none survive).
//! - Determinism via `to_bits()`.
//! - Caller-provided `dst` capacity (too-big or zero) doesn't change
//!   the result.
//! - Width/height swap distinguishability.
//!
//! Like the reduce file, pure-numeric DC-preservation / reconstruction
//! quality is NOT pinned here — it would require the wrap-around edge
//! behavior to be analyzed and is the parity goldens' job.

use cvvdp_gpu::kernels::pyramid::gausspyr_expand_scalar;

fn ramp(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| (i as f32) * 0.01).collect()
}

#[test]
fn dst_len_matches_user_specified_dimensions() {
    // Per the debug_assert range: out_w ∈ [2*sw - 1, 2*sw],
    // out_h ∈ [2*sh - 1, 2*sh]. Both ends of each range must work.
    let src = ramp(4, 4);

    for &(out_w, out_h) in &[
        (7_usize, 7_usize), // 2*sw - 1, 2*sh - 1 (both odd)
        (7, 8),             // 2*sw - 1, 2*sh
        (8, 7),             // 2*sw, 2*sh - 1
        (8, 8),             // 2*sw, 2*sh
    ] {
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, 4, 4, out_w, out_h, &mut dst);
        assert_eq!(
            dst.len(),
            out_w * out_h,
            "(out_w={out_w}, out_h={out_h}): dst.len() = {}",
            dst.len()
        );
    }
}

#[test]
fn dst_is_fully_overwritten_even_when_prefilled() {
    let src = ramp(4, 4);
    let mut dst: Vec<f32> = vec![f32::NAN; 9999];
    gausspyr_expand_scalar(&src, 4, 4, 8, 8, &mut dst);
    assert_eq!(dst.len(), 64);
    for (i, &v) in dst.iter().enumerate() {
        assert!(!v.is_nan(), "dst[{i}] = NaN — pre-fill leaked");
    }
}

#[test]
fn determinism_across_repeated_calls() {
    let src = ramp(8, 4);
    let mut a = Vec::new();
    let mut b = Vec::new();
    gausspyr_expand_scalar(&src, 8, 4, 16, 8, &mut a);
    gausspyr_expand_scalar(&src, 8, 4, 16, 8, &mut b);
    assert_eq!(a.len(), b.len());
    for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "[{i}] = {va} vs {vb} (bit-mismatch)"
        );
    }
}

#[test]
fn dst_vec_capacity_can_grow_or_shrink() {
    let src = ramp(4, 4);

    let mut dst_big: Vec<f32> = Vec::with_capacity(1_000_000);
    gausspyr_expand_scalar(&src, 4, 4, 8, 8, &mut dst_big);
    assert_eq!(dst_big.len(), 64);

    let mut dst_small: Vec<f32> = Vec::new();
    gausspyr_expand_scalar(&src, 4, 4, 8, 8, &mut dst_small);
    assert_eq!(dst_small.len(), 64);

    for (i, (&va, &vb)) in dst_big.iter().zip(dst_small.iter()).enumerate() {
        assert_eq!(va.to_bits(), vb.to_bits(), "[{i}] capacity affected output");
    }
}

#[test]
fn non_square_inputs_distinguish_width_from_height() {
    // (sw=4, sh=2) → (8, 4) is the natural expand. Swap to (sw=2, sh=4)
    // → (4, 8) and the output dims AND content must differ.
    let src_4x2 = ramp(4, 2);
    let src_2x4 = ramp(2, 4);

    let mut dst_4x2 = Vec::new();
    let mut dst_2x4 = Vec::new();
    gausspyr_expand_scalar(&src_4x2, 4, 2, 8, 4, &mut dst_4x2);
    gausspyr_expand_scalar(&src_2x4, 2, 4, 4, 8, &mut dst_2x4);

    assert_eq!(dst_4x2.len(), 32, "(sw=4, sh=2) → 8×4");
    assert_eq!(dst_2x4.len(), 32, "(sw=2, sh=4) → 4×8");

    // Same lengths, different ramps + different convolution paths —
    // any bit overlap would be coincidence.
    let bit_overlap = dst_4x2
        .iter()
        .zip(dst_2x4.iter())
        .filter(|&(&a, &b)| a.to_bits() == b.to_bits())
        .count();
    assert!(
        bit_overlap < dst_4x2.len(),
        "(sw=4, sh=2) and (sw=2, sh=4) produced fully-identical bits — \
         width/height collapse?"
    );
}

#[test]
fn even_and_odd_target_dims_both_succeed() {
    // The `odd_h` and `odd_w` branches inside the function select
    // different `back_idx` values. Both must produce valid output —
    // pin via 5×5 → 9×9 (odd) AND 5×5 → 10×10 (even).
    let src = ramp(5, 5);

    let mut dst_odd = Vec::new();
    gausspyr_expand_scalar(&src, 5, 5, 9, 9, &mut dst_odd);
    assert_eq!(dst_odd.len(), 81);
    assert!(
        dst_odd.iter().all(|v| v.is_finite()),
        "odd target dims produced non-finite output"
    );

    let mut dst_even = Vec::new();
    gausspyr_expand_scalar(&src, 5, 5, 10, 10, &mut dst_even);
    assert_eq!(dst_even.len(), 100);
    assert!(
        dst_even.iter().all(|v| v.is_finite()),
        "even target dims produced non-finite output"
    );
}

#[test]
fn output_finite_for_typical_pyramid_dim_pairs() {
    // Sweep the (sw, sh) → (out_w, out_h) pairs that naturally arise
    // in a Laplacian-pyramid expand of a non-power-of-two image
    // (where each level's "expand to fine size" uses 2*coarse or
    // 2*coarse - 1 depending on the fine's parity).
    for &(sw, sh, out_w, out_h) in &[
        (4_usize, 4_usize, 7_usize, 7_usize),
        (4, 4, 8, 8),
        (8, 4, 16, 7),
        (5, 3, 9, 5),
        (16, 16, 32, 32),
        (3, 3, 6, 5),
        (3, 3, 5, 5),
    ] {
        let src = ramp(sw, sh);
        let mut dst = Vec::new();
        gausspyr_expand_scalar(&src, sw, sh, out_w, out_h, &mut dst);
        assert_eq!(dst.len(), out_w * out_h);
        for (i, &v) in dst.iter().enumerate() {
            assert!(
                v.is_finite(),
                "({sw}, {sh}) → ({out_w}, {out_h}) dst[{i}] = {v}"
            );
        }
    }
}
