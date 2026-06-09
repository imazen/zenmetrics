//! Structural invariant pins on [`gausspyr_reduce_scalar`]. The
//! existing `pyramid_scalar.rs::reduce_matches_pycvvdp` is a single
//! pointwise-numeric goldens check against pycvvdp v0.5.4 on a fixed
//! 8×8 input — it catches kernel-coefficient drift but won't catch:
//!
//! - Output dimension contract (`(ceil(sw/2), ceil(sh/2))`) being
//!   broken for odd source dimensions.
//! - The returned `(dw, dh)` tuple disagreeing with the `dst.len()`
//!   the function writes.
//! - `dst` being only partially overwritten (a future refactor that
//!   uses `resize_with` and skips some cells).
//! - Non-determinism creeping in (e.g., if the reduce ever moves to
//!   parallel/SIMD with non-associative accumulation).
//! - Width/height swap bugs (the pycvvdp `x.shape[-2]` parity quirk
//!   already wires `sh` parity into the horizontal-patch path, so
//!   accidentally reading `sw` parity would silently corrupt
//!   non-square inputs).
//!
//! Pure-numeric reconstruction quality is intentionally NOT pinned
//! here — that's `reduce_matches_pycvvdp`'s job.

use cvvdp_gpu::kernels::pyramid::gausspyr_reduce_scalar;

/// Reusable per-pixel ramp `(y * w + x) as f32 * 0.01` — deterministic,
/// small magnitude, distinguishable across positions.
fn ramp(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| (i as f32) * 0.01).collect()
}

#[test]
fn output_dimensions_are_ceil_halved_for_even_inputs() {
    let src = ramp(8, 8);
    let mut dst = Vec::new();
    let (dw, dh) = gausspyr_reduce_scalar(&src, 8, 8, &mut dst);
    assert_eq!((dw, dh), (4, 4));
    assert_eq!(dst.len(), 16, "dst.len() must equal dw * dh");
}

#[test]
fn output_dimensions_are_ceil_halved_for_odd_inputs() {
    // Odd dims: ceil(7/2) = 4 → dst is 4 wide, 4 tall for 7×7.
    let src = ramp(7, 7);
    let mut dst = Vec::new();
    let (dw, dh) = gausspyr_reduce_scalar(&src, 7, 7, &mut dst);
    assert_eq!((dw, dh), (4, 4));
    assert_eq!(dst.len(), 16);

    // Non-square odd dims: 17×13 → ceil(17/2)=9, ceil(13/2)=7.
    let src2 = ramp(17, 13);
    let mut dst2 = Vec::new();
    let (dw2, dh2) = gausspyr_reduce_scalar(&src2, 17, 13, &mut dst2);
    assert_eq!((dw2, dh2), (9, 7));
    assert_eq!(dst2.len(), 63);
}

#[test]
fn returned_tuple_agrees_with_dst_length() {
    // The (dw, dh) tuple is documented as the new dimensions. The
    // dst vector must be exactly `dw * dh` after the call — any
    // mismatch means a caller will silently index past valid data
    // or skip valid cells.
    for &(sw, sh) in &[
        (2_usize, 2_usize),
        (3, 3),
        (5, 7),
        (8, 16),
        (16, 8),
        (17, 13),
        (32, 32),
        (64, 48),
        (100, 100),
    ] {
        let src = ramp(sw, sh);
        let mut dst = Vec::new();
        let (dw, dh) = gausspyr_reduce_scalar(&src, sw, sh, &mut dst);
        assert_eq!(
            dst.len(),
            dw * dh,
            "({sw}, {sh}) → ({dw}, {dh}) but dst.len() = {}",
            dst.len()
        );
    }
}

#[test]
fn dst_is_fully_overwritten_even_when_prefilled() {
    // `dst.clear()` + `resize(_, 0.0)` is the documented contract.
    // If a future refactor switches to `resize_with(_, |..| sentinel)`
    // or `set_len` + uninit write, this test trips when a sentinel
    // leaks through.
    let src = ramp(16, 16);
    let mut dst: Vec<f32> = vec![f32::NAN; 9999]; // garbage pre-fill
    let (dw, dh) = gausspyr_reduce_scalar(&src, 16, 16, &mut dst);
    assert_eq!(dst.len(), dw * dh);
    for (i, &v) in dst.iter().enumerate() {
        assert!(
            !v.is_nan(),
            "dst[{i}] = NaN — pre-fill sentinel leaked through"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    // Pure function of (src, sw, sh). Repeated calls must yield
    // bit-identical output via to_bits(); relaxing to relative
    // tolerance would let a non-associative reduction sneak in.
    let src = ramp(64, 32);
    let mut a = Vec::new();
    let mut b = Vec::new();
    let (dwa, dha) = gausspyr_reduce_scalar(&src, 64, 32, &mut a);
    let (dwb, dhb) = gausspyr_reduce_scalar(&src, 64, 32, &mut b);
    assert_eq!((dwa, dha), (dwb, dhb));
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
fn non_square_inputs_distinguish_width_from_height() {
    // Swapping (sw, sh) should produce different (dw, dh) AND
    // different content. This catches a refactor that internally
    // swaps the two dimensions (e.g., reads `sh` where it should
    // read `sw` in a stride computation).
    let src_4x8 = ramp(4, 8);
    let src_8x4 = ramp(8, 4);
    let mut dst_4x8 = Vec::new();
    let mut dst_8x4 = Vec::new();
    let dims_4x8 = gausspyr_reduce_scalar(&src_4x8, 4, 8, &mut dst_4x8);
    let dims_8x4 = gausspyr_reduce_scalar(&src_8x4, 8, 4, &mut dst_8x4);

    assert_eq!(dims_4x8, (2, 4), "(4, 8) → (2, 4)");
    assert_eq!(dims_8x4, (4, 2), "(8, 4) → (4, 2)");
    assert_eq!(dst_4x8.len(), 8);
    assert_eq!(dst_8x4.len(), 8);

    // Confirm the data differs — same length but populated by
    // different ramps, so distinct content.
    let any_diff = dst_4x8
        .iter()
        .zip(dst_8x4.iter())
        .any(|(&a, &b)| a.to_bits() != b.to_bits());
    assert!(
        any_diff,
        "(4, 8) and (8, 4) reduces produced identical output — width/height collapse?"
    );
}

#[test]
fn dst_vec_capacity_can_grow_or_shrink() {
    // The function `clear()`s then `resize()`s — it must work with
    // pre-allocated capacity larger OR smaller than the result.
    // Pin this so a refactor that asserts a specific capacity trips
    // here.
    let src = ramp(8, 8);

    // Pre-allocated way too big:
    let mut dst_big: Vec<f32> = Vec::with_capacity(1_000_000);
    let (dw1, dh1) = gausspyr_reduce_scalar(&src, 8, 8, &mut dst_big);
    assert_eq!((dw1, dh1), (4, 4));
    assert_eq!(dst_big.len(), 16);

    // Pre-allocated way too small (zero):
    let mut dst_small: Vec<f32> = Vec::new();
    let (dw2, dh2) = gausspyr_reduce_scalar(&src, 8, 8, &mut dst_small);
    assert_eq!((dw2, dh2), (4, 4));
    assert_eq!(dst_small.len(), 16);

    // Content matches across the two paths (capacity must not
    // affect the result).
    for (i, (&va, &vb)) in dst_big.iter().zip(dst_small.iter()).enumerate() {
        assert_eq!(
            va.to_bits(),
            vb.to_bits(),
            "[{i}] different output for different starting capacity"
        );
    }
}
