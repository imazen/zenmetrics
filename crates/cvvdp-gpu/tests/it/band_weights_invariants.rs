//! Invariant pins on [`flatten_band_weights`] and
//! [`precomputed_band_weights`] beyond their pointwise-match tests in
//! `csf_scalar.rs`. These cover edges and structural properties that
//! the existing tests don't:
//!
//! - `flatten_band_weights`: empty input, large input, layout
//!   reconstruction, capacity-correctness.
//! - `precomputed_band_weights`: length agrees with `band_frequencies`
//!   across image sizes, output is finite + positive across realistic
//!   `(ppd, l_bkg)` ranges, deterministic for repeated calls.
//!
//! The pointwise-numeric check lives in `csf_scalar.rs`'s
//! `precomputed_band_weights_match_pointwise`; this file is the
//! structural/edge-case complement.

use cvvdp_gpu::kernels::csf::{flatten_band_weights, precomputed_band_weights};
use cvvdp_gpu::kernels::pyramid::band_frequencies;
use cvvdp_gpu::params::DisplayGeometry;

#[test]
fn flatten_band_weights_empty_input_returns_empty() {
    // The helper allocates `weights.len() * 3` capacity up front;
    // for an empty slice that must be 0 and not panic. The function
    // is `#[must_use]` so returning a borrowed/aliased buffer would
    // also be wrong — verify it's a fresh owned Vec.
    let flat = flatten_band_weights(&[]);
    assert!(flat.is_empty(), "empty input must yield empty output");
    assert_eq!(flat.capacity(), 0, "no over-allocation for zero bands");
}

#[test]
fn flatten_band_weights_length_invariant_holds_for_many_sizes() {
    // For any input of length N, output length must be N * 3 (no
    // padding, no truncation, no off-by-one at array boundaries).
    for n in [0_usize, 1, 2, 3, 5, 8, 16, 50] {
        let weights: Vec<[f32; 3]> = (0..n)
            .map(|i| [i as f32, i as f32 + 0.5, i as f32 + 0.75])
            .collect();
        let flat = flatten_band_weights(&weights);
        assert_eq!(
            flat.len(),
            n * 3,
            "len({n}) = {} but expected {}",
            flat.len(),
            n * 3
        );
    }
}

#[test]
fn flatten_band_weights_indexing_invariant() {
    // Documented index convention: `weight_idx = level * 3 + channel`,
    // i.e. flat[level*3 + 0] = A, flat[level*3 + 1] = Rg,
    // flat[level*3 + 2] = Vy. Generate distinguishable values across
    // levels AND channels so a swap (level-major vs channel-major)
    // would trip.
    let weights: Vec<[f32; 3]> = (0..6)
        .map(|lvl| {
            [
                100.0 + (lvl as f32),
                200.0 + (lvl as f32),
                300.0 + (lvl as f32),
            ]
        })
        .collect();
    let flat = flatten_band_weights(&weights);

    for (lvl, w) in weights.iter().enumerate() {
        for ch in 0..3 {
            assert_eq!(flat[lvl * 3 + ch], w[ch], "flat[{lvl} * 3 + {ch}] mismatch");
        }
    }
}

#[test]
fn flatten_band_weights_preserves_nan_inf() {
    // The helper does pure memcpy via `extend_from_slice` — special
    // float values must pass through unchanged. If a future refactor
    // ever filters / clamps / normalizes here, this trips.
    let weights = vec![
        [f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        [0.0, -0.0, 1.0],
    ];
    let flat = flatten_band_weights(&weights);
    assert!(flat[0].is_nan(), "NaN passthrough at [0]");
    assert_eq!(flat[1], f32::INFINITY);
    assert_eq!(flat[2], f32::NEG_INFINITY);
    assert_eq!(flat[3].to_bits(), 0_u32, "+0.0 bit-exact");
    assert_eq!(flat[4].to_bits(), 0x8000_0000_u32, "-0.0 bit-exact");
    assert_eq!(flat[5], 1.0);
}

#[test]
fn precomputed_band_weights_length_tracks_band_frequencies() {
    // Per docstring, the helper returns one [A, Rg, Vy] triple per
    // pyramid band. That count comes from `band_frequencies`, which
    // depends on (ppd, min(width, height)). Sweep a range of image
    // sizes and confirm the lengths agree exactly — a single-off-by-one
    // here would silently shift CSF weights across all bands.
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let l_bkg = 100.0_f32.log10();

    for (w, h) in [
        (16_usize, 16_usize),
        (32, 32),
        (64, 64),
        (256, 256),
        (1024, 768),
        (1920, 1080),
        (3840, 2160),
        // Non-square edges; band count depends on min dim.
        (64, 16),
        (16, 64),
    ] {
        let weights = precomputed_band_weights(ppd, w, h, l_bkg);
        let freqs = band_frequencies(ppd, w, h);
        assert_eq!(
            weights.len(),
            freqs.len(),
            "(w={w}, h={h}): weights.len()={} vs band_frequencies.len()={}",
            weights.len(),
            freqs.len()
        );
    }
}

#[test]
fn precomputed_band_weights_all_finite_and_positive() {
    // CSF sensitivity is a physical contrast-sensitivity value —
    // strictly positive at every realistic operating point. NaN,
    // ±infinity, or non-positive entries would propagate downstream
    // into `weight_band_kernel` and corrupt scoring.
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    // Sweep typical display backgrounds: 0.1 cd/m² (dim) to
    // 1000 cd/m² (HDR peak). log10 → -1.0 .. 3.0.
    for l in [-1.0_f32, 0.0, 1.0, 2.0, 3.0] {
        let weights = precomputed_band_weights(ppd, 512, 512, l);
        for (lvl, [a, rg, vy]) in weights.iter().copied().enumerate() {
            for (tag, v) in [("A", a), ("Rg", rg), ("Vy", vy)] {
                assert!(
                    v.is_finite(),
                    "non-finite {tag} at lvl {lvl}, log_l_bkg={l}: {v}"
                );
                assert!(
                    v > 0.0,
                    "non-positive {tag} at lvl {lvl}, log_l_bkg={l}: {v}"
                );
            }
        }
    }
}

#[test]
fn precomputed_band_weights_is_deterministic() {
    // The helper is a pure function of its arguments. Repeated calls
    // with identical (ppd, w, h, l_bkg) must yield bit-identical
    // results — no internal RNG, no time-dependent state, no
    // floating-point accumulation order drift.
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let l_bkg = 100.0_f32.log10();
    let a = precomputed_band_weights(ppd, 1024, 768, l_bkg);
    let b = precomputed_band_weights(ppd, 1024, 768, l_bkg);
    assert_eq!(a.len(), b.len());
    for (i, ([a_a, a_rg, a_vy], [b_a, b_rg, b_vy])) in
        a.iter().copied().zip(b.iter().copied()).enumerate()
    {
        assert_eq!(a_a.to_bits(), b_a.to_bits(), "lvl {i} A bit-equality");
        assert_eq!(a_rg.to_bits(), b_rg.to_bits(), "lvl {i} Rg bit-equality");
        assert_eq!(a_vy.to_bits(), b_vy.to_bits(), "lvl {i} Vy bit-equality");
    }
}

#[test]
fn flatten_then_index_roundtrip_matches_band_layout() {
    // End-to-end: feed `precomputed_band_weights` into
    // `flatten_band_weights` and confirm that the documented index
    // `level * N_CHANNELS + channel` recovers the original [A, Rg, Vy]
    // triple. This is the contract that `weight_band_kernel` relies
    // on at runtime — getting it wrong would shift channel weights
    // by one slot per level.
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let l_bkg = 100.0_f32.log10();
    let weights = precomputed_band_weights(ppd, 256, 256, l_bkg);
    let flat = flatten_band_weights(&weights);

    assert_eq!(flat.len(), weights.len() * 3);
    for (lvl, &[a, rg, vy]) in weights.iter().enumerate() {
        assert_eq!(flat[lvl * 3], a, "lvl {lvl} A");
        assert_eq!(flat[lvl * 3 + 1], rg, "lvl {lvl} Rg");
        assert_eq!(flat[lvl * 3 + 2], vy, "lvl {lvl} Vy");
    }
}
