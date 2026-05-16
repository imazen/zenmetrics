//! Structural invariant pins on
//! [`laplacian_pyramid_dec_scalar`]. The existing
//! `pyramid_scalar.rs::laplacian_3_levels_matches_pycvvdp` and
//! `one_band_laplacian_matches_pycvvdp` are pointwise-numeric parity
//! checks against pycvvdp v0.5.4 on a single fixed 8×8 input — they
//! catch coefficient drift but won't catch:
//!
//! - Output band count desync from requested `n_levels`.
//! - Auto-level (`n_levels = 0`) selecting the wrong height for
//!   non-square or non-power-of-two inputs.
//! - Band dimensions diverging from the underlying Gaussian pyramid
//!   (which `gausspyr_reduce_scalar`'s ceil-halving fixes).
//! - The baseband (last band) being constructed differently from the
//!   coarsest Gaussian (the docstring guarantees it IS the coarsest
//!   Gaussian, not `gauss - 0`).
//! - Determinism drift from any future internal RNG / accumulation-
//!   order change.
//!
//! Numeric reconstruction quality is intentionally out of scope —
//! that's covered by the parity goldens in `pyramid_scalar.rs`.

use cvvdp_gpu::kernels::pyramid::{Band, gausspyr_reduce_scalar, laplacian_pyramid_dec_scalar};

/// Helper: deterministic per-pixel f32 fill `(y * w + x) as f32 * 0.01`
/// so different sizes produce distinguishable but small-magnitude inputs.
fn ramp(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| (i as f32) * 0.01).collect()
}

#[test]
fn output_band_count_matches_requested_n_levels() {
    let src = ramp(16, 16);
    for n_levels in 1_usize..=4 {
        let bands = laplacian_pyramid_dec_scalar(&src, 16, 16, n_levels);
        assert_eq!(
            bands.len(),
            n_levels,
            "n_levels={n_levels}: got {} bands",
            bands.len()
        );
    }
}

#[test]
fn auto_n_levels_zero_picks_log2_min_dim() {
    // Per source: `n = if n_levels == 0 { min(sw, sh).ilog2() }`.
    // Square 64 → log2(64) = 6 bands.
    let src = ramp(64, 64);
    let bands = laplacian_pyramid_dec_scalar(&src, 64, 64, 0);
    assert_eq!(bands.len(), 6, "64x64 auto → 6 bands");

    // 32 → 5. Non-square — min dimension drives the level count.
    let src_64x32 = ramp(64, 32);
    let bands_64x32 = laplacian_pyramid_dec_scalar(&src_64x32, 64, 32, 0);
    assert_eq!(bands_64x32.len(), 5, "64x32 auto → 5 bands (min=32)");

    let src_32x64 = ramp(32, 64);
    let bands_32x64 = laplacian_pyramid_dec_scalar(&src_32x64, 32, 64, 0);
    assert_eq!(bands_32x64.len(), 5, "32x64 auto → 5 bands (min=32)");
}

#[test]
fn band_dimensions_track_gausspyr_reduce_chain() {
    // Each band[k] has the spatial dims of gauss[k]. gauss[k+1] dims
    // are `ceil(gauss[k].w / 2) × ceil(gauss[k].h / 2)` from
    // `gausspyr_reduce_scalar`. Verify the entire chain.
    let (sw, sh) = (17_usize, 13_usize); // intentionally odd
    let src = ramp(sw, sh);
    let n_levels = 4;
    let bands = laplacian_pyramid_dec_scalar(&src, sw, sh, n_levels);

    // Reproduce the Gaussian chain by hand.
    let mut gw = sw;
    let mut gh = sh;
    let mut prev = src.clone();
    for (k, band) in bands.iter().enumerate() {
        assert_eq!(
            (band.w, band.h),
            (gw, gh),
            "band[{k}] dims = ({}, {}) but gauss[{k}] dims = ({gw}, {gh})",
            band.w,
            band.h
        );
        // Advance to next level via reduce.
        if k + 1 < bands.len() {
            let mut next = Vec::new();
            let (nw, nh) = gausspyr_reduce_scalar(&prev, gw, gh, &mut next);
            gw = nw;
            gh = nh;
            prev = next;
        }
    }
}

#[test]
fn baseband_is_coarsest_gaussian_not_zero() {
    // Per source: the last band is the coarsest gaussian
    // (no subtraction). For a flat-zero input the entire Gaussian
    // pyramid is zero, so this test uses a non-trivial input and
    // confirms that band[N-1] reproduces what an independent
    // `gausspyr_reduce_scalar` chain yields.
    let (sw, sh) = (16_usize, 16_usize);
    let src = ramp(sw, sh);
    let n_levels = 3;
    let bands = laplacian_pyramid_dec_scalar(&src, sw, sh, n_levels);

    // Hand-build the Gaussian chain to the coarsest level.
    let mut gw = sw;
    let mut gh = sh;
    let mut current = src.clone();
    for _ in 0..(n_levels - 1) {
        let mut next = Vec::new();
        let (nw, nh) = gausspyr_reduce_scalar(&current, gw, gh, &mut next);
        gw = nw;
        gh = nh;
        current = next;
    }

    let baseband = bands.last().expect("≥1 band");
    assert_eq!((baseband.w, baseband.h), (gw, gh));
    assert_eq!(
        baseband.data.len(),
        current.len(),
        "baseband data length must match coarsest gaussian"
    );
    for (i, (&got, &exp)) in baseband.data.iter().zip(current.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "baseband[{i}] = {got} but coarsest gaussian = {exp}"
        );
    }
}

#[test]
fn determinism_across_repeated_calls() {
    // Pure function of (src, sw, sh, n_levels). Two identical calls
    // must yield bit-identical outputs — no RNG, no accumulation-order
    // drift across runs. Bit-equality via `to_bits()` is the strongest
    // form of this invariant; relaxing to relative-tolerance would let
    // a non-deterministic reduction sneak in.
    let src = ramp(64, 32);
    let a = laplacian_pyramid_dec_scalar(&src, 64, 32, 4);
    let b = laplacian_pyramid_dec_scalar(&src, 64, 32, 4);
    assert_eq!(a.len(), b.len(), "band count differs across calls");
    for (k, (band_a, band_b)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            (band_a.w, band_a.h),
            (band_b.w, band_b.h),
            "band[{k}] dim mismatch"
        );
        assert_eq!(
            band_a.data.len(),
            band_b.data.len(),
            "band[{k}] data-len mismatch"
        );
        for (i, (&va, &vb)) in band_a.data.iter().zip(band_b.data.iter()).enumerate() {
            assert_eq!(
                va.to_bits(),
                vb.to_bits(),
                "band[{k}][{i}] = {va} vs {vb} (bit-mismatch)"
            );
        }
    }
}

#[test]
fn n_levels_one_returns_single_band_equal_to_input() {
    // Edge case: n_levels=1 means the loop `for k in 0..(n-1)` is
    // empty and the only band pushed is the coarsest gaussian, which
    // IS the input (no reduce ran). Pin this contract so a future
    // refactor that always runs at least one reduce trips here.
    let src = ramp(8, 8);
    let bands = laplacian_pyramid_dec_scalar(&src, 8, 8, 1);
    assert_eq!(bands.len(), 1, "n_levels=1 → 1 band");
    let only = &bands[0];
    assert_eq!((only.w, only.h), (8, 8));
    for (i, (&got, &exp)) in only.data.iter().zip(src.iter()).enumerate() {
        assert_eq!(
            got.to_bits(),
            exp.to_bits(),
            "n_levels=1 band[0][{i}] should equal input: {got} vs {exp}"
        );
    }
}

#[test]
fn band_data_length_equals_w_times_h() {
    // Trivial but worth pinning: a Band struct invariant. Catches a
    // future refactor that resizes width/height fields but forgets to
    // resize the data vector (e.g., uses uninitialized capacity).
    let src = ramp(24, 17);
    let bands = laplacian_pyramid_dec_scalar(&src, 24, 17, 0);
    for (k, Band { w, h, data }) in bands.iter().map(|b| (0, b)).enumerate().map(|(_, b)| b) {
        assert_eq!(
            data.len(),
            w * h,
            "band[{k}]: data.len() = {} but w*h = {}",
            data.len(),
            w * h
        );
    }
}
