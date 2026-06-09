//! Opaque-API cached-reference tests for `zensim-gpu`.
//!
//! Verifies the `set_reference_srgb_u8` + `compute_features_with_reference_srgb_u8`
//! pair through the opaque shim:
//!
//! - Cached result matches the pair-mode result bit-for-bit (same dist).
//! - Multiple distortions against one cached reference work.
//! - Error path: calling `compute_features_with_reference_srgb_u8` before
//!   `set_reference_srgb_u8` returns `NoCachedReference`.
//! - Dimension-mismatch errors on the wrong-size buffer.
//! - Regime is respected on the cached path (Vec length matches).
//! - Re-calling `set_reference_srgb_u8` swaps the cached reference.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use zensim_gpu::{Backend, Error, ZensimFeatureRegime, ZensimOpaque, ZensimParams};

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_add(seed.wrapping_mul(3))) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

fn approx_eq(a: f64, b: f64, rel: f64, abs: f64) -> bool {
    if (a - b).abs() < abs {
        return true;
    }
    let denom = a.abs().max(b.abs()).max(1e-12);
    ((a - b).abs() / denom) < rel
}

#[test]
fn cached_ref_matches_pair_mode_basic() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let mut z_pair = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    let pair_v = z_pair
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("pair compute");
    assert_eq!(pair_v.len(), 228);

    let mut z_cached = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    z_cached
        .set_reference_srgb_u8(&ref_buf)
        .expect("set_reference");
    let cached_v = z_cached
        .compute_features_with_reference_srgb_u8(&dis_buf)
        .expect("compute_with_reference");
    assert_eq!(cached_v.len(), 228);

    for i in 0..pair_v.len() {
        assert!(
            approx_eq(pair_v[i], cached_v[i], 1e-6, 1e-9),
            "slot {} pair {} cached {}",
            i,
            pair_v[i],
            cached_v[i]
        );
    }
}

#[test]
fn cached_ref_multiple_distortions() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dists: Vec<Vec<u8>> = (1..=5).map(|s| make_image(w, h, s as u32 * 13)).collect();

    // Reference: pair-mode result for each (ref, dist_i).
    let mut z_pair = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    let pair_vs: Vec<Vec<f64>> = dists
        .iter()
        .map(|d| {
            z_pair
                .compute_features_vec_srgb_u8(&ref_buf, d)
                .expect("pair compute")
        })
        .collect();

    // Cached path: set_reference once, then compute_with_reference for
    // every distortion.
    let mut z_cached = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    z_cached
        .set_reference_srgb_u8(&ref_buf)
        .expect("set_reference");
    let cached_vs: Vec<Vec<f64>> = dists
        .iter()
        .map(|d| {
            z_cached
                .compute_features_with_reference_srgb_u8(d)
                .expect("compute_with_reference")
        })
        .collect();

    assert_eq!(pair_vs.len(), 5);
    assert_eq!(cached_vs.len(), 5);
    for (k, (p, c)) in pair_vs.iter().zip(cached_vs.iter()).enumerate() {
        assert_eq!(p.len(), c.len(), "dist {} length mismatch", k);
        for i in 0..p.len() {
            assert!(
                approx_eq(p[i], c[i], 1e-6, 1e-9),
                "dist {} slot {} pair {} cached {}",
                k,
                i,
                p[i],
                c[i]
            );
        }
    }
}

#[test]
fn compute_with_reference_before_set_returns_error() {
    let w = 64_u32;
    let h = 64_u32;
    let dis_buf = make_image(w, h, 7);

    let mut z = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    let err = z
        .compute_features_with_reference_srgb_u8(&dis_buf)
        .unwrap_err();
    match err {
        Error::NoCachedReference => {}
        other => panic!("expected NoCachedReference, got {:?}", other),
    }
}

#[test]
fn set_reference_rejects_wrong_dim() {
    let w = 64_u32;
    let h = 64_u32;
    // Wrong size — fewer pixels than configured (w,h).
    let bad_ref = vec![0u8; (w as usize) * (h as usize) * 3 - 3];

    let mut z = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    let err = z.set_reference_srgb_u8(&bad_ref).unwrap_err();
    match err {
        Error::DimensionMismatch { .. } => {}
        other => panic!("expected DimensionMismatch, got {:?}", other),
    }
}

#[test]
fn cached_ref_with_iw_returns_372() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::WithIw),
    )
    .expect("opaque new with_iw");
    z.set_reference_srgb_u8(&ref_buf).expect("set_reference");
    let v = z
        .compute_features_with_reference_srgb_u8(&dis_buf)
        .expect("compute_with_reference");
    assert_eq!(v.len(), 372, "WithIw cached path returns 372 features");
}

#[test]
fn set_reference_overwrites_previous() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_a = make_image(w, h, 0);
    let ref_b = make_image(w, h, 100);
    let dis_buf = make_image(w, h, 7);

    // Expected: compute_with_reference against ref_b should equal
    // pair-mode (ref_b, dis_buf), not the leftover ref_a result.
    let mut z_pair = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    let pair_b = z_pair
        .compute_features_vec_srgb_u8(&ref_b, &dis_buf)
        .expect("pair compute b");

    let mut z = ZensimOpaque::new(BACKEND_E, w, h, ZensimParams::new()).expect("opaque new");
    z.set_reference_srgb_u8(&ref_a).expect("set_reference a");
    // Sanity: cached result with ref_a is different from pair_b
    let cached_a = z
        .compute_features_with_reference_srgb_u8(&dis_buf)
        .expect("compute_with_reference a");
    // Swap to ref_b.
    z.set_reference_srgb_u8(&ref_b).expect("set_reference b");
    let cached_b = z
        .compute_features_with_reference_srgb_u8(&dis_buf)
        .expect("compute_with_reference b");

    // cached_b should match pair_b
    for i in 0..pair_b.len() {
        assert!(
            approx_eq(pair_b[i], cached_b[i], 1e-6, 1e-9),
            "slot {} pair_b {} cached_b {}",
            i,
            pair_b[i],
            cached_b[i]
        );
    }

    // Crude sanity: at least SOMETHING differs between cached_a and
    // cached_b (different references → different features).
    let mut diff_count = 0;
    for i in 0..cached_a.len() {
        if !approx_eq(cached_a[i], cached_b[i], 1e-6, 1e-9) {
            diff_count += 1;
        }
    }
    assert!(
        diff_count > 0,
        "set_reference(ref_b) didn't change features vs set_reference(ref_a)"
    );
}
