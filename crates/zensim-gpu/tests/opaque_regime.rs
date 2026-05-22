//! Opaque-API regime selection tests for `zensim-gpu`.
//!
//! Verifies that `ZensimParams::with_regime(...)` plumbs the regime
//! through to the underlying pipeline so callers configuring
//! `Extended` / `WithIw` see all 300 / 372 features via
//! `ZensimOpaque::compute_features_vec_srgb_u8(...)`.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use zensim_gpu::{Backend, ZensimFeatureRegime, ZensimOpaque, ZensimParams};

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

#[test]
fn defaults_to_basic_regime() {
    let params = ZensimParams::new();
    assert_eq!(params.regime, ZensimFeatureRegime::Basic);
}

#[test]
fn with_regime_basic_returns_228() {
    let w = 64_u32;
    let h = 64_u32;
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::Basic),
    )
    .expect("opaque new basic");
    assert_eq!(z.regime(), ZensimFeatureRegime::Basic);

    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let v = z
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute_features_vec_srgb_u8");
    assert_eq!(v.len(), 228, "Basic regime returns 228 features");
}

#[test]
fn with_regime_extended_returns_300() {
    let w = 64_u32;
    let h = 64_u32;
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::Extended),
    )
    .expect("opaque new extended");
    assert_eq!(z.regime(), ZensimFeatureRegime::Extended);

    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let v = z
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute_features_vec_srgb_u8");
    assert_eq!(v.len(), 300, "Extended regime returns 300 features");
}

#[test]
fn with_regime_with_iw_returns_372() {
    let w = 64_u32;
    let h = 64_u32;
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::WithIw),
    )
    .expect("opaque new with_iw");
    assert_eq!(z.regime(), ZensimFeatureRegime::WithIw);

    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let v = z
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute_features_vec_srgb_u8");
    assert_eq!(v.len(), 372, "WithIw regime returns 372 features");
}

#[test]
fn fixed_array_path_truncates_to_228_under_extended() {
    // The fixed-length `compute_features_srgb_u8` path returns
    // exactly 228 floats regardless of regime — that's the documented
    // backwards-compat behaviour. Verify the WithIw-configured opaque
    // still satisfies that contract.
    let w = 64_u32;
    let h = 64_u32;
    let mut z = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::WithIw),
    )
    .expect("opaque new with_iw");

    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let arr = z
        .compute_features_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute_features_srgb_u8");
    assert_eq!(arr.len(), 228);
}

#[test]
fn extended_first_228_match_basic() {
    // The Extended regime's first 228 slots ARE the Basic vector —
    // verify the wider pipeline doesn't perturb the basic block.
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let mut z_basic = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::Basic),
    )
    .expect("opaque new basic");
    let v_basic = z_basic
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute basic");
    assert_eq!(v_basic.len(), 228);

    let mut z_ext = ZensimOpaque::new(
        BACKEND_E,
        w,
        h,
        ZensimParams::new().with_regime(ZensimFeatureRegime::Extended),
    )
    .expect("opaque new extended");
    let v_ext = z_ext
        .compute_features_vec_srgb_u8(&ref_buf, &dis_buf)
        .expect("compute extended");
    assert_eq!(v_ext.len(), 300);

    for i in 0..228 {
        let a = v_basic[i];
        let b = v_ext[i];
        let rel = (a - b).abs() / a.abs().max(1e-12);
        assert!(
            rel < 1e-5 || (a - b).abs() < 1e-9,
            "slot {} basic {} ext {} rel {}",
            i,
            a,
            b,
            rel
        );
    }
}
