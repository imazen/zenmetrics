//! Parity test: cvvdp's `Cvvdp::score` must match cvvdp-gpu's
//! `host_scalar::predict_jod_still_3ch` within ≤ 1e-4 JOD on every
//! synthetic fixture and corpus image. host_scalar IS the
//! f32-precision contract; the GPU implementation is itself locked
//! to host_scalar in cvvdp-gpu's tests.
//!
//! If this test fails the CPU port has diverged from the canonical
//! algorithm — investigate which stage drifted (color / pyramid /
//! csf / masking / pool).

use cvvdp::{Cvvdp, CvvdpParams, DisplayGeometry};
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::DisplayModel;

fn score_via_host_scalar(ref_srgb: &[u8], dist_srgb: &[u8], w: usize, h: usize) -> f32 {
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    predict_jod_still_3ch(ref_srgb, dist_srgb, w, h, display, ppd)
}

fn make_grid(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = vec![0u8; w * h * 3];
    for i in 0..w * h {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3] = (s >> 16) as u8;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3 + 1] = (s >> 16) as u8;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        out[i * 3 + 2] = (s >> 16) as u8;
    }
    out
}

#[test]
fn identical_inputs_yield_jod_10() {
    let w = 64;
    let h = 64;
    let img = make_grid(w, h, 7);
    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let jod = cv.score(&img, &img).unwrap();
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "JOD for identical inputs should be ≈10, got {jod}"
    );
}

#[test]
fn matches_host_scalar_on_random_pairs() {
    // 10 synthetic pairs spanning 64²..512² and a few aspect ratios.
    let cases: &[(usize, usize)] = &[
        (16, 16),
        (24, 32),
        (32, 32),
        (48, 48),
        (64, 64),
        (96, 96),
        (128, 128),
        (192, 128),
        (256, 256),
        (384, 256),
        (512, 512),
    ];
    for (idx, &(w, h)) in cases.iter().enumerate() {
        let r = make_grid(w, h, 1000 + idx as u32);
        // Distort by adding noise.
        let mut d = r.clone();
        let mut s = 42_u32 + idx as u32;
        for v in d.iter_mut() {
            s = s.wrapping_mul(48271);
            let delta = ((s >> 24) as i32 - 128) / 4;
            *v = ((*v as i32 + delta).clamp(0, 255)) as u8;
        }
        let want = score_via_host_scalar(&r, &d, w, h);
        let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
        let got = cv.score(&r, &d).unwrap();
        let diff = (want - got).abs();
        assert!(
            diff < 1e-4,
            "case {w}x{h}: cpu={got} host_scalar={want} diff={diff}"
        );
    }
}

#[test]
fn warm_ref_matches_cold_path() {
    // warm_reference + score_with_warm_ref must produce IDENTICAL
    // output to a one-shot score (within f32 noise; both call the
    // same fold_bands).
    let w = 128;
    let h = 128;
    let r = make_grid(w, h, 333);
    let d_a = make_grid(w, h, 334);
    let d_b = make_grid(w, h, 335);

    let mut cv_cold = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let cold_a = cv_cold.score(&r, &d_a).unwrap();
    let cold_b = cv_cold.score(&r, &d_b).unwrap();

    let mut cv_warm = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    cv_warm.warm_reference(&r).unwrap();
    let warm_a = cv_warm.score_with_warm_ref(&d_a).unwrap();
    let warm_b = cv_warm.score_with_warm_ref(&d_b).unwrap();
    assert!(cv_warm.has_warm_reference());

    let diff_a = (cold_a - warm_a).abs();
    let diff_b = (cold_b - warm_b).abs();
    assert!(
        diff_a < 1e-5,
        "warm vs cold A: {warm_a} vs {cold_a} diff {diff_a}"
    );
    assert!(
        diff_b < 1e-5,
        "warm vs cold B: {warm_b} vs {cold_b} diff {diff_b}"
    );
}

#[test]
fn linear_planes_entry_matches_srgb_byte_entry() {
    use cvvdp_gpu::kernels::color::SRGB8_TO_LINEAR_LUT;
    let w = 64;
    let h = 48;
    let r = make_grid(w, h, 71);
    let d = make_grid(w, h, 72);
    // Build linear planes from sRGB bytes via the LUT.
    let mut rr = vec![0.0_f32; w * h];
    let mut rg = vec![0.0_f32; w * h];
    let mut rb = vec![0.0_f32; w * h];
    let mut dr = vec![0.0_f32; w * h];
    let mut dg = vec![0.0_f32; w * h];
    let mut db = vec![0.0_f32; w * h];
    for i in 0..w * h {
        rr[i] = SRGB8_TO_LINEAR_LUT[r[i * 3] as usize];
        rg[i] = SRGB8_TO_LINEAR_LUT[r[i * 3 + 1] as usize];
        rb[i] = SRGB8_TO_LINEAR_LUT[r[i * 3 + 2] as usize];
        dr[i] = SRGB8_TO_LINEAR_LUT[d[i * 3] as usize];
        dg[i] = SRGB8_TO_LINEAR_LUT[d[i * 3 + 1] as usize];
        db[i] = SRGB8_TO_LINEAR_LUT[d[i * 3 + 2] as usize];
    }
    let mut cv1 = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let mut cv2 = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let jod_byte = cv1.score(&r, &d).unwrap();
    let jod_lin = cv2
        .score_from_linear_planes(&rr, &rg, &rb, &dr, &dg, &db, w)
        .unwrap();
    let diff = (jod_byte - jod_lin).abs();
    assert!(diff < 1e-5, "byte={jod_byte} linear={jod_lin} diff={diff}");
}
