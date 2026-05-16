//! Flow invariants on [`predict_jod_still_3ch`] — the composed
//! host-scalar still-image pipeline. Existing coverage in
//! `shadow_jod.rs` is pycvvdp v1 golden-manifest parity; this file
//! pins shape + boundary contracts:
//!
//! - Byte-identical inputs → JOD == 10 (perfect quality).
//! - Output bounded above by 10 + ε.
//! - Determinism.
//! - Sensitivity to magnitude (catches stuck-at-constant refactors).
//! - Panic-on-dim-mismatch contract (the `assert_eq!` guards at the
//!   function entry).

use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

fn ppd() -> f32 {
    DisplayGeometry::STANDARD_4K.pixels_per_degree()
}

fn dm() -> DisplayModel {
    DisplayModel::STANDARD_4K
}

#[test]
fn identical_inputs_yield_max_jod_ten() {
    // byte-for-byte identical means every per-channel diff is zero,
    // the masking → pool → met2jod chain returns 10.0 (the JOD ceiling).
    let (w, h) = (32_usize, 32_usize);
    let src: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
    let jod = predict_jod_still_3ch(&src, &src, w, h, dm(), ppd());
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "identical inputs should give JOD ≈ 10, got {jod}"
    );
}

#[test]
fn output_bounded_above_by_ten() {
    // For any (ref, dist) pair the JOD must be ≤ 10 (with small
    // f32 rounding allowance).
    let (w, h) = (16_usize, 16_usize);
    let ref_: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();

    // Distorted = small perturbation of ref.
    let dist: Vec<u8> = ref_.iter().map(|&b| b.saturating_add(4)).collect();
    let jod = predict_jod_still_3ch(&ref_, &dist, w, h, dm(), ppd());
    assert!(
        jod <= 10.0 + 1e-3,
        "JOD = {jod} should be ≤ 10 + ε"
    );
    assert!(jod.is_finite(), "JOD = {jod} non-finite");
}

#[test]
fn determinism_across_repeated_calls() {
    // Pure function — same inputs yield bit-identical output.
    let (w, h) = (16_usize, 16_usize);
    let ref_: Vec<u8> = (0..w * h * 3).map(|i| ((i * 7) % 256) as u8).collect();
    let dist: Vec<u8> = (0..w * h * 3).map(|i| ((i * 11) % 256) as u8).collect();
    let a = predict_jod_still_3ch(&ref_, &dist, w, h, dm(), ppd());
    let b = predict_jod_still_3ch(&ref_, &dist, w, h, dm(), ppd());
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "non-deterministic: {a} vs {b}"
    );
}

#[test]
fn output_responds_to_distortion_magnitude() {
    // Catches a refactor that accidentally returns a constant. Use
    // a textured (non-flat) reference so the Weber-contrast pyramid
    // has non-zero band content — otherwise a flat ref + uniform
    // shift gives JOD=10 regardless (no high-frequency contrast to
    // perturb, the masking/pool stage sees zero).
    let (w, h) = (16_usize, 16_usize);
    let ref_: Vec<u8> = (0..w * h * 3)
        .map(|i| ((i * 17 + 91) % 256) as u8)
        .collect();
    let dist_small: Vec<u8> = ref_
        .iter()
        .enumerate()
        .map(|(i, &b)| if i % 7 == 0 { b.wrapping_add(2) } else { b })
        .collect();
    let dist_big: Vec<u8> = ref_
        .iter()
        .enumerate()
        .map(|(i, &b)| if i % 7 == 0 { b.wrapping_add(80) } else { b })
        .collect();

    let jod_small = predict_jod_still_3ch(&ref_, &dist_small, w, h, dm(), ppd());
    let jod_big = predict_jod_still_3ch(&ref_, &dist_big, w, h, dm(), ppd());

    assert!(
        (jod_small - jod_big).abs() > 1e-3,
        "JOD insensitive to ±2 vs ±80 sparse distortion: small={jod_small} big={jod_big}"
    );
    assert!(
        jod_big < jod_small,
        "larger distortion should give smaller JOD: small={jod_small} big={jod_big}"
    );
}

#[test]
#[should_panic(expected = "assertion")]
fn panics_on_ref_dim_mismatch() {
    // The function asserts `ref_srgb.len() == width * height * 3`.
    // Pass a 31-byte buffer for a 32×32 image (should be 3072 bytes).
    let (w, h) = (32_usize, 32_usize);
    let bad_ref: Vec<u8> = vec![0; 31]; // way too small
    let dist: Vec<u8> = vec![0; w * h * 3];
    let _ = predict_jod_still_3ch(&bad_ref, &dist, w, h, dm(), ppd());
}

#[test]
#[should_panic(expected = "assertion")]
fn panics_on_dist_dim_mismatch() {
    let (w, h) = (32_usize, 32_usize);
    let ref_: Vec<u8> = vec![0; w * h * 3];
    let bad_dist: Vec<u8> = vec![0; 31];
    let _ = predict_jod_still_3ch(&ref_, &bad_dist, w, h, dm(), ppd());
}

#[test]
fn small_image_smoke() {
    // The smallest image the function should accept is one where
    // band_frequencies returns at least 1 band — practically
    // 8×8 (min(w, h).ilog2() = 3, but band_frequencies clamps to
    // 0.2 cpd cutoff). Confirm 8×8 doesn't panic and yields finite
    // output for identical inputs (= 10) and non-identical (< 10).
    let (w, h) = (8_usize, 8_usize);
    let src: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
    let jod_ident = predict_jod_still_3ch(&src, &src, w, h, dm(), ppd());
    assert!(jod_ident.is_finite() && (jod_ident - 10.0).abs() < 1e-2);

    let dist: Vec<u8> = src.iter().map(|&b| b.wrapping_add(50)).collect();
    let jod_diff = predict_jod_still_3ch(&src, &dist, w, h, dm(), ppd());
    assert!(jod_diff.is_finite());
    assert!(jod_diff < jod_ident + 1e-3);
}
