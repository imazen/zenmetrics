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

#[path = "common/mod.rs"]
mod common;

fn ppd() -> f32 {
    DisplayGeometry::STANDARD_4K.pixels_per_degree()
}

fn dm() -> DisplayModel {
    DisplayModel::STANDARD_4K
}

#[test]
fn predict_jod_matches_pycvvdp_at_1024x1024_noise() {
    // Tick 635: host-scalar parity vs pycvvdp v0.5.4 on a 1024×1024
    // per-pixel-per-channel noise distortion. Mirrors the 256² noise
    // fixture at MAX_LEVELS=9-clamped pyramid depth. Tests that
    // high-frequency masking + CSF response is consistent across
    // pyramid depths — noise is the worst-case input for the
    // high-freq pyramid bands (uncorrelated per-pixel, full
    // bandwidth).
    //
    // Noise formula `(x*73 + y*137 + c*211) % 64 - 32` matches the
    // existing 256² noise tests in pipeline_color.rs.
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_1024x1024_noise");
    let (w, h) = (1024_usize, 1024_usize);
    let ref_b = common::synth_pair_ref(w, h);
    let mut dist_b = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            for c in 0..3 {
                let noise = ((x as i64 * 73 + y as i64 * 137 + c as i64 * 211) % 64) - 32;
                let v = (i64::from(ref_b[i + c]) + noise).clamp(0, 255) as u8;
                dist_b[i + c] = v;
            }
        }
    }
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 1024x1024 noise: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_1024x1024_chroma_shift() {
    // Tick 634: host-scalar parity vs pycvvdp v0.5.4 on a 1024×1024
    // chroma-only distortion (G+16). Completes the 128²+256²+1024²
    // chroma_shift triple, pinning chroma behavior across pyramid
    // depths from 6 levels (128²) through 7-8 levels (256²) to
    // MAX_LEVELS=9-clamped (1024²).
    //
    // A refactor that introduced a depth-dependent RG/VY bug would
    // surface at one specific depth, narrowing the regression. The
    // 1024² case specifically tests the MAX_LEVELS-clamp interaction
    // with the chroma-only path.
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_1024x1024_chroma_shift");
    let (w, h) = (1024_usize, 1024_usize);
    let ref_b = common::synth_pair_ref(w, h);
    let dist_b: Vec<u8> = ref_b
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 1024x1024 chroma_shift: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_128x128_chroma_shift() {
    // Tick 633: host-scalar parity vs pycvvdp v0.5.4 on a 128×128
    // chroma-only distortion (G channel +16, R/B unchanged).
    // Mirrors the 256² chroma_shift fixture at a different pyramid
    // depth (6 levels at 128² vs 7-8 at 256²) — tests that the
    // RG/VY-isolation behavior of the DKL stage is consistent
    // across pyramid depths.
    //
    // The pycvvdp golden was generated locally via the pinned
    // `.venv` (pycvvdp 0.5.4 on CPU torch). Stored in
    // `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`.
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_128x128_chroma_shift");
    let (w, h) = (128_usize, 128_usize);
    let ref_b = common::synth_pair_ref(w, h);
    // Same +16 G-channel construction as the 256² chroma_shift tests
    // in pipeline_color.rs.
    let dist_b: Vec<u8> = ref_b
        .chunks_exact(3)
        .flat_map(|p| [p[0], (i16::from(p[1]) + 16).clamp(0, 255) as u8, p[2]])
        .collect();
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 128x128 chroma_shift: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_11x19_tiny_odd() {
    // Tick 632: host-scalar parity vs pycvvdp v0.5.4 on an 11×19
    // (~209 px) TINY odd-dim synth pair. Tiniest viable odd-dim
    // pyramid case — min dim = 11 > PYRAMID_MIN_DIM*2 = 8 just
    // barely. The pyramid is only 2 levels deep here
    // (floor(log2(11)) - 1 = 2), so every band exercises edge
    // handling.
    //
    // Sister to the 73×91 odd-dim fixture (~6.6k px, 5 levels):
    // together they pin odd-dim parity at BOTH extremes of
    // pyramid depth. A refactor that changed the edge-padding
    // semantics or the pycvvdp gausspyr_reduce parity-check bug
    // replication (tick 206) would surface here differently than
    // at 73×91, narrowing the regression.
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_11x19_offset");
    let (w, h) = (11_usize, 19_usize);
    let (ref_b, dist_b) = common::synth_pair_with_offset_dist(w, h);
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 11x19 tiny odd: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_720x1280_offset() {
    // Tick 631: host-scalar parity vs pycvvdp v0.5.4 on a 720×1280
    // (TALL HD aspect, h > w) non-square synth pair. Mirrors tick
    // 630's 1280×720 wide-HD fixture by swapping aspect — same
    // total pixel count, same per-pixel distortion, but pyramid
    // strides are width/height-asymmetric.
    //
    // The two side-by-side pin that the pyramid kernels are
    // width-height-symmetric (no `w >= h` assumption baked in).
    // A refactor that introduced an asymmetric bug would surface
    // as a JOD diff between tall and wide that exceeds the f32
    // noise floor. Measured locally: tall=9.445360, wide=9.454182
    // (diff ~0.009 — non-trivial, reflecting genuinely different
    // per-band downsampling order; both are bit-stable against
    // pycvvdp v0.5.4 individually).
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_720x1280_offset");
    let (w, h) = (720_usize, 1280_usize);
    let (ref_b, dist_b) = common::synth_pair_with_offset_dist(w, h);
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 720x1280 offset: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_1280x720_offset() {
    // Tick 630: host-scalar parity vs pycvvdp v0.5.4 on a 1280×720
    // (HD aspect, ~1 MP) non-square synth pair. Sister to the
    // 1024² fixture (tick 629): min(w, h) = 720 → floor(log2(720))
    // - 1 = 8 raw pyramid levels, NOT MAX_LEVELS=9-clamped. Tests
    // the un-clamped asymmetric pyramid-depth path.
    //
    // The pycvvdp golden was generated locally via the pinned
    // `.venv` (pycvvdp 0.5.4 on CPU torch). Stored in
    // `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`.
    //
    // Tolerance: 0.005 JOD — canonical manifest-parity gate.
    let golden = common::pycvvdp_synth_golden_jod("synth_1280x720_offset");
    let (w, h) = (1280_usize, 720_usize);
    let (ref_b, dist_b) = common::synth_pair_with_offset_dist(w, h);
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 1280x720 offset: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_1024x1024_offset() {
    // Tick 629: host-scalar parity vs pycvvdp v0.5.4 on a 1024×1024
    // (1 MP) synth pair using the offset construction. Fills the
    // size gap between the 256² and 4000×3000 fixtures with a
    // clean power-of-2 1 MP case — exercises the MAX_LEVELS=9
    // pyramid-depth clamp (raw band_frequencies would suggest 10
    // levels for 1024²; pyramid_levels caps to 9).
    //
    // The pycvvdp golden was generated locally via the pinned
    // `.venv` (pycvvdp 0.5.4 on CPU torch). Stored in
    // `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`.
    //
    // Tolerance: 0.005 JOD — same canonical manifest-parity gate
    // as the other fixtures. Runtime: ~10s on a modern x86 host
    // (host-scalar at 1 MP is sequential pyramid + per-pixel CSF).
    let golden = common::pycvvdp_synth_golden_jod("synth_1024x1024_offset");
    let (w, h) = (1024_usize, 1024_usize);
    let (ref_b, dist_b) = common::synth_pair_with_offset_dist(w, h);
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 1024x1024 offset: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
}

#[test]
fn predict_jod_matches_pycvvdp_at_128x128_offset() {
    // Tick 628: host-scalar parity vs pycvvdp v0.5.4 on a 128×128
    // synth pair using the same offset-construction as the 12 MP
    // fixture. Fills the size gap between the 73×91 odd-dim and
    // 256² fixtures with a clean power-of-2 case — exercises a
    // shallower pyramid (one fewer level than 256²) without
    // odd-dim edge handling.
    //
    // The pycvvdp golden was generated locally via
    // `scripts/cvvdp_goldens/.venv/bin/python` (pycvvdp 0.5.4 on
    // CPU torch backend) — same construction is
    // `synth_pair_128_offset` in `bench_12mp_cuda.py`. Stored in
    // `scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`.
    //
    // Tolerance: 0.005 JOD — same canonical manifest-parity gate
    // as the 12 MP / 256² / 73×91 fixtures.
    let golden = common::pycvvdp_synth_golden_jod("synth_128x128_offset");
    let (w, h) = (128_usize, 128_usize);
    let (ref_b, dist_b) = common::synth_pair_with_offset_dist(w, h);
    let jod = predict_jod_still_3ch(&ref_b, &dist_b, w, h, dm(), ppd());
    let diff = (jod - golden).abs();
    eprintln!(
        "host_scalar 128x128 offset: jod = {jod:.6}, pycvvdp golden = {golden:.6}, |diff| = {diff:.6}"
    );
    assert!(jod.is_finite(), "JOD must be finite, got {jod}");
    assert!(
        (0.0..=10.0).contains(&jod),
        "JOD must be in [0, 10], got {jod}"
    );
    assert!(
        diff < 0.005,
        "host_scalar JOD {jod:.6} drifts from pycvvdp golden {golden:.6} by {diff:.6} > 0.005"
    );
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
    assert!(jod <= 10.0 + 1e-3, "JOD = {jod} should be ≤ 10 + ε");
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
    assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic: {a} vs {b}");
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
fn textured_ref_vs_flat_dist_detects_detail_loss() {
    // Tick 543: pin cvvdp's blur/detail-loss detection. A textured
    // ref + flat dist (catastrophic blur — all spatial detail
    // collapsed to a constant) MUST give JOD significantly below 10
    // because the ref carries Weber-pyramid energy that the dist
    // lacks. The masking → pool chain converts this missing-band
    // energy into a non-trivial Q, which met2jod maps below 10.
    //
    // Sibling to flat_vs_flat_yields_max_jod_regardless_of_brightness
    // (tick 542): together they bracket cvvdp's spatial-contrast
    // contract — flat vs flat is JOD=10 (no perceptible distortion),
    // textured vs flat is JOD ≪ 10 (catastrophic blur is caught).
    //
    // A refactor that swallows missing-band energy (e.g. masking
    // that always returns 0, or pool that doesn't weight by
    // BASEBAND_W) would re-promote this to JOD ≈ 10 — a critical
    // regression that this pin surfaces.
    let (w, h) = (32_usize, 32_usize);
    // Textured ref: every pixel different.
    let ref_textured: Vec<u8> = (0..w * h * 3).map(|i| (i % 256) as u8).collect();
    // Flat dist: mid-gray, no spatial variation.
    let dist_flat: Vec<u8> = vec![128u8; w * h * 3];
    let jod = predict_jod_still_3ch(&ref_textured, &dist_flat, w, h, dm(), ppd());
    eprintln!("textured ref vs flat dist: jod = {jod:.4}");
    assert!(jod.is_finite(), "blur JOD must be finite, got {jod}");
    assert!(
        jod < 9.0,
        "textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    // Sanity floor — cvvdp's met2jod is unbounded below but for
    // 32×32 inputs we'd never expect anything truly extreme; pin
    // that we're in the realistic blur-detection regime.
    assert!(
        jod > -10.0,
        "blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn flat_vs_flat_yields_max_jod_regardless_of_brightness() {
    // Tick 542: pin cvvdp's spatial-contrast contract. A flat ref +
    // flat dist (even pure-black vs pure-white) gives JOD ≈ 10
    // because the Weber-contrast pyramid measures spatial contrast
    // WITHIN an image, not absolute differences BETWEEN two images.
    // Both flat inputs have zero contrast at every band → D = 0 →
    // JOD = 10 (the ceiling).
    //
    // A naive intuition would expect "pure black vs pure white" to
    // be a maximally-different pair and produce JOD ≪ 10. cvvdp's
    // design (matching the perceptual phenomenon of luminance
    // adaptation) explicitly returns "no perceptible distortion"
    // for this case. The pin guards against a refactor that adds
    // an absolute-difference term — which would be wrong vs the
    // pycvvdp reference and break parity tests.
    let (w, h) = (32_usize, 32_usize);
    let ref_black: Vec<u8> = vec![0u8; w * h * 3];
    let dist_white: Vec<u8> = vec![255u8; w * h * 3];
    let jod = predict_jod_still_3ch(&ref_black, &dist_white, w, h, dm(), ppd());
    eprintln!("flat-vs-flat (black vs white): jod = {jod:.4}");
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "flat-vs-flat should give JOD ≈ 10 (cvvdp is a spatial-contrast metric, not \
         absolute-difference); got {jod}",
    );

    // Equivalent test with mid-gray vs mid-gray — also flat.
    let ref_gray: Vec<u8> = vec![128u8; w * h * 3];
    let dist_gray: Vec<u8> = vec![64u8; w * h * 3];
    let jod_gray = predict_jod_still_3ch(&ref_gray, &dist_gray, w, h, dm(), ppd());
    assert!(
        (jod_gray - 10.0).abs() < 1e-3,
        "flat 128 vs flat 64 should give JOD ≈ 10 (same reason); got {jod_gray}",
    );
}

#[test]
fn jod_monotonically_decreases_with_noise_amplitude() {
    // Tick 544: pin cvvdp's noise-amplitude monotonicity. A textured
    // reference + dist = ref + alternating-sign noise of amplitude A
    // should give JOD that strictly decreases as A grows. Different
    // hypothesis from `output_responds_to_distortion_magnitude` (which
    // uses *sparse* distortion every 7th byte) — here every byte
    // carries a ± perturbation, so we're probing the dense-noise
    // regime of the masking + pool chain.
    //
    // A bug where the masking stage saturates (clamps Q above some
    // threshold) would flatten the curve at high amplitudes. Three
    // sample points are enough to surface a plateau; the assertion
    // is strict monotonicity across them.
    let (w, h) = (32_usize, 32_usize);
    let ref_: Vec<u8> = (0..w * h * 3).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
                // deterministic ± noise: sign by hash parity, magnitude by amplitude
                let sign: i16 = if ((i * 31).wrapping_add(17)) % 2 == 0 {
                    1
                } else {
                    -1
                };
                let delta = sign * amplitude as i16;
                (b as i16 + delta).clamp(0, 255) as u8
            })
            .collect()
    }

    let jod_a2 = predict_jod_still_3ch(&ref_, &add_alt_noise(&ref_, 2), w, h, dm(), ppd());
    let jod_a8 = predict_jod_still_3ch(&ref_, &add_alt_noise(&ref_, 8), w, h, dm(), ppd());
    let jod_a32 = predict_jod_still_3ch(&ref_, &add_alt_noise(&ref_, 32), w, h, dm(), ppd());

    eprintln!("noise amplitude sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}");

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    // Strict monotonicity: bigger noise → lower JOD.
    assert!(
        jod_a2 > jod_a8,
        "JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8} (more noise = lower JOD)",
    );
    assert!(
        jod_a8 > jod_a32,
        "JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32} (more noise = lower JOD)",
    );
    // All distorted samples below the JOD=10 ceiling.
    assert!(
        jod_a2 < 10.0 - 1e-3,
        "a=2 noise should detectably drop JOD below 10, got {jod_a2}",
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

#[test]
fn non_square_dimensions_are_supported() {
    // The Weber-contrast pyramid + band_frequencies are driven by
    // `min(width, height)` for level count but reduce on both dims
    // independently. Pin that non-square inputs don't panic and
    // produce a finite JOD that respects the identical → 10 contract.
    for (w, h) in [
        (32_usize, 16_usize),
        (16_usize, 32_usize),
        (64, 24),
        (24, 64),
    ] {
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i * 13 + 5) % 256) as u8).collect();
        let jod_ident = predict_jod_still_3ch(&src, &src, w, h, dm(), ppd());
        assert!(
            jod_ident.is_finite() && (jod_ident - 10.0).abs() < 1e-2,
            "({w}, {h}) identical: JOD = {jod_ident}"
        );

        let dist: Vec<u8> = src
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 5 == 0 { b.wrapping_add(64) } else { b })
            .collect();
        let jod_diff = predict_jod_still_3ch(&src, &dist, w, h, dm(), ppd());
        assert!(
            jod_diff.is_finite(),
            "({w}, {h}) diff: JOD = {jod_diff} non-finite"
        );
        assert!(
            jod_diff < jod_ident + 1e-3,
            "({w}, {h}) diff: JOD = {jod_diff} >= ident = {jod_ident}"
        );
    }
}

#[test]
fn odd_dimensions_are_supported() {
    // gausspyr_reduce_scalar's ceil-halving + boundary patches were
    // historically buggy at odd dims (tick 206 — `x.shape[-2]` parity
    // quirk in pycvvdp source). Pin that the composed pipeline still
    // accepts odd inputs through to a finite JOD.
    for (w, h) in [
        (13_usize, 17_usize), // both odd, prime-like
        (15_usize, 15_usize), // square odd
        (73, 91),             // the historical regression case
    ] {
        let src: Vec<u8> = (0..w * h * 3).map(|i| ((i * 17 + 3) % 256) as u8).collect();
        let jod_ident = predict_jod_still_3ch(&src, &src, w, h, dm(), ppd());
        assert!(
            jod_ident.is_finite() && (jod_ident - 10.0).abs() < 1e-2,
            "({w}, {h}) odd identical: JOD = {jod_ident}"
        );

        let dist: Vec<u8> = src
            .iter()
            .enumerate()
            .map(|(i, &b)| if i % 4 == 0 { b.wrapping_add(30) } else { b })
            .collect();
        let jod_diff = predict_jod_still_3ch(&src, &dist, w, h, dm(), ppd());
        assert!(
            jod_diff.is_finite(),
            "({w}, {h}) odd diff: JOD = {jod_diff} non-finite"
        );
    }
}
