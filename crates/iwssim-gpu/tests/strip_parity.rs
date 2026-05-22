//! Strip-processing parity tests for `iwssim-gpu`.
//!
//! `Iwssim::new_strip` builds a strip-mode pipeline; `compute_gray_stripped`
//! loops over per-strip body+halo slices and accumulates partial sums on
//! the host. These tests verify that the strip path produces the same
//! IW-SSIM score as the whole-image path within f32 noise.
//!
//! ## Tolerance band
//!
//! Single-strip degenerate runs (image fits in one body) match the
//! whole-image score to ~0 rel — no reduction reorder, no per-thread
//! accumulation rounding difference. (`measure_drifts` reported 0.000e0
//! for the 256² body=256 case post-f64-cov-finalize.)
//!
//! Multi-strip runs reorder f32 adds across strips; the dominant drift
//! source used to be `cov_finalize_kernel`'s f32 cross-thread sum over
//! 16384 partials per cell (√N · ε_f32 ≈ 7.7e-6 per cell relative,
//! propagated through eigendecomp + Π|wmcs|^β to ~2-3e-4 final score).
//! After the 2026-05-22 tighten-tolerances pass promoted that sum to
//! f64 (the per-thread accumulator stays f32 — small-N register
//! arithmetic), the per-cell floor is now bounded by the f32
//! per-thread accumulator (small N, error ~ε_f32 directly). Measured
//! max rel drift across this test grid (CUDA): 3.6e-6 at 1024² body=512.
//! The 1e-5 tolerance leaves ~3× margin while still catching any real
//! mode-switch / orchestration bug — those would land at 1e-3+.
//!
//! Tests are gated on `cubecl-types` + at least one runtime feature.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use iwssim_gpu::{Error, Iwssim};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

/// Strip-vs-whole multi-strip tolerance. See module docs.
/// Post-f64-cov-finalize: max measured rel drift = 3.6e-6 across the
/// test grid. Set to 1e-5 with ~3× margin.
const STRIP_VS_WHOLE_REL: f64 = 1e-5;
/// Single-strip degenerate (no reduction reorder, identical algorithm
/// trip) tolerance. Post-f64-cov-finalize: measured 0.0 — gate at 1e-7
/// to keep some headroom against future micro-noise.
const STRIP_SINGLE_REL: f64 = 1e-7;
/// Cross-tile-size tolerance: both strip configurations reorder vs
/// the whole-image path, so their mutual drift can be up to
/// ~2 × STRIP_VS_WHOLE_REL. Bound at 5e-5 with margin.
const STRIP_VS_STRIP_REL: f64 = 5e-5;

/// Generate a deterministic grayscale-in-the-f32-0..255-range image.
/// Mixes a low-frequency gradient with a high-frequency texture so
/// the LP cascade actually has work to do (a flat field would give
/// degenerate cs/iw maps that hide reduction bugs).
fn make_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let lf = ((x as f32 / w as f32) * 200.0) + ((y as f32 / h as f32) * 50.0);
            let hf = (((x.wrapping_mul(7).wrapping_add(seed)) & 0x1f) as f32) * 1.5
                + (((y.wrapping_mul(11).wrapping_add(seed.wrapping_mul(3))) & 0x1f)
                    as f32)
                    * 1.5;
            let v = (lf + hf).clamp(0.0, 255.0);
            out.push(v);
        }
    }
    out
}

/// Generate a deterministic RGB-u8 buffer with the same content pattern
/// as `make_gray`, replicated into all three channels (so BT.601 gray
/// extraction yields the same plane as `make_gray`).
fn make_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let g = make_gray(w, h, seed);
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for &v in &g {
        let b = v.clamp(0.0, 255.0).round() as u8;
        out.push(b);
        out.push(b);
        out.push(b);
    }
    out
}

/// Run both paths (whole + strip with given body) and return (whole, strip).
fn run_pair(w: u32, h: u32, seed_a: u32, seed_b: u32, body: u32) -> (f64, f64) {
    let r = make_gray(w, h, seed_a);
    let d = make_gray(w, h, seed_b);

    let client = BackendT::client(&Default::default());
    let mut whole = Iwssim::<BackendT>::new(client.clone(), w, h).expect("whole new");
    let s_whole = whole.compute_gray(&r, &d).expect("whole compute");

    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, body).expect("strip new");
    let s_strip = strip
        .compute_gray_stripped(&r, &d)
        .expect("strip compute");

    (s_whole.score, s_strip.score)
}

// ─────────────────────────────────────────────────────────────────
// Single-strip degenerate case (image fits in one body)
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_matches_whole_at_256() {
    // 256² fits in a single strip (body=256). No reduction reorder →
    // tighter tolerance than the multi-strip cases.
    let (whole, strip) = run_pair(256, 256, 0, 7, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_SINGLE_REL,
        "256² strip {strip} differs from whole {whole} by rel={rel}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Multi-strip cases — gray path
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_matches_whole_at_1024_square() {
    // 1024² with body=256, halo=256 → 4 body strips
    // (rows 0..256, 256..512, 512..768, 768..1024). Exercises real
    // multi-strip orchestration on a square image.
    let (whole, strip) = run_pair(1024, 1024, 0, 13, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "1024² strip {strip} differs from whole {whole} by rel={rel}"
    );
}

#[test]
fn strip_mode_matches_whole_at_1024x768() {
    // Non-square: 1024 wide × 768 tall with body=256 / halo=256.
    // 3 body strips, all aligned.
    let (whole, strip) = run_pair(1024, 768, 0, 19, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "1024×768 strip {strip} differs from whole {whole} by rel={rel}"
    );
}

#[test]
fn strip_mode_matches_whole_at_1024_body_512() {
    // 1024² with body=512: 2 body strips. Tighter loop than body=256.
    let (whole, strip) = run_pair(1024, 1024, 0, 23, 512);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "1024² body=512 strip {strip} differs from whole {whole} by rel={rel}"
    );
}

#[test]
fn strip_mode_matches_whole_at_512_square() {
    // 512² with body=256: 2 body strips. A middle-of-the-band size
    // that catches issues only visible past the single-strip degenerate.
    let (whole, strip) = run_pair(512, 512, 0, 29, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "512² body=256 strip {strip} differs from whole {whole} by rel={rel}"
    );
}

#[test]
fn strip_mode_matches_whole_at_768_square() {
    // 768² with body=256: 3 body strips. image_h IS divisible by h_body.
    let (whole, strip) = run_pair(768, 768, 0, 31, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "768² body=256 strip {strip} differs from whole {whole} by rel={rel}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Edge cases: image_h NOT divisible by h_body
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_handles_uneven_last_strip_896x896_body_256() {
    // 896 / 256 = 3.5 → 3 full strips of 256 + 1 last strip of 128.
    // The last strip's actual upload region is body=128 + halo=256 =
    // 384 rows (or less if image_h doesn't extend the trailing halo).
    let (whole, strip) = run_pair(896, 896, 0, 37, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "896² body=256 (uneven) strip {strip} differs from whole {whole} by rel={rel}"
    );
}

#[test]
fn strip_mode_handles_uneven_last_strip_640x640_body_256() {
    // 640 / 256 = 2.5 → 2 full body strips + 1 last strip of 128 rows.
    let (whole, strip) = run_pair(640, 640, 0, 41, 256);
    let rel = ((strip - whole) / whole).abs();
    assert!(
        rel < STRIP_VS_WHOLE_REL,
        "640² body=256 (uneven) strip {strip} differs from whole {whole} by rel={rel}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Cross-tile-size parity: same image, different body sizes
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_cross_tile_size_parity_1024_square() {
    // The same 1024² pair scored with body=256 vs body=512 should
    // agree within the f32 precision band. Both reorder vs whole, so
    // their mutual drift can be up to 2×STRIP_VS_WHOLE_REL in the
    // worst case — bound to the same precision floor.
    let w = 1024_u32;
    let h = 1024_u32;
    let r = make_gray(w, h, 0);
    let d = make_gray(w, h, 47);
    let client = BackendT::client(&Default::default());

    let mut s256 = Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("s256");
    let v256 = s256.compute_gray_stripped(&r, &d).expect("s256 compute").score;

    let mut s512 = Iwssim::<BackendT>::new_strip(client.clone(), w, h, 512).expect("s512");
    let v512 = s512.compute_gray_stripped(&r, &d).expect("s512 compute").score;

    let rel = ((v256 - v512) / v512).abs();
    assert!(
        rel < STRIP_VS_STRIP_REL,
        "1024² body=256 ({v256}) vs body=512 ({v512}) drift rel={rel}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Self-identity (any path returns 1.0 within f32 noise)
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_self_identity_is_one() {
    // ref == dis → score must be 1.0 (within f32 noise).
    let w = 512_u32;
    let h = 512_u32;
    let r = make_gray(w, h, 0);

    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    let s = strip.compute_gray_stripped(&r, &r).expect("strip compute").score;
    assert!((s - 1.0).abs() < 1e-5, "strip self-identity {s} ≠ 1");
}

#[test]
fn strip_mode_self_identity_multi_strip_uneven() {
    // ref == dis with an uneven last strip; should still hit 1.0
    // exactly modulo f32 noise (every cs / iw is 1.0; reductions sum
    // 1.0 across the body row range).
    let w = 896_u32;
    let h = 896_u32;
    let r = make_gray(w, h, 11);

    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    let s = strip.compute_gray_stripped(&r, &r).expect("strip compute").score;
    assert!(
        (s - 1.0).abs() < 1e-5,
        "uneven-strip self-identity {s} ≠ 1"
    );
}

// ─────────────────────────────────────────────────────────────────
// Constructor validation
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_constructor_rejects_small_image() {
    let client = BackendT::client(&Default::default());
    let r = Iwssim::<BackendT>::new_strip(client, 100, 100, 256);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}

#[test]
fn strip_mode_constructor_rejects_misaligned_body() {
    // Body must be a non-zero multiple of 16 (2^(NUM_SCALES-1)).
    let client = BackendT::client(&Default::default());
    let r = Iwssim::<BackendT>::new_strip(client, 256, 256, 100);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}

#[test]
fn strip_mode_constructor_rejects_zero_body() {
    let client = BackendT::client(&Default::default());
    let r = Iwssim::<BackendT>::new_strip(client, 256, 256, 0);
    assert!(matches!(r, Err(Error::InvalidImageSize)));
}

// ─────────────────────────────────────────────────────────────────
// Cached-reference path: must error in strip mode
// ─────────────────────────────────────────────────────────────────

#[test]
fn strip_mode_set_reference_errors_with_typed_variant() {
    // Strip mode does NOT support the cached-ref fast path. Calling
    // set_reference on a new_strip instance must return the dedicated
    // CachedRefNotSupportedInStripMode variant — never silently
    // produce wrong scores by trying to upload a full-image buffer
    // into a strip-sized scale-0 plane.
    let w = 512_u32;
    let h = 512_u32;
    let r = make_gray(w, h, 0);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    let err = iw.set_reference(&r).expect_err("set_reference must error in strip mode");
    assert!(
        matches!(err, Error::CachedRefNotSupportedInStripMode),
        "expected CachedRefNotSupportedInStripMode, got {err:?}"
    );
    // And the Display impl must not be empty / panicky.
    let msg = format!("{err}");
    assert!(msg.contains("strip mode"), "Display message: {msg}");
    // Cached state must not have been flipped.
    assert!(!iw.has_cached_reference());
}

#[test]
fn strip_mode_compute_with_reference_errors_with_typed_variant() {
    let w = 512_u32;
    let h = 512_u32;
    let d = make_gray(w, h, 1);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    let err = iw
        .compute_with_reference(&d)
        .expect_err("compute_with_reference must error in strip mode");
    assert!(
        matches!(err, Error::CachedRefNotSupportedInStripMode),
        "expected CachedRefNotSupportedInStripMode, got {err:?}"
    );
}

#[test]
fn whole_image_compute_gray_stripped_errors_with_typed_variant() {
    // The reverse direction: calling compute_gray_stripped on a
    // whole-image instance must return NotStripMode, not silently
    // misbehave.
    let w = 256_u32;
    let h = 256_u32;
    let r = make_gray(w, h, 0);
    let d = make_gray(w, h, 5);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new(client, w, h).expect("whole new");
    let err = iw
        .compute_gray_stripped(&r, &d)
        .expect_err("compute_gray_stripped must error on whole-image instance");
    assert!(
        matches!(err, Error::NotStripMode),
        "expected NotStripMode, got {err:?}"
    );
}

// ─────────────────────────────────────────────────────────────────
// Whole-image path regression: the strip work must not have
// changed any pre-existing pair-mode or cached-ref behavior.
// ─────────────────────────────────────────────────────────────────

#[test]
fn whole_image_path_unchanged_by_strip_addition() {
    // Sanity: the new_strip code shouldn't accidentally affect the
    // whole-image path. Score on a fixed seed pair should be finite
    // in [0, 1].
    let w = 256_u32;
    let h = 256_u32;
    let r = make_gray(w, h, 0);
    let d = make_gray(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new(client, w, h).expect("new");
    let s = iw.compute_gray(&r, &d).expect("compute").score;
    assert!(s.is_finite(), "non-finite score {s}");
    assert!(s > 0.0 && s <= 1.0, "score {s} outside [0, 1]");
}

#[test]
fn whole_image_compute_rgb_unchanged_by_strip_addition() {
    // Sanity check that compute_rgb on the whole-image path still
    // operates normally after the strip additions (which only added
    // new methods + an enum variant).
    let w = 256_u32;
    let h = 256_u32;
    let r = make_rgb(w, h, 0);
    let d = make_rgb(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new(client, w, h).expect("new");
    let s = iw.compute_rgb(&r, &d).expect("compute_rgb").score;
    assert!(s.is_finite(), "non-finite rgb score {s}");
    assert!(s > 0.0 && s <= 1.0, "rgb score {s} outside [0, 1]");
}

#[test]
fn whole_image_cached_ref_path_unchanged_by_strip_addition() {
    // set_reference + compute_with_reference still works on the
    // whole-image path. (The strip path returns
    // CachedRefNotSupportedInStripMode; this test only exercises
    // whole-image.)
    let w = 256_u32;
    let h = 256_u32;
    let r = make_gray(w, h, 0);
    let d = make_gray(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut iw = Iwssim::<BackendT>::new(client, w, h).expect("new");
    iw.set_reference(&r).expect("set_reference");
    assert!(iw.has_cached_reference());
    let s = iw.compute_with_reference(&d).expect("cwr").score;
    assert!(s.is_finite() && s > 0.0 && s <= 1.0);
    // Compare against the pair-mode score on the same content — they
    // must agree to floating-point noise.
    let s_pair = iw.compute_gray(&r, &d).expect("pair").score;
    let rel = ((s - s_pair) / s_pair).abs();
    assert!(
        rel < 1e-6,
        "cached-ref ({s}) vs pair ({s_pair}) drift rel={rel}"
    );
}
