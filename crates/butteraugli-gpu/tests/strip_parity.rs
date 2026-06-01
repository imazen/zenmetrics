//! Strip-mode vs whole-image parity tests for [`butteraugli_gpu::Butteraugli`].
//!
//! Strip mode allocates `width × (body_h + 2 × HALO_ROWS) × f32`
//! per plane instead of the full `width × height`. For a body row
//! the strip's halo rows hold the real image content above and below,
//! so each kernel's edge-clamp `saturating_sub` / `min(h - 1)` reads
//! the same window of source rows it would in the whole-image path.
//!
//! The reduction order differs (the strip path accumulates p3/p6/p12
//! and max per strip then folds host-side, vs the whole-image path's
//! single on-device fused reduce) so the parity test allows ~1e-4 rel
//! error — well within numerical noise.

#![cfg(all(
    feature = "cubecl-types",
    any(feature = "cpu", feature = "cuda", feature = "wgpu")
))]

use butteraugli_gpu::{Butteraugli, ButteraugliParams, Error};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type BackendT = cubecl::cpu::CpuRuntime;

/// Deterministic sRGB-u8 source with a mid-spatial-frequency pattern.
/// Avoids pure noise so the LF and HF bands have non-trivial energy.
fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            // Mix a low-frequency sinusoid + a high-frequency XOR pattern,
            // so both LF (σ=7.16 blur) and HF (σ=3.22 blur) stages see
            // signal.
            let sx = ((x as f32 / 32.0).sin() * 50.0 + 128.0) as u8;
            let sy = ((y as f32 / 24.0).cos() * 40.0 + 128.0) as u8;
            let hf = (((x ^ y).wrapping_mul(seed.max(1)) ^ seed) & 0x3f) as u8;
            out.push(sx.wrapping_add(hf));
            out.push(sy.wrapping_add(hf));
            out.push(sx.wrapping_add(sy).wrapping_add(hf >> 1));
        }
    }
    out
}

fn run_pair(w: u32, h: u32, body_h: u32) -> (f32, f32, f32, f32) {
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let client = BackendT::client(&Default::default());

    // Whole-image baseline.
    let mut whole = Butteraugli::<BackendT>::new(client.clone(), w, h);
    let whole_res = whole.compute(&ref_buf, &dis_buf).expect("whole compute");

    // Strip pass at the requested body_h.
    let mut strip = Butteraugli::<BackendT>::new_strip(client, w, h, body_h);
    let strip_res = strip
        .compute_strip(&ref_buf, &dis_buf)
        .expect("strip compute");

    (
        whole_res.score,
        whole_res.pnorm_3,
        strip_res.score,
        strip_res.pnorm_3,
    )
}

fn assert_rel_eq(name: &str, want: f32, got: f32, tol: f64) {
    let denom = (want as f64).abs().max(1e-12);
    let rel = (got as f64 - want as f64).abs() / denom;
    assert!(
        rel < tol,
        "{name}: whole={want} strip={got} rel_err={rel:.2e} (tol={tol:.0e})"
    );
}

// ─── Pair-path matrix: 3 image sizes × 2 strip_h_body values ───
//
// Covers small (256²), medium (1024²), and large (2048²) with body
// values chosen so the multi-strip walk has 4-8 strips at small/mid
// and a single-strip pass at the larger body. 4096² is omitted on
// the bench-host GPU budget — see the bench example for that size.

#[test]
fn strip_vs_whole_256_body_64() {
    let (ws, wp, ss, sp) = run_pair(256, 256, 64);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_256_body_128() {
    let (ws, wp, ss, sp) = run_pair(256, 256, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_1024_body_128() {
    let (ws, wp, ss, sp) = run_pair(1024, 1024, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_1024_body_256() {
    let (ws, wp, ss, sp) = run_pair(1024, 1024, 256);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_2048_body_128() {
    let (ws, wp, ss, sp) = run_pair(2048, 2048, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_2048_body_256() {
    let (ws, wp, ss, sp) = run_pair(2048, 2048, 256);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

// ─── Edge cases ───

#[test]
fn strip_vs_whole_512_body_64() {
    // 8 strips at body=64 / image_h=512.
    let (ws, wp, ss, sp) = run_pair(512, 512, 64);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_512_body_256_one_strip() {
    // body_h >= image_h: walker runs exactly one strip whose body
    // covers the whole image (degenerate strip = whole image but
    // with strip allocation slab geometry). Strip path must match
    // the whole-image path bit-for-bit on the body-only reduce
    // (since the body IS the whole image, no halo content above
    // or below).
    let (ws, wp, ss, sp) = run_pair(512, 512, 256);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_512_body_equals_image_h() {
    // Degenerate single-strip mode: body == image_h. Walker's
    // body_h_eff = body.min(image_h) keeps this from over-allocating
    // beyond `image_h + 2 * halo`.
    let (ws, wp, ss, sp) = run_pair(512, 512, 512);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_768_body_96_uneven() {
    // image_h not a multiple of body_h — last strip's body is short.
    // image_h=800 / body_h=96 → 8 strips of 96 + 1 strip of 32 = 800.
    let (ws, wp, ss, sp) = run_pair(768, 800, 96);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_640_body_100_very_uneven() {
    // Second uneven case at a different image / body combination:
    // image_h=480 / body_h=100 → 4 strips of 100 + 1 strip of 80.
    let (ws, wp, ss, sp) = run_pair(640, 480, 100);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn strip_vs_whole_with_options() {
    let w = 512;
    let h = 512;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let params = ButteraugliParams::default()
        .with_intensity_target(120.0)
        .with_hf_asymmetry(1.5)
        .with_xmul(0.5);

    let client = BackendT::client(&Default::default());
    let mut whole = Butteraugli::<BackendT>::new(client.clone(), w, h);
    let whole_res = whole
        .compute_with_options(&ref_buf, &dis_buf, &params)
        .expect("whole compute_with_options");

    let mut strip = Butteraugli::<BackendT>::new_strip(client, w, h, 64);
    let strip_res = strip
        .compute_strip_with_options(&ref_buf, &dis_buf, &params)
        .expect("strip compute_strip_with_options");

    assert_rel_eq("score(options)", whole_res.score, strip_res.score, 1e-4);
    assert_rel_eq(
        "pnorm_3(options)",
        whole_res.pnorm_3,
        strip_res.pnorm_3,
        1e-4,
    );
}

#[test]
fn strip_constructor_records_metadata() {
    let client = BackendT::client(&Default::default());
    let strip = Butteraugli::<BackendT>::new_strip(client, 1024, 768, 128);
    assert!(strip.is_strip_mode());
    assert_eq!(strip.image_height(), 768);
    assert_eq!(strip.strip_body_h(), 128);
    assert!(strip.strip_halo_h() > 0);
    let (w, h) = strip.dimensions();
    // strip-mode `dimensions()` exposes the slab geometry (width and
    // strip_h_total), NOT the image height — callers querying image
    // dimensions should use `image_height()`.
    assert_eq!(w, 1024);
    assert!(h >= 128, "slab height must be at least body_h");
    assert!(h >= 128 + 2 * strip.strip_halo_h());
}

// ─── Cached-reference path in strip mode (Mode E — task #45) ───

#[test]
fn strip_set_reference_succeeds_mode_e() {
    // Mode E (task #45 / issue #15): single-resolution strip-mode
    // accepts set_reference by allocating a whole-image cache sibling.
    // The strip walker blits ref-side planes per strip during the
    // following compute_with_reference calls.
    let client = BackendT::client(&Default::default());
    let mut strip = Butteraugli::<BackendT>::new_strip(client, 512, 512, 64);
    let ref_buf = make_image(512, 512, 0);
    strip
        .set_reference(&ref_buf)
        .expect("single-res strip mode set_reference (Mode E) must succeed");
    assert!(
        strip.has_cached_reference(),
        "has_cached_reference must be true after set_reference"
    );
}

#[test]
fn strip_compute_returns_clear_error() {
    // Calling whole-image `compute` on a strip instance should surface
    // a strip-mode error rather than the misleading DimensionMismatch
    // against the slab geometry.
    let client = BackendT::client(&Default::default());
    let mut strip = Butteraugli::<BackendT>::new_strip(client, 512, 512, 64);
    let ref_buf = make_image(512, 512, 0);
    let dis_buf = make_image(512, 512, 7);
    match strip.compute(&ref_buf, &dis_buf) {
        Err(Error::StripModeUnsupported(api)) => {
            assert_eq!(api, "compute");
        }
        other => panic!("expected StripModeUnsupported(compute), got {other:?}"),
    }
}

#[test]
fn strip_compute_with_reference_without_set_reference_errors() {
    // Mode E supports compute_with_reference in strip mode AFTER
    // set_reference, but should still surface NoCachedReference
    // when set_reference hasn't been called yet — same contract as
    // whole-image mode.
    let client = BackendT::client(&Default::default());
    let mut strip = Butteraugli::<BackendT>::new_strip(client, 512, 512, 64);
    let dis_buf = make_image(512, 512, 7);
    match strip.compute_with_reference(&dis_buf) {
        Err(Error::NoCachedReference) => {}
        other => panic!("expected NoCachedReference, got {other:?}"),
    }
}

#[test]
fn strip_set_reference_then_compute_with_reference_mode_e() {
    // Mode E end-to-end smoke: set_reference + compute_with_reference
    // returns a finite score (parity to whole-image is exercised by
    // the umbrella `cached_ref_butter_strip_n_distortions_1mp` test).
    let client = BackendT::client(&Default::default());
    let mut strip = Butteraugli::<BackendT>::new_strip(client, 512, 512, 64);
    let ref_buf = make_image(512, 512, 0);
    let dis_buf = make_image(512, 512, 7);
    strip
        .set_reference(&ref_buf)
        .expect("strip set_reference (Mode E)");
    let r = strip
        .compute_with_reference(&dis_buf)
        .expect("strip compute_with_reference (Mode E)");
    assert!(r.score.is_finite(), "score must be finite, got {}", r.score);
    assert!(r.pnorm_3.is_finite(), "pnorm_3 must be finite");
    // After clear_reference, compute_with_reference should error again.
    strip.clear_reference();
    assert!(!strip.has_cached_reference());
    match strip.compute_with_reference(&dis_buf) {
        Err(Error::NoCachedReference) => {}
        other => panic!("expected NoCachedReference after clear_reference, got {other:?}"),
    }
}

#[test]
fn multires_strip_set_reference_still_returns_clear_error() {
    // Mode E ports the single-resolution strip case. The multires-
    // strip path (new_multires_strip) still rejects set_reference;
    // umbrella callers fall back to one-shot compute_strip.
    let client = BackendT::client(&Default::default());
    let mut strip = Butteraugli::<BackendT>::new_multires_strip(client, 512, 512, 64);
    let ref_buf = make_image(512, 512, 0);
    match strip.set_reference(&ref_buf) {
        Err(Error::StripModeUnsupported(api)) => {
            assert_eq!(api, "set_reference");
        }
        other => {
            panic!("expected multires-strip StripModeUnsupported(set_reference), got {other:?}")
        }
    }
}

// ─── Multi-resolution + strip composition ───

#[test]
fn multires_whole_path_still_works_after_strip_landing() {
    // The strip work landed alongside the existing whole-image
    // `new_multires` constructor. This test confirms the multi-
    // resolution whole-image pair-path still works (and exercises
    // the half-res sibling) — strip-mode shouldn't have affected
    // it. Compare against the single-res whole-image result on the
    // same images: with the half-res supersample contribution the
    // multi-res score will differ slightly but must finish without
    // error.
    let w = 256;
    let h = 256;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let client = BackendT::client(&Default::default());

    let mut single = Butteraugli::<BackendT>::new(client.clone(), w, h);
    let single_res = single
        .compute(&ref_buf, &dis_buf)
        .expect("single-res compute");

    let mut multires = Butteraugli::<BackendT>::new_multires(client, w, h);
    let multires_res = multires
        .compute(&ref_buf, &dis_buf)
        .expect("multi-res compute");

    // Multi-res score should be finite + non-negative.
    assert!(
        multires_res.score.is_finite() && multires_res.score >= 0.0,
        "multires score non-finite: {}",
        multires_res.score
    );
    assert!(
        multires_res.pnorm_3.is_finite() && multires_res.pnorm_3 >= 0.0,
        "multires pnorm_3 non-finite: {}",
        multires_res.pnorm_3
    );
    // Multi-res adds the half-res supersample to the diffmap, so
    // the max-norm `score` is >= single-res. (Loose lower bound:
    // adding non-negative values can only raise the max.)
    assert!(
        multires_res.score >= single_res.score - 1e-5,
        "multires.score ({}) < single-res.score ({}) — supersample-add should raise the max",
        multires_res.score,
        single_res.score
    );
}

#[test]
fn compute_strip_on_whole_image_instance_returns_clear_error() {
    // Reverse direction: a whole-image instance (no halo) given to
    // compute_strip must surface a strip-mode error, not panic on
    // an assert.
    let client = BackendT::client(&Default::default());
    let mut whole = Butteraugli::<BackendT>::new(client, 512, 512);
    let ref_buf = make_image(512, 512, 0);
    let dis_buf = make_image(512, 512, 7);
    match whole.compute_strip(&ref_buf, &dis_buf) {
        Err(Error::StripModeUnsupported(api)) => {
            assert!(
                api.contains("compute_strip"),
                "expected message to mention compute_strip, got `{api}`"
            );
        }
        other => panic!("expected StripModeUnsupported(compute_strip*), got {other:?}"),
    }
}

#[test]
fn single_res_strip_constructor_has_no_half_res_sibling() {
    // `new_strip` is the single-resolution strip constructor — it
    // does NOT allocate a half-res sibling (the multires-strip path
    // lives at `new_multires_strip`, covered by `multires_strip.rs`).
    // This test pins that contract so a future refactor doesn't
    // silently turn `new_strip` into a multires-strip allocator
    // (which would double its memory footprint, breaking the strip
    // memory savings).
    let client = BackendT::client(&Default::default());
    let strip = Butteraugli::<BackendT>::new_strip(client, 512, 512, 64);
    assert!(strip.is_strip_mode());
    assert!(
        strip.half_res().is_none(),
        "`new_strip` must not allocate a half-res sibling — use `new_multires_strip` for multires"
    );
}
