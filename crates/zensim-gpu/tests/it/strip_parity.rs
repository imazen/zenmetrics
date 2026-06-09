//! Strip-mode parity tests: GPU `MemoryMode::Strip` vs GPU `MemoryMode::Full`.
//!
//! These tests construct two pipelines for the same image dimensions
//! — one Full, one Strip — and compare the per-feature output. The
//! kernels are the same in both modes; the only differences are
//!   * strip mode runs the kernel multiple times on strip-sized
//!     buffers with body-row gating
//!   * the host accumulates raw sums across strips before normalising
//!
//! Both effects are mathematically equivalent to the full-image path
//! when the strip's halo rows preserve V-blur reach (default halo=40
//! covers R=5 at scale 3). Numerical drift comes from f32 reduction
//! order differences and is expected to be ≤ 5e-3 rel.
//!
//! Phase 1 covers `ZensimFeatureRegime::Basic` (228 features). Phase 2
//! extends to Extended (300), Phase 3 to WithIw (372).

use cubecl::Runtime;
use zensim_gpu::{Zensim, ZensimFeatureRegime};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("zensim-gpu strip_parity test requires either the `cuda` or `wgpu` feature");

macro_rules! make_client {
    () => {
        Backend::client(&Default::default())
    };
}

fn gradient(w: usize, h: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            v.push(r);
            v.push(g);
            v.push(b);
        }
    }
    v
}

fn add_noise(data: &[u8], amount: i16) -> Vec<u8> {
    use std::num::Wrapping;
    let mut out = Vec::with_capacity(data.len());
    let mut seed = Wrapping(12345_u32);
    for &v in data {
        seed = seed * Wrapping(1103515245_u32) + Wrapping(12345_u32);
        let noise = ((seed.0 >> 16) as i16 % (amount * 2 + 1)) - amount;
        out.push((v as i16 + noise).clamp(0, 255) as u8);
    }
    out
}

/// Default relative tolerance. The kernel uses f32 for the V-blur
/// sliding sums (`sum_m1`/`sum_m2`/`sum_sq`/`sum_s12`); strip mode
/// changes the slide-start position vs Full mode, so the slide
/// trajectory takes a different rounding path through f32 land. For
/// pyramid-aligned inputs (h_body and image_h multiples of 8) the
/// drift stays within ~2e-3 rel. For unaligned image heights the
/// boundary strip can drift to ~2e-2 rel — see
/// `boundary_strip_drift_documented` below.
const TOL_REL: f64 = 5e-3;
const TOL_ABS: f64 = 5e-4;

/// Tolerance for the unaligned-height case where strip 2's actual_h
/// is less than h_body + 2 × halo. The f32 V-blur sliding sum drifts
/// more here because Full mode's n_strips at scale 2 may produce a
/// different sliding path. ~2% rel is the empirical bound.
const TOL_REL_BOUNDARY: f64 = 3e-2;
const TOL_ABS_BOUNDARY: f64 = 1e-1;

fn assert_features_close(full: &[f64], strip: &[f64], context: &str) {
    assert_features_close_with_tol(full, strip, context, TOL_REL, TOL_ABS);
}

fn assert_features_close_with_tol(
    full: &[f64],
    strip: &[f64],
    context: &str,
    tol_rel: f64,
    tol_abs: f64,
) {
    assert_eq!(
        full.len(),
        strip.len(),
        "{context}: feature-vec length mismatch"
    );
    let mut max_rel = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut worst_idx = 0;
    let mut failures = Vec::new();
    for (i, (&a, &b)) in full.iter().zip(strip.iter()).enumerate() {
        let abs = (a - b).abs();
        let rel = if a.abs() > 1e-12 { abs / a.abs() } else { abs };
        if rel > max_rel {
            max_rel = rel;
            worst_idx = i;
        }
        if abs > max_abs {
            max_abs = abs;
        }
        if abs > tol_abs && rel > tol_rel {
            failures.push((i, a, b, abs, rel));
        }
    }
    if !failures.is_empty() {
        eprintln!(
            "{context}: {} failures (max |rel| = {:.2e} at idx {})",
            failures.len(),
            max_rel,
            worst_idx
        );
        for (i, a, b, abs, rel) in failures.iter().take(20) {
            eprintln!("  feature[{i}] full={a:.6e} strip={b:.6e} abs={abs:.3e} rel={rel:.3e}");
        }
        panic!("{context}: parity violations");
    }
    eprintln!(
        "{context}: ok ({} features, max |rel| = {:.2e} at idx {}, max |abs| = {:.2e})",
        full.len(),
        max_rel,
        worst_idx,
        max_abs
    );
}

#[test]
fn basic_strip_matches_full_512x512_h_body_128() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 8);

    let mut full = Zensim::<Backend>::new(client.clone(), w as u32, h as u32).unwrap();
    let feat_full = full.compute_features_vec(&ref_img, &dist_img).unwrap();

    let mut strip = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 128).unwrap();
    let feat_strip = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    assert_features_close(&feat_full, &feat_strip, "basic 512x512 h_body=128");
}

#[test]
fn basic_strip_matches_full_768x384_h_body_64() {
    let client = make_client!();
    let w = 768;
    let h = 384;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 12);

    let mut full = Zensim::<Backend>::new(client.clone(), w as u32, h as u32).unwrap();
    let feat_full = full.compute_features_vec(&ref_img, &dist_img).unwrap();

    let mut strip = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 64).unwrap();
    let feat_strip = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    assert_features_close(&feat_full, &feat_strip, "basic 768x384 h_body=64");
}

#[test]
fn basic_strip_matches_full_400x300_h_body_120() {
    // Non-pyramid-aligned image dims (height=300 isn't divisible by
    // 2^(SCALES-1)=8) with boundary strips that have < strip_alloc_h
    // rows. The boundary strip's V-blur sliding sum starts from a
    // different image row than Full mode's GPU sub-strip — f32 drift
    // is larger here (~2% rel for HF features) than for aligned
    // dimensions. See TOL_REL_BOUNDARY docstring.
    let client = make_client!();
    let w = 400;
    let h = 300;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 6);

    let mut full = Zensim::<Backend>::new(client.clone(), w as u32, h as u32).unwrap();
    let feat_full = full.compute_features_vec(&ref_img, &dist_img).unwrap();

    let mut strip = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 120).unwrap();
    let feat_strip = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    assert_features_close_with_tol(
        &feat_full,
        &feat_strip,
        "basic 400x300 h_body=120 (unaligned)",
        TOL_REL_BOUNDARY,
        TOL_ABS_BOUNDARY,
    );
}

#[test]
fn basic_strip_cached_ref_matches_full_512x512() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img1 = add_noise(&ref_img, 4);
    let dist_img2 = add_noise(&ref_img, 12);

    let mut full = Zensim::<Backend>::new(client.clone(), w as u32, h as u32).unwrap();
    full.set_reference(&ref_img).unwrap();
    let feat_full_1 = full.compute_with_reference_vec(&dist_img1).unwrap();
    let feat_full_2 = full.compute_with_reference_vec(&dist_img2).unwrap();

    let mut strip = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 128).unwrap();
    strip.set_reference(&ref_img).unwrap();
    let feat_strip_1 = strip.compute_with_reference_vec(&dist_img1).unwrap();
    let feat_strip_2 = strip.compute_with_reference_vec(&dist_img2).unwrap();
    // Cached-ref strip has slightly higher drift than one-shot
    // because the host-side ref-cache splits the ref re-upload into
    // strips on every dist call — same per-strip f32 V-blur drift.
    // 2e-2 rel handles low-distortion image where peak features are
    // small and ratios amplify drift.
    assert_features_close_with_tol(
        &feat_full_1,
        &feat_strip_1,
        "cached-ref strip dist1",
        TOL_REL_BOUNDARY,
        TOL_ABS_BOUNDARY,
    );
    assert_features_close(&feat_full_2, &feat_strip_2, "cached-ref strip dist2");
}

#[test]
fn strip_constructs_with_default_body() {
    // Smoke test: 8 KP image with default constructor — verify it works
    // for a non-aligned size.
    let client = make_client!();
    let mut strip = Zensim::<Backend>::new_strip(client, 1024, 768, 256).unwrap();
    assert!(strip.is_strip_mode());
    let ref_img = gradient(1024, 768);
    let dist_img = add_noise(&ref_img, 3);
    let feat = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    // 228 features.
    assert_eq!(feat.len(), 228);
    // No NaNs, score-relevant features finite.
    assert!(feat.iter().all(|f| f.is_finite()), "NaN in features");
}

#[test]
fn strip_rejects_misaligned_body() {
    // h_body must be a multiple of STRIP_ALIGN=8.
    let client = make_client!();
    let r = Zensim::<Backend>::new_strip(client, 256, 256, 100);
    assert!(
        r.is_err(),
        "h_body=100 should be rejected (not multiple of 8)"
    );
}

#[test]
fn extended_strip_matches_full_512x512() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 8);

    let mut full = Zensim::<Backend>::new_with_regime(
        client.clone(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let feat_full = full.compute_features_vec(&ref_img, &dist_img).unwrap();

    let mut strip = Zensim::<Backend>::new_strip_with_halo_and_regime(
        client,
        w as u32,
        h as u32,
        128,
        40,
        ZensimFeatureRegime::Extended,
    )
    .unwrap();
    let feat_strip = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    assert_features_close(&feat_full, &feat_strip, "extended 512x512 h_body=128");
}

/// Task #76: diffmap output is identical between strip and full modes
/// because the diffmap production routes through the CPU
/// `zensim::Zensim::compute_with_ref_and_diffmap_linear_planar` path
/// (see `docs/DIFFMAP_DIVERGENCES.md`). Both modes mirror the same
/// reference into `diffmap_state.warm_ref` from `set_reference`, then
/// `score_with_warm_ref_diffmap` calls CPU regardless of GPU mode.
/// This test pins that identity so future changes (e.g., a GPU-native
/// diffmap kernel) don't silently break it.
///
/// Follows the documented usage contract: invoke a diffmap entry-point
/// (here `score_with_diffmap` with a dummy distorted) once before
/// `set_reference` so the lazy `diffmap_state` is allocated; then
/// subsequent `set_reference` calls mirror the reference into
/// `state.warm_ref`. This mirrors the production zensim-fork
/// buttloop's startup sequence.
#[test]
fn diffmap_strip_matches_full_bit_for_bit_512x512() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 8);
    let mut dummy_diff: Vec<f32> = Vec::new();

    let mut full = Zensim::<Backend>::new(client.clone(), w as u32, h as u32).unwrap();
    // Lazy-init diffmap state, then warm via set_reference.
    full.score_with_diffmap(&ref_img, &ref_img, &mut dummy_diff)
        .unwrap();
    full.set_reference(&ref_img).unwrap();
    let mut diff_full: Vec<f32> = Vec::new();
    let score_full = full
        .score_with_warm_ref_diffmap(&dist_img, &mut diff_full)
        .unwrap();

    let mut strip = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 128).unwrap();
    strip
        .score_with_diffmap(&ref_img, &ref_img, &mut dummy_diff)
        .unwrap();
    strip.set_reference(&ref_img).unwrap();
    let mut diff_strip: Vec<f32> = Vec::new();
    let score_strip = strip
        .score_with_warm_ref_diffmap(&dist_img, &mut diff_strip)
        .unwrap();

    assert_eq!(diff_full.len(), diff_strip.len(), "diffmap length mismatch");
    assert_eq!(diff_full.len(), w * h, "diffmap should be width x height");
    // CPU is deterministic; the diffmap should be bit-identical.
    for (i, (&a, &b)) in diff_full.iter().zip(diff_strip.iter()).enumerate() {
        assert_eq!(a, b, "diffmap[{i}] differs: full={a:?} strip={b:?}");
    }
    // Score is f32; deterministic CPU path → bit-identical.
    assert_eq!(score_full, score_strip, "warm-ref score mismatch");
}

/// Task #75: opting out of the device-cached ref XYB pyramid (via
/// `set_reference_host_cached_only`) must yield the SAME features
/// as the device-cached path (within strip's normal f32 drift).
/// Both compute over the same image and identical kernels — they
/// just allocate the ref pyramid differently. Documents that the
/// opt-out lever doesn't change scoring semantics.
#[test]
fn host_cached_only_matches_device_cached_512x512() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 8);

    let mut strip_dev =
        Zensim::<Backend>::new_strip(client.clone(), w as u32, h as u32, 128).unwrap();
    strip_dev.set_reference(&ref_img).unwrap();
    let feat_dev = strip_dev.compute_with_reference_vec(&dist_img).unwrap();

    let mut strip_host = Zensim::<Backend>::new_strip(client, w as u32, h as u32, 128).unwrap();
    strip_host.set_reference_host_cached_only(&ref_img).unwrap();
    let feat_host = strip_host.compute_with_reference_vec(&dist_img).unwrap();

    // Device-cache + host-rebuild paths should agree within strip's
    // own f32 reordering noise — the kernel chain is identical;
    // only the source rows for the ref XYB pyramid differ in
    // origin (full-image scratch vs strip-local scratch). The
    // downscale is bit-exact on aligned strips so we expect tight
    // agreement.
    assert_features_close(
        &feat_dev,
        &feat_host,
        "device-cached vs host-cached ref strip 512x512",
    );
}

#[test]
fn with_iw_strip_matches_full_512x512() {
    let client = make_client!();
    let w = 512;
    let h = 512;
    let ref_img = gradient(w, h);
    let dist_img = add_noise(&ref_img, 8);

    let mut full = Zensim::<Backend>::new_with_regime(
        client.clone(),
        w as u32,
        h as u32,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let feat_full = full.compute_features_vec(&ref_img, &dist_img).unwrap();

    let mut strip = Zensim::<Backend>::new_strip_with_halo_and_regime(
        client,
        w as u32,
        h as u32,
        128,
        40,
        ZensimFeatureRegime::WithIw,
    )
    .unwrap();
    let feat_strip = strip.compute_features_vec(&ref_img, &dist_img).unwrap();
    assert_features_close(&feat_full, &feat_strip, "with_iw 512x512 h_body=128");
}
