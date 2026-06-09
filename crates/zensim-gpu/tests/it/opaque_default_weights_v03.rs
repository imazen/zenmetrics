//! Regression test for the v0.3 shim replacement (task #71).
//!
//! Before the shim was deleted, every `ZensimOpaque::compute_srgb_u8`
//! call with `ZensimParams::default_weights()` silently routed through
//! `score_features_with_profile_and_codec_compat`, which dropped the
//! profile parameter on the floor and recomputed the legacy 228-element
//! linear score under a v0.3 label. The fix wires
//! [`zensim::score_features_with_profile_and_codec`] into
//! [`zensim_gpu::ZensimOpaque`]'s profile-aware scoring path.
//!
//! This test proves the wiring is live by:
//!
//! 1. Constructing a `ZensimOpaque` with `ZensimParams::default_weights()`
//!    (which selects `ZensimProfile::PreviewV0_3` + the 372-feature
//!    `WithIw` regime).
//! 2. Scoring a synthetic 256×256 reference / distorted pair (a
//!    deterministic gradient + noise pattern — no corpus dependency).
//! 3. Computing what the deleted shim would have returned: the legacy
//!    `score_from_features(features[..228], WEIGHTS_PREVIEW_V0_2)`
//!    linear dot product mapped through `100 − 18·d^0.7`.
//! 4. Asserting the opaque score and the legacy linear score are
//!    materially different. If they ever match again, the shim is
//!    back (or the v0.3 spline + per-codec affine collapsed onto the
//!    legacy formula by coincidence, which is also worth surfacing).

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use zensim_gpu::{
    Backend, TOTAL_FEATURES, WEIGHTS_PREVIEW_V0_2, Zensim, ZensimFeatureRegime, ZensimOpaque,
    ZensimParams, score_from_features,
};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

/// Deterministic 256×256 gradient pair: `ref` is a smooth gradient,
/// `dist` is the same gradient plus a low-frequency XOR pattern that
/// produces non-zero values in every feature slot. Both buffers
/// avoid all-zero / all-max corners that short-circuit the metric.
fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let mut r = Vec::with_capacity((w * h * 3) as usize);
    let mut d = Vec::with_capacity((w * h * 3) as usize);
    let mut seed: u32 = 0xC0FFEE_u32;
    for y in 0..h {
        for x in 0..w {
            let rr = ((x.wrapping_add(y)) & 0xff) as u8;
            let gg = ((x.wrapping_mul(3)) & 0xff) as u8;
            let bb = ((y.wrapping_mul(5)) & 0xff) as u8;
            r.extend_from_slice(&[rr, gg, bb]);
            // xorshift32 noise — bounded, deterministic, no_std-friendly.
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let n = (seed & 0x1f) as i32 - 16; // ±16
            d.extend_from_slice(&[
                (rr as i32 + n).clamp(0, 255) as u8,
                (gg as i32 + n).clamp(0, 255) as u8,
                (bb as i32 + n).clamp(0, 255) as u8,
            ]);
        }
    }
    (r, d)
}

/// Mirror of the deleted v0.2 shim path: take the first 228 slots of
/// the feature vector and compute `score_from_features(..)`
/// with `WEIGHTS_PREVIEW_V0_2`. Used to prove the opaque path is
/// NOT producing this number any more.
fn legacy_v0_2_linear_score(features: &[f64]) -> f64 {
    assert!(
        features.len() >= TOTAL_FEATURES,
        "feature vector too short ({} < {})",
        features.len(),
        TOTAL_FEATURES
    );
    let mut head = [0.0_f64; TOTAL_FEATURES];
    head.copy_from_slice(&features[..TOTAL_FEATURES]);
    score_from_features(&head, &WEIGHTS_PREVIEW_V0_2)
}

#[test]
fn default_weights_routes_through_profile_not_shim() {
    let w = 256_u32;
    let h = 256_u32;
    let (ref_rgb, dist_rgb) = synth_pair(w, h);

    // Sanity: `default_weights()` is the path that triggered the bug —
    // it bakes in `Some(ZensimProfile::latest())` AND a `Some(...)`
    // weights vector. Both branches of `score_from_profile_vec` /
    // `score_from_linear` would otherwise short-circuit to NaN.
    let params = ZensimParams::default_weights();
    assert!(
        params.profile.is_some(),
        "default_weights() must carry a profile so the opaque API \
         routes through score_from_profile_vec — if this is None the \
         opaque path silently falls through to score_from_linear and \
         the v0.3 wiring isn't exercised at all"
    );
    assert_eq!(
        params.regime,
        ZensimFeatureRegime::WithIw,
        "default_weights() must select WithIw so the 372-feature \
         vector matches the v0.3 MLP's input shape"
    );

    let mut opaque = match ZensimOpaque::new(BACKEND_E, w, h, params) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[skip] couldn't open GPU at {w}×{h}: {e}");
            return;
        }
    };
    let opaque_score = opaque
        .compute_srgb_u8(&ref_rgb, &dist_rgb)
        .expect("opaque compute_srgb_u8")
        .value;

    // Extract the 372-feature vector the opaque path used (typed
    // call, same regime). The CPU `score_features_with_profile_and_codec`
    // dispatch lives on these features.
    let client = BackendT::client(&Default::default());
    let mut typed = Zensim::<BackendT>::new_with_regime(client, w, h, ZensimFeatureRegime::WithIw)
        .expect("typed new_with_regime");
    let features = typed
        .compute_features_vec(&ref_rgb, &dist_rgb)
        .expect("typed compute_features_vec");
    assert_eq!(features.len(), 372, "WithIw must emit 372 features");

    // What the shim would have returned: legacy v0.2 linear score.
    let legacy = legacy_v0_2_linear_score(&features);

    eprintln!("opaque (v0.3 MLP+spline): {opaque_score:.5}");
    eprintln!("legacy (v0.2 linear)    : {legacy:.5}");
    eprintln!(
        "|Δ|                     : {:.5}",
        (opaque_score - legacy).abs()
    );

    // Sanity checks: both numbers must be finite and inside the dial.
    assert!(
        opaque_score.is_finite(),
        "opaque v0.3 score is non-finite: {opaque_score}"
    );
    assert!(
        legacy.is_finite(),
        "legacy v0.2 score is non-finite: {legacy}"
    );

    // The real check: if the shim is back, opaque_score == legacy
    // (or bit-equivalent). They must differ by a clearly measurable
    // amount on a generic gradient+noise fixture — the v0.3 MLP head
    // + tanh-pin + PCHIP spline shifts the score by O(1) JOD on
    // typical synthetic content. Allow a generous 0.5 floor: bigger
    // than any conceivable f32-vs-f64 drift, small enough that even
    // a near-zero distortion case would still trigger if the shim
    // got reintroduced.
    let diff = (opaque_score - legacy).abs();
    assert!(
        diff > 0.5,
        "opaque v0.3 score ({opaque_score:.5}) and legacy v0.2 \
         linear score ({legacy:.5}) differ by only {diff:.5} — \
         this is the smoking gun that the v0.3 shim is live again \
         (the opaque path is producing the legacy linear formula \
         instead of routing through \
         `zensim::score_features_with_profile_and_codec`). \
         See task #71 / opaque.rs commit history."
    );
}

#[test]
fn opaque_v03_matches_cpu_score_features_with_profile_and_codec() {
    // Direct positive control: the opaque path must produce the same
    // number that `zensim::score_features_with_profile_and_codec`
    // returns when fed the GPU's feature vector — i.e. zensim-gpu and
    // zensim-CPU's MLP head agree on the score given identical
    // features. This pins the wiring shape (right function, right
    // arguments, right error handling) independent of the corpus.
    let w = 256_u32;
    let h = 256_u32;
    let (ref_rgb, dist_rgb) = synth_pair(w, h);

    let params = ZensimParams::default_weights();
    let profile = params.profile.expect("default_weights() has a profile");

    let mut opaque = match ZensimOpaque::new(BACKEND_E, w, h, params) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("[skip] couldn't open GPU at {w}×{h}: {e}");
            return;
        }
    };
    let opaque_score = opaque
        .compute_srgb_u8(&ref_rgb, &dist_rgb)
        .expect("opaque compute_srgb_u8")
        .value;

    let client = BackendT::client(&Default::default());
    let mut typed = Zensim::<BackendT>::new_with_regime(client, w, h, ZensimFeatureRegime::WithIw)
        .expect("typed new_with_regime");
    let features = typed
        .compute_features_vec(&ref_rgb, &dist_rgb)
        .expect("typed compute_features_vec");

    let cpu_score = zensim::score_features_with_profile_and_codec(profile, &features, w, h, None)
        .expect("cpu score_features_with_profile_and_codec");

    eprintln!("opaque: {opaque_score:.5}  cpu(profile_and_codec): {cpu_score:.5}");
    let abs = (opaque_score - cpu_score).abs();
    // Bit-identity expected — both paths feed the same f64 features
    // through the same CPU helper. Any deviation here is a bug in
    // the opaque path's argument plumbing.
    assert!(
        abs < 1e-9,
        "opaque score {opaque_score} diverged from \
         zensim::score_features_with_profile_and_codec {cpu_score} \
         (|Δ| = {abs}) — opaque path is not forwarding features / \
         dims / codec_hint correctly"
    );
}
