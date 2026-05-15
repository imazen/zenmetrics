//! `Cvvdp::score` end-to-end against the v1 R2 manifest values.
//!
//! Routes through `Cvvdp::compute_dkl_jod` as of tick 213; the
//! prior host-scalar path is still callable directly via
//! `host_scalar::predict_jod_still_3ch`. Measured manifest diffs
//! across q=1, 5, 20, 45, 70, 90 are 0.0000–0.0033 JOD, well
//! within the canonical 0.005 manifest-parity tolerance.

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "common/mod.rs"]
mod common;

use common::{Backend, load_rgb_bytes};

#[test]
fn cvvdp_score_matches_v1_manifest() {
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // (q, pycvvdp_manifest_jod) — loaded from
    // scripts/cvvdp_goldens/v1_corpus_jods.json (mirrors R2 v1).
    let qs = common::v1_corpus_qs();
    let cases: Vec<(u32, f32)> = qs
        .iter()
        .map(|&q| (q, common::v1_corpus_jod_golden(q)))
        .collect();
    for &(q, expected) in &cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        let diff = (jod as f32 - expected).abs();
        eprintln!("q={q:>2}: JOD = {jod:.4} (pycvvdp {expected:.4}, |diff| {diff:.4})");
        assert!(
            diff < 0.005,
            "q={q}: Cvvdp::score returned {jod}, pycvvdp manifest {expected}, |diff| {diff:.4} > 0.005"
        );
    }
}

#[test]
fn cvvdp_score_respects_custom_geometry() {
    // Same image pair, two different display geometries — the JOD
    // should differ because PPD differs (higher PPD = more pixels
    // per degree = lower spatial frequency per pyramid band =
    // different CSF weighting). The exact deltas depend on the
    // image; we just assert that (a) both calls succeed, (b) both
    // are in the valid JOD range, and (c) different geometries
    // produce a measurable difference.
    let client_4k = Backend::client(&Default::default());
    let client_phone = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);

    let mut cvvdp_4k =
        Cvvdp::<Backend>::new(client_4k, w, h, CvvdpParams::PLACEHOLDER).expect("new 4k");

    let phone_geom = DisplayGeometry {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.40,
        diagonal_inches: 5.5,
    };
    let mut cvvdp_phone = Cvvdp::<Backend>::new_with_geometry(
        client_phone,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        phone_geom,
    )
    .expect("new phone");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);

    let jod_4k = cvvdp_4k.score(&ref_bytes, &dist_bytes).expect("4k");
    let jod_phone = cvvdp_phone.score(&ref_bytes, &dist_bytes).expect("phone");
    eprintln!("q20 @ standard_4k: JOD = {jod_4k:.4}");
    eprintln!("q20 @ phone:       JOD = {jod_phone:.4}");

    assert!((0.0..=10.0).contains(&jod_4k));
    assert!((0.0..=10.0).contains(&jod_phone));
    assert!(
        (jod_4k - jod_phone).abs() > 1e-3,
        "geometries differ; JODs should not be identical: 4k={jod_4k}, phone={jod_phone}"
    );
}

#[test]
fn score_with_reference_matches_score() {
    // Tick 213 routed both score and score_with_reference through
    // the GPU compute_dkl_jod path. The contract is exact parity
    // with `score(ref, dist)` — pin it across the full v1 corpus
    // q-grid.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // set_reference + score_with_reference against several
    // distorted candidates — that's the call pattern that motivates
    // having a cached fast path in the first place. Tick 261:
    // expanded from a hand-picked &[1, 20, 90] to the full
    // v1_corpus_qs() set so all 6 q-levels are covered.
    cvvdp
        .set_reference(&ref_bytes)
        .expect("set_reference should succeed on valid bytes");
    for &q in &common::v1_corpus_qs() {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod_direct = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        let jod_cached = cvvdp
            .score_with_reference(&dist_bytes)
            .expect("score_with_reference");
        assert!(
            (jod_direct - jod_cached).abs() < 1e-6,
            "q={q}: cached path {jod_cached} != direct {jod_direct}"
        );
    }
}

#[test]
fn set_reference_replaces_prior_cache() {
    // Tick 249: pin the documented-by-convention semantics that
    // `Cvvdp::set_reference` replaces any prior cached reference
    // (rather than e.g. accumulating or no-op'ing on second call).
    // The contract isn't spelled out in the docstring but is the
    // natural cache-replace shape callers expect.
    //
    // Test pattern: stash ref_a, stash ref_b, score against dist.
    // The cached-path JOD must equal score(ref_b, dist), not
    // score(ref_a, dist) — they differ because the two refs are
    // different.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_a = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    // ref_b is q=20-degraded ref_a — a perceptually different ref.
    let ref_b = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);
    // dist is q=70 of source — used to score against both refs.
    let dist = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(70), w, h);

    // Direct: score(ref_b, dist) — what the cached path should produce.
    let jod_direct_b = cvvdp.score(&ref_a, &dist).expect("warm-up score");
    let _ = jod_direct_b; // discard; the warm-up is to flush any first-call costs

    let jod_direct_against_a = cvvdp.score(&ref_a, &dist).expect("score(ref_a, dist)");
    let jod_direct_against_b = cvvdp.score(&ref_b, &dist).expect("score(ref_b, dist)");

    // The two refs should produce different JODs vs the same dist
    // (test premise — if they coincide the test can't distinguish
    // replace-vs-no-op semantics).
    assert!(
        (jod_direct_against_a - jod_direct_against_b).abs() > 1e-3,
        "test premise: score(ref_a) {jod_direct_against_a} and score(ref_b) {jod_direct_against_b} \
         differ too little to distinguish cache-replace semantics"
    );

    // Replace pattern: set ref_a, then set ref_b, then score.
    cvvdp.set_reference(&ref_a).expect("set_reference(ref_a)");
    cvvdp
        .set_reference(&ref_b)
        .expect("set_reference(ref_b) must replace ref_a");
    let jod_cached = cvvdp
        .score_with_reference(&dist)
        .expect("score_with_reference");

    // Must match the ref_b direct path, not ref_a.
    assert!(
        (jod_cached - jod_direct_against_b).abs() < 1e-6,
        "cached path JOD {jod_cached} should equal score(ref_b, dist) \
         {jod_direct_against_b}, not score(ref_a, dist) {jod_direct_against_a} — \
         set_reference must replace prior cache"
    );
}

#[test]
fn score_with_reference_errors_without_set_reference() {
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);
    let err = cvvdp
        .score_with_reference(&dist_bytes)
        .expect_err("must error without prior set_reference");
    // Don't lock the exact Debug repr; just ensure we got a
    // structured error rather than a 0.0 placeholder.
    let msg = format!("{err:?}");
    assert!(
        msg.contains("NoCachedReference"),
        "unexpected error kind: {msg}"
    );
}

#[test]
fn invalid_image_size_surfaces_on_too_small_dims() {
    // Tick 241: pin the InvalidImageSize error path on Cvvdp::new
    // and Cvvdp::new_with_geometry. The construction-time guard at
    // `if width < PYRAMID_MIN_DIM * 2 || height < PYRAMID_MIN_DIM * 2`
    // had zero test coverage before this — a refactor that swapped
    // the check for `width < PYRAMID_MIN_DIM` (off-by-2× threshold,
    // would accept 4×4 which has no usable pyramid) would not have
    // surfaced in CI.
    //
    // PYRAMID_MIN_DIM = 4, so the lower bound is 4×2 = 8. Cases:
    //   - 8×8 must succeed (boundary)
    //   - 7×8, 8×7, 7×7 must each fail
    //   - 4×4 must fail (smaller than threshold)
    //   - 0×0 must fail (degenerate)
    // One client shared across all subcases — the guard at
    // `width < PYRAMID_MIN_DIM * 2 || ...` runs before any GPU
    // alloc, so failing cases never touch the cubecl backend.
    // The boundary 8×8 case fully constructs (does touch the
    // backend); do it last so an early failure doesn't leak a
    // partial Cvvdp.
    let client = Backend::client(&Default::default());

    let check_invalid = |w: u32, h: u32, label: &str| {
        let err = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected InvalidImageSize, got Ok"));
        match err {
            cvvdp_gpu::Error::InvalidImageSize => {}
            other => panic!("{label}: expected InvalidImageSize, got {other:?}"),
        }
    };

    // Sub-threshold cases (guard rejects before GPU touch).
    check_invalid(7, 8, "Cvvdp::new(7, 8)");
    check_invalid(8, 7, "Cvvdp::new(8, 7)");
    check_invalid(7, 7, "Cvvdp::new(7, 7)");
    check_invalid(4, 4, "Cvvdp::new(4, 4)");
    check_invalid(0, 0, "Cvvdp::new(0, 0)");

    // new_with_geometry shares the same guard — pin one case to
    // catch a future copy-paste mistake that drops the check from
    // one constructor.
    let phone_geom = cvvdp_gpu::params::DisplayGeometry {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.40,
        diagonal_inches: 5.5,
    };
    let err = Cvvdp::<Backend>::new_with_geometry(
        client.clone(),
        4,
        4,
        CvvdpParams::PLACEHOLDER,
        phone_geom,
    )
    .err()
    .expect("new_with_geometry(4, 4) should fail");
    match err {
        cvvdp_gpu::Error::InvalidImageSize => {}
        other => panic!("new_with_geometry(4, 4): expected InvalidImageSize, got {other:?}"),
    }

    // Boundary: exact minimum dims must construct successfully.
    let cvvdp_ok = Cvvdp::<Backend>::new(client, 8, 8, CvvdpParams::PLACEHOLDER);
    if let Err(e) = &cvvdp_ok {
        panic!("8×8 should succeed (PYRAMID_MIN_DIM * 2 boundary), got Err({e:?})");
    }
}

#[test]
fn dimension_mismatch_surfaces_on_wrong_size_inputs() {
    // Tick 239: pin the DimensionMismatch error-path on every public
    // entry that validates buffer length. The 8 sites (lib::Error::
    // DimensionMismatch in pipeline.rs at score, set_reference,
    // score_with_reference, compute_dkl_planes, warm_reference,
    // compute_dkl_jod_with_warm_ref, and the GPU compute_dkl_*
    // helpers via `_dispatch_dkl_planes_gpu`) had zero direct test
    // coverage before this — a refactor that swapped the != check
    // for a < check (silently accepting smaller buffers and reading
    // garbage past srgb.len()) would not surface in CI.
    //
    // Test pattern: build a Cvvdp configured for 64×64, then call
    // each entry with a buffer sized for 32×32 (length n/4); expect
    // DimensionMismatch with the right expected/got values.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let expected_len = (w as usize) * (h as usize) * 3;
    let wrong_bytes = vec![128u8; expected_len / 4]; // 32×32 sized
    let right_bytes = vec![128u8; expected_len];

    // Closure to extract (expected, got) from a DimensionMismatch.
    let check_dim_err = |err: cvvdp_gpu::Error, label: &str| match err {
        cvvdp_gpu::Error::DimensionMismatch { expected, got } => {
            assert_eq!(expected, expected_len, "{label}: expected field mismatched",);
            assert_eq!(got, expected_len / 4, "{label}: got field mismatched",);
        }
        other => panic!("{label}: expected DimensionMismatch, got {other:?}"),
    };

    // score: both args validated; wrong reference triggers first.
    let err = cvvdp
        .score(&wrong_bytes, &right_bytes)
        .expect_err("score with short reference must error");
    check_dim_err(err, "score(short_ref, ok_dist)");

    let err = cvvdp
        .score(&right_bytes, &wrong_bytes)
        .expect_err("score with short distorted must error");
    check_dim_err(err, "score(ok_ref, short_dist)");

    // set_reference: validates the stored buffer.
    let err = cvvdp
        .set_reference(&wrong_bytes)
        .expect_err("set_reference with short buffer must error");
    check_dim_err(err, "set_reference(short)");

    // score_with_reference: validates the dist buffer. set_reference
    // a correct ref first so we don't hit NoCachedReference.
    cvvdp
        .set_reference(&right_bytes)
        .expect("set_reference(ok)");
    let err = cvvdp
        .score_with_reference(&wrong_bytes)
        .expect_err("score_with_reference with short dist must error");
    check_dim_err(err, "score_with_reference(short)");

    // warm_reference: validates the ref buffer.
    let err = cvvdp
        .warm_reference(&wrong_bytes)
        .expect_err("warm_reference with short buffer must error");
    check_dim_err(err, "warm_reference(short)");

    // compute_dkl_jod_with_warm_ref: validates dist buffer. Need a
    // valid warm state first to not collide with NoWarmReference.
    cvvdp
        .warm_reference(&right_bytes)
        .expect("warm_reference(ok)");
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let err = cvvdp
        .compute_dkl_jod_with_warm_ref(&wrong_bytes, ppd)
        .expect_err("compute_dkl_jod_with_warm_ref with short dist must error");
    check_dim_err(err, "compute_dkl_jod_with_warm_ref(short)");
}

#[test]
fn compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm() {
    // Tick 248: pin the dim-check-before-NoWarmReference ordering.
    // When a caller has BOTH a wrong-size dist buffer AND no warm
    // state set, the wrong-size buffer is the more actionable error.
    // Pre-tick-248 the function returned NoWarmReference first,
    // masking the dim mismatch until the caller re-armed.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // No warm_reference call. Pass a buffer sized for 32×32 against
    // a Cvvdp configured for 64×64. Expect DimensionMismatch, not
    // NoWarmReference.
    let expected_len = (w as usize) * (h as usize) * 3;
    let wrong_bytes = vec![128u8; expected_len / 4];
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let err = cvvdp
        .compute_dkl_jod_with_warm_ref(&wrong_bytes, ppd)
        .expect_err("must error on wrong-size dist regardless of warm state");
    match err {
        cvvdp_gpu::Error::DimensionMismatch { expected, got } => {
            assert_eq!(expected, expected_len);
            assert_eq!(got, expected_len / 4);
        }
        cvvdp_gpu::Error::NoWarmReference => {
            panic!(
                "tick-248 regression: ordering changed back — \
                 NoWarmReference reported before DimensionMismatch on a \
                 wrong-size + no-warm call"
            );
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "ppd=")]
fn debug_assert_fires_when_ppd_mismatches_geometry() {
    // Tick 244: pin the tick-243 debug_assert. Builds Cvvdp with
    // the default STANDARD_4K geometry (75.4 PPD), then calls
    // compute_dkl_jod with a phone PPD (110.087282 derived from
    // a 5.5″ 1080p phone at 0.40m). Expects the debug-only ppd-
    // mismatch assertion to fire.
    //
    // #[cfg(debug_assertions)] guards the test definition: release
    // builds skip it (the assertion compiles out, so the call
    // wouldn't panic and #[should_panic] would itself fail). The
    // assert is informational only — a refactor that drops the
    // ppd safety net would silently regress without this test.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_bytes = vec![128u8; n];
    let dist_bytes = vec![128u8; n];

    // Phone PPD for a 5.5″ 1080p display at 0.40m (110.087 ≠ 75.4).
    let phone_geom = cvvdp_gpu::params::DisplayGeometry {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.40,
        diagonal_inches: 5.5,
    };
    let phone_ppd = phone_geom.pixels_per_degree();
    let standard_4k_ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    assert!(
        (phone_ppd - standard_4k_ppd).abs() > 1.0,
        "test premise broken: phone_ppd {phone_ppd} too close to STANDARD_4K {standard_4k_ppd}"
    );

    // Should panic at the debug_assert in compute_dkl_jod via
    // debug_assert_ppd_matches_geometry.
    let _ = cvvdp.compute_dkl_jod(&ref_bytes, &dist_bytes, phone_ppd);
}

#[test]
fn compute_dkl_jod_on_v1_manifest_corpus() {
    // GPU-composed compute_dkl_jod against the v1 R2 manifest values.
    // shadow_jod pins the all-host path to ≤0.006 JOD; this test
    // measures the GPU path's drift on real corpus images vs pycvvdp.
    //
    // Observed 2026-05-15 (cuda backend, post tick-181 band-count
    // alignment + tick-175 ceil-div pyramid):
    //
    // ```text
    //   q    pycvvdp manifest   GPU JOD    |drift|
    //   1    7.6536             7.6471     0.0065
    //   5    8.8889             8.8909     0.0020
    //   20   9.7076             9.7088     0.0012
    //   45   9.8273             9.8295     0.0022
    //   70   9.8915             9.8945     0.0030
    //   90   9.9930             9.9929     0.0001
    // ```
    //
    // Max drift 0.0065 at q=1 — comfortably inside f32 accumulation
    // noise across the full q range. The old q=1 drift of 0.3992
    // came from the pre-tick-175 floor-div pyramid bug; q=5 was
    // 0.0545. Both collapsed to <0.01 once the pyramid was fixed.
    //
    // Per-q diffs report to stdout so future ticks can watch the
    // drift profile if upstream changes shift it.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // Loaded from scripts/cvvdp_goldens/v1_corpus_jods.json.
    let qs = common::v1_corpus_qs();
    let cases: Vec<(u32, f32)> = qs
        .iter()
        .map(|&q| (q, common::v1_corpus_jod_golden(q)))
        .collect();

    let mut max_drift = 0.0_f32;
    for &(q, expected) in &cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let diff = (jod - expected).abs();
        if diff > max_drift {
            max_drift = diff;
        }
        eprintln!(
            "compute_dkl_jod q={q:>2}: GPU JOD = {jod:.4} (pycvvdp manifest {expected:.4}, |drift| {diff:.4})"
        );
        assert!(jod.is_finite(), "q={q}: JOD = {jod} (not finite)");
        assert!(
            (0.0..=10.0).contains(&jod),
            "q={q}: JOD = {jod} out of [0, 10]"
        );
    }
    eprintln!("compute_dkl_jod max drift vs v1 manifest: {max_drift:.4}");
    // Tightened to the canonical 0.005 JOD manifest tolerance in
    // tick 224. Was 0.02 (set in tick 185); post-tick-204/206/207
    // drift closures the observed max is 0.0031 at q=70 (was
    // 0.0065 at tick 185, before chroma_shift fix). 0.005 matches
    // the tolerance every other manifest-parity test in the suite
    // uses (shadow_jod_gpu, cvvdp_score_matches_v1_manifest).
    assert!(
        max_drift < 0.005,
        "GPU JOD drifts > 0.005 from v1 manifest: {max_drift} (observed 0.0031 at q=70 post-tick-207)"
    );
}

#[test]
fn compute_dkl_jod_vs_host_scalar_on_corpus() {
    // Direct GPU-vs-HOST comparison on the v1 manifest corpus (real
    // 256×256 images). shadow_jod pins both paths against pycvvdp at
    // 0.005 JOD (tick 207); compute_dkl_jod_on_v1_manifest_corpus
    // measures GPU-vs-pycvvdp directly. This test answers the
    // remaining question: does GPU-vs-HOST agree better than either
    // of those, or does the drift compound? (Measured: GPU vs host
    // ≤ 0.003 JOD — f32 noise from the atomic pool's accumulation
    // order.)
    use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
    use cvvdp_gpu::params::DisplayModel;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    let qs = common::v1_corpus_qs();
    eprintln!("  q   pycvvdp    host_scalar   GPU JOD   GPU-host   GPU-pycvvdp");
    let mut max_gpu_host_drift = 0.0_f32;
    let pycvvdp_manifest: Vec<(u32, f32)> = qs
        .iter()
        .map(|&q| (q, common::v1_corpus_jod_golden(q)))
        .collect();
    for &q in &qs {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let gpu_jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let host_jod = predict_jod_still_3ch(
            &ref_bytes,
            &dist_bytes,
            w as usize,
            h as usize,
            display,
            ppd,
        );
        let manifest = pycvvdp_manifest.iter().find(|&&(qq, _)| qq == q).unwrap().1;
        let gpu_host = (gpu_jod - host_jod).abs();
        let gpu_pyc = (gpu_jod - manifest).abs();
        eprintln!(
            "  {q:>2}  {manifest:>8.4}   {host_jod:>9.4}   {gpu_jod:>7.4}    {gpu_host:>7.4}      {gpu_pyc:>7.4}"
        );
        if gpu_host > max_gpu_host_drift {
            max_gpu_host_drift = gpu_host;
        }
        assert!(gpu_jod.is_finite(), "q={q}: GPU JOD not finite");
        assert!(
            (0.0..=10.0).contains(&gpu_jod),
            "q={q}: GPU JOD = {gpu_jod} out of range"
        );
    }
    eprintln!("compute_dkl_jod max drift vs host scalar: {max_gpu_host_drift:.4}");
    // Tightened in tick 185. Post tick-181's band-count alignment,
    // observed max drift = 0.0006 JOD across q1..q90. The earlier
    // 1.0 JOD tolerance dated to when the GPU pipeline was partial
    // (host fold + masking). 0.005 gives ~8× margin while still
    // gating real regressions (pre-tick-175 ceil-div bug was 0.5+
    // JOD at deeper pyramids).
    assert!(
        max_gpu_host_drift < 0.005,
        "GPU JOD drifts > 0.005 from host scalar: {max_gpu_host_drift} (was 0.0006 at tick 185)"
    );
}

#[test]
fn compute_dkl_weber_pyramid_matches_host_on_corpus_256x256() {
    // The 32×32 synthetic parity test (in pipeline_color.rs) shows
    // Weber bands match host within 5e-4 max-abs. On 256×256 corpus
    // images with deeper pyramids the f32 accumulation across 5+
    // reduce/expand levels may compound. This test surfaces per-
    // band max-abs error at scale, narrowing where the q=1 JOD
    // drift comes from.
    use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
    use cvvdp_gpu::kernels::pyramid::weber_contrast_pyr_dec_scalar;
    use cvvdp_gpu::params::DisplayModel;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    // Test on the most-distorted (q=1) dist image — heaviest
    // signal stress on the pyramid. Pre-tick-175 q=1 had a 0.4 JOD
    // GPU-vs-host drift (closed by the ceil-div fix); the test
    // remains useful as a stress-point even though the modern
    // pipeline tracks host within f32 noise across all q-levels.
    let srgb = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(1), w, h);
    let (gpu_bands, gpu_log_l_bkg) = cvvdp
        .compute_dkl_weber_pyramid(&srgb)
        .expect("compute_dkl_weber_pyramid");

    // Host reference: replay color → 3 weber pyramids (one per channel,
    // achromatic as L_bkg).
    let display = DisplayModel::STANDARD_4K;
    let n = (w * h) as usize;
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (i, chunk) in srgb.chunks_exact(3).enumerate() {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            chunk[0],
            chunk[1],
            chunk[2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }
    let n_levels = gpu_bands.len();
    let host_per_ch = [
        weber_contrast_pyr_dec_scalar(&planes[0], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[1], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[2], &planes[0], w as usize, h as usize, n_levels),
    ];

    eprintln!("level   shape    | A max-abs  RG max-abs  VY max-abs  log_l_bkg max-abs");
    let mut overall_max_band = 0.0_f32;
    let mut overall_max_log = 0.0_f32;
    for k in 0..n_levels {
        let bw = host_per_ch[0].bands[k].w;
        let bh = host_per_ch[0].bands[k].h;
        let mut max_bands = [0.0_f32; 3];
        for (c, max_b) in max_bands.iter_mut().enumerate() {
            let host = &host_per_ch[c].bands[k].data;
            let gpu = &gpu_bands[k][c];
            let max_err = gpu
                .iter()
                .zip(host)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            *max_b = max_err;
        }
        let host_log = &host_per_ch[0].log_l_bkg[k];
        let gpu_log = &gpu_log_l_bkg[k];
        let max_log = gpu_log
            .iter()
            .zip(host_log)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "  {k}   {bw:>3}x{bh:<3}  | {:<10.4e} {:<11.4e} {:<11.4e} {:<10.4e}",
            max_bands[0], max_bands[1], max_bands[2], max_log
        );
        let band_max = max_bands.iter().fold(0.0_f32, |a, &b| a.max(b));
        if band_max > overall_max_band {
            overall_max_band = band_max;
        }
        if max_log > overall_max_log {
            overall_max_log = max_log;
        }
    }
    eprintln!(
        "max-abs over all bands: weber = {overall_max_band:.4e}, log_l_bkg = {overall_max_log:.4e}"
    );

    // Tolerances calibrated from the observed values; tightens if the
    // upstream stages get bit-stable, surfaces a regression if either
    // stage starts to drift further.
    assert!(
        overall_max_band < 1e-2,
        "Weber band max-abs vs host on corpus 256×256 = {overall_max_band:.4e}"
    );
    assert!(
        overall_max_log < 1e-2,
        "log_l_bkg max-abs vs host on corpus 256×256 = {overall_max_log:.4e}"
    );
}

#[test]
fn compute_dkl_t_p_bands_matches_host_on_corpus_256x256() {
    // Original purpose (pre-tick-175): characterize the per-pixel
    // CSF apply + CH_GAIN + band_mul step on the q=1 corpus image,
    // to narrow where the 0.4 JOD GPU-vs-host drift lived. Ticks
    // 175/204/206 closed that drift; the test now pins T_p
    // bit-stability vs host at scale (256×256 q=1 corpus) so a
    // future regression in the CSF apply chain surfaces here.
    use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
    use cvvdp_gpu::kernels::masking::CH_GAIN;
    use cvvdp_gpu::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};
    use cvvdp_gpu::params::DisplayModel;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let srgb = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(1), w, h);

    // GPU T_p (per-side Weber → CSF → CH_GAIN × band_mul).
    let t_p_gpu = cvvdp
        .compute_dkl_t_p_bands(&srgb, ppd)
        .expect("compute_dkl_t_p_bands");

    // Host T_p: same formula reproduced from host_scalar at scale.
    let display = DisplayModel::STANDARD_4K;
    let n = (w * h) as usize;
    let mut planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (i, chunk) in srgb.chunks_exact(3).enumerate() {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            chunk[0],
            chunk[1],
            chunk[2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        planes[0][i] = a;
        planes[1][i] = rg;
        planes[2][i] = vy;
    }
    let n_levels = t_p_gpu.len();
    let host_per_ch = [
        weber_contrast_pyr_dec_scalar(&planes[0], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[1], &planes[0], w as usize, h as usize, n_levels),
        weber_contrast_pyr_dec_scalar(&planes[2], &planes[0], w as usize, h as usize, n_levels),
    ];
    let freqs = band_frequencies(ppd, w as usize, h as usize);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    eprintln!(
        "level   shape    | A max-abs   RG max-abs   VY max-abs   A band-rel  RG band-rel  VY band-rel"
    );
    let mut overall_max_band_rel = 0.0_f32;
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = host_per_ch[0].bands[k].w;
        let bh = host_per_ch[0].bands[k].h;
        let n_band = bw * bh;
        let log_l_bkg_band = &host_per_ch[0].log_l_bkg[k];
        // Tick 204: pycvvdp overrides baseband CSF rho to 0.1 cy/deg
        // (cvvdp_metric.py:628); host reference applies the same.
        let rho_eff = if is_baseband {
            cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };

        let mut max_abs = [0.0_f32; 3];
        let mut max_band_t_p = [0.0_f32; 3];
        for c in 0..3 {
            let weber_c = &host_per_ch[c].bands[k].data;
            let ch_gain_eff = if is_baseband {
                1.0
            } else {
                band_mul * CH_GAIN[c]
            };
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(rho_eff, log_l_bkg_band[i], channels[c]);
                let host_t_p = weber_c[i] * s * ch_gain_eff;
                let abs = (t_p_gpu[k][c][i] - host_t_p).abs();
                if abs > max_abs[c] {
                    max_abs[c] = abs;
                }
                let mag = host_t_p.abs();
                if mag > max_band_t_p[c] {
                    max_band_t_p[c] = mag;
                }
            }
        }
        // Band-normalized rel: max-abs over the band divided by the
        // band's max |T_p|. More meaningful than per-pixel rel which
        // blows up near zero-crossings.
        let band_rel = [
            max_abs[0] / max_band_t_p[0].max(1e-6),
            max_abs[1] / max_band_t_p[1].max(1e-6),
            max_abs[2] / max_band_t_p[2].max(1e-6),
        ];
        eprintln!(
            "  {k}   {bw:>3}x{bh:<3}  | {:<11.4e} {:<12.4e} {:<12.4e} {:<11.4e} {:<12.4e} {:<11.4e}",
            max_abs[0], max_abs[1], max_abs[2], band_rel[0], band_rel[1], band_rel[2]
        );
        let local_max = band_rel.iter().fold(0.0_f32, |a, &b| a.max(b));
        if local_max > overall_max_band_rel {
            overall_max_band_rel = local_max;
        }
    }
    eprintln!("max band-normalized rel over all bands: {overall_max_band_rel:.4e}");

    // Tightened in tick 186. Post tick 175 (ceil-div) + tick 181
    // (band-count), observed max band-normalized rel = 7.6e-4. 5e-3
    // gives ~6× margin while catching a real regression (pre-fix
    // we observed 8e-4 in this test's original comment, but the
    // 1e-1 tolerance allowed a much larger silent drift to slip
    // by during the pre-tick-175 ceil-div bug).
    assert!(
        overall_max_band_rel < 5e-3,
        "T_p max band-normalized rel vs host on corpus 256×256 = {overall_max_band_rel:.4e} (was 7.6e-4 at tick 186)"
    );
}

#[test]
fn compute_dkl_d_bands_matches_host_on_corpus_256x256() {
    // Original purpose (pre-tick-175): isolate where the 0.4 JOD
    // GPU-vs-host drift lived by walking the pipeline stage by
    // stage. The Weber + log_l_bkg + T_p companion tests pin
    // bit-stability earlier in the chain; this test pins D bands
    // (the masking + soft-clamp output). Ticks 175/204/206 closed
    // that drift; the test now pins D-band bit-stability at scale
    // so any future masking-chain regression surfaces here.
    use cvvdp_gpu::kernels::color::srgb_byte_to_dkl_scalar;
    use cvvdp_gpu::kernels::csf::{CsfChannel, sensitivity_corrected_scalar};
    use cvvdp_gpu::kernels::masking::{CH_GAIN, mult_mutual_band};
    use cvvdp_gpu::kernels::pyramid::{band_frequencies, weber_contrast_pyr_dec_scalar};
    use cvvdp_gpu::params::DisplayModel;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(1), w, h);

    let gpu_d = cvvdp
        .compute_dkl_d_bands(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_d_bands");

    // Host reference: full replay of host_scalar's per-band masking.
    let display = DisplayModel::STANDARD_4K;
    let n = (w * h) as usize;
    let mut ref_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    let mut dis_planes: [Vec<f32>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for i in 0..n {
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            ref_bytes[i * 3],
            ref_bytes[i * 3 + 1],
            ref_bytes[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        ref_planes[0][i] = a;
        ref_planes[1][i] = rg;
        ref_planes[2][i] = vy;
        let (a, rg, vy) = srgb_byte_to_dkl_scalar(
            dist_bytes[i * 3],
            dist_bytes[i * 3 + 1],
            dist_bytes[i * 3 + 2],
            display.y_peak,
            display.y_black,
            display.y_refl,
        );
        dis_planes[0][i] = a;
        dis_planes[1][i] = rg;
        dis_planes[2][i] = vy;
    }
    let n_levels = gpu_d.len();
    let ref_weber = [
        weber_contrast_pyr_dec_scalar(
            &ref_planes[0],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[1],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &ref_planes[2],
            &ref_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
    ];
    let dis_weber = [
        weber_contrast_pyr_dec_scalar(
            &dis_planes[0],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[1],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
        weber_contrast_pyr_dec_scalar(
            &dis_planes[2],
            &dis_planes[0],
            w as usize,
            h as usize,
            n_levels,
        ),
    ];
    let freqs = band_frequencies(ppd, w as usize, h as usize);
    let channels = [CsfChannel::A, CsfChannel::Rg, CsfChannel::Vy];

    eprintln!(
        "level   shape    | A max-abs    RG max-abs    VY max-abs    A band-rel   RG band-rel   VY band-rel"
    );
    let mut overall_max_band_rel = 0.0_f32;
    for k in 0..n_levels {
        let is_first = k == 0;
        let is_baseband = k == n_levels - 1;
        let band_mul: f32 = if is_first || is_baseband { 1.0 } else { 2.0 };
        let bw = ref_weber[0].bands[k].w;
        let bh = ref_weber[0].bands[k].h;
        let n_band = bw * bh;
        let log_l_bkg_band = &ref_weber[0].log_l_bkg[k];
        // Tick 204: baseband CSF rho override (see compute_dkl_t_p_*
        // sibling above).
        let rho_eff = if is_baseband {
            cvvdp_gpu::kernels::csf::CSF_BASEBAND_RHO
        } else {
            freqs[k]
        };

        let mut t_p_dis: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        let mut t_p_ref: [Vec<f32>; 3] = [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
        for c in 0..3 {
            for i in 0..n_band {
                let s = sensitivity_corrected_scalar(rho_eff, log_l_bkg_band[i], channels[c]);
                let ch_gain_eff = if is_baseband {
                    1.0
                } else {
                    band_mul * CH_GAIN[c]
                };
                t_p_dis[c][i] = dis_weber[c].bands[k].data[i] * s * ch_gain_eff;
                t_p_ref[c][i] = ref_weber[c].bands[k].data[i] * s * ch_gain_eff;
            }
        }
        let host_d = if is_baseband {
            let mut planes: [Vec<f32>; 3] =
                [vec![0.0; n_band], vec![0.0; n_band], vec![0.0; n_band]];
            for c in 0..3 {
                for i in 0..n_band {
                    planes[c][i] = (t_p_dis[c][i] - t_p_ref[c][i]).abs();
                }
            }
            planes
        } else {
            mult_mutual_band(&t_p_dis, &t_p_ref, bw, bh)
        };

        let mut max_abs = [0.0_f32; 3];
        let mut max_band_d = [0.0_f32; 3];
        for c in 0..3 {
            for i in 0..n_band {
                let abs = (gpu_d[k][c][i] - host_d[c][i]).abs();
                if abs > max_abs[c] {
                    max_abs[c] = abs;
                }
                let mag = host_d[c][i].abs();
                if mag > max_band_d[c] {
                    max_band_d[c] = mag;
                }
            }
        }
        let band_rel = [
            max_abs[0] / max_band_d[0].max(1e-6),
            max_abs[1] / max_band_d[1].max(1e-6),
            max_abs[2] / max_band_d[2].max(1e-6),
        ];
        eprintln!(
            "  {k}   {bw:>3}x{bh:<3}  | {:<12.4e} {:<13.4e} {:<13.4e} {:<12.4e} {:<13.4e} {:<12.4e}",
            max_abs[0], max_abs[1], max_abs[2], band_rel[0], band_rel[1], band_rel[2]
        );
        let local_max = band_rel.iter().fold(0.0_f32, |a, &b| a.max(b));
        if local_max > overall_max_band_rel {
            overall_max_band_rel = local_max;
        }
    }
    eprintln!("max band-normalized rel over all bands: {overall_max_band_rel:.4e}");

    // Tightened in tick 186. Post tick 175 + tick 181, observed max
    // band-normalized rel = 1.3e-3. 5e-3 gives ~4× margin while
    // surfacing a regression — pre-fix ceil-div drift would have
    // pushed this well above 5e-3 on the deeper-pyramid bands.
    assert!(
        overall_max_band_rel < 5e-3,
        "D max band-normalized rel vs host on corpus 256×256 = {overall_max_band_rel:.4e} (was 1.3e-3 at tick 186)"
    );
}
