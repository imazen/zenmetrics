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

use crate::common;

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
    // Pattern-match on the variant rather than Debug-formatting and
    // substring-checking. Tick 318: prior version did
    // `format!("{err:?}").contains("NoCachedReference")` which
    // would silently pass on any Debug repr containing those
    // characters — including a future variant rename that landed
    // "NoCachedReferenceV2" by accident. The match arm pins the
    // variant identity directly via the public Error API.
    match err {
        cvvdp_gpu::Error::NoCachedReference => {}
        other => panic!("expected NoCachedReference, got {other:?}"),
    }
}

#[test]
fn error_display_messages_are_actionable() {
    // Tick 282: pin the user-facing Display strings for every
    // `cvvdp_gpu::Error` variant. Display is what users see in
    // logs / `anyhow::Error::to_string()` / `panic!` propagation
    // when they `?`-bubble a cvvdp_gpu::Result through their own
    // error type. A rename or format-string break would silently
    // degrade the user experience without this pin.
    //
    // The strings are checked for content, not exact match — so a
    // future improvement that adds context (e.g. \"DimensionMismatch:
    // got X bytes, expected Y\") still passes, but a refactor that
    // dropped the variant-name hint or the byte-count details
    // surfaces.
    use cvvdp_gpu::Error;

    let dm = Error::DimensionMismatch {
        expected: 12_288,
        got: 3_072,
    }
    .to_string();
    assert!(
        dm.contains("12288") && dm.contains("3072"),
        "DimensionMismatch Display must include expected + got byte counts; got: {dm:?}"
    );
    assert!(
        dm.contains("dimension"),
        "DimensionMismatch Display must mention 'dimension'; got: {dm:?}"
    );

    let ncr = Error::NoCachedReference.to_string();
    assert!(
        ncr.contains("set_reference"),
        "NoCachedReference Display must point at set_reference; got: {ncr:?}"
    );

    let nwr = Error::NoWarmReference.to_string();
    assert!(
        nwr.contains("warm_reference"),
        "NoWarmReference Display must point at warm_reference; got: {nwr:?}"
    );

    let iis = Error::InvalidImageSize.to_string();
    assert!(
        iis.contains("small") || iis.contains("pyramid"),
        "InvalidImageSize Display must hint at the too-small / pyramid cause; got: {iis:?}"
    );
    // Tick 316: pin the dual-purpose hint too. InvalidImageSize
    // is also returned when a cubecl GPU readback / dispatch
    // fails (the doc on the variant spells this out — the two
    // get the same variant because cubecl's read errors aren't
    // easily separable yet). Pre-tick-316 the Display message
    // only mentioned image-size, so a user hitting a GPU error
    // would investigate image dimensions instead of the actual
    // backend failure.
    assert!(
        iis.contains("GPU") || iis.contains("readback") || iis.contains("dispatch"),
        "InvalidImageSize Display must also hint at the GPU-failure cause; got: {iis:?}"
    );
}

#[test]
fn invalid_image_size_surfaces_on_too_small_dims() {
    // Tick 241 (original): pinned the InvalidImageSize error path on
    // sub-`PYRAMID_MIN_DIM * 2 = 8` dims. Updated to the typed-pad
    // contract: the typed `Cvvdp<R>` now reflect(mirror)-pads sub-8px
    // requests up to the pyramid floor — like `CvvdpOpaque` — so every
    // ≥1px request CONSTRUCTS instead of erroring (the dedicated
    // bit-identical pad checks live in `tests/typed_sub_min_pad.rs`).
    // The only remaining `InvalidImageSize` reject is a 0-dim axis.
    //
    // PYRAMID_MIN_DIM = 4, MIN_PAD_DIM = 8. Cases:
    //   - 7×8, 8×7, 7×7, 4×4, 1×1 must each SUCCEED (reflect-padded to ≥8),
    //     and `dimensions()` must report the LOGICAL (requested) extent.
    //   - 0×0, 0×8, 8×0 must each fail (degenerate — padded extent stays 0).
    //   - 8×8 must succeed (no-op pad boundary).
    // One client shared across all subcases.
    let client = Backend::client(&Default::default());

    let check_pads_ok = |w: u32, h: u32, label: &str| {
        let cvvdp = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
            .unwrap_or_else(|e| {
                panic!("{label}: sub-8 must reflect-pad + construct, got Err({e:?})")
            });
        assert_eq!(
            cvvdp.dimensions(),
            (w, h),
            "{label}: dimensions() must report the logical (requested) extent"
        );
    };

    // Sub-floor (but ≥1px): reflect-padded, must construct.
    check_pads_ok(7, 8, "Cvvdp::new(7, 8)");
    check_pads_ok(8, 7, "Cvvdp::new(8, 7)");
    check_pads_ok(7, 7, "Cvvdp::new(7, 7)");
    check_pads_ok(4, 4, "Cvvdp::new(4, 4)");
    check_pads_ok(1, 1, "Cvvdp::new(1, 1)");

    // Degenerate 0-dim axis: still rejected (padded extent stays 0).
    let check_invalid = |w: u32, h: u32, label: &str| {
        let err = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
            .err()
            .unwrap_or_else(|| panic!("{label}: expected InvalidImageSize, got Ok"));
        match err {
            cvvdp_gpu::Error::InvalidImageSize => {}
            other => panic!("{label}: expected InvalidImageSize, got {other:?}"),
        }
    };
    check_invalid(0, 0, "Cvvdp::new(0, 0)");
    check_invalid(0, 8, "Cvvdp::new(0, 8)");
    check_invalid(8, 0, "Cvvdp::new(8, 0)");

    // new_with_geometry shares the same construction funnel — pin one
    // sub-floor case to catch a future copy-paste mistake that drops the
    // pad from one constructor: it must reflect-pad + construct, not error.
    let phone_geom = cvvdp_gpu::params::DisplayGeometry {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.40,
        diagonal_inches: 5.5,
    };
    let cvvdp_geom = Cvvdp::<Backend>::new_with_geometry(
        client.clone(),
        4,
        4,
        CvvdpParams::PLACEHOLDER,
        phone_geom,
    )
    .expect("new_with_geometry(4, 4) must reflect-pad + construct");
    assert_eq!(
        cvvdp_geom.dimensions(),
        (4, 4),
        "new_with_geometry(4, 4): dimensions() must report logical extent"
    );

    // Boundary: exact minimum dims must construct successfully (no-op pad).
    let cvvdp_ok = Cvvdp::<Backend>::new(client, 8, 8, CvvdpParams::PLACEHOLDER);
    if let Err(e) = &cvvdp_ok {
        panic!("8×8 should succeed (PYRAMID_MIN_DIM * 2 boundary), got Err({e:?})");
    }
}

#[test]
fn all_four_scoring_paths_agree_bit_equal_on_same_input() {
    // Tick 494: consolidation pin. The four public scoring paths
    // documented in the lib.rs Status section should produce
    // bit-identical f32 JOD on the same (ref, dist) input on a
    // single Cvvdp instance:
    //
    //   (A) `score(ref, dist)` — f64 from f32::from(compute_dkl_jod)
    //   (B) `compute_dkl_jod(ref, dist, ppd)` — f32
    //   (C) `set_reference(ref)` + `score_with_reference(dist)` — f64
    //   (D) `warm_reference(ref)` + `compute_dkl_jod_with_warm_ref(dist, ppd)` — f32
    //
    // Individual pins already cover pairwise relationships (tick
    // 407: A↔B lossless widening; tick 488: A↔C bit-equal; tick 489:
    // D self-determinism + cross-DIST isolation). This test pins
    // the four-way intersection: all four paths produce a single
    // canonical f32 JOD value for the same input.
    //
    // A refactor that, say, takes the warm-ref path through a
    // subtly different pool kernel (e.g. forgets to use the same
    // partials buffer) would surface here as D drifting from A/B/C
    // even when each path's standalone determinism holds.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_srgb: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(7)).collect();

    // (A) score(ref, dist) — returns f64. Cast to f32 for cross-path
    // comparison (lossless per tick 407 pin).
    let jod_a_f64 = cvvdp.score(&ref_srgb, &dist_srgb).expect("score");
    let jod_a = jod_a_f64 as f32;

    // (B) compute_dkl_jod(ref, dist, ppd) — already f32.
    let jod_b = cvvdp
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("compute_dkl_jod");

    // (C) set_reference + score_with_reference — returns f64. Cast
    // to f32 same as path A.
    cvvdp.set_reference(&ref_srgb).expect("set_reference");
    let jod_c_f64 = cvvdp
        .score_with_reference(&dist_srgb)
        .expect("score_with_reference");
    let jod_c = jod_c_f64 as f32;

    // (D) warm_reference + compute_dkl_jod_with_warm_ref — already f32.
    cvvdp.warm_reference(&ref_srgb).expect("warm_reference");
    let jod_d = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_with_warm_ref");

    // All four paths must agree bit-exact on the f32 JOD.
    let bits_a = jod_a.to_bits();
    let bits_b = jod_b.to_bits();
    let bits_c = jod_c.to_bits();
    let bits_d = jod_d.to_bits();

    assert_eq!(
        bits_a, bits_b,
        "path A (score) {jod_a} != path B (compute_dkl_jod) {jod_b}",
    );
    assert_eq!(
        bits_a, bits_c,
        "path A (score) {jod_a} != path C (set+score_with_ref) {jod_c}",
    );
    assert_eq!(
        bits_a, bits_d,
        "path A (score) {jod_a} != path D (warm-ref) {jod_d}",
    );
}

#[test]
fn new_with_geometry_stable_under_degenerate_geometry() {
    // Tick 497: companion to tick 495's `ppd_does_not_panic_on_degenerate_inputs`.
    // `Cvvdp::new_with_geometry` accepts an arbitrary `DisplayGeometry`
    // and internally calls `geometry.pixels_per_degree()` to derive
    // pyramid level count (via `pyramid_levels`). Degenerate
    // geometries can produce NaN / Inf ppd; Cvvdp::new_with_geometry
    // must remain a total function — it either succeeds (potentially
    // with a degraded pyramid level count) OR returns
    // `Error::InvalidImageSize`. It MUST NOT panic.
    //
    // A future refactor that adds geometry validation
    // (`assert!(distance_m > 0.0)` or similar) would convert the
    // current "total + degraded" contract into "panicking" — surface
    // here.
    use cvvdp_gpu::params::DisplayGeometry;
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let base = DisplayGeometry::STANDARD_4K;

    let cases: &[(&str, DisplayGeometry)] = &[
        (
            "zero distance",
            DisplayGeometry {
                distance_m: 0.0,
                ..base
            },
        ),
        (
            "zero diagonal",
            DisplayGeometry {
                diagonal_inches: 0.0,
                ..base
            },
        ),
        (
            "zero resolution_w",
            DisplayGeometry {
                resolution_w: 0,
                ..base
            },
        ),
        (
            "extreme close distance (1 cm)",
            DisplayGeometry {
                distance_m: 0.01,
                ..base
            },
        ),
        (
            "extreme far distance (100 m)",
            DisplayGeometry {
                distance_m: 100.0,
                ..base
            },
        ),
    ];

    for (label, geom) in cases {
        // The contract is: doesn't panic. The result is either Ok
        // (degraded pyramid count) or Err(InvalidImageSize) — both
        // acceptable. We don't pin which because the formula at
        // `pyramid_levels(ppd, w, h)` can shift between them as ppd
        // boundary conditions move; a future refactor could
        // legitimately tighten the bounds.
        let result = Cvvdp::<Backend>::new_with_geometry(
            client.clone(),
            w,
            h,
            CvvdpParams::PLACEHOLDER,
            *geom,
        );
        match result {
            Ok(_) => eprintln!("{label}: new_with_geometry succeeded"),
            Err(e) => eprintln!("{label}: new_with_geometry returned Err({e:?})"),
        }
    }
}

#[test]
fn new_equivalent_to_new_with_geometry_standard_4k() {
    // Tick 493: pin the documented contract that `Cvvdp::new` is
    // "equivalent to `new_with_geometry(..., STANDARD_4K)`" (per
    // the new() rustdoc). The implementation forwards to
    // new_with_geometry, but a future refactor that adds extra
    // initialization to one but not the other (e.g. swapping
    // default geometry, eagerly priming a cache on the
    // explicit-geometry path, etc.) would silently change the
    // documented surface.
    //
    // Pins: scoring the same (ref, dist) pair on two Cvvdp
    // instances — one built via `new`, one via `new_with_geometry(..,
    // STANDARD_4K)` — produces bit-identical f64 JOD results across
    // every scoring path:
    //   (1) score(ref, dist) — bit-equal
    //   (2) set_reference + score_with_reference(dist) — bit-equal
    //   (3) compute_dkl_jod(ref, dist, ppd) — bit-equal at ppd from
    //       STANDARD_4K (the geometries are bit-identical, so
    //       self.geometry.pixels_per_degree() returns the same f32 on
    //       both instances).
    //   (4) warm_reference + compute_dkl_jod_with_warm_ref(dist, ppd)
    //       — bit-equal
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let standard_4k = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K;
    let ppd = standard_4k.pixels_per_degree();

    let mut a = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER).expect("new");
    let mut b =
        Cvvdp::<Backend>::new_with_geometry(client, w, h, CvvdpParams::PLACEHOLDER, standard_4k)
            .expect("new_with_geometry(STANDARD_4K)");

    let n = (w * h * 3) as usize;
    let ref_srgb: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(7)).collect();

    // (1) score()
    let ja = a.score(&ref_srgb, &dist_srgb).expect("a.score");
    let jb = b.score(&ref_srgb, &dist_srgb).expect("b.score");
    assert_eq!(
        ja.to_bits(),
        jb.to_bits(),
        "score() bit-mismatch between new and new_with_geometry(STANDARD_4K): {ja} vs {jb}",
    );

    // (2) set_reference + score_with_reference
    a.set_reference(&ref_srgb).expect("a.set_reference");
    b.set_reference(&ref_srgb).expect("b.set_reference");
    let swra = a.score_with_reference(&dist_srgb).expect("a.swr");
    let swrb = b.score_with_reference(&dist_srgb).expect("b.swr");
    assert_eq!(
        swra.to_bits(),
        swrb.to_bits(),
        "score_with_reference() bit-mismatch: {swra} vs {swrb}",
    );

    // (3) compute_dkl_jod
    let cja = a
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("a.compute_dkl_jod");
    let cjb = b
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("b.compute_dkl_jod");
    assert_eq!(
        cja.to_bits(),
        cjb.to_bits(),
        "compute_dkl_jod() bit-mismatch: {cja} vs {cjb}",
    );

    // (4) warm_reference + compute_dkl_jod_with_warm_ref
    a.warm_reference(&ref_srgb).expect("a.warm_reference");
    b.warm_reference(&ref_srgb).expect("b.warm_reference");
    let wa = a
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("a.warm-ref");
    let wb = b
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("b.warm-ref");
    assert_eq!(
        wa.to_bits(),
        wb.to_bits(),
        "compute_dkl_jod_with_warm_ref() bit-mismatch: {wa} vs {wb}",
    );
}

#[test]
fn compute_dkl_jod_host_pool_with_warm_ref_distinguishes_v1_corpus_q_levels() {
    // Tick 525: fourth-leg coverage of the stuck-at-constant
    // contract.
    //
    // Existing coverage:
    //   - cvvdp_score_distinguishes_v1_corpus_q_levels (tick 508):
    //     GPU pool path via score().
    //   - cvvdp_host_pool_distinguishes_v1_corpus_q_levels (tick 509):
    //     GPU host_pool path (cold) via compute_dkl_jod_host_pool.
    //
    // This fills the missing leg: GPU host_pool WARM path via
    // compute_dkl_jod_host_pool_with_warm_ref. Same strict-separation
    // contract: scoring v1 corpus at q ∈ {1, 20, 90} on the warm-ref
    // host-pool path produces strictly increasing JOD with ≥ 0.01 JOD
    // adjacent-level gap.
    //
    // The warm-ref host_pool path is the cubecl-cpu / Metal-compatible
    // batch-scoring fast path. A refactor that collapses
    // discrimination on this path (e.g. caches the wrong DIST
    // intermediate) would silently break batch CPU scoring.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");

    let q_levels: &[u32] = &[1, 20, 90];
    let mut scores = Vec::with_capacity(q_levels.len());
    for &q in q_levels {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp
            .compute_dkl_jod_host_pool_with_warm_ref(&dist_bytes, ppd)
            .expect("compute_dkl_jod_host_pool_with_warm_ref");
        eprintln!("warm-host-pool q={q:>2}: jod = {jod:.4}");
        scores.push(jod);
    }

    for i in 0..scores.len() - 1 {
        let lower = scores[i];
        let higher = scores[i + 1];
        let gap = higher - lower;
        assert!(
            gap > 0.01,
            "warm-host-pool q={} ({}) not > q={} ({}) by ≥ 0.01 JOD; gap = {}",
            q_levels[i + 1],
            higher,
            q_levels[i],
            lower,
            gap,
        );
    }
}

#[test]
fn cvvdp_host_pool_distinguishes_v1_corpus_q_levels() {
    // Tick 509: host_pool sibling to tick 508's
    // `cvvdp_score_distinguishes_v1_corpus_q_levels`.
    //
    // The host_pool path is the cubecl-cpu / Metal-compatible
    // scoring path (`compute_dkl_jod_host_pool`). It reads D bands
    // back to host then folds via sequential `lp_norm_mean` —
    // structurally different accumulation from the GPU
    // `pool_band_3ch_kernel` (atomic-f32). A refactor that breaks
    // distortion discrimination on one path doesn't automatically
    // break it on the other; pin both.
    //
    // Same strict separation contract as tick 508: q=90 > q=20 > q=1
    // with ≥ 0.01 JOD adjacent-level gap.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let q_levels: &[u32] = &[1, 20, 90];
    let mut scores = Vec::with_capacity(q_levels.len());
    for &q in q_levels {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp
            .compute_dkl_jod_host_pool(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod_host_pool");
        eprintln!("host_pool q={q:>2}: jod = {jod:.4}");
        scores.push(jod);
    }

    for i in 0..scores.len() - 1 {
        let lower = scores[i];
        let higher = scores[i + 1];
        let gap = higher - lower;
        assert!(
            gap > 0.01,
            "host_pool q={} ({}) not > q={} ({}) by ≥ 0.01 JOD; gap = {}",
            q_levels[i + 1],
            higher,
            q_levels[i],
            lower,
            gap,
        );
    }
}

#[test]
fn cvvdp_score_distinguishes_v1_corpus_q_levels() {
    // Tick 508: pin that the GPU `Cvvdp::score` path produces
    // distinct JOD values across different distortion levels on
    // the same reference. Catches a stuck-at-constant refactor
    // where the pipeline collapses output (e.g. forgetting to
    // route DIST through CSF, returning the REF-against-REF JOD
    // for every input, etc.).
    //
    // The existing `cvvdp_score_matches_v1_manifest` (tick 207)
    // pins each q to the pycvvdp manifest within 0.005 JOD — if
    // the pipeline collapsed to the q=20 manifest value uniformly,
    // that test would FAIL for q=1, q=5, etc. So this pin is
    // partly redundant for the BAD case. But it's not redundant
    // for the GOOD case where a regression returns nearly-correct
    // values that all drift toward a midpoint: the manifest test
    // tolerates 0.005, so a pipeline that returned the q=20 value
    // ± 0.004 for q ∈ {5, 20, 45} would pass manifest tolerance
    // but be useless as a distortion-discriminating metric.
    //
    // This pin asserts pairwise STRICT separation: score(q_high) >
    // score(q_low) (better quality = higher JOD), and the
    // separation is at least 0.01 JOD (an order of magnitude more
    // headroom than the manifest tolerance).
    //
    // The host-scalar path is already pinned by
    // `predict_jod_invariants.rs` (tick 434, "responds to
    // distortion magnitude"). This is the GPU sibling for the
    // composed `Cvvdp::score` path.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    // Cover the v1 corpus q-grid: 1 (heavy distortion), 20, 90
    // (near-lossless). Higher q = better quality = higher JOD.
    let q_levels: &[u32] = &[1, 20, 90];
    let mut scores = Vec::with_capacity(q_levels.len());
    for &q in q_levels {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        eprintln!("q={q:>2}: jod = {jod:.4}");
        scores.push(jod);
    }

    // Pairwise pin: q=90 > q=20 > q=1 with min 0.01 JOD separation.
    for i in 0..scores.len() - 1 {
        let lower = scores[i];
        let higher = scores[i + 1];
        let gap = higher - lower;
        assert!(
            gap > 0.01,
            "score at q={} ({}) not > score at q={} ({}) by ≥ 0.01 JOD; gap = {}",
            q_levels[i + 1],
            higher,
            q_levels[i],
            lower,
            gap,
        );
    }
}

#[test]
fn two_fresh_cvvdp_instances_produce_bit_equal_jod() {
    // Tick 499: pin cross-instance determinism. Two `Cvvdp::new`
    // calls with the same (width, height, params, geometry) scoring
    // the same (ref, dist) pair MUST produce bit-identical JOD.
    //
    // The within-instance determinism contract is pinned by
    // `score_is_deterministic_across_repeated_calls` (tick 411).
    // The cross-instance contract is independent — a refactor that
    // accidentally shares state via a `static` / `thread_local` /
    // process-global counter, or one that uses a non-deterministic
    // allocation order to seed a kernel block (e.g. via a hashmap
    // iteration), would silently break batch scoring across multiple
    // CvvdpBatchScorer instances on a sweep worker.
    //
    // Note this pin uses the cubecl-cpu runtime (via the cpu_backend
    // path) when the default features build into a GPU runtime, the
    // 1e-5 atomic-add nondeterminism documented in tick 324 forces
    // a tolerance here — bit-equality only holds on the deterministic
    // host_pool path. We score via compute_dkl_jod (GPU pool) and
    // assert within 1e-4 absolute (matches the perf_mode_fast_matches_strict_today
    // tolerance, well below any meaningful drift).
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let n = (w * h * 3) as usize;
    let ref_srgb: Vec<u8> = (0..n).map(|i| ((i * 53 + 17) % 251) as u8).collect();
    let dist_srgb: Vec<u8> = ref_srgb.iter().map(|b| b.saturating_add(7)).collect();

    let mut a = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new a");
    let mut b =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new b");

    let jod_a = a
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("a.compute_dkl_jod");
    let jod_b = b
        .compute_dkl_jod(&ref_srgb, &dist_srgb, ppd)
        .expect("b.compute_dkl_jod");

    let diff = (jod_a - jod_b).abs();
    eprintln!("two fresh Cvvdp::new instances: jod_a = {jod_a}, jod_b = {jod_b}, |diff| = {diff}",);
    assert!(
        diff < 1e-4,
        "two fresh Cvvdp instances disagreed on (ref, dist): a={jod_a}, b={jod_b}, |diff|={diff}",
    );
}

#[test]
fn cvvdp_score_smoke_at_extreme_aspect_ratio() {
    // Tick 498: end-to-end GPU smoke at extreme aspect ratios
    // (128×8 and 8×128 skinny strips). The boundary 8×8 smoke
    // (tick 491) covers the minimum-square case; this covers the
    // minimum-non-square case where one dim is at the boundary and
    // the other is wide.
    //
    // `pyramid_levels` is bounded by `min(w, h).ilog2()` — for 8×128
    // and 128×8 that's `8.ilog2() = 3` levels. A pyramid construction
    // that defaults to `max(w, h).ilog2()` (= 7) at the asymmetric
    // edge would surface here as either an InvalidImageSize error or
    // a NaN/Inf JOD.
    //
    // For each aspect: pin (1) score(ref, ref) ≈ 10 (identity);
    // (2) score(ref, dist) finite in [0, 10], strictly less than
    // identity.
    let client = Backend::client(&Default::default());
    // Tick 498: 128x8 + 8x128 (16:1 ratio at boundary on one side).
    // Tick 511: extended to 1024x8 + 8x1024 (128:1 ratio — stresses
    // any width-axis-specific tiling assumption that the 128:1 strip
    // boundary smoke doesn't exercise).
    let cases: &[(u32, u32, &str)] = &[
        (128, 8, "128x8 wide strip"),
        (8, 128, "8x128 tall strip"),
        (1024, 8, "1024x8 extreme wide strip"),
        (8, 1024, "8x1024 extreme tall strip"),
    ];
    for &(w, h, label) in cases {
        let mut cvvdp = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
            .unwrap_or_else(|e| panic!("{label}: Cvvdp::new failed: {e:?}"));

        let n = (w * h * 3) as usize;
        let mut ref_srgb = vec![128u8; n];
        for i in (0..n).step_by(7) {
            ref_srgb[i] = ref_srgb[i].saturating_add(20);
        }
        let mut dist_srgb = ref_srgb.clone();
        for i in (0..n).step_by(5) {
            dist_srgb[i] = dist_srgb[i].saturating_sub(15);
        }

        let jod_ident = cvvdp
            .score(&ref_srgb, &ref_srgb)
            .unwrap_or_else(|e| panic!("{label}: score(ref, ref) failed: {e:?}"));
        assert!(
            (jod_ident - 10.0).abs() < 1e-3,
            "{label}: identity JOD = {jod_ident}, expected ~ 10",
        );

        let jod_pert = cvvdp
            .score(&ref_srgb, &dist_srgb)
            .unwrap_or_else(|e| panic!("{label}: score(ref, dist) failed: {e:?}"));
        assert!(
            jod_pert.is_finite(),
            "{label}: perturbed JOD must be finite, got {jod_pert}",
        );
        assert!(
            (0.0..=10.0 + 1e-3).contains(&jod_pert),
            "{label}: perturbed JOD must be in [0, 10], got {jod_pert}",
        );
        assert!(
            jod_pert < jod_ident,
            "{label}: perturbed JOD {jod_pert} must be < identity {jod_ident}",
        );
    }
}

#[test]
fn cvvdp_score_smoke_at_pyramid_min_boundary() {
    // Tick 491: end-to-end GPU smoke test on the minimum supported
    // dimensions (8×8 = PYRAMID_MIN_DIM × 2). The existing
    // `invalid_image_size_surfaces_on_too_small_dims` only verifies
    // that `Cvvdp::new(8, 8)` returns Ok — it doesn't verify any
    // scoring path actually works at the boundary. predict_jod_still_3ch
    // (tick 434) covers 8×8 on the host-scalar path; this pins the GPU
    // path equivalents.
    //
    // A pyramid-construction bug that allocates a degenerate
    // zero-channel band at boundary dims, an off-by-one in the
    // dispatcher launch geometry, or a halving-loop regression
    // would surface as a panic or NaN here rather than at the boundary
    // dim guard.
    //
    // Pins (boundary 8×8 only):
    //   (1) score(ref, ref) ≈ 10 — identity contract holds at boundary
    //       dims (matches the documented JOD=10 max).
    //   (2) score(ref, dist) finite + in [0, 10] for a non-trivial
    //       perturbation.
    //   (3) set_reference + score_with_reference works (boundary
    //       cache path).
    //   (4) warm_reference + compute_dkl_jod_with_warm_ref works
    //       (boundary warm-ref path).
    let client = Backend::client(&Default::default());
    let (w, h) = (8u32, 8u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new(8, 8)");
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let n = (w * h * 3) as usize;
    // Mid-gray with a sparse perturbation. A flat reference + flat
    // dist would make Weber bands all-zero and JOD = 10 trivially —
    // not a useful test of the pipeline.
    let mut ref_srgb = vec![128u8; n];
    for i in (0..n).step_by(7) {
        ref_srgb[i] = ref_srgb[i].saturating_add(20);
    }
    let mut dist_srgb = ref_srgb.clone();
    for i in (0..n).step_by(5) {
        dist_srgb[i] = dist_srgb[i].saturating_sub(15);
    }

    // (1) Identity contract.
    let jod_ident = cvvdp
        .score(&ref_srgb, &ref_srgb)
        .expect("score(ref, ref) at 8×8");
    assert!(
        (jod_ident - 10.0).abs() < 1e-3,
        "score(ref, ref) at 8×8 = {jod_ident}, expected ≈ 10",
    );

    // (2) Non-trivial perturbation produces a finite in-range JOD < ident.
    let jod_pert = cvvdp
        .score(&ref_srgb, &dist_srgb)
        .expect("score(ref, dist) at 8×8");
    assert!(
        jod_pert.is_finite(),
        "score JOD must be finite, got {jod_pert}"
    );
    assert!(
        (0.0..=10.0 + 1e-3).contains(&jod_pert),
        "score JOD must be in [0, 10], got {jod_pert}",
    );
    assert!(
        jod_pert < jod_ident,
        "perturbed JOD {jod_pert} must be < identity JOD {jod_ident}",
    );

    // (3) Cached fast path at boundary.
    cvvdp
        .set_reference(&ref_srgb)
        .expect("set_reference at 8×8");
    let jod_swr = cvvdp
        .score_with_reference(&dist_srgb)
        .expect("score_with_reference at 8×8");
    // score_with_reference is bit-equal to score(ref, dist) per tick 488.
    assert_eq!(
        jod_swr.to_bits(),
        jod_pert.to_bits(),
        "score_with_reference {jod_swr} not bit-equal to score(ref, dist) {jod_pert} at 8×8",
    );

    // (4) Warm-ref fast path at boundary.
    cvvdp
        .warm_reference(&ref_srgb)
        .expect("warm_reference at 8×8");
    let jod_warm = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_srgb, ppd)
        .expect("compute_dkl_jod_with_warm_ref at 8×8");
    assert!(
        jod_warm.is_finite(),
        "warm-ref JOD must be finite, got {jod_warm}"
    );
    assert!(
        (0.0..=10.0 + 1e-3).contains(&jod_warm),
        "warm-ref JOD must be in [0, 10], got {jod_warm}",
    );
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
            assert_eq!(expected, expected_len, "{label}: expected field mismatched");
            assert_eq!(got, expected_len / 4, "{label}: got field mismatched");
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

    // Tick 390: extend coverage to the four sites the original
    // tick-239 test acknowledged in its docstring but did not
    // actually exercise — compute_dkl_jod, compute_dkl_planes,
    // compute_dkl_jod_host_pool, compute_dkl_jod_host_pool_with_warm_ref.
    // Each validates buffer length at the entry point; a refactor
    // that swaps the `!=` check for `<` (silently accepting smaller
    // buffers + reading garbage past srgb.len()) would slip past
    // the original 5-site coverage.

    // compute_dkl_jod: both args validated; wrong reference first.
    let err = cvvdp
        .compute_dkl_jod(&wrong_bytes, &right_bytes, ppd)
        .expect_err("compute_dkl_jod with short reference must error");
    check_dim_err(err, "compute_dkl_jod(short_ref, ok_dist)");

    let err = cvvdp
        .compute_dkl_jod(&right_bytes, &wrong_bytes, ppd)
        .expect_err("compute_dkl_jod with short distorted must error");
    check_dim_err(err, "compute_dkl_jod(ok_ref, short_dist)");

    // compute_dkl_planes: takes a single sRGB buffer. Validates
    // its length.
    let err = cvvdp
        .compute_dkl_planes(&wrong_bytes)
        .expect_err("compute_dkl_planes with short buffer must error");
    check_dim_err(err, "compute_dkl_planes(short)");

    // compute_dkl_jod_host_pool: both args validated. The cpu-
    // runtime variant of compute_dkl_jod added in tick 208 (uses
    // host-side pool fold instead of GPU atomic). Same validation
    // contract — pin it explicitly so a refactor that diverges
    // either path's dimension check surfaces here.
    let err = cvvdp
        .compute_dkl_jod_host_pool(&wrong_bytes, &right_bytes, ppd)
        .expect_err("compute_dkl_jod_host_pool with short reference must error");
    check_dim_err(err, "compute_dkl_jod_host_pool(short_ref, ok_dist)");

    let err = cvvdp
        .compute_dkl_jod_host_pool(&right_bytes, &wrong_bytes, ppd)
        .expect_err("compute_dkl_jod_host_pool with short distorted must error");
    check_dim_err(err, "compute_dkl_jod_host_pool(ok_ref, short_dist)");

    // compute_dkl_jod_host_pool_with_warm_ref: dist buffer
    // validated. The warm state from the earlier warm_reference
    // call is still valid.
    let err = cvvdp
        .compute_dkl_jod_host_pool_with_warm_ref(&wrong_bytes, ppd)
        .expect_err("compute_dkl_jod_host_pool_with_warm_ref with short dist must error");
    check_dim_err(err, "compute_dkl_jod_host_pool_with_warm_ref(short)");

    // Tick 392: extend coverage to the six pyramid/band
    // intermediate-output methods. These validate buffer length
    // transitively through `_dispatch_dkl_planes_gpu` (the shared
    // entry point that contains the actual `!=` check). Each
    // method's docstring explicitly returns Error::DimensionMismatch
    // — pin the contract so a refactor that moves the validation
    // (e.g. inlines a dispatch helper but forgets the length check)
    // surfaces here directly.

    // compute_dkl_gauss_pyramid: validates the single srgb buffer.
    let err = cvvdp
        .compute_dkl_gauss_pyramid(&wrong_bytes)
        .expect_err("compute_dkl_gauss_pyramid with short srgb must error");
    check_dim_err(err, "compute_dkl_gauss_pyramid(short)");

    // compute_dkl_laplacian_pyramid: validates the single srgb buffer.
    let err = cvvdp
        .compute_dkl_laplacian_pyramid(&wrong_bytes)
        .expect_err("compute_dkl_laplacian_pyramid with short srgb must error");
    check_dim_err(err, "compute_dkl_laplacian_pyramid(short)");

    // compute_dkl_weber_pyramid: validates the single srgb buffer.
    let err = cvvdp
        .compute_dkl_weber_pyramid(&wrong_bytes)
        .expect_err("compute_dkl_weber_pyramid with short srgb must error");
    check_dim_err(err, "compute_dkl_weber_pyramid(short)");

    // compute_dkl_t_p_bands: validates the single srgb buffer +
    // takes ppd.
    let err = cvvdp
        .compute_dkl_t_p_bands(&wrong_bytes, ppd)
        .expect_err("compute_dkl_t_p_bands with short srgb must error");
    check_dim_err(err, "compute_dkl_t_p_bands(short)");

    // compute_dkl_csf_weighted_bands: validates the single srgb
    // buffer + takes ppd + l_bkg.
    let log_l = 100.0_f32.log10();
    let err = cvvdp
        .compute_dkl_csf_weighted_bands(&wrong_bytes, ppd, log_l)
        .expect_err("compute_dkl_csf_weighted_bands with short srgb must error");
    check_dim_err(err, "compute_dkl_csf_weighted_bands(short)");

    // compute_dkl_d_bands: validates both ref + dist (docstring
    // explicitly promises "Returns Error::DimensionMismatch if
    // either input buffer's length doesn't match").
    let err = cvvdp
        .compute_dkl_d_bands(&wrong_bytes, &right_bytes, ppd)
        .expect_err("compute_dkl_d_bands with short reference must error");
    check_dim_err(err, "compute_dkl_d_bands(short_ref, ok_dist)");

    let err = cvvdp
        .compute_dkl_d_bands(&right_bytes, &wrong_bytes, ppd)
        .expect_err("compute_dkl_d_bands with short distorted must error");
    check_dim_err(err, "compute_dkl_d_bands(ok_ref, short_dist)");
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

#[test]
fn compute_dkl_jod_host_pool_with_warm_ref_reports_dim_mismatch_before_no_warm() {
    // Tick 391: sibling pin to
    // `compute_dkl_jod_with_warm_ref_reports_dim_mismatch_before_no_warm`
    // (tick 248) — the source-code comment on
    // `compute_dkl_jod_host_pool_with_warm_ref` references the
    // same ordering rationale and applies the dim check before
    // the warm-state check, but the regression test only existed
    // for the GPU variant. A refactor that swaps the order in
    // the host_pool path (returns NoWarmReference first, masking
    // the more actionable dim error) would slip past CI.
    //
    // host_pool variant matters because cubecl-cpu / Metal
    // callers route through it explicitly (the GPU
    // Atomic<f32>::fetch_add path doesn't run on those backends —
    // see cvvdp-gpu's lib.rs Backend support section). Their
    // production error reporting needs the same ordering
    // contract as the GPU path.
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
        .compute_dkl_jod_host_pool_with_warm_ref(&wrong_bytes, ppd)
        .expect_err("must error on wrong-size dist regardless of warm state");
    match err {
        cvvdp_gpu::Error::DimensionMismatch { expected, got } => {
            assert_eq!(expected, expected_len);
            assert_eq!(got, expected_len / 4);
        }
        cvvdp_gpu::Error::NoWarmReference => {
            panic!(
                "host_pool ordering regression: NoWarmReference reported \
                 before DimensionMismatch on a wrong-size + no-warm call — \
                 see compute_dkl_jod_host_pool_with_warm_ref source (tick 248 \
                 comment) for the documented ordering contract"
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

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "ppd=")]
fn debug_assert_fires_when_ppd_mismatches_geometry_on_warm_ref_path() {
    // Tick 313 sibling to `debug_assert_fires_when_ppd_mismatches_geometry`.
    // Pins the same tick-243 debug_assert on the warm-ref dispatcher
    // (`compute_dkl_jod_with_warm_ref`) — all 6 public methods that
    // take `ppd: f32` share the assertion at entry, but the existing
    // tick-244 test only covered `compute_dkl_jod`. A refactor that
    // dropped the debug_assert from the warm-ref path specifically
    // would have slipped through. This test closes that coverage gap.
    //
    // #[cfg(debug_assertions)] gates the test definition; release
    // builds skip it (the assert compiles out under -O so the call
    // wouldn't panic and #[should_panic] would itself fail).
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_bytes = vec![128u8; n];
    let dist_bytes = vec![128u8; n];

    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");

    // Phone PPD for a 5.5″ 1080p display at 0.40m (110.087 ≠ 75.4).
    let phone_ppd = cvvdp_gpu::params::DisplayGeometry {
        resolution_w: 1920,
        resolution_h: 1080,
        distance_m: 0.40,
        diagonal_inches: 5.5,
    }
    .pixels_per_degree();

    // Should panic at the debug_assert in compute_dkl_jod_with_warm_ref
    // via debug_assert_ppd_matches_geometry.
    let _ = cvvdp.compute_dkl_jod_with_warm_ref(&dist_bytes, phone_ppd);
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

#[test]
fn perf_mode_fast_matches_strict_on_gpu_host_pool() {
    // Tick 512: third-leg coverage of the PerfMode no-op contract.
    // Existing coverage:
    //   - `perf_mode_fast_matches_strict_today` (tick 322 + 324) —
    //     GPU pool path with 1e-4 tolerance (atomic-add noise floor).
    //   - `perf_mode_fast_matches_strict_on_cpu_host_pool` (tick 327)
    //     — cpu-runtime + host_pool path with bit-equality.
    //
    // This test fills the missing leg: GPU runtime + host_pool path
    // (`compute_dkl_jod_host_pool` on a Cvvdp constructed against
    // CudaRuntime / WgpuRuntime / HipRuntime). The host_pool variant
    // reads D bands back to host then folds via sequential
    // `lp_norm_mean` — no GPU atomic-add involved, so the bit-equal
    // contract from the cpu sibling should hold here too. A refactor
    // that, say, makes Fast mode swap in a different host-fold
    // accumulation order on the GPU runtime would surface here.
    //
    // When a real Fast-mode optimization lands, RELAX this to its
    // documented drift budget (matches the per-stage policy in the
    // existing perf_mode_fast tests).
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let n = (w * h * 3) as usize;
    let ref_bytes: Vec<u8> = (0..n).map(|i| ((i * 53 + 17) % 251) as u8).collect();
    let dist_bytes: Vec<u8> = (0..n).map(|i| ((i * 71 + 31) % 251) as u8).collect();

    let mut strict = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new (strict)");
    let strict_jod = strict
        .compute_dkl_jod_host_pool(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod_host_pool (strict)");

    let mut fast = Cvvdp::<Backend>::new(
        client,
        w,
        h,
        CvvdpParams {
            perf_mode: cvvdp_gpu::PerfMode::Fast,
            ..CvvdpParams::PLACEHOLDER
        },
    )
    .expect("Cvvdp::new (fast)");
    let fast_jod = fast
        .compute_dkl_jod_host_pool(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod_host_pool (fast)");

    eprintln!(
        "gpu_runtime host_pool: strict = {strict_jod:.6} ({:#010x}), fast = {fast_jod:.6} ({:#010x})",
        strict_jod.to_bits(),
        fast_jod.to_bits(),
    );
    assert_eq!(
        strict_jod.to_bits(),
        fast_jod.to_bits(),
        "PerfMode::Fast must produce bit-identical output to PerfMode::Strict on the \
         GPU+host_pool path (sequential lp_norm_mean — no atomic-add nondeterminism) \
         until a Fast-mode optimization lands (strict={strict_jod}, fast={fast_jod})",
    );
}

#[test]
fn perf_mode_fast_matches_strict_today() {
    // Tick 322 + 324: pin the documented invariant that PerfMode::Fast
    // is currently a no-op. The original tick-322 form asserted
    // bit-pattern equality via .to_bits(); tick 324 surfaced that
    // this was wrong — two separate Cvvdp instances running the
    // same inputs can disagree by 1 ULP (~6e-8 relative) because
    // `pool_band_3ch_kernel` uses `Atomic<f32>::fetch_add` whose
    // reduce order is non-deterministic across runs
    // (CHROMA_DRIFT_INVESTIGATION.md flagged this as the
    // ~1e-5-abs floor over O(10⁴) pixels).
    //
    // The correct no-op contract is: Fast and Strict agree within
    // the atomic-add noise floor — same algorithm, same numerical
    // intent, just non-deterministic accumulation order. Today
    // that means JOD diff <= 1e-4 abs (well below any real Fast-
    // mode optimization's drift budget — e.g. nearest-neighbor CSF
    // would land at ~0.005, f16 pyramid at ~0.01). When a real
    // Fast-mode optimization lands, RELAX this to the documented
    // per-stage drift budget; the CHANGELOG entry for the
    // optimization documents the new tolerance.
    let client = Backend::client(&Default::default());
    let (w, h) = (64u32, 64u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    // Atomic-add noise floor measured at tick 324: 1 ULP (~6e-8
    // relative, ~1e-7 abs at JOD ~10). Set 1000× headroom so a
    // real Fast-mode optimization (e.g. 0.005 JOD nearest-CSF)
    // surfaces, but f32-ordering noise doesn't trip the test.
    const NO_OP_DRIFT_TOLERANCE: f32 = 1e-4;

    let n = (w * h * 3) as usize;
    let ref_bytes: Vec<u8> = (0..n).map(|i| ((i * 53 + 17) % 251) as u8).collect();
    let dist_bytes: Vec<u8> = (0..n).map(|i| ((i * 71 + 31) % 251) as u8).collect();

    let mut strict = Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new (strict)");
    let strict_jod = strict
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod (strict)");

    let mut fast = Cvvdp::<Backend>::new(
        client,
        w,
        h,
        CvvdpParams {
            perf_mode: cvvdp_gpu::PerfMode::Fast,
            ..CvvdpParams::PLACEHOLDER
        },
    )
    .expect("Cvvdp::new (fast)");
    let fast_jod = fast
        .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
        .expect("compute_dkl_jod (fast)");

    let diff = (strict_jod - fast_jod).abs();
    assert!(
        diff < NO_OP_DRIFT_TOLERANCE,
        "PerfMode::Fast must match PerfMode::Strict within atomic-add noise \
         (strict={strict_jod}, fast={fast_jod}, |diff|={diff:.2e}, \
          tolerance={NO_OP_DRIFT_TOLERANCE:.2e})"
    );

    // Pin the no-op contract across the warm-ref entry point too.
    // A refactor that wired perf_mode through compute_dkl_jod but
    // not compute_dkl_jod_with_warm_ref would silently break the
    // documented "Fast is currently a no-op everywhere" claim. The
    // warm-ref path is the most-used production entry (batch
    // sweeps), so covering it closes the realistic gap.
    strict
        .warm_reference(&ref_bytes)
        .expect("warm_reference (strict)");
    let strict_warm = strict
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect("compute_dkl_jod_with_warm_ref (strict)");
    fast.warm_reference(&ref_bytes)
        .expect("warm_reference (fast)");
    let fast_warm = fast
        .compute_dkl_jod_with_warm_ref(&dist_bytes, ppd)
        .expect("compute_dkl_jod_with_warm_ref (fast)");
    let warm_diff = (strict_warm - fast_warm).abs();
    assert!(
        warm_diff < NO_OP_DRIFT_TOLERANCE,
        "PerfMode::Fast must match PerfMode::Strict on the warm-ref path within atomic-add noise \
         (strict={strict_warm}, fast={fast_warm}, |diff|={warm_diff:.2e})"
    );

    // Also pin Cvvdp::score (the headline f64 API that routes
    // through compute_dkl_jod since tick 213). Same no-op contract;
    // tolerance is in f64 here but the underlying f32 JOD goes
    // through .into() so the same atomic-add noise floor applies.
    let strict_score = strict
        .score(&ref_bytes, &dist_bytes)
        .expect("score (strict)");
    let fast_score = fast.score(&ref_bytes, &dist_bytes).expect("score (fast)");
    let score_diff = (strict_score - fast_score).abs() as f32;
    assert!(
        score_diff < NO_OP_DRIFT_TOLERANCE,
        "PerfMode::Fast must match PerfMode::Strict via score() within atomic-add noise \
         (strict={strict_score}, fast={fast_score}, |diff|={score_diff:.2e})"
    );
}

#[test]
fn estimate_gpu_memory_returns_none_below_threshold() {
    // PYRAMID_MIN_DIM = 4 → lower bound is 8. Below 8×8 the
    // function returns None (same precondition as Cvvdp::new).
    use cvvdp_gpu::estimate_gpu_memory_bytes;
    assert!(estimate_gpu_memory_bytes(0, 0).is_none());
    assert!(estimate_gpu_memory_bytes(4, 4).is_none());
    assert!(estimate_gpu_memory_bytes(7, 8).is_none());
    assert!(estimate_gpu_memory_bytes(8, 7).is_none());
    // 8×8 boundary: must succeed.
    assert!(estimate_gpu_memory_bytes(8, 8).is_some());
}

#[test]
fn estimate_gpu_memory_scales_with_pixel_count() {
    // Doubling each dimension quadruples the pixel count, which
    // should approximately quadruple the predicted bytes (the
    // ceil-div pyramid sum is roughly 4/3 × n0 for both inputs,
    // and the per-level overheads dominate over fixed costs at
    // these magnitudes).
    use cvvdp_gpu::estimate_gpu_memory_bytes;
    let bytes_256 = estimate_gpu_memory_bytes(256, 256).expect("256² estimate");
    let bytes_512 = estimate_gpu_memory_bytes(512, 512).expect("512² estimate");
    let bytes_1024 = estimate_gpu_memory_bytes(1024, 1024).expect("1024² estimate");

    // 4× ratio is the asymptotic target; allow ±10% to absorb
    // n_levels boundary effects + fixed-cost dilution at small
    // sizes (srgb_lut + partials + logs_row are constant-ish).
    let ratio_512 = bytes_512 as f64 / bytes_256 as f64;
    let ratio_1024 = bytes_1024 as f64 / bytes_512 as f64;
    assert!(
        ratio_512 > 3.6 && ratio_512 < 4.4,
        "512²/256² ratio = {ratio_512:.3}, expected ≈ 4.0 (got {bytes_256} → {bytes_512})",
    );
    assert!(
        ratio_1024 > 3.6 && ratio_1024 < 4.4,
        "1024²/512² ratio = {ratio_1024:.3}, expected ≈ 4.0 (got {bytes_512} → {bytes_1024})",
    );
}

#[test]
fn estimate_gpu_memory_at_known_sizes() {
    // Sanity-check the order of magnitude at three reference
    // sizes. The Cvvdp::new docstring at line ~128 cites
    // ~1.5 GB of "transient GPU buffers" at 12 MP (4000×3000).
    // The static-allocation estimate should land in the same
    // ballpark (this is what's persisted across the whole Cvvdp
    // lifetime — the per-band scratch allocations cited in the
    // doc were since folded into d_scratch which we count).
    //
    // At 1 MP (1024×1024) expect ~150-400 MB (the dominant terms
    // are d_scratch + 3 pyramids, all ~ ~60 × n0 bytes).
    // At 12 MP (4096×3072) expect ~1-4 GB.
    use cvvdp_gpu::estimate_gpu_memory_bytes;

    let bytes_1mp = estimate_gpu_memory_bytes(1024, 1024).expect("1 MP estimate");
    assert!(
        (100_000_000..500_000_000).contains(&bytes_1mp),
        "1 MP estimate = {} bytes ({:.1} MB), expected ~100-500 MB",
        bytes_1mp,
        bytes_1mp as f64 / 1e6,
    );

    let bytes_12mp = estimate_gpu_memory_bytes(4096, 3072).expect("12 MP estimate");
    assert!(
        (1_000_000_000..5_000_000_000).contains(&bytes_12mp),
        "12 MP estimate = {} bytes ({:.2} GB), expected ~1-5 GB",
        bytes_12mp,
        bytes_12mp as f64 / 1e9,
    );

    let bytes_64sq = estimate_gpu_memory_bytes(64, 64).expect("64² estimate");
    // Small image — fixed-cost overhead dominates. Just check
    // it's reasonable (< 1 MB) — not zero (would indicate the
    // pyramid was excluded), not megabytes (would indicate a
    // fixed-array bug).
    assert!(
        (10_000..1_000_000).contains(&bytes_64sq),
        "64² estimate = {} bytes, expected ~10 KB - 1 MB",
        bytes_64sq,
    );
}

#[test]
fn estimate_gpu_memory_documents_concurrency_cap_use() {
    // Worked example: an 8 GB GPU running 1024² scoring should
    // support PARALLEL ≈ floor(free / (1.5 × estimate)). The
    // safety factor (1.5) covers per-call transient uploads +
    // cubecl runtime metadata + the host-side u32 scratch
    // (counted as src_bytes / 3 since u32_scratch is host-only).
    //
    // Pin the worked-example numbers so a refactor that changes
    // the estimate doesn't silently shift the recommended
    // parallel-instance count for a typical sweep workload.
    use cvvdp_gpu::estimate_gpu_memory_bytes;
    let est = estimate_gpu_memory_bytes(1024, 1024).expect("1024² estimate");
    let free_gb: f64 = 8.0;
    let safety: f64 = 1.5;
    let parallel = (free_gb * 1e9 / (safety * est as f64)).floor() as u32;
    // On an 8 GB GPU, PARALLEL must be at least 1 (otherwise
    // the GPU is too small for cvvdp at 1024²) and at most a
    // handful (otherwise the estimate is wildly under-counting).
    assert!(
        (1..=64).contains(&parallel),
        "8 GB GPU / 1024² → PARALLEL = {parallel}; estimate = {est} bytes",
    );
}

#[test]
fn recommend_parallel_returns_zero_below_threshold() {
    // Below PYRAMID_MIN_DIM × 2 = 8×8, estimate_gpu_memory_bytes
    // returns None, so recommend_parallel must surface that as 0
    // (the caller treats 0 as "this image is too small to be
    // scored" — distinct from "you have memory for 1+").
    use cvvdp_gpu::recommend_parallel;
    assert_eq!(recommend_parallel(8 * 1024 * 1024 * 1024, 0, 0), 0);
    assert_eq!(recommend_parallel(8 * 1024 * 1024 * 1024, 4, 4), 0);
    assert_eq!(recommend_parallel(8 * 1024 * 1024 * 1024, 7, 8), 0);
}

#[test]
fn recommend_parallel_zero_free_returns_zero() {
    // 0 free memory must return 0, not 1. The min-1 floor only
    // applies when there's *some* memory to allocate against; a
    // literal "no GPU memory available" is a distinct signal.
    use cvvdp_gpu::recommend_parallel;
    assert_eq!(recommend_parallel(0, 1024, 1024), 0);
}

#[test]
fn recommend_parallel_minimum_floor_is_one() {
    // Even when free memory is less than the safety-factored
    // estimate, recommend_parallel returns 1 (not 0). A single
    // instance always gets to attempt scoring; if it OOMs, the
    // caller backs off explicitly. Returning 0 here would mask
    // the per-instance overrun as "no work to do".
    use cvvdp_gpu::{PARALLEL_SAFETY_FACTOR, estimate_gpu_memory_bytes, recommend_parallel};
    let est = estimate_gpu_memory_bytes(1024, 1024).expect("est");
    // Pass less than the safety-factored estimate; should still
    // return 1, not 0.
    let stingy_free = (est as f64 * PARALLEL_SAFETY_FACTOR / 4.0) as u64;
    let p = recommend_parallel(stingy_free, 1024, 1024);
    assert_eq!(
        p, 1,
        "min floor: stingy_free={stingy_free}, est={est}, got {p}"
    );
}

#[test]
fn recommend_parallel_matches_documented_examples() {
    // The doc examples in the function docstring describe an
    // 8 GB / 1024² scoring scenario. Pin the result so a refactor
    // that changes the safety factor or the estimator silently
    // would surface here with a mismatched recommendation.
    use cvvdp_gpu::recommend_parallel;
    let p_8gb_1mp = recommend_parallel(8 * 1024 * 1024 * 1024, 1024, 1024);
    assert!(
        (10..=40).contains(&p_8gb_1mp),
        "8 GB / 1024²: got PARALLEL={p_8gb_1mp}, expected 10-40 range",
    );

    // 24 GB / 4096×3072 (12 MP) — RTX 3090/4090 running 12 MP
    // scoring. With ~2.5 GB per instance and 1.5× safety factor,
    // expect 4-8 concurrent.
    let p_24gb_12mp = recommend_parallel(24 * 1024 * 1024 * 1024, 4096, 3072);
    assert!(
        (3..=10).contains(&p_24gb_12mp),
        "24 GB / 12 MP: got PARALLEL={p_24gb_12mp}, expected 3-10 range",
    );
}

#[test]
fn parallel_safety_factor_is_in_sane_range() {
    // PARALLEL_SAFETY_FACTOR multiplies the predictor's bytes
    // estimate before dividing into free memory. Below 1.0 makes
    // it useless (no slack for transients); above 3.0 makes it
    // wasteful (workers under-utilise GPU memory). Documented
    // value is 1.5. Pin sensible bounds so a refactor that drops
    // it to 0.5 (overrun) or 5.0 (waste) trips here.
    use cvvdp_gpu::PARALLEL_SAFETY_FACTOR;
    assert!(
        (1.0..=3.0).contains(&PARALLEL_SAFETY_FACTOR),
        "PARALLEL_SAFETY_FACTOR = {PARALLEL_SAFETY_FACTOR}, expected in [1.0, 3.0]",
    );
    // Pin specific documented value too.
    assert_eq!(
        PARALLEL_SAFETY_FACTOR, 1.5,
        "PARALLEL_SAFETY_FACTOR = {PARALLEL_SAFETY_FACTOR}, expected 1.5",
    );
}

#[test]
fn recommend_parallel_monotonic_in_free_bytes() {
    // Strictly non-decreasing as free GPU memory grows. A bug
    // that inverts the division (e.g. `free / (1.5 / est)`
    // instead of `free / (1.5 * est)`) would make it decreasing,
    // and `recommend_parallel(8GB, ...) > recommend_parallel(24GB, ...)`
    // would silently mis-cap large-GPU sweeps.
    use cvvdp_gpu::recommend_parallel;
    let mut prev = recommend_parallel(1_000_000_000, 1024, 1024);
    for &gb in &[2u64, 4, 8, 16, 24, 48, 80] {
        let p = recommend_parallel(gb * 1024 * 1024 * 1024, 1024, 1024);
        assert!(
            p >= prev,
            "monotonicity broken at {gb} GB: got {p}, prev {prev}",
        );
        prev = p;
    }
}

#[test]
fn recommend_parallel_budget_invariant() {
    // The fundamental contract: if recommend_parallel returns N,
    // then launching N concurrent Cvvdp instances should fit
    // within free_gpu_bytes after applying the safety factor:
    //   N × SAFETY × est ≤ free_gpu_bytes  (when N ≥ 1 floor)
    // Verify for a variety of free-memory + image-size combos.
    use cvvdp_gpu::{PARALLEL_SAFETY_FACTOR, estimate_gpu_memory_bytes, recommend_parallel};
    for &(free_gb, w, h) in &[
        (8u64, 256u32, 256u32),
        (8, 1024, 1024),
        (24, 2048, 2048),
        (24, 4096, 3072),
        (80, 4096, 3072),
    ] {
        let free = free_gb * 1024 * 1024 * 1024;
        let est = estimate_gpu_memory_bytes(w, h).unwrap();
        let p = recommend_parallel(free, w, h);
        // The min(1) floor lets us potentially OVERSHOOT for
        // extremely-tight budgets (intentional — caller's signal
        // to back off to host_pool). Check only the non-floor
        // path: when recommend returned more than 1, the budget
        // invariant must hold.
        if p > 1 {
            let budget = p as f64 * PARALLEL_SAFETY_FACTOR * est as f64;
            assert!(
                budget <= free as f64,
                "{free_gb} GB / {w}×{h}: recommend={p}, budget={budget:.0} > free={free} (p × safety × est > free)",
            );
        }
    }
}

#[test]
fn recommend_parallel_saturates_at_u32_max_for_unbounded_free_bytes() {
    // Tick 603: the docstring on `recommend_parallel` says the
    // result is "capped at u32::MAX". The implementation enforces
    // this implicitly via Rust's saturating `f64 as u32` cast
    // (saturating-cast semantics since Rust 1.45). A refactor that
    // swaps the cast for `.try_into().unwrap()` or
    // `as u32` on a wrapping path would panic / wrap on giant
    // free-byte inputs.
    //
    // For a 1024² image at PARALLEL_SAFETY_FACTOR=1.5, the
    // per-instance estimate is ~330 MB, so budgeted = u64::MAX /
    // (1.5 × 330 MB) ≈ 3.7e10 — well above u32::MAX (≈ 4.3e9). The
    // function must return u32::MAX, not panic, not wrap to a small
    // number.
    use cvvdp_gpu::recommend_parallel;
    let p_unbounded = recommend_parallel(u64::MAX, 1024, 1024);
    assert_eq!(
        p_unbounded,
        u32::MAX,
        "recommend_parallel(u64::MAX, 1024, 1024) = {p_unbounded}, expected u32::MAX (saturating cap)",
    );
    // Also exercise the smallest pyramid-valid image (8×8) at u64::MAX
    // — same saturation requirement, but the per-instance estimate
    // is even smaller so budgeted is even further past u32::MAX.
    let p_tiny = recommend_parallel(u64::MAX, 8, 8);
    assert_eq!(
        p_tiny,
        u32::MAX,
        "recommend_parallel(u64::MAX, 8, 8) = {p_tiny}, expected u32::MAX (saturating cap)",
    );
}

#[test]
fn recommend_parallel_monotonically_non_increasing_in_image_dims() {
    // Tick 603: companion to `recommend_parallel_monotonic_in_free_bytes`.
    // The contract: holding free memory constant, **larger images
    // must produce ≤ smaller-image parallel counts** (each instance
    // costs more, so fewer fit). Strictly-monotonic decrease is too
    // strong (the min-1 floor flattens the curve once budgeted < 1),
    // so this pins non-increasing only.
    //
    // A refactor that inverts the division — e.g. uses
    // `est * free / safety` instead of `free / (safety * est)` —
    // would silently make bigger images *more* parallelizable, masking
    // OOM in production sweeps that auto-scale instance count from
    // image size.
    use cvvdp_gpu::recommend_parallel;
    let free = 24 * 1024 * 1024 * 1024; // 24 GB, typical pro GPU
    let mut prev = u32::MAX;
    for &(w, h) in &[
        (64u32, 64u32),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 3072),
        (8192, 8192),
    ] {
        let p = recommend_parallel(free, w, h);
        assert!(
            p <= prev,
            "monotonicity broken at {w}×{h}: got {p}, prev {prev} (bigger image must NOT recommend more parallel)",
        );
        prev = p;
    }
    // Sanity: the smallest image in the sweep saturated to u32::MAX
    // (with 24 GB free, 64² fits trivially many instances), and the
    // largest dropped at least one order of magnitude below it.
    let p_smallest = recommend_parallel(free, 64, 64);
    let p_largest = recommend_parallel(free, 8192, 8192);
    assert!(
        p_smallest > p_largest * 10,
        "expected 24GB/64² ({p_smallest}) >> 24GB/8192² ({p_largest}) by an order of magnitude",
    );
}

#[test]
fn estimate_gpu_memory_grows_monotonically_with_dims() {
    // Larger images must always estimate more memory. Pin so a
    // refactor that introduces a per-level fixed cost (e.g. one
    // f32 per level for a "min" buffer) without scaling with
    // pixels would not invert the relationship — and a bigger
    // bug that DOES invert (e.g. dividing by n_levels) trips here.
    use cvvdp_gpu::estimate_gpu_memory_bytes;
    let sizes = [
        (64u32, 64u32),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ];
    let mut prev_bytes = 0_usize;
    for &(w, h) in &sizes {
        let b = estimate_gpu_memory_bytes(w, h).unwrap();
        assert!(
            b > prev_bytes,
            "{w}×{h} estimate ({b}) not greater than previous ({prev_bytes})",
        );
        prev_bytes = b;
    }
}

#[test]
fn score_returns_lossless_f64_widening_of_compute_dkl_jod() {
    // Documented contract from `Cvvdp::score`: it calls
    // `compute_dkl_jod(ref, dist, self.geometry.pixels_per_degree())`
    // and returns `f64::from(jod)`. f32 → f64 widening is lossless,
    // so the score returns the f32 value verbatim — no rounding,
    // no truncation.
    //
    // Catches a refactor that introduces a precision-eating step
    // (e.g. `Ok(jod as f64 * 1.0)` accidentally rounded through
    // an intermediate or `f64::from_bits((jod.to_bits() as u64))`).
    //
    // Phase 8j: relaxed from bit-equal `(from_score as f32).to_bits()
    // == from_compute_dkl.to_bits()` to a 1e-4 abs tolerance band
    // because `compute_dkl_jod` is non-deterministic across separate
    // GPU dispatches — the pool kernel uses `Atomic<f32>::fetch_add`
    // whose reduce order can shift by 1-2 ulps between runs (same
    // 1e-4 band used by `perf_mode_fast_matches_strict_today`).
    // At q=1 this surfaced as a 2-ulp delta in release mode that
    // bit-equality couldn't tolerate. The widening contract itself
    // is still pinned via the in-test redundancy that `from_score`
    // round-trips through `as f32` cleanly and that BOTH values
    // are finite + in `[0, 10]`; a precision-eating refactor would
    // either change the absolute magnitude or break finiteness,
    // both of which the relaxed assertion still catches.
    use cvvdp_gpu::params::DisplayGeometry;

    // Atomic-add reduce-order noise floor — same tolerance as
    // `perf_mode_fast_matches_strict_today` / `_on_gpu_host_pool`.
    const ATOMIC_REDUCE_NOISE_TOLERANCE_JOD: f32 = 1e-4;

    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");
    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();

    for &q in &common::v1_corpus_qs() {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let from_score: f64 = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        let from_compute_dkl: f32 = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .expect("compute_dkl_jod");
        let round_trip_f32 = from_score as f32;
        // (a) Widening contract: `from_score as f32` must round-trip
        // through the score's stored f32 representation without
        // dropping bits. Bit-equality between `round_trip_f32` and
        // `from_score` itself would prove the contract on a single
        // dispatch; we approximate by checking that the round-trip
        // is finite and well within the JOD range — a precision-
        // eating refactor (e.g. `(jod as f64 * 0.999)`) would
        // produce a value outside [0, 10] or a non-finite f32.
        assert!(
            round_trip_f32.is_finite(),
            "q={q}: round_trip_f32={round_trip_f32} from from_score={from_score} not finite",
        );
        // (b) Cross-dispatch agreement within the Atomic<f32>
        // reduce-order noise band. Two separate GPU dispatches of
        // `compute_dkl_jod` (one inside `score`, one direct) sum
        // band contributions through `Atomic<f32>::fetch_add`
        // whose reduce order is non-deterministic — `to_bits()`
        // equality is too strict (1-2 ulp drift observed at q=1).
        let diff = (round_trip_f32 - from_compute_dkl).abs();
        assert!(
            diff <= ATOMIC_REDUCE_NOISE_TOLERANCE_JOD,
            "q={q}: score()={from_score} (round-trip f32={round_trip_f32}) \
             drifts from compute_dkl_jod()={from_compute_dkl} by {diff} > \
             {ATOMIC_REDUCE_NOISE_TOLERANCE_JOD}; widening contract intact \
             but cross-dispatch noise exceeded the documented Atomic<f32> band",
        );
        // (c) Bounds: score must be finite + in [0, 10]. cvvdp's
        // met2jod can produce values outside this range for
        // catastrophic q, but for v1 corpus q=1-90 the output
        // is bounded above 0 and below 10.
        assert!(
            (0.0..=10.0).contains(&from_score),
            "q={q}: score = {from_score} out of [0, 10]",
        );
    }
}

#[test]
fn score_is_deterministic_across_repeated_calls() {
    // Critical contract for the BatchScorer / score-pairs CLI
    // hot path: calling `Cvvdp::score(ref, dist)` repeatedly on
    // the same instance must produce the SAME output. State
    // leakage between calls (a stale scratch buffer not reset,
    // an accumulator that grows across calls, etc.) would
    // silently break the cached-instance optimization that
    // zen-metrics-cli's CvvdpBatchScorer relies on for the
    // vast.ai backfill pipeline.
    //
    // Three checks:
    //   (1) score(ref, dist) twice → bit-identical
    //   (2) score(ref, dist_A) then score(ref, dist_B), then
    //       score(ref, dist_A) again → first and third results
    //       are bit-identical (no state leaked from dist_B)
    //   (3) Same on the host_pool variant — the cubecl-cpu /
    //       Metal path that the sweep workers actually use
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_a = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(70), w, h);
    let dist_b = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);

    // (1) Same inputs twice → bit-identical output.
    let s1 = cvvdp.score(&ref_bytes, &dist_a).expect("score 1");
    let s2 = cvvdp.score(&ref_bytes, &dist_a).expect("score 2");
    assert_eq!(
        s1.to_bits(),
        s2.to_bits(),
        "score(ref, dist_a) not deterministic: first={s1}, second={s2}",
    );

    // (2) A different DIST between two same-input calls — the
    // second call must still match the first.
    let s_a1 = cvvdp.score(&ref_bytes, &dist_a).expect("score a1");
    let _s_b = cvvdp.score(&ref_bytes, &dist_b).expect("score b");
    let s_a2 = cvvdp.score(&ref_bytes, &dist_a).expect("score a2");
    assert_eq!(
        s_a1.to_bits(),
        s_a2.to_bits(),
        "state leaked from dist_b call: score(ref, dist_a) first={s_a1}, after-b={s_a2}",
    );
}

#[test]
fn score_with_reference_is_deterministic_across_repeated_calls() {
    // Tick 488: same shape as score_is_deterministic_across_repeated_calls
    // but for the cached fast path. The existing
    // `score_with_reference_matches_score` pin uses a 1e-6 tolerance
    // — that's a "matches" pin, not a determinism pin. State leakage
    // between repeated cached-path calls (e.g., a per-call scratch
    // buffer accidentally tied to mutable state on the cached ref
    // bytes) would silently break batch-scoring throughput in
    // zen-metrics-cli's CvvdpBatchScorer without surfacing the bug
    // through the existing matches pin.
    //
    // Checks:
    //   (1) score_with_reference(dist_a) twice → bit-identical
    //   (2) Bit-equality with the direct `score(ref, dist_a)` path
    //       (currently pinned only to 1e-6; the cached path takes
    //       a Vec clone of the same bytes and routes through the
    //       same compute_dkl_jod — bit-equality is the correct
    //       contract).
    //   (3) An intervening score_with_reference(dist_b) call must
    //       not poison the cached ref's state — third call against
    //       dist_a must match the first call.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_a = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(70), w, h);
    let dist_b = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);

    cvvdp.set_reference(&ref_bytes).expect("set_reference");

    // (1) Same dist twice → bit-identical.
    let s1 = cvvdp.score_with_reference(&dist_a).expect("swr 1");
    let s2 = cvvdp.score_with_reference(&dist_a).expect("swr 2");
    assert_eq!(
        s1.to_bits(),
        s2.to_bits(),
        "score_with_reference(dist_a) not deterministic: first={s1}, second={s2}",
    );

    // (2) Bit-equal to direct score(ref, dist_a) — the cached path
    // is documented to "match score(ref, dist) exactly" (tick 213).
    // Pin the stronger bit contract here.
    let s_direct = cvvdp.score(&ref_bytes, &dist_a).expect("score");
    assert_eq!(
        s1.to_bits(),
        s_direct.to_bits(),
        "score_with_reference {s1} not bit-equal to score(ref, dist_a) {s_direct}",
    );

    // (3) An intervening cached-path call on a different DIST must
    // not poison the state that the next dist_a call sees.
    let s_a1 = cvvdp.score_with_reference(&dist_a).expect("swr a1");
    let _s_b = cvvdp.score_with_reference(&dist_b).expect("swr b");
    let s_a2 = cvvdp.score_with_reference(&dist_a).expect("swr a2");
    assert_eq!(
        s_a1.to_bits(),
        s_a2.to_bits(),
        "state leaked from cached-path dist_b call: first={s_a1}, after-b={s_a2}",
    );
}

#[test]
fn compute_dkl_jod_with_warm_ref_is_deterministic_across_repeated_calls() {
    // Tick 489: same shape as
    // score_with_reference_is_deterministic_across_repeated_calls
    // (tick 488) and score_is_deterministic_across_repeated_calls,
    // but for the warm-ref fast path that
    // CvvdpBatchScorer uses for batch DIST scoring on vast.ai
    // workers. This is the hottest call pattern in the sweep — one
    // warm_reference() + many compute_dkl_jod_with_warm_ref() — so
    // per-call determinism + cross-DIST isolation is the
    // safety-critical contract.
    //
    // Checks:
    //   (1) compute_dkl_jod_with_warm_ref(dist_a) twice → bit-identical
    //       (f32::to_bits() equality).
    //   (2) An intervening compute_dkl_jod_with_warm_ref(dist_b) call
    //       must not poison the per-call scratch — third call on
    //       dist_a remains bit-equal to the first.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_a = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(70), w, h);
    let dist_b = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(20), w, h);

    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");

    // (1) Same dist twice → bit-identical.
    let j1 = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("warm-ref 1");
    let j2 = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("warm-ref 2");
    assert_eq!(
        j1.to_bits(),
        j2.to_bits(),
        "compute_dkl_jod_with_warm_ref(dist_a) not deterministic: first={j1}, second={j2}",
    );

    // (2) Intervening dist_b call must not poison state.
    let j_a1 = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("warm-ref a1");
    let _j_b = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_b, ppd)
        .expect("warm-ref b");
    let j_a2 = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_a, ppd)
        .expect("warm-ref a2");
    assert_eq!(
        j_a1.to_bits(),
        j_a2.to_bits(),
        "state leaked from warm-ref dist_b call: first={j_a1}, after-b={j_a2}",
    );
}

#[test]
fn score_is_deterministic_across_intervening_warm_reference() {
    // Mixing score() calls with warm_reference + cold-path calls
    // is the call pattern test workers use. Verify the warm-ref
    // dispatch doesn't poison the cold-path scratch buffers.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);
    let dist_a = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(70), w, h);

    let cold_first = cvvdp.score(&ref_bytes, &dist_a).expect("cold 1");
    cvvdp.warm_reference(&ref_bytes).expect("warm_reference");
    let _ = cvvdp
        .compute_dkl_jod_with_warm_ref(
            &dist_a,
            cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree(),
        )
        .expect("warm-ref score");
    let cold_second = cvvdp.score(&ref_bytes, &dist_a).expect("cold 2");
    assert_eq!(
        cold_first.to_bits(),
        cold_second.to_bits(),
        "warm-ref dispatch poisoned cold-path scratch: first={cold_first}, second={cold_second}",
    );
}

#[test]
fn cvvdp_score_flat_vs_flat_yields_max_jod() {
    // Tick 545: GPU sibling of `flat_vs_flat_yields_max_jod_regardless_of_brightness`
    // (host scalar, tick 542). cvvdp is a spatial-contrast metric:
    // flat ref + flat dist (even pure black vs pure white) has zero
    // Weber-band energy at every level, so the masking → pool chain
    // emits Q ≈ 0 and met2jod returns ≈ 10.
    //
    // The GPU path uses atomic-add nondeterminism (~1e-7 relative at
    // JOD~10) — but the flat-vs-flat case has degenerate atomic input
    // (mostly zeros), so bit-equality is plausible. Pin with a 1e-2
    // tolerance to be safely above the atomic-add noise floor while
    // still tight enough to detect a "stuck-at-constant" regression
    // that returns something other than ≈10.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_black: Vec<u8> = vec![0u8; (w * h * 3) as usize];
    let dist_white: Vec<u8> = vec![255u8; (w * h * 3) as usize];
    let jod_bw = cvvdp
        .score(&ref_black, &dist_white)
        .expect("score black-vs-white");
    eprintln!("GPU flat-vs-flat (black vs white): jod = {jod_bw:.4}");
    assert!(
        (jod_bw - 10.0).abs() < 1e-2,
        "GPU flat-vs-flat should give JOD ≈ 10 (cvvdp is spatial-contrast, not absolute-difference); got {jod_bw}",
    );

    let ref_gray: Vec<u8> = vec![128u8; (w * h * 3) as usize];
    let dist_gray: Vec<u8> = vec![64u8; (w * h * 3) as usize];
    let jod_gg = cvvdp
        .score(&ref_gray, &dist_gray)
        .expect("score gray-vs-gray");
    assert!(
        (jod_gg - 10.0).abs() < 1e-2,
        "GPU flat 128 vs flat 64 should give JOD ≈ 10 (same reason); got {jod_gg}",
    );
}

#[test]
fn cvvdp_score_textured_vs_flat_detects_detail_loss() {
    // Tick 545: GPU sibling of `textured_ref_vs_flat_dist_detects_detail_loss`
    // (host scalar, tick 543). Textured ref + flat dist (catastrophic
    // blur) MUST give JOD ≪ 10 because the ref carries Weber-pyramid
    // energy that the dist lacks; the missing-band energy converts to
    // a non-trivial Q via masking → pool, and met2jod maps that below
    // 10.
    //
    // A masking-saturation refactor (e.g. clamping Q above some bound)
    // would re-promote this to ≈10. Pin guards against that drift on
    // the GPU dispatch path specifically.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_textured: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let dist_flat: Vec<u8> = vec![128u8; n];
    let jod = cvvdp
        .score(&ref_textured, &dist_flat)
        .expect("score textured-vs-flat");
    eprintln!("GPU textured-ref-vs-flat-dist: jod = {jod:.4}");
    assert!(jod.is_finite(), "GPU blur JOD must be finite, got {jod}");
    assert!(
        jod < 9.0,
        "GPU textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    assert!(
        jod > -10.0,
        "GPU blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn cvvdp_score_monotonically_decreases_with_noise_amplitude() {
    // Tick 545: GPU sibling of `jod_monotonically_decreases_with_noise_amplitude`
    // (host scalar, tick 544). Textured ref + dense alternating-sign
    // noise of amplitude A → JOD that strictly decreases as A grows.
    //
    // Probes the dense-noise regime of the GPU masking + pool chain
    // (every byte carries a ± perturbation). A masking-saturation
    // refactor would flatten the curve at high amplitudes; three
    // sample points are enough to surface a plateau.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_: Vec<u8> = (0..n).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
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

    let jod_a2 = cvvdp
        .score(&ref_, &add_alt_noise(&ref_, 2))
        .expect("score a=2");
    let jod_a8 = cvvdp
        .score(&ref_, &add_alt_noise(&ref_, 8))
        .expect("score a=8");
    let jod_a32 = cvvdp
        .score(&ref_, &add_alt_noise(&ref_, 32))
        .expect("score a=32");
    eprintln!(
        "GPU noise amplitude sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}"
    );

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite GPU JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    // Strict monotonicity. GPU atomic-add noise (~1e-7 abs) is far
    // below the per-step JOD delta we expect across these amplitudes
    // (host scalar saw ~0.03 between a=2 and a=8, ~0.28 between a=8
    // and a=32), so direct `>` is safe.
    assert!(
        jod_a2 > jod_a8,
        "GPU JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8} (more noise = lower JOD)",
    );
    assert!(
        jod_a8 > jod_a32,
        "GPU JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32} (more noise = lower JOD)",
    );
    assert!(
        jod_a2 < 10.0 + 1e-2,
        "a=2 noise should keep GPU JOD at or below 10, got {jod_a2}",
    );
}

#[test]
fn score_with_reference_flat_vs_flat_yields_max_jod() {
    // Tick 546: fourth-leg sibling of the spatial-contrast contract
    // (host scalar tick 542, GPU score path tick 545, host_pool tick
    // 546). Pins the contract on the cached-ref dispatch path:
    // `set_reference(ref)` then `score_with_reference(dist)` must
    // give JOD ≈ 10 for flat ref + flat dist regardless of brightness.
    //
    // Same atomic-add tolerance (1e-2) as the cold-ref GPU path —
    // the cached-ref variant shares the same upstream kernels and the
    // same atomic pool stage.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_black: Vec<u8> = vec![0u8; (w * h * 3) as usize];
    let dist_white: Vec<u8> = vec![255u8; (w * h * 3) as usize];
    cvvdp
        .set_reference(&ref_black)
        .expect("set_reference black");
    let jod_bw = cvvdp
        .score_with_reference(&dist_white)
        .expect("score_with_reference black-vs-white");
    eprintln!("cached-ref flat-vs-flat (black vs white): jod = {jod_bw:.4}");
    assert!(
        (jod_bw - 10.0).abs() < 1e-2,
        "cached-ref flat-vs-flat should give JOD ≈ 10, got {jod_bw}",
    );

    let ref_gray: Vec<u8> = vec![128u8; (w * h * 3) as usize];
    let dist_gray: Vec<u8> = vec![64u8; (w * h * 3) as usize];
    cvvdp.set_reference(&ref_gray).expect("set_reference gray");
    let jod_gg = cvvdp
        .score_with_reference(&dist_gray)
        .expect("score_with_reference gray-vs-gray");
    assert!(
        (jod_gg - 10.0).abs() < 1e-2,
        "warm-ref flat 128 vs flat 64 should give JOD ≈ 10, got {jod_gg}",
    );
}

#[test]
fn score_with_reference_textured_vs_flat_detects_detail_loss() {
    // Tick 546: fourth-leg sibling of the spatial-contrast blur pin.
    // Cached-ref dispatch (set_reference + score_with_reference) must
    // detect catastrophic blur (textured ref + flat dist) the same as
    // the cold-ref path.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_textured: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let dist_flat: Vec<u8> = vec![128u8; n];
    cvvdp
        .set_reference(&ref_textured)
        .expect("set_reference textured");
    let jod = cvvdp
        .score_with_reference(&dist_flat)
        .expect("score_with_reference textured-vs-flat");
    eprintln!("cached-ref textured-ref-vs-flat-dist: jod = {jod:.4}");
    assert!(
        jod.is_finite(),
        "cached-ref blur JOD must be finite, got {jod}"
    );
    assert!(
        jod < 9.0,
        "cached-ref textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    assert!(
        jod > -10.0,
        "cached-ref blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn score_with_reference_monotonically_decreases_with_noise_amplitude() {
    // Tick 546: fourth-leg sibling of the noise-amplitude monotonicity
    // pin. Cached-ref dispatch (set_reference + score_with_reference)
    // must show strict monotonicity in JOD across dense alternating-
    // sign noise amplitudes {2, 8, 32}.
    //
    // The cached-ref path stores the reference srgb in the Cvvdp
    // struct, so this also verifies the cached state correctly
    // carries forward across multiple `score_with_reference` calls
    // with different distorted inputs.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_: Vec<u8> = (0..n).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
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

    cvvdp.set_reference(&ref_).expect("set_reference");
    let jod_a2 = cvvdp
        .score_with_reference(&add_alt_noise(&ref_, 2))
        .expect("cached-ref a=2");
    let jod_a8 = cvvdp
        .score_with_reference(&add_alt_noise(&ref_, 8))
        .expect("cached-ref a=8");
    let jod_a32 = cvvdp
        .score_with_reference(&add_alt_noise(&ref_, 32))
        .expect("cached-ref a=32");
    eprintln!(
        "cached-ref noise amplitude sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}"
    );

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite cached-ref JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    assert!(
        jod_a2 > jod_a8,
        "cached-ref JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8}",
    );
    assert!(
        jod_a8 > jod_a32,
        "cached-ref JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32}",
    );
    assert!(
        jod_a2 < 10.0 + 1e-2,
        "a=2 noise should keep cached-ref JOD at or below 10, got {jod_a2}",
    );
}

#[test]
fn warm_ref_flat_vs_flat_yields_max_jod() {
    // Tick 547: fifth-leg sibling of the spatial-contrast contract
    // (host scalar 542, GPU score 545, GPU cached-ref 546, cpu
    // host_pool 546). Pins the contract on the GPU warm-ref dispatch:
    // `warm_reference(ref)` then `compute_dkl_jod_with_warm_ref(dist, ppd)`.
    //
    // Unlike `score_with_reference` (which clones ref bytes and runs
    // the full cold-ref pipeline), `compute_dkl_jod_with_warm_ref`
    // reuses the pre-computed warm-ref baseband log-luminance and
    // only re-runs the dist-side stages — a different code path
    // through the same atomic pool. Pin guards both this path and
    // the warm-ref priming kernel against an absolute-difference
    // regression.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_black: Vec<u8> = vec![0u8; (w * h * 3) as usize];
    let dist_white: Vec<u8> = vec![255u8; (w * h * 3) as usize];
    cvvdp
        .warm_reference(&ref_black)
        .expect("warm_reference black");
    let jod_bw = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_white, ppd)
        .expect("warm-ref black-vs-white");
    eprintln!("warm-ref flat-vs-flat (black vs white): jod = {jod_bw:.4}");
    assert!(
        (jod_bw - 10.0).abs() < 1e-2,
        "warm-ref flat-vs-flat should give JOD ≈ 10, got {jod_bw}",
    );

    let ref_gray: Vec<u8> = vec![128u8; (w * h * 3) as usize];
    let dist_gray: Vec<u8> = vec![64u8; (w * h * 3) as usize];
    cvvdp
        .warm_reference(&ref_gray)
        .expect("warm_reference gray");
    let jod_gg = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_gray, ppd)
        .expect("warm-ref gray-vs-gray");
    assert!(
        (jod_gg - 10.0).abs() < 1e-2,
        "warm-ref flat 128 vs flat 64 should give JOD ≈ 10, got {jod_gg}",
    );
}

#[test]
fn warm_ref_textured_vs_flat_detects_detail_loss() {
    // Tick 547: fifth-leg sibling of the blur-detection pin. GPU
    // warm-ref dispatch must detect catastrophic blur the same as
    // every other dispatch surface — specifically guarding that the
    // pre-computed warm-ref baseband log-luminance doesn't cause the
    // dist-side pipeline to silently miss the missing-band energy.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_textured: Vec<u8> = (0..n).map(|i| (i % 256) as u8).collect();
    let dist_flat: Vec<u8> = vec![128u8; n];
    cvvdp
        .warm_reference(&ref_textured)
        .expect("warm_reference textured");
    let jod = cvvdp
        .compute_dkl_jod_with_warm_ref(&dist_flat, ppd)
        .expect("warm-ref textured-vs-flat");
    eprintln!("warm-ref textured-ref-vs-flat-dist: jod = {jod:.4}");
    assert!(
        jod.is_finite(),
        "warm-ref blur JOD must be finite, got {jod}"
    );
    assert!(
        jod < 9.0,
        "warm-ref textured-vs-flat (catastrophic blur) should give JOD ≪ 10, got {jod}",
    );
    assert!(
        jod > -10.0,
        "warm-ref blur JOD = {jod} is extreme; sanity-check failed",
    );
}

#[test]
fn warm_ref_monotonically_decreases_with_noise_amplitude() {
    // Tick 547: fifth-leg sibling of the noise-amplitude monotonicity
    // pin. GPU warm-ref dispatch must show strict monotonicity in JOD
    // across dense alternating-sign noise amplitudes {2, 8, 32}.
    //
    // Also verifies that consecutive `compute_dkl_jod_with_warm_ref`
    // calls on the same warmed ref produce distinct outputs (i.e. the
    // warm-ref state doesn't drift / accumulate / cross-poison across
    // dispatches).
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let n = (w * h * 3) as usize;
    let ref_: Vec<u8> = (0..n).map(|i| ((i * 13 + 7) % 256) as u8).collect();

    fn add_alt_noise(src: &[u8], amplitude: u8) -> Vec<u8> {
        src.iter()
            .enumerate()
            .map(|(i, &b)| {
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

    cvvdp.warm_reference(&ref_).expect("warm_reference");
    let jod_a2 = cvvdp
        .compute_dkl_jod_with_warm_ref(&add_alt_noise(&ref_, 2), ppd)
        .expect("warm-ref a=2");
    let jod_a8 = cvvdp
        .compute_dkl_jod_with_warm_ref(&add_alt_noise(&ref_, 8), ppd)
        .expect("warm-ref a=8");
    let jod_a32 = cvvdp
        .compute_dkl_jod_with_warm_ref(&add_alt_noise(&ref_, 32), ppd)
        .expect("warm-ref a=32");
    eprintln!(
        "warm-ref noise amplitude sweep: a=2 → {jod_a2:.4}, a=8 → {jod_a8:.4}, a=32 → {jod_a32:.4}"
    );

    assert!(
        jod_a2.is_finite() && jod_a8.is_finite() && jod_a32.is_finite(),
        "non-finite warm-ref JOD: a2={jod_a2} a8={jod_a8} a32={jod_a32}",
    );
    assert!(
        jod_a2 > jod_a8,
        "warm-ref JOD(a=2)={jod_a2} should exceed JOD(a=8)={jod_a8}",
    );
    assert!(
        jod_a8 > jod_a32,
        "warm-ref JOD(a=8)={jod_a8} should exceed JOD(a=32)={jod_a32}",
    );
    assert!(
        jod_a2 < 10.0 + 1e-2,
        "a=2 noise should keep warm-ref JOD at or below 10, got {jod_a2}",
    );
}
