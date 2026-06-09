//! Composed host-scalar pipeline integration test on the
//! zenmetrics-corpus. Each (ref, dist) pair runs through the full
//! still-image cvvdp chain (color → pyramid → CSF → masking →
//! pooling → JOD) at the standard_4k display config.
//!
//! Post-ticks 204/206 (chroma_shift + 73×91 odd-dim drifts closed),
//! the host-scalar shadow matches the v1 R2 manifest values
//! (standard_4k display) within 0.005 JOD across all 6 q-levels:
//!
//! ```text
//!   q    pycvvdp manifest   shadow scalar   |diff|
//!   1    7.6536             7.6538          0.0002
//!   5    8.8889             8.8903          0.0014
//!   20   9.7076             9.7091          0.0015
//!   45   9.8273             9.8295          0.0022
//!   70   9.8915             9.8946          0.0031
//!   90   9.9930             9.9929          0.0001
//! ```
//!
//! Test assertions:
//! - JOD is finite and in `[0, 10]`.
//! - Each shadow JOD is within 0.005 of the pycvvdp manifest value.
//!   Was 0.05 before ticks 204/206 closed the chroma_shift and
//!   73×91 odd-dim drifts; tightened to the standard 0.005
//!   tolerance in tick 207 once all 6 q levels measured ≤ 0.0031
//!   vs manifest.
//! - JOD is monotonic non-decreasing across q.

use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

use crate::common;

use common::load_rgb_bytes;

#[test]
fn shadow_jod_runs_and_is_monotonic_on_corpus() {
    let (w, h) = (256u32, 256u32);

    let display = DisplayModel::STANDARD_4K;
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // pycvvdp_manifest_jod loaded from
    // scripts/cvvdp_goldens/v1_corpus_jods.json via the common
    // helper — single source of truth so a build_goldens.py
    // rerun + JSON bump propagates without hand-editing test
    // constants. Tick 253 dedup.
    let qs = common::v1_corpus_qs();
    let cases: Vec<(u32, f32)> = qs
        .iter()
        .map(|&q| (q, common::v1_corpus_jod_golden(q)))
        .collect();
    let mut jods = Vec::with_capacity(cases.len());
    for &(q, expected) in &cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = predict_jod_still_3ch(
            &ref_bytes,
            &dist_bytes,
            w as usize,
            h as usize,
            display,
            ppd,
        );
        let diff = (jod - expected).abs();
        eprintln!("q={q:>2}: shadow JOD = {jod:.4} (pycvvdp {expected:.4}, |diff| {diff:.4})");
        assert!(jod.is_finite(), "q={q}: JOD = {jod} (not finite)");
        assert!(
            (0.0..=10.0).contains(&jod),
            "q={q}: JOD = {jod} out of [0, 10]"
        );
        assert!(
            diff < 0.005,
            "q={q}: shadow JOD {jod:.4} diverges from pycvvdp manifest {expected:.4} by {diff:.4} > 0.005"
        );
        jods.push((q, jod));
    }

    // Monotone non-decreasing across q.
    for win in jods.windows(2) {
        let (q_lo, j_lo) = win[0];
        let (q_hi, j_hi) = win[1];
        assert!(
            j_hi + 1e-3 >= j_lo,
            "non-monotone: q={q_lo} JOD={j_lo:.4} > q={q_hi} JOD={j_hi:.4}"
        );
    }
}

/// Same manifest-parity check as `shadow_jod_runs_and_is_monotonic_on_corpus`
/// but routed through the GPU pipeline via [`Cvvdp::compute_dkl_jod`].
/// Anchors the full GPU composition path (color → weber → CSF →
/// masking → spatial pool → host fold) directly against pycvvdp
/// v0.5.4's published manifest values, complementing the
/// `compute_dkl_jod_matches_host_scalar` test (which only pins GPU
/// vs host scalar at f32 precision, not vs the manifest).
///
/// Tick 207: tightened from the old per-q tolerance schedule
/// (0.5 at q=1, 0.1 at q=5, 0.05 at q≥20) to a flat 0.005 across
/// the manifest after ticks 204/206 closed the chroma_shift and
/// 73×91 odd-dim drifts. Measured diffs are 0.0000–0.0031 across
/// all 6 q levels.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]
#[test]
fn shadow_jod_gpu_runs_and_is_close_to_manifest_on_corpus() {
    use common::Backend;
    use cubecl::Runtime;
    use cvvdp_gpu::Cvvdp;
    use cvvdp_gpu::params::CvvdpParams;

    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // pycvvdp_manifest_jod loaded from the canonical
    // scripts/cvvdp_goldens/v1_corpus_jods.json via the common
    // helper (tick 253 dedup — was hand-mirroring the values
    // alongside the sibling shadow_jod_runs_and_is_monotonic_on_corpus
    // test).
    let qs = common::v1_corpus_qs();
    let cases: Vec<(u32, f32)> = qs
        .iter()
        .map(|&q| (q, common::v1_corpus_jod_golden(q)))
        .collect();

    let client = Backend::client(&Default::default());
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp on GPU backend");

    // Tick 207: flat 0.005 tolerance across all q levels. Was a
    // per-q schedule (0.5 / 0.1 / 0.05) before ticks 204/206 closed
    // the chroma_shift and 73×91 odd-dim drifts; measured GPU vs
    // pycvvdp diffs are now 0.0000–0.0031 across the manifest.
    let tol_for = |_q: u32| -> f32 { 0.005 };
    let mut jods = Vec::with_capacity(cases.len());
    for &(q, expected) in &cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp
            .compute_dkl_jod(&ref_bytes, &dist_bytes, ppd)
            .unwrap_or_else(|e| panic!("compute_dkl_jod failed at q={q}: {e:?}"));
        let diff = (jod - expected).abs();
        let tol = tol_for(q);
        eprintln!(
            "q={q:>2}: GPU JOD = {jod:.4} (pycvvdp {expected:.4}, |diff| {diff:.4}, tol {tol:.2})"
        );
        assert!(jod.is_finite(), "q={q}: JOD = {jod} (not finite)");
        assert!(
            (0.0..=10.0).contains(&jod),
            "q={q}: JOD = {jod} out of [0, 10]"
        );
        assert!(
            diff < tol,
            "q={q}: GPU JOD {jod:.4} diverges from pycvvdp manifest {expected:.4} by {diff:.4} > {tol:.2}"
        );
        jods.push((q, jod));
    }

    for win in jods.windows(2) {
        let (q_lo, j_lo) = win[0];
        let (q_hi, j_hi) = win[1];
        assert!(
            j_hi + 1e-3 >= j_lo,
            "non-monotone: q={q_lo} JOD={j_lo:.4} > q={q_hi} JOD={j_hi:.4}"
        );
    }
}

// The `# Panics` section on `predict_jod_still_3ch` documents two
// dim-mismatch assertions. Pin both as separate `#[should_panic]`
// tests so a refactor that drops one assertion is caught — they
// guard against the most likely caller bug (sRGB buffer with the
// wrong stride / width / height tuple). `Cvvdp::score` returns
// `Error::DimensionMismatch` instead of panicking; `predict_jod_still_3ch`
// keeps the panic contract because it's the host-scalar reference
// path that fixtures and shadow tests invoke with hand-rolled
// buffers.

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn predict_jod_still_3ch_panics_on_ref_dim_mismatch() {
    let (w, h) = (8usize, 8usize);
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    // ref short by 1 byte.
    let bad_ref = vec![128u8; w * h * 3 - 1];
    let ok_dist = vec![128u8; w * h * 3];
    let _ = predict_jod_still_3ch(&bad_ref, &ok_dist, w, h, display, ppd);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn predict_jod_still_3ch_panics_on_dist_dim_mismatch() {
    let (w, h) = (8usize, 8usize);
    let display = DisplayModel::STANDARD_4K;
    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let ok_ref = vec![128u8; w * h * 3];
    // dist short by 1 byte — second assertion fires only when the
    // first passed.
    let bad_dist = vec![128u8; w * h * 3 - 1];
    let _ = predict_jod_still_3ch(&ok_ref, &bad_dist, w, h, display, ppd);
}

#[test]
fn predict_jod_still_3ch_returns_max_jod_on_identical_inputs() {
    // Identity contract: scoring a buffer against itself yields
    // the maximum JOD (≈ 10.0 — "imperceptible difference"). The
    // doctest on `predict_jod_still_3ch` asserts this on a 64×64
    // gray buffer; promote that to an integration test so it
    // runs in the standard `cargo test --test shadow_jod` path
    // (doctests are skipped when filtering by `--test <name>`).
    //
    // Companion to `cpu_backend.rs::compute_dkl_jod_host_pool_returns_max_jod_on_identical_inputs`
    // (tick 350) which pins the same contract on the GPU host-pool
    // path. This is the host-scalar reference twin: a refactor
    // that diverges either path's identity output surfaces
    // independently.
    use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
    use cvvdp_gpu::params::{DisplayGeometry, DisplayModel};

    // Test three sizes spanning small/mid/edge:
    //   - 8×8: the PYRAMID_MIN_DIM × 2 boundary
    //   - 64×64: matches the lib.rs doctest example
    //   - 73×91: the odd-dim 'gausspyr_reduce' parity case from
    //     ticks 204-206 (pycvvdp's column-parity bug). Identity
    //     output should still equal 10 there — if the bug-compat
    //     boundary patches break identity, this trips.
    for (w, h, label) in [
        (8_usize, 8_usize, "8×8 boundary"),
        (64, 64, "64×64 doctest size"),
        (73, 91, "73×91 odd-dim"),
    ] {
        for &val in &[0_u8, 128, 255] {
            let bytes = vec![val; w * h * 3];
            let jod = predict_jod_still_3ch(
                &bytes,
                &bytes,
                w,
                h,
                DisplayModel::STANDARD_4K,
                DisplayGeometry::STANDARD_4K.pixels_per_degree(),
            );
            assert!(
                (jod - 10.0).abs() < 1e-3,
                "{label} val={val}: predict_jod_still_3ch identity = {jod}, expected ≈ 10.0",
            );
        }
    }
}
