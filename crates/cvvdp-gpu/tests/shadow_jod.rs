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

#[path = "common/mod.rs"]
mod common;

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
    use cubecl::Runtime;
    use cvvdp_gpu::Cvvdp;
    use cvvdp_gpu::params::CvvdpParams;

    // Prefer cuda when both backends are compiled in; fall back to
    // wgpu otherwise. Matches the type-alias pattern in
    // `pipeline_score.rs` / `pipeline_color.rs`.
    #[cfg(feature = "cuda")]
    type Backend = cubecl::cuda::CudaRuntime;
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    type Backend = cubecl::wgpu::WgpuRuntime;
    #[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
    type Backend = cubecl::hip::HipRuntime;

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
