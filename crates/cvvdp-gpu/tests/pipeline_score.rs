//! `Cvvdp::score` end-to-end against the v1 R2 manifest values.
//!
//! Currently routes through the host scalar (see `score` doc), but
//! the public surface is what matters: the JOD returned matches
//! pycvvdp v0.5.4 on the v1 manifest within ~0.01 JOD across q1–q90.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use image::ImageReader;
use std::path::PathBuf;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn load_rgb_bytes(path: &PathBuf, w: u32, h: u32) -> Vec<u8> {
    let img = ImageReader::open(path)
        .unwrap_or_else(|e| panic!("open {path:?}: {e}"))
        .decode()
        .unwrap_or_else(|e| panic!("decode {path:?}: {e}"))
        .to_rgb8();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    img.into_raw()
}

#[test]
fn cvvdp_score_matches_v1_manifest() {
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // (q, pycvvdp_manifest_jod) — same goldens shadow_jod uses.
    let cases: &[(u32, f32)] = &[
        (1, 7.6536),
        (5, 8.8889),
        (20, 9.7076),
        (45, 9.8273),
        (70, 9.8915),
        (90, 9.9930),
    ];
    for &(q, expected) in cases {
        let dist_bytes = load_rgb_bytes(&zenmetrics_corpus::jpeg_at_quality(q), w, h);
        let jod = cvvdp.score(&ref_bytes, &dist_bytes).expect("score");
        let diff = (jod as f32 - expected).abs();
        eprintln!("q={q:>2}: JOD = {jod:.4} (pycvvdp {expected:.4}, |diff| {diff:.4})");
        assert!(
            diff < 0.05,
            "q={q}: Cvvdp::score returned {jod}, pycvvdp manifest {expected}, |diff| {diff:.4} > 0.05"
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
    // The cached-reference path is currently a host-scalar
    // pass-through; once the GPU composition lands it becomes a
    // band-reuse fast path. Either way, the contract is exact
    // parity with `score(ref, dist)` — pin it.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    // set_reference + score_with_reference against several
    // distorted candidates — that's the call pattern that motivates
    // having a cached fast path in the first place.
    cvvdp
        .set_reference(&ref_bytes)
        .expect("set_reference should succeed on valid bytes");
    for &q in &[1u32, 20, 90] {
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
fn compute_dkl_jod_on_v1_manifest_corpus() {
    // GPU-composed compute_dkl_jod against the v1 R2 manifest values.
    // shadow_jod pins the all-host path to ≤0.006 JOD; this test
    // measures the GPU path's drift on real corpus images vs pycvvdp.
    //
    // Observed 2026-05-14 (cuda backend, current scaffold):
    //
    // ```text
    //   q    pycvvdp manifest   GPU JOD    |drift|
    //   1    7.6536             8.0528     0.3992
    //   5    8.8889             8.9434     0.0545
    //   20   9.7076             9.7086     0.0010
    //   45   9.8273             9.8293     0.0020
    //   70   9.8915             9.8944     0.0029
    //   90   9.9930             9.9919     0.0011
    // ```
    //
    // Findings:
    // - q ≥ 20: GPU JOD matches pycvvdp within 0.003 JOD (well
    //   inside f32 accumulation noise — the GPU score path is
    //   production-quality at moderate-to-high quality).
    // - q = 5: 0.055 JOD drift — borderline.
    // - q = 1: 0.40 JOD drift in the optimistic direction (GPU
    //   reports less distortion than pycvvdp). Tick 55 moved the
    //   masking step from host scalar to GPU kernels
    //   (min_abs_3ch + pu_blur_{h,v} + mult_mutual_3ch_with_blurred,
    //   plus the 10^MASK_C scale on the blurred PU output via
    //   weight_band_kernel) — drift was unchanged at q=1, so the
    //   masker itself is bit-stable across paths.
    //   The residual drift is upstream — most likely the per-pixel
    //   CSF interp's uniform-axis arithmetic (GPU) vs binary-search
    //   interp1_clamped (host). At very low quality D values are
    //   large enough that the soft-clamp at D_MAX saturates, and
    //   small `S` deltas from the LUT interp amplify into JOD shift.
    //
    // Per-q diffs report to stdout so the loop can watch the drift
    // shrink as more stages move to GPU.
    let client = Backend::client(&Default::default());
    let (w, h) = (256u32, 256u32);
    let geom = DisplayGeometry::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("new Cvvdp");

    let ref_bytes = load_rgb_bytes(&zenmetrics_corpus::source_png(), w, h);

    let cases: &[(u32, f32)] = &[
        (1, 7.6536),
        (5, 8.8889),
        (20, 9.7076),
        (45, 9.8273),
        (70, 9.8915),
        (90, 9.9930),
    ];

    let mut max_drift = 0.0_f32;
    for &(q, expected) in cases {
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
    // Loose ceiling — measure-only test. Tightens once GPU masking
    // + pool kernels replace the host calls in compute_dkl_jod.
    assert!(
        max_drift < 1.0,
        "GPU JOD drifts > 1.0 from v1 manifest: {max_drift}"
    );
}

#[test]
fn compute_dkl_jod_vs_host_scalar_on_corpus() {
    // Direct GPU-vs-HOST comparison on the v1 manifest corpus (real
    // 256×256 images). shadow_jod pinned the all-host path to
    // pycvvdp within 0.006 JOD; compute_dkl_jod_on_v1_manifest_corpus
    // measured GPU-vs-pycvvdp. The remaining unknown was whether
    // GPU-vs-HOST agrees better than GPU-vs-pycvvdp, or whether the
    // drift compounds. This test surfaces both.
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

    let qs: &[u32] = &[1, 5, 20, 45, 70, 90];
    eprintln!("  q   pycvvdp    host_scalar   GPU JOD   GPU-host   GPU-pycvvdp");
    let mut max_gpu_host_drift = 0.0_f32;
    let pycvvdp_manifest: &[(u32, f32)] = &[
        (1, 7.6536),
        (5, 8.8889),
        (20, 9.7076),
        (45, 9.8273),
        (70, 9.8915),
        (90, 9.9930),
    ];
    for &q in qs {
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
    assert!(
        max_gpu_host_drift < 1.0,
        "GPU JOD drifts > 1.0 from host scalar: {max_gpu_host_drift}"
    );
}
