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
    let mut cvvdp = Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER)
        .expect("new Cvvdp");

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
