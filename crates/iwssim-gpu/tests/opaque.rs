//! Opaque-API unit tests for `iwssim-gpu`.
//!
//! IW-SSIM requires `min(W,H) >= 176` (paper's 5-level pyramid +
//! 11×11 valid blur). Tests use 256×256.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use iwssim_gpu::{Backend, Iwssim, IwssimOpaque, IwssimParams};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_add(seed.wrapping_mul(3))) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

#[test]
fn opaque_srgb_u8_matches_typed() {
    let w = 256_u32;
    let h = 256_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let client = BackendT::client(&Default::default());
    let mut typed = Iwssim::<BackendT>::new(client, w, h).expect("typed new");
    let typed_score = typed
        .compute_rgb(&ref_buf, &dis_buf)
        .expect("typed compute_rgb");

    let mut opaque = IwssimOpaque::new(BACKEND_E, w, h, IwssimParams::DEFAULT).expect("opaque new");
    let opaque_score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque compute_srgb_u8");

    let rel = (opaque_score.value - typed_score.score).abs() / typed_score.score.abs().max(1e-12);
    eprintln!(
        "opaque_srgb_u8_matches_typed: opaque={} typed={} rel={:.3e}",
        opaque_score.value, typed_score.score, rel
    );
    // Post-2026-05-22 cov_finalize f64 promotion: opaque vs typed measured
    // bit-identical (rel=0.0) on CUDA. The 1e-7 gate leaves ~1 ULP@1.0
    // headroom for wgpu / cubecl-cpu where the reduction order may
    // differ but the math is the same.
    assert!(
        rel < 1e-7,
        "opaque {} vs typed {} differ by rel {}",
        opaque_score.value,
        typed_score.score,
        rel
    );
    assert_eq!(opaque_score.metric_name, "iwssim");
    assert_eq!(opaque_score.metric_version, env!("CARGO_PKG_VERSION"));
}

#[cfg(feature = "pixels")]
#[test]
fn opaque_pixels_handles_stride() {
    use zenpixels::{PixelDescriptor, PixelSlice};

    let w = 256_u32;
    let h = 256_u32;
    let tight = make_image(w, h, 11);
    let dist_tight = make_image(w, h, 23);

    let descriptor = PixelDescriptor::RGB8_SRGB;
    let row_bytes = w as usize * 3;
    let pad = 15usize;
    let stride = row_bytes + pad;
    assert_eq!(stride % 3, 0);

    let mut padded_ref = vec![0xAA_u8; stride * h as usize];
    let mut padded_dis = vec![0xBB_u8; stride * h as usize];
    for y in 0..h as usize {
        padded_ref[y * stride..y * stride + row_bytes]
            .copy_from_slice(&tight[y * row_bytes..(y + 1) * row_bytes]);
        padded_dis[y * stride..y * stride + row_bytes]
            .copy_from_slice(&dist_tight[y * row_bytes..(y + 1) * row_bytes]);
    }

    let mut opaque = IwssimOpaque::new(BACKEND_E, w, h, IwssimParams::DEFAULT).expect("opaque new");

    let r_tight = PixelSlice::new(&tight, w, h, row_bytes, descriptor).expect("tight ref");
    let d_tight = PixelSlice::new(&dist_tight, w, h, row_bytes, descriptor).expect("tight dist");
    let tight_score = opaque
        .compute_pixels(r_tight, d_tight)
        .expect("tight compute_pixels");

    let r_padded = PixelSlice::new(&padded_ref, w, h, stride, descriptor).expect("padded ref");
    let d_padded = PixelSlice::new(&padded_dis, w, h, stride, descriptor).expect("padded dist");
    let padded_score = opaque
        .compute_pixels(r_padded, d_padded)
        .expect("padded compute_pixels");

    let rel = (tight_score.value - padded_score.value).abs() / tight_score.value.abs().max(1e-12);
    eprintln!(
        "opaque_pixels_stride: tight={} padded={} rel={:.3e}",
        tight_score.value, padded_score.value, rel
    );
    // Two calls on the same instance with the same content (tight vs
    // strided buffer that the PixelSlice reads with a per-row gather
    // into the same internal tight buffer) should produce bit-
    // identical scores. Post-2026-05-22 cov_finalize f64 promotion:
    // 1e-7 gate (was 5e-5 in the pre-f64 era).
    assert!(
        rel < 1e-7,
        "strided {} vs tight {} differ by rel {}",
        padded_score.value,
        tight_score.value,
        rel
    );
}

/// Identity pair must score 1.0 (within f32 noise). The per-scale
/// information-weighted ratio Σ(cs·iw)/Σ(iw) is 0/0 in the degenerate
/// case — pipeline-level handling collapses both `Σ(iw) == 0` and a
/// non-finite numerator to the perfect-score value (1.0) so the final
/// Π |wmcs_j|^β_j product lands on 1.0 instead of 0 or NaN.
#[test]
fn compute_on_identical_returns_1() {
    let w = 256_u32;
    let h = 256_u32;
    // Use a spatially-structured sRGB image (not a flat constant) so
    // the test exercises the real pyramid path; the "identical"
    // contract has to hold for any input, not just trivial ones.
    let img = make_image(w, h, 42);

    let mut opaque = IwssimOpaque::new(BACKEND_E, w, h, IwssimParams::DEFAULT).expect("opaque new");
    let score = opaque
        .compute_srgb_u8(&img, &img)
        .expect("opaque compute_srgb_u8 identical");

    assert!(
        score.value.is_finite(),
        "identical pair must produce a finite score, got {}",
        score.value,
    );
    // Tolerance 1e-7 ≈ one f32 ULP at 1.0. CUDA's atomic-stable reduce
    // gets ≤1e-9 on this input, but wgpu's Vulkan compute pipeline
    // shifts the cs/iw reduction sum by ~2.5e-8 vs CUDA after the
    // per-thread-partials cov refactor — well under f32 epsilon, well
    // above any tolerance that would still surface the original
    // NaN-on-identical bug (which scored 0, not ~1).
    assert!(
        (score.value - 1.0).abs() < 1e-7,
        "identical pair must score 1.0 within 1e-7, got {}",
        score.value,
    );
    assert_eq!(score.metric_name, "iwssim");
}
