//! Opaque-API unit tests for `ssim2-gpu`.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use ssim2_gpu::{Backend, Ssim2, Ssim2Mode, Ssim2Opaque, Ssim2Params};

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
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);

    let client = BackendT::client(&Default::default());
    let mut typed = Ssim2::<BackendT>::new(client, w, h).expect("typed new");
    let typed_score = typed
        .compute_with_mode(Ssim2Mode::Faster, &ref_buf, &dis_buf)
        .expect("typed compute");

    let mut opaque = Ssim2Opaque::new(BACKEND_E, w, h, Ssim2Params::DEFAULT).expect("opaque new");
    let opaque_score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque compute_srgb_u8");

    let rel = (opaque_score.value - typed_score.score).abs() / typed_score.score.abs().max(1e-12);
    assert!(
        rel < 1e-5,
        "opaque {} vs typed {} differ by rel {}",
        opaque_score.value,
        typed_score.score,
        rel
    );
    assert_eq!(opaque_score.metric_name, "ssim2");
    assert_eq!(opaque_score.metric_version, env!("CARGO_PKG_VERSION"));
}

#[cfg(feature = "pixels")]
#[test]
fn opaque_pixels_handles_stride() {
    use zenpixels::{PixelDescriptor, PixelSlice};

    let w = 64_u32;
    let h = 64_u32;
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

    let mut opaque = Ssim2Opaque::new(BACKEND_E, w, h, Ssim2Params::DEFAULT).expect("opaque new");

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
    assert!(
        rel < 1e-5,
        "strided {} vs tight {} differ by rel {}",
        padded_score.value,
        tight_score.value,
        rel
    );
}

/// Synthetic absolute-luminance HDR field (cd/m²): smooth gradients
/// spanning ~0.1–1000 nits with per-channel phase offsets, so every
/// pyramid scale sees structure across the PU21 operating range.
fn make_hdr_nits(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = (x + seed) as f32 / w as f32;
            let fy = (y + seed * 3) as f32 / h as f32;
            // log-spaced luminance ramp: 0.1 → 1000 cd/m²
            let l = 0.1f32 * 10f32.powf(4.0 * (0.5 * (fx + fy)).clamp(0.0, 1.0));
            out.extend_from_slice(&[l, l * 0.8 + 1.0, l * 0.6 + 2.0]);
        }
    }
    out
}

/// Opaque PU21-integrated HDR entry: identical 64×64 pair scores ~100,
/// a distorted pair scores finite and lower — proof the routed
/// `Ssim2Opaque::compute_linear_nits` path works end-to-end on-device.
#[test]
fn opaque_compute_linear_nits_identical_scores_100() {
    let (w, h) = (64_u32, 64_u32);
    let a = make_hdr_nits(w, h, 0);

    let mut opaque = Ssim2Opaque::new(BACKEND_E, w, h, Ssim2Params::DEFAULT).expect("opaque new");
    let same = opaque
        .compute_linear_nits(&a, &a)
        .expect("identical compute_linear_nits");
    assert!(
        same.value >= 99.0 && same.value <= 100.05,
        "identical HDR pair: score={}, expected [99, 100.05]",
        same.value
    );
    assert_eq!(same.metric_name, "ssim2");

    // Luminance-shifted distorted copy: finite score, clearly below identical.
    let b: Vec<f32> = a.iter().map(|&v| v * 1.35 + 0.5).collect();
    let diff = opaque
        .compute_linear_nits(&a, &b)
        .expect("distorted compute_linear_nits");
    assert!(
        diff.value.is_finite() && diff.value < same.value,
        "distorted HDR pair: score={} (identical={})",
        diff.value,
        same.value
    );
}

/// Sub-8px opaque instances reflect-pad u8 inputs, but the f32 PU21
/// ingress has no pad path — the opaque must reject, not corrupt.
#[test]
fn opaque_compute_linear_nits_rejects_sub_min() {
    let (w, h) = (4_u32, 4_u32);
    let a = make_hdr_nits(w, h, 0);
    let mut opaque = Ssim2Opaque::new(BACKEND_E, w, h, Ssim2Params::DEFAULT).expect("opaque new");
    let err = opaque.compute_linear_nits(&a, &a);
    assert!(
        err.is_err(),
        "sub-8px compute_linear_nits must error, got {err:?}"
    );
}
