//! Opaque-API unit tests for `butteraugli-gpu`.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use butteraugli_gpu::{Backend, Butteraugli, ButteraugliOpaque, ButteraugliParams};
use cubecl::Runtime;

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
    let mut typed = Butteraugli::<BackendT>::new_multires(client, w, h);
    let typed_score = typed.compute(&ref_buf, &dis_buf).expect("typed compute");

    let mut opaque =
        ButteraugliOpaque::new(BACKEND_E, w, h, ButteraugliParams::default()).expect("opaque new");
    let opaque_score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque compute_srgb_u8");

    let rel = (opaque_score.value - typed_score.score as f64).abs()
        / (typed_score.score as f64).abs().max(1e-12);
    assert!(
        rel < 1e-5,
        "opaque {} vs typed {} differ by rel {}",
        opaque_score.value,
        typed_score.score,
        rel
    );
    assert_eq!(opaque_score.metric_name, "butter");
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

    // Padding aligned to bpp=3.
    let pad = 15usize;
    let stride = row_bytes + pad;
    assert_eq!(stride % 3, 0, "stride must be pixel-aligned");

    let mut padded_ref = vec![0xAA_u8; stride * h as usize];
    let mut padded_dis = vec![0xBB_u8; stride * h as usize];
    for y in 0..h as usize {
        padded_ref[y * stride..y * stride + row_bytes]
            .copy_from_slice(&tight[y * row_bytes..(y + 1) * row_bytes]);
        padded_dis[y * stride..y * stride + row_bytes]
            .copy_from_slice(&dist_tight[y * row_bytes..(y + 1) * row_bytes]);
    }

    let mut opaque =
        ButteraugliOpaque::new(BACKEND_E, w, h, ButteraugliParams::default()).expect("opaque new");

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
