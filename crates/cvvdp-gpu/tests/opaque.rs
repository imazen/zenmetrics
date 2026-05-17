//! Opaque-API unit tests for `cvvdp-gpu`.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use cvvdp_gpu::{Backend, Cvvdp, CvvdpOpaque, CvvdpParams};

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
    let mut typed =
        Cvvdp::<BackendT>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("typed new");
    let typed_score = typed.score(&ref_buf, &dis_buf).expect("typed score");

    let mut opaque =
        CvvdpOpaque::new(BACKEND_E, w, h, CvvdpParams::PLACEHOLDER).expect("opaque new");
    let opaque_score = opaque
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("opaque compute_srgb_u8");

    let rel = (opaque_score.value - typed_score).abs() / typed_score.abs().max(1e-12);
    assert!(
        rel < 1e-5,
        "opaque {} vs typed {} differ by rel {}",
        opaque_score.value,
        typed_score,
        rel
    );
    assert_eq!(opaque_score.metric_name, "cvvdp");
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

    let mut opaque =
        CvvdpOpaque::new(BACKEND_E, w, h, CvvdpParams::PLACEHOLDER).expect("opaque new");

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
