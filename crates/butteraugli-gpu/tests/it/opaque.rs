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

/// PROTOTYPE proof: the HDR/SDR API split can collapse into one
/// descriptor-driven entry. The SAME image expressed two ways — as an
/// `RGB8_SRGB` slice and as an `RGBF32_LINEAR` slice (its sRGB-decoded linear
/// light) — scored through the SINGLE `compute_pixels_display` yields the SAME
/// butteraugli score. The descriptor (not a separate HDR API) carries the
/// encoding; zenpixels-convert does the transfer. HDR is then just the same
/// call with a PQ/linear descriptor + a 1000-nit peak.
#[cfg(all(feature = "pixels", feature = "internals"))]
#[test]
fn unified_compute_pixels_display_one_image_two_descriptors() {
    use zenpixels::{PixelDescriptor, PixelSlice};

    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = make_image(w, h, 0);
    let dis_u8 = make_image(w, h, 7);
    let row_u8 = (w * 3) as usize;

    // The IEC sRGB EOTF — decode the same u8 content to relative linear [0,1].
    let srgb_eotf = |c: u8| -> f32 {
        let v = c as f32 / 255.0;
        if v <= 0.040_449_936 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    };
    let to_lin_bytes = |buf: &[u8]| -> Vec<u8> {
        let mut out = Vec::with_capacity(buf.len() * 4);
        for &c in buf {
            out.extend_from_slice(&srgb_eotf(c).to_ne_bytes());
        }
        out
    };
    let ref_lin = to_lin_bytes(&ref_u8);
    let dis_lin = to_lin_bytes(&dis_u8);
    let row_f32 = (w * 3 * 4) as usize;

    // Whole-image instance (the linear-planes path is Full-only).
    let mut opaque = ButteraugliOpaque::new_with_memory_mode(
        BACKEND_E,
        w,
        h,
        ButteraugliParams::default(),
        butteraugli_gpu::MemoryMode::Full,
    )
    .expect("opaque Full");

    let peak = 80.0_f32;
    let (s_sdr, p_sdr) = opaque
        .compute_pixels_display(
            PixelSlice::new(&ref_u8, w, h, row_u8, PixelDescriptor::RGB8_SRGB).unwrap(),
            PixelSlice::new(&dis_u8, w, h, row_u8, PixelDescriptor::RGB8_SRGB).unwrap(),
            peak,
        )
        .expect("sdr descriptor");
    let (s_lin, p_lin) = opaque
        .compute_pixels_display(
            PixelSlice::new(&ref_lin, w, h, row_f32, PixelDescriptor::RGBF32_LINEAR).unwrap(),
            PixelSlice::new(&dis_lin, w, h, row_f32, PixelDescriptor::RGBF32_LINEAR).unwrap(),
            peak,
        )
        .expect("linear descriptor");

    assert!(s_sdr.value > 0.0, "non-trivial distortion expected");
    let rel = (s_sdr.value - s_lin.value).abs() / s_sdr.value.abs().max(1e-6);
    assert!(
        rel < 2e-3,
        "ONE method, two descriptors, same image must agree: sRGB8 {} vs linear {} (rel {rel})",
        s_sdr.value,
        s_lin.value,
    );
    let prel = (p_sdr - p_lin).abs() / p_sdr.abs().max(1e-6);
    assert!(
        prel < 2e-3,
        "pnorm_3 mismatch: {p_sdr} vs {p_lin} (rel {prel})"
    );
}
