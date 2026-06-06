//! The unified descriptor-driven entry `HdrScorer::compute_pixels_multi` — proof
//! that SDR and HDR are one call, and that it preserves BOTH validated baselines:
//!   - an `RGB8_SRGB` slice → bit-identical to the native `compute_srgb_u8`,
//!   - an HDR linear slice → identical to the `compute_multi(nits)` faithful path.
//! CUDA-gated; NO GRACEFUL SKIPS.
#![cfg(all(
    feature = "cuda",
    feature = "hdr",
    feature = "pixels",
    feature = "butter",
    feature = "ssim2"
))]

use zenmetrics_api::hdr::{HDR_PEAK_NITS, HdrScorer};
use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams};
use zenpixels::{PixelDescriptor, PixelSlice};

fn gradient_u8(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push(((x.wrapping_add(seed)) & 0xff) as u8);
            out.push(((y.wrapping_add(seed * 3)) & 0xff) as u8);
            out.push(((x ^ y ^ seed) & 0xff) as u8);
        }
    }
    out
}

fn srgb_eotf(c: u8) -> f32 {
    let v = c as f32 / 255.0;
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn f32_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 4);
    for &v in vals {
        out.extend_from_slice(&v.to_ne_bytes());
    }
    out
}

/// An sRGB8 `PixelSlice` through `compute_pixels_multi` is bit-identical to the
/// native `compute_srgb_u8` (SDR scores preserved — the descriptor takes the
/// native path, no conversion).
#[test]
fn sdr_srgb8_slice_matches_native_compute_srgb_u8() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 9);
    let row = (w * 3) as usize;

    // Native baseline — whole-image so it's comparable to the scorer's Full butter.
    let mut native = Metric::new_with_memory_mode(
        MetricKind::Butter,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Butter),
        MemoryMode::Full,
    )
    .expect("native");
    let native_score = native
        .compute_srgb_u8(&ref_u8, &dis_u8)
        .expect("native srgb")
        .value;

    // Unified entry, sRGB8 descriptor. Peak matches butter's default
    // intensity_target (80) so the native path's params line up.
    let mut hs = HdrScorer::new(MetricKind::Butter, Backend::Cuda, w, h, 80.0).expect("scorer");
    let unified = hs
        .compute_pixels_multi(
            PixelSlice::new(&ref_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
            PixelSlice::new(&dis_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
        )
        .expect("unified srgb8")
        .primary();

    let rel = (unified - native_score).abs() / native_score.abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "sRGB8 slice via compute_pixels_multi ({unified}) must match native compute_srgb_u8 ({native_score}); rel {rel}"
    );
}

/// An HDR linear `PixelSlice` (display-relative [0,1]) through
/// `compute_pixels_multi` is identical to the faithful `compute_multi(nits)`
/// path — the descriptor carries the encoding, same score.
#[test]
fn hdr_linear_slice_matches_nits_path() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 9);

    // Display-relative [0,1] linear (sRGB-decoded), and the same content as nits.
    let lin = |buf: &[u8]| -> Vec<f32> { buf.iter().map(|&c| srgb_eotf(c)).collect() };
    let ref_lin = lin(&ref_u8);
    let dis_lin = lin(&dis_u8);
    let ref_nits: Vec<f32> = ref_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let dis_nits: Vec<f32> = dis_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let ref_bytes = f32_bytes(&ref_lin);
    let dis_bytes = f32_bytes(&dis_lin);
    let row_f32 = (w * 3 * 4) as usize;

    let mut hs =
        HdrScorer::new(MetricKind::Butter, Backend::Cuda, w, h, HDR_PEAK_NITS).expect("scorer");

    // (a) the existing nits-array faithful path.
    let via_nits = hs.compute_multi(&ref_nits, &dis_nits).expect("nits");
    // (b) the unified entry with a linear PixelSlice descriptor.
    let via_pixels = hs
        .compute_pixels_multi(
            PixelSlice::new(&ref_bytes, w, h, row_f32, PixelDescriptor::RGBF32_LINEAR).unwrap(),
            PixelSlice::new(&dis_bytes, w, h, row_f32, PixelDescriptor::RGBF32_LINEAR).unwrap(),
        )
        .expect("pixels");

    assert_eq!(via_nits.scores.len(), 2);
    assert_eq!(via_pixels.scores.len(), 2);
    let rel =
        (via_nits.primary() - via_pixels.primary()).abs() / via_nits.primary().abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "HDR linear slice ({}) must match nits path ({}); rel {rel}",
        via_pixels.primary(),
        via_nits.primary(),
    );
    let prel = (via_nits.get("pnorm_3").unwrap() - via_pixels.get("pnorm_3").unwrap()).abs()
        / via_nits.get("pnorm_3").unwrap().abs().max(1e-6);
    assert!(prel < 1e-4, "pnorm_3 mismatch (rel {prel})");
}

/// SSIM-family path: an sRGB8 slice → native `compute_srgb_u8` (SDR preserved);
/// an HDR linear slice → the pu-rescale u8 feeding == `compute_multi(nits)`.
#[test]
fn ssim2_srgb8_native_and_hdr_pu_rescale() {
    let (w, h) = (64_u32, 64_u32);
    let ref_u8 = gradient_u8(w, h, 0);
    let dis_u8 = gradient_u8(w, h, 9);
    let row = (w * 3) as usize;

    // SDR: sRGB8 slice == native compute_srgb_u8.
    let mut native = Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        w,
        h,
        MetricParams::default_for(MetricKind::Ssim2),
    )
    .expect("native ssim2");
    let native_sdr = native
        .compute_srgb_u8(&ref_u8, &dis_u8)
        .expect("native srgb")
        .value;

    let mut hs =
        HdrScorer::new(MetricKind::Ssim2, Backend::Cuda, w, h, HDR_PEAK_NITS).expect("scorer");
    let unified_sdr = hs
        .compute_pixels_multi(
            PixelSlice::new(&ref_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
            PixelSlice::new(&dis_u8, w, h, row, PixelDescriptor::RGB8_SRGB).unwrap(),
        )
        .expect("unified srgb8")
        .primary();
    let rel = (unified_sdr - native_sdr).abs() / native_sdr.abs().max(1e-6);
    assert!(
        rel < 1e-4,
        "ssim2 sRGB8 slice {unified_sdr} != native {native_sdr} (rel {rel})"
    );

    // HDR: linear slice (display-relative) == nits path (both pu-rescale).
    let ref_lin: Vec<f32> = ref_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let dis_lin: Vec<f32> = dis_u8.iter().map(|&c| srgb_eotf(c)).collect();
    let ref_nits: Vec<f32> = ref_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let dis_nits: Vec<f32> = dis_lin.iter().map(|&v| v * HDR_PEAK_NITS).collect();
    let row_f32 = (w * 3 * 4) as usize;
    let via_nits = hs
        .compute_multi(&ref_nits, &dis_nits)
        .expect("nits")
        .primary();
    let via_pixels = hs
        .compute_pixels_multi(
            PixelSlice::new(
                &f32_bytes(&ref_lin),
                w,
                h,
                row_f32,
                PixelDescriptor::RGBF32_LINEAR,
            )
            .unwrap(),
            PixelSlice::new(
                &f32_bytes(&dis_lin),
                w,
                h,
                row_f32,
                PixelDescriptor::RGBF32_LINEAR,
            )
            .unwrap(),
        )
        .expect("pixels")
        .primary();
    let hrel = (via_nits - via_pixels).abs() / via_nits.abs().max(1e-6);
    assert!(
        hrel < 1e-4,
        "ssim2 HDR linear {via_pixels} != nits {via_nits} (rel {hrel})"
    );
}
