//! Smoke test for the `compute_pixels` path. Builds a sRGB-RGB8
//! `PixelSlice` and feeds it through three different metrics through
//! the umbrella. Requires `pixels` + `cuda` features.

#![cfg(all(feature = "pixels", feature = "cuda"))]

use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};
use zenpixels::{PixelDescriptor, PixelSlice};

const W: u32 = 256;
const H: u32 = 256;

fn make_image() -> Vec<u8> {
    let n = (W as usize) * (H as usize) * 3;
    let mut v = vec![0u8; n];
    for (i, b) in v.iter_mut().enumerate() {
        // Distinct pattern per channel so the metric isn't computing
        // on a constant image (would short-circuit reduction paths).
        *b = ((i * 3539) & 0xFF) as u8;
    }
    v
}

fn run_pixels(kind: MetricKind) -> zenmetrics_api::Score {
    let r_bytes = make_image();
    let d_bytes = make_image();
    let descriptor = PixelDescriptor::RGB8_SRGB;
    let row_bytes = (W as usize) * 3;
    let r_slice =
        PixelSlice::new(&r_bytes, W, H, row_bytes, descriptor).expect("ref slice construction");
    let d_slice =
        PixelSlice::new(&d_bytes, W, H, row_bytes, descriptor).expect("dist slice construction");

    let params = MetricParams::default_for(kind);
    let mut m = Metric::new(kind, Backend::Cuda, W, H, params)
        .unwrap_or_else(|e| panic!("Metric::new({kind:?}) failed: {e}"));
    m.compute_pixels(r_slice, d_slice)
        .unwrap_or_else(|e| panic!("compute_pixels({kind:?}) failed: {e}"))
}

#[cfg(feature = "cvvdp")]
#[test]
fn pixels_cvvdp() {
    let s = run_pixels(MetricKind::Cvvdp);
    assert_eq!(s.metric_name, "cvvdp");
    // Identical RGB8 inputs → JOD ~10.
    assert!(
        (s.value - 10.0).abs() < 1e-3,
        "cvvdp identity must be ~10 from pixels path, got {}",
        s.value
    );
}

#[cfg(feature = "ssim2")]
#[test]
fn pixels_ssim2() {
    let s = run_pixels(MetricKind::Ssim2);
    assert_eq!(s.metric_name, "ssim2");
    assert!(
        (s.value - 100.0).abs() < 1e-1,
        "ssim2 identity must be ~100 from pixels path, got {}",
        s.value
    );
}

#[cfg(feature = "dssim")]
#[test]
fn pixels_dssim() {
    let s = run_pixels(MetricKind::Dssim);
    assert_eq!(s.metric_name, "dssim");
    assert!(
        s.value < 1e-4,
        "dssim identity must be ~0 from pixels path, got {}",
        s.value
    );
}
