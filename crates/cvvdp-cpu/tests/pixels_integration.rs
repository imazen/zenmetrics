//! `zenpixels::PixelSlice` integration smoke test.
//!
//! Only built under `--features pixels`. Verifies that scoring
//! through `Cvvdp::score_pixels` produces the same result as the
//! raw-bytes `Cvvdp::score` entry point.

#![cfg(feature = "pixels")]

use cvvdp_cpu::{Cvvdp, CvvdpParams};
use zenpixels::{PixelDescriptor, PixelSlice};

fn make_bytes(w: usize, h: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut out = vec![0u8; w * h * 3];
    for v in out.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (s >> 16) as u8;
    }
    out
}

#[test]
fn score_pixels_matches_score_bytes() {
    let w = 64;
    let h = 64;
    let r = make_bytes(w, h, 17);
    let d = make_bytes(w, h, 18);

    let mut cv = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let jod_bytes = cv.score(&r, &d).unwrap();

    let row_bytes = (w as usize) * PixelDescriptor::RGB8_SRGB.bytes_per_pixel();
    let r_slice = PixelSlice::new(&r, w as u32, h as u32, row_bytes, PixelDescriptor::RGB8_SRGB)
        .expect("ref slice");
    let d_slice = PixelSlice::new(&d, w as u32, h as u32, row_bytes, PixelDescriptor::RGB8_SRGB)
        .expect("dist slice");
    let mut cv2 = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let jod_pixels = cv2.score_pixels(r_slice, d_slice).unwrap();

    let diff = (jod_bytes - jod_pixels).abs();
    assert!(
        diff < 1e-5,
        "byte JOD {jod_bytes} vs pixels JOD {jod_pixels} diff {diff}"
    );
}
