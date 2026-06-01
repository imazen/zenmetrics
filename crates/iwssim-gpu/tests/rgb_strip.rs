//! RGB-strip tests for `iwssim-gpu`.
//!
//! Adds [`Iwssim::compute_rgb_stripped`] (the symmetric counterpart of
//! [`Iwssim::compute_rgb`]). The implementation does host-side BT.601
//! rgb→gray, then routes through [`Iwssim::compute_gray_stripped`].
//! These tests pin that routing: an RGB input pre-converted to gray
//! via the same BT.601 formula must produce the same score as routing
//! the raw RGB through `compute_rgb_stripped`.
//!
//! The on-device `rgb_u32_to_gray_kernel` and the host-side
//! `rgb_u8_to_gray_bt601` helper share the same coefficients
//! `(0.2989, 0.5870, 0.1140)` plus the same `(y + 0.5).floor()`
//! rounding rule, so the two paths produce identical gray planes
//! to integer precision.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use iwssim_gpu::{Error, Iwssim};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

/// RGB-strip → gray-strip parity: the host-side BT.601 conversion is
/// deterministic, so two paths through identical pixels must agree
/// to the f32 precision floor. Bound at 5e-5 rel — tighter than
/// cross-tile-size, because there's no reduction-order shift.
const RGB_VS_GRAY_REL: f64 = 5e-5;

/// Host-side BT.601 rgb→gray with the same rounding rule as the
/// kernel + `rgb_u8_to_gray_bt601`. Pulled into tests so we can
/// pre-render the gray plane and route it through
/// `compute_gray_stripped` for direct comparison with
/// `compute_rgb_stripped`.
fn rgb_to_gray_bt601(rgb: &[u8]) -> Vec<f32> {
    let n = rgb.len() / 3;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let r = rgb[i * 3] as f32;
        let g = rgb[i * 3 + 1] as f32;
        let b = rgb[i * 3 + 2] as f32;
        let y = 0.2989_f32 * r + 0.5870_f32 * g + 0.1140_f32 * b;
        out.push((y + 0.5_f32).floor());
    }
    out
}

/// Make a non-trivial color RGB image with distinct per-channel
/// patterns so the BT.601 weights actually have something to mix.
fn make_color_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(3).wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_mul(5).wrapping_add(seed.wrapping_mul(7))) & 0xff) as u8;
            let b = ((x
                .wrapping_add(y)
                .wrapping_mul(11)
                .wrapping_add(seed.wrapping_mul(13)))
                & 0xff) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

#[test]
fn rgb_strip_matches_gray_strip_of_prerendered_gray_256() {
    // Convert RGB → gray host-side, run gray-strip. Compare to
    // running compute_rgb_stripped on the same RGB. Scores must agree.
    let w = 256_u32;
    let h = 256_u32;
    let r_rgb = make_color_rgb(w, h, 0);
    let d_rgb = make_color_rgb(w, h, 11);
    let r_gray = rgb_to_gray_bt601(&r_rgb);
    let d_gray = rgb_to_gray_bt601(&d_rgb);

    let client = BackendT::client(&Default::default());
    let mut s_rgb = Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip");
    let v_rgb = s_rgb
        .compute_rgb_stripped(&r_rgb, &d_rgb)
        .expect("rgb stripped")
        .score;

    let mut s_gray = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    let v_gray = s_gray
        .compute_gray_stripped(&r_gray, &d_gray)
        .expect("gray stripped")
        .score;

    let rel = ((v_rgb - v_gray) / v_gray).abs();
    assert!(
        rel < RGB_VS_GRAY_REL,
        "rgb_strip {v_rgb} vs gray_strip {v_gray} rel={rel}"
    );
}

#[test]
fn rgb_strip_matches_gray_strip_512_body_256() {
    // Multi-strip 512² body=256 (2 strips). Same content, different
    // path → same score.
    let w = 512_u32;
    let h = 512_u32;
    let r_rgb = make_color_rgb(w, h, 19);
    let d_rgb = make_color_rgb(w, h, 23);
    let r_gray = rgb_to_gray_bt601(&r_rgb);
    let d_gray = rgb_to_gray_bt601(&d_rgb);

    let client = BackendT::client(&Default::default());
    let mut s_rgb = Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip");
    let v_rgb = s_rgb
        .compute_rgb_stripped(&r_rgb, &d_rgb)
        .expect("rgb stripped")
        .score;
    let mut s_gray = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    let v_gray = s_gray
        .compute_gray_stripped(&r_gray, &d_gray)
        .expect("gray stripped")
        .score;
    let rel = ((v_rgb - v_gray) / v_gray).abs();
    assert!(
        rel < RGB_VS_GRAY_REL,
        "512² rgb_strip {v_rgb} vs gray_strip {v_gray} rel={rel}"
    );
}

#[test]
fn rgb_strip_matches_gray_strip_1024_body_256() {
    // Production-ish 1024² body=256 (4 strips).
    let w = 1024_u32;
    let h = 1024_u32;
    let r_rgb = make_color_rgb(w, h, 31);
    let d_rgb = make_color_rgb(w, h, 37);
    let r_gray = rgb_to_gray_bt601(&r_rgb);
    let d_gray = rgb_to_gray_bt601(&d_rgb);

    let client = BackendT::client(&Default::default());
    let mut s_rgb = Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip");
    let v_rgb = s_rgb
        .compute_rgb_stripped(&r_rgb, &d_rgb)
        .expect("rgb stripped")
        .score;
    let mut s_gray = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    let v_gray = s_gray
        .compute_gray_stripped(&r_gray, &d_gray)
        .expect("gray stripped")
        .score;
    let rel = ((v_rgb - v_gray) / v_gray).abs();
    assert!(
        rel < RGB_VS_GRAY_REL,
        "1024² rgb_strip {v_rgb} vs gray_strip {v_gray} rel={rel}"
    );
}

#[test]
fn rgb_strip_errors_on_whole_image_instance() {
    // Calling compute_rgb_stripped on a whole-image instance must
    // NotStripMode.
    let w = 256_u32;
    let h = 256_u32;
    let r = make_color_rgb(w, h, 0);
    let d = make_color_rgb(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut whole = Iwssim::<BackendT>::new(client, w, h).expect("whole new");
    let err = whole.compute_rgb_stripped(&r, &d).expect_err("must error");
    assert!(matches!(err, Error::NotStripMode));
}

#[test]
fn rgb_strip_errors_on_dim_mismatch() {
    let w = 256_u32;
    let h = 256_u32;
    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    let too_small = vec![0_u8; ((w - 1) * h * 3) as usize];
    let dummy = vec![0_u8; (w * h * 3) as usize];
    let err = strip
        .compute_rgb_stripped(&too_small, &dummy)
        .expect_err("must error on small ref");
    assert!(matches!(err, Error::DimensionMismatch { .. }));
    let err = strip
        .compute_rgb_stripped(&dummy, &too_small)
        .expect_err("must error on small dis");
    assert!(matches!(err, Error::DimensionMismatch { .. }));
}

#[test]
fn rgb_strip_self_identity() {
    // RGB self-identity: same RGB input ref and dis must score 1.0.
    let w = 512_u32;
    let h = 512_u32;
    let r_rgb = make_color_rgb(w, h, 41);
    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    let s = strip
        .compute_rgb_stripped(&r_rgb, &r_rgb)
        .expect("rgb stripped")
        .score;
    assert!((s - 1.0).abs() < 1e-5, "rgb strip self-identity {s} ≠ 1");
}
