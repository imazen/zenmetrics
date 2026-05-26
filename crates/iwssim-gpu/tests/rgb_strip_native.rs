//! Parity lock for the native-RGB strip path
//! ([`Iwssim::compute_rgb_with_reference_stripped_native`]).
//!
//! The native path skips the host-side BT.601 rgb→gray conversion and
//! instead packs each strip's sRGB rows into pinned packed-u32 and
//! launches the existing `rgb_u32_to_gray_kernel` to produce strip-
//! sized gray-f32 on the device. The math is identical to the host
//! path by construction (same BT.601 coefficients, same half-up
//! rounding) so scores must match the host-converted variant.
//!
//! Motivation (measurement): `benchmarks/iwssim_native_rgb_perf_2026-05-26.csv`
//! shows host conversion consumes 35–41% of per-call wall time at
//! 1024² through 4096². See `examples/native_rgb_perf_probe.rs`.

#![cfg(all(feature = "cubecl-types", any(feature = "cuda", feature = "wgpu")))]

use cubecl::Runtime;
use iwssim_gpu::{Error, Iwssim};

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;

/// Native and host paths produce numerically identical gray planes
/// (same BT.601 formula + half-up rounding), so scores must agree to
/// the f32 precision floor. Bound chosen to match `rgb_strip.rs`'s
/// 5e-5 relative tolerance.
const NATIVE_VS_HOST_REL: f64 = 5e-5;

fn make_color_rgb(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_mul(3).wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_mul(5).wrapping_add(seed.wrapping_mul(7))) & 0xff) as u8;
            let b = ((x.wrapping_add(y).wrapping_mul(11).wrapping_add(seed.wrapping_mul(13)))
                & 0xff) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

#[test]
fn native_strip_matches_host_strip_256() {
    let w = 256_u32;
    let h = 256_u32;
    let r_rgb = make_color_rgb(w, h, 0);
    let d_rgb = make_color_rgb(w, h, 11);

    let client = BackendT::client(&Default::default());

    let mut s_host =
        Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip host");
    s_host.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref host");
    let v_host = s_host
        .compute_rgb_with_reference_stripped(&d_rgb)
        .expect("compute host")
        .score;

    let mut s_native =
        Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip native");
    s_native.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref native");
    let v_native = s_native
        .compute_rgb_with_reference_stripped_native(&d_rgb)
        .expect("compute native")
        .score;

    let rel = ((v_native - v_host) / v_host).abs();
    assert!(
        rel < NATIVE_VS_HOST_REL,
        "256² native_strip {v_native} vs host_strip {v_host} rel={rel}"
    );
}

#[test]
fn native_strip_matches_host_strip_512_body_256() {
    let w = 512_u32;
    let h = 512_u32;
    let r_rgb = make_color_rgb(w, h, 19);
    let d_rgb = make_color_rgb(w, h, 23);

    let client = BackendT::client(&Default::default());

    let mut s_host =
        Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip host");
    s_host.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref host");
    let v_host = s_host
        .compute_rgb_with_reference_stripped(&d_rgb)
        .expect("compute host")
        .score;

    let mut s_native =
        Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip native");
    s_native.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref native");
    let v_native = s_native
        .compute_rgb_with_reference_stripped_native(&d_rgb)
        .expect("compute native")
        .score;

    let rel = ((v_native - v_host) / v_host).abs();
    assert!(
        rel < NATIVE_VS_HOST_REL,
        "512² native_strip {v_native} vs host_strip {v_host} rel={rel}"
    );
}

#[test]
fn native_strip_matches_host_strip_1024_body_256() {
    // Production-ish 1024² body=256 (4 strips).
    let w = 1024_u32;
    let h = 1024_u32;
    let r_rgb = make_color_rgb(w, h, 31);
    let d_rgb = make_color_rgb(w, h, 37);

    let client = BackendT::client(&Default::default());

    let mut s_host =
        Iwssim::<BackendT>::new_strip(client.clone(), w, h, 256).expect("strip host");
    s_host.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref host");
    let v_host = s_host
        .compute_rgb_with_reference_stripped(&d_rgb)
        .expect("compute host")
        .score;

    let mut s_native =
        Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip native");
    s_native.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref native");
    let v_native = s_native
        .compute_rgb_with_reference_stripped_native(&d_rgb)
        .expect("compute native")
        .score;

    let rel = ((v_native - v_host) / v_host).abs();
    assert!(
        rel < NATIVE_VS_HOST_REL,
        "1024² native_strip {v_native} vs host_strip {v_host} rel={rel}"
    );
}

#[test]
fn native_strip_self_identity_512() {
    let w = 512_u32;
    let h = 512_u32;
    let r_rgb = make_color_rgb(w, h, 41);
    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    strip.set_rgb_reference_stripped(&r_rgb).expect("set_rgb_ref");
    let s = strip
        .compute_rgb_with_reference_stripped_native(&r_rgb)
        .expect("native compute")
        .score;
    assert!(
        (s - 1.0).abs() < 1e-5,
        "native rgb strip self-identity {s} ≠ 1"
    );
}

#[test]
fn native_strip_errors_on_whole_image_instance() {
    let w = 256_u32;
    let h = 256_u32;
    let d = make_color_rgb(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut whole = Iwssim::<BackendT>::new(client, w, h).expect("whole new");
    let err = whole
        .compute_rgb_with_reference_stripped_native(&d)
        .expect_err("must error");
    assert!(matches!(err, Error::NotStripMode));
}

#[test]
fn native_strip_errors_on_no_cached_reference() {
    let w = 256_u32;
    let h = 256_u32;
    let d = make_color_rgb(w, h, 7);
    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip");
    let err = strip
        .compute_rgb_with_reference_stripped_native(&d)
        .expect_err("must error");
    assert!(
        matches!(err, Error::NoCachedReference),
        "expected NoCachedReference, got {err:?}"
    );
}

#[test]
fn native_strip_errors_on_dim_mismatch() {
    let w = 256_u32;
    let h = 256_u32;
    let r = make_color_rgb(w, h, 0);
    let client = BackendT::client(&Default::default());
    let mut strip = Iwssim::<BackendT>::new_strip(client, w, h, 256).expect("strip new");
    strip.set_rgb_reference_stripped(&r).expect("set_rgb_ref");
    let too_small = vec![0_u8; ((w - 1) * h * 3) as usize];
    let err = strip
        .compute_rgb_with_reference_stripped_native(&too_small)
        .expect_err("must error on small dis");
    assert!(matches!(err, Error::DimensionMismatch { .. }));
}
