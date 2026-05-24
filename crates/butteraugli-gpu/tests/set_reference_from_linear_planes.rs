//! Parity test for `set_reference_from_linear_planes` (W44-phase3-B4).
//!
//! Two paths must produce identical butteraugli scores:
//! 1. **Legacy**: `set_reference(srgb_u8)` then `compute_with_reference(srgb_u8)`.
//!    Internally: GPU uploads u8, runs srgb_byte_to_linear kernel, runs full
//!    opsin / freq / mask / diff pipeline.
//! 2. **Linear-planes**: host-side srgb→linear conversion (same formula as
//!    the GPU kernel), upload f32 plane handles via `create_from_slice`,
//!    then `set_reference_from_linear_planes(linear_handles)` then
//!    `compute_with_reference(srgb_u8)` for the distorted side. The
//!    reference-side srgb→linear kernel is bypassed; the distorted-side
//!    path is unchanged.
//!
//! Tolerance: within 0.01% relative score. The two paths differ only in
//! where the srgb→linear conversion happens (host vs GPU) — the formula
//! is bit-identical because both use the same IEC 61966-2-1 piecewise
//! transfer. Any drift > 0.01% indicates a real bug.

#![cfg(all(feature = "cubecl-types", feature = "internals", feature = "cuda"))]

use butteraugli_gpu::Butteraugli;
use cubecl::Runtime;
use cubecl::prelude::*;

type Backend = cubecl::cuda::CudaRuntime;

/// Host-side sRGB byte → linear-f32 — bit-identical to
/// `srgb_byte_to_linear` in `src/kernels/colors.rs`.
fn srgb_byte_to_linear(v: u8) -> f32 {
    let f = (v as f32) * (1.0 / 255.0);
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

/// Unpack an interleaved sRGB-u8 buffer (`width*height*3` bytes,
/// `[R,G,B,R,G,B,...]`) into three tight planar f32 linear-RGB buffers.
fn srgb_to_linear_planes(srgb: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = width * height;
    assert_eq!(srgb.len(), n * 3);
    let mut r = vec![0.0_f32; n];
    let mut g = vec![0.0_f32; n];
    let mut b = vec![0.0_f32; n];
    for i in 0..n {
        r[i] = srgb_byte_to_linear(srgb[i * 3]);
        g[i] = srgb_byte_to_linear(srgb[i * 3 + 1]);
        b[i] = srgb_byte_to_linear(srgb[i * 3 + 2]);
    }
    (r, g, b)
}

/// Build a deterministic synthetic sRGB image with gradients + a center
/// perturbation so butteraugli reports non-zero.
fn make_ref_image(width: usize, height: usize) -> Vec<u8> {
    let mut buf = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let i = (y * width + x) * 3;
            buf[i] = ((x * 255) / width.max(1)) as u8;
            buf[i + 1] = ((y * 255) / height.max(1)) as u8;
            buf[i + 2] = (((x + y) * 255) / (width + height).max(1)) as u8;
        }
    }
    buf
}

fn perturb(src: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out = src.to_vec();
    // Inject a JPEG-like blocking perturbation in the centre region.
    for y in (height / 4)..(3 * height / 4) {
        for x in (width / 4)..(3 * width / 4) {
            let i = (y * width + x) * 3;
            let bx = x / 8;
            let by = y / 8;
            let stripe = (((bx + by) % 3) as i32 - 1).signum();
            for c in 0..3 {
                let v = src[i + c] as i32 + stripe * 12;
                out[i + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    out
}

fn upload_plane(client: &cubecl::prelude::ComputeClient<Backend>, plane: &[f32]) -> cubecl::server::Handle {
    client.create_from_slice(f32::as_bytes(plane))
}

fn run_parity(width: u32, height: u32) {
    let w = width as usize;
    let h = height as usize;
    let ref_srgb = make_ref_image(w, h);
    let dist_srgb = perturb(&ref_srgb, w, h);

    let device = <Backend as Runtime>::Device::default();
    let client = <Backend as Runtime>::client(&device);

    // Path 1: legacy set_reference(srgb_u8) → compute_with_reference(srgb_u8)
    let mut legacy = Butteraugli::<Backend>::new_multires(client.clone(), width, height);
    legacy.set_reference(&ref_srgb).expect("legacy set_reference");
    let r_legacy = legacy
        .compute_with_reference(&dist_srgb)
        .expect("legacy compute");

    // Path 2: host srgb→linear → upload f32 planes → set_reference_from_linear_planes
    //         → compute_with_reference(srgb_u8) for distorted (same as legacy).
    let mut linp = Butteraugli::<Backend>::new_multires(client.clone(), width, height);
    let (r_plane, g_plane, b_plane) = srgb_to_linear_planes(&ref_srgb, w, h);
    let ref_r_h = upload_plane(&client, &r_plane);
    let ref_g_h = upload_plane(&client, &g_plane);
    let ref_b_h = upload_plane(&client, &b_plane);
    linp.set_reference_from_linear_planes(ref_r_h, ref_g_h, ref_b_h)
        .expect("set_reference_from_linear_planes");
    let r_lin = linp
        .compute_with_reference(&dist_srgb)
        .expect("lin compute");

    // Tolerance: 0.01% relative on score and pnorm_3. The only difference
    // between paths is where the srgb→linear conversion happens (host vs
    // GPU); both use the bit-identical IEC 61966-2-1 formula.
    let rel_score = (r_lin.score as f64 - r_legacy.score as f64).abs() / r_legacy.score.max(1e-12) as f64;
    let rel_pnorm = (r_lin.pnorm_3 as f64 - r_legacy.pnorm_3 as f64).abs() / r_legacy.pnorm_3.max(1e-12) as f64;
    println!(
        "{}×{} legacy: score={:.6} pnorm3={:.6} | lin: score={:.6} pnorm3={:.6} | rel: score={:.2e} pnorm3={:.2e}",
        width, height, r_legacy.score, r_legacy.pnorm_3, r_lin.score, r_lin.pnorm_3, rel_score, rel_pnorm,
    );
    assert!(
        rel_score < 1e-4,
        "{}×{}: score divergence {:.4}% (legacy={} lin={})",
        width,
        height,
        rel_score * 100.0,
        r_legacy.score,
        r_lin.score,
    );
    assert!(
        rel_pnorm < 1e-4,
        "{}×{}: pnorm_3 divergence {:.4}% (legacy={} lin={})",
        width,
        height,
        rel_pnorm * 100.0,
        r_legacy.pnorm_3,
        r_lin.pnorm_3,
    );
}

#[test]
fn parity_64x64() {
    run_parity(64, 64);
}

#[test]
fn parity_128x128() {
    run_parity(128, 128);
}

#[test]
fn parity_256x256() {
    run_parity(256, 256);
}
