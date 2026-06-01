//! End-to-end diffmap dispatch tests for `Cvvdp<R>`.
//!
//! Runs the full GPU pipeline (color → weber pyramid → CSF → masking
//! → diffmap accumulate → channel pool → host readback) and pins the
//! invariants documented in `kernels::diffmap` module docs:
//!
//! - **Shape**: `diffmap.len() == width * height`
//! - **Identity**: `ref ≡ dist` → all-zero diffmap to 1e-7 absolute
//! - **Non-negative**: `diffmap.iter().all(|&v| v >= 0)`
//! - **JOD invariance**: `score_with_diffmap` returns the same JOD
//!   scalar as `score` on the same input pair (the diffmap fold is
//!   side-channel; the scalar pool path is unchanged)
//! - **Linear-planes equivalence**: `score_from_linear_planes` on
//!   sRGB-decoded linear planes equals `score` on the original sRGB
//!   bytes to a documented tolerance (the host LUT path and the
//!   skip-LUT linear path differ in their inputs by at most f32
//!   precision after sRGB→linear conversion).
//!
//! The cubecl-cpu backend is used by default (no GPU required for
//! CI). The CUDA backend tests live in `tests/pipeline_score.rs`;
//! we mirror the diffmap-specific invariants here at small sizes
//! to keep cubecl-cpu's runtime cost low (large sizes take
//! seconds-to-minutes on cubecl-cpu).
//!
//! Note: `Cvvdp::compute_dkl_jod` uses `pool_band_3ch_kernel` which
//! requires `Atomic<f32>::fetch_add` — unsupported on cubecl-cpu.
//! These tests therefore use the GPU runtime that's compiled in
//! (default features: prefer CUDA if available, else wgpu). For
//! a cpu-only environment, see `tests/cpu_backend.rs` for the
//! host-pool variant that bypasses the atomic kernel; the diffmap
//! path uses `_compute_diffmap_into` which calls
//! `_pool_and_finalize_jod` internally, so it inherits the same
//! atomic-f32 constraint.

#![cfg(any(feature = "cuda", feature = "wgpu"))]
#![allow(clippy::excessive_precision)]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n_pix = (w as usize) * (h as usize);
    let mut r = Vec::with_capacity(n_pix * 3);
    let mut d = Vec::with_capacity(n_pix * 3);
    for y in 0..h {
        for x in 0..w {
            let base_r = ((x * 5) % 256) as u8;
            let base_g = ((y * 7) % 256) as u8;
            let base_b = (((x + y) * 3) % 256) as u8;
            r.push(base_r);
            r.push(base_g);
            r.push(base_b);
            // Add a tiny pixel-wise offset for the distorted side.
            d.push(base_r.wrapping_add(10));
            d.push(base_g.wrapping_add(6));
            d.push(base_b.wrapping_add(8));
        }
    }
    (r, d)
}

fn srgb_to_linear(b: u8) -> f32 {
    // Match `kernels::color::SRGB8_TO_LINEAR_LUT` semantics. Inlined
    // here to keep the test self-contained.
    let p = (b as f32) / 255.0;
    if p > 0.04045 {
        ((p + 0.055) / 1.055).powf(2.4)
    } else {
        p / 12.92
    }
}

fn srgb_bytes_to_linear_planes(bytes: &[u8], n_pix: usize) -> [Vec<f32>; 3] {
    let mut r = vec![0.0_f32; n_pix];
    let mut g = vec![0.0_f32; n_pix];
    let mut b = vec![0.0_f32; n_pix];
    for i in 0..n_pix {
        r[i] = srgb_to_linear(bytes[i * 3]);
        g[i] = srgb_to_linear(bytes[i * 3 + 1]);
        b[i] = srgb_to_linear(bytes[i * 3 + 2]);
    }
    [r, g, b]
}

#[test]
fn score_with_diffmap_returns_correct_shape_and_jod() {
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let (r, d) = synth_pair(w, h);
    let mut diffmap = Vec::new();
    let jod = cvvdp
        .score_with_diffmap(&r, &d, &mut diffmap)
        .expect("score_with_diffmap");

    assert_eq!(diffmap.len(), n_pix);
    assert!(jod.is_finite() && (0.0..=10.0).contains(&jod));
    for &v in &diffmap {
        assert!(v >= 0.0, "diffmap value {v} must be non-negative");
        assert!(v.is_finite(), "diffmap value {v} must be finite");
    }
}

#[test]
fn score_with_diffmap_identity_yields_zero_diffmap() {
    // RFC §3 contract: ref ≡ dist → diffmap must be all zeros to
    // 1e-7 absolute. This is the single most important invariant for
    // the jxl-encoder buttloop — false-floor in the diffmap would
    // cause spurious quantization changes on already-perfect blocks.
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let buf = vec![128u8; n_pix * 3];
    let mut diffmap = Vec::new();
    let jod = cvvdp
        .score_with_diffmap(&buf, &buf, &mut diffmap)
        .expect("score_with_diffmap");

    assert_eq!(diffmap.len(), n_pix);
    assert!(
        (jod - 10.0).abs() < 1e-3,
        "identity JOD = {jod}, expected ≈ 10"
    );
    let max_abs = diffmap.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    assert!(
        max_abs < 1e-7,
        "identity diffmap has non-zero entry, max |.| = {max_abs}",
    );
}

#[test]
fn score_with_diffmap_jod_matches_score_scalar() {
    // `score_with_diffmap` should produce the same JOD scalar as
    // plain `score` on the same input pair — the diffmap fold is a
    // side-channel; the scalar pool path goes through the same
    // _pool_and_finalize_jod.
    let client = Backend::client(&Default::default());
    let (w, h) = (24u32, 24u32);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let (r, d) = synth_pair(w, h);
    let scalar_jod = cvvdp.score(&r, &d).expect("score scalar") as f32;

    let mut diffmap = Vec::new();
    let diffmap_jod = cvvdp
        .score_with_diffmap(&r, &d, &mut diffmap)
        .expect("score_with_diffmap");

    assert!(
        (scalar_jod - diffmap_jod).abs() < 1e-4,
        "scalar_jod = {scalar_jod} vs diffmap_jod = {diffmap_jod}",
    );
}

#[test]
fn score_from_linear_planes_matches_score_to_srgb_lut_tolerance() {
    // Path A: feed sRGB bytes through the LUT kernel (existing path).
    // Path B: feed pre-LUT'd linear-RGB f32 planes through the new
    //         from-linear-planes kernel.
    // Both paths must produce the same JOD scalar to within a small
    // tolerance — the only numerical difference is the f32 round-trip
    // through the sRGB→linear LUT: bytes lookup a 256-entry f32 table
    // (the same values our host helper computes), so the two paths
    // see identical-to-1-ULP linear inputs.
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let (r_bytes, d_bytes) = synth_pair(w, h);
    let [r_r, r_g, r_b] = srgb_bytes_to_linear_planes(&r_bytes, n_pix);
    let [d_r, d_g, d_b] = srgb_bytes_to_linear_planes(&d_bytes, n_pix);

    let jod_bytes = cvvdp.score(&r_bytes, &d_bytes).expect("score sRGB-bytes") as f32;
    let jod_planes = cvvdp
        .score_from_linear_planes(&r_r, &r_g, &r_b, &d_r, &d_g, &d_b)
        .expect("score_from_linear_planes");

    // The LUT itself rounds at f32 — feeding host-computed linear-f32
    // through the from-linear-planes path may differ by a few ULPs
    // from the in-kernel LUT lookup. 1e-3 JOD is well above the f32-
    // precision floor and well below any meaningful drift.
    assert!(
        (jod_bytes - jod_planes).abs() < 1e-3,
        "sRGB path JOD = {jod_bytes} vs linear-planes path JOD = {jod_planes}",
    );
}

#[test]
fn warm_reference_diffmap_matches_one_shot() {
    // The warm-ref variant must produce the same JOD + diffmap as a
    // one-shot dispatch on the same DIST input.
    let client = Backend::client(&Default::default());
    let (w, h) = (24u32, 24u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let (r, d) = synth_pair(w, h);
    let mut diffmap_oneshot = Vec::new();
    let jod_oneshot = cvvdp
        .score_with_diffmap(&r, &d, &mut diffmap_oneshot)
        .expect("score_with_diffmap oneshot");

    cvvdp.warm_reference(&r).expect("warm_reference");
    let mut diffmap_warm = Vec::new();
    let jod_warm = cvvdp
        .score_with_warm_ref_diffmap(&d, &mut diffmap_warm)
        .expect("score_with_warm_ref_diffmap");

    assert!(
        (jod_oneshot - jod_warm).abs() < 1e-4,
        "oneshot JOD = {jod_oneshot} vs warm JOD = {jod_warm}",
    );
    assert_eq!(diffmap_oneshot.len(), n_pix);
    assert_eq!(diffmap_warm.len(), n_pix);
    let max_dev = diffmap_oneshot
        .iter()
        .zip(diffmap_warm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_dev < 1e-4,
        "warm vs oneshot diffmap max deviation = {max_dev}",
    );
}

#[test]
fn warm_reference_from_linear_planes_matches_one_shot() {
    let client = Backend::client(&Default::default());
    let (w, h) = (32u32, 32u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let (r_bytes, d_bytes) = synth_pair(w, h);
    let [r_r, r_g, r_b] = srgb_bytes_to_linear_planes(&r_bytes, n_pix);
    let [d_r, d_g, d_b] = srgb_bytes_to_linear_planes(&d_bytes, n_pix);

    let mut diffmap_oneshot = Vec::new();
    let jod_oneshot = cvvdp
        .score_from_linear_planes_with_diffmap(
            &r_r,
            &r_g,
            &r_b,
            &d_r,
            &d_g,
            &d_b,
            &mut diffmap_oneshot,
        )
        .expect("score_from_linear_planes_with_diffmap");

    cvvdp
        .warm_reference_from_linear_planes(&r_r, &r_g, &r_b)
        .expect("warm_reference_from_linear_planes");
    let mut diffmap_warm = Vec::new();
    let jod_warm = cvvdp
        .score_from_linear_planes_with_warm_ref_diffmap(&d_r, &d_g, &d_b, &mut diffmap_warm)
        .expect("score_from_linear_planes_with_warm_ref_diffmap");

    assert!(
        (jod_oneshot - jod_warm).abs() < 1e-4,
        "linear-planes warm vs oneshot JOD: {jod_oneshot} vs {jod_warm}",
    );
    let max_dev = diffmap_oneshot
        .iter()
        .zip(diffmap_warm.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_dev < 1e-4,
        "warm vs oneshot linear-planes diffmap max dev = {max_dev}",
    );
}

#[test]
fn dimension_mismatch_surfaced_on_linear_planes() {
    // The from-linear-planes family validates each plane is exactly
    // `width * height` long, returning DimensionMismatch with the
    // expected/got byte counts.
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    let good = vec![0.5_f32; n_pix];
    let bad = vec![0.5_f32; n_pix - 1]; // one short
    let result = cvvdp.score_from_linear_planes(&bad, &good, &good, &good, &good, &good);
    assert!(matches!(
        result,
        Err(cvvdp_gpu::Error::DimensionMismatch { .. })
    ));

    // Also fires from a wrong-size DIST plane.
    let result = cvvdp.score_from_linear_planes(&good, &good, &good, &good, &good, &bad);
    assert!(matches!(
        result,
        Err(cvvdp_gpu::Error::DimensionMismatch { .. })
    ));
}

#[test]
fn diffmap_reuses_caller_vec_buffer() {
    // The diffmap_out Vec is overwritten via clear() + extend, so
    // callers can keep a long-lived buffer that's reused across
    // many score calls without per-call allocation.
    let client = Backend::client(&Default::default());
    let (w, h) = (16u32, 16u32);
    let n_pix = (w as usize) * (h as usize);
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, w, h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");

    // Pre-allocate a Vec with junk content; verify the diffmap call
    // overwrites it cleanly.
    let mut diffmap = vec![999.0_f32; n_pix * 4]; // 4x oversize on purpose
    let pre_cap = diffmap.capacity();

    let (r, d) = synth_pair(w, h);
    let _ = cvvdp
        .score_with_diffmap(&r, &d, &mut diffmap)
        .expect("score_with_diffmap");

    assert_eq!(diffmap.len(), n_pix);
    // Capacity may stay >= pre_cap (Vec doesn't shrink on clear+extend
    // when the new content fits). Just check we didn't realloc smaller.
    assert!(diffmap.capacity() >= pre_cap.max(n_pix));
    // No 999.0 stragglers — clear() drops the prior content.
    for &v in &diffmap {
        assert_ne!(v, 999.0);
    }
}
