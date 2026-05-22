//! Multires-strip vs multires-whole parity tests.
//!
//! [`Butteraugli::new_multires`] (whole-image) and
//! [`Butteraugli::new_multires_strip`] both compute the CPU-butter
//! default multi-resolution score: full-res + half-resolution
//! sibling whose diffmap is supersample-added into the full-res
//! diffmap before reduction. The strip variant runs the same kernel
//! chain over strip-sized slabs and stitches the per-strip body
//! diffmap reduction into the final partials host-side.
//!
//! The reduction order differs (per-strip max+p3/p6/p12 folded
//! host-side vs single fused on-device reduce), so a small
//! numerical drift is expected — same `1e-4` rel tolerance used by
//! the single-resolution `strip_parity` tests.

#![cfg(all(
    feature = "cubecl-types",
    any(feature = "cpu", feature = "cuda", feature = "wgpu")
))]

use butteraugli_gpu::{Butteraugli, ButteraugliParams, MemoryMode};
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type BackendT = cubecl::cpu::CpuRuntime;

/// Same deterministic mid-spatial-frequency image as the
/// single-resolution `strip_parity` tests: both LF (σ=7.16) and HF
/// (σ=3.22) bands see signal AND the half-res sibling has visible
/// content too (the half-res image is the 2× downsample of this).
fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let sx = ((x as f32 / 32.0).sin() * 50.0 + 128.0) as u8;
            let sy = ((y as f32 / 24.0).cos() * 40.0 + 128.0) as u8;
            let hf = (((x ^ y).wrapping_mul(seed.max(1)) ^ seed) & 0x3f) as u8;
            out.push(sx.wrapping_add(hf));
            out.push(sy.wrapping_add(hf));
            out.push(sx.wrapping_add(sy).wrapping_add(hf >> 1));
        }
    }
    out
}

fn run_pair(w: u32, h: u32, body_h: u32) -> (f32, f32, f32, f32) {
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let client = BackendT::client(&Default::default());

    let mut whole = Butteraugli::<BackendT>::new_multires(client.clone(), w, h);
    let whole_res = whole.compute(&ref_buf, &dis_buf).expect("multires-whole");

    let mut strip = Butteraugli::<BackendT>::new_multires_strip(client, w, h, body_h);
    let strip_res = strip
        .compute_strip(&ref_buf, &dis_buf)
        .expect("multires-strip");

    (
        whole_res.score,
        whole_res.pnorm_3,
        strip_res.score,
        strip_res.pnorm_3,
    )
}

fn assert_rel_eq(name: &str, want: f32, got: f32, tol: f64) {
    let denom = (want as f64).abs().max(1e-12);
    let rel = (got as f64 - want as f64).abs() / denom;
    assert!(
        rel < tol,
        "{name}: whole={want} strip={got} rel_err={rel:.2e} (tol={tol:.0e})"
    );
}

// ─── Multires-strip vs multires-whole parity matrix ───
//
// Sweeps small (256²), medium (1024²), large (2048²) at body sizes
// that produce both multi-strip walks (4-8 strips) and degenerate
// single-strip passes. 4096² is omitted on the bench-host GPU
// memory budget — covered by the bench example.

#[test]
fn multires_strip_vs_whole_256_body_64() {
    let (ws, wp, ss, sp) = run_pair(256, 256, 64);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn multires_strip_vs_whole_512_body_128() {
    let (ws, wp, ss, sp) = run_pair(512, 512, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn multires_strip_vs_whole_1024_body_128() {
    let (ws, wp, ss, sp) = run_pair(1024, 1024, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn multires_strip_vs_whole_1024_body_256() {
    let (ws, wp, ss, sp) = run_pair(1024, 1024, 256);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn multires_strip_vs_whole_2048_body_128() {
    let (ws, wp, ss, sp) = run_pair(2048, 2048, 128);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

#[test]
fn multires_strip_vs_whole_2048_body_256() {
    let (ws, wp, ss, sp) = run_pair(2048, 2048, 256);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

// ─── Non-square / 4000×3000 photo-aspect case ───
//
// Real photos aren't 4096² — 4000×3000 (12 MP, 4:3) is the standard
// camera intermediate. The strip walker has to handle non-square
// dims correctly (separately because the half-res's body_top
// alignment uses width and height independently).

#[test]
fn multires_strip_vs_whole_4000x3000_body_300() {
    let (ws, wp, ss, sp) = run_pair(4000, 3000, 300);
    assert_rel_eq("score", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3", wp, sp, 1e-4);
}

// ─── Edge / contract checks ───

#[test]
fn multires_strip_uneven_body_768x800_body_96() {
    // image_h=800 isn't a multiple of body=96 → 8 strips of 96 + 1
    // strip of 32 = 800. The last strip's body is short; its
    // half-res counterpart needs to clamp body_end_half to the
    // half-res image height (800 / 2 = 400) cleanly.
    let (ws, wp, ss, sp) = run_pair(768, 800, 96);
    assert_rel_eq("score(uneven)", ws, ss, 1e-4);
    assert_rel_eq("pnorm_3(uneven)", wp, sp, 1e-4);
}

#[test]
fn multires_strip_with_options() {
    let w = 512;
    let h = 512;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 7);
    let params = ButteraugliParams::default()
        .with_intensity_target(120.0)
        .with_hf_asymmetry(1.5)
        .with_xmul(0.5);

    let client = BackendT::client(&Default::default());
    let mut whole = Butteraugli::<BackendT>::new_multires(client.clone(), w, h);
    let whole_res = whole
        .compute_with_options(&ref_buf, &dis_buf, &params)
        .expect("multires-whole compute_with_options");

    let mut strip = Butteraugli::<BackendT>::new_multires_strip(client, w, h, 128);
    let strip_res = strip
        .compute_strip_with_options(&ref_buf, &dis_buf, &params)
        .expect("multires-strip compute_strip_with_options");

    assert_rel_eq("score(options)", whole_res.score, strip_res.score, 1e-4);
    assert_rel_eq("pnorm_3(options)", whole_res.pnorm_3, strip_res.pnorm_3, 1e-4);
}

#[test]
fn multires_strip_records_half_res_sibling() {
    let client = BackendT::client(&Default::default());
    let strip = Butteraugli::<BackendT>::new_multires_strip(client, 1024, 1024, 128);
    assert!(strip.is_strip_mode(), "multires_strip must be strip-mode");
    assert!(
        strip.half_res().is_some(),
        "multires_strip must allocate a half-res sibling"
    );
    let half = strip.half_res().unwrap();
    assert!(
        half.is_strip_mode(),
        "the half-res sibling itself must be strip-mode"
    );
    // Half-res image dims are ceiling-halved.
    assert_eq!(half.image_height(), 512);
    // Half-res body is body/2 (constructor enforces body even).
    assert_eq!(half.strip_body_h(), 64);
}

#[test]
fn multires_strip_via_memory_mode_constructor() {
    // The mode-aware constructor MUST route MemoryMode::Strip to
    // new_multires_strip (used to surface
    // StripModeUnsupported("new_multires") — that error is gone now
    // that the strip-multires path exists).
    let client = BackendT::client(&Default::default());
    let strip = Butteraugli::<BackendT>::new_multires_with_memory_mode(
        client,
        1024,
        1024,
        MemoryMode::Strip { h_body: Some(128) },
    )
    .expect("new_multires_with_memory_mode Strip must succeed");
    assert!(strip.is_strip_mode());
    assert!(strip.half_res().is_some());
}
