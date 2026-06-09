//! HF-checkerboard strip-parity GATE for butteraugli-gpu (task #158).
//!
//! The mode_wall sweep (`benchmarks/mode_wall_2026-05-31.{csv,md}`) found
//! butter's Strip score diverging ~8% from Full on an aggressive
//! high-frequency checkerboard, while the other five -gpu crates were
//! score-safe. Root cause (see `crates/butteraugli-gpu/src/strip.rs`
//! module docs + the fix commit): the umbrella `ButteraugliOpaque` ran
//! `MemoryMode::Full` through the multi-resolution path (full-res +
//! half-res supersample) but `MemoryMode::Strip` through single-resolution
//! — silently dropping the half-res band — and the multires-strip
//! half-res sibling was under-haloed (`HALO_ROWS/2` real rows < the 34 the
//! half-res blur cascade needs).
//!
//! These tests pin BOTH halves of the fix on the EXACT content the sweep
//! used (`structured_pair`: smooth RGB gradient ref + period-8 ±12
//! checkerboard dist — the worst case for a finite strip halo):
//!
//! 1. typed `new_multires` (Full) == `new_multires_strip` (Strip), and
//! 2. the umbrella `ButteraugliOpaque` `MemoryMode::Full` ==
//!    `MemoryMode::Strip` (the precise surface the sweep measured).
//!
//! Both the max-norm `score` AND the `pnorm_3` aggregate must agree within
//! `1e-4` rel — which is the existing `strip_parity` tolerance, far tighter
//! than the 8% bug (and tighter than the ~7e-4 max-norm half-res-halo
//! regression). With the fix, the measured agreement is bit-identical
//! (0.0e0 max-norm, ≤2e-7 pnorm_3) at every size/body tested.
//!
//! NOTE: the smooth+moderate-HF `make_image` content the other strip
//! tests use is NOT a substitute here — the period-8 ±12 checkerboard is
//! the adversarial HF input that exposed the bug. This file deliberately
//! uses that content.

#![cfg(all(
    feature = "cubecl-types",
    any(feature = "cpu", feature = "cuda", feature = "wgpu")
))]

use butteraugli_gpu::Butteraugli;
#[cfg(feature = "cuda")]
use butteraugli_gpu::ButteraugliParams;
use cubecl::Runtime;

#[cfg(feature = "cuda")]
type BackendT = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type BackendT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type BackendT = cubecl::cpu::CpuRuntime;

/// EXACT copy of `mode_wall.rs`'s `structured_pair` (the content the
/// divergence was measured on): smooth RGB gradient reference + a
/// period-8 (8×8 block) ±mag checkerboard perturbation on the distorted
/// image. mag=12 is the sweep's value.
fn structured_pair(w: u32, h: u32, mag: u8) -> (Vec<u8>, Vec<u8>) {
    let (width, height) = (w as usize, h as usize);
    let mut a = vec![0u8; width * height * 3];
    let mut b = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 220 / width.max(1)) & 0xff) as u8;
            let g = ((y * 220 / height.max(1)) & 0xff) as u8;
            let bb = (((x + y) * 200 / (width + height).max(1)) & 0xff) as u8;
            let i = (y * width + x) * 3;
            a[i] = r;
            a[i + 1] = g;
            a[i + 2] = bb;
            let bx = x / 8;
            let by = y / 8;
            let pert = if (bx ^ by) & 1 == 0 {
                mag as i32
            } else {
                -(mag as i32)
            };
            b[i] = (r as i32 + pert).clamp(0, 255) as u8;
            b[i + 1] = (g as i32 + pert).clamp(0, 255) as u8;
            b[i + 2] = (bb as i32 + pert).clamp(0, 255) as u8;
        }
    }
    (a, b)
}

fn assert_rel_eq(name: &str, want: f64, got: f64, tol: f64) {
    let denom = want.abs().max(1e-12);
    let rel = (got - want).abs() / denom;
    assert!(
        rel < tol,
        "{name}: full={want} strip={got} rel_err={rel:.3e} (tol={tol:.0e}) \
         — HF-checkerboard strip parity regressed (task #158)"
    );
}

/// Typed multires path: `new_multires` (Full) vs `new_multires_strip`
/// (Strip) on the HF checkerboard. Both score and pnorm_3 must match.
fn check_typed_multires(w: u32, h: u32, body: u32) {
    let (r, d) = structured_pair(w, h, 12);
    let client = BackendT::client(&Default::default());

    let mut whole = Butteraugli::<BackendT>::new_multires(client.clone(), w, h);
    let wr = whole.compute(&r, &d).expect("multires-whole compute");

    let mut strip = Butteraugli::<BackendT>::new_multires_strip(client, w, h, body);
    let sr = strip.compute_strip(&r, &d).expect("multires-strip compute");

    assert_rel_eq(
        &format!("typed score {w}x{h}/body{body}"),
        wr.score as f64,
        sr.score as f64,
        1e-4,
    );
    assert_rel_eq(
        &format!("typed pnorm_3 {w}x{h}/body{body}"),
        wr.pnorm_3 as f64,
        sr.pnorm_3 as f64,
        1e-4,
    );
}

#[test]
fn hf_checkerboard_typed_512_body_256() {
    // The mode_wall sweep's worst HF-checkerboard cell (max-norm drifted
    // ~7e-4 at HALO_ROWS=40 here before the bump).
    check_typed_multires(512, 512, 256);
}

#[test]
fn hf_checkerboard_typed_512_body_64() {
    check_typed_multires(512, 512, 64);
}

#[test]
fn hf_checkerboard_typed_1024_body_256() {
    // The exact MW_PARITY=1024 cell that reported butter strip 15.007 vs
    // full 16.317 (the 8% headline divergence).
    check_typed_multires(1024, 1024, 256);
}

#[test]
fn hf_checkerboard_typed_1024_body_128() {
    check_typed_multires(1024, 1024, 128);
}

#[test]
fn hf_checkerboard_typed_768x800_body_96_uneven() {
    // Non-square + body not dividing height: the last strip's body is
    // short and its half-res counterpart clamps. Exercises the boundary
    // arithmetic on HF content.
    check_typed_multires(768, 800, 96);
}

// ── Umbrella opaque path: the EXACT surface the sweep measured ──
//
// `ButteraugliOpaque` is what `zenmetrics-api`'s `Metric::Butter` wraps.
// `MemoryMode::Full` -> new_multires; `MemoryMode::Strip` -> (after the
// fix) new_multires_strip. Their `.value` (max-norm) must agree on the HF
// checkerboard. This is the precise comparison that produced the 8% number.

#[cfg(feature = "cuda")]
fn check_opaque(w: u32, h: u32, body: u32) {
    use butteraugli_gpu::{Backend, ButteraugliOpaque, MemoryMode};

    let (r, d) = structured_pair(w, h, 12);

    let mut full = ButteraugliOpaque::new_with_memory_mode(
        Backend::Cuda,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Full,
    )
    .expect("opaque Full");
    let fv = full
        .compute_srgb_u8(&r, &d)
        .expect("opaque Full compute")
        .value;

    let mut strip = ButteraugliOpaque::new_with_memory_mode(
        Backend::Cuda,
        w,
        h,
        ButteraugliParams::default(),
        MemoryMode::Strip { h_body: Some(body) },
    )
    .expect("opaque Strip");
    let sv = strip
        .compute_srgb_u8(&r, &d)
        .expect("opaque Strip compute")
        .value;

    assert_rel_eq(&format!("opaque value {w}x{h}/body{body}"), fv, sv, 1e-4);
}

#[cfg(feature = "cuda")]
#[test]
fn hf_checkerboard_opaque_512_body_256() {
    check_opaque(512, 512, 256);
}

#[cfg(feature = "cuda")]
#[test]
fn hf_checkerboard_opaque_1024_body_256() {
    check_opaque(1024, 1024, 256);
}

#[cfg(feature = "cuda")]
#[test]
fn hf_checkerboard_opaque_1024_body_128() {
    check_opaque(1024, 1024, 128);
}
