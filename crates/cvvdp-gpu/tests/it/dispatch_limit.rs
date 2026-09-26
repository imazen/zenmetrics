//! JOD above wgpu's 65 535-workgroups-per-axis dispatch limit.
//!
//! A 1-D per-pixel launch of 64-thread workgroups needs more than
//! 65 535 of them once `width × height > 4 194 240`. Before
//! `cube_count_1d` folded such grids into 2-D, wgpu rejected every one of
//! those dispatches on its device thread, no error reached the caller,
//! and the JOD came back as exactly 10.0: on Vulkan, 2026-09-26, the
//! noise pair scored 10.000000 at 2048×2048 and at 3000×2000 (CUDA:
//! 1.854579 / 3.804536).
//!
//! The strip cases use `h_body = 2048`, so each strip window covers the
//! whole 3000×2000 band (93 750 workgroups) and the strip walkers'
//! launches cross the limit too. Multi-strip Mode B is not covered here:
//! it still panics on wgpu for unrelated reasons (unaligned sub-views
//! outside the masking walker; see CLAUDE.md Known Bugs).

#![cfg(feature = "wgpu")]

use crate::common::noise_pair;

use cubecl::Runtime;
use cubecl::wgpu::WgpuRuntime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

/// GPU-path parity tolerance used across this suite (Full vs strip
/// modes, `strip_mode_e_parity.rs`); wgpu and CUDA agree to ~3e-6 here.
const PARITY_TOL_JOD: f32 = 1e-4;

/// Sizes whose level-0 per-pixel launches exceed 65 535 workgroups.
const SIZES: [(u32, u32); 2] = [(2048, 2048), (3000, 2000)];

fn ppd() -> f32 {
    DisplayGeometry::STANDARD_4K.pixels_per_degree()
}

fn full_jod<R: Runtime>(w: u32, h: u32) -> f32 {
    let (r, d) = noise_pair(w as usize, h as usize);
    let mut c = Cvvdp::<R>::new(
        R::client(&Default::default()),
        w,
        h,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("Cvvdp::new");
    c.compute_dkl_jod(&r, &d, ppd()).expect("compute_dkl_jod")
}

/// The pair is two unrelated byte patterns, so any real score sits far
/// below the 10.0 an unwritten D plane produces.
fn assert_real_score(jod: f32, what: &str) {
    assert!(jod.is_finite(), "{what}: JOD {jod} is not finite");
    assert!(
        jod < 9.0,
        "{what}: JOD {jod} on a heavily distorted pair (dispatches dropped?)"
    );
}

#[test]
fn wgpu_full_scores_above_the_dispatch_limit() {
    for (w, h) in SIZES {
        assert!(u64::from(w) * u64::from(h) > 4_194_240);
        let jod = full_jod::<WgpuRuntime>(w, h);
        eprintln!("wgpu Full {w}x{h}: JOD {jod:.6}");
        assert_real_score(jod, &format!("wgpu Full {w}x{h}"));
    }
}

#[test]
fn wgpu_strip_modes_match_full_above_the_dispatch_limit() {
    let (w, h) = (3000_u32, 2000_u32);
    let (r, d) = noise_pair(w as usize, h as usize);
    let jod_full = full_jod::<WgpuRuntime>(w, h);
    assert_real_score(jod_full, "wgpu Full 3000x2000");

    let mut e = Cvvdp::<WgpuRuntime>::new_strip(
        WgpuRuntime::client(&Default::default()),
        w,
        h,
        2048,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip");
    e.warm_reference(&r).expect("warm_reference");
    let jod_e = e.compute_dkl_jod_with_warm_ref(&d, ppd()).expect("Mode E");
    drop(e);

    let mut b = Cvvdp::<WgpuRuntime>::new_strip_pair(
        WgpuRuntime::client(&Default::default()),
        w,
        h,
        2048,
        CvvdpParams::PLACEHOLDER,
    )
    .expect("new_strip_pair");
    let jod_b = b.compute_dkl_jod(&r, &d, ppd()).expect("Mode B");

    eprintln!(
        "wgpu 3000x2000 h_body=2048: Full {jod_full:.6}, Mode E {jod_e:.6}, Mode B {jod_b:.6}"
    );
    for (mode, jod) in [("Mode E", jod_e), ("Mode B", jod_b)] {
        assert_real_score(jod, &format!("wgpu {mode} 3000x2000"));
        let diff = (jod - jod_full).abs();
        assert!(
            diff <= PARITY_TOL_JOD,
            "wgpu {mode} 3000x2000: JOD {jod} vs Full {jod_full}, |diff| {diff} > {PARITY_TOL_JOD}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn wgpu_matches_cuda_above_the_dispatch_limit() {
    for (w, h) in SIZES {
        let jod_wgpu = full_jod::<WgpuRuntime>(w, h);
        let jod_cuda = full_jod::<cubecl::cuda::CudaRuntime>(w, h);
        let diff = (jod_wgpu - jod_cuda).abs();
        eprintln!("{w}x{h}: wgpu {jod_wgpu:.6}, CUDA {jod_cuda:.6}, |diff| {diff:.3e}");
        assert_real_score(jod_wgpu, &format!("wgpu {w}x{h}"));
        assert_real_score(jod_cuda, &format!("CUDA {w}x{h}"));
        assert!(
            diff <= PARITY_TOL_JOD,
            "{w}x{h}: wgpu {jod_wgpu} vs CUDA {jod_cuda}, |diff| {diff} > {PARITY_TOL_JOD}"
        );
    }
}
