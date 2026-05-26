//! Tests for `CvvdpOpaque::new_with_geometry` /
//! `new_with_geometry_and_memory_mode` — pins that the new geometry-
//! aware constructors actually feed through to
//! `Cvvdp::<R>::new_with_geometry` instead of silently downcasting to
//! `STANDARD_4K` (the bug `CvvdpOpaque` had until 2026-05-26).
//!
//! Same compile-time backend gating as `tests/opaque.rs` — runtime
//! tests need a real GPU/CPU backend (CUDA/wgpu/cpu); skipped at
//! compile time when none is enabled.
//!
//! Run with:
//!
//!     cargo test -p cvvdp-gpu --features cuda --test opaque_geometry_api

#![cfg(all(
    feature = "cubecl-types",
    any(feature = "cuda", feature = "wgpu", feature = "cpu")
))]

use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};
use cvvdp_gpu::{Backend, CvvdpOpaque};

#[cfg(feature = "cuda")]
const BACKEND_E: Backend = Backend::Cuda;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
const BACKEND_E: Backend = Backend::Wgpu;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
const BACKEND_E: Backend = Backend::Cpu;

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = ((x.wrapping_add(seed)) & 0xff) as u8;
            let g = ((y.wrapping_add(seed.wrapping_mul(3))) & 0xff) as u8;
            let b = ((x ^ y ^ seed) & 0xff) as u8;
            out.extend_from_slice(&[r, g, b]);
        }
    }
    out
}

/// Gate 1 (bit-identical default equivalence): `new(backend, w, h, params)`
/// and `new_with_geometry(.., STANDARD_4K)` must produce the same JOD on
/// the same input. Both code paths route through
/// `Cvvdp::<R>::new_with_geometry(.., STANDARD_4K)` after the refactor,
/// so any divergence here indicates the dispatch is broken.
#[test]
fn opaque_new_matches_new_with_geometry_standard_4k() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 11);

    let mut implicit = CvvdpOpaque::new(BACKEND_E, w, h, CvvdpParams::PLACEHOLDER)
        .expect("CvvdpOpaque::new (implicit STANDARD_4K)");
    let implicit_score = implicit
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("implicit compute_srgb_u8");

    let mut explicit = CvvdpOpaque::new_with_geometry(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::STANDARD_4K,
    )
    .expect("CvvdpOpaque::new_with_geometry(STANDARD_4K)");
    let explicit_score = explicit
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("explicit compute_srgb_u8");

    let abs = (implicit_score.value - explicit_score.value).abs();
    // Both paths run the same kernels on the same client-allocated
    // buffers with the same PPD — JOD must match within f32 noise.
    // Use abs (not rel) because JOD ≈ 10 for matching pairs and the
    // rel-tol denominator vanishes near JOD = 10. 1e-6 is well below
    // the 1e-4 STANDARD_4K parity floor used elsewhere.
    assert!(
        abs < 1e-6,
        "implicit-STANDARD_4K JOD {} vs explicit-STANDARD_4K JOD {} differ by {} \
         — `new_with_geometry(STANDARD_4K)` must be bit-identical to `new()`",
        implicit_score.value,
        explicit_score.value,
        abs
    );
}

/// Gate 2 (geometry is actually consumed): two distinct
/// `DisplayGeometry` values must produce DIFFERENT JOD scores on the
/// same input. PPD shifts the spatial frequencies the castleCSF table
/// is queried with, which materially changes the metric.
///
/// IPHONE_14_PRO has ≈454 PPD (5.5″ panel at 0.508 m); PANEL_65IN_4K
/// has ≈57 PPD (65″ panel at 1.98 m) — an 8× PPD ratio, which lands
/// the two geometries in very different parts of the CSF.
#[test]
fn opaque_new_with_geometry_actually_uses_geometry() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 11);

    let mut phone = CvvdpOpaque::new_with_geometry(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::IPHONE_14_PRO,
    )
    .expect("CvvdpOpaque::new_with_geometry(IPHONE_14_PRO)");
    let phone_score = phone
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("phone compute_srgb_u8");

    let mut tv = CvvdpOpaque::new_with_geometry(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::PANEL_65IN_4K,
    )
    .expect("CvvdpOpaque::new_with_geometry(PANEL_65IN_4K)");
    let tv_score = tv
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("tv compute_srgb_u8");

    let diff = (phone_score.value - tv_score.value).abs();
    // Pre-refactor `CvvdpOpaque` permanently downcast geometry to
    // STANDARD_4K, making phone_score == tv_score. After the refactor
    // the two must differ — the test threshold (1e-3 JOD) is well
    // above f32 noise but well below the actual PPD-induced shift
    // (typically 0.1-1.0 JOD between these two geometries).
    assert!(
        diff > 1e-3,
        "IPHONE_14_PRO JOD {} and PANEL_65IN_4K JOD {} are too close (diff {} ≤ 1e-3) \
         — `new_with_geometry` may not be threading the geometry to the kernels",
        phone_score.value,
        tv_score.value,
        diff
    );
}

/// Gate 3 (memory-mode + geometry composition): the
/// `new_with_geometry_and_memory_mode` constructor accepts both a
/// geometry AND a [`MemoryMode`]. Auto + Full both succeed on a
/// fixture that fits the default VRAM budget; Strip / Tile surface
/// `ModeUnsupported` (mirrors the no-geometry `new_with_memory_mode`
/// behaviour). Pin both successful and rejection paths so a future
/// refactor of the mode-dispatch can't silently change behaviour.
#[test]
fn opaque_new_with_geometry_and_memory_mode_modes() {
    let w = 64_u32;
    let h = 64_u32;
    let ref_buf = make_image(w, h, 0);
    let dis_buf = make_image(w, h, 11);

    // Full + custom geometry: must succeed and produce a score.
    let mut full = CvvdpOpaque::new_with_geometry_and_memory_mode(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::IPHONE_14_PRO,
        cvvdp_gpu::MemoryMode::Full,
    )
    .expect("CvvdpOpaque::new_with_geometry_and_memory_mode(Full, IPHONE_14_PRO)");
    let _ = full
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("Full+IPHONE compute_srgb_u8");

    // Auto + custom geometry: must succeed (64×64 always fits Auto's
    // VRAM budget).
    let mut auto = CvvdpOpaque::new_with_geometry_and_memory_mode(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::IPHONE_14_PRO,
        cvvdp_gpu::MemoryMode::Auto,
    )
    .expect("CvvdpOpaque::new_with_geometry_and_memory_mode(Auto, IPHONE_14_PRO)");
    let _ = auto
        .compute_srgb_u8(&ref_buf, &dis_buf)
        .expect("Auto+IPHONE compute_srgb_u8");

    // Strip: ModeUnsupported (cvvdp-gpu has no Strip impl on the
    // opaque API surface). The error must surface BEFORE backend
    // allocation, matching `new_with_memory_mode`.
    // (Use `match` rather than `expect_err` because `CvvdpOpaque`
    // doesn't impl `Debug`.)
    let strip_result = CvvdpOpaque::new_with_geometry_and_memory_mode(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::IPHONE_14_PRO,
        cvvdp_gpu::MemoryMode::Strip {
            h_body: None,
            capped_levels: None,
        },
    );
    match strip_result {
        Err(cvvdp_gpu::Error::ModeUnsupported(_)) => {}
        Err(other) => panic!("Strip: expected ModeUnsupported, got {other:?}"),
        Ok(_) => panic!("Strip: expected ModeUnsupported, got Ok(_)"),
    }

    // Tile: ModeUnsupported (cvvdp-gpu has no Tile impl either).
    let tile_result = CvvdpOpaque::new_with_geometry_and_memory_mode(
        BACKEND_E,
        w,
        h,
        CvvdpParams::PLACEHOLDER,
        DisplayGeometry::IPHONE_14_PRO,
        cvvdp_gpu::MemoryMode::Tile { w: 64, h: 64 },
    );
    match tile_result {
        Err(cvvdp_gpu::Error::ModeUnsupported(_)) => {}
        Err(other) => panic!("Tile: expected ModeUnsupported, got {other:?}"),
        Ok(_) => panic!("Tile: expected ModeUnsupported, got Ok(_)"),
    }
}
