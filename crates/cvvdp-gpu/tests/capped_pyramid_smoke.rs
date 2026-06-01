//! Smoke tests for the restored
//! [`MemoryMode::CappedPyramid { levels }`] variant (Option B safety
//! net, task #79 follow-on 2026-05-26).
//!
//! `CappedPyramid` is **JOD-shifting** — it does not preserve the
//! same JOD value as Full mode. These tests verify only:
//!   - The constructor succeeds for valid `levels` values.
//!   - `compute_dkl_jod` returns a finite, in-range JOD score.
//!   - The capped-pyramid memory estimate is strictly less than the
//!     full-pyramid estimate when `levels` is below the natural depth.
//!
//! Full JOD parity bounds vs the natural-depth pipeline are
//! deliberately *not* asserted here — that's the JOD-shift the user
//! is opting into. The pre-rollback bench measured ≤ 0.005 JOD at
//! k=8 (very close to natural 9 in the typical 4K case); that bound
//! is documented in `crate::MemoryMode::CappedPyramid` but is not a
//! gate here because the goldens needed for it sit behind the
//! `parity-goldens` feature.

#![cfg(feature = "cubecl-types")]

mod common;
use common::{Backend, apply_offset_dist, synth_pair_ref};

use cubecl::Runtime;
use cvvdp_gpu::{
    Cvvdp, CvvdpParams, MemoryMode, estimate_gpu_memory_bytes, estimate_gpu_memory_bytes_capped,
};

#[test]
fn capped_pyramid_constructor_succeeds_at_256_squared_levels_5() {
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_capped_pyramid(client, 256, 256, CvvdpParams::PLACEHOLDER, 5)
        .expect("new_capped_pyramid");
    let (w, h) = cvvdp.dimensions();
    assert_eq!((w, h), (256, 256));
    // is_strip_mode should be false — CappedPyramid is independent
    // of strip mode.
    assert!(!cvvdp.is_strip_mode());
}

#[test]
fn capped_pyramid_compute_dkl_jod_returns_finite_reasonable_score() {
    let client = Backend::client(&Default::default());
    let mut cvvdp =
        Cvvdp::<Backend>::new_capped_pyramid(client, 256, 256, CvvdpParams::PLACEHOLDER, 5)
            .expect("new_capped_pyramid");
    let r = synth_pair_ref(256, 256);
    let d = apply_offset_dist(&r);
    let ppd = cvvdp_gpu::params::DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let jod = cvvdp.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod");
    assert!(
        jod.is_finite() && (0.0..=10.0 + 1e-3).contains(&jod),
        "jod = {jod}",
    );
}

#[test]
fn capped_pyramid_estimate_less_than_full_when_cap_below_natural() {
    // At 4096² the natural pyramid is ~9 levels. Capping to 5 should
    // shrink memory substantially (deeper levels are tiny but the
    // d_scratch / pyramid / weber buffers still pay for them).
    let full = estimate_gpu_memory_bytes(4096, 4096).expect("full estimate");
    let capped5 = estimate_gpu_memory_bytes_capped(4096, 4096, 5).expect("capped estimate");
    assert!(
        capped5 < full,
        "capped5 ({capped5}) should be < full ({full})"
    );
    // The savings should be noticeable — deepest bands aren't free.
    // Bound is conservative; the actual ratio depends on per-level
    // pixel-count contribution.
    assert!(
        capped5 + 1024 < full,
        "expected at least small savings, got capped5={capped5} full={full}"
    );
}

#[test]
fn capped_pyramid_levels_above_natural_clamps_to_full() {
    // 64×64 has only a few natural pyramid levels (< 9). Asking for
    // 20 levels should silently clamp to the natural depth and
    // match the full-pyramid estimate byte-for-byte (modulo the
    // constructor's branch — the estimator clamps the same way).
    let full = estimate_gpu_memory_bytes(64, 64).expect("full estimate");
    let cap_huge = estimate_gpu_memory_bytes_capped(64, 64, 20).expect("capped estimate");
    assert_eq!(cap_huge, full);
}

#[test]
fn capped_pyramid_levels_zero_errors_at_construction() {
    let client = Backend::client(&Default::default());
    let r = Cvvdp::<Backend>::new_capped_pyramid(client, 64, 64, CvvdpParams::PLACEHOLDER, 0);
    // Don't pin the exact error variant text — only that we got an
    // error (avoids brittle string comparisons against the
    // ModeUnsupported message). Use is_err() not expect_err because
    // Cvvdp<R> doesn't impl Debug.
    assert!(r.is_err());
}

#[test]
fn capped_pyramid_levels_zero_estimator_returns_none() {
    assert_eq!(estimate_gpu_memory_bytes_capped(64, 64, 0), None);
}

#[test]
fn capped_pyramid_too_small_image_returns_none() {
    // Below PYRAMID_MIN_DIM × 2 = 8 — same precondition as
    // `estimate_gpu_memory_bytes`.
    assert_eq!(estimate_gpu_memory_bytes_capped(4, 4, 5), None);
}

#[test]
fn capped_pyramid_via_memory_mode_constructor() {
    // The unified MemoryMode entry point should dispatch to
    // new_capped_pyramid when given CappedPyramid { levels }.
    let client = Backend::client(&Default::default());
    let cvvdp = Cvvdp::<Backend>::new_with_memory_mode(
        client,
        128,
        128,
        CvvdpParams::PLACEHOLDER,
        MemoryMode::CappedPyramid { levels: 4 },
    )
    .expect("new_with_memory_mode(CappedPyramid)");
    assert_eq!(cvvdp.dimensions(), (128, 128));
    assert!(!cvvdp.is_strip_mode());
}
