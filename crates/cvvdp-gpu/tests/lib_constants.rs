//! Pin the three crate-level constants exposed from `lib.rs`:
//! `N_CHANNELS`, `MAX_LEVELS`, `PYRAMID_MIN_DIM`. These are
//! load-bearing for every dimension calculation in the pipeline:
//! the GPU buffer sizing in `Cvvdp::new`, the per-level pyramid
//! halving in `pyramid_levels`, the bounds check on construction,
//! and downstream tests' arithmetic on `n_levels × N_CHANNELS`
//! partials buffers (which would silently mis-size if N_CHANNELS
//! drifted).
//!
//! Sibling to ticks 393-397 (per-kernel-module constant pins) +
//! tick 401 (masking constants). Same loud-failure-on-silent-edit
//! pattern: a refactor that bumps `MAX_LEVELS = 10` without
//! resizing every test's expected level count, or `N_CHANNELS = 4`
//! to support a future luminance-only test variant, would surface
//! here with the specific constant name + expected value.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::{MAX_LEVELS, N_CHANNELS, PYRAMID_MIN_DIM};

// Tick 523: promote the runtime asserts below to compile-time
// static asserts. The constants are fundamental dimension parameters
// that fan out through every dim calculation in the pipeline; a
// silent edit cannot be allowed to survive even a `cargo build`
// against this test target. Same pattern as tick 522 on lib_reexports.rs
// and tick 505 on N_L_BKG.
const _: () = assert!(
    N_CHANNELS == 3,
    "N_CHANNELS contract — still-image DKL (Achromatic, Red-Green, Violet-Yellow)",
);
const _: () = assert!(
    MAX_LEVELS == 9,
    "MAX_LEVELS cap — bumping requires resizing logs_row, partials_h, weights buffers",
);
const _: () = assert!(
    PYRAMID_MIN_DIM == 4,
    "PYRAMID_MIN_DIM contract — smallest logical level dim",
);
const _: () = assert!(
    PYRAMID_MIN_DIM * 2 == 8,
    "PYRAMID_MIN_DIM × 2 boundary — Cvvdp::new's minimum-image-dim guard",
);

#[test]
fn n_channels_is_three_for_still_image_dkl() {
    // cvvdp's still-image path operates on three DKL opponent
    // channels (Achromatic, Red-Green, Violet-Yellow). The bound
    // is pinned at compile time via the const _: () = assert!(...)
    // block above; this test exercises the import + leaves the
    // CHANGELOG-tested 'tests/lib_constants.rs::n_channels_is_three_for_still_image_dkl'
    // name resolvable from the test runner.
    assert_eq!(N_CHANNELS, 3);
}

#[test]
fn max_levels_cap_at_nine() {
    // See the module-level static assert. This test exists for
    // test-runner-visible naming (the CHANGELOG referenced
    // 'max_levels_cap_at_nine' as the pin name in tick 402).
    assert_eq!(MAX_LEVELS, 9);
}

#[test]
fn pyramid_min_dim_is_four() {
    // See the module-level static assert. Same naming-preservation
    // rationale as the siblings.
    assert_eq!(PYRAMID_MIN_DIM, 4);
}

#[test]
fn pyramid_min_image_size_is_eight() {
    // See the module-level static assert. Same naming-preservation
    // rationale as the siblings.
    assert_eq!(PYRAMID_MIN_DIM * 2, 8);
}
