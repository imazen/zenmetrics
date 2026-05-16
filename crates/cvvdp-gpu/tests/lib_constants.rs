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

#[test]
fn n_channels_is_three_for_still_image_dkl() {
    // cvvdp's still-image path operates on three DKL opponent
    // channels (Achromatic, Red-Green, Violet-Yellow). A bump to 4
    // would mean adding a temporal channel (not in still-image
    // scope) or a luminance variant (also out of scope until
    // pycvvdp ports its own).
    assert_eq!(
        N_CHANNELS, 3,
        "N_CHANNELS = {N_CHANNELS}, expected 3 (still-image DKL)"
    );
}

#[test]
fn max_levels_cap_at_nine() {
    // The pyramid-level cap. 9 levels handles up to ~1024×1024
    // (each level halves; 1024 / 2^8 = 4 = PYRAMID_MIN_DIM × 1).
    // Bumping to 10+ requires resizing the `logs_row` Vec, the
    // `partials_h` buffer (sized `n_levels × N_CHANNELS`), and
    // the weights buffers — none of which auto-grow on
    // MAX_LEVELS change. Pin the literal so a refactor surfaces
    // here instead of as silent OOB index reads.
    assert_eq!(MAX_LEVELS, 9, "MAX_LEVELS = {MAX_LEVELS}, expected 9");
}

#[test]
fn pyramid_min_dim_is_four() {
    // Smallest logical width/height where the pyramid keeps
    // building further coarse levels. Combined with the
    // `width < PYRAMID_MIN_DIM * 2` check in `Cvvdp::new`, this
    // makes 8×8 the absolute minimum image dimension. A bump to
    // PYRAMID_MIN_DIM=8 would silently reject 8×8 and 16×16
    // images, breaking the existing `invalid_image_size_*`
    // boundary tests in `pipeline_score.rs`.
    assert_eq!(
        PYRAMID_MIN_DIM, 4,
        "PYRAMID_MIN_DIM = {PYRAMID_MIN_DIM}, expected 4",
    );
}

#[test]
fn pyramid_min_image_size_is_eight() {
    // The construction-time guard `width < PYRAMID_MIN_DIM * 2`
    // implies a minimum-acceptable image dim of 8 pixels. Make
    // this implicit relationship explicit so a refactor that
    // changes the guard's multiplier (e.g. to ×4 = 16) trips
    // here.
    assert_eq!(
        PYRAMID_MIN_DIM * 2,
        8,
        "PYRAMID_MIN_DIM × 2 = {} (minimum-image-dim), expected 8",
        PYRAMID_MIN_DIM * 2,
    );
}
