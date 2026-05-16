//! Pin the two `CvvdpParams::PLACEHOLDER` fields that the pipeline
//! actually consumes: `display` and `perf_mode`. The other fields
//! (`csf`, `masking`, `pooling`, `jod`) are scaffolding for a
//! planned vendored-JSON load path that hasn't landed — those
//! numbers are placeholder and unread.
//!
//! Documented contract from params.rs: "Most callers want
//! [`PerfMode::Strict`]" + the display defaults to
//! `DisplayModel::STANDARD_4K`. A refactor that flipped the
//! placeholder default to `PerfMode::Fast` would silently change
//! the parity-test calibration baseline — every parity test
//! constructs `Cvvdp::new(..., PLACEHOLDER)` and would now run
//! under Fast instead of Strict. Pin so that flip surfaces here
//! before invalidating dozens of golden tests.
#![allow(clippy::excessive_precision)]

use cvvdp_gpu::CvvdpParams;
use cvvdp_gpu::PerfMode;
use cvvdp_gpu::params::DisplayModel;

// Tick 552: compile-time pin of PLACEHOLDER.perf_mode = Strict.
// Every parity test constructs `Cvvdp::new(..., PLACEHOLDER)`
// and inherits this perf-mode default. A refactor that flipped
// it to Fast would silently change the calibration baseline for
// every golden test. `matches!` is `const`-callable (derived
// `PartialEq` on enums isn't yet, in stable Rust as of this
// crate's MSRV 1.93), so the pattern-match form is the const-
// compatible way to pin this without an enum-equality call.
// Same load-bearing pattern as the STANDARD_4K display field
// pins (tick 551).
const _: () = assert!(
    matches!(CvvdpParams::PLACEHOLDER.perf_mode, PerfMode::Strict),
    "PLACEHOLDER.perf_mode drifted from PerfMode::Strict (the parity-calibrated baseline)",
);

#[test]
fn placeholder_display_is_standard_4k() {
    // PLACEHOLDER.display is the DisplayModel consumed by
    // `Cvvdp::new`. STANDARD_4K is what every parity golden was
    // captured against. A refactor that drops in a different
    // display would silently shift every JOD score because the
    // sRGB → luminance step depends on y_peak/y_black/y_refl.
    let p = CvvdpParams::PLACEHOLDER;
    // STANDARD_4K is a const Self; compare field-by-field via
    // bit pattern. (Both literals are f32; `.to_bits()` sidesteps
    // PartialEq on the struct + clippy::float_cmp.)
    let std4k = DisplayModel::STANDARD_4K;
    assert_eq!(
        p.display.y_peak.to_bits(),
        std4k.y_peak.to_bits(),
        "PLACEHOLDER.display.y_peak = {}, expected {} (STANDARD_4K)",
        p.display.y_peak,
        std4k.y_peak,
    );
    assert_eq!(
        p.display.y_black.to_bits(),
        std4k.y_black.to_bits(),
        "PLACEHOLDER.display.y_black = {}, expected {} (STANDARD_4K)",
        p.display.y_black,
        std4k.y_black,
    );
    assert_eq!(
        p.display.y_refl.to_bits(),
        std4k.y_refl.to_bits(),
        "PLACEHOLDER.display.y_refl = {}, expected {} (STANDARD_4K)",
        p.display.y_refl,
        std4k.y_refl,
    );
}

#[test]
fn placeholder_perf_mode_is_strict() {
    // PLACEHOLDER.perf_mode is what every parity test inherits
    // (each constructs `Cvvdp::new(..., PLACEHOLDER)`). A flip to
    // Fast would silently change the calibration baseline. Pin
    // so it trips here, with a specific message, before the
    // parity gates run their many-second GPU dispatches.
    assert_eq!(
        CvvdpParams::PLACEHOLDER.perf_mode,
        PerfMode::Strict,
        "PLACEHOLDER.perf_mode = {:?}, expected PerfMode::Strict (the parity-calibrated baseline)",
        CvvdpParams::PLACEHOLDER.perf_mode,
    );
}

#[test]
fn perf_mode_strict_is_default_used_throughout_parity_tests() {
    // Document the relationship: every parity test in this crate
    // (shadow_jod, pipeline_score, pipeline_color, cpu_backend)
    // constructs `Cvvdp::new(..., PLACEHOLDER)`, which inherits
    // `perf_mode: PerfMode::Strict`. The `compute_dkl_jod`
    // dispatch then gates fast-path optimizations on
    // `params.perf_mode == Fast`. So the parity baseline is
    // tied at compile time to PerfMode::Strict via PLACEHOLDER.
    //
    // This test pins the implied transitive contract — if someone
    // changes the default-construction path, this test articulates
    // why that change has cascading consequences.
    let p = CvvdpParams::PLACEHOLDER;
    assert_eq!(
        p.perf_mode,
        PerfMode::Strict,
        "every parity-test baseline depends on PerfMode::Strict via PLACEHOLDER",
    );
    // PerfMode is also Copy + PartialEq + Eq derive — pin those
    // by exercising them so a refactor that drops one of the
    // derives breaks build here.
    let copy_a = PerfMode::Strict;
    let copy_b = copy_a;
    assert_eq!(copy_a, copy_b);
    assert_ne!(PerfMode::Strict, PerfMode::Fast);
}
