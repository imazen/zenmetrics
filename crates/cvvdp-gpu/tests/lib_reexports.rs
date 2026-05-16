//! Pin the `lib.rs` re-export surface for the public types and
//! helpers that downstream callers reach via the crate root:
//!
//! ```text
//! pub use params::{CvvdpParams, PerfMode};
//! pub use pipeline::{Cvvdp, PARALLEL_SAFETY_FACTOR,
//!                    estimate_gpu_memory_bytes, recommend_parallel};
//! ```
//!
//! These re-exports are the canonical import paths in production
//! callers (zen-metrics-cli, downstream CvvdpBatchScorer). A
//! refactor that drops one — or moves it under a feature gate —
//! would break callers silently if no test referenced the crate-root
//! path. This file pins each re-export via a compile-time use site.

// Crate-root imports — would fail to compile if any of these
// items stopped being re-exported.
use cvvdp_gpu::{
    CvvdpParams, PARALLEL_SAFETY_FACTOR, PerfMode, estimate_gpu_memory_bytes, recommend_parallel,
};

#[test]
fn perf_mode_reexport_resolves() {
    // PerfMode is re-exported from params. The Default impl is what
    // CvvdpParams::PLACEHOLDER consumes.
    let _ = PerfMode::default();
}

#[test]
fn cvvdp_params_placeholder_reexport_resolves() {
    // PLACEHOLDER is the canonical default; downstream callers use
    // `CvvdpParams::PLACEHOLDER` to construct a Cvvdp without
    // hand-rolling each field.
    let _ = CvvdpParams::PLACEHOLDER;
}

#[test]
fn parallel_safety_factor_reexport_matches_pipeline_const() {
    // The re-export and the original pipeline::PARALLEL_SAFETY_FACTOR
    // must be the same value. A future refactor that splits them
    // would silently break the `recommend_parallel` doctest math.
    assert_eq!(
        PARALLEL_SAFETY_FACTOR,
        cvvdp_gpu::pipeline::PARALLEL_SAFETY_FACTOR,
        "crate-root and pipeline:: re-exports of PARALLEL_SAFETY_FACTOR diverged",
    );
}

#[test]
fn estimate_gpu_memory_bytes_reexport_matches_pipeline_fn() {
    // Both paths must return the same value for the same input.
    let a = estimate_gpu_memory_bytes(1024, 1024);
    let b = cvvdp_gpu::pipeline::estimate_gpu_memory_bytes(1024, 1024);
    assert_eq!(a, b, "re-export and pipeline:: paths diverged");
}

#[test]
fn recommend_parallel_reexport_matches_pipeline_fn() {
    // Same contract: re-export must delegate to the pipeline:: original.
    let a = recommend_parallel(8 * 1024 * 1024 * 1024, 1024, 1024);
    let b = cvvdp_gpu::pipeline::recommend_parallel(8 * 1024 * 1024 * 1024, 1024, 1024);
    assert_eq!(a, b, "re-export and pipeline:: paths diverged");
}
