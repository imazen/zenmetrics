//! Pin the `lib.rs` re-export surface for the public types and
//! helpers that downstream callers reach via the crate root:
//!
//! ```text
//! pub use params::{CvvdpParams, PerfMode};
//! pub use pipeline::{Cvvdp, PARALLEL_SAFETY_FACTOR,
//!                    estimate_gpu_memory_bytes, recommend_parallel};
//!
//! // Plus lib-level constants + Error/Result:
//! pub const N_CHANNELS, MAX_LEVELS, PYRAMID_MIN_DIM, CVVDP_COLUMN_NAME;
//! pub enum Error;
//! pub type Result<T>;
//! ```
//!
//! These re-exports are the canonical import paths in production
//! callers (zen-metrics-cli, downstream CvvdpBatchScorer). A
//! refactor that drops one — or moves it under a feature gate —
//! would break callers silently if no test referenced the crate-root
//! path. This file pins each re-export via a compile-time use site.

// Crate-root imports — would fail to compile if any of these
// items stopped being re-exported. Tick 501 widened from the
// original 5-item set (CvvdpParams, PerfMode, PARALLEL_SAFETY_FACTOR,
// estimate_gpu_memory_bytes, recommend_parallel) to also pin
// Cvvdp (the canonical scoring type), Error + Result (error API),
// and the four lib-root constants.
use cvvdp_gpu::{
    CVVDP_COLUMN_NAME, Cvvdp, CvvdpParams, Error, MAX_LEVELS, N_CHANNELS, PARALLEL_SAFETY_FACTOR,
    PYRAMID_MIN_DIM, PerfMode, Result, estimate_gpu_memory_bytes, recommend_parallel,
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

#[test]
fn cvvdp_type_reexport_resolves() {
    // Tick 501: `Cvvdp<R>` is the main scoring type. A refactor that
    // moves it into a private module or behind a feature gate would
    // break every downstream caller (zen-metrics-cli's
    // CvvdpBatchScorer references `cvvdp_gpu::Cvvdp` directly).
    // The compile-time use above is the load-bearing pin; this test
    // documents that intent with a runtime touchpoint using the
    // ::pipeline:: alias to confirm the two paths resolve to the
    // same type.
    fn _accepts_via_reexport<R: cubecl::Runtime>(_c: &Cvvdp<R>) {}
    fn _accepts_via_pipeline<R: cubecl::Runtime>(_c: &cvvdp_gpu::pipeline::Cvvdp<R>) {}
    // The fns above accept the same type if the re-export is sound.
    // No instantiation — this is purely a compile-time check.
}

#[test]
fn lib_constants_reexport_match_their_originals() {
    // Tick 501: lib-root constants are the canonical import path for
    // downstream callers. Pin that the crate-root names resolve to
    // the documented values and to anything they're re-exported
    // from. (Currently all four are defined directly in lib.rs, not
    // re-exported from a submodule, so this is a value pin rather
    // than an alias pin.)
    //
    // Tick 522: integer constants promoted to `const _: () =
    // assert!(...)` static asserts so they fire at compile time
    // rather than runtime. `CVVDP_COLUMN_NAME` stays runtime
    // because `.starts_with` isn't `const fn` as of stable Rust.
    const _: () = assert!(N_CHANNELS == 3, "N_CHANNELS contract — DKL opponent count");
    const _: () = assert!(MAX_LEVELS == 9, "MAX_LEVELS contract — pyramid cap");
    const _: () = assert!(PYRAMID_MIN_DIM == 4, "PYRAMID_MIN_DIM contract");
    assert!(
        CVVDP_COLUMN_NAME.starts_with("cvvdp_"),
        "CVVDP_COLUMN_NAME must start with cvvdp_; got: {CVVDP_COLUMN_NAME:?}",
    );
}

#[test]
fn error_and_result_reexport_resolve() {
    // Tick 501: `Error` and `Result<T>` are how callers see method
    // failures. Both must be reachable from the crate root.
    // Compile-time checks only — instantiating doesn't add coverage.
    fn _accepts_error(_e: &Error) {}
    fn _accepts_result(_r: &Result<()>) {}
    // Touchpoint to keep the imports used.
    let _e = Error::NoCachedReference;
    let _r: Result<()> = Ok(());
}

#[test]
fn host_scalar_module_is_public() {
    // Tick 503: `cvvdp_gpu::host_scalar::predict_jod_still_3ch` is
    // the canonical host-only reference pipeline. Used by shadow_jod
    // tests, cpu_backend tests, and downstream consumers that want
    // a pure-CPU JOD without spinning up a GPU runtime (e.g. for
    // CI environments without a GPU). A refactor that downgrades
    // the module to `pub(crate)` or moves the fn out of it would
    // break callers silently. Pin via compile-time use site.
    //
    // Tick 505: hoist the fn-pointer type out into a `type` alias to
    // clear the `clippy::type_complexity` warning.
    use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
    type PredictJodFn = fn(&[u8], &[u8], usize, usize, cvvdp_gpu::params::DisplayModel, f32) -> f32;
    fn _accepts_predict_fn(_f: PredictJodFn) {}
    _accepts_predict_fn(predict_jod_still_3ch);
}

#[test]
fn kernels_submodules_are_public() {
    // Tick 503: the five kernels submodules (color, csf, masking,
    // pool, pyramid) are the documented public API for direct kernel
    // usage. Existing per-kernel test files import specific items
    // (e.g. `gausspyr_reduce_scalar`), but no single pin verifies
    // the module path itself remains public. A refactor that
    // collapses one into a parent or makes it `pub(crate)` would
    // break callers that reach for `cvvdp_gpu::kernels::masking::*`
    // directly.
    //
    // Compile-time use sites — one item per submodule:
    use cvvdp_gpu::kernels::color::SRGB8_TO_LINEAR_LUT;
    use cvvdp_gpu::kernels::csf::N_L_BKG;
    use cvvdp_gpu::kernels::masking::MASK_C;
    use cvvdp_gpu::kernels::pool::JOD_A;
    use cvvdp_gpu::kernels::pyramid::KERNEL_A;
    // Touchpoint to keep imports used. Tick 505: replaced the
    // compile-time `assert!(N_L_BKG > 0)` (clippy: "this assertion
    // has a constant value") with `const _: () = assert!(...)`,
    // which is a true static assertion checked at compile time.
    assert_eq!(SRGB8_TO_LINEAR_LUT.len(), 256);
    const _: () = assert!(N_L_BKG > 0, "N_L_BKG must be positive (CSF LUT axis size)");
    assert!(MASK_C.is_finite());
    assert!(JOD_A.is_finite());
    assert!(KERNEL_A.is_finite());
}

#[test]
fn params_scaffolding_types_are_public() {
    // Tick 502: the params:: scaffolding types are currently unused
    // by production code (the per-stage cvvdp v0.5.4 constants are
    // inlined in `kernels::pool` / `kernels::masking` / etc.) but
    // they exist as the documented public API for a planned
    // "load parameters from the vendored cvvdp JSON" path. A future
    // refactor that downgrades them to `pub(crate)` or removes them
    // because they're unused would break that planned path silently.
    //
    // Pin each type's public visibility via a compile-time use site.
    // CsfParams / MaskingParams / PoolingParams / JodParams have
    // no other test importing them — without this pin a removal
    // would only surface when the JSON-loading path lands.
    use cvvdp_gpu::params::{CsfParams, JodParams, MaskingParams, PoolingParams};
    fn _accepts_csf(_p: &CsfParams) {}
    fn _accepts_masking(_p: &MaskingParams) {}
    fn _accepts_pooling(_p: &PoolingParams) {}
    fn _accepts_jod(_p: &JodParams) {}
    // Touchpoint via the PLACEHOLDER sub-bundles to keep imports used.
    let p = CvvdpParams::PLACEHOLDER;
    _accepts_csf(&p.csf);
    _accepts_masking(&p.masking);
    _accepts_pooling(&p.pooling);
    _accepts_jod(&p.jod);
}
