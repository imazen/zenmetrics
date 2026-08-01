//! Tests for the `ZENMETRICS_FORCE_BACKEND_INIT_FAIL` seam of
//! [`zenmetrics_gpu_core::validate_backend`] (imazen/zenmetrics#37).
//!
//! Everything lives in ONE `#[test]` fn: the seam is process-global env
//! state and the scenarios set *different* values, so parallel test
//! threads would race. No scenario here runs an **unforced** probe — that
//! would touch real GPU state and make the test host-dependent; the
//! forced paths return before any runtime is constructed.
//!
//! Env mutation is `unsafe` on edition 2024; this integration test is a
//! separate compilation unit from the library crate (same pattern as
//! zenmetrics-api's `tests/it/backend_resolve.rs`).
#![cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]

use zenmetrics_gpu_core::{Backend, validate_backend};

fn set_seam(value: Option<&str>) {
    // SAFETY: confined to this single serial test fn, the only place in
    // this binary that touches the variable.
    unsafe {
        match value {
            Some(v) => std::env::set_var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL", v),
            None => std::env::remove_var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL"),
        }
    }
}

#[test]
fn force_seam_fails_matching_backends_before_any_probe() {
    let prev = std::env::var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL").ok();

    #[cfg(feature = "cuda")]
    {
        // Tag list containing the backend → forced failure, and the reason
        // names the seam (so a fleet log reads as "forced", not "broken").
        set_seam(Some("cuda"));
        let err =
            validate_backend(Backend::Cuda).expect_err("cuda must fail when the seam names it");
        assert!(
            err.contains("ZENMETRICS_FORCE_BACKEND_INIT_FAIL"),
            "reason must name the seam: {err}"
        );

        // `all` and `1` force every backend.
        set_seam(Some("all"));
        assert!(validate_backend(Backend::Cuda).is_err());
        set_seam(Some("1"));
        assert!(validate_backend(Backend::Cuda).is_err());

        // Comma-separated lists and case-insensitivity.
        set_seam(Some("wgpu, CUDA"));
        assert!(validate_backend(Backend::Cuda).is_err());

        // A list NOT naming cuda must not force it. We can't assert the
        // unforced result (that would run a real probe); instead assert
        // via the cpu arm below, which shares the seam parsing.
    }

    #[cfg(feature = "cpu")]
    {
        // The cubecl-cpu arm validates trivially (in-process runtime), so
        // it can prove both seam polarities without touching a GPU.
        set_seam(Some("cuda,wgpu"));
        assert!(
            validate_backend(Backend::Cpu).is_ok(),
            "a seam list not naming 'cpu' must not force the cpu arm"
        );
        set_seam(Some("cpu"));
        assert!(validate_backend(Backend::Cpu).is_err());
        set_seam(Some("all"));
        assert!(validate_backend(Backend::Cpu).is_err());
        set_seam(None);
        assert!(
            validate_backend(Backend::Cpu).is_ok(),
            "cpu must validate trivially with the seam unset"
        );
    }

    // Restore the caller's environment so a later-added test can't inherit
    // a stale override.
    set_seam(prev.as_deref());
}
