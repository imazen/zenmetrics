//! Regression tests for imazen/zenmetrics#37: an **explicit** GPU backend
//! request must surface `Err` when that runtime cannot actually operate —
//! never a silent fallback, and never a plausible-looking score (the field
//! failure was `Backend::Cuda` on a driver-without-toolkit box returning
//! ssim2 = 100.0 from zero-initialized, never-written accumulators).
//!
//! The real broken-box condition (NVIDIA driver present, CUDA toolkit
//! absent) cannot be reproduced on a box that has the toolkit installed,
//! so these tests drive the dispatch/init-failure path through the
//! `ZENMETRICS_FORCE_BACKEND_INIT_FAIL` seam
//! (`zenmetrics_gpu_core::validate_backend` consults it per call, before
//! the probe cache and before touching any runtime). The seam exercises
//! the same code path a real probe failure takes: constructor →
//! `verify_gpu_backend_operational` → `Error::BackendUnavailable`.
//!
//! This is a DEDICATED test file (its own test binary = its own process)
//! because the seam is process-global env state: every test here wants it
//! set, and no test in any *other* binary can be affected. Env mutation is
//! `unsafe` on edition 2024; the library crate is
//! `#![forbid(unsafe_code)]` but integration tests are separate
//! compilation units (same pattern as `tests/it/backend_resolve.rs`).
#![cfg(all(feature = "ssim2", feature = "cuda"))]

use zenmetrics_api::{Backend, Error, MemoryMode, Metric, MetricKind, MetricParams, MetricSession};

/// Force every backend's liveness validation to fail for this process.
/// Idempotent (always the same value), so concurrent `#[test]` threads in
/// this binary can each call it without racing on distinct values.
fn force_backend_init_failure() {
    // SAFETY: single-binary-scoped, write-only, and every test in this
    // file sets the identical value — no test here ever needs the
    // variable unset, and no other test binary shares this process.
    unsafe {
        std::env::set_var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL", "all");
    }
}

fn expect_backend_unavailable(res: Result<Metric, Error>, context: &str) {
    match res {
        Ok(_) => panic!(
            "{context}: constructed a scorer on a force-failed explicit GPU backend — \
             a bogus score could have been produced (the #37 failure mode)"
        ),
        Err(Error::BackendUnavailable { backend, reason }) => {
            assert_eq!(backend, "cuda", "{context}: wrong backend tag");
            assert!(
                reason.contains("ZENMETRICS_FORCE_BACKEND_INIT_FAIL"),
                "{context}: reason should carry the probe detail, got: {reason}"
            );
        }
        Err(other) => {
            panic!("{context}: expected Error::BackendUnavailable, got: {other} ({other:?})")
        }
    }
}

/// The exact repro shape from issue #37: `new_with_memory_mode(Ssim2,
/// Cuda, 512, 512, .., Full)` returned `Ok` and then scored 100.0. With a
/// failing runtime it must now be `Err` at construction.
#[test]
fn issue_37_repro_shape_errs_at_construction() {
    force_backend_init_failure();
    let res = Metric::new_with_memory_mode(
        MetricKind::Ssim2,
        Backend::Cuda,
        512,
        512,
        MetricParams::default_for(MetricKind::Ssim2),
        MemoryMode::Full,
    );
    expect_backend_unavailable(res, "new_with_memory_mode(Ssim2, Cuda, 512x512, Full)");
}

/// Every enabled metric must refuse an explicit `Backend::Cuda` whose
/// runtime fails validation, through BOTH umbrella constructors — the
/// verification is metric-independent and sits before the per-metric
/// dispatch.
#[test]
fn all_enabled_metrics_err_on_broken_explicit_cuda() {
    force_backend_init_failure();
    let kinds = [
        MetricKind::Cvvdp,
        MetricKind::Butter,
        MetricKind::Ssim2,
        MetricKind::Dssim,
        MetricKind::Iwssim,
        MetricKind::Zensim,
    ];
    let mut asserted = 0usize;
    for kind in kinds {
        // Metrics not compiled into this build can't even construct their
        // params (feature-conditional coverage, decided at build time, not
        // a runtime data skip); the enabled ones must all hard-error.
        let Ok(params) = MetricParams::try_default_for(kind) else {
            continue;
        };
        expect_backend_unavailable(
            Metric::new(kind, Backend::Cuda, 64, 64, params.clone()),
            &format!("Metric::new({kind:?}, Cuda)"),
        );
        expect_backend_unavailable(
            Metric::new_with_memory_mode(kind, Backend::Cuda, 64, 64, params, MemoryMode::Auto),
            &format!("Metric::new_with_memory_mode({kind:?}, Cuda)"),
        );
        asserted += 1;
    }
    // The file's cfg gate guarantees ssim2 is enabled, so this loop can
    // never pass vacuously.
    assert!(asserted >= 1, "no metric was actually exercised");
}

/// `Backend::Auto` may fall back **by design**: with every GPU backend's
/// validation failing, `Auto` must resolve to a non-GPU backend instead of
/// erroring or picking a broken GPU.
#[test]
fn auto_resolution_falls_back_away_from_broken_gpus() {
    force_backend_init_failure();
    let resolved = Backend::resolve_auto();
    assert!(
        !matches!(resolved, Backend::Cuda | Backend::Wgpu),
        "Auto must not resolve to a GPU backend whose liveness probe fails, got {resolved:?}"
    );
}

/// Session-scoped construction is a construction too: a stream-bound
/// metric on an explicit broken GPU backend must fail at
/// `MetricSession::metric`, not hand back a scorer.
#[test]
fn session_metric_construction_errs_on_broken_explicit_cuda() {
    force_backend_init_failure();
    let session = MetricSession::acquire(Backend::Cuda)
        .expect("acquire only claims a slot; the runtime probe belongs to construction");
    let res = session.metric(
        MetricKind::Ssim2,
        64,
        64,
        MetricParams::default_for(MetricKind::Ssim2),
    );
    match res {
        Ok(_) => {
            panic!("session.metric constructed a scorer on a force-failed explicit GPU backend")
        }
        Err(Error::BackendUnavailable { backend, .. }) => assert_eq!(backend, "cuda"),
        Err(other) => panic!("expected Error::BackendUnavailable, got: {other} ({other:?})"),
    }
}

/// The error's `Display` must state the no-fallback contract so operators
/// reading a fleet log understand an explicit request intentionally
/// refused to degrade.
#[test]
fn backend_unavailable_display_names_the_contract() {
    force_backend_init_failure();
    let err = match Metric::new(
        MetricKind::Ssim2,
        Backend::Cuda,
        32,
        32,
        MetricParams::default_for(MetricKind::Ssim2),
    ) {
        Err(e) => e,
        Ok(_) => panic!("forced validation failure must error"),
    };
    let msg = err.to_string();
    assert!(msg.contains("cuda"), "message must name the backend: {msg}");
    assert!(
        msg.contains("never fall back"),
        "message must state the explicit-no-fallback contract: {msg}"
    );
}
