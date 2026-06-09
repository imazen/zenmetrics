//! Public-surface tests for [`zenmetrics_api::Backend::resolve_auto`]
//! (task #159 phase 1).
//!
//! These exercise the observable `Auto` resolution. No graceful skips:
//! every code path asserts a concrete invariant. The GPU-present arm and
//! the forced-no-GPU arm are both checked — which one fires depends on
//! the host, but each makes an assertion, so the test never passes
//! vacuously.
//!
//! NOTE on `ZENMETRICS_FORCE_NO_GPU`: the override is exercised via
//! [`std::env::set_var`], which is `unsafe` on edition-2024. The library
//! crate is `#![forbid(unsafe_code)]`; this integration test is a
//! separate compilation unit without that lint, so it can drive the
//! override the same way `zenmetrics-orchestrator`'s `no_gpu_fallback`
//! test does. To avoid a data race with the GPU-presence assertion (cargo
//! runs a test binary's `#[test]` fns on multiple threads), ALL env
//! mutation + every env-sensitive assertion lives in a single
//! `#[test]` fn (`resolve_auto_host_and_force_no_gpu`). The invariant
//! test below touches no env and is race-free.

use zenmetrics_api::Backend;

/// The GPU-less fallback `Auto` must resolve to, mirroring
/// `capability::cpu_fallback_backend`: the optimized native [`Backend::Cpu`]
/// when any `cpu-*` metric is compiled (it's the fast, non-panicking CPU path),
/// else the cubecl-cpu reference [`Backend::CubeclCpu`]. Keeping this in lockstep
/// with the library is why the assertion can't hard-code `CubeclCpu`.
#[cfg(any(
    feature = "cpu-ssim2",
    feature = "cpu-cvvdp",
    feature = "cpu-iwssim",
    feature = "cpu-zensim",
    feature = "cpu-dssim",
    feature = "cpu-butter"
))]
const EXPECTED_NO_GPU: Backend = Backend::Cpu;
#[cfg(not(any(
    feature = "cpu-ssim2",
    feature = "cpu-cvvdp",
    feature = "cpu-iwssim",
    feature = "cpu-zensim",
    feature = "cpu-dssim",
    feature = "cpu-butter"
)))]
const EXPECTED_NO_GPU: Backend = Backend::CubeclCpu;

/// Probe `nvidia-smi` ourselves so the test can branch on ground truth
/// instead of assuming a GPU. Mirrors the umbrella's internal probe but
/// is independent of it (a real cross-check, not a tautology).
fn host_has_nvidia_gpu() -> bool {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name", "--format=csv,noheader"])
        .output()
    else {
        return false;
    };
    out.status.success()
        && String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|l| !l.trim().is_empty())
}

/// `resolve_auto()` must always terminate on a concrete backend and
/// never panic, on any host / feature set. Touches no environment, so it
/// is safe to run concurrently with everything else.
#[test]
fn resolve_auto_never_auto_never_panics() {
    let b = Backend::resolve_auto();
    assert_ne!(b, Backend::Auto, "resolve_auto must never return Auto");
    // `Backend::Auto.resolve()` must agree with the free function.
    assert_eq!(b, Backend::Auto.resolve());
    // A concrete backend resolves to itself.
    assert_eq!(Backend::Cuda.resolve(), Backend::Cuda);
    assert_eq!(Backend::Wgpu.resolve(), Backend::Wgpu);
    assert_eq!(Backend::Hip.resolve(), Backend::Hip);
    assert_eq!(Backend::CubeclCpu.resolve(), Backend::CubeclCpu);
}

/// Two env-sensitive checks, kept in ONE test fn so the process-global
/// `ZENMETRICS_FORCE_NO_GPU` mutation can't race a sibling test:
///
/// 1. **Host-presence (no override).** With the `cuda` feature built and
///    a usable NVIDIA GPU present, `Auto` resolves to [`Backend::Cuda`];
///    with no NVIDIA GPU it falls back to [`Backend::CubeclCpu`] (phase 1
///    has no optimized `Cpu` variant yet). Both arms assert.
/// 2. **Forced no-GPU.** `ZENMETRICS_FORCE_NO_GPU=1` must force `Auto`
///    away from any GPU backend to [`Backend::CubeclCpu`], regardless of
///    real hardware — the no-GPU CI fixture, matching the orchestrator's
///    detector.
#[test]
fn resolve_auto_host_and_force_no_gpu() {
    // --- 1. host-presence, override guaranteed unset ---
    let prev = std::env::var("ZENMETRICS_FORCE_NO_GPU").ok();
    // SAFETY: env mutation is confined to this single serial test fn,
    // which is the only place in this binary that touches this variable;
    // the prior value is restored before the fn returns.
    unsafe {
        std::env::remove_var("ZENMETRICS_FORCE_NO_GPU");
    }

    let has_gpu = host_has_nvidia_gpu();
    let resolved = Backend::resolve_auto();

    // --- 2. forced no-GPU ---
    unsafe {
        std::env::set_var("ZENMETRICS_FORCE_NO_GPU", "1");
    }
    let resolved_forced = Backend::resolve_auto();

    // Restore the caller's environment BEFORE asserting, so a panic in
    // an assert can't leak the override to any later test.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("ZENMETRICS_FORCE_NO_GPU", v),
            None => std::env::remove_var("ZENMETRICS_FORCE_NO_GPU"),
        }
    }

    // host-presence assertions: only assert the "→ Cuda" expectation
    // when the cuda backend is actually compiled in (default on this
    // box); without it, a present GPU still can't be selected.
    if has_gpu && cfg!(feature = "cuda") {
        assert_eq!(
            resolved,
            Backend::Cuda,
            "CUDA GPU present + `cuda` feature → Auto must resolve to Cuda"
        );
    } else {
        // No CUDA selection: the CPU fallback (`EXPECTED_NO_GPU`) — or `Wgpu`
        // if a wgpu device is present and the `wgpu` backend is built.
        assert!(
            resolved == EXPECTED_NO_GPU || resolved == Backend::Wgpu,
            "no CUDA selection → Auto must fall back to {EXPECTED_NO_GPU:?} \
             (or Wgpu if a wgpu device is present), got {resolved:?}"
        );
    }

    // forced-no-GPU assertion (host-independent): every GPU probe is forced
    // absent, so `Auto` resolves to the pure CPU fallback.
    assert_eq!(
        resolved_forced, EXPECTED_NO_GPU,
        "ZENMETRICS_FORCE_NO_GPU=1 must force Auto to the CPU fallback ({EXPECTED_NO_GPU:?})"
    );
}
