//! Public-surface tests for [`zenmetrics_api::Backend::resolve_auto`]
//! (task #159 phase 1).
//!
//! These exercise the observable `Auto` resolution. No graceful skips:
//! every code path asserts a concrete invariant. The GPU-present arm and
//! the forced-no-GPU arm are both checked — which one fires depends on
//! the host, but each makes an assertion, so the test never passes
//! vacuously.
//!
//! NOTE on `ZENMETRICS_FORCE_NO_GPU`: `Backend::Auto` resolution reads it
//! process-globally, and the tests in the `it` binary run concurrently, so
//! no test in the shared process mutates it. The env-sensitive test below
//! re-execs this test binary as a child per case, with the variable set or
//! cleared at spawn (see `crate::run_in_child`). The invariant test below
//! touches no env and is race-free.

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

/// Independent CUDA-*toolkit* presence cross-check, mirroring
/// `cubecl_cuda::install::cuda_path` (`CUDA_PATH`, `/usr/local/cuda`,
/// `/opt/cuda`, `/usr/bin/nvcc`). A box can have a driver (so
/// `nvidia-smi` lists a GPU) but no toolkit — the imazen/zenmetrics#37
/// trap where every kernel launch silently no-ops. `Auto` must NOT
/// resolve to Cuda on such a box.
fn host_has_cuda_toolkit() -> bool {
    if std::env::var("CUDA_PATH").is_ok() {
        return true;
    }
    ["/usr/local/cuda", "/opt/cuda", "/usr/bin/nvcc"]
        .iter()
        .any(|p| std::path::Path::new(p).exists())
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

/// Two env-sensitive checks, each run in its own CHILD copy of this test
/// binary (see [`crate::run_in_child`]) so the shared process's environment
/// is never mutated and no concurrent `Backend::Auto` reader can observe a
/// flipped `ZENMETRICS_FORCE_NO_GPU`:
///
/// 1. **`host` — host-presence (override unset at spawn).** With the `cuda`
///    feature built, a NVIDIA GPU present, AND the CUDA toolkit installed
///    (the liveness probe passes), `Auto` resolves to [`Backend::Cuda`].
///    With a driver but no toolkit (issue #37) it must NOT pick Cuda. With
///    no NVIDIA GPU it falls back to the CPU ladder. Every arm asserts.
/// 2. **`forced` — forced no-GPU (`ZENMETRICS_FORCE_NO_GPU=1` at spawn).**
///    The override must force `Auto` away from any GPU backend to the CPU
///    fallback ([`EXPECTED_NO_GPU`]), regardless of real hardware — the
///    no-GPU CI fixture, matching the orchestrator's detector.
#[test]
fn resolve_auto_host_and_force_no_gpu() {
    const TEST: &str = "backend_resolve::resolve_auto_host_and_force_no_gpu";
    match crate::child_case().as_deref() {
        None => {
            crate::run_in_child(TEST, "host", crate::ForceNoGpu::Unset);
            crate::run_in_child(TEST, "forced", crate::ForceNoGpu::Set);
        }
        Some("host") => {
            assert!(
                std::env::var_os("ZENMETRICS_FORCE_NO_GPU").is_none(),
                "host case must be spawned with ZENMETRICS_FORCE_NO_GPU unset"
            );
            assert_host_presence();
        }
        Some("forced") => {
            assert_eq!(
                std::env::var("ZENMETRICS_FORCE_NO_GPU").as_deref(),
                Ok("1"),
                "forced case must be spawned with ZENMETRICS_FORCE_NO_GPU=1"
            );
            assert_forced_no_gpu();
        }
        Some(other) => panic!("unknown child case {other:?}"),
    }
}

/// Case 1 body: override unset, so resolution reflects the real host.
fn assert_host_presence() {
    let has_gpu = host_has_nvidia_gpu();
    let resolved = Backend::resolve_auto();

    // Host-presence assertions: only assert the "→ Cuda" expectation
    // when the cuda backend is actually compiled in (default on this
    // box); without it, a present GPU still can't be selected. Since #37,
    // presence alone is not enough: `Auto` additionally requires the
    // cached liveness probe to pass, so the healthy-box expectation is
    // gated on an independent toolkit cross-check, and a
    // driver-without-toolkit host asserts the NEW invariant (never Cuda).
    if has_gpu && cfg!(feature = "cuda") && host_has_cuda_toolkit() {
        assert_eq!(
            resolved,
            Backend::Cuda,
            "CUDA GPU + toolkit present + `cuda` feature → Auto must resolve to Cuda"
        );
    } else if has_gpu && cfg!(feature = "cuda") {
        // Driver present, toolkit absent (the lianli / issue #37 box):
        // kernel launches silently no-op, so resolution must NOT pick
        // Cuda — the liveness probe fails and Auto falls down the ladder.
        assert_ne!(
            resolved,
            Backend::Cuda,
            "CUDA driver without toolkit → Auto must NOT resolve to Cuda (issue #37)"
        );
        assert!(
            resolved == EXPECTED_NO_GPU || resolved == Backend::Wgpu,
            "broken-CUDA fallback must land on {EXPECTED_NO_GPU:?} (or Wgpu), got {resolved:?}"
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
}

/// Case 2 body: override set to `1`, every GPU probe is forced absent, so
/// `Auto` resolves to the pure CPU fallback (host-independent).
fn assert_forced_no_gpu() {
    let resolved_forced = Backend::resolve_auto();
    assert_eq!(
        resolved_forced, EXPECTED_NO_GPU,
        "ZENMETRICS_FORCE_NO_GPU=1 must force Auto to the CPU fallback ({EXPECTED_NO_GPU:?})"
    );
}
