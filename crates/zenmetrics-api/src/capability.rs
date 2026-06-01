//! Minimal backend-availability detection for [`Backend::resolve_auto`].
//!
//! This is the small, dependency-light probe the umbrella uses to turn
//! [`Backend::Auto`] into a concrete backend. It deliberately does **not**
//! pull in `zenmetrics-orchestrator`'s full capability stack
//! (`GpuCapability` / `CpuCapability`, VRAM totals, compute-cap parsing,
//! `nvcc` version extraction). It answers exactly one question per
//! backend: *is a usable device present right now?*
//!
//! ## Why a subprocess probe (and not a cubecl client init)
//!
//! Phase 1 wants resolution to be cheap and side-effect-free — building a
//! real cubecl CUDA/wgpu client allocates a device context and is far too
//! heavy to run on every `Auto` resolution. So, mirroring
//! `zenmetrics-orchestrator::gpu::detect_gpu`, the CUDA probe shells out to
//! `nvidia-smi` once. That tells us a CUDA device is *present*; it does not
//! prove the cubecl client will initialize. Construction still surfaces
//! [`crate::Error::BackendNotEnabled`] when a backend's Cargo feature is
//! off, and a device that's present-but-unusable will fail at the
//! per-crate construction boundary, not here.
//!
//! ## `ZENMETRICS_FORCE_NO_GPU=1` override
//!
//! Honored identically to the orchestrator: when the environment variable
//! `ZENMETRICS_FORCE_NO_GPU` is `"1"`, every GPU probe reports *absent*
//! without spawning any subprocess. This is the test/CI fixture for the
//! no-GPU fallback path. Only reads the variable — never sets it — so the
//! crate stays `#![forbid(unsafe_code)]`-clean.

use crate::metric::Backend;

/// True when `ZENMETRICS_FORCE_NO_GPU=1` is set in the environment.
fn forced_no_gpu() -> bool {
    std::env::var("ZENMETRICS_FORCE_NO_GPU")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Best-effort CUDA-device presence check.
///
/// Returns `false` when `ZENMETRICS_FORCE_NO_GPU=1`, when the `cuda`
/// umbrella feature is off (no point reporting a CUDA device the build
/// can't use), or when `nvidia-smi` is missing / exits nonzero / lists no
/// GPU. Never panics. Matches the `nvidia-smi --query-gpu=gpu_name` probe
/// in `zenmetrics-orchestrator::gpu`.
pub(crate) fn cuda_device_present() -> bool {
    if !cfg!(feature = "cuda") || forced_no_gpu() {
        return false;
    }
    nvidia_smi_lists_a_gpu()
}

/// True if a wgpu-capable GPU adapter is detectable.
///
/// Phase 1 has no standalone wgpu enumeration dependency (pulling `wgpu`
/// in just to count adapters would violate "small + dependency-light").
/// `nvidia-smi` presence is a sufficient *positive* signal that some GPU
/// exists, which wgpu can drive via Vulkan/DX12; absence is treated as no
/// wgpu adapter for `Auto` purposes. Returns `false` when the `wgpu`
/// feature is off or `ZENMETRICS_FORCE_NO_GPU=1`.
pub(crate) fn wgpu_device_present() -> bool {
    if !cfg!(feature = "wgpu") || forced_no_gpu() {
        return false;
    }
    nvidia_smi_lists_a_gpu()
}

/// True if a ROCm/HIP device is detectable.
///
/// Probes `rocm-smi` (the ROCm analogue of `nvidia-smi`); absence is
/// treated as no HIP device. Returns `false` when the `hip` feature is
/// off or `ZENMETRICS_FORCE_NO_GPU=1`.
pub(crate) fn hip_device_present() -> bool {
    if !cfg!(feature = "hip") || forced_no_gpu() {
        return false;
    }
    rocm_smi_lists_a_gpu()
}

/// Resolve [`Backend::Auto`] to a concrete backend in preference order
/// CUDA → WGPU → HIP → optimized native [`Backend::Cpu`] → cubecl-cpu
/// [`Backend::CubeclCpu`]. Each GPU arm is gated on both its Cargo feature
/// (via the `*_device_present` helpers) and live detection, so a build
/// without `cuda` never resolves to `Cuda`. With no GPU present it resolves
/// to the optimized native `Cpu` path when any `cpu-*` metric is compiled in
/// (task #159 phase 2), else to `CubeclCpu` so `Auto` always lands on a
/// runnable backend. Guaranteed to return a non-`Auto` variant; never panics.
pub(crate) fn resolve_auto_backend() -> Backend {
    if cuda_device_present() {
        Backend::Cuda
    } else if wgpu_device_present() {
        Backend::Wgpu
    } else if hip_device_present() {
        Backend::Hip
    } else {
        cpu_fallback_backend()
    }
}

/// The GPU-less fallback backend (pure; no device probing). Prefers the
/// optimized native [`Backend::Cpu`] path when any `cpu-*` metric is built —
/// it is the fast CPU path and, unlike cubecl-cpu, never panics on
/// `atomic<f32>`. Falls back to [`Backend::CubeclCpu`] only when no
/// optimized-CPU metric is compiled in, so `Auto` always resolves to a
/// runnable backend. `Auto` resolving here never changes the score: it only
/// selects a backend, and per-metric construction still surfaces
/// [`crate::Error::BackendNotEnabled`] if the *chosen* metric lacks its
/// `cpu-*` feature.
fn cpu_fallback_backend() -> Backend {
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-iwssim",
        feature = "cpu-zensim",
        feature = "cpu-dssim",
        feature = "cpu-butter"
    ))]
    {
        Backend::Cpu
    }
    #[cfg(not(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-iwssim",
        feature = "cpu-zensim",
        feature = "cpu-dssim",
        feature = "cpu-butter"
    )))]
    {
        Backend::CubeclCpu
    }
}

/// Run `nvidia-smi --query-gpu=gpu_name --format=csv,noheader` and report
/// whether at least one non-empty GPU name came back. One short
/// subprocess; any spawn failure / nonzero exit / empty output collapses
/// to `false`.
fn nvidia_smi_lists_a_gpu() -> bool {
    smi_lists_a_gpu(
        "nvidia-smi",
        &["--query-gpu=gpu_name", "--format=csv,noheader"],
    )
}

/// ROCm analogue: `rocm-smi --showproductname` lists each GPU. We only
/// need presence, so any successful run with non-blank stdout counts.
fn rocm_smi_lists_a_gpu() -> bool {
    smi_lists_a_gpu("rocm-smi", &["--showproductname"])
}

/// Shared subprocess helper: run `program args…`, return `true` iff it
/// exits successfully and prints at least one non-whitespace line. Never
/// panics — a missing binary returns `false`.
fn smi_lists_a_gpu(program: &str, args: &[&str]) -> bool {
    let Ok(out) = std::process::Command::new(program).args(args).output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout.lines().any(|l| !l.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_auto_never_returns_auto_and_never_panics() {
        // Whatever this host has, resolution must terminate on a
        // concrete backend.
        let b = resolve_auto_backend();
        assert_ne!(b, Backend::Auto, "resolve_auto must not return Auto");
    }

    #[test]
    fn missing_binary_reports_no_gpu() {
        // A program that doesn't exist must collapse to `false`, not
        // panic.
        assert!(!smi_lists_a_gpu(
            "definitely-not-a-real-smi-binary-xyzzy",
            &["--whatever"]
        ));
    }

    #[test]
    fn rocm_probe_is_false_without_hip_feature() {
        // The `hip` feature is not in the default set, so on a default
        // build the HIP arm is gated off regardless of any rocm-smi on
        // PATH.
        if !cfg!(feature = "hip") {
            assert!(!hip_device_present());
        }
    }

    // GPU-less fallback (task #159 phase 2): with an optimized-CPU metric
    // built, `Auto` must prefer the fast native `Cpu` path, not cubecl-cpu.
    #[cfg(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-iwssim",
        feature = "cpu-zensim",
        feature = "cpu-dssim",
        feature = "cpu-butter"
    ))]
    #[test]
    fn cpu_fallback_prefers_optimized_cpu_when_built() {
        assert_eq!(cpu_fallback_backend(), Backend::Cpu);
    }

    // Without any optimized-CPU metric, the GPU-less fallback stays
    // cubecl-cpu so `Auto` still resolves to a runnable backend.
    #[cfg(not(any(
        feature = "cpu-ssim2",
        feature = "cpu-cvvdp",
        feature = "cpu-iwssim",
        feature = "cpu-zensim",
        feature = "cpu-dssim",
        feature = "cpu-butter"
    )))]
    #[test]
    fn cpu_fallback_is_cubecl_cpu_without_optimized_metric() {
        assert_eq!(cpu_fallback_backend(), Backend::CubeclCpu);
    }
}
