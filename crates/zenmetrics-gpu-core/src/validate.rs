//! Backend liveness validation — proves a cubecl runtime can actually
//! **compile and execute** a kernel before any metric trusts it
//! (imazen/zenmetrics#37).
//!
//! ## Why constructing a client is not enough
//!
//! `CudaRuntime::client(..)` succeeds on any box with an NVIDIA *driver* —
//! including boxes with **no CUDA toolkit** (no `/usr/local/cuda`, no
//! `CUDA_PATH`). On such a box every kernel launch fails at NVRTC-compile
//! time inside cubecl's device-service thread (`DSD-*`), where the panic is
//! caught per-task (`cubecl-common` device channel: `catch_unwind` →
//! `log::warn!`) and the fire-and-forget launch silently no-ops. Uploads and
//! readbacks are pure driver-API calls, so they keep working — a metric then
//! reads back its **zero-initialized, never-written** accumulator buffers
//! and aggregates them into a plausible-looking score (ssim2's weighted sum
//! of 108 all-zero sub-stats lands in the `ssim <= 0.0` branch of the
//! reference remap → exactly `100.0`). See the issue for the field report.
//!
//! ## What the probe does
//!
//! Once per process per backend (cached), [`validate_backend`]:
//!
//! 1. builds the runtime's default-device client (caller-thread panics —
//!    e.g. no driver at all — are caught and surfaced as `Err`),
//! 2. uploads a 2-element `u32` sentinel buffer,
//! 3. launches a trivial kernel that derives element 1 from element 0
//!    (forcing a real backend compile + dispatch),
//! 4. reads the buffer back and verifies the kernel's write is present.
//!
//! A swallowed device-thread failure leaves the sentinel untouched, which
//! the readback check converts into a hard `Err` — so "the runtime accepted
//! the dispatch but never ran it" can no longer masquerade as a score.
//!
//! The result is cached per backend (`OnceLock`): the failure modes probed
//! here (missing driver, missing toolkit, no adapter) are process-global
//! and don't heal mid-process, and construction stays cheap after the first
//! call. On a healthy box the probe costs one trivial kernel compile that
//! warms the same caches real kernels use.
//!
//! ## Test seam
//!
//! `ZENMETRICS_FORCE_BACKEND_INIT_FAIL` (checked per call, **before** the
//! cache and before touching any runtime) forces validation failure so the
//! explicit-backend error path is testable on boxes whose GPU stack works:
//! set it to `all` / `1` (every backend) or a comma-separated list of
//! backend tags (`cuda`, `wgpu`, `cpu`). Mirrors the existing
//! `ZENMETRICS_FORCE_NO_GPU` fixture pattern.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::sync::OnceLock;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use cubecl::prelude::*;

use crate::Backend;

/// Sentinel uploaded at element 0; the probe kernel must write
/// `PROBE_INPUT + 1` into element 1. Arbitrary non-zero bit pattern so a
/// zero-filled or stale buffer can't false-pass.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
const PROBE_INPUT: u32 = 0x5EED_CAFE;

/// Trivial liveness kernel: derive element 1 from element 0. Forces a real
/// backend compile (NVRTC / naga / MLIR) + dispatch + device write — the
/// exact machinery that silently no-ops when the toolkit is absent.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[cube(launch_unchecked)]
fn backend_probe_kernel(buf: &mut Array<u32>) {
    if ABSOLUTE_POS == 0 {
        buf[1] = buf[0] + 1u32;
    }
}

/// Validate that `backend`'s runtime is actually operational on this
/// machine: client init + kernel compile + dispatch + verified readback.
///
/// Returns `Err(reason)` when the runtime cannot initialize **or** when it
/// initializes but kernel dispatch is non-functional (the
/// driver-without-toolkit trap of imazen/zenmetrics#37). Results are cached
/// per backend for the process lifetime; the
/// `ZENMETRICS_FORCE_BACKEND_INIT_FAIL` test seam is consulted per call,
/// before the cache.
///
/// [`Backend::Cpu`] (the cubecl-cpu reference runtime) is in-process — it
/// has no external installation whose absence could be swallowed — so it
/// validates trivially (test seam still honored). Its known per-kernel
/// gaps (`Atomic<f32>`) are documented per metric crate and are not
/// detectable by a liveness probe.
pub fn validate_backend(backend: Backend) -> Result<(), String> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            if let Some(forced) = forced_failure("cuda") {
                return Err(forced);
            }
            static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
            PROBE
                .get_or_init(|| probe_runtime::<cubecl::cuda::CudaRuntime>("cuda"))
                .clone()
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            if let Some(forced) = forced_failure("wgpu") {
                return Err(forced);
            }
            static PROBE: OnceLock<Result<(), String>> = OnceLock::new();
            PROBE
                .get_or_init(|| probe_runtime::<cubecl::wgpu::WgpuRuntime>("wgpu"))
                .clone()
        }
        #[cfg(feature = "cpu")]
        Backend::Cpu => match forced_failure("cpu") {
            Some(forced) => Err(forced),
            None => Ok(()),
        },
    }
}

/// The `ZENMETRICS_FORCE_BACKEND_INIT_FAIL` test seam: `Some(reason)` when
/// the variable forces `tag` to fail (`all`/`1`, or a comma-separated tag
/// list containing `tag`).
fn forced_failure(tag: &str) -> Option<String> {
    let v = std::env::var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL").ok()?;
    let hit = v == "1"
        || v.eq_ignore_ascii_case("all")
        || v.split(',').any(|t| t.trim().eq_ignore_ascii_case(tag));
    hit.then(|| {
        format!(
            "backend '{tag}' initialization failure forced by \
             ZENMETRICS_FORCE_BACKEND_INIT_FAIL={v} (test seam)"
        )
    })
}

/// Run the sentinel probe for runtime `R`, converting caller-thread panics
/// (cubecl inits liberally `unwrap`/`expect` — e.g. `cuInit` with no
/// driver, wgpu with no adapter) into `Err`.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn probe_runtime<R: cubecl::Runtime>(tag: &'static str) -> Result<(), String>
where
    R::Device: Default,
{
    match std::panic::catch_unwind(|| probe_client::<R>()) {
        Ok(r) => r.map_err(|e| format!("backend '{tag}' failed its liveness probe: {e}")),
        Err(payload) => Err(format!(
            "backend '{tag}' panicked during its liveness probe: {}",
            panic_message(payload.as_ref())
        )),
    }
}

/// Upload → dispatch → verified readback on `R`'s default-device client.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn probe_client<R: cubecl::Runtime>() -> Result<(), String>
where
    R::Device: Default,
{
    let client = R::client(&Default::default());
    let handle = client.create_from_slice(u32::as_bytes(&[PROBE_INPUT, 0]));
    // SAFETY: `handle` is a live 2-element u32 buffer on `client`; the
    // launch geometry (1 cube × 1 thread) only touches indices 0 and 1.
    unsafe {
        backend_probe_kernel::launch_unchecked::<R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(handle.clone(), 2),
        );
    }
    let bytes = client
        .read_one(handle)
        .map_err(|e| format!("sentinel readback failed: {e:?}"))?;
    let vals = u32::from_bytes(&bytes);
    match (vals.first().copied(), vals.get(1).copied()) {
        (Some(PROBE_INPUT), Some(v)) if v == PROBE_INPUT.wrapping_add(1) => Ok(()),
        (a, b) => Err(format!(
            "probe kernel never executed (readback {a:?}/{b:?}, expected \
             Some({PROBE_INPUT:#010x})/Some({:#010x})). The runtime accepted the \
             dispatch but the kernel did not run — on CUDA this is the \
             driver-without-toolkit trap (no /usr/local/cuda and no CUDA_PATH), \
             where cubecl's device-service thread swallows the NVRTC compile \
             panic. Install the CUDA toolkit or select another backend.",
            PROBE_INPUT.wrapping_add(1),
        )),
    }
}

/// Best-effort panic-payload → message (payloads are `&str` or `String`
/// in practice).
#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

#[cfg(test)]
mod tests {
    use super::forced_failure;

    // NOTE: these tests only exercise the pure seam-parsing helper — env
    // *mutation* lives in the dedicated `tests/validate_hook.rs` process
    // so it can't race sibling tests in this binary.

    #[test]
    fn forced_failure_is_none_when_var_unset() {
        // The var is not set in a normal test environment; if a caller
        // exported it globally the other integration tests would fail
        // loudly anyway, so this stays honest.
        if std::env::var("ZENMETRICS_FORCE_BACKEND_INIT_FAIL").is_err() {
            assert!(forced_failure("cuda").is_none());
            assert!(forced_failure("wgpu").is_none());
        }
    }
}
