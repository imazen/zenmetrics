//! Stream-bound session plumbing for the umbrella `MetricSession`
//! (issue imazen/zenmetrics#17, design
//! `zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md`).
//!
//! **This is internal-plumbing surface, not a supported per-crate API.**
//! Every item here is `#[doc(hidden)]` and gated on `cubecl-types`. The
//! supported way to get ironclad VRAM-on-drop is
//! `zenmetrics_api::MetricSession` — this module exists only so that
//! umbrella can bind a cvvdp scorer to a private cubecl stream without
//! pulling the `unsafe` `set_stream` call into `zenmetrics-api` (which
//! is `#![forbid(unsafe_code)]`).
//!
//! ## Why the unsafe lives here
//!
//! `ComputeClient::set_stream` is `unsafe` (cubecl
//! `client.rs:97` — *"highly unsafe and should probably only be used by
//! the CubeCL/Burn projects."*). The umbrella forbids `unsafe`, so the
//! one `set_stream` call — guarded by a process-global allocator that
//! hands out collision-free `StreamId.value`s within `max_streams` (so
//! no two live sessions ever alias the same `value % 128` pool) — is
//! confined to this metric crate.
//!
//! ## What each helper does
//!
//! - [`new_opaque_on_stream`] builds a [`CvvdpOpaque`] whose internal
//!   cubecl client is bound to `stream_value`, so every device
//!   allocation it makes lands on that stream's private pool.
//! - [`cleanup_stream`] binds a fresh client to `stream_value` and runs
//!   `memory_cleanup()` + `sync()` on it — returning that stream's
//!   fully-free pool pages to the driver. Spike-confirmed isolated:
//!   cleaning one stream does not touch another's pool
//!   (`examples/vram_isolation_spike.rs`).
//! - [`stream_reserved_bytes`] binds a fresh client to `stream_value`
//!   and reads `memory_usage().bytes_reserved` — the in-API truth of
//!   what that stream's pool holds on the device. Used by the umbrella's
//!   VRAM-isolation test to assert a dropped session's pool went to 0.

use crate::opaque::{Backend, CvvdpOpaque};
use crate::params::{CvvdpParams, DisplayGeometry};
use crate::Result;

/// Bind a CLONE of the cached per-device client for `backend` to the
/// explicit cubecl stream `stream_value`, so every allocation /
/// cleanup on the returned client lands on that stream's private pool.
///
/// SAFETY: the caller (the umbrella `MetricSession` allocator) MUST
/// hand out a `stream_value` that is unique among live sessions and
/// `< max_streams` (128), so no two live clients alias the same
/// `value % max_streams` physical stream + pool. Aliasing would
/// reintroduce the shared-pool partial-page reclaim hazard the session
/// design exists to eliminate.
#[cfg(feature = "cuda")]
fn cuda_client_on_stream(stream_value: u64) -> cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::cuda::CudaRuntime::client(&Default::default());
    // SAFETY: stream_value is unique-per-live-session and < 128 by the
    // umbrella allocator's contract (see module docs).
    unsafe { c.set_stream(StreamId { value: stream_value }) };
    c
}

/// wgpu counterpart of [`cuda_client_on_stream`].
#[cfg(feature = "wgpu")]
fn wgpu_client_on_stream(stream_value: u64) -> cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    // SAFETY: see cuda_client_on_stream.
    unsafe { c.set_stream(StreamId { value: stream_value }) };
    c
}

/// cpu counterpart of [`cuda_client_on_stream`].
#[cfg(feature = "cpu")]
fn cpu_client_on_stream(stream_value: u64) -> cubecl::prelude::ComputeClient<cubecl::cpu::CpuRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::cpu::CpuRuntime::client(&Default::default());
    // SAFETY: see cuda_client_on_stream.
    unsafe { c.set_stream(StreamId { value: stream_value }) };
    c
}

/// Build an opaque cvvdp scorer bound to the private cubecl stream
/// `stream_value`. Mirrors
/// [`CvvdpOpaque::new_with_geometry_and_memory_mode`] but threads a
/// stream-bound client through the same `build_cvvdp_inner` seam the
/// default constructor uses, so the scorer's device working set lives
/// on `stream_value`'s isolated pool.
///
/// `#[doc(hidden)]` — internal plumbing for `zenmetrics_api::MetricSession`.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn new_opaque_on_stream(
    backend: Backend,
    stream_value: u64,
    width: u32,
    height: u32,
    params: CvvdpParams,
    geometry: DisplayGeometry,
    mode: crate::MemoryMode,
) -> Result<CvvdpOpaque> {
    let resolved_mode = crate::opaque::resolve_mode_for_construction(width, height, mode)?;
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let client = cuda_client_on_stream(stream_value);
            crate::opaque::build_opaque_from_client::<cubecl::cuda::CudaRuntime>(
                client,
                backend,
                width,
                height,
                params,
                geometry,
                resolved_mode,
            )
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            let client = wgpu_client_on_stream(stream_value);
            crate::opaque::build_opaque_from_client::<cubecl::wgpu::WgpuRuntime>(
                client,
                backend,
                width,
                height,
                params,
                geometry,
                resolved_mode,
            )
        }
        #[cfg(feature = "cpu")]
        Backend::Cpu => {
            let client = cpu_client_on_stream(stream_value);
            crate::opaque::build_opaque_from_client::<cubecl::cpu::CpuRuntime>(
                client,
                backend,
                width,
                height,
                params,
                geometry,
                resolved_mode,
            )
        }
    }
}

/// Run `memory_cleanup()` + `sync()` on `backend`'s pool for the
/// explicit stream `stream_value`, returning that stream's fully-free
/// pages to the driver. Isolated: does not touch any other stream's
/// pool (spike-confirmed).
///
/// Call only when NO live binding into `stream_value`'s pool exists on
/// any thread (the umbrella `MetricSession` guarantees this by dropping
/// the session's metric state before calling cleanup).
///
/// `#[doc(hidden)]` — internal plumbing for `zenmetrics_api::MetricSession`.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn cleanup_stream(backend: Backend, stream_value: u64) {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let client = cuda_client_on_stream(stream_value);
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            let client = wgpu_client_on_stream(stream_value);
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
        #[cfg(feature = "cpu")]
        Backend::Cpu => {
            let client = cpu_client_on_stream(stream_value);
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
    }
}

/// Read `memory_usage().bytes_reserved` for `backend`'s pool on the
/// explicit stream `stream_value` (after a `sync()` so the deferred
/// free queue has drained). This is the in-API truth of how much device
/// memory that stream's pool currently holds — the load-bearing signal
/// for the umbrella's VRAM-isolation test.
///
/// Returns `None` if the backend isn't compiled in or the client query
/// fails.
///
/// `#[doc(hidden)]` — internal plumbing for `zenmetrics_api::MetricSession`.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn stream_reserved_bytes(backend: Backend, stream_value: u64) -> Option<u64> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            let client = cuda_client_on_stream(stream_value);
            let _ = cubecl::future::block_on(client.sync());
            client.memory_usage().ok().map(|u| u.bytes_reserved)
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            let client = wgpu_client_on_stream(stream_value);
            let _ = cubecl::future::block_on(client.sync());
            client.memory_usage().ok().map(|u| u.bytes_reserved)
        }
        #[cfg(feature = "cpu")]
        Backend::Cpu => {
            let client = cpu_client_on_stream(stream_value);
            let _ = cubecl::future::block_on(client.sync());
            client.memory_usage().ok().map(|u| u.bytes_reserved)
        }
    }
}
