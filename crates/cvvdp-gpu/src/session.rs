//! Stream-bound session plumbing for the umbrella `MetricSession`
//! (issue imazen/zenmetrics#17, design
//! `zenmetrics-api/docs/VRAM_LIFECYCLE_DESIGN_2026-05-30.md`).
//!
//! **This is internal-plumbing surface, not a supported per-crate API.**
//! Every item here is `#[doc(hidden)]` and gated on `cubecl-types`. The
//! supported way to get ironclad VRAM-on-drop is
//! `zenmetrics_api::MetricSession` — this module exists only so that
//! umbrella can bind a cvvdp scorer to a private cubecl stream.
//!
//! ## Why the `unsafe` lives in `zenmetrics-gpu-core`
//!
//! `ComputeClient::set_stream` is `unsafe` (cubecl
//! `client.rs:97` — *"highly unsafe and should probably only be used by
//! the CubeCL/Burn projects."*). The umbrella forbids `unsafe`, so the
//! one `set_stream` call — guarded by a process-global allocator that
//! hands out collision-free `StreamId.value`s within `max_streams` (so
//! no two live sessions ever alias the same `value % 128` pool) — is
//! confined to the shared [`zenmetrics_gpu_core`] crate (the six `*-gpu`
//! crates carried byte-identical copies before it existed). The
//! client-on-stream helpers + [`cleanup_stream`] + [`stream_reserved_bytes`]
//! live there; only the metric-specific [`new_opaque_on_stream`] builder
//! stays here.
//!
//! ## What each helper does
//!
//! - [`new_opaque_on_stream`] builds a [`CvvdpOpaque`] whose internal
//!   cubecl client is bound to `stream_value`, so every device
//!   allocation it makes lands on that stream's private pool.
//! - [`cleanup_stream`] (re-exported from core) binds a fresh client to
//!   `stream_value` and runs `memory_cleanup()` + `sync()` on it —
//!   returning that stream's fully-free pool pages to the driver.
//!   Spike-confirmed isolated: cleaning one stream does not touch
//!   another's pool (`examples/vram_isolation_spike.rs`).
//! - [`stream_reserved_bytes`] (re-exported from core) binds a fresh
//!   client to `stream_value` and reads `memory_usage().bytes_reserved`
//!   — the in-API truth of what that stream's pool holds on the device.
//!   Used by the umbrella's VRAM-isolation test to assert a dropped
//!   session's pool went to 0.

use crate::Result;
use crate::opaque::{Backend, CvvdpOpaque};
use crate::params::{CvvdpParams, DisplayGeometry};

// Pool cleanup / usage are metric-agnostic — re-export the shared impls so
// `cvvdp_gpu::session::{cleanup_stream, stream_reserved_bytes}` keep resolving.
#[doc(hidden)]
pub use zenmetrics_gpu_core::{cleanup_stream, stream_reserved_bytes};

/// Build an opaque cvvdp scorer bound to the private cubecl stream
/// `stream_value`. Mirrors
/// [`CvvdpOpaque::new_with_geometry_and_memory_mode`] but threads a
/// stream-bound client through the same `build_opaque_from_client` seam
/// the default constructor uses, so the scorer's device working set
/// lives on `stream_value`'s isolated pool.
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
            let client = zenmetrics_gpu_core::cuda_client_on_stream(stream_value);
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
            let client = zenmetrics_gpu_core::wgpu_client_on_stream(stream_value);
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
            let client = zenmetrics_gpu_core::cpu_client_on_stream(stream_value);
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
