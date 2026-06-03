//! Stream-bound session plumbing for the umbrella `MetricSession`
//! (issue imazen/zenmetrics#17). `#[doc(hidden)]`, gated `cubecl-types`.
//! See `cvvdp-gpu/src/session.rs` for the full rationale. The supported
//! API is `zenmetrics_api::MetricSession`.
//!
//! The client-on-stream helpers + pool `cleanup_stream` /
//! `stream_reserved_bytes` are byte-identical across the six `*-gpu`
//! crates and live in [`zenmetrics_gpu_core`] (which confines the one
//! `unsafe set_stream`). Only the metric-specific `new_opaque_on_stream`
//! builder stays here.

use crate::Result;
use crate::opaque::{Backend, IwssimOpaque, IwssimParams};

// Pool cleanup / usage are metric-agnostic — re-export the shared impls so
// `iwssim_gpu::session::{cleanup_stream, stream_reserved_bytes}` keep resolving.
#[doc(hidden)]
pub use zenmetrics_gpu_core::{cleanup_stream, stream_reserved_bytes};

/// Build an opaque iwssim scorer bound to the private cubecl stream
/// `stream_value`. `#[doc(hidden)]` — internal plumbing for
/// `zenmetrics_api::MetricSession`.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn new_opaque_on_stream(
    backend: Backend,
    stream_value: u64,
    width: u32,
    height: u32,
    params: IwssimParams,
    mode: crate::MemoryMode,
) -> Result<IwssimOpaque> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => IwssimOpaque::build_from_client::<cubecl::cuda::CudaRuntime>(
            zenmetrics_gpu_core::cuda_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => IwssimOpaque::build_from_client::<cubecl::wgpu::WgpuRuntime>(
            zenmetrics_gpu_core::wgpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "cpu")]
        Backend::Cpu => IwssimOpaque::build_from_client::<cubecl::cpu::CpuRuntime>(
            zenmetrics_gpu_core::cpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
    }
}
