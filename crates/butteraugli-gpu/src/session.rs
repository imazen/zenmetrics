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

use crate::opaque::{Backend, ButteraugliOpaque};
use crate::{ButteraugliParams, Result};

// Pool cleanup / usage are metric-agnostic — re-export the shared impls so
// `butteraugli_gpu::session::{cleanup_stream, stream_reserved_bytes}` keep
// resolving.
#[doc(hidden)]
pub use zenmetrics_gpu_core::{cleanup_stream, stream_reserved_bytes};

/// Build an opaque butteraugli scorer bound to the private cubecl stream
/// `stream_value`. `#[doc(hidden)]` — internal plumbing for
/// `zenmetrics_api::MetricSession`.
#[doc(hidden)]
#[allow(unused_variables)]
pub fn new_opaque_on_stream(
    backend: Backend,
    stream_value: u64,
    width: u32,
    height: u32,
    params: ButteraugliParams,
    mode: crate::MemoryMode,
) -> Result<ButteraugliOpaque> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => ButteraugliOpaque::build_from_client::<cubecl::cuda::CudaRuntime>(
            zenmetrics_gpu_core::cuda_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => ButteraugliOpaque::build_from_client::<cubecl::wgpu::WgpuRuntime>(
            zenmetrics_gpu_core::wgpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "cpu")]
        Backend::Cpu => ButteraugliOpaque::build_from_client::<cubecl::cpu::CpuRuntime>(
            zenmetrics_gpu_core::cpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
    }
}
