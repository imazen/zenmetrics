//! Stream-bound session plumbing for the umbrella `MetricSession`
//! (issue imazen/zenmetrics#17). `#[doc(hidden)]`, gated `cubecl-types`.
//! See `cvvdp-gpu/src/session.rs` for the full rationale. The supported
//! API is `zenmetrics_api::MetricSession`. The one `unsafe set_stream`
//! call is confined here so the umbrella stays `#![forbid(unsafe_code)]`.

use crate::opaque::{Backend, ButteraugliOpaque};
use crate::{ButteraugliParams, Result};

#[cfg(feature = "cuda")]
fn cuda_client_on_stream(
    stream_value: u64,
) -> cubecl::prelude::ComputeClient<cubecl::cuda::CudaRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::cuda::CudaRuntime::client(&Default::default());
    unsafe {
        c.set_stream(StreamId {
            value: stream_value,
        })
    };
    c
}

#[cfg(feature = "wgpu")]
fn wgpu_client_on_stream(
    stream_value: u64,
) -> cubecl::prelude::ComputeClient<cubecl::wgpu::WgpuRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::wgpu::WgpuRuntime::client(&Default::default());
    unsafe {
        c.set_stream(StreamId {
            value: stream_value,
        })
    };
    c
}

#[cfg(feature = "cpu")]
fn cpu_client_on_stream(
    stream_value: u64,
) -> cubecl::prelude::ComputeClient<cubecl::cpu::CpuRuntime> {
    use cubecl::Runtime;
    use cubecl::stream_id::StreamId;
    let mut c = cubecl::cpu::CpuRuntime::client(&Default::default());
    unsafe {
        c.set_stream(StreamId {
            value: stream_value,
        })
    };
    c
}

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
            cuda_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => ButteraugliOpaque::build_from_client::<cubecl::wgpu::WgpuRuntime>(
            wgpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
        #[cfg(feature = "cpu")]
        Backend::Cpu => ButteraugliOpaque::build_from_client::<cubecl::cpu::CpuRuntime>(
            cpu_client_on_stream(stream_value),
            backend,
            width,
            height,
            params,
            mode,
        ),
    }
}

/// Run `memory_cleanup()` + `sync()` on `backend`'s pool for the stream
/// `stream_value`. `#[doc(hidden)]` internal plumbing.
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
/// stream `stream_value` (after a `sync()`). `#[doc(hidden)]` internal
/// plumbing.
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
