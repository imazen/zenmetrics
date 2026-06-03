//! Shared plumbing for the zenmetrics `*-gpu` metric crates.
//!
//! The six GPU metric crates (`butteraugli-gpu`, `cvvdp-gpu`, `dssim-gpu`,
//! `iwssim-gpu`, `ssim2-gpu`, `zensim-gpu`) were each carrying byte-identical
//! copies of:
//!
//! - the [`Backend`] enum (which CubeCL runtime an opaque shim dispatches to),
//! - the uniform [`Score`] struct returned by every opaque shim,
//! - the [`convert_to_srgb_rgb8`] zenpixels conversion helper, and
//! - the stream-bound session plumbing ([`cuda_client_on_stream`] &c.,
//!   [`cleanup_stream`], [`stream_reserved_bytes`]) backing the umbrella
//!   `zenmetrics_api::MetricSession` (issue imazen/zenmetrics#17).
//!
//! This crate is the single source of truth for those. Each `*-gpu` crate
//! re-exports [`Backend`] / [`Score`] (so `crate::Backend` keeps resolving)
//! and calls the helpers here. Metric-specific types (`*Params`, `*Opaque`,
//! per-crate `Error`, the `new_opaque_on_stream` builder) stay in their crate.
//!
//! These are internal-plumbing types for `publish = false` workspace crates;
//! the supported public surface is `zenmetrics_api`. They deliberately drop
//! `#[non_exhaustive]` (carried by the per-crate copies before this crate
//! existed) so the metric crates can construct/match them directly — the
//! umbrella's `Backend`/`Score`/`MemoryMode` remain the stability surface.
//!
//! The one `unsafe` here is the confined `set_stream` call in the
//! client-on-stream helpers (same as the per-crate `session.rs` modules); the
//! umbrella stays `#![forbid(unsafe_code)]` by funnelling stream binding here.

/// Selects the GPU/CPU backend an opaque metric shim dispatches to.
///
/// Variants are Cargo-feature-gated to match the runtimes CubeCL was built
/// with. Re-exported by each `*-gpu` crate as `crate::Backend`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend (NVIDIA, requires the `cuda` Cargo feature).
    #[cfg(feature = "cuda")]
    Cuda,
    /// WGPU backend (cross-vendor: Vulkan/Metal/DX12, requires the `wgpu`
    /// Cargo feature).
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU reference backend (requires the `cpu` Cargo feature). Some metric
    /// kernels use `Atomic<f32>` reductions the cubecl-cpu backend does not
    /// support; those crates accept `Cpu` for API uniformity but may panic at
    /// first dispatch.
    #[cfg(feature = "cpu")]
    Cpu,
}

/// Uniform metric score value returned by every opaque shim.
///
/// Re-exported by each `*-gpu` crate as `crate::Score` / `crate::opaque::Score`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    /// The numeric score. Its meaning is metric-specific — see each crate's
    /// opaque-shim docs (e.g. butteraugli max-norm, zensim MLP score).
    pub value: f64,
    /// Short metric identifier (e.g. `"zensim"`, `"butter"`, `"cvvdp"`).
    pub metric_name: &'static str,
    /// Implementation version tag.
    pub metric_version: &'static str,
}

/// Convert a [`zenpixels::PixelSlice`] to a tight, row-major `RGB8` buffer in
/// the requested `target` descriptor (`RGB8_SRGB` for the metric shims).
///
/// Row-strided input is handled natively — each source row is converted into
/// its tight destination row, so SIMD-padded / sub-region slices work without
/// a fast-path bail. Returns the contiguous `width × height × bpp` bytes.
///
/// The per-crate `to_srgb_rgb8` wrappers own the dimension check and the
/// short-circuit when the slice is already `RGB8_SRGB`; this owns the actual
/// `zenpixels_convert` row conversion.
#[cfg(feature = "pixels")]
pub fn convert_to_srgb_rgb8(
    s: &zenpixels::PixelSlice<'_>,
    target: zenpixels::PixelDescriptor,
) -> core::result::Result<Vec<u8>, zenpixels_convert::ConvertError> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let plan = ConvertPlan::new(s.descriptor(), target).map_err(|e| e.decompose().0)?;
    let w = s.width();
    let h = s.rows();
    let row_bytes = (w as usize) * target.bytes_per_pixel();
    let mut out = vec![0u8; row_bytes * (h as usize)];
    for y in 0..h {
        let src_row = s.row(y);
        let start = (y as usize) * row_bytes;
        let dst_row = &mut out[start..start + row_bytes];
        convert_row(&plan, src_row, dst_row, w);
    }
    Ok(out)
}

// ───────────────────────── stream-bound session plumbing ─────────────────────
//
// Backs the umbrella `zenmetrics_api::MetricSession` (issue #17). Each helper
// clones the cached per-device client and binds it to an explicit CubeCL
// stream so the umbrella's 128-slot allocator can isolate live sessions. The
// single `unsafe set_stream` is confined here.

#[cfg(all(feature = "cubecl-types", feature = "cuda"))]
#[doc(hidden)]
pub fn cuda_client_on_stream(
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

#[cfg(all(feature = "cubecl-types", feature = "wgpu"))]
#[doc(hidden)]
pub fn wgpu_client_on_stream(
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

#[cfg(all(feature = "cubecl-types", feature = "cpu"))]
#[doc(hidden)]
pub fn cpu_client_on_stream(
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

/// Run `memory_cleanup()` + `sync()` on `backend`'s pool for the stream
/// `stream_value`. `#[doc(hidden)]` internal plumbing.
#[cfg(feature = "cubecl-types")]
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

/// Read `memory_usage().bytes_reserved` for `backend`'s pool on the stream
/// `stream_value` (after a `sync()`). `#[doc(hidden)]` internal plumbing.
#[cfg(feature = "cubecl-types")]
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
