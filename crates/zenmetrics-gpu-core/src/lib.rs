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

// ───────────────────────── reflect-pad (sub-minimum images) ──────────────────
//
// Every `*-gpu` metric has a minimum image dimension below which its pyramid
// can't form (8×8 for ssim2/dssim/cvvdp, 176 for iwssim, 64 for zensim's
// 4-scale bake). Rather than reject those inputs, the opaque shims
// reflect(mirror)-pad them up to that floor and score the padded image — so a
// metric returns a finite score down to 1×1 instead of `InvalidImageSize`.
// This is the single source of truth for that padding (matches the CPU
// `zensim::metric` reflect-pad funnel byte-for-byte: same reflect-101 rule),
// shared so the metric crates don't each carry a copy.

/// Reflect-101 index map: fold an out-of-range index `i` back into `[0, n)` by
/// mirroring at the borders **without** repeating the edge sample (OpenCV
/// `BORDER_REFLECT_101`, the rule used by the CPU `zensim` reflect-pad). For
/// `i < n` this is the identity, so the original pixels land at `[0, n)` after
/// padding. `n <= 1` collapses to 0 (a single row/column replicates).
#[inline]
pub fn reflect_index(i: usize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = 2 * (n - 1);
    let mut k = i % period;
    if k >= n {
        k = period - k;
    }
    k
}

/// Reflect(mirror)-pad an interleaved buffer of `ch`-element pixels from a
/// logical `lw × lh` extent up to a padded `pw × ph` extent, using
/// [`reflect_index`] on each axis. Used for `RGB8` (`ch = 3`) and single
/// linear planes (`ch = 1`). The original samples occupy `[0, lw) × [0, lh)`
/// of the output, so a result computed on the padded buffer can be cropped
/// back to the logical extent by taking the top-left sub-rectangle.
///
/// Assumes `pw >= lw`, `ph >= lh`, and `src.len() == lw * lh * ch` — callers
/// validate the input length.
pub fn reflect_pad<T: Copy>(
    src: &[T],
    lw: usize,
    lh: usize,
    pw: usize,
    ph: usize,
    ch: usize,
) -> Vec<T> {
    debug_assert_eq!(src.len(), lw * lh * ch);
    let mut out = Vec::with_capacity(pw * ph * ch);
    for y in 0..ph {
        let sy = reflect_index(y, lh);
        let row = sy * lw;
        for x in 0..pw {
            let sx = reflect_index(x, lw);
            let s = (row + sx) * ch;
            out.extend_from_slice(&src[s..s + ch]);
        }
    }
    out
}

/// Convert a [`PixelSlice`](zenpixels::PixelSlice) of **any** descriptor (sRGB8,
/// PQ, HLG, linear-f32, …) to interleaved **linear-light RGB f32**, letting the
/// descriptor's transfer + primaries drive the conversion via zenpixels-convert.
///
/// This is the descriptor-driven front-end that lets a single metric entry
/// handle SDR and HDR alike: an `RGB8_SRGB` slice decodes to relative linear
/// `[0,1]`; a PQ/HLG/linear HDR slice decodes to its linear light. The caller
/// then maps to display-relative values (e.g. `÷ peak` for absolute transfers)
/// — see the `butteraugli-gpu` `compute_pixels_display` prototype. Returns
/// interleaved `[R,G,B, …]` of length `width·height·3`.
#[cfg(feature = "pixels")]
pub fn convert_to_linear_f32(
    s: &zenpixels::PixelSlice<'_>,
) -> core::result::Result<Vec<f32>, zenpixels_convert::ConvertError> {
    use zenpixels_convert::{ConvertPlan, convert_row};
    let target = zenpixels::PixelDescriptor::RGBF32_LINEAR;
    let plan = ConvertPlan::new(s.descriptor(), target).map_err(|e| e.decompose().0)?;
    let w = s.width();
    let h = s.rows();
    let row_bytes = (w as usize) * target.bytes_per_pixel();
    let mut out = vec![0u8; row_bytes * (h as usize)];
    for y in 0..h {
        let src_row = s.row(y);
        let start = (y as usize) * row_bytes;
        convert_row(&plan, src_row, &mut out[start..start + row_bytes], w);
    }
    // Reinterpret the f32 bytes without an alignment assumption.
    Ok(out
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Split an interleaved RGB `f32` buffer (`[R,G,B, R,G,B, …]`, length `3·n`)
/// into three planar `f32` buffers (`R…`, `G…`, `B…`, each length `n`). Backs
/// the metrics' non-planar linear-RGB entry points (`*_from_linear_interleaved`),
/// whose planar kernels want one tight plane per channel.
///
/// Returns `None` if `rgb.len()` isn't a multiple of 3.
pub fn deinterleave_rgb_f32(rgb: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if !rgb.len().is_multiple_of(3) {
        return None;
    }
    let n = rgb.len() / 3;
    let mut r = Vec::with_capacity(n);
    let mut g = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for px in rgb.chunks_exact(3) {
        r.push(px[0]);
        g.push(px[1]);
        b.push(px[2]);
    }
    Some((r, g, b))
}

#[cfg(test)]
mod deinterleave_tests {
    use super::deinterleave_rgb_f32;

    #[test]
    fn splits_rgb_and_rejects_non_multiple_of_3() {
        let rgb = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (r, g, b) = deinterleave_rgb_f32(&rgb).expect("len 6 is 2 px");
        assert_eq!(r, [1.0, 4.0]);
        assert_eq!(g, [2.0, 5.0]);
        assert_eq!(b, [3.0, 6.0]);
        assert!(deinterleave_rgb_f32(&[1.0, 2.0]).is_none());
        assert_eq!(deinterleave_rgb_f32(&[]), Some((vec![], vec![], vec![])));
    }
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
