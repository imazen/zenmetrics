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
//! The confined `unsafe` here: the `set_stream` call in the client-on-stream
//! helpers (same as the per-crate `session.rs` modules) and the sentinel-kernel
//! launch inside [`validate_backend`]; the umbrella stays
//! `#![forbid(unsafe_code)]` by funnelling both here.

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

/// Release pooled device memory back to the driver after a failed GPU compute.
///
/// **Every `*-gpu` metric crate must call this on its compute failure paths.**
///
/// cubecl's allocator keeps freed slices in a per-stream pool, so a compute that
/// dies part-way — a device OOM on an oversized image — leaves the whole
/// reservation parked in that pool even after the pipeline drops. MEASURED on a
/// 6 GB card (ssim2, 8192x8192): the failed attempt left **6,261,084,160 bytes**
/// reserved, and every later score in the same process then failed too,
/// including a 1024x1024 needing 99 MB. One oversized cell poisoned the worker
/// for every cell after it — the shape of an OOM storm.
///
/// After calling this, reserved drops to 0 and the next score succeeds normally
/// (measured: 1024x1024 back to 10.4 ms on CUDA, 11.9 ms on Vulkan).
///
/// Safe on an unhealthy stream: the CUDA server runs `memory_cleanup` with
/// `ignore: true, flush: false`, and the client submits it fire-and-forget
/// rather than blocking.
///
/// # Do not call this while panicking
/// The underlying `submit` takes the device-service mutex with `.lock().unwrap()`,
/// so calling it during an unwind on a poisoned mutex would abort the process.
/// Reclaim on the **error** path; convert panicking readbacks into errors rather
/// than reclaiming from a `Drop` guard. See imazen/zenmetrics#41.
pub fn release_device_pool<R: cubecl::Runtime>(client: &cubecl::client::ComputeClient<R>) {
    debug_assert!(
        !std::thread::panicking(),
        "release_device_pool must not run during unwind: cubecl's submit path \
         locks a mutex with unwrap(), which would abort"
    );
    client.memory_cleanup();
}

/// What a device can say about a prospective working set, from
/// [`device_fit`].
///
/// Every "no" names the number it failed against, so a caller can log an
/// actionable message or size a fallback rather than reporting "it did not
/// fit".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceFit {
    /// Within both the per-allocation ceiling and the device's capacity.
    Fits,
    /// Larger than the device's total memory. No amount of splitting helps —
    /// a strip/streaming path over the same working set will not rescue this,
    /// and a CPU path is the remaining option.
    ExceedsDeviceMemory {
        /// What was asked for, in bytes.
        requested: u64,
        /// What the device has, in bytes.
        total_memory: u64,
    },
    /// Larger than a single allocation may be, but not necessarily larger than
    /// the device. **This is the case a strip/streaming path exists for**: the
    /// same work in smaller pieces may well fit.
    ExceedsMaxAllocation {
        /// What was asked for, in bytes.
        requested: u64,
        /// The largest single allocation this device permits, in bytes.
        max_page_size: u64,
    },
    /// The backend does not report its capacity, so this could not be decided.
    ///
    /// Not a yes. Treat it as "proceed, but be ready to handle the failure" —
    /// never as unlimited memory.
    Unknown {
        /// What was asked for, in bytes.
        requested: u64,
    },
}

impl DeviceFit {
    /// Whether the device can be expected to hold this. False for
    /// [`Unknown`](DeviceFit::Unknown) — an undecidable answer is not a yes.
    pub fn fits(&self) -> bool {
        matches!(self, Self::Fits)
    }

    /// Whether splitting the work into smaller allocations could plausibly
    /// help — i.e. whether it is worth trying a strip/streaming path.
    ///
    /// False when the request exceeds the device outright, which is the signal
    /// to fall back to CPU rather than to a slower GPU path that will fail the
    /// same way.
    pub fn splitting_may_help(&self) -> bool {
        matches!(self, Self::ExceedsMaxAllocation { .. })
    }
}

/// Ask the device whether a working set of `bytes` is plausible, **before**
/// building anything.
///
/// The point is to choose an implementation rather than to discover the answer
/// by failing: finding out through a failed allocation pays the setup cost
/// first, and on some backends the failure surfaces only at a later checkpoint.
/// [`DeviceFit::splitting_may_help`] distinguishes "try the strip path" from
/// "this device cannot hold it at all, use CPU".
///
/// This is a static check against what the driver reports. It does not consult
/// free memory, so a [`DeviceFit::Fits`] answer is not a guarantee against a
/// device another process is using — allocation failures still have to be
/// handled. It rules out the impossible cheaply, which is the expensive half.
pub fn device_fit<R: cubecl::Runtime>(
    client: &cubecl::client::ComputeClient<R>,
    bytes: u64,
) -> DeviceFit {
    let mem = &client.properties().memory;
    match mem.can_allocate(bytes) {
        cubecl::ir::AllocationVerdict::Fits => DeviceFit::Fits,
        cubecl::ir::AllocationVerdict::ExceedsMaxPageSize { max_page_size } => {
            DeviceFit::ExceedsMaxAllocation {
                requested: bytes,
                max_page_size,
            }
        }
        cubecl::ir::AllocationVerdict::ExceedsDeviceMemory { total_memory } => {
            DeviceFit::ExceedsDeviceMemory {
                requested: bytes,
                total_memory,
            }
        }
        cubecl::ir::AllocationVerdict::Unknown => DeviceFit::Unknown { requested: bytes },
    }
}

/// Cheap non-cryptographic digest of an input buffer.
///
/// Used for exactly one question — "were these two inputs byte-identical?" —
/// on the rare degenerate-score path, and for retaining that answer across a
/// cached reference whose bytes the caller no longer holds. FNV-1a over
/// 8-byte chunks: a single streaming pass, ~2-3 ms on a 12 MP RGB frame,
/// negligible beside the pyramid build `set_reference` already does.
///
/// Not a security hash and not a content address — do not use it as either.
pub fn input_digest(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut h = OFFSET ^ (bytes.len() as u64);
    let (chunks, remainder) = bytes.as_chunks::<8>();
    for c in chunks {
        h = (h ^ u64::from_le_bytes(*c)).wrapping_mul(PRIME);
    }
    for &b in remainder {
        h = (h ^ b as u64).wrapping_mul(PRIME);
    }
    h
}

/// Which direction of a metric's scale means "more similar".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleDirection {
    /// Larger is more similar (ssim2 → 100, iwssim → 1, cvvdp → 10 JOD).
    HigherIsBetter,
    /// Smaller is more similar (dssim → 0, butteraugli → 0).
    LowerIsBetter,
}

/// **A metric must never report "these images are identical" for inputs that
/// are not byte-identical.** Returns `true` when it just did — i.e. the caller
/// is holding a silent wrong result and must return an error instead.
///
/// Why this exists, measured: with a device OOM upstream, `dssim-gpu` returned
/// **0.000000** — its exact identical-value — for two different synthetic
/// images, exited 0, and printed it as a normal score. Nothing downstream can
/// tell that from a real lossless cell. A dead reduction does NOT reliably
/// leave an all-zero buffer either (dssim computes `1/ssim - 1`, so zeroed
/// sums would give a huge number, not 0) — which is why this guards the
/// OUTPUT rather than the accumulator.
///
/// No false positives by construction: a genuinely byte-identical pair is
/// *supposed* to score the extremum, and this only fires when the digests
/// differ. Lossless corpora — where every cell is identical and scores the
/// extremum — are unaffected. That matters: they are common here, so a naive
/// "perfect score is suspicious" check would reject real data.
///
/// Cost is nil on the normal path: the float comparison short-circuits, and
/// the digest comparison only runs for a score that already claims identity.
pub fn is_silent_identical_claim(
    score: f64,
    identical_value: f64,
    direction: ScaleDirection,
    ref_digest: u64,
    dist_digest: u64,
) -> bool {
    let claims_identical = match direction {
        ScaleDirection::HigherIsBetter => score >= identical_value,
        ScaleDirection::LowerIsBetter => score <= identical_value,
    };
    claims_identical && ref_digest != dist_digest
}

/// Closure-based sibling of [`is_silent_identical_claim`], for callers that
/// still hold both inputs and can compare them exactly.
///
/// Prefer this where the bytes are in hand: comparing the two slices is exact,
/// where a digest match is only near-certain. Use the digest form on
/// warm-reference paths, where the reference bytes are gone by scoring time and
/// only the digest survives.
///
/// `same` is a closure so the comparison runs only on the degenerate path. The
/// float test short-circuits first, so an ordinary score pays nothing.
///
/// Returns `true` when the caller is holding a silent wrong result: the score
/// claims the images are identical, and they are not.
pub fn is_identical_claim(
    score: f64,
    identical_value: f64,
    direction: ScaleDirection,
    same: impl FnOnce() -> bool,
) -> bool {
    let claims_identical = match direction {
        ScaleDirection::HigherIsBetter => score >= identical_value,
        ScaleDirection::LowerIsBetter => score <= identical_value,
    };
    claims_identical && !same()
}

/// Backend liveness validation (imazen/zenmetrics#37): proves a runtime can
/// compile + dispatch a kernel before a metric trusts it. See the module docs.
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
mod validate;
#[cfg(any(feature = "cuda", feature = "wgpu", feature = "cpu"))]
pub use validate::validate_backend;

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

// ───────────────────────── flagged GPU timing (CI diagnosis) ─────────────────
//
// macos-latest Metal CI is dominated by per-crate release *compilation* (each
// `cargo test -p <crate>` rebuilds; 4–10 min each), but the runtime portion —
// Metal shader compilation on first kernel dispatch, upload, readback — is worth
// breaking down too. Set `ZENMETRICS_GPU_TIMING=1` and run with `--nocapture` to
// `eprintln!` the wall-time of timed phases. Off by default (one env read, no
// output), so it never affects normal runs or scores.

/// `true` when `ZENMETRICS_GPU_TIMING=1` — enables [`time_phase`] output.
pub fn gpu_timing_enabled() -> bool {
    std::env::var("ZENMETRICS_GPU_TIMING").as_deref() == Ok("1")
}

/// Time `f` and, when [`gpu_timing_enabled`], `eprintln!` `[gpu-timing] label:
/// <secs>s`. Returns `f`'s result unchanged. Zero-overhead (just the closure
/// call + one env read) when the flag is off. Wrap GPU phases (client build,
/// upload, dispatch, readback) to see where slow CI time goes — visible in test
/// output only with `--nocapture`.
pub fn time_phase<T>(label: &str, f: impl FnOnce() -> T) -> T {
    if !gpu_timing_enabled() {
        return f();
    }
    let t = std::time::Instant::now();
    let out = f();
    eprintln!("[gpu-timing] {label}: {:.3}s", t.elapsed().as_secs_f64());
    out
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

/// Sub-minimum-image pad plan shared by the opaque **and** typed `*-gpu`
/// entry points, so no public interface silently scores a degenerate
/// sub-floor pyramid: a request below the pyramid floor is reflect-padded
/// up to it and scored, returning a finite result down to 1×1.
///
/// Maps the caller's *logical* extent to the *padded* extent the pyramid
/// needs (`max(dim, min)` per axis) and pads/crops buffers between them
/// via [`reflect_pad`] / [`reflect_index`]. At ≥`min` on both axes it's a
/// no-op (borrows, never copies). A 0-dim axis stays 0 — callers reject
/// empty images before scoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PadPlan {
    logical_w: u32,
    logical_h: u32,
    padded_w: u32,
    padded_h: u32,
}

impl PadPlan {
    /// Pad each non-zero axis up to `min` (the pyramid floor).
    pub fn to_min(width: u32, height: u32, min: u32) -> Self {
        Self {
            logical_w: width,
            logical_h: height,
            padded_w: if width == 0 { 0 } else { width.max(min) },
            padded_h: if height == 0 { 0 } else { height.max(min) },
        }
    }

    /// Caller-requested `(width, height)` — what `dims()` should report.
    pub fn logical(&self) -> (u32, u32) {
        (self.logical_w, self.logical_h)
    }

    /// Padded `(width, height)` the inner pipeline allocates buffers for.
    pub fn padded(&self) -> (u32, u32) {
        (self.padded_w, self.padded_h)
    }

    /// `true` when the padded extent exceeds the logical one (the input
    /// must be reflect-padded). `false` is the zero-overhead fast path.
    pub fn is_padded(&self) -> bool {
        self.padded_w != self.logical_w || self.padded_h != self.logical_h
    }

    /// Expected packed length of a logical-extent buffer at `channels`
    /// elements per pixel (callers validate inputs against this).
    pub fn logical_len(&self, channels: usize) -> usize {
        self.logical_w as usize * self.logical_h as usize * channels
    }

    /// Reflect-pad a packed `channels`-per-pixel buffer (RGB8 → 3, a
    /// single linear plane → 1) from the logical extent up to the padded
    /// extent. Borrows unchanged when [`is_padded`](Self::is_padded) is
    /// false. Caller validates `src.len() == self.logical_len(channels)`.
    pub fn pad<'a, T: Copy>(&self, src: &'a [T], channels: usize) -> std::borrow::Cow<'a, [T]> {
        if !self.is_padded() {
            return std::borrow::Cow::Borrowed(src);
        }
        std::borrow::Cow::Owned(reflect_pad(
            src,
            self.logical_w as usize,
            self.logical_h as usize,
            self.padded_w as usize,
            self.padded_h as usize,
            channels,
        ))
    }

    /// Crop a padded-extent single-channel `f32` map (row-major, padded
    /// width stride) back to the logical extent — the top-left
    /// sub-rectangle, where the original pixels live after reflect-pad.
    /// No-op when not padded.
    pub fn crop_logical_plane(&self, buf: &mut Vec<f32>) {
        if !self.is_padded() {
            return;
        }
        let (lw, lh) = (self.logical_w as usize, self.logical_h as usize);
        let pw = self.padded_w as usize;
        if buf.len() < pw * lh {
            return;
        }
        let mut out = Vec::with_capacity(lw * lh);
        for y in 0..lh {
            let row = y * pw;
            out.extend_from_slice(&buf[row..row + lw]);
        }
        *buf = out;
    }
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
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
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
    for px in rgb.as_chunks::<3>().0 {
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

#[cfg(test)]
mod identical_claim_tests {
    //! One tested implementation of the guard predicate, shared by every metric
    //! crate, rather than a copy per crate that could drift.
    //!
    //! The property that matters most is the absence of false positives: a
    //! byte-identical pair *should* score the extremum, and lossless corpora —
    //! where every cell is identical — are common here. A guard that rejected
    //! those would throw away real data, which is worse than the bug it fixes.

    use super::*;

    /// The real metric constants, so a wrong threshold or direction in any
    /// crate is caught here rather than in production.
    const SSIM2: (f64, ScaleDirection) = (100.0, ScaleDirection::HigherIsBetter);
    const IWSSIM: (f64, ScaleDirection) = (1.0, ScaleDirection::HigherIsBetter);
    const CVVDP: (f64, ScaleDirection) = (10.0, ScaleDirection::HigherIsBetter);
    const BUTTER: (f64, ScaleDirection) = (0.0, ScaleDirection::LowerIsBetter);
    const DSSIM: (f64, ScaleDirection) = (0.0, ScaleDirection::LowerIsBetter);

    #[test]
    fn a_genuinely_identical_pair_is_never_rejected() {
        for (identical, dir) in [SSIM2, IWSSIM, CVVDP, BUTTER, DSSIM] {
            assert!(
                !is_identical_claim(identical, identical, dir, || true),
                "identical inputs scoring the extremum ({identical}, {dir:?}) must be accepted \
                 -- this is every cell of a lossless corpus"
            );
        }
    }

    #[test]
    fn the_extremum_claimed_for_differing_inputs_is_rejected() {
        for (identical, dir) in [SSIM2, IWSSIM, CVVDP, BUTTER, DSSIM] {
            assert!(
                is_identical_claim(identical, identical, dir, || false),
                "the extremum ({identical}, {dir:?}) claimed for inputs that differ is a \
                 silently wrong result and must be refused"
            );
        }
    }

    /// A corrupted reduction can overshoot the extremum, not just land on it.
    #[test]
    fn overshooting_the_extremum_is_rejected_too() {
        assert!(is_identical_claim(
            100.5,
            100.0,
            ScaleDirection::HigherIsBetter,
            || false
        ));
        assert!(is_identical_claim(
            -1.0e-9,
            0.0,
            ScaleDirection::LowerIsBetter,
            || false
        ));
    }

    #[test]
    fn ordinary_scores_are_untouched_either_way() {
        // Nothing short of the extremum ever fires, whether inputs match or not.
        for same in [true, false] {
            assert!(!is_identical_claim(
                87.3,
                100.0,
                ScaleDirection::HigherIsBetter,
                || same
            ));
            assert!(!is_identical_claim(
                0.42,
                0.0,
                ScaleDirection::LowerIsBetter,
                || same
            ));
            assert!(!is_identical_claim(
                9.1,
                10.0,
                ScaleDirection::HigherIsBetter,
                || same
            ));
        }
    }

    /// The closure must not run when the score is ordinary -- that is what keeps
    /// the guard free on the normal path.
    #[test]
    fn the_comparison_is_skipped_for_an_ordinary_score() {
        let mut ran = false;
        let fired = is_identical_claim(50.0, 100.0, ScaleDirection::HigherIsBetter, || {
            ran = true;
            false
        });
        assert!(!fired);
        assert!(
            !ran,
            "the input comparison must not run for an ordinary score"
        );
    }
}

#[cfg(test)]
mod device_fit_tests {
    //! The decision these encode is which implementation to run, so the two
    //! "too large" answers must stay distinguishable: one says try smaller
    //! pieces, the other says this device cannot hold it however you slice it.

    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn exceeding_one_allocation_points_at_the_streaming_path() {
        let f = DeviceFit::ExceedsMaxAllocation {
            requested: 4 * GIB,
            max_page_size: GIB,
        };
        assert!(!f.fits());
        assert!(
            f.splitting_may_help(),
            "the work may still fit in strips; this is what a streaming path is for"
        );
    }

    #[test]
    fn exceeding_the_device_points_at_cpu_instead() {
        let f = DeviceFit::ExceedsDeviceMemory {
            requested: 10 * GIB,
            total_memory: 2 * GIB,
        };
        assert!(!f.fits());
        assert!(
            !f.splitting_may_help(),
            "a strip path would fail the same way -- falling back to it wastes the attempt"
        );
    }

    /// An unreported capacity must not read as permission, and must not be
    /// mistaken for a reason to take the streaming path either.
    #[test]
    fn unknown_is_neither_a_yes_nor_a_reason_to_stripe() {
        let f = DeviceFit::Unknown { requested: 4 * GIB };
        assert!(!f.fits());
        assert!(!f.splitting_may_help());
    }

    #[test]
    fn fitting_is_a_yes_and_needs_no_fallback() {
        assert!(DeviceFit::Fits.fits());
        assert!(!DeviceFit::Fits.splitting_may_help());
    }
}
