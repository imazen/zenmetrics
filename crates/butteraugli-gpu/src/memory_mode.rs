//! Unified memory-mode API. Surfaces the [`MemoryMode`] enum that
//! every metric crate in zenmetrics exposes, plus the per-crate
//! Auto-resolution policy that picks between Full and Strip based on
//! an estimated working-set vs. an explicit (or env-var) VRAM cap.
//!
//! butteraugli-gpu is **strip-preferred**: when both modes fit the
//! cap, Auto picks Strip — the strip walker is 1.9-4.9× faster than
//! whole-image on this crate per the bench at
//! `benchmarks/butter_strip_vs_whole_2026-05-21.md`.
//!
//! ## Backwards compatibility
//!
//! [`crate::pipeline::Butteraugli::new`] keeps its historical
//! `(client, width, height)` signature and now delegates through
//! `new_with_memory_mode(.., MemoryMode::Auto)`. Existing callers see
//! no source-level change.

/// Optional cap (in bytes) on the GPU working set Auto is allowed to
/// allocate. Read from `ZENMETRICS_VRAM_CAP_BYTES` (decimal usize);
/// returns `None` when unset or unparseable so the resolver can fall
/// back to its default heuristics.
fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. The query takes
/// 50-200 ms per invocation; we cache process-wide so the hot path
/// stays sub-microsecond after first init.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// Probe live free-VRAM via `nvidia-smi --query-gpu=memory.free`.
/// Returns `Some(bytes)` on success, `None` when nvidia-smi is
/// unavailable or its output can't be parsed (AMD/Intel GPUs, CI
/// runners without a CUDA driver, exotic distros).
///
/// Cached process-wide; subsequent calls return the same value
/// without re-querying. Mirrors `iwssim_gpu::memory_mode::live_vram_probe_bytes`.
pub fn live_vram_probe_bytes() -> Option<usize> {
    *LIVE_PROBE_CACHE.get_or_init(query_nvidia_smi_memory_free)
}

fn query_nvidia_smi_memory_free() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mb: u64 = s.lines().next()?.trim().parse().ok()?;
    let bytes = (mb as usize).saturating_mul(1024 * 1024);
    // 10% safety margin so a freshly-probed cap doesn't put us at
    // 99% occupancy. Sibling cubecl clients (other metrics + the
    // runtime's own kernel cache) share the pool.
    Some(bytes.saturating_sub(bytes / 10))
}

/// Effective cap policy (task #51 — live VRAM probe across all 6 crates):
/// 1. `ZENMETRICS_VRAM_CAP_BYTES` env var (always wins).
/// 2. Live `nvidia-smi` probe (cached process-wide, 10% headroom).
/// 3. 8 GB default for non-NVIDIA / CI / exotic environments.
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    if let Some(probed) = live_vram_probe_bytes() {
        return probed;
    }
    8 * 1024 * 1024 * 1024
}

/// Reclaim pooled-but-unreferenced device memory back to the driver
/// for `backend`.
///
/// cubecl pools GPU buffers: dropping a metric instance returns its
/// `Handle`s to the pool's free list, but the underlying device pages
/// stay resident for reuse — so a user who drops a metric does **not**
/// immediately get VRAM back, and an orchestrator that swaps between
/// metrics sees peak trend toward the SUM of their working sets instead
/// of the MAX. This function issues cubecl's
/// `ComputeClient::memory_cleanup` hint (which deallocates fully-free
/// pool pages) followed by a `sync` (which flushes the CUDA
/// deferred-free queue so `cuMemFree*` actually runs), returning the
/// freed pages to the driver.
///
/// **Thread/stream scoped.** cubecl's CUDA memory pool is per-stream
/// and the stream is selected by the *calling thread's* id, so this
/// only reclaims the pool owned by the thread that calls it. Call it
/// from the same thread that dropped the metric instance.
///
/// **Do NOT call between scores of the same warm metric.** Reclaiming
/// while a live instance still references pool pages can deallocate /
/// relocate pages that an in-flight binding points at (the cubecl
/// allocator panics on the next dispatch), and it discards the warm
/// working set the next score would have reused — regressing the warm
/// per-call path. The intended call sites are: after a metric is
/// dropped (user reclaim), and at an orchestrator metric-signature
/// swap (after dropping the old instance, before constructing the new
/// one) or when going idle. Best-effort: cubecl frees only what its
/// allocator deems beneficial.
#[allow(unused_variables)]
pub fn reclaim_pooled_vram(backend: crate::opaque::Backend) {
    use crate::opaque::Backend;
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => {
            use cubecl::Runtime;
            let client = cubecl::cuda::CudaRuntime::client(&Default::default());
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
        #[cfg(feature = "wgpu")]
        Backend::Wgpu => {
            use cubecl::Runtime;
            let client = cubecl::wgpu::WgpuRuntime::client(&Default::default());
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
        #[cfg(feature = "cpu")]
        Backend::Cpu => {
            use cubecl::Runtime;
            let client = cubecl::cpu::CpuRuntime::client(&Default::default());
            client.memory_cleanup();
            let _ = cubecl::future::block_on(client.sync());
        }
    }
}

/// How the GPU pipeline should partition its working set.
///
/// See module-level docs for the Auto policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full or Strip automatically based on
    /// [`vram_cap_bytes()`]. butteraugli-gpu prefers Strip when it
    /// fits — see module-level docs.
    Auto,
    /// Allocate one working set sized for the whole image. Equivalent
    /// to historical `Butteraugli::new`.
    Full,
    /// Allocate one working set sized for a strip of `h_body` body
    /// rows + the crate's halo per side. `h_body: None` lets the
    /// resolver auto-size; `Some(n)` pins to `n`.
    Strip { h_body: Option<u32> },
    /// 2-D tiling. Reserved for a future implementation; currently
    /// returns [`crate::Error::ModeUnsupported`] at construction.
    Tile { h: u32, w: u32 },
}

/// Outcome of resolving [`MemoryMode::Auto`] for a given
/// `(width, height)` and crate. Internal-use; surfaced for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMode {
    Full,
    Strip { h_body: u32 },
}

/// Minimum h_body the butteraugli strip walker accepts. Must be a
/// positive integer; the strip path itself has no hard lower bound,
/// but very small bodies waste launch overhead. 64 rows mirrors what
/// the bench at `benchmarks/butter_strip_vs_whole_2026-05-21.md` was
/// tuned against.
const MIN_STRIP_BODY: u32 = 64;

/// Resolve [`MemoryMode::Auto`] for butteraugli-gpu (strip-preferred).
///
/// Policy:
/// 1. If Strip fits the cap with a body of at least [`MIN_STRIP_BODY`]
///    rows, pick Strip. (Strip is faster than Full on this crate per
///    the published bench, so we prefer it whenever it fits.)
/// 2. Else if Full fits the cap, pick Full.
/// 3. Else return [`crate::Error::TooBigForFull`] with the smaller of
///    the two estimates so the caller knows the gap.
pub fn resolve_auto(width: u32, height: u32, cap: usize) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);

    // Strip is only worthwhile when image_h is large enough that the
    // strip working set (body + 2 × halo) is meaningfully smaller than
    // the whole image. At image_h ≤ MIN_STRIP_BODY + 2 × HALO_ROWS the
    // single-strip case fully covers the image with the halo "spilling"
    // past the edges via reflection — strip allocates AT LEAST as much
    // as Full, the dispatch path degenerates, and small-image edge
    // cases (sub-128 px thumbnails, opaque-shim tests) hit awkward
    // dimension-check paths in the walker. Fall through to Full there.
    let min_strip_image_h = MIN_STRIP_BODY + 2 * crate::strip::HALO_ROWS;
    if height > min_strip_image_h {
        // Try strip first — even if Full fits, butter is strip-preferred.
        if let Some(h_body) = auto_size_strip_body(width, height, cap) {
            return Ok(ResolvedMode::Strip { h_body });
        }
    }
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Last-ditch strip attempt for big images that don't fit Full —
    // even at suboptimal small h_body, strip beats OOM.
    if let Some(h_body) = auto_size_strip_body(width, height, cap) {
        return Ok(ResolvedMode::Strip { h_body });
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
}

/// Public auto-sizer for callers that pass
/// `MemoryMode::Strip { h_body: None }`. Returns a body size that fits
/// the cap when possible; falls back to [`MIN_STRIP_BODY`] (clamped to
/// `height`) when the cap is too tight for the linearized estimate.
/// Never returns 0 — the strip constructor itself rejects `h_body == 0`.
#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap).unwrap_or_else(|| MIN_STRIP_BODY.min(height).max(1))
}

/// Pick the largest h_body that fits the cap, clamped to
/// `[MIN_STRIP_BODY, image_h]`. Returns `None` if even `MIN_STRIP_BODY`
/// rows of body exceed the cap.
fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    // Strip constant overhead = halo bytes + LUT bytes + scratch that
    // doesn't scale with body height. Approximate as the bytes used
    // by a body-0 strip (= 2 × HALO_ROWS rows of working set).
    let halo_bytes = estimate_strip_gpu_memory_bytes(width, 0)?;
    if halo_bytes >= cap {
        return None;
    }
    // Per-body-row marginal cost = estimate_strip(_, h_body=1) − halo_bytes.
    let one_row = estimate_strip_gpu_memory_bytes(width, 1)?;
    let per_row = one_row.saturating_sub(halo_bytes);
    if per_row == 0 {
        // Pathological: estimator didn't grow with body. Fall back to
        // a fixed body that's likely to fit.
        let fallback = MIN_STRIP_BODY.min(height);
        let est = estimate_strip_gpu_memory_bytes(width, fallback)?;
        return if est <= cap { Some(fallback) } else { None };
    }
    let max_body = ((cap - halo_bytes) / per_row).min(height as usize) as u32;
    if max_body < MIN_STRIP_BODY {
        // Try MIN_STRIP_BODY explicitly — the linearization might
        // underestimate; the real allocator may still fit.
        let est = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY)?;
        if est <= cap {
            return Some(MIN_STRIP_BODY.min(height));
        }
        return None;
    }
    Some(max_body)
}

/// Estimate the GPU working-set bytes [`Butteraugli::new`](crate::pipeline::Butteraugli::new)
/// allocates for a `width × height` image. Pure function; performs no
/// allocation. Mirrors `cvvdp_gpu::estimate_gpu_memory_bytes` in
/// purpose.
///
/// Counted buffers (per the `new` constructor in
/// `crates/butteraugli-gpu/src/pipeline.rs`):
/// - `src_u8_a`, `src_u8_b`: packed-u32 sRGB staging, `n × 4` bytes each
/// - 4 plane-3 bundles (`lin_a/b`, `blur_a/b`): 4 × 3 = 12 planes × `n × 4`
/// - 2 freq plane-3 bundles × 4 sub-bands: 24 planes × `n × 4`
/// - `block_diff_dc/ac`: 6 planes × `n × 4`
/// - `mask`, `mask_scratch`, `cached_blurred_a`, `diffmap_buf`,
///   `temp1`, `temp2`: 6 planes × `n × 4`
/// - 5 blur LUTs: ≤ 67 floats each → negligible (< 2 KB total)
///
/// = 2 packed + 12 + 24 + 6 + 6 = 50 planes × `n × 4` bytes (ignoring
/// LUTs).
///
/// **Caveat**: ignores the per-call transient pinned upload buffer and
/// cubecl metadata; callers concerned with hitting hard VRAM ceilings
/// should add a ~10-15% safety factor on top.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let n = (width as usize).saturating_mul(height as usize);
    const PLANES: usize = 50;
    PLANES.saturating_mul(n).saturating_mul(4)
}

/// Strip-mode estimator. Same set of planes as
/// [`estimate_gpu_memory_bytes`] but sized for
/// `width × (h_body + 2 × HALO_ROWS)`.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    #[cfg(feature = "cubecl-types")]
    let halo = crate::strip::HALO_ROWS;
    // Keep in sync with `crate::strip::HALO_ROWS` for the
    // `cubecl-types`-off build (the const lives behind that feature).
    #[cfg(not(feature = "cubecl-types"))]
    let halo: u32 = 80;
    let strip_h = (h_body as usize).saturating_add((halo as usize).saturating_mul(2));
    let n = (width as usize).saturating_mul(strip_h);
    const PLANES: usize = 50;
    Some(PLANES.saturating_mul(n).saturating_mul(4))
}
