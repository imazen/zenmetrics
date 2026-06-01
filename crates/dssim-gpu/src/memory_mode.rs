//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for the shared design rationale.
//!
//! dssim-gpu is **NOT strip-preferred**: strip mode is 2-5× slower
//! than whole-image on this crate (bench at
//! `benchmarks/dssim_strip_vs_whole_2026-05-22.md`). Auto picks Full
//! whenever it fits the cap; Strip only when Full would exceed the
//! cap.

use crate::NUM_SCALES;

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. Process-wide.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// Probe live free-VRAM. See `iwssim_gpu::memory_mode::live_vram_probe_bytes`.
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
    Some(bytes.saturating_sub(bytes / 10))
}

/// Effective cap policy: env → live probe → 8 GB. Identical to the
/// sibling metric crates (task #51).
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

/// How the GPU pipeline should partition its working set. dssim-gpu
/// supports `Auto`, `Full`, and `Strip`; `Tile` is reserved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    Auto,
    Full,
    Strip { h_body: Option<u32> },
    Tile { h: u32, w: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMode {
    Full,
    Strip { h_body: u32 },
}

/// Minimum strip body for dssim — must be a multiple of
/// `2^(NUM_SCALES − 1) = 16` per [`crate::pipeline::Dssim::new_strip`].
const PYRAMID_ALIGN: u32 = 1 << (NUM_SCALES as u32 - 1);
const MIN_STRIP_BODY: u32 = PYRAMID_ALIGN * 4; // 64 rows

/// Auto policy: prefer Full. dssim's strip walker is 2-5× slower than
/// whole-image, so we only switch to Strip when Full would exceed the
/// cap.
pub fn resolve_auto(width: u32, height: u32, cap: usize) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    if let Some(h_body) = auto_size_strip_body(width, height, cap) {
        return Ok(ResolvedMode::Strip { h_body });
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
}

/// Public auto-sizer for callers passing
/// `MemoryMode::Strip { h_body: None }`. Always returns a positive,
/// pyramid-aligned body.
#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap).unwrap_or_else(|| {
        MIN_STRIP_BODY
            .min(round_up_align(height))
            .max(PYRAMID_ALIGN)
    })
}

fn round_up_align(h: u32) -> u32 {
    h.div_ceil(PYRAMID_ALIGN) * PYRAMID_ALIGN
}

fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    let halo_bytes = estimate_strip_gpu_memory_bytes(width, 0)?;
    if halo_bytes >= cap {
        return None;
    }
    let one_row = estimate_strip_gpu_memory_bytes(width, PYRAMID_ALIGN)?;
    let per_align_unit = one_row.saturating_sub(halo_bytes);
    if per_align_unit == 0 {
        let fb = MIN_STRIP_BODY.min(round_up_align(height));
        let est = estimate_strip_gpu_memory_bytes(width, fb)?;
        return if est <= cap { Some(fb) } else { None };
    }
    let max_units = (cap - halo_bytes) / per_align_unit;
    let raw = (max_units as u32) * PYRAMID_ALIGN;
    let max_body = raw.min(round_up_align(height));
    if max_body < MIN_STRIP_BODY {
        // Try MIN_STRIP_BODY itself in case the linearization
        // undershoots — strip allocation has bounded constants.
        let est = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY)?;
        if est <= cap {
            return Some(MIN_STRIP_BODY.min(round_up_align(height)));
        }
        return None;
    }
    Some(max_body)
}

/// Number of f32 planes allocated per pyramid scale, counted directly
/// from [`crate::pipeline::Scale::new`] (verified against source
/// 2026-05-28): nine `alloc_3` triples (`ref_lin`, `dis_lin`,
/// `ref_lab`, `dis_lab`, `ref_mu`, `ref_sq_blur`, `dis_mu`,
/// `dis_sq_blur`, `cross_blur` = 27 planes) plus four single planes
/// (`temp1`, `temp2`, `ssim_map`, `mad_map`) = **31**. The prior value
/// of 13 was ~2.4× too low and is the main reason the estimator
/// under-predicted the measured peak by 55-64%.
const PLANES_PER_SCALE: usize = 31;

/// Fixed GPU-context base term added to every estimate.
///
/// The CUDA/WGPU context, cubecl kernel cache, and the per-instance
/// reduction `partials`/`sums` buffers cost VRAM that the working-set
/// pyramid sum does not capture. The previously-diagnosed flat
/// "256 MiB context" choice was wrong: the measured residual
/// (`measured_peak − raw_31_plane_working_set`) is NOT flat — it grows
/// 212 → 268 → 462 → 594 MiB across 1/4/16/40 MP (see
/// `benchmarks/gpu_metrics_sweep_2026-05-28.tsv`). A flat 256 MiB
/// constant under-predicts at 16 MP (−6.4%) and 40 MP (−4.7%) — the
/// UNSAFE direction for `resolve_auto`. The residual fits cleanly to a
/// linear model `≈ 234 MiB + 9.7 MiB/MP`, so we model it as a base
/// term plus a per-pixel overhead ([`CONTEXT_PER_PIXEL_BYTES`]) rather
/// than a single flat constant. Calibrated so every size OVER-predicts
/// within ±20%:
///   1 MP   418.7 MiB est / 385 MiB meas  (+3.7%)
///   4 MP   972.5 MiB est / 961 MiB meas  (+1.2%)
///  16 MP  3265.5 MiB est / 3233 MiB meas (+1.0%)
///  40 MP  7470.6 MiB est / 7169 MiB meas (+4.2%)
const CONTEXT_BASE_BYTES: usize = 208 * 1024 * 1024;

/// Per-pixel slice of the context/allocator overhead (page rounding,
/// per-buffer alignment slack, runtime staging that scales with image
/// area). See [`CONTEXT_BASE_BYTES`] for the residual-fit rationale.
const CONTEXT_PER_PIXEL_BYTES: usize = 18;

/// Estimate the GPU working-set bytes [`crate::pipeline::Dssim::new`]
/// allocates for `width × height` images, plus the fixed + per-pixel
/// GPU-context overhead.
///
/// 31 f32 planes per scale ([`PLANES_PER_SCALE`], counted from
/// `Scale::new`) across [`NUM_SCALES`] = 5 pyramid levels, two staging
/// packed-u32 sRGB buffers at scale 0 (`n0 × 4` each), plus the
/// context term ([`CONTEXT_BASE_BYTES`] + [`CONTEXT_PER_PIXEL_BYTES`] ·
/// `n0`). The context term over-predicts at all four calibration sizes,
/// which is the safe budgeting bias for `resolve_auto` — see the
/// constant docs.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    for _ in 0..NUM_SCALES {
        let w_eff = w.max(8) as usize;
        let h_eff = h.max(8) as usize;
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    // Two staging packed-u32 sRGB buffers (4 B per pixel).
    let n0 = (width as usize) * (height as usize);
    total = total.saturating_add(n0 * 4 * 2);
    // Fixed + per-pixel GPU-context overhead.
    total = total.saturating_add(CONTEXT_BASE_BYTES);
    total = total.saturating_add(n0.saturating_mul(CONTEXT_PER_PIXEL_BYTES));
    total
}

/// Strip-mode estimator. Same 31 planes/scale + context overhead as
/// [`estimate_gpu_memory_bytes`] but the pyramid is sized for
/// `width × (h_body + 2 × HALO)`. HALO = 256 is fixed in
/// [`crate::pipeline::Dssim::new_strip`]. The per-pixel context term is
/// billed on the strip pixel count, so the marginal cost per added
/// pyramid-aligned body row stays positive and [`auto_size_strip_body`]
/// can linearize it.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    const HALO: u32 = 256;
    let strip_h = (h_body as usize).saturating_add((HALO as usize) * 2);
    let mut w = width as usize;
    let mut h = strip_h;
    let mut total: usize = 0;
    for _ in 0..NUM_SCALES {
        let w_eff = w.max(8);
        let h_eff = h.max(8);
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    // Two staging packed-u32 sRGB buffers at the strip pixel count.
    let n = (width as usize).saturating_mul(strip_h);
    total = total.saturating_add(n * 4 * 2);
    // Fixed + per-strip-pixel GPU-context overhead.
    total = total.saturating_add(CONTEXT_BASE_BYTES);
    total = total.saturating_add(n.saturating_mul(CONTEXT_PER_PIXEL_BYTES));
    Some(total)
}
