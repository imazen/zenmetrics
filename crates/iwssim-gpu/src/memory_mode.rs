//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! iwssim-gpu is **NOT strip-preferred**: strip mode is ~1.7× slower
//! than whole-image on this crate (the cached-reference strip path
//! is deferred — see `docs/STRIP_PROCESSING.md`). Auto picks Full
//! whenever it fits the cap.

use crate::{MIN_NATIVE_DIM, NUM_SCALES};

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. The query takes
/// 50–200 ms per invocation; we read it at most once per process
/// run so [`vram_cap_bytes`] stays sub-microsecond on the hot path.
/// Wrapped in `OnceLock` so the cache is thread-safe and lock-free
/// after first init.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> =
    std::sync::OnceLock::new();

/// Probe live free-VRAM via `nvidia-smi --query-gpu=memory.free`.
/// Returns `Some(bytes)` on success, `None` when `nvidia-smi` is
/// unavailable or its output can't be parsed (e.g. AMD/Intel GPUs,
/// CI runners without a CUDA driver, exotic distros).
///
/// The result is **cached process-wide** — subsequent calls return
/// the same value without re-querying. This matches the intent: the
/// cap is a budgeting hint, not a live tracker. If the GPU's free
/// memory drops between calls (other processes allocating) the cap
/// stays at the probed value; that's intentional, since refusing
/// work mid-sweep would be worse than over-committing slightly.
///
/// Override via `ZENMETRICS_VRAM_CAP_BYTES` if the cached value
/// becomes stale — env var always wins over the probe.
pub fn live_vram_probe_bytes() -> Option<usize> {
    *LIVE_PROBE_CACHE.get_or_init(query_nvidia_smi_memory_free)
}

/// Single-shot query of `nvidia-smi --query-gpu=memory.free`. Internal
/// helper — callers should use [`live_vram_probe_bytes`] which caches
/// the result. Mirrors `zen_cloud_vastai::worker::adapt::nvidia_smi_total_memory_mb`
/// but queries `memory.free` (what we actually want for capacity
/// planning) rather than `memory.total`.
fn query_nvidia_smi_memory_free() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let mb: u64 = s.lines().next()?.trim().parse().ok()?;
    // Apply a safety factor: keep 10% headroom so a freshly-probed
    // cap doesn't immediately put us at 99% occupancy. The IW-SSIM
    // pipeline allocates a chunk of staging buffers on top of the
    // estimated working set, and other live cubecl clients (other
    // metrics, the runtime's own kernel cache) share the pool.
    let bytes = (mb as usize).saturating_mul(1024 * 1024);
    Some(bytes.saturating_sub(bytes / 10))
}

/// Cap policy: env var first (`ZENMETRICS_VRAM_CAP_BYTES`), then
/// live `nvidia-smi` probe (cached process-wide, 10% safety factor),
/// then the 8 GiB default for environments without an NVIDIA GPU
/// (CI, AMD/Intel boxes, WGPU backend on macOS/etc.).
///
/// The probe is **best-effort** — when `nvidia-smi` is missing or
/// fails (no CUDA driver, snap-docker, etc.) we fall through to the
/// 8 GiB default and the existing strip/full resolver does the
/// right thing. The probe is never a hard requirement.
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    if let Some(probed) = live_vram_probe_bytes() {
        return probed;
    }
    8 * 1024 * 1024 * 1024
}

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

const PYRAMID_ALIGN: u32 = 1 << (NUM_SCALES as u32 - 1); // 16
const STRIP_DEFAULT_HALO: u32 = 256;

/// Resolve [`MemoryMode::Auto`] for iwssim-gpu (Full-preferred).
///
/// Policy (matches the canonical shape from
/// `butteraugli_gpu::memory_mode::resolve_auto` minus the
/// strip-preferred first-pass):
///
/// 1. If Full fits the cap, pick Full. iwssim's strip walker is ~1.7×
///    slower than whole-image (see `docs/STRIP_PROCESSING.md`), so we
///    only fall back to Strip when Full is impossible.
/// 2. Else if a pyramid-aligned strip body fits the cap **and** both
///    axes are ≥ [`MIN_NATIVE_DIM`] (the floor enforced by
///    [`crate::pipeline::Iwssim::new_strip_with_halo`]), pick Strip
///    with the auto-sized body. This is the "2-pass iwssim fallback"
///    that prevents TooBigForFull on large images.
/// 3. Else return [`crate::Error::TooBigForFull`] with the Full
///    estimate so callers can see the gap.
///
/// The MIN_NATIVE_DIM guard is **not** a relaxation of step 2 — strip
/// genuinely cannot construct below that floor (`new_strip_with_halo`
/// rejects with `InvalidImageSize`). For sub-MIN_NATIVE_DIM inputs the
/// only memory-fitting paths are (a) explicit `MemoryMode::Full` with
/// `IwssimConfig::allow_small`, or (b) raising the cap. Auto cannot
/// pick either on the caller's behalf; surfacing TooBigForFull is the
/// honest answer.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Full exceeds the cap — try Strip before giving up. This is the
    // 2-pass-iwssim auto-fallback: `Iwssim::new_strip` works whenever
    // both axes are ≥ MIN_NATIVE_DIM and an aligned body fits, and
    // it's the only way to score images that don't fit Full.
    if width >= MIN_NATIVE_DIM && height >= MIN_NATIVE_DIM {
        if let Some(h_body) = auto_size_strip_body(width, height, cap) {
            return Ok(ResolvedMode::Strip { h_body });
        }
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
}

#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap)
        .unwrap_or(crate::pipeline::STRIP_DEFAULT_BODY.min(round_align(height)))
        .max(PYRAMID_ALIGN)
}

fn round_align(h: u32) -> u32 {
    h.div_ceil(PYRAMID_ALIGN) * PYRAMID_ALIGN
}

fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    let halo_bytes = estimate_strip_gpu_memory_bytes(width, 0)?;
    if halo_bytes >= cap {
        return None;
    }
    let one_unit = estimate_strip_gpu_memory_bytes(width, PYRAMID_ALIGN)?;
    let per_unit = one_unit.saturating_sub(halo_bytes);
    if per_unit == 0 {
        let fb = crate::pipeline::STRIP_DEFAULT_BODY.min(round_align(height));
        let est = estimate_strip_gpu_memory_bytes(width, fb)?;
        return if est <= cap { Some(fb) } else { None };
    }
    let max_units = (cap - halo_bytes) / per_unit;
    let raw = (max_units as u32) * PYRAMID_ALIGN;
    let body = raw.min(round_align(height));
    if body < PYRAMID_ALIGN {
        return None;
    }
    Some(body)
}

/// Reduction/cov scratch buffers allocated once per `Iwssim` instance,
/// independent of image size. Hand-counted from
/// [`crate::pipeline::Iwssim::new`] (constants verified against source
/// 2026-05-28):
///
/// - `partials` = `NUM_SLOTS(9) · NUM_BLOCKS(16) · BLOCK_SIZE(256)` f32
/// - `sums`     = `NUM_SLOTS(9)` f32
/// - `cov_partials` = `COV_MAX_CELLS(100) · COV_N_THREADS(64·256)` f32
///
/// Total ≈ 6.39 MiB. Negligible at 16/40 MP but a real fixed cost the
/// previous estimator omitted entirely.
const REDUCTION_SCRATCH_BYTES: usize = (9 * 16 * 256) * 4 + 9 * 4 + (100 * 64 * 256) * 4;

/// Pool / runtime-overhead multiplier applied to the raw working-set
/// sum. cubecl's allocator page-rounds each buffer, keeps a kernel
/// cache, and the CUDA/WGPU context itself reserves VRAM that scales
/// with (but is not captured by) the working-set arithmetic. The
/// previous estimator applied no factor at all and under-predicted the
/// measured peak by 55–88% (see `benchmarks/gpu_metrics_sweep_2026-05-28.tsv`).
///
/// Calibrated to 1.40 — **not** the 1.18 first proposed — because at
/// 1.18 the 4 MP full estimate lands at −22.3% vs the measured peak,
/// which is the UNSAFE (under-prediction) direction: `resolve_auto`
/// would pick Full when a memory-bounded mode is actually needed. At
/// 1.40 the estimator over-predicts at 16/40 MP and stays within ±20%
/// at 4 MP, which is the safe budgeting bias. Validated against the
/// committed cuda full-mode peaks:
///   1 MP  256.0 MiB est / 513 MiB meas  (floor-bound, exempt — see [`POOL_FLOOR_BYTES`])
///   4 MP  620.7 MiB est / 673 MiB meas  (−7.8%)
///  16 MP 2455.8 MiB est / 2209 MiB meas (+11.2%)
///  40 MP 5815.4 MiB est / 5025 MiB meas (+15.7%)
const POOL_FACTOR_NUM: usize = 7;
const POOL_FACTOR_DEN: usize = 5; // 7/5 = 1.40

/// Floor: the smallest VRAM an `Iwssim` instance realistically pins
/// once the cubecl context + kernel cache are warm. At 1 MP the raw
/// working set (≈116 MiB × pool) sits below the measured 513 MiB
/// because the context/runtime fixed cost dominates; clamp up so we
/// never wildly under-budget tiny inputs. The 1 MP cell remains
/// ≈−50% vs measured even with this floor (the fixed context cost is
/// irreducible at that size), which is the documented small-size
/// exemption — `resolve_auto` always picks Full for 1 MP anyway, so
/// the under-budget there is harmless.
const POOL_FLOOR_BYTES: usize = 256 * 1024 * 1024;

fn apply_pool_and_floor(raw: usize) -> usize {
    let pooled = raw.saturating_mul(POOL_FACTOR_NUM) / POOL_FACTOR_DEN;
    pooled.max(POOL_FLOOR_BYTES)
}

/// Estimate the GPU working-set bytes [`crate::pipeline::Iwssim::new`]
/// allocates for `width × height` images.
///
/// This is **Wang & Li Laplacian-pyramid IW-SSIM** (`NUM_SCALES = 5`,
/// no orientation dimension — there is no steerable-pyramid /
/// orientation subband decomposition here). Each of the 5 pyramid
/// scales allocates 19 f32 planes in [`crate::pipeline::Scale::new`]
/// (`lp_ref`/`lp_dis`/`g_ref`/`g_dis` at LP shape, `gh_*` ×5 at the
/// horizontal-pass shape, `mu1`/`mu2`/`m11`/`m22`/`m12`/`cs` at SSIM
/// shape, `g_buf`/`vv_buf`/`parent_band` at LP shape, `iw` at IW
/// shape, plus tiny fixed `cu`/`cu_inv_dev`/`lambda_dev`). We bill all
/// 19 at full `w × h` per scale — a deliberate slight over-count, since
/// the `gh_*`/`mu*`/`cs`/`iw` planes are a few rows/cols smaller. Two
/// packed-u32 sRGB staging buffers are sized at the scale-0 image
/// (`n0 × 4 × 2`). The fixed reduction/cov scratch
/// ([`REDUCTION_SCRATCH_BYTES`], ≈6.39 MiB) is added once. The raw sum
/// is then scaled by the pool factor and clamped to the floor — see
/// [`apply_pool_and_floor`].
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 19;
    for _ in 0..NUM_SCALES {
        let w_eff = w as usize;
        let h_eff = h as usize;
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n0 = (width as usize) * (height as usize);
    total = total.saturating_add(n0 * 4 * 2);
    total = total.saturating_add(REDUCTION_SCRATCH_BYTES);
    apply_pool_and_floor(total)
}

/// Strip-mode estimator. Same 19 planes/scale + reduction scratch +
/// pool + floor as [`estimate_gpu_memory_bytes`], but the pyramid is
/// sized for `width × (h_body + 2 × halo)` (default halo = 256) rather
/// than the full image height. Used by [`auto_size_strip_body`] to find
/// the largest pyramid-aligned body that fits the cap.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    let strip_h = (h_body as usize).saturating_add((STRIP_DEFAULT_HALO as usize) * 2);
    let mut w = width as usize;
    let mut h = strip_h;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 19;
    for _ in 0..NUM_SCALES {
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w * h * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n = (width as usize).saturating_mul(strip_h);
    total = total.saturating_add(n * 4 * 2);
    total = total.saturating_add(REDUCTION_SCRATCH_BYTES);
    Some(apply_pool_and_floor(total))
}
