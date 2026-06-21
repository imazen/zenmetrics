//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! **Phase 2 (strip processing) SHIPPED 2026-05-22.** Strip mode now
//! sizes buffers to `(image_w, h_body + 2*halo)` per the design in
//! `docs/STRIP_PROCESSING.md`; the previous "Strip unsupported"
//! behaviour is gone. Auto resolves to Strip when Full doesn't fit
//! the cap. Tile remains unsupported.

use crate::NUM_SCALES;

/// Halo budget at the finest scale, per side, in original-frame rows.
/// Per `docs/STRIP_PROCESSING.md`: cumulative reach from the IIR (4
/// taps per scale) and the LP downscale (1 row per cascade) across the
/// 6-level pyramid gives a theoretical max of ~160 rows; we round up
/// to 256 for f32-noise headroom AND to keep halo divisible by every
/// downscale-by-2 cascade (256 = 2^8). The same number iwssim uses.
pub const STRIP_HALO_ROWS: u32 = 256;

/// Default body height when callers pass `MemoryMode::Strip { h_body: None }`.
/// Per `docs/STRIP_PROCESSING.md` sweet spot: 1024 body + 2×256 halo = 1536
/// strip height, ~50% halo overhead at 24 MP, ~2.87 GB working set on a
/// 24 MP image across all 6 pyramid scales (down from ~7.5 GB Full —
/// measured via `examples/bench_strip_vs_whole.rs` 2026-05-22). The
/// scale-0 alone is ~2.1 GB; the pyramid geometric series adds ~37% on
/// top of that. Earlier design-doc estimate of 1.4 GB counted scale 0
/// only.
pub const STRIP_H_BODY_DEFAULT: u32 = 1024;

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. Process-wide so the
/// hot path stays sub-microsecond after first init.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// Probe live free-VRAM via `nvidia-smi --query-gpu=memory.free`.
/// Mirrors `iwssim_gpu::memory_mode::live_vram_probe_bytes`; see that
/// for cache semantics + 10% headroom rationale.
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

/// Effective cap policy (task #51): env var → live nvidia-smi probe → 8 GB.
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
    /// Strip-mode resolution carrying the body height that fits the
    /// cap. `h_body + 2 * STRIP_HALO_ROWS` is the per-strip allocation
    /// height; the constructor uses `min(h_body, image_h)` so for tiny
    /// images it degenerates to a single-strip computation.
    Strip {
        h_body: u32,
    },
}

/// Auto policy. Picks Full when it fits the cap; otherwise tries Strip
/// with an auto-sized body that fits the cap. If neither Full nor any
/// strip body fits, surfaces [`crate::Error::TooBigForFull`] with the
/// Full estimate — Tile isn't implemented in ssim2-gpu so there's no
/// smaller option.
///
/// Auto-sizing the strip body (rather than using only the default
/// [`STRIP_H_BODY_DEFAULT`]) matches the canonical shape from
/// `butteraugli_gpu::memory_mode::resolve_auto`: when the default body
/// doesn't fit, walk down to the largest aligned body that does. This
/// is the 2-pass-fallback the unified MemoryMode API promises.
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
/// `MemoryMode::Strip { h_body: None }`. Returns a body height that
/// fits the cap when possible; falls back to
/// [`STRIP_H_BODY_DEFAULT`] (clamped to `height`) when the cap is too
/// tight for the linearized estimate so the strip constructor itself
/// has a chance to allocate.
#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap).unwrap_or_else(|| {
        STRIP_H_BODY_DEFAULT
            .min(height.max(MIN_STRIP_BODY))
            .max(MIN_STRIP_BODY)
    })
}

/// Minimum body height. Smaller than the default but still safely
/// above the per-scale 8-row floor enforced by
/// [`crate::pipeline::Ssim2::new_strip`]. 64 rows mirrors the
/// butteraugli/dssim convention and gives the IIR cascade enough
/// rows to settle at every pyramid scale.
const MIN_STRIP_BODY: u32 = 64;

/// Pick the largest body that fits the cap, clamped to
/// `[MIN_STRIP_BODY, height]`. Returns `None` if even
/// `MIN_STRIP_BODY` exceeds the cap.
fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    if width < 8 {
        return None;
    }
    // Halo-only baseline (body = 0 isn't valid per the estimator's own
    // contract, but we treat its limit as "everything except the body
    // rows"). Approximate by the smallest valid body — the result is
    // a linear lower bound the search starts from.
    let one_unit = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY)?;
    if one_unit > cap {
        return None;
    }
    // Try the default body first — most images fit there.
    let h_default = STRIP_H_BODY_DEFAULT.min(height.max(MIN_STRIP_BODY));
    if let Some(b) = estimate_strip_gpu_memory_bytes(width, h_default)
        && b <= cap
    {
        return Some(h_default);
    }
    // Default doesn't fit — linear-extrapolate to the largest body
    // that fits and clamp to the image height.
    let two_unit = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY * 2)?;
    let per_extra = two_unit.saturating_sub(one_unit);
    if per_extra == 0 {
        // Pathological: estimator didn't grow with body. Try a few
        // discrete sizes in decreasing order.
        for candidate in [
            STRIP_H_BODY_DEFAULT / 2,
            STRIP_H_BODY_DEFAULT / 4,
            MIN_STRIP_BODY,
        ] {
            let h = candidate.min(height.max(MIN_STRIP_BODY));
            if let Some(b) = estimate_strip_gpu_memory_bytes(width, h)
                && b <= cap
            {
                return Some(h);
            }
        }
        return None;
    }
    // one_unit covers MIN_STRIP_BODY rows; each additional MIN_STRIP_BODY
    // rows costs `per_extra` bytes. Solve for the multiplier.
    let headroom = cap - one_unit;
    let extra_units = headroom / per_extra;
    let body = MIN_STRIP_BODY.saturating_add((extra_units as u32).saturating_mul(MIN_STRIP_BODY));
    let body = body.min(height.max(MIN_STRIP_BODY)).max(MIN_STRIP_BODY);
    // Verify the result actually fits — the linearization is exact
    // for this estimator but the saturating arithmetic above might
    // have over-counted at extreme cap values. Walk back if needed.
    if let Some(b) = estimate_strip_gpu_memory_bytes(width, body)
        && b <= cap
    {
        return Some(body);
    }
    // Final fallback — try MIN_STRIP_BODY itself.
    if let Some(b) = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY)
        && b <= cap
    {
        return Some(MIN_STRIP_BODY);
    }
    None
}

/// Estimate the GPU working-set bytes
/// [`crate::pipeline::Ssim2::new`] allocates for `width × height`
/// images.
///
/// After the 2026-05-21 plane-aliasing Phase 1 each `Scale` allocates
/// 57 planes (was 81 before), per the docstring on `Scale::new`. Six
/// scales total, plus 2 packed-u32 sRGB staging buffers at scale 0.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 57;
    for _ in 0..NUM_SCALES {
        if w < 8 || h < 8 {
            break;
        }
        let n = (w as usize) * (h as usize);
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(n * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n0 = (width as usize) * (height as usize);
    total = total.saturating_add(n0 * 4 * 2);
    total
}

/// Strip-mode estimator — Phase 2 (2026-05-22). Returns the per-strip
/// working-set bytes the [`crate::pipeline::Ssim2::new_strip`]
/// constructor allocates given `(width, h_body)`. The strip height at
/// scale 0 is `h_body + 2 * STRIP_HALO_ROWS`; subsequent pyramid
/// levels halve in both dimensions. Same 57-planes-per-scale +
/// 2-staging-buffers layout as Full.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    if width < 8 || h_body == 0 {
        return None;
    }
    let strip_h = h_body.saturating_add(2 * STRIP_HALO_ROWS);
    let mut w = width;
    let mut h = strip_h;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 57;
    for _ in 0..NUM_SCALES {
        if w < 8 || h < 8 {
            break;
        }
        let n = (w as usize) * (h as usize);
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(n * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n0 = (width as usize) * (strip_h as usize);
    total = total.saturating_add(n0 * 4 * 2);
    Some(total)
}

// ---------------------------------------------------------------------------
// Score-time estimate (calibrated — single-size anchor)
// ---------------------------------------------------------------------------

/// Fixed per-call overhead (ms) of an SSIMULACRA2 GPU score, isolated from
/// the per-pixel work via the batch-amortization curve (see below).
///
/// Provenance: `crates/ssim2-gpu/benchmarks/bench_batch_2026-05-02.md`,
/// CUDA RTX 5070 + CUDA 13.2 on the 7950X host. A single 256×256 pair
/// (65 536 px) scores at **4.10 ms** sequentially (batch=1, `seq /img`),
/// while the batched throughput floors at **~1.68 ms/img** at N=16 (work
/// fully overlapped, per-pixel cost only). Reading the floor as the
/// per-pixel term and the difference as the fixed launch/upload/readback
/// overhead: `fixed = 4.10 − 1.68 = 2.42 ms`, `slope = 1.68 ms / 65 536 px
/// = 25.6 ns/px`.
///
/// **CALIBRATED FROM A SINGLE IMAGE SIZE (256×256)** — unlike cvvdp/zensim
/// this is not size-swept, so the per-pixel slope is anchored at one point
/// and assumed constant. SSIMULACRA2's 6-scale pyramid does sub-linear
/// per-pixel work at larger sizes, so this likely *over*-estimates at
/// medium/large (conservative for capacity planning). A multi-size sweep
/// would tighten the slope; flagged for a future GPU-box run.
const SSIM2_FIXED_MS: f64 = 2.42;
/// Per-pixel slope (ns/px) — see [`SSIM2_FIXED_MS`] for derivation.
const SSIM2_NS_PER_PX: f64 = 25.6;

/// Estimated single-GPU score wall time (ms) for a `width × height`
/// SSIMULACRA2 pair. Pure size-math — no GPU, no allocation; safe on a
/// GPU-less host. `time_ms = SSIM2_FIXED_MS + SSIM2_NS_PER_PX · pixels /
/// 1e6` (a fixed-overhead + per-pixel model; see [`SSIM2_FIXED_MS`] for the
/// single-size calibration and its caveats). Returns `0.0` for a degenerate
/// image.
#[must_use]
pub fn estimate_score_time_ms(width: u32, height: u32) -> f32 {
    let pixels = (width as u64).saturating_mul(height as u64) as f64;
    if pixels == 0.0 {
        return 0.0;
    }
    (SSIM2_FIXED_MS + SSIM2_NS_PER_PX * pixels / 1.0e6) as f32
}

/// A metric's predicted per-pair resource use: GPU working-set bytes + score
/// wall time. The fleet planner sums `time_ms` across a cell's metrics and
/// takes the `max` of `vram_bytes` (GPU scoring serializes on the device).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreResourceEstimate {
    /// Peak GPU working-set in bytes for this metric at this size.
    pub vram_bytes: usize,
    /// Estimated single-GPU score wall time in milliseconds.
    pub time_ms: f32,
}

/// Bundle the VRAM estimate ([`estimate_gpu_memory_bytes`]) and score-time
/// estimate ([`estimate_score_time_ms`]) into one [`ScoreResourceEstimate`].
/// Pure math; no GPU required.
#[must_use]
pub fn estimate_score_resources(width: u32, height: u32) -> ScoreResourceEstimate {
    ScoreResourceEstimate {
        vram_bytes: estimate_gpu_memory_bytes(width, height),
        time_ms: estimate_score_time_ms(width, height),
    }
}

#[cfg(test)]
mod score_time_tests {
    use super::*;

    #[test]
    fn time_is_positive_and_scales_with_pixels() {
        assert!(estimate_score_time_ms(64, 64) > 0.0);
        assert!(estimate_score_time_ms(512, 512) > estimate_score_time_ms(256, 256));
        assert!(estimate_score_time_ms(4096, 4096) > estimate_score_time_ms(1024, 1024));
        assert_eq!(estimate_score_time_ms(0, 0), 0.0);
    }

    #[test]
    fn matches_calibration_anchor_256() {
        // The 256×256 anchor: fixed + slope·px = 2.42 + 25.6e-9·65536·1e3
        // = 2.42 + 1.678 ≈ 4.10 ms (the measured seq /img at batch=1).
        let got = estimate_score_time_ms(256, 256) as f64;
        assert!((got - 4.10).abs() < 0.05, "256x256 = {got} ms vs 4.10");
    }

    #[test]
    fn fixed_overhead_dominates_at_tiny() {
        // At 64×64 (4096 px) the per-pixel term is ~0.1 ms — small next to
        // the 2.42 ms fixed cost. Assert time is just above fixed and the
        // per-pixel contribution is < 10% of the total.
        let got = estimate_score_time_ms(64, 64) as f64;
        let per_pixel_term = got - SSIM2_FIXED_MS;
        assert!(got > SSIM2_FIXED_MS, "64x64 = {got} ms must exceed fixed");
        assert!(
            per_pixel_term < 0.2 && per_pixel_term / got < 0.10,
            "64x64 per-pixel term {per_pixel_term} ms (total {got}) should be tiny vs fixed"
        );
    }

    #[test]
    fn resources_bundle_matches_parts() {
        let r = estimate_score_resources(1024, 1024);
        assert_eq!(r.time_ms, estimate_score_time_ms(1024, 1024));
        assert_eq!(r.vram_bytes, estimate_gpu_memory_bytes(1024, 1024));
    }
}
