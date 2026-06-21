//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! zensim-gpu supports [`MemoryMode::Strip`] as of 2026-05-26: the
//! strip walker re-uses the per-scale fused / persist / masked-IW
//! kernels with strip-sized buffers, walks the image in
//! `h_body + 2 × halo` strips, and accumulates per-feature raw sums
//! on the host across strips (gated to body rows only via the
//! `y_body_start / y_body_end` kernel parameters). See
//! `docs/STRIP_PROCESSING.md` for the design + measured numbers.
//!
//! ## Regime-aware GPU memory estimator
//!
//! [`estimate_gpu_memory_bytes`] takes the active
//! [`ZensimFeatureRegime`](crate::ZensimFeatureRegime) and returns
//! the *metric's own* allocation (NOT including cubecl's runtime
//! pool initialisation overhead). [`CUBECL_OVERHEAD_BYTES`] documents
//! the fixed-per-process cubecl pool floor; [`resolve_auto`] reserves
//! it from the caller's cap before comparing the metric's estimate.
//!
//! Calibration data: `benchmarks/mem_per_metric_2026-05-26.csv`
//! (24 zensim rows, 8 sizes × 3 regimes). Each row's
//! `peak_delta_gpu_mb` was sampled via nvidia-smi between a baseline
//! process start (cubecl init done) and the metric's warm-iter peak.
//! See [`estimate_gpu_memory_bytes`] for the per-regime fit. Max
//! validation residual: 10.2 % (basic), 20.3 % (extended), 18.9 %
//! (withiw) — all within the ±25 % budget asserted by the unit test
//! `memory_mode::estimator_matches_measured`.

use crate::{SCALES, ZensimFeatureRegime};

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
    Strip { h_body: u32 },
}

/// Fixed-per-process cubecl runtime pool init overhead measured on
/// the calibration host (Ryzen 9 7950X + RTX 5070, CUDA 13.2).
///
/// This is *not* the metric's own allocation — it's the cubecl
/// arena+staging buffer that the runtime pre-warms on the first
/// kernel launch and keeps around for subsequent launches. The
/// smallest measured `peak_delta_gpu_mb` across 24 zensim rows was
/// 193 MB at 64×64 (where the metric's own buffers fit *inside* the
/// pool), so we attribute that 193 MB to the runtime, not the metric.
///
/// [`resolve_auto`] reserves this from the caller's cap before
/// comparing the metric's estimate; [`estimate_gpu_memory_bytes`]
/// does NOT add it (it reports the metric's contribution above the
/// runtime floor).
///
/// Source: `benchmarks/mem_per_metric_2026-05-26.csv`, basic regime
/// at 64×64 (`peak_delta_gpu_mb = 193.0`).
pub const CUBECL_OVERHEAD_BYTES: usize = 193 * 1024 * 1024;

/// Auto policy: prefer Full when it fits the cap, fall back to Strip
/// otherwise. Mirrors the shape from
/// `iwssim_gpu::memory_mode::resolve_auto`.
///
/// The cap is interpreted as the *total* VRAM budget the caller has
/// available; this routine reserves [`CUBECL_OVERHEAD_BYTES`] for the
/// cubecl runtime before deciding whether the metric's own estimate
/// fits.
pub fn resolve_auto(
    width: u32,
    height: u32,
    regime: ZensimFeatureRegime,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let metric_bytes = estimate_gpu_memory_bytes(width, height, regime);
    let needed_total = metric_bytes.saturating_add(CUBECL_OVERHEAD_BYTES);
    if needed_total <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Full exceeds the cap — try Strip before giving up. zensim-gpu's
    // strip walker handles any image height ≥ STRIP_ALIGN, so as long
    // as a strip-sized footprint fits we can score the image.
    let cap_for_strip = cap.saturating_sub(CUBECL_OVERHEAD_BYTES);
    if let Some(h_body) = auto_size_strip_body(width, height, regime, cap_for_strip) {
        return Ok(ResolvedMode::Strip { h_body });
    }
    Err(crate::Error::TooBigForFull {
        needed: needed_total,
        cap,
    })
}

/// Helper for callers needing the resolved h_body without going
/// through [`resolve_auto`] — e.g. `Zensim::new_with_memory_mode`'s
/// `MemoryMode::Strip { h_body: None }` path.
#[must_use]
pub fn auto_strip_body_for(
    width: u32,
    height: u32,
    regime: ZensimFeatureRegime,
    cap: usize,
) -> u32 {
    let cap_for_strip = cap.saturating_sub(CUBECL_OVERHEAD_BYTES);
    auto_size_strip_body(width, height, regime, cap_for_strip)
        .unwrap_or_else(|| {
            let default = crate::pipeline::STRIP_DEFAULT_BODY;
            round_align(height.min(default))
        })
        .max(crate::pipeline::STRIP_ALIGN)
}

/// Returns the largest `h_body` that fits the cap, or None if even
/// the minimum body doesn't fit. Always a multiple of
/// [`crate::pipeline::STRIP_ALIGN`].
fn round_align(h: u32) -> u32 {
    let align = crate::pipeline::STRIP_ALIGN;
    h.div_ceil(align) * align
}

/// Pick the largest body height (multiple of [`crate::pipeline::STRIP_ALIGN`])
/// such that the strip estimator fits the cap. Returns None when even
/// a one-unit (`STRIP_ALIGN`-tall) body wouldn't fit.
fn auto_size_strip_body(
    width: u32,
    height: u32,
    regime: ZensimFeatureRegime,
    cap: usize,
) -> Option<u32> {
    let align = crate::pipeline::STRIP_ALIGN;
    let halo_bytes = estimate_strip_gpu_memory_bytes_for(width, 0, regime)?;
    if halo_bytes >= cap {
        return None;
    }
    let one_unit = estimate_strip_gpu_memory_bytes_for(width, align, regime)?;
    let per_unit = one_unit.saturating_sub(halo_bytes);
    if per_unit == 0 {
        let fb = crate::pipeline::STRIP_DEFAULT_BODY.min(round_align(height));
        return Some(fb.max(align));
    }
    let max_units = (cap - halo_bytes) / per_unit;
    let raw = (max_units as u32) * align;
    let body = raw.min(round_align(height));
    if body < align {
        return None;
    }
    Some(body)
}

/// Helper variant of [`estimate_strip_gpu_memory_bytes`] that takes
/// the regime explicitly. Returns the strip working-set estimate
/// (without [`CUBECL_OVERHEAD_BYTES`]).
fn estimate_strip_gpu_memory_bytes_for(
    width: u32,
    h_body: u32,
    regime: ZensimFeatureRegime,
) -> Option<usize> {
    let halo = crate::pipeline::STRIP_DEFAULT_HALO;
    let strip_h = h_body.saturating_add(2 * halo);
    Some(estimate_gpu_memory_bytes(width, strip_h, regime))
}

/// Sum of `w_s × h_s` across the 4 pyramid scales (s=0..3) using
/// `div_ceil` halving — matches the host-side scale walk in
/// [`crate::pipeline::Zensim::new_with_regime`]. Returns 0 once a
/// scale drops below the 8×8 floor (same termination criterion as
/// the runtime).
#[inline]
fn pyramid_pixels(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    for _ in 0..SCALES {
        if w < 8 || h < 8 {
            break;
        }
        total = total.saturating_add((w as usize).saturating_mul(h as usize));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    total
}

/// Regime-aware estimate of the metric's own peak GPU allocation in
/// bytes for `width × height` images. Does NOT include
/// [`CUBECL_OVERHEAD_BYTES`] — callers that want total VRAM pressure
/// (e.g. comparing against [`vram_cap_bytes`]) should sum the two
/// constants; [`resolve_auto`] does that internally.
///
/// ## Model
///
/// For each regime, the estimate is a linear fit through the pyramid
/// pixel sum `Σ_s w_s · h_s` (4 scales, `div_ceil(2)` halving):
///
/// ```text
/// estimate_bytes = BASE[regime] + BETA[regime] · pyramid_pixels(w, h)
/// ```
///
/// Coefficients fit via grid search on
/// `benchmarks/mem_per_metric_2026-05-26.csv` (8 sizes × 3 regimes,
/// 24 rows total). The fit minimises `max |total_pred − measured|`
/// where `total_pred = estimate + CUBECL_OVERHEAD_BYTES`:
///
/// | regime   | BASE_MB | BETA (B/pyr_pix) | planes-equiv | max abs %err |
/// |----------|---------|------------------|--------------|--------------|
/// | basic    |    0    | 41               | 10.25        | 10.2 %       |
/// | extended | 38      | 139              | 34.75        | 20.3 %       |
/// | withiw   | 71      | 136              | 34.0         | 18.9 %       |
///
/// All 24 calibration rows land within ±25 % — the budget asserted by
/// the `memory_mode::estimator_matches_measured` regression test.
/// The "planes-equiv" column reads the per-pyramid-pixel cost as
/// `BETA / 4` (one f32 plane per pyramid pixel). Extended/WithIw add
/// 4 persist planes per channel × 3 channels = 12 planes for the
/// masked-IW kernel, plus 2 staging planes for the σ²/σ12 atomics —
/// the empirical ~35 planes-equiv reflects both the persist planes
/// and cubecl's per-launch transient buffers.
///
/// ## Why no `floor()` clamp?
///
/// Tiny inputs (64×64, 256×256) measure 0 MB above the cubecl
/// overhead — the metric's allocation fits inside the pre-warmed
/// pool. The linear estimate stays in [0, few MB] for those sizes,
/// `resolve_auto` clamps to `CUBECL_OVERHEAD_BYTES` via the
/// `saturating_add` + cap check, and the unit test asserts the
/// combined `estimate + CUBECL_OVERHEAD` lands within ±25 % of the
/// measurement. Adding an explicit floor inside the estimator would
/// double-count the runtime overhead.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32, regime: ZensimFeatureRegime) -> usize {
    let pyramid = pyramid_pixels(width, height);
    let (base_mb, beta_b_per_pyr) = match regime {
        ZensimFeatureRegime::Basic => (0_usize, 41_usize),
        ZensimFeatureRegime::Extended => (38_usize, 139_usize),
        ZensimFeatureRegime::WithIw => (71_usize, 136_usize),
    };
    let base_bytes = base_mb.saturating_mul(1024 * 1024);
    let scale_bytes = pyramid.saturating_mul(beta_b_per_pyr);
    base_bytes.saturating_add(scale_bytes)
}

/// Strip-mode estimator. Returns the working-set bytes for one strip
/// of `h_body + 2 × halo` rows at `width` pixels (default halo =
/// [`crate::pipeline::STRIP_DEFAULT_HALO`] = 40). Uses the same
/// per-pyramid-pixel coefficients as [`estimate_gpu_memory_bytes`]
/// but evaluates the pyramid on the strip's allocation height instead
/// of the full image height.
///
/// Does NOT include [`CUBECL_OVERHEAD_BYTES`] — callers that want
/// total VRAM pressure should sum the two constants.
///
/// `regime` defaults to [`ZensimFeatureRegime::Basic`]. Use
/// [`estimate_strip_gpu_memory_bytes_with_regime`] when the caller
/// needs the Extended / WithIw budget.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    estimate_strip_gpu_memory_bytes_with_regime(width, h_body, ZensimFeatureRegime::Basic)
}

/// Regime-aware strip estimator. See [`estimate_strip_gpu_memory_bytes`].
#[must_use]
pub fn estimate_strip_gpu_memory_bytes_with_regime(
    width: u32,
    h_body: u32,
    regime: ZensimFeatureRegime,
) -> Option<usize> {
    estimate_strip_gpu_memory_bytes_for(width, h_body, regime)
}

// ---------------------------------------------------------------------------
// Score-time estimate (calibrated)
// ---------------------------------------------------------------------------

/// zensim per-pixel scalar-feature score cost (ns/px), anchored at four
/// measured sizes. Monotone-decreasing: per-call dispatch overhead amortizes
/// as the image grows. This is the **default scalar GPU feature-extraction**
/// path (`compute_features_vec`, no diffmap, Basic-regime kernels).
///
/// Provenance: `crates/zensim-gpu/benchmarks/zensim_diffmap_overhead_2026-05-27.tsv`,
/// `score_only_ms` column (the no-diffmap baseline), `gradient_identity`
/// fixture, CUDA RTX 5070 on the 7950X host, parent commit `f9c567a2`. The
/// four `(pixels, ns_per_px)` anchors:
///
/// | dims        | pixels    | score_only_ms | ns/px  |
/// | ----        | ----      | ----          | ----   |
/// |  256 ×  256 |    65 536 | 1.002         | 15.29  |
/// |  512 ×  512 |   262 144 | 1.610         |  6.14  |
/// | 1024 × 1024 | 1 048 576 | 3.122         |  2.98  |
/// | 2048 × 2048 | 4 194 304 | 10.293        |  2.45  |
///
/// The Extended / WithIw masked-IW regimes do more per-pixel work; this
/// estimate (Basic/default) is a lower bound for those. The diffmap path
/// (`score_with_diffmap`) is much heavier (Phase-1 CPU fallback, 3-23×) and
/// is **not** modeled here — the planner scores the scalar path.
const ZENSIM_NS_PER_PX_ANCHORS: &[(f64, f64)] = &[
    (65_536.0, 15.29),
    (262_144.0, 6.14),
    (1_048_576.0, 2.98),
    (4_194_304.0, 2.45),
];

/// Interpolate a per-pixel cost (ns/px) from a `(pixels, ns_per_px)` anchor
/// table, **piecewise-linear in `log2(pixels)`**. Outside the anchored range
/// the endpoint value is held flat (no slope extrapolation past measured
/// data). `anchors` must be sorted ascending by pixel count and non-empty.
#[must_use]
fn interp_ns_per_px_log(anchors: &[(f64, f64)], pixels: f64) -> f64 {
    debug_assert!(!anchors.is_empty());
    let lx = pixels.max(1.0).log2();
    let first = anchors[0];
    let last = anchors[anchors.len() - 1];
    if lx <= first.0.log2() {
        return first.1;
    }
    if lx >= last.0.log2() {
        return last.1;
    }
    for w in anchors.windows(2) {
        let (p0, v0) = w[0];
        let (p1, v1) = w[1];
        let (l0, l1) = (p0.log2(), p1.log2());
        if lx >= l0 && lx <= l1 {
            let t = if (l1 - l0).abs() < f64::EPSILON {
                0.0
            } else {
                (lx - l0) / (l1 - l0)
            };
            return v0 + t * (v1 - v0);
        }
    }
    last.1
}

/// Estimated single-GPU score wall time (ms) for a `width × height` zensim
/// scalar-feature score. Pure size-math — no GPU, no allocation; safe on a
/// GPU-less host (fleet planning, CI). `time_ms = ns_per_px(pixels) · pixels
/// / 1e6`, interpolated from [`ZENSIM_NS_PER_PX_ANCHORS`] (default scalar
/// path, Basic regime). Returns `0.0` for a degenerate image.
#[must_use]
pub fn estimate_score_time_ms(width: u32, height: u32) -> f32 {
    let pixels = (width as u64).saturating_mul(height as u64) as f64;
    if pixels == 0.0 {
        return 0.0;
    }
    let ns_per_px = interp_ns_per_px_log(ZENSIM_NS_PER_PX_ANCHORS, pixels);
    ((ns_per_px * pixels) / 1.0e6) as f32
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
/// estimate ([`estimate_score_time_ms`]) into one [`ScoreResourceEstimate`]
/// for the given feature regime. Pure math; no GPU required.
#[must_use]
pub fn estimate_score_resources(
    width: u32,
    height: u32,
    regime: ZensimFeatureRegime,
) -> ScoreResourceEstimate {
    ScoreResourceEstimate {
        vram_bytes: estimate_gpu_memory_bytes(width, height, regime),
        time_ms: estimate_score_time_ms(width, height),
    }
}

#[cfg(test)]
mod score_time_tests {
    use super::*;

    fn ns_per_px(width: u32, height: u32) -> f64 {
        let px = (width as u64 * height as u64) as f64;
        (estimate_score_time_ms(width, height) as f64) * 1.0e6 / px
    }

    #[test]
    fn time_is_positive_and_scales_with_pixels() {
        assert!(estimate_score_time_ms(256, 256) > 0.0);
        assert!(estimate_score_time_ms(512, 512) > estimate_score_time_ms(256, 256));
        assert!(estimate_score_time_ms(1024, 1024) > estimate_score_time_ms(512, 512));
        assert!(estimate_score_time_ms(2048, 2048) > estimate_score_time_ms(1024, 1024));
        assert_eq!(estimate_score_time_ms(0, 0), 0.0);
    }

    #[test]
    fn matches_calibration_anchor_points() {
        for &(w, h, expect) in &[
            (256u32, 256u32, 15.29f64),
            (512, 512, 6.14),
            (1024, 1024, 2.98),
            (2048, 2048, 2.45),
        ] {
            let got = ns_per_px(w, h);
            let rel = (got - expect).abs() / expect;
            assert!(rel < 0.01, "{w}x{h}: ns/px {got} vs anchor {expect}");
        }
    }

    #[test]
    fn per_pixel_cost_is_monotone_decreasing() {
        // Fixed-overhead amortization: bigger images cost less per pixel.
        assert!(ns_per_px(512, 512) < ns_per_px(256, 256));
        assert!(ns_per_px(1024, 1024) < ns_per_px(512, 512));
        assert!(ns_per_px(2048, 2048) < ns_per_px(1024, 1024));
    }

    #[test]
    fn resources_bundle_matches_parts() {
        let r = estimate_score_resources(1024, 1024, ZensimFeatureRegime::Basic);
        assert_eq!(r.time_ms, estimate_score_time_ms(1024, 1024));
        assert_eq!(
            r.vram_bytes,
            estimate_gpu_memory_bytes(1024, 1024, ZensimFeatureRegime::Basic)
        );
    }
}
