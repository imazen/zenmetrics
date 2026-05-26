//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! zensim-gpu does **NOT** implement Strip processing — the
//! 4-channel + 4-scale + Extended-regime allocator is contiguous and
//! interlocked enough that strip processing would need a dedicated
//! design pass. Until then `MemoryMode::Strip { .. }` returns
//! [`crate::Error::ModeUnsupported`].
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

/// Auto policy: Strip unimplemented, so Auto resolves to Full when
/// it fits and errors otherwise.
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
    Err(crate::Error::TooBigForFull {
        needed: needed_total,
        cap,
    })
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
pub fn estimate_gpu_memory_bytes(
    width: u32,
    height: u32,
    regime: ZensimFeatureRegime,
) -> usize {
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

/// Strip-mode estimator. Always returns `None` because zensim-gpu
/// has no Strip implementation.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(_width: u32, _h_body: u32) -> Option<usize> {
    None
}
