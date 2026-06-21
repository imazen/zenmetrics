//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! cvvdp-gpu supports four memory modes:
//!
//! - **Full** — whole-image working set on device. Bit-stable with
//!   the host-scalar reference. Default; preferred when the image
//!   fits the VRAM cap.
//! - **Strip { h_body }** — **Mode E only** (ref-full + dist-strip
//!   cached-ref). Task #79 reintroduces a Strip variant that is
//!   **JOD-preserving**: the reference-side state stays at full
//!   image resolution on device (so per-band masking has the
//!   correct neighbour pixels at every level); the dist side walks
//!   the image in vertical strips. Per-band atomic-pool sums are
//!   associative across strips, so the final JOD equals Full-mode
//!   JOD within the documented Atomic<f32> reduction-order noise
//!   band.
//! - **StripPair { h_body }** — **Mode B** (one-shot pair stripwise).
//!   Both ref AND dist sides walk through strips together, no ref
//!   cache. Peak memory ≈ 2 × per-strip working set (REF gauss/weber
//!   built fresh per strip alongside DIST). Better than Strip for
//!   one-shot CLI callers (no cached-ref overhead) and worse for
//!   batch workloads (REF pyramid recomputed every dist).
//! - **CappedPyramid { levels }** — Option B safety net. Reduces
//!   the natural pyramid depth to `levels` so the deepest band's
//!   σ=3 PU blur halo shrinks and per-level d_scratch / pyramid /
//!   weber buffers stop allocating for the truncated levels.
//!   **NOT JOD-bit-identical** to Full (capping pyramid depth
//!   changes JOD at any level shorter than the natural depth).
//!   Opt-in only — never picked by `Auto`. Use when memory pressure
//!   forces a metric-value tradeoff (e.g. cvvdp on 6 GB VRAM at
//!   >16 MP). See the [`Self::CappedPyramid`] variant docstring for
//!   > historical bench (≤0.005 JOD at k=8, archived) and current
//!   > memory savings (estimator-based; runtime nvsmi not pinned).
//!
//! The earlier capped-pyramid Strip variant that lived here before
//! task #77 was rolled back because **capping the pyramid depth
//! changes the JOD value** at any k < 9. See `docs/STRIP_PROCESSING.md`
//! for the full rationale on what was rolled back vs what mode E
//! preserves.
//!
//! Strip mode is **only valid for the cached-ref code path**
//! ([`crate::pipeline::Cvvdp::warm_reference`] +
//! [`crate::pipeline::Cvvdp::compute_dkl_jod_with_warm_ref`] and the
//! umbrella `MetricCache` surface). One-shot scoring
//! ([`crate::pipeline::Cvvdp::score`]) is still Full-only because
//! its memory profile is the dist working set that mode E aims to
//! shrink anyway.
//!
//! `MemoryMode::Auto` picks Full when it fits the cap, else Strip
//! with a crate-default `h_body`. Callers can override `h_body`
//! explicitly via `MemoryMode::Strip { h_body: Some(N) }`.

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
    let bytes: u64 = mb.saturating_mul(1024 * 1024);
    let bytes_headroom: u64 = bytes.saturating_sub(bytes / 10);
    Some(usize::try_from(bytes_headroom).unwrap_or(usize::MAX))
}

/// Effective VRAM cap in bytes (task #51 cap policy):
/// 1. `ZENMETRICS_VRAM_CAP_BYTES` env var if set
/// 2. Live `nvidia-smi` probe (cached, 10% headroom)
/// 3. 8 GB default
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    if let Some(probed) = live_vram_probe_bytes() {
        return probed;
    }
    // 8 GiB as u64 to avoid usize overflow on 32-bit targets; saturate
    // to usize::MAX (which the resolver treats as "too big") on such hosts.
    usize::try_from(8u64 << 30).unwrap_or(usize::MAX)
}

/// Reclaim pooled-but-unreferenced device memory back to the driver
/// for `backend`.
///
/// cubecl pools GPU buffers: dropping a [`crate::CvvdpOpaque`] (or any
/// metric) returns its `Handle`s to the pool's free list, but the
/// underlying device pages stay resident for reuse — so a user who
/// drops a metric does **not** immediately get VRAM back, and an
/// orchestrator that swaps between metrics sees peak trend toward the
/// SUM of their working sets instead of the MAX. This function issues
/// cubecl's `ComputeClient::memory_cleanup` hint (which deallocates
/// fully-free pool pages) followed by a `sync` (which flushes the
/// CUDA deferred-free queue so `cuMemFree*` actually runs), returning
/// the freed pages to the driver.
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

/// Crate-default Strip body height. Picked at the small side so the
/// dist-side strip working set stays modest even on tiny VRAM caps.
/// The Auto resolver can pick a larger value when the cap permits.
///
/// Must be a multiple of `2^(MAX_LEVELS - 1) = 256` so the per-level
/// halving in the Weber pyramid doesn't drift through the strip
/// boundary. 512 = 2 * 256 — small enough to fit in 1 GB at 12 MP
/// (`512 × 4096 × 9 levels × N_CHANNELS × 4 bytes ≈ 224 MB` for the
/// dist-pyramid working set, plus halo).
pub const STRIP_H_BODY_DEFAULT: u32 = 512;

/// Strip body height alignment factor (legacy). Multiples of
/// `2^(MAX_LEVELS - 1)` would let `h_body` halve cleanly through
/// every Weber pyramid level in the deepest possible image; in
/// practice the strip walker only needs `h_body` to be a positive
/// power of two (which is also the alignment the underlying
/// kernels assume). The constructors validate `h_body` against the
/// power-of-two rule directly (see
/// [`crate::pipeline::Cvvdp::new_strip_pair`]); this constant is
/// retained for the [`STRIP_H_BODY_DEFAULT`] derivation and for any
/// downstream callers that want a safe-for-MAX_LEVELS alignment.
pub const STRIP_ALIGN: u32 = 1 << (crate::MAX_LEVELS as u32 - 1);

/// How the GPU pipeline should partition its working set.
///
/// cvvdp-gpu supports three variants:
///
/// - [`Self::Full`] — whole-image working set. Default.
/// - [`Self::Strip`] — Mode E (ref-full + dist-strip cached-ref).
///   Only valid for the cached-ref code path
///   ([`crate::pipeline::Cvvdp::warm_reference`] +
///   [`crate::pipeline::Cvvdp::compute_dkl_jod_with_warm_ref`]).
/// - [`Self::CappedPyramid`] — JOD-shifting capped-pyramid safety
///   net. Opt-in only; not picked by [`Self::Auto`].
///
/// See module-level docs for the JOD-preservation rationale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full when it fits the cap, else Strip with a crate-
    /// default `h_body`. See [`vram_cap_bytes()`] for the cap source.
    Auto,
    /// Allocate the whole-image working set.
    Full,
    /// Mode E strip walker: full ref state on device + per-strip dist
    /// working set. `h_body` is the dist-side strip body height in
    /// rows. `None` lets the [`Self::Auto`] policy pick a default
    /// (= [`STRIP_H_BODY_DEFAULT`]).
    ///
    /// `h_body` must be a positive power of two so the per-level
    /// halving in the strip walker halves cleanly.
    Strip {
        /// Dist-side strip body height in rows. `None` → crate-default.
        h_body: Option<u32>,
    },
    /// Mode B one-shot pair strip walker: both ref and dist sides walk
    /// in strips together (no full ref cache). Peak memory ≈ 2 × per-
    /// strip working set. Best for one-shot CLI callers; worse than
    /// `Strip` for batch workloads (REF pyramid recomputed per dist).
    ///
    /// `h_body` must be a positive power of two so the per-level
    /// halving in the strip walker halves cleanly.
    StripPair {
        /// Strip body height in rows for both ref and dist. `None` →
        /// crate-default ([`STRIP_H_BODY_DEFAULT`]).
        h_body: Option<u32>,
    },
    /// JOD-shifting capped-pyramid mode. Reduces natural pyramid depth
    /// to `levels` to shrink the σ=3 PU blur halo at the deepest band
    /// and skip allocating per-level d_scratch / pyramid / weber
    /// buffers for the truncated levels.
    ///
    /// **NOT JOD-bit-identical to Full** — opt-in only. [`Self::Auto`]
    /// does not pick this variant. Use when memory pressure forces a
    /// metric-value tradeoff (e.g. cvvdp on 6 GB VRAM at >16 MP).
    ///
    /// **Historical JOD bench (pre-task-#77 rollback, no longer
    /// runnable in-tree)**: ≤0.005 JOD parity gate at `k=8` vs Full's
    /// natural depth of 9. The sweep data file
    /// (`archived/cvvdp_capped_levels_2026-05-22.csv`) was removed
    /// alongside the capped-levels Strip variant; treat the 0.005
    /// figure as historical methodology, not a current contract.
    ///
    /// **Memory savings**: estimator-based — for natural depth 9
    /// capped to 5 at 4096², `estimate_gpu_memory_bytes_capped`
    /// returns substantially less than Full (`tests/capped_pyramid_smoke.rs:60`
    /// asserts `capped5 + 1024 < full` as a conservative gate). The
    /// exact ratio depends on the per-level pixel-count contribution;
    /// no recent runtime nvsmi number is committed.
    ///
    /// `levels` must be `>= 1` and is clamped from above by the
    /// natural pyramid depth (`pipeline::pyramid_levels`) at
    /// construction time.
    CappedPyramid {
        /// Maximum pyramid depth. Clamped by the natural depth.
        levels: u32,
    },
}

/// Outcome of resolving [`MemoryMode::Auto`]. cvvdp-gpu can resolve
/// to either Full or Strip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMode {
    /// Whole-image allocation.
    Full,
    /// Mode E with the picked `h_body`.
    Strip {
        /// Resolved dist-side strip body height in rows.
        h_body: u32,
    },
}

/// Auto-resolve policy: prefer Full when it fits the cap, else Strip
/// with a crate-default `h_body`. See module-level docs.
pub fn resolve_auto(width: u32, height: u32, cap: usize) -> crate::Result<ResolvedMode> {
    let Some(full_bytes) = crate::pipeline::estimate_gpu_memory_bytes(width, height) else {
        // Below the pyramid minimum — `Cvvdp::new` would reject too;
        // surface a TooBigForFull-shaped error with `needed: 0` to
        // signal "image too small to allocate at all".
        return Err(crate::Error::TooBigForFull { needed: 0, cap });
    };
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Try Strip with the crate-default h_body. The strip-mode
    // estimator returns the ref-full footprint + one-strip dist
    // working set; if even that doesn't fit, surface TooBigForFull.
    let strip_bytes =
        crate::pipeline::estimate_gpu_memory_bytes_strip(width, height, STRIP_H_BODY_DEFAULT)
            .unwrap_or(usize::MAX);
    if strip_bytes <= cap {
        return Ok(ResolvedMode::Strip {
            h_body: STRIP_H_BODY_DEFAULT,
        });
    }
    Err(crate::Error::TooBigForFull {
        needed: strip_bytes,
        cap,
    })
}

/// Unified-API wrapper around
/// [`crate::pipeline::estimate_gpu_memory_bytes`]. Returns a `usize`
/// (matching the other metric crates' signature); below-pyramid-minimum
/// images surface `usize::MAX` so the resolver classifies them as
/// "too big for the cap". cvvdp's own pipeline-level helper still
/// returns `Option<usize>` for callers that prefer the explicit
/// invalid-size encoding.
#[must_use]
pub fn estimate_gpu_memory_bytes_usize(width: u32, height: u32) -> usize {
    crate::pipeline::estimate_gpu_memory_bytes(width, height).unwrap_or(usize::MAX)
}

/// Unified-API wrapper that selects between Full / Strip estimates
/// based on the supplied [`MemoryMode`]. For [`MemoryMode::Auto`] this
/// returns the Full estimate (mirroring the umbrella `Auto`-prefers-
/// Full behavior at small sizes); the resolver consults
/// [`resolve_auto`] for the actual pick.
#[must_use]
pub fn estimate_gpu_memory_bytes_for_mode(width: u32, height: u32, mode: MemoryMode) -> usize {
    match mode {
        MemoryMode::Full | MemoryMode::Auto => {
            crate::pipeline::estimate_gpu_memory_bytes(width, height).unwrap_or(usize::MAX)
        }
        MemoryMode::Strip { h_body } => {
            let body = h_body.unwrap_or(STRIP_H_BODY_DEFAULT);
            crate::pipeline::estimate_gpu_memory_bytes_strip(width, height, body)
                .unwrap_or(usize::MAX)
        }
        MemoryMode::StripPair { h_body } => {
            let body = h_body.unwrap_or(STRIP_H_BODY_DEFAULT);
            crate::pipeline::estimate_gpu_memory_bytes_strip_pair(width, height, body)
                .unwrap_or(usize::MAX)
        }
        MemoryMode::CappedPyramid { levels } => {
            crate::pipeline::estimate_gpu_memory_bytes_capped(width, height, levels)
                .unwrap_or(usize::MAX)
        }
    }
}

// ---------------------------------------------------------------------------
// Score-time estimate (calibrated)
// ---------------------------------------------------------------------------

/// CVVDP JOD per-pixel cost (ns/px) anchored at the four size buckets of
/// the committed size-sweep. The curve is **non-monotone (U-shaped)**:
/// per-call dispatch + cubecl-init overhead dominates at tiny/small sizes,
/// and DRAM bandwidth saturation lifts it again at large sizes; the minimum
/// sits near 1 MP.
///
/// Provenance: `crates/cvvdp-gpu/benchmarks/time_size_sweep_tick164_2026-05-14.md`
/// (commit `8a6de7be` on `feat/cvvdp-gpu-scaffold`, RTX-class CUDA on the
/// 7950X host, 5-iter median, `compute_dkl_jod` full pipeline). The four
/// `(pixels, ns_per_px)` anchors are the `jod` column of that table:
///
/// | bucket | dims        | pixels     | jod ns/px |
/// | ----   | ----        | ----       | ----      |
/// | tiny   |   64 ×   64 |      4 096 |    526.81 |
/// | small  |  256 ×  256 |     65 536 |     90.81 |
/// | medium | 1024 × 1024 |  1 048 576 |     28.48 |
/// | large  | 4000 × 3000 | 12 000 000 |     38.83 |
const CVVDP_NS_PER_PX_ANCHORS: &[(f64, f64)] = &[
    (4_096.0, 526.81),
    (65_536.0, 90.81),
    (1_048_576.0, 28.48),
    (12_000_000.0, 38.83),
];

/// Interpolate a per-pixel cost (ns/px) from a `(pixels, ns_per_px)` anchor
/// table, **piecewise-linear in `log2(pixels)`** (the natural axis for a
/// fixed-overhead-amortizing curve spanning 4 K → 12 M pixels). Outside the
/// anchored range the endpoint value is held flat — we never extrapolate the
/// slope past measured data. `anchors` must be sorted ascending by pixel
/// count and non-empty.
#[must_use]
pub(crate) fn interp_ns_per_px_log(anchors: &[(f64, f64)], pixels: f64) -> f64 {
    debug_assert!(!anchors.is_empty());
    let px = pixels.max(1.0);
    let lx = px.log2();
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

/// Estimated single-GPU score wall time (ms) for a `width × height` CVVDP JOD
/// pair. Pure size-math — no GPU, no allocation; safe to call on a GPU-less
/// host (fleet planning, CI). `time_ms = ns_per_px(pixels) · pixels / 1e6`,
/// with `ns_per_px` interpolated from [`CVVDP_NS_PER_PX_ANCHORS`].
///
/// This is the Full-mode steady-state estimate (the measured pipeline is
/// `compute_dkl_jod`). Strip modes trade VRAM for some extra per-strip
/// launch overhead; this estimate does not model that delta and is treated
/// as a lower bound for strip scoring.
///
/// Returns `0.0` for a degenerate (zero-area) image.
#[must_use]
pub fn estimate_score_time_ms(width: u32, height: u32) -> f32 {
    let pixels = (width as u64).saturating_mul(height as u64) as f64;
    if pixels == 0.0 {
        return 0.0;
    }
    let ns_per_px = interp_ns_per_px_log(CVVDP_NS_PER_PX_ANCHORS, pixels);
    ((ns_per_px * pixels) / 1.0e6) as f32
}

/// A metric's predicted per-pair resource use: GPU working-set bytes + score
/// wall time. The fleet planner sums `time_ms` across a cell's metrics and
/// takes the `max` of `vram_bytes` (GPU scoring is serialized on the device).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoreResourceEstimate {
    /// Peak GPU working-set in bytes for this metric at this size (Full mode).
    pub vram_bytes: usize,
    /// Estimated single-GPU score wall time in milliseconds.
    pub time_ms: f32,
}

/// Bundle the VRAM estimate ([`estimate_gpu_memory_bytes_usize`]) and the
/// score-time estimate ([`estimate_score_time_ms`]) into one
/// [`ScoreResourceEstimate`] — the clean per-metric call for the fleet
/// planner. Pure math; no GPU required.
#[must_use]
pub fn estimate_score_resources(width: u32, height: u32) -> ScoreResourceEstimate {
    ScoreResourceEstimate {
        vram_bytes: estimate_gpu_memory_bytes_usize(width, height),
        time_ms: estimate_score_time_ms(width, height),
    }
}

#[cfg(test)]
mod score_time_tests {
    use super::*;

    /// ns/px = time_ms × 1e6 / pixels — recover the per-pixel cost so we can
    /// compare against the calibration anchors directly.
    fn ns_per_px(width: u32, height: u32) -> f64 {
        let px = (width as u64 * height as u64) as f64;
        (estimate_score_time_ms(width, height) as f64) * 1.0e6 / px
    }

    #[test]
    fn time_is_positive_and_scales_with_pixels() {
        assert!(estimate_score_time_ms(64, 64) > 0.0);
        // More pixels → more total time (despite the U-shaped per-pixel cost,
        // total wall is monotone in pixels across these buckets).
        assert!(estimate_score_time_ms(256, 256) > estimate_score_time_ms(64, 64));
        assert!(estimate_score_time_ms(1024, 1024) > estimate_score_time_ms(256, 256));
        assert!(estimate_score_time_ms(4000, 3000) > estimate_score_time_ms(1024, 1024));
        // Degenerate image → zero.
        assert_eq!(estimate_score_time_ms(0, 0), 0.0);
        assert_eq!(estimate_score_time_ms(100, 0), 0.0);
    }

    #[test]
    fn matches_calibration_anchor_points() {
        // At each measured bucket the recovered ns/px must equal the anchor
        // (interpolation is exact at the knots). 1% tolerance for the
        // f64→f32 round-trip through time_ms.
        for &(w, h, expect) in &[
            (64u32, 64u32, 526.81f64),
            (256, 256, 90.81),
            (1024, 1024, 28.48),
            (4000, 3000, 38.83),
        ] {
            let got = ns_per_px(w, h);
            let rel = (got - expect).abs() / expect;
            assert!(rel < 0.01, "{w}x{h}: ns/px {got} vs anchor {expect}");
        }
    }

    #[test]
    fn u_shape_minimum_is_near_medium() {
        // The per-pixel cost dips to its minimum near 1 MP then rises again —
        // medium must be cheaper per pixel than both small and large.
        assert!(ns_per_px(1024, 1024) < ns_per_px(256, 256));
        assert!(ns_per_px(1024, 1024) < ns_per_px(4000, 3000));
    }

    #[test]
    fn flat_extrapolation_beyond_anchor_endpoints() {
        // Below the tiny anchor (4096 px): held at the tiny ns/px.
        assert!((ns_per_px(32, 32) - 526.81).abs() / 526.81 < 0.01);
        // Above the large anchor (12 MP): held at the large ns/px.
        assert!((ns_per_px(8000, 6000) - 38.83).abs() / 38.83 < 0.01);
    }

    #[test]
    fn resources_bundle_matches_parts() {
        let r = estimate_score_resources(1024, 1024);
        assert_eq!(r.time_ms, estimate_score_time_ms(1024, 1024));
        assert_eq!(r.vram_bytes, estimate_gpu_memory_bytes_usize(1024, 1024));
        assert!(r.vram_bytes > 0);
    }
}
