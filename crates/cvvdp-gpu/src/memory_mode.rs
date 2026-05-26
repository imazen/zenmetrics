//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! cvvdp-gpu's strip-mode plan is documented in
//! `docs/STRIP_PROCESSING.md`. Two paths are tracked there:
//!
//! 1. **Capped pyramid depth** — `MemoryMode::Strip { capped_levels:
//!    Some(k) }`. Reduces the natural pyramid depth from the
//!    `band_frequencies` cutoff (typically 9 at 4K-class viewing) to
//!    `k`, shrinking the σ=3 PU-blur halo. Constructor delegates to
//!    [`crate::Cvvdp::new_with_geometry_and_cap`]. Fidelity-vs-memory
//!    tradeoff: changes JOD score outside the canonical ≤ 0.005 JOD
//!    pycvvdp v0.5.4 parity gate for some fixtures at small caps.
//!    Sweep data at `benchmarks/cvvdp_capped_levels_2026-05-22.csv`
//!    shows `k = 8` keeps every measured fixture under the gate;
//!    `k <= 7` fails on the 720×1280 offset fixture.
//! 2. **Panorama strip** — tall/wide images via two-pass walker. Not
//!    yet implemented; the panorama case currently falls through to
//!    `Full` via `Auto` resolution. See `docs/STRIP_PROCESSING.md`.
//!
//! `MemoryMode::Tile` returns [`crate::Error::ModeUnsupported`].
//! `MemoryMode::Auto` resolves to `Full` whenever Full fits the cap;
//! it does NOT autoselect capped-Strip — capping changes the
//! metric value, so the caller must opt in explicitly.
//!
//! This module wraps the pre-existing
//! [`crate::pipeline::estimate_gpu_memory_bytes`] (which already
//! computes the working-set bytes) and exposes the unified
//! [`MemoryMode`] enum.

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Cache for the live nvidia-smi probe result. Process-wide.
static LIVE_PROBE_CACHE: std::sync::OnceLock<Option<usize>> =
    std::sync::OnceLock::new();

/// Probe live free-VRAM. See `iwssim_gpu::memory_mode::live_vram_probe_bytes`.
pub fn live_vram_probe_bytes() -> Option<usize> {
    *LIVE_PROBE_CACHE.get_or_init(query_nvidia_smi_memory_free)
}

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
    let bytes = (mb as usize).saturating_mul(1024 * 1024);
    Some(bytes.saturating_sub(bytes / 10))
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
    8 * 1024 * 1024 * 1024
}

/// How the GPU pipeline should partition its working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full or Strip automatically based on
    /// [`vram_cap_bytes()`]. cvvdp-gpu always picks Full when it
    /// fits the cap. **Does not** autoselect capped-Strip — capping
    /// changes the JOD value (see `docs/STRIP_PROCESSING.md`), so
    /// callers must opt in via `Strip { capped_levels: Some(_) }`
    /// explicitly.
    Auto,
    /// Allocate the whole-image working set.
    Full,
    /// Strip mode.
    ///
    /// **Single-pass strip (`h_body = Some(_)` or `None` and
    /// `capped_levels = None`)**: currently returns
    /// [`crate::Error::ModeUnsupported`] — see
    /// `docs/STRIP_PROCESSING.md` for the planned panorama-strip
    /// design.
    ///
    /// **Capped-depth Full (`capped_levels = Some(k)`)**: not a true
    /// strip walker — instead, builds a Full pipeline with the
    /// pyramid depth clamped to `k`. The σ=3 phase-uncertainty blur
    /// halo at non-baseband bands shrinks proportionally to
    /// `6 × 2^(k-2)` rows, making 24 MP square viable on smaller
    /// VRAM budgets. **Capping changes the JOD value** — see
    /// `docs/STRIP_PROCESSING.md` for the cap-vs-JOD-drift sweep
    /// data and which caps fit the canonical ≤ 0.005 JOD pycvvdp
    /// parity gate per fixture.
    Strip {
        /// Optional body row count override. **Ignored for now** —
        /// the single-pass-strip path is not implemented yet.
        h_body: Option<u32>,
        /// Optional pyramid-depth cap. `Some(k)` clamps the pyramid
        /// to `min(k, natural_n_levels)` bands; `None` defers to the
        /// natural depth from `band_frequencies`.
        capped_levels: Option<u32>,
    },
    /// 2-D tile mode. **Unsupported** — reserved for a future
    /// implementation.
    Tile {
        /// Tile height in rows.
        h: u32,
        /// Tile width in pixels.
        w: u32,
    },
}

/// Outcome of resolving [`MemoryMode::Auto`]. cvvdp-gpu can only
/// resolve to Full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedMode {
    /// Whole-image allocation.
    Full,
}

/// Auto-resolve policy. See module-level docs.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let Some(full_bytes) = crate::pipeline::estimate_gpu_memory_bytes(width, height) else {
        // Below the pyramid minimum — `Cvvdp::new` would reject too;
        // surface a TooBigForFull-shaped error with `needed: 0` to
        // signal "image too small to allocate at all".
        return Err(crate::Error::TooBigForFull { needed: 0, cap });
    };
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
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

/// Strip-mode estimator. Always returns `None` because cvvdp-gpu has
/// no Strip path (the spatial-frequency decomposition is full-image
/// by construction).
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(_width: u32, _h_body: u32) -> Option<usize> {
    None
}
