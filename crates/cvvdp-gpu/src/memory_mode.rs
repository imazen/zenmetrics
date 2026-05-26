//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! cvvdp-gpu supports only **Full** image processing. There is no
//! partitioned working-set path — the 9-level Weber-contrast pyramid
//! + σ=3 PU-blur halo accumulation makes a true walker a major
//! redesign, and the capped-pyramid variant that previously lived
//! here was rolled back (task #77) because **capping the pyramid
//! depth changes the JOD value** — every k < 9 produces a different
//! metric output. See `docs/STRIP_PROCESSING.md` for the full
//! rationale.
//!
//! `MemoryMode::Auto` resolves to `Full` whenever Full fits the cap;
//! otherwise it surfaces [`crate::Error::TooBigForFull`] and lets the
//! caller decide whether to pick a different metric or split the
//! image at the application layer.
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
///
/// cvvdp-gpu only supports whole-image processing — partitioned
/// variants were removed in task #77 (the previous capped-pyramid
/// implementation changed the JOD value at any k < 9, which is
/// unacceptable for a metric crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full automatically based on [`vram_cap_bytes()`]. cvvdp-gpu
    /// always picks Full when it fits; otherwise surfaces
    /// [`crate::Error::TooBigForFull`].
    Auto,
    /// Allocate the whole-image working set.
    Full,
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
