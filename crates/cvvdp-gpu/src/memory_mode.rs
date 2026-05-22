//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! cvvdp-gpu is **architecturally blocked** at 24 MP square per
//! `docs/STRIP_PROCESSING.md` — the spatial-frequency channel uses
//! a full-image FFT-like decomposition that doesn't decompose into
//! independently scorable strips. `MemoryMode::Strip` and `Tile`
//! return [`crate::Error::ModeUnsupported`].
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

/// Effective VRAM cap in bytes. Reads `ZENMETRICS_VRAM_CAP_BYTES` if
/// set, else returns the 8 GB default.
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    8 * 1024 * 1024 * 1024
}

/// How the GPU pipeline should partition its working set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryMode {
    /// Pick Full or Strip automatically based on
    /// [`vram_cap_bytes()`]. cvvdp-gpu always picks Full (no Strip
    /// implementation).
    Auto,
    /// Allocate the whole-image working set.
    Full,
    /// Strip mode. **Unsupported in cvvdp-gpu** — the constructor
    /// returns [`crate::Error::ModeUnsupported`].
    Strip {
        /// Optional body row count override.
        h_body: Option<u32>,
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
