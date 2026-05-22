//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! ssim2-gpu does **NOT** implement Strip processing yet — Phase 2 of
//! the strip design (see `docs/STRIP_PROCESSING.md`) is the planned
//! follow-up. Until then `MemoryMode::Strip { .. }` returns
//! [`crate::Error::ModeUnsupported`], and Auto can only resolve to
//! Full. If the image's working set exceeds the cap, Auto surfaces
//! [`crate::Error::TooBigForFull`].

use crate::NUM_SCALES;

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
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

/// Auto policy: Strip isn't implemented, so the only feasible
/// resolution is Full. Errors with [`crate::Error::TooBigForFull`]
/// when Full exceeds the cap.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
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

/// Strip-mode estimator. Always returns `None` because Strip isn't
/// implemented in ssim2-gpu — the unified API requires the signature
/// for cross-crate parity, but there is nothing to estimate.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(_width: u32, _h_body: u32) -> Option<usize> {
    None
}
