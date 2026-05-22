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
    /// Strip-mode resolution carrying the body height that fits the
    /// cap. `h_body + 2 * STRIP_HALO_ROWS` is the per-strip allocation
    /// height; the constructor uses `min(h_body, image_h)` so for tiny
    /// images it degenerates to a single-strip computation.
    Strip { h_body: u32 },
}

/// Auto policy. Picks Full when it fits the cap; otherwise tries Strip
/// with the default body height (1024 rows). If even Strip exceeds the
/// cap, surfaces [`crate::Error::TooBigForFull`] with the Full estimate
/// — Tile isn't implemented in ssim2-gpu so there's no smaller option.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    // Try Strip with the default body height. If even that's too
    // large, we can't auto-shrink further. (Halving body height once
    // more drops working set ~2× at scale 0, but at < ~512 rows the
    // scale-5 strip becomes too small for safe IIR boundary handling.)
    let h_body = STRIP_H_BODY_DEFAULT.min(height.max(8));
    if let Some(strip_bytes) = estimate_strip_gpu_memory_bytes(width, h_body) {
        if strip_bytes <= cap {
            return Ok(ResolvedMode::Strip { h_body });
        }
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
