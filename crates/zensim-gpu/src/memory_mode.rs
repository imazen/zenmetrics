//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! zensim-gpu does **NOT** implement Strip processing — the
//! 4-channel + 4-scale + Extended-regime allocator is contiguous and
//! interlocked enough that strip processing would need a dedicated
//! design pass. Until then `MemoryMode::Strip { .. }` returns
//! [`crate::Error::ModeUnsupported`].

use crate::SCALES;

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

/// Auto policy: Strip unimplemented, so Auto resolves to Full when
/// it fits and errors otherwise.
pub fn resolve_auto(width: u32, height: u32, cap: usize) -> crate::Result<ResolvedMode> {
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
/// [`crate::pipeline::Zensim::new`] allocates for `width × height`
/// images in the `Basic` feature regime. Extended / WithIw add
/// additional persist planes per scale (`4 planes × 3 channels ×
/// padded_pixels × 4 bytes` per scale) — see
/// [`crate::pipeline::Zensim::new_with_regime_budget`] for the cap
/// helper.
///
/// Per-scale: roughly 18 f32 planes (3 channels × ~6 planes each)
/// across the Basic kernel set. 4 scales total. Plus 2 packed-u32
/// sRGB staging buffers at scale 0.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 18;
    for _ in 0..SCALES {
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

/// Strip-mode estimator. Always returns `None` because zensim-gpu
/// has no Strip implementation.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(_width: u32, _h_body: u32) -> Option<usize> {
    None
}
