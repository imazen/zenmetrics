//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for shared design rationale.
//!
//! iwssim-gpu is **NOT strip-preferred**: strip mode is ~1.7× slower
//! than whole-image on this crate (the cached-reference strip path
//! is deferred — see `docs/STRIP_PROCESSING.md`). Auto picks Full
//! whenever it fits the cap.

use crate::{MIN_NATIVE_DIM, NUM_SCALES};

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
    Strip { h_body: u32 },
}

const PYRAMID_ALIGN: u32 = 1 << (NUM_SCALES as u32 - 1); // 16
const STRIP_DEFAULT_HALO: u32 = 256;

pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    if width < MIN_NATIVE_DIM || height < MIN_NATIVE_DIM {
        // Small images don't have strip support — fall back to Full
        // and let the caller surface the TooBigForFull.
        return Err(crate::Error::TooBigForFull {
            needed: full_bytes,
            cap,
        });
    }
    if let Some(h_body) = auto_size_strip_body(width, height, cap) {
        return Ok(ResolvedMode::Strip { h_body });
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
}

#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap)
        .unwrap_or(crate::pipeline::STRIP_DEFAULT_BODY.min(round_align(height)))
        .max(PYRAMID_ALIGN)
}

fn round_align(h: u32) -> u32 {
    h.div_ceil(PYRAMID_ALIGN) * PYRAMID_ALIGN
}

fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    let halo_bytes = estimate_strip_gpu_memory_bytes(width, 0)?;
    if halo_bytes >= cap {
        return None;
    }
    let one_unit = estimate_strip_gpu_memory_bytes(width, PYRAMID_ALIGN)?;
    let per_unit = one_unit.saturating_sub(halo_bytes);
    if per_unit == 0 {
        let fb = crate::pipeline::STRIP_DEFAULT_BODY.min(round_align(height));
        let est = estimate_strip_gpu_memory_bytes(width, fb)?;
        return if est <= cap { Some(fb) } else { None };
    }
    let max_units = (cap - halo_bytes) / per_unit;
    let raw = (max_units as u32) * PYRAMID_ALIGN;
    let body = raw.min(round_align(height));
    if body < PYRAMID_ALIGN {
        return None;
    }
    Some(body)
}

/// Estimate the GPU working-set bytes [`crate::pipeline::Iwssim::new`]
/// allocates for `width × height` images.
///
/// Counted buffers per scale (5 scales): the `Scale` struct allocates
/// roughly 10 planes of f32 (`lp_ref`, `lp_dis`, `mu1`, `mu2`,
/// `sig1_sq`, `sig2_sq`, `sig12`, `cs`, `iw`, scratch). Two packed-u32
/// sRGB staging buffers at scale 0. Plus small reduction buffers
/// (partials, sums, cov_partials) which are negligible.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 10;
    for _ in 0..NUM_SCALES {
        let w_eff = w as usize;
        let h_eff = h as usize;
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n0 = (width as usize) * (height as usize);
    total = total.saturating_add(n0 * 4 * 2);
    total
}

/// Strip-mode estimator. Same planes as
/// [`estimate_gpu_memory_bytes`] but sized for
/// `width × (h_body + 2 × halo)`, default halo = 256.
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    let strip_h = (h_body as usize).saturating_add((STRIP_DEFAULT_HALO as usize) * 2);
    let mut w = width as usize;
    let mut h = strip_h;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 10;
    for _ in 0..NUM_SCALES {
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w * h * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    let n = (width as usize).saturating_mul(strip_h);
    total = total.saturating_add(n * 4 * 2);
    Some(total)
}
