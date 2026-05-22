//! Unified memory-mode API. See `butteraugli-gpu/src/memory_mode.rs`
//! for the shared design rationale.
//!
//! dssim-gpu is **NOT strip-preferred**: strip mode is 2-5× slower
//! than whole-image on this crate (bench at
//! `benchmarks/dssim_strip_vs_whole_2026-05-22.md`). Auto picks Full
//! whenever it fits the cap; Strip only when Full would exceed the
//! cap.

use crate::NUM_SCALES;

fn env_cap_bytes() -> Option<usize> {
    std::env::var("ZENMETRICS_VRAM_CAP_BYTES")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Effective cap policy. See `butteraugli_gpu::memory_mode::vram_cap_bytes`
/// for details — identical semantics here.
pub fn vram_cap_bytes() -> usize {
    if let Some(cap) = env_cap_bytes() {
        return cap;
    }
    8 * 1024 * 1024 * 1024
}

/// How the GPU pipeline should partition its working set. dssim-gpu
/// supports `Auto`, `Full`, and `Strip`; `Tile` is reserved.
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

/// Minimum strip body for dssim — must be a multiple of
/// `2^(NUM_SCALES − 1) = 16` per [`crate::pipeline::Dssim::new_strip`].
const PYRAMID_ALIGN: u32 = 1 << (NUM_SCALES as u32 - 1);
const MIN_STRIP_BODY: u32 = PYRAMID_ALIGN * 4; // 64 rows

/// Auto policy: prefer Full. dssim's strip walker is 2-5× slower than
/// whole-image, so we only switch to Strip when Full would exceed the
/// cap.
pub fn resolve_auto(
    width: u32,
    height: u32,
    cap: usize,
) -> crate::Result<ResolvedMode> {
    let full_bytes = estimate_gpu_memory_bytes(width, height);
    if full_bytes <= cap {
        return Ok(ResolvedMode::Full);
    }
    if let Some(h_body) = auto_size_strip_body(width, height, cap) {
        return Ok(ResolvedMode::Strip { h_body });
    }
    Err(crate::Error::TooBigForFull {
        needed: full_bytes,
        cap,
    })
}

/// Public auto-sizer for callers passing
/// `MemoryMode::Strip { h_body: None }`. Always returns a positive,
/// pyramid-aligned body.
#[must_use]
pub fn auto_strip_body_for(width: u32, height: u32, cap: usize) -> u32 {
    auto_size_strip_body(width, height, cap)
        .unwrap_or_else(|| MIN_STRIP_BODY.min(round_up_align(height)).max(PYRAMID_ALIGN))
}

fn round_up_align(h: u32) -> u32 {
    h.div_ceil(PYRAMID_ALIGN) * PYRAMID_ALIGN
}

fn auto_size_strip_body(width: u32, height: u32, cap: usize) -> Option<u32> {
    let halo_bytes = estimate_strip_gpu_memory_bytes(width, 0)?;
    if halo_bytes >= cap {
        return None;
    }
    let one_row = estimate_strip_gpu_memory_bytes(width, PYRAMID_ALIGN)?;
    let per_align_unit = one_row.saturating_sub(halo_bytes);
    if per_align_unit == 0 {
        let fb = MIN_STRIP_BODY.min(round_up_align(height));
        let est = estimate_strip_gpu_memory_bytes(width, fb)?;
        return if est <= cap { Some(fb) } else { None };
    }
    let max_units = (cap - halo_bytes) / per_align_unit;
    let raw = (max_units as u32) * PYRAMID_ALIGN;
    let max_body = raw.min(round_up_align(height));
    if max_body < MIN_STRIP_BODY {
        // Try MIN_STRIP_BODY itself in case the linearization
        // undershoots — strip allocation has bounded constants.
        let est = estimate_strip_gpu_memory_bytes(width, MIN_STRIP_BODY)?;
        if est <= cap {
            return Some(MIN_STRIP_BODY.min(round_up_align(height)));
        }
        return None;
    }
    Some(max_body)
}

/// Estimate the GPU working-set bytes [`crate::pipeline::Dssim::new`]
/// allocates for `width × height` images.
///
/// Counted buffers per scale (5 scales): the `Scale` struct allocates
/// roughly 13 planes of `(w_s × h_s) × f32` (LP_ref, LP_dis, mu1, mu2,
/// sig11, sig22, sig12, ssim, ssim_mu, scratch_a/b/c, plus a couple
/// of small reduction buffers). Two staging packed-u32 sRGB buffers
/// at scale 0 (`n × 4` each) round out the global state. Constants
/// from `crates/dssim-gpu/src/pipeline.rs::Scale::new`.
#[must_use]
pub fn estimate_gpu_memory_bytes(width: u32, height: u32) -> usize {
    let mut w = width;
    let mut h = height;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 13;
    for _ in 0..NUM_SCALES {
        let w_eff = w.max(8) as usize;
        let h_eff = h.max(8) as usize;
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    // Two staging packed-u32 sRGB buffers (4 B per pixel).
    let n0 = (width as usize) * (height as usize);
    total = total.saturating_add(n0 * 4 * 2);
    total
}

/// Strip-mode estimator. Same planes as
/// [`estimate_gpu_memory_bytes`] but sized for
/// `width × (h_body + 2 × HALO)`. HALO = 256 is fixed in
/// [`crate::pipeline::Dssim::new_strip`].
#[must_use]
pub fn estimate_strip_gpu_memory_bytes(width: u32, h_body: u32) -> Option<usize> {
    const HALO: u32 = 256;
    let strip_h = (h_body as usize).saturating_add((HALO as usize) * 2);
    let mut w = width as usize;
    let mut h = strip_h;
    let mut total: usize = 0;
    const PLANES_PER_SCALE: usize = 13;
    for _ in 0..NUM_SCALES {
        let w_eff = w.max(8);
        let h_eff = h.max(8);
        total = total.saturating_add(PLANES_PER_SCALE.saturating_mul(w_eff * h_eff * 4));
        w = w.div_ceil(2);
        h = h.div_ceil(2);
    }
    // Two staging packed-u32 sRGB buffers at the strip pixel count.
    let n = (width as usize).saturating_mul(strip_h);
    total = total.saturating_add(n * 4 * 2);
    Some(total)
}
