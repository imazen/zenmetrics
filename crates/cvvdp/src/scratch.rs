//! Persistent scratch buffers used by `Cvvdp` to avoid per-call
//! allocations.
//!
//! Four classes of scratch:
//!
//! - **Color planes**: 3× `Vec<f32>` per side (REF + DIST) at full
//!   resolution. Reused across calls; sized to `w * h` once on
//!   `new()`.
//! - **Pyramid scratch**: 5 buffers used by the reduce/expand
//!   passes — see `pyramid::PyramidScratch`.
//! - **Per-band scratch**: per-band `T_p`, `R_p`, `D` buffers sized
//!   to each band's pixel count. Re-used across bands by resizing.
//! - **WeberPyramid slots + per-band workspaces**: 6 `WeberPyramid`
//!   output buffers (one per channel × side) and `n_levels` band
//!   workspaces, each holding the 13 per-band Vec<f32> slots
//!   consumed by the band loop. Indexed by band index, so the
//!   parallel band loop can write into them without aliasing.

use alloc::vec;
use alloc::vec::Vec;

use crate::pyramid::{PyramidScratch, WeberPyramid, WeberPyramidCache};

/// Per-band workspace owned by `Scratch`. Each band index in
/// `fold_bands_parallel` gets its own slot, so the rayon parallel
/// iterator writes into mutually-exclusive memory.
///
/// All fields are `pub(crate)` so the band loop can directly resize
/// + write into them. Sizes match the per-band pixel count.
///
/// **Strip variant (Phase 9.Z.F / chunk 4)**: see
/// [`StripBandWorkspace`] below — a slimmed-down version sized at
/// `R_k × bw` (strip-shaped) rather than `bh × bw` (full-image). For
/// shallow levels (`k < k_split`) the strip walker allocates one
/// `StripBandWorkspace` per shallow level; deep levels reuse the
/// existing full-image `BandWorkspace`.
#[derive(Default)]
pub(crate) struct BandWorkspace {
    // T_p / R_p per channel: CSF-weighted contrasts.
    pub t_p_a: Vec<f32>,
    pub t_p_rg: Vec<f32>,
    pub t_p_vy: Vec<f32>,
    pub r_p_a: Vec<f32>,
    pub r_p_rg: Vec<f32>,
    pub r_p_vy: Vec<f32>,
    // D per channel: post-masking diff.
    pub d_a: Vec<f32>,
    pub d_rg: Vec<f32>,
    pub d_vy: Vec<f32>,
    // mult_mutual intermediates.
    pub m_mm_a: Vec<f32>,
    pub m_mm_rg: Vec<f32>,
    pub m_mm_vy: Vec<f32>,
    pub term_a: Vec<f32>,
    pub term_rg: Vec<f32>,
    pub term_vy: Vec<f32>,
    pub pu_h: Vec<f32>,
}

/// Strip-shaped per-band workspace owned by `Scratch`.
///
/// Chunk 4 of the cvvdp CPU K_SPLIT walker port. For shallow levels
/// (`k < k_split`) the strip walker dispatches band-fold work in
/// row-strips of `R_k` rows × `bw` columns instead of the full
/// `bh × bw` rectangle. Each shallow level gets one
/// `StripBandWorkspace` slot sized at `R_k × bw` (a strict subset of
/// the full band's pixel count for `R_k < bh`).
///
/// At `h_body = 512, k = 0, W = H = 4096`, this is `1148 × 4096 × 4
/// bytes = 18.4 MB` per buffer vs the full-image's `4096 × 4096 × 4
/// bytes = 64 MB` per buffer — a ~3.5× reduction at the level
/// dominating peak heap.
///
/// Mirrors the GPU's `DBandsTransient::new_strip` (cvvdp-gpu
/// `pipeline.rs:6388` allocator). Allocated by `Scratch::ensure_strip_band_ws`
/// (added in chunk 6's dispatcher wiring); unused until then.
#[derive(Default)]
#[allow(dead_code)] // wired in by chunk 6 (strip-major dispatcher)
pub(crate) struct StripBandWorkspace {
    pub t_p_a: Vec<f32>,
    pub t_p_rg: Vec<f32>,
    pub t_p_vy: Vec<f32>,
    pub r_p_a: Vec<f32>,
    pub r_p_rg: Vec<f32>,
    pub r_p_vy: Vec<f32>,
    pub d_a: Vec<f32>,
    pub d_rg: Vec<f32>,
    pub d_vy: Vec<f32>,
    pub m_mm_a: Vec<f32>,
    pub m_mm_rg: Vec<f32>,
    pub m_mm_vy: Vec<f32>,
    pub term_a: Vec<f32>,
    pub term_rg: Vec<f32>,
    pub term_vy: Vec<f32>,
    pub pu_h: Vec<f32>,
}

#[allow(dead_code)] // wired in by chunk 6
impl StripBandWorkspace {
    /// Allocate (or resize) every per-band slot to `n_strip = R_k × bw`
    /// f32 entries.
    pub(crate) fn ensure_strip_sized(&mut self, n_strip: usize) {
        self.t_p_a.clear();
        self.t_p_a.resize(n_strip, 0.0);
        self.t_p_rg.clear();
        self.t_p_rg.resize(n_strip, 0.0);
        self.t_p_vy.clear();
        self.t_p_vy.resize(n_strip, 0.0);
        self.r_p_a.clear();
        self.r_p_a.resize(n_strip, 0.0);
        self.r_p_rg.clear();
        self.r_p_rg.resize(n_strip, 0.0);
        self.r_p_vy.clear();
        self.r_p_vy.resize(n_strip, 0.0);
        self.d_a.clear();
        self.d_a.resize(n_strip, 0.0);
        self.d_rg.clear();
        self.d_rg.resize(n_strip, 0.0);
        self.d_vy.clear();
        self.d_vy.resize(n_strip, 0.0);
        self.m_mm_a.clear();
        self.m_mm_a.resize(n_strip, 0.0);
        self.m_mm_rg.clear();
        self.m_mm_rg.resize(n_strip, 0.0);
        self.m_mm_vy.clear();
        self.m_mm_vy.resize(n_strip, 0.0);
        self.term_a.clear();
        self.term_a.resize(n_strip, 0.0);
        self.term_rg.clear();
        self.term_rg.resize(n_strip, 0.0);
        self.term_vy.clear();
        self.term_vy.resize(n_strip, 0.0);
        self.pu_h.clear();
        self.pu_h.resize(n_strip, 0.0);
    }
}

/// All scratch state owned by a `Cvvdp` instance.
///
/// Layout:
///
/// - `dist_*` / `ref_*`: per-side DKL planes (full resolution).
/// - `pyr`: shared pyramid reduce/expand scratch (legacy slot, used
///   by the cold sequential test path).
/// - `weber_ref` / `weber_dist`: 3 `WeberPyramid` output slots per
///   side. Buffers persist across calls; `weber_contrast_pyr_into`
///   resizes-in-place.
/// - `weber_cache_ref` / `weber_cache_dist`: per-channel pyramid
///   caches (the two intermediate Gaussian pyramids + per-channel
///   PyramidScratch). One cache per channel per side.
/// - `band_ws`: one `BandWorkspace` per band, indexed by band id.
///   Each holds the 16 per-band working Vec<f32>s so the parallel
///   band loop can write into mutually-exclusive memory without
///   aliasing. Grown lazily inside `fold_bands_parallel` to match
///   `n_levels`.
pub(crate) struct Scratch {
    // Source-derived DKL planes for the DIST side (REF lives in the
    // warm reference state when warm_reference has been called).
    pub dist_a: Vec<f32>,
    pub dist_rg: Vec<f32>,
    pub dist_vy: Vec<f32>,
    // Same for REF when not using warm-ref.
    pub ref_a: Vec<f32>,
    pub ref_rg: Vec<f32>,
    pub ref_vy: Vec<f32>,
    // Legacy pyramid scratch — kept for the warm_reference path which
    // builds its own single-side pyramid outside the recycle scaffolding.
    #[allow(dead_code)]
    pub pyr: PyramidScratch,
    // Persistent WeberPyramid output slots (3 per side).
    pub weber_ref: [WeberPyramid; 3],
    pub weber_dist: [WeberPyramid; 3],
    // Per-channel pyramid caches (3 per side).
    pub weber_cache_ref: [WeberPyramidCache; 3],
    pub weber_cache_dist: [WeberPyramidCache; 3],
    // Per-band workspaces. Indexed by band id; grown lazily.
    pub band_ws: Vec<BandWorkspace>,
    // Per-shallow-level strip workspaces. `Some(slots)` only when strip
    // mode is active and the dispatcher has called
    // `ensure_strip_band_ws`. `slots[k]` is a strip-shaped
    // `StripBandWorkspace` for shallow level `k`; deep levels reuse
    // `band_ws[k]` (which is small at deep levels).
    //
    // Chunk 4 of the CPU K_SPLIT walker port. Allocated lazily so
    // Full-mode `Scratch::new` doesn't pay the strip allocator's
    // cost.
    #[allow(dead_code)] // wired in by chunk 6
    pub strip_band_ws: Option<Vec<StripBandWorkspace>>,
}

impl Scratch {
    /// Construct with full pre-allocation for `width × height` image and
    /// `n_levels` weber pyramid bands.
    ///
    /// Phase 9.YA Part 2: every per-level Vec<f32> in
    /// `weber_ref` / `weber_dist` (output bands + log_l_bkg) and in
    /// `weber_cache_ref` / `weber_cache_dist` (gauss_img + gauss_l +
    /// inner PyramidScratch) is sized at construction time so the
    /// first `score()` call doesn't take the cold-allocation hit. This
    /// pre-allocation is large (~3 GB at 40 MP for the 6 caches + 6
    /// outputs combined) but the alternative was paying it inside the
    /// first per-call build path. Both `score()` cold path and
    /// `warm_reference` cold-path callers benefit.
    pub fn new(width: usize, height: usize, n_levels: usize) -> Self {
        let n = width * height;
        Self {
            dist_a: vec![0.0; n],
            dist_rg: vec![0.0; n],
            dist_vy: vec![0.0; n],
            ref_a: vec![0.0; n],
            ref_rg: vec![0.0; n],
            ref_vy: vec![0.0; n],
            pyr: PyramidScratch::default(),
            weber_ref: [
                WeberPyramid::with_capacity(width, height, n_levels),
                WeberPyramid::with_capacity(width, height, n_levels),
                WeberPyramid::with_capacity(width, height, n_levels),
            ],
            weber_dist: [
                WeberPyramid::with_capacity(width, height, n_levels),
                WeberPyramid::with_capacity(width, height, n_levels),
                WeberPyramid::with_capacity(width, height, n_levels),
            ],
            // weber_cache_ref is NOT pre-allocated — it's only used by
            // the cold `score()` path, never by the warm
            // `warm_reference` + `score_with_warm_ref` path (which uses
            // local caches in `build_one_side_warm_ref_into` from
            // Phase 9.YA Part 1). Pre-allocating it would burn ~640 MB
            // of peak heap in the warm path with no benefit.
            //
            // The cold score() path's first call still pays the
            // alloc cost for weber_cache_ref's gauss_img + gauss_l
            // band growth (~213 MB × 6 = 1.3 GB across 3 ref channels'
            // gauss_img + gauss_l). Subsequent calls reuse capacity.
            weber_cache_ref: [
                WeberPyramidCache::default(),
                WeberPyramidCache::default(),
                WeberPyramidCache::default(),
            ],
            // weber_cache_dist IS pre-allocated — it's used by both
            // cold `score()` and warm `score_with_warm_ref` paths.
            // Pre-allocation removes the first-call cost for the
            // distorted-side pyramid build in both modes.
            weber_cache_dist: [
                WeberPyramidCache::with_capacity(width, height, n_levels),
                WeberPyramidCache::with_capacity(width, height, n_levels),
                WeberPyramidCache::with_capacity(width, height, n_levels),
            ],
            band_ws: Vec::new(),
            strip_band_ws: None,
        }
    }

    /// Grow the per-band workspace slot vector to at least `n_levels`
    /// entries. Called by the band loop before parallel dispatch.
    pub fn ensure_band_ws(&mut self, n_levels: usize) {
        while self.band_ws.len() < n_levels {
            self.band_ws.push(BandWorkspace::default());
        }
    }

    /// Grow the per-shallow-level strip workspace vector to at least
    /// `k_split` entries. Called by the strip-major dispatcher before
    /// strip iteration. Each shallow level's slot is sized at
    /// `R_k × bw` via [`StripBandWorkspace::ensure_strip_sized`] when
    /// the strip is dispatched.
    ///
    /// Chunk 4 of the CPU K_SPLIT walker port; populated lazily so
    /// the Full-mode path's `Scratch::new` doesn't pay the strip
    /// allocator's cost.
    #[allow(dead_code)] // wired in by chunk 6
    pub fn ensure_strip_band_ws(&mut self, k_split: usize) {
        if self.strip_band_ws.is_none() {
            self.strip_band_ws = Some(Vec::new());
        }
        let slots = self.strip_band_ws.as_mut().unwrap();
        while slots.len() < k_split {
            slots.push(StripBandWorkspace::default());
        }
    }
}
