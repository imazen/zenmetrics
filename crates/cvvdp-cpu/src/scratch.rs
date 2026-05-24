//! Persistent scratch buffers used by `Cvvdp` to avoid per-call
//! allocations.
//!
//! Three classes of scratch:
//!
//! - **Color planes**: 3× `Vec<f32>` per side (REF + DIST) at full
//!   resolution. Reused across calls; sized to `w * h` once on
//!   `new()`.
//! - **Pyramid scratch**: 5 buffers used by the reduce/expand
//!   passes — see `pyramid::PyramidScratch`.
//! - **Per-band scratch**: per-band `T_p`, `R_p`, `D` buffers sized
//!   to each band's pixel count. Re-used across bands by resizing.

use alloc::vec;
use alloc::vec::Vec;

use crate::pyramid::PyramidScratch;

/// All scratch state owned by a `Cvvdp` instance.
///
/// Per-band T_p / R_p / D / m_mm fields are reserved for future
/// SIMD-vectorized in-place masking passes (B7c-style buffer
/// recycling). The current pipeline allocates per-band working
/// buffers because the upstream `mult_mutual_band` helper consumes
/// owned `[Vec<f32>; 3]` triples. Keep the fields for the next
/// optimization round.
#[allow(dead_code)]
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
    // Pyramid scratch — reused.
    pub pyr: PyramidScratch,
    // Per-band working buffers (T_p/R_p per channel, D per channel).
    pub t_p_a: Vec<f32>,
    pub t_p_rg: Vec<f32>,
    pub t_p_vy: Vec<f32>,
    pub r_p_a: Vec<f32>,
    pub r_p_rg: Vec<f32>,
    pub r_p_vy: Vec<f32>,
    pub d_a: Vec<f32>,
    pub d_rg: Vec<f32>,
    pub d_vy: Vec<f32>,
    // For mult_mutual_band: min(|T|, |R|) intermediate.
    pub m_mm_a: Vec<f32>,
    pub m_mm_rg: Vec<f32>,
    pub m_mm_vy: Vec<f32>,
    // Phase-uncertainty horizontal pass scratch.
    pub pu_h: Vec<f32>,
}

impl Scratch {
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height;
        Self {
            dist_a: vec![0.0; n],
            dist_rg: vec![0.0; n],
            dist_vy: vec![0.0; n],
            ref_a: vec![0.0; n],
            ref_rg: vec![0.0; n],
            ref_vy: vec![0.0; n],
            pyr: PyramidScratch::default(),
            t_p_a: Vec::new(),
            t_p_rg: Vec::new(),
            t_p_vy: Vec::new(),
            r_p_a: Vec::new(),
            r_p_rg: Vec::new(),
            r_p_vy: Vec::new(),
            d_a: Vec::new(),
            d_rg: Vec::new(),
            d_vy: Vec::new(),
            m_mm_a: Vec::new(),
            m_mm_rg: Vec::new(),
            m_mm_vy: Vec::new(),
            pu_h: Vec::new(),
        }
    }
}
