//! Filter coefficients for the IW-SSIM kernels, as crate-level
//! `pub(crate) const` so kernels can read them at compile time.
//!
//! Why iwssim-gpu keeps its own copy (instead of `pub use
//! iwssim::filters::*`): cube-macros that consume these constants
//! (`lap_pyramid`, `gauss11` `#[cube(launch_unchecked)]` kernels)
//! require them to resolve through the **local** `crate::filters` path
//! at macro-expansion time — the cube codegen captures `crate`-relative
//! paths and re-emits them device-side, so a cross-crate re-export is
//! not name-resolvable inside a `#[cube]` body.
//!
//! Bit-identical to `iwssim/src/filters.rs` (the same literal values).
//! Previously emitted at build time by the `iwssim-filter-codegen`
//! helper crate; flattened to committed literals on 2026-06-26 (the
//! values are fixed mathematical constants — see `iwssim/src/filters.rs`
//! for the derivation and the invariant-guard tests). The literals are
//! the exact bytes the codegen emitted.

// A few companion scalars (`BINOM5_LEN`, `SSIM_WIN_LEN`,
// `SSIM_WIN_RADIUS`) are read only by some kernels; allow dead_code.
#![allow(dead_code)]

// BINOM5 — pyrtools `binom5` taps: `sqrt(2) * [1,4,6,4,1] / 16`.
pub(crate) const BINOM5_LEN: usize = 5;
pub(crate) const BINOM5_RADIUS: i32 = 2;
pub(crate) const BINOM5: [f32; 5] = [
    8.83883476483184465922e-2_f32,
    3.53553390593273786369e-1_f32,
    5.30330085889910707309e-1_f32,
    3.53553390593273786369e-1_f32,
    8.83883476483184465922e-2_f32,
];

// SSIM_WIN_1D — `fspecial('gaussian', 11, 1.5)` applied separably.
pub(crate) const SSIM_WIN_LEN: usize = 11;
pub(crate) const SSIM_WIN_RADIUS: i32 = 5;
pub(crate) const SSIM_WIN_1D: [f32; 11] = [
    1.02838008447911008481e-3_f32,
    7.59875813523918502979e-3_f32,
    3.60007721284308288001e-2_f32,
    1.09360689509700015343e-1_f32,
    2.13005537711253689626e-1_f32,
    2.66011724861794363051e-1_f32,
    2.13005537711253689626e-1_f32,
    1.09360689509700015343e-1_f32,
    3.60007721284308288001e-2_f32,
    7.59875813523918502979e-3_f32,
    1.02838008447911008481e-3_f32,
];

// SCALE_WEIGHTS — per-scale MS-SSIM combination weights (β in eq 47
// of Wang & Li 2011) verbatim from `iwssim.m` / `IW_SSIM_PyTorch.py`.
pub(crate) const SCALE_WEIGHTS: [f32; 5] = [
    4.48e-2_f32,
    2.856e-1_f32,
    3.001e-1_f32,
    2.363e-1_f32,
    1.333e-1_f32,
];
