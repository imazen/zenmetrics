//! Scalar kernel helpers for the ColorVideoVDP still-image pipeline.
//!
//! Phase 8c.1-B moved these out of `cvvdp-gpu::kernels` so the CPU
//! crate owns the canonical scalar implementations + shared
//! constants. cvvdp-gpu continues to re-export the same paths via
//! shims, so existing `cvvdp_gpu::kernels::*` callsites resolve
//! unchanged.
//!
//! The GPU-side `#[cube(launch)]` kernels (52 in total) remain in
//! `cvvdp-gpu::kernels` — they depend on cubecl which is GPU-side
//! only.

pub mod color;
pub mod csf;
pub mod diffmap;
pub mod masking;
pub mod pool;
pub mod pyramid;
