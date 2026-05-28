//! Phase 8c.1-B re-export shim: `presets` now lives in the `cvvdp`
//! crate (CPU side) as the canonical owner of the JSON-loaded display
//! registry. This module re-exports the entire `cvvdp::presets`
//! surface so existing `cvvdp_gpu::presets::*` callsites resolve
//! unchanged.

pub use cvvdp::presets::*;
