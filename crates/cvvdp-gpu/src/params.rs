//! Phase 8c.1-B re-export shim: `params` now lives in the `cvvdp`
//! crate (CPU side), which is the canonical owner of the shared
//! parameter types + display model. This module re-exports the
//! entire `cvvdp::params` surface so existing
//! `cvvdp_gpu::params::*` callsites continue to resolve unchanged.

pub use cvvdp::params::*;
