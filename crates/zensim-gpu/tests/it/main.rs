//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

mod auto_fallback;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
mod cached_ref_slot_rebuild;
#[cfg(feature = "cubecl-types")]
mod cpu_gpu_diffmap_parity;
#[cfg(feature = "cubecl-types")]
mod cpu_gpu_feature_sweep;
#[cfg(feature = "cubecl-types")]
mod cpu_parity;
#[cfg(feature = "cubecl-types")]
mod diffmap_invariants;
mod extended_parity;
mod memory_mode;
#[cfg(feature = "cubecl-types")]
mod opaque;
#[cfg(feature = "cubecl-types")]
mod opaque_cached_ref;
mod opaque_default_weights_v03;
#[cfg(feature = "cubecl-types")]
mod opaque_regime;
#[cfg(feature = "cubecl-types")]
mod parity_lock;
#[cfg(feature = "cubecl-types")]
mod pu_xyb_parity;
mod strip_parity;
mod sub64_reflect_pad;
mod typed_sub_min_pad;
mod weights_parity;
