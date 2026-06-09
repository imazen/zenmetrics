//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

#[cfg(feature = "cubecl-types")]
mod aliasing_invariants;
mod auto_fallback;
#[cfg(feature = "cubecl-types")]
mod blur_mode_api;
#[cfg(all(feature = "cubecl-types", feature = "fir"))]
mod fir_path;
mod memory_mode;
#[cfg(feature = "cubecl-types")]
mod opaque;
#[cfg(feature = "cubecl-types")]
mod parity_lock;
#[cfg(feature = "cubecl-types")]
mod reduction_determinism;
#[cfg(feature = "cubecl-types")]
mod ssim2_skipmap_audit;
#[cfg(feature = "cubecl-types")]
mod strip_parity;
mod sub_min_reflect_pad;
mod typed_sub_min_pad;
