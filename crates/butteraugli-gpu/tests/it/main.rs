//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

mod auto_fallback;
mod memory_mode;
mod multires_strip;
#[cfg(feature = "cubecl-types")]
mod opaque;
mod opaque_strip_parity;
#[cfg(feature = "cubecl-types")]
mod reduction_parity;
#[cfg(all(feature = "cubecl-types", feature = "cuda", feature = "internals"))]
mod set_reference_from_linear_planes;
mod strip_hf_checkerboard;
#[cfg(feature = "cubecl-types")]
mod strip_parity;
mod vram_no_leak;
