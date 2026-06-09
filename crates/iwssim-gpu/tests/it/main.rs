//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

mod auto_fallback;
#[cfg(feature = "cubecl-types")]
mod cached_ref_strip;
mod memory_mode;
#[cfg(feature = "cubecl-types")]
mod opaque;
#[cfg(feature = "cuda")]
mod parity_cpu;
#[cfg(feature = "cubecl-types")]
mod parity_lock;
#[cfg(feature = "cubecl-types")]
mod rgb_strip;
#[cfg(feature = "cubecl-types")]
mod rgb_strip_native;
#[cfg(feature = "cubecl-types")]
mod small_image_adaptive;
#[cfg(feature = "cubecl-types")]
mod strip_parity;
mod sub_min_score;
mod vram_probe;
