//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

#[cfg(feature = "cuda")]
mod cached_ref;
#[cfg(feature = "bench")]
mod chooser;
#[cfg(feature = "cuda")]
mod cpu_backend;
#[cfg(feature = "cuda")]
mod executor;
#[cfg(feature = "cuda")]
mod gpu_concurrency;
mod multiwarm_pool;
#[cfg(feature = "cuda")]
mod no_gpu_fallback;
#[cfg(feature = "cuda")]
mod reorder;
#[cfg(feature = "cuda")]
mod streaming;
