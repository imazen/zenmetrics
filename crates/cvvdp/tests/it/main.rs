//! Consolidated integration-test entry point.
//!
//! Every former `tests/<name>.rs` is a submodule here, compiled into one
//! `it` test binary instead of N separate binaries (one link step, not N).
//! Per-test gating that used to live in `[[test]] required-features` is now
//! a `#[cfg(...)]` on each `mod` line. Select a former target with a module
//! filter: `cargo test --test it <name>::`.

mod diffmap_invariants;
mod k_split_gpu_parity;
mod parity_against_host_scalar;
mod parity_corpus;
mod pixels_integration;
mod simd_equivalence;
mod strip_parity;
mod strip_stub;
mod upstream_parity_extended;
