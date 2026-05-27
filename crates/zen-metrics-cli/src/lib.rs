#![forbid(unsafe_code)]

//! Library facade for the `zen-metrics` binary.
//!
//! The vast majority of users want the CLI (`zen-metrics` binary). This
//! library surface exists so the unified fleet worker
//! (`crates/zen-cloud-vastai`) can call the same code paths in-process and
//! share one `cubecl` device across hundreds of metric evaluations
//! within a worker's lifetime — eliminating the ~3-5 s cubecl init
//! that costs ~30× per chunk when the dispatcher subprocess-spawns
//! `zen-metrics sweep` once per (codec, knob_tuple) group.
//!
//! ## What's re-exported
//!
//! Only the public APIs the worker actually consumes:
//!
//! - [`sweep::run_sweep`] + [`sweep::SweepConfig`] — the encode +
//!   score + emit pipeline.
//! - [`sweep::CodecKind`] + [`sweep::KnobGrid`] + [`sweep::parse_knob_grid`]
//!   + [`sweep::parse_q_grid`] — config parsing helpers so callers
//!   build a `SweepConfig` without re-implementing the JSON
//!   structure the CLI accepts.
//! - [`metrics::MetricKind`] + [`metrics::GpuRuntime`] — the
//!   enum types `SweepConfig` carries.
//!
//! ## What's NOT re-exported
//!
//! Internal modules (`compare`, `decode`, `output`) stay private to
//! the binary. If a future caller needs them, the library facade
//! can be widened — but doing so on demand keeps the public surface
//! minimal.
//!
//! ## Feature gating
//!
//! This lib's feature flags mirror the binary's. Build with the
//! same feature set the binary was built with for consistent
//! codec + metric support:
//!
//! ```toml
//! zen-metrics-cli = { path = "...", default-features = false,
//!                     features = ["sweep","png","jpeg","webp","gpu-cuda"] }
//! ```

// Module declarations duplicated from main.rs. We can't simply
// `pub mod` from a binary entry point; lib.rs needs its own root.
// The modules themselves are unchanged — same source files, two
// crate roots referencing them.
pub mod metrics;
pub mod output;

/// Typed full-key corpus assembler — see the [`assemble`] module docs. Gated
/// on the lean `assemble` feature (arrow/parquet only); `sweep` enables it too.
#[cfg(feature = "assemble")]
pub mod assemble;

#[cfg(feature = "sweep")]
pub mod sweep;

#[allow(dead_code)] // Only used by the binary, but compiled for the lib.
pub mod compare;
#[allow(dead_code)]
pub mod decode;
