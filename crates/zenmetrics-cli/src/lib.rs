#![forbid(unsafe_code)]
#![allow(clippy::doc_lazy_continuation)]

//! Library facade for the `zenmetrics` binary.
//!
//! The vast majority of users want the CLI (`zenmetrics` binary). This
//! library surface exists so the unified fleet worker
//! (`crates/zenfleet-vastai`) can call the same code paths in-process and
//! share one `cubecl` device across hundreds of metric evaluations
//! within a worker's lifetime — eliminating the ~3-5 s cubecl init
//! that costs ~30× per chunk when the dispatcher subprocess-spawns
//! `zenmetrics sweep` once per (codec, knob_tuple) group.
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
//! zenmetrics-cli = { path = "...", default-features = false,
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

/// HDR decode + per-metric feeding front-end (PQ-PNG / EXR / gain-map
/// sources → absolute nits). The sweep's `--hdr` mode (`sweep::hdr`)
/// builds on this, so the lib compiles it when the `hdr` feature is on.
#[cfg(feature = "hdr")]
#[allow(dead_code)]
pub mod hdr;

/// Orchestrator integration (Phase 7). Bridges the CLI's
/// [`metrics::MetricKind`] enum and the
/// [`zenmetrics_orchestrator::Orchestrator`] surface. Library callers
/// that want OOM-safe, perf-aware multi-task scoring should drive the
/// orchestrator directly via this module's helpers; the legacy
/// per-subcommand handlers below the surface here still work for
/// simple cases.
#[cfg(feature = "orchestrator")]
pub mod orchestrator_glue;

/// Orchestrator-driven re-implementations of the legacy subcommand
/// handlers. Lives behind the `orchestrator` feature and only fires
/// when `--use-orchestrator` (CLI flag) or
/// `ZENMETRICS_USE_ORCHESTRATOR=1` (env) is set.
#[cfg(feature = "orchestrator")]
pub mod orchestrator_runner;
