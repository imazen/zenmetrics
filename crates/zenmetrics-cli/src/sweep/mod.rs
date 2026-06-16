#![forbid(unsafe_code)]

//! `zenmetrics sweep` — encode a corpus of source images across a Cartesian
//! grid of (codec, quality, knob-tuple) triples and score every encoded
//! variant with one or more perceptual metrics.
//!
//! The driver lives entirely outside any codec source tree: codecs stay
//! "dumb" (no in-codec picker glue, no `.bin` shipped alongside the encoder)
//! and we orchestrate science from this binary by poking each codec's
//! `__expert` / public encode API. One encoded variant per cell is held in
//! memory just long enough to (a) decode it back to RGB8 for scoring and
//! (b) write a Pareto row, then dropped.
//!
//! Output schema (TSV, one row per cell):
//! ```text
//! image_path codec q knob_tuple_json encoded_bytes encode_ms decode_ms \
//!     score_<col_1> score_<col_2> ...
//! ```
//!
//! Metric columns come from [`crate::metrics::MetricKind::column_names`].
//! Most metrics emit a single column; butteraugli (CPU and GPU) emits two
//! — `butteraugli_max{,_gpu}` and `butteraugli_pnorm3{,_gpu}` — because
//! one `compute()` call yields both aggregations. Cells that fail to encode
//! or score are emitted with `score_*` columns blank but the `encode_ms` /
//! `decode_ms` columns still populated where possible, so a downstream
//! filter step can drop or quarantine partial rows without having to re-run
//! the sweep.

pub mod encode;
pub mod feature_writer;
pub mod grid;
/// HDR sweep mode (`--hdr`): PQ-PNG refs -> nits, HDR-capable codec
/// round-trip, validated per-metric HDR feedings. See the module docs
/// for which codecs are wired (zenjxl today) and why SDR-only codecs
/// are refused rather than approximated.
#[cfg(feature = "hdr")]
pub mod hdr;
#[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
pub mod plan;
pub mod run;

pub use encode::CodecKind;
#[allow(unused_imports)]
pub use grid::{KnobGrid, KnobTuple, parse_knob_grid, parse_q_grid};
pub use run::{PlanSpec, SweepConfig, run_sweep, try_init_thread_pool};

/// Re-export of the orchestrator handle type used by the sweep
/// runner — pulled only when the `orchestrator` feature is on.
/// `cmd_sweep` builds the wrapped orchestrator at sweep entry and
/// passes it into `run_sweep` so the per-cell loop can dispatch
/// through it instead of `MetricCache`.
#[cfg(feature = "orchestrator")]
pub use run::SweepOrchestratorHandle;
