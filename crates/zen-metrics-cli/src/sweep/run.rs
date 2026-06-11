#![forbid(unsafe_code)]

//! Sweep execution: walk the (image × q × knobs) Cartesian grid, encode
//! each cell, decode-back, score against the source for every selected
//! metric, and write a Pareto TSV.
//!
//! ## Concurrency model
//!
//! - **Outer loop** (over source images) is **serial** by design. Each
//!   image is decoded once into an `Rgb8Image` shared across all of its
//!   cells via `&Rgb8Image`. Holding only one source's pixels in memory
//!   at a time keeps peak RAM at `1× source + N_threads × decoded_cell`,
//!   which is the bound that lets us run on 12-vCPU vast.ai boxes with
//!   modest memory.
//! - **Inner loop** (over `q × knob_tuple` for the current source) is
//!   **parallel** via rayon. Each rayon task encodes the cell, decodes
//!   it back, scores every metric, and emits a row through a `Mutex<csv
//!   Writer>` plus a `Mutex<FeatureParquetWriter>`. Rows land out-of-
//!   order; downstream tools group by `(image_path, q, knob_tuple)` and
//!   don't depend on order.
//! - **Thread budget** is set by `cfg.jobs` (or rayon's default = num
//!   cpus when `jobs = 0`). The setter is `try_init_thread_pool`, called
//!   exactly once per process from `cmd_sweep`.
//!
//! ## RAM
//!
//! At any moment in flight we hold:
//!   - 1 × source `Rgb8Image` (decoded once per image)
//!   - up to `N_threads` × `(encoded_bytes + decoded_cell + metric scratch)`
//!
//! The encoded bytes are short-lived (KB), the decoded cell is a `Vec<u8>`
//! the same size as the source. We deliberately **do not** pre-collect
//! per-cell results into a `Vec<CellOutcome>` before writing — the rayon
//! `for_each` walks one cell at a time per thread and emits immediately.
//!
//! ## Failure isolation
//!
//! A panic or error in one cell only invalidates that row; surrounding
//! cells continue. Stat counters use `AtomicU64`, so multiple parallel
//! failures don't lose increments to torn writes.

use std::collections::hash_map::DefaultHasher;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use crate::decode::{Rgb8Image, decode_image_to_rgb8};
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
use crate::metrics::cache::MetricCache;
use crate::metrics::{GpuRuntime, MetricKind, ZensimFeatureRegime, run_zensim_with_features};
// `run_metric` is the no-GPU-features fall-through AND the gpu-feature-off-
// but-gpu-zensim-on branch inside `compute_cell`. The MetricCache also calls
// it for CPU / disabled metrics. Mark unused to handle the all-GPU-on build
// where the cache replaces every direct call site in this module.
#[allow(unused_imports)]
use crate::metrics::run_metric;
// Re-exported for callers (lib consumers) — not directly called in this
// file once the cache was introduced.
#[allow(unused_imports)]
use crate::metrics::run_zensim_gpu_with_features;
use crate::sweep::encode::EncodedCell;
use crate::sweep::encode::{CodecKind, encode};
use crate::sweep::feature_writer::FeatureParquetWriter;
use crate::sweep::grid::{KnobGrid, KnobTuple};
#[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
use crate::sweep::plan::{BuiltPlan, PlannedCell, build_plan};

/// One unit of sweep work: row identity (`q` + canonical knob JSON)
/// plus how to produce the encoded bytes.
enum SweepUnit<'a> {
    /// Classic `(q, knob-tuple)` cell — per-codec JSON knob dispatch.
    Tuple(f64, KnobTuple),
    /// Plan-driven cell carrying a fully-resolved per-codec config
    /// from the codec's own sweep planner (`sweep::plan`).
    #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
    Planned(&'a PlannedCell),
    /// Unconstructable; keeps the lifetime parameter in play when the
    /// jpeg feature is off.
    #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
    _Never(std::convert::Infallible, std::marker::PhantomData<&'a ()>),
}

impl SweepUnit<'_> {
    fn q(&self) -> f64 {
        match self {
            Self::Tuple(q, _) => *q,
            #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
            Self::Planned(cell) => cell.q,
            #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
            Self::_Never(never, _) => match *never {},
        }
    }

    fn knob_json(&self) -> String {
        match self {
            Self::Tuple(_, tuple) => tuple.to_canonical_json(),
            #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
            Self::Planned(cell) => cell.knob_json.clone(),
            #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
            Self::_Never(never, _) => match *never {},
        }
    }

    fn encode_cell(
        &self,
        codec: CodecKind,
        source: &Rgb8Image,
    ) -> Result<EncodedCell, Box<dyn std::error::Error>> {
        match self {
            Self::Tuple(q, tuple) => encode(codec, source, *q, &tuple.0),
            #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
            Self::Planned(cell) => Ok(cell.config.encode_bytes(source)?),
            #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
            Self::_Never(never, _) => match *never {},
        }
    }
}

/// The classic `q_grid x knob_grid` cross product as sweep units.
fn tuple_units(cfg: &SweepConfig) -> Vec<SweepUnit<'static>> {
    cfg.q_grid
        .iter()
        .flat_map(|&q| {
            cfg.knob_grid
                .iter_tuples()
                .map(move |t| SweepUnit::Tuple(q, t))
        })
        .collect()
}

/// Selector for a plan-driven zenjpeg sweep (see
/// [`SweepConfig::plan`]).
#[derive(Debug, Clone)]
pub struct PlanSpec {
    /// `"rd_core"` or `"modes_full"`.
    pub name: String,
    /// Optional cell budget — zenjpeg's reduction ladder sheds
    /// lowest-priority axis values one at a time and reports every drop
    /// in the plan manifest; nothing is sampled away silently.
    pub budget: Option<usize>,
}

/// Runtime parameters for a sweep invocation.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    pub codec: CodecKind,
    pub sources: Vec<PathBuf>,
    pub q_grid: Vec<f64>,
    pub knob_grid: KnobGrid,
    /// Plan-driven zenjpeg cells: when set (requires
    /// [`CodecKind::Zenjpeg`] and an empty `knob_grid`), cells come from
    /// `zenjpeg::encode::sweep` — the codec's curated, provenance-stamped
    /// axes (`rd_core` / `modes_full`) over `q_grid`, fingerprint-
    /// deduplicated, validity-filtered, and ordered main-effects-first —
    /// instead of the `knob_grid` Cartesian product. The plan's audit
    /// manifest (alias merges, invalid strata, budget drops) is written
    /// to `<output>.plan.json`. Requires `--features sweep,jpeg`.
    pub plan: Option<PlanSpec>,
    pub metrics: Vec<MetricKind>,
    pub gpu_runtime: GpuRuntime,
    pub output: PathBuf,
    /// When set, every cell that runs the [`MetricKind::Zensim`] (CPU)
    /// **or** [`MetricKind::ZensimGpu`] metric also persists its
    /// regime-appropriate feature vector to a parquet sidecar at this
    /// path. Joins back to `output` (TSV) by
    /// `(image_path, codec, q, knob_tuple_json)`.
    ///
    /// The vector length is `feature_regime.total_features()`
    /// (228 / 300 / 372). CPU zensim always emits 300 features (its
    /// `compute_extended_features` returns the Extended block); the GPU
    /// path honours [`Self::feature_regime`].
    ///
    /// Cells that do not run any zensim variant emit nothing to the
    /// parquet. If the metric list contains neither variant, the parquet
    /// file is created but receives no rows; we don't auto-add zensim
    /// because callers may have explicit reasons for the metric set
    /// they passed.
    pub feature_output: Option<PathBuf>,
    /// Feature regime for the **GPU** zensim feature path. Defaults to
    /// [`ZensimFeatureRegime::WithIw`] (372) — the v26+ training schema.
    /// Ignored when GPU zensim is not in the metric set OR
    /// `feature_output` is None. CPU zensim always emits 300 features
    /// regardless of this flag (its CPU API doesn't expose the regime
    /// knob).
    pub feature_regime: ZensimFeatureRegime,
    /// When set, every successfully decoded cell writes its
    /// **distorted** image (the result of encode + decode-back) as
    /// PNG into this directory. Filenames are
    /// `<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.png`
    /// — deterministic per cell so reruns overwrite the same files
    /// rather than producing duplicates.
    ///
    /// Pairs with `pairs_tsv` to feed external scorers (e.g. the
    /// pycvvdp worker at `scripts/sweep/pycvvdp_worker.py`) that
    /// need on-disk `(ref, dist)` pairs. See
    /// `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`.
    pub distorted_out_dir: Option<PathBuf>,
    /// When set, every successfully encoded cell writes the **encoded
    /// codec bytes** (the actual .jpg / .webp / .avif / .jxl / .png the
    /// codec produced) into this directory. Filenames are
    /// `<src_stem>_<src_path_hash16>_<codec>_q<q>_<knob_hash16>.<ext>`
    /// — same hash scheme as `distorted_out_dir` so a row in the
    /// output TSV can address both the decoded PNG and the encoded
    /// blob from the same `(src_path, codec, q, knob_json)` tuple.
    ///
    /// Pairs with the new `encoded_filename` column added to the
    /// output TSV. Designed for sweeps that intend to upload encoded
    /// variants to R2 once and reuse them across N future metric
    /// backfills (skip the encode step in subsequent runs).
    pub encoded_out_dir: Option<PathBuf>,
    /// When set, a parallel TSV is written with the columns
    /// `image_path  codec  q  knob_tuple_json  ref_path  dist_path`
    /// — one row per successfully-decoded cell, mirroring the main
    /// `output` TSV's identity tuple. `ref_path` is the source
    /// image; `dist_path` is the cell's distorted PNG (empty when
    /// `distorted_out_dir` is not set).
    pub pairs_tsv: Option<PathBuf>,
    /// Number of CPU threads for the per-image inner cell loop. `0`
    /// defers to rayon's default (one per logical core). `1` runs cells
    /// serially, useful for debugging.
    pub jobs: usize,
}

/// Initialise the global rayon thread pool. Safe to call multiple times
/// — the first call wins. Returns `Ok` regardless because subsequent
/// initialisations from the same process are a no-op for rayon.
pub fn try_init_thread_pool(jobs: usize) -> Result<(), Box<dyn Error>> {
    // `build_global` errors if already-initialised; we silently swallow
    // because the harness is run as a one-shot binary — nobody else has
    // initialised the global pool.
    //
    // 32 MB worker stacks are mandatory, not a tuning choice: zenavif's
    // engine (zenrav1e) recurses deeply in partition RDO and overflows
    // rayon's default 2 MB workers (observed SIGABRT at 256², speed 2).
    // The cost is reserved address space, not resident memory.
    let mut builder = rayon::ThreadPoolBuilder::new().stack_size(32 * 1024 * 1024);
    if jobs > 0 {
        builder = builder.num_threads(jobs);
    }
    let _ = builder.build_global();
    Ok(())
}

/// Phase 7.5: score one (ref, dist) pair via the shared
/// orchestrator handle. Returns `(column_name, value)` tuples
/// matching the shape the legacy `MetricCache::run_metric_cached`
/// produces, so the sweep loop's row-writer doesn't branch.
///
/// Column names are determined by
/// [`crate::orchestrator_runner::rekey_orchestrator_columns`] —
/// which maps from the orchestrator's canonical GPU-variant keys
/// back to the CLI-variant-specific names (e.g. `ssim2_gpu` stays
/// the same for `Ssim2Gpu`; `ssim2_gpu` becomes `ssim2` for
/// `Ssim2`). The column names are leaked to `&'static str` to
/// match the existing row-builder signature; the leak is bounded
/// by metric-count × cells × variants (single-digit) and the
/// process exits shortly after.
///
/// The orchestrator's `run_single` is synchronous and blocks the
/// rayon worker thread for the duration of the GPU dispatch. The
/// `Mutex<Orchestrator>` lock contention is bounded by GPU compute
/// time per call — the same shape as the legacy
/// `MetricCache::lock_global` path.
#[cfg(feature = "orchestrator")]
fn score_via_orchestrator(
    orch: &SweepOrchestratorHandle,
    cli_metric: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn Error>> {
    use crate::orchestrator_glue::OrchestratorMetricSpec;
    use crate::orchestrator_runner::rekey_orchestrator_columns;
    use zenmetrics_orchestrator::{Task, TaskData};

    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
            cli_metric.name(),
            reference.width,
            reference.height,
            distorted.width,
            distorted.height
        )
        .into());
    }

    let spec = OrchestratorMetricSpec::from_cli(cli_metric);
    let task = Task {
        task_id: 1,
        ref_data: TaskData::Srgb8(reference.pixels.clone()),
        dist_data: TaskData::Srgb8(distorted.pixels.clone()),
        width: reference.width,
        height: reference.height,
        metric: spec.kind,
        params: None,
        ref_hash: 0,
    };
    let result = {
        let mut g = orch.lock().expect("orchestrator handle poisoned");
        g.run_single(task)
    };

    match result.outcome {
        Ok(_score) => {
            let rekeyed = rekey_orchestrator_columns(cli_metric, &result.output_columns);
            // Leak Strings to &'static str to match the
            // legacy row-builder's column-name lifetime.
            let leaked: Vec<(&'static str, f64)> = rekeyed
                .into_iter()
                .map(|(k, v)| (String::leak(k) as &'static str, v))
                .collect();
            Ok(leaked)
        }
        Err(e) => Err(format!(
            "orchestrator: {} ({e}; backends_attempted={:?})",
            cli_metric.name(),
            result.backends_attempted
        )
        .into()),
    }
}

/// Optional handle to an orchestrator-driven scoring backend. When
/// `Some`, the sweep per-cell loop dispatches every metric through
/// the orchestrator instead of the legacy [`MetricCache`]. The two
/// paths produce **bit-identical parquet sidecars** for the same
/// inputs — see `crates/zenmetrics-orchestrator/src/executor.rs::build_output_columns`.
///
/// **Critical: when this is `Some`, MetricCache MUST NOT be used in
/// the same process.** Both maintain warm cubecl `Metric` instances;
/// activating both simultaneously double-allocates GPU buffers and
/// triggers cubecl-cuda's pool to saturate ~2× faster. The sweep
/// loop's branching enforces this: orchestrator-on means
/// `cache.run_metric_cached(...)` is never called.
///
/// The handle is shared across rayon threads via the wrapped
/// `Arc<Mutex<...>>` — GPU dispatch is serialised by the driver
/// anyway, so the lock contention is bounded by GPU compute time
/// (the same shape as the legacy `MetricCache::lock_global` path).
#[cfg(feature = "orchestrator")]
pub type SweepOrchestratorHandle =
    std::sync::Arc<std::sync::Mutex<zenmetrics_orchestrator::Orchestrator>>;

/// Drive the sweep end-to-end. Outer loop over sources is serial (one
/// decoded image in memory at a time); inner cell loop is parallel via
/// rayon, with row writes funnelled through a `Mutex`.
///
/// `orch` is an optional orchestrator handle (Phase 7.5). When
/// `Some`, every cell's metric scoring routes through
/// `Orchestrator::run_single` instead of the legacy `MetricCache`
/// (which would otherwise warm a separate set of cubecl instances).
/// `None` keeps the legacy code path (the production default until
/// Phase 7.6 flips it).
pub fn run_sweep(
    cfg: &SweepConfig,
    #[cfg(feature = "orchestrator")] orch: Option<&SweepOrchestratorHandle>,
) -> Result<SweepStats, Box<dyn Error>> {
    // Honour the configured thread budget. No-op if the rayon global
    // pool is already initialised; first call wins.
    try_init_thread_pool(cfg.jobs)?;

    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&cfg.output)?;
    write_header(&mut wtr, &cfg.metrics)?;

    let zensim_in_metrics = cfg.metrics.contains(&MetricKind::Zensim);
    let zensim_gpu_in_metrics = cfg.metrics.contains(&MetricKind::ZensimGpu);
    // Feature-vector width depends on which path is producing them.
    // CPU zensim's `compute_extended_features` returns 300 floats; GPU
    // honours `feature_regime`. When both are in the metric set we
    // prefer the GPU regime (matches the v26+ schema) and CPU zensim's
    // extra emit (if any) is ignored — only one variant is allowed to
    // write per cell to keep the sidecar's `feat_*` count consistent.
    // CPU zensim's `score_with_features` runs `ZensimProfile::latest()`,
    // whose extended-feature block now carries the IW features (372), not
    // the legacy 300. Size the writer from the active regime for BOTH the
    // CPU and GPU paths (default WithIw = 372). If a future profile's CPU
    // emit ever diverges from the regime count, the per-cell push errors
    // loudly rather than silently writing a mis-shaped sidecar.
    let _ = zensim_gpu_in_metrics;
    let feature_n: usize = cfg.feature_regime.total_features();
    let feature_writer_inner = match &cfg.feature_output {
        Some(path) => Some(FeatureParquetWriter::create_with_n(path, feature_n)?),
        None => None,
    };

    // Optional pairs TSV — header row first so failures partway
    // through a sweep still leave a parseable file.
    let pairs_writer_inner = match &cfg.pairs_tsv {
        Some(path) => {
            let mut w = csv::WriterBuilder::new().delimiter(b'\t').from_path(path)?;
            w.write_record([
                "image_path",
                "codec",
                "q",
                "knob_tuple_json",
                "ref_path",
                "dist_path",
            ])?;
            Some(w)
        }
        None => None,
    };

    // Plan-driven zenjpeg cells (zenjpeg::encode::sweep): validated and
    // built ONCE per sweep — cells are image-independent — with the
    // audit manifest written next to the TSV before any encode runs.
    #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
    let built_plan: Option<BuiltPlan> = match &cfg.plan {
        Some(spec) => {
            if !cfg.knob_grid.axes.is_empty() {
                return Err("--plan and --knob-grid are mutually exclusive".into());
            }
            // Codec dispatch (and feature availability) live in
            // sweep::plan::build_plan — unsupported codecs error there.
            let built = build_plan(cfg.codec, &spec.name, spec.budget, &cfg.q_grid)?;
            let manifest_path = cfg.output.with_extension("plan.json");
            std::fs::write(&manifest_path, &built.manifest_json)?;
            eprintln!(
                "[sweep] plan {}: {} cells/image (manifest: {})",
                spec.name,
                built.cells.len(),
                manifest_path.display()
            );
            Some(built)
        }
        None => None,
    };
    #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
    if cfg.plan.is_some() {
        return Err(
            "plan-driven sweeps require building with --features sweep and the codec feature (jpeg/avif)"
                .into(),
        );
    }

    #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
    let cells_per_image: usize = built_plan
        .as_ref()
        .map(|p| p.cells.len())
        .unwrap_or_else(|| cfg.q_grid.len() * cfg.knob_grid.cell_count());
    #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
    let cells_per_image: usize = cfg.q_grid.len() * cfg.knob_grid.cell_count();

    let cells_total = (cfg.sources.len() * cells_per_image) as u64;
    let stats = AtomicSweepStats::new(cells_total);

    // Wrap writers in Mutex so rayon tasks can flush rows under a lock.
    // Lock contention is dominated by encode/decode/score work — order
    // of milliseconds — so the critical section is tiny in comparison.
    let wtr = Mutex::new(wtr);
    let feature_writer = Mutex::new(feature_writer_inner);
    let pairs_writer = Mutex::new(pairs_writer_inner);

    // GPU metric instance cache — **process-static**. The cache
    // outlives individual `run_sweep` calls so cached `Metric`
    // instances are reused across all calls (groups within a chunk,
    // chunks within a worker process). This is the only point at
    // which the cubecl-cuda persist-plane footprint stays bounded:
    // a local cache per `run_sweep` would still drop 6 metrics ×
    // ~200 MB / instance back to the pool between groups, and
    // cubecl-cuda's pool does not promptly return those pages to
    // the driver — after ~4 groups the 12 GB RTX 3060 OOMs on
    // tiny (12 MB) follow-on allocations. The first refactor
    // (commit 11d374dd) hoisted per-cell — necessary but not
    // sufficient; this commit hoists per-process. See
    // `metrics::cache::MetricCache::global` for the OnceLock
    // construction.
    //
    // The lock is taken only for GPU score calls; encode +
    // decode-back run in parallel as before. Poisoned-lock
    // recovery is in `MetricCache::lock_global` — a panic inside
    // one cell (e.g. cubecl OOM bubbling up through the metric
    // crate) used to cascade into every subsequent cell on the
    // same source.
    #[cfg(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    ))]
    let gpu_runtime_for_cache = cfg.gpu_runtime;

    // SWEEP_CLEANUP_BETWEEN_SOURCES — opt-in cubecl pool flush hint.
    // Default OFF because with PC>1 (multiple chunks sharing the
    // global cubecl client) the cleanup_all flush of one chunk's
    // sources races with another chunk's in-flight kernel calls.
    // Observed failure mode on v26 24 GB smoke: thread 'DSD-0-0'
    // panicked at cubecl-cuda/src/compute/stream.rs:101 with
    // "Memory page 0 doesn't exist" once AIMD ramped PC past 1 +
    // cleanup_all fired between sources. The chunk-cap respawn
    // (MAX_CHUNKS_PER_PROCESS, default 20) resets the pool to zero
    // every N chunks, which is safer than the in-process cleanup.
    //
    // Set SWEEP_CLEANUP_BETWEEN_SOURCES=1 only when running PC=1
    // (e.g. single-instance smokes that need to bound pool footprint
    // on 12 GB cards without respawning).
    #[cfg(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    ))]
    let cleanup_between_sources = std::env::var("SWEEP_CLEANUP_BETWEEN_SOURCES")
        .ok()
        .as_deref()
        .map(|s| matches!(s.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);

    #[cfg_attr(
        not(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    for (src_idx, src_path) in cfg.sources.iter().enumerate() {
        #[cfg(any(
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        ))]
        if src_idx > 0 && cleanup_between_sources {
            // Skip the MetricCache flush when the orchestrator drives
            // scoring — MetricCache isn't touched in that mode, so
            // nothing to flush. Touching it here would just wake up
            // the OnceLock for nothing.
            #[cfg(feature = "orchestrator")]
            let skip_cache = orch.is_some();
            #[cfg(not(feature = "orchestrator"))]
            let skip_cache = false;
            if !skip_cache {
                let mut cache = MetricCache::lock_global(cfg.gpu_runtime);
                let _ = cache.cleanup_all();
            }
        }
        // Decode the source once per image so we don't re-PNG-decode for
        // every cell. The bytes are freed when we move to the next image
        // (drops at the end of this loop iteration). This is the entire
        // RAM-discipline knob: one source resident at a time.
        let source = match decode_image_to_rgb8(src_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!(
                    "[sweep] skipping {} (decode failed: {e})",
                    src_path.display()
                );
                stats.add_failed_decode(cells_per_image as u64);
                continue;
            }
        };

        // Build a flat list of sweep units for rayon to walk. Cell count
        // is bounded (≤ a few thousand per image typically), so the Vec
        // is cheap. Plan-driven cells borrow from `built_plan` (built
        // once above); tuple cells own their small knob maps. We don't
        // clone the source either way.
        #[cfg(all(feature = "sweep", any(feature = "jpeg", feature = "avif")))]
        let units: Vec<SweepUnit<'_>> = match &built_plan {
            Some(p) => p.cells.iter().map(SweepUnit::Planned).collect(),
            None => tuple_units(cfg),
        };
        #[cfg(not(all(feature = "sweep", any(feature = "jpeg", feature = "avif"))))]
        let units: Vec<SweepUnit<'_>> = tuple_units(cfg);

        units.par_iter().for_each(|unit| {
            // Wrap each cell in catch_unwind so a panic in encode/decode/
            // metric scoring doesn't abort sibling cells. Without this,
            // a single bad knob combo would tear down a chunk's worth of
            // good rows mid-flight.
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute_cell(
                    cfg,
                    src_path,
                    &source,
                    unit,
                    zensim_in_metrics,
                    zensim_gpu_in_metrics,
                    #[cfg(any(
                        feature = "gpu-butteraugli",
                        feature = "gpu-ssim2",
                        feature = "gpu-dssim",
                        feature = "gpu-iwssim",
                        feature = "gpu-zensim",
                        feature = "gpu-cvvdp"
                    ))]
                    gpu_runtime_for_cache,
                    #[cfg(feature = "orchestrator")]
                    orch,
                )
            }))
            .unwrap_or_else(|panic_payload| {
                let msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<non-string panic payload>".to_string()
                };
                eprintln!(
                    "[sweep] cell panicked: {} q={} knobs={}: {msg}",
                    src_path.display(),
                    unit.q(),
                    unit.knob_json(),
                );
                // Emit a row with blank score columns so the panic is
                // visible in downstream tooling (rather than silently
                // dropped). Treat as a decode failure for stat-counting.
                let mut row: Vec<String> = vec![
                    src_path.display().to_string(),
                    cfg.codec.name().to_string(),
                    unit.q().to_string(),
                    unit.knob_json(),
                    "".to_string(), // encoded_bytes
                    "".to_string(), // encode_ms
                    "".to_string(), // decode_ms
                ];
                for m in &cfg.metrics {
                    for _ in m.column_names() {
                        row.push("".to_string());
                    }
                }
                CellOutcome::DecodeFailed { row }
            });
            // Emit row + feature row + update stats. The `wtr` lock is
            // held for the duration of one TSV record; `feature_writer`
            // for one parquet push.
            match outcome {
                CellOutcome::Ok {
                    row,
                    feature,
                    pair_row,
                    score_failed,
                } => {
                    if let Ok(mut w) = wtr.lock() {
                        if w.write_record(&row).is_ok() {
                            stats.add_emitted();
                        } else {
                            eprintln!("[sweep] write_record failed");
                        }
                    }
                    if let Some((image, codec, q_, knob_json, score, features)) = feature
                        && let Ok(mut fw_guard) = feature_writer.lock()
                        && let Some(fw) = fw_guard.as_mut()
                        && let Err(e) = fw.push_row(&image, codec, q_, &knob_json, score, &features)
                    {
                        eprintln!("[sweep] feature_writer push failed: {} q={q_}: {e}", image,);
                    }
                    if let Some(pair) = pair_row
                        && let Ok(mut pw_guard) = pairs_writer.lock()
                        && let Some(pw) = pw_guard.as_mut()
                        && let Err(e) = pw.write_record(&pair)
                    {
                        eprintln!(
                            "[sweep] pairs writer failed: {} q={}: {e}",
                            pair[0],
                            unit.q(),
                        );
                    }
                    if score_failed {
                        stats.add_failed_score();
                    }
                }
                CellOutcome::EncodeFailed { row } => {
                    if let Ok(mut w) = wtr.lock() {
                        let _ = w.write_record(&row);
                    }
                    stats.add_failed_encode();
                }
                CellOutcome::DecodeFailed { row } => {
                    if let Ok(mut w) = wtr.lock() {
                        let _ = w.write_record(&row);
                    }
                    stats.add_failed_decode(1);
                }
            }
        });
    }

    // Drop locks and finalize.
    let mut wtr = wtr
        .into_inner()
        .map_err(|e| format!("wtr lock poisoned: {e}"))?;
    wtr.flush()?;
    if let Some(fw) = feature_writer
        .into_inner()
        .map_err(|e| format!("feature_writer lock poisoned: {e}"))?
    {
        fw.finish()?;
    }
    if let Some(mut pw) = pairs_writer
        .into_inner()
        .map_err(|e| format!("pairs_writer lock poisoned: {e}"))?
    {
        pw.flush()?;
    }
    Ok(stats.snapshot())
}

/// Feature-sidecar row payload:
/// `(image_path, codec, q, knob_json, zensim_score, features)`.
type FeatureRow = (String, &'static str, f64, String, f32, Vec<f64>);

/// Per-cell outcome — pure result type, written to disk by the caller.
// One transient value per cell, matched immediately by the emit loop —
// boxing the large Ok payload would buy nothing but an allocation.
#[allow(clippy::large_enum_variant)]
enum CellOutcome {
    Ok {
        row: Vec<String>,
        feature: Option<FeatureRow>,
        /// `[image_path, codec, q, knob_tuple_json, ref_path, dist_path]`
        /// — emitted when the sweep config requests a pairs TSV. The
        /// outer loop writes this row to the pairs writer.
        pair_row: Option<[String; 6]>,
        score_failed: bool,
    },
    EncodeFailed {
        row: Vec<String>,
    },
    DecodeFailed {
        row: Vec<String>,
    },
}

/// Pure per-cell compute — no shared mutable state. Allocations:
///   - encoded bytes (small, dropped before scoring returns)
///   - decoded `Rgb8Image` (≈ source dimensions × 3 bytes; held during
///     metric scoring then dropped)
///   - row `Vec<String>` (small)
///   - optional feature `Vec<f64>` (228 / 300 / 372 entries when a
///     zensim variant is in the metric set and `feature_output` is on)
fn compute_cell(
    cfg: &SweepConfig,
    src_path: &Path,
    source: &Rgb8Image,
    unit: &SweepUnit<'_>,
    zensim_in_metrics: bool,
    zensim_gpu_in_metrics: bool,
    #[cfg(any(
        feature = "gpu-butteraugli",
        feature = "gpu-ssim2",
        feature = "gpu-dssim",
        feature = "gpu-iwssim",
        feature = "gpu-zensim",
        feature = "gpu-cvvdp"
    ))]
    gpu_runtime_for_cache: GpuRuntime,
    #[cfg(feature = "orchestrator")] orch: Option<&SweepOrchestratorHandle>,
) -> CellOutcome {
    let q = unit.q();
    let knob_json = unit.knob_json();
    let mut row: Vec<String> = vec![
        src_path.display().to_string(),
        cfg.codec.name().to_string(),
        q.to_string(),
        knob_json.clone(),
    ];

    // Encode.
    let cell = match unit.encode_cell(cfg.codec, source) {
        Ok(c) => c,
        Err(e) => {
            eprintln!(
                "[sweep] encode failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            row.push("".to_string()); // encoded_bytes
            row.push("".to_string()); // encode_ms
            row.push("".to_string()); // encoded_filename
            row.push("".to_string()); // decode_ms
            for m in &cfg.metrics {
                for _ in m.column_names() {
                    row.push("".to_string());
                }
            }
            return CellOutcome::EncodeFailed { row };
        }
    };

    row.push(cell.bytes.len().to_string());
    row.push(format!("{:.3}", cell.encode_ms));

    // Optionally persist the encoded codec bytes. The filename matches
    // the same `<stem>_<src_hash>_<codec>_q<q>_<knob_hash>.<ext>` scheme
    // as save_distorted_png so an external tool can pair them up by
    // identity tuple alone. Failure to write demotes the encoded_filename
    // column to empty — the score columns are still valid.
    let encoded_filename = match &cfg.encoded_out_dir {
        Some(dir) => save_encoded_variant(dir, src_path, cfg.codec, q, &knob_json, &cell.bytes)
            .unwrap_or_else(|e| {
                eprintln!(
                    "[sweep] save encoded variant failed: {} q={q}: {e}",
                    src_path.display(),
                );
                String::new()
            }),
        None => String::new(),
    };
    row.push(encoded_filename);

    // Decode-back through the path-based decoder for format-sniff parity
    // with production. Tempfile lifetime ends when this function returns.
    let decode_start = Instant::now();
    let decoded = match decode_encoded_bytes(&cell.bytes, cfg.codec) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "[sweep] decode-back failed: {} q={q} knobs={knob_json}: {e}",
                src_path.display()
            );
            row.push("".to_string()); // decode_ms
            for m in &cfg.metrics {
                for _ in m.column_names() {
                    row.push("".to_string());
                }
            }
            return CellOutcome::DecodeFailed { row };
        }
    };
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    row.push(format!("{decode_ms:.3}"));

    // Dimension check — skip scoring on size mismatch (chroma upsampling
    // bug or wrong pixel-format conversion).
    if decoded.width != source.width || decoded.height != source.height {
        eprintln!(
            "[sweep] dimension mismatch: {} q={q} src={}x{} decoded={}x{}",
            src_path.display(),
            source.width,
            source.height,
            decoded.width,
            decoded.height,
        );
        for m in &cfg.metrics {
            for _ in m.column_names() {
                row.push("".to_string());
            }
        }
        return CellOutcome::DecodeFailed { row };
    }

    // Score every selected metric.
    let mut any_score_failed = false;
    // GPU zensim takes precedence when both variants are in the
    // metric set — its `feature_regime` matches the v26+ schema
    // (typically 372). CPU zensim's `compute_extended_features`
    // emits 300, so collecting features from both at once would
    // require two distinct sidecar widths (we don't support that;
    // ship one).
    let want_features_gpu = cfg.feature_output.is_some() && zensim_gpu_in_metrics;
    let want_features_cpu =
        cfg.feature_output.is_some() && zensim_in_metrics && !zensim_gpu_in_metrics;
    let mut zensim_features: Option<(f32, Vec<f64>)> = None;

    // Phase 7.5 routing: when an orchestrator handle is provided,
    // dispatch every metric through it INSTEAD of MetricCache to
    // avoid double-allocating warm cubecl instances on the GPU.
    // The exception is zensim with feature-vector emit: the
    // orchestrator's API doesn't yet expose
    // `compute_features_srgb_u8`, so a sweep that needs the
    // sidecar parquet of zensim features still falls back to
    // MetricCache. This is documented in `INTEGRATION_NOTES.md`
    // as a Phase 7.5+ enhancement.
    #[cfg(feature = "orchestrator")]
    let use_orch_for_cell = orch.is_some();
    #[cfg(not(feature = "orchestrator"))]
    let use_orch_for_cell = false;

    for &metric in &cfg.metrics {
        let result: Result<Vec<(&'static str, f64)>, Box<dyn Error>> = if metric
            == MetricKind::Zensim
            && want_features_cpu
        {
            // CPU zensim — does not pressure the cubecl pool;
            // keep the uncached per-call path.
            match run_zensim_with_features(source, &decoded) {
                Ok((score, features)) => {
                    zensim_features = Some((score as f32, features));
                    Ok(vec![("zensim", score)])
                }
                Err(e) => Err(e),
            }
        } else if use_orch_for_cell && !(metric == MetricKind::ZensimGpu && want_features_gpu) && {
            #[cfg(feature = "orchestrator")]
            {
                crate::orchestrator_runner::metric_orchestrator_eligible(metric)
            }
            #[cfg(not(feature = "orchestrator"))]
            {
                false
            }
        } {
            // Phase 7.5: orchestrator-driven scoring. Skipped for
            // ZensimGpu+want_features_gpu because the orchestrator
            // doesn't yet expose feature emission; that branch
            // still uses MetricCache below. Phase 7.7.1: also
            // skipped for `Butteraugli` / `ButteraugliGpu`
            // because the orchestrator's strip-preferred Auto
            // resolver picks single-resolution scoring which
            // diverges from the legacy CLI's
            // `Butteraugli::new_multires`-always output by
            // ~14-30 %. See
            // `crate::orchestrator_runner::metric_orchestrator_eligible`.
            #[cfg(feature = "orchestrator")]
            {
                score_via_orchestrator(
                    orch.expect("use_orch_for_cell true => orch Some"),
                    metric,
                    source,
                    &decoded,
                )
            }
            #[cfg(not(feature = "orchestrator"))]
            {
                // Unreachable — use_orch_for_cell is `false` when
                // the feature is off.
                run_metric(metric, source, &decoded, cfg.gpu_runtime)
            }
        } else if metric == MetricKind::ZensimGpu && want_features_gpu {
            // GPU zensim with feature emit — go through the cache
            // so the WithIw persist planes (~200 MB at 1080p) are
            // allocated once per (dims, regime) instead of per
            // cell. Without this, repeated cell-level construction
            // saturates cubecl-cuda's pool on the 12 GB RTX 3060
            // after ~80 cells. See `metrics::cache` module docs.
            #[cfg(feature = "gpu-zensim")]
            {
                let mut cache = MetricCache::lock_global(gpu_runtime_for_cache);
                match cache.compute_zensim_features(source, &decoded, cfg.feature_regime) {
                    Ok((score, features)) => {
                        zensim_features = Some((score as f32, features));
                        Ok(vec![("zensim_gpu", score)])
                    }
                    Err(e) => Err(e),
                }
            }
            #[cfg(not(feature = "gpu-zensim"))]
            {
                run_metric(metric, source, &decoded, cfg.gpu_runtime)
            }
        } else {
            // All other GPU metrics route through the cache; CPU
            // metrics (and unknown / disabled GPU metrics) fall
            // through to the uncached `run_metric` path inside
            // `run_metric_cached`.
            #[cfg(any(
                feature = "gpu-butteraugli",
                feature = "gpu-ssim2",
                feature = "gpu-dssim",
                feature = "gpu-iwssim",
                feature = "gpu-zensim",
                feature = "gpu-cvvdp"
            ))]
            {
                let mut cache = MetricCache::lock_global(gpu_runtime_for_cache);
                cache.run_metric_cached(metric, source, &decoded)
            }
            #[cfg(not(any(
                feature = "gpu-butteraugli",
                feature = "gpu-ssim2",
                feature = "gpu-dssim",
                feature = "gpu-iwssim",
                feature = "gpu-zensim",
                feature = "gpu-cvvdp"
            )))]
            {
                run_metric(metric, source, &decoded, cfg.gpu_runtime)
            }
        };
        match result {
            Ok(values) => {
                for (_, v) in &values {
                    row.push(format!("{v:.6}"));
                }
            }
            Err(e) => {
                eprintln!(
                    "[sweep] metric {} failed on {} q={q}: {e}",
                    metric.name(),
                    src_path.display()
                );
                for _ in metric.column_names() {
                    row.push("".to_string());
                }
                any_score_failed = true;
            }
        }
    }

    let feature = zensim_features.map(|(score, features)| {
        (
            src_path.display().to_string(),
            cfg.codec.name(),
            q,
            knob_json.clone(),
            score,
            features,
        )
    });

    // Emit the pair row + (optionally) save the distorted PNG when
    // requested. Failure to write the PNG demotes the pair_row to
    // an empty dist_path so downstream tooling can spot it; we don't
    // hard-fail the whole cell because the score columns are still
    // valid.
    let pair_row = if cfg.pairs_tsv.is_some() {
        let dist_path = match &cfg.distorted_out_dir {
            Some(dir) => {
                save_distorted_png(dir, src_path, cfg.codec.name(), q, &knob_json, &decoded)
                    .unwrap_or_else(|e| {
                        eprintln!(
                            "[sweep] save distorted PNG failed: {} q={q}: {e}",
                            src_path.display(),
                        );
                        String::new()
                    })
            }
            None => String::new(),
        };
        Some([
            src_path.display().to_string(),
            cfg.codec.name().to_string(),
            q.to_string(),
            knob_json.clone(),
            src_path.display().to_string(),
            dist_path,
        ])
    } else {
        None
    };

    CellOutcome::Ok {
        row,
        feature,
        pair_row,
        score_failed: any_score_failed,
    }
}

/// Hash a value to a fixed-width hex string. Used to produce
/// collision-resistant filename fragments for `(src_path, knobs)`
/// without dragging in a cryptographic dep.
fn hex_hash16<H: Hash + ?Sized>(value: &H) -> String {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Write `decoded` as a fast-effort PNG into `dir` at the canonical
/// filename. Returns the saved path as a string for the pair-row.
fn save_distorted_png(
    dir: &Path,
    src_path: &Path,
    codec_name: &str,
    q: f64,
    knob_json: &str,
    decoded: &Rgb8Image,
) -> Result<String, Box<dyn Error>> {
    #[cfg(feature = "png")]
    {
        use enough::Unstoppable;
        use imgref::ImgRef;
        use zenpng::{Compression, EncodeConfig};

        std::fs::create_dir_all(dir)?;

        let stem = src_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("source");
        let src_hash = hex_hash16(src_path.to_string_lossy().as_ref());
        let knob_hash = hex_hash16(knob_json);
        let filename = format!("{stem}_{src_hash}_{codec_name}_q{q}_{knob_hash}.png");
        let out_path = dir.join(&filename);

        // Fastest effort — these PNGs feed scorers, not end users.
        let cfg = EncodeConfig::default().with_compression(Compression::Fastest);

        use rgb::FromSlice;
        let pixels: &[rgb::Rgb<u8>] = decoded.pixels.as_rgb();
        let img = ImgRef::new(pixels, decoded.width as usize, decoded.height as usize);

        let bytes = zenpng::encode_rgb8(img, None, &cfg, &Unstoppable, &Unstoppable)
            .map_err(|e| format!("zenpng encode failed: {e}"))?;
        std::fs::write(&out_path, bytes)?;
        Ok(out_path.display().to_string())
    }
    #[cfg(not(feature = "png"))]
    {
        let _ = (dir, src_path, codec_name, q, knob_json, decoded);
        Err("distorted-out-dir requires the `png` feature to be enabled".into())
    }
}

/// Write the raw encoded codec bytes into `dir` using the same hash
/// scheme as `save_distorted_png`. Returns the saved file's basename
/// (relative to `dir`) so the per-cell row can record an addressable
/// reference without leaking the absolute path.
fn save_encoded_variant(
    dir: &Path,
    src_path: &Path,
    codec: CodecKind,
    q: f64,
    knob_json: &str,
    bytes: &[u8],
) -> Result<String, Box<dyn Error>> {
    std::fs::create_dir_all(dir)?;
    let stem = src_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("source");
    let src_hash = hex_hash16(src_path.to_string_lossy().as_ref());
    let knob_hash = hex_hash16(knob_json);
    let ext = match codec {
        CodecKind::Zenpng => "png",
        CodecKind::Zenjpeg => "jpg",
        CodecKind::Zenwebp => "webp",
        CodecKind::Zenavif => "avif",
        CodecKind::Zenjxl => "jxl",
    };
    let codec_name = codec.name();
    let filename = format!("{stem}_{src_hash}_{codec_name}_q{q}_{knob_hash}.{ext}");
    let out_path = dir.join(&filename);
    std::fs::write(&out_path, bytes)?;
    Ok(filename)
}

fn decode_encoded_bytes(bytes: &[u8], codec: CodecKind) -> Result<Rgb8Image, Box<dyn Error>> {
    // Path-based decode_image_to_rgb8 sniffs format and dispatches through
    // the per-codec decoder. We write to a tempfile to reuse it unchanged.
    // Performance: write+read is on the order of microseconds for
    // typical encoded sizes (10 KB - 1 MB) and dominates neither encode
    // nor decode wall time, so we don't optimize this away.
    let suffix = match codec {
        CodecKind::Zenpng => ".png",
        CodecKind::Zenjpeg => ".jpg",
        CodecKind::Zenwebp => ".webp",
        CodecKind::Zenavif => ".avif",
        CodecKind::Zenjxl => ".jxl",
    };
    let tmp = tempfile::Builder::new()
        .prefix("zen-metrics-sweep-")
        .suffix(suffix)
        .tempfile()?;
    std::fs::write(tmp.path(), bytes)?;
    decode_image_to_rgb8(tmp.path())
}

fn write_header(
    wtr: &mut csv::Writer<std::fs::File>,
    metrics: &[MetricKind],
) -> Result<(), Box<dyn Error>> {
    let mut headers: Vec<String> = vec![
        "image_path".to_string(),
        "codec".to_string(),
        "q".to_string(),
        "knob_tuple_json".to_string(),
        "encoded_bytes".to_string(),
        "encode_ms".to_string(),
        "encoded_filename".to_string(),
        "decode_ms".to_string(),
    ];
    // Each metric expands to one column per name in `column_names()`. For
    // most metrics that's a single column; butteraugli (CPU and GPU) emits
    // two — `butteraugli_max{,_gpu}` + `butteraugli_pnorm3{,_gpu}`. The
    // sweep TSV prefixes every score column with `score_` to disambiguate
    // from later columns the harness may add (per-cell timings, etc.).
    for m in metrics {
        for col in m.column_names() {
            headers.push(format!("score_{col}"));
        }
    }
    wtr.write_record(&headers)?;
    Ok(())
}

/// Aggregate counters from a sweep run. Useful for the EOM report.
#[derive(Debug, Clone, Copy, Default)]
pub struct SweepStats {
    pub cells_total: usize,
    pub cells_emitted: usize,
    pub cells_failed_encode: usize,
    pub cells_failed_decode: usize,
    pub cells_failed_score: usize,
}

/// Atomic counters used during the parallel sweep. Snapshotted into a
/// plain `SweepStats` once the sweep finishes.
struct AtomicSweepStats {
    cells_total: u64,
    cells_emitted: AtomicU64,
    cells_failed_encode: AtomicU64,
    cells_failed_decode: AtomicU64,
    cells_failed_score: AtomicU64,
}

impl AtomicSweepStats {
    fn new(cells_total: u64) -> Self {
        Self {
            cells_total,
            cells_emitted: AtomicU64::new(0),
            cells_failed_encode: AtomicU64::new(0),
            cells_failed_decode: AtomicU64::new(0),
            cells_failed_score: AtomicU64::new(0),
        }
    }

    fn add_emitted(&self) {
        self.cells_emitted.fetch_add(1, Ordering::Relaxed);
    }
    fn add_failed_encode(&self) {
        self.cells_failed_encode.fetch_add(1, Ordering::Relaxed);
    }
    fn add_failed_decode(&self, n: u64) {
        self.cells_failed_decode.fetch_add(n, Ordering::Relaxed);
    }
    fn add_failed_score(&self) {
        self.cells_failed_score.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot(&self) -> SweepStats {
        SweepStats {
            cells_total: self.cells_total as usize,
            cells_emitted: self.cells_emitted.load(Ordering::Relaxed) as usize,
            cells_failed_encode: self.cells_failed_encode.load(Ordering::Relaxed) as usize,
            cells_failed_decode: self.cells_failed_decode.load(Ordering::Relaxed) as usize,
            cells_failed_score: self.cells_failed_score.load(Ordering::Relaxed) as usize,
        }
    }
}
