//! In-process replacement for the bash `zenmetrics sweep` subprocess.
//!
//! The bash chunk worker calls `zenmetrics sweep --codec X --sources S
//! --q-grid Q --knob-grid K --output T --metric m1 --metric m2 ...`
//! once per (codec, knob_tuple) group — roughly 30 times per chunk.
//! Each subprocess launches a fresh `zenmetrics` PROCESS, which means
//! a fresh cubecl device init (~3-5 s on RTX 3060 / 1080 Ti class
//! GPUs). 30 inits per chunk × 4-5 min chunk time = 30-50 % of wall
//! time spent on cubecl init alone.
//!
//! By linking `zenmetrics_cli` as a library and calling `run_sweep`
//! in-process, all 30 group calls — across all chunks the worker
//! processes during its lifetime — reuse cubecl's process-static
//! device cache. The cost goes from `30 × N_chunks × init_s` down to
//! `1 × init_s`. On a 2 568-chunk run that's the difference between
//! ~5 hr and ~7 hr of cubecl init overhead saved.
//!
//! ## API shape
//!
//! [`run_group_inline`] is the per-group entry point. The chunk
//! dispatcher (in `worker/chunk.rs`) builds one [`InlineGroupSpec`]
//! per (codec, knob_tuple) pairing from the chunk's input parquet,
//! then calls this for each. The output TSV path is caller-supplied
//! so the dispatcher controls scratch-dir layout.

#![cfg(feature = "inline-sweep")]

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use zenmetrics_cli::metrics::{GpuRuntime, MetricKind, ZensimFeatureRegime};
use zenmetrics_cli::sweep::{
    CodecKind, KnobGrid, PlanSpec, SweepConfig, SweepOrchestratorHandle, parse_knob_grid,
    parse_q_grid, run_sweep,
};

/// Process-wide orchestrator handle. Built lazily on the first
/// [`run_group_inline`] call and reused for every subsequent group
/// within the worker process. Phase A wire-through (2026-05-28):
/// the inline-sweep path previously skipped the orchestrator entirely,
/// landing every cell on the legacy `MetricCache` — bypassing the OOM
/// ladder, capability cache, and cached-ref auto-detect. Constructing
/// once at process scope keeps the warm cubecl instances shared across
/// all chunks the worker processes (same lifecycle the existing
/// `MetricCache::global` OnceLock used).
///
/// Build failures are recorded as `None` and the call site falls back
/// to the legacy MetricCache path with a one-time warning. Refusing to
/// fall back would surface as a hard chunk failure on machines whose
/// capability-cache directory isn't writeable — better to keep working
/// with the legacy path than to lose the chunk.
static ORCHESTRATOR: OnceLock<Option<SweepOrchestratorHandle>> = OnceLock::new();

/// Force-disable the orchestrator wire-through via env. Set
/// `ZENMETRICS_SWEEP_LEGACY=1` to take the historical MetricCache path
/// even when this binary is built with the `orchestrator` features.
/// Kept as an escape hatch in case the wire-through surfaces a sweep
/// parity regression in production before the next image rebuild.
fn legacy_sweep_from_env() -> bool {
    std::env::var("ZENMETRICS_SWEEP_LEGACY")
        .ok()
        .as_deref()
        .map(|s| matches!(s.trim(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false)
}

/// Lazily build a single orchestrator handle for the worker process.
/// First call pays orchestrator init (~50-200 ms — capability cache
/// hydrate, optional warm-bench). Subsequent calls return the cached
/// handle. Returns `None` (and logs a warning ONCE) when:
///   * `ZENMETRICS_SWEEP_LEGACY=1` is set in env, OR
///   * `OrchestratorRuntimeOpts::build()` errors (cache I/O failure,
///     etc.). In that case `run_group_inline` falls back to the
///     legacy MetricCache path.
fn orchestrator_handle() -> Option<&'static SweepOrchestratorHandle> {
    ORCHESTRATOR
        .get_or_init(|| {
            if legacy_sweep_from_env() {
                tracing::warn!(
                    "ZENMETRICS_SWEEP_LEGACY=1 — sweep cells will use legacy MetricCache path"
                );
                return None;
            }
            // Default runtime opts: auto bench-on-start (re-bench only if
            // cache is stale), default cache directory, no CPU-feature
            // whitelist. The sweep worker fixes the GPU runtime to CUDA
            // (see `InlineGroupSpec::gpu_runtime`), so we don't need the
            // CPU paths.
            let opts = zenmetrics_cli::orchestrator_glue::OrchestratorRuntimeOpts::default();
            match zenmetrics_cli::orchestrator_runner::build_orchestrator(&opts) {
                Ok(orch) => {
                    zenmetrics_cli::orchestrator_runner::print_capability_summary(&orch);
                    Some(std::sync::Arc::new(std::sync::Mutex::new(orch)))
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "orchestrator init failed — sweep cells will fall back to legacy MetricCache"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// One sweep-call's worth of work: a single (codec, knob_tuple)
/// group's worth of cells.
///
/// All fields mirror the bash worker's `zenmetrics sweep` flag
/// shape so the bash → Rust transition is verifiable: a chunk's
/// per-group args become one InlineGroupSpec.
pub struct InlineGroupSpec {
    /// Codec to drive. e.g. `zenjpeg`, `zenwebp`, `zenavif`.
    pub codec: CodecKind,
    /// Directory of source images. Same shape the bash worker
    /// hands `--sources`.
    pub sources_dir: PathBuf,
    /// Comma-list of q values to sweep, e.g. `"5,10,15,20"`.
    pub q_grid: String,
    /// JSON object `{axis: [values]}` describing the knob Cartesian
    /// product for THIS group only. The chunk's full knob_tuple_json
    /// is converted by lifting each scalar value to a single-element
    /// list (`{"effort": 1, ...}` → `{"effort": [1], ...}`).
    pub knob_grid_json: String,
    /// Plan-driven zenjpeg cells (`"rd_core"` / `"modes_full"`): when
    /// set, cells come from `zenjpeg::encode::sweep` over `q_grid`
    /// instead of `knob_grid_json` (which must then be empty). The
    /// plan's audit manifest lands at `<output_tsv>.plan.json` next to
    /// the group TSV.
    pub plan: Option<String>,
    /// Optional cell budget for `plan`.
    pub plan_budget: Option<usize>,
    /// Metrics to score with. e.g. `[Cvvdp, Ssim2Gpu, ...]`.
    pub metrics: Vec<MetricKind>,
    /// GPU runtime selector. `Cuda` in production; `Auto` for local
    /// dev so wgpu can fall back when CUDA isn't compiled in.
    pub gpu_runtime: GpuRuntime,
    /// Where to write the per-group output TSV. Caller is expected
    /// to concat these across all groups in the chunk.
    pub output_tsv: PathBuf,
    /// Optional path for the zensim feature parquet sidecar.
    /// Only meaningful if `metrics` contains `Zensim` (CPU, 300-D) or
    /// `ZensimGpu` (GPU, [`Self::feature_regime`]-controlled: 228 /
    /// 300 / 372). When both are in the metric set, GPU wins.
    pub feature_output: Option<PathBuf>,
    /// Zensim feature regime for the GPU path. Defaults to
    /// [`ZensimFeatureRegime::WithIw`] (372). Ignored when GPU zensim
    /// is not in `metrics` OR `feature_output` is None.
    pub feature_regime: ZensimFeatureRegime,
    /// Optional directory to receive the raw encoded codec bytes
    /// per cell (.jpg / .webp / .avif / .jxl / .png). Phase A
    /// disabled this because the v21 binary built before the
    /// `--encoded-out-dir` flag landed didn't expose it. The Rust
    /// in-process call doesn't have that limitation — the field
    /// is on `SweepConfig` regardless of when the binary was
    /// built.
    pub encoded_out_dir: Option<PathBuf>,
    /// Rayon thread budget passed via `--jobs`. 0 = auto-detect.
    pub jobs: usize,
    /// HDR sweep mode (v27 schema addition, 2026-06-12): sources are
    /// 16-bit PQ PNGs decoded to absolute nits, cells run the HDR
    /// codec round-trip (zenjxl today) and score via the validated
    /// per-metric HDR feedings. `SweepConfig.hdr` validates the whole
    /// combination at startup and errors loudly when unsupported — a
    /// chunk can never silently degrade to SDR semantics. Default
    /// `false` keeps every existing chunk byte-identical.
    pub hdr: bool,
}

/// Run one group's worth of sweep work in-process. The first call
/// pays cubecl device init (~3-5 s); subsequent calls within the
/// same worker process reuse the cached device.
pub fn run_group_inline(spec: InlineGroupSpec) -> Result<()> {
    // Convert string-shaped CLI args into zenmetrics-cli's typed
    // forms. parse_q_grid + parse_knob_grid are the exact parsers
    // the binary uses for its --q-grid / --knob-grid flags, so the
    // semantics match by construction.
    //
    // zenmetrics-cli's parsers return `Box<dyn StdError>` which
    // isn't Send+Sync — anyhow's Context trait won't bind to it
    // directly. Convert via the error message instead; the caller
    // doesn't need the full error chain.
    let q_grid: Vec<f64> = parse_q_grid(&spec.q_grid)
        .map_err(|e| anyhow::anyhow!("parse q_grid {:?}: {e}", spec.q_grid))?;
    let knob_grid: KnobGrid = parse_knob_grid(&spec.knob_grid_json)
        .map_err(|e| anyhow::anyhow!("parse knob_grid {:?}: {e}", spec.knob_grid_json))?;

    // Enumerate source files. The bash equivalent passes a dir; the
    // Rust API expects a Vec<PathBuf>. List once per group; for a
    // typical chunk with 1-3 unique source images this is cheap.
    let mut sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&spec.sources_dir)
        .with_context(|| format!("read sources dir: {}", spec.sources_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            sources.push(path);
        }
    }
    // Stable order for determinism — sweeps with random source
    // order would produce different-but-equivalent TSVs that don't
    // diff cleanly across reruns.
    sources.sort();

    let cfg = SweepConfig {
        codec: spec.codec,
        sources,
        q_grid,
        knob_grid,
        plan: spec.plan.as_ref().map(|name| PlanSpec {
            name: name.clone(),
            budget: spec.plan_budget,
        }),
        metrics: spec.metrics,
        gpu_runtime: spec.gpu_runtime,
        output: spec.output_tsv.clone(),
        feature_output: spec.feature_output,
        feature_regime: spec.feature_regime,
        // The bash worker added --distorted-out-dir for sidecar
        // PNGs but the v21 binary lacked --encoded-out-dir. The
        // Rust in-process call has direct access to both fields.
        // We default distorted/pairs to None unless callers wire
        // them through; encoded_out_dir comes from the spec so
        // chunks that want encoded blobs can request them.
        distorted_out_dir: None,
        encoded_out_dir: spec.encoded_out_dir,
        pairs_tsv: None,
        jobs: spec.jobs,
        hdr: spec.hdr,
    };

    // run_sweep returns a SweepStats struct with cell counts; we
    // discard it for now (the parquet conversion stage doesn't
    // need it). Phase B.4 will plumb it through for per-chunk
    // success metrics.
    //
    // Phase A wire-through (2026-05-28): pass the process-wide
    // orchestrator handle so each cell flows through the
    // OOM-ladder/capability-cache code path instead of the legacy
    // MetricCache. `orchestrator_handle()` returns `None` when env
    // overrides or init fails; `run_sweep` accepts that and falls
    // back to MetricCache automatically.
    let orch = orchestrator_handle();
    run_sweep(&cfg, orch).map_err(|e| anyhow::anyhow!("run_sweep: {e}"))?;
    Ok(())
}

/// Convert one chunk-input's `knob_tuple_json` (a single point in
/// knob-space, e.g. `{"effort": 1, "subsampling": "444"}`) into the
/// `{axis: [value]}` Cartesian-product shape `run_sweep` expects.
///
/// We lift each scalar value to a single-element list, so the
/// Cartesian product collapses to that single point. The bash
/// worker does the same `with_entries(.value |= [.])` jq step.
pub fn knob_tuple_to_grid_json(tuple_json: &str) -> Result<String> {
    let v: serde_json::Value = serde_json::from_str(tuple_json)
        .with_context(|| format!("parse knob_tuple_json: {tuple_json}"))?;
    let obj = v.as_object().context("knob_tuple_json must be an object")?;
    let mut out = serde_json::Map::with_capacity(obj.len());
    for (k, val) in obj {
        out.insert(k.clone(), serde_json::Value::Array(vec![val.clone()]));
    }
    serde_json::to_string(&serde_json::Value::Object(out)).context("serialise knob_grid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn knob_tuple_lifted_to_single_element_array() {
        let tup = r#"{"effort":1,"subsampling":"444","aq_enabled":true}"#;
        let grid = knob_tuple_to_grid_json(tup).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&grid).unwrap();
        let obj = parsed.as_object().unwrap();
        assert_eq!(obj["effort"].as_array().unwrap()[0].as_i64(), Some(1));
        assert_eq!(
            obj["subsampling"].as_array().unwrap()[0].as_str(),
            Some("444")
        );
        assert_eq!(
            obj["aq_enabled"].as_array().unwrap()[0].as_bool(),
            Some(true)
        );
    }

    #[test]
    fn empty_knob_tuple_yields_empty_grid() {
        let grid = knob_tuple_to_grid_json("{}").unwrap();
        assert_eq!(grid, "{}");
    }

    #[test]
    fn non_object_knob_tuple_errors() {
        assert!(knob_tuple_to_grid_json("[1,2,3]").is_err());
        assert!(knob_tuple_to_grid_json("\"str\"").is_err());
    }
}
