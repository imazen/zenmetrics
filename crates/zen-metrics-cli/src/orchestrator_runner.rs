#![cfg(feature = "orchestrator")]
#![forbid(unsafe_code)]

//! Orchestrator-driven implementations of the CLI's scoring
//! subcommands. Live behind the `orchestrator` feature; the legacy
//! per-subcommand handlers in `main.rs` remain compiled and are the
//! default. Either the `--use-orchestrator` CLI flag or
//! `ZENMETRICS_USE_ORCHESTRATOR=1` env var routes traffic here.
//!
//! ## Why an opt-in path
//!
//! The orchestrator is the recommended entry point per the Phase 7
//! plan — it adds OOM-safe fallback, persistent capability cache, and
//! cached-ref auto-detect, all of which lift sweep workers off the
//! per-cell hand-rolled construction matrix that current production
//! workers rely on.
//!
//! But: existing sweeps in flight (and CI smokes) depend on the
//! legacy code path's exact output shape and timing characteristics.
//! Making the orchestrator path opt-in lets fleets migrate at their
//! own pace and lets us keep both paths green during the transition.
//!
//! ## Per-subcommand routing
//!
//! - `score` / `compare` / `batch`: build a `Vec<Task>` from the
//!   (ref, dist, metric) tuples, drive `Orchestrator::run_all`,
//!   re-emit results in the original output shape.
//! - `sweep`: the per-cell scoring loop inside `sweep::run` is
//!   already tightly integrated with the GPU instance cache; for
//!   Phase 7 we leave that loop in place and instead add a
//!   `--use-orchestrator` gate that warms a persistent
//!   `Orchestrator` at sweep start (so the capability cache is
//!   populated for future workers) but keeps the per-cell scoring
//!   on the existing fast path. This is the additive
//!   compatibility-preserving choice; a follow-up phase can switch
//!   the per-cell loop to `run_all` once the perf characteristics
//!   are validated.

use std::path::PathBuf;
use std::sync::Arc;

use zenmetrics_api::MetricKind as ApiMetricKind;
use zenmetrics_orchestrator::Orchestrator;

use crate::decode::Rgb8Image;
use crate::metrics::MetricKind as CliMetricKind;
use crate::orchestrator_glue::{
    OrchestratorBuildError, OrchestratorMetricSpec, OrchestratorRuntimeOpts,
};

/// Result of scoring one `(ref, dist, metric)` triple via the
/// orchestrator. Mirrors the legacy `run_metric` shape (one row per
/// emitted column) so output writers don't branch on the path.
///
/// **Phase 7.5 change**: `column` is now `String` (was `&'static str`)
/// to accommodate cvvdp's versioned column name (computed at runtime
/// from the `CVVDP_IMPL_TAG` build env var), and butter's two-column
/// emit which produces one row per column.
#[derive(Debug, Clone)]
pub struct OrchestratorScoreRow {
    /// Metric column name (e.g. `"ssim2_gpu"`, `"butteraugli_max_gpu"`,
    /// `"butteraugli_pnorm3_gpu"`, or
    /// `"cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>"`).
    pub column: String,
    /// Score value.
    pub value: f64,
}

/// Run one `(ref, dist, metric)` pair through the orchestrator.
/// Returns the score columns the CLI's output writers expect.
///
/// **Phase 7.5**: multi-column metrics (butter GPU max + pnorm_3) are
/// supported by reading from `TaskResult.output_columns`. The legacy
/// hard-coded column-name mapping is now a fallback for the case
/// where `output_columns` is empty (older orchestrator builds that
/// predate Phase 7.5 — unlikely in practice since both ship from the
/// same workspace, but kept for graceful degradation).
#[cfg(feature = "orchestrator-cuda")]
pub fn orchestrator_score_one(
    orch: &mut Orchestrator,
    cli_kind: CliMetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<OrchestratorScoreRow>, Box<dyn std::error::Error>> {
    use zenmetrics_orchestrator::{Task, TaskData};

    let spec = OrchestratorMetricSpec::from_cli(cli_kind);
    let width = reference.width;
    let height = reference.height;

    // Build the task — orchestrator API consumes owned byte vectors.
    let task = Task {
        task_id: 1,
        ref_data: TaskData::Srgb8(reference.pixels.clone()),
        dist_data: TaskData::Srgb8(distorted.pixels.clone()),
        width,
        height,
        metric: spec.kind,
        params: None,
        ref_hash: 0,
    };

    // Single-task path: drive through `run_single` (synchronous,
    // includes the OOM ladder). This avoids spinning up the worker
    // pool for one-shot CLI calls.
    let result = orch.run_single(task);

    match result.outcome {
        Ok(score) => {
            // Phase 7.5: consume output_columns when populated, then
            // re-key per CLI variant so the column names match the
            // legacy path's output exactly. Empty output_columns
            // (older orchestrator build) falls back to the static
            // primary-column-name mapping.
            let rows: Vec<OrchestratorScoreRow> = if result.output_columns.is_empty() {
                vec![OrchestratorScoreRow {
                    column: cli_metric_to_column_name(cli_kind).to_string(),
                    value: score.value,
                }]
            } else {
                rekey_orchestrator_columns(cli_kind, &result.output_columns)
                    .into_iter()
                    .map(|(column, value)| OrchestratorScoreRow { column, value })
                    .collect()
            };
            Ok(rows)
        }
        Err(e) => Err(format!(
            "orchestrator: {e} (backends tried: {:?})",
            result.backends_attempted
        )
        .into()),
    }
}

/// Stub for builds without `orchestrator-cuda`. The orchestrator's
/// executor + run_single live behind `cuda`, so without that feature
/// every orchestrator-driven score call surfaces a "rebuild with CUDA"
/// error rather than silently falling back to legacy dispatch.
#[cfg(not(feature = "orchestrator-cuda"))]
pub fn orchestrator_score_one(
    _orch: &mut Orchestrator,
    _cli_kind: CliMetricKind,
    _reference: &Rgb8Image,
    _distorted: &Rgb8Image,
) -> Result<Vec<OrchestratorScoreRow>, Box<dyn std::error::Error>> {
    Err("orchestrator-driven scoring requires the `orchestrator-cuda` feature".into())
}

/// Map a CLI metric kind to its primary output column name. Used as
/// the fallback when `TaskResult.output_columns` is empty (an older
/// orchestrator build without Phase 7.5's column emission); modern
/// builds drive column names from `output_columns` directly so this
/// function only fires for graceful degradation.
fn cli_metric_to_column_name(kind: CliMetricKind) -> &'static str {
    match kind {
        CliMetricKind::Ssim2 => "ssim2",
        CliMetricKind::Ssim2Gpu => "ssim2_gpu",
        CliMetricKind::Dssim => "dssim",
        CliMetricKind::DssimGpu => "dssim_gpu",
        CliMetricKind::IwssimGpu => "iwssim_gpu",
        CliMetricKind::Zensim => "zensim",
        CliMetricKind::ZensimGpu => "zensim_gpu",
        CliMetricKind::Iwssim => "iwssim",
        CliMetricKind::Cvvdp => "cvvdp",
        CliMetricKind::Butteraugli => "butteraugli_max",
        CliMetricKind::ButteraugliGpu => "butteraugli_max_gpu",
    }
}

/// Re-key the orchestrator's `output_columns` BTreeMap to match the
/// legacy CLI variant's column names. The orchestrator emits canonical
/// (GPU-variant-style) keys via `executor::build_output_columns`;
/// sweep / batch / score callers using a CPU-variant CLI metric (e.g.
/// `--metric ssim2` not `--metric ssim2-gpu`) need to drop the `_gpu`
/// suffix to match the legacy MetricCache's CPU path.
///
/// **Bit-identical parquet contract**: the keys produced here MUST
/// match what the legacy `run_metric` writes for the same CLI metric.
/// The mapping is `gpu_cli_variant -> orchestrator_key -> sweep_key`.
/// For GPU variants the orchestrator key == sweep key (no rename).
/// For CPU variants we strip the `_gpu` suffix.
///
/// Returns a `Vec<(String, f64)>` in the input BTreeMap's iteration
/// order (sorted-by-key), so callers iterating the result emit
/// deterministic column order.
pub fn rekey_orchestrator_columns(
    cli_kind: CliMetricKind,
    columns: &std::collections::BTreeMap<String, f64>,
) -> Vec<(String, f64)> {
    // Renames applied per CLI variant. GPU variants need no renames
    // (their column name matches what the orchestrator produces).
    let rename: &[(&str, &str)] = match cli_kind {
        CliMetricKind::Ssim2 => &[("ssim2_gpu", "ssim2")],
        CliMetricKind::Dssim => &[("dssim_gpu", "dssim")],
        CliMetricKind::Zensim => &[("zensim_gpu", "zensim")],
        CliMetricKind::Butteraugli => {
            &[
                ("butteraugli_max_gpu", "butteraugli_max"),
                ("butteraugli_pnorm3_gpu", "butteraugli_pnorm3"),
            ]
        }
        CliMetricKind::Iwssim
        | CliMetricKind::IwssimGpu
        | CliMetricKind::Cvvdp
        | CliMetricKind::Ssim2Gpu
        | CliMetricKind::DssimGpu
        | CliMetricKind::ZensimGpu
        | CliMetricKind::ButteraugliGpu => &[],
    };
    columns
        .iter()
        .map(|(k, v)| {
            let renamed = rename
                .iter()
                .find(|(orig, _)| *orig == k.as_str())
                .map(|(_, new)| new.to_string())
                .unwrap_or_else(|| k.clone());
            (renamed, *v)
        })
        .collect()
}

/// Decide whether a given CLI metric kind can flow through the
/// orchestrator path.
///
/// **Phase 7.5 change**: butteraugli + cvvdp are now eligible.
///
/// - **Butteraugli**: the orchestrator's `TaskResult.output_columns`
///   includes `butteraugli_pnorm3_gpu` alongside the max-norm column,
///   so the two-column emit survives end-to-end. The columns are
///   bit-identical to the legacy `MetricCache::compute_butter` path
///   (same fused reduction kernel, same field names — see
///   `executor::build_output_columns` and
///   `butteraugli_gpu::opaque::compute_srgb_u8_with_pnorm3`).
/// - **Cvvdp**: the orchestrator surfaces the versioned column tag
///   from `Score::metric_version`, then
///   `executor::build_output_columns` keys it into the parquet under
///   `cvvdp_gpu::CVVDP_COLUMN_NAME` — same shape the legacy
///   `CvvdpBatchScorer` emits.
///
/// **Phase 7.5 leaves `CliMetricKind::Butteraugli` (CPU)** still on
/// the legacy path because the CPU butteraugli adapter doesn't expose
/// `pnorm_3` today; routing it through the orchestrator would drop
/// the second column silently. Future work: add `pnorm_3` to the
/// `cpu-butter` adapter or document that CPU butter is single-column.
pub fn metric_orchestrator_eligible(kind: CliMetricKind) -> bool {
    !matches!(kind, CliMetricKind::Butteraugli)
}

/// Build the orchestrator at the start of a CLI command. Wraps the
/// glue helper with a uniform error type so subcommand handlers don't
/// need to import the glue module directly.
pub fn build_orchestrator(opts: &OrchestratorRuntimeOpts) -> Result<Orchestrator, Box<dyn std::error::Error>> {
    opts.build().map_err(|e: OrchestratorBuildError| {
        let msg: Box<dyn std::error::Error> = format!("{e}").into();
        msg
    })
}

/// Helper struct for the runner-level cache — keeps the orchestrator
/// alive across multiple subcommand calls (e.g. when used as a
/// library by the sweep worker, which scores many cells per process).
pub struct OrchestratorHandle {
    inner: Arc<std::sync::Mutex<Orchestrator>>,
}

impl OrchestratorHandle {
    /// Wrap a freshly-built orchestrator.
    pub fn new(orch: Orchestrator) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(orch)),
        }
    }

    /// Borrow the inner orchestrator under the lock.
    pub fn lock(&self) -> std::sync::MutexGuard<'_, Orchestrator> {
        self.inner.lock().expect("orchestrator handle poisoned")
    }
}

impl Clone for OrchestratorHandle {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// Print a one-line summary of the orchestrator's capability cache to
/// stderr. Called at the start of orchestrator-driven sessions so
/// users can audit which machine profile is in use.
pub fn print_capability_summary(orch: &Orchestrator) {
    let cap = orch.capability();
    eprintln!(
        "[orchestrator] capability profile: machine={} gpu={} cpu={} cache={}",
        &cap.machine_hash[..16.min(cap.machine_hash.len())],
        if cap.gpu.present {
            cap.gpu.model.as_str()
        } else {
            "<absent>"
        },
        cap.cpu.brand,
        orch.cache_path().display(),
    );
}

/// Resolve the `--cpu-features` CLI argument into a vector of metric
/// tags. Recognised tokens: `cvvdp`, `ssim2`, `dssim`, `butter`,
/// `zensim`, `all`. Empty input → empty Vec. Unknown tokens are
/// surfaced as a clear error so the user can fix the typo.
pub fn parse_cpu_features(raw: &str) -> Result<Vec<&'static str>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let mut out: Vec<&'static str> = Vec::new();
    for tok in trimmed.split(',') {
        let tok = tok.trim().to_ascii_lowercase();
        let mapped: &'static str = match tok.as_str() {
            "all" => "all",
            "cvvdp" => "cvvdp",
            "ssim2" => "ssim2",
            "dssim" => "dssim",
            "butter" | "butteraugli" => "butter",
            "zensim" => "zensim",
            other => {
                return Err(format!(
                    "unknown --cpu-features token '{other}'; valid: cvvdp,ssim2,dssim,butter,zensim,all"
                ));
            }
        };
        out.push(mapped);
    }
    Ok(out)
}

/// Construct an `OrchestratorRuntimeOpts` from raw CLI strings + the
/// `--bench-on-start` selector.
pub fn runtime_opts_from_cli(
    cache_dir: Option<PathBuf>,
    bench_on_start: BenchOnStart,
    cpu_features_raw: &str,
) -> Result<OrchestratorRuntimeOpts, String> {
    let cpu_features = parse_cpu_features(cpu_features_raw)?
        .into_iter()
        .map(String::from)
        .collect();
    let bench_on_start = match bench_on_start {
        BenchOnStart::Auto => None,
        BenchOnStart::Yes => Some(true),
        BenchOnStart::No => Some(false),
    };
    Ok(OrchestratorRuntimeOpts {
        cache_dir,
        bench_on_start,
        cpu_features,
    })
}

/// `--bench-on-start` selector.
#[derive(Debug, Clone, Copy)]
pub enum BenchOnStart {
    /// Auto-detect (default): re-bench only if the cache is stale.
    Auto,
    /// Force a bench at startup.
    Yes,
    /// Skip the bench entirely (use whatever's in the cache).
    No,
}

/// Map the CLI's tri-state `--bench-on-start <auto|yes|no>` flag
/// (parsed as `Option<bool>`-via-clap) into this module's enum.
pub fn bench_on_start_from_flag(flag: Option<&str>) -> Result<BenchOnStart, String> {
    let Some(v) = flag else {
        return Ok(BenchOnStart::Auto);
    };
    match v.trim().to_ascii_lowercase().as_str() {
        "auto" | "" => Ok(BenchOnStart::Auto),
        "yes" | "true" | "on" | "1" => Ok(BenchOnStart::Yes),
        "no" | "false" | "off" | "0" => Ok(BenchOnStart::No),
        other => Err(format!(
            "--bench-on-start expects auto|yes|no, got '{other}'"
        )),
    }
}

/// Re-export the ApiMetricKind to make doctest examples in the
/// migration guide compile without a transitive zenmetrics-api dep.
pub use zenmetrics_api::MetricKind;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_features_basic() {
        assert_eq!(parse_cpu_features("").unwrap(), Vec::<&str>::new());
        assert_eq!(parse_cpu_features("all").unwrap(), vec!["all"]);
        assert_eq!(
            parse_cpu_features("ssim2, dssim ,cvvdp").unwrap(),
            vec!["ssim2", "dssim", "cvvdp"]
        );
        assert!(parse_cpu_features("unknown").is_err());
    }

    #[test]
    fn bench_on_start_parses_yes_no_auto() {
        assert!(matches!(
            bench_on_start_from_flag(None).unwrap(),
            BenchOnStart::Auto
        ));
        assert!(matches!(
            bench_on_start_from_flag(Some("yes")).unwrap(),
            BenchOnStart::Yes
        ));
        assert!(matches!(
            bench_on_start_from_flag(Some("no")).unwrap(),
            BenchOnStart::No
        ));
        assert!(matches!(
            bench_on_start_from_flag(Some("auto")).unwrap(),
            BenchOnStart::Auto
        ));
        assert!(bench_on_start_from_flag(Some("maybe")).is_err());
    }

    #[test]
    fn rekey_orchestrator_columns_phase_7_5_renames_cpu_variants() {
        use std::collections::BTreeMap;

        // Orchestrator emits GPU-variant key for ssim2; the CLI's
        // `Ssim2Gpu` variant should pass it through unchanged.
        let mut cols = BTreeMap::new();
        cols.insert("ssim2_gpu".to_string(), 95.5_f64);
        let gpu_rekeyed = rekey_orchestrator_columns(CliMetricKind::Ssim2Gpu, &cols);
        assert_eq!(gpu_rekeyed, vec![("ssim2_gpu".to_string(), 95.5)]);

        // CPU variant of ssim2 (CLI `Ssim2`) strips the _gpu suffix
        // to match what the legacy `MetricCache` would emit.
        let cpu_rekeyed = rekey_orchestrator_columns(CliMetricKind::Ssim2, &cols);
        assert_eq!(cpu_rekeyed, vec![("ssim2".to_string(), 95.5)]);

        // Butter GPU keeps two columns (max + pnorm3) as-is.
        let mut butter_cols = BTreeMap::new();
        butter_cols.insert("butteraugli_max_gpu".to_string(), 1.5_f64);
        butter_cols.insert("butteraugli_pnorm3_gpu".to_string(), 2.5_f64);
        let butter_gpu = rekey_orchestrator_columns(CliMetricKind::ButteraugliGpu, &butter_cols);
        // BTreeMap iteration is sorted by key — max before pnorm3
        // alphabetically.
        assert_eq!(
            butter_gpu,
            vec![
                ("butteraugli_max_gpu".to_string(), 1.5),
                ("butteraugli_pnorm3_gpu".to_string(), 2.5),
            ]
        );

        // Butter CPU strips _gpu from both columns.
        let butter_cpu = rekey_orchestrator_columns(CliMetricKind::Butteraugli, &butter_cols);
        assert_eq!(
            butter_cpu,
            vec![
                ("butteraugli_max".to_string(), 1.5),
                ("butteraugli_pnorm3".to_string(), 2.5),
            ]
        );

        // Cvvdp uses a versioned column; no rename ever.
        let mut cvvdp_cols = BTreeMap::new();
        cvvdp_cols.insert("cvvdp_imazen_v0_0_1".to_string(), 9.2_f64);
        let cvvdp = rekey_orchestrator_columns(CliMetricKind::Cvvdp, &cvvdp_cols);
        assert_eq!(cvvdp, vec![("cvvdp_imazen_v0_0_1".to_string(), 9.2)]);
    }

    #[test]
    fn metric_eligibility_phase_7_5_admits_butter_gpu_and_cvvdp() {
        // Phase 7.5: butter (GPU) + cvvdp now flow through the
        // orchestrator. Only CPU butteraugli still uses the legacy
        // path (the CPU adapter doesn't expose `pnorm_3`).
        assert!(!metric_orchestrator_eligible(CliMetricKind::Butteraugli));
        assert!(metric_orchestrator_eligible(CliMetricKind::ButteraugliGpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Cvvdp));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2Gpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Dssim));
        assert!(metric_orchestrator_eligible(CliMetricKind::Iwssim));
    }
}
