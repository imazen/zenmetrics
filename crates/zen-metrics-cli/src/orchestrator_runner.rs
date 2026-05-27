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
#[derive(Debug, Clone)]
pub struct OrchestratorScoreRow {
    /// Metric column name (e.g. `"ssim2_gpu"` or `"butteraugli_max"`).
    pub column: &'static str,
    /// Score value.
    pub value: f64,
}

/// Run one `(ref, dist, metric)` pair through the orchestrator.
/// Returns the score columns the CLI's output writers expect.
///
/// Phase 7 only supports single-column metrics through this path —
/// butteraugli's two-column emit (`max` + `pnorm3`) still flows
/// through the legacy direct-dispatch handler. The CLI router checks
/// this constraint before electing the orchestrator path.
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
    };

    // Single-task path: drive through `run_single` (synchronous,
    // includes the OOM ladder). This avoids spinning up the worker
    // pool for one-shot CLI calls.
    let result = orch.run_single(task);

    match result.outcome {
        Ok(score) => Ok(vec![OrchestratorScoreRow {
            column: cli_metric_to_column_name(cli_kind),
            value: score.value,
        }]),
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

/// Map a CLI metric kind to its primary output column name. Butteraugli
/// is excluded — its two-column shape is handled by the legacy path.
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
        // Cvvdp's column name is versioned at the umbrella level —
        // fall back to a stable bare name for the orchestrator path
        // since the upstream tag is only available with `gpu-cvvdp`
        // enabled. The legacy path still produces the versioned name.
        CliMetricKind::Cvvdp => "cvvdp",
        // Butteraugli single-column fallback (the legacy path also
        // emits a `_pnorm3` column; for orchestrator-driven scoring
        // Phase 7 surfaces only the max-norm).
        CliMetricKind::Butteraugli => "butteraugli_max",
        CliMetricKind::ButteraugliGpu => "butteraugli_max_gpu",
    }
}

/// Decide whether a given CLI metric kind can flow through the
/// orchestrator path. Butteraugli's two-column emit and CVVDP's
/// versioned column logic need the legacy handler to preserve the
/// historical output shape; everything else is fair game.
pub fn metric_orchestrator_eligible(kind: CliMetricKind) -> bool {
    !matches!(
        kind,
        CliMetricKind::Butteraugli | CliMetricKind::ButteraugliGpu | CliMetricKind::Cvvdp
    )
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
pub use ApiMetricKind as MetricKind;

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
    fn metric_eligibility_keeps_butter_and_cvvdp_legacy() {
        assert!(!metric_orchestrator_eligible(CliMetricKind::Butteraugli));
        assert!(!metric_orchestrator_eligible(CliMetricKind::ButteraugliGpu));
        assert!(!metric_orchestrator_eligible(CliMetricKind::Cvvdp));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2Gpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Dssim));
        assert!(metric_orchestrator_eligible(CliMetricKind::Iwssim));
    }
}
