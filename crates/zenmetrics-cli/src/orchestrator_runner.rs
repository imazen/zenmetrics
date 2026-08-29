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
        CliMetricKind::CvvdpGpu => "cvvdp",
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
    // Phase 7.7.1: the orchestrator's `executor::build_output_columns`
    // emits the *versioned* iwssim column name (e.g.
    // `iwssim_imazen_v0_0_1` via `iwssim_gpu::IWSSIM_COLUMN_NAME`),
    // matching the cvvdp pattern. The legacy
    // `MetricCache::run_metric_cached` path historically emitted the
    // unversioned `iwssim_gpu` column. The bit-identical parquet
    // sidecar contract is what production tooling locked onto, so
    // re-key the versioned name back to the legacy
    // unversioned name for both `Iwssim` and `IwssimGpu` CLI variants.
    //
    // Discover the versioned key at compile-time when the `iwssim`
    // feature lights up zenmetrics_api::iwssim; in builds where it
    // doesn't (rare), the rename is a no-op since the orchestrator
    // wouldn't have emitted that column either.
    #[cfg(feature = "gpu-iwssim")]
    let iwssim_versioned: &str = zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME;
    #[cfg(not(feature = "gpu-iwssim"))]
    let iwssim_versioned: &str = "iwssim_imazen_vX_X_X";

    // Renames applied per CLI variant. GPU variants need no renames
    // for ssim2/dssim/zensim/butteraugli (their column name matches
    // what the orchestrator produces). Iwssim is the exception:
    // even `IwssimGpu` strips the versioned suffix because the legacy
    // CLI path uses `iwssim_gpu` and downstream parquet readers depend
    // on that exact column name.
    let renames: Vec<(String, String)> = match cli_kind {
        CliMetricKind::Ssim2 => vec![("ssim2_gpu".to_string(), "ssim2".to_string())],
        CliMetricKind::Dssim => vec![("dssim_gpu".to_string(), "dssim".to_string())],
        CliMetricKind::Zensim => vec![("zensim_gpu".to_string(), "zensim".to_string())],
        CliMetricKind::Butteraugli => vec![
            (
                "butteraugli_max_gpu".to_string(),
                "butteraugli_max".to_string(),
            ),
            (
                "butteraugli_pnorm3_gpu".to_string(),
                "butteraugli_pnorm3".to_string(),
            ),
        ],
        // Iwssim-GPU: the legacy CLI's `IwssimGpu` arm emits the
        // unversioned `iwssim_gpu` column (see
        // `crates/zenmetrics-cli/src/metrics/mod.rs` IwssimGpu match
        // arm). The orchestrator's `executor::build_output_columns`
        // routes through `IwssimOpaque::column_name()` which yields
        // the *versioned* `iwssim_imazen_v<MAJOR>_<MINOR>_<PATCH>`
        // name. Re-key it back to `iwssim_gpu` so the parquet sidecar
        // shape is bit-identical to legacy.
        CliMetricKind::IwssimGpu => {
            vec![(iwssim_versioned.to_string(), "iwssim_gpu".to_string())]
        }
        // Iwssim (the bare CLI variant): the legacy CLI's `Iwssim`
        // arm emits the *versioned* column name straight through
        // (`IWSSIM_COLUMN_NAME`). The orchestrator already emits the
        // versioned name — so this case is a no-op rename. We list
        // it here explicitly (rather than falling into the empty arm
        // below) so future readers see the contract.
        CliMetricKind::Iwssim => Vec::new(),
        CliMetricKind::Cvvdp
        | CliMetricKind::CvvdpGpu
        | CliMetricKind::Ssim2Gpu
        | CliMetricKind::DssimGpu
        | CliMetricKind::ZensimGpu
        | CliMetricKind::ButteraugliGpu => Vec::new(),
    };
    columns
        .iter()
        .map(|(k, v)| {
            let renamed = renames
                .iter()
                .find(|(orig, _)| orig.as_str() == k.as_str())
                .map(|(_, new)| new.clone())
                .unwrap_or_else(|| k.clone());
            (renamed, *v)
        })
        .collect()
}

/// Decide whether a given CLI metric kind can flow through the
/// orchestrator path.
///
/// **Every metric kind is eligible as of 2026-08-28.** Butteraugli was
/// the last holdout; the history and the accepted tolerance are below
/// because the reason it was excluded is easy to re-derive wrongly.
///
/// - **Butteraugli (CPU + GPU)** — eligible since 2026-08-28
///   (zenmetrics#47 item 4). Phase 7.7.1 had reverted it to the legacy
///   path: the legacy GPU helper `butter_pnorm3::score_both` calls
///   `Butteraugli::new_multires` and ALWAYS returns the
///   multi-resolution score, while `ButteraugliOpaque`'s strip-preferred
///   `MemoryMode::Auto` resolver at the time dropped to a
///   SINGLE-resolution walker — a ~14–30 % divergence. That defect was
///   fixed in butteraugli-gpu `77020757` (opaque Strip → the
///   multi-resolution `new_multires_strip`, `HALO_ROWS` 40 → 80).
///
///   What kept the flag `false` afterwards was the *gate*, not missing
///   code: the strip walker was documented as drifting "up to ~1e-4
///   rel", above the Phase 7.7.1 "bit-exact or ≤ ~5e-5 atomic-reorder
///   noise" acceptance. **That ~1e-4 was the tests' assertion
///   TOLERANCE, not a measured drift.** The only measurement that ever
///   produced it (`benchmarks/butter_strip_halo_2026-05-31.md`) was the
///   PRE-fix `HALO_ROWS = 40` half-res sibling; the same document's
///   post-fix table already read `0.00e0`.
///
///   **Accepted tolerance (measured, `benchmarks/butter_strip_drift_2026-08-28.{csv,md}`):**
///   6,810 cells — 95 real images (photo / screen / illustration /
///   line-art, 0.004–10.5 MP) × 15 JPEG quality levels (butteraugli
///   score 0.32–31.62) × strip bodies {auto, 128, 256, 512}, i.e. 1–75
///   dispatched strips:
///   * `new_multires_strip` vs `new_multires`, **max-norm score:
///     bit-identical in 6,810 / 6,810 cells** (`rel = 0.00e0`);
///   * the same pair's `pnorm_3`: **≤ 1.72e-6** rel (p99 4.6e-7) —
///     host-side partial-fold ordering;
///   * the production surface (`ButteraugliOpaque` `Full` vs `Auto`):
///     **bit-identical in 6,810 / 6,810**;
///   * flat on every axis — magnitude, megapixels, strip count,
///     quality, content class — and invariant to the strip body itself
///     (2,118 / 2,118 groups identical across all bodies tested).
///   That is inside the ≤ 5e-5 gate by two orders of magnitude, so no
///   `MemoryMode::Full` pin is needed in `executor::construct`.
///
///   Bit-identity is the *expected* outcome, not luck: `HALO_ROWS = 80`
///   gives the half-res sibling 40 real halo rows against a 34-row blur
///   requirement (`butteraugli_gpu::strip` module docs), so each body
///   pixel's diffmap value sees exactly the whole-image inputs, and the
///   max-norm fold (`max`) is exact and order-independent. Only
///   `pnorm_3` sums, hence its ~1e-7…1e-6 residual.
///
///   **Scope of that evidence:** `cubecl-wgpu`/Metal, with the portable
///   per-thread-partials reduction (the crate default). CUDA was not
///   available on the measuring host, so the 54-cell CUDA parity sweep
///   (`scripts/orchestrator_parity_sweep.py`) is still outstanding. The
///   argument above is backend-independent, but butteraugli-gpu's
///   opt-in `fast-reduction` feature (CUDA-only) replaces the
///   deterministic fold with `Atomic<f32>::fetch_add`, whose ordering is
///   nondeterministic — under that flag neither path is reproducible
///   run-to-run and this tolerance does not apply.
///
///   **Separate axis, deliberately not gated here:** eligibility also
///   hands the backend choice to `zenmetrics_orchestrator::chooser`.
///   `OrchestratorMetricSpec::prefer_cpu` is NOT forwarded into the
///   `Task` (it only drives a build-config error in
///   `orchestrator_glue.rs`), and `chooser::cpu_wins_oneshot_max_pixels`
///   returns `u64::MAX` for butter, so an `ExecContext::OneShot` task
///   prefers CPU at any size when `cpu-butter` is compiled in. That is
///   true of all ten eligible metric kinds, not butter specifically.
///   Measured envelope for butter on the same corpus: CPU
///   `butteraugli_strip(256)` vs CPU whole ≤ 9.4e-8, but GPU-whole vs
///   CPU-whole is median 2.02e-4 / p95 2.37e-4 / **max 1.29e-2** — far
///   larger than the strip term. Sweeps that need byte-identical butter
///   columns against archived sidecars should pin the backend or pass
///   `--use-legacy-scheduler`.
/// - **Cvvdp**: orchestrator-eligible. The orchestrator surfaces the
///   versioned column tag from `Score::metric_version`, then
///   `executor::build_output_columns` keys it under
///   `cvvdp_gpu::CVVDP_COLUMN_NAME`.
pub fn metric_orchestrator_eligible(_kind: CliMetricKind) -> bool {
    true
}

/// Build the orchestrator at the start of a CLI command. Wraps the
/// glue helper with a uniform error type so subcommand handlers don't
/// need to import the glue module directly.
pub fn build_orchestrator(
    opts: &OrchestratorRuntimeOpts,
) -> Result<Orchestrator, Box<dyn std::error::Error>> {
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

    /// Phase 7.7.1: iwssim's versioned column gets re-keyed back to the
    /// legacy `iwssim_gpu` (GPU CLI variant) / `iwssim` (CPU CLI
    /// variant). This is the parity fix — the orchestrator path emits
    /// the versioned name that `zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME`
    /// returns; the legacy `MetricCache` path emits the unversioned
    /// `iwssim_gpu`. Production parquet readers depend on the legacy
    /// column name, so the orchestrator path harmonises on it.
    #[test]
    #[cfg(feature = "gpu-iwssim")]
    fn rekey_orchestrator_columns_phase_7_7_1_renames_iwssim() {
        use std::collections::BTreeMap;
        // The exact versioned key depends on `IWSSIM_IMPL_TAG` /
        // `CARGO_PKG_VERSION`. Build the key dynamically from the
        // constant so this test tracks the source of truth.
        let versioned = zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME.to_string();
        let mut cols = BTreeMap::new();
        cols.insert(versioned.clone(), 0.952_f64);

        let gpu_rekeyed = rekey_orchestrator_columns(CliMetricKind::IwssimGpu, &cols);
        assert_eq!(
            gpu_rekeyed,
            vec![("iwssim_gpu".to_string(), 0.952)],
            "iwssim-gpu should re-key versioned column to legacy iwssim_gpu"
        );

        // The bare `Iwssim` CLI variant is a passthrough — legacy
        // CLI's matching arm ALSO emits the versioned name, so
        // there's nothing to re-key. See the inline comment in
        // `rekey_orchestrator_columns` for the source-of-truth path.
        let cpu_rekeyed = rekey_orchestrator_columns(CliMetricKind::Iwssim, &cols);
        assert_eq!(
            cpu_rekeyed,
            vec![(versioned.clone(), 0.952)],
            "iwssim (bare CLI variant) should pass versioned column through"
        );
    }

    #[test]
    fn every_metric_kind_is_orchestrator_eligible() {
        // Butteraugli was the last holdout (Phase 7.7.1 reverted it
        // when `ButteraugliOpaque`'s Auto resolver still dropped to a
        // single-resolution walker, ~14-30 % off multires — see
        // `benchmarks/orchestrator_parity_2026-05-27_phase771_run2.csv`).
        // butteraugli-gpu `77020757` routed opaque Strip to
        // `new_multires_strip` with `HALO_ROWS = 80`, and the 6,810-cell
        // corpus measurement in
        // `benchmarks/butter_strip_drift_2026-08-28.{csv,md}` shows that
        // walker is BIT-IDENTICAL to `new_multires` on the max-norm
        // score (0.00e0 in 6,810/6,810) and <= 1.72e-6 rel on pnorm_3 —
        // inside the <= 5e-5 Phase 7.7.1 gate. Accepted tolerance and
        // the scope of that evidence (wgpu/Metal; CUDA sweep still
        // outstanding) are documented on
        // `metric_orchestrator_eligible` itself.
        assert!(metric_orchestrator_eligible(CliMetricKind::Butteraugli));
        assert!(metric_orchestrator_eligible(CliMetricKind::ButteraugliGpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Cvvdp));
        assert!(metric_orchestrator_eligible(CliMetricKind::CvvdpGpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2));
        assert!(metric_orchestrator_eligible(CliMetricKind::Ssim2Gpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Dssim));
        assert!(metric_orchestrator_eligible(CliMetricKind::DssimGpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Iwssim));
        assert!(metric_orchestrator_eligible(CliMetricKind::IwssimGpu));
        assert!(metric_orchestrator_eligible(CliMetricKind::Zensim));
        assert!(metric_orchestrator_eligible(CliMetricKind::ZensimGpu));
    }
}
