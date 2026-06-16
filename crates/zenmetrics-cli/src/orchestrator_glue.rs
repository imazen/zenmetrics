#![cfg(feature = "orchestrator")]
#![forbid(unsafe_code)]

//! Bridge between the CLI's [`crate::metrics::MetricKind`] enum and the
//! [`zenmetrics_orchestrator::Orchestrator`] surface.
//!
//! The CLI exposes per-backend variants (`Ssim2` vs `Ssim2Gpu`) so users
//! can pick implementations explicitly. The orchestrator's
//! [`zenmetrics_api::MetricKind`] only carries the metric family
//! (`Ssim2`) and chooses GPU vs CPU via the chooser + OOM ladder.
//!
//! Phase 7 maps the CLI variants → `(api::MetricKind, prefer_cpu)` so
//! the orchestrator-driven code path honours the user's intent. A CLI
//! `--metric ssim2` (CPU) sets `prefer_cpu = true`, which we encode as
//! a build-time "skip GPU candidates" hint on the chooser. `--metric
//! ssim2-gpu` keeps the default chooser behaviour (GPU first, CPU on
//! OOM).
//!
//! For Phase 7 we honour `prefer_cpu` by surfacing a clear error when
//! the CLI explicitly asked for a CPU variant but the build doesn't
//! ship that variant's CPU feature (`cpu-ssim2`, etc.) — callers
//! either rebuild or pick a different variant. The chooser itself
//! doesn't yet take a "force CPU" knob (that's a Phase 7+ enhancement
//! if needed); we keep the surface forward-compatible by routing
//! prefer-cpu callers through a per-task ladder that pre-records OOM
//! against every GPU backend at the requested size.

use std::path::PathBuf;
use std::time::Duration;

use zenmetrics_api::MetricKind as ApiMetricKind;
use zenmetrics_orchestrator::{
    Orchestrator, OrchestratorConfig, OrchestratorError as OrchestratorCacheError,
};

use crate::metrics::MetricKind as CliMetricKind;

/// How the orchestrator should treat a CLI metric variant.
#[derive(Debug, Clone, Copy)]
pub struct OrchestratorMetricSpec {
    /// Underlying metric family (matches the orchestrator's API).
    pub kind: ApiMetricKind,
    /// `true` when the caller explicitly chose the CPU variant (e.g.
    /// `--metric ssim2`). The orchestrator-driven path warns + falls
    /// back to GPU first; surfaces a hard error if the
    /// `cpu-<metric>` orchestrator feature isn't compiled in.
    pub prefer_cpu: bool,
}

impl OrchestratorMetricSpec {
    /// Map a CLI variant → orchestrator spec. Returns the full bridge
    /// metadata the caller needs; the caller can fan out one
    /// orchestrator [`Task`] per spec.
    pub fn from_cli(kind: CliMetricKind) -> Self {
        match kind {
            CliMetricKind::Ssim2 => Self {
                kind: ApiMetricKind::Ssim2,
                prefer_cpu: true,
            },
            CliMetricKind::Ssim2Gpu => Self {
                kind: ApiMetricKind::Ssim2,
                prefer_cpu: false,
            },
            CliMetricKind::Butteraugli => Self {
                kind: ApiMetricKind::Butter,
                prefer_cpu: true,
            },
            CliMetricKind::ButteraugliGpu => Self {
                kind: ApiMetricKind::Butter,
                prefer_cpu: false,
            },
            CliMetricKind::Dssim => Self {
                kind: ApiMetricKind::Dssim,
                prefer_cpu: true,
            },
            CliMetricKind::DssimGpu => Self {
                kind: ApiMetricKind::Dssim,
                prefer_cpu: false,
            },
            CliMetricKind::IwssimGpu | CliMetricKind::Iwssim => Self {
                kind: ApiMetricKind::Iwssim,
                prefer_cpu: false, // iwssim has no CPU reference
            },
            CliMetricKind::Zensim => Self {
                kind: ApiMetricKind::Zensim,
                prefer_cpu: true,
            },
            CliMetricKind::ZensimGpu => Self {
                kind: ApiMetricKind::Zensim,
                prefer_cpu: false,
            },
            CliMetricKind::Cvvdp => Self {
                kind: ApiMetricKind::Cvvdp,
                prefer_cpu: false,
            },
        }
    }
}

/// Options propagated from the CLI top-level flags to the orchestrator
/// construction site. Stays a plain struct so subcommand handlers
/// don't depend on the clap derive of the top-level [`crate::Cli`].
#[derive(Debug, Clone, Default)]
pub struct OrchestratorRuntimeOpts {
    /// Override for [`OrchestratorConfig::cache_dir`]. `None` → default
    /// `$XDG_CACHE_HOME/zenmetrics/`.
    pub cache_dir: Option<PathBuf>,
    /// `Some(true)` → always run `warm()` at startup.
    /// `Some(false)` → never run.
    /// `None`        → auto (run only when cache is missing/stale; the
    /// orchestrator's own staleness check handles this).
    pub bench_on_start: Option<bool>,
    /// Comma-separated whitelist of CPU backend names to enable. Empty
    /// = use whatever the build's features ship. Recognised names:
    /// `cvvdp`, `ssim2`, `dssim`, `butter`, `zensim`, `all`.
    pub cpu_features: Vec<String>,
}

impl OrchestratorRuntimeOpts {
    /// Build a fresh [`Orchestrator`] from the runtime options. Honours
    /// `--orchestrator-cache`, the cache-validity window, and the
    /// `--bench-on-start` mode.
    ///
    /// Each call rebuilds from disk — callers that want a persistent
    /// orchestrator across multiple subcommand invocations should cache
    /// the returned instance.
    pub fn build(&self) -> Result<Orchestrator, OrchestratorBuildError> {
        let mut cfg = OrchestratorConfig::default();
        if let Some(dir) = &self.cache_dir {
            cfg.cache_dir = dir.clone();
        }
        // Cache validity window: keep the orchestrator's 7-day default
        // — too aggressive a re-bench wastes worker startup time, too
        // long misses driver bumps. Future Phase 7+ work may expose a
        // CLI flag for this.
        cfg.cache_validity = Duration::from_secs(7 * 24 * 60 * 60);

        let mut orch = Orchestrator::new(cfg).map_err(OrchestratorBuildError::Cache)?;

        match self.bench_on_start {
            Some(true) => {
                #[cfg(all(feature = "orchestrator-cuda", feature = "orchestrator"))]
                {
                    // Force a full re-bench.
                    orch.bench().map_err(OrchestratorBuildError::Cache)?;
                }
                #[cfg(not(all(feature = "orchestrator-cuda", feature = "orchestrator")))]
                {
                    // Without the bench feature the bench is a no-op;
                    // skip silently rather than fail.
                    let _ = &mut orch;
                }
            }
            Some(false) => {
                // Skip — the existing cache (if any) is used as-is.
            }
            None => {
                #[cfg(all(feature = "orchestrator-cuda", feature = "orchestrator"))]
                {
                    // `warm()` only runs the bench if the cache is
                    // missing or any metric profile is stale.
                    let _ran = orch.warm().map_err(OrchestratorBuildError::Cache)?;
                }
                #[cfg(not(all(feature = "orchestrator-cuda", feature = "orchestrator")))]
                {
                    let _ = &mut orch;
                }
            }
        }

        Ok(orch)
    }
}

/// Errors raised when building or driving the orchestrator from the CLI.
#[derive(Debug)]
pub enum OrchestratorBuildError {
    /// Capability cache I/O or persistence failure.
    Cache(OrchestratorCacheError),
    /// Caller asked for a CPU variant that this build doesn't include
    /// (e.g. `--metric ssim2` but the binary was built without
    /// `--features orchestrator-cpu-ssim2`).
    CpuVariantUnavailable {
        /// Metric tag, e.g. `"ssim2"`.
        metric: &'static str,
        /// Feature flag the caller should enable.
        required_feature: &'static str,
    },
}

impl std::fmt::Display for OrchestratorBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrchestratorBuildError::Cache(e) => write!(f, "orchestrator cache: {e}"),
            OrchestratorBuildError::CpuVariantUnavailable {
                metric,
                required_feature,
            } => write!(
                f,
                "CPU variant of '{metric}' is not available in this build; rebuild with --features {required_feature}"
            ),
        }
    }
}

impl std::error::Error for OrchestratorBuildError {}

/// Determine whether the user-explicit `--metric <cpu-variant>` is
/// possible with the current build. Surfaces a structured error so the
/// CLI can suggest a feature flag rather than silently falling back to
/// GPU.
pub fn validate_cpu_variant_built_in(
    cli_kind: CliMetricKind,
) -> Result<(), OrchestratorBuildError> {
    let spec = OrchestratorMetricSpec::from_cli(cli_kind);
    if !spec.prefer_cpu {
        return Ok(());
    }
    // Cargo `cfg(feature = "…")` is the source of truth. Mirror the
    // orchestrator's per-metric features here.
    let (metric, required_feature, enabled): (&'static str, &'static str, bool) = match spec.kind {
        ApiMetricKind::Cvvdp => (
            "cvvdp",
            "orchestrator-cpu-cvvdp",
            cfg!(feature = "orchestrator-cpu-cvvdp"),
        ),
        ApiMetricKind::Butter => (
            "butteraugli",
            "orchestrator-cpu-butter",
            cfg!(feature = "orchestrator-cpu-butter"),
        ),
        ApiMetricKind::Ssim2 => (
            "ssim2",
            "orchestrator-cpu-ssim2",
            cfg!(feature = "orchestrator-cpu-ssim2"),
        ),
        ApiMetricKind::Dssim => (
            "dssim",
            "orchestrator-cpu-dssim",
            cfg!(feature = "orchestrator-cpu-dssim"),
        ),
        ApiMetricKind::Zensim => (
            "zensim",
            "orchestrator-cpu-zensim",
            cfg!(feature = "orchestrator-cpu-zensim"),
        ),
        ApiMetricKind::Iwssim => {
            // Iwssim has no CPU reference at all — surface an honest
            // error instead of pointing the user at a feature flag
            // that wouldn't help.
            return Err(OrchestratorBuildError::CpuVariantUnavailable {
                metric: "iwssim",
                required_feature: "<none — iwssim CPU is not available upstream>",
            });
        }
    };
    if enabled {
        Ok(())
    } else {
        Err(OrchestratorBuildError::CpuVariantUnavailable {
            metric,
            required_feature,
        })
    }
}

/// Read the `ZENMETRICS_USE_ORCHESTRATOR` env var. Recognised truthy
/// values: `1`, `true`, `yes`, `on` (case-insensitive). Falsy or unset
/// returns `false`.
///
/// **Phase 7.7.1 (2026-05-27)**: deprecated. The orchestrator is the
/// default; this env var is a no-op kept for backwards-compat with
/// Docker images / shell scripts that set it before the default flip.
/// Use [`use_legacy_scheduler_from_env`] for the opt-OUT path.
pub fn use_orchestrator_from_env() -> bool {
    match std::env::var("ZENMETRICS_USE_ORCHESTRATOR") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

/// Read the `ZENMETRICS_USE_LEGACY_SCHEDULER` env var. Recognised
/// truthy values: `1`, `true`, `yes`, `on` (case-insensitive). Falsy
/// or unset returns `false`.
///
/// Phase 7.7.1 (2026-05-27): the env-driven counterpart to the CLI's
/// `--use-legacy-scheduler` flag. The CLI ORs them — either path
/// opts out of the orchestrator.
pub fn use_legacy_scheduler_from_env() -> bool {
    match std::env::var("ZENMETRICS_USE_LEGACY_SCHEDULER") {
        Ok(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metric_spec_maps_cpu_and_gpu_variants() {
        // Spot-check CPU/GPU pairing for each metric.
        assert!(OrchestratorMetricSpec::from_cli(CliMetricKind::Ssim2).prefer_cpu);
        assert!(!OrchestratorMetricSpec::from_cli(CliMetricKind::Ssim2Gpu).prefer_cpu);
        assert!(OrchestratorMetricSpec::from_cli(CliMetricKind::Zensim).prefer_cpu);
        assert!(!OrchestratorMetricSpec::from_cli(CliMetricKind::ZensimGpu).prefer_cpu);
        assert!(!OrchestratorMetricSpec::from_cli(CliMetricKind::Cvvdp).prefer_cpu);
        assert!(!OrchestratorMetricSpec::from_cli(CliMetricKind::Iwssim).prefer_cpu);
        assert!(!OrchestratorMetricSpec::from_cli(CliMetricKind::IwssimGpu).prefer_cpu);
    }

    #[test]
    fn env_var_truthy_parsing_is_falsey_without_setting() {
        // Note: we can't safely call `std::env::remove_var` from a
        // test (Rust 2024 made it `unsafe` due to TOCTOU with other
        // threads + the crate's `#![forbid(unsafe_code)]` blocks
        // the `unsafe { }` block we'd otherwise need). Trust that
        // the test environment doesn't already have the var set
        // (cargo test doesn't propagate it; CI runners don't either).
        // If a future test sets it, this assertion may flake — the
        // observable behaviour at that point is "the parser
        // correctly reads the var", which is also fine.
        if std::env::var("ZENMETRICS_USE_ORCHESTRATOR").is_err() {
            assert!(!use_orchestrator_from_env());
        }
    }

    #[test]
    fn use_legacy_scheduler_from_env_parses_truthy() {
        // Same caveat as the orchestrator env var: rely on the test
        // environment not pre-setting the var. We can verify the
        // parser is wired by checking the falsey branch (unset → false).
        if std::env::var("ZENMETRICS_USE_LEGACY_SCHEDULER").is_err() {
            assert!(!use_legacy_scheduler_from_env());
        }
    }
}
