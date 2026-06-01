#![forbid(unsafe_code)]

//! Metric backend dispatch.
//!
//! `MetricKind` is the user-facing enum (parsed from the CLI's `--metric`
//! flag). `run_metric` resolves it to the actual backend and runs the
//! comparison on the supplied RGB8 image pair.
//!
//! ## GPU path: one umbrella, one switch
//!
//! Every GPU metric routes through the [`zenmetrics_api`] umbrella crate.
//! The umbrella exposes an opaque `Metric::new(MetricKind, Backend, w, h,
//! params)` constructor and a `Metric::compute_srgb_u8(&r, &d) -> Score`
//! method that hide the per-crate `<Metric><R: Runtime>::new` /
//! `<Metric><R: Runtime>::compute` typed surface. The CLI keeps a single
//! `match kind { ... }` to translate its [`MetricKind`] into the
//! umbrella's `MetricKind` + per-metric default `MetricParams`. There is
//! no per-metric `gpu` module any more; one [`run_gpu_via_umbrella`]
//! helper handles ssim2/dssim/iwssim/zensim/cvvdp single-shot and butter
//! max-norm scoring.
//!
//! Two typed-path escape hatches stay for the cases the opaque API does
//! not cover today:
//!
//! - [`butter_pnorm3`] uses the typed `butteraugli_gpu::Butteraugli<R>`
//!   pipeline (reached via the umbrella's `zenmetrics_api::butter`
//!   re-export) to extract the libjxl 3-norm aggregation
//!   (`ButteraugliResult.pnorm_3`) alongside the max-norm in one
//!   `compute()` call. The umbrella's opaque `Score` only carries the
//!   max-norm value.
//! - [`cvvdp_gpu`] (this directory) keeps `CvvdpBatchScorer`, which
//!   caches a `cvvdp_gpu::Cvvdp<R>` instance across pairs of matching
//!   dims to avoid the ~200 MB / NVRTC compile per-pair cost (fleet
//!   OOMs at 100-pair chunks otherwise). The instance type comes from
//!   the umbrella's `zenmetrics_api::cvvdp` re-export, so the CLI
//!   still has no direct `cvvdp-gpu` dependency.

use clap::ValueEnum;

use crate::decode::Rgb8Image;

#[cfg(feature = "cpu-metrics")]
mod butteraugli;
#[cfg(feature = "cpu-metrics")]
mod dssim;
#[cfg(feature = "cpu-metrics")]
mod ssim2;
#[cfg(feature = "cpu-metrics")]
mod zensim;

#[cfg(feature = "gpu-butteraugli")]
mod butter_pnorm3;
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) mod cache;
#[cfg(feature = "gpu-cvvdp")]
pub mod cvvdp_gpu;

/// Metric identifier exposed on the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MetricKind {
    /// SSIMULACRA2 — CPU implementation via the `ssimulacra2` crate.
    #[value(name = "ssim2")]
    Ssim2,
    /// SSIMULACRA2 — GPU implementation via the `ssim2-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    #[value(name = "ssim2-gpu")]
    Ssim2Gpu,
    /// Butteraugli — CPU implementation via the `butteraugli` crate.
    /// Emits **two** columns per cell: `butteraugli_max`
    /// (`ButteraugliResult::score`, the per-block maximum) and
    /// `butteraugli_pnorm3` (`ButteraugliResult::pnorm_3`, the libjxl-style
    /// 3-norm aggregation matching `butteraugli_main --pnorm` and the
    /// Cloudinary CID22 paper). One `compute()` call yields both numbers,
    /// so emitting both is free.
    #[value(name = "butteraugli")]
    Butteraugli,
    /// Butteraugli — GPU implementation via the `butteraugli-gpu` crate.
    /// Emits two columns: `butteraugli_max_gpu` and `butteraugli_pnorm3_gpu`.
    /// Same single-`compute()` rationale as the CPU variant.
    #[value(name = "butteraugli-gpu")]
    ButteraugliGpu,
    /// DSSIM — CPU implementation via the `dssim-core` crate. Distance metric: 0 = identical.
    #[value(name = "dssim")]
    Dssim,
    /// DSSIM — GPU implementation via the `dssim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    /// Distance metric: 0 = identical.
    #[value(name = "dssim-gpu")]
    DssimGpu,
    /// IW-SSIM — GPU implementation via the `iwssim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    /// Range `[0, 1]`; 1.0 = identical.
    #[value(name = "iwssim-gpu")]
    IwssimGpu,
    /// Zensim — CPU implementation via the `zensim` crate.
    #[value(name = "zensim")]
    Zensim,
    /// Zensim — GPU implementation via the `zensim-gpu` crate
    /// (dispatched through the `zenmetrics-api` umbrella).
    #[value(name = "zensim-gpu")]
    ZensimGpu,
    /// ColorVideoVDP (still-image, JOD scale 0–10, 10 = imperceptible) via
    /// the `cvvdp-gpu` crate (dispatched through the umbrella by default;
    /// `score-pairs` uses the typed `CvvdpBatchScorer` for instance
    /// reuse across pairs).
    #[value(name = "cvvdp")]
    Cvvdp,
    /// IW-SSIM (Information-Content Weighted SSIM, Wang & Li 2011) via
    /// the `iwssim-gpu` crate. Score in `[0, 1]` where 1 = identical.
    /// Requires `min(W, H) >= 176` per the paper's 5-level pyramid + 11×11
    /// valid-mode SSIM stats constraint; smaller images return an error.
    #[value(name = "iwssim")]
    Iwssim,
}

impl MetricKind {
    pub fn all() -> &'static [MetricKind] {
        &[
            MetricKind::Ssim2,
            MetricKind::Ssim2Gpu,
            MetricKind::Butteraugli,
            MetricKind::ButteraugliGpu,
            MetricKind::Dssim,
            MetricKind::DssimGpu,
            MetricKind::IwssimGpu,
            MetricKind::Zensim,
            MetricKind::ZensimGpu,
            MetricKind::Cvvdp,
            MetricKind::Iwssim,
        ]
    }

    pub fn name(self) -> &'static str {
        match self {
            MetricKind::Ssim2 => "ssim2",
            MetricKind::Ssim2Gpu => "ssim2-gpu",
            MetricKind::Butteraugli => "butteraugli",
            MetricKind::ButteraugliGpu => "butteraugli-gpu",
            MetricKind::Dssim => "dssim",
            MetricKind::DssimGpu => "dssim-gpu",
            MetricKind::IwssimGpu => "iwssim-gpu",
            MetricKind::Zensim => "zensim",
            MetricKind::ZensimGpu => "zensim-gpu",
            MetricKind::Cvvdp => "cvvdp",
            MetricKind::Iwssim => "iwssim",
        }
    }

    pub fn backend(self) -> &'static str {
        match self {
            MetricKind::Ssim2Gpu
            | MetricKind::ButteraugliGpu
            | MetricKind::DssimGpu
            | MetricKind::IwssimGpu
            | MetricKind::ZensimGpu
            | MetricKind::Cvvdp
            | MetricKind::Iwssim => "GPU",
            _ => "CPU",
        }
    }

    pub fn requires_gpu(self) -> bool {
        matches!(
            self,
            MetricKind::Ssim2Gpu
                | MetricKind::ButteraugliGpu
                | MetricKind::DssimGpu
                | MetricKind::IwssimGpu
                | MetricKind::ZensimGpu
                | MetricKind::Cvvdp
                | MetricKind::Iwssim
        )
    }

    /// Static list of TSV / parquet column suffixes emitted by this metric,
    /// in the order [`run_metric`] produces them. For most metrics this is a
    /// single column matching [`MetricKind::name`] with `-` rewritten to
    /// `_`. Butteraugli (CPU and GPU) emits two columns — `_max` and
    /// `_pnorm3` — because one `compute()` call yields both aggregations.
    ///
    /// Headers in the sweep TSV are formatted as `score_<column>` for each
    /// entry returned here.
    pub fn column_names(self) -> &'static [&'static str] {
        match self {
            MetricKind::Ssim2 => &["ssim2"],
            MetricKind::Ssim2Gpu => &["ssim2_gpu"],
            MetricKind::Butteraugli => &["butteraugli_max", "butteraugli_pnorm3"],
            MetricKind::ButteraugliGpu => &["butteraugli_max_gpu", "butteraugli_pnorm3_gpu"],
            MetricKind::Dssim => &["dssim"],
            MetricKind::DssimGpu => &["dssim_gpu"],
            MetricKind::IwssimGpu => &["iwssim_gpu"],
            MetricKind::Zensim => &["zensim"],
            MetricKind::ZensimGpu => &["zensim_gpu"],
            MetricKind::Cvvdp => CVVDP_COLUMNS,
            MetricKind::Iwssim => IWSSIM_COLUMNS,
        }
    }
}

// Versioned cvvdp column name. With the `gpu-cvvdp` feature enabled,
// pulls the per-implementation tag from
// [`zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME`] (defaults to
// `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>`; overridable at build time
// via `CVVDP_IMPL_TAG`). Without the feature, falls back to a bare
// `"cvvdp"` so callers that list metric names without invoking the
// backend still see a usable identifier.
//
// Why versioned: parquet sidecars store cvvdp scores from multiple
// implementations side-by-side (e.g. `cvvdp_pycvvdp_v054` for the
// pycvvdp reference, `cvvdp_imazen_v0_0_1` for this crate's host
// scalar path). A bare `cvvdp` column would collide on join. See the
// PINNED TASK section in the repo-root `CLAUDE.md`.
#[cfg(feature = "gpu-cvvdp")]
const CVVDP_COLUMNS: &[&str] = &[zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME];
#[cfg(not(feature = "gpu-cvvdp"))]
const CVVDP_COLUMNS: &[&str] = &["cvvdp"];

// Versioned iwssim column name — mirrors the cvvdp pattern. With the
// `gpu-iwssim` feature on, pulls from `::iwssim_gpu::IWSSIM_COLUMN_NAME`
// (default `iwssim_imazen_v<MAJOR>_<MINOR>_<PATCH>`, overridable at
// build time via `IWSSIM_IMPL_TAG`). Without the feature, falls back
// to a bare `iwssim` so callers listing metric names without the
// backend still see a usable identifier.
#[cfg(feature = "gpu-iwssim")]
const IWSSIM_COLUMNS: &[&str] = &[zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME];
#[cfg(not(feature = "gpu-iwssim"))]
const IWSSIM_COLUMNS: &[&str] = &["iwssim"];

/// CubeCL runtime selector for GPU metrics.
///
/// Maps onto [`zenmetrics_api::Backend`] inside [`run_gpu_via_umbrella`];
/// kept on the CLI side as its own enum so the `--gpu-runtime` flag
/// surface stays stable and the existing `auto` discovery logic does
/// not leak through to callers that only want a fixed backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GpuRuntime {
    /// First runtime that initialises successfully.
    Auto,
    /// CUDA. Requires the `gpu-cuda` cargo feature.
    Cuda,
    /// wgpu (Vulkan / Metal / DX12 / WebGPU). Requires `gpu-wgpu`.
    Wgpu,
    /// AMD HIP / ROCm. Requires `gpu-hip`.
    Hip,
    /// CPU-fallback runtime in CubeCL. Requires `gpu-cpu`.
    Cpu,
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn auto_order() -> &'static [GpuRuntime] {
    &[
        GpuRuntime::Cuda,
        GpuRuntime::Wgpu,
        GpuRuntime::Hip,
        GpuRuntime::Cpu,
    ]
}

#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn runtime_label(rt: GpuRuntime) -> &'static str {
    match rt {
        GpuRuntime::Auto => "auto",
        GpuRuntime::Cuda => "cuda",
        GpuRuntime::Wgpu => "wgpu",
        GpuRuntime::Hip => "hip",
        GpuRuntime::Cpu => "cpu",
    }
}

/// Translate a CLI [`GpuRuntime`] selection into the umbrella's
/// [`zenmetrics_api::Backend`]. `Auto` is expanded by the caller.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn gpu_runtime_to_backend(rt: GpuRuntime) -> Result<zenmetrics_api::Backend, String> {
    match rt {
        GpuRuntime::Auto => Err("Auto is expanded by the caller".to_string()),
        GpuRuntime::Cuda => Ok(zenmetrics_api::Backend::Cuda),
        GpuRuntime::Wgpu => Ok(zenmetrics_api::Backend::Wgpu),
        GpuRuntime::Hip => Ok(zenmetrics_api::Backend::Hip),
        // The umbrella's old `Backend::Cpu` (cubecl-cpu reference path)
        // is now `Backend::CubeclCpu`; the CLI's `GpuRuntime::Cpu` keeps
        // meaning "run the kernels on CPU via cubecl-cpu".
        GpuRuntime::Cpu => Ok(zenmetrics_api::Backend::CubeclCpu),
    }
}

/// Process-wide flag toggled by `score-pairs --allow-small-images`.
/// Read by [`resolve_default_params`] to switch iwssim into adaptive
/// reflect-pad mode at metric construction. Defaults to false; tests
/// never set it. Using `AtomicBool` so we can flip-then-spawn-threads
/// without a `&mut`.
static ALLOW_SMALL_IMAGES: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Set the process-wide allow-small-images flag. Called once from
/// `cmd_score_pairs` when `--allow-small-images` is on the CLI. Safe
/// to call from non-main if scoping changes.
pub fn set_allow_small_images() {
    ALLOW_SMALL_IMAGES.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Read the flag. Used by [`resolve_default_params`] and re-exposed
/// for the integration tests' debug paths.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn allow_small_images() -> bool {
    ALLOW_SMALL_IMAGES.load(std::sync::atomic::Ordering::Relaxed)
}

/// Resolve the default [`zenmetrics_api::MetricParams`] for a kind,
/// applying per-CLI overrides from the CLI's process-wide flags where
/// it exposes them. Currently this is just `--allow-small-images` for
/// the iwssim metric.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
pub(crate) fn resolve_default_params(
    kind: zenmetrics_api::MetricKind,
) -> Result<zenmetrics_api::MetricParams, zenmetrics_api::Error> {
    #[cfg(feature = "gpu-iwssim")]
    {
        if matches!(kind, zenmetrics_api::MetricKind::Iwssim) && allow_small_images() {
            return Ok(zenmetrics_api::MetricParams::Iwssim(
                zenmetrics_api::iwssim::IwssimParams::allow_small(true),
            ));
        }
    }
    zenmetrics_api::MetricParams::try_default_for(kind)
}

/// Single dispatch point for every GPU metric that fits the
/// "construct → `compute_srgb_u8` → unwrap one `Score`" shape:
/// ssim2-gpu, dssim-gpu, iwssim-gpu, zensim-gpu, cvvdp single-shot,
/// and butter max-norm. `auto` walks the compiled-in runtime list and
/// returns the first that produces a finite score. Replaces the
/// per-metric `score_with_runtime` / `run::<R>` cascade that used to
/// live one file per metric.
#[cfg(any(
    feature = "gpu-butteraugli",
    feature = "gpu-ssim2",
    feature = "gpu-dssim",
    feature = "gpu-iwssim",
    feature = "gpu-zensim",
    feature = "gpu-cvvdp"
))]
fn run_gpu_via_umbrella(
    umbrella_kind: zenmetrics_api::MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "{}: reference ({}×{}) and distorted ({}×{}) differ in size",
            umbrella_kind.tag(),
            reference.width,
            reference.height,
            distorted.width,
            distorted.height
        )
        .into());
    }
    let candidates: Vec<GpuRuntime> = match gpu_runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };
    // Collect EVERY runtime's failure so the user can see what each one
    // tried — surfacing only the last error hides the actual problem
    // when (e.g.) cuda fails for an interesting reason but cpu trips
    // on a feature-not-enabled error later in the fallback chain.
    let mut errors: Vec<String> = Vec::with_capacity(candidates.len());
    for rt in candidates {
        let backend = match gpu_runtime_to_backend(rt) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        // Per-metric default params from the umbrella. The CLI does
        // not expose per-metric tuning today — the score subcommand is
        // a "is the metric wired in" smoke test.
        //
        // Exception: for iwssim, honour the `IWSSIM_ALLOW_SMALL=1`
        // env var by switching to `IwssimParams::allow_small(true)`.
        // The score-pairs `--allow-small-images` flag sets that env var
        // in-process before invoking this dispatcher. Default behaviour
        // (env unset) is unchanged — `IwssimParams::DEFAULT` rejects
        // sub-176 inputs exactly as before.
        let params = match resolve_default_params(umbrella_kind) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        let metric = zenmetrics_api::Metric::new(
            umbrella_kind,
            backend,
            reference.width,
            reference.height,
            params,
        );
        let mut m = match metric {
            Ok(m) => m,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        match m.compute_srgb_u8(&reference.pixels, &distorted.pixels) {
            Ok(score) => {
                if !score.value.is_finite() {
                    errors.push(format!(
                        "{}: non-finite score {}",
                        runtime_label(rt),
                        score.value
                    ));
                    continue;
                }
                return Ok(score.value);
            }
            Err(e) => errors.push(format!("{}: {e}", runtime_label(rt))),
        }
    }
    Err(format!(
        "{}: no runtime succeeded; tried [{}]",
        umbrella_kind.tag(),
        if errors.is_empty() {
            "none".to_string()
        } else {
            errors.join("; ")
        }
    )
    .into())
}

/// Run `kind` on a `(reference, distorted)` RGB8 pair. GPU metrics route
/// `gpu_runtime` through the CubeCL backend; CPU metrics ignore it.
///
/// Returns `(column_name, value)` pairs in the order declared by
/// [`MetricKind::column_names`]. For most metrics this is a single pair;
/// for butteraugli (CPU and GPU) it is two pairs (max-norm + 3-norm) yielded
/// from a single `compute()` call so callers don't pay twice.
pub fn run_metric(
    kind: MetricKind,
    #[cfg_attr(
        not(any(
            feature = "cpu-metrics",
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    reference: &Rgb8Image,
    #[cfg_attr(
        not(any(
            feature = "cpu-metrics",
            feature = "gpu-butteraugli",
            feature = "gpu-ssim2",
            feature = "gpu-dssim",
            feature = "gpu-iwssim",
            feature = "gpu-zensim",
            feature = "gpu-cvvdp"
        )),
        allow(unused_variables)
    )]
    distorted: &Rgb8Image,
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
    gpu_runtime: GpuRuntime,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    match kind {
        #[cfg(feature = "cpu-metrics")]
        MetricKind::Ssim2 => Ok(vec![("ssim2", ssim2::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Ssim2 => Err(disabled_msg("ssim2", "cpu-metrics")),

        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => Ok(vec![(
            "ssim2_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Ssim2,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-ssim2"))]
        MetricKind::Ssim2Gpu => Err(disabled_msg("ssim2-gpu", "gpu-ssim2")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Butteraugli => {
            let (max, pnorm3) = butteraugli::score_both(reference, distorted)?;
            Ok(vec![
                ("butteraugli_max", max),
                ("butteraugli_pnorm3", pnorm3),
            ])
        }
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Butteraugli => Err(disabled_msg("butteraugli", "cpu-metrics")),

        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => {
            // Butteraugli is the only GPU metric the CLI still drives
            // through the typed cubecl-types surface — the opaque
            // `Score` only carries the max-norm, but our TSV emits
            // both max-norm AND the libjxl 3-norm (`pnorm_3`) so the
            // sweep parquet schema stays unchanged. The typed call
            // reaches `ButteraugliResult.pnorm_3` directly. Lives in
            // `butter_pnorm3.rs` to keep this dispatch readable.
            let (max, pnorm3) = butter_pnorm3::score_both(reference, distorted, gpu_runtime)?;
            Ok(vec![
                ("butteraugli_max_gpu", max),
                ("butteraugli_pnorm3_gpu", pnorm3),
            ])
        }
        #[cfg(not(feature = "gpu-butteraugli"))]
        MetricKind::ButteraugliGpu => Err(disabled_msg("butteraugli-gpu", "gpu-butteraugli")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Dssim => Ok(vec![("dssim", dssim::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Dssim => Err(disabled_msg("dssim", "cpu-metrics")),

        #[cfg(feature = "gpu-dssim")]
        MetricKind::DssimGpu => Ok(vec![(
            "dssim_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Dssim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-dssim"))]
        MetricKind::DssimGpu => Err(disabled_msg("dssim-gpu", "gpu-dssim")),

        #[cfg(feature = "gpu-iwssim")]
        MetricKind::IwssimGpu => Ok(vec![(
            "iwssim_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Iwssim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-iwssim"))]
        MetricKind::IwssimGpu => Err(disabled_msg("iwssim-gpu", "gpu-iwssim")),

        #[cfg(feature = "cpu-metrics")]
        MetricKind::Zensim => Ok(vec![("zensim", zensim::score(reference, distorted)?)]),
        #[cfg(not(feature = "cpu-metrics"))]
        MetricKind::Zensim => Err(disabled_msg("zensim", "cpu-metrics")),

        #[cfg(feature = "gpu-zensim")]
        MetricKind::ZensimGpu => Ok(vec![(
            "zensim_gpu",
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Zensim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-zensim"))]
        MetricKind::ZensimGpu => Err(disabled_msg("zensim-gpu", "gpu-zensim")),

        #[cfg(feature = "gpu-cvvdp")]
        MetricKind::Cvvdp => Ok(vec![(
            zenmetrics_api::cvvdp::CVVDP_COLUMN_NAME,
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Cvvdp,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-cvvdp"))]
        MetricKind::Cvvdp => Err(disabled_msg("cvvdp", "gpu-cvvdp")),

        #[cfg(feature = "gpu-iwssim")]
        MetricKind::Iwssim => Ok(vec![(
            zenmetrics_api::iwssim::IWSSIM_COLUMN_NAME,
            run_gpu_via_umbrella(
                zenmetrics_api::MetricKind::Iwssim,
                reference,
                distorted,
                gpu_runtime,
            )?,
        )]),
        #[cfg(not(feature = "gpu-iwssim"))]
        MetricKind::Iwssim => Err(disabled_msg("iwssim", "gpu-iwssim")),
    }
}

#[allow(dead_code)]
fn disabled_msg(metric: &str, feature: &str) -> Box<dyn std::error::Error> {
    format!("metric '{metric}' is disabled (rebuild with `--features {feature}`)").into()
}

/// Run the zensim metric and additionally return the 300-feature extended
/// vector that the score is derived from. Only the zensim metric exposes a
/// feature vector — other metrics return `None` so callers can still use a
/// uniform call site. Callers that want feature output must therefore include
/// `MetricKind::Zensim` in their metric list.
///
/// Score values match what [`run_metric`] would return for `MetricKind::Zensim`,
/// so a sweep that scores zensim today and migrates to this entry point
/// later sees no shift in the TSV `score_zensim` column.
#[cfg(feature = "sweep")]
#[allow(unused_variables)]
pub fn run_zensim_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    #[cfg(feature = "cpu-metrics")]
    {
        zensim::score_with_features(reference, distorted)
    }
    #[cfg(not(feature = "cpu-metrics"))]
    {
        Err(disabled_msg("zensim", "cpu-metrics"))
    }
}

/// Selects which zensim feature regime the GPU path emits.
///
/// Maps onto [`zenmetrics_api::zensim::ZensimFeatureRegime`]. Each
/// variant determines the per-cell feature-vector width (228 / 300 /
/// 372) and the parquet sidecar's `feat_<i>` column count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ZensimFeatureRegime {
    /// 228 features — basic + peak only. Matches the legacy
    /// `compute_features` output of CPU zensim's default profile.
    Basic,
    /// 300 features — basic + peak + masked. Matches CPU
    /// `compute_extended_features` (pre-v26 sweep schema).
    Extended,
    /// 372 features — basic + peak + masked + IW. v26+ default for
    /// picker training; supersedes Extended.
    WithIw,
}

impl ZensimFeatureRegime {
    /// Feature-vector length (228 / 300 / 372).
    pub fn total_features(self) -> usize {
        match self {
            ZensimFeatureRegime::Basic => 228,
            ZensimFeatureRegime::Extended => 300,
            ZensimFeatureRegime::WithIw => 372,
        }
    }
}

#[cfg(feature = "gpu-zensim")]
impl From<ZensimFeatureRegime> for zenmetrics_api::zensim::ZensimFeatureRegime {
    fn from(r: ZensimFeatureRegime) -> Self {
        match r {
            ZensimFeatureRegime::Basic => {
                zenmetrics_api::zensim::ZensimFeatureRegime::Basic
            }
            ZensimFeatureRegime::Extended => {
                zenmetrics_api::zensim::ZensimFeatureRegime::Extended
            }
            ZensimFeatureRegime::WithIw => {
                zenmetrics_api::zensim::ZensimFeatureRegime::WithIw
            }
        }
    }
}

/// Run **GPU** zensim and return the score + the regime-appropriate
/// feature vector (228 / 300 / 372). Mirrors
/// [`run_zensim_with_features`] but goes through the GPU pipeline so
/// the encoded sweep doesn't pay the CPU-zensim cost twice (once for
/// the score column, once for the features).
///
/// The `gpu_runtime = Auto` case walks the compiled-in runtime list
/// and returns the first that produces a finite score.
#[cfg(feature = "sweep")]
#[cfg(feature = "gpu-zensim")]
#[allow(dead_code)] // superseded by `metrics::cache::MetricCache::compute_zensim_features`
                    // for the sweep path; retained as a library entry point.
pub fn run_zensim_gpu_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
    regime: ZensimFeatureRegime,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "zensim-gpu: reference ({}×{}) and distorted ({}×{}) differ in size",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let candidates: Vec<GpuRuntime> = match gpu_runtime {
        GpuRuntime::Auto => auto_order().to_vec(),
        other => vec![other],
    };
    let mut errors: Vec<String> = Vec::with_capacity(candidates.len());
    for rt in candidates {
        let backend = match gpu_runtime_to_backend(rt) {
            Ok(b) => b,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        // Construct ZensimParams with the requested regime + the
        // canonical default weights (so the basic-block score matches
        // `run_gpu_via_umbrella(MetricKind::Zensim, ...)` exactly).
        let zp = zenmetrics_api::zensim::ZensimParams::default_weights()
            .with_regime(regime.into());
        let params = zenmetrics_api::MetricParams::Zensim(zp);
        let metric = zenmetrics_api::Metric::new(
            zenmetrics_api::MetricKind::Zensim,
            backend,
            reference.width,
            reference.height,
            params,
        );
        let mut m = match metric {
            Ok(m) => m,
            Err(e) => {
                errors.push(format!("{}: {e}", runtime_label(rt)));
                continue;
            }
        };
        match m.compute_features_srgb_u8(&reference.pixels, &distorted.pixels) {
            Ok((score, features)) => {
                if !score.value.is_finite() {
                    errors.push(format!(
                        "{}: non-finite score {}",
                        runtime_label(rt),
                        score.value
                    ));
                    continue;
                }
                if features.len() != regime.total_features() {
                    errors.push(format!(
                        "{}: expected {} features, got {}",
                        runtime_label(rt),
                        regime.total_features(),
                        features.len()
                    ));
                    continue;
                }
                return Ok((score.value, features));
            }
            Err(e) => errors.push(format!("{}: {e}", runtime_label(rt))),
        }
    }
    Err(format!(
        "zensim-gpu: no runtime succeeded; tried [{}]",
        if errors.is_empty() {
            "none".to_string()
        } else {
            errors.join("; ")
        }
    )
    .into())
}

#[cfg(feature = "sweep")]
#[cfg(not(feature = "gpu-zensim"))]
#[allow(unused_variables)]
pub fn run_zensim_gpu_with_features(
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    gpu_runtime: GpuRuntime,
    regime: ZensimFeatureRegime,
) -> Result<(f64, Vec<f64>), Box<dyn std::error::Error>> {
    Err(disabled_msg("zensim-gpu", "gpu-zensim"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cvvdp_column_name_is_versioned_when_feature_on() {
        let cols = MetricKind::Cvvdp.column_names();
        assert_eq!(cols.len(), 1);
        // With `gpu-cvvdp` enabled, the column carries an
        // implementation tag so multiple cvvdp variants don't
        // collide in joined parquet sidecars. The user-facing CLI
        // flag (`--metric cvvdp`) remains stable.
        #[cfg(feature = "gpu-cvvdp")]
        {
            assert!(
                cols[0].starts_with("cvvdp_imazen_v")
                    || std::env::var("CVVDP_IMPL_TAG")
                        .map(|t| cols[0] == t)
                        .unwrap_or(false),
                "expected cvvdp column to start with cvvdp_imazen_v or match \
                 CVVDP_IMPL_TAG override, got {:?}",
                cols[0]
            );
        }
        #[cfg(not(feature = "gpu-cvvdp"))]
        {
            assert_eq!(cols[0], "cvvdp");
        }
    }

    #[test]
    fn cvvdp_cli_flag_name_is_stable() {
        // User-facing identifier stays "cvvdp" regardless of which
        // implementation is wired in — only the parquet column
        // changes.
        assert_eq!(MetricKind::Cvvdp.name(), "cvvdp");
    }
}
