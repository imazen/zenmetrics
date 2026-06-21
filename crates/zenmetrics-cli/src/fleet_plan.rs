//! `fleet-plan` — turn a sweep description into a recommended instance spec
//! and box count.
//!
//! This is the thin driver over [`zenfleet_core::recommend_instance`]: it
//! builds a [`zenfleet_core::CellCost`] per work item from
//!
//! - the **encode** side — `zencodec::estimate::ResourceEstimate` via a
//!   codec's `estimate_encode_resources` (when a codec feature is compiled
//!   in and `--codec` is given), or explicit `--encode-*` flags otherwise;
//!   and
//! - the **score** side — each GPU metric's pure-math
//!   `estimate_gpu_memory_bytes` + `estimate_score_time_ms` (reached through
//!   the `zenmetrics-api` re-exports, gated per `gpu-<metric>` feature).
//!
//! then prints the [`zenfleet_core::InstanceRecommendation`] (human text or
//! `--json`).
//!
//! ## Why the encode side can come from flags
//!
//! The codec `estimate_encode_resources` path requires the codec crates to
//! compile (the `sweep` + per-codec features). When they are not compiled —
//! or while a cross-repo codec/`zencodec` API skew blocks that build — the
//! planner still works: pass the encode footprint directly with
//! `--encode-peak-ram-mb`, `--encode-threads`, `--encode-ms` (read these off
//! a `ResourceEstimate` you obtained elsewhere, or estimate them). The
//! metric (score) side is the genuinely new estimator surface and is always
//! reachable under the relevant `gpu-<metric>` feature.

use std::error::Error;

use clap::Parser;

use crate::metrics::MetricKind;

/// One source-image size in the sweep (`WIDTHxHEIGHT`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl std::str::FromStr for Size {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (w, h) = s
            .split_once(['x', 'X', '*'])
            .ok_or_else(|| format!("size '{s}' must be WIDTHxHEIGHT (e.g. 1024x768)"))?;
        let width: u32 = w
            .trim()
            .parse()
            .map_err(|_| format!("bad width in '{s}'"))?;
        let height: u32 = h
            .trim()
            .parse()
            .map_err(|_| format!("bad height in '{s}'"))?;
        if width == 0 || height == 0 {
            return Err(format!("size '{s}' has a zero dimension"));
        }
        Ok(Size { width, height })
    }
}

/// The "tiny + small + medium + large" discipline default when `--sizes` is
/// omitted: per the size-sweep guidance, fixed overhead dominates at tiny and
/// per-pixel cost at large, so a default plan must span both ends.
const DEFAULT_SIZES: &[Size] = &[
    Size {
        width: 64,
        height: 64,
    },
    Size {
        width: 256,
        height: 256,
    },
    Size {
        width: 1024,
        height: 1024,
    },
    Size {
        width: 4096,
        height: 4096,
    },
];

#[derive(Parser, Debug)]
pub struct FleetPlanArgs {
    /// Source-image sizes to plan for, each `WIDTHxHEIGHT` (repeat or comma-
    /// separate). Defaults to the tiny/small/medium/large sweep buckets
    /// (64², 256², 1024², 4096²).
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    sizes: Vec<Size>,

    /// How many variants are encoded per source size (the `q × knob-tuple`
    /// grid cardinality). Each (size × variant) is one cell. Defaults to 1.
    #[arg(long, default_value = "1")]
    variants_per_size: u32,

    /// Metrics each cell is scored with (repeat or comma-separate). The
    /// cell's score VRAM is the max over these; its score time is the sum.
    /// Only GPU metrics carry an estimator; CPU-only metric kinds are
    /// rejected.
    #[arg(long, value_enum, value_delimiter = ',', num_args = 1..)]
    metrics: Vec<MetricKind>,

    /// Target wall-clock to finish the whole sweep within, seconds. Drives
    /// the recommended box count.
    #[arg(long, default_value = "3600")]
    target_wall_clock: f64,

    /// CPU cores per box for the instance class being sized.
    #[arg(long, default_value = "16")]
    cores_per_box: u32,

    // --- encode-side footprint (explicit, codec-free path) --------------
    /// Worst-case host RAM one encode needs, MiB
    /// (`ResourceEstimate::peak_memory_bytes_max`). Used when no `--codec`
    /// path is compiled in. Defaults to a light 128 MiB.
    #[arg(long, default_value = "128")]
    encode_peak_ram_mb: u64,
    /// CPU threads one encode uses
    /// (`ResourceEstimate::threading().effective_threads(cores)`).
    #[arg(long, default_value = "1")]
    encode_threads: u32,
    /// Encode wall time per cell, milliseconds (`ResourceEstimate::wall_ms`,
    /// optionally already `at_cores`-scaled). Defaults to 20 ms.
    #[arg(long, default_value = "20.0")]
    encode_ms: f32,

    /// Emit the recommendation as JSON instead of human-readable text.
    #[arg(long)]
    json: bool,
}

/// The score-side estimate for one metric at one size: `(vram_bytes,
/// time_ms)`. Dispatches to the per-crate pure-math estimators through the
/// `zenmetrics-api` re-exports; each arm is gated on the metric's
/// `gpu-<metric>` feature and errors with a rebuild hint when absent.
fn metric_score_estimate(
    metric: MetricKind,
    width: u32,
    height: u32,
) -> Result<(u64, f32), Box<dyn Error>> {
    // `width`/`height` are consumed only inside the `gpu-<metric>` cfg arms;
    // in a build with no GPU metric features every such arm is stripped and
    // they would read as unused. Discard explicitly (Copy, harmless when the
    // arms are present) to keep the `-D warnings` CI gate green.
    let _ = (width, height);
    match metric {
        #[cfg(feature = "gpu-cvvdp")]
        MetricKind::Cvvdp => {
            let r = zenmetrics_api::cvvdp::estimate_score_resources(width, height);
            Ok((r.vram_bytes as u64, r.time_ms))
        }
        #[cfg(feature = "gpu-ssim2")]
        MetricKind::Ssim2Gpu => {
            let r = zenmetrics_api::ssim2::estimate_score_resources(width, height);
            Ok((r.vram_bytes as u64, r.time_ms))
        }
        #[cfg(feature = "gpu-dssim")]
        MetricKind::DssimGpu => {
            let r = zenmetrics_api::dssim::estimate_score_resources(width, height);
            Ok((r.vram_bytes as u64, r.time_ms))
        }
        #[cfg(feature = "gpu-iwssim")]
        MetricKind::IwssimGpu => {
            let r = zenmetrics_api::iwssim::estimate_score_resources(width, height);
            Ok((r.vram_bytes as u64, r.time_ms))
        }
        #[cfg(feature = "gpu-butteraugli")]
        MetricKind::ButteraugliGpu => {
            let r = zenmetrics_api::butter::estimate_score_resources(width, height);
            Ok((r.vram_bytes as u64, r.time_ms))
        }
        #[cfg(feature = "gpu-zensim")]
        MetricKind::ZensimGpu => {
            // zensim's estimators take a feature regime; the default scalar
            // (Basic) path is what the sweep scores.
            let regime = zenmetrics_api::zensim::ZensimFeatureRegime::Basic;
            let vram = zenmetrics_api::zensim::estimate_gpu_memory_bytes(width, height, regime);
            let time = zenmetrics_api::zensim::estimate_score_time_ms(width, height);
            Ok((vram as u64, time))
        }
        // CPU-only metric kinds have no GPU working set / device-time model.
        MetricKind::Ssim2
        | MetricKind::Butteraugli
        | MetricKind::Dssim
        | MetricKind::Zensim
        | MetricKind::Iwssim => Err(format!(
            "metric '{metric:?}' is a CPU metric — fleet-plan models GPU score \
             cost; pass a -gpu metric (e.g. ssim2-gpu, cvvdp, zensim-gpu)"
        )
        .into()),
        // GPU metric requested but its feature wasn't compiled in.
        #[allow(unreachable_patterns)]
        other => Err(format!(
            "metric '{other:?}' GPU estimator is not compiled in; rebuild with \
             the matching --features gpu-<metric> (e.g. gpu-cvvdp, gpu-ssim2, \
             gpu-zensim, gpu-dssim, gpu-iwssim, gpu-butteraugli)"
        )
        .into()),
    }
}

/// Run `fleet-plan`: build per-cell costs, aggregate, print the
/// recommendation.
pub fn run(args: FleetPlanArgs) -> Result<(), Box<dyn Error>> {
    let sizes: Vec<Size> = if args.sizes.is_empty() {
        DEFAULT_SIZES.to_vec()
    } else {
        args.sizes.clone()
    };
    if args.metrics.is_empty() {
        return Err("at least one --metrics value is required".into());
    }
    let variants = args.variants_per_size.max(1);
    let encode_peak_ram_bytes = args.encode_peak_ram_mb.saturating_mul(1024 * 1024);

    // Build the CellCost list: one cell per (size × variant). The encode
    // footprint is per the explicit flags (codec-free path); the score side
    // is max(vram) / sum(time) over the cell's metrics.
    let mut cells: Vec<zenfleet_core::CellCost> = Vec::new();
    for size in &sizes {
        let mut score_vram_max: u64 = 0;
        let mut score_ms_sum: f32 = 0.0;
        for &m in &args.metrics {
            let (vram, time) = metric_score_estimate(m, size.width, size.height)?;
            score_vram_max = score_vram_max.max(vram);
            score_ms_sum += time;
        }
        let cell = zenfleet_core::CellCost::new(
            encode_peak_ram_bytes,
            args.encode_threads,
            args.encode_ms,
            score_vram_max,
            score_ms_sum,
        );
        for _ in 0..variants {
            cells.push(cell);
        }
    }

    let rec = zenfleet_core::recommend_instance(&cells, args.target_wall_clock, args.cores_per_box);

    if args.json {
        print_json(&rec, &sizes, variants, &args.metrics, cells.len());
    } else {
        print_human(&rec, &sizes, variants, &args.metrics, cells.len());
    }
    Ok(())
}

fn metric_names(metrics: &[MetricKind]) -> String {
    metrics
        .iter()
        .map(|m| format!("{m:?}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

fn print_human(
    rec: &zenfleet_core::InstanceRecommendation,
    sizes: &[Size],
    variants: u32,
    metrics: &[MetricKind],
    n_cells: usize,
) {
    let size_list = sizes
        .iter()
        .map(|s| format!("{}x{}", s.width, s.height))
        .collect::<Vec<_>>()
        .join(", ");
    println!("fleet-plan recommendation");
    println!("  manifest:");
    println!("    sizes              : {size_list}");
    println!("    variants per size  : {variants}");
    println!("    metrics            : {}", metric_names(metrics));
    println!("    total cells        : {n_cells}");
    println!("  per box:");
    println!("    cores              : {}", rec.cores);
    println!(
        "    host RAM           : {:.2} GiB ({} bytes)",
        gib(rec.host_ram_bytes),
        rec.host_ram_bytes
    );
    println!("    concurrent encodes : {}", rec.recommended_concurrency);
    println!("    needs GPU          : {}", rec.needs_gpu);
    if rec.needs_gpu {
        println!(
            "    GPU VRAM (card)    : {:.2} GiB ({} bytes){}",
            gib(rec.gpu_vram_bytes),
            rec.gpu_vram_bytes,
            if rec.vram_exceeds_sane_card {
                "  [exceeds a 24 GiB card → metric resolve_auto picks Strip mode]"
            } else {
                ""
            }
        );
    }
    println!("  fleet:");
    println!("    boxes              : {}", rec.box_count);
    println!(
        "    est wall-clock     : {:.1} s ({:.2} min)",
        rec.est_wall_clock_s,
        rec.est_wall_clock_s / 60.0
    );
    println!(
        "    total encode time  : {:.1} s (sum, pre-parallelism)",
        rec.total_encode_ms / 1000.0
    );
    println!(
        "    total score time   : {:.1} s (sum, serialized GPU)",
        rec.total_score_ms / 1000.0
    );
}

fn print_json(
    rec: &zenfleet_core::InstanceRecommendation,
    sizes: &[Size],
    variants: u32,
    metrics: &[MetricKind],
    n_cells: usize,
) {
    let size_list: Vec<String> = sizes
        .iter()
        .map(|s| format!("{}x{}", s.width, s.height))
        .collect();
    let v = serde_json::json!({
        "manifest": {
            "sizes": size_list,
            "variants_per_size": variants,
            "metrics": metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>(),
            "total_cells": n_cells,
        },
        "per_box": {
            "cores": rec.cores,
            "host_ram_bytes": rec.host_ram_bytes,
            "recommended_concurrency": rec.recommended_concurrency,
            "needs_gpu": rec.needs_gpu,
            "gpu_vram_bytes": rec.gpu_vram_bytes,
            "vram_exceeds_sane_card": rec.vram_exceeds_sane_card,
        },
        "fleet": {
            "box_count": rec.box_count,
            "est_wall_clock_s": rec.est_wall_clock_s,
            "total_encode_ms": rec.total_encode_ms,
            "total_score_ms": rec.total_score_ms,
        }
    });
    println!("{}", serde_json::to_string_pretty(&v).unwrap_or_default());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_parses_wxh() {
        assert_eq!(
            "1024x768".parse::<Size>().unwrap(),
            Size {
                width: 1024,
                height: 768
            }
        );
        assert_eq!(
            "64X64".parse::<Size>().unwrap(),
            Size {
                width: 64,
                height: 64
            }
        );
        assert!("1024".parse::<Size>().is_err());
        assert!("0x10".parse::<Size>().is_err());
        assert!("ax b".parse::<Size>().is_err());
    }

    #[test]
    fn cpu_metric_kinds_are_rejected() {
        // CPU-only kinds have no GPU device cost — must error clearly.
        let e = metric_score_estimate(MetricKind::Ssim2, 256, 256).unwrap_err();
        assert!(e.to_string().contains("CPU metric"));
    }
}
