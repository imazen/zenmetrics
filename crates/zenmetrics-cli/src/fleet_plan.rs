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
//! ## Encode side: codec estimate vs. flags
//!
//! With `--codec <name>` (under the `sweep` + matching per-codec feature) the
//! planner asks the codec itself for the encode footprint: it builds a
//! **representative default** `EncoderConfig` and calls
//! [`estimate_encode_resources`](zencodec::encode::EncoderConfig::estimate_encode_resources),
//! which returns a [`ResourceEstimate`](zencodec::estimate::ResourceEstimate)
//! whose `wall_ms` is already scaled to `--cores-per-box`. Six of the seven
//! codecs (jpeg / webp / avif / png / gif / jxl) carry a calibrated
//! `heuristics` model; zentiff returns a structural (uncalibrated) estimate.
//! The estimate uses one representative config per `--codec` — per-variant
//! config accuracy (driving the planner off an actual `--plan`) is a future
//! refinement.
//!
//! The explicit `--encode-peak-ram-mb` / `--encode-threads` / `--encode-ms`
//! flags are **overrides**: any flag that is set wins over the codec estimate
//! (and they are the only way to size the encode when no `--codec` path is
//! compiled in, e.g. a cross-repo codec/`zencodec` API skew, or a bare
//! CPU-metrics build). At least one of `--codec` or the encode flags must be
//! given. The metric (score) side is the genuinely new estimator surface and
//! is always reachable under the relevant `gpu-<metric>` feature.

use std::error::Error;

use clap::Parser;

use crate::metrics::MetricKind;
#[cfg(feature = "sweep")]
use crate::sweep::encode::CodecKind;

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

/// Parse a `--codec` value into the sweep's [`CodecKind`], accepting both the
/// short codec names the task expects (`jpeg`, `webp`, `avif`, `png`, `gif`,
/// `tiff`, `jxl`) and the `zen`-prefixed enum names (`zenjpeg`, …) that the
/// `sweep` subcommand's `ValueEnum` already uses. Reusing `CodecKind` keeps a
/// single codec vocabulary; this parser just gives `fleet-plan` the friendlier
/// short aliases without disturbing the sweep subcommand's accepted values.
#[cfg(feature = "sweep")]
fn parse_codec(s: &str) -> Result<CodecKind, String> {
    match s.to_ascii_lowercase().as_str() {
        "jpeg" | "zenjpeg" => Ok(CodecKind::Zenjpeg),
        "webp" | "zenwebp" => Ok(CodecKind::Zenwebp),
        "avif" | "zenavif" => Ok(CodecKind::Zenavif),
        "png" | "zenpng" => Ok(CodecKind::Zenpng),
        "gif" | "zengif" => Ok(CodecKind::Zengif),
        "tiff" | "zentiff" => Ok(CodecKind::Zentiff),
        "jxl" | "zenjxl" => Ok(CodecKind::Zenjxl),
        other => Err(format!(
            "unknown codec {other:?}; expected one of jpeg, webp, avif, png, gif, tiff, jxl"
        )),
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

    /// Codec to size the encode side for: `jpeg`, `webp`, `avif`, `png`,
    /// `gif`, `tiff`, or `jxl` (the `zen`-prefixed forms are also accepted).
    /// With this set, the planner builds a representative default
    /// `EncoderConfig` and asks the codec's `estimate_encode_resources` for
    /// the per-cell peak RAM / threads / wall time (already scaled to
    /// `--cores-per-box`). One representative config is used per codec —
    /// per-variant config accuracy (via an actual `--plan`) is a future
    /// refinement. Requires building with `--features sweep` plus the matching
    /// per-codec feature. The `--encode-*` flags below override individual
    /// fields.
    #[cfg(feature = "sweep")]
    #[arg(long, value_parser = parse_codec)]
    codec: Option<CodecKind>,

    // --- encode-side footprint (explicit overrides / codec-free path) ----
    //
    // These are `Option`: a set flag OVERRIDES the codec estimate for that
    // field; when no `--codec` is given they are the sole encode source (and
    // each unset field falls back to a light conservative default). Keeping
    // them `Option` is what lets "flag wins over estimate" work per-field.
    /// Override: worst-case host RAM one encode needs, MiB
    /// (`ResourceEstimate::peak_memory_bytes_max`). Wins over the `--codec`
    /// estimate; without `--codec`, defaults to a light 128 MiB.
    #[arg(long)]
    encode_peak_ram_mb: Option<u64>,
    /// Override: CPU threads one encode uses
    /// (`ResourceEstimate::threading().effective_threads(cores)`). Wins over
    /// the `--codec` estimate; without `--codec`, defaults to 1.
    #[arg(long)]
    encode_threads: Option<u32>,
    /// Override: encode wall time per cell, milliseconds
    /// (`ResourceEstimate::wall_ms`, already `at_cores`-scaled). Wins over the
    /// `--codec` estimate; without `--codec`, defaults to 20 ms.
    #[arg(long)]
    encode_ms: Option<f32>,

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

/// The codec's encode estimate for one cell at one size: the per-field encode
/// cost the planner needs, plus whether the codec actually modelled it.
#[cfg(feature = "sweep")]
#[derive(Clone, Copy, Debug)]
struct CodecEncodeEstimate {
    /// Worst-case host RAM for one encode, bytes (max, falling back to est).
    peak_ram_bytes: u64,
    /// CPU threads the encode uses at `cores_per_box`.
    threads: u32,
    /// Encode wall time per cell, ms (already `cores`-scaled by the codec).
    ms: f32,
    /// `true` when the codec returned a real (calibrated or structural)
    /// estimate; `false` when it fell through to `ResourceEstimate::unknown()`
    /// (every field `None`), which means the planner must use the flags /
    /// conservative defaults instead.
    modelled: bool,
}

/// Build a representative default `EncoderConfig` for `codec` and ask it for
/// the encode resources at `width × height` on `cores` cores. The returned
/// [`ResourceEstimate`](zencodec::estimate::ResourceEstimate) is already
/// scaled to `cores` by the codec's override (each codec calls `at_cores`
/// internally), so `wall_ms` is read straight through. Each arm is gated on
/// the codec's cargo feature and constructs the codec's public,
/// trait-implementing config type at a representative default — NOT through a
/// `SweepVariant`, so it sizes the codec's typical encode independent of any
/// `--plan`.
#[cfg(feature = "sweep")]
fn codec_encode_estimate(
    codec: CodecKind,
    width: u32,
    height: u32,
    cores: u32,
) -> Result<CodecEncodeEstimate, Box<dyn Error>> {
    use zencodec::encode::EncoderConfig as _;
    use zencodec::estimate::{ComputeEnvironment, ImageCharacteristics};

    // RGB8 sRGB is the representative source format for the sweep (opaque
    // stills); a codec whose estimate cares about alpha still gets a sound
    // typical figure from the opaque path.
    let image = ImageCharacteristics::new(width, height, zenpixels::PixelDescriptor::RGB8_SRGB);
    let env = ComputeEnvironment::new().with_cores(cores as usize);

    // Each arm builds the codec's public trait-config type at a representative
    // mid-quality default and calls the trait method. `est` is `cores`-scaled.
    let est = match codec {
        #[cfg(feature = "jpeg")]
        CodecKind::Zenjpeg => {
            // Default = YCbCr 4:2:0 @ q85 (calibrated heuristics estimate).
            zenjpeg::JpegEncoderConfig::default().estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "webp")]
        CodecKind::Zenwebp => {
            // zenwebp re-exports its wrapper under the `zencodec` submodule
            // (not the crate root). No `Default`; `lossy()` is the
            // representative q75 method-4 path.
            zenwebp::zencodec::WebpEncoderConfig::lossy().estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "avif")]
        CodecKind::Zenavif => {
            // Default = quality 75, speed 4 (calibrated heuristics estimate).
            zenavif::AvifEncoderConfig::default().estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "png")]
        CodecKind::Zenpng => {
            // Default effort (calibrated heuristics; lossless — q is N/A).
            zenpng::PngEncoderConfig::default().estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "gif")]
        CodecKind::Zengif => {
            // Default quantizer backend (calibrated heuristics estimate).
            zengif::GifEncoderConfig::default().estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "tiff")]
        CodecKind::Zentiff => {
            // The trait-config type isn't re-exported at the crate root; reach
            // it via `zentiff::codec`. Estimate is structural (uncalibrated)
            // but returns real numbers (input + output + scratch, ~200 Mpix/s).
            zentiff::codec::TiffEncoderCodecConfig::default()
                .estimate_encode_resources(&image, &env)
        }
        #[cfg(feature = "jxl")]
        CodecKind::Zenjxl => {
            // Default = lossy distance 1.0 (calibrated via jxl-encoder).
            zenjxl::JxlEncoderConfig::default().estimate_encode_resources(&image, &env)
        }
        // Codec selected but its feature wasn't compiled in. Reachable in a
        // partial build (e.g. `--features sweep` without `jxl`/`png`/…).
        #[allow(unreachable_patterns)]
        other => {
            return Err(format!(
                "codec {:?} is not compiled into this build; rebuild with the \
                 matching --features (zenjpeg needs jpeg, zenwebp webp, zenavif \
                 avif, zenpng png, zengif gif, zentiff tiff, zenjxl jxl) — or \
                 supply the encode footprint with --encode-peak-ram-mb / \
                 --encode-threads / --encode-ms instead",
                other.name()
            )
            .into());
        }
    };

    // The estimate is `unknown()` (all-`None`) only if a codec did NOT override
    // the trait method; the six calibrated codecs + zentiff's structural arm
    // all fill at least `wall_ms`. Detect that so the caller can fall back.
    let modelled = est.peak_memory_bytes_max().is_some()
        || est.peak_memory_bytes_est().is_some()
        || est.wall_ms().is_some()
        || est.threading().is_some();

    // Per the task mapping (the refined-API accessors are all `Option`):
    //   peak  = peak_max OR peak_est OR 0
    //   ms    = wall_ms OR 0   (already `cores`-scaled by the codec override)
    //   thr   = threading.effective_threads(cores) OR 1
    let peak_ram_bytes = est
        .peak_memory_bytes_max()
        .or_else(|| est.peak_memory_bytes_est())
        .unwrap_or(0);
    let ms = est.wall_ms().unwrap_or(0) as f32;
    let threads = est
        .threading()
        .map(|t| t.effective_threads(cores as usize) as u32)
        .unwrap_or(1);

    Ok(CodecEncodeEstimate {
        peak_ram_bytes,
        threads,
        ms,
        modelled,
    })
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
    let cores = args.cores_per_box;

    // The `--codec` flag only exists in a `sweep` build; without it the
    // selector is always `None` and the encode side must come from the flags.
    #[cfg(feature = "sweep")]
    let codec = args.codec;
    #[cfg(not(feature = "sweep"))]
    let codec: Option<()> = None;

    // Require at least one encode source: a `--codec` (codec estimate) or any
    // of the `--encode-*` override flags. Neither = the planner has no encode
    // footprint to plan with, so error clearly rather than silently using a
    // light default the caller didn't ask for.
    let any_encode_flag = args.encode_peak_ram_mb.is_some()
        || args.encode_threads.is_some()
        || args.encode_ms.is_some();
    if codec.is_none() && !any_encode_flag {
        #[cfg(feature = "sweep")]
        let hint = "provide --codec <name> (jpeg/webp/avif/png/gif/tiff/jxl) or \
                    the --encode-peak-ram-mb / --encode-threads / --encode-ms flags";
        #[cfg(not(feature = "sweep"))]
        let hint = "provide the --encode-peak-ram-mb / --encode-threads / \
                    --encode-ms flags (or rebuild with --features sweep plus a \
                    per-codec feature to use --codec <name>)";
        return Err(hint.into());
    }

    // Light conservative defaults for any encode field left unspecified when
    // there is no `--codec` estimate to fall back to (mirrors the old flag
    // defaults: 128 MiB peak, single-thread, 20 ms).
    let flag_peak_ram_bytes = args
        .encode_peak_ram_mb
        .map(|mb| mb.saturating_mul(1024 * 1024));

    // Build the CellCost list: one cell per (size × variant). The encode
    // footprint per size comes from the `--codec` estimate (when set),
    // overlaid by any `--encode-*` flag (flag wins per-field); the score side
    // is max(vram) / sum(time) over the cell's metrics.
    let mut cells: Vec<zenfleet_core::CellCost> = Vec::new();
    // Tracks whether the `--codec` estimate was actually modelled (vs an
    // all-`None` `unknown()` fall-through) — reported in the summary.
    #[cfg(feature = "sweep")]
    let mut codec_modelled: Option<bool> = None;
    for size in &sizes {
        // Encode side: start from the codec estimate (if any), then let each
        // set flag override its field. In a non-`sweep` build there is no
        // codec path, so the base is always unset and only the flags apply.
        #[cfg(feature = "sweep")]
        let (base_peak, base_threads, base_ms) = match codec {
            Some(c) => {
                let e = codec_encode_estimate(c, size.width, size.height, cores)?;
                codec_modelled = Some(codec_modelled.unwrap_or(true) && e.modelled);
                (Some(e.peak_ram_bytes), Some(e.threads), Some(e.ms))
            }
            None => (None, None, None),
        };
        #[cfg(not(feature = "sweep"))]
        let (base_peak, base_threads, base_ms): (Option<u64>, Option<u32>, Option<f32>) =
            (None, None, None);

        let encode_peak_ram_bytes = flag_peak_ram_bytes
            .or(base_peak)
            .unwrap_or(128 * 1024 * 1024);
        let encode_threads = args.encode_threads.or(base_threads).unwrap_or(1);
        let encode_ms = args.encode_ms.or(base_ms).unwrap_or(20.0);

        let mut score_vram_max: u64 = 0;
        let mut score_ms_sum: f32 = 0.0;
        for &m in &args.metrics {
            let (vram, time) = metric_score_estimate(m, size.width, size.height)?;
            score_vram_max = score_vram_max.max(vram);
            score_ms_sum += time;
        }
        let cell = zenfleet_core::CellCost::new(
            encode_peak_ram_bytes,
            encode_threads,
            encode_ms,
            score_vram_max,
            score_ms_sum,
        );
        for _ in 0..variants {
            cells.push(cell);
        }
    }

    let rec = zenfleet_core::recommend_instance(&cells, args.target_wall_clock, cores);

    // Describe where the encode footprint came from (codec estimate, override
    // flags, or both) for the summary. Only consumed by the `sweep`-gated
    // `encode_source` below, so gate the binding too — otherwise it is an
    // unused variable under `--no-default-features` (CI clippy is -D warnings).
    #[cfg(feature = "sweep")]
    let overridden: Vec<&str> = [
        args.encode_peak_ram_mb.map(|_| "peak-ram"),
        args.encode_threads.map(|_| "threads"),
        args.encode_ms.map(|_| "wall-ms"),
    ]
    .into_iter()
    .flatten()
    .collect();
    #[cfg(feature = "sweep")]
    let encode_source = match codec {
        Some(c) if overridden.is_empty() => {
            format!("codec estimate ({})", c.name())
        }
        Some(c) => format!(
            "codec estimate ({}), overridden flags: {}",
            c.name(),
            overridden.join(", ")
        ),
        None => "explicit --encode-* flags".to_string(),
    };
    #[cfg(not(feature = "sweep"))]
    let encode_source = "explicit --encode-* flags".to_string();
    #[cfg(feature = "sweep")]
    let codec_calibrated = codec_modelled;
    #[cfg(not(feature = "sweep"))]
    let codec_calibrated: Option<bool> = None;

    if args.json {
        print_json(
            &rec,
            &sizes,
            variants,
            &args.metrics,
            cells.len(),
            &encode_source,
            codec_calibrated,
        );
    } else {
        print_human(
            &rec,
            &sizes,
            variants,
            &args.metrics,
            cells.len(),
            &encode_source,
            codec_calibrated,
        );
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

#[allow(clippy::too_many_arguments)]
fn print_human(
    rec: &zenfleet_core::InstanceRecommendation,
    sizes: &[Size],
    variants: u32,
    metrics: &[MetricKind],
    n_cells: usize,
    encode_source: &str,
    codec_calibrated: Option<bool>,
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
    println!("    encode source      : {encode_source}");
    if let Some(modelled) = codec_calibrated {
        println!(
            "    codec estimate     : {}",
            if modelled {
                "modelled (calibrated or structural)"
            } else {
                "unknown() — codec did not model encode cost; using fallbacks"
            }
        );
    }
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

#[allow(clippy::too_many_arguments)]
fn print_json(
    rec: &zenfleet_core::InstanceRecommendation,
    sizes: &[Size],
    variants: u32,
    metrics: &[MetricKind],
    n_cells: usize,
    encode_source: &str,
    codec_calibrated: Option<bool>,
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
            "encode_source": encode_source,
            "codec_estimate_modelled": codec_calibrated,
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

    // The codec estimate path needs the `sweep` feature (CodecKind +
    // zencodec). zenjpeg is force-pulled by `sweep`, so it is always present
    // in a `sweep` test build.
    #[cfg(all(feature = "sweep", feature = "jpeg"))]
    #[test]
    fn codec_estimate_returns_real_numbers() {
        // jpeg carries a calibrated heuristics estimate: a 1024² encode must
        // produce non-zero peak RAM and wall time, and at least one thread.
        let e = codec_encode_estimate(CodecKind::Zenjpeg, 1024, 1024, 8).unwrap();
        assert!(e.modelled, "zenjpeg should model its encode cost");
        assert!(e.peak_ram_bytes > 0, "peak RAM must be a real figure");
        assert!(e.ms > 0.0, "wall time must be a real figure");
        assert!(e.threads >= 1, "at least one thread");
        // Larger images cost more wall time than tiny ones (monotone, sane).
        let small = codec_encode_estimate(CodecKind::Zenjpeg, 64, 64, 8).unwrap();
        assert!(
            e.ms >= small.ms,
            "1024² ({} ms) should cost at least as much as 64² ({} ms)",
            e.ms,
            small.ms
        );
    }
}
