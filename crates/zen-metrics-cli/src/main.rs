#![forbid(unsafe_code)]

//! `zen-metrics` — a focused CLI for decoding zen image formats and scoring
//! perceptual quality metrics on either CPU or GPU.
//!
//! Built so encoder rate-distortion experiments and selector training runs
//! can call one binary instead of stitching together half a dozen
//! per-metric scripts. Format support and metrics are gated behind cargo
//! features; see `Cargo.toml` for the feature matrix.
//!
//! Subcommands at a glance:
//! - `score` — one (reference, distorted) pair, one metric.
//! - `batch` — N rows from a TSV, one metric, output TSV with a score column.
//! - `compare` — M references × N variants × K metrics in one invocation,
//!   with each unique image decoded only once. Use this when you have one
//!   reference and several encoded variants you want scored across multiple
//!   metrics for picker / RD-curve work.
//! - `list-metrics` / `list-formats` — environment introspection.
//!
//! See `--help` on any subcommand for full options.

mod compare;
mod decode;
mod metrics;
mod output;

#[cfg(feature = "sweep")]
mod sweep;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
// `Path` is consumed only by `score_one_pair`, which lives behind
// the `sweep` feature. Under `--no-default-features --features wgpu`
// (CI's clippy invocation) `Path` would be an unused import and
// trip `-D warnings`. Gating the import alongside its sole caller
// keeps the wgpu-only build clean while the sweep build still gets
// it.
#[cfg(feature = "sweep")]
use std::path::Path;
use std::process::ExitCode;

use crate::compare::{print_report, run_compare};
use crate::metrics::{GpuRuntime, MetricKind, run_metric};
use crate::output::{OutputFormat, print_score};

/// Top-level CLI parser.
#[derive(Parser, Debug)]
#[command(version, author = "Lilith River", about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Score a single (reference, distorted) image pair.
    Score(ScoreArgs),
    /// Run a metric over a TSV of image pairs.
    Batch(BatchArgs),
    /// Score every (reference, variant) pair across multiple metrics in
    /// one invocation. Decodes each unique image once, then runs all
    /// metrics back-to-back per pair.
    Compare(CompareArgs),
    /// Drive a codec across a (image × q × knob-tuple) Cartesian grid
    /// and score each encoded variant with one or more metrics. Emits
    /// a Pareto TSV. Only available when the binary is built with
    /// `--features sweep`.
    #[cfg(feature = "sweep")]
    Sweep(SweepArgs),
    /// Score (ref, dist) pairs from a TSV and emit a parquet sidecar
    /// per `crates/cvvdp-gpu/docs/CVVDP_SIDECAR_SCHEMA.md`. Symmetric
    /// with `scripts/sweep/pycvvdp_worker.py score-pairs`: same input
    /// TSV format, same output parquet schema, just the
    /// `zen-metrics`-side of the bake. Only available when built with
    /// `--features sweep`.
    #[cfg(feature = "sweep")]
    ScorePairs(ScorePairsArgs),
    /// Print available metrics and which require a GPU.
    ListMetrics,
    /// Print supported input formats.
    ListFormats,
}

#[derive(Parser, Debug)]
struct ScoreArgs {
    /// Metric to evaluate.
    #[arg(long, value_enum)]
    metric: MetricKind,
    /// Reference image path.
    #[arg(long)]
    reference: PathBuf,
    /// Distorted image path.
    #[arg(long)]
    distorted: PathBuf,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Output format. Defaults to plain text on stdout.
    #[arg(long, value_enum, default_value = "plain")]
    output: OutputFormat,
}

#[derive(Parser, Debug)]
struct BatchArgs {
    /// Metric to evaluate.
    #[arg(long, value_enum)]
    metric: MetricKind,
    /// Input TSV with at least two columns: ref_path and dist_path. The
    /// header row is required. Extra columns are passed through.
    #[arg(long)]
    pairs: PathBuf,
    /// Output TSV with the same columns plus a metric score column.
    #[arg(long)]
    output: PathBuf,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Number of CPU jobs (CPU metrics only). GPU metrics always serialize
    /// through one CubeCL stream.
    #[arg(long, default_value = "1")]
    jobs: usize,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// Reference image. Pass once per reference: `--reference a.png --reference b.png`.
    /// Every reference is paired with every `--variant`.
    #[arg(long = "reference", action = ArgAction::Append, required = true)]
    references: Vec<PathBuf>,
    /// Variant image. Pass once per variant. Every variant is scored
    /// against every `--reference`.
    #[arg(long = "variant", action = ArgAction::Append, required = true)]
    variants: Vec<PathBuf>,
    /// Metric to evaluate. Pass once per metric — every metric is run on
    /// every (reference, variant) pair.
    #[arg(long = "metric", value_enum, action = ArgAction::Append, required = true)]
    metrics: Vec<MetricKind>,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Output format. Plain is the default human-readable layout; JSON
    /// emits a structured object with `metrics` + `results`; TSV gives a
    /// flat table with one row per pair and one column per metric.
    #[arg(long, value_enum, default_value = "plain")]
    output: OutputFormat,
    /// Reserved for CPU parallelism. Currently serial — see `batch`.
    #[arg(long, default_value = "1")]
    jobs: usize,
}

#[cfg(feature = "sweep")]
#[derive(Parser, Debug)]
struct SweepArgs {
    /// Codec to drive.
    #[arg(long, value_enum)]
    codec: crate::sweep::CodecKind,
    /// Directory of source images. Every file the path-based decoder
    /// recognises (PNG / WebP / AVIF / JXL / JPEG when enabled) is
    /// included. Subdirectories are not walked.
    #[arg(long)]
    sources: PathBuf,
    /// Comma-separated list of integer qualities (0..=100). e.g.
    /// `5,10,15,20,...,95`.
    #[arg(long)]
    q_grid: String,
    /// JSON object `{axis: [values]}` describing the knob Cartesian
    /// product. See `crates/zen-metrics-cli/src/sweep/encode.rs` for
    /// the per-codec axis names.
    #[arg(long, default_value = "")]
    knob_grid: String,
    /// One or more metrics to score each cell with. Pass once per
    /// metric. Defaults to `zensim` if omitted.
    #[arg(long = "metric", value_enum, action = ArgAction::Append)]
    metrics: Vec<MetricKind>,
    /// Output Pareto TSV path.
    #[arg(long)]
    output: PathBuf,
    /// Optional path for a per-cell zensim feature parquet sidecar. When
    /// set, every cell that runs the `zensim` metric also persists its
    /// 300-feature extended vector to this parquet file. Joins back to
    /// `--output` (TSV) on `(image_path, codec, q, knob_tuple_json)`.
    /// The metric list must include `zensim` for any rows to be written.
    #[arg(long)]
    feature_output: Option<PathBuf>,
    /// Optional directory to receive a PNG of every successfully
    /// decoded cell's distorted image. Filenames are deterministic
    /// per `(src_path, codec, q, knobs)`. Pairs with `--pairs-tsv`
    /// to feed external scorers (e.g. pycvvdp) that need on-disk
    /// `(ref, dist)` image pairs.
    #[arg(long)]
    distorted_out_dir: Option<PathBuf>,
    /// Optional TSV path emitting one row per successfully decoded
    /// cell with columns `image_path codec q knob_tuple_json
    /// ref_path dist_path`. The `ref_path` is the source image's
    /// path; `dist_path` is the distorted PNG written under
    /// `--distorted-out-dir` (empty when that flag is unset).
    #[arg(long)]
    pairs_tsv: Option<PathBuf>,
    /// CubeCL runtime selector for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// CPU thread budget for the per-image inner cell loop. `0` (default)
    /// uses rayon's auto-detection (one thread per logical core). `1`
    /// forces serial execution. Higher values cap the rayon pool.
    /// GPU metrics still serialize through one CubeCL stream regardless.
    #[arg(long, default_value = "0")]
    jobs: usize,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
#[allow(dead_code)]
enum FormatLabel {
    Png,
    Jpeg,
    Webp,
    Avif,
    Jxl,
}

#[cfg(feature = "sweep")]
#[derive(Parser, Debug)]
struct ScorePairsArgs {
    /// Metric to evaluate. The output parquet's score column name is
    /// taken from `MetricKind::column_names()` — e.g. cvvdp uses
    /// `cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>` (the
    /// `CVVDP_IMPL_TAG`-overridable per-implementation tag), keeping
    /// implementations distinguishable in joined sidecars.
    #[arg(long, value_enum)]
    metric: MetricKind,
    /// Input pairs TSV. Required header columns: `ref_path` and
    /// `dist_path` (aliases `reference` / `distorted` also accepted —
    /// matches `batch`'s parser). Identity-tuple columns
    /// `image_path`, `codec`, `q`, `knob_tuple_json` are passed
    /// through to the output when present; when absent, the
    /// `ref_path` is used as `image_path` and the rest get blank
    /// defaults (`""`, `0`, `"{}"`).
    #[arg(long)]
    pairs_tsv: PathBuf,
    /// Output parquet sidecar path. Schema:
    ///   image_path:string  codec:string  q:int64  knob_tuple_json:string
    ///   <metric_score_col>:float64
    /// Zstd compression. Caller joins back to the unified source
    /// parquet by the identity tuple.
    #[arg(long)]
    out_parquet: PathBuf,
    /// CubeCL runtime selection for GPU metrics.
    #[arg(long, value_enum, default_value = "auto")]
    gpu_runtime: GpuRuntime,
    /// Allow sub-176-pixel images for IW-SSIM via reflect-pad adaptive
    /// mode. Default `false` rejects small inputs (stock IW-SSIM
    /// requires `min(W, H) ≥ 176` per the 5-level pyramid + 11×11 valid
    /// blur). When set, the pipeline reflect-pads short axes up to 176
    /// before evaluation — the resulting score is the IW-SSIM of the
    /// padded image and is **informational, not bit-exact stock
    /// IW-SSIM**. Stock-size inputs (≥ 176 on both axes) are unaffected.
    /// Only iwssim honours this flag today; other metrics ignore it.
    #[arg(long, default_value_t = false)]
    allow_small_images: bool,
    /// Gate sidecar emission on a post-scoring distribution sanity check.
    /// When set, after the parquet is written, [`bogus_check`] inspects the
    /// score column and exits with rc=2 (NOT rc=1) if any of these hold:
    ///
    /// - any `NaN` rows (workers should not silently retain NaN);
    /// - ≥ 50% of scores are exactly 0 (or, for distance metrics, exactly the
    ///   identity value) — a strong sign the kernel hit a default-fail
    ///   short-circuit path;
    /// - score range `max - min < 0.01` over ≥ 4 rows (constant output is
    ///   the iwssim "NaN-on-identical → 0" failure mode);
    /// - mean falls outside the metric's documented valid range.
    ///
    /// On rc=2 the parquet is still written (callers can inspect it) and a
    /// structured warning goes to stderr listing every failed check. The
    /// distinct exit code lets the chunk worker upload a failure log to
    /// `s3://zentrain/<run>/failures/<chunk>.log` instead of the bogus
    /// sidecar.
    #[arg(long, default_value_t = false)]
    fail_on_bogus: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match cli.command {
        Command::Score(args) => match cmd_score(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Batch(args) => match cmd_batch(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::Compare(args) => match cmd_compare(args) {
            Ok(success) => {
                if success {
                    ExitCode::SUCCESS
                } else {
                    // Some cells produced errors. Output is still complete
                    // and rendered, but signal a non-zero exit so callers
                    // (CI / driver scripts) can detect partial failures.
                    ExitCode::FAILURE
                }
            }
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        #[cfg(feature = "sweep")]
        Command::Sweep(args) => match cmd_sweep(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        #[cfg(feature = "sweep")]
        Command::ScorePairs(args) => match cmd_score_pairs(args) {
            Ok(ScorePairsOutcome::Ok) => ExitCode::SUCCESS,
            // rc=2 means scores were written but failed the bogus-data
            // sanity check. Distinct from rc=1 (hard error before any
            // parquet was written) so the chunk worker can route the
            // chunk to a failure-log upload instead of treating the
            // sidecar as authoritative training data.
            Ok(ScorePairsOutcome::Bogus) => ExitCode::from(2),
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Command::ListMetrics => {
            print_metric_list();
            ExitCode::SUCCESS
        }
        Command::ListFormats => {
            print_format_list();
            ExitCode::SUCCESS
        }
    }
}

#[cfg(feature = "sweep")]
fn cmd_sweep(args: SweepArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::sweep::{SweepConfig, parse_knob_grid, parse_q_grid, run_sweep};

    let q_grid = parse_q_grid(&args.q_grid)?;
    let knob_grid = parse_knob_grid(&args.knob_grid)?;
    let mut metrics = args.metrics;
    if metrics.is_empty() {
        // Default metric set keeps the binary useful when invoked without
        // a `--metric` flag. zensim is the cheapest defensible default —
        // CPU-only, no GPU runtime needed, and already exposed by the
        // crate.
        metrics.push(MetricKind::Zensim);
    }

    // Walk the source directory (no recursion). Every file we can sniff
    // is included; everything else is skipped silently.
    let mut sources: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&args.sources)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_file() {
            sources.push(p);
        }
    }
    sources.sort();
    if sources.is_empty() {
        return Err(format!("no source files found in {}", args.sources.display()).into());
    }

    // Default: 0 → use rayon's auto-detected num_cpus. Allow override
    // via --jobs N. Old behaviour (`--jobs 1`, serial) preserved for
    // debugging.
    let jobs = if args.jobs == 0 { 0 } else { args.jobs };
    crate::sweep::try_init_thread_pool(jobs)?;
    let cfg = SweepConfig {
        codec: args.codec,
        sources,
        q_grid,
        knob_grid,
        metrics,
        gpu_runtime: args.gpu_runtime,
        output: args.output,
        feature_output: args.feature_output,
        distorted_out_dir: args.distorted_out_dir,
        pairs_tsv: args.pairs_tsv,
        jobs,
    };
    let stats = run_sweep(&cfg)?;
    eprintln!(
        "[sweep] done: {emitted}/{total} cells emitted; \
         encode-fail={ef} decode-fail={df} score-fail={sf}",
        emitted = stats.cells_emitted,
        total = stats.cells_total,
        ef = stats.cells_failed_encode,
        df = stats.cells_failed_decode,
        sf = stats.cells_failed_score,
    );
    Ok(())
}

/// Outcome of [`cmd_score_pairs`] that lets `main` distinguish a successful
/// scoring run from a "data is bogus, retry/fail this chunk" result.
///
/// `rc=0` (Ok) means the parquet was written and the score distribution
/// passed every sanity check (or `--fail-on-bogus` was not set).
/// `rc=2` (Bogus) means the parquet was still written, but one or more
/// distribution checks failed and the caller should treat the chunk as
/// poisoned — chunk workers translate this into a failure-log upload to
/// `s3://zentrain/<run>/failures/<chunk>.log` instead of accepting the
/// sidecar.
#[cfg(feature = "sweep")]
enum ScorePairsOutcome {
    Ok,
    Bogus,
}

/// Per-metric expected-range bounds for the [`bogus_check`] mean-in-range
/// gate. Returns `(min_mean, max_mean, identity_value)` where:
///
/// - `min_mean..=max_mean` is the metric's documented valid score range.
///   `bogus_check` flags a chunk if the column mean falls outside this.
/// - `identity_value` is what the metric outputs for "identical inputs".
///   The "majority-of-rows-at-identity" check (≥ 50% rows exactly at the
///   identity value) catches the iwssim NaN-on-identical → 0 mode and
///   the cvvdp-on-cpu atomic-panic → JOD 10.0 default-fail mode.
///
/// Returns `None` for metrics whose range we haven't characterised yet —
/// callers must then skip the mean / identity checks.
#[cfg(feature = "sweep")]
fn metric_range_bounds(metric: crate::metrics::MetricKind) -> Option<(f64, f64, f64)> {
    use crate::metrics::MetricKind;
    match metric {
        // SSIMULACRA2: [0, 100] roughly, 100 = identical. Real corpus
        // means usually fall in 30..95. Bound widely so we only catch
        // truly bogus distributions.
        MetricKind::Ssim2 | MetricKind::Ssim2Gpu => Some((-50.0, 100.5, 100.0)),
        // Butteraugli: [0, ~30+]; 0 = identical. Distance metric, no
        // upper bound is hard but real-world rarely exceeds 30.
        MetricKind::Butteraugli | MetricKind::ButteraugliGpu => Some((-0.001, 100.0, 0.0)),
        // DSSIM: [0, 1ish]; 0 = identical.
        MetricKind::Dssim | MetricKind::DssimGpu => Some((-0.001, 1.5, 0.0)),
        // IW-SSIM: [0, 1]; 1 = identical. Real distributions hover
        // 0.6..0.99 — anything < 0 or > 1.001 is suspicious.
        MetricKind::IwssimGpu | MetricKind::Iwssim => Some((-0.001, 1.001, 1.0)),
        // Zensim: [0, ~100]; 100 = identical (similarity).
        MetricKind::Zensim | MetricKind::ZensimGpu => Some((-1.0, 110.0, 100.0)),
        // CVVDP: JOD scale, [0, 10]; 10 = imperceptible (identical).
        MetricKind::Cvvdp => Some((-0.5, 10.5, 10.0)),
    }
}

/// Inspect the post-scoring score column and report bogus-data failures.
///
/// Returns `Ok(true)` if every check passed, `Ok(false)` if at least one
/// failed (caller should treat the sidecar as poisoned). Always emits one
/// line per failing check to stderr so vast.ai worker logs explain why a
/// chunk got marked failed.
///
/// Checks implemented:
///
/// 1. `n_nan == 0` — score writer never emits NaN except on per-pair
///    decode/score errors. > 0 NaNs means the kernel failed on at least
///    one pair without surfacing a hard error.
/// 2. `n_at_identity / n_total < 0.5` — for metrics with an identity
///    value (`IwssimGpu` → 1.0, `Cvvdp` → 10.0, `Ssim2*` → 100.0, etc.).
///    50% of rows at exactly the identity value means the kernel hit a
///    default-fail short-circuit on at least half the chunk.
/// 3. `max - min > 0.01` (over ≥ 4 rows) — a real metric on a quality
///    sweep produces variance; a constant column means the kernel never
///    ran (returned the same default every time).
/// 4. Mean is within `metric_range_bounds(metric)`.
///
/// All checks are tolerant of unknown / experimental metrics: if
/// `metric_range_bounds` returns `None`, checks #2 and #4 are skipped.
#[cfg(feature = "sweep")]
fn bogus_check(
    metric: crate::metrics::MetricKind,
    scores: &[f64],
    out_parquet: &Path,
) -> bool {
    let n_total = scores.len();
    if n_total == 0 {
        eprintln!("[fail-on-bogus] FAIL: empty score column in {}", out_parquet.display());
        return false;
    }

    let n_nan = scores.iter().filter(|s| s.is_nan()).count();
    let finite: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();

    let mut ok = true;

    if n_nan > 0 {
        eprintln!(
            "[fail-on-bogus] FAIL ({}): {n_nan}/{n_total} rows are NaN — kernel failed without surfacing a hard error",
            out_parquet.display()
        );
        ok = false;
    }

    if finite.is_empty() {
        eprintln!(
            "[fail-on-bogus] FAIL ({}): zero finite rows after NaN filter — column is unusable",
            out_parquet.display()
        );
        return false;
    }

    if let Some((min_mean, max_mean, identity)) = metric_range_bounds(metric) {
        let eps = 1e-9_f64;
        let n_at_identity = finite
            .iter()
            .filter(|s| (**s - identity).abs() <= eps)
            .count();
        // 50% threshold — flag clearly-pathological distributions like the
        // cvvdp-on-cpu atomic-panic mode (all rows fall through to JOD
        // 10.0). Real sweeps over quality grids never put half the chunk
        // at the identity value (that would mean half the encodes were
        // bit-identical to the source, which only happens at q=100 +
        // lossless, not a realistic backfill chunk).
        let identity_frac = n_at_identity as f64 / finite.len() as f64;
        if identity_frac >= 0.5 {
            eprintln!(
                "[fail-on-bogus] FAIL ({}): {n_at_identity}/{} rows at identity value {} ({:.1}% ≥ 50%) — default-fail short-circuit suspected",
                out_parquet.display(),
                finite.len(),
                identity,
                identity_frac * 100.0
            );
            ok = false;
        }

        let mean: f64 = finite.iter().sum::<f64>() / finite.len() as f64;
        if mean < min_mean || mean > max_mean {
            eprintln!(
                "[fail-on-bogus] FAIL ({}): mean {mean:.4} outside expected range [{min_mean}, {max_mean}] for {}",
                out_parquet.display(),
                metric.name()
            );
            ok = false;
        }
    }

    if finite.len() >= 4 {
        let min = finite.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range < 0.01 {
            eprintln!(
                "[fail-on-bogus] FAIL ({}): score range {range:.6} < 0.01 across {} finite rows — constant output (kernel never ran?)",
                out_parquet.display(),
                finite.len()
            );
            ok = false;
        }
    }

    ok
}

#[cfg(feature = "sweep")]
fn cmd_score_pairs(args: ScorePairsArgs) -> Result<ScorePairsOutcome, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::basic::{Compression, ZstdLevel};
    use parquet::file::properties::WriterProperties;

    // Propagate `--allow-small-images` to the metric construction site
    // via a process-wide `OnceLock` flag set by the CLI. Read by
    // `resolve_default_params` in the metrics dispatcher; today only
    // iwssim honours it.
    if args.allow_small_images {
        crate::metrics::set_allow_small_images();
        eprintln!(
            "[score-pairs] --allow-small-images set: IW-SSIM will tile-pad sub-176 inputs to 176×176"
        );
    }

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(&args.pairs_tsv)?;
    let headers = rdr.headers()?.clone();
    let ref_idx = find_col(&headers, &["ref_path", "reference"])?;
    let dist_idx = find_col(&headers, &["dist_path", "distorted"])?;
    let image_path_idx = find_col(&headers, &["image_path"]).ok();
    let codec_idx = find_col(&headers, &["codec"]).ok();
    let q_idx = find_col(&headers, &["q"]).ok();
    let knob_idx = find_col(&headers, &["knob_tuple_json"]).ok();

    let metric_cols = args.metric.column_names();
    if metric_cols.len() != 1 {
        // CVVDP_SIDECAR_SCHEMA.md fixes one score column per sidecar.
        // Butteraugli's 2-column metric isn't useful as a "cvvdp-like"
        // sidecar — callers should use the `batch` subcommand for that
        // case until we extend the spec to multi-column sidecars.
        return Err(format!(
            "score-pairs supports single-column metrics only; \
             {} emits {} columns ({:?}). Use `batch` for now.",
            args.metric.name(),
            metric_cols.len(),
            metric_cols,
        )
        .into());
    }
    let score_col_name = metric_cols[0];

    // Buffer everything in memory — score-pairs runs over a bounded
    // pairs TSV (one sweep's worth of cells, typically ≤ 10⁵ rows).
    // For larger jobs the producer should partition the TSV by chunk
    // and call score-pairs per chunk.
    let mut image_paths: Vec<String> = Vec::new();
    let mut codecs: Vec<String> = Vec::new();
    let mut qs: Vec<i64> = Vec::new();
    let mut knobs: Vec<String> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();

    let mut failed = 0usize;
    let mut succeeded = 0usize;

    // Cvvdp's Cvvdp::new is expensive (allocates ~200 MB GPU at 1024²
    // + triggers NVRTC kernel compilation). The per-pair `score_one_pair`
    // path recreates it on every row → fleet OOMs at 100-pair chunks
    // even with PARALLEL=1 + 16 GB RAM. Use the batched scorer for cvvdp
    // so the instance survives across pairs of matching dims.
    let mut cvvdp_scorer: Option<crate::metrics::cvvdp_gpu::CvvdpBatchScorer> = None;
    if args.metric == crate::metrics::MetricKind::Cvvdp {
        cvvdp_scorer = Some(
            crate::metrics::cvvdp_gpu::CvvdpBatchScorer::new(args.gpu_runtime)
                .map_err(|e| format!("CvvdpBatchScorer init: {e}"))?,
        );
    }
    // NOTE: IwssimBatchScorer used to be wired here for per-(W,H) JIT
    // caching, but the local CLI iwssim_gpu module depended on the
    // deleted gpu_runtime_dispatch infra. Iwssim now goes through the
    // umbrella's per-pair Metric::compute_srgb_u8 path — slower in
    // batch mode but correct. TODO: port the caching scorer to use
    // zenmetrics_api::iwssim re-export when batch perf matters.

    for record in rdr.records() {
        let record = record?;
        let ref_path = PathBuf::from(record.get(ref_idx).ok_or("missing ref_path")?);
        let dist_path = PathBuf::from(record.get(dist_idx).ok_or("missing dist_path")?);

        // Identity-tuple passthrough with explicit fallbacks. Producer
        // contracts (the sweep's pairs-tsv mode, pycvvdp_worker's
        // input) provide all four; callers feeding a bare ref/dist
        // TSV get sensible defaults that still round-trip the schema.
        let image_path = image_path_idx
            .and_then(|i| record.get(i))
            .map(String::from)
            .unwrap_or_else(|| ref_path.display().to_string());
        let codec = codec_idx
            .and_then(|i| record.get(i))
            .map(String::from)
            .unwrap_or_default();
        let q = q_idx
            .and_then(|i| record.get(i))
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0);
        let knob = knob_idx
            .and_then(|i| record.get(i))
            .map(String::from)
            .unwrap_or_else(|| "{}".to_string());

        let pair_result: Result<f64, Box<dyn std::error::Error>> = {
            if let Some(scorer) = cvvdp_scorer.as_mut() {
                // Cvvdp fast path: decode + reuse cached Cvvdp instance.
                match (
                    decode::decode_image_to_rgb8(&ref_path),
                    decode::decode_image_to_rgb8(&dist_path),
                ) {
                    (Ok(r), Ok(d)) => scorer.score(&r, &d),
                    (Err(e), _) | (_, Err(e)) => Err(e),
                }
            } else {
                score_one_pair(args.metric, &ref_path, &dist_path, args.gpu_runtime)
            }
            #[cfg(not(feature = "gpu-iwssim"))]
            if let Some(scorer) = cvvdp_scorer.as_mut() {
                match (
                    decode::decode_image_to_rgb8(&ref_path),
                    decode::decode_image_to_rgb8(&dist_path),
                ) {
                    (Ok(r), Ok(d)) => scorer.score(&r, &d),
                    (Err(e), _) | (_, Err(e)) => Err(e),
                }
            } else {
                score_one_pair(args.metric, &ref_path, &dist_path, args.gpu_runtime)
            }
        };
        let jod = match pair_result {
            Ok(v) => {
                succeeded += 1;
                v
            }
            Err(e) => {
                eprintln!("[score-pairs] {} q={q} failed: {e}", image_path,);
                failed += 1;
                f64::NAN
            }
        };

        image_paths.push(image_path);
        codecs.push(codec);
        qs.push(q);
        knobs.push(knob);
        scores.push(jod);

        let total = succeeded + failed;
        if total % 100 == 0 && total > 0 {
            eprintln!("[score-pairs] {total} pairs scored, {failed} failed",);
        }
    }

    if image_paths.is_empty() {
        return Err("score-pairs: input TSV produced no rows".into());
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("image_path", DataType::Utf8, false),
        Field::new("codec", DataType::Utf8, false),
        Field::new("q", DataType::Int64, false),
        Field::new("knob_tuple_json", DataType::Utf8, false),
        Field::new(score_col_name, DataType::Float64, false),
    ]));

    // Snapshot the score column before it's moved into the Arrow array so
    // `bogus_check` can inspect it post-write without needing to round-trip
    // through parquet. Cheap (one Vec<f64> copy) and keeps the sanity
    // check in-process — important on workers that may not have R2 set up.
    let scores_snapshot: Vec<f64> = scores.clone();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(image_paths)),
        Arc::new(StringArray::from(codecs)),
        Arc::new(Int64Array::from(qs)),
        Arc::new(StringArray::from(knobs)),
        Arc::new(Float64Array::from(scores)),
    ];

    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    if let Some(parent) = args.out_parquet.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(&args.out_parquet)?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    eprintln!(
        "[score-pairs] wrote {succeeded} rows ({failed} NaN-failures) \
         to {} with score column `{score_col_name}`",
        args.out_parquet.display(),
    );

    if args.fail_on_bogus {
        let passed = bogus_check(args.metric, &scores_snapshot, &args.out_parquet);
        if !passed {
            eprintln!(
                "[score-pairs] --fail-on-bogus: sanity checks FAILED — exiting rc=2; sidecar at {} is suspect",
                args.out_parquet.display()
            );
            return Ok(ScorePairsOutcome::Bogus);
        }
        eprintln!(
            "[score-pairs] --fail-on-bogus: sanity checks PASSED for {}",
            args.out_parquet.display()
        );
    }

    Ok(ScorePairsOutcome::Ok)
}

#[cfg(feature = "sweep")]
fn score_one_pair(
    metric: MetricKind,
    ref_path: &Path,
    dist_path: &Path,
    gpu_runtime: GpuRuntime,
) -> Result<f64, Box<dyn std::error::Error>> {
    use crate::metrics::run_metric;
    let reference = decode::decode_image_to_rgb8(ref_path)?;
    let distorted = decode::decode_image_to_rgb8(dist_path)?;
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "dimension mismatch: {} ({}x{}) vs {} ({}x{})",
            ref_path.display(),
            reference.width,
            reference.height,
            dist_path.display(),
            distorted.width,
            distorted.height,
        )
        .into());
    }
    let scores = run_metric(metric, &reference, &distorted, gpu_runtime)?;
    let (_, value) = scores
        .first()
        .copied()
        .ok_or("metric returned zero scores")?;
    Ok(value)
}

fn cmd_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    let reference = decode::decode_image_to_rgb8(&args.reference)?;
    let distorted = decode::decode_image_to_rgb8(&args.distorted)?;
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(format!(
            "dimension mismatch: reference is {}x{}, distorted is {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )
        .into());
    }
    let scores = run_metric(args.metric, &reference, &distorted, args.gpu_runtime)?;
    print_score(args.output, args.metric, &scores);
    Ok(())
}

fn cmd_batch(args: BatchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_path(&args.pairs)?;
    let headers = rdr.headers()?.clone();
    let ref_idx = find_col(&headers, &["ref_path", "reference"])?;
    let dist_idx = find_col(&headers, &["dist_path", "distorted"])?;

    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(&args.output)?;
    let metric_cols = args.metric.column_names();
    let mut new_headers: Vec<String> = headers.iter().map(String::from).collect();
    for col in metric_cols {
        new_headers.push((*col).to_string());
    }
    wtr.write_record(&new_headers)?;

    let _ = args.jobs; // Reserved: CPU parallelism. Currently serial; rayon
    // integration left for a follow-up because zen decoders and CPU metrics
    // mix Send + non-Send internals in ways that need per-metric review.

    for record in rdr.records() {
        let record = record?;
        let ref_path = PathBuf::from(record.get(ref_idx).ok_or("missing ref_path")?);
        let dist_path = PathBuf::from(record.get(dist_idx).ok_or("missing dist_path")?);
        let reference = decode::decode_image_to_rgb8(&ref_path)?;
        let distorted = decode::decode_image_to_rgb8(&dist_path)?;
        if reference.width != distorted.width || reference.height != distorted.height {
            return Err(format!(
                "dimension mismatch on row: {} ({}x{}) vs {} ({}x{})",
                ref_path.display(),
                reference.width,
                reference.height,
                dist_path.display(),
                distorted.width,
                distorted.height
            )
            .into());
        }
        let scores = run_metric(args.metric, &reference, &distorted, args.gpu_runtime)?;
        let mut row: Vec<String> = record.iter().map(String::from).collect();
        for (_, value) in &scores {
            row.push(format!("{value:.6}"));
        }
        wtr.write_record(&row)?;
    }
    wtr.flush()?;
    Ok(())
}

/// Returns `Ok(true)` when every cell succeeded and `Ok(false)` when at
/// least one cell failed (the report is still rendered in full). `Err` is
/// reserved for setup failures that prevent any output (currently: empty
/// argument lists, which clap also enforces via `required = true`).
fn cmd_compare(args: CompareArgs) -> Result<bool, Box<dyn std::error::Error>> {
    if args.references.is_empty() {
        return Err("at least one --reference is required".into());
    }
    if args.variants.is_empty() {
        return Err("at least one --variant is required".into());
    }
    if args.metrics.is_empty() {
        return Err("at least one --metric is required".into());
    }
    let report = run_compare(
        &args.references,
        &args.variants,
        &args.metrics,
        args.gpu_runtime,
        args.jobs,
    );
    print_report(args.output, &args.metrics, &report)?;
    Ok(!report.had_failures)
}

fn find_col(headers: &csv::StringRecord, names: &[&str]) -> Result<usize, String> {
    for (idx, h) in headers.iter().enumerate() {
        for n in names {
            if h.eq_ignore_ascii_case(n) {
                return Ok(idx);
            }
        }
    }
    Err(format!(
        "input TSV is missing one of the expected columns: {names:?}"
    ))
}

fn print_metric_list() {
    println!("name                 backend  requires_gpu");
    for m in MetricKind::all() {
        println!(
            "{:<20} {:<8} {}",
            m.name(),
            m.backend(),
            if m.requires_gpu() { "yes" } else { "no" }
        );
    }
    println!();
    println!("GPU runtimes (--gpu-runtime): auto, cuda, wgpu, hip, cpu");
}

fn print_format_list() {
    let mut formats: Vec<&str> = Vec::new();
    if cfg!(feature = "png") {
        formats.push("png  (zenpng)");
    }
    if cfg!(feature = "jpeg") {
        formats.push("jpeg (zenjpeg)");
    }
    if cfg!(feature = "webp") {
        formats.push("webp (zenwebp)");
    }
    if cfg!(feature = "avif") {
        formats.push("avif (zenavif)");
    }
    if cfg!(feature = "jxl") {
        formats.push("jxl  (zenjxl)");
    }
    if formats.is_empty() {
        println!("(no decoders enabled — rebuild with `--features png,jpeg,webp,avif,jxl`)");
    } else {
        for f in formats {
            println!("{f}");
        }
    }
}

#[cfg(all(test, feature = "sweep"))]
mod fail_on_bogus_tests {
    //! Unit tests for [`bogus_check`]. These do not need GPU — they
    //! synthesise score vectors directly and check the gating logic.
    //!
    //! Backstop for the iwssim NaN-on-identical regression: 525 sidecars
    //! were uploaded with every score at 0 or NaN, undetected for 3 hr
    //! until V_24 training failed. Each test below corresponds to one
    //! failure mode that should trip the gate.
    use super::*;
    use crate::metrics::MetricKind;

    fn p() -> std::path::PathBuf {
        std::path::PathBuf::from("/tmp/test_sidecar.parquet")
    }

    #[test]
    fn bogus_check_passes_clean_iwssim_distribution() {
        // Real iwssim distribution from a 100-cell quality sweep: scores
        // spread 0.7..0.99 with mean ~0.88. Should pass every gate.
        let scores: Vec<f64> = (0..100)
            .map(|i| 0.70 + (i as f64) * 0.0028) // 0.70..0.978
            .collect();
        assert!(bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }

    #[test]
    fn bogus_check_rejects_any_nan() {
        // Even one NaN means the kernel failed silently on at least one
        // pair without a hard error — must trip the gate.
        let mut scores: Vec<f64> = (0..50).map(|i| 0.70 + (i as f64) * 0.005).collect();
        scores[12] = f64::NAN;
        assert!(!bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }

    #[test]
    fn bogus_check_rejects_all_zero_iwssim() {
        // The exact 525-sidecar failure mode: NaN-on-identical wrote 0.0
        // for every row. Identity for iwssim is 1.0, so this trips the
        // "constant + mean out of range" but NOT the identity check.
        let scores: Vec<f64> = vec![0.0; 100];
        assert!(!bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }

    #[test]
    fn bogus_check_rejects_majority_at_identity() {
        // The cvvdp-on-cpu atomic-panic mode: kernel panics, default-fail
        // path writes JOD=10.0 (the identity for CVVDP). 60% at identity
        // means the gate must fire.
        let mut scores: Vec<f64> = vec![10.0; 60];
        scores.extend((0..40).map(|i| 7.0 + (i as f64) * 0.02));
        assert!(!bogus_check(MetricKind::Cvvdp, &scores, &p()));
    }

    #[test]
    fn bogus_check_rejects_constant_output() {
        // 100 rows all at exactly 0.8 — no kernel variation. The
        // "max - min < 0.01" check trips even though 0.8 isn't the
        // identity value, because real metrics on real data ALWAYS
        // produce some variance across a quality sweep.
        let scores: Vec<f64> = vec![0.8; 100];
        assert!(!bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }

    #[test]
    fn bogus_check_rejects_mean_out_of_range() {
        // Iwssim mean of -0.5 with variance: clearly broken.
        let scores: Vec<f64> = (0..100).map(|i| -0.6 + (i as f64) * 0.001).collect();
        assert!(!bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }

    #[test]
    fn bogus_check_passes_clean_cvvdp_distribution() {
        // Real cvvdp distribution: JOD scores spread 6.0..9.8 with mean
        // ~8.0. Should pass every gate.
        let scores: Vec<f64> = (0..50).map(|i| 6.0 + (i as f64) * 0.076).collect();
        assert!(bogus_check(MetricKind::Cvvdp, &scores, &p()));
    }

    #[test]
    fn bogus_check_passes_clean_ssim2_distribution() {
        let scores: Vec<f64> = (0..100).map(|i| 30.0 + (i as f64) * 0.65).collect();
        assert!(bogus_check(MetricKind::Ssim2, &scores, &p()));
    }

    #[test]
    fn bogus_check_handles_few_rows() {
        // < 4 rows: the constant-output check is skipped (can't
        // distinguish a real 3-pair chunk from kernel-failure). Other
        // checks still apply. A 3-cell chunk producing 0.85, 0.85, 0.85
        // is plausibly real (q=95 outputs are often near-identical for
        // simple sources) so the gate cannot reject it. This is the
        // intended trade-off for tiny chunks: a real production chunk
        // is hundreds of rows, where the constant-output check fires
        // reliably on kernel failures.
        let scores: Vec<f64> = vec![0.85, 0.85, 0.85];
        assert!(bogus_check(MetricKind::IwssimGpu, &scores, &p()));
        // Cvvdp 3 rows all at JOD 10.0 (identity) trips the identity
        // check (3/3 = 100% ≥ 50%).
        let cvvdp_identity: Vec<f64> = vec![10.0, 10.0, 10.0];
        assert!(!bogus_check(MetricKind::Cvvdp, &cvvdp_identity, &p()));
    }

    #[test]
    fn bogus_check_rejects_empty_column() {
        let scores: Vec<f64> = vec![];
        assert!(!bogus_check(MetricKind::IwssimGpu, &scores, &p()));
    }
}
