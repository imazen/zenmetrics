#![forbid(unsafe_code)]

//! `score-video` subcommand — cvvdp's video path over two directories
//! of still-image frames (`--reference-dir` / `--distorted-dir`).
//!
//! Frames are paired by lexicographic filename order (name your frames
//! `f0000.png`, `f0001.png`, …). Decoding streams frame-by-frame
//! through `cvvdp::VideoScorer`, so peak memory is the decoder's frame
//! buffers plus the scorer's bounded temporal window — never the whole
//! clip. Requires `--features cpu-cvvdp` (the default build has it).
//!
//! This is the CLI face of the `cvvdp` crate's video API — the
//! byte-slice analog of `pycvvdp.predict(test, reference, dim_order,
//! frames_per_second)`; see `crates/cvvdp/docs/VIDEO.md`.

use std::fs;
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use zenmetrics_api::cvvdp_cpu::{
    CvvdpParams, DisplayGeometry, DisplayModel, FrameLayout, TempPadding, VideoScorer,
    VideoScorerOptions,
};

use crate::decode::decode_image_to_rgb8;
use crate::output::OutputFormat;

/// `score-video` arguments — see [`Command::ScoreVideo`].
#[derive(Parser, Debug)]
pub(crate) struct ScoreVideoArgs {
    /// Directory holding the reference clip's frames (any format
    /// `decode` supports; files are sorted by name — zero-pad frame
    /// numbers, e.g. `f0000.png`).
    #[arg(long)]
    reference_dir: PathBuf,
    /// Directory holding the distorted clip's frames. Must contain the
    /// same number of decodable frames as `--reference-dir`.
    #[arg(long)]
    distorted_dir: PathBuf,
    /// Clip frame rate in Hz — drives the sustained/transient temporal
    /// filters (pycvvdp `frames_per_second`).
    #[arg(long)]
    fps: f32,
    /// Display preset name from cvvdp's vendored `display_models.json`
    /// (photometry AND geometry), e.g. `standard_4k` (default),
    /// `standard_fhd` (the AIC-4 CTC anchor), `standard_phone`,
    /// `standard_hdr_pq`.
    #[arg(long, default_value = "standard_4k")]
    display_model: String,
    /// Temporal padding for the filter's left edge — pycvvdp's
    /// `temp_padding`. `replicate` (default) copies frame 0 backwards;
    /// `symmetric` mirrors (`frame[-k] = frame[k]`, ping-pong on clips
    /// shorter than the filter). The scorer defers the first `fl−1`
    /// outputs under symmetric until the lookahead frames exist.
    #[arg(long, value_enum, default_value = "replicate")]
    temp_padding: CliTempPadding,
    /// Output format: `plain` prints `metric=… jod=… loss=…`, `tsv`
    /// prints a two-row table, `json` prints one object (add `--stats`
    /// for the full `Q_per_ch`/`rho_band` bundle).
    #[arg(long, value_enum, default_value = "plain")]
    output: OutputFormat,
    /// Include the full stats bundle (per-frame/per-band/per-channel
    /// `q_per_ch`, `rho_band`) — JSON output only.
    #[arg(long)]
    stats: bool,
    /// Store the temporal window as u8 sRGB instead of f32 DKL — a 4×
    /// smaller ring (e.g. ~450→113 MB at 1080p/30 fps) at the cost of
    /// re-converting window slots on each emit (~+10–20 % CPU). Scores
    /// are bit-identical to the default path.
    #[arg(long)]
    low_memory: bool,
}

/// clap `ValueEnum` mirror of [`TempPadding`] (kept crate-local so the
/// upstream enum stays dependency-light).
#[derive(ValueEnum, Debug, Clone, Copy)]
enum CliTempPadding {
    /// Replicate the first frame (upstream default).
    Replicate,
    /// Mirror frames before index 0 (`frame[-k] → frame[k]`).
    Symmetric,
}

impl From<CliTempPadding> for TempPadding {
    fn from(p: CliTempPadding) -> Self {
        match p {
            CliTempPadding::Replicate => TempPadding::Replicate,
            CliTempPadding::Symmetric => TempPadding::Symmetric,
        }
    }
}

/// Sorted list of regular files in `dir` — frame order for the clip.
fn frame_list(dir: &Path, which: &str) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut frames: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| {
            format!(
                "score-video: cannot read {which} dir {}: {e}",
                dir.display()
            )
        })?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    frames.sort();
    if frames.is_empty() {
        return Err(format!(
            "score-video: {which} dir {} contains no files",
            dir.display()
        )
        .into());
    }
    Ok(frames)
}

/// `score-video` entry point.
pub(crate) fn run(args: &ScoreVideoArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.fps.is_finite() || args.fps <= 0.0 {
        return Err(format!("score-video: --fps must be > 0 (got {})", args.fps).into());
    }
    let display = DisplayModel::by_name(&args.display_model).ok_or_else(|| {
        format!(
            "score-video: unknown --display-model {:?}; see cvvdp's vendored \
             display_models.json (e.g. standard_4k, standard_fhd, standard_phone)",
            args.display_model
        )
    })?;
    let geometry = DisplayGeometry::by_name(&args.display_model).ok_or_else(|| {
        format!(
            "score-video: --display-model {:?} has photometry but no geometry",
            args.display_model
        )
    })?;

    let ref_frames = frame_list(&args.reference_dir, "reference")?;
    let dist_frames = frame_list(&args.distorted_dir, "distorted")?;
    if ref_frames.len() != dist_frames.len() {
        return Err(format!(
            "score-video: frame count differs — {} has {}, {} has {}",
            args.reference_dir.display(),
            ref_frames.len(),
            args.distorted_dir.display(),
            dist_frames.len()
        )
        .into());
    }

    let params = CvvdpParams {
        display,
        ..CvvdpParams::default()
    };
    let mut scorer: Option<VideoScorer> = None;
    for (i, (rp, dp)) in ref_frames.iter().zip(dist_frames.iter()).enumerate() {
        let r = decode_image_to_rgb8(rp)
            .map_err(|e| format!("score-video: decode {}: {e}", rp.display()))?;
        let d = decode_image_to_rgb8(dp)
            .map_err(|e| format!("score-video: decode {}: {e}", dp.display()))?;
        if r.width != d.width || r.height != d.height {
            return Err(format!(
                "score-video: frame {i} size differs — ref {}×{}, dist {}×{}",
                r.width, r.height, d.width, d.height
            )
            .into());
        }
        let v = match scorer.as_mut() {
            Some(v) => v,
            None => {
                let v = VideoScorer::with_options(
                    r.width,
                    r.height,
                    args.fps,
                    params,
                    geometry,
                    VideoScorerOptions {
                        layout: FrameLayout::Interleaved,
                        temp_padding: args.temp_padding.into(),
                        low_memory: args.low_memory,
                    },
                )
                .map_err(|e| format!("score-video: VideoScorer: {e}"))?;
                scorer.insert(v)
            }
        };
        v.push_frame(&r.pixels, &d.pixels)
            .map_err(|e| format!("score-video: frame {i}: {e}"))?;
    }
    let stats = scorer
        .ok_or("score-video: no frames decoded")?
        .finish_with_stats()
        .map_err(|e| format!("score-video: finish: {e}"))?;

    match args.output {
        OutputFormat::Plain => {
            println!(
                "metric=cvvdp-video jod={:.6} loss={:.6} frames={} fps={} display={}",
                stats.jod,
                stats.loss(),
                stats.n_frames,
                stats.frames_per_second,
                args.display_model
            );
        }
        OutputFormat::Tsv => {
            println!("jod\tloss\tn_frames\tfps\tdisplay");
            println!(
                "{:.6}\t{:.6}\t{}\t{}\t{}",
                stats.jod,
                stats.loss(),
                stats.n_frames,
                stats.frames_per_second,
                args.display_model
            );
        }
        OutputFormat::Json => {
            let mut v = serde_json::json!({
                "metric": "cvvdp-video",
                "scores": {
                    "jod": stats.jod,
                    "loss": stats.loss(),
                },
                "display_model": args.display_model,
                "frames_per_second": stats.frames_per_second,
                "width": stats.width,
                "height": stats.height,
                "n_frames": stats.n_frames,
                "temp_padding": format!("{:?}", args.temp_padding).to_lowercase(),
            });
            if args.stats {
                v["rho_band"] = serde_json::json!(stats.rho_band);
                v["q_per_ch"] = serde_json::json!(stats.q_per_ch);
            }
            println!("{v}");
        }
    }
    Ok(())
}
