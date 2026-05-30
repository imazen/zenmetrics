#![forbid(unsafe_code)]
//! `zen-metrics jobexec` — the reference executor for the zen job system's `ZEN_EXEC` contract.
//!
//! The worker (`zen-jobworker::exec_command`) pipes ONE `DesiredJob` as JSON to this process's stdin
//! and content-addresses whatever we write to stdout. We do the real encode/score for that one cell:
//!
//!   stdin  <- {"kind":{"kind":"encode"|"metric",...}, "inputs":[...], "cell":{image_path,codec,q,knob_tuple_json}}
//!   stdout -> Encode: the encoded image bytes;  Metric: a one-line JSON score row
//!   exit 0  = success; non-zero = deterministic FAILED row.
//!
//! Source resolution (the source image named by `cell.image_path`):
//!   - `s3://…` path                          -> fetched with s5cmd
//!   - else if `$ZEN_CORPUS_PREFIX` is set     -> s3://$ZEN_BUCKET/$ZEN_CORPUS_PREFIX/<image_path>
//!   - else if the local file exists           -> used directly
//! A `metric` job is self-contained: it re-encodes the cell (deterministic) and scores
//! (reference=source, distorted=encode). CPU metrics only (ssim2/butteraugli/zensim) — GPU metrics
//! need a GPU build/tier. This keeps the basement + CPU burst tiers fully useful.
use crate::decode::{decode_image_to_rgb8, Rgb8Image};
use crate::metrics::{run_metric, GpuRuntime, MetricKind};
use crate::sweep::encode::{encode, CodecKind};
use clap::{Parser, ValueEnum};
use serde_json::{Map, Value};
use std::error::Error;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser, Debug)]
pub struct JobexecArgs {
    /// R2 prefix under $ZEN_BUCKET where source images live (overrides $ZEN_CORPUS_PREFIX). The job's
    /// `cell.image_path` is appended to it. Omit if image_path is already an `s3://…` or local path.
    #[arg(long)]
    pub corpus_prefix: Option<String>,
}

fn codec_from_name(name: &str) -> Result<CodecKind, Box<dyn Error>> {
    Ok(match name {
        "zenpng" => CodecKind::Zenpng,
        "zenjpeg" => CodecKind::Zenjpeg,
        "zenwebp" => CodecKind::Zenwebp,
        "zenavif" => CodecKind::Zenavif,
        "zenjxl" => CodecKind::Zenjxl,
        other => return Err(format!("unknown codec {other:?}").into()),
    })
}

fn ext_for(name: &str) -> &'static str {
    match name {
        "zenpng" => "png",
        "zenjpeg" => "jpg",
        "zenwebp" => "webp",
        "zenavif" => "avif",
        "zenjxl" => "jxl",
        _ => "bin",
    }
}

/// Map the job's metric string to a `MetricKind` (clap value names: ssim2, butteraugli, zensim,
/// cvvdp, …), tolerating the `ssimulacra2` alias. Then `run_metric` dispatches to the right backend —
/// CPU metrics work in a default build; GPU metrics return a clear "needs a GPU build" error there.
fn metric_kind(metric: &str) -> Result<MetricKind, Box<dyn Error>> {
    let canon = if metric == "ssimulacra2" { "ssim2" } else { metric };
    MetricKind::from_str(canon, true).map_err(|e| format!("unknown metric {metric:?}: {e}").into())
}

/// Score `(reference, distorted)` with `metric`, returning all `(column, value)` pairs run_metric
/// yields (butteraugli yields max-norm + 3-norm; most yield one).
fn score(metric: &str, reference: &Rgb8Image, distorted: &Rgb8Image)
    -> Result<Vec<(&'static str, f64)>, Box<dyn Error>>
{
    run_metric(metric_kind(metric)?, reference, distorted, GpuRuntime::Auto)
}

/// Resolve `cell.image_path` to a readable local file, fetching from R2 if needed.
fn resolve_source(image_path: &str, corpus_prefix: Option<&str>) -> Result<PathBuf, Box<dyn Error>> {
    let local = PathBuf::from(image_path);
    if corpus_prefix.is_none() && !image_path.starts_with("s3://") && local.exists() {
        return Ok(local);
    }
    let endpoint = std::env::var("ZEN_R2_ENDPOINT")
        .map_err(|_| "ZEN_R2_ENDPOINT unset — cannot fetch source from R2")?;
    let uri = if image_path.starts_with("s3://") {
        image_path.to_string()
    } else {
        let bucket = std::env::var("ZEN_BUCKET").map_err(|_| "ZEN_BUCKET unset")?;
        match corpus_prefix {
            Some(p) if !p.is_empty() => format!("s3://{bucket}/{}/{image_path}", p.trim_end_matches('/')),
            _ => format!("s3://{bucket}/{image_path}"),
        }
    };
    let dst = std::env::temp_dir().join(format!(
        "jobexec_src_{}_{}",
        std::process::id(),
        image_path.rsplit('/').next().unwrap_or("src")
    ));
    let st = Command::new("s5cmd")
        .arg("--endpoint-url")
        .arg(&endpoint)
        .arg("cp")
        .arg(&uri)
        .arg(&dst)
        .status()
        .map_err(|e| format!("spawn s5cmd: {e}"))?;
    if !st.success() {
        return Err(format!("s5cmd cp {uri} failed").into());
    }
    Ok(dst)
}

pub fn run(args: JobexecArgs) -> Result<(), Box<dyn Error>> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let job: Value = serde_json::from_str(&buf).map_err(|e| format!("parse DesiredJob: {e}"))?;

    let kind = job["kind"]["kind"].as_str().ok_or("job.kind.kind missing")?;
    let cell = &job["cell"];
    let image_path = cell["image_path"].as_str().ok_or("cell.image_path missing")?;
    let codec_name = cell["codec"].as_str().ok_or("cell.codec missing")?;
    let q = cell["q"].as_i64().ok_or("cell.q missing")? as f64;
    let knob_json = cell["knob_tuple_json"].as_str().unwrap_or("{}");
    let knobs: Map<String, Value> =
        serde_json::from_str(knob_json).map_err(|e| format!("parse knob_tuple_json: {e}"))?;

    let corpus_prefix = args
        .corpus_prefix
        .or_else(|| std::env::var("ZEN_CORPUS_PREFIX").ok());
    let src_path = resolve_source(image_path, corpus_prefix.as_deref())?;
    let reference = decode_image_to_rgb8(&src_path)?;

    let codec = codec_from_name(codec_name)?;
    let encoded = encode(codec, &reference, q, &knobs)?;

    let mut out = std::io::stdout().lock();
    match kind {
        "encode" => {
            // The encoded bytes ARE the output → content-addressed to blobs/<sha256> (goal G).
            out.write_all(&encoded.bytes)?;
        }
        "metric" => {
            let metric = job["kind"]["metric"].as_str().ok_or("metric job missing kind.metric")?;
            let dist_path = std::env::temp_dir().join(format!(
                "jobexec_dist_{}.{}",
                std::process::id(),
                ext_for(codec_name)
            ));
            std::fs::write(&dist_path, &encoded.bytes)?;
            let distorted = decode_image_to_rgb8(&dist_path)?;
            let pairs = score(metric, &reference, &distorted)?;
            let mut scores = Map::new();
            for (name, value) in &pairs {
                scores.insert((*name).to_string(), serde_json::json!(value));
            }
            let row = serde_json::json!({
                "kind": "metric",
                "metric": metric,
                "image_path": image_path,
                "codec": codec_name,
                "q": cell["q"],
                "knob_tuple_json": knob_json,
                "score": pairs.first().map(|(_, v)| *v),
                "scores": scores,
                "encoded_bytes": encoded.bytes.len(),
                "encode_ms": encoded.encode_ms,
            });
            out.write_all(serde_json::to_string(&row)?.as_bytes())?;
            let _ = std::fs::remove_file(&dist_path);
        }
        other => return Err(format!("jobexec: unhandled job kind {other:?}").into()),
    }
    out.flush()?;
    Ok(())
}
