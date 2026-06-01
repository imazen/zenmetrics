#![forbid(unsafe_code)]
//! A *real* encode executor honoring the worker's contract: read a `DesiredJob` JSON on stdin, write
//! the produced bytes to stdout (exit 0 = success). For an `Encode` job it loads `cell.image_path`
//! and encodes it via the `image` crate (JPEG with the job's quality, or PNG) — genuine codec work,
//! no stub. Wire it as `zen-jobworker --exec <path>/zen-jobexec`. (Light path: JPEG/PNG via `image`;
//! the zen codecs / metric scoring are a heavier executor for later.)

use std::io::{Read, Write};

use image::codecs::jpeg::JpegEncoder;
use zen_job_core::{DesiredJob, JobKind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf)?;
    let job: DesiredJob = serde_json::from_slice(&buf)?;

    let out = match &job.kind {
        JobKind::Encode { codec, q, .. } => encode(&job.cell.image_path, codec, *q)?,
        JobKind::Metric { metric } => score(&job, metric)?,
        other => {
            return Err(format!("zen-jobexec handles Encode/Metric jobs (got {other:?})").into());
        }
    };
    std::io::stdout().write_all(&out)?;
    Ok(())
}

/// Score a distorted encode against its reference (real CPU work — PSNR). Reference is
/// `cell.image_path`; the distorted image is the content-addressed blob `inputs[0]`, fetched from the
/// local blob dir given by `ZEN_BLOBS_DIR` (the worker's `--blobs`). Output is a small JSON score blob.
fn score(job: &DesiredJob, metric: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let blobs = std::env::var("ZEN_BLOBS_DIR")
        .map_err(|_| "Metric job needs ZEN_BLOBS_DIR (dir of content-addressed blobs)")?;
    let dist_sha = job.inputs.first().ok_or("Metric job has no input encode")?;
    let dist_path = std::path::Path::new(&blobs).join(dist_sha.as_str());

    let reference = image::ImageReader::open(&job.cell.image_path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let distorted = image::ImageReader::open(&dist_path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    if reference.dimensions() != distorted.dimensions() {
        return Err(format!(
            "ref {:?} vs dist {:?} dimension mismatch",
            reference.dimensions(),
            distorted.dimensions()
        )
        .into());
    }
    let psnr = psnr_db(&reference, &distorted);
    Ok(format!("{{\"metric\":\"{metric}\",\"psnr_db\":{psnr:.4}}}\n").into_bytes())
}

/// Peak signal-to-noise ratio (dB) over RGB. Higher = closer; identical → capped at 99.
fn psnr_db(a: &image::RgbImage, b: &image::RgbImage) -> f64 {
    let mut se = 0f64;
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        for c in 0..3 {
            let d = pa[c] as f64 - pb[c] as f64;
            se += d * d;
        }
    }
    let n = (a.width() as u64 * a.height() as u64 * 3) as f64;
    let mse = se / n;
    if mse <= 0.0 {
        return 99.0;
    }
    10.0 * ((255.0 * 255.0) / mse).log10()
}

fn encode(path: &str, codec: &str, q: i64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?
        .with_guessed_format()?
        .decode()?;
    let c = codec.to_ascii_lowercase();
    let mut out = Vec::new();
    if c.contains("jpeg") || c.contains("jpg") {
        let quality = q.clamp(1, 100) as u8;
        let mut enc = JpegEncoder::new_with_quality(&mut out, quality);
        enc.encode_image(&img)?;
    } else if c.contains("png") {
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    } else {
        return Err(
            format!("codec {codec:?} unsupported by this light executor (jpeg/png only)").into(),
        );
    }
    Ok(out)
}
