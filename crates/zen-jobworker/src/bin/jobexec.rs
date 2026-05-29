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
        other => {
            return Err(format!("zen-jobexec only handles Encode jobs (got {other:?})").into());
        }
    };
    std::io::stdout().write_all(&out)?;
    Ok(())
}

fn encode(path: &str, codec: &str, q: i64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = image::ImageReader::open(path)?.with_guessed_format()?.decode()?;
    let c = codec.to_ascii_lowercase();
    let mut out = Vec::new();
    if c.contains("jpeg") || c.contains("jpg") {
        let quality = q.clamp(1, 100) as u8;
        let mut enc = JpegEncoder::new_with_quality(&mut out, quality);
        enc.encode_image(&img)?;
    } else if c.contains("png") {
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    } else {
        return Err(format!("codec {codec:?} unsupported by this light executor (jpeg/png only)").into());
    }
    Ok(out)
}
