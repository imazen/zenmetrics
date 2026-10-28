//! Transport for the same frozen 0.8.2 metric: one raw reference, many PNG candidates.
//! The canonical implementation identity remains the replayed B0 teacher;
//! run receipts additionally bind this adapter's actual executable SHA256.
#![forbid(unsafe_code)]
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, BufRead},
};
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut lines = io::stdin().lock().lines();
    let header: serde_json::Value =
        serde_json::from_str(&lines.next().ok_or("batch header missing")??)?;
    let w = header["width"].as_u64().ok_or("width missing")? as usize;
    let h = header["height"].as_u64().ok_or("height missing")? as usize;
    if w == 0 || h == 0 || w.checked_mul(h).is_none_or(|n| n >= 1_000_000) {
        return Err("invalid batch geometry".into());
    }
    let raw = fs::read(
        header["reference_rgb8"]
            .as_str()
            .ok_or("raw reference missing")?,
    )?;
    if raw.len() != w * h * 3 {
        return Err("raw reference length differs".into());
    }
    let reference_hash = hash(&raw);
    if header["reference_pixels_sha256"].as_str() != Some(&reference_hash) {
        return Err("canonical reference pixels differ".into());
    }
    let reference = raw.as_chunks::<3>().0;
    for line in lines {
        let row: serde_json::Value = serde_json::from_str(&line?)?;
        let decoded = zenpng::decode(
            &fs::read(row["distorted"].as_str().ok_or("candidate missing")?)?,
            &zenpng::PngDecodeConfig::strict(),
            &enough::Unstoppable,
        )?;
        if decoded.pixels.width() as usize != w || decoded.pixels.height() as usize != h {
            return Err("candidate geometry differs".into());
        }
        // Same legacy projection as the canonical B0 helper's Image::load.
        let source = decoded.pixels.as_slice();
        let desc = decoded.pixels.descriptor();
        let mut bytes = Vec::with_capacity(w * h * 3);
        if desc.layout_compatible(zenpixels::PixelDescriptor::RGB8_SRGB) {
            for y in 0..h {
                bytes.extend_from_slice(&source.row(y as u32)[..w * 3]);
            }
        } else if desc.layout_compatible(zenpixels::PixelDescriptor::RGBA8_SRGB) {
            for y in 0..h {
                for pixel in source.row(y as u32)[..w * 4].as_chunks::<4>().0 {
                    bytes.extend_from_slice(&pixel[..3]);
                }
            }
        } else {
            bytes.resize(w * h * 3, 0);
            let mut converter =
                zenpixels_convert::RowConverter::new(desc, zenpixels::PixelDescriptor::RGB8_SRGB)?;
            converter.convert_rows(
                source.as_strided_bytes(),
                source.stride(),
                &mut bytes,
                w * 3,
                w as u32,
                h as u32,
            )?;
        }
        let value = fast_ssim2::compute_ssimulacra2(
            imgref::Img::new(reference, w, h),
            imgref::Img::new(bytes.as_chunks::<3>().0, w, h),
        )?;
        if !value.is_finite() {
            return Err("nonfinite frozen target".into());
        }
        println!(
            "{}",
            serde_json::json!({"reference":header["reference"],"distorted":row["distorted"],"value":value,"value_bits":format!("{:016x}",value.to_bits()),"reference_pixels_sha256":reference_hash,"distorted_pixels_sha256":hash(&bytes),"implementation":"fast-ssim2@b7c2b4b33f3a26b753caabd5264a7fef9c4f4545;crate=0.8.2"})
        );
    }
    Ok(())
}
