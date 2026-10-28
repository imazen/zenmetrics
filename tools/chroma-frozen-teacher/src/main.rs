//! Source replay of the SafeSyn teacher. No current-metric fallback.
#![forbid(unsafe_code)]
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, BufRead},
    path::Path,
};
struct Image {
    width: usize,
    height: usize,
    pixels: Vec<[u8; 3]>,
}
impl Image {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let out = zenpng::decode(
            &fs::read(path)?,
            &zenpng::PngDecodeConfig::strict(),
            &enough::Unstoppable,
        )?;
        // Preserve the archived extract_features_372col/shared/zen_decode.rs
        // legacy projection: direct RGB8, alpha drop for RGBA8, canonical row
        // conversion for other native layouts (including lossless PNG palettes).
        let (width, height) = (out.pixels.width() as usize, out.pixels.height() as usize);
        let desc = out.pixels.descriptor();
        let source = out.pixels.as_slice();
        let mut bytes = Vec::with_capacity(width * height * 3);
        if desc.layout_compatible(zenpixels::PixelDescriptor::RGB8_SRGB) {
            for y in 0..height {
                bytes.extend_from_slice(&source.row(y as u32)[..width * 3]);
            }
        } else if desc.layout_compatible(zenpixels::PixelDescriptor::RGBA8_SRGB) {
            for y in 0..height {
                for p in source.row(y as u32)[..width * 4].as_chunks::<4>().0 {
                    bytes.extend_from_slice(&p[..3]);
                }
            }
        } else {
            bytes.resize(width * height * 3, 0);
            let mut converter =
                zenpixels_convert::RowConverter::new(desc, zenpixels::PixelDescriptor::RGB8_SRGB)?;
            converter.convert_rows(
                source.as_strided_bytes(),
                source.stride(),
                &mut bytes,
                width * 3,
                width as u32,
                height as u32,
            )?;
        }
        let pixels = bytes.as_chunks::<3>().0.to_vec();
        Ok(Self {
            width,
            height,
            pixels,
        })
    }
    fn hash(&self) -> String {
        let mut digest = Sha256::new();
        for p in &self.pixels {
            digest.update(p);
        }
        digest
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // JSONL pairs on stdin allow one warm process for replay and job execution.
    for line in io::stdin().lock().lines() {
        let p: serde_json::Value = serde_json::from_str(&line?)?;
        let r = Image::load(Path::new(
            p["reference"].as_str().ok_or("reference missing")?,
        ))?;
        let d = Image::load(Path::new(
            p["distorted"].as_str().ok_or("distorted missing")?,
        ))?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err("reference/candidate geometry differs".into());
        }
        let score = fast_ssim2::compute_ssimulacra2(
            imgref::Img::new(r.pixels.as_slice(), r.width, r.height),
            imgref::Img::new(d.pixels.as_slice(), d.width, d.height),
        )?;
        if !score.is_finite() {
            return Err("nonfinite frozen teacher score".into());
        }
        println!(
            "{}",
            serde_json::json!({"reference":p["reference"],"distorted":p["distorted"],"value":score,"value_bits":format!("{:016x}",score.to_bits()),"reference_pixels_sha256":r.hash(),"distorted_pixels_sha256":d.hash(),"implementation":"fast-ssim2@b7c2b4b33f3a26b753caabd5264a7fef9c4f4545;crate=0.8.2"})
        );
    }
    Ok(())
}
