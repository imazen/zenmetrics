//! VMAF scoring. The model scores (v0.6.1, v0.6.1neg, 4k v0.6.1,
//! v1.0.16 3d0h) run in-process through the in-tree `vmaf` crate, a
//! pure-Rust match of libvmaf v3.2.1. The auxiliary libvmaf feature
//! extractors (`float_ssim`, `float_ms_ssim`, `psnr`) have no in-tree
//! port, so they still shell out to the libvmaf `vmaf` executable;
//! `ZENMETRICS_VMAF_BIN` can point to a non-PATH build.
//! Input is converted from encoded sRGB to 8-bit BT.709 limited-range YUV420.

use crate::decode::Rgb8Image;
use std::{fs::File, io::Write, process::Command};

#[derive(Clone, Copy)]
pub enum Model {
    V061,
    V061Neg,
    V061FourK,
    V1,
}

/// Score one pair with a built-in VMAF model via the in-tree `vmaf` crate.
#[cfg(feature = "cpu-vmaf")]
pub fn score(
    r: &Rgb8Image,
    d: &Rgb8Image,
    model: Model,
) -> Result<f64, Box<dyn std::error::Error>> {
    check_dims(r, d)?;
    let widen = |p: Vec<u8>| p.into_iter().map(u16::from).collect::<Vec<u16>>();
    let (ry, ru, rv) = to_yuv420(r)?;
    let (dy, du, dv) = to_yuv420(d)?;
    let (ry, ru, rv) = (widen(ry), widen(ru), widen(rv));
    let (dy, du, dv) = (widen(dy), widen(du), widen(dv));
    let reference = [vmaf::Yuv420Frame {
        y: &ry,
        u: &ru,
        v: &rv,
    }];
    let distorted = [vmaf::Yuv420Frame {
        y: &dy,
        u: &du,
        v: &dv,
    }];
    let (w, h) = (r.width as usize, r.height as usize);
    let v0 = |variant| -> Result<f64, Box<dyn std::error::Error>> {
        let frames = vmaf::score_v0_420(&reference, &distorted, w, h, 8, variant)?;
        Ok(frames.first().ok_or("vmaf returned no frame")?.score)
    };
    match model {
        Model::V061 => v0(vmaf::VmafV0Variant::Standard),
        Model::V061Neg => v0(vmaf::VmafV0Variant::StandardNeg),
        Model::V061FourK => v0(vmaf::VmafV0Variant::FourK),
        Model::V1 => {
            let frames = vmaf::score_v1_420(
                &reference,
                &distorted,
                w,
                h,
                8,
                vmaf::ModelVariant::Standard1080p,
            )?;
            Ok(frames.first().ok_or("vmaf returned no frame")?.score)
        }
    }
}

/// Extract a single auxiliary libvmaf feature for AIC2026-style parity.
/// Requires the libvmaf `vmaf` executable.
pub fn feature(
    r: &Rgb8Image,
    d: &Rgb8Image,
    feature: &'static str,
    key: &'static str,
) -> Result<f64, Box<dyn std::error::Error>> {
    check_dims(r, d)?;
    let dir = tempfile::tempdir()?;
    let rp = dir.path().join("reference.yuv");
    let dp = dir.path().join("distorted.yuv");
    let out = dir.path().join("score.json");
    write_yuv420(&rp, r)?;
    write_yuv420(&dp, d)?;
    let program = std::env::var_os("ZENMETRICS_VMAF_BIN").unwrap_or_else(|| "vmaf".into());
    let run = Command::new(program)
        .arg("--reference")
        .arg(&rp)
        .arg("--distorted")
        .arg(&dp)
        .arg("--width")
        .arg(r.width.to_string())
        .arg("--height")
        .arg(r.height.to_string())
        .arg("--pixel_format")
        .arg("420")
        .arg("--bitdepth")
        .arg("8")
        .arg("--model")
        .arg("version=vmaf_v0.6.1")
        .arg("--feature")
        .arg(feature)
        .arg("--output")
        .arg(&out)
        .arg("--json")
        .arg("--quiet")
        .output()
        .map_err(|e| format!("cannot run libvmaf tool (set ZENMETRICS_VMAF_BIN): {e}"))?;
    if !run.status.success() {
        return Err(format!("libvmaf failed: {}", String::from_utf8_lossy(&run.stderr)).into());
    }
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&out)?)?;
    let score = report["pooled_metrics"][key]["mean"]
        .as_f64()
        .or_else(|| report["frames"][0]["metrics"][key].as_f64())
        .ok_or_else(|| format!("libvmaf JSON lacks {key} score"))?;
    Ok(score)
}

fn check_dims(r: &Rgb8Image, d: &Rgb8Image) -> Result<(), Box<dyn std::error::Error>> {
    if r.width != d.width
        || r.height != d.height
        || r.width < 32
        || r.height < 32
        || r.width % 2 != 0
        || r.height % 2 != 0
    {
        return Err("VMAF requires matching, even dimensions >= 32".into());
    }
    Ok(())
}

/// Write `image` as a single-frame 8-bit BT.709 limited-range YUV420
/// plane file (the libvmaf `--pixel_format 420` input layout: Y plane,
/// then Cb, then Cr). `pub` so benchmarks/heaptrack drivers can time the
/// RGB→YUV conversion + serialization half of the adapter without a
/// libvmaf binary present.
pub fn write_yuv420(
    path: &std::path::Path,
    image: &Rgb8Image,
) -> Result<(), Box<dyn std::error::Error>> {
    let (y, cb, cr) = to_yuv420(image)?;
    let mut f = File::create(path)?;
    f.write_all(&y)?;
    f.write_all(&cb)?;
    f.write_all(&cr)?;
    Ok(())
}

/// Convert `image` to 8-bit BT.709 limited-range YUV420 planes (Y, Cb, Cr).
#[allow(clippy::type_complexity)]
fn to_yuv420(image: &Rgb8Image) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), Box<dyn std::error::Error>> {
    let w = image.width as usize;
    let h = image.height as usize;
    if image.pixels.len() != w * h * 3 {
        return Err("VMAF expected RGB8 pixel buffer".into());
    }
    let mut y = vec![0u8; w * h];
    let mut cb = vec![0u8; w * h / 4];
    let mut cr = vec![0u8; w * h / 4];
    for by in 0..h / 2 {
        for bx in 0..w / 2 {
            let mut u_sum = 0.0;
            let mut v_sum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = (2 * by + dy) * w + 2 * bx + dx;
                    let rgb = &image.pixels[3 * p..3 * p + 3];
                    let r = f64::from(rgb[0]) / 255.0;
                    let g = f64::from(rgb[1]) / 255.0;
                    let b = f64::from(rgb[2]) / 255.0;
                    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    y[p] = (16.0_f64 + 219.0 * lum).round().clamp(16.0, 235.0) as u8;
                    u_sum += -0.114572 * r - 0.385428 * g + 0.5 * b;
                    v_sum += 0.5 * r - 0.454153 * g - 0.045847 * b;
                }
            }
            cb[by * (w / 2) + bx] =
                (128.0_f64 + 224.0 * u_sum / 4.0).round().clamp(16.0, 240.0) as u8;
            cr[by * (w / 2) + bx] =
                (128.0_f64 + 224.0 * v_sum / 4.0).round().clamp(16.0, 240.0) as u8;
        }
    }
    Ok((y, cb, cr))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> (Rgb8Image, Rgb8Image) {
        let pixels: Vec<u8> = (0..256 * 256)
            .flat_map(|i| {
                let v = ((i * 73) % 256) as u8;
                [v, v.wrapping_add(17), v.wrapping_add(39)]
            })
            .collect();
        let r = Rgb8Image {
            pixels: pixels.clone(),
            width: 256,
            height: 256,
        };
        let mut d = Rgb8Image {
            pixels,
            width: 256,
            height: 256,
        };
        for (i, value) in d.pixels.iter_mut().enumerate() {
            if i % 7 == 0 {
                *value = value.saturating_add(12);
            }
        }
        (r, d)
    }

    #[cfg(feature = "cpu-vmaf")]
    #[test]
    fn in_tree_models_score_distortion_below_identity() {
        let (r, d) = pair();
        for model in [Model::V061, Model::V061Neg, Model::V061FourK, Model::V1] {
            let same = score(&r, &r, model).unwrap();
            let dist = score(&r, &d, model).unwrap();
            assert!(same.is_finite() && dist.is_finite());
            assert!(
                dist < same,
                "distorted {dist} should score below identical {same}"
            );
        }
    }

    #[test]
    fn libvmaf_features_smoke_when_configured() {
        if std::env::var_os("ZENMETRICS_VMAF_BIN").is_none() {
            return;
        }
        let (r, d) = pair();
        for (name, key) in [
            ("float_ssim", "float_ssim"),
            ("float_ms_ssim", "float_ms_ssim"),
            ("psnr", "psnr_y"),
        ] {
            assert!(feature(&r, &d, name, key).unwrap().is_finite());
        }
    }
}
