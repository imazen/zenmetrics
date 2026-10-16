//! Internal indexed-only job executor. No orchestration or source discovery.
use crate::decode::Rgb8Image;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::time::Instant;
use zenpng::Quantizer;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema: String,
    joint: bool,
    format: String,
    quality: String,
    dither: String,
    colors: u32,
    effort: u32,
    priority: String,
    strength: Option<f32>,
    lambda: Option<f32>,
    sample_cap: usize,
    tolerance: f32,
    target: Option<f32>,
    minimum: Option<f32>,
    metrics: bool,
    joint_effort: u32,
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

impl Config {
    fn resolve(&self) -> Result<zenquant::QuantizeConfig, Box<dyn Error>> {
        if self.schema != "indexed-ladder-v1" || self.joint != cfg!(feature = "palette-joint") {
            return Err("palette schema/build closure mismatch".into());
        }
        if !(2..=256).contains(&self.colors)
            || !(self.effort <= 31 || self.effort == 200)
            || ![0, 131_072].contains(&self.sample_cap)
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.
            || !(1..=22).contains(&self.joint_effort)
            || self
                .strength
                .is_some_and(|v| !v.is_finite() || !(0. ..=1.).contains(&v))
            || self.lambda.is_some_and(|v| !v.is_finite() || v < 0.)
            || [self.target, self.minimum]
                .into_iter()
                .flatten()
                .any(|v| !v.is_finite() || !(0. ..=100.).contains(&v))
        {
            return Err("invalid palette ladder parameters".into());
        }
        let format = match self.format.as_str() {
            "png" => zenquant::OutputFormat::Png,
            "joint" if self.joint => zenquant::OutputFormat::PngJoint,
            "min-size" if self.joint => zenquant::OutputFormat::PngMinSize,
            _ => return Err("unsupported palette format/build".into()),
        };
        let quality = match self.quality.as_str() {
            "fast" => zenquant::Quality::Fast,
            "balanced" => zenquant::Quality::Balanced,
            "best" => zenquant::Quality::Best,
            _ => return Err("unsupported palette quality".into()),
        };
        let mut c = zenquant::QuantizeConfig::new(format)
            .with_max_colors(self.colors)
            .with_quality(quality)
            .with_compute_quality_metric(self.metrics)
            .with_kmeans_sample_cap(self.sample_cap)
            ._with_joint_tolerance(self.tolerance)
            ._with_joint_deflate_effort(self.joint_effort);
        c = match self.dither.as_str() {
            "none" => c._with_no_dither(),
            "adaptive" => c._with_adaptive_dither(),
            "sierra-lite" => c._with_sierra_lite_dither(),
            "blue-noise" => c._with_blue_noise_dither(),
            "linear" => c._with_linear_dither(),
            "ordered" => c._with_ordered_dither(),
            _ => return Err("unsupported palette dither".into()),
        };
        c = match self.priority.as_str() {
            "quality" => c._with_run_priority_quality(),
            "compression" => c._with_run_priority_compression(),
            _ => return Err("unsupported palette priority".into()),
        };
        if let Some(v) = self.strength {
            c = c._with_dither_strength(v);
        }
        if let Some(v) = self.lambda {
            c = c._with_viterbi_lambda(v);
        }
        if let Some(v) = self.target {
            c = c.with_target_ssim2(v);
        }
        if let Some(v) = self.minimum {
            c = c.with_min_ssim2(v);
        }
        Ok(c)
    }
}

// encode_indexed calls this quantizer, keeping precisely the already captured
// palette/indices across compression efforts and avoiding any truecolor fallback.
struct Captured(zenpng::QuantizeOutput);
impl Quantizer for Captured {
    fn quantize_rgba(
        &self,
        _: &[[u8; 4]],
        _: usize,
        _: usize,
    ) -> Result<zenpng::QuantizeOutput, zenpng::PngError> {
        Ok(self.0.clone())
    }
    fn name(&self) -> &str {
        "captured-zenquant"
    }
}

pub(crate) fn execute(
    source: &Rgb8Image,
    value: &Value,
) -> Result<(Option<Vec<u8>>, Value), Box<dyn Error>> {
    let c: Config = serde_json::from_value(value.clone())?;
    let config = c.resolve()?; // Refuse configuration before touching pixels.
    let (w, h) = (source.width as usize, source.height as usize);
    if w == 0
        || h == 0
        || w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(source.pixels.len())
    {
        return Err("palette source geometry mismatch".into());
    }
    let pixels: Vec<_> = source
        .pixels
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| rgb::Rgba::new(p[0], p[1], p[2], 255))
        .collect();
    let quantizer = zenpng::ZenquantQuantizer::from_config(config);
    let started = Instant::now();
    let output = match quantizer.quantize_rgba(
        &pixels
            .iter()
            .map(|p| [p.r, p.g, p.b, p.a])
            .collect::<Vec<_>>(),
        w,
        h,
    ) {
        Ok(v) => v,
        Err(zenpng::PngError::Quantize(zenquant::QuantizeError::QualityNotMet {
            min_ssim2,
            achieved_ssim2,
        })) if c.minimum.is_some() => {
            return Ok((
                None,
                json!({
                    "schema":"indexed-receipt-v1", "status":"quality-floor", "requested":value,
                    "minimum":min_ssim2, "achieved":achieved_ssim2,
                    "quantize_seconds":started.elapsed().as_secs_f64()
                }),
            ));
        }
        Err(e) => return Err(e.into()),
    };
    let quantize_seconds = started.elapsed().as_secs_f64();
    let decoded: Vec<u8> = output
        .indices
        .iter()
        .flat_map(|&i| output.palette_rgba[i as usize][..3].to_vec())
        .collect();
    let mut receipt = json!({
        "schema":"indexed-receipt-v1", "status":"ok", "requested":value,
        "width":w, "height":h, "stride":w*3, "colour":"rgb8-srgb-opaque",
        "source_pixels_sha256":hash(&source.pixels), "decoded_pixels_sha256":hash(&decoded),
        "palette_rgba":output.palette_rgba, "indices_sha256":hash(&output.indices),
        "indices":output.indices, "quantize_seconds":quantize_seconds,
        "mpe":output.mpe_score,"ssim2_estimate":output.ssim2_estimate,
        "butteraugli_estimate":output.butteraugli_estimate,
        "resolved_requested_config_debug":format!("{:?}", quantizer.config()),
        "build_commit":env!("ZEN_METRICS_COMMIT")
    });
    let started = Instant::now();
    let cfg = zenpng::EncodeConfig::default()
        .with_compression(zenpng::Compression::Effort(c.effort))
        .with_parallel(false);
    let bytes = zenpng::encode_indexed(
        imgref::ImgRef::new(&pixels, w, h),
        &cfg,
        &Captured(output),
        None,
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    receipt["encode_seconds"] = json!(started.elapsed().as_secs_f64());
    receipt["encoded_sha256"] = json!(hash(&bytes));
    receipt["encoded_bytes"] = json!(bytes.len());
    let actual = crate::decode::decode_rgb8_from_bytes(&bytes, "indexed.png")?;
    if actual.width != source.width || actual.height != source.height || actual.pixels != decoded {
        return Err("indexed decode disagrees with palette/indices".into());
    }
    Ok((Some(bytes), receipt))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn config() -> Value {
        json!({"schema":"indexed-ladder-v1", "joint":cfg!(feature="palette-joint"),
            "format":"png","quality":"best","dither":"adaptive","colors":2,
            "effort":0,"priority":"quality","strength":null,"lambda":null,
            "sample_cap":131072,"tolerance":0.01,"target":null,"minimum":null,
            "metrics":false,"joint_effort":10})
    }
    #[test]
    fn indexed_pixels_match_independent_decoder_and_receipt() {
        let source = Rgb8Image {
            width: 4,
            height: 2,
            pixels: vec![
                0, 0, 0, 255, 255, 255, 25, 32, 56, 89, 43, 21, 12, 245, 54, 23, 44, 192, 182, 72,
                32, 255, 0, 0,
            ],
        };
        let (bytes, receipt) = execute(&source, &config()).unwrap();
        let bytes = bytes.unwrap();
        assert_eq!(bytes[25], 3, "PNG IHDR must be indexed");
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .unwrap()
            .into_rgb8();
        assert_eq!(hash(decoded.as_raw()), receipt["decoded_pixels_sha256"]);
        assert_eq!(hash(&bytes), receipt["encoded_sha256"]);
        assert_eq!(receipt["palette_rgba"].as_array().unwrap().len(), 2);
        assert_eq!(receipt["indices"].as_array().unwrap().len(), 8);
    }
    #[test]
    fn bad_build_and_unknown_parameters_refuse_before_pixels() {
        let source = Rgb8Image {
            width: 0,
            height: 0,
            pixels: vec![],
        };
        let mut bad = config();
        bad["joint"] = json!(!cfg!(feature = "palette-joint"));
        assert!(
            execute(&source, &bad)
                .unwrap_err()
                .to_string()
                .contains("closure")
        );
        let mut bad = config();
        bad["misspelled"] = json!(true);
        assert!(
            execute(&source, &bad)
                .unwrap_err()
                .to_string()
                .contains("unknown field")
        );
        let mut bad = config();
        bad["colors"] = json!(1);
        assert!(
            execute(&source, &bad)
                .unwrap_err()
                .to_string()
                .contains("parameters")
        );
    }
    #[test]
    fn ignored_joint_effort_retains_request_but_preserves_bytes() {
        let source = Rgb8Image {
            width: 2,
            height: 2,
            pixels: vec![0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255],
        };
        let mut a = config();
        a["joint_effort"] = json!(1);
        let mut b = a.clone();
        b["joint_effort"] = json!(22);
        let (ab, ar) = execute(&source, &a).unwrap();
        let (bb, br) = execute(&source, &b).unwrap();
        assert_eq!(ab, bb);
        assert_ne!(ar["requested"], br["requested"]);
    }
}
