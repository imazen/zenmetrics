#![forbid(unsafe_code)]
//! Opaque sRGB8 rows, a cached reference and raw spatial error maps.
use wasm_bindgen::prelude::*;
// These are the native CLI adapters themselves, so browser variants preserve
// each metric's ingress and implementation identity.
#[path = "../../zenmetrics-cli/src/decode.rs"]
#[allow(dead_code, unreachable_pub, unexpected_cfgs)]
mod decode;
#[path = "../../zenmetrics-cli/src/metrics/mod.rs"]
#[allow(dead_code, unreachable_pub, unexpected_cfgs)]
mod metrics;

use zensim::{DiffmapWeighting, PrecomputedReference, RgbSlice, Zensim, ZensimProfile};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

fn rgb(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Vec<[u8; 3]>, String> {
    let row = (width as usize).checked_mul(4).ok_or("width overflow")?;
    let required = (height as usize)
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(row))
        .ok_or("empty image or dimension overflow")?;
    if width == 0 || stride < row || rgba.len() < required {
        return Err("invalid dimensions, stride or buffer length".into());
    }
    let count = (width as usize)
        .checked_mul(height as usize)
        .ok_or("image size overflow")?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(count).map_err(|e| e.to_string())?;
    for y in 0..height as usize {
        for p in rgba[y * stride..y * stride + row].as_chunks::<4>().0 {
            if p[3] != 255 {
                return Err(
                    "opaque sRGB8 required; choose an explicit alpha composite before scoring"
                        .into(),
                );
            }
            pixels.push([p[0], p[1], p[2]]);
        }
    }
    Ok(pixels)
}

/// Reference preparation retained by the coordinator worker across candidates.
#[wasm_bindgen]
pub struct Reference {
    scorer: Zensim,
    reference: PrecomputedReference,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl Reference {
    /// Accept declared, opaque sRGB8 rows with a byte stride.
    #[wasm_bindgen(constructor)]
    pub fn new(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Reference, JsError> {
        let pixels = rgb(rgba, width, height, stride).map_err(|e| JsError::new(&e))?;
        let scorer = Zensim::new(ZensimProfile::codec_target());
        let source = RgbSlice::new(&pixels, width as usize, height as usize);
        let reference = scorer
            .precompute_reference(&source)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            scorer,
            reference,
            width,
            height,
        })
    }

    /// Score once, retaining both the scalar and the unnormalized map.
    pub fn compare(&self, rgba: &[u8], stride: usize) -> Result<Comparison, JsError> {
        let pixels = rgb(rgba, self.width, self.height, stride).map_err(|e| JsError::new(&e))?;
        let source = RgbSlice::new(&pixels, self.width as usize, self.height as usize);
        let result = self
            .scorer
            .compute_with_ref_and_diffmap(&self.reference, &source, DiffmapWeighting::default())
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Comparison {
            score: result.score(),
            map: result.diffmap().to_vec(),
        })
    }
}

/// Scalar and raw map from the same comparison.
#[wasm_bindgen]
pub struct Comparison {
    score: f64,
    map: Vec<f32>,
}

#[wasm_bindgen]
impl Comparison {
    /// The codec-target zensim score.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.score
    }
    /// Raw row-major error values; no visualization normalization.
    #[wasm_bindgen(getter)]
    pub fn map(&self) -> Vec<f32> {
        self.map.clone()
    }
}

/// Actual initialized Rayon pool size, for browser qualification.
#[wasm_bindgen]
pub fn thread_count() -> usize {
    rayon::current_num_threads()
}

#[cfg(test)]
mod tests {
    use super::rgb;
    #[test]
    fn ingress_preserves_strided_rows_and_refuses_alpha() {
        let packed = [1, 2, 3, 255, 4, 5, 6, 255];
        let padded = [1, 2, 3, 255, 99, 99, 4, 5, 6, 255];
        assert_eq!(
            rgb(&packed, 1, 2, 4).unwrap(),
            rgb(&padded, 1, 2, 6).unwrap()
        );
        assert_eq!(rgb(&packed, 1, 2, 4).unwrap(), vec![[1, 2, 3], [4, 5, 6]]);
        assert!(rgb(&[1, 2, 3, 0], 1, 1, 4).is_err());
        assert!(rgb(&packed, 1, 2, 3).is_err());
        assert!(rgb(&packed, 1, 0, 4).is_err());
        assert!(rgb(&packed, u32::MAX, u32::MAX, usize::MAX).is_err());
    }
}

/// The SDR CPU metrics accepted by this browser module; HDR and GPU names are
/// refused rather than substituted. CVVDP requires an explicit display name.
#[wasm_bindgen]
pub fn available_metrics() -> String {
    use clap::ValueEnum;
    serde_json::to_string(
        &metrics::MetricKind::value_variants()
            .iter()
            .filter(|m| {
                !m.requires_gpu()
                    && !matches!(
                        m,
                        metrics::MetricKind::Hdrvdp | metrics::MetricKind::Hdrvdp3
                    )
            })
            .map(|m| m.name())
            .collect::<Vec<_>>(),
    )
    .expect("static metric names serialize")
}

/// Score a selected SDR CPU metric using the same adapter as the native CLI.
/// Reference and distorted rows both use the supplied byte stride.
#[wasm_bindgen]
pub fn score_rgba(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<String, JsError> {
    let kind = <metrics::MetricKind as clap::ValueEnum>::from_str(metric, false)
        .map_err(|e| JsError::new(&e))?;
    if kind.requires_gpu()
        || matches!(
            kind,
            metrics::MetricKind::Hdrvdp | metrics::MetricKind::Hdrvdp3
        )
    {
        return Err(JsError::new(
            "Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable",
        ));
    }
    let image = |bytes: &[u8]| -> Result<decode::Rgb8Image, JsError> {
        let pixels = rgb(bytes, width, height, stride).map_err(|e| JsError::new(&e))?;
        Ok(decode::Rgb8Image {
            pixels: pixels.into_iter().flatten().collect(),
            width,
            height,
        })
    };
    let r = image(reference)?;
    let d = image(distorted)?;
    let screen = if kind.needs_display() {
        Some(metrics::display::parse_display(display).map_err(|e| JsError::new(&e))?)
    } else {
        None
    };
    let values =
        metrics::run_metric_display(kind, &r, &d, metrics::GpuRuntime::Cpu, screen.as_ref())
            .map_err(|e| JsError::new(&e.to_string()))?;
    let values: Vec<_> = values.into_iter().map(|(column,value)| serde_json::json!({"column":column,
        "value":if value.is_finite() {serde_json::json!(value)} else {serde_json::json!(value.to_string())}})).collect();
    Ok(
        serde_json::json!({"metric":metric,"backend":"native-cpu-wasm","values":values})
            .to_string(),
    )
}

/// Decode SDR file bytes with Zen codecs, preserving PNG/WebP/JXL alpha.
/// Returns dimensions followed by little-endian packed RGBA8 pixels.
/// ICC profiles are not applied; scoring still requires explicit opaque input.
#[wasm_bindgen]
pub fn decode_image(bytes: &[u8], filename: &str) -> Result<Vec<u32>, JsError> {
    fn rgba(buffer: &zenpixels::PixelBuffer) -> Result<(Vec<u8>, u32, u32), String> {
        let pixels = buffer.as_slice();
        if matches!(
            pixels.descriptor().transfer,
            zenpixels::TransferFunction::Pq | zenpixels::TransferFunction::Hlg
        ) {
            return Err("HDR input requires an absolute-nits scoring path".into());
        }
        let (w, h) = (pixels.width(), pixels.rows());
        let mut output = vec![0; w as usize * h as usize * 4];
        let mut converter = zenpixels_convert::converter::RowConverter::new(
            pixels.descriptor(),
            zenpixels::PixelDescriptor::RGBA8_SRGB,
        )
        .map_err(|e| e.to_string())?;
        converter
            .convert_rows(
                pixels.as_strided_bytes(),
                pixels.stride(),
                &mut output,
                w as usize * 4,
                w,
                h,
            )
            .map_err(|e| e.to_string())?;
        Ok((output, w, h))
    }
    let decoded: Result<(Vec<u8>, u32, u32), String> = if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        zenpng::decode(
            bytes,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )
        .map_err(|e| e.to_string())
        .and_then(|output| {
            if output
                .info
                .cicp
                .as_ref()
                .is_some_and(|c| matches!(c.transfer_characteristics, 16 | 18))
            {
                return Err("HDR PNG requires an absolute-nits scoring path".into());
            }
            rgba(&output.pixels)
        })
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        zenwebp::decoder::decode_rgba(bytes).map_err(|e| e.to_string())
    } else if bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(b"\0\0\0\x0cJXL ") {
        zenjxl::decode(bytes, None, &[])
            .map_err(|e| e.to_string())
            .and_then(|output| rgba(&output.pixels))
    } else {
        decode::decode_rgb8_from_bytes(bytes, filename)
            .map_err(|e| e.to_string())
            .map(|image| {
                let mut output =
                    Vec::with_capacity(image.width as usize * image.height as usize * 4);
                for pixel in image.pixels.as_chunks::<3>().0 {
                    output.extend_from_slice(pixel);
                    output.push(255);
                }
                (output, image.width, image.height)
            })
    };
    let (pixels, width, height) = decoded.map_err(|e| JsError::new(&e))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(width as usize * height as usize + 2)
        .map_err(|e| JsError::new(&e.to_string()))?;
    output.extend([width, height]);
    output.extend(
        pixels
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| u32::from_le_bytes([p[0], p[1], p[2], p[3]])),
    );
    Ok(output)
}

/// Generate an actual JPEG candidate with zenjpeg. Rows support byte stride.
#[wasm_bindgen]
pub fn encode_jpeg(
    rgba: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    quality: f32,
) -> Result<Vec<u8>, JsError> {
    use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
    let pixels = rgb(rgba, width, height, stride).map_err(|e| JsError::new(&e))?;
    let packed: Vec<_> = pixels.into_iter().flatten().collect();
    let mut encoder = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .map_err(|e| JsError::new(&e.to_string()))?;
    encoder
        .push_packed(&packed, enough::Unstoppable)
        .map_err(|e| JsError::new(&e.to_string()))?;
    encoder.finish().map_err(|e| JsError::new(&e.to_string()))
}

/// Native spatial output, with its actual sampling grid and units.
#[wasm_bindgen]
pub struct SpatialMap {
    width: u32,
    height: u32,
    similarity: bool,
    values: Vec<f32>,
}
#[wasm_bindgen]
impl SpatialMap {
    /// Native map width in samples.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Native map height in samples.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// Similarity maps are visualized as 1-value; raw values remain unchanged.
    #[wasm_bindgen(getter)]
    pub fn similarity(&self) -> bool {
        self.similarity
    }
    /// Unnormalized row-major native samples.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
}

/// Compute a metric's native spatial output. Unsupported map requests fail
/// explicitly; a different metric is never substituted.
#[wasm_bindgen]
pub fn spatial_map(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<SpatialMap, JsError> {
    let r = rgb(reference, width, height, stride).map_err(|e| JsError::new(&e))?;
    let d = rgb(distorted, width, height, stride).map_err(|e| JsError::new(&e))?;
    let (w, h) = (width as usize, height as usize);
    let fail = |e: String| JsError::new(&e);
    let mut result = SpatialMap {
        width,
        height,
        similarity: false,
        values: Vec::new(),
    };
    match metric {
        "zensim" => {
            let prepared = Reference::new(reference, width, height, stride)?;
            result.values = prepared.compare(distorted, stride)?.map;
        }
        "butteraugli" => {
            let r: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
            let d: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
            let scored = butteraugli::butteraugli(
                butteraugli::ImgRef::new(r, w, h),
                butteraugli::ImgRef::new(d, w, h),
                &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
            )
            .map_err(|e| fail(e.to_string()))?;
            result.values = scored
                .diffmap
                .ok_or_else(|| fail("Butteraugli returned no spatial map".into()))?
                .buf()
                .to_vec();
        }
        "dssim" => {
            let mut scorer = dssim_core::Dssim::new();
            scorer.set_save_ssim_maps(1);
            let r = scorer
                .create_image_rgb(bytemuck::cast_slice(&r), w, h)
                .ok_or_else(|| fail("DSSIM rejected reference dimensions".into()))?;
            let d = scorer
                .create_image_rgb(bytemuck::cast_slice(&d), w, h)
                .ok_or_else(|| fail("DSSIM rejected distorted dimensions".into()))?;
            let (_, maps) = scorer.compare(&r, d);
            let first = maps
                .into_iter()
                .next()
                .ok_or_else(|| fail("DSSIM returned no spatial map".into()))?;
            result.width = first.map.width() as u32;
            result.height = first.map.height() as u32;
            result.similarity = true;
            result.values = first.map.buf().to_vec();
        }
        "cvvdp" => {
            let display = metrics::display::parse_display(display).map_err(fail)?;
            let mut scorer = cvvdp::Cvvdp::with_geometry(
                width,
                height,
                cvvdp::CvvdpParams {
                    display: display.model(),
                    ..Default::default()
                },
                display.geometry(),
            )
            .map_err(|e| fail(e.to_string()))?;
            scorer
                .score_with_diffmap(
                    bytemuck::cast_slice(&r),
                    bytemuck::cast_slice(&d),
                    &mut result.values,
                )
                .map_err(|e| fail(e.to_string()))?;
        }
        "gmsd" => {
            let mut rg = vec![0.; w * h];
            let mut dg = vec![0.; w * h];
            gmsd::rgb8_to_gray(bytemuck::cast_slice(&r), w, h, w * 3, &mut rg)
                .map_err(|e| fail(e.to_string()))?;
            gmsd::rgb8_to_gray(bytemuck::cast_slice(&d), w, h, w * 3, &mut dg)
                .map_err(|e| fail(e.to_string()))?;
            let (mw, mh) = gmsd::map_dims(w, h);
            result.width = mw as u32;
            result.height = mh as u32;
            result.similarity = true;
            result.values = vec![0.; mw * mh];
            gmsd::gmsd_with_map(
                gmsd::GrayImage::packed(&rg, w, h).map_err(|e| fail(e.to_string()))?,
                gmsd::GrayImage::packed(&dg, w, h).map_err(|e| fail(e.to_string()))?,
                &mut result.values,
            )
            .map_err(|e| fail(e.to_string()))?;
        }
        "psnr" => {
            // Per-pixel mean squared sRGB code-value error, in normalized units.
            result.values = r
                .iter()
                .zip(&d)
                .map(|(a, b)| {
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| {
                            let delta = (*x as f32 - *y as f32) / 255.;
                            delta * delta
                        })
                        .sum::<f32>()
                        / 3.
                })
                .collect();
        }
        _ => {
            return Err(fail(format!(
                "{metric} has no native spatial-map export in this build"
            )));
        }
    }
    if result.values.len() != result.width as usize * result.height as usize
        || result.values.iter().any(|v| !v.is_finite())
    {
        return Err(fail(
            "Native spatial map has invalid dimensions or non-finite samples".into(),
        ));
    }
    Ok(result)
}
