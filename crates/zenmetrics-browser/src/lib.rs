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
mod ssim2_map;

use zensim::{DiffmapWeighting, PrecomputedReference, RgbSlice, Zensim, ZensimProfile};

#[cfg(target_arch = "wasm32")]
pub use wasm_bindgen_rayon::init_thread_pool;

use how_far::{Report, Stop, StopReason};
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering},
};

thread_local! {
    static ACTIVE_CONTROL: RefCell<Option<JobControl>> = const { RefCell::new(None) };
}

/// Shared cancellation and completed-work counters, accessible while WASM runs.
/// Layout: cancelled, phase, completed, total. A zero total is indeterminate.
#[wasm_bindgen]
#[derive(Clone)]
pub struct JobControl {
    state: Arc<[AtomicU32; 4]>,
}

impl Stop for JobControl {
    fn check(&self) -> Result<(), StopReason> {
        if self.state[0].load(Ordering::Acquire) != 0 {
            Err(StopReason::Cancelled)
        } else {
            Ok(())
        }
    }
}
impl Report for JobControl {
    fn advance(&self, completed: u64) {
        self.state[2].fetch_add(completed.min(u32::MAX as u64) as u32, Ordering::Relaxed);
    }
}

#[wasm_bindgen]
impl JobControl {
    /// Install a control retained for this coordinator's subsequent calls.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let control = Self {
            state: Arc::new(std::array::from_fn(|_| AtomicU32::new(0))),
        };
        ACTIVE_CONTROL.with(|slot| *slot.borrow_mut() = Some(control.clone()));
        control
    }
    /// Byte offset into the shared WASM memory; four aligned atomic u32 words.
    pub fn state_ptr(&self) -> usize {
        self.state.as_ptr() as usize
    }
    /// Begin a phase. Only actual completed units may advance its counter.
    pub fn begin(&self, phase: u32, total: u32) -> Result<(), JsError> {
        self.check().map_err(|e| JsError::new(&e.to_string()))?;
        self.state[2].store(0, Ordering::Relaxed);
        self.state[3].store(total, Ordering::Relaxed);
        self.state[1].store(phase, Ordering::Release);
        Ok(())
    }
}
impl Default for JobControl {
    fn default() -> Self {
        Self::new()
    }
}
fn control() -> JobControl {
    ACTIVE_CONTROL
        .with(|slot| slot.borrow().clone())
        .unwrap_or_else(|| JobControl {
            state: Arc::new(std::array::from_fn(|_| AtomicU32::new(0))),
        })
}

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
    let progress = control();
    for y in 0..height as usize {
        progress.check().map_err(|e| e.to_string())?;
        for p in rgba[y * stride..y * stride + row].as_chunks::<4>().0 {
            if p[3] != 255 {
                return Err(
                    "opaque sRGB8 required; choose an explicit alpha composite before scoring"
                        .into(),
                );
            }
            pixels.push([p[0], p[1], p[2]]);
        }
        progress.advance(1);
    }
    Ok(pixels)
}

/// Reference preparation retained by the coordinator worker across candidates.
#[wasm_bindgen]
pub struct Reference {
    scorer: Zensim,
    reference: Option<PrecomputedReference>,
    raw_reference: Option<Vec<[u8; 3]>>,
    width: u32,
    height: u32,
}

#[wasm_bindgen]
impl Reference {
    /// Accept declared, opaque sRGB8 rows with a byte stride.
    #[wasm_bindgen(constructor)]
    pub fn new(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Reference, JsError> {
        Self::with_profile(rgba, width, height, stride, "b")
    }

    /// Prepare an explicit SDR model. HDR models require a different ingress.
    pub fn with_profile(
        rgba: &[u8],
        width: u32,
        height: u32,
        stride: usize,
        profile: &str,
    ) -> Result<Reference, JsError> {
        #[allow(deprecated)]
        let profile = match profile {
            "b" => ZensimProfile::B,
            "a" => ZensimProfile::A,
            "c" => ZensimProfile::C,
            "d" => ZensimProfile::D,
            "preview-v0.1" => ZensimProfile::PreviewV0_1,
            "preview-v0.2" => ZensimProfile::PreviewV0_2,
            _ => return Err(JsError::new("Unsupported SDR Zensim model")),
        };
        let pixels = rgb(rgba, width, height, stride).map_err(|e| JsError::new(&e))?;
        let scorer = Zensim::new(profile).with_stop(control());
        let source = RgbSlice::new(&pixels, width as usize, height as usize);
        let reference = if profile == ZensimProfile::C {
            None
        } else {
            Some(
                scorer
                    .precompute_reference(&source)
                    .map_err(|e| JsError::new(&e.to_string()))?,
            )
        };
        let raw_reference = (profile == ZensimProfile::C).then_some(pixels);
        Ok(Self {
            scorer,
            reference,
            raw_reference,
            width,
            height,
        })
    }

    /// Score without waiting for optional map construction.
    pub fn score(&self, rgba: &[u8], stride: usize) -> Result<f64, JsError> {
        let pixels = rgb(rgba, self.width, self.height, stride).map_err(|e| JsError::new(&e))?;
        let source = RgbSlice::new(&pixels, self.width as usize, self.height as usize);
        let scorer = self.scorer.clone();
        let result = if let Some(reference) = &self.raw_reference {
            // C's canonical fold is available on the full pair entry only at
            // this pin. Worker disposal provides cancellation without changing
            // the selected model or scoring through an incomplete feature vector.
            scorer.with_stop(enough::Unstoppable).compute(
                &RgbSlice::new(reference, self.width as usize, self.height as usize),
                &source,
            )
        } else {
            let reference = self
                .reference
                .as_ref()
                .ok_or_else(|| JsError::new("Missing prepared reference"))?;
            scorer
                .with_stop(control())
                .compute_with_ref(reference, &source)
        };
        result
            .map(|result| result.score())
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Score once, retaining both the scalar and the unnormalized map.
    pub fn compare(&self, rgba: &[u8], stride: usize) -> Result<Comparison, JsError> {
        if self.scorer.profile() == ZensimProfile::C {
            return Err(JsError::new(
                "Model C spatial maps are unavailable in this pinned library; choose another map metric explicitly",
            ));
        }
        let pixels = rgb(rgba, self.width, self.height, stride).map_err(|e| JsError::new(&e))?;
        let source = RgbSlice::new(&pixels, self.width as usize, self.height as usize);
        let result = self
            .scorer
            .clone()
            .with_stop(control())
            .compute_with_ref_and_diffmap(
                self.reference
                    .as_ref()
                    .ok_or_else(|| JsError::new("Missing prepared reference"))?,
                &source,
                DiffmapWeighting::default(),
            )
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
    /// The selected Zensim model score.
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
    use super::{JobControl, rgb};
    use how_far::{Report, Stop};
    use std::sync::atomic::Ordering;

    #[test]
    fn completed_work_is_distinct_from_cancellation_polls() {
        let progress = JobControl::new();
        for _ in 0..10 {
            progress.check().unwrap();
        }
        assert_eq!(progress.state[2].load(Ordering::Relaxed), 0);
        progress.advance(7);
        assert_eq!(progress.state[2].load(Ordering::Relaxed), 7);
        progress.state[0].store(1, Ordering::Release);
        assert!(progress.check().is_err());
        assert!(
            rgb(&[1, 2, 3, 255], 1, 1, 4)
                .unwrap_err()
                .contains("cancel")
        );
    }

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

    /// Opaque RGBA rows `pad` bytes wider than the packed RGB pixels.
    #[cfg(feature = "cpu-metrics")]
    fn rgba(px: &[u8], w: u32, pad: usize) -> (Vec<u8>, usize) {
        let stride = w as usize * 4 + pad;
        let mut out = vec![7u8; stride * (px.len() / (w as usize * 3))];
        for (y, row) in px.chunks_exact(w as usize * 3).enumerate() {
            for (x, p) in row.as_chunks::<3>().0.iter().enumerate() {
                out[y * stride + x * 4..][..4].copy_from_slice(&[p[0], p[1], p[2], 255]);
            }
        }
        (out, stride)
    }

    #[cfg(feature = "cpu-metrics")]
    fn direct(
        metric: &mut dyn zenmetrics_abstractions::Metric,
        r: zenmetrics_abstractions::Image<'_>,
        d: zenmetrics_abstractions::Image<'_>,
    ) -> (zenmetrics_abstractions::MapGeometry, bool, Vec<f32>) {
        use zenmetrics_abstractions as za;
        let e = metric
            .evaluate(r, d, za::EvalOpts::SCORE_ONLY.with_native_map())
            .unwrap();
        let maps = e.maps();
        let m = maps
            .iter()
            .find(|m| m.name == "diffmap")
            .unwrap_or(&maps[0]);
        (
            m.geometry,
            m.polarity == za::MapPolarity::Similarity,
            m.values().into_owned(),
        )
    }

    /// Each newly routed name serves its own metric's primary native map,
    /// sample for sample, through strided RGBA ingress.
    #[cfg(all(
        feature = "cpu-metrics",
        feature = "cpu-psnrhvs",
        feature = "cpu-mdctpsnr",
        feature = "cpu-mad",
        feature = "cpu-iwssim",
        feature = "cpu-vif",
        feature = "cpu-vifvec",
        feature = "cpu-vmaf"
    ))]
    #[test]
    fn native_map_routes_serve_their_own_metric() {
        use zenmetrics_abstractions::{self as za, zenpixels};
        let (w, h) = (224u32, 180u32);
        let (a, b) = (
            za::conformance::textured_rgb8(w as usize, h as usize, 5),
            za::conformance::textured_rgb8(w as usize, h as usize, 6),
        );
        let ((ra, stride), (rb, _)) = (rgba(&a, w, 12), rgba(&b, w, 12));
        let packed = |px| {
            za::Image::from(
                zenpixels::PixelSlice::new(
                    px,
                    w,
                    h,
                    w as usize * 3,
                    zenpixels::PixelDescriptor::RGB8_SRGB,
                )
                .unwrap(),
            )
        };
        let mut iw = iwssim::Iwssim::new(w, h).unwrap();
        let routes: [(&str, &mut dyn za::Metric); 7] = [
            ("nlpd", &mut nlpd::Nlpd),
            ("psnrhvs", &mut psnrhvs::Psnrhvs),
            ("mdctpsnr", &mut mdctpsnr::MdctPsnr),
            ("mad", &mut mad_iqa::Mad),
            ("iwssim", &mut iw),
            ("vif", &mut vif::Vif),
            ("vifvec", &mut vif::Vifvec),
        ];
        let catalog: Vec<String> = serde_json::from_str(&super::available_spatial_maps()).unwrap();
        for (name, metric) in routes {
            assert!(
                catalog.iter().any(|c| c == name),
                "{name} missing from catalog"
            );
            let (geometry, similarity, values) = direct(metric, packed(&a), packed(&b));
            let got = super::spatial_map(name, &ra, &rb, w, h, stride, "").unwrap();
            assert_eq!(got.geometry, Some(geometry), "{name}");
            assert_eq!(
                (got.width, got.height),
                (geometry.cells_x, geometry.cells_y)
            );
            assert_eq!(got.similarity, similarity, "{name}");
            // Every route serves a fused or difference map.
            assert!(!got.similarity, "{name}");
            assert_eq!(
                got.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}"
            );
        }
        // VMAF v1 sees the 4:2:0 frames its score converts to. SPEED cells
        // are 160 pixels, so this pair is larger and differs in chroma.
        assert!(catalog.iter().any(|c| c == "vmaf-v1"));
        let (w, h) = (480u32, 352u32);
        let a = za::conformance::textured_rgb8(w as usize, h as usize, 9);
        let b: Vec<u8> = a
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .flat_map(|(i, p)| [p[0], p[1], if (i / 37) % 3 == 0 { p[2] / 2 } else { p[2] }])
            .collect();
        let yuv = |px: &[u8]| {
            super::metrics::vmaf::to_yuv420(&super::decode::Rgb8Image {
                pixels: px.to_vec(),
                width: w,
                height: h,
            })
            .unwrap()
        };
        let (ya, yb) = (yuv(&a), yuv(&b));
        let (geometry, similarity, values) = direct(
            &mut vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p),
            super::yuv420_image(&ya, w, h).unwrap(),
            super::yuv420_image(&yb, w, h).unwrap(),
        );
        let ((ra, stride), (rb, _)) = (rgba(&a, w, 8), rgba(&b, w, 8));
        let got = super::spatial_map("vmaf-v1", &ra, &rb, w, h, stride, "").unwrap();
        assert_eq!(got.geometry, Some(geometry));
        assert_eq!((geometry.cells_x, geometry.cells_y), (3, 2));
        assert!(!similarity && !got.similarity);
        assert_eq!(got.values, values);
        assert!(got.values.iter().any(|&v| v > 0.0));
    }

    /// Identical odd-sized inputs: NLPD, PSNR-HVS, mDCT-PSNR and MAD vanish
    /// exactly and IW-SSIM's contrast-structure is one to f32 rounding. VIF
    /// keeps a tiny positive loss, because the reference regularizes the gain
    /// `σ12/(σ1² + 1e-10)` and floors `σv²` at 1e-10 (measured: 2.2e-11 for
    /// VIFp, 4.3e-14 for vifvec).
    #[cfg(all(
        feature = "cpu-metrics",
        feature = "cpu-psnrhvs",
        feature = "cpu-mdctpsnr",
        feature = "cpu-mad",
        feature = "cpu-iwssim",
        feature = "cpu-vif",
        feature = "cpu-vifvec"
    ))]
    #[test]
    fn native_map_routes_on_identical_inputs() {
        let (w, h) = (257u32, 253u32);
        let px = zenmetrics_abstractions::conformance::textured_rgb8(w as usize, h as usize, 3);
        let (px, stride) = rgba(&px, w, 4);
        for name in [
            "nlpd", "psnrhvs", "mdctpsnr", "mad", "iwssim", "vif", "vifvec",
        ] {
            let got = super::spatial_map(name, &px, &px, w, h, stride, "").unwrap();
            let g = got.geometry.unwrap();
            assert_eq!((g.input_width, g.input_height), (w, h), "{name}");
            let (lo, hi) = got
                .values
                .iter()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            match name {
                // Identical inputs leave cs within f32 rounding of 1.
                "iwssim" => assert!(!got.similarity && lo >= 0.0 && hi <= 1e-6, "{lo:e} {hi:e}"),
                "vif" | "vifvec" => assert!(lo > 0.0 && hi < 1e-9, "{name}: {lo:e} {hi:e}"),
                _ => assert_eq!((lo, hi), (0.0, 0.0), "{name}"),
            }
        }
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
        zenpng::decode(bytes, &zenpng::PngDecodeConfig::default(), &control())
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
        .push_packed(&packed, control())
        .map_err(|e| JsError::new(&e.to_string()))?;
    encoder.finish().map_err(|e| JsError::new(&e.to_string()))
}

/// Native spatial output, with its actual sampling grid and units.
#[wasm_bindgen]
pub struct SpatialMap {
    width: u32,
    height: u32,
    similarity: bool,
    geometry: Option<zenmetrics_abstractions::MapGeometry>,
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
    /// Native cell ownership; null for legacy exports without this metadata.
    #[wasm_bindgen(getter)]
    pub fn geometry(&self) -> String {
        self.geometry.map_or_else(
            || "null".into(),
            |g| {
                serde_json::json!({
                    "inputWidth":g.input_width,"inputHeight":g.input_height,
                    "originX":g.origin_x,"originY":g.origin_y,
                    "cellWidth":g.cell_width,"cellHeight":g.cell_height,
                    "cellsX":g.cells_x,"cellsY":g.cells_y,
                })
                .to_string()
            },
        )
    }
    /// Unnormalized row-major native samples.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
}

fn export_evaluation_map(
    e: &dyn zenmetrics_abstractions::Evaluation,
    out: &mut SpatialMap,
) -> Result<(), JsError> {
    // A metric's fused `diffmap` (its whole score broken down per pixel)
    // when it publishes one, else its primary map.
    let maps = e.maps();
    let map = maps
        .into_iter()
        .find(|m| m.name == "diffmap")
        .or_else(|| e.map())
        .ok_or_else(|| JsError::new("Native evaluation returned no spatial map"))?;
    out.width = map.width();
    out.height = map.height();
    out.similarity = map.polarity == zenmetrics_abstractions::MapPolarity::Similarity;
    out.geometry = Some(map.geometry);
    out.values = map.values().into_owned();
    Ok(())
}

fn evaluate_spatial_map(
    metric: &mut dyn zenmetrics_abstractions::Metric,
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
    out: &mut SpatialMap,
) -> Result<(), JsError> {
    use za::zenpixels::{PixelDescriptor, PixelSlice};
    use zenmetrics_abstractions as za;
    let image = |pixels| {
        PixelSlice::new(
            pixels,
            width,
            height,
            width as usize * 3,
            PixelDescriptor::RGB8_SRGB,
        )
        .map(za::Image::from)
        .map_err(|e| JsError::new(&e.to_string()))
    };
    evaluate_native_map(
        metric,
        image(bytemuck::cast_slice(r))?,
        image(bytemuck::cast_slice(d))?,
        out,
    )
}

/// The metric's own primary map for an already shaped input pair.
fn evaluate_native_map(
    metric: &mut dyn zenmetrics_abstractions::Metric,
    r: zenmetrics_abstractions::Image<'_>,
    d: zenmetrics_abstractions::Image<'_>,
    out: &mut SpatialMap,
) -> Result<(), JsError> {
    use zenmetrics_abstractions as za;
    let pulse = how_far::StopOnly::new(control());
    let opts = za::EvalOpts::SCORE_ONLY
        .with_native_map()
        .with_pulse(&pulse);
    let evaluation = metric
        .evaluate(r, d, opts)
        .map_err(|e| JsError::new(&e.to_string()))?;
    opts.checkpoint()
        .map_err(|e| JsError::new(&e.to_string()))?;
    export_evaluation_map(evaluation.as_ref(), out)
}

/// VMAF v1.0.16 reads 8-bit BT.709 limited-range 4:2:0, converted exactly as
/// the `vmaf-v1` score converts it.
#[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
fn vmaf_v1_map(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
    out: &mut SpatialMap,
) -> Result<(), JsError> {
    let fail = |e: Box<dyn std::error::Error>| JsError::new(&e.to_string());
    let rgb8 = |p: &[[u8; 3]]| decode::Rgb8Image {
        pixels: p.as_flattened().to_vec(),
        width,
        height,
    };
    let (r, d) = (rgb8(r), rgb8(d));
    metrics::vmaf::check_dims(&r, &d).map_err(fail)?;
    let (r, d) = (
        metrics::vmaf::to_yuv420(&r).map_err(fail)?,
        metrics::vmaf::to_yuv420(&d).map_err(fail)?,
    );
    let r = yuv420_image(&r, width, height).map_err(|e| JsError::new(&e))?;
    let d = yuv420_image(&d, width, height).map_err(|e| JsError::new(&e))?;
    evaluate_native_map(
        &mut vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p),
        r,
        d,
        out,
    )
}

/// 8-bit 4:2:0 planes (Y, Cb, Cr) as a planar metric input.
#[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
fn yuv420_image(
    (y, u, v): &(Vec<u8>, Vec<u8>, Vec<u8>),
    width: u32,
    height: u32,
) -> Result<zenmetrics_abstractions::Image<'_>, String> {
    use za::zenpixels::{ChannelType, PixelDescriptor, PixelSlice, PlaneLayout};
    use zenmetrics_abstractions as za;
    let plane = |p, w: u32, h: u32| {
        PixelSlice::new(p, w, h, w as usize, PixelDescriptor::GRAY8).map_err(|e| e.to_string())
    };
    za::Planes::new(
        PlaneLayout::ycbcr_420(ChannelType::U8),
        vec![
            plane(y, width, height)?,
            plane(u, width / 2, height / 2)?,
            plane(v, width / 2, height / 2)?,
        ],
    )
    .map(za::Image::from)
    .map_err(|e| e.to_string())
}

/// Spatial exports compiled in this browser build. No alternate metric is substituted.
#[wasm_bindgen]
pub fn available_spatial_maps() -> String {
    let mut names = vec!["zensim", "butteraugli", "dssim", "gmsd", "psnr", "ssim2"];
    #[cfg(feature = "cpu-cvvdp")]
    names.push("cvvdp");
    #[cfg(feature = "cpu-fsim")]
    names.extend(["fsim", "fsimc"]);
    #[cfg(feature = "cpu-haarpsi")]
    names.push("haarpsi");
    #[cfg(feature = "cpu-vsi")]
    names.push("vsi");
    #[cfg(feature = "cpu-msssim")]
    names.push("msssim");
    #[cfg(feature = "cpu-metrics")]
    names.push("nlpd");
    #[cfg(feature = "cpu-psnrhvs")]
    names.push("psnrhvs");
    #[cfg(feature = "cpu-mdctpsnr")]
    names.push("mdctpsnr");
    #[cfg(feature = "cpu-mad")]
    names.push("mad");
    #[cfg(feature = "cpu-iwssim")]
    names.push("iwssim");
    #[cfg(feature = "cpu-vif")]
    names.push("vif");
    #[cfg(feature = "cpu-vifvec")]
    names.push("vifvec");
    #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
    names.push("vmaf-v1");
    names.sort_unstable();
    serde_json::json!(names).to_string()
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
    let _ = display; // Also valid when the CVVDP feature is disabled.
    let r = rgb(reference, width, height, stride).map_err(|e| JsError::new(&e))?;
    let d = rgb(distorted, width, height, stride).map_err(|e| JsError::new(&e))?;
    let (w, h) = (width as usize, height as usize);
    let fail = |e: String| JsError::new(&e);
    let mut result = SpatialMap {
        width,
        height,
        similarity: false,
        geometry: None,
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
            // DSSIM's saved maps hold per-pixel SSIM: higher = more similar.
            result.similarity = true;
            result.values = first.map.buf().to_vec();
        }
        #[cfg(feature = "cpu-fsim")]
        "fsim" | "fsimc" => {
            evaluate_spatial_map(&mut fsim::Fsim, &r, &d, width, height, &mut result)?
        }
        #[cfg(feature = "cpu-haarpsi")]
        "haarpsi" => {
            evaluate_spatial_map(&mut haarpsi::Haarpsi, &r, &d, width, height, &mut result)?
        }
        #[cfg(feature = "cpu-vsi")]
        "vsi" => evaluate_spatial_map(&mut vsi::Vsi, &r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-msssim")]
        "msssim" => evaluate_spatial_map(&mut msssim::Msssim, &r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-metrics")]
        "nlpd" => evaluate_spatial_map(&mut nlpd::Nlpd, &r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-psnrhvs")]
        "psnrhvs" => {
            evaluate_spatial_map(&mut psnrhvs::Psnrhvs, &r, &d, width, height, &mut result)?
        }
        #[cfg(feature = "cpu-mdctpsnr")]
        "mdctpsnr" => {
            evaluate_spatial_map(&mut mdctpsnr::MdctPsnr, &r, &d, width, height, &mut result)?
        }
        #[cfg(feature = "cpu-mad")]
        "mad" => evaluate_spatial_map(&mut mad_iqa::Mad, &r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-iwssim")]
        "iwssim" => {
            let mut metric = iwssim::Iwssim::new(width, height).map_err(|e| fail(e.to_string()))?;
            evaluate_spatial_map(&mut metric, &r, &d, width, height, &mut result)?
        }
        #[cfg(feature = "cpu-vif")]
        "vif" => evaluate_spatial_map(&mut vif::Vif, &r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-vifvec")]
        "vifvec" => evaluate_spatial_map(&mut vif::Vifvec, &r, &d, width, height, &mut result)?,
        #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
        "vmaf-v1" => vmaf_v1_map(&r, &d, width, height, &mut result)?,
        #[cfg(feature = "cpu-cvvdp")]
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
        "gmsd" => evaluate_spatial_map(&mut gmsd::Gmsd, &r, &d, width, height, &mut result)?,
        "ssim2" => evaluate_spatial_map(
            &mut ssim2_map::Ssimulacra2,
            &r,
            &d,
            width,
            height,
            &mut result,
        )?,
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
        || result
            .values
            .iter()
            .any(|v| v.is_infinite() || (v.is_nan() && metric != "haarpsi"))
    {
        return Err(fail(
            "Native spatial map has invalid dimensions or non-finite samples".into(),
        ));
    }
    Ok(result)
}
