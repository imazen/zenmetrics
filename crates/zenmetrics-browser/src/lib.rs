#![forbid(unsafe_code)]
//! Opaque sRGB8 rows, a cached reference and raw spatial error maps.
use wasm_bindgen::prelude::*;

// These are the native CLI adapters themselves, so browser variants preserve
// each metric's ingress and implementation identity.
#[path = "../../zenmetrics-cli/src/decode.rs"]
#[allow(dead_code, unreachable_pub, unexpected_cfgs)]
mod decode;
mod loop_map;
// Learned-metric input tensors, built with zenmetrics-nn's own conversion
// and resize sources. Those files address `crate::par`, `crate::za` and
// `crate::engine::workers`, so they sit at this crate's root.
#[cfg(feature = "nn-prep")]
mod nn_prep;
#[cfg(feature = "nn-prep")]
pub use nn_prep::{NnPrepared, nn_prepare, nn_wd_layer, prepare as nn_prepare_native, wd_layer};
#[cfg(feature = "nn-prep")]
#[path = "../../zenmetrics-nn/src/par.rs"]
#[allow(dead_code, unreachable_pub)]
mod par;
#[cfg(feature = "nn-prep")]
#[path = "../../zenmetrics-nn/src/preprocess.rs"]
#[allow(dead_code, unreachable_pub)]
mod preprocess;
#[cfg(feature = "nn-prep")]
#[path = "../../zenmetrics-nn/src/resize.rs"]
#[allow(dead_code, unreachable_pub)]
mod resize;
#[cfg(feature = "nn-prep")]
#[path = "../../zenmetrics-nn/src/tfresize.rs"]
#[allow(dead_code, unreachable_pub)]
mod tfresize;
#[cfg(feature = "nn-prep")]
use zenmetrics_abstractions as za;
/// zenmetrics-nn's worker cap for its scoped-thread helpers. The browser
/// module has no `std::thread`, so its preprocessing runs on one thread;
/// zenmetrics-nn states each element's value is independent of the split.
#[cfg(feature = "nn-prep")]
mod engine {
    pub(crate) fn workers() -> usize {
        1
    }
}
#[path = "../../zenmetrics-cli/src/metrics/mod.rs"]
#[allow(dead_code, unreachable_pub, unexpected_cfgs)]
mod metrics;
#[cfg(feature = "cpu-metrics")]
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
        feature = "cpu-iwssim-piq",
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
        let mut iw_piq =
            iwssim::Iwssim::with_params(w, h, iwssim::IwssimParams::piq_luma()).unwrap();
        let routes: [(&str, &mut dyn za::Metric); 8] = [
            ("nlpd", &mut nlpd::Nlpd),
            ("psnrhvs", &mut psnrhvs::Psnrhvs),
            ("mdctpsnr", &mut mdctpsnr::MdctPsnr),
            ("mad", &mut mad_iqa::Mad),
            ("iwssim", &mut iw),
            ("iwssim-piq", &mut iw_piq),
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
            let got = super::spatial_map(name, "", &ra, &rb, w, h, stride, "").unwrap();
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
        let got = super::spatial_map("vmaf-v1", "", &ra, &rb, w, h, stride, "").unwrap();
        assert_eq!(got.geometry, Some(geometry));
        assert_eq!((geometry.cells_x, geometry.cells_y), (3, 2));
        assert!(!similarity && !got.similarity);
        assert_eq!(got.values, values);
        assert!(got.values.iter().any(|&v| v > 0.0));
    }

    /// The weights the fused DSSIM map uses are DSSIM's defaults (same score
    /// bits), and the fused map's mean is `1 − SSIM` with `DSSIM = 1/SSIM − 1`.
    #[cfg(feature = "cpu-metrics")]
    #[test]
    fn dssim_weights_match_the_default_score() {
        let (w, h) = (203u32, 117u32);
        let a = zenmetrics_abstractions::conformance::textured_rgb8(w as usize, h as usize, 11);
        let b = zenmetrics_abstractions::conformance::textured_rgb8(w as usize, h as usize, 12);
        let score = |weights: Option<&[f64]>| {
            let mut s = dssim_core::Dssim::new();
            if let Some(weights) = weights {
                s.set_scales(weights);
            }
            let r = s
                .create_image_rgb(bytemuck::cast_slice(&a), w as usize, h as usize)
                .unwrap();
            let d = s
                .create_image_rgb(bytemuck::cast_slice(&b), w as usize, h as usize)
                .unwrap();
            f64::from(s.compare(&r, d).0)
        };
        let dssim = score(None);
        assert_eq!(
            dssim.to_bits(),
            score(Some(&super::DSSIM_SCALE_WEIGHTS)).to_bits()
        );
        let (ra, stride) = rgba(&a, w, 0);
        let (rb, _) = rgba(&b, w, 0);
        let fused = super::spatial_map("dssim", "", &ra, &rb, w, h, stride, "").unwrap();
        assert_eq!(
            (fused.name.as_str(), fused.polarity),
            ("diffmap", "difference")
        );
        assert_eq!((fused.width, fused.height), (w, h));
        let mean = fused.values.iter().map(|&v| f64::from(v)).sum::<f64>() / f64::from(w * h);
        let from_map = 1.0 / (1.0 - mean) - 1.0;
        assert!(
            (from_map - dssim).abs() <= 1e-6 * dssim.max(1e-3),
            "{from_map} vs {dssim}"
        );
        let names: Vec<String> = serde_json::from_str(&fused.names()).unwrap();
        assert_eq!(
            names,
            ["diffmap", "ssim_0", "ssim_1", "ssim_2", "ssim_3", "ssim_4"]
        );
        let s0 = super::cached_spatial_map("ssim_0").unwrap();
        assert_eq!((s0.width, s0.height, s0.polarity), (w, h, "similarity"));
        assert!(s0.similarity);
    }

    /// Every map an evaluation publishes is selectable by name, by request
    /// or from the cache, and carries the metric's own values.
    #[cfg(all(
        feature = "cpu-metrics",
        feature = "cpu-msssim",
        feature = "cpu-iwssim"
    ))]
    #[test]
    fn every_published_map_is_selectable() {
        use zenmetrics_abstractions::{self as za, zenpixels};
        let (w, h) = (224u32, 180u32);
        let (a, b) = (
            za::conformance::textured_rgb8(w as usize, h as usize, 5),
            za::conformance::textured_rgb8(w as usize, h as usize, 6),
        );
        let ((ra, stride), (rb, _)) = (rgba(&a, w, 4), rgba(&b, w, 4));
        let image = |px| {
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
        let routes: [(&str, &mut dyn za::Metric); 3] = [
            ("msssim", &mut msssim::Msssim),
            ("iwssim", &mut iw),
            ("nlpd", &mut nlpd::Nlpd),
        ];
        for (route, metric) in routes {
            let e = metric
                .evaluate(
                    image(&a),
                    image(&b),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            let direct = e.maps();
            assert!(direct.len() > 1, "{route} publishes one map");
            let first = super::spatial_map(route, "", &ra, &rb, w, h, stride, "").unwrap();
            let names: Vec<String> = serde_json::from_str(&first.names()).unwrap();
            assert_eq!(
                names,
                direct.iter().map(|m| m.name).collect::<Vec<_>>(),
                "{route}"
            );
            for m in &direct {
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                let cached = super::cached_spatial_map(m.name).unwrap();
                let fresh = super::spatial_map(route, m.name, &ra, &rb, w, h, stride, "").unwrap();
                for got in [&cached, &fresh] {
                    assert_eq!(got.name, m.name);
                    assert_eq!(got.geometry, Some(m.geometry), "{route}/{}", m.name);
                    assert_eq!(bits(&got.values), bits(&m.values()), "{route}/{}", m.name);
                }
            }
        }
    }

    /// Each metric's first fitted loop default is the map `spatial_map`
    /// serves by default, which the viewer relies on, and every fitted map
    /// still exists and reads as error.
    #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf", feature = "cpu-iwssim"))]
    #[test]
    fn loop_defaults_name_existing_maps_default_first() {
        let (w, h) = (240u32, 224u32);
        let a = zenmetrics_abstractions::conformance::textured_rgb8(w as usize, h as usize, 21);
        let b = zenmetrics_abstractions::conformance::textured_rgb8(w as usize, h as usize, 22);
        let ((ra, stride), (rb, _)) = (rgba(&a, w, 0), rgba(&b, w, 0));
        let mut seen = Vec::new();
        for n in super::loop_map::DEFAULTS {
            assert!(
                n.scale > 0.0 && n.exponent > 0.0 && n.r.is_finite(),
                "{n:?}"
            );
            let e = super::loop_error_map(n.metric, n.map, &ra, &rb, w, h, stride, "standard_4k")
                .unwrap_or_else(|e| panic!("{}/{}: {e}", n.metric, n.map));
            if !seen.contains(&n.metric) {
                seen.push(n.metric);
                let first = super::spatial_map(n.metric, "", &ra, &rb, w, h, stride, "standard_4k")
                    .unwrap();
                assert_eq!(first.name, n.map, "{}", n.metric);
            }
            assert_eq!(e.len(), (w * h) as usize);
        }
        assert!(seen.len() >= 19, "{seen:?}");
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
            let got = super::spatial_map(name, "", &px, &px, w, h, stride, "").unwrap();
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

    /// The `iwssim-piq` map route evaluates the metric the CLI scores: its
    /// evaluation's score equals `score_json("iwssim-piq")` exactly.
    #[cfg(all(feature = "cpu-metrics", feature = "cpu-iwssim-piq"))]
    #[test]
    fn iwssim_piq_map_route_scores_the_cli_metric() {
        use zenmetrics_abstractions::{self as za, Metric as _, zenpixels};
        let (w, h) = (224u32, 180u32);
        let (a, b) = (
            za::conformance::textured_rgb8(w as usize, h as usize, 5),
            za::conformance::textured_rgb8(w as usize, h as usize, 6),
        );
        let ((ra, stride), (rb, _)) = (rgba(&a, w, 0), rgba(&b, w, 0));
        let cli: serde_json::Value = serde_json::from_str(
            &super::score_json("iwssim-piq", &ra, &rb, w, h, stride, "").unwrap(),
        )
        .unwrap();
        let image = |px| {
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
        let mut metric =
            iwssim::Iwssim::with_params(w, h, iwssim::IwssimParams::piq_luma()).unwrap();
        let e = metric
            .evaluate(
                image(&a),
                image(&b),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        assert_eq!(cli["values"].as_array().unwrap().len(), 1);
        assert_eq!(
            e.score().to_bits(),
            cli["values"][0]["value"].as_f64().unwrap().to_bits()
        );
    }
}

#[cfg(test)]
mod listed_metrics_score {
    /// Every metric `available_metrics` lists must be scoreable by this
    /// build: a listed metric whose scoring arm is compiled out only answers
    /// "disabled". Size refusals on this small pair are allowed.
    #[test]
    fn listed_metrics_are_compiled_in() {
        let (w, h) = (64u32, 64u32);
        let r: Vec<u8> = (0..w * h * 3)
            .map(|i| ((i * 13 + i / 192 * 17) % 256) as u8)
            .collect();
        let d: Vec<u8> = r.iter().map(|v| v.saturating_sub(7)).collect();
        let img = |px: &[u8]| super::decode::Rgb8Image {
            pixels: px.to_vec(),
            width: w,
            height: h,
        };
        let (ri, di) = (img(&r), img(&d));
        let names: Vec<String> = serde_json::from_str(&super::available_metrics()).unwrap();
        let mut disabled = Vec::new();
        for name in &names {
            let kind =
                <super::metrics::MetricKind as clap::ValueEnum>::from_str(name, false).unwrap();
            let screen = kind
                .needs_display()
                .then(|| super::metrics::display::parse_display("standard_4k").unwrap());
            if let Err(e) = super::metrics::run_metric_display(
                kind,
                &ri,
                &di,
                super::metrics::GpuRuntime::Cpu,
                screen.as_ref(),
            ) && e.to_string().contains("is disabled")
            {
                disabled.push(format!("{name}: {e}"));
            }
        }
        assert!(
            disabled.is_empty(),
            "listed but not compiled in: {disabled:#?}"
        );
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
                // The core module scores Zensim through `Reference` only;
                // its CLI dispatch has every other metric compiled out.
                (cfg!(feature = "cpu-metrics") || **m == metrics::MetricKind::Zensim)
                    && !m.requires_gpu()
                    && m.scoring_compiled_in()
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

/// Every metric name of the native CLI catalog (`MetricKind`), in its
/// order, whether or not this module scores it.
#[wasm_bindgen]
pub fn all_metric_names() -> String {
    use clap::ValueEnum;
    serde_json::to_string(
        &metrics::MetricKind::value_variants()
            .iter()
            .map(|m| m.name())
            .collect::<Vec<_>>(),
    )
    .expect("static metric names serialize")
}

/// Which build this is: `"core"` (first load: decoders, JPEG encoder and
/// Zensim) or `"full"` (every CPU metric).
#[wasm_bindgen]
pub fn module_name() -> String {
    if cfg!(feature = "cpu-metrics") {
        "full"
    } else {
        "core"
    }
    .into()
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
    score_json(metric, reference, distorted, width, height, stride, display)
        .map_err(|e| JsError::new(&e))
}

/// [`score_rgba`] with a plain error, callable natively (the browser oracle).
pub fn score_json(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<String, String> {
    let kind = <metrics::MetricKind as clap::ValueEnum>::from_str(metric, false)?;
    if kind.requires_gpu()
        || matches!(
            kind,
            metrics::MetricKind::Hdrvdp | metrics::MetricKind::Hdrvdp3
        )
    {
        return Err(
            "Browser image pairs use SDR CPU metrics; GPU and absolute-nits HDR requests are unavailable"
                .into(),
        );
    }
    let image = |bytes: &[u8]| -> Result<decode::Rgb8Image, String> {
        let pixels = rgb(bytes, width, height, stride)?;
        Ok(decode::Rgb8Image {
            pixels: pixels.into_iter().flatten().collect(),
            width,
            height,
        })
    };
    let r = image(reference)?;
    let d = image(distorted)?;
    let screen = if kind.needs_display() {
        Some(metrics::display::parse_display(display)?)
    } else {
        None
    };
    let values =
        metrics::run_metric_display(kind, &r, &d, metrics::GpuRuntime::Cpu, screen.as_ref())
            .map_err(|e| e.to_string())?;
    let values: Vec<_> = values.into_iter().map(|(column,value)| serde_json::json!({"column":column,
        "value":if value.is_finite() {serde_json::json!(value)} else {serde_json::json!(value.to_string())}})).collect();
    Ok(
        serde_json::json!({"metric":metric,"backend":"native-cpu-wasm","values":values})
            .to_string(),
    )
}

/// The decoder for this format is in the other module.
#[allow(dead_code)]
const NOT_IN_MODULE: &str = "decoder not in this module";

/// Decode SDR file bytes with Zen codecs, preserving PNG/WebP/JXL alpha.
/// Returns dimensions followed by little-endian packed RGBA8 pixels.
/// ICC profiles are not applied; scoring still requires explicit opaque input.
#[wasm_bindgen]
pub fn decode_image(bytes: &[u8], filename: &str) -> Result<Vec<u32>, JsError> {
    #[allow(dead_code)]
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
    let png = bytes.starts_with(b"\x89PNG\r\n\x1a\n");
    let webp = bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP");
    let jxl = bytes.starts_with(&[0xff, 0x0a]) || bytes.starts_with(b"\0\0\0\x0cJXL ");
    let decoded: Result<(Vec<u8>, u32, u32), String> = if png {
        #[cfg(feature = "png")]
        {
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
        }
        #[cfg(not(feature = "png"))]
        Err(NOT_IN_MODULE.into())
    } else if webp {
        #[cfg(feature = "webp")]
        {
            zenwebp::decoder::decode_rgba(bytes).map_err(|e| e.to_string())
        }
        #[cfg(not(feature = "webp"))]
        Err(NOT_IN_MODULE.into())
    } else if jxl {
        #[cfg(feature = "jxl")]
        {
            zenjxl::decode(bytes, None, &[])
                .map_err(|e| e.to_string())
                .and_then(|output| rgba(&output.pixels))
        }
        #[cfg(not(feature = "jxl"))]
        Err(NOT_IN_MODULE.into())
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
#[derive(Clone)]
pub struct SpatialMap {
    width: u32,
    height: u32,
    similarity: bool,
    geometry: Option<zenmetrics_abstractions::MapGeometry>,
    values: Vec<f32>,
    /// The metric's own name for this map (`"diffmap"`, `"ssim_0"`, …).
    name: String,
    /// `"similarity"`, `"difference"`, `"probability"` or `"feature"`.
    polarity: &'static str,
    /// Every map the same evaluation published, in the metric's order.
    names: Vec<String>,
}

thread_local! {
    /// Every map of the last [`spatial_map`] evaluation, so choosing another
    /// map of the same pair does not evaluate the metric again.
    static LAST_MAPS: RefCell<Vec<SpatialMap>> = const { RefCell::new(Vec::new()) };
}

impl SpatialMap {
    fn empty(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            similarity: false,
            geometry: None,
            values: Vec::new(),
            name: String::new(),
            polarity: "difference",
            names: Vec::new(),
        }
    }
    /// A full-resolution difference map named `name`.
    fn difference(name: &str, width: u32, height: u32, values: Vec<f32>) -> Self {
        Self {
            values,
            name: name.into(),
            ..Self::empty(width, height)
        }
    }
}

/// Pick `requested` (or, for an empty request, the metric's fused `diffmap`
/// when it publishes one, else its first map) from one evaluation's maps and
/// keep them all for [`cached_spatial_map`].
fn select_map(mut maps: Vec<SpatialMap>, requested: &str) -> Result<SpatialMap, JsError> {
    let names: Vec<String> = maps.iter().map(|m| m.name.clone()).collect();
    for m in &mut maps {
        m.names = names.clone();
    }
    let index = if requested.is_empty() {
        maps.iter().position(|m| m.name == "diffmap").unwrap_or(0)
    } else {
        maps.iter()
            .position(|m| m.name == requested)
            .ok_or_else(|| {
                JsError::new(&format!(
                    "this metric publishes no map named {requested} (it has {})",
                    names.join(", ")
                ))
            })?
    };
    let chosen = maps
        .get(index)
        .cloned()
        .ok_or_else(|| JsError::new("Native evaluation returned no spatial map"))?;
    LAST_MAPS.with(|slot| *slot.borrow_mut() = maps);
    check_map(chosen)
}

/// Another map of the last [`spatial_map`] evaluation, without evaluating
/// again. Fails when that evaluation published no map named `name`.
#[wasm_bindgen]
pub fn cached_spatial_map(name: &str) -> Result<SpatialMap, JsError> {
    LAST_MAPS
        .with(|slot| {
            slot.borrow()
                .iter()
                .find(|m| m.name == name)
                .cloned()
                .ok_or_else(|| JsError::new(&format!("no cached map named {name}")))
        })
        .and_then(check_map)
}

#[wasm_bindgen]
impl SpatialMap {
    /// The metric's own name for this map.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    /// Every map name the evaluation published, as a JSON array.
    #[wasm_bindgen(getter)]
    pub fn names(&self) -> String {
        serde_json::json!(self.names).to_string()
    }
    /// `"similarity"`, `"difference"`, `"probability"` or `"feature"`.
    #[wasm_bindgen(getter)]
    pub fn polarity(&self) -> String {
        self.polarity.into()
    }
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

/// Every map an evaluation published, primary first.
// The core module has no unified-interface metric.
#[cfg_attr(not(feature = "cpu-metrics"), allow(dead_code))]
fn evaluation_maps(e: &dyn zenmetrics_abstractions::Evaluation) -> Vec<SpatialMap> {
    use zenmetrics_abstractions::MapPolarity;
    e.maps()
        .into_iter()
        .map(|m| SpatialMap {
            width: m.width(),
            height: m.height(),
            similarity: m.polarity == MapPolarity::Similarity,
            geometry: Some(m.geometry),
            values: m.values().into_owned(),
            name: m.name.into(),
            polarity: match m.polarity {
                MapPolarity::Similarity => "similarity",
                MapPolarity::Difference => "difference",
                MapPolarity::DetectionProbability => "probability",
                _ => "feature",
            },
            names: Vec::new(),
        })
        .collect()
}

// The core module has no unified-interface metric.
#[cfg_attr(not(feature = "cpu-metrics"), allow(dead_code))]
fn evaluate_spatial_map(
    metric: &mut dyn zenmetrics_abstractions::Metric,
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
) -> Result<Vec<SpatialMap>, String> {
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
        .map_err(|e| e.to_string())
    };
    evaluate_native_map(
        metric,
        image(bytemuck::cast_slice(r))?,
        image(bytemuck::cast_slice(d))?,
    )
}

/// Every map the metric publishes for an already shaped input pair.
// The core module has no unified-interface metric.
#[cfg_attr(not(feature = "cpu-metrics"), allow(dead_code))]
fn evaluate_native_map(
    metric: &mut dyn zenmetrics_abstractions::Metric,
    r: zenmetrics_abstractions::Image<'_>,
    d: zenmetrics_abstractions::Image<'_>,
) -> Result<Vec<SpatialMap>, String> {
    use zenmetrics_abstractions as za;
    let pulse = how_far::StopOnly::new(control());
    let opts = za::EvalOpts::SCORE_ONLY
        .with_native_map()
        .with_pulse(&pulse);
    let evaluation = metric.evaluate(r, d, opts).map_err(|e| e.to_string())?;
    opts.checkpoint().map_err(|e| e.to_string())?;
    Ok(evaluation_maps(evaluation.as_ref()))
}

/// VMAF v1.0.16 reads 8-bit BT.709 limited-range 4:2:0, converted exactly as
/// the `vmaf-v1` score converts it.
#[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
fn vmaf_v1_map(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
) -> Result<Vec<SpatialMap>, String> {
    let fail = |e: Box<dyn std::error::Error>| e.to_string();
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
    let r = yuv420_image(&r, width, height)?;
    let d = yuv420_image(&d, width, height)?;
    evaluate_native_map(
        &mut vmaf::Vmaf::new(vmaf::ModelVariant::Standard1080p),
        r,
        d,
    )
}

/// 8-bit 4:2:0 planes (Y, Cb, Cr) as a planar metric input.
#[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
fn yuv420_image(
    (y, u, v): &(Vec<u8>, Vec<u8>, Vec<u8>),
    width: u32,
    height: u32,
) -> Result<zenmetrics_abstractions::Image<'_>, String> {
    use za::PlaneLayout;
    use za::zenpixels::{ChannelType, PixelDescriptor, PixelSlice};
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
    let mut names = vec!["zensim"];
    #[cfg(feature = "cpu-metrics")]
    names.extend(["butteraugli", "dssim", "psnr", "ssim2"]);
    #[cfg(feature = "cpu-gmsd")]
    names.push("gmsd");
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
    #[cfg(feature = "cpu-iwssim-piq")]
    names.push("iwssim-piq");
    #[cfg(feature = "cpu-vif")]
    names.push("vif");
    #[cfg(feature = "cpu-vifvec")]
    names.push("vifvec");
    #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
    names.push("vmaf-v1");
    names.sort_unstable();
    serde_json::json!(names).to_string()
}

/// Compute a metric's native spatial output: the map named `map`, or for an
/// empty `map` the metric's fused `diffmap` when it publishes one, else its
/// primary map. Every map of the evaluation stays available through
/// [`cached_spatial_map`] until the next call. Unsupported requests fail
/// explicitly; a different metric is never substituted.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn spatial_map(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<SpatialMap, JsError> {
    LAST_MAPS.with(|slot| slot.borrow_mut().clear());
    let maps = compute_maps(metric, reference, distorted, width, height, stride, display)
        .map_err(|e| JsError::new(&e))?;
    LAST_NAN_OK.with(|slot| slot.set(metric == "haarpsi"));
    select_map(maps, map)
}

/// Every map `metric` publishes for one pair, primary first.
fn compute_maps(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<Vec<SpatialMap>, String> {
    let _ = display; // Also valid when the CVVDP feature is disabled.
    let r = rgb(reference, width, height, stride)?;
    let d = rgb(distorted, width, height, stride)?;
    let fail = |e: String| e;
    let maps: Vec<SpatialMap> = match metric {
        "zensim" => vec![SpatialMap::difference(
            "diffmap",
            width,
            height,
            zensim_diffmap(&r, &d, width, height)?,
        )],
        #[cfg(feature = "cpu-metrics")]
        "butteraugli" => {
            let (w, h) = (width as usize, height as usize);
            let r: &[rgb::RGB<u8>] = bytemuck::cast_slice(&r);
            let d: &[rgb::RGB<u8>] = bytemuck::cast_slice(&d);
            let scored = butteraugli::butteraugli(
                butteraugli::ImgRef::new(r, w, h),
                butteraugli::ImgRef::new(d, w, h),
                &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
            )
            .map_err(|e| fail(e.to_string()))?;
            let values = scored
                .diffmap
                .ok_or_else(|| fail("Butteraugli returned no spatial map".into()))?
                .buf()
                .to_vec();
            vec![SpatialMap::difference("diffmap", width, height, values)]
        }
        #[cfg(feature = "cpu-metrics")]
        "dssim" => dssim_maps(&r, &d, width, height)?,
        #[cfg(feature = "cpu-fsim")]
        "fsim" | "fsimc" => {
            let mut scorer = fsim::Fsim::default();
            evaluate_spatial_map(&mut scorer, &r, &d, width, height)?
        }
        #[cfg(feature = "cpu-haarpsi")]
        "haarpsi" => evaluate_spatial_map(&mut haarpsi::Haarpsi, &r, &d, width, height)?,
        #[cfg(feature = "cpu-vsi")]
        "vsi" => {
            let mut scorer = vsi::Vsi::default();
            evaluate_spatial_map(&mut scorer, &r, &d, width, height)?
        }
        #[cfg(feature = "cpu-msssim")]
        "msssim" => evaluate_spatial_map(&mut msssim::Msssim, &r, &d, width, height)?,
        #[cfg(feature = "cpu-metrics")]
        "nlpd" => evaluate_spatial_map(&mut nlpd::Nlpd, &r, &d, width, height)?,
        #[cfg(feature = "cpu-psnrhvs")]
        "psnrhvs" => evaluate_spatial_map(&mut psnrhvs::Psnrhvs, &r, &d, width, height)?,
        #[cfg(feature = "cpu-mdctpsnr")]
        "mdctpsnr" => evaluate_spatial_map(&mut mdctpsnr::MdctPsnr, &r, &d, width, height)?,
        #[cfg(feature = "cpu-mad")]
        "mad" => evaluate_spatial_map(&mut mad_iqa::Mad, &r, &d, width, height)?,
        #[cfg(feature = "cpu-iwssim")]
        "iwssim" => {
            let mut metric = iwssim::Iwssim::new(width, height).map_err(|e| fail(e.to_string()))?;
            evaluate_spatial_map(&mut metric, &r, &d, width, height)?
        }
        // The CLI's `iwssim-piq` score: the same metric with piq's
        // unrounded-YIQ luma (`run_cpu_iwssim_piq`).
        #[cfg(feature = "cpu-iwssim-piq")]
        "iwssim-piq" => {
            let mut metric =
                iwssim::Iwssim::with_params(width, height, iwssim::IwssimParams::piq_luma())
                    .map_err(|e| fail(e.to_string()))?;
            evaluate_spatial_map(&mut metric, &r, &d, width, height)?
        }
        #[cfg(feature = "cpu-vif")]
        "vif" => evaluate_spatial_map(&mut vif::Vif, &r, &d, width, height)?,
        #[cfg(feature = "cpu-vifvec")]
        "vifvec" => evaluate_spatial_map(&mut vif::Vifvec, &r, &d, width, height)?,
        #[cfg(all(feature = "cpu-metrics", feature = "cpu-vmaf"))]
        "vmaf-v1" => vmaf_v1_map(&r, &d, width, height)?,
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
            let mut values = Vec::new();
            scorer
                .score_with_diffmap(
                    bytemuck::cast_slice(&r),
                    bytemuck::cast_slice(&d),
                    &mut values,
                )
                .map_err(|e| fail(e.to_string()))?;
            vec![SpatialMap::difference("diffmap", width, height, values)]
        }
        #[cfg(feature = "cpu-gmsd")]
        "gmsd" => evaluate_spatial_map(&mut gmsd::Gmsd, &r, &d, width, height)?,
        #[cfg(feature = "cpu-metrics")]
        "ssim2" => evaluate_spatial_map(&mut ssim2_map::Ssimulacra2, &r, &d, width, height)?,
        #[cfg(feature = "cpu-metrics")]
        "psnr" => {
            // Per-pixel mean squared sRGB code-value error, in normalized units.
            let values = r
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
            vec![SpatialMap::difference(
                "squared_error",
                width,
                height,
                values,
            )]
        }
        _ => {
            return Err(fail(format!(
                "{metric} has no native spatial-map export in this build"
            )));
        }
    };
    Ok(maps)
}

thread_local! {
    /// HaarPSI leaves local similarity undefined (NaN) in zero-weight windows.
    static LAST_NAN_OK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn check_map(m: SpatialMap) -> Result<SpatialMap, JsError> {
    let nan_ok = LAST_NAN_OK.with(std::cell::Cell::get);
    if m.values.len() != m.width as usize * m.height as usize
        || m.values
            .iter()
            .any(|v| v.is_infinite() || (v.is_nan() && !nan_ok))
    {
        return Err(JsError::new(&format!(
            "Native spatial map {} has invalid dimensions or non-finite samples",
            m.name
        )));
    }
    Ok(m)
}

thread_local! {
    /// The last Zensim reference prepared for a map, kept while a loop scores
    /// many candidates against one source.
    static ZENSIM_REFERENCE: RefCell<Option<(Vec<[u8; 3]>, u32, u32, PrecomputedReference)>> =
        const { RefCell::new(None) };
}

/// Zensim's (model B) diffmap, the same values as [`Reference::compare`].
/// The cached reference is taken out of its slot while comparing: rayon
/// work-stealing can run another call on this thread meanwhile, which then
/// prepares its own.
fn zensim_diffmap(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, String> {
    let (w, h) = (width as usize, height as usize);
    let scorer = Zensim::new(ZensimProfile::B).with_stop(control());
    let cached = ZENSIM_REFERENCE
        .with(|slot| slot.borrow_mut().take())
        .filter(|(pixels, sw, sh, _)| (*sw, *sh) == (width, height) && pixels == r);
    let (pixels, prepared) = match cached {
        Some((pixels, _, _, prepared)) => (pixels, prepared),
        None => (
            r.to_vec(),
            scorer
                .precompute_reference(&RgbSlice::new(r, w, h))
                .map_err(|e| e.to_string())?,
        ),
    };
    let result = scorer
        .compute_with_ref_and_diffmap(
            &prepared,
            &RgbSlice::new(d, w, h),
            DiffmapWeighting::default(),
        )
        .map(|result| result.diffmap().to_vec())
        .map_err(|e| e.to_string());
    ZENSIM_REFERENCE.with(|slot| *slot.borrow_mut() = Some((pixels, width, height, prepared)));
    result
}

/// DSSIM's own default scale weights (`dssim-core` `DEFAULT_WEIGHTS`, which
/// it does not export). Set explicitly so the fused map uses the weights the
/// score uses; `dssim_weights_match_the_default_score` pins them.
#[cfg(feature = "cpu-metrics")]
const DSSIM_SCALE_WEIGHTS: [f64; 5] = [0.028, 0.197, 0.322, 0.298, 0.155];

/// DSSIM's per-scale SSIM maps (`ssim_0` full resolution … `ssim_4`) and a
/// fused `diffmap` whose mean is `1 − SSIM`, the combined similarity DSSIM
/// transforms (`DSSIM = 1/SSIM − 1`). Scale `n` scores
/// `1 − mean|avg_n − s|`, with `avg_n` its mean SSIM raised to `0.5ⁿ`; each
/// cell's `w_n/Σw · |avg_n − s|` is spread evenly over the full-resolution
/// pixels it covers, so the fused mean is the weighted sum exactly.
#[cfg(feature = "cpu-metrics")]
fn dssim_maps(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    width: u32,
    height: u32,
) -> Result<Vec<SpatialMap>, String> {
    let fail = |e: &str| e.to_string();
    let (w, h) = (width as usize, height as usize);
    let mut scorer = dssim_core::Dssim::new();
    scorer.set_scales(&DSSIM_SCALE_WEIGHTS);
    scorer.set_save_ssim_maps(DSSIM_SCALE_WEIGHTS.len() as u8);
    let r = scorer
        .create_image_rgb(bytemuck::cast_slice(r), w, h)
        .ok_or_else(|| fail("DSSIM rejected reference dimensions"))?;
    let d = scorer
        .create_image_rgb(bytemuck::cast_slice(d), w, h)
        .ok_or_else(|| fail("DSSIM rejected distorted dimensions"))?;
    let (_, scales) = scorer.compare(&r, d);
    if scales.is_empty() {
        return Err(fail("DSSIM returned no spatial map"));
    }
    let total: f64 = DSSIM_SCALE_WEIGHTS[..scales.len()].iter().sum();
    let mut fused = vec![0f64; w * h];
    for (n, scale) in scales.iter().enumerate() {
        let (sw, sh) = (scale.map.width(), scale.map.height());
        let cells = (sw * sh) as f64;
        let mean = scale.map.pixels().map(f64::from).sum::<f64>() / cells;
        let avg = mean.max(0.0).powf(0.5f64.powi(n as i32));
        let col = |x: usize| (x >> n).min(sw - 1);
        let row = |y: usize| (y >> n).min(sh - 1);
        // Full-resolution pixels per cell, so each cell's share is spread
        // over exactly the pixels it covers.
        let mut per_col = vec![0usize; sw];
        let mut per_row = vec![0usize; sh];
        (0..w).for_each(|x| per_col[col(x)] += 1);
        (0..h).for_each(|y| per_row[row(y)] += 1);
        let k = DSSIM_SCALE_WEIGHTS[n] / total * (w * h) as f64 / cells;
        for y in 0..h {
            let (cy, ry) = (row(y), per_row[row(y)] as f64);
            for x in 0..w {
                let cx = col(x);
                let s = f64::from(scale.map[(cx, cy)]);
                fused[y * w + x] += k * (avg - s).abs() / (per_col[cx] as f64 * ry);
            }
        }
    }
    const NAMES: [&str; 5] = ["ssim_0", "ssim_1", "ssim_2", "ssim_3", "ssim_4"];
    let mut maps = vec![SpatialMap::difference(
        "diffmap",
        width,
        height,
        fused.into_iter().map(|v| v as f32).collect(),
    )];
    maps.extend(
        scales
            .into_iter()
            .zip(NAMES)
            .map(|(scale, name)| SpatialMap {
                width: scale.map.width() as u32,
                height: scale.map.height() as u32,
                similarity: true,
                polarity: "similarity",
                values: scale.map.buf().to_vec(),
                ..SpatialMap::difference(name, width, height, Vec::new())
            }),
    );
    Ok(maps)
}

/// A metric map put on butteraugli's scale for jxl-encoder's loop: one
/// value per pixel and the map's maximum, which the loop reads as the score.
#[wasm_bindgen]
pub struct LoopMap {
    values: Vec<f32>,
    score: f64,
}

#[wasm_bindgen]
impl LoopMap {
    /// `width × height` normalized values, row-major.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
    /// The maximum value, the loop's score (butteraugli's score is its
    /// diffmap's maximum).
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.score
    }
}

/// The error reading of `metric`'s map `map` (empty: its default map) at
/// full resolution, before normalization. See [`loop_map`].
#[allow(clippy::too_many_arguments)]
pub fn loop_error_map(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<Vec<f32>, String> {
    let maps = compute_maps(metric, reference, distorted, width, height, stride, display)?;
    let chosen = if map.is_empty() {
        maps.iter()
            .find(|m| m.name == "diffmap")
            .or_else(|| maps.first())
    } else {
        maps.iter().find(|m| m.name == map)
    }
    .ok_or_else(|| format!("{metric} publishes no map named {map}"))?;
    loop_map::error_pixels(chosen, width, height)
}

/// jxl-encoder's per-block distances for a full-resolution map: each 8×8
/// block's 16-norm times libjxl's 1.2.
pub fn loop_tile_norms(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    loop_map::tile_norms(values, width, height)
}

/// The steering map for one loop iteration: `metric`'s map `map` read as
/// error, one value per pixel, then `scale · e^exponent`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen]
pub fn loop_map(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
    scale: f64,
    exponent: f64,
) -> Result<LoopMap, JsError> {
    if !(scale.is_finite() && scale > 0.0 && exponent.is_finite() && exponent > 0.0) {
        return Err(JsError::new(
            "scale and exponent must be finite and positive",
        ));
    }
    let mut values = loop_error_map(
        metric, map, reference, distorted, width, height, stride, display,
    )
    .map_err(|e| JsError::new(&e))?;
    loop_map::normalize(&mut values, scale, exponent);
    let score = values.iter().copied().fold(0f32, f32::max).into();
    Ok(LoopMap { values, score })
}

/// The fitted loop normalization defaults, as a JSON array of
/// `{metric, map, scale, exponent, r}`, each metric's default map first.
#[wasm_bindgen]
pub fn loop_map_defaults() -> String {
    serde_json::json!(
        loop_map::DEFAULTS
            .iter()
            .map(|n| serde_json::json!({"metric":n.metric,"map":n.map,"scale":n.scale,"exponent":n.exponent,"r":n.r}))
            .collect::<Vec<_>>()
    )
    .to_string()
}
