//! HyperIQA (S. Su, Q. Yan, Y. Zhu, C. Zhang, X. Ge, J. Sun, Y. Zhang,
//! "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive
//! Hyper Network", CVPR 2020; upstream `SSL92/hyperIQA` at `c42e7279`,
//! MIT code — **the released KonIQ checkpoint carries no stated licence**:
//! `WeightsLicence::Unknown`, consent-gated). Parity target: upstream
//! `demo.py` / test loader path on CPU fp32, `koniq_pretrained.pkl`.
//!
//! The eval transform is upstream's `Resize((512, 384))` (Pillow bilinear
//! u8 — [`resize_bilinear_pil_u8`], LICENCE in LICENSE-pillow) then
//! /255, ImageNet normalisation — **but** upstream's test protocol crops
//! `test_patch_num = 25` 224×224 patches *at random* (`RandomCrop`) and
//! averages: nondeterministic. This port uses a deterministic 5×5 crop
//! grid instead — x-origins `round(i·160/4)` ∈ {0,40,80,120,160},
//! y-origins `round(j·288/4)` ∈ {0,72,144,216,288} over the 384×512
//! (W×H) frame, same crop count. The deviation is documented in the
//! card; the model and weights are upstream's, byte for byte.
//!
//! `models/hyperiqa/export.py` loads upstream's `models.py` verbatim,
//! wraps `HyperNet` + `TargetNet` as one ONNX graph — the target net's
//! per-crop generated weights run as batched matmuls (upstream uses
//! `groups=batch` 1×1 convs, which cannot export a dynamic batch) — and
//! emits the batch mean as `hyperiqa [1]`. rten runs it through
//! [`engine::load_image`], ONNX Runtime engines through the named-input
//! whole-image graph.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, NamedInput};
use crate::preprocess::rgb8_to_planar_u8;
use crate::resize::resize_bilinear_pil_u8;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "hyperiqa";
/// Crop size (upstream `RandomCrop(224)`).
const CROP: usize = 224;
/// `Resize((512, 384))` output, height × width.
const RH: usize = 512;
const RW: usize = 384;
/// Deterministic 5×5 crop origins, evenly spaced (`round(i·(dim−224)/4)`):
/// upstream draws 25 *random* crops; our deviation is documented.
const XS: [usize; 5] = [0, 40, 80, 120, 160];
const YS: [usize; 5] = [0, 72, 144, 216, 288];
/// torchvision's ImageNet normalisation.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// HyperIQA, KonIQ checkpoint. Higher is better — upstream's 0–100
/// MOS-scale training target; the final layer is linear, so scores are
/// unbounded rather than clamped to [0,100]. Input: packed sRGB8 (any stride); every nonzero
/// size is accepted — the frame always resizes to 384×512.
pub struct Hyperiqa {
    backend: Backend,
    min_size: usize,
}

impl Hyperiqa {
    /// This model's weights' published terms: the README's
    /// `pretrained/koniq_pretrained.pkl` link (Google Drive / Baidu)
    /// states no licence; the code is MIT. Consent is required to load.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Unknown,
        terms_url: "https://github.com/SSL92/hyperIQA#testing-a-single-image",
    };

    /// Load an exported model directory (`hyperiqa.onnx`,
    /// `hyperiqa.meta`, as written by `models/hyperiqa/export.py`) on
    /// `engine`.
    ///
    /// `consent` must accept [`Self::WEIGHTS`] (the weights state no
    /// licence).
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`], plus `consent` does not accept
    /// [`Self::WEIGHTS`].
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (_channels, min_size) = read_meta(&dir.join("hyperiqa.meta"))?;
        let backend = match engine {
            Engine::Rten => Backend::Image(engine::load_image(&dir.join("hyperiqa.onnx"))?),
            #[cfg(feature = "ort")]
            gpu => Backend::Image(Box::new(engine::ort_engine::OrtGpuImage::load(
                &dir.join("hyperiqa.onnx"),
                "hyperiqa",
                gpu,
            )?)),
        };
        Ok(Self { backend, min_size })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    /// `Resize((512,384))` + ToTensor + Normalize + the deterministic
    /// 5×5 crop grid: `x [25, 3, 224, 224]` flattened.
    fn prep(img: &za::Rgb8In<'_>, min_size: usize) -> Result<Vec<f32>, Error> {
        let (h, w) = (img.height, img.width);
        if h < min_size || w < min_size {
            return Err(Error::TooSmall { min: min_size });
        }
        let planar = rgb8_to_planar_u8(img);
        let r = resize_bilinear_pil_u8(&planar, 3, h, w, RH, RW);
        let plane = RH * RW;
        let mut out = Vec::with_capacity(25 * 3 * CROP * CROP);
        for &y in &YS {
            for &x in &XS {
                for c in 0..3 {
                    let (m, s) = (MEAN[c], STD[c]);
                    for yy in 0..CROP {
                        let row =
                            &r[c * plane + (y + yy) * RW + x..c * plane + (y + yy) * RW + x + CROP];
                        for &v in row {
                            out.push((v as f32 / 255.0 - m) / s);
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Score one prepared image: `x [25,3,224,224]` in, scalar MOS out.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let x = Self::prep(img, self.min_size)?;
        match &self.backend {
            Backend::Image(g) => g.score(&[NamedInput {
                name: "x",
                shape: vec![25, 3, CROP, CROP],
                data: x,
            }]),
            Backend::Trunk(_) | Backend::Pair(_) => {
                Err(Error::Engine("hyperiqa needs an image-graph engine".into()))
            }
        }
    }
}

impl za::NoReferenceMetric for Hyperiqa {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "SSL92/hyperIQA @c42e7279 koniq_pretrained, 384x512 + deterministic 5x5 crop grid; engine {}",
            self.engine_name()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Quality)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate_image(
        &mut self,
        image: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let img = za::rgb8_view(&image)?;
        let s = self
            .score_image(&img)
            .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view<'a>(px: &'a [u8], w: usize, h: usize) -> za::Rgb8In<'a> {
        za::Rgb8In {
            pixels: px,
            width: w,
            height: h,
            stride: w * 3,
        }
    }

    /// Zero dimensions are refused before the resize; every nonzero
    /// size is accepted (the frame always resizes to 384x512).
    #[test]
    fn prep_size_bounds() {
        for (w, h) in [(0, 5), (5, 0)] {
            assert!(matches!(
                Hyperiqa::prep(&view(&[], w, h), 1),
                Err(Error::TooSmall { .. })
            ));
        }
        assert!(Hyperiqa::prep(&view(&[0u8; 3], 1, 1), 1).is_ok());
    }

    fn texture(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 3)
            .map(|i| ((i * 73 + i / 7 * 19 + i / (w * 3) * 31) % 256) as u8)
            .collect()
    }

    #[test]
    fn refuses_without_weights_consent() {
        let err = Hyperiqa::from_dir(
            "/nonexistent-zenmetrics-hyperiqa",
            Engine::Rten,
            &WeightsConsent::new(),
        )
        .err()
        .expect("must refuse before reading model files");
        assert!(matches!(err, Error::WeightsNotAccepted(_)), "{err}");
    }

    /// No resize is needed: each crop's full contents identify its source origin.
    #[test]
    fn prep_crop_grid() {
        let px = texture(384, 512);
        let actual = Hyperiqa::prep(&view(&px, 384, 512), 1).unwrap();
        assert_eq!(actual.len(), 25 * 3 * CROP * CROP);
        for crop in 0..25 {
            let x = (crop % 5) * 40;
            let y = (crop / 5) * 72; // crop 6 starts at (40,72).
            for c in 0..3 {
                for yy in 0..224 {
                    for xx in 0..224 {
                        let v = px[((y + yy) * 384 + x + xx) * 3 + c];
                        let expected = (f32::from(v) / 255.0 - MEAN[c]) / STD[c];
                        assert_eq!(
                            actual[((crop * 3 + c) * 224 + yy) * 224 + xx],
                            expected,
                            "crop {crop} origin ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    /// Compare the entire prepared tensor, with padding deliberately unrelated to pixels.
    #[test]
    fn prep_strided_rows() {
        let (w, h, stride) = (37, 29, 37 * 3 + 17);
        let packed = texture(w, h);
        let mut strided = vec![251; stride * h];
        for y in 0..h {
            strided[y * stride..y * stride + w * 3]
                .copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
        }
        let image = za::Rgb8In {
            pixels: &strided,
            width: w,
            height: h,
            stride,
        };
        assert_eq!(
            Hyperiqa::prep(&image, 1).unwrap(),
            Hyperiqa::prep(&view(&packed, w, h), 1).unwrap()
        );
    }
}
