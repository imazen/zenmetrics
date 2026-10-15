//! MANIQA (Yang, Ding, Wang, Dai et al., "MANIQA: Multi-dimension
//! Attention Network for No-Reference Image Quality Assessment",
//! CVPRW 2022; upstream `IIGROUP/MANIQA` at `f573d86`, Apache-2.0,
//! release checkpoint `ckpt_koniq10k.pt`). Parity target: upstream's
//! deterministic eval path — `train_maniqa.py` validation for
//! `dataset_name = "koniq10k"` (`num_avg_val = 1`, `crop_size = 224`),
//! where `five_point_crop(0)` on the pre-resized 224 image is the image
//! itself — on CPU fp32.
//!
//! The eval transform is upstream's dataset loader (`data/koniq10k.py`):
//! the sRGB8 image is resized to 224×224 with OpenCV `INTER_CUBIC` on u8
//! ([`resize_bicubic_cv2_u8`]), `/255` to 0..1, then
//! `Normalize(0.5, 0.5)` — `(x − 0.5)/0.5` — into the `[1, 3, 224, 224]`
//! input. Upstream's `predict_one_image.py` instead averages 20
//! `np.random` crops with no resize; it is not reproducible host-side,
//! so the scored convention here is the deterministic one that produced
//! upstream's published koniq10k numbers.
//!
//! `models/maniqa/export.py` exports `maniqa_full.onnx`: ViT-B/8 blocks
//! 6..9 token features → TABlocks → SwinTransformer stages →
//! per-token `fc_score·fc_weight` weighted mean → `score[1,1,1]`. rten
//! runs it as a one-tap trunk graph; ONNX Runtime engines run it as a
//! named-input whole-image graph.
//!
//! The output is the raw koniq10k checkpoint score — an unbounded
//! quality number (roughly MOS 1..5 territory, the trained head applies
//! ReLU), higher is better.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, NamedInput};
use crate::preprocess::rgb8_to_planar_u8;
use crate::resize::resize_bicubic_cv2_u8;
use crate::{Engine, Error, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "maniqa";
/// The upstream fixed input size.
const SIDE: usize = 224;

/// MANIQA, koniq10k checkpoint. Higher is better. Input: packed sRGB8
/// (any stride); every input resizes to 224×224 first, so any size is
/// accepted.
pub struct Maniqa {
    backend: Backend,
}

impl Maniqa {
    /// This model's weights' published terms: `ckpt_koniq10k.pt` is a
    /// GitHub release asset covered by the repository's Apache-2.0
    /// licence.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Permissive,
        terms_url: "https://github.com/IIGROUP/MANIQA#model-zoo",
    };

    /// Load an exported model directory (`maniqa_full.onnx`,
    /// `maniqa.meta`, as written by `models/maniqa/export.py`) on
    /// `engine`.
    ///
    /// The weights are permissively licensed; `from_dir` takes no
    /// consent.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (_channels, _min_size) = read_meta(&dir.join("maniqa.meta"))?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(&dir.join("maniqa_full.onnx"), 1)?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Image(Box::new(engine::ort_engine::OrtCudaImage::load(
                &dir.join("maniqa_full.onnx"),
                "maniqa",
            )?)),
        };
        Ok(Self { backend })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    /// The dataset-loader transform: cv2 INTER_CUBIC u8 resize to
    /// 224×224, `/255`, `(x − 0.5)/0.5` — `[1, 3, 224, 224]` planar.
    fn prep(img: &za::Rgb8In<'_>) -> Result<Vec<f32>, Error> {
        let full = rgb8_to_planar_u8(img);
        let r = resize_bicubic_cv2_u8(&full, 3, img.height, img.width, SIDE, SIDE);
        Ok(r.iter()
            .map(|&v| (f32::from(v) / 255.0 - 0.5) / 0.5)
            .collect())
    }

    /// Score one prepared image: the whole-image graph ends in
    /// `score[1,1,1]` — one tap for trunk engines, the scalar for
    /// image-graph engines.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        let x = Self::prep(img)?;
        match &self.backend {
            Backend::Trunk(t) => {
                let taps = t.run(x, SIDE, SIDE)?;
                let mut it = taps.into_iter();
                let (Some(tap), None) = (it.next(), it.next()) else {
                    return Err(Error::BadModel("expected exactly one tap".into()));
                };
                if tap.data.len() != 1 {
                    return Err(Error::BadModel(format!(
                        "score tap length {}, expected 1",
                        tap.data.len()
                    )));
                }
                Ok(f64::from(tap.data[0]))
            }
            Backend::Image(g) => g.score(&[NamedInput {
                name: "x",
                shape: [1, 3, SIDE, SIDE],
                data: x,
            }]),
            Backend::Pair(_) => Err(Error::Engine("pair graph on an NR model".into())),
        }
    }
}

impl za::NoReferenceMetric for Maniqa {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "IIGROUP/MANIQA @f573d86 ckpt_koniq10k, 224 INTER_CUBIC single forward; engine {}",
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

    /// Every size goes through the 224 resize — even 1×1.
    #[test]
    fn prep_accepts_any_size() {
        for (w, h) in [(1, 1), (7, 3), (2, 2)] {
            let px = vec![0u8; w * h * 3];
            assert_eq!(
                Maniqa::prep(&view(&px, w, h)).unwrap().len(),
                3 * SIDE * SIDE
            );
        }
    }

    /// `(x − 0.5)/0.5` after `/255`: u8 0 → −1.0, 255 → +1.0.
    #[test]
    fn prep_normalises() {
        let px = vec![0u8; 3 * 4];
        let x = Maniqa::prep(&view(&px, 2, 2)).unwrap();
        assert_eq!(x[0], -1.0);
        let px = vec![255u8; 3 * 4];
        let x = Maniqa::prep(&view(&px, 2, 2)).unwrap();
        assert_eq!(x[0], 1.0);
    }
}
