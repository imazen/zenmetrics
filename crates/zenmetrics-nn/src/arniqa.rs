//! ARNIQA (Agnolucci, Galteri, Bertini, Del Bimbo, "ARNIQA: Learning
//! Distortion Manifold for No-Reference Image Quality Assessment", WACV
//! 2024; upstream `miccunifi/ARNIQA` at `66d16eb`, Apache-2.0, release
//! weights `ARNIQA.pth` + `regressor_kadid10k.pth`). Parity target:
//! upstream's `single_image_inference.py` on CPU fp32 (its autocast block is
//! a no-op off CUDA) with the default `kadid10k` regressor.
//!
//! The eval transform is upstream's: the sRGB8 image and its half-scale copy
//! (torchvision `Resize((h//2, w//2))` on a PIL image — bilinear, fixed-point
//! u8, [`resize_bilinear_pil_u8`]) each give five 224×224 crops — centre,
//! top-left, bottom-left, top-right, bottom-right — zero-padded where a crop
//! exceeds the image (TF.crop), `/255` to 0..1. The graphs carry the ImageNet
//! normalisation.
//!
//! `models/arniqa/export.py` splits the model in two: `arniqa_trunk.onnx`
//! maps one normalised-inside crop to the L2-normalised 2048-d ResNet-50
//! embedding `f`; the Rust head concatenates `[f, f_ds]`, applies the
//! exported Ridge regressor (`arniqa_head.f32`), rescales the kadid10k MOS
//! range 1..5 to 0..1 (`(s − 1)/4`) and means the five scores. On
//! whole-graph engines `arniqa_full.onnx` takes the two `[5, 3, 224, 224]`
//! crop batches and returns that scalar directly.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_f32, read_meta};
use crate::engine::{self, NamedInput, Tap};
use crate::preprocess::rgb8_to_planar_u8;
use crate::resize::resize_bilinear_pil_u8;
use crate::{Engine, Error, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "arniqa";
/// `center_corners_crop` size.
const CROP: usize = 224;
/// Regressor input: `[f, f_ds]` for two 2048-d embeddings.
const HEAD_LEN: usize = 4096;

/// ARNIQA, kadid10k regressor. Higher is better — a quality score in
/// `[0, 1]` (upstream's scaled MOS). Input: packed sRGB8 (any stride).
pub struct Arniqa {
    backend: Backend,
    /// `arniqa_head.f32`: regressor weights [4096] then bias.
    head: Vec<f32>,
    min_size: usize,
}

impl Arniqa {
    /// This model's weights' published terms: `ARNIQA.pth` and the
    /// regressors are released under upstream's "Model weights" heading and
    /// covered by the repository's Apache-2.0 licence.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Permissive,
        // The README publishes the encoder/regressor downloads; there is
        // no dedicated weights section to anchor to.
        terms_url: "https://github.com/miccunifi/ARNIQA",
    };

    /// Load an exported model directory (`arniqa_trunk.onnx`,
    /// `arniqa_full.onnx`, `arniqa_head.f32`, `arniqa.meta`, as written by
    /// `models/arniqa/export.py`) on `engine`. The ONNX Runtime engines read
    /// `arniqa_full.onnx` instead of the trunk graph.
    ///
    /// The weights are permissively licensed; `from_dir` takes no consent.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join("arniqa.meta"))?;
        if channels != [HEAD_LEN / 2] {
            return Err(Error::BadModel(format!(
                "arniqa.meta: channels {:?}, expected [{}]",
                channels,
                HEAD_LEN / 2
            )));
        }
        let head = read_f32(&dir.join("arniqa_head.f32"), HEAD_LEN + 1)?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(&dir.join("arniqa_trunk.onnx"), 1)?),
            #[cfg(feature = "ort")]
            gpu => Backend::Image(Box::new(engine::ort_engine::OrtGpuImage::load(
                &dir.join("arniqa_full.onnx"),
                "arniqa",
                gpu,
            )?)),
        };
        Ok(Self {
            backend,
            head,
            min_size,
        })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    /// One 224×224 crop of a planar u8 `[3, h, w]`, `top`/`left` signed,
    /// zero-padded past the edges (torchvision `TF.crop`), as f32 0..1.
    fn crop01(img: &[u8], h: usize, w: usize, top: i64, left: i64) -> Vec<f32> {
        let mut out = vec![0.0f32; 3 * CROP * CROP];
        for c in 0..3 {
            let src = &img[c * h * w..(c + 1) * h * w];
            let dst = &mut out[c * CROP * CROP..(c + 1) * CROP * CROP];
            for y in 0..CROP as i64 {
                let sy = top + y;
                if sy < 0 || sy >= h as i64 {
                    continue;
                }
                for x in 0..CROP as i64 {
                    let sx = left + x;
                    if sx >= 0 && sx < w as i64 {
                        dst[y as usize * CROP + x as usize] =
                            src[sy as usize * w + sx as usize] as f32 / 255.0;
                    }
                }
            }
        }
        out
    }

    /// The eval transform's two crop batches: the image and its
    /// PIL-bilinear half-scale copy, each `[5, 3, 224, 224]` f32 0..1 in
    /// `center_corners_crop` order (centre, TL, BL, TR, BR).
    fn prep(img: &za::Rgb8In<'_>) -> Result<(Vec<f32>, Vec<f32>), Error> {
        let (h, w) = (img.height, img.width);
        if h < 2 || w < 2 {
            return Err(Error::TooSmall { min: 2 });
        }
        let full = rgb8_to_planar_u8(img);
        let half = resize_bilinear_pil_u8(&full, 3, h, w, h / 2, w / 2);
        let crops = |img: &[u8], h: usize, w: usize| -> Vec<f32> {
            let (cx, cy) = (w / 2, h / 2);
            let (k, h64, w64) = (CROP as i64, h as i64, w as i64);
            let origins = [
                (cy as i64 - k / 2, cx as i64 - k / 2),
                (0, 0),
                (h64 - k, 0),
                (0, w64 - k),
                (h64 - k, w64 - k),
            ];
            let mut out = Vec::with_capacity(5 * 3 * CROP * CROP);
            for &(top, left) in &origins {
                out.extend_from_slice(&Self::crop01(img, h, w, top, left));
            }
            out
        };
        Ok((crops(&full, h, w), crops(&half, h / 2, w / 2)))
    }

    /// A trunk run's single `[channels, 1, 1]` tap as a 2048-d embedding.
    fn embed_tap(taps: Vec<Tap>) -> Result<Vec<f32>, Error> {
        let mut it = taps.into_iter();
        let (Some(t), None) = (it.next(), it.next()) else {
            return Err(Error::BadModel("expected exactly one tap".into()));
        };
        if t.data.len() != HEAD_LEN / 2 {
            return Err(Error::BadModel(format!(
                "embedding length {}, expected {}",
                t.data.len(),
                HEAD_LEN / 2
            )));
        }
        Ok(t.data)
    }

    /// Regressor on one `[f, f_ds]` pair — `x @ w.T + b` with the bias at
    /// `head[HEAD_LEN]`, accumulated in f64 (a host-side linear layer);
    /// upstream's `_scale_score` maps kadid10k MOS 1..5 to `[0, 1]`.
    fn head(head: &[f32], f: &[f32], f_ds: &[f32]) -> f32 {
        let (w, b) = (&head[..HEAD_LEN], f64::from(head[HEAD_LEN]));
        let mut s = b;
        for i in 0..HEAD_LEN / 2 {
            s += f64::from(w[i]) * f64::from(f[i]);
            s += f64::from(w[i + HEAD_LEN / 2]) * f64::from(f_ds[i]);
        }
        ((s as f32) - 1.0) / 4.0
    }

    /// Score one prepared image. Trunk engines run the ten crops and fold
    /// the head here; whole-graph engines take the two crop batches.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if img.height < self.min_size || img.width < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        let (x, x_ds) = Self::prep(img)?;
        match &self.backend {
            Backend::Trunk(t) => {
                let mut scores = Vec::with_capacity(5);
                for i in 0..5 {
                    let span = i * 3 * CROP * CROP..(i + 1) * 3 * CROP * CROP;
                    // The full- and half-scale crop runs are independent:
                    // pair them so each pool half covers one — same taps,
                    // same head, same accumulation order.
                    let (ft, ft_ds) =
                        t.run_pair(x[span.clone()].to_vec(), x_ds[span].to_vec(), CROP, CROP)?;
                    let f = Self::embed_tap(ft)?;
                    let f_ds = Self::embed_tap(ft_ds)?;
                    scores.push(Self::head(&self.head, &f, &f_ds));
                }
                Ok(f64::from(scores.iter().sum::<f32>() / 5.0))
            }
            Backend::Image(g) => g.score(vec![
                NamedInput {
                    name: "x",
                    shape: vec![5, 3, CROP, CROP],
                    data: x,
                },
                NamedInput {
                    name: "x_ds",
                    shape: vec![5, 3, CROP, CROP],
                    data: x_ds,
                },
            ]),
            Backend::Pair(_) => Err(Error::Engine("pair graph on an NR model".into())),
        }
    }
}

impl za::NoReferenceMetric for Arniqa {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "miccunifi/ARNIQA @66d16eb kadid10k regressor, 5x224 crops at full + PIL-bilinear half scale; engine {}",
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

    /// An image under half-scale minimum (the half view needs `h//2 >= 1`)
    /// is refused, as is an empty one.
    #[test]
    fn prep_refuses_small() {
        for (w, h) in [(1, 1), (1, 10), (10, 1), (0, 4)] {
            let px = vec![0u8; w * h * 3];
            assert!(matches!(
                Arniqa::prep(&view(&px, w, h)),
                Err(Error::TooSmall { .. })
            ));
        }
        assert!(Arniqa::prep(&view(&[0u8; 2 * 2 * 3], 2, 2)).is_ok());
    }

    /// A row-stride larger than `w*3` reads the same pixels.
    #[test]
    fn prep_strided_rows() {
        let (w, h, stride) = (5, 4, 5 * 3 + 7);
        let mut packed = vec![0u8; w * h * 3];
        let mut buf = vec![0xEEu8; stride * h];
        for (i, v) in packed.iter_mut().enumerate() {
            *v = (i * 13 + 7) as u8;
        }
        for y in 0..h {
            buf[y * stride..y * stride + w * 3]
                .copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
        }
        let s = za::Rgb8In {
            pixels: &buf,
            width: w,
            height: h,
            stride,
        };
        assert_eq!(
            Arniqa::prep(&s).unwrap(),
            Arniqa::prep(&view(&packed, w, h)).unwrap()
        );
    }

    /// The five crops come out in `center_corners_crop` order — centre,
    /// TL, BL, TR, BR — and sub-224 images zero-pad where the crop exceeds
    /// the image.
    #[test]
    fn crops_order_and_padding() {
        // 4x4 planar, one channel worth of pixels per position y*4+x+1.
        let (w, h) = (4, 4);
        let mut img = vec![0u8; 3 * h * w];
        for c in 0..3 {
            for i in 0..h * w {
                img[c * h * w + i] = (i + 1) as u8;
            }
        }
        let v = |top: i64, left: i64, y: usize, x: usize| {
            Arniqa::crop01(&img, h, w, top, left)[y * CROP + x]
        };
        // TL crop (0,0): pixel (0,0) = 1/255.
        assert_eq!(v(0, 0, 0, 0), 1.0 / 255.0);
        // BR crop starts at (h-224, w-224) = (-220, -220): its (220,220) is
        // the image's (0,0); (0,0) of the crop is padding.
        assert_eq!(v(-220, -220, 0, 0), 0.0);
        assert_eq!(v(-220, -220, 220, 220), 1.0 / 255.0);
        assert_eq!(v(-220, -220, 223, 223), 16.0 / 255.0);
    }

    /// The Ridge head: bias plus the two half-weight dots, rescaled by the
    /// kadid10k MOS range `(s − 1)/4`.
    #[test]
    fn head_is_linear_scaled() {
        let mut head = vec![0.0f32; HEAD_LEN + 1];
        head[0] = 2.0;
        head[HEAD_LEN / 2] = 4.0;
        head[HEAD_LEN] = 3.0; // bias
        let (mut f, mut f_ds) = (vec![0.0f32; 2048], vec![0.0f32; 2048]);
        f[0] = 1.0;
        f_ds[0] = 0.5;
        // s = 3 + 2*1 + 4*0.5 = 7; (7 - 1)/4 = 1.5
        assert_eq!(Arniqa::head(&head, &f, &f_ds), 1.5);
    }
}
