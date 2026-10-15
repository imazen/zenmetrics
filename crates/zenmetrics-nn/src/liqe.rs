//! LIQE (Zhang, Zhai, Wei, Yang, Ma, "Blind Image Quality Assessment via
//! Vision-Language Correspondence: A Multitask Learning Perspective",
//! CVPR 2023, pp. 14071–14081; upstream
//! `zwx8981/LIQE` at `a64655f`, MIT code — **weights carry no stated
//! licence**: `WeightsLicence::Unknown`, consent-gated). Parity target:
//! upstream `LIQE.forward` (the module `demo2.py` drives) on CPU fp32,
//! `LIQE.pt` over OpenAI CLIP ViT-B/32.
//!
//! The eval transform is upstream's: `ToTensor` (u8 → 0..1, no resize),
//! then the module CLIP-normalises and unfolds 224×224 patches at
//! stride 32, partial borders dropped — and selects
//! `sel[i] = (batch//15)·i`: for the single-image call batch is 1, so
//! `sel = [0; 15]` — the top-left 224×224 patch scored 15 times.
//! Images under 224×224 yield zero patches (upstream indexes an empty
//! patch dim) → `TooSmall`.
//!
//! `models/liqe/export.py` splits the model: `liqe_trunk.onnx` maps a
//! 0..1 crop to the L2-normalised 512-d CLIP image-tower embedding;
//! `liqe_full.onnx` runs the 15 selected crops to the quality scalar
//! end to end. The head (`liqe_head.f32`: `logit_scale.exp()` plus the
//! 495 L2-normalised joint-prompt text features) computes
//! `softmax(mean_patches(scale · f·tᵀ))` marginalised to the 5 quality
//! buckets, scored `Σ (k+1)·p_k` — a 1..5 MOS.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_f32, read_meta};
use crate::engine::{self, NamedInput, Trunk};
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "liqe";
/// Patch size and selection count (`self.step`, `self.num_patch`).
const CROP: usize = 224;
const NUM_PATCH: usize = 15;
/// CLIP embedding width and joint-prompt count (5×9×11).
const DIM: usize = 512;
const NPROMPTS: usize = 495;
const HEAD_LEN: usize = 1 + NPROMPTS * DIM;

/// LIQE on `LIQE.pt`. Higher is better — a 1..5 MOS. Input: packed
/// sRGB8 (any stride), `min_size` 224.
pub struct Liqe {
    backend: Backend,
    /// `liqe_head.f32`: logit_scale.exp() then text features [495,512].
    head: Vec<f32>,
    min_size: usize,
}

impl Liqe {
    /// This model's weights' published terms: the README's
    /// pre-trained-weight link (Google Drive / Baidu) states no
    /// licence; the code is MIT. Consent is required to load.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Unknown,
        terms_url: "https://github.com/zwx8981/LIQE#pre-trained-weights",
    };

    /// Load an exported model directory (`liqe_trunk.onnx`,
    /// `liqe_full.onnx`, `liqe_head.f32`, `liqe.meta`, as written by
    /// `models/liqe/export.py`) on `engine`.
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
        let (channels, min_size) = read_meta(&dir.join("liqe.meta"))?;
        if channels != [DIM] {
            return Err(Error::BadModel(format!(
                "liqe.meta: channels {channels:?}, expected [{DIM}]"
            )));
        }
        let head = read_f32(&dir.join("liqe_head.f32"), HEAD_LEN)?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(&dir.join("liqe_trunk.onnx"), 1)?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Image(Box::new(engine::ort_engine::OrtCudaImage::load(
                &dir.join("liqe_full.onnx"),
                "liqe",
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

    /// The selected patch: `unfold(224, 32)` then `sel = [0; 15]` for a
    /// single-image batch picks the top-left 224×224 crop 15 times. Returns
    /// that crop once as planar f32 0..1 `[3, 224, 224]`, read straight from
    /// the strided rows (no full-image copy).
    fn prep(img: &za::Rgb8In<'_>) -> Result<Vec<f32>, Error> {
        if img.height < CROP || img.width < CROP {
            return Err(Error::TooSmall { min: CROP });
        }
        let mut out = vec![0.0f32; 3 * CROP * CROP];
        for y in 0..CROP {
            let row = &img.pixels[y * img.stride..y * img.stride + 3 * CROP];
            for (x, px) in row.as_chunks::<3>().0.iter().enumerate() {
                for c in 0..3 {
                    out[c * CROP * CROP + y * CROP + x] = f32::from(px[c]) / 255.0;
                }
            }
        }
        Ok(out)
    }

    /// A trunk run's `[512,1,1]` tap as an embedding.
    fn embed(trunk: &dyn Trunk, crop: &[f32]) -> Result<Vec<f32>, Error> {
        let taps = trunk.run(crop.to_vec(), CROP, CROP)?;
        let mut it = taps.into_iter();
        let (Some(t), None) = (it.next(), it.next()) else {
            return Err(Error::BadModel("expected exactly one tap".into()));
        };
        if t.data.len() != DIM {
            return Err(Error::BadModel(format!(
                "embedding length {}, expected {DIM}",
                t.data.len()
            )));
        }
        Ok(t.data)
    }

    /// The head on the selected patches' embeddings: per-patch logits
    /// `scale·f·tᵀ` over the 495 prompts, mean over patches, softmax,
    /// marginalised to 5 quality buckets, `Σ (k+1)·p_k`. f64 on host.
    fn head(head: &[f32], embs: &[Vec<f32>]) -> f64 {
        let (scale, text) = (f64::from(head[0]), &head[1..]);
        let mut logits = vec![0.0f64; NPROMPTS];
        for emb in embs {
            for (j, l) in logits.iter_mut().enumerate() {
                let mut d = 0.0f64;
                for i in 0..DIM {
                    d += f64::from(emb[i]) * f64::from(text[j * DIM + i]);
                }
                *l += scale * d;
            }
        }
        for l in &mut logits {
            *l /= embs.len() as f64;
        }
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for l in &mut logits {
            *l = (*l - max).exp();
            sum += *l;
        }
        // Quality marginal: view (5,9,11), sum over distortion & scene.
        let mut q = [0.0f64; 5];
        for (i, &l) in logits.iter().enumerate() {
            q[i / (9 * 11)] += l;
        }
        (1..=5).map(|k| k as f64 * q[k - 1] / sum).sum()
    }

    /// Score one prepared image.
    fn score_image(&self, img: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if img.height < self.min_size || img.width < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        let x = Self::prep(img)?;
        match &self.backend {
            Backend::Trunk(t) => {
                // All 15 selected patches are the same crop: run the trunk once
                // and give the head 15 copies, so its arithmetic is unchanged.
                let emb = Self::embed(&**t, &x)?;
                Ok(Self::head(&self.head, &vec![emb; NUM_PATCH]))
            }
            Backend::Image(g) => g.score(&[NamedInput {
                name: "x",
                shape: vec![NUM_PATCH, 3, CROP, CROP],
                // The full graph takes the 15 selected patches explicitly.
                data: x.repeat(NUM_PATCH),
            }]),
            Backend::Pair(_) => Err(Error::Engine("pair graph on an NR model".into())),
        }
    }
}

impl za::NoReferenceMetric for Liqe {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "zwx8981/LIQE @a64655f LIQE.pt over CLIP ViT-B/32, LIQE.forward 15x sel patches; engine {}",
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

    /// Without consent to its `Unknown`-licence weights, loading fails before
    /// any file is read (the directory does not exist).
    #[test]
    fn refuses_without_weights_consent() {
        let err = Liqe::from_dir(
            "/nonexistent-zenmetrics-liqe",
            Engine::Rten,
            &WeightsConsent::new(),
        )
        .err()
        .expect("must refuse");
        assert!(matches!(err, Error::WeightsNotAccepted(_)), "{err}");
    }

    /// Images under the 224 patch size are refused.
    #[test]
    fn prep_refuses_small() {
        for (w, h) in [(223, 300), (300, 223), (1, 1)] {
            let px = vec![0u8; w * h * 3];
            assert!(matches!(
                Liqe::prep(&view(&px, w, h)),
                Err(Error::TooSmall { .. })
            ));
        }
        assert!(Liqe::prep(&view(&vec![7u8; 3 * 224 * 224], 224, 224)).is_ok());
    }

    /// `sel = [0; 15]` for B=1 → all 15 patches are the top-left 224 crop;
    /// `prep` returns that crop once, planar and scaled to 0..1, and every
    /// element equals the image pixel it came from.
    #[test]
    fn prep_is_top_left_crop() {
        let (w, h) = (300, 300);
        let mut px = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    px[(y * w + x) * 3 + c] = (x * 3 + y * 5 + c * 7) as u8;
                }
            }
        }
        let x = Liqe::prep(&view(&px, w, h)).unwrap();
        assert_eq!(x.len(), 3 * CROP * CROP);
        for c in 0..3 {
            for y in 0..CROP {
                for xx in 0..CROP {
                    let want = f32::from(px[(y * w + xx) * 3 + c]) / 255.0;
                    assert_eq!(x[c * CROP * CROP + y * CROP + xx], want, "c{c} y{y} x{xx}");
                }
            }
        }
    }

    /// Prep is stride-agnostic: a strided view of a wider buffer equals
    /// the packed view.
    #[test]
    fn prep_strided_matches_packed() {
        let (w, h, stride) = (250, 260, 300 * 3);
        let mut wide = vec![0u8; stride * h];
        let mut packed = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = (x * 7 + y * 11 + c * 13) as u8;
                    wide[y * stride + x * 3 + c] = v;
                    packed[(y * w + x) * 3 + c] = v;
                }
            }
        }
        let strided = za::Rgb8In {
            pixels: &wide,
            width: w,
            height: h,
            stride,
        };
        assert_eq!(
            Liqe::prep(&view(&packed, w, h)).unwrap(),
            Liqe::prep(&strided).unwrap()
        );
    }

    /// The head: softmax over 495 prompt logits marginalised to the 5
    /// quality buckets.
    #[test]
    fn head_marginalises() {
        let mut head = vec![0.0f32; HEAD_LEN];
        head[0] = 1.0; // scale
        // prompt 0 (quality bucket 0) aligns with emb[0]; all others orth.
        head[1] = 1.0;
        let mut emb = vec![0.0f32; DIM];
        emb[0] = 2.0;
        // logit0 = 2, others 0 → after max-subtraction bucket 0 holds
        // 1 + 98·e⁻² (prompt 0 plus 98 siblings), buckets 1..4 hold
        // 99·e⁻² each, over sum = 1 + 494·e⁻².
        let e2 = (-2.0f64).exp();
        let s = 1.0 + 494.0 * e2;
        let want = (1.0 + 98.0 * e2) / s + (99.0 * e2 / s) * (2.0 + 3.0 + 4.0 + 5.0);
        let got = Liqe::head(&head, &[emb]);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
