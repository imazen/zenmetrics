//! DreamSim ensemble (Steiner, Sundaram et al., "DreamSim: Learning New
//! Dimensions of Human Visual Similarity using Synthetic Data", NeurIPS 2023;
//! upstream `ssundaram21/dreamsim`, MIT). Parity target: `PerceptualModel`'s
//! `forward` — the learned ensemble distance in `[0, 2]`, typically `0..0.3`.
//!
//! The ensemble is three LoRA-finetuned ViT-B/16 extractors: DINO (class
//! token of the last block, pre final LayerNorm, 768d), OpenAI CLIP and
//! OpenCLIP (`visual @ proj`, 512d each). Upstream concatenates the three
//! embeddings to 1792d, mean-centres and L2-normalises the vector, and
//! reports `1 - cosine(a, b)` between the two images' embeddings. The LoRA
//! adapters (rank 16, alpha 1.0, `qkv` target on all 12 blocks of each
//! extractor) are folded into the ViT weights at export by
//! `models/dreamsim/export.py` (peft `merge_and_unload`, verified to ~6e-8
//! on the model's own output).
//!
//! `dreamsim_trunk.onnx` runs the three extractors on trunk engines, emitting
//! `[768,1,1]`, `[512,1,1]`, `[512,1,1]` taps; the concat/normalise/cosine
//! head is this file. `dreamsim_full.onnx` is the whole pipeline for whole-graph
//! engines. Each graph normalises its input internally with its own stats
//! (ImageNet for DINO, CLIP for the other two).
//!
//! Input preprocessing is upstream's `preprocess`: PIL bicubic (antialiased)
//! resize to exactly 224x224 — anamorphic, ignoring aspect — then `ToTensor`
//! (0..1).

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, Tap};
use crate::preprocess::rgb8_to_planar_u8;
use crate::resize::resize_bicubic_pil_u8;
use crate::{Engine, Error, WeightsConsent, WeightsLicence, WeightsTerms, za};

/// Bump whenever any output can change.
const SCORE_REVISION: u32 = 1;
const COLUMN: &str = "dreamsim";
/// Upstream resizes both axes to this size (anamorphic).
const IMG: usize = 224;
/// Concatenated embedding length: 768 + 512 + 512.
const EMBED: usize = 1792;

/// DreamSim ensemble distance. Lower is closer (`0` = identical). Input:
/// packed sRGB8 (any stride).
pub struct Dreamsim {
    backend: Backend,
    channels: [usize; 3],
}

impl Dreamsim {
    /// This model's weights' published terms: the ensemble checkpoint
    /// (`dreamsim_ensemble.zip`, hosted on upstream's bucket and downloaded
    /// by the package at init) carries no stated licence, so loading needs a
    /// [`WeightsConsent`] accepting these terms.
    pub const WEIGHTS: WeightsTerms = WeightsTerms {
        model: COLUMN,
        licence: WeightsLicence::Unknown,
        terms_url: "https://github.com/ssundaram21/dreamsim",
    };

    /// Load an exported model directory (`dreamsim_trunk.onnx`, `dreamsim_full.onnx`,
    /// `dreamsim.meta`, as written by `models/dreamsim/export.py`) on `engine`.
    /// The ONNX Runtime engines read `dreamsim_full.onnx` instead of the trunk
    /// graph.
    ///
    /// `consent` must accept [`Self::WEIGHTS`] (the weights state no
    /// licence); it is checked before any file is read.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`], plus [`Error::WeightsNotAccepted`]
    /// when `consent` does not accept [`Self::WEIGHTS`].
    pub fn from_dir(
        dir: impl AsRef<Path>,
        engine: Engine,
        consent: &WeightsConsent,
    ) -> Result<Self, Error> {
        consent.check(&Self::WEIGHTS)?;
        let dir = dir.as_ref();
        let (channels, _min_size) = read_meta(&dir.join("dreamsim.meta"))?;
        let channels: [usize; 3] = channels
            .as_slice()
            .try_into()
            .map_err(|_| Error::BadModel("dreamsim.meta: expected 3 tap widths".into()))?;
        if channels.iter().sum::<usize>() != EMBED {
            return Err(Error::BadModel(format!(
                "dreamsim.meta: tap widths {channels:?} do not sum to {EMBED}"
            )));
        }
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(&dir.join("dreamsim_trunk.onnx"), 3)?),
            #[cfg(feature = "ort")]
            gpu => Backend::Pair(Box::new(engine::ort_engine::OrtGpuPair::load(
                &dir.join("dreamsim_full.onnx"),
                "dreamsim",
                gpu,
            )?)),
        };
        Ok(Self { backend, channels })
    }

    fn engine_name(&self) -> &'static str {
        match &self.backend {
            Backend::Trunk(t) => t.engine(),
            Backend::Pair(p) => p.engine(),
            Backend::Image(g) => g.engine(),
        }
    }

    /// Upstream's `preprocess`: torchvision `Resize((224, 224), BICUBIC)`
    /// on the PIL image — Pillow's anamorphic fixed-point bicubic on u8 —
    /// then `ToTensor` (`v/255` planar f32).
    fn prep(img: &za::Rgb8In<'_>) -> Result<Vec<f32>, Error> {
        if img.width == 0 || img.height == 0 {
            return Err(Error::TooSmall { min: 1 });
        }
        let planar = rgb8_to_planar_u8(img);
        let u8r = resize_bicubic_pil_u8(&planar, 3, img.height, img.width, IMG, IMG);
        Ok(u8r.into_iter().map(|v| v as f32 * (1.0 / 255.0)).collect())
    }

    /// The three trunk taps concatenated into upstream's 1792d embedding.
    fn embed_of(&self, taps: Vec<Tap>) -> Result<Vec<f32>, Error> {
        if taps.len() != 3 {
            return Err(Error::BadModel(format!(
                "expected 3 taps, got {}",
                taps.len()
            )));
        }
        let mut e = Vec::with_capacity(EMBED);
        for (tap, &c) in taps.iter().zip(&self.channels) {
            if tap.c != c || tap.plane() != 1 || tap.data.len() != c {
                return Err(Error::BadModel(format!(
                    "tap shape [{}, {}, {}], expected [{c}, 1, 1]",
                    tap.c, tap.h, tap.w
                )));
            }
            e.extend_from_slice(&tap.data);
        }
        Ok(e)
    }

    /// `ds_trunk(img)` on a trunk engine — the reference-side embedding.
    fn embed(&self, img: &za::Rgb8In<'_>) -> Result<Vec<f32>, Error> {
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        self.embed_of(t.run(Self::prep(img)?, IMG, IMG)?)
    }

    /// Upstream's `normalize_embedding` + cosine distance: mean-centre,
    /// L2-normalise, `1 - dot` on unit vectors.
    fn distance(a: &[f32], b: &[f32]) -> f64 {
        fn unit(e: &[f32]) -> Vec<f64> {
            let n = e.len() as f64;
            let mean = e.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
            let v: Vec<f64> = e.iter().map(|&x| f64::from(x) - mean).collect();
            let norm = v
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(f64::MIN_POSITIVE);
            v.into_iter().map(|x| x / norm).collect()
        }
        let (ua, ub) = (unit(a), unit(b));
        let cos: f64 = ua.iter().zip(&ub).map(|(x, y)| x * y).sum();
        1.0 - cos.clamp(-1.0, 1.0)
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        match &self.backend {
            Backend::Trunk(t) => {
                let (pr, pd) = (Self::prep(r)?, Self::prep(d)?);
                let (tr, td) = std::thread::scope(|s| {
                    let hr = s.spawn(|| t.run(pr, IMG, IMG));
                    let td = t.run(pd, IMG, IMG);
                    (
                        hr.join()
                            .map_err(|_| Error::Engine("trunk run panicked".into()))
                            .and_then(|x| x),
                        td,
                    )
                });
                Ok(Self::distance(&self.embed_of(tr?)?, &self.embed_of(td?)?))
            }
            Backend::Pair(p) => {
                // The graph emits unclamped 1-cos; mirror distance()'s
                // cosine clamp so identical pairs score exactly 0.
                Ok(p.score(Self::prep(r)?, Self::prep(d)?, IMG, IMG)?
                    .clamp(0.0, 2.0))
            }
            Backend::Image(_) => Err(Error::Engine("image graph on a pair model".into())),
        }
    }
}

impl za::Metric for Dreamsim {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "ssundaram21/dreamsim ensemble (dino_vitb16 cls + clip_vitb16 + open_clip_vitb16 embeddings, LoRA folded), 1-cos of centred unit concat, anamorphic 224x224 bicubic-aa; engine {}",
            self.engine_name()
        ))
    }

    fn outputs(&self) -> Cow<'static, [za::ScoreSpec]> {
        const OUTPUTS: &[za::ScoreSpec] = &[za::ScoreSpec::new(COLUMN, za::ScoreKind::Difference)];
        Cow::Borrowed(OUTPUTS)
    }

    fn accepts(&self, format: &za::InputFormat) -> bool {
        za::is_rgb8_srgb(format)
    }

    fn evaluate(
        &mut self,
        r: za::Image<'_>,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let (r, d) = za::rgb8_pair(&r, &d)?;
        let s = self
            .score_pair(&r, &d)
            .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::WarmReference + 'm>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        let view = za::rgb8_view(&r)?;
        let reference = match &self.backend {
            Backend::Trunk(_) => {
                WarmState::Embed(self.embed(&view).metric_err_with(COLUMN, Error::kind)?)
            }
            Backend::Pair(_) => {
                WarmState::Pixels(Self::prep(&view).metric_err_with(COLUMN, Error::kind)?)
            }
            Backend::Image(_) => {
                return Err(Error::Engine("image graph on a pair model".into()))
                    .metric_err_with(COLUMN, Error::kind);
            }
        };
        Ok(Box::new(WarmDreamsim {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its embedding on a trunk engine, the
/// converted input on a whole-graph engine (which recomputes features).
struct WarmDreamsim<'m> {
    metric: &'m Dreamsim,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    Embed(Vec<f32>),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmDreamsim<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let view = za::rgb8_view(&d)?;
        if (view.width, view.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        let s = match (&self.metric.backend, &self.reference) {
            (Backend::Trunk(_), WarmState::Embed(er)) => {
                let ed = self
                    .metric
                    .embed(&view)
                    .metric_err_with(COLUMN, Error::kind)?;
                Ok(Dreamsim::distance(er, &ed))
            }
            (Backend::Pair(p), WarmState::Pixels(pr)) => p
                .score(
                    pr.clone(),
                    Dreamsim::prep(&view).metric_err_with(COLUMN, Error::kind)?,
                    IMG,
                    IMG,
                )
                .map(|s| s.clamp(0.0, 2.0)),
            _ => Err(Error::Engine("warm state does not match engine".into())),
        }
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Embed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prep_refuses_zero_dimensions() {
        for (width, height) in [(0, 5), (5, 0), (0, 0)] {
            let image = za::Rgb8In {
                pixels: &[],
                width,
                height,
                stride: 0,
            };
            assert!(matches!(
                Dreamsim::prep(&image),
                Err(Error::TooSmall { min: 1 })
            ));
        }
    }
}
