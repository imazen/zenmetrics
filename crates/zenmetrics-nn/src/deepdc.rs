//! DeepDC (Zhu, Chen, Zhu, Wang, Lin, "DeepDC: Deep Distance Correlation as
//! a Perceptual Image Quality Evaluator", arXiv:2211.04927; upstream
//! `DeepDC_PyTorch` 0.2, MIT). Parity target: that package, without the
//! resize in its demo's `prepare_image`.
//!
//! The exported trunk (`models/deepdc/export.py`) runs VGG19 to the five taps
//! (conv1_2, conv2_2, conv3_4, conv4_4, conv5_4, before ReLU) with upstream's
//! ImageNet input normalisation switched on (`use_input_norm=True`: the
//! package default is off, but its source says correlation wants it and the
//! JPEG AIC-4 `DeepDC` column matches only with it), and the upstream `_DCDM`
//! on each, so it returns one
//! double-centred C × C distance matrix per tap (as a `[1, C, C, 1]` map).
//! The head is the rest of `DeepDC.forward`: per tap, the distance
//! correlation `r = (Σ A∘B + c) / (√(Σ A∘A · Σ B∘B) + c)` with `c = 1e-6`, and
//! `DeepDC = 1 − mean(r)`. Sums are accumulated in f64; each `r`, the mean
//! and the final subtraction are f32, as upstream's tensors are, so an
//! identical pair scores exactly 0 as it does upstream.

use std::borrow::Cow;
use std::path::Path;

use crate::common::{Backend, read_meta};
use crate::engine::{self, Tap};
use crate::preprocess::rgb8_to_planar_01;
use crate::{Engine, Error, za};

const C: f64 = 1e-6;
/// Bump whenever any output can change. 2: ImageNet input normalisation
/// switched on (revision 1, pushed briefly on 2026-10-15, had it off).
const SCORE_REVISION: u32 = 2;
const COLUMN: &str = "deepdc";

/// DeepDC. Lower is closer; `0` for identical inputs. Input: packed sRGB8
/// (any stride).
pub struct DeepDc {
    backend: Backend,
    channels: Vec<usize>,
    min_size: usize,
}

/// One tap's distance matrix and its self-product `Σ A∘A`.
struct Dcdm {
    tap: Tap,
    self_dot: f64,
}

impl Dcdm {
    fn new(tap: Tap) -> Self {
        let self_dot = tap.data.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        Self { tap, self_dot }
    }
}

impl DeepDc {
    /// Load an exported model directory (`deepdc_trunk.onnx`, `deepdc.meta`,
    /// as written by `models/deepdc/export.py`) on `engine`. The ONNX Runtime
    /// engines read `deepdc_full.onnx` instead of the trunk.
    ///
    /// # Errors
    /// As [`crate::Lpips::from_dir`].
    pub fn from_dir(dir: impl AsRef<Path>, engine: Engine) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let (channels, min_size) = read_meta(&dir.join("deepdc.meta"))?;
        let backend = match engine {
            Engine::Rten => Backend::Trunk(engine::load(
                &dir.join("deepdc_trunk.onnx"),
                channels.len(),
            )?),
            #[cfg(feature = "ort")]
            Engine::OrtCuda => Backend::Pair(Box::new(engine::ort_engine::OrtCudaPair::load(
                &dir.join("deepdc_full.onnx"),
                "deepdc",
            )?)),
        };
        Ok(Self {
            backend,
            channels,
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

    fn check_size(&self, img: &za::Rgb8In<'_>) -> Result<(), Error> {
        if img.width < self.min_size || img.height < self.min_size {
            return Err(Error::TooSmall { min: self.min_size });
        }
        Ok(())
    }

    fn check(&self, taps: Vec<Tap>) -> Result<Vec<Dcdm>, Error> {
        if taps.len() != self.channels.len() {
            return Err(Error::BadModel(format!(
                "{} taps, expected {}",
                taps.len(),
                self.channels.len()
            )));
        }
        for (t, &c) in taps.iter().zip(&self.channels) {
            if (t.c, t.h, t.w) != (c, c, 1) || t.data.len() != c * c {
                return Err(Error::BadModel(format!(
                    "tap shape [{}, {}, {}], expected [{c}, {c}, 1]",
                    t.c, t.h, t.w
                )));
            }
        }
        Ok(taps.into_iter().map(Dcdm::new).collect())
    }

    fn matrices(&self, img: &za::Rgb8In<'_>) -> Result<Vec<Dcdm>, Error> {
        self.check_size(img)?;
        let Backend::Trunk(t) = &self.backend else {
            return Err(Error::Engine("no trunk graph on this engine".into()));
        };
        self.check(t.run(rgb8_to_planar_01(img), img.height, img.width)?)
    }

    fn score_matrices(r: &[Dcdm], d: &[Dcdm]) -> f64 {
        let mut sum = 0.0f32;
        for (a, b) in r.iter().zip(d) {
            let cross: f64 = a
                .tap
                .data
                .iter()
                .zip(&b.tap.data)
                .map(|(&x, &y)| f64::from(x) * f64::from(y))
                .sum();
            let corr = (cross + C) / ((a.self_dot * b.self_dot).sqrt() + C);
            sum += corr as f32;
        }
        f64::from(1.0f32 - sum / r.len() as f32)
    }

    fn score_pair(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        self.check_size(r)?;
        if (r.width, r.height) != (d.width, d.height) {
            return Err(Error::SizeMismatch);
        }
        let (h, w) = (r.height, r.width);
        let (xr, xd) = (rgb8_to_planar_01(r), rgb8_to_planar_01(d));
        match &self.backend {
            Backend::Trunk(t) => {
                let (tr, td) = t.run_pair(xr, xd, h, w)?;
                Ok(Self::score_matrices(&self.check(tr)?, &self.check(td)?))
            }
            Backend::Pair(p) => p.score(xr, xd, h, w),
            Backend::Image(_) => Err(Error::Engine("image graph on a pair model".into())),
        }
    }
}

impl za::Metric for DeepDc {
    fn id(&self) -> za::MetricId {
        za::MetricId::new(COLUMN, za::implementation!(SCORE_REVISION)).with_reference(format!(
            "DeepDC_PyTorch 0.2, use_input_norm=True, no resize; engine {}",
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
                WarmState::Matrices(self.matrices(&view).metric_err_with(COLUMN, Error::kind)?)
            }
            Backend::Pair(_) => {
                self.check_size(&view)
                    .metric_err_with(COLUMN, Error::kind)?;
                WarmState::Pixels(rgb8_to_planar_01(&view))
            }
            Backend::Image(_) => {
                return Err(Error::Engine("image graph on a pair model".into()))
                    .metric_err_with(COLUMN, Error::kind);
            }
        };
        Ok(Box::new(WarmDeepDc {
            metric: self,
            reference,
            size: (view.width, view.height),
        }))
    }
}

/// A reference prepared once: its distance matrices on a trunk engine, the
/// converted input on a whole-graph engine (which recomputes its features).
struct WarmDeepDc<'m> {
    metric: &'m DeepDc,
    reference: WarmState,
    size: (usize, usize),
}

enum WarmState {
    Matrices(Vec<Dcdm>),
    Pixels(Vec<f32>),
}

impl za::WarmReference for WarmDeepDc<'_> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(COLUMN)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != self.size {
            return Err(Error::SizeMismatch).metric_err_with(COLUMN, Error::kind);
        }
        let s = match &self.reference {
            WarmState::Matrices(rm) => self
                .metric
                .matrices(&d)
                .map(|dm| DeepDc::score_matrices(rm, &dm)),
            WarmState::Pixels(px) => match &self.metric.backend {
                Backend::Pair(p) => p.score(px.clone(), rgb8_to_planar_01(&d), d.height, d.width),
                Backend::Trunk(_) | Backend::Image(_) => {
                    Err(Error::Engine("warm state does not match engine".into()))
                }
            },
        }
        .metric_err_with(COLUMN, Error::kind)?;
        Ok(Box::new(za::ScalarEval::single(COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        matches!(self.reference, WarmState::Matrices(_))
    }
}
