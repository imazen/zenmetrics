//! Bridge between [`crate::Metric`] / [`crate::NoReferenceMetric`] and
//! [`zenpsy_api`].
//!
//! Lets any metric implementing this crate's object-safe traits execute
//! natively through [`zenpsy_api::score`], [`zenpsy_api::score_image`],
//! [`zenpsy_api::score_planar`], [`zenpsy_api::prepare_reference`], and
//! `zenmetrics_orchestrator::psy::ContractExecutor` with zero per-metric
//! boilerplate, while preserving exact score bits, score-only no-map default,
//! opt-in native or named maps, and native result downcasting.

use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use zenpixels::{PixelBuf, PixelView, PlanarView, PlaneFormat, PlaneRole, PlaneView, SampleType};
use zenpsy_api::{
    Direction, ErrorKind, EvalOpts as PsyEvalOpts, Evaluation as PsyEvaluation, MapGrid,
    MapRequest as PsyMapRequest, MapView as PsyMapView, MetricError, MetricResult, Score,
    ScoreSpec as PsyScoreSpec, WarmReference as PsyWarmReference, fail,
};

use crate::eval::{Evaluation, ScoreKind, ScoreSpec};
use crate::input::Image;
use crate::map::MapPolarity;
use crate::metric::{EvalOpts, Metric, WarmReference};
use crate::no_reference::NoReferenceMetric;

/// Convert a [`ScoreSpec`] into its [`zenpsy_api::ScoreSpec`] equivalent.
pub fn score_spec_to_psy(spec: &ScoreSpec) -> PsyScoreSpec {
    let base = match spec.kind {
        ScoreKind::Similarity => PsyScoreSpec::new("", Direction::Higher).with_perfect(1.0),
        ScoreKind::Difference => PsyScoreSpec::new("", Direction::Lower).with_perfect(0.0),
        ScoreKind::Jod => PsyScoreSpec::new("", Direction::Higher)
            .with_unit("jod")
            .with_perfect(10.0),
        ScoreKind::Quality100 => PsyScoreSpec::new("", Direction::Higher).with_perfect(100.0),
        ScoreKind::Decibels => PsyScoreSpec::new("", Direction::Higher)
            .with_unit("dB")
            .with_perfect(f64::INFINITY),
        ScoreKind::Probability => PsyScoreSpec::new("", Direction::Lower).with_perfect(0.0),
        ScoreKind::Feature => PsyScoreSpec::new("", Direction::Unordered),
        ScoreKind::Quality => PsyScoreSpec::new("", Direction::Higher),
        ScoreKind::Distortion => PsyScoreSpec::new("", Direction::Lower),
    };
    base.with_key(spec.key.clone())
}

fn map_spec_to_psy(name: &'static str, polarity: MapPolarity) -> PsyScoreSpec {
    match polarity {
        MapPolarity::Similarity => PsyScoreSpec::new(name, Direction::Higher).with_perfect(1.0),
        MapPolarity::Difference => PsyScoreSpec::new(name, Direction::Lower).with_perfect(0.0),
        MapPolarity::DetectionProbability => {
            PsyScoreSpec::new(name, Direction::Lower).with_perfect(0.0)
        }
        MapPolarity::Feature => PsyScoreSpec::new(name, Direction::Unordered),
    }
}

struct PsyMap {
    spec: PsyScoreSpec,
    grid: MapGrid,
    pixels: PixelBuf,
}

/// [`zenpsy_api::Evaluation`] adapter over a boxed [`crate::Evaluation`].
///
/// Score-only evaluations allocate and retain zero diagnostic maps. When maps
/// are explicitly requested via [`zenpsy_api::EvalOpts::with_native_map`] or
/// [`zenpsy_api::EvalOpts::with_map`], the matching maps are exposed through
/// [`zenpsy_api::Evaluation::maps`].
pub struct PsyEval {
    inner: Box<dyn Evaluation>,
    scores: Vec<(PsyScoreSpec, f64)>,
    maps: Vec<PsyMap>,
}

impl PsyEval {
    /// Wrap a completed [`Evaluation`] according to the declared outputs and
    /// requested diagnostic options.
    #[track_caller]
    pub fn from_eval(
        family: &'static str,
        declared_outputs: &[ScoreSpec],
        inner: Box<dyn Evaluation>,
        opts: PsyEvalOpts<'_>,
    ) -> MetricResult<Self> {
        let mut scores = Vec::new();
        for (key, value) in inner.scores() {
            let spec = declared_outputs
                .iter()
                .find(|s| s.key == key)
                .map(score_spec_to_psy)
                .unwrap_or_else(|| {
                    PsyScoreSpec::new("", Direction::Unordered).with_key(key.clone())
                });
            scores.push((spec, value));
        }

        let mut maps = Vec::new();
        match opts.map_request() {
            PsyMapRequest::None => {}
            PsyMapRequest::Native => {
                for m in inner.maps() {
                    let g = m.geometry;
                    let grid = MapGrid::new(g.input_width, g.input_height)?.with_cells(
                        g.origin_x,
                        g.origin_y,
                        g.cell_width,
                        g.cell_height,
                    )?;
                    let buf = m.pixels.into_owned().map_err(|e| {
                        e.map_error(|b| {
                            MetricError::from_detail(ErrorKind::Resource, b).with_metric(family)
                        })
                    })?;
                    maps.push(PsyMap {
                        spec: map_spec_to_psy(m.name, m.polarity),
                        grid,
                        pixels: buf,
                    });
                }
            }
            PsyMapRequest::Named(wanted) => {
                let mut found = false;
                for m in inner.maps() {
                    if m.name != wanted {
                        continue;
                    }
                    found = true;
                    let g = m.geometry;
                    let grid = MapGrid::new(g.input_width, g.input_height)?.with_cells(
                        g.origin_x,
                        g.origin_y,
                        g.cell_width,
                        g.cell_height,
                    )?;
                    let buf = m.pixels.into_owned().map_err(|e| {
                        e.map_error(|b| {
                            MetricError::from_detail(ErrorKind::Resource, b).with_metric(family)
                        })
                    })?;
                    maps.push(PsyMap {
                        spec: map_spec_to_psy(m.name, m.polarity),
                        grid,
                        pixels: buf,
                    });
                }
                if !found {
                    return fail(
                        MetricError::new(ErrorKind::NoMap)
                            .with_metric(family)
                            .with_message(format!("map {wanted:?} not produced")),
                    );
                }
            }
            _ => {
                return fail(
                    MetricError::new(ErrorKind::UnsupportedRequest)
                        .with_metric(family)
                        .with_message("unsupported zenpsy map request"),
                );
            }
        }

        Ok(Self {
            inner,
            scores,
            maps,
        })
    }

    /// Borrow the underlying [`crate::Evaluation`].
    pub fn inner(&self) -> &dyn Evaluation {
        &*self.inner
    }

    /// Downcast to the concrete native result type (e.g. `GmsdResult`).
    pub fn downcast_native<T: 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }
}

impl PsyEvaluation for PsyEval {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        self.scores
            .iter()
            .map(|(spec, value)| Score::new(spec, *value))
    }

    fn maps(&self) -> impl Iterator<Item = PsyMapView<'_>> {
        self.maps.iter().map(|m| {
            let v = m.pixels.as_view();
            let plane = PlaneView::new(
                v.as_strided_bytes(),
                v.width(),
                v.height(),
                v.stride_bytes(),
                PlaneFormat::new(PlaneRole::Y, SampleType::F32),
            )
            .expect("validated GRAYF32_LINEAR map buffer");
            PsyMapView::new(&m.spec, plane, m.grid)
        })
    }
}

fn to_za_opts(opts: PsyEvalOpts<'_>) -> EvalOpts<'_> {
    let base = EvalOpts::SCORE_ONLY.with_pulse(opts.pulse());
    if opts.diagnostics() {
        base.with_native_map()
    } else {
        base
    }
}

/// Evaluate a packed [`PixelView`] pair through any [`crate::Metric`].
#[track_caller]
pub fn psy_evaluate<M: Metric + ?Sized>(
    metric: &mut M,
    reference: &PixelView<'_>,
    candidate: &PixelView<'_>,
    opts: PsyEvalOpts<'_>,
) -> MetricResult<PsyEval> {
    let family = metric.id().family();
    let outputs = metric.outputs();
    let eval = metric.evaluate(
        Image::Packed(reference.clone()),
        Image::Packed(candidate.clone()),
        to_za_opts(opts),
    )?;
    PsyEval::from_eval(family, &outputs, eval, opts)
}

/// Evaluate a [`PlanarView`] pair through any [`crate::Metric`].
#[track_caller]
pub fn psy_evaluate_planar<M: Metric + ?Sized>(
    metric: &mut M,
    reference: &PlanarView<'_>,
    candidate: &PlanarView<'_>,
    opts: PsyEvalOpts<'_>,
) -> MetricResult<PsyEval> {
    let family = metric.id().family();
    let outputs = metric.outputs();
    let eval = metric.evaluate(
        Image::Planar(reference.clone()),
        Image::Planar(candidate.clone()),
        to_za_opts(opts),
    )?;
    PsyEval::from_eval(family, &outputs, eval, opts)
}

/// Evaluate a single packed [`PixelView`] through any [`NoReferenceMetric`].
#[track_caller]
pub fn psy_evaluate_image<M: NoReferenceMetric + ?Sized>(
    metric: &mut M,
    image: &PixelView<'_>,
    opts: PsyEvalOpts<'_>,
) -> MetricResult<PsyEval> {
    let family = metric.id().family();
    let outputs = metric.outputs();
    let eval = metric.evaluate_image(Image::Packed(image.clone()), to_za_opts(opts))?;
    PsyEval::from_eval(family, &outputs, eval, opts)
}

/// [`zenpsy_api::WarmReference`] adapter over a boxed [`crate::WarmReference`]
/// that genuinely reuses reference work.
pub struct PsyWarm<'a> {
    family: &'static str,
    outputs: alloc::borrow::Cow<'static, [ScoreSpec]>,
    inner: Box<dyn WarmReference + 'a>,
}

/// Prepare a reusable reference through [`crate::Metric::warm_reference`].
///
/// Refuses metrics whose `warm_reference` falls back to cold replay
/// (`!warm.reuses_reference_work()`), upholding [`zenpsy_api::PrepareReference`].
#[track_caller]
pub fn psy_warm_reference<'a, M: Metric + ?Sized>(
    metric: &'a mut M,
    reference: &PixelView<'_>,
    opts: PsyEvalOpts<'_>,
) -> MetricResult<PsyWarm<'a>> {
    let family = metric.id().family();
    let outputs = metric.outputs();
    let inner = metric.warm_reference(
        Image::Packed(reference.clone()),
        EvalOpts::SCORE_ONLY.with_pulse(opts.pulse()),
    )?;
    if !inner.reuses_reference_work() {
        return fail(
            MetricError::new(ErrorKind::UnsupportedRequest)
                .with_metric(family)
                .with_message("metric does not genuinely reuse reference work"),
        );
    }
    Ok(PsyWarm {
        family,
        outputs,
        inner,
    })
}

impl PsyWarmReference for PsyWarm<'_> {
    type Output = PsyEval;

    fn evaluate(
        &mut self,
        candidate: &PixelView<'_>,
        opts: PsyEvalOpts<'_>,
    ) -> MetricResult<PsyEval> {
        let eval = self
            .inner
            .evaluate(Image::Packed(candidate.clone()), to_za_opts(opts))?;
        PsyEval::from_eval(self.family, &self.outputs, eval, opts)
    }
}

/// Implement [`zenpsy_api::Metric`] for a type that implements [`crate::Metric`].
#[macro_export]
macro_rules! impl_zenpsy_metric {
    ($ty:ty) => {
        impl $crate::zenpsy_api::Metric for $ty {
            type Output = $crate::PsyEval;

            fn id(&self) -> $crate::MetricId {
                <Self as $crate::Metric>::id(self)
            }

            fn evaluate(
                &mut self,
                reference: &$crate::zenpixels::PixelView<'_>,
                candidate: &$crate::zenpixels::PixelView<'_>,
                opts: $crate::zenpsy_api::EvalOpts<'_>,
            ) -> $crate::MetricResult<Self::Output> {
                $crate::psy_evaluate(self, reference, candidate, opts)
            }
        }
    };
}

/// Implement [`zenpsy_api::PrepareReference`] for a type that implements both
/// [`crate::Metric`] (with genuine reference reuse) and [`zenpsy_api::Metric`].
#[macro_export]
macro_rules! impl_zenpsy_prepare_reference {
    ($ty:ty) => {
        impl $crate::zenpsy_api::PrepareReference for $ty {
            type Prepared<'a>
                = $crate::PsyWarm<'a>
            where
                Self: 'a;

            fn warm_reference(
                &mut self,
                reference: &$crate::zenpixels::PixelView<'_>,
                opts: $crate::zenpsy_api::EvalOpts<'_>,
            ) -> $crate::MetricResult<Self::Prepared<'_>> {
                $crate::psy_warm_reference(self, reference, opts)
            }
        }
    };
}

/// Implement [`zenpsy_api::NoReferenceMetric`] for a type that implements
/// [`crate::NoReferenceMetric`].
#[macro_export]
macro_rules! impl_zenpsy_no_reference {
    ($ty:ty) => {
        impl $crate::zenpsy_api::NoReferenceMetric for $ty {
            type Output = $crate::PsyEval;

            fn id(&self) -> $crate::MetricId {
                <Self as $crate::NoReferenceMetric>::id(self)
            }

            fn evaluate_image(
                &mut self,
                image: &$crate::zenpixels::PixelView<'_>,
                opts: $crate::zenpsy_api::EvalOpts<'_>,
            ) -> $crate::MetricResult<Self::Output> {
                $crate::psy_evaluate_image(self, image, opts)
            }
        }
    };
}

/// Implement [`zenpsy_api::Metric`] and [`zenpsy_api::PlanarMetric`] for a
/// planar metric that implements [`crate::Metric`].
#[macro_export]
macro_rules! impl_zenpsy_planar_metric {
    ($ty:ty) => {
        $crate::impl_zenpsy_metric!($ty);

        impl $crate::zenpsy_api::PlanarMetric for $ty {
            fn evaluate_planar(
                &mut self,
                reference: &$crate::zenpixels::PlanarView<'_>,
                candidate: &$crate::zenpixels::PlanarView<'_>,
                opts: $crate::zenpsy_api::EvalOpts<'_>,
            ) -> $crate::MetricResult<Self::Output> {
                $crate::psy_evaluate_planar(self, reference, candidate, opts)
            }
        }
    };
}
