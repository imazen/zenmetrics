//! No-reference metrics: one image in, a quality estimate out.
//!
//! [`NoReferenceMetric`] is the single-image counterpart of [`Metric`].
//! [`PairedNoReference`] runs one wherever a pair [`Metric`] is expected
//! (batch scorers, the CLI, fleet executors): it scores the distorted image
//! under each declared key, and the reference under `<key>_ref`, so a sweep
//! row carries both and consumers can form the change themselves. A warm
//! reference scores the reference once.

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::format;
use alloc::vec::Vec;

use crate::{EvalOpts, Image};
use crate::{
    Evaluation, Extras, InputFormat, MapFold, MapView, Metric, MetricId, MetricResult, ScoreSpec,
    Scores, WarmReference, check_same_size, fail, unsupported,
};

/// A configured no-reference metric. Object-safe and `Send`, like
/// [`Metric`].
pub trait NoReferenceMetric: Send {
    /// Identity and provenance of this configured instance.
    fn id(&self) -> MetricId;

    /// Every scalar output, headline first. Stable for the instance. NR
    /// scores use [`crate::ScoreKind::Quality`] (higher is better) or
    /// [`crate::ScoreKind::Distortion`] (higher is worse).
    fn outputs(&self) -> Cow<'static, [ScoreSpec]>;

    /// Whether [`Self::evaluate_image`] takes inputs of this format
    /// natively. Conversion is the caller's job.
    fn accepts(&self, format: &InputFormat) -> bool;

    /// Score one image. Maps, when requested and available, describe it.
    ///
    /// # Errors
    /// Classified by [`crate::ErrorKind`]; a stopped pulse is
    /// [`crate::ErrorKind::Stopped`].
    fn evaluate_image(
        &mut self,
        image: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>>;
}

/// Suffix of the reference's copy of each output key.
pub const REFERENCE_SUFFIX: &str = "_ref";

/// A [`NoReferenceMetric`] as a pair [`Metric`]: outputs are the inner
/// outputs on the distorted image (headline first), then the same outputs on
/// the reference with keys suffixed [`REFERENCE_SUFFIX`]. Maps come from the
/// distorted image. Both images must be the same size, as for every pair
/// metric.
pub struct PairedNoReference<M> {
    inner: M,
}

impl<M: NoReferenceMetric> PairedNoReference<M> {
    /// Wrap `inner`.
    pub fn new(inner: M) -> Self {
        Self { inner }
    }

    /// The wrapped metric.
    pub fn inner(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Unwrap.
    pub fn into_inner(self) -> M {
        self.inner
    }

    /// Score the reference alone, scores only.
    fn reference_scores(
        &mut self,
        r: Image<'_>,
        opts: &EvalOpts<'_>,
    ) -> MetricResult<Vec<(Cow<'static, str>, f64)>> {
        let e = self
            .inner
            .evaluate_image(r, EvalOpts::new().with_pulse(opts.pulse))?;
        Ok(e.scores()
            .map(|(k, v)| (Cow::Owned(format!("{k}{REFERENCE_SUFFIX}")), v))
            .collect())
    }
}

impl<M: NoReferenceMetric> Metric for PairedNoReference<M> {
    fn id(&self) -> MetricId {
        self.inner.id()
    }

    fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
        let inner = self.inner.outputs();
        let refs = inner
            .iter()
            .map(|s| ScoreSpec::owned(format!("{}{REFERENCE_SUFFIX}", s.key), s.kind));
        let all: Vec<ScoreSpec> = inner.iter().cloned().chain(refs).collect();
        Cow::Owned(all)
    }

    fn accepts(&self, format: &InputFormat) -> bool {
        self.inner.accepts(format)
    }

    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>> {
        opts.checkpoint()?;
        check_same_size(&r, &d)?;
        let reference = self.reference_scores(r, &opts)?;
        let distorted = self.inner.evaluate_image(d, opts)?;
        Ok(Box::new(PairedEval {
            distorted,
            reference,
        }))
    }

    fn warm_reference<'m>(
        &'m mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        opts.checkpoint()?;
        if !self.accepts(&r.format()) {
            return fail(unsupported(&r.format(), "an accepted format"));
        }
        let reference = self.reference_scores(r.clone(), &opts)?;
        Ok(Box::new(WarmPaired {
            size: (r.width(), r.height()),
            reference,
            metric: self,
        }))
    }
}

struct WarmPaired<'m, M> {
    metric: &'m mut PairedNoReference<M>,
    reference: Vec<(Cow<'static, str>, f64)>,
    size: (u32, u32),
}

impl<M: NoReferenceMetric> WarmReference for WarmPaired<'_, M> {
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        opts.checkpoint()?;
        if (d.width(), d.height()) != self.size {
            return fail(crate::MetricError::new(crate::ErrorKind::DimensionMismatch));
        }
        let distorted = self.metric.inner.evaluate_image(d, opts)?;
        Ok(Box::new(PairedEval {
            distorted,
            reference: self.reference.clone(),
        }))
    }

    fn reuses_reference_work(&self) -> bool {
        true
    }
}

/// The result of [`PairedNoReference`]: the distorted image's evaluation
/// plus the reference's scores.
pub struct PairedEval {
    distorted: Box<dyn Evaluation>,
    reference: Vec<(Cow<'static, str>, f64)>,
}

impl PairedEval {
    /// The distorted image's own evaluation.
    pub fn distorted(&self) -> &dyn Evaluation {
        &*self.distorted
    }
}

impl Evaluation for PairedEval {
    fn score(&self) -> f64 {
        self.distorted.score()
    }

    fn scores(&self) -> Scores<'_> {
        Box::new(
            self.distorted
                .scores()
                .chain(self.reference.iter().cloned()),
        )
    }

    fn map(&self) -> Option<MapView<'_>> {
        self.distorted.map()
    }

    fn maps(&self) -> Vec<MapView<'_>> {
        self.distorted.maps()
    }

    fn map_score(&self, fold: MapFold<'_>) -> MetricResult<f64> {
        self.distorted.map_score(fold)
    }

    fn extras(&self) -> Extras<'_> {
        self.distorted.extras()
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::{assert_conforms, distort_rgb8, rgb8_image, scores, textured_rgb8};
    use crate::{ScalarEval, ScoreKind, rgb8_view};

    /// Toy NR metric: mean absolute horizontal gradient (higher = busier,
    /// read as "distortion") and its count of evaluations.
    struct Busy {
        calls: usize,
    }

    impl NoReferenceMetric for Busy {
        fn id(&self) -> MetricId {
            MetricId::new("busy", crate::implementation!(1))
        }
        fn outputs(&self) -> Cow<'static, [ScoreSpec]> {
            const O: &[ScoreSpec] = &[ScoreSpec::new("busy", ScoreKind::Distortion)];
            Cow::Borrowed(O)
        }
        fn accepts(&self, f: &InputFormat) -> bool {
            crate::is_rgb8_srgb(f)
        }
        fn evaluate_image(
            &mut self,
            image: Image<'_>,
            opts: EvalOpts<'_>,
        ) -> MetricResult<Box<dyn Evaluation>> {
            opts.checkpoint()?;
            opts.refuse_map("busy")?;
            self.calls += 1;
            let v = rgb8_view(&image)?;
            let (mut sum, mut n) = (0.0f64, 0usize);
            for y in 0..v.height {
                let row = &v.pixels[y * v.stride..y * v.stride + v.width * 3];
                for x in 3..v.width * 3 {
                    sum += f64::from(row[x].abs_diff(row[x - 3]));
                    n += 1;
                }
            }
            Ok(Box::new(ScalarEval::single("busy", sum / n.max(1) as f64)))
        }
    }

    #[test]
    fn paired_scores_distorted_then_reference() {
        let (w, h) = (48, 40);
        let a = textured_rgb8(w, h, 3);
        let b = distort_rgb8(&a, w, h);
        let (ra, rb) = (rgb8_image(&a, w, h), rgb8_image(&b, w, h));
        let mut m = PairedNoReference::new(Busy { calls: 0 });
        let keys: Vec<_> = m.outputs().iter().map(|s| s.key.clone()).collect();
        assert_eq!(keys, ["busy", "busy_ref"]);
        assert_eq!(m.outputs()[1].kind, ScoreKind::Distortion);

        let pair = scores(&*m.evaluate(ra.clone(), rb.clone(), EvalOpts::new()).unwrap());
        let mut solo = Busy { calls: 0 };
        let sd = solo
            .evaluate_image(rb.clone(), EvalOpts::new())
            .unwrap()
            .score();
        let sr = solo
            .evaluate_image(ra.clone(), EvalOpts::new())
            .unwrap()
            .score();
        assert_eq!(pair[0], (Cow::Borrowed("busy"), sd));
        assert_eq!(pair[1].0, "busy_ref");
        assert_eq!(pair[1].1.to_bits(), sr.to_bits());

        // The warm reference scores the reference once, then only candidates.
        let before = m.inner().calls;
        let mut warm = m.warm_reference(ra.clone(), EvalOpts::new()).unwrap();
        assert!(warm.reuses_reference_work());
        for _ in 0..3 {
            let s = scores(&*warm.evaluate(rb.clone(), EvalOpts::new()).unwrap());
            assert_eq!(s, pair, "warm must equal cold bit for bit");
        }
        drop(warm);
        assert_eq!(m.inner().calls - before, 1 + 3);
    }

    #[test]
    fn paired_conforms() {
        let (w, h) = (64, 48);
        let a = textured_rgb8(w, h, 5);
        let b = distort_rgb8(&a, w, h);
        let mut m = PairedNoReference::new(Busy { calls: 0 });
        assert_conforms(&mut m, &rgb8_image(&a, w, h), &rgb8_image(&b, w, h));
    }

    #[test]
    fn nr_kinds_have_directions_and_no_perfect_value() {
        assert_eq!(ScoreKind::Quality.higher_is_better(), Some(true));
        assert_eq!(ScoreKind::Distortion.higher_is_better(), Some(false));
        assert_eq!(ScoreKind::Quality.perfect(), None);
        assert_eq!(ScoreKind::Distortion.perfect(), None);
    }
}
