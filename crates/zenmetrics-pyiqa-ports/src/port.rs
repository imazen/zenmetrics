//! The `za::Metric` plumbing shared by the single-column pair ports
//! (wadiqam, ahiq, cw_ssim, dmm, pieapp): each implements [`Port`] (a warm
//! reference plus a score against it) and [`impl_metric!`] turns that into
//! `za::Metric` and `za::WarmReference`. The cold score is the warm path
//! unless a port overrides [`Port::score_cold`] with a computation that sums
//! identically, so warm and cold scores are bit-identical.

use zenmetrics_abstractions as za;
use zenmetrics_nn::Error;

use crate::engine::kind;

/// A full-reference port scoring one column.
pub(crate) trait Port: Send + Sync {
    /// What the reference keeps between calls.
    type Warm: Send;
    /// Score column.
    const COLUMN: &'static str;
    /// Score direction.
    const KIND: za::ScoreKind;
    /// Bump whenever any output can change.
    const REVISION: u32;
    /// Provenance text for `za::MetricId::with_reference`.
    fn reference(&self) -> String;
    /// Prepare a reference.
    fn warm(&self, r: &za::Rgb8In<'_>) -> Result<Self::Warm, Error>;
    /// Score a distorted image against a prepared reference.
    fn score_warm(&self, w: &Self::Warm, d: &za::Rgb8In<'_>) -> Result<f64, Error>;
    /// Score a pair without keeping a warm reference. Must equal
    /// `score_warm(&warm(r)?, d)` bit for bit; the default is exactly that.
    fn score_cold(&self, r: &za::Rgb8In<'_>, d: &za::Rgb8In<'_>) -> Result<f64, Error> {
        self.score_warm(&self.warm(r)?, d)
    }
    /// Reference size, for the mismatch check.
    fn warm_size(w: &Self::Warm) -> (usize, usize);
    /// Whether the warm reference saves work (false when it only copies).
    fn reuses_reference_work() -> bool {
        true
    }
}

/// Packed RGB8 geometry check shared by the ports: `stride ≥ 3w` and the
/// buffer covers `(h − 1)·stride + 3w` bytes.
pub(crate) fn check_buffer(img: &za::Rgb8In<'_>) -> Result<(), Error> {
    let need = img
        .height
        .checked_sub(1)
        .and_then(|n| n.checked_mul(img.stride))
        .and_then(|n| n.checked_add(img.width.checked_mul(3)?));
    if img.width == 0
        || img.height == 0
        || img.stride < img.width.saturating_mul(3)
        || need.is_none_or(|n| img.pixels.len() < n)
    {
        return Err(Error::BadModel(
            "pixel buffer shorter than its geometry".into(),
        ));
    }
    Ok(())
}

/// The warm reference of a [`Port`].
pub(crate) struct WarmPort<'m, P: Port> {
    pub(crate) metric: &'m P,
    pub(crate) warm: P::Warm,
}

impl<P: Port> za::WarmReference for WarmPort<'_, P> {
    fn evaluate(
        &mut self,
        d: za::Image<'_>,
        opts: za::EvalOpts<'_>,
    ) -> za::MetricResult<Box<dyn za::Evaluation>> {
        use za::ResultMetricExt as _;
        opts.checkpoint()?;
        opts.refuse_map(P::COLUMN)?;
        let d = za::rgb8_view(&d)?;
        if (d.width, d.height) != P::warm_size(&self.warm) {
            return Err(Error::SizeMismatch).metric_err_with(P::COLUMN, kind);
        }
        let s = self
            .metric
            .score_warm(&self.warm, &d)
            .metric_err_with(P::COLUMN, kind)?;
        Ok(Box::new(za::ScalarEval::single(P::COLUMN, s)))
    }

    fn reuses_reference_work(&self) -> bool {
        P::reuses_reference_work()
    }
}

/// `za::Metric` for a [`Port`] type.
macro_rules! impl_metric {
    ($t:ty) => {
        impl zenmetrics_abstractions::Metric for $t {
            fn id(&self) -> zenmetrics_abstractions::MetricId {
                use $crate::port::Port as _;
                zenmetrics_abstractions::MetricId::new(
                    <$t>::COLUMN,
                    zenmetrics_abstractions::implementation!(<$t>::REVISION),
                )
                .with_reference(self.reference())
            }

            fn outputs(&self) -> std::borrow::Cow<'static, [zenmetrics_abstractions::ScoreSpec]> {
                use $crate::port::Port as _;
                std::borrow::Cow::Owned(vec![zenmetrics_abstractions::ScoreSpec::new(
                    <$t>::COLUMN,
                    <$t>::KIND,
                )])
            }

            fn accepts(&self, format: &zenmetrics_abstractions::InputFormat) -> bool {
                zenmetrics_abstractions::is_rgb8_srgb(format)
            }

            fn evaluate(
                &mut self,
                r: zenmetrics_abstractions::Image<'_>,
                d: zenmetrics_abstractions::Image<'_>,
                opts: zenmetrics_abstractions::EvalOpts<'_>,
            ) -> zenmetrics_abstractions::MetricResult<Box<dyn zenmetrics_abstractions::Evaluation>>
            {
                use $crate::port::Port as _;
                use zenmetrics_abstractions::ResultMetricExt as _;
                opts.checkpoint()?;
                opts.refuse_map(<$t>::COLUMN)?;
                let (r, d) = zenmetrics_abstractions::rgb8_pair(&r, &d)?;
                let s = self
                    .score_cold(&r, &d)
                    .metric_err_with(<$t>::COLUMN, $crate::engine::kind)?;
                Ok(Box::new(zenmetrics_abstractions::ScalarEval::single(
                    <$t>::COLUMN,
                    s,
                )))
            }

            fn warm_reference<'m>(
                &'m mut self,
                r: zenmetrics_abstractions::Image<'_>,
                opts: zenmetrics_abstractions::EvalOpts<'_>,
            ) -> zenmetrics_abstractions::MetricResult<
                Box<dyn zenmetrics_abstractions::WarmReference + 'm>,
            > {
                use $crate::port::Port as _;
                use zenmetrics_abstractions::ResultMetricExt as _;
                opts.checkpoint()?;
                let view = zenmetrics_abstractions::rgb8_view(&r)?;
                let warm = self
                    .warm(&view)
                    .metric_err_with(<$t>::COLUMN, $crate::engine::kind)?;
                Ok(Box::new($crate::port::WarmPort {
                    metric: &*self,
                    warm,
                }))
            }
        }
    };
}
pub(crate) use impl_metric;
