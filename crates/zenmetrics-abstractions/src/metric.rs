//! The metric surface: one-shot [`Metric::evaluate`], warm-reference
//! sessions ([`Metric::warm_reference`]) and temporal sequences
//! ([`Metric::sequence`]).

use alloc::borrow::Cow;
use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use how_far::{NoPulse, Pulse, Stop, StopReason};

use crate::error::{ErrorKind, MetricError, MetricResult, fail};
use crate::eval::{Evaluation, ScoreKind, ScoreSpec};
use crate::identity::MetricId;
use crate::input::{Image, InputFormat, OwnedImage, check_same_size};
use crate::map::MapRequest;

/// What one evaluation should compute, and how the caller can stop it.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct EvalOpts<'a> {
    /// `None` = scores only; `Some` = also produce maps.
    pub map: Option<MapRequest>,
    /// Cancellation and progress. Metrics call [`Self::checkpoint`] (or
    /// pass this to how-far stages) between phases. Default: `&NoPulse`.
    pub pulse: &'a dyn Pulse,
}

impl Default for EvalOpts<'_> {
    fn default() -> Self {
        Self::SCORE_ONLY
    }
}

impl fmt::Debug for EvalOpts<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalOpts")
            .field("map", &self.map)
            .finish_non_exhaustive()
    }
}

impl<'a> EvalOpts<'a> {
    /// Scores only, never stopped.
    pub const SCORE_ONLY: EvalOpts<'static> = EvalOpts {
        map: None,
        pulse: &NoPulse,
    };

    /// Same as [`Self::SCORE_ONLY`].
    pub const fn new() -> EvalOpts<'static> {
        Self::SCORE_ONLY
    }

    /// Request maps.
    #[must_use]
    pub const fn with_map(mut self, request: MapRequest) -> Self {
        self.map = Some(request);
        self
    }

    /// Request native maps.
    #[must_use]
    pub const fn with_native_map(self) -> Self {
        self.with_map(MapRequest::Native)
    }

    /// Scores only (drop any map request).
    #[must_use]
    pub const fn without_map(mut self) -> Self {
        self.map = None;
        self
    }

    /// Observe and stop through `pulse`.
    #[must_use]
    pub fn with_pulse<'b>(self, pulse: &'b dyn Pulse) -> EvalOpts<'b> {
        EvalOpts {
            map: self.map,
            pulse,
        }
    }

    /// The pulse as a [`Stop`], for metric internals that take one.
    pub fn stop(&self) -> &'a dyn Stop {
        self.pulse
    }

    /// Return [`ErrorKind::Stopped`] if the caller stopped.
    ///
    /// # Errors
    /// As described.
    #[track_caller]
    pub fn checkpoint(&self) -> MetricResult<()> {
        match self.pulse.check() {
            Ok(()) => Ok(()),
            Err(r) => fail(MetricError::stopped(r)),
        }
    }

    /// For metrics without maps: fail before any work when a map was
    /// requested.
    ///
    /// # Errors
    /// [`ErrorKind::NoMap`] with `family` stamped.
    #[track_caller]
    pub fn refuse_map(&self, family: &'static str) -> MetricResult<()> {
        match self.map {
            None => Ok(()),
            Some(_) => fail(MetricError::new(ErrorKind::NoMap).with_metric(family)),
        }
    }
}

/// Typed viewing conditions of a viewing-conditioned metric. Richer
/// settings (surround, observer age, EOTF) stay metric-specific and appear
/// in [`MetricId::config`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewingConditions {
    /// Pixels per visual degree.
    pub ppd: f32,
    /// Display peak luminance, cd/m², when the metric models display
    /// light.
    pub display_peak_nits: Option<f32>,
}

impl ViewingConditions {
    /// `ppd` and an optional peak; extend with the `with_*` builders.
    pub const fn new(ppd: f32, display_peak_nits: Option<f32>) -> Self {
        Self {
            ppd,
            display_peak_nits,
        }
    }

    /// Set the display peak.
    #[must_use]
    pub const fn with_display_peak_nits(mut self, nits: f32) -> Self {
        self.display_peak_nits = Some(nits);
        self
    }
}

/// How a metric relates successive frames.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Temporal {
    /// Each frame scores alone; a sequence is a loop over
    /// [`Metric::evaluate`].
    Stateless,
    /// Frame scores depend on neighboring frames.
    Stateful {
        /// Past frames each score depends on.
        history: u32,
        /// Most frames a result can trail its [`Sequence::push`].
        latency: u32,
    },
}

/// A frame rate as a rational.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrameRate {
    /// Frames …
    pub num: u32,
    /// … per this many seconds.
    pub den: u32,
}

/// What a caller knows about a sequence before it starts.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SequenceInfo {
    /// Frame rate, when known.
    pub frame_rate: Option<FrameRate>,
    /// Total frames, when known.
    pub frame_count: Option<u64>,
}

impl SequenceInfo {
    /// Nothing known.
    pub const fn new() -> Self {
        Self {
            frame_rate: None,
            frame_count: None,
        }
    }

    /// Set the frame rate.
    #[must_use]
    pub const fn with_frame_rate(mut self, num: u32, den: u32) -> Self {
        self.frame_rate = Some(FrameRate { num, den });
        self
    }

    /// Set the frame count.
    #[must_use]
    pub const fn with_frame_count(mut self, n: u64) -> Self {
        self.frame_count = Some(n);
        self
    }
}

/// One frame's result from a [`Sequence`].
#[non_exhaustive]
pub struct FrameEvaluation {
    /// Zero-based index of the pushed frame this describes.
    pub index: u64,
    /// Its result.
    pub evaluation: Box<dyn Evaluation>,
}

impl FrameEvaluation {
    /// Result `evaluation` for frame `index`.
    pub fn new(index: u64, evaluation: Box<dyn Evaluation>) -> Self {
        Self { index, evaluation }
    }
}

impl fmt::Debug for FrameEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameEvaluation")
            .field("index", &self.index)
            .field("score", &self.evaluation.score())
            .finish()
    }
}

/// What [`Sequence::finish`] returns.
#[non_exhaustive]
pub struct SequenceSummary {
    /// Frames whose results were still held back at finish, in order.
    pub frames: Vec<FrameEvaluation>,
    /// The metric's own pooled result over the sequence, when it defines
    /// one (VMAF's mean); `None` for stateless metrics, which leave
    /// pooling to the caller.
    pub pooled: Option<Box<dyn Evaluation>>,
    /// Frames pushed in total.
    pub frame_count: u64,
}

impl SequenceSummary {
    /// A summary.
    pub fn new(
        frames: Vec<FrameEvaluation>,
        pooled: Option<Box<dyn Evaluation>>,
        frame_count: u64,
    ) -> Self {
        Self {
            frames,
            pooled,
            frame_count,
        }
    }
}

/// A configured metric.
///
/// Object-safe and `Send`: a `Box<dyn Metric>` can move to a worker
/// thread. Methods take `&mut self` so scorers can keep scratch.
pub trait Metric: Send {
    /// Identity and provenance of this configured instance.
    fn id(&self) -> MetricId;

    /// Every scalar output, headline first. Stable for the instance.
    fn outputs(&self) -> Cow<'static, [ScoreSpec]>;

    /// The headline's meaning. Default: the first output's kind.
    fn score_kind(&self) -> ScoreKind {
        self.outputs()
            .first()
            .map_or(ScoreKind::Feature, |s| s.kind)
    }

    /// Fully qualified column names for [`Self::outputs`], in order.
    fn columns(&self) -> Vec<String> {
        let id = self.id();
        self.outputs().iter().map(|s| id.column(&s.key)).collect()
    }

    /// Typed viewing conditions, for viewing-conditioned metrics.
    fn viewing_conditions(&self) -> Option<ViewingConditions> {
        None
    }

    /// Whether [`Self::evaluate`] takes inputs of this format natively.
    /// Conversion is the caller's job (zenpixels-convert).
    fn accepts(&self, format: &InputFormat) -> bool;

    /// Compare a pair. `r` and `d` cover the same region at one size.
    ///
    /// # Errors
    /// Classified by [`ErrorKind`]; a stopped pulse is
    /// [`ErrorKind::Stopped`].
    fn evaluate(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn Evaluation>>;

    /// Prepare `r` once to score many candidates against it.
    ///
    /// Every candidate's result must be bit-identical to
    /// [`Self::evaluate`] on the same pair. The default copies `r` and
    /// replays `evaluate` ([`WarmReference::reuses_reference_work`] is
    /// `false`); metrics with reference-side work to hoist override it.
    /// Only `opts.pulse` applies here; map requests go to each candidate.
    ///
    /// # Errors
    /// As [`Self::evaluate`] for the reference alone.
    fn warm_reference<'m>(
        &'m mut self,
        r: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Box<dyn WarmReference + 'm>> {
        opts.checkpoint()?;
        if !self.accepts(&r.format()) {
            return fail(crate::input::unsupported(&r.format(), "an accepted format"));
        }
        Ok(Box::new(ColdReplay {
            metric: self,
            reference: r.to_owned_image(),
        }))
    }

    /// How this metric relates successive frames.
    fn temporal(&self) -> Temporal {
        Temporal::Stateless
    }

    /// Start a sequence. The default, for stateless metrics, evaluates each
    /// pushed pair at once and pools nothing.
    ///
    /// # Errors
    /// Metric-specific (an unsupported frame rate, say).
    fn sequence<'m>(&'m mut self, info: SequenceInfo) -> MetricResult<Box<dyn Sequence + 'm>> {
        Ok(Box::new(StatelessSequence {
            metric: self,
            info,
            pushed: 0,
        }))
    }
}

/// A prepared reference. See [`Metric::warm_reference`].
pub trait WarmReference: Send {
    /// Compare candidate `d` against the prepared reference.
    ///
    /// # Errors
    /// As [`Metric::evaluate`]; a size differing from the reference is
    /// [`ErrorKind::DimensionMismatch`].
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>>;

    /// `true` when reference-side work is actually reused; `false` for the
    /// cold replay.
    fn reuses_reference_work(&self) -> bool;
}

/// The default [`WarmReference`]: a copied reference and full replays.
pub struct ColdReplay<'m, M: ?Sized + Metric> {
    metric: &'m mut M,
    reference: OwnedImage,
}

impl<'m, M: ?Sized + Metric> ColdReplay<'m, M> {
    /// Replay `metric` against a copy of `reference`.
    pub fn new(metric: &'m mut M, reference: &Image<'_>) -> Self {
        Self {
            metric,
            reference: reference.to_owned_image(),
        }
    }
}

impl<M: ?Sized + Metric> WarmReference for ColdReplay<'_, M> {
    fn evaluate(&mut self, d: Image<'_>, opts: EvalOpts<'_>) -> MetricResult<Box<dyn Evaluation>> {
        let r = self.reference.as_image();
        check_same_size(&r, &d)?;
        self.metric.evaluate(r, d, opts)
    }

    fn reuses_reference_work(&self) -> bool {
        false
    }
}

/// A sequence session. See [`Metric::sequence`].
///
/// Results may trail their push by up to the metric's
/// [`Temporal::Stateful`] latency; every pushed frame's result appears
/// exactly once, from a `push` or from [`Self::finish`], in index order.
pub trait Sequence: Send {
    /// Push the next frame pair; returns the results that became ready.
    ///
    /// # Errors
    /// [`ErrorKind::DimensionMismatch`] when the size changes mid-sequence;
    /// otherwise as [`Metric::evaluate`]. After an error the session is
    /// unusable: later pushes fail with [`ErrorKind::SequenceOrder`].
    fn push(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Vec<FrameEvaluation>>;

    /// Flush held-back frames and pool.
    ///
    /// # Errors
    /// Metric-specific; [`ErrorKind::SequenceOrder`] after a failed push.
    fn finish(self: Box<Self>, opts: EvalOpts<'_>) -> MetricResult<SequenceSummary>;

    /// Frames pushed so far.
    fn frames_pushed(&self) -> u64;
}

struct StatelessSequence<'m, M: ?Sized + Metric> {
    metric: &'m mut M,
    #[allow(dead_code)]
    info: SequenceInfo,
    pushed: u64,
}

impl<M: ?Sized + Metric> Sequence for StatelessSequence<'_, M> {
    fn push(
        &mut self,
        r: Image<'_>,
        d: Image<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Vec<FrameEvaluation>> {
        let e = self.metric.evaluate(r, d, opts)?;
        let index = self.pushed;
        self.pushed += 1;
        Ok(alloc::vec![FrameEvaluation::new(index, e)])
    }

    fn finish(self: Box<Self>, opts: EvalOpts<'_>) -> MetricResult<SequenceSummary> {
        opts.checkpoint()?;
        Ok(SequenceSummary::new(Vec::new(), None, self.pushed))
    }

    fn frames_pushed(&self) -> u64 {
        self.pushed
    }
}

/// A [`Stop`] that always reports `reason` — for tests of a metric's
/// cancellation path.
#[derive(Clone, Copy, Debug)]
pub struct AlwaysStop(pub StopReason);

impl Stop for AlwaysStop {
    fn check(&self) -> Result<(), StopReason> {
        Err(self.0)
    }
}
