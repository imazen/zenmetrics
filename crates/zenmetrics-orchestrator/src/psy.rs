//! Shared erased wrappers normalize results; orchestration owns factories and
//! fallback policy. Factories run on
//! the calling worker, so even thread-local native scorers are supported.
//!
//! Register caller-selected implementations in preference order. A batch shares
//! one reference; prepared registrations actually precompute it once per attempt.
//! Resource fallback is explicit, preserves completed results and records the
//! actual implementation of every result. Cancellation and computation failures
//! stop the batch. Existing GPU admission/benchmark scheduling remains separate.
use iqa::{whereat::At, zenpixels::PixelSlice};
use zenpsy_api::{
    self as iqa, ErrorKind, EvalOpts, Evaluation, Metric, MetricError, MetricId, MetricResult,
    PrepareReference, ScoreSpec,
};

/// An owned score view crossing the orchestration boundary.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct OutputScore {
    /// Native output semantics.
    pub spec: ScoreSpec,
    /// Unmodified native value.
    pub value: f64,
}
/// Runtime result retaining its complete native assessment.
#[non_exhaustive]
pub struct Assessment {
    /// All named scalar outputs, in native order.
    pub scores: Vec<OutputScore>,
    native: iqa::ErasedEvaluation,
}
impl Assessment {
    /// Recover the concrete native assessment, when the caller knows its type.
    pub fn native<T: 'static>(&self) -> Option<&T> {
        self.native.native()
    }
}
/// One completed candidate, bound to its actual successful implementation.
#[non_exhaustive]
pub struct Completed {
    /// Index into the submitted candidate slice.
    pub index: usize,
    /// Effective configuration and actual implementation provenance.
    pub identity: MetricId,
    /// Structured result and common named-score view.
    pub assessment: Assessment,
}
/// One attempted registered implementation, including its original failure.
#[non_exhaustive]
pub struct Attempt {
    /// Caller-selected registration name.
    pub registration: String,
    /// Actual identity, absent when construction failed.
    pub identity: Option<MetricId>,
    /// Whether this registration uses native prepared-reference work.
    pub prepared: bool,
    /// Original located error and cause, if the attempt failed.
    pub error: Option<At<MetricError>>,
}
/// Completed prefix and attempt history. On failure, pending candidates have no
/// invented scores; inspect the final attempt's error.
#[non_exhaustive]
pub struct BatchReport {
    /// Successfully completed candidate results, in submission order.
    pub completed: Vec<Completed>,
    /// Every attempted factory, including construction/preparation failures.
    pub attempts: Vec<Attempt>,
    /// Number of submitted candidates.
    pub requested: usize,
}
impl BatchReport {
    /// Whether every submitted candidate completed successfully.
    pub fn is_complete(&self) -> bool {
        self.completed.len() == self.requested
    }
}
/// Caller policy for retrying pending work on another registered implementation.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub enum Fallback {
    /// Use only the first selected implementation.
    #[default]
    Refuse,
    /// On classified resource exhaustion, try the next caller-authorized
    /// implementation. This does not assert CPU/GPU numerical equivalence.
    Resource,
}

struct Partial {
    outputs: Vec<Assessment>,
    error: Option<At<MetricError>>,
}
trait Driver {
    fn id(&self) -> MetricId;
    fn run(
        &mut self,
        reference: &PixelSlice<'_>,
        candidates: &[PixelSlice<'_>],
        opts: EvalOpts<'_>,
    ) -> Partial;
}
fn erase(native: iqa::ErasedEvaluation) -> Assessment {
    let scores = native
        .scores()
        .map(|s| OutputScore {
            spec: s.spec.clone(),
            value: s.value,
        })
        .collect();
    Assessment { scores, native }
}
fn collect(calls: impl Iterator<Item = MetricResult<iqa::ErasedEvaluation>>) -> Partial {
    let mut outputs = Vec::new();
    for result in calls {
        match result {
            Ok(native) => outputs.push(erase(native)),
            Err(error) => {
                return Partial {
                    outputs,
                    error: Some(error),
                };
            }
        }
    }
    Partial {
        outputs,
        error: None,
    }
}
struct Cold(Box<iqa::DynMetric<'static>>);
impl Driver for Cold {
    fn id(&self) -> MetricId {
        self.0.id()
    }
    fn run(&mut self, r: &PixelSlice<'_>, ds: &[PixelSlice<'_>], opts: EvalOpts<'_>) -> Partial {
        collect(ds.iter().map(|d| iqa::score(&mut self.0, r, d, opts)))
    }
}
struct Prepared(Box<dyn iqa::DynPrepareReference>);
impl Driver for Prepared {
    fn id(&self) -> MetricId {
        self.0.id()
    }
    fn run(&mut self, r: &PixelSlice<'_>, ds: &[PixelSlice<'_>], opts: EvalOpts<'_>) -> Partial {
        match self.0.prepare(r, opts) {
            Ok(mut warm) => collect(ds.iter().map(|d| warm.evaluate(d, opts))),
            Err(error) => Partial {
                outputs: Vec::new(),
                error: Some(error),
            },
        }
    }
}
struct Registration {
    name: String,
    prepared: bool,
    factory: Box<dyn FnMut() -> MetricResult<Box<dyn Driver>>>,
}
/// Worker-local backend-neutral batch executor. The surrounding caller controls
/// placement, CPU/RAM admission and implementation order. No hardware heuristics
/// or global metric enum is imposed on independent metric implementations.
#[derive(Default)]
pub struct ContractExecutor {
    registrations: Vec<Registration>,
}
impl ContractExecutor {
    /// Register a cold implementation. Its factory runs on the executing worker.
    pub fn register<M, F>(&mut self, name: impl Into<String>, mut factory: F)
    where
        M: Metric + 'static,
        M::Output: 'static,
        F: FnMut() -> MetricResult<M> + 'static,
    {
        self.registrations.push(Registration {
            name: name.into(),
            prepared: false,
            factory: Box::new(move || {
                factory().map(|m| Box::new(Cold(Box::new(iqa::Erased::new(m)))) as Box<dyn Driver>)
            }),
        });
    }
    /// Register actual native reference reuse. One preparation serves the
    /// remaining candidate batch; no synthetic warm path is installed.
    pub fn register_prepared<M, F>(&mut self, name: impl Into<String>, mut factory: F)
    where
        M: PrepareReference + 'static,
        M::Output: 'static,
        F: FnMut() -> MetricResult<M> + 'static,
    {
        self.registrations.push(Registration {
            name: name.into(),
            prepared: true,
            factory: Box::new(move || {
                factory()
                    .map(|m| Box::new(Prepared(Box::new(iqa::Erased::new(m)))) as Box<dyn Driver>)
            }),
        });
    }
    /// Run score-only work. Completed candidates survive resource fallback and
    /// are never replayed. Diagnostics use the typed metric API with explicitly
    /// bounded witnesses and storage, rather than the routine batch route.
    /// A missing registration or diagnostic request is an immediate error;
    /// attempted native failures are retained in the returned report.
    #[track_caller]
    pub fn run(
        &mut self,
        r: &PixelSlice<'_>,
        ds: &[PixelSlice<'_>],
        opts: EvalOpts<'_>,
        fallback: Fallback,
    ) -> MetricResult<BatchReport> {
        opts.checkpoint()?;
        if opts.diagnostics() {
            return iqa::fail(
                MetricError::new(ErrorKind::UnsupportedRequest)
                    .with_message("routine batches are score-only"),
            );
        }
        if self.registrations.is_empty() {
            return iqa::fail(
                MetricError::new(ErrorKind::InvalidConfig)
                    .with_message("no metric implementation registered"),
            );
        }
        let mut report = BatchReport {
            completed: Vec::new(),
            attempts: Vec::new(),
            requested: ds.len(),
        };
        if ds.is_empty() {
            return Ok(report);
        }
        let mut scoring_identity: Option<MetricId> = None;
        for registration in &mut self.registrations {
            // A cancelled batch must never construct the next backend.
            if let Err(error) = opts.checkpoint() {
                report.attempts.push(Attempt {
                    registration: registration.name.clone(),
                    identity: None,
                    prepared: registration.prepared,
                    error: Some(error),
                });
                break;
            }
            let (identity, partial) = match (registration.factory)() {
                Ok(mut driver) => {
                    let id = driver.id();
                    if scoring_identity.as_ref().is_some_and(|previous| {
                        previous.family() != id.family()
                            || previous.variant() != id.variant()
                            || previous.reference() != id.reference()
                            || previous.config() != id.config()
                    }) {
                        let error = iqa::whereat::at(
                            MetricError::new(ErrorKind::InvalidConfig)
                                .with_message("fallback changed the scoring configuration"),
                        );
                        (
                            Some(id),
                            Partial {
                                outputs: Vec::new(),
                                error: Some(error),
                            },
                        )
                    } else {
                        scoring_identity = Some(id.clone());
                        (Some(id), driver.run(r, &ds[report.completed.len()..], opts))
                    }
                }
                Err(error) => (
                    None,
                    Partial {
                        outputs: Vec::new(),
                        error: Some(error),
                    },
                ),
            };
            let offset = report.completed.len();
            for (index, assessment) in partial.outputs.into_iter().enumerate() {
                report.completed.push(Completed {
                    index: offset + index,
                    identity: identity.clone().expect("constructed scorer identity"),
                    assessment,
                });
            }
            let retry = matches!(fallback, Fallback::Resource)
                && partial
                    .error
                    .as_ref()
                    .is_some_and(|e| e.error().kind() == ErrorKind::Resource);
            report.attempts.push(Attempt {
                registration: registration.name.clone(),
                identity,
                prepared: registration.prepared,
                error: partial.error,
            });
            if report.is_complete() || !retry {
                break;
            }
        }
        Ok(report)
    }
}
