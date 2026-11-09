use iqa_contract_validation::{Butter, Ssim2, Zen, fixture};
use std::{cell::Cell, rc::Rc};
use zenmetrics_orchestrator::psy::{ContractExecutor, Fallback};
use zenpsy_api::{
    self as iqa,
    zenpixels::{PixelDescriptor, PixelSlice},
    *,
};

fn image(bytes: &[u8], w: u32, h: u32, pad: usize) -> PixelSlice<'_> {
    PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * 3 + pad,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}
fn real<M: PrepareReference + 'static>(factory: fn() -> M)
where
    M::Output: 'static,
{
    let (w, h) = (97, 73);
    let r = fixture(w, h, 9, false);
    let d = fixture(w, h, 15, true);
    let r = image(&r, w, h, 9);
    let d = image(&d, w, h, 15);
    let expected = iqa::score(&mut factory(), &r, &d, EvalOpts::new())
        .ok()
        .unwrap();
    let mut executor = ContractExecutor::default();
    executor.register_prepared("native", move || Ok(factory()));
    let out = executor
        .run(&r, &[d.clone(), d], EvalOpts::new(), Fallback::Refuse)
        .ok()
        .unwrap();
    assert!(out.is_complete());
    assert_eq!(out.attempts.len(), 1);
    assert!(out.attempts[0].prepared);
    for (i, result) in out.completed.iter().enumerate() {
        assert_eq!(result.index, i);
        assert_eq!(result.identity, factory().id());
        assert!(result.assessment.native::<M::Output>().is_some());
        assert_eq!(
            result
                .assessment
                .scores
                .iter()
                .map(|s| (s.spec.clone(), s.value.to_bits()))
                .collect::<Vec<_>>(),
            expected
                .scores()
                .map(|s| (s.spec.clone(), s.value.to_bits()))
                .collect::<Vec<_>>()
        );
    }
}
#[test]
fn orchestrates_three_native_prepared_implementations_with_all_outputs_and_provenance() {
    real(|| Butter);
    real(|| Ssim2);
    real(Zen::default);
}

#[derive(Debug)]
struct NativeCause;
impl std::fmt::Display for NativeCause {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("native sentinel")
    }
}
impl std::error::Error for NativeCause {}
struct ResultData {
    value: f64,
}
impl Evaluation for ResultData {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        static SPEC: ScoreSpec = ScoreSpec::new("native", Direction::Lower);
        core::iter::once(Score::new(&SPEC, self.value))
    }
}
struct Control {
    calls: usize,
    fail: Option<(usize, ErrorKind)>,
    prepared: Rc<Cell<usize>>,
    backend: Backend,
    config: &'static str,
}
impl Metric for Control {
    type Output = ResultData;
    fn id(&self) -> MetricId {
        MetricId::new(
            "control",
            Implementation::new("control", "1", 1).with_backend(self.backend),
        )
        .with_config("mode", self.config)
    }
    fn evaluate(
        &mut self,
        _: &PixelSlice<'_>,
        d: &PixelSlice<'_>,
        _: EvalOpts<'_>,
    ) -> MetricResult<ResultData> {
        self.calls += 1;
        if let Some((at, kind)) = self.fail
            && self.calls == at
        {
            return fail(MetricError::from_detail(kind, NativeCause));
        }
        Ok(ResultData {
            value: d.row(0)[0] as f64,
        })
    }
}
impl PrepareReference for Control {
    type Prepared<'a> = &'a mut Self;
    fn warm_reference(&mut self, _: &PixelSlice<'_>, _: EvalOpts<'_>) -> MetricResult<&mut Self> {
        self.prepared.set(self.prepared.get() + 1);
        Ok(self)
    }
}
impl WarmReference for &mut Control {
    type Output = ResultData;
    fn evaluate(&mut self, d: &PixelSlice<'_>, opts: EvalOpts<'_>) -> MetricResult<ResultData> {
        Metric::evaluate(*self, d, d, opts)
    }
}
fn control(
    fail: Option<(usize, ErrorKind)>,
    prepared: Rc<Cell<usize>>,
    backend: Backend,
) -> Control {
    Control {
        calls: 0,
        fail,
        prepared,
        backend,
        config: "same",
    }
}
fn inputs() -> (PixelSlice<'static>, Vec<PixelSlice<'static>>) {
    (
        image(&[0; 12], 2, 2, 0),
        vec![
            image(&[11; 12], 2, 2, 0),
            image(&[22; 12], 2, 2, 0),
            image(&[33; 12], 2, 2, 0),
        ],
    )
}
#[test]
fn resource_fallback_keeps_completed_prefix_and_real_preparation() {
    let counter = Rc::new(Cell::new(0));
    let c = counter.clone();
    let mut executor = ContractExecutor::default();
    executor.register("first", move || {
        Ok(control(
            Some((2, ErrorKind::Resource)),
            c.clone(),
            Backend::Cpu,
        ))
    });
    let c = counter.clone();
    executor.register_prepared("second", move || {
        Ok(control(None, c.clone(), Backend::gpu("control")))
    });
    let (r, ds) = inputs();
    let report = executor
        .run(&r, &ds, EvalOpts::new(), Fallback::Resource)
        .ok()
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(counter.get(), 1);
    assert_eq!(report.attempts.len(), 2);
    assert_eq!(
        report
            .completed
            .iter()
            .map(|r| (r.index, r.assessment.scores[0].value))
            .collect::<Vec<_>>(),
        [(0, 11.0), (1, 22.0), (2, 33.0)]
    );
    assert_eq!(
        report.completed[0].identity.implementation().backend(),
        Backend::Cpu
    );
    assert_eq!(
        report.completed[1].identity.implementation().backend(),
        Backend::gpu("control")
    );
    let error = report.attempts[0].error.as_ref().unwrap();
    assert_eq!(error.error().kind(), ErrorKind::Resource);
    assert!(
        std::error::Error::source(error.error())
            .unwrap()
            .downcast_ref::<NativeCause>()
            .is_some()
    );
    assert!(!error.is_empty());
}
#[test]
fn cancellation_backend_errors_and_default_policy_never_trigger_resource_fallback() {
    for (kind, policy) in [
        (
            ErrorKind::Stopped(how_far::StopReason::TimedOut),
            Fallback::Resource,
        ),
        (ErrorKind::Backend, Fallback::Resource),
        (ErrorKind::UnsupportedInput, Fallback::Resource),
        (ErrorKind::Resource, Fallback::Refuse),
    ] {
        let built = Rc::new(Cell::new(0));
        let mut executor = ContractExecutor::default();
        executor.register("first", move || {
            Ok(control(
                Some((2, kind)),
                Rc::new(Cell::new(0)),
                Backend::Cpu,
            ))
        });
        let c = built.clone();
        executor.register("must not construct", move || {
            c.set(c.get() + 1);
            Ok(control(None, Rc::new(Cell::new(0)), Backend::Cpu))
        });
        let (r, ds) = inputs();
        let report = executor.run(&r, &ds, EvalOpts::new(), policy).ok().unwrap();
        assert!(!report.is_complete());
        assert_eq!(report.completed.len(), 1);
        assert_eq!(report.attempts.len(), 1);
        assert_eq!(built.get(), 0);
        assert_eq!(
            report.attempts[0].error.as_ref().unwrap().error().kind(),
            kind
        );
    }
}
#[test]
fn fallback_cannot_change_the_effective_scoring_configuration() {
    let mut executor = ContractExecutor::default();
    executor.register("first", || {
        Ok(control(
            Some((1, ErrorKind::Resource)),
            Rc::new(Cell::new(0)),
            Backend::Cpu,
        ))
    });
    executor.register("different model", || {
        let mut m = control(None, Rc::new(Cell::new(0)), Backend::Cpu);
        m.config = "different";
        Ok(m)
    });
    let (r, ds) = inputs();
    let report = executor
        .run(&r, &ds, EvalOpts::new(), Fallback::Resource)
        .ok()
        .unwrap();
    assert!(!report.is_complete());
    assert!(report.completed.is_empty());
    assert_eq!(
        report.attempts[1].error.as_ref().unwrap().error().kind(),
        ErrorKind::InvalidConfig
    );
}
#[test]
fn routine_batch_refuses_diagnostic_arrays_before_construction() {
    let built = Rc::new(Cell::new(0));
    let c = built.clone();
    let mut executor = ContractExecutor::default();
    executor.register("native", move || {
        c.set(c.get() + 1);
        Ok(Butter)
    });
    let (r, ds) = inputs();
    let error = executor
        .run(&r, &ds, EvalOpts::new().with_native_map(), Fallback::Refuse)
        .err()
        .unwrap();
    assert_eq!(error.error().kind(), ErrorKind::UnsupportedRequest);
    assert_eq!(built.get(), 0);
}
