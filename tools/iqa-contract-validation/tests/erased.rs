//! Heterogeneous native collections without the orchestrator.
use iqa_contract_validation::{Assessment, Butter, Ssim2, SsimNative, Zen, ZenNative, fixture};
use zenpsy_api::{
    self as iqa, DynMetric, DynPrepareReference, Erased, EvalOpts, Evaluation, MetricResult,
    zenpixels::{PixelDescriptor, PixelSlice},
};
fn image(bytes: &[u8], padding: usize) -> PixelSlice<'_> {
    PixelSlice::new(bytes, 97, 73, 97 * 3 + padding, PixelDescriptor::RGB8_SRGB).unwrap()
}
fn scores(e: &impl Evaluation) -> Vec<(String, u64)> {
    e.scores()
        .map(|s| (s.spec.key().to_string(), s.value.to_bits()))
        .collect()
}
#[test]
fn runs_all_native_metrics_in_a_heterogeneous_array_without_scheduling() {
    let r = fixture(97, 73, 9, false);
    let d = fixture(97, 73, 15, true);
    let r = image(&r, 9);
    let d = image(&d, 15);
    let expected = [
        scores(&iqa::score(&mut Butter, &r, &d, EvalOpts::new()).unwrap()),
        scores(&iqa::score(&mut Zen::default(), &r, &d, EvalOpts::new()).unwrap()),
        scores(&iqa::score(&mut Ssim2, &r, &d, EvalOpts::new()).unwrap()),
    ];
    let mut metrics: [Box<DynMetric<'_>>; 3] = [
        Box::new(Erased::new(Erased::new(Butter))),
        Box::new(Erased::new(Zen::default())),
        Box::new(Erased::new(Ssim2)),
    ];
    let mut families = Vec::new();
    for (i, metric) in metrics.iter_mut().enumerate() {
        families.push(metric.id().family());
        let result = iqa::score(metric, &r, &d, EvalOpts::new()).unwrap();
        assert_eq!(scores(&result), expected[i]);
        match i {
            0 => assert!(
                result
                    .native::<Assessment<butteraugli::ButteraugliResult, 2>>()
                    .unwrap()
                    .native
                    .diffmap
                    .is_none()
            ),
            1 => assert!(matches!(
                result.native::<Assessment<ZenNative, 1>>().unwrap().native,
                ZenNative::Scalar(_)
            )),
            2 => assert!(matches!(
                result.native::<Assessment<SsimNative, 1>>().unwrap().native,
                SsimNative::Scalar(_)
            )),
            _ => unreachable!(),
        }
    }
    assert_eq!(families, ["butteraugli", "zensim", "ssim2"]);
}
#[test]
fn prepares_all_native_metrics_and_retains_each_concrete_result() {
    let r = fixture(97, 73, 9, false);
    let d = fixture(97, 73, 15, true);
    let r = image(&r, 9);
    let d = image(&d, 15);
    let mut metrics: Vec<Box<dyn DynPrepareReference>> = vec![
        Box::new(Erased::new(Erased::new(Butter))),
        Box::new(Erased::new(Zen::default())),
        Box::new(Erased::new(Ssim2)),
    ];
    let expected = metrics
        .iter_mut()
        .map(|m| iqa::score(m, &r, &d, EvalOpts::new()).map(|e| scores(&e)))
        .collect::<MetricResult<Vec<_>>>()
        .unwrap();
    let mut prepared = metrics
        .iter_mut()
        .map(|m| m.prepare(&r, EvalOpts::new()))
        .collect::<MetricResult<Vec<_>>>()
        .unwrap();
    for _ in 0..2 {
        for (i, reference) in prepared.iter_mut().enumerate() {
            let result = reference.evaluate(&d, EvalOpts::new()).unwrap();
            assert_eq!(scores(&result), expected[i]);
            assert_eq!(
                result
                    .native::<Assessment<butteraugli::ButteraugliResult, 2>>()
                    .is_some(),
                i == 0
            );
            assert_eq!(
                result.native::<Assessment<ZenNative, 1>>().is_some(),
                i == 1
            );
            assert_eq!(
                result.native::<Assessment<SsimNative, 1>>().is_some(),
                i == 2
            );
        }
    }
}
