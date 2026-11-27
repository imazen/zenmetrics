//! Real map storage forwarding and a legacy PIQE bridge into the one-image API.
use iqa_contract_validation::{Assessment, Butter, Ssim2, SsimNative, Zen, ZenNative, fixture};
use zenmetrics_abstractions as legacy;
use zenpsy_api::{
    self as iqa,
    zenpixels::{
        AlphaMode, ColorSpace, PixelFormat, PixelView, RenderingIntent, SampleLayout, SampleType,
        formats,
    },
    *,
};

fn image(bytes: &[u8], w: u32, h: u32, padding: usize) -> PixelView<'_> {
    PixelView::new(bytes, w, h, w as usize * 3 + padding, &formats::RGB8_SRGB).unwrap()
}
fn pointer(map: MapView<'_>) -> *const f32 {
    assert_eq!(map.values.format().sample, SampleType::F32);
    map.values.data().as_ptr().cast::<f32>()
}
#[test]
fn three_native_results_expose_original_map_planes_through_stacked_erasure() {
    for (w, h) in [(5, 3), (97, 73), (130, 9)] {
        let rb = fixture(w, h, 9, false);
        let db = fixture(w, h, 15, true);
        let r = image(&rb, w, h, 9);
        let d = image(&db, w, h, 15);
        let mut methods: Vec<Box<DynMetric<'_>>> = vec![
            Box::new(Erased::new(Erased::new(Zen::default()))),
            Box::new(Erased::new(Ssim2)),
        ];
        // Butteraugli's native minimum geometry is retained.
        if w >= 8 && h >= 8 {
            methods.push(Box::new(Erased::new(Butter)));
        }
        for m in &mut methods {
            let scalar = iqa::score(m, &r, &d, EvalOpts::new()).unwrap();
            assert_eq!(scalar.maps().count(), 0);
            let mapped = iqa::score(m, &r, &d, EvalOpts::new().with_native_map()).unwrap();
            assert_eq!(
                mapped
                    .scores()
                    .map(|s| s.value.to_bits())
                    .collect::<Vec<_>>(),
                scalar
                    .scores()
                    .map(|s| s.value.to_bits())
                    .collect::<Vec<_>>()
            );
            let planes = mapped.maps().collect::<Vec<_>>();
            if let Some(native) = mapped.native::<Assessment<ZenNative, 1>>() {
                let ZenNative::Map(n) = &native.native else {
                    panic!("map request")
                };
                assert_eq!(planes.len(), 1);
                assert_eq!(pointer(planes[0]), n.diffmap().as_ptr());
            } else if let Some(native) = mapped.native::<Assessment<SsimNative, 1>>() {
                let SsimNative::Maps(n) = &native.native else {
                    panic!("map request")
                };
                assert_eq!(planes.len(), n.scales.len() * 9);
                let mut at = 0;
                for (s, level) in n.scales.iter().enumerate() {
                    for (term, data) in [
                        ("ssim", &level.ssim),
                        ("artifact", &level.artifact),
                        ("detail_lost", &level.detail_lost),
                    ] {
                        for (c, values) in ["x", "y", "b"].into_iter().zip(data) {
                            let m = planes[at];
                            at += 1;
                            assert_eq!(m.spec.key(), format!("{term}_{c}_{s}"));
                            assert_eq!(pointer(m), values.as_ptr());
                            assert_eq!(
                                (m.values.width() as usize, m.values.height() as usize),
                                (level.width, level.height)
                            );
                            assert_eq!(
                                m.grid,
                                MapGrid::new(w, h)
                                    .expect("native map geometry")
                                    .with_cells(0, 0, 1 << s, 1 << s)
                                    .expect("native map geometry")
                            );
                        }
                    }
                }
            } else {
                let n = &mapped
                    .native::<Assessment<butteraugli::ButteraugliResult, 2>>()
                    .unwrap()
                    .native;
                assert_eq!(planes.len(), 1);
                assert_eq!(
                    pointer(planes[0]),
                    n.diffmap.as_ref().unwrap().buf().as_ptr()
                );
            }
            for m in planes {
                assert_eq!((m.grid.input_width(), m.grid.input_height()), (w, h));
            }
        }
    }
}

struct Piqe(piqe::Piqe);
struct PiqeResult {
    native: Box<dyn legacy::Evaluation>,
}
impl Evaluation for PiqeResult {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        static SPEC: ScoreSpec = ScoreSpec::new("piqe", Direction::Lower);
        core::iter::once(Score::new(&SPEC, self.native.score()))
    }
}
impl NoReferenceMetric for Piqe {
    type Output = PiqeResult;
    fn id(&self) -> MetricId {
        MetricId::new("piqe", Implementation::new("piqe", "0.1.0", 1))
    }
    fn evaluate_image(
        &mut self,
        p: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<PiqeResult> {
        opts.checkpoint()?;
        if opts.diagnostics() {
            return fail(MetricError::new(ErrorKind::UnsupportedRequest));
        }
        let native = legacy::NoReferenceMetric::evaluate_image(
            &mut self.0,
            legacy::Image::from(p.clone()),
            legacy::EvalOpts::new().with_pulse(opts.pulse()),
        )
        .metric_err_with("piqe", |e| e.error().kind())?;
        Ok(PiqeResult { native })
    }
}
#[test]
fn real_piqe_one_image_paired_and_cached_results_keep_native_score_bits() {
    for (w, h) in [(64, 64), (97, 73), (131, 129)] {
        let rb = fixture(w, h, 9, false);
        let db = fixture(w, h, 15, true);
        let r = image(&rb, w, h, 9);
        let d = image(&db, w, h, 15);
        let direct = |p: &PixelView<'_>| {
            legacy::NoReferenceMetric::evaluate_image(
                &mut piqe::Piqe::new(),
                legacy::Image::from(p.clone()),
                legacy::EvalOpts::SCORE_ONLY,
            )
            .unwrap()
            .score()
        };
        let expected = [direct(&d), direct(&r)].map(f64::to_bits);
        let out = score_image(
            &mut Erased::new(Piqe(piqe::Piqe::new())),
            &d,
            EvalOpts::new(),
        )
        .unwrap();
        assert_eq!(
            out.native::<PiqeResult>().unwrap().native.score().to_bits(),
            expected[0]
        );
        // Also verify `piqe::Piqe` executes directly through `zenpsy_api::score_image`.
        let native_out = score_image(&mut piqe::Piqe::new(), &d, EvalOpts::new()).unwrap();
        assert_eq!(
            native_out.scores().next().unwrap().value.to_bits(),
            expected[0]
        );
        let mut paired = Erased::new(PairedNoReference::new(Piqe(piqe::Piqe::new())));
        let cold = iqa::score(&mut paired, &r, &d, EvalOpts::new()).unwrap();
        let mut prepared = paired.prepare(&r, EvalOpts::new()).unwrap();
        for out in [cold, prepared.evaluate(&d, EvalOpts::new()).unwrap()] {
            assert_eq!(
                out.scores().map(|s| s.value.to_bits()).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                out.scores().map(|s| s.spec.key()).collect::<Vec<_>>(),
                ["piqe", "piqe_ref"]
            );
            assert_eq!(out.maps().count(), 0);
            let n = out.native::<PairedEvaluation<PiqeResult>>().unwrap();
            assert_eq!(n.reference().native.score().to_bits(), expected[1]);
        }
    }
}

#[test]
fn piqe_native_refusals_keep_classification_and_original_cause() {
    let small = fixture(4, 4, 0, false);
    let unsupported = vec![0u8; 4 * 4 * 3 * 4];
    let image_float =
        PixelView::new(&unsupported, 4, 4, 4 * 3 * 4, &formats::RGBF32_LINEAR).unwrap();
    for (input, kind) in [
        (image(&small, 4, 4, 0), ErrorKind::TooSmall),
        (image_float, ErrorKind::UnsupportedInput),
    ] {
        let error = score_image(&mut Piqe(piqe::Piqe::new()), &input, EvalOpts::new())
            .err()
            .unwrap();
        assert_eq!(error.error().kind(), kind);
        assert!(std::error::Error::source(error.error()).is_some());
    }
}

#[test]
fn piqe_bridge_refuses_overriding_color_context_without_reinterpretation() {
    let bytes = fixture(64, 64, 0, false);
    // Minimal structurally valid 132-byte ICC Display/RGB header (tag_count = 0).
    let mut icc = vec![0u8; 132];
    icc[0..4].copy_from_slice(&132u32.to_be_bytes());
    icc[12..16].copy_from_slice(b"mntr");
    icc[16..20].copy_from_slice(b"RGB ");
    icc[20..24].copy_from_slice(b"XYZ ");
    icc[36..40].copy_from_slice(b"acsp");
    let icc_fmt = PixelFormat::new(
        SampleLayout::RGB8,
        ColorSpace::Icc {
            profile: std::sync::Arc::from(icc),
            color_channels: 3,
            alpha: AlphaMode::None,
            intent: RenderingIntent::RelativeColorimetric,
        },
    );
    let p = PixelView::new(&bytes, 64, 64, 64 * 3, &icc_fmt).unwrap();
    let error = score_image(&mut Piqe(piqe::Piqe::new()), &p, EvalOpts::new())
        .err()
        .unwrap();
    assert_eq!(error.error().kind(), ErrorKind::UnsupportedInput);
}
