//! End-to-end contract tests verifying real migrated `zenmetrics` metric
//! crates execute natively through `zenpsy_api` (`score`, `score_image`,
//! `score_planar`, `prepare_reference`, `PairedNoReference`) and
//! `zenmetrics_orchestrator::psy::ContractExecutor` on `zenpixels 0.5`
//! `PixelView` and `PlanarView`.

use cvvdp::{Cvvdp, CvvdpParams};
use gmsd::{Gmsd, GmsdDiffmap};
use piqe::Piqe;
use vmaf::{ModelVariant, Vmaf};
use zenmetrics_abstractions::{
    self as za, PsyEval,
    conformance::{distort_rgb8, textured_rgb8},
};
use zenmetrics_orchestrator::psy::{ContractExecutor, Fallback};
use zenpsy_api::{
    self as psy, ErrorKind, EvalOpts, Evaluation as _, Metric as _, PairedNoReference,
    zenpixels::{PixelView, PlanarView, PlaneFormat, PlaneRole, PlaneView, SampleType, formats},
};

fn strided_rgb8(packed: &[u8], w: usize, h: usize, pad_bytes: usize) -> (Vec<u8>, usize) {
    let stride = w * 3 + pad_bytes;
    let mut out = vec![0x5Au8; stride * h];
    for y in 0..h {
        out[y * stride..y * stride + w * 3].copy_from_slice(&packed[y * w * 3..(y + 1) * w * 3]);
    }
    (out, stride)
}

#[test]
fn gmsd_scores_maps_and_executor_on_pixel_view() {
    let (w, h) = (64usize, 48usize);
    let r_bytes = textured_rgb8(w, h, 11);
    let d_bytes = distort_rgb8(&r_bytes, w, h);
    let (r_pad, r_stride) = strided_rgb8(&r_bytes, w, h, 15);
    let (d_pad, d_stride) = strided_rgb8(&d_bytes, w, h, 9);

    let r_view = PixelView::new(&r_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();
    let d_view = PixelView::new(&d_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();
    let r_strided =
        PixelView::new(&r_pad, w as u32, h as u32, r_stride, &formats::RGB8_SRGB).unwrap();
    let d_strided =
        PixelView::new(&d_pad, w as u32, h as u32, d_stride, &formats::RGB8_SRGB).unwrap();

    // Direct score-only via zenpsy_api::score — zero maps retained, bit-exact vs raw gmsd_rgb8.
    let direct = gmsd::gmsd_rgb8(&r_bytes, &d_bytes, w, h, w * 3).unwrap();
    let mut metric = Gmsd;
    let eval = psy::score(&mut metric, &r_view, &d_view, EvalOpts::new()).unwrap();
    let scores: Vec<_> = eval.scores().collect();
    assert_eq!(scores[0].spec.key(), "gmsd");
    assert_eq!(scores[0].value.to_bits(), direct.gmsd.to_bits());
    assert_eq!(eval.maps().count(), 0, "score-only must retain zero maps");
    assert!(eval.downcast_native::<za::ScalarEval>().is_some());

    // Strided PixelView yields bit-identical scalar score.
    let strided_eval = psy::score(&mut metric, &r_strided, &d_strided, EvalOpts::new()).unwrap();
    assert_eq!(
        strided_eval.scores().next().unwrap().value.to_bits(),
        direct.gmsd.to_bits()
    );

    // Opt-in native and named maps on zenpsy_api::score preserve exact scalar bits.
    let with_native = psy::score(
        &mut metric,
        &r_view,
        &d_view,
        EvalOpts::new().with_native_map(),
    )
    .unwrap();
    assert_eq!(
        with_native.scores().next().unwrap().value.to_bits(),
        direct.gmsd.to_bits()
    );
    assert!(with_native.downcast_native::<GmsdDiffmap>().is_some());
    let maps: Vec<_> = with_native.maps().collect();
    assert!(!maps.is_empty());
    assert_eq!(maps[0].spec.key(), "gms");
    assert_eq!(maps[0].values.format().sample, SampleType::F32);
    assert_eq!(maps[0].grid.input_width(), w as u32);
    assert_eq!(maps[0].grid.input_height(), h as u32);

    let with_named = psy::score(
        &mut metric,
        &r_view,
        &d_view,
        EvalOpts::new().with_map("gms"),
    )
    .unwrap();
    assert_eq!(with_named.maps().count(), 1);
    assert_eq!(with_named.maps().next().unwrap().spec.key(), "gms");

    // Requesting a nonexistent named map must fail loudly with NoMap.
    let missing = psy::score(
        &mut metric,
        &r_view,
        &d_view,
        EvalOpts::new().with_map("does_not_exist"),
    )
    .err()
    .unwrap();
    assert_eq!(missing.error().kind(), ErrorKind::NoMap);

    // Execute through ContractExecutor (cold registration).
    let mut exec = ContractExecutor::default();
    exec.register("gmsd-cpu", || Ok(Gmsd));
    let report = exec
        .run(
            &r_view,
            &[d_view.clone(), d_strided],
            EvalOpts::new(),
            Fallback::Refuse,
        )
        .unwrap();
    assert!(report.is_complete());
    assert_eq!(report.completed.len(), 2);
    for c in &report.completed {
        assert_eq!(c.identity, Gmsd.id());
        assert_eq!(
            c.assessment.scores[0].value.to_bits(),
            direct.gmsd.to_bits()
        );
        let psy_eval = c.assessment.native::<PsyEval>().unwrap();
        assert!(psy_eval.downcast_native::<za::ScalarEval>().is_some());
    }

    // Routine ContractExecutor batches refuse diagnostic map requests.
    let diag_err = exec
        .run(
            &r_view,
            &[d_view],
            EvalOpts::new().with_native_map(),
            Fallback::Refuse,
        )
        .err()
        .unwrap();
    assert_eq!(diag_err.error().kind(), ErrorKind::UnsupportedRequest);
}

#[test]
fn piqe_scores_via_score_image_and_paired_no_reference() {
    let (w, h) = (48usize, 40usize);
    let r_bytes = textured_rgb8(w, h, 3);
    let d_bytes = distort_rgb8(&r_bytes, w, h);
    let r_view = PixelView::new(&r_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();
    let d_view = PixelView::new(&d_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();

    let mut nr = Piqe::new();
    let r_solo = psy::score_image(&mut nr, &r_view, EvalOpts::new()).unwrap();
    let d_solo = psy::score_image(&mut nr, &d_view, EvalOpts::new()).unwrap();
    let r_val = r_solo.scores().next().unwrap().value;
    let d_val = d_solo.scores().next().unwrap().value;

    // PairedNoReference caches the reference via prepare_reference and matches solo scores bit-for-bit.
    let mut paired = PairedNoReference::new(Piqe::new());
    let cold = psy::score(&mut paired, &r_view, &d_view, EvalOpts::new()).unwrap();
    let cold_scores: Vec<_> = cold
        .scores()
        .map(|s| (s.spec.key().to_owned(), s.value.to_bits()))
        .collect();
    assert_eq!(
        cold_scores,
        vec![
            ("piqe".to_owned(), d_val.to_bits()),
            ("piqe_ref".to_owned(), r_val.to_bits()),
        ]
    );

    let mut prepared = psy::prepare_reference(&mut paired, &r_view, EvalOpts::new()).unwrap();
    let warm = prepared.score(&d_view, EvalOpts::new()).unwrap();
    let warm_scores: Vec<_> = warm
        .scores()
        .map(|s| (s.spec.key().to_owned(), s.value.to_bits()))
        .collect();
    assert_eq!(warm_scores, cold_scores);

    // Run PairedNoReference through ContractExecutor::register_prepared.
    let mut exec = ContractExecutor::default();
    exec.register_prepared("piqe-paired", || Ok(PairedNoReference::new(Piqe::new())));
    let report = exec
        .run(&r_view, &[d_view], EvalOpts::new(), Fallback::Refuse)
        .unwrap();
    assert!(report.is_complete());
    assert!(report.attempts[0].prepared);
    assert_eq!(
        report.completed[0].assessment.scores[0].value.to_bits(),
        d_val.to_bits()
    );
    assert_eq!(
        report.completed[0].assessment.scores[1].value.to_bits(),
        r_val.to_bits()
    );
}

#[test]
fn cvvdp_prepared_reference_executes_through_contract_executor() {
    let (w, h) = (64usize, 64usize);
    let r_bytes = textured_rgb8(w, h, 21);
    let d1_bytes = distort_rgb8(&r_bytes, w, h);
    let d2_bytes = textured_rgb8(w, h, 29);

    let r_view = PixelView::new(&r_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();
    let d1_view =
        PixelView::new(&d1_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();
    let d2_view =
        PixelView::new(&d2_bytes, w as u32, h as u32, w * 3, &formats::RGB8_SRGB).unwrap();

    let mut cold_metric = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    let cold1 = psy::score(&mut cold_metric, &r_view, &d1_view, EvalOpts::new())
        .unwrap()
        .scores()
        .next()
        .unwrap()
        .value;
    let cold2 = psy::score(&mut cold_metric, &r_view, &d2_view, EvalOpts::new())
        .unwrap()
        .scores()
        .next()
        .unwrap()
        .value;

    let mut exec = ContractExecutor::default();
    exec.register_prepared("cvvdp-cpu", move || {
        Ok(Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap())
    });
    let report = exec
        .run(
            &r_view,
            &[d1_view, d2_view],
            EvalOpts::new(),
            Fallback::Refuse,
        )
        .unwrap();
    assert!(report.is_complete());
    assert!(report.attempts[0].prepared);
    assert_eq!(
        report.completed[0].assessment.scores[0].value.to_bits(),
        cold1.to_bits()
    );
    assert_eq!(
        report.completed[1].assessment.scores[0].value.to_bits(),
        cold2.to_bits()
    );
}

#[test]
fn vmaf_scores_planar_view_and_refuses_packed_pixel_view() {
    let (w, h) = (640usize, 360usize);
    let make_frame = |noise: u8| -> [Vec<u8>; 3] {
        let y = (0..w * h)
            .map(|i| {
                let (x, yy) = (i % w, i / w);
                ((x * 3 + yy * 2) % 220
                    + (i * 7) % 11
                    + usize::from(i % 5 == 0) * usize::from(noise))
                .min(255) as u8
            })
            .collect();
        let c = |k: usize| {
            (0..(w / 2) * (h / 2))
                .map(|i| (100 + (i * k) % 50) as u8)
                .collect()
        };
        [y, c(3), c(5)]
    };
    let r_yuv = make_frame(0);
    let d_yuv = make_frame(12);

    fn make_planar(p: &[Vec<u8>; 3], w: usize, h: usize) -> PlanarView<'_> {
        let py = PlaneView::new(
            &p[0],
            w as u32,
            h as u32,
            w,
            PlaneFormat::new(PlaneRole::Y, SampleType::U8),
        )
        .unwrap();
        let pu = PlaneView::new(
            &p[1],
            (w / 2) as u32,
            (h / 2) as u32,
            w / 2,
            PlaneFormat::new(PlaneRole::Cb, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        let pv = PlaneView::new(
            &p[2],
            (w / 2) as u32,
            (h / 2) as u32,
            w / 2,
            PlaneFormat::new(PlaneRole::Cr, SampleType::U8).with_subsampling(2, 2),
        )
        .unwrap();
        PlanarView::new(w as u32, h as u32, &[py, pu, pv]).unwrap()
    }

    let r_planar = make_planar(&r_yuv, w, h);
    let d_planar = make_planar(&d_yuv, w, h);

    let mut vmaf = Vmaf::new(ModelVariant::Standard1080p);

    // Packed RGB8 is explicitly refused with UnsupportedInput.
    let rgb = vec![128u8; 64 * 64 * 3];
    let packed = PixelView::new(&rgb, 64, 64, 64 * 3, &formats::RGB8_SRGB).unwrap();
    let err = psy::score(&mut vmaf, &packed, &packed, EvalOpts::new())
        .err()
        .unwrap();
    assert_eq!(err.error().kind(), ErrorKind::UnsupportedInput);

    // PlanarView executes natively through zenpsy_api::score_planar and matches za::Metric bit-for-bit.
    let psy_out = psy::score_planar(&mut vmaf, &r_planar, &d_planar, EvalOpts::new()).unwrap();
    let za_out = za::Metric::evaluate(
        &mut vmaf,
        za::Image::from(r_planar),
        za::Image::from(d_planar),
        za::EvalOpts::SCORE_ONLY,
    )
    .unwrap();
    assert_eq!(
        psy_out.scores().next().unwrap().value.to_bits(),
        za_out.score().to_bits()
    );
    assert_eq!(psy_out.maps().count(), 0);
}
