//! Native scalar/map parity for the standard planar ingress and its explicit
//! packed fallbacks. Diagnostic witnesses are bounded and remain in memory.
use iqa_contract_validation::{Butter, Ssim2, Zen};
use zenpsy_api::{
    zenpixels::{
        AlphaMode, ColorSpace, DiffuseWhite, PixelView, PlanarView, PlaneFormat, PlaneRole,
        PlaneView, Primaries, SampleStride, SampleType, SignalRange, Transfer, formats,
    },
    *,
};
fn color() -> ColorSpace {
    ColorSpace::Rgb {
        primaries: Primaries::Bt709,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: Some(DiffuseWhite::new(80.0).unwrap()),
    }
}
fn input(w: usize, h: usize, delta: f32) -> [Vec<f32>; 3] {
    core::array::from_fn(|c| {
        let stride = w + c + 3;
        let mut v = vec![f32::NAN; (h - 1) * stride + w];
        for y in 0..h {
            for x in 0..w {
                v[y * stride + x] = ((x * 31 + y * 47 + c * 17) % 211 + 13) as f32 / 255.0
                    + if (x + y + c) % 7 == 0 { delta } else { 0.0 }
            }
        }
        v
    })
}
fn views(v: &[Vec<f32>; 3], w: usize, h: usize) -> [PlaneView<'_>; 3] {
    let roles = [PlaneRole::R, PlaneRole::G, PlaneRole::B];
    core::array::from_fn(|c| {
        PlaneView::from_samples_strided(
            &v[c][..],
            w as u32,
            h as u32,
            SampleStride(w + c + 3),
            PlaneFormat::new(roles[c], SampleType::F32),
        )
        .unwrap()
    })
}
fn plane_bits(p: PlaneView<'_>) -> Vec<u32> {
    assert_eq!(p.format().sample, SampleType::F32);
    (0..p.height())
        .flat_map(|y| p.row_as::<f32>(y).unwrap().iter().map(|v| v.to_bits()))
        .collect()
}
fn scores(e: &impl Evaluation) -> Vec<(String, u64)> {
    e.scores()
        .map(|s| (s.spec.key().to_string(), s.value.to_bits()))
        .collect()
}
fn check<M: PreparePlanarReference>(mut metric: M, key: &str)
where
    M::Output: 'static,
{
    let (w, h) = (97, 73);
    let c = color();
    let a = input(w, h, 0.0);
    let b = input(w, h, 0.02);
    let ap = views(&a, w, h);
    let bp = views(&b, w, h);
    let r = PlanarView::new(w as u32, h as u32, &ap)
        .unwrap()
        .with_color(c.clone());
    let d = PlanarView::new(w as u32, h as u32, &bp)
        .unwrap()
        .with_color(c);
    // Construct the direct packed oracle independently of the adapter's
    // interleave helper, so channel/stride/transfer mistakes cannot self-agree.
    let packed = |v: &[Vec<f32>; 3]| -> Vec<[f32; 4]> {
        (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    [
                        v[0][y * (w + 3) + x],
                        v[1][y * (w + 4) + x],
                        v[2][y * (w + 5) + x],
                        1.0,
                    ]
                })
            })
            .collect()
    };
    let ar = packed(&a);
    let bd = packed(&b);
    fn packed_view(v: &[[f32; 4]], w: usize, h: usize) -> PixelView<'_> {
        PixelView::new(
            bytemuck::cast_slice(v),
            w as u32,
            h as u32,
            w * 16,
            &formats::RGBAF32_LINEAR,
        )
        .unwrap()
    }
    let (direct_scores, direct_map) = if metric.id().family() == "butteraugli" {
        let refs: [Vec<f32>; 3] = core::array::from_fn(|c| ar.iter().map(|p| p[c]).collect());
        let dsts: [Vec<f32>; 3] = core::array::from_fn(|c| bd.iter().map(|p| p[c]).collect());
        let direct = butteraugli::ButteraugliReference::new_linear_planar(
            &refs[0],
            &refs[1],
            &refs[2],
            w,
            h,
            w,
            butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
        )
        .unwrap()
        .compare_linear_planar(&dsts[0], &dsts[1], &dsts[2], w)
        .unwrap();
        (
            vec![
                ("max".into(), direct.score.to_bits()),
                ("pnorm3".into(), direct.pnorm_3.to_bits()),
            ],
            direct
                .diffmap
                .as_ref()
                .unwrap()
                .rows()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        )
    } else {
        let direct = metric
            .evaluate(
                &packed_view(&ar, w, h),
                &packed_view(&bd, w, h),
                EvalOpts::new().with_map(key),
            )
            .unwrap();
        let map = direct.maps().find(|m| m.spec.key() == key).unwrap();
        (scores(&direct), plane_bits(map.values))
    };
    let cold = score_planar(&mut metric, &r, &d, EvalOpts::new().with_map(key)).unwrap();
    let expected = scores(&cold);
    assert_eq!(expected, direct_scores);
    let mut warm = prepare_planar_reference(&mut metric, &r, EvalOpts::new()).unwrap();
    let output = warm.score(&d, EvalOpts::new().with_map(key)).unwrap();
    assert_eq!(scores(&output), expected);
    let cold_map = cold.maps().find(|m| m.spec.key() == key).unwrap();
    let wanted = plane_bits(cold_map.values);
    assert_eq!(wanted, direct_map);
    warm.with_evaluation(&d, EvalOpts::new().with_map(key), |view| {
        let mut got = Vec::new();
        view.visit_scores(|s| got.push((s.spec.key().to_string(), s.value.to_bits())));
        assert_eq!(got, expected);
        view.visit_maps(|m| {
            if m.spec.key() == key {
                assert_eq!(plane_bits(m.values), wanted);
            }
        });
        Ok(())
    })
    .unwrap();
    let changed = ColorSpace::Rgb {
        primaries: Primaries::Bt2020,
        transfer: Transfer::Linear,
        range: SignalRange::Full,
        alpha: AlphaMode::None,
        white: None,
    };
    let wrong = PlanarView::new(w as u32, h as u32, &bp)
        .unwrap()
        .with_color(changed);
    assert_eq!(
        warm.score(&wrong, EvalOpts::new())
            .err()
            .unwrap()
            .error()
            .kind(),
        ErrorKind::UnsupportedInput
    );
    let mut erased = Erased::new(Erased::new(metric));
    assert_eq!(
        scores(&score_planar(&mut erased, &r, &d, EvalOpts::new()).unwrap()),
        expected
    );
}
#[test]
fn butteraugli_borrows_native_planar_maps() {
    check(Butter, "diffmap");
}
#[test]
fn ssimulacra2_reuses_explicit_interleave_storage_and_preserves_components() {
    check(Ssim2, "ssim_y_0");
}
#[test]
fn zensim_reuses_explicit_interleave_storage_and_preserves_native_map() {
    check(Zen::default(), "diffmap");
}

#[test]
fn heterogeneous_native_planar_preparation_preserves_all_named_scores() {
    let (w, h) = (97, 73);
    let c = color();
    let input = input(w, h, 0.0);
    let planes = views(&input, w, h);
    let image = PlanarView::new(w as u32, h as u32, &planes)
        .unwrap()
        .with_color(c);
    let mut metrics: Vec<Box<dyn DynPreparePlanarReference>> = vec![
        Box::new(Erased::new(Butter)),
        Box::new(Erased::new(Ssim2)),
        Box::new(Erased::new(Zen::default())),
    ];
    for metric in &mut metrics {
        let cold = score_planar(metric, &image, &image, EvalOpts::new()).unwrap();
        let expected = scores(&cold);
        let mut prepared = metric.prepare_planar(&image, EvalOpts::new()).unwrap();
        let mut actual = Vec::new();
        prepared
            .with_evaluation(&image, EvalOpts::new(), &mut |view| {
                view.visit_scores(|s| actual.push((s.spec.key().to_string(), s.value.to_bits())))
            })
            .unwrap();
        assert_eq!(actual, expected);
    }
}
