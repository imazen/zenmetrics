//! Native scalar/map parity for the standard planar ingress and its explicit
//! packed fallbacks. Diagnostic witnesses are bounded and remain in memory.
use iqa_contract_validation::{Butter, Ssim2, Zen};
use zenpsy_api::{
    zenpixels::{hdr::DiffuseWhite, plane::*, *},
    *,
};
fn layout() -> PlanarLayout {
    PlanarLayout::new(
        [PlaneRole::Red, PlaneRole::Green, PlaneRole::Blue]
            .map(|r| PlaneSpec::new(r, PlaneEncoding::Float(FloatEncoding::NOMINAL)))
            .to_vec(),
        sample::ChromaSubsampling::Yuv444,
        sample::ChromaSiting::Unknown,
    )
    .unwrap()
}
fn color() -> ColorContext {
    ColorContext::from_cicp(Cicp::new(1, 8, 0, true)).with_diffuse_white(DiffuseWhite::new(80.0))
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
fn views(v: &[Vec<f32>; 3], w: usize, h: usize) -> [AnyPlane<'_>; 3] {
    core::array::from_fn(|c| AnyPlane::F32(PlaneRef::new(&v[c][..], w, h, w + c + 3).unwrap()))
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
    let l = layout();
    let c = color();
    let a = input(w, h, 0.0);
    let b = input(w, h, 0.02);
    let ap = views(&a, w, h);
    let bp = views(&b, w, h);
    let r = PlanarSlice::new(&ap, &l, w as u32, h as u32, &c, None).unwrap();
    let d = PlanarSlice::new(&bp, &l, w as u32, h as u32, &c, None).unwrap();
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
    fn packed_view(v: &[[f32; 4]], w: usize, h: usize) -> PixelSlice<'_> {
        PixelSlice::new(
            bytemuck::cast_slice(v),
            w as u32,
            h as u32,
            w * 16,
            PixelDescriptor::RGBAF32_LINEAR,
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
        let AnyPlane::F32(p) = map.values else {
            panic!("f32")
        };
        (
            scores(&direct),
            p.rows().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(),
        )
    };
    let cold = score_planar(&mut metric, &r, &d, EvalOpts::new().with_map(key)).unwrap();
    let expected = scores(&cold);
    assert_eq!(expected, direct_scores);
    let mut warm = prepare_planar_reference(&mut metric, &r, EvalOpts::new()).unwrap();
    let output = warm.score(&d, EvalOpts::new().with_map(key)).unwrap();
    assert_eq!(scores(&output), expected);
    let cold_map = cold.maps().find(|m| m.spec.key() == key).unwrap();
    let AnyPlane::F32(cold_plane) = cold_map.values else {
        panic!("f32")
    };
    let wanted: Vec<u32> = cold_plane.rows().flatten().map(|v| v.to_bits()).collect();
    assert_eq!(wanted, direct_map);
    warm.with_evaluation(&d, EvalOpts::new().with_map(key), |view| {
        let mut got = Vec::new();
        view.visit_scores(|s| got.push((s.spec.key().to_string(), s.value.to_bits())));
        assert_eq!(got, expected);
        view.visit_maps(|m| {
            if m.spec.key() == key {
                let AnyPlane::F32(p) = m.values else {
                    panic!("f32")
                };
                assert_eq!(
                    p.rows().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    wanted
                );
            }
        });
        Ok(())
    })
    .unwrap();
    let changed = ColorContext::from_cicp(Cicp::new(9, 8, 0, true));
    let wrong = PlanarSlice::new(&bp, &l, w as u32, h as u32, &changed, None).unwrap();
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
    let l = layout();
    let c = color();
    let input = input(w, h, 0.0);
    let planes = views(&input, w, h);
    let image = PlanarSlice::new(&planes, &l, w as u32, h as u32, &c, None).unwrap();
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
