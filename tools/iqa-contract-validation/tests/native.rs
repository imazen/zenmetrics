use iqa::zenpixels::{PixelDescriptor, PixelSlice};
use iqa::{ErrorKind, EvalOpts, Image, Metric};
use iqa_contract_validation::{Butter, Ssim2, Zen, fixture};
use zenpsychovisual_api as iqa;
fn image(bytes: &[u8], w: u32, h: u32, padding: usize) -> Image<'_> {
    PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * 3 + padding,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
    .into()
}
fn bits(e: &iqa::Evaluated) -> Vec<u64> {
    e.result.scores().map(|(_, v)| v.to_bits()).collect()
}
#[test]
fn three_native_implementations_conform_and_preserve_strided_cold_warm_bits() {
    for (w, h) in [(64, 64), (97, 73), (131, 129)] {
        let r = fixture(w, h, 0, false);
        let d = fixture(w, h, 0, true);
        let rp = fixture(w, h, 9, false);
        let dp = fixture(w, h, 15, true);
        for mut m in [
            Box::new(Butter) as Box<dyn Metric>,
            Box::new(Zen),
            Box::new(Ssim2),
        ] {
            let name = m.id().family();
            let ri = image(&r, w, h, 0);
            let di = image(&d, w, h, 0);
            let violations = iqa::conformance::check(m.as_mut(), &ri, &di);
            assert!(violations.is_empty(), "{name} {w}x{h}: {violations:#?}");
            let cold = iqa::score(m.as_mut(), ri.clone(), di.clone(), EvalOpts::new()).unwrap();
            let padded = iqa::score(
                m.as_mut(),
                image(&rp, w, h, 9),
                image(&dp, w, h, 15),
                EvalOpts::new(),
            )
            .unwrap();
            assert_eq!(
                bits(&cold),
                bits(&padded),
                "{name}: active rows must ignore padding"
            );
            assert_eq!(cold.id.canonical(), padded.id.canonical());
            let mut warm =
                iqa::prepare_reference(m.as_mut(), image(&rp, w, h, 9), EvalOpts::new()).unwrap();
            let warm = warm.score(image(&dp, w, h, 15), EvalOpts::new()).unwrap();
            assert_eq!(
                bits(&cold),
                bits(&warm),
                "{name}: real warm work must preserve cold bits"
            );
        }
    }
}
#[test]
fn wrapper_scores_and_maps_match_direct_native_calls() {
    let (w, h) = (97, 73);
    let r = fixture(w, h, 0, false);
    let d = fixture(w, h, 0, true);
    let rp = PixelSlice::new(&r, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let dp = PixelSlice::new(&d, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
    let ss = fast_ssim2::compute_ssimulacra2(&rp, &dp).unwrap();
    let wrapped = iqa::score(
        &mut Ssim2,
        rp.clone().into(),
        dp.clone().into(),
        EvalOpts::new(),
    )
    .unwrap();
    assert_eq!(wrapped.result.score().to_bits(), ss.to_bits());
    let native = butteraugli::butteraugli(
        butteraugli::Img::new(
            bytemuck::cast_slice::<u8, butteraugli::RGB8>(&r),
            w as usize,
            h as usize,
        ),
        butteraugli::Img::new(
            bytemuck::cast_slice::<u8, butteraugli::RGB8>(&d),
            w as usize,
            h as usize,
        ),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
    )
    .unwrap();
    let wrapped = iqa::score(
        &mut Butter,
        rp.clone().into(),
        dp.clone().into(),
        EvalOpts::new().with_native_map(),
    )
    .unwrap();
    assert_eq!(
        bits(&wrapped),
        vec![native.score.to_bits(), native.pnorm_3.to_bits()]
    );
    assert_eq!(
        wrapped
            .result
            .map()
            .unwrap()
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        native
            .diffmap
            .unwrap()
            .pixels()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    );
    let z = zensim::Zensim::new(zensim::ZensimProfile::latest()).with_parallel(false);
    let native = z
        .compute_with_diffmap(
            &zensim::ZenpixelsSource::try_from_slice(&rp).unwrap(),
            &zensim::ZenpixelsSource::try_from_slice(&dp).unwrap(),
            zensim::DiffmapOptions::default(),
        )
        .unwrap();
    let wrapped = iqa::score(
        &mut Zen,
        rp.into(),
        dp.into(),
        EvalOpts::new().with_native_map(),
    )
    .unwrap();
    assert_eq!(wrapped.result.score().to_bits(), native.score().to_bits());
    assert_eq!(
        wrapped
            .result
            .map()
            .unwrap()
            .values()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        native
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
#[test]
fn geometry_remains_classified_native_causes_remain_typed() {
    let r = fixture(64, 64, 0, false);
    let ri = image(&r, 64, 64, 0);
    let e = iqa::score(
        &mut Butter,
        ri.clone(),
        ri.clone(),
        EvalOpts::new().with_max_pixels(4095),
    )
    .unwrap_err();
    assert_eq!(e.error().kind(), ErrorKind::Resource);
    // A valid packed view below Butteraugli's warm-reference minimum reaches the
    // native constructor: its original error is retained, not reduced to text.
    let tiny = [1u8; 3];
    let tiny = image(&tiny, 1, 1, 0);
    let e = match iqa::prepare_reference(&mut Butter, tiny, EvalOpts::new()) {
        Err(e) => e,
        Ok(_) => panic!("native tiny-reference admission changed"),
    };
    assert!(
        std::error::Error::source(e.error())
            .unwrap()
            .is::<butteraugli::ButteraugliError>()
    );
    assert!(!e.is_empty());
}
#[test]
fn cancellation_during_native_work_keeps_the_first_reason() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    #[derive(Clone)]
    struct Budget(Arc<AtomicUsize>);
    impl iqa::how_far::Stop for Budget {
        fn check(&self) -> Result<(), iqa::how_far::StopReason> {
            if self.0.fetch_add(1, Ordering::SeqCst) >= 3 {
                Err(iqa::how_far::StopReason::TimedOut)
            } else {
                Ok(())
            }
        }
    }
    impl iqa::how_far::Report for Budget {
        fn advance(&self, _: u64) {}
    }
    impl iqa::how_far::Pulse for Budget {
        fn split(
            &self,
            _: iqa::how_far::Execution,
            _: &[iqa::how_far::PhaseSpec<'_>],
        ) -> Result<Vec<iqa::how_far::Child<'_>>, iqa::how_far::PlanError> {
            Err(iqa::how_far::PlanError::Unsupported)
        }
        fn share(&self) -> Result<iqa::how_far::SharedPulse, iqa::how_far::PlanError> {
            Ok(iqa::how_far::SharedPulse::new(self.clone()))
        }
    }
    let r = fixture(131, 129, 0, false);
    let d = fixture(131, 129, 0, true);
    for mut m in [
        Box::new(Butter) as Box<dyn Metric>,
        Box::new(Zen),
        Box::new(Ssim2),
    ] {
        for maps in [false, true] {
            let count = Arc::new(AtomicUsize::new(0));
            let pulse = Budget(count.clone());
            let opts = EvalOpts::new().with_pulse(&pulse);
            let opts = if maps { opts.with_native_map() } else { opts };
            let e = iqa::score(
                m.as_mut(),
                image(&r, 131, 129, 0),
                image(&d, 131, 129, 0),
                opts,
            )
            .unwrap_err();
            assert_eq!(
                iqa::stop_reason(&e),
                Some(iqa::how_far::StopReason::TimedOut),
                "{}",
                m.id().family()
            );
            assert!(
                std::error::Error::source(e.error()).is_some(),
                "native cancellation must retain its typed cause"
            );
            assert!(count.load(Ordering::SeqCst) > 3);
        }
    }
}

/// Every native term, including coarse grids, survives both the core and the
/// actual cached-reference path. Odd, padded and sub-8 inputs exercise edges.
#[test]
fn ssim2_native_maps_match_direct_strided_and_prepared_calls() {
    for (w, h) in [(64, 64), (97, 73), (131, 129), (5, 3), (130, 9)] {
        let r = fixture(w, h, 0, false);
        let d = fixture(w, h, 0, true);
        let rp = fixture(w, h, 9, false);
        let dp = fixture(w, h, 15, true);
        let ri = image(&r, w, h, 0);
        let di = image(&d, w, h, 0);
        let native =
            fast_ssim2::compute_ssimulacra2_maps(ri.as_packed().unwrap(), di.as_packed().unwrap())
                .unwrap();
        let scalar =
            fast_ssim2::compute_ssimulacra2(ri.as_packed().unwrap(), di.as_packed().unwrap())
                .unwrap();
        assert_eq!(native.score.to_bits(), scalar.to_bits());
        let opts = EvalOpts::new().with_native_map();
        let cold = iqa::score(&mut Ssim2, ri, di, opts).unwrap();
        let strided =
            iqa::score(&mut Ssim2, image(&rp, w, h, 9), image(&dp, w, h, 15), opts).unwrap();
        let mut metric = Ssim2;
        let mut warm =
            iqa::prepare_reference(&mut metric, image(&rp, w, h, 9), EvalOpts::new()).unwrap();
        let prepared = warm.score(image(&dp, w, h, 15), opts).unwrap();
        let fused = native.diffmap();
        for evaluated in [&cold, &strided, &prepared] {
            assert_eq!(evaluated.result.score().to_bits(), scalar.to_bits());
            let maps = evaluated.result.maps();
            assert_eq!(maps.len(), 1 + 9 * native.scales.len());
            assert_eq!(maps[0].name, "diffmap");
            assert_eq!(
                maps[0].geometry,
                iqa::MapGeometry::identity(w, h).with_support(iqa::Support::Global)
            );
            assert_map_bits(&maps[0], &fused);
            for (s, scale) in native.scales.iter().enumerate() {
                for (t, (term, planes)) in [
                    ("ssim", &scale.ssim),
                    ("artifact", &scale.artifact),
                    ("detail_lost", &scale.detail_lost),
                ]
                .into_iter()
                .enumerate()
                {
                    for (c, (channel, values)) in
                        ["x", "y", "b"].into_iter().zip(planes).enumerate()
                    {
                        let map = &maps[1 + s * 9 + t * 3 + c];
                        assert_eq!(map.name, format!("{term}_{channel}_{s}"));
                        let cell = 1 << s;
                        let geometry = iqa::MapGeometry::grid(
                            w,
                            h,
                            0,
                            0,
                            cell,
                            cell,
                            scale.width as u32,
                            scale.height as u32,
                        );
                        let geometry = if w < 8 || h < 8 {
                            geometry.with_support(iqa::Support::Global)
                        } else {
                            geometry.with_halo(5 * cell, 5 * cell)
                        };
                        assert_eq!(map.geometry, geometry);
                        assert_map_bits(map, values);
                    }
                }
            }
        }
        let identity =
            iqa::score(&mut Ssim2, image(&r, w, h, 0), image(&r, w, h, 0), opts).unwrap();
        assert!(
            identity
                .result
                .maps()
                .iter()
                .all(|m| m.values().iter().all(|&v| v == 0.0))
        );
    }
}
fn assert_map_bits(map: &iqa::MapView<'_>, expected: &[f32]) {
    assert_eq!(map.polarity, iqa::MapPolarity::Difference);
    assert_eq!(
        map.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "{}",
        map.name
    );
}
