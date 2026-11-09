use iqa::zenpixels::{PixelDescriptor, PixelSlice};
use iqa::{EvalOpts, Evaluation, Metric, PrepareReference};
use iqa_contract_validation::{Butter, Ssim2, SsimNative, Zen, ZenNative, fixture};
use zenpsy_api as iqa;
fn image(bytes: &[u8], w: u32, h: u32, padding: usize) -> PixelSlice<'_> {
    PixelSlice::new(
        bytes,
        w,
        h,
        w as usize * 3 + padding,
        PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}
fn bits(e: &impl Evaluation) -> Vec<u64> {
    e.scores().map(|s| s.value.to_bits()).collect()
}
fn check_native<M: PrepareReference>(mut m: M)
where
    M::Output: std::fmt::Debug,
{
    for (w, h) in [(64, 64), (97, 73), (131, 129)] {
        let r = fixture(w, h, 0, false);
        let d = fixture(w, h, 0, true);
        let rp = fixture(w, h, 9, false);
        let dp = fixture(w, h, 15, true);
        let cold = iqa::score(
            &mut m,
            &image(&r, w, h, 0),
            &image(&d, w, h, 0),
            EvalOpts::new(),
        )
        .unwrap();
        let padded = iqa::score(
            &mut m,
            &image(&rp, w, h, 9),
            &image(&dp, w, h, 15),
            EvalOpts::new(),
        )
        .unwrap();
        assert_eq!(bits(&cold), bits(&padded), "{} stride", m.id().family());
        let mut warm =
            iqa::prepare_reference(&mut m, &image(&rp, w, h, 9), EvalOpts::new()).unwrap();
        assert_eq!(
            bits(&cold),
            bits(&warm.score(&image(&dp, w, h, 15), EvalOpts::new()).unwrap())
        );
    }
}
#[test]
fn typed_cold_strided_and_actual_prepared_results_agree() {
    check_native(Butter);
    check_native(Ssim2);
    check_native(Zen::default());
}
#[test]
fn native_scalars_and_opt_in_maps_survive_without_repackaging() {
    let (w, h) = (97, 73);
    let r = fixture(w, h, 0, false);
    let d = fixture(w, h, 0, true);
    let ri = image(&r, w, h, 0);
    let di = image(&d, w, h, 0);
    let mut b = Butter;
    let cold = iqa::score(&mut b, &ri, &di, EvalOpts::new()).unwrap();
    assert!(
        cold.native.diffmap.is_none(),
        "routine scoring must not allocate a diagnostic result"
    );
    let opts = EvalOpts::new().with_native_map();
    let out = iqa::score(&mut b, &ri, &di, opts).unwrap();
    let rgb = |bytes: &[u8]| bytemuck::cast_slice::<u8, butteraugli::RGB8>(bytes).to_vec();
    let rb = rgb(&r);
    let db = rgb(&d);
    let direct = butteraugli::butteraugli(
        butteraugli::Img::new(rb.as_slice(), w as usize, h as usize),
        butteraugli::Img::new(db.as_slice(), w as usize, h as usize),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
    )
    .unwrap();
    assert_eq!(bits(&out), [direct.score, direct.pnorm_3].map(f64::to_bits));
    let expected = direct
        .diffmap
        .unwrap()
        .pixels()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        out.native
            .diffmap
            .unwrap()
            .pixels()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected
    );
    let mut warm = iqa::prepare_reference(&mut b, &ri, opts).unwrap();
    assert_eq!(
        warm.score(&di, opts)
            .unwrap()
            .native
            .diffmap
            .unwrap()
            .pixels()
            .map(f32::to_bits)
            .collect::<Vec<_>>(),
        expected
    );
    let source_r = zensim::ZenpixelsSource::try_from_slice(&ri).unwrap();
    let source_d = zensim::ZenpixelsSource::try_from_slice(&di).unwrap();
    let direct = zensim::Zensim::new(zensim::ZensimProfile::latest())
        .with_parallel(false)
        .compute_with_diffmap(&source_r, &source_d, zensim::DiffmapOptions::default())
        .unwrap();
    let out = iqa::score(&mut Zen::default(), &ri, &di, opts).unwrap();
    let ZenNative::Map(native) = out.native else {
        panic!("requested native map missing")
    };
    assert_eq!(native.score().to_bits(), direct.score().to_bits());
    assert_eq!(
        native
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        direct
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    let mut z = Zen::default();
    let mut warm = iqa::prepare_reference(&mut z, &ri, EvalOpts::new()).unwrap();
    let ZenNative::Map(native) = warm.score(&di, opts).unwrap().native else {
        panic!("prepared native map missing")
    };
    assert_eq!(
        native
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        direct
            .diffmap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
}
#[test]
fn ssim2_preserves_every_native_component_and_grid_on_edges_and_prepared_calls() {
    for (w, h) in [(64, 64), (97, 73), (131, 129), (5, 3), (130, 9)] {
        let r = fixture(w, h, 0, false);
        let d = fixture(w, h, 0, true);
        let rp = fixture(w, h, 9, false);
        let dp = fixture(w, h, 15, true);
        let ri = image(&r, w, h, 0);
        let di = image(&d, w, h, 0);
        let opts = EvalOpts::new().with_native_map();
        let direct = fast_ssim2::compute_ssimulacra2_maps(&ri, &di).unwrap();
        let scalar = fast_ssim2::compute_ssimulacra2(&ri, &di).unwrap();
        assert_eq!(direct.score.to_bits(), scalar.to_bits());
        let mut m = Ssim2;
        let cold = iqa::score(&mut m, &ri, &di, opts).unwrap();
        let strided =
            iqa::score(&mut m, &image(&rp, w, h, 9), &image(&dp, w, h, 15), opts).unwrap();
        let mut warm = iqa::prepare_reference(&mut m, &ri, EvalOpts::new()).unwrap();
        let prepared = warm.score(&image(&dp, w, h, 15), opts).unwrap();
        for e in [cold, strided, prepared] {
            assert_eq!(e.values[0].to_bits(), scalar.to_bits());
            let SsimNative::Maps(n) = e.native else {
                panic!("native map missing")
            };
            assert_eq!((n.width, n.height), (direct.width, direct.height));
            assert_eq!(
                n.diffmap().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                direct
                    .diffmap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(n.scales.len(), direct.scales.len());
            for (s, d) in n.scales.iter().zip(&direct.scales) {
                assert_eq!((s.width, s.height), (d.width, d.height));
                for (a, b) in [&s.ssim, &s.artifact, &s.detail_lost].into_iter().zip([
                    &d.ssim,
                    &d.artifact,
                    &d.detail_lost,
                ]) {
                    for (a, b) in a.iter().zip(b) {
                        assert_eq!(
                            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
}
fn cancellation<M: Metric>(mut m: M)
where
    M::Output: std::fmt::Debug,
{
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
    for maps in [false, true] {
        let count = Arc::new(AtomicUsize::new(0));
        let pulse = Budget(count.clone());
        let opts = EvalOpts::new().with_pulse(&pulse);
        let opts = if maps { opts.with_native_map() } else { opts };
        let e = iqa::score(
            &mut m,
            &image(&r, 131, 129, 0),
            &image(&d, 131, 129, 0),
            opts,
        )
        .unwrap_err();
        assert_eq!(
            iqa::stop_reason(&e),
            Some(iqa::how_far::StopReason::TimedOut)
        );
        assert!(
            std::error::Error::source(e.error()).is_some(),
            "native cause must survive"
        );
        assert!(count.load(Ordering::SeqCst) > 3);
    }
}
#[test]
fn cancellation_reaches_all_three_native_kernels_and_keeps_causes() {
    cancellation(Butter);
    cancellation(Ssim2);
    cancellation(Zen::default());
}
#[test]
fn zensim_hdr_preserves_native_pq_hlg_and_absolute_linear_scores_and_stride() {
    use iqa::zenpixels::{AlphaMode, ColorPrimaries, TransferFunction};
    use zensim::HdrEncoding;
    let (w, h) = (97, 73);
    for encoding in [
        HdrEncoding::Linear,
        HdrEncoding::Pq { peak_nits: 1000.0 },
        HdrEncoding::Hlg {
            peak_nits: 1000.0,
            ambient_lux: 5.0,
        },
    ] {
        let (transfer, gain) = match encoding {
            HdrEncoding::Linear => (TransferFunction::Linear, 1000.0),
            HdrEncoding::Pq { .. } => (TransferFunction::Pq, 1.0),
            HdrEncoding::Hlg { .. } => (TransferFunction::Hlg, 1.0),
            _ => unreachable!(),
        };
        for primaries in [
            ColorPrimaries::Bt709,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
        ] {
            let desc = PixelDescriptor::RGBAF32_LINEAR
                .with_transfer(transfer)
                .with_primaries(primaries)
                .with_alpha(Some(AlphaMode::Opaque));
            let make = |padding: usize, dist: bool| {
                let stride = w as usize * 4 + padding;
                let mut v = vec![12345.0f32; (h as usize - 1) * stride + w as usize * 4];
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        for c in 0..3 {
                            let base = ((x * 31 + y * 47 + c * 63 + (x * y) % 71) % 700 + 50)
                                as f32
                                / 1000.0;
                            v[y * stride + x * 4 + c] = (base
                                + if dist && (x + 3 * y + c) % 7 == 0 {
                                    0.025
                                } else {
                                    0.0
                                })
                                * gain;
                        }
                        v[y * stride + x * 4 + 3] = 1.0;
                    }
                }
                v
            };
            let r = make(0, false);
            let d = make(0, true);
            let rp = make(4, false);
            let dp = make(8, true);
            let ri = hdr_view(&r, w, h, 0, desc);
            let di = hdr_view(&d, w, h, 0, desc);
            let direct = zensim::Zensim::new(zensim::ZensimProfile::latest())
                .with_parallel(false)
                .compute_hdr(
                    &zensim::StridedBytes::try_with_alpha_mode(
                        bytemuck::cast_slice(&r),
                        w as usize,
                        h as usize,
                        w as usize * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    )
                    .unwrap()
                    .with_color_primaries(match primaries {
                        ColorPrimaries::Bt709 => zensim::ColorPrimaries::Srgb,
                        ColorPrimaries::DisplayP3 => zensim::ColorPrimaries::DisplayP3,
                        ColorPrimaries::Bt2020 => zensim::ColorPrimaries::Bt2020,
                        _ => unreachable!(),
                    }),
                    &zensim::StridedBytes::try_with_alpha_mode(
                        bytemuck::cast_slice(&d),
                        w as usize,
                        h as usize,
                        w as usize * 16,
                        zensim::PixelFormat::LinearF32Rgba,
                        zensim::AlphaMode::Opaque,
                    )
                    .unwrap()
                    .with_color_primaries(match primaries {
                        ColorPrimaries::Bt709 => zensim::ColorPrimaries::Srgb,
                        ColorPrimaries::DisplayP3 => zensim::ColorPrimaries::DisplayP3,
                        ColorPrimaries::Bt2020 => zensim::ColorPrimaries::Bt2020,
                        _ => unreachable!(),
                    }),
                    encoding,
                )
                .unwrap();
            let mut m = Zen {
                hdr: Some(encoding),
            };
            let out = iqa::score(&mut m, &ri, &di, EvalOpts::new()).unwrap();
            let padded = iqa::score(
                &mut m,
                &hdr_view(&rp, w, h, 4, desc),
                &hdr_view(&dp, w, h, 8, desc),
                EvalOpts::new(),
            )
            .unwrap();
            assert_eq!(out.values[0].to_bits(), direct.score().to_bits());
            assert_eq!(bits(&out), bits(&padded));
            let mut erased: Box<iqa::DynMetric<'_>> =
                Box::new(iqa::Erased::new(iqa::Erased::new(Zen {
                    hdr: Some(encoding),
                })));
            let erased = iqa::score(&mut erased, &ri, &di, EvalOpts::new()).unwrap();
            assert_eq!(
                erased
                    .native::<iqa_contract_validation::Assessment<ZenNative, 1>>()
                    .unwrap()
                    .values[0]
                    .to_bits(),
                direct.score().to_bits()
            );
            let wrong = PixelSlice::new(
                bytemuck::cast_slice(&r),
                w,
                h,
                w as usize * 16,
                desc.with_transfer(TransferFunction::Srgb),
            )
            .unwrap();
            assert!(iqa::score(&mut m, &wrong, &wrong, EvalOpts::new()).is_err());
        }
    }
}

fn hdr_view(v: &[f32], w: u32, h: u32, padding: usize, desc: PixelDescriptor) -> PixelSlice<'_> {
    PixelSlice::new(
        bytemuck::cast_slice(v),
        w,
        h,
        (w as usize * 4 + padding) * 4,
        desc,
    )
    .unwrap()
}
