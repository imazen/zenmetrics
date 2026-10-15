//! Native PSNR-HVS block maps on the generic surface.

use psnrhvs::Psnrhvs;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

const NAMES: [&str; 8] = [
    "csf_error",
    "masked_error",
    "csf_error_r",
    "csf_error_g",
    "csf_error_b",
    "masked_error_r",
    "masked_error_g",
    "masked_error_b",
];

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}

fn eval(r: &za::Image<'_>, d: &za::Image<'_>, opts: za::EvalOpts<'_>) -> Box<dyn za::Evaluation> {
    Psnrhvs.evaluate(r.clone(), d.clone(), opts).unwrap()
}

fn scores(e: &dyn za::Evaluation) -> Vec<u64> {
    e.scores().map(|(_, v)| v.to_bits()).collect()
}

#[test]
fn maps_keep_every_score_and_describe_the_block_grid() {
    for (w, h) in [(64usize, 48usize), (37, 29), (8, 8), (131, 70)] {
        let a = za::conformance::textured_rgb8(w, h, 3);
        let b = za::conformance::textured_rgb8(w, h, 4);
        let flat = vec![77u8; w * h * 3];
        for (r, d) in [(&a, &b), (&a, &a), (&flat, &a), (&flat, &flat)] {
            let (r, d) = (image(r, w, h, w * 3), image(d, w, h, w * 3));
            let plain = eval(&r, &d, za::EvalOpts::SCORE_ONLY);
            let mapped = eval(&r, &d, za::EvalOpts::SCORE_ONLY.with_native_map());
            assert_eq!(scores(&*mapped), scores(&*plain), "{w}x{h}");
            let maps = mapped.maps();
            assert_eq!(maps.iter().map(|m| m.name).collect::<Vec<_>>(), NAMES);
            let want = za::MapGeometry::grid(
                w as u32,
                h as u32,
                0,
                0,
                8,
                8,
                ((w - 8) / 8 + 1) as u32,
                ((h - 8) / 8 + 1) as u32,
            );
            for m in &maps {
                assert_eq!(m.geometry, want);
                assert_eq!(m.polarity, za::MapPolarity::Difference);
                assert!(m.values().iter().all(|v| v.is_finite() && *v >= 0.0));
            }
            // Per-channel maps fold back to the per-channel scores (the
            // exact fold, with the kernel's band split, is a unit test).
            for (c, key) in ["psnrhvs_r", "psnrhvs_g", "psnrhvs_b"].iter().enumerate() {
                let v = maps[2 + c].values();
                let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
                let got = if mean == 0.0 {
                    100000.0
                } else {
                    10.0 * (255.0f64 * 255.0 / mean).log10()
                };
                let want = plain.scores().find(|(k, _)| k == key).unwrap().1;
                assert!(
                    (got - want).abs() < 1e-9 * want.abs(),
                    "{w}x{h} {key}: {got} vs {want}"
                );
            }
        }
    }
}

#[test]
fn strided_input_gives_the_packed_maps() {
    let (w, h) = (41, 33);
    let (a, b) = (
        za::conformance::textured_rgb8(w, h, 8),
        za::conformance::textured_rgb8(w, h, 9),
    );
    let stride = (w + 2) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0xC3u8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
    let packed = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), opts);
    let strided = eval(&image(&pa, w, h, stride), &image(&pb, w, h, stride), opts);
    assert_eq!(scores(&*packed), scores(&*strided));
    for (p, s) in packed.maps().iter().zip(strided.maps()) {
        assert_eq!(p.values(), s.values(), "{}", p.name);
    }
}

#[test]
fn subsampled_and_stopped_requests() {
    let (w, h) = (64, 64);
    let (a, b) = (
        za::conformance::textured_rgb8(w, h, 1),
        za::conformance::textured_rgb8(w, h, 2),
    );
    let (r, d) = (image(&a, w, h, w * 3), image(&b, w, h, w * 3));
    let native = eval(&r, &d, za::EvalOpts::SCORE_ONLY.with_native_map());
    let sub = eval(
        &r,
        &d,
        za::EvalOpts::SCORE_ONLY.with_map(za::MapRequest::subsampled(2)),
    );
    assert_eq!(
        sub.maps()[0].geometry,
        native.maps()[0].geometry.subsampled(2)
    );
    assert_eq!(
        sub.map_score(za::MapFold::Mean).unwrap().to_bits(),
        native.map_score(za::MapFold::Mean).unwrap().to_bits()
    );
    let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
    let err = Psnrhvs
        .evaluate(
            r,
            d,
            za::EvalOpts::SCORE_ONLY.with_native_map().with_pulse(&stop),
        )
        .err()
        .unwrap();
    assert!(za::is_stopped(&err));
}
