//! MAD's strategy maps on the generic surface.

use mad_iqa::Mad;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}

fn eval(r: &za::Image<'_>, d: &za::Image<'_>, opts: za::EvalOpts<'_>) -> Box<dyn za::Evaluation> {
    Mad.evaluate(r.clone(), d.clone(), opts).unwrap()
}

fn bits(e: &dyn za::Evaluation) -> Vec<u64> {
    e.scores().map(|(_, v)| v.to_bits()).collect()
}

#[test]
fn maps_keep_the_scores_and_describe_the_interior() {
    for (w, h) in [(96usize, 96usize), (61, 75), (34, 34)] {
        let a = za::conformance::textured_rgb8(w, h, 1);
        let b = za::conformance::textured_rgb8(w, h, 2);
        let flat = vec![100u8; w * h * 3];
        for (r, d) in [(&a, &b), (&a, &a), (&flat, &a)] {
            let identical = r == d;
            let (r, d) = (image(r, w, h, w * 3), image(d, w, h, w * 3));
            let plain = eval(&r, &d, za::EvalOpts::SCORE_ONLY);
            let mapped = eval(&r, &d, za::EvalOpts::SCORE_ONLY.with_native_map());
            assert_eq!(bits(&*mapped), bits(&*plain), "{w}x{h}");
            let maps = mapped.maps();
            assert_eq!(
                maps.iter().map(|m| m.name).collect::<Vec<_>>(),
                ["hi", "lo"]
            );
            let (w32, h32) = (w as u32, h as u32);
            assert_eq!(
                maps[0].geometry,
                za::MapGeometry::grid(w32, h32, 16, 16, 1, 1, w32 - 33, h32 - 33)
                    .with_support(za::Support::Global)
            );
            assert_eq!(
                maps[1].geometry,
                za::MapGeometry::grid(
                    w32,
                    h32,
                    16,
                    16,
                    4,
                    4,
                    (w32 - 18) / 4 - 3,
                    (h32 - 18) / 4 - 3
                )
                .with_support(za::Support::Global)
            );
            for m in &maps {
                assert_eq!(m.polarity, za::MapPolarity::Difference);
                assert!(
                    m.values().iter().all(|v| v.is_finite() && *v >= 0.0),
                    "{}",
                    m.name
                );
            }
            // MAD = HI^σ·LO^(1−σ) is 0 whenever HI is, so only identity
            // forces both maps to 0.
            if identical {
                assert_eq!(plain.score(), 0.0);
                assert!(maps.iter().all(|m| m.values().iter().all(|&v| v == 0.0)));
            }
        }
    }
}

#[test]
fn below_the_interior_there_is_no_map() {
    let (w, h) = (33, 50);
    let a = za::conformance::textured_rgb8(w, h, 1);
    let (r, d) = (image(&a, w, h, w * 3), image(&a, w, h, w * 3));
    assert!(eval(&r, &d, za::EvalOpts::SCORE_ONLY).score().is_nan());
    let e = Mad
        .evaluate(r, d, za::EvalOpts::SCORE_ONLY.with_native_map())
        .err()
        .unwrap();
    assert_eq!(e.error().kind(), za::ErrorKind::NoMap);
}

/// A localized patch peaks on the `lo` tile grid where its window meets
/// the patch: the tile alignment is the reference's.
#[test]
fn lo_peaks_at_the_patch() {
    let (w, h) = (128usize, 112usize);
    let a = za::conformance::textured_rgb8(w, h, 5);
    let (px, py, size) = (72usize, 48usize, 12usize);
    let mut b = a.clone();
    for y in py..py + size {
        for x in px..px + size {
            for c in 0..3 {
                b[(y * w + x) * 3 + c] = 255 - b[(y * w + x) * 3 + c];
            }
        }
    }
    let e = eval(
        &image(&a, w, h, w * 3),
        &image(&b, w, h, w * 3),
        za::EvalOpts::SCORE_ONLY.with_native_map(),
    );
    for m in e.maps() {
        let g = m.geometry;
        let v = m.values();
        let (k, _) =
            v.iter().enumerate().fold(
                (0, f32::MIN),
                |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) },
            );
        let cell = g
            .cell_rect(k as u32 % g.cells_x, k as u32 / g.cells_x)
            .unwrap();
        // The tile statistics read a 16×16 window from the tile origin;
        // the local MSE reads {x−7 … x+8}. Either way the peak must sit
        // within that reach of the patch.
        let reach = if m.name == "lo" { 16 } else { 8 };
        let (cx0, cy0) = (cell.x as usize, cell.y as usize);
        assert!(
            cx0 < px + size + reach
                && px < cx0 + reach
                && cy0 < py + size + reach
                && py < cy0 + reach,
            "{}: peak at ({cx0},{cy0}) away from the patch",
            m.name
        );
    }
}

#[test]
fn strided_subsampled_and_stopped() {
    let (w, h) = (70, 66);
    let (a, b) = (
        za::conformance::textured_rgb8(w, h, 3),
        za::conformance::textured_rgb8(w, h, 4),
    );
    let stride = (w + 1) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0x77u8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
    let packed = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), opts);
    let strided = eval(&image(&pa, w, h, stride), &image(&pb, w, h, stride), opts);
    assert_eq!(bits(&*packed), bits(&*strided));
    for (p, s) in packed.maps().iter().zip(strided.maps()) {
        assert_eq!(p.values(), s.values(), "{}", p.name);
    }
    let sub = eval(
        &image(&a, w, h, w * 3),
        &image(&b, w, h, w * 3),
        za::EvalOpts::SCORE_ONLY.with_map(za::MapRequest::subsampled(2)),
    );
    assert_eq!(
        sub.maps()[1].geometry,
        packed.maps()[1].geometry.subsampled(2)
    );
    let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
    let err = Mad
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            opts.with_pulse(&stop),
        )
        .err()
        .unwrap();
    assert!(za::is_stopped(&err));
}
