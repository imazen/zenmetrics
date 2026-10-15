//! The mDCT-PSNR detection map on the generic surface.

use mdctpsnr::MdctPsnr;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}

fn eval(r: &za::Image<'_>, d: &za::Image<'_>, opts: za::EvalOpts<'_>) -> Box<dyn za::Evaluation> {
    MdctPsnr.evaluate(r.clone(), d.clone(), opts).unwrap()
}

#[test]
fn detection_map_keeps_the_score_and_names_its_grid() {
    for (w, h) in [(64usize, 48usize), (37, 29), (14, 14), (77, 40)] {
        let a = za::conformance::textured_rgb8(w, h, 1);
        let b = za::conformance::textured_rgb8(w, h, 2);
        let flat = vec![140u8; w * h * 3];
        for (r, d) in [(&a, &b), (&a, &a), (&flat, &a), (&flat, &flat)] {
            let (r, d) = (image(r, w, h, w * 3), image(d, w, h, w * 3));
            let plain = eval(&r, &d, za::EvalOpts::SCORE_ONLY).score();
            let mapped = eval(&r, &d, za::EvalOpts::SCORE_ONLY.with_native_map());
            assert_eq!(mapped.score().to_bits(), plain.to_bits(), "{w}x{h}");
            let m = mapped.map().unwrap();
            assert_eq!(m.name, "detection");
            assert_eq!(m.polarity, za::MapPolarity::DetectionProbability);
            assert_eq!(
                m.geometry,
                za::MapGeometry::grid(
                    w as u32,
                    h as u32,
                    5,
                    6,
                    1,
                    1,
                    (w - 13) as u32,
                    (h - 12) as u32
                )
                .with_halo(6, 6)
            );
            let v = m.values();
            assert!(v.iter().all(|p| (0.0..=1.0).contains(p)));
            if plain.is_infinite() {
                assert!(
                    v.iter().all(|&p| p == 0.0),
                    "identical pair must not detect"
                );
            }
        }
    }
}

/// Cell `(i, j)` reads pixels `[i, i+12) × [j+1, j+13)` and nothing else:
/// cells whose window misses a changed patch are exactly 0, and the cells
/// that respond stay inside the box of windows that meet it. For an 8×8
/// patch they fill that box exactly. With fewer changed pixels, a window
/// that holds a single patch column or row can see so small a change that
/// `exp(−err^3.5)` rounds to 1 in f32, and that edge cell stays 0.
#[test]
fn the_window_is_exactly_where_a_patch_can_reach() {
    let (w, h) = (90usize, 70usize);
    let a = za::conformance::textured_rgb8(w, h, 7);
    for (px, py, size, tight) in [
        (40usize, 30usize, 8usize, true),
        (40, 30, 3, false),
        (0, 0, 2, false),
        (w - 3, h - 2, 2, false),
        (20, 50, 1, false),
    ] {
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
        let m = e.map().unwrap();
        let (cx, cy) = (m.width() as usize, m.height() as usize);
        let v = m.values();
        let meets =
            |i: usize, j: usize| i < px + size && px < i + 12 && j + 1 < py + size && py < j + 13;
        let (mut lo, mut hi) = ((usize::MAX, usize::MAX), (0, 0));
        for j in 0..cy {
            for i in 0..cx {
                let p = v[j * cx + i];
                if !meets(i, j) {
                    assert_eq!(
                        p, 0.0,
                        "patch ({px},{py}): cell ({i},{j}) outside its window"
                    );
                } else if p > 0.0 {
                    (lo, hi) = ((lo.0.min(i), lo.1.min(j)), (hi.0.max(i), hi.1.max(j)));
                }
            }
        }
        let want_lo = (px.saturating_sub(11), (py + 1).saturating_sub(13));
        let want_hi = (
            (px + size - 1).min(cx - 1),
            (py + size - 1).saturating_sub(1).min(cy - 1),
        );
        assert!(
            lo.0 >= want_lo.0 && lo.1 >= want_lo.1 && hi.0 <= want_hi.0 && hi.1 <= want_hi.1,
            "patch ({px},{py}) size {size}: responding box {lo:?}..{hi:?}"
        );
        if tight {
            assert_eq!(
                (lo, hi),
                (want_lo, want_hi),
                "patch ({px},{py}) size {size}"
            );
        }
    }
}

#[test]
fn strided_subsampled_and_stopped() {
    let (w, h) = (53, 41);
    let (a, b) = (
        za::conformance::textured_rgb8(w, h, 3),
        za::conformance::textured_rgb8(w, h, 4),
    );
    let stride = (w + 4) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0x11u8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
    let packed = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), opts);
    let strided = eval(&image(&pa, w, h, stride), &image(&pb, w, h, stride), opts);
    assert_eq!(packed.score().to_bits(), strided.score().to_bits());
    assert_eq!(
        packed.map().unwrap().values(),
        strided.map().unwrap().values()
    );
    let sub = eval(
        &image(&a, w, h, w * 3),
        &image(&b, w, h, w * 3),
        za::EvalOpts::SCORE_ONLY.with_map(za::MapRequest::subsampled(3)),
    );
    assert_eq!(
        sub.map().unwrap().geometry,
        packed.map().unwrap().geometry.subsampled(3)
    );
    let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
    let err = MdctPsnr
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            opts.with_pulse(&stop),
        )
        .err()
        .unwrap();
    assert!(za::is_stopped(&err));
}
