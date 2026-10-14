//! IW-SSIM's per-scale contrast-structure maps on the generic surface.

use iwssim::{Iwssim, IwssimParams};
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

const NAMES: [&str; 5] = ["cs_0", "cs_1", "cs_2", "cs_3", "csl_4"];

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}

fn metric(params: IwssimParams) -> Iwssim {
    Iwssim::with_params(
        8,
        8,
        IwssimParams {
            allow_small: true,
            ..params
        },
    )
    .unwrap()
}

fn bits(e: &dyn za::Evaluation) -> Vec<u64> {
    e.scores().map(|(_, v)| v.to_bits()).collect()
}

fn with_patch(src: &[u8], w: usize, px: usize, py: usize, size: usize) -> Vec<u8> {
    let mut out = src.to_vec();
    for y in py..py + size {
        for x in px..px + size {
            for c in 0..3 {
                out[(y * w + x) * 3 + c] = 255 - out[(y * w + x) * 3 + c];
            }
        }
    }
    out
}

#[test]
fn maps_keep_the_scores_and_describe_each_scale() {
    for (w, h) in [(176usize, 176usize), (200, 183), (256, 192), (64, 48)] {
        let a = za::conformance::textured_rgb8(w, h, 1);
        let b = za::conformance::textured_rgb8(w, h, 2);
        let flat = vec![120u8; w * h * 3];
        let patched = with_patch(&a, w, w / 2, h / 3, 6);
        for (r, d) in [(&a, &b), (&a, &a), (&flat, &a), (&a, &patched)] {
            let (r, d) = (image(r, w, h, w * 3), image(d, w, h, w * 3));
            let mut m = metric(IwssimParams::default());
            let plain = m
                .evaluate(r.clone(), d.clone(), za::EvalOpts::SCORE_ONLY)
                .unwrap();
            let mapped = m
                .evaluate(
                    r.clone(),
                    d.clone(),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            assert_eq!(bits(&*mapped), bits(&*plain), "{w}x{h}");
            let maps = mapped.maps();
            assert_eq!(maps.iter().map(|m| m.name).collect::<Vec<_>>(), NAMES);
            let tiled = w.min(h) < 176;
            let (mut sw, mut sh) = (w.max(176), h.max(176));
            for (s, m) in maps.iter().enumerate() {
                let cell = 1u32 << s;
                assert_eq!(
                    (m.width(), m.height()),
                    ((sw - 10) as u32, (sh - 10) as u32)
                );
                assert_eq!(m.polarity, za::MapPolarity::Similarity);
                let g = m.geometry;
                assert_eq!(
                    (g.cell_width, g.origin_x),
                    (cell, 5 * i64::from(cell) - i64::from(cell / 2))
                );
                assert_eq!(g.support == za::Support::Global, tiled);
                assert!(
                    m.values()
                        .iter()
                        .all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6)
                );
                (sw, sh) = (sw.div_ceil(2), sh.div_ceil(2));
            }
        }
    }
}

/// Without information weighting every scale pools its map by a plain
/// mean, so the maps reproduce `wmcs_0..4` bit for bit (the coarsest
/// scale pools that way in both modes).
#[test]
fn maps_fold_to_the_per_scale_terms() {
    for iw_flag in [false, true] {
        let (w, h) = (200usize, 190usize);
        let a = za::conformance::textured_rgb8(w, h, 3);
        let b = za::conformance::textured_rgb8(w, h, 4);
        let mut m = metric(IwssimParams {
            iw_flag,
            ..IwssimParams::default()
        });
        let e = m
            .evaluate(
                image(&a, w, h, w * 3),
                image(&b, w, h, w * 3),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        let wmcs: Vec<f64> = e.scores().skip(1).map(|(_, v)| v).collect();
        for (s, map) in e.maps().iter().enumerate() {
            if iw_flag && s < 4 {
                continue;
            }
            let v = map.values();
            let mut sum = 0.0f64;
            for &x in v.iter() {
                sum += x as f64;
            }
            let mean = sum / v.len() as f64;
            assert_eq!(
                mean.to_bits(),
                wmcs[s].to_bits(),
                "iw_flag {iw_flag} scale {s}"
            );
        }
    }
}

#[test]
fn warm_candidates_serve_the_cold_maps() {
    let (w, h) = (190usize, 180usize);
    let a = za::conformance::textured_rgb8(w, h, 5);
    let cands = [
        za::conformance::textured_rgb8(w, h, 6),
        with_patch(&a, w, 30, 40, 9),
    ];
    let mut cold = metric(IwssimParams::default());
    let want: Vec<_> = cands
        .iter()
        .map(|d| {
            let e = cold
                .evaluate(
                    image(&a, w, h, w * 3),
                    image(d, w, h, w * 3),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            (
                bits(&*e),
                e.maps()
                    .iter()
                    .map(|m| m.values().into_owned())
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut m = metric(IwssimParams::default());
    // `Iwssim::warm_reference` (inherent) shadows the trait method.
    let mut warm =
        za::Metric::warm_reference(&mut m, image(&a, w, h, w * 3), za::EvalOpts::SCORE_ONLY)
            .unwrap();
    assert!(warm.reuses_reference_work());
    for _round in 0..2 {
        for (d, (sb, maps)) in cands.iter().zip(&want) {
            let e = warm
                .evaluate(
                    image(d, w, h, w * 3),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            assert_eq!(&bits(&*e), sb);
            let got: Vec<_> = e.maps().iter().map(|m| m.values().into_owned()).collect();
            assert_eq!(&got, maps);
        }
        // Score-only warm candidates still work after a map request.
        let e = warm
            .evaluate(image(&cands[0], w, h, w * 3), za::EvalOpts::SCORE_ONLY)
            .unwrap();
        assert_eq!(&bits(&*e), &want[0].0);
        assert!(e.map().is_none());
    }
}

/// All filters are finite, so a cell whose declared support misses a
/// changed patch is computed from the same samples as for the unchanged
/// pair: bit-identical. A too-small halo or a shifted origin fails here.
#[test]
fn support_bounds_where_a_patch_can_reach() {
    for (w, h) in [(200usize, 180usize), (177, 191)] {
        let a = za::conformance::textured_rgb8(w, h, 9);
        let mut m = metric(IwssimParams::default());
        let base = m
            .evaluate(
                image(&a, w, h, w * 3),
                image(&a, w, h, w * 3),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        for (px, py, size) in [(90usize, 70usize, 4usize), (0, 0, 3), (w - 4, h - 5, 3)] {
            let b = with_patch(&a, w, px, py, size);
            let e = m
                .evaluate(
                    image(&a, w, h, w * 3),
                    image(&b, w, h, w * 3),
                    za::EvalOpts::SCORE_ONLY.with_native_map(),
                )
                .unwrap();
            for (mm, mb) in e.maps().iter().zip(base.maps()) {
                let g = mm.geometry;
                let (v, vb) = (mm.values(), mb.values());
                let mut changed = 0;
                for j in 0..g.cells_y {
                    for i in 0..g.cells_x {
                        let k = (j * g.cells_x + i) as usize;
                        let hits = g.support_rect(i, j).is_some_and(|s| {
                            s.x < (px + size) as u32
                                && px < (s.x + s.width) as usize
                                && s.y < (py + size) as u32
                                && py < (s.y + s.height) as usize
                        });
                        if hits {
                            changed += usize::from(v[k].to_bits() != vb[k].to_bits());
                        } else {
                            assert_eq!(
                                v[k].to_bits(),
                                vb[k].to_bits(),
                                "{w}x{h} ({px},{py}) {} ({i},{j})",
                                mm.name
                            );
                        }
                    }
                }
                assert!(changed > 0, "{w}x{h} ({px},{py}) {}: no response", mm.name);
            }
        }
    }
}

/// A single changed pixel dips `cs_0` most in a cell that owns that pixel
/// or its neighbour: cell `i` is centred on pixel `i + 5`.
#[test]
fn finest_scale_dips_at_the_changed_pixel() {
    let (w, h) = (192usize, 184usize);
    let a = za::conformance::textured_rgb8(w, h, 11);
    for (px, py) in [(60usize, 50usize), (101, 133)] {
        let b = with_patch(&a, w, px, py, 1);
        let mut m = metric(IwssimParams::default());
        let e = m
            .evaluate(
                image(&a, w, h, w * 3),
                image(&b, w, h, w * 3),
                za::EvalOpts::SCORE_ONLY.with_native_map(),
            )
            .unwrap();
        let m0 = e.map().unwrap();
        let v = m0.values();
        let k = (0..v.len()).min_by(|&x, &y| v[x].total_cmp(&v[y])).unwrap();
        let (i, j) = (k as u32 % m0.width(), k as u32 / m0.width());
        let r = m0.geometry.cell_rect(i, j).unwrap();
        let (dx, dy) = (r.x as i64 - px as i64, r.y as i64 - py as i64);
        assert!(
            dx.abs() <= 1 && dy.abs() <= 1,
            "pixel ({px},{py}): dip at cell ({i},{j}) = pixel ({},{})",
            r.x,
            r.y
        );
    }
}

#[test]
fn stopped_requests_and_strided_input() {
    let (w, h) = (180usize, 176usize);
    let (a, b) = (
        za::conformance::textured_rgb8(w, h, 1),
        za::conformance::textured_rgb8(w, h, 2),
    );
    let stride = (w + 3) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0x3Cu8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let mut m = metric(IwssimParams::default());
    let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
    let packed = m
        .evaluate(image(&a, w, h, w * 3), image(&b, w, h, w * 3), opts)
        .unwrap();
    let strided = m
        .evaluate(image(&pa, w, h, stride), image(&pb, w, h, stride), opts)
        .unwrap();
    assert_eq!(bits(&*packed), bits(&*strided));
    for (p, s) in packed.maps().iter().zip(strided.maps()) {
        assert_eq!(p.values(), s.values());
    }
    let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
    let err = m
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            opts.with_pulse(&stop),
        )
        .err()
        .unwrap();
    assert!(za::is_stopped(&err));
}
