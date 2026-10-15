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
            let all = mapped.maps();
            let names: Vec<_> = all.iter().map(|m| m.name).collect();
            assert_eq!(&names[..5], NAMES);
            assert_eq!(&names[5..9], ["iw_0", "iw_1", "iw_2", "iw_3"]);
            assert_eq!(&names[9..], ["diffmap"]);
            for (cs, iw) in all[..4].iter().zip(&all[5..]) {
                assert_eq!(iw.polarity, za::MapPolarity::Feature);
                assert_eq!(iw.geometry, cs.geometry.with_support(za::Support::Global));
                assert!(iw.values().iter().all(|v| v.is_finite() && *v >= 0.0));
            }
            let maps = &all[..5];
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
        for (s, map) in e.maps().iter().take(5).enumerate() {
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
            // The fused map's support is global (the scale means).
            for (mm, mb) in e
                .maps()
                .iter()
                .zip(base.maps())
                .filter(|(m, _)| m.name != "diffmap")
            {
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

/// With IW pooling, each fine scale's term is the `iw`-weighted mean of its
/// `cs` map: `wmcs_s = Σ cs·iw / Σ iw` (the kernel narrows both f64 sums to
/// f32 before dividing, so the fold holds to f32 precision).
#[test]
fn weights_pool_the_contrast_structure_maps() {
    let (w, h) = (210usize, 190usize);
    let a = za::conformance::textured_rgb8(w, h, 31);
    let b = with_patch(&za::conformance::textured_rgb8(w, h, 32), w, 50, 60, 20);
    let mut m = metric(IwssimParams::default());
    let e = m
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .unwrap();
    let wmcs: Vec<f64> = e.scores().skip(1).map(|(_, v)| v).collect();
    let maps = e.maps();
    for s in 0..4 {
        let (cs, iw) = (maps[s].values(), maps[5 + s].values());
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&c, &wgt) in cs.iter().zip(iw.iter()) {
            num += f64::from(c) * f64::from(wgt);
            den += f64::from(wgt);
        }
        let folded = f64::from(num as f32 / den as f32);
        assert!(
            (folded - wmcs[s]).abs() <= 1e-6 * wmcs[s].abs(),
            "scale {s}: {folded} vs {}",
            wmcs[s]
        );
    }
    let mut plain = metric(IwssimParams {
        iw_flag: false,
        ..IwssimParams::default()
    });
    let e = plain
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .unwrap();
    let names: Vec<_> = e.maps().iter().map(|m| m.name).collect();
    assert_eq!(names.len(), 6);
    assert!(
        names.iter().all(|n| !n.starts_with("iw_")),
        "no weights without IW pooling"
    );
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

/// The fused `diffmap` breaks down `−ln IW-SSIM` at input resolution,
/// vanishes for identical inputs and peaks at a local change.
#[test]
fn fused_diffmap_breaks_down_minus_log_iwssim() {
    let (w, h) = (192usize, 176usize);
    let a = za::conformance::textured_rgb8(w, h, 13);
    let (px, py, size) = (100usize, 80usize, 10usize);
    let b = with_patch(&a, w, px, py, size);
    let e = metric(IwssimParams::default())
        .evaluate(
            image(&a, w, h, w * 3),
            image(&b, w, h, w * 3),
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .unwrap();
    let maps = e.maps();
    let d = maps.iter().find(|m| m.name == "diffmap").unwrap();
    assert_eq!(d.polarity, za::MapPolarity::Difference);
    assert_eq!(d.geometry.support, za::Support::Global);
    let v = d.values();
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
    let want = -e.score().ln();
    assert!((mean - want).abs() <= 1e-5 * want, "{mean} vs {want}");
    let k = (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap() as u32;
    let g = d.geometry;
    let (x, y) = (
        (g.origin_x + i64::from(k % g.cells_x)) as usize,
        (g.origin_y + i64::from(k / g.cells_x)) as usize,
    );
    assert!(
        (px.saturating_sub(12)..px + size + 12).contains(&x)
            && (py.saturating_sub(12)..py + size + 12).contains(&y),
        "peak at ({x}, {y})"
    );
    let same = metric(IwssimParams::default())
        .evaluate(
            image(&a, w, h, w * 3),
            image(&a, w, h, w * 3),
            za::EvalOpts::SCORE_ONLY.with_native_map(),
        )
        .unwrap();
    let maps = same.maps();
    let d = maps.iter().find(|m| m.name == "diffmap").unwrap();
    let worst = d.values().iter().fold(0f32, |m, &x| m.max(x.abs()));
    eprintln!("identity diffmap max |v| {worst:e}");
    assert!(worst <= 1e-6, "{worst}");
}
