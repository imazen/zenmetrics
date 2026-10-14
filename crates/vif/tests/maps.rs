//! VIF information-loss maps on the generic surface.

use vif::{Vif, Vifvec};
use zenmetrics_abstractions as za;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
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

fn eval(
    m: &mut dyn za::Metric,
    r: &[u8],
    d: &[u8],
    w: usize,
    h: usize,
    maps: bool,
) -> Box<dyn za::Evaluation> {
    let opts = if maps {
        za::EvalOpts::SCORE_ONLY.with_native_map()
    } else {
        za::EvalOpts::SCORE_ONLY
    };
    m.evaluate(image(r, w, h, w * 3), image(d, w, h, w * 3), opts)
        .unwrap()
}

/// Scale `s` keeps `⌈…⌉` of the previous 'valid' grid; maps exist for
/// scales whose statistics grid is non-empty.
fn vifp_grids(w: usize, h: usize) -> Vec<(usize, usize)> {
    let (mut w, mut h) = (w, h);
    let mut out = Vec::new();
    for s in 0..4 {
        let nk = (1 << (4 - s)) + 1;
        if s > 0 {
            if w < nk || h < nk {
                break;
            }
            (w, h) = ((w - nk + 1).div_ceil(2), (h - nk + 1).div_ceil(2));
        }
        if w < nk || h < nk {
            break;
        }
        out.push((w - nk + 1, h - nk + 1));
    }
    out
}

#[test]
fn vifp_maps_keep_the_score_and_sum_to_the_loss() {
    for (w, h) in [(64usize, 48usize), (97, 61), (17, 17), (130, 129)] {
        let a = za::conformance::textured_rgb8(w, h, 1);
        let b = za::conformance::textured_rgb8(w, h, 2);
        let flat = vec![150u8; w * h * 3];
        let patched = with_patch(&a, w, w / 3, h / 3, 5);
        for (r, d) in [(&a, &b), (&a, &a), (&a, &patched), (&flat, &a)] {
            let plain = eval(&mut Vif, r, d, w, h, false).score();
            let mapped = eval(&mut Vif, r, d, w, h, true);
            assert_eq!(mapped.score().to_bits(), plain.to_bits(), "{w}x{h}");
            let all = mapped.maps();
            let maps: Vec<_> = all
                .iter()
                .filter(|m| m.name.starts_with("info_loss"))
                .collect();
            let (num, den): (Vec<_>, Vec<_>) = all
                .iter()
                .filter(|m| !m.name.starts_with("info_loss"))
                .partition(|m| m.name.starts_with("num"));
            assert_eq!((num.len(), den.len()), (maps.len(), maps.len()));
            let (mut sn, mut sd) = (0.0f64, 0.0f64);
            for ((l, n), d) in maps.iter().zip(&num).zip(&den) {
                assert_eq!(
                    (n.polarity, d.polarity),
                    (za::MapPolarity::Feature, za::MapPolarity::Feature)
                );
                assert_eq!((n.geometry, d.geometry), (l.geometry, l.geometry));
                for ((&lv, &nv), &dv) in l
                    .values()
                    .iter()
                    .zip(n.values().iter())
                    .zip(d.values().iter())
                {
                    assert!(
                        (lv - (dv - nv)).abs() <= 1e-5 * dv.abs().max(1.0),
                        "{}",
                        l.name
                    );
                    sn += f64::from(nv);
                    sd += f64::from(dv);
                }
            }
            if sd > 0.0 {
                let folded = sn / sd;
                let vif = plain;
                assert!(
                    (folded - vif).abs() <= 1e-5 * vif.abs().max(1e-6),
                    "{w}x{h}: {folded} vs {vif}"
                );
            }
            let grids: Vec<_> = maps
                .iter()
                .map(|m| (m.width() as usize, m.height() as usize))
                .collect();
            assert_eq!(grids, vifp_grids(w, h), "{w}x{h}");
            for (s, m) in maps.iter().enumerate() {
                assert_eq!(
                    m.name,
                    ["info_loss_0", "info_loss_1", "info_loss_2", "info_loss_3"][s]
                );
                assert_eq!(m.polarity, za::MapPolarity::Difference);
                let g = m.geometry;
                let cell = 1u32 << s;
                assert_eq!(g.cell_width, cell);
                assert_eq!(g.origin_x, 8 + 4 * s as i64 - i64::from(cell / 2));
                assert!(m.values().iter().all(|v| v.is_finite()), "{}", m.name);
            }
        }
    }
}

/// Every filter is finite and every statistic local, so a cell whose
/// declared support misses a changed patch matches the unchanged pair's
/// cell bit for bit.
#[test]
fn vifp_support_bounds_where_a_patch_can_reach() {
    let (w, h) = (150usize, 140usize);
    let a = za::conformance::textured_rgb8(w, h, 9);
    let base = eval(&mut Vif, &a, &a, w, h, true);
    for (px, py, size) in [(70usize, 60usize, 4usize), (0, 0, 3), (w - 4, h - 3, 3)] {
        let b = with_patch(&a, w, px, py, size);
        let e = eval(&mut Vif, &a, &b, w, h, true);
        for (m, mb) in e.maps().iter().zip(base.maps()) {
            let g = m.geometry;
            let (v, vb) = (m.values(), mb.values());
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
                            "({px},{py}) {} ({i},{j})",
                            m.name
                        );
                    }
                }
            }
            // Declared support is a bound, not a promise of dependence: at
            // coarse scales the 'valid' decimation chain need not reach a
            // corner, so only the interior patch must register everywhere.
            if px == 70 && !m.name.starts_with("den") {
                assert!(changed > 0, "({px},{py}) {}: no response", m.name);
            }
            // `den` reads the reference alone, which the patch never touches.
            if m.name.starts_with("den") {
                assert_eq!(changed, 0, "{}", m.name);
            }
        }
    }
}

/// A single changed pixel loses the most information in the full-resolution
/// cell whose window is centred on it: cell `i` owns pixel `i + 8`.
#[test]
fn vifp_finest_loss_peaks_at_the_changed_pixel() {
    let (w, h) = (120usize, 100usize);
    let a = za::conformance::textured_rgb8(w, h, 13);
    for (px, py) in [(50usize, 40usize), (33, 71)] {
        let b = with_patch(&a, w, px, py, 1);
        let e = eval(&mut Vif, &a, &b, w, h, true);
        let m = e.map().unwrap();
        let v = m.values();
        let k = (0..v.len()).max_by(|&x, &y| v[x].total_cmp(&v[y])).unwrap();
        let r = m
            .geometry
            .cell_rect(k as u32 % m.width(), k as u32 / m.width())
            .unwrap();
        let (dx, dy) = (r.x as i64 - px as i64, r.y as i64 - py as i64);
        assert!(
            dx.abs() <= 1 && dy.abs() <= 1,
            "({px},{py}): peak at pixel ({},{})",
            r.x,
            r.y
        );
    }
}

#[test]
fn vif_strided_subsampled_and_stopped() {
    for metric in [&mut Vif as &mut dyn za::Metric, &mut Vifvec] {
        let (w, h) = (100, 90);
        let (a, b) = (
            za::conformance::textured_rgb8(w, h, 3),
            za::conformance::textured_rgb8(w, h, 4),
        );
        let stride = (w + 2) * 3;
        let pad = |src: &[u8]| {
            let mut out = vec![0x42u8; stride * h];
            for y in 0..h {
                out[y * stride..y * stride + w * 3]
                    .copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
            }
            out
        };
        let (pa, pb) = (pad(&a), pad(&b));
        let opts = za::EvalOpts::SCORE_ONLY.with_native_map();
        let packed = metric
            .evaluate(image(&a, w, h, w * 3), image(&b, w, h, w * 3), opts)
            .unwrap();
        let strided = metric
            .evaluate(image(&pa, w, h, stride), image(&pb, w, h, stride), opts)
            .unwrap();
        assert_eq!(packed.score().to_bits(), strided.score().to_bits());
        for (p, s) in packed.maps().iter().zip(strided.maps()) {
            assert_eq!(p.values(), s.values(), "{}", p.name);
        }
        let sub = metric
            .evaluate(
                image(&a, w, h, w * 3),
                image(&b, w, h, w * 3),
                za::EvalOpts::SCORE_ONLY.with_map(za::MapRequest::subsampled(2)),
            )
            .unwrap();
        assert_eq!(
            sub.map().unwrap().geometry,
            packed.map().unwrap().geometry.subsampled(2)
        );
        let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
        let err = metric
            .evaluate(
                image(&a, w, h, w * 3),
                image(&b, w, h, w * 3),
                opts.with_pulse(&stop),
            )
            .err()
            .unwrap();
        assert!(za::is_stopped(&err));
    }
}

/// `vifvec` serves its eight pooled subbands finest first, each on its
/// own 3×3-block grid; a localized patch loses the most information in a
/// finest-level block near it.
#[test]
fn vifvec_maps_name_their_subbands_and_localize_a_patch() {
    let names = [
        "info_loss_l1_b1",
        "info_loss_l1_b4",
        "info_loss_l2_b1",
        "info_loss_l2_b4",
        "info_loss_l3_b1",
        "info_loss_l3_b4",
        "info_loss_l4_b1",
        "info_loss_l4_b4",
    ];
    let (w, h) = (160usize, 144usize);
    let a = za::conformance::textured_rgb8(w, h, 21);
    let (px, py, size) = (90usize, 60usize, 8usize);
    let b = with_patch(&a, w, px, py, size);
    let plain = eval(&mut Vifvec, &a, &b, w, h, false).score();
    let e = eval(&mut Vifvec, &a, &b, w, h, true);
    assert_eq!(e.score().to_bits(), plain.to_bits());
    let all = e.maps();
    assert_eq!(all.len(), 24);
    let maps = &all[..8];
    assert_eq!(maps.iter().map(|m| m.name).collect::<Vec<_>>(), names);
    for (k, m) in all[8..].iter().enumerate() {
        let (kind, sub) = if k < 8 { ("num", k) } else { ("den", k - 8) };
        assert_eq!(m.name, names[sub].replace("info_loss", kind));
        assert_eq!(m.polarity, za::MapPolarity::Feature);
        assert_eq!(m.geometry, maps[sub].geometry);
    }
    for (k, m) in maps.iter().enumerate() {
        let level = k / 2 + 1;
        assert_eq!(m.geometry.cell_width, 3 << (level - 1), "{}", m.name);
        assert_eq!(m.geometry.support, za::Support::Global);
        assert_eq!(m.polarity, za::MapPolarity::Difference);
        assert!(m.values().iter().all(|v| v.is_finite()), "{}", m.name);
    }
    for m in &maps[..2] {
        let v = m.values();
        let i = (0..v.len()).max_by(|&x, &y| v[x].total_cmp(&v[y])).unwrap();
        let r = m
            .geometry
            .cell_rect(i as u32 % m.width(), i as u32 / m.width())
            .unwrap();
        // The block's local statistics window reaches 8 pixels around it.
        let (x0, y0) = (r.x as usize, r.y as usize);
        assert!(
            x0 < px + size + 8
                && px < x0 + r.width as usize + 8
                && y0 < py + size + 8
                && py < y0 + r.height as usize + 8,
            "{}: peak block at ({x0},{y0}) away from the patch",
            m.name
        );
    }
}
