//! Native NLPD maps on the generic surface: requesting them never changes
//! the score, they fold back to it, and their geometry bounds where a
//! local change can reach.

use nlpd::Nlpd;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{self as za, Metric as _};

fn textured(w: usize, h: usize, seed: u32) -> Vec<u8> {
    za::conformance::textured_rgb8(w, h, seed)
}

fn image(px: &[u8], w: usize, h: usize, stride: usize) -> za::Image<'_> {
    PixelSlice::new(px, w as u32, h as u32, stride, PixelDescriptor::RGB8_SRGB)
        .unwrap()
        .into()
}

fn with_patch(src: &[u8], w: usize, x0: usize, y0: usize, size: usize) -> Vec<u8> {
    let mut out = src.to_vec();
    for y in y0..y0 + size {
        for x in x0..x0 + size {
            for c in 0..3 {
                let i = (y * w + x) * 3 + c;
                out[i] = out[i].wrapping_add(50);
            }
        }
    }
    out
}

fn eval(r: &za::Image<'_>, d: &za::Image<'_>, maps: bool) -> Box<dyn za::Evaluation> {
    let opts = if maps {
        za::EvalOpts::SCORE_ONLY.with_native_map()
    } else {
        za::EvalOpts::SCORE_ONLY
    };
    Nlpd.evaluate(r.clone(), d.clone(), opts).unwrap()
}

#[test]
fn maps_never_change_the_score_and_fold_back_to_it() {
    for (w, h) in [(112usize, 100usize), (97, 101), (96, 98), (131, 96)] {
        let a = textured(w, h, 1);
        let b = textured(w, h, 2);
        let flat_a = vec![90u8; w * h * 3];
        let flat_b = vec![97u8; w * h * 3];
        let patched = with_patch(&a, w, w / 3, h / 4, 9);
        for (r, d) in [(&a, &b), (&a, &a), (&flat_a, &flat_b), (&a, &patched)] {
            let (r, d) = (image(r, w, h, w * 3), image(d, w, h, w * 3));
            let plain = eval(&r, &d, false).score();
            let mapped = eval(&r, &d, true);
            assert_eq!(mapped.score().to_bits(), plain.to_bits(), "{w}x{h}");
            let all = mapped.maps();
            assert_eq!(all.len(), 7);
            // The fused map is input-sized and its mean is NLPD.
            let d = &all[0];
            assert_eq!(d.name, "diffmap");
            assert_eq!(d.polarity, za::MapPolarity::Difference);
            assert_eq!((d.width(), d.height()), (w as u32, h as u32));
            assert_eq!(d.geometry.support, za::Support::Global);
            let dv = d.values();
            assert!(dv.iter().all(|&x| x.is_finite() && x >= 0.0));
            let dmean = dv.iter().map(|&x| f64::from(x)).sum::<f64>() / dv.len() as f64;
            assert!(
                (dmean - plain).abs() <= 1e-6 * plain.abs().max(1e-9),
                "{w}x{h}: diffmap mean {dmean} vs score {plain}"
            );
            let maps = &all[1..];
            // NLPD = (Σ_l rms_l^0.6)^(1/0.6), rms_l = sqrt(mean_l / 3).
            let mut sum = 0.0f64;
            for (l, m) in maps.iter().enumerate() {
                assert_eq!(
                    m.name,
                    [
                        "level_0", "level_1", "level_2", "level_3", "level_4", "level_5"
                    ][l]
                );
                assert_eq!(m.polarity, za::MapPolarity::Difference);
                assert_eq!(
                    (m.width(), m.height()),
                    (w.div_ceil(1 << l) as u32, h.div_ceil(1 << l) as u32)
                );
                let v = m.values();
                let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / v.len() as f64;
                sum += (mean / 3.0).sqrt().powf(0.6);
            }
            let folded = sum.powf(1.0 / 0.6);
            assert!(
                (folded - plain).abs() <= 1e-6 * plain.abs().max(1e-9),
                "{w}x{h}: folded {folded} vs score {plain}"
            );
        }
    }
}

#[test]
fn identical_pairs_map_to_zero() {
    let (w, h) = (100, 97);
    let a = textured(w, h, 5);
    let e = eval(&image(&a, w, h, w * 3), &image(&a, w, h, w * 3), true);
    assert_eq!(e.score(), 0.0);
    for m in e.maps() {
        assert!(m.values().iter().all(|&v| v == 0.0), "{}", m.name);
    }
}

#[test]
fn strided_input_gives_the_packed_maps() {
    let (w, h) = (101, 99);
    let (a, b) = (textured(w, h, 3), textured(w, h, 4));
    let stride = (w + 5) * 3;
    let pad = |src: &[u8]| {
        let mut out = vec![0x5Au8; stride * h];
        for y in 0..h {
            out[y * stride..y * stride + w * 3].copy_from_slice(&src[y * w * 3..(y + 1) * w * 3]);
        }
        out
    };
    let (pa, pb) = (pad(&a), pad(&b));
    let packed = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), true);
    let strided = eval(&image(&pa, w, h, stride), &image(&pb, w, h, stride), true);
    assert_eq!(packed.score().to_bits(), strided.score().to_bits());
    for (p, s) in packed.maps().iter().zip(strided.maps()) {
        assert_eq!(p.values(), s.values());
        assert_eq!(p.geometry, s.geometry);
    }
}

/// The pipeline is all finite filters, so a cell whose declared support
/// misses the changed patch is computed from identical samples: exactly
/// zero. A too-small halo or a shifted origin fails here.
#[test]
fn support_bounds_where_a_patch_can_reach() {
    for (w, h) in [(160usize, 128usize), (157, 131), (150, 133)] {
        let a = textured(w, h, 9);
        for (px, py, size) in [(70usize, 50usize, 3usize), (0, 0, 4), (w - 5, h - 6, 5)] {
            let b = with_patch(&a, w, px, py, size);
            let e = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), true);
            // The fused map's support is global (the rms normalization).
            for m in e.maps().into_iter().filter(|m| m.name != "diffmap") {
                let g = m.geometry;
                let v = m.values();
                let mut touched = 0;
                for j in 0..g.cells_y {
                    for i in 0..g.cells_x {
                        let s = g
                            .support_rect(i, j)
                            .expect("every level cell overlaps the image");
                        let hits = s.x < (px + size) as u32
                            && px < (s.x + s.width) as usize
                            && s.y < (py + size) as u32
                            && py < (s.y + s.height) as usize;
                        let val = v[(j * g.cells_x + i) as usize];
                        if hits {
                            touched += usize::from(val > 0.0);
                        } else {
                            assert_eq!(
                                val, 0.0,
                                "{w}x{h} patch ({px},{py}) {}: cell ({i},{j})",
                                m.name
                            );
                        }
                    }
                }
                assert!(
                    touched > 0,
                    "{w}x{h} patch ({px},{py}) {}: no response",
                    m.name
                );
            }
        }
    }
}

#[test]
fn subsampled_requests_reduce_but_fold_the_native_map() {
    let (w, h) = (112, 100);
    let (a, b) = (textured(w, h, 1), textured(w, h, 2));
    let (r, d) = (image(&a, w, h, w * 3), image(&b, w, h, w * 3));
    let native = eval(&r, &d, true);
    let sub = Nlpd
        .evaluate(
            r,
            d,
            za::EvalOpts::SCORE_ONLY.with_map(za::MapRequest::subsampled(4)),
        )
        .unwrap();
    assert_eq!(sub.score().to_bits(), native.score().to_bits());
    let (n0, s0) = (&native.maps()[0], &sub.maps()[0]);
    assert_eq!(s0.geometry, n0.geometry.subsampled(4));
    assert_eq!(
        sub.map_score(za::MapFold::Max).unwrap(),
        native.map_score(za::MapFold::Max).unwrap()
    );
}

#[test]
fn a_stopped_pulse_refuses_map_requests_too() {
    let (w, h) = (100, 100);
    let a = textured(w, h, 1);
    let stop = za::how_far::StopOnly::new(za::AlwaysStop(za::how_far::StopReason::Cancelled));
    let err = Nlpd
        .evaluate(
            image(&a, w, h, w * 3),
            image(&a, w, h, w * 3),
            za::EvalOpts::SCORE_ONLY.with_native_map().with_pulse(&stop),
        )
        .err()
        .unwrap();
    assert!(za::is_stopped(&err));
}

/// A local change peaks in the fused map over it.
#[test]
fn the_fused_map_peaks_at_a_local_change() {
    let (w, h) = (160usize, 128usize);
    let a = textured(w, h, 9);
    let (px, py, size) = (100usize, 40usize, 8usize);
    let b = with_patch(&a, w, px, py, size);
    let e = eval(&image(&a, w, h, w * 3), &image(&b, w, h, w * 3), true);
    let d = e.map().unwrap();
    assert_eq!(d.name, "diffmap");
    let v = d.values();
    let k = (0..v.len()).max_by(|&i, &j| v[i].total_cmp(&v[j])).unwrap();
    let (x, y) = (k % w, k / w);
    assert!(
        (px.saturating_sub(4)..px + size + 4).contains(&x)
            && (py.saturating_sub(4)..py + size + 4).contains(&y),
        "peak at ({x}, {y})"
    );
}
