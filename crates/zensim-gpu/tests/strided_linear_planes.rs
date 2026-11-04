//! The strided linear-plane entry points give the tight form's score and diffmap,
//! to the bit: padding holds out-of-range garbage and is never read. Covers the
//! pass-through (planes span `stride * height`), the pack (exact-minimum buffers),
//! and the sub-64px instance that reflect-pads.

#![cfg(any(feature = "cpu", feature = "wgpu"))]

use zenmetrics_gpu_core::Backend;
use zensim_gpu::{ZensimOpaque, ZensimParams};

#[cfg(feature = "cpu")]
const BACKEND: Backend = Backend::Cpu;
#[cfg(all(not(feature = "cpu"), feature = "wgpu"))]
const BACKEND: Backend = Backend::Wgpu;

type Planes = [Vec<f32>; 3];

fn planes(w: usize, h: usize, seed: usize) -> Planes {
    std::array::from_fn(|c| {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x * 3 + y * 5 + (x * y) / 7 + c * 31 + seed * 13) % 241;
                0.05 + 0.9 * t as f32 / 240.0
            })
            .collect()
    })
}

/// Exact minimum when `full` is false, `stride * height` samples when true.
fn padded(tight: &[f32], w: usize, stride: usize, full: bool) -> Vec<f32> {
    let h = tight.len() / w;
    let n = if full {
        h * stride
    } else {
        (h - 1) * stride + w
    };
    let mut out = vec![f32::NAN; n];
    for (row, line) in tight.chunks_exact(w).enumerate() {
        out[row * stride..row * stride + w].copy_from_slice(line);
    }
    out
}

fn padded3(p: &Planes, w: usize, stride: usize, full: bool) -> Planes {
    std::array::from_fn(|c| padded(&p[c], w, stride, full))
}

fn refs(p: &Planes) -> [&[f32]; 3] {
    [&p[0], &p[1], &p[2]]
}

fn opaque(w: usize, h: usize) -> ZensimOpaque {
    ZensimOpaque::new(BACKEND, w as u32, h as u32, ZensimParams::default()).unwrap()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn every_entry_point_matches_tight_to_the_bit() {
    // At and above 64px read in place or pack; below it the opaque pads.
    for (w, h) in [(64usize, 64usize), (65, 67), (32, 40)] {
        let (r, d) = (planes(w, h, 0), planes(w, h, 1));
        let (rs, ds) = (w + 5, w + 9);
        for full in [true, false] {
            let label = format!("{w}x{h} full={full}");
            let (pr, pd) = (padded3(&r, w, rs, full), padded3(&d, w, ds, full));

            let mut a = opaque(w, h);
            let mut b = opaque(w, h);
            let tight = a
                .score_from_linear_planes(refs(&r), w, refs(&d), w)
                .unwrap();
            let strided = b
                .score_from_linear_planes(refs(&pr), rs, refs(&pd), ds)
                .unwrap();
            assert_eq!(tight.to_bits(), strided.to_bits(), "{label} score");

            let (mut dm_a, mut dm_b) = (Vec::new(), Vec::new());
            let tight = a
                .score_from_linear_planes_with_diffmap(refs(&r), w, refs(&d), w, &mut dm_a)
                .unwrap();
            let strided = b
                .score_from_linear_planes_with_diffmap(refs(&pr), rs, refs(&pd), ds, &mut dm_b)
                .unwrap();
            assert_eq!(tight.to_bits(), strided.to_bits(), "{label} diffmap score");
            assert_eq!(bits(&dm_a), bits(&dm_b), "{label} diffmap");

            a.warm_reference_from_linear_planes(refs(&r), w).unwrap();
            b.warm_reference_from_linear_planes(refs(&pr), rs).unwrap();
            let tight = a
                .score_from_linear_planes_with_warm_ref(refs(&d), w)
                .unwrap();
            let strided = b
                .score_from_linear_planes_with_warm_ref(refs(&pd), ds)
                .unwrap();
            assert_eq!(tight.to_bits(), strided.to_bits(), "{label} warm score");

            let (mut dm_a, mut dm_b) = (Vec::new(), Vec::new());
            let tight = a
                .score_from_linear_planes_with_warm_ref_diffmap(refs(&d), w, &mut dm_a)
                .unwrap();
            let strided = b
                .score_from_linear_planes_with_warm_ref_diffmap(refs(&pd), ds, &mut dm_b)
                .unwrap();
            assert_eq!(tight.to_bits(), strided.to_bits(), "{label} warm diffmap");
            assert_eq!(bits(&dm_a), bits(&dm_b), "{label} warm diffmap map");
        }

        // A tight view through the strided entry reads the same samples.
        let mut a = opaque(w, h);
        let mut b = opaque(w, h);
        let tight = a
            .score_from_linear_planes(refs(&r), w, refs(&d), w)
            .unwrap();
        let view = b
            .score_from_linear_planes(refs(&r), w, refs(&d), w)
            .unwrap();
        assert_eq!(tight.to_bits(), view.to_bits(), "{w}x{h} tight view");
    }
}

#[test]
fn bad_geometry_is_a_dimension_mismatch() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded3(&r, w, rs, false), padded3(&d, w, ds, false));
    let mut m = opaque(w, h);
    let mismatch = |e: zensim_gpu::Error| matches!(e, zensim_gpu::Error::DimensionMismatch { .. });
    // One sample short of the minimum, on each plane of each image.
    for plane in 0..3 {
        let mut short = refs(&pr);
        short[plane] = &short[plane][..short[plane].len() - 1];
        let e = m
            .score_from_linear_planes(short, rs, refs(&pd), ds)
            .unwrap_err();
        assert!(mismatch(e), "short ref {plane}");
        let mut short = refs(&pd);
        short[plane] = &short[plane][..short[plane].len() - 1];
        let e = m
            .score_from_linear_planes(refs(&pr), rs, short, ds)
            .unwrap_err();
        assert!(mismatch(e), "short dist {plane}");
    }
    // A stride below the width, and one whose arithmetic overflows.
    for stride in [w - 1, usize::MAX] {
        let e = m
            .score_from_linear_planes(refs(&pr), stride, refs(&pd), ds)
            .unwrap_err();
        assert!(mismatch(e), "stride {stride}");
        let e = m
            .warm_reference_from_linear_planes(refs(&pr), stride)
            .unwrap_err();
        assert!(mismatch(e), "warm stride {stride}");
    }
}
