//! The strided linear-plane entry points give the tight form's score and diffmap,
//! to the bit: padding holds garbage and is never read, and exact-minimum buffers
//! are accepted. Runs on wgpu when CUDA is not compiled in, as `diffmap_dispatch`
//! does.

#![cfg(any(feature = "cuda", feature = "wgpu"))]

use cvvdp_gpu::params::CvvdpParams;
use cvvdp_gpu::{Backend, CvvdpOpaque};

#[cfg(feature = "cuda")]
const BACKEND: Backend = Backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
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

fn padded(p: &Planes, w: usize, stride: usize) -> Planes {
    std::array::from_fn(|c| {
        let h = p[c].len() / w;
        let mut out = vec![f32::NAN; (h - 1) * stride + w];
        for (row, line) in p[c].chunks_exact(w).enumerate() {
            out[row * stride..row * stride + w].copy_from_slice(line);
        }
        out
    })
}

fn refs(p: &Planes) -> [&[f32]; 3] {
    [&p[0], &p[1], &p[2]]
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn opaque(w: usize, h: usize) -> CvvdpOpaque {
    CvvdpOpaque::new(BACKEND, w as u32, h as u32, CvvdpParams::PLACEHOLDER).unwrap()
}

#[test]
fn every_entry_point_matches_tight_to_the_bit() {
    // 5x6 runs on an instance that reflect-pads below 8px.
    for (w, h) in [(32usize, 32usize), (33, 35), (5, 6)] {
        let (r, d) = (planes(w, h, 0), planes(w, h, 1));
        let (rs, ds) = (w + 5, w + 9);
        let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
        let mut a = opaque(w, h);
        let mut b = opaque(w, h);

        let (mut dm_a, mut dm_b) = (Vec::new(), Vec::new());
        let tight = a
            .compute_from_linear_planes(refs(&r), w, refs(&d), w, Some(&mut dm_a))
            .unwrap();
        let strided = b
            .compute_from_linear_planes(refs(&pr), rs, refs(&pd), ds, Some(&mut dm_b))
            .unwrap();
        assert_eq!(tight.value.to_bits(), strided.value.to_bits(), "{w}x{h}");
        assert_eq!(bits(&dm_a), bits(&dm_b), "{w}x{h} diffmap");

        a.warm_reference_from_linear_planes(refs(&r), w).unwrap();
        b.warm_reference_from_linear_planes(refs(&pr), rs).unwrap();
        let (mut dm_a, mut dm_b) = (Vec::new(), Vec::new());
        let tight = a
            .compute_with_warm_ref_from_linear_planes(refs(&d), w, Some(&mut dm_a))
            .unwrap();
        let strided = b
            .compute_with_warm_ref_from_linear_planes(refs(&pd), ds, Some(&mut dm_b))
            .unwrap();
        assert_eq!(
            tight.value.to_bits(),
            strided.value.to_bits(),
            "{w}x{h} warm"
        );
        assert_eq!(bits(&dm_a), bits(&dm_b), "{w}x{h} warm diffmap");

        // Scratch is reused across calls: a second distorted image must not see
        // the first one's samples.
        let d2 = planes(w, h, 2);
        let pd2 = padded(&d2, w, ds);
        let tight = a
            .compute_with_warm_ref_from_linear_planes(refs(&d2), w, None)
            .unwrap();
        let strided = b
            .compute_with_warm_ref_from_linear_planes(refs(&pd2), ds, None)
            .unwrap();
        assert_eq!(
            tight.value.to_bits(),
            strided.value.to_bits(),
            "{w}x{h} second"
        );

        // A tight view through the strided entry reads the same samples.
        let view = b
            .compute_with_warm_ref_from_linear_planes(refs(&d2), w, None)
            .unwrap();
        assert_eq!(tight.value.to_bits(), view.value.to_bits(), "{w}x{h} view");
    }
}

#[test]
fn bad_geometry_is_a_dimension_mismatch() {
    let (w, h) = (33usize, 35usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
    let mut m = opaque(w, h);
    let mismatch = |e: cvvdp_gpu::Error| matches!(e, cvvdp_gpu::Error::DimensionMismatch { .. });
    for plane in 0..3 {
        let mut short = refs(&pr);
        short[plane] = &short[plane][..short[plane].len() - 1];
        let e = m
            .compute_from_linear_planes(short, rs, refs(&pd), ds, None)
            .unwrap_err();
        assert!(mismatch(e), "short ref {plane}");
        let mut short = refs(&pd);
        short[plane] = &short[plane][..short[plane].len() - 1];
        let e = m
            .compute_from_linear_planes(refs(&pr), rs, short, ds, None)
            .unwrap_err();
        assert!(mismatch(e), "short dist {plane}");
    }
    for stride in [w - 1, usize::MAX] {
        let e = m
            .compute_from_linear_planes(refs(&pr), stride, refs(&pd), ds, None)
            .unwrap_err();
        assert!(mismatch(e), "stride {stride}");
        let e = m
            .warm_reference_from_linear_planes(refs(&pr), stride)
            .unwrap_err();
        assert!(mismatch(e), "warm stride {stride}");
    }
}
