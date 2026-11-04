//! The strided linear-plane entry points give the tight form's score and diffmap,
//! to the bit: padding holds garbage and is never read, reference and distorted
//! images may differ in stride, and exact-minimum buffers are accepted.

use cvvdp::{Cvvdp, CvvdpParams, Error};

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

#[test]
fn linear_planes_match_tight_to_the_bit() {
    for (w, h) in [(64usize, 64usize), (65, 67), (33, 27)] {
        let (r, d) = (planes(w, h, 0), planes(w, h, 1));
        let (rs, ds) = (w + 5, w + 9);
        let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
        let new = || Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();

        let tight = new()
            .score_from_linear_planes(refs(&r), w, refs(&d), w)
            .unwrap();
        let strided = new()
            .score_from_linear_planes(refs(&pr), rs, refs(&pd), ds)
            .unwrap();
        assert_eq!(tight.to_bits(), strided.to_bits(), "{w}x{h}");

        let (mut dm_a, mut dm_b) = (Vec::new(), Vec::new());
        let tight = new()
            .score_from_linear_planes_with_diffmap(refs(&r), w, refs(&d), w, &mut dm_a)
            .unwrap();
        let strided = new()
            .score_from_linear_planes_with_diffmap(refs(&r), w, refs(&pd), ds, &mut dm_b)
            .unwrap();
        assert_eq!(tight.to_bits(), strided.to_bits(), "{w}x{h} diffmap score");
        assert_eq!(bits(&dm_a), bits(&dm_b), "{w}x{h} diffmap");
    }
}

#[test]
fn bad_geometry_is_a_plane_shape_mismatch() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
    let mut m = Cvvdp::new(w as u32, h as u32, CvvdpParams::default()).unwrap();
    for plane in 0..3 {
        let mut short = refs(&pr);
        short[plane] = &short[plane][..short[plane].len() - 1];
        assert!(matches!(
            m.score_from_linear_planes(short, rs, refs(&pd), ds),
            Err(Error::PlaneShapeMismatch { .. })
        ));
        let mut short = refs(&pd);
        short[plane] = &short[plane][..short[plane].len() - 1];
        assert!(matches!(
            m.score_from_linear_planes(refs(&pr), rs, short, ds),
            Err(Error::PlaneShapeMismatch { .. })
        ));
    }
    for stride in [w - 1, usize::MAX] {
        assert!(matches!(
            m.score_from_linear_planes(refs(&pr), stride, refs(&pd), ds),
            Err(Error::PlaneShapeMismatch { .. })
        ));
    }
}
