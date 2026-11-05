//! The strided linear-plane entry point gives the tight form's scores, to the bit:
//! padding holds garbage and is never read, and exact-minimum buffers are accepted.

#![cfg(all(feature = "internals", any(feature = "cuda", feature = "wgpu")))]

use butteraugli_gpu::{Backend, ButteraugliOpaque, ButteraugliParams, MemoryMode};

#[cfg(feature = "cuda")]
const BACKEND: Backend = Backend::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
const BACKEND: Backend = Backend::Wgpu;

type Planes = [Vec<f32>; 3];

fn planes(w: usize, h: usize, seed: usize) -> Planes {
    std::array::from_fn(|c| {
        (0..w * h)
            .map(|i| {
                let t = (i % w + (i / w) * 2 + c * 11 + seed * 7) as f32;
                0.1 + 0.8 * t / (w + 2 * h + 40) as f32
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

fn opaque(w: usize, h: usize) -> ButteraugliOpaque {
    ButteraugliOpaque::new_with_memory_mode(
        BACKEND,
        w as u32,
        h as u32,
        ButteraugliParams::default(),
        MemoryMode::Full,
    )
    .unwrap()
}

#[test]
fn linear_planes_match_tight_to_the_bit() {
    for (w, h) in [(64usize, 64usize), (65, 67)] {
        let (r, d) = (planes(w, h, 0), planes(w, h, 1));
        let (rs, ds) = (w + 5, w + 9);
        let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
        let (tight, tight_p3) = opaque(w, h)
            .compute_from_linear_planes(refs(&r), w, refs(&d), w)
            .unwrap();
        let mut m = opaque(w, h);
        let (strided, strided_p3) = m
            .compute_from_linear_planes(refs(&pr), rs, refs(&pd), ds)
            .unwrap();
        assert_eq!(tight.value.to_bits(), strided.value.to_bits(), "{w}x{h}");
        assert_eq!(tight_p3.to_bits(), strided_p3.to_bits(), "{w}x{h} pnorm3");
        // The reused scratch and a tight view both give the same again.
        let (again, _) = m
            .compute_from_linear_planes(refs(&r), w, refs(&pd), ds)
            .unwrap();
        assert_eq!(
            tight.value.to_bits(),
            again.value.to_bits(),
            "{w}x{h} again"
        );
    }
}

#[test]
fn bad_geometry_is_a_dimension_mismatch() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
    let mut m = opaque(w, h);
    for plane in 0..3 {
        let mut short = refs(&pd);
        short[plane] = &short[plane][..short[plane].len() - 1];
        let e = m
            .compute_from_linear_planes(refs(&pr), rs, short, ds)
            .unwrap_err();
        assert!(matches!(
            e,
            butteraugli_gpu::Error::DimensionMismatch { .. }
        ));
    }
    for stride in [w - 1, usize::MAX] {
        let e = m
            .compute_from_linear_planes(refs(&pr), stride, refs(&pd), ds)
            .unwrap_err();
        assert!(matches!(
            e,
            butteraugli_gpu::Error::DimensionMismatch { .. }
        ));
    }
}
