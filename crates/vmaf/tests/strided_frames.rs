//! A strided frame scores exactly as its tight form: padding holds samples out of
//! range and is never read, and exact-minimum buffers are accepted. Covers the v0
//! and v1 scorers and streams at 8 and 10 bits, and the geometry refusals.

use vmaf::{
    ModelVariant, VmafV0Scorer, VmafV0Stream, VmafV0Variant, VmafV1Scorer, VmafV1Stream,
    Yuv420Frame,
};

const W: usize = 320;
const H: usize = 240;

struct Planes {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

fn sample(x: usize, y: usize, t: usize, c: usize, max: u32) -> u16 {
    (((x * 3 + y * 5 + (x * y) / 7 + t * 11 + c * 37) % 200 + 20) as u32 * max / 255) as u16
}

fn plane(w: usize, h: usize, t: usize, c: usize, max: u32) -> Vec<u16> {
    (0..w * h)
        .map(|i| sample(i % w, i / w, t, c, max))
        .collect()
}

fn planes(t: usize, depth: u8) -> Planes {
    let max = (1u32 << depth) - 1;
    Planes {
        y: plane(W, H, t, 0, max),
        u: plane(W / 2, H / 2, t, 1, max),
        v: plane(W / 2, H / 2, t, 2, max),
    }
}

fn padded(p: &[u16], w: usize, stride: usize) -> Vec<u16> {
    let h = p.len() / w;
    let mut out = vec![0xFFFF; (h - 1) * stride + w];
    for (row, line) in p.chunks_exact(w).enumerate() {
        out[row * stride..row * stride + w].copy_from_slice(line);
    }
    out
}

const YS: usize = W + 7;
const US: usize = W / 2 + 3;
const VS: usize = W / 2 + 11;

fn padded_planes(p: &Planes) -> Planes {
    Planes {
        y: padded(&p.y, W, YS),
        u: padded(&p.u, W / 2, US),
        v: padded(&p.v, W / 2, VS),
    }
}

fn tight(p: &Planes) -> Yuv420Frame<'_> {
    Yuv420Frame {
        y: &p.y,
        u: &p.u,
        v: &p.v,
        strides: [W, W / 2, W / 2],
    }
}

fn strided(p: &Planes) -> Yuv420Frame<'_> {
    Yuv420Frame {
        y: &p.y,
        u: &p.u,
        v: &p.v,
        strides: [YS, US, VS],
    }
}

#[test]
fn v1_scorer_and_stream_match_tight_exactly() {
    for depth in [8u8, 10] {
        let reference: Vec<_> = (0..3).map(|t| planes(t, depth)).collect();
        let distorted: Vec<_> = (0..3).map(|t| planes(t + 1, depth)).collect();
        let rp: Vec<_> = reference.iter().map(padded_planes).collect();
        let dp: Vec<_> = distorted.iter().map(padded_planes).collect();
        let rt: Vec<_> = reference.iter().map(tight).collect();
        let dt: Vec<_> = distorted.iter().map(tight).collect();
        let rs: Vec<_> = rp.iter().map(strided).collect();
        let ds: Vec<_> = dp.iter().map(strided).collect();
        let variant = ModelVariant::Standard1080p;
        let scorer = VmafV1Scorer::new(W, H, depth, variant).unwrap();
        let a = scorer.score(&rt, &dt).unwrap();
        let b = scorer.score(&rs, &ds).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.score.to_bits(), y.score.to_bits(), "bd{depth}");
        }
        let run = |r: &[Yuv420Frame<'_>], d: &[Yuv420Frame<'_>]| {
            let mut stream = VmafV1Stream::new(W, H, depth, variant).unwrap();
            let mut out = Vec::new();
            for (x, y) in r.iter().zip(d) {
                out.extend(stream.push(*x, *y).unwrap());
            }
            out.extend(stream.finish().unwrap());
            out.iter().map(|r| r.score.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(run(&rt, &dt), run(&rs, &ds), "bd{depth} stream");
    }
}

#[test]
fn v0_scorer_and_stream_match_tight_exactly() {
    for depth in [8u8, 10] {
        let reference: Vec<_> = (0..3).map(|t| planes(t, depth)).collect();
        let distorted: Vec<_> = (0..3).map(|t| planes(t + 1, depth)).collect();
        let rp: Vec<_> = reference.iter().map(padded_planes).collect();
        let dp: Vec<_> = distorted.iter().map(padded_planes).collect();
        let rt: Vec<_> = reference.iter().map(tight).collect();
        let dt: Vec<_> = distorted.iter().map(tight).collect();
        let rs: Vec<_> = rp.iter().map(strided).collect();
        let ds: Vec<_> = dp.iter().map(strided).collect();
        let variant = VmafV0Variant::Standard;
        let scorer = VmafV0Scorer::new(W, H, depth, variant).unwrap();
        let a = scorer.score(&rt, &dt).unwrap();
        let b = scorer.score(&rs, &ds).unwrap();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.score.to_bits(), y.score.to_bits(), "bd{depth}");
        }
        let run = |r: &[Yuv420Frame<'_>], d: &[Yuv420Frame<'_>]| {
            let mut stream = VmafV0Stream::new(W, H, depth, variant).unwrap();
            let mut out = Vec::new();
            for (x, y) in r.iter().zip(d) {
                out.extend(stream.push(*x, *y).unwrap());
            }
            out.extend(stream.finish().unwrap());
            out.iter().map(|r| r.score.to_bits()).collect::<Vec<_>>()
        };
        assert_eq!(run(&rt, &dt), run(&rs, &ds), "bd{depth} stream");
    }
}

#[test]
fn v0_range_check_reads_image_samples_only() {
    let p = padded_planes(&planes(0, 8));
    let good = strided(&p);
    let scorer = VmafV0Scorer::new(W, H, 8, VmafV0Variant::Standard).unwrap();
    scorer.score(&[good], &[good]).unwrap();
    let mut bad = padded_planes(&planes(0, 8));
    bad.y[YS + 3] = 300;
    assert!(scorer.score(&[strided(&bad)], &[good]).is_err());
}

#[test]
fn bad_geometry_is_refused() {
    let p = padded_planes(&planes(0, 8));
    let good = strided(&p);
    let v1 = VmafV1Scorer::new(W, H, 8, ModelVariant::Standard1080p).unwrap();
    let v0 = VmafV0Scorer::new(W, H, 8, VmafV0Variant::Standard).unwrap();
    let cases = [
        Yuv420Frame {
            y: &p.y[..p.y.len() - 1],
            ..good
        },
        Yuv420Frame {
            u: &p.u[..p.u.len() - 1],
            ..good
        },
        Yuv420Frame {
            v: &p.v[..p.v.len() - 1],
            ..good
        },
        Yuv420Frame {
            strides: [W - 1, US, VS],
            ..good
        },
        Yuv420Frame {
            strides: [YS, W / 2 - 1, VS],
            ..good
        },
        Yuv420Frame {
            strides: [YS, US, W / 2 - 1],
            ..good
        },
        Yuv420Frame {
            strides: [usize::MAX, US, VS],
            ..good
        },
    ];
    for (i, frame) in cases.iter().enumerate() {
        assert!(v1.score(&[*frame], &[good]).is_err(), "v1 case {i}");
        assert!(v0.score(&[good], &[*frame]).is_err(), "v0 case {i}");
    }
}
