//! `Metric`'s strided linear-plane entry points route to the tight forms' results,
//! and metrics with no linear path refuse alike.

use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind};

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

fn metric(kind: MetricKind, w: usize, h: usize) -> Metric {
    Metric::new_with_memory_mode(
        kind,
        BACKEND,
        w as u32,
        h as u32,
        crate::params_for(kind),
        MemoryMode::Full,
    )
    .unwrap()
}

#[test]
fn linear_metrics_match_tight_to_the_bit() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
    for kind in [MetricKind::Cvvdp, MetricKind::Butter, MetricKind::Zensim] {
        let tight = metric(kind, w, h)
            .compute_from_linear_planes(refs(&r), w, refs(&d), w)
            .unwrap();
        let strided = metric(kind, w, h)
            .compute_from_linear_planes(refs(&pr), rs, refs(&pd), ds)
            .unwrap();
        assert_eq!(tight.value.to_bits(), strided.value.to_bits(), "{kind:?}");

        let tight = metric(kind, w, h)
            .compute_from_linear_planes_multi(refs(&r), w, refs(&d), w)
            .unwrap();
        let strided = metric(kind, w, h)
            .compute_from_linear_planes_multi(refs(&pr), rs, refs(&pd), ds)
            .unwrap();
        assert_eq!(tight.metric_name, strided.metric_name, "{kind:?} multi");
        assert_eq!(tight.scores.len(), strided.scores.len(), "{kind:?} multi");
        for (a, b) in tight.scores.iter().zip(&strided.scores) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.value.to_bits(), b.value.to_bits(), "{kind:?} {}", a.name);
        }
    }
}

#[test]
fn metrics_with_no_linear_path_refuse_alike() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let mut m = metric(MetricKind::Ssim2, w, h);
    let tight = m
        .compute_from_linear_planes(refs(&r), w, refs(&d), w)
        .unwrap_err();
    let strided = m
        .compute_from_linear_planes(refs(&r), w, refs(&d), w)
        .unwrap_err();
    assert_eq!(tight.to_string(), strided.to_string());
    let multi = m
        .compute_from_linear_planes_multi(refs(&r), w, refs(&d), w)
        .unwrap_err();
    assert_eq!(tight.to_string(), multi.to_string());
}

#[test]
fn bad_geometry_is_refused() {
    let (w, h) = (65usize, 67usize);
    let (r, d) = (planes(w, h, 0), planes(w, h, 1));
    let (rs, ds) = (w + 5, w + 9);
    let (pr, pd) = (padded(&r, w, rs), padded(&d, w, ds));
    for kind in [MetricKind::Cvvdp, MetricKind::Butter, MetricKind::Zensim] {
        let mut m = metric(kind, w, h);
        let mut short = refs(&pd);
        short[0] = &short[0][..short[0].len() - 1];
        assert!(
            m.compute_from_linear_planes(refs(&pr), rs, short, ds)
                .is_err(),
            "{kind:?} short"
        );
        assert!(
            m.compute_from_linear_planes(refs(&pr), w - 1, refs(&pd), ds)
                .is_err(),
            "{kind:?} stride"
        );
    }
}
