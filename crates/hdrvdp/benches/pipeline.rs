//! zenbench suite for the HDR-VDP-2.2 pipeline: end-to-end scoring at the
//! sizes where interleaved rounds are affordable, plus the kernels profiling
//! showed hottest (corr_dn / up_conv / fft2 / imresize) for regression gating.
//!
//! Larger sizes (1024², 4096²) are swept with `examples/perf_probe` instead —
//! a single 4096² score is ~10 s even optimised, which interleaved rounds
//! would multiply into hours. The size sweep + `α + β·pixels` fits live in
//! `benchmarks/hdrvdp_perf_2026-08-29.md`.
//!
//! Run: `cargo bench -p hdrvdp --bench pipeline`

use hdrvdp::spyr::Band;
use hdrvdp::{ColorEncoding, Params, hdrvdp};
use zenbench::prelude::*;

/// Deterministic textured HDR field (same generator as `examples/perf_probe`).
fn hdr_field(w: usize, h: usize, seed: u64) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    let mut s = seed | 1;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let base = 60.0
                * (1.0 + 0.4 * (2.0 * pi * x / 17.0).sin() * (2.0 * pi * y / 23.0).cos())
                + 6.0 * (rng() - 0.5);
            let (cx, cy) = (w as f64 * 0.7, h as f64 * 0.3);
            let r2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (0.02 * (w * h) as f64 + 1.0);
            (base + 3000.0 * (-r2).exp()).max(0.05)
        })
        .collect()
}

fn distorted(reference: &[f64], w: usize) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    reference
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            v * (1.0 + 0.03 * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin())
        })
        .collect()
}

fn bench_end_to_end(suite: &mut Suite) {
    for side in [64usize, 256] {
        let reference = hdr_field(side, side, 0xfeed);
        let test = distorted(&reference, side);
        suite.group(format!("hdrvdp_pair/{side}x{side}"), move |g| {
            g.throughput(Throughput::Elements((side * side) as u64));
            g.bench("score", move |b| {
                let (r, t) = (reference.clone(), test.clone());
                let par = Params::new(30.0);
                b.iter(move || hdrvdp(&t, &r, side, side, ColorEncoding::Luminance, &par).unwrap())
            });
        });
    }
}

fn bench_kernels(suite: &mut Suite) {
    use hdrvdp::sp3_filters::{BFILTS, LOFILT};
    const SIDE: usize = 256;

    suite.group("corr_dn/256x256", move |g| {
        g.throughput(Throughput::Elements((SIDE * SIDE) as u64));
        g.bench("bfilt_9x9_step1", move |b| {
            let im = hdr_field(SIDE, SIDE, 0xbead);
            let f: Vec<f64> = BFILTS[0].iter().flatten().copied().collect();
            b.iter(move || hdrvdp::spyr::corr_dn(&im, SIDE, SIDE, &f, 9, 1))
        });
        g.bench("lofilt_17x17_step2", move |b| {
            let im = hdr_field(SIDE, SIDE, 0xbead);
            let f: Vec<f64> = LOFILT.iter().flatten().copied().collect();
            b.iter(move || hdrvdp::spyr::corr_dn(&im, SIDE, SIDE, &f, 17, 2))
        });
    });

    suite.group("up_conv/256x256", move |g| {
        g.throughput(Throughput::Elements((SIDE * SIDE) as u64));
        g.bench("bfilt_9x9_step1", move |b| {
            let im = hdr_field(SIDE, SIDE, 0xbead);
            let f: Vec<f64> = BFILTS[0].iter().flatten().copied().collect();
            let band: Band = hdrvdp::spyr::corr_dn(&im, SIDE, SIDE, &f, 9, 1);
            b.iter(move || {
                let mut res = vec![0.0; SIDE * SIDE];
                hdrvdp::spyr::up_conv(&band, &f, 9, 1, SIDE, SIDE, &mut res);
                res
            })
        });
    });

    suite.group("fft/512x512", move |g| {
        // The metric transforms 2×-padded planes, so 512 is the fft size a
        // 256² image pays.
        g.throughput(Throughput::Elements((512 * 512) as u64));
        g.bench("fft2_complex", move |b| {
            let src: Vec<hdrvdp::fft::Complex> = hdr_field(512, 512, 9)
                .into_iter()
                .map(|v| hdrvdp::fft::Complex::new(v, 0.0))
                .collect();
            b.with_input(move || src.clone()).run(|mut buf| {
                hdrvdp::fft::fft2(&mut buf, 512, 512);
                buf
            })
        });
    });

    suite.group("imresize/512->181", move |g| {
        g.throughput(Throughput::Elements((512 * 512) as u64));
        g.bench("downscale", move |b| {
            let src = hdr_field(512, 512, 11);
            b.iter(move || hdrvdp::resize::imresize(&src, 512, 512, 181, 181))
        });
    });
}

zenbench::main!(bench_end_to_end, bench_kernels);
