//! Per-stage wall-time breakdown of `hdrvdp()` on a synthetic pair.
//!
//! ```text
//! cargo run --release -p hdrvdp --example stage_probe -- <side> [reps]
//! ```

#![forbid(unsafe_code)]

use hdrvdp::Params;
use hdrvdp::bands::decompose;
use hdrvdp::display::{ColorEncoding, to_nits};
use hdrvdp::masking::{self, diff_mask};
use hdrvdp::pathway::{surround_per_channel, visual_pathway};
use hdrvdp::photoreceptor::Photoreceptor;
use hdrvdp::pool::visibility;
use hdrvdp::spectral::{emission_spectra, lmsr_matrix};
use std::hint::black_box;
use std::time::Instant;

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

fn main() {
    let mut args = std::env::args().skip(1);
    let side: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1024);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);

    let reference = hdr_field(side, side, 0xfeed);
    let test = hdr_field(side, side, 0xbeef);
    let par = Params::new(30.0);
    let channels = 1;

    // One warm run for lazy init + a score() parity check.
    let full = hdrvdp::hdrvdp(
        black_box(&test),
        black_box(&reference),
        side,
        side,
        ColorEncoding::Luminance,
        &par,
    )
    .unwrap();
    let s = hdrvdp::score(
        black_box(&test),
        black_box(&reference),
        side,
        side,
        ColorEncoding::Luminance,
        &par,
    )
    .unwrap();
    assert_eq!(s, full.q, "score() must equal hdrvdp().q bit-for-bit");
    println!("score() == hdrvdp().q: {s:.10}");

    let names = [
        "to_nits",
        "surround+lmsr",
        "pathway_ref",
        "pathway_test",
        "decompose_ref",
        "decompose_test",
        "diff_mask+l_adapt",
        "masking",
        "visibility",
    ];
    let mut acc = [0.0f64; 9];

    for _ in 0..reps {
        let t0 = Instant::now();
        let ref_nits = to_nits(&reference, side, side, ColorEncoding::Luminance).unwrap();
        let test_nits = to_nits(&test, side, side, ColorEncoding::Luminance).unwrap();
        acc[0] += t0.elapsed().as_secs_f64();

        let t = Instant::now();
        let surround = surround_per_channel(&ref_nits, channels, par.surround_l);
        let lmsr = lmsr_matrix(&emission_spectra(
            ColorEncoding::Luminance.spectra(),
            channels,
        ));
        let pn = Photoreceptor::new(&par);
        acc[1] += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let path_ref = visual_pathway(&ref_nits, side, side, &par, &pn, &lmsr, &surround);
        acc[2] += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let path_test = visual_pathway(&test_nits, side, side, &par, &pn, &lmsr, &surround);
        acc[3] += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let (bands_ref, pad) = decompose(&path_ref, &par, None);
        acc[4] += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (bands_test, _) = decompose(&path_test, &par, Some(pad));
        acc[5] += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let l_adapt: Vec<f64> = path_ref
            .l_adapt
            .iter()
            .zip(&path_test.l_adapt)
            .map(|(a, b)| 0.5 * (a + b))
            .collect();
        let dm = diff_mask(&test_nits, &ref_nits, channels);
        acc[6] += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let m = masking::run(&bands_test, &bands_ref, &l_adapt, &dm, &par);
        acc[7] += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let v = visibility(&m.d_bands, &par);
        acc[8] += t.elapsed().as_secs_f64();
        black_box(&v);
    }

    let total: f64 = acc.iter().sum();
    println!("{side}x{side}, {reps} reps, pipeline total {total:.3} s avg");
    for (n, a) in names.iter().zip(&acc) {
        println!(
            "  {n:<16} {:>8.1} ms  {:>5.1}%",
            a / reps as f64 * 1e3,
            a / total * 100.0
        );
    }

    // And the score-only path end to end.
    let t = Instant::now();
    for _ in 0..reps {
        black_box(
            hdrvdp::score(
                &test,
                &reference,
                side,
                side,
                ColorEncoding::Luminance,
                &par,
            )
            .unwrap(),
        );
    }
    println!(
        "score() total: {:.1} ms avg",
        t.elapsed().as_secs_f64() / reps as f64 * 1e3
    );
}
