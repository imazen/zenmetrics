//! Optimized MDSI against the straight-line evaluation of the paper's
//! equations: asserts bit-identical scores and reports best-of-N wall time.
//! Usage: `mdsi_vs_reference [output.tsv]`; plain GMSD is timed alongside.
#![allow(clippy::too_many_arguments)]
extern crate alloc;
#[path = "../src/mdsi.rs"]
#[allow(dead_code)]
mod mdsi;
use gmsd::{Error, Result};
const BAND_ROWS: usize = 64;
fn check_rgb8(rgb: &[u8], w: usize, h: usize, stride: usize) -> Result<()> {
    let mut gray = vec![0.0; w * h];
    gmsd::rgb8_to_gray(rgb, w, h, stride, &mut gray)
}
fn best_ms<F: FnMut() -> f64>(mut f: F, reps: usize) -> (f64, f64) {
    let (mut best, mut v) = (f64::MAX, 0.0);
    for _ in 0..reps {
        let t = std::time::Instant::now();
        v = f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, v)
}
fn main() -> std::io::Result<()> {
    use std::io::Write;
    let mut out: Box<dyn Write> = match std::env::args().nth(1) {
        Some(p) => Box::new(std::fs::File::create(p)?),
        None => Box::new(std::io::stdout()),
    };
    writeln!(
        out,
        "size\tthreads\treference_ms\toptimized_ms\tspeedup\tplain_gmsd_ms"
    )?;
    for size in [64usize, 256, 1024, 4096] {
        let mut s = 7u32;
        let r: Vec<u8> = (0..size * size * 3)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 24) as u8
            })
            .collect();
        let d: Vec<u8> = r
            .iter()
            .enumerate()
            .map(|(i, &v)| if i % 3 == 0 { v / 2 } else { v })
            .collect();
        for threads in [1usize, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let reps = if size >= 4096 { 3 } else { 8 };
            let params = mdsi::Params::default();
            let (t_ref, v_ref) = best_ms(
                || {
                    mdsi::reference::score_and_map(&r, &d, size, size, size * 3, &params)
                        .unwrap()
                        .0
                },
                reps,
            );
            let (t_fast, v_fast) = pool.install(|| {
                best_ms(
                    || gmsd::mdsi_rgb8(&r, &d, size, size, size * 3).unwrap(),
                    reps,
                )
            });
            let (t_gmsd, _) = pool.install(|| {
                best_ms(
                    || gmsd::gmsd_rgb8(&r, &d, size, size, size * 3).unwrap().gmsd,
                    reps,
                )
            });
            assert_eq!(v_ref.to_bits(), v_fast.to_bits(), "{size} t{threads}");
            writeln!(
                out,
                "{size}\t{threads}\t{t_ref:.3}\t{t_fast:.3}\t{:.2}\t{t_gmsd:.3}",
                t_ref / t_fast
            )?;
        }
    }
    Ok(())
}
