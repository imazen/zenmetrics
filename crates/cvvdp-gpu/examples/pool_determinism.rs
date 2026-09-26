//! Run-to-run determinism probe for the GPU JOD (the pool reduction).
//!
//! Scores fixed (ref, dist) pairs `DET_REPS` times per (pair, size,
//! memory mode) in one process and reports how many distinct f32 bit
//! patterns came back. With a fixed-order pool every cell must show
//! `distinct=1`, and repeated processes must print the same bits.
//! With the old `Atomic<f32>::fetch_add` pool the low bits wander.
//!
//! Full mode also scores through `score_with_band_breakdown` and
//! counts distinct per-band score vectors (`q_distinct`): the pool
//! noise shows there even when a mild pair's JOD (≈ 10) rounds it
//! away.
//!
//! Uses only API that predates the deterministic pool, so the same
//! file builds against the old kernels for the negative control.
//!
//! Environment:
//! - `DET_PAIRS`  comma list of `noise` (two unrelated byte patterns, the
//!   zenmetrics-api `cancel.rs` pair; JOD ≈ 2) and `mild` (textured ref
//!   + ±12 hash offset; JOD ≈ 10) (default both)
//! - `DET_SIZES`  comma list of `WxH` (default `256x256,1000x750,1024x1024`)
//! - `DET_MODES`  comma list of `full`, `strip` (Mode E, warm ref),
//!   `pair` (Mode B) (default all three)
//! - `DET_REPS`   scores per cell (default 50)
//! - `DET_HBODY`  strip body rows for `strip` / `pair` (default 128)
//! - `DET_OUT`    optional TSV path: one row per score
//!   (`pair size mode rep jod bits qhash`; `qhash` is `-` outside Full)
//!
//! Stdout: one `CELL` line per (pair, size, mode):
//! `CELL <pair> <WxH> <mode> reps=<n> distinct=<k> q_distinct=<k|-> first=<jod> first_bits=<hex> min=<jod> max=<jod>`

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use std::io::Write;

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry};

#[path = "../tests/it/common/mod.rs"]
mod common;
use common::Backend;

fn env_or(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

/// `noise`: the zenmetrics-api `cancel.rs` pair, two unrelated byte
/// patterns (JOD ≈ 2). `mild`: textured reference (the parity fixtures'
/// `synth_pair_ref`) plus a fixed per-pixel hash offset in ±12 (JOD ≈ 10).
fn pair(kind: &str, w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    if kind == "noise" {
        let n = w * h * 3;
        let r = (0..n)
            .map(|i| ((i as u64).wrapping_mul(7919) & 0xFF) as u8)
            .collect();
        let d = (0..n)
            .map(|i| ((i as u64).wrapping_mul(2_147_483_647) & 0xFF) as u8)
            .collect();
        return (r, d);
    }
    assert_eq!(kind, "mild", "unknown DET_PAIRS entry {kind:?}");
    let r = common::synth_pair_ref(w, h);
    let d = r
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let mut x = (i as u32).wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
            x ^= x >> 15;
            x = x.wrapping_mul(0x2C1B_3C6D);
            x ^= x >> 12;
            let off = (x % 25) as i32 - 12;
            (i32::from(v) + off).clamp(0, 255) as u8
        })
        .collect();
    (r, d)
}

fn fnv(q: &[[f32; 3]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for v in q.iter().flatten() {
        for byte in v.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn main() {
    let pairs = env_or("DET_PAIRS", "noise,mild");
    let sizes = env_or("DET_SIZES", "256x256,1000x750,1024x1024");
    let modes = env_or("DET_MODES", "full,strip,pair");
    let reps: usize = env_or("DET_REPS", "50").parse().expect("DET_REPS");
    let h_body: u32 = env_or("DET_HBODY", "128").parse().expect("DET_HBODY");
    let mut out = std::env::var("DET_OUT")
        .ok()
        .map(|p| std::fs::File::create(p).expect("create DET_OUT"));

    let ppd = DisplayGeometry::STANDARD_4K.pixels_per_degree();
    let client = Backend::client(&Default::default());

    for kind in pairs.split(',') {
        for size in sizes.split(',') {
            let (w, h) = size.split_once('x').expect("size WxH");
            let (w, h): (u32, u32) = (w.parse().expect("W"), h.parse().expect("H"));
            let (r, d) = pair(kind, w as usize, h as usize);
            for mode in modes.split(',') {
                let mut jods: Vec<f32> = Vec::with_capacity(reps);
                let mut qhash: Vec<u64> = Vec::new();
                match mode {
                    "full" => {
                        let mut c =
                            Cvvdp::<Backend>::new(client.clone(), w, h, CvvdpParams::PLACEHOLDER)
                                .expect("Cvvdp::new");
                        for _ in 0..reps {
                            jods.push(c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod"));
                            let bb = c.score_with_band_breakdown(&r, &d).expect("band breakdown");
                            qhash.push(fnv(&bb.q_per_ch));
                        }
                    }
                    "strip" => {
                        let mut c = Cvvdp::<Backend>::new_strip(
                            client.clone(),
                            w,
                            h,
                            h_body,
                            CvvdpParams::PLACEHOLDER,
                        )
                        .expect("Cvvdp::new_strip");
                        c.warm_reference(&r).expect("warm_reference");
                        for _ in 0..reps {
                            jods.push(
                                c.compute_dkl_jod_with_warm_ref(&d, ppd)
                                    .expect("compute_dkl_jod_with_warm_ref"),
                            );
                        }
                    }
                    "pair" => {
                        let mut c = Cvvdp::<Backend>::new_strip_pair(
                            client.clone(),
                            w,
                            h,
                            h_body,
                            CvvdpParams::PLACEHOLDER,
                        )
                        .expect("Cvvdp::new_strip_pair");
                        for _ in 0..reps {
                            jods.push(c.compute_dkl_jod(&r, &d, ppd).expect("compute_dkl_jod"));
                        }
                    }
                    other => panic!("unknown DET_MODES entry {other:?}"),
                }
                if let Some(f) = out.as_mut() {
                    for (i, j) in jods.iter().enumerate() {
                        let q = qhash
                            .get(i)
                            .map_or_else(|| "-".to_string(), |h| format!("{h:016x}"));
                        writeln!(
                            f,
                            "{kind}\t{size}\t{mode}\t{i}\t{j:.9}\t{:08x}\t{q}",
                            j.to_bits()
                        )
                        .expect("write");
                    }
                }
                let mut bits: Vec<u32> = jods.iter().map(|j| j.to_bits()).collect();
                bits.sort_unstable();
                bits.dedup();
                qhash.sort_unstable();
                qhash.dedup();
                let q_distinct = if mode == "full" {
                    qhash.len().to_string()
                } else {
                    "-".into()
                };
                let min = jods.iter().copied().fold(f32::INFINITY, f32::min);
                let max = jods.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                println!(
                    "CELL {kind} {size} {mode} reps={reps} distinct={} q_distinct={q_distinct} first={:.9} first_bits={:08x} min={min:.9} max={max:.9}",
                    bits.len(),
                    jods[0],
                    jods[0].to_bits(),
                );
                std::io::stdout().flush().expect("flush");
            }
        }
    }
}
