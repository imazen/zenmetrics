//! Multi-fixture parity probe vs pycvvdp v0.5.4 goldens.
//!
//! Walks every synth fixture in the goldens manifest
//! (`scripts/cvvdp_goldens/pycvvdp_synth_goldens.json`) and prints
//! `gpu_jod`, `host_scalar_jod`, the pycvvdp golden, and absolute
//! diffs side-by-side — a fast way to confirm manifest parity
//! hasn't regressed after a kernel edit. Exits non-zero if any
//! abs diff exceeds 0.005 JOD — the canonical tolerance the
//! manifest-parity tests use.
//!
//! Originally written (tick 191) as a single-fixture
//! `chroma_shift_drift_probe.rs` while investigating the 0.117 JOD
//! divergence chased through ticks 191-204. Ticks 204 (baseband
//! CSF rho) and 206 (gauss-reduce parity bug) closed every drift
//! to f32 precision; tick 210 expanded the probe to all 6 manifest
//! fixtures, and tick 229 renamed it to reflect what it actually
//! does. References to the old name in
//! `CHROMA_DRIFT_INVESTIGATION.md` are preserved as historical
//! context — the verification command in this file is the
//! authoritative one.
//!
//! Run:
//!     cargo run --release --example manifest_parity_probe \
//!         -p cvvdp-gpu --features cuda

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

#[path = "../tests/it/common/mod.rs"]
mod common;

use common::{Backend, synth_pair_ref};

const TOLERANCE: f32 = 0.005;

fn synth_chroma_shift_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_ref(w, h);
    let mut d = r.clone();
    for i in (1..(w * h * 3)).step_by(3) {
        d[i] = (i16::from(r[i]) + 16).clamp(0, 255) as u8;
    }
    (r, d)
}

fn synth_blur3x1_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_ref(w, h);
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let x1 = (x + 1) % w;
            let x2 = (x + 2) % w;
            for c in 0..3 {
                let a = u16::from(r[(y * w + x) * 3 + c]);
                let b = u16::from(r[(y * w + x1) * 3 + c]);
                let cc = u16::from(r[(y * w + x2) * 3 + c]);
                d[(y * w + x) * 3 + c] = ((a + b + cc) / 3) as u8;
            }
        }
    }
    (r, d)
}

fn synth_blur1x3_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_ref(w, h);
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let y1 = (y + 1) % h;
            let y2 = (y + 2) % h;
            for c in 0..3 {
                let a = u16::from(r[(y * w + x) * 3 + c]);
                let b = u16::from(r[(y1 * w + x) * 3 + c]);
                let cc = u16::from(r[(y2 * w + x) * 3 + c]);
                d[(y * w + x) * 3 + c] = ((a + b + cc) / 3) as u8;
            }
        }
    }
    (r, d)
}

fn synth_odd_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    common::synth_pair_odd_dim_with_offset_dist(w, h)
}

fn synth_noise_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    // pycvvdp `synth_pair_256_noise` (bench_12mp_cuda.py:102):
    //   noise[y,x,c] = ((x * 73 + y * 137 + c * 211) % 64) - 32
    //   dist[y,x,c]  = clamp(ref[y,x,c] + noise[y,x,c], 0, 255)
    // Pure integer arithmetic — bit-stable across NumPy + Rust.
    let r = synth_pair_ref(w, h);
    let mut d = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let noise = (((x * 73 + y * 137 + c * 211) % 64) as i32) - 32;
                let v = i32::from(r[(y * w + x) * 3 + c]) + noise;
                d[(y * w + x) * 3 + c] = v.clamp(0, 255) as u8;
            }
        }
    }
    (r, d)
}

fn synth_pair_12mp(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    // pycvvdp `synth_pair_12mp` (bench_12mp_cuda.py:38) — same ref
    // construction as synth_pair_ref, with R-8 / G-4 / B+12 saturated
    // distortion. Identical formula to synth_odd_pair, just at 12 MP.
    common::synth_pair_with_offset_dist(w, h)
}

struct Fixture {
    name: &'static str,
    w: u32,
    h: u32,
    builder: fn(usize, usize) -> (Vec<u8>, Vec<u8>),
}

fn run_fixture(f: &Fixture) -> bool {
    let geom = DisplayGeometry::STANDARD_4K;
    let display = DisplayModel::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_b, dist_b) = (f.builder)(f.w as usize, f.h as usize);
    let golden = common::pycvvdp_synth_golden_jod(f.name);

    let client = Backend::client(&Default::default());
    let mut cvvdp =
        Cvvdp::<Backend>::new(client, f.w, f.h, CvvdpParams::PLACEHOLDER).expect("Cvvdp::new");
    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod");
    let host_jod = predict_jod_still_3ch(&ref_b, &dist_b, f.w as usize, f.h as usize, display, ppd);
    let d_gpu_golden = (gpu_jod - golden).abs();
    let d_host_golden = (host_jod - golden).abs();
    let d_gpu_host = (gpu_jod - host_jod).abs();
    let pass = d_gpu_golden < TOLERANCE && d_host_golden < TOLERANCE;
    println!(
        "{:30}  {:>5}x{:<5}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {}",
        f.name,
        f.w,
        f.h,
        gpu_jod,
        host_jod,
        golden,
        d_gpu_golden,
        d_host_golden,
        d_gpu_host,
        if pass { "ok" } else { "FAIL" }
    );
    pass
}

fn main() {
    println!(
        "{:30}  {:>11}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  status",
        "fixture", "dims", "gpu_jod", "host_jod", "pycvvdp", "d_gpu", "d_host", "d_g-h",
    );
    // Bit-stable dist constructions across pycvvdp's
    // `scripts/cvvdp_goldens/bench_12mp_cuda.py` and Rust. Every
    // synth fixture in the goldens manifest now has a matching
    // builder here.
    let fixtures: &[Fixture] = &[
        Fixture {
            name: "synth_73x91_odd",
            w: 73,
            h: 91,
            builder: synth_odd_pair,
        },
        Fixture {
            name: "synth_256x256_chroma_shift",
            w: 256,
            h: 256,
            builder: synth_chroma_shift_pair,
        },
        Fixture {
            name: "synth_256x256_blur3x1",
            w: 256,
            h: 256,
            builder: synth_blur3x1_pair,
        },
        Fixture {
            name: "synth_256x256_blur1x3",
            w: 256,
            h: 256,
            builder: synth_blur1x3_pair,
        },
        Fixture {
            name: "synth_256x256_noise",
            w: 256,
            h: 256,
            builder: synth_noise_pair,
        },
        Fixture {
            name: "synth_4000x3000",
            w: 4000,
            h: 3000,
            builder: synth_pair_12mp,
        },
    ];

    let mut all_pass = true;
    for f in fixtures {
        if !run_fixture(f) {
            all_pass = false;
        }
    }
    println!();
    if all_pass {
        println!(
            "STATUS: all {} fixtures within {TOLERANCE} JOD of pycvvdp golden. \
             Manifest parity intact.",
            fixtures.len()
        );
    } else {
        println!(
            "STATUS: at least one fixture diverged > {TOLERANCE} JOD. \
             See per-fixture rows above."
        );
        std::process::exit(1);
    }
}
