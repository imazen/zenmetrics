//! Multi-fixture parity probe vs pycvvdp v0.5.4 goldens.
//!
//! Originally written (tick 191) as a single-fixture
//! `chroma_shift` drift probe while investigating the 0.117 JOD
//! divergence chased through ticks 191-204. Now that ticks 204
//! (baseband CSF rho) and 206 (gauss-reduce parity bug) closed
//! the chroma_shift AND 73×91 odd-dim drifts to f32 precision,
//! this example walks every synth fixture in the goldens manifest
//! and reports the diff side-by-side — a fast way to confirm
//! manifest parity hasn't regressed after a kernel edit.
//!
//! Each row prints `gpu_jod`, `host_scalar_jod`, the pycvvdp
//! golden, and absolute diffs (vs golden, GPU vs host). Exits
//! non-zero if any abs diff exceeds 0.005 JOD — the canonical
//! tolerance used by the manifest-parity tests.
//!
//! Run:
//!     cargo run --release --example chroma_shift_drift_probe \
//!         -p cvvdp-gpu --features cuda

#![cfg(any(feature = "cuda", feature = "wgpu", feature = "hip"))]

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::host_scalar::predict_jod_still_3ch;
use cvvdp_gpu::params::{CvvdpParams, DisplayGeometry, DisplayModel};

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "hip", not(feature = "cuda"), not(feature = "wgpu")))]
type Backend = cubecl::hip::HipRuntime;

const TOLERANCE: f32 = 0.005;

/// Build the same byte-level synth pair `bench_12mp_cuda.py`'s
/// `synth_pair_12mp` constructs (called for the 4000×3000 and
/// 256×256 fixtures). Pure xx/yy modular arithmetic — no PRNG.
fn synth_pair_ref(w: usize, h: usize) -> Vec<u8> {
    let n = w * h * 3;
    let mut b = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let r = (((x * 17 + y * 5) % 251) as u8).wrapping_add(40);
            let g = (((x * 11 + y * 13) % 247) as u8).wrapping_add(40);
            let bb = (((x * 7 + y * 19) % 241) as u8).wrapping_add(40);
            let i = (y * w + x) * 3;
            b[i] = r;
            b[i + 1] = g;
            b[i + 2] = bb;
        }
    }
    b
}

fn synth_chroma_shift_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let r = synth_pair_ref(w, h);
    let mut d = r.clone();
    for i in (1..(w * h * 3)).step_by(3) {
        d[i] = (r[i] as i16 + 16).clamp(0, 255) as u8;
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
                let a = r[(y * w + x) * 3 + c] as u16;
                let b = r[(y * w + x1) * 3 + c] as u16;
                let cc = r[(y * w + x2) * 3 + c] as u16;
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
                let a = r[(y * w + x) * 3 + c] as u16;
                let b = r[(y1 * w + x) * 3 + c] as u16;
                let cc = r[(y2 * w + x) * 3 + c] as u16;
                d[(y * w + x) * 3 + c] = ((a + b + cc) / 3) as u8;
            }
        }
    }
    (r, d)
}

fn synth_odd_pair(w: usize, h: usize) -> (Vec<u8>, Vec<u8>) {
    let n = w * h * 3;
    let mut r = vec![0u8; n];
    let mut d = vec![0u8; n];
    for y in 0..h {
        for x in 0..w {
            let rr = ((x * 8) % 256) as u8;
            let g = ((y * 8) % 256) as u8;
            let bb = (((x + y) * 4) % 256) as u8;
            let i = (y * w + x) * 3;
            r[i] = rr;
            r[i + 1] = g;
            r[i + 2] = bb;
            d[i] = rr.saturating_sub(8);
            d[i + 1] = g.saturating_sub(4);
            d[i + 2] = bb.saturating_add(12);
        }
    }
    (r, d)
}

struct Fixture {
    name: &'static str,
    w: u32,
    h: u32,
    golden: f32,
    builder: fn(usize, usize) -> (Vec<u8>, Vec<u8>),
}

fn run_fixture(f: &Fixture) -> bool {
    let geom = DisplayGeometry::STANDARD_4K;
    let display = DisplayModel::STANDARD_4K;
    let ppd = geom.pixels_per_degree();
    let (ref_b, dist_b) = (f.builder)(f.w as usize, f.h as usize);

    let client = Backend::client(&Default::default());
    let mut cvvdp = Cvvdp::<Backend>::new(client, f.w, f.h, CvvdpParams::PLACEHOLDER)
        .expect("Cvvdp::new");
    let gpu_jod = cvvdp
        .compute_dkl_jod(&ref_b, &dist_b, ppd)
        .expect("compute_dkl_jod");
    let host_jod = predict_jod_still_3ch(
        &ref_b,
        &dist_b,
        f.w as usize,
        f.h as usize,
        display,
        ppd,
    );
    let d_gpu_golden = (gpu_jod - f.golden).abs();
    let d_host_golden = (host_jod - f.golden).abs();
    let d_gpu_host = (gpu_jod - host_jod).abs();
    let pass = d_gpu_golden < TOLERANCE && d_host_golden < TOLERANCE;
    println!(
        "{:30}  {:>5}x{:<5}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {:8.6}  {}",
        f.name,
        f.w,
        f.h,
        gpu_jod,
        host_jod,
        f.golden,
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
    // Fixtures whose dist construction is bit-stable across Python
    // (`scripts/cvvdp_goldens/bench_12mp_cuda.py`) and Rust. The 12 MP
    // and 256² noise fixtures use pycvvdp constructions that aren't
    // ported here yet; they're covered by the test-side parity
    // (tests/pipeline_color.rs).
    let fixtures: &[Fixture] = &[
        Fixture {
            name: "synth_73x91_odd",
            w: 73,
            h: 91,
            golden: 9.390370,
            builder: synth_odd_pair,
        },
        Fixture {
            name: "synth_256x256_chroma_shift",
            w: 256,
            h: 256,
            golden: 9.664865,
            builder: synth_chroma_shift_pair,
        },
        Fixture {
            name: "synth_256x256_blur3x1",
            w: 256,
            h: 256,
            golden: 8.441194,
            builder: synth_blur3x1_pair,
        },
        Fixture {
            name: "synth_256x256_blur1x3",
            w: 256,
            h: 256,
            golden: 8.124331,
            builder: synth_blur1x3_pair,
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
