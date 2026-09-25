//! VMAF wall-time + thread-scaling matrix: v0.6.1 and v1 across
//! {512², 1024², 4K, 8K} × {serial, T4, T8} on synthetic 8-bit 4:2:0 frames.
//!
//!   cargo run --release -p vmaf --features simd,avx512,parallel \
//!       --example wall_matrix -- <size_label> [v0|v1|both]
//!   size_label: 512 1024 4K 8K  (or `all` for the full matrix)
//!
//! `heap <size>` prints per-phase allocations once — run under heaptrack.
//! Frame-parallel scoring uses Vmaf*Scorer::with_threads (rayon over frames,
//! the same axis libvmaf's --threads uses), so NFRAMES >= 8 keeps T8 fed.

use std::env;
use std::hint::black_box;
use std::time::Instant;
use vmaf::{ModelVariant, VmafV0Scorer, VmafV0Variant, VmafV1Scorer, Yuv420Frame};

const NFRAMES: usize = 8;
const ROUNDS: usize = 3;

fn synth(w: usize, h: usize, index: usize, distorted: bool) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let mut y = vec![0u16; w * h];
    let mut u = vec![0u16; w * h / 4];
    let mut v = vec![0u16; w * h / 4];
    for r in 0..h {
        for c in 0..w {
            let mut val = 16 + ((c * 3 + r * 5 + index * 7 + (c / 17) * 9) % 220) as u16;
            if distorted {
                val = val.saturating_add_signed(((c % 7) as i16) - 3);
            }
            y[r * w + c] = val;
        }
    }
    let jitter = if distorted { 4u16 } else { 0u16 };
    for r in 0..h / 2 {
        for c in 0..w / 2 {
            let i = r * (w / 2) + c;
            u[i] = 100 + ((c * 2 + r * 3 + index) % 56) as u16 + jitter;
            v[i] = 120 + ((c * 5 + r * 2 + index) % 56) as u16 + jitter;
        }
    }
    (y, u, v)
}

fn dims(label: &str) -> Option<(usize, usize)> {
    match label {
        "512" => Some((512, 512)),
        "1024" => Some((1024, 1024)),
        "4K" => Some((3840, 2160)),
        "8K" => Some((7680, 4320)),
        _ => None,
    }
}

fn timeit<F: FnMut() -> Vec<f64>>(mut f: F) -> (f64, f64) {
    let mut times = Vec::with_capacity(ROUNDS);
    for _ in 0..ROUNDS {
        let t = Instant::now();
        black_box(f());
        times.push(t.elapsed().as_secs_f64() * 1e3 / NFRAMES as f64);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (times[ROUNDS / 2], times[0])
}

fn run(label: &str, which: &str) {
    let (w, h) = dims(label).unwrap();
    let refs: Vec<_> = (0..NFRAMES).map(|i| synth(w, h, i, false)).collect();
    let diss: Vec<_> = (0..NFRAMES).map(|i| synth(w, h, i, true)).collect();
    let rf: Vec<Yuv420Frame<'_>> = refs.iter().map(|f| Yuv420Frame { y: &f.0, u: &f.1, v: &f.2 }).collect();
    let df: Vec<Yuv420Frame<'_>> = diss.iter().map(|f| Yuv420Frame { y: &f.0, u: &f.1, v: &f.2 }).collect();

    println!("model\tsize\tthreads\tmedian_ms/frame\tmin_ms/frame");
    for model in ["v0", "v1"].iter().filter(|m| which == "both" || **m == which) {
        for &threads in &[1usize, 4, 8] {
            // Scorer setup (model parse + pool) is outside the timed region.
            let (med, min) = if *model == "v0" {
                let s = VmafV0Scorer::new(w, h, 8, VmafV0Variant::Standard)
                    .unwrap()
                    .with_threads(threads)
                    .unwrap();
                timeit(|| s.score(&rf, &df).unwrap().iter().map(|r| r.score).collect())
            } else {
                let s = VmafV1Scorer::new(w, h, 8, ModelVariant::Phone)
                    .unwrap()
                    .with_threads(threads)
                    .unwrap();
                timeit(|| s.score(&rf, &df).unwrap().iter().map(|r| r.score).collect())
            };
            println!("{model}\t{label}\t{threads}\t{med:.3}\t{min:.3}");
        }
    }
}

fn heap(label: &str) {
    let (w, h) = dims(label).unwrap();
    let refs: Vec<_> = (0..2).map(|i| synth(w, h, i, false)).collect();
    let diss: Vec<_> = (0..2).map(|i| synth(w, h, i, true)).collect();
    let rf: Vec<Yuv420Frame<'_>> = refs.iter().map(|f| Yuv420Frame { y: &f.0, u: &f.1, v: &f.2 }).collect();
    let df: Vec<Yuv420Frame<'_>> = diss.iter().map(|f| Yuv420Frame { y: &f.0, u: &f.1, v: &f.2 }).collect();
    for model in ["v0", "v1"] {
        let s0 = VmafV0Scorer::new(w, h, 8, VmafV0Variant::Standard).unwrap();
        let s1 = VmafV1Scorer::new(w, h, 8, ModelVariant::Phone).unwrap();
        for _ in 0..2 {
            if model == "v0" {
                black_box(s0.score(&rf, &df).unwrap());
            } else {
                black_box(s1.score(&rf, &df).unwrap());
            }
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() >= 3 && args[1] == "heap" {
        heap(&args[2]);
        return;
    }
    if args.len() < 2 {
        eprintln!("usage: wall_matrix <size|all> [v0|v1|both]   |   wall_matrix heap <size>");
        std::process::exit(64);
    }
    let which = args.get(2).map(String::as_str).unwrap_or("both");
    if args[1] == "all" {
        for l in ["512", "1024", "4K", "8K"] {
            run(l, which);
        }
    } else {
        run(&args[1], which);
    }
}
