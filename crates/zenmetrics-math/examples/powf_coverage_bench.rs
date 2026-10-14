//! What each coverage gate of the lane `e_powf`/`e_pow` costs in ns/elem on
//! positive-finite input (the metric domain), plus a hostile-domain
//! correctness probe showing what each gate *buys*. Drives the shipped
//! `vpow_libm_into_gated` / `vpow64_libm_into_gated` API — no kernel copy.
//!
//!   EDGE  — early-out compares + `one_out` blend + scalar fallback routing
//!   SUBN  — subnormal-input normalize (f32) / subnormal-x scalar routing (f64)
//!   RANGE — overflow/underflow `z` gates
//!   TAIL  — sub-ulp precision machinery (~12 flops)
//!
//! `cargo run --release -p zenmetrics-math --features avx512 --example powf_coverage_bench`

use std::time::Instant;
use zenmetrics_math::{fdlibm::coverage::*, vpow64_libm_into_gated, vpow_libm_into_gated};

fn bench<const G: u8>(label: &str, xs: &[f32], p: f32, reps: usize) {
    let n = xs.len();
    let mut out = vec![0.0f32; n];
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        vpow_libm_into_gated::<G>(std::hint::black_box(xs), &mut out, p);
        times.push(t.elapsed().as_secs_f64());
        std::hint::black_box(&out);
    }
    times.sort_by(f64::total_cmp);
    let med = times[times.len() / 2] * 1e9 / n as f64;
    let mut ulp = 0u64;
    for i in 0..n {
        let w = libm::powf(xs[i], p);
        if out[i].is_nan() && w.is_nan() {
            continue;
        }
        ulp = ulp.max((out[i].to_bits() as i64 - w.to_bits() as i64).unsigned_abs());
    }
    println!("  {label:<34} {med:7.3} ns/elem   max-ulp {ulp}");
}

fn bench64<const G: u8>(label: &str, xs: &[f64], p: f64, reps: usize) {
    let n = xs.len();
    let mut out = vec![0.0f64; n];
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        vpow64_libm_into_gated::<G>(std::hint::black_box(xs), &mut out, p);
        times.push(t.elapsed().as_secs_f64());
        std::hint::black_box(&out);
    }
    times.sort_by(f64::total_cmp);
    let med = times[times.len() / 2] * 1e9 / n as f64;
    let mut ulp = 0u64;
    for i in 0..n {
        let w = libm::pow(xs[i], p);
        if out[i].is_nan() && w.is_nan() {
            continue;
        }
        ulp = ulp.max((out[i].to_bits() as i64 - w.to_bits() as i64).unsigned_abs());
    }
    println!("  {label:<34} {med:7.3} ns/elem   max-ulp {ulp}");
}

fn hostile32<const G: u8>(label: &str, xs: &[f32], ys: &[f32]) {
    // Drive the fixed-p driver once per distinct y, merging per index.
    let n = xs.len();
    let mut got = vec![0.0f32; n];
    let mut done = vec![false; n];
    for &yv in ys.iter() {
        let idxs: Vec<usize> = (0..n)
            .filter(|&i| !done[i] && ys[i].to_bits() == yv.to_bits())
            .collect();
        if idxs.is_empty() {
            continue;
        }
        for &i in &idxs {
            done[i] = true;
        }
        let sub: Vec<f32> = idxs.iter().map(|&i| xs[i]).collect();
        let mut so = vec![0.0f32; sub.len()];
        vpow_libm_into_gated::<G>(&sub, &mut so, yv);
        for (j, &i) in idxs.iter().enumerate() {
            got[i] = so[j];
        }
    }
    let (mut mism, mut ulp) = (0u32, 0u64);
    for i in 0..n {
        let w = libm::powf(xs[i], ys[i]);
        if got[i].is_nan() && w.is_nan() {
            continue;
        }
        if got[i].to_bits() != w.to_bits() {
            mism += 1;
            ulp = ulp.max((got[i].to_bits() as i64 - w.to_bits() as i64).unsigned_abs());
        }
    }
    println!("  {label:<34} mismatched {mism:>6}/{n}  max-ulp {ulp}");
}

fn main() {
    let n: usize = 1 << 20;
    let reps: usize = 41;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut lcg = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f64 / (1u64 << 24) as f64
    };
    let xs32: Vec<f32> = (0..n).map(|_| (lcg() * 2.0) as f32 + f32::MIN_POSITIVE).collect();
    let xs64: Vec<f64> = (0..n).map(|_| lcg() * 2.0 + f64::MIN_POSITIVE).collect();
    let p = 2.4f32;
    let p64 = 2.4f64;

    println!("n={n} p={p} — coverage gates, ns/elem on the best native tier");
    println!("f32:");
    let mut so = vec![0.0f32; n];
    let t = Instant::now();
    for (o, &x) in so.iter_mut().zip(&xs32) {
        *o = libm::powf(x, p);
    }
    println!(
        "  {:<34} {:7.3} ns/elem",
        "scalar libm::powf",
        t.elapsed().as_secs_f64() * 1e9 / n as f64
    );
    bench::<{ EXACT }>("full port (EDGE|SUBN|RANGE|TAIL)", &xs32, p, reps);
    bench::<{ POSITIVE_FINITE }>("no EDGE bookkeeping", &xs32, p, reps);
    bench::<{ POSITIVE_NORMAL }>("POSITIVE_NORMAL (no SUBN)", &xs32, p, reps);
    bench::<{ BULK }>("BULK (TAIL only)", &xs32, p, reps);
    bench::<{ EDGE | SUBN | RANGE }>("no TAIL corrections", &xs32, p, reps);
    bench::<0u8>("bare core (nothing gated)", &xs32, p, reps);

    println!("f64:");
    let mut so = vec![0.0f64; n];
    let t = Instant::now();
    for (o, &x) in so.iter_mut().zip(&xs64) {
        *o = libm::pow(x, p64);
    }
    println!(
        "  {:<34} {:7.3} ns/elem",
        "scalar libm::pow",
        t.elapsed().as_secs_f64() * 1e9 / n as f64
    );
    bench64::<{ EXACT }>("full port (EDGE|SUBN|RANGE|TAIL)", &xs64, p64, reps);
    bench64::<{ POSITIVE_FINITE }>("no EDGE bookkeeping", &xs64, p64, reps);
    bench64::<{ POSITIVE_NORMAL }>("POSITIVE_NORMAL (no SUBN)", &xs64, p64, reps);
    bench64::<{ BULK }>("BULK (TAIL only)", &xs64, p64, reps);
    bench64::<{ EDGE | SUBN | RANGE }>("no TAIL corrections", &xs64, p64, reps);
    bench64::<0u8>("bare core (nothing gated)", &xs64, p64, reps);

    // Hostile domain (f32): what each gate's coverage buys.
    let m = n / 16;
    let mut h: Vec<f32> = Vec::with_capacity(m);
    let mut hy: Vec<f32> = Vec::with_capacity(m);
    for i in 0..m {
        h.push(match i % 8 {
            0 => 0.0,
            1 => -1.5,
            2 => 1.0,
            3 => 5.9e-39, // subnormal
            4 => f32::INFINITY,
            5 => f32::NAN,
            6 => 8.0e33,
            _ => 0.7071,
        });
        hy.push(match i % 6 {
            0 => 0.0,
            1 => 1.0,
            2 => 2.0,
            3 => 0.5,
            4 => 1.0e8,
            _ => -150.0,
        });
    }
    println!("hostile domain ({} lanes): mismatches vs scalar libm::powf", h.len());
    hostile32::<{ EXACT }>("full", &h, &hy);
    hostile32::<{ POSITIVE_FINITE }>("no EDGE", &h, &hy);
    hostile32::<{ POSITIVE_NORMAL }>("POSITIVE_NORMAL", &h, &hy);
    hostile32::<{ EDGE | SUBN | TAIL }>("no RANGE", &h, &hy);
    hostile32::<{ EDGE | SUBN | RANGE }>("no TAIL", &h, &hy);
    hostile32::<0u8>("bare", &h, &hy);
}
