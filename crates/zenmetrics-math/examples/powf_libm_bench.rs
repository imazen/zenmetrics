//! Throughput of the bit-exact lane `powf` (`vpow_libm_into`, best native
//! tier) against a scalar `libm::powf` loop over the same positive-normal
//! slice. Verifies bit equality before timing; reps interleave both sides.
//!
//! `cargo run --release -p zenmetrics-math [--features avx512] --example powf_libm_bench [n reps]`

use std::time::Instant;

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1 << 20);
    let reps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(31);

    // Metric-domain inputs: positive normals in (0, 2], a fixed LCG stream.
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let xs: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 + f32::MIN_POSITIVE
        })
        .collect();
    let mut lane = vec![0.0f32; n];
    let mut scalar = vec![0.0f32; n];

    println!("n={n} reps={reps} (median / min ns per element)");
    for p in [0.4f32, 0.02, 2.4, -1.5] {
        zenmetrics_math::vpow_libm_into(&xs, &mut lane, p);
        for (o, &x) in scalar.iter_mut().zip(&xs) {
            *o = libm::powf(x, p);
        }
        let mismatches = lane
            .iter()
            .zip(&scalar)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mismatches, 0, "p={p}: lane powf differs from libm::powf");

        let (mut tl, mut ts) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let t = Instant::now();
            zenmetrics_math::vpow_libm_into(std::hint::black_box(&xs), &mut lane, p);
            tl.push(t.elapsed().as_secs_f64());
            std::hint::black_box(&lane);
            let t = Instant::now();
            for (o, &x) in scalar.iter_mut().zip(std::hint::black_box(&xs)) {
                *o = libm::powf(x, std::hint::black_box(p));
            }
            ts.push(t.elapsed().as_secs_f64());
            std::hint::black_box(&scalar);
        }
        let ns = |s: f64| s * 1e9 / n as f64;
        let (ml, ms) = (median(&mut tl), median(&mut ts));
        println!(
            "p={p:>5}: lane {:.3} / {:.3} ns  scalar {:.3} / {:.3} ns  speedup {:.2}x (median)",
            ns(ml),
            ns(tl[0]),
            ns(ms),
            ns(ts[0]),
            ms / ml
        );
    }
}
