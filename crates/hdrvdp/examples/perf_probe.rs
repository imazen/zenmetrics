//! Profiling / wall-time probe: score one synthetic pair end to end.
//!
//! Not a benchmark (no interleaving, no statistics — `benches/` is that);
//! this exists so a profiler has a single long-running, deterministic target:
//!
//! ```text
//! cargo run --release -p hdrvdp --example perf_probe -- <side> [reps]
//! sample <pid> 10 -f /tmp/prof.txt        # macOS
//! ```
//!
//! Prints per-call wall time and a bit-digest of the outputs so two builds
//! can be compared for bit-identity at any size without a debugger.

#![forbid(unsafe_code)]

use hdrvdp::{ColorEncoding, Params, hdrvdp};

/// FNV-1a-64 over the bit patterns of an f64 stream.
fn fnv1a(values: impl IntoIterator<Item = f64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for v in values {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

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
            let base = 60.0 * (1.0 + 0.4 * (2.0 * pi * x / 17.0).sin() * (2.0 * pi * y / 23.0).cos())
                + 6.0 * (rng() - 0.5);
            let (cx, cy) = (w as f64 * 0.7, h as f64 * 0.3);
            let r2 = ((x - cx).powi(2) + (y - cy).powi(2)) / (0.02 * (w * h) as f64 + 1.0);
            (base + 3000.0 * (-r2).exp()).max(0.05)
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let side: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(512);
    let reps: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1);

    let par = Params::new(30.0);
    let reference = hdr_field(side, side, 0xfeed);
    let pi = core::f64::consts::PI;
    let test: Vec<f64> = reference
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (x, y) = ((i % side) as f64, (i / side) as f64);
            v * (1.0 + 0.03 * (2.0 * pi * x / 5.0).sin() * (2.0 * pi * y / 7.0).sin())
        })
        .collect();

    for rep in 0..reps {
        let t0 = std::time::Instant::now();
        let r = hdrvdp(&test, &reference, side, side, ColorEncoding::Luminance, &par)
            .expect("perf_probe: metric failed");
        let dt = t0.elapsed();
        let digest = fnv1a(
            r.p_map
                .iter()
                .copied()
                .chain(r.c_map.iter().copied())
                .chain([r.q, r.q_mos, r.p_det, r.c_max]),
        );
        println!(
            "rep {rep}: {side}x{side} in {:.3} s  q_mos={:.6}  digest={digest:#018x}",
            dt.as_secs_f64(),
            r.q_mos
        );
    }
}
