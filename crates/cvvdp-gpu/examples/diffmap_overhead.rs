//! Wall-time overhead measurement: `score_with_diffmap` vs `score`.
//!
//! Reports median + p95 wall-time per call across `--iters` runs at
//! `--sizes` (default 256, 512, 1024). Prints a TSV row per size so
//! the result is paste-able into `benchmarks/`.
//!
//! Usage:
//!
//! ```bash
//! cargo run --release -p cvvdp-gpu --features cubecl-types \
//!     --example diffmap_overhead -- --iters 10
//! ```
//!
//! Output (TSV columns):
//!
//! ```text
//! size  scoring_median_us  scoring_p95_us  diffmap_median_us  diffmap_p95_us  overhead_pct
//! ```
//!
//! Overhead is `(diffmap_median - scoring_median) / scoring_median * 100`.
//!
//! Acceptance gate from the diffmap-API task: if overhead > 30%
//! wall, surface it; > 50% is worth a follow-on optimization.

use std::time::Instant;

use cubecl::Runtime;
use cvvdp_gpu::Cvvdp;
use cvvdp_gpu::params::CvvdpParams;

#[cfg(feature = "cuda")]
type Backend = cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
type Backend = cubecl::wgpu::WgpuRuntime;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "wgpu"))))]
type Backend = cubecl::cpu::CpuRuntime;

fn synth_pair(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    let n = (w as usize) * (h as usize);
    let mut r = Vec::with_capacity(n * 3);
    let mut d = Vec::with_capacity(n * 3);
    for y in 0..h {
        for x in 0..w {
            r.push(((x * 5) % 256) as u8);
            r.push(((y * 7) % 256) as u8);
            r.push((((x + y) * 3) % 256) as u8);
            d.push(((x * 5 + 10) % 256) as u8);
            d.push(((y * 7 + 6) % 256) as u8);
            d.push((((x + y) * 3 + 8) % 256) as u8);
        }
    }
    (r, d)
}

fn percentile(times_us: &mut [f64], p: f64) -> f64 {
    if times_us.is_empty() {
        return f64::NAN;
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((times_us.len() as f64 - 1.0) * p).round() as usize;
    times_us[idx]
}

fn main() {
    let iters: usize = std::env::args()
        .skip_while(|a| a != "--iters")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let sizes: Vec<u32> = std::env::args()
        .skip_while(|a| a != "--sizes")
        .nth(1)
        .map(|s| s.split(',').filter_map(|t| t.parse().ok()).collect())
        .unwrap_or_else(|| vec![256u32, 512, 1024]);

    eprintln!(
        "# diffmap_overhead: backend = {:?}",
        std::any::type_name::<Backend>()
    );
    eprintln!("# iters per cell: {iters}");
    println!(
        "size\tscoring_median_us\tscoring_p95_us\tdiffmap_median_us\tdiffmap_p95_us\toverhead_pct"
    );

    for size in sizes {
        let client = Backend::client(&Default::default());
        let mut cvvdp = match Cvvdp::<Backend>::new(client, size, size, CvvdpParams::PLACEHOLDER) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Cvvdp::new({size}, {size}) failed: {e:?}");
                continue;
            }
        };
        let (r, d) = synth_pair(size, size);

        // Warmup — first call pays compile-cache + lazy-alloc costs.
        let _ = cvvdp.score(&r, &d).unwrap();
        let mut diffmap = Vec::with_capacity((size * size) as usize);
        let _ = cvvdp.score_with_diffmap(&r, &d, &mut diffmap).unwrap();

        let mut scoring_us: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = cvvdp.score(&r, &d).unwrap();
            scoring_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        let mut diffmap_us: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let _ = cvvdp.score_with_diffmap(&r, &d, &mut diffmap).unwrap();
            diffmap_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        let s_med = percentile(&mut scoring_us.clone(), 0.50);
        let s_p95 = percentile(&mut scoring_us, 0.95);
        let d_med = percentile(&mut diffmap_us.clone(), 0.50);
        let d_p95 = percentile(&mut diffmap_us, 0.95);
        let overhead_pct = (d_med - s_med) / s_med * 100.0;
        println!("{size}\t{s_med:.2}\t{s_p95:.2}\t{d_med:.2}\t{d_p95:.2}\t{overhead_pct:+.2}",);
    }
}
