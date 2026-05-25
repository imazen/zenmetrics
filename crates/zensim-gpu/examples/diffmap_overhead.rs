//! Phase 1 diffmap wall-time overhead bench for zensim-gpu.
//!
//! Mirrors `cvvdp-gpu/examples/diffmap_overhead.rs` shape: paired
//! `score` vs `score_with_diffmap` wall-time at 4 sizes × 2
//! fixtures. Writes a TSV row per cell to stdout (caller pipes to
//! `benchmarks/zensim_diffmap_overhead_<date>.tsv`).
//!
//! ## Running
//!
//! ```bash
//! cd ~/work/zen/zenmetrics--zensim-diffmap
//! cargo run --example diffmap_overhead \
//!     --features cubecl-types --release \
//!     > crates/zensim-gpu/benchmarks/zensim_diffmap_overhead_$(date +%Y-%m-%d).tsv
//! ```
//!
//! Runs on whichever CubeCL backend is available (CUDA preferred).
//! Phase 1 is CPU-fallback for the diffmap path; the overhead
//! measured here is the wall hit of building the
//! `PrecomputedReference` + running zensim CPU's
//! `compute_with_ref_and_diffmap_linear_planar` on top of the GPU's
//! scalar feature-vector compute.
//!
//! Phase 1b will replace the CPU side with pure-GPU kernels —
//! re-run this bench after that lands to capture the win.

#![cfg(feature = "cubecl-types")]

use std::time::{Duration, Instant};

use cubecl::Runtime;
use zensim_gpu::Zensim;

#[cfg(feature = "cuda")]
type RT = cubecl::cuda::CudaRuntime;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
type RT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "cpu"))]
type RT = cubecl::cpu::CpuRuntime;

const SIZES: &[(u32, u32)] = &[(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

/// Warm-up + N-iter measurement loop. Returns (median, p95) wall in
/// milliseconds.
fn time_iters<F: FnMut()>(mut f: F, iters: usize) -> (f64, f64) {
    // Warm.
    for _ in 0..2 {
        f();
    }
    let mut times: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    times.sort();
    let median = times[times.len() / 2].as_secs_f64() * 1000.0;
    let p95 = times[(times.len() * 95) / 100].as_secs_f64() * 1000.0;
    (median, p95)
}

fn make_gradient(w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

fn perturb(src: &[u8], delta: i32) -> Vec<u8> {
    src.iter()
        .map(|&v| (v as i32 + delta).clamp(0, 255) as u8)
        .collect()
}

fn main() {
    let client = RT::client(&Default::default());
    let backend_name = std::any::type_name::<RT>();

    // TSV header.
    println!(
        "size_w\tsize_h\tfixture\tscore_only_median_ms\tscore_only_p95_ms\tscore_with_diffmap_median_ms\tscore_with_diffmap_p95_ms\toverhead_pct"
    );
    eprintln!(
        "# zensim-gpu diffmap overhead bench (Phase 1: CPU-fallback diffmap)\n# backend: {backend_name}\n# date: {}",
        chrono_now()
    );

    for &(w, h) in SIZES {
        // 2 fixtures per size: smooth gradient + perturbed gradient.
        let img = make_gradient(w, h);
        let dist = perturb(&img, 15);

        // Re-create the Zensim per size so the diffmap_state's lazy
        // alloc fires cleanly between fixtures.
        let mut z_a: Zensim<RT> = Zensim::new(client.clone(), w, h).expect("Zensim::new");
        let mut z_b: Zensim<RT> = Zensim::new(client.clone(), w, h).expect("Zensim::new");

        let iters = match w {
            0..=256 => 10,
            257..=512 => 8,
            513..=1024 => 5,
            _ => 3,
        };

        // Fixture 1: gradient self (identity).
        let (so_med, so_p95) = time_iters(
            || {
                let _ = z_a.compute_features_vec(&img, &img).expect("score-only");
            },
            iters,
        );
        let mut diffmap = Vec::new();
        let (sd_med, sd_p95) = time_iters(
            || {
                let _ = z_b
                    .score_with_diffmap(&img, &img, &mut diffmap)
                    .expect("score+diffmap");
            },
            iters,
        );
        let overhead_pct = (sd_med - so_med) / so_med * 100.0;
        println!(
            "{w}\t{h}\tgradient_identity\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
            so_med, so_p95, sd_med, sd_p95, overhead_pct
        );

        // Fixture 2: gradient vs perturbed.
        let (so_med, so_p95) = time_iters(
            || {
                let _ = z_a.compute_features_vec(&img, &dist).expect("score-only");
            },
            iters,
        );
        let (sd_med, sd_p95) = time_iters(
            || {
                let _ = z_b
                    .score_with_diffmap(&img, &dist, &mut diffmap)
                    .expect("score+diffmap");
            },
            iters,
        );
        let overhead_pct = (sd_med - so_med) / so_med * 100.0;
        println!(
            "{w}\t{h}\tgradient_perturbed\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.2}",
            so_med, so_p95, sd_med, sd_p95, overhead_pct
        );
    }
}

fn chrono_now() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix-epoch-seconds={now}")
}
