//! Phase 1b chunk 3+ diffmap wall bench: scalar `score` baseline vs
//! the CPU-default diffmap path vs the opt-in GPU diffmap path.
//!
//! Emits a TSV row per (size, fixture) with:
//!   • `score_only_ms`  — GPU scalar feature extraction (the +1006%
//!     overhead baseline denominator);
//!   • `cpu_diffmap_ms` — `score_with_diffmap` with the CPU default
//!     (Phase 1 behaviour; ZENSIM_GPU_DIFFMAP unset);
//!   • `gpu_diffmap_ms` — `score_with_diffmap` with the opt-in GPU
//!     kernel chain (ZENSIM_GPU_DIFFMAP=1; score still CPU-sourced);
//!   • overhead-% of each diffmap path vs the scalar baseline.
//!
//! Honest-stop context: the GPU path keeps the CPU canonical SCORE
//! (the GPU-feature → V0_3 MLP score is broken on the pinned zensim
//! 0.3.0 — see `docs/DIFFMAP_DIVERGENCES.md` §2b + §9), so GPU-diffmap
//! + CPU-score runs BOTH pipelines and is measured here to be slower
//! than the CPU-only default. This bench documents that finding and is
//! the regression baseline the chunk-N+1 score-path fix optimises
//! against.
//!
//! ## Running
//!
//! ```bash
//! cargo run -p zensim-gpu --release --no-default-features \
//!     --features "cuda,fast-reduction" --example diffmap_gpu_vs_cpu \
//!     > crates/zensim-gpu/benchmarks/zensim_diffmap_overhead_$(date +%Y-%m-%d).tsv
//! ```

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

fn time_iters<F: FnMut()>(mut f: F, iters: usize) -> f64 {
    for _ in 0..3 {
        f();
    }
    let mut times: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed());
    }
    times.sort();
    times[times.len() / 2].as_secs_f64() * 1000.0
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

fn set_gpu_gate(on: bool) {
    // SAFETY: single-threaded bench; the env var is read lazily by the
    // pipeline on the next call. Set BEFORE constructing the Zensim
    // used for that measurement.
    unsafe {
        if on {
            std::env::set_var("ZENSIM_GPU_DIFFMAP", "1");
        } else {
            std::env::remove_var("ZENSIM_GPU_DIFFMAP");
        }
    }
}

fn main() {
    let client = RT::client(&Default::default());
    let backend_name = std::any::type_name::<RT>();

    println!(
        "size_w\tsize_h\tfixture\tscore_only_ms\tcpu_diffmap_ms\tgpu_diffmap_ms\tcpu_overhead_pct\tgpu_overhead_pct"
    );
    eprintln!(
        "# zensim-gpu diffmap wall bench (Phase 1b chunk 3: CPU-default vs GPU-opt-in)\n# backend: {backend_name}\n# date: unix-epoch-seconds={}",
        std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );

    for &(w, h) in SIZES {
        let img = make_gradient(w, h);
        let dist = perturb(&img, 15);
        let iters = match w {
            0..=256 => 10,
            257..=512 => 8,
            513..=1024 => 5,
            _ => 3,
        };

        for (fixture, d) in [("gradient_identity", &img), ("gradient_perturbed", &dist)] {
            // Scalar baseline (no env gate relevance — pure GPU features).
            set_gpu_gate(false);
            let mut z_score: Zensim<RT> = Zensim::new(client.clone(), w, h).expect("Zensim::new");
            let score_ms = time_iters(
                || {
                    let _ = z_score.compute_features_vec(&img, d).expect("score-only");
                },
                iters,
            );

            // CPU-default diffmap path.
            set_gpu_gate(false);
            let mut z_cpu: Zensim<RT> = Zensim::new(client.clone(), w, h).expect("Zensim::new");
            let mut dm = Vec::new();
            let cpu_ms = time_iters(
                || {
                    let _ = z_cpu
                        .score_with_diffmap(&img, d, &mut dm)
                        .expect("cpu diffmap");
                },
                iters,
            );

            // GPU-opt-in diffmap path.
            set_gpu_gate(true);
            let mut z_gpu: Zensim<RT> = Zensim::new(client.clone(), w, h).expect("Zensim::new");
            let gpu_ms = time_iters(
                || {
                    let _ = z_gpu
                        .score_with_diffmap(&img, d, &mut dm)
                        .expect("gpu diffmap");
                },
                iters,
            );
            set_gpu_gate(false);

            let cpu_oh = (cpu_ms - score_ms) / score_ms * 100.0;
            let gpu_oh = (gpu_ms - score_ms) / score_ms * 100.0;
            println!(
                "{w}\t{h}\t{fixture}\t{score_ms:.3}\t{cpu_ms:.3}\t{gpu_ms:.3}\t{cpu_oh:.1}\t{gpu_oh:.1}"
            );
        }
    }
}
