//! Phase 1 diffmap per-pixel distribution capture for zensim-gpu.
//!
//! Captures `mean / median / p25 / p75 / p95 / max` of the per-pixel
//! diffmap across (fixture × distortion-level) cells. Used by Phase 4
//! (zensim_targets calibration) to derive `ZENSIM_DIFFMAP_RENORM_SCALE`
//! per `RFC_PERCEPTUAL_METRIC_REQUIREMENTS.md` §4 and the per-block
//! reducer constants per §3.
//!
//! Mirrors `cvvdp-gpu`-equivalent shape for `JXL_PHASE8B_DIFFMAP_DUMP`
//! data collection. Writes one TSV row per (fixture, distortion).
//!
//! ## Running
//!
//! ```bash
//! cd ~/work/zen/zenmetrics--zensim-diffmap
//! cargo run --example diffmap_distribution \
//!     --features cubecl-types --release \
//!     > crates/zensim-gpu/benchmarks/zensim_diffmap_distribution_$(date +%Y-%m-%d).tsv
//! ```

#![cfg(feature = "cubecl-types")]

use cubecl::Runtime;
use zensim_gpu::Zensim;

#[cfg(feature = "cuda")]
type RT = cubecl::cuda::CudaRuntime;
#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
type RT = cubecl::wgpu::WgpuRuntime;
#[cfg(all(not(feature = "cuda"), not(feature = "wgpu"), feature = "cpu"))]
type RT = cubecl::cpu::CpuRuntime;

const SIZE: u32 = 512;
const DELTAS: &[i32] = &[1, 4, 16, 32, 64];

fn make_gradient(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y + seed as usize) * 255) / (w + h).max(1)) as u8;
            out.push(r);
            out.push(g);
            out.push(b);
        }
    }
    out
}

fn make_checker(w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let v = if ((x / 16) + (y / 16)) % 2 == 0 {
                255
            } else {
                0
            };
            out.push(v);
            out.push(v);
            out.push(v);
        }
    }
    out
}

fn make_noise(w: u32, h: u32) -> Vec<u8> {
    let w = w as usize;
    let h = h as usize;
    let mut out = Vec::with_capacity(w * h * 3);
    let mut rng_state: u32 = 0xDEAD_BEEF;
    for _ in 0..(w * h) {
        // Simple LCG for reproducibility.
        for _ in 0..3 {
            rng_state = rng_state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            out.push((rng_state >> 16) as u8);
        }
    }
    out
}

fn perturb(src: &[u8], delta: i32) -> Vec<u8> {
    src.iter()
        .map(|&v| (v as i32 + delta).clamp(0, 255) as u8)
        .collect()
}

fn diffmap_stats(diffmap: &[f32]) -> (f64, f64, f64, f64, f64, f64) {
    let n = diffmap.len();
    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let mut sorted: Vec<f32> = diffmap.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = diffmap.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
    let pct = |p: f64| -> f64 {
        let idx = ((p * (n as f64 - 1.0)) as usize).min(n - 1);
        sorted[idx] as f64
    };
    (mean, pct(0.25), pct(0.50), pct(0.75), pct(0.95), pct(1.00))
}

fn main() {
    let client = RT::client(&Default::default());
    let backend = std::any::type_name::<RT>();

    println!("fixture\tdelta\tscore\tmean\tp25\tp50\tp75\tp95\tmax\tn_pixels");
    eprintln!(
        "# zensim-gpu diffmap distribution capture (Phase 1)\n\
         # backend: {backend}\n\
         # size: {SIZE}x{SIZE}\n\
         # purpose: input to Phase 4 ZENSIM_DIFFMAP_RENORM_SCALE + K_TILE_NORM calibration"
    );

    let fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("gradient_a", make_gradient(SIZE, SIZE, 0)),
        ("gradient_b", make_gradient(SIZE, SIZE, 31)),
        ("checker_16", make_checker(SIZE, SIZE)),
        ("noise_lcg", make_noise(SIZE, SIZE)),
    ];

    let mut z: Zensim<RT> = Zensim::new(client, SIZE, SIZE).expect("Zensim::new");
    let mut diffmap = Vec::new();

    for (name, img) in &fixtures {
        for &delta in DELTAS {
            let dist = perturb(img, delta);
            let score = z
                .score_with_diffmap(img, &dist, &mut diffmap)
                .expect("score_with_diffmap");
            let (mean, p25, p50, p75, p95, max) = diffmap_stats(&diffmap);
            println!(
                "{name}\t{delta}\t{score:.6}\t{mean:.6}\t{p25:.6}\t{p50:.6}\t{p75:.6}\t{p95:.6}\t{max:.6}\t{}",
                diffmap.len()
            );
        }
    }
}
