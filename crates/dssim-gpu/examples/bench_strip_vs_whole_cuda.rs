//! Whole-image vs strip DSSIM throughput benchmark.
//!
//! Compares `Dssim::new()` + `compute()` (whole-image, single
//! pyramid sized at image_w × image_h) against `Dssim::new_strip()`
//! + `compute()` (strip-mode, pyramid sized at image_w × (h_body +
//! 2*halo), reused across strips).
//!
//! Writes a CSV row per (size, h_body) to
//! `benchmarks/dssim_strip_vs_whole_<YYYY-MM-DD>.csv`.
//!
//! Run:
//! ```bash
//! CUDA_PATH=/usr/local/cuda cargo run --release -p dssim-gpu \
//!     --example bench_strip_vs_whole_cuda \
//!     --features cuda,cubecl-types
//! ```

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;

use dssim_gpu::Dssim;

fn make_image(w: usize, h: usize, seed: u32) -> Vec<u8> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity(w * h * 3);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        for _ in 0..3 {
            s = s * Wrapping(1664525u32) + Wrapping(1013904223u32);
            v.push(((s.0 >> 16) & 0xFF) as u8);
        }
    }
    v
}

#[derive(Debug, Clone)]
struct Row {
    width: u32,
    height: u32,
    mp: f64,
    h_body: u32,
    whole_ms: f64,
    strip_ms: f64,
    whole_score: f64,
    strip_score: f64,
}

impl Row {
    fn header() -> &'static str {
        "width,height,megapixels,h_body,whole_ms,strip_ms,strip_overhead_pct,whole_score,strip_score,score_rel_diff\n"
    }

    fn to_csv(&self) -> String {
        let overhead = if self.whole_ms > 0.0 {
            (self.strip_ms - self.whole_ms) / self.whole_ms * 100.0
        } else {
            0.0
        };
        let rel = (self.strip_score - self.whole_score).abs() / self.whole_score.max(1e-6);
        format!(
            "{},{},{:.6},{},{:.3},{:.3},{:.2},{:.8},{:.8},{:.6}\n",
            self.width,
            self.height,
            self.mp,
            self.h_body,
            self.whole_ms,
            self.strip_ms,
            overhead,
            self.whole_score,
            self.strip_score,
            rel
        )
    }
}

fn bench(w: u32, h: u32, h_body: u32, n_warmup: usize, n_measure: usize) -> Row {
    let img_a = make_image(w as usize, h as usize, 42);
    let img_b = make_image(w as usize, h as usize, 137);

    // Whole-image baseline.
    let client = Backend::client(&Default::default());
    let mut whole = Dssim::<Backend>::new(client.clone(), w, h).unwrap();
    for _ in 0..n_warmup {
        let _ = whole.compute(&img_a, &img_b).unwrap();
    }
    let t = Instant::now();
    let mut whole_score = 0.0;
    for _ in 0..n_measure {
        whole_score = whole.compute(&img_a, &img_b).unwrap().score;
    }
    let whole_ms = t.elapsed().as_secs_f64() / n_measure as f64 * 1e3;
    drop(whole);

    // Strip path.
    let client = Backend::client(&Default::default());
    let mut strip = Dssim::<Backend>::new_strip(client.clone(), w, h, h_body).unwrap();
    for _ in 0..n_warmup {
        let _ = strip.compute(&img_a, &img_b).unwrap();
    }
    let t = Instant::now();
    let mut strip_score = 0.0;
    for _ in 0..n_measure {
        strip_score = strip.compute(&img_a, &img_b).unwrap().score;
    }
    let strip_ms = t.elapsed().as_secs_f64() / n_measure as f64 * 1e3;

    Row {
        width: w,
        height: h,
        mp: (w as f64 * h as f64) / 1e6,
        h_body,
        whole_ms,
        strip_ms,
        whole_score,
        strip_score,
    }
}

fn main() {
    let date = "2026-05-22"; // bake date for reproducibility
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benchmarks");
    fs::create_dir_all(&out_dir).ok();
    let out_path = out_dir.join(format!("dssim_strip_vs_whole_{date}.csv"));

    eprintln!("dssim-gpu strip-vs-whole benchmark (RTX 5070 + Ryzen 9 7950X)");
    eprintln!("writing: {}", out_path.display());

    // (w, h, h_body) — h_body=256 is the default sweep value; we
    // also probe h_body=512 at large sizes to see whether the
    // tradeoff shifts.
    let mut grid: Vec<(u32, u32, u32)> = vec![
        (1024, 1024, 256), // 1 MP
        (2048, 2048, 256), // 4 MP
        (3464, 3464, 256), // 12 MP
        (3464, 3464, 512),
    ];

    // 24 MP — only if the whole-image baseline can fit on the GPU.
    // GPU memory budget is ~12 GB on RTX 5070; whole-image at 24 MP
    // needs ~3.66 GB of working set, comfortable.
    grid.push((4898, 4898, 256));
    grid.push((4898, 4898, 512));

    let mut f = fs::File::create(&out_path).unwrap();
    f.write_all(Row::header().as_bytes()).unwrap();

    println!(
        "{:>10}  {:>5}  {:>10}  {:>10}  {:>10}  {:>16}",
        "size", "h_body", "whole_ms", "strip_ms", "overhead_%", "score_rel_diff"
    );
    for (w, h, h_body) in grid {
        let row = bench(w, h, h_body, 2, 8);
        let overhead = if row.whole_ms > 0.0 {
            (row.strip_ms - row.whole_ms) / row.whole_ms * 100.0
        } else {
            0.0
        };
        let rel = (row.strip_score - row.whole_score).abs() / row.whole_score.max(1e-6);
        println!(
            "{:>5}x{:<4}  {:>5}  {:>10.3}  {:>10.3}  {:>9.2}%  {:>16.6e}",
            w, h, h_body, row.whole_ms, row.strip_ms, overhead, rel
        );
        f.write_all(row.to_csv().as_bytes()).unwrap();
        f.flush().unwrap();
    }

    println!("\nWrote {}", out_path.display());
}
