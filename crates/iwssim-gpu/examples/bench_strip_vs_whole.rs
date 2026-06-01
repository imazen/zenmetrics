//! Wall-clock and memory comparison of whole-image vs strip-mode
//! IW-SSIM on a fixed CUDA backend.
//!
//! Build + run:
//! ```bash
//! cargo run --release -p iwssim-gpu --example bench_strip_vs_whole \
//!     --no-default-features --features cubecl-types,cuda
//! ```
//!
//! Output: CSV-style rows printed to stdout. The first run also
//! writes the same rows to
//! `benchmarks/iwssim_strip_vs_whole_<YYYY-MM-DD>.csv` for archival.
//!
//! Why this exists: production sweep workers OOM at 24 MP because the
//! whole-image path pre-allocates `O(image_h × image_w × 4B × ~19)`
//! scale-0 working planes. Strip mode bounds the working set to a
//! single strip's allocation. This bench shows the wall-time cost (if
//! any) of the strip path and the memory delta — see the GPU memory
//! tally printed at the end of each row.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use cubecl::wgpu::WgpuRuntime as Backend;
use iwssim_gpu::Iwssim;

/// Deterministic LCG image generator — same content for each (seed, w, h).
fn make_gray(w: u32, h: u32, seed: u32) -> Vec<f32> {
    use std::num::Wrapping;
    let mut v = Vec::with_capacity((w * h) as usize);
    let mut s = Wrapping(seed);
    for _ in 0..w * h {
        s = s * Wrapping(1_664_525_u32) + Wrapping(1_013_904_223_u32);
        v.push(((s.0 >> 16) & 0xFF) as f32);
    }
    v
}

/// Per-scale buffer count used by the iwssim-gpu pipeline. Matches the
/// `Scale` struct's field count (19 f32 planes per scale). Hand-counted
/// from `crates/iwssim-gpu/src/pipeline.rs::struct Scale`. Used for the
/// approximate GPU-memory tally — the cov_partials + sums + src_u32_*
/// allocations are tiny vs the per-scale planes and are amortised away.
const PLANES_PER_SCALE_APPROX: u64 = 19;
const NUM_SCALES: u64 = 5;

/// Approximate the per-scale, scale-0-sized working set in MB. Halves
/// per axis at each scale, so the geometric series is `~1.33 ×
/// scale_0_bytes`. Returns megabytes (10⁶ bytes), not mebibytes.
fn working_set_mb(strip_alloc_h: u32, image_w: u32) -> f64 {
    // Cumulative across 5 scales — each scale halves in both axes, so
    // the factor is roughly 1 + 1/4 + 1/16 + 1/64 + 1/256 = 341/256 ≈ 1.332.
    // Use exact pyramid scaling here — accounts for div_ceil(2) at each
    // step instead of the closed-form approximation.
    let mut total: u64 = 0;
    let mut h = strip_alloc_h as u64;
    let mut w = image_w as u64;
    for _ in 0..NUM_SCALES {
        total += h * w * 4 * PLANES_PER_SCALE_APPROX;
        h = h.div_ceil(2);
        w = w.div_ceil(2);
    }
    (total as f64) / 1e6
}

struct Row {
    w: u32,
    h: u32,
    /// (alloc-strip-h, h_body) — None for the whole-image baseline row.
    strip: Option<(u32, u32)>,
    /// Wall time per pair (mean over `n_measure`), milliseconds.
    /// `None` for OOM-skipped rows (24 MP whole-image).
    mean_ms: Option<f64>,
    /// Min wall time per pair across `n_measure` iters, milliseconds.
    /// `None` for OOM-skipped rows.
    min_ms: Option<f64>,
    /// Final IW-SSIM score (kept beside the timing so a perf-regressing
    /// change that also breaks correctness is visible in one line).
    /// `None` for OOM-skipped rows.
    score: Option<f64>,
    /// Approximate GPU working set (geometric series across 5 scales)
    /// in megabytes.
    working_set_mb: f64,
}

fn bench_whole(w: u32, h: u32, n_warmup: usize, n_measure: usize) -> Row {
    let ref_gray = make_gray(w, h, 42);
    let dis_gray = make_gray(w, h, 137);

    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new(client.clone(), w, h).unwrap();

    for _ in 0..n_warmup {
        let _ = iw.compute_gray(&ref_gray, &dis_gray).unwrap();
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");

    let mut min_ms = f64::INFINITY;
    let mut last_score = 0.0_f64;
    let t = Instant::now();
    for _ in 0..n_measure {
        let t_iter = Instant::now();
        last_score = iw.compute_gray(&ref_gray, &dis_gray).unwrap().score;
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let dt_iter = t_iter.elapsed().as_secs_f64();
        if dt_iter < min_ms {
            min_ms = dt_iter;
        }
    }
    let mean_ms = t.elapsed().as_secs_f64() / n_measure as f64 * 1e3;
    let min_ms = min_ms * 1e3;

    let ws = working_set_mb(h, w);
    Row {
        w,
        h,
        strip: None,
        mean_ms: Some(mean_ms),
        min_ms: Some(min_ms),
        score: Some(last_score),
        working_set_mb: ws,
    }
}

fn bench_strip(w: u32, h: u32, h_body: u32, n_warmup: usize, n_measure: usize) -> Row {
    let ref_gray = make_gray(w, h, 42);
    let dis_gray = make_gray(w, h, 137);

    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body).unwrap();

    for _ in 0..n_warmup {
        let _ = iw.compute_gray_stripped(&ref_gray, &dis_gray).unwrap();
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");

    let mut min_ms = f64::INFINITY;
    let mut last_score = 0.0_f64;
    let t = Instant::now();
    for _ in 0..n_measure {
        let t_iter = Instant::now();
        last_score = iw
            .compute_gray_stripped(&ref_gray, &dis_gray)
            .unwrap()
            .score;
        cubecl::future::block_on(client.sync()).expect("client.sync");
        let dt_iter = t_iter.elapsed().as_secs_f64();
        if dt_iter < min_ms {
            min_ms = dt_iter;
        }
    }
    let mean_ms = t.elapsed().as_secs_f64() / n_measure as f64 * 1e3;
    let min_ms = min_ms * 1e3;

    // Default halo is 256 (see STRIP_DEFAULT_HALO). The buffer allocation
    // is sized for h_body + 2 * 256.
    let strip_alloc_h = h_body + 2 * 256;
    let ws = working_set_mb(strip_alloc_h, w);
    Row {
        w,
        h,
        strip: Some((strip_alloc_h, h_body)),
        mean_ms: Some(mean_ms),
        min_ms: Some(min_ms),
        score: Some(last_score),
        working_set_mb: ws,
    }
}

fn fmt_row(r: &Row) -> String {
    let mode = match r.strip {
        None => "whole".to_string(),
        Some((alloc, body)) => format!("strip(body={body},alloc={alloc})"),
    };
    let mp = (r.w as f64 * r.h as f64) / 1e6;
    let mean = r
        .mean_ms
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "SKIP_OOM".into());
    let min = r
        .min_ms
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "SKIP_OOM".into());
    let score = r
        .score
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "n/a".into());
    format!(
        "{w}x{h},{mp:.2},{mode},{mean},{min},{score},{ws:.1}",
        w = r.w,
        h = r.h,
        mp = mp,
        mode = mode,
        ws = r.working_set_mb,
    )
}

fn main() {
    println!("iwssim-gpu bench: whole-image vs strip-mode");
    let header = "w_h,mp,mode,mean_ms,min_ms,score,working_set_mb";
    println!("{header}");

    // Iteration count is small because we want this bench to finish
    // in a reasonable time at 12 MP (~80 ms / pair × 8 iters ≈ 1 s).
    let n_warmup = 3;
    let n_measure = 8;

    // Image sizes to sweep. Each is paired with a strip body of 1024.
    // 4MP and 12MP are the focus — 1MP is a baseline (single-strip
    // degenerate when body=1024) and 24MP is the OOM target.
    let sizes: &[(u32, u32)] = &[
        (1024, 1024), // 1 MP, single-strip degenerate
        (2048, 2048), // 4 MP, 2 body strips at body=1024
        (4000, 3000), // 12 MP, 3 body strips at body=1024
        (6000, 4000), // 24 MP, 4 body strips at body=1024
    ];

    let mut rows: Vec<Row> = Vec::new();

    for &(w, h) in sizes {
        let mp = (w as f64 * h as f64) / 1e6;

        // Whole-image baseline. Skip at 24 MP — the allocations would
        // OOM 12 GB GPUs; this is the precise reason the strip path
        // exists. Push a SKIP_OOM marker row.
        if mp <= 13.0 {
            let r = bench_whole(w, h, n_warmup, n_measure);
            println!("{}", fmt_row(&r));
            rows.push(r);
        } else {
            let r = Row {
                w,
                h,
                strip: None,
                mean_ms: None,
                min_ms: None,
                score: None,
                working_set_mb: working_set_mb(h, w),
            };
            println!("{}", fmt_row(&r));
            rows.push(r);
        }

        // Strip path with body=1024.
        let r_strip = bench_strip(w, h, 1024, n_warmup, n_measure);
        println!("{}", fmt_row(&r_strip));
        rows.push(r_strip);
    }

    // Write the same data to benchmarks/iwssim_strip_vs_whole_<date>.csv.
    let date = std::env::var("BENCH_DATE").unwrap_or_else(|_| "2026-05-22".to_string());
    let out_path = format!("benchmarks/iwssim_strip_vs_whole_{date}.csv");
    match File::create(&out_path) {
        Ok(mut f) => {
            writeln!(f, "{header}").ok();
            for r in &rows {
                writeln!(f, "{}", fmt_row(r)).ok();
            }
            eprintln!("wrote {out_path}");
        }
        Err(e) => eprintln!("could not write {out_path}: {e}"),
    }
}
