//! Wall-clock comparison of cached-reference vs uncached strip-mode
//! IW-SSIM for the RD-search hot loop: one reference + N distortions.
//!
//! Build + run:
//! ```bash
//! cargo run --release -p iwssim-gpu --example bench_cachedref_strip \
//!     --no-default-features --features cubecl-types,cuda
//! ```
//!
//! Output: CSV-style rows printed to stdout; the first run also
//! writes `benchmarks/iwssim_cachedref_strip_<YYYY-MM-DD>.csv`.
//!
//! Why this exists: RD-search workloads score a single reference
//! against many distortions (q sweep, knob sweep). The uncached strip
//! path re-uploads + re-pyramids the reference for every dist, while
//! the cached path uploads once via `set_reference_stripped` and
//! amortises the per-strip LP pyramid across all dist calls. This
//! bench measures the per-dist savings at 4 MP (the size the
//! recovery brief specified) plus a 12 MP point for context, and
//! both a cached vs uncached strip column and a pair-mode (whole)
//! column where memory allows.

use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime as Backend;
use cubecl::Runtime;
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

struct Row {
    w: u32,
    h: u32,
    h_body: u32,
    /// Mode label.
    mode: &'static str,
    /// Number of distortions per reference.
    n_dist: usize,
    /// Total wall time over the full (1-ref + N-dist) workload, ms.
    total_ms: f64,
    /// Per-distortion wall time, ms. For the cached path this excludes
    /// the one-time `set_reference_stripped` cost; for the uncached
    /// path each call rebuilds the pyramid.
    per_dist_ms: f64,
    /// Final IW-SSIM score of the last dist (kept beside the timing
    /// so a perf change that silently breaks correctness shows up).
    last_score: f64,
}

fn bench_cached(w: u32, h: u32, h_body: u32, n_dist: usize, n_warmup: usize) -> Row {
    let ref_gray = make_gray(w, h, 42);
    let dists: Vec<Vec<f32>> = (0..n_dist).map(|i| make_gray(w, h, 137 + i as u32)).collect();

    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body).unwrap();

    // Warmup: prime caches, JIT, allocator. We re-run set_reference_stripped
    // here to mirror the actual workload's order.
    for _ in 0..n_warmup {
        iw.set_reference_stripped(&ref_gray).unwrap();
        for d in &dists[..n_dist.min(2)] {
            let _ = iw.compute_with_reference_stripped(d).unwrap();
        }
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");

    let mut last_score = 0.0_f64;
    let t = Instant::now();
    iw.set_reference_stripped(&ref_gray).unwrap();
    let t_dist = Instant::now();
    for d in &dists {
        last_score = iw.compute_with_reference_stripped(d).unwrap().score;
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");
    let per_dist_ms = t_dist.elapsed().as_secs_f64() / n_dist as f64 * 1e3;
    let total_ms = t.elapsed().as_secs_f64() * 1e3;

    Row {
        w,
        h,
        h_body,
        mode: "cached_ref_strip",
        n_dist,
        total_ms,
        per_dist_ms,
        last_score,
    }
}

fn bench_uncached(w: u32, h: u32, h_body: u32, n_dist: usize, n_warmup: usize) -> Row {
    let ref_gray = make_gray(w, h, 42);
    let dists: Vec<Vec<f32>> = (0..n_dist).map(|i| make_gray(w, h, 137 + i as u32)).collect();

    let client = Backend::client(&Default::default());
    let mut iw = Iwssim::<Backend>::new_strip(client.clone(), w, h, h_body).unwrap();

    for _ in 0..n_warmup {
        for d in &dists[..n_dist.min(2)] {
            let _ = iw.compute_gray_stripped(&ref_gray, d).unwrap();
        }
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");

    let mut last_score = 0.0_f64;
    let t = Instant::now();
    for d in &dists {
        last_score = iw.compute_gray_stripped(&ref_gray, d).unwrap().score;
    }
    cubecl::future::block_on(client.sync()).expect("client.sync");
    let total_ms = t.elapsed().as_secs_f64() * 1e3;
    let per_dist_ms = total_ms / n_dist as f64;

    Row {
        w,
        h,
        h_body,
        mode: "uncached_strip",
        n_dist,
        total_ms,
        per_dist_ms,
        last_score,
    }
}

fn fmt_row(r: &Row) -> String {
    let mp = (r.w as f64 * r.h as f64) / 1e6;
    format!(
        "{w}x{h},{mp:.2},{mode},body={body},{n_dist},{total:.3},{per:.3},{score:.6}",
        w = r.w,
        h = r.h,
        mp = mp,
        mode = r.mode,
        body = r.h_body,
        n_dist = r.n_dist,
        total = r.total_ms,
        per = r.per_dist_ms,
        score = r.last_score,
    )
}

fn main() {
    println!("iwssim-gpu bench: cached-ref vs uncached strip (1 ref + N dist)");
    let header = "w_h,mp,mode,strip_body,n_dist,total_ms,per_dist_ms,score";
    println!("{header}");

    let n_warmup = 2;
    let n_dist = 16; // RD-search-ish: 1 ref + 16 dists per row.

    // 4 MP is the recovery brief's target. 12 MP is a sanity point.
    // Strip body 1024 matches the production sweep config.
    let configs: &[(u32, u32, u32)] = &[
        (2048, 2048, 1024), // 4 MP, body=1024
        (4000, 3000, 1024), // 12 MP, body=1024
    ];

    let mut rows: Vec<Row> = Vec::new();

    for &(w, h, body) in configs {
        let r_uncached = bench_uncached(w, h, body, n_dist, n_warmup);
        println!("{}", fmt_row(&r_uncached));
        rows.push(r_uncached);

        let r_cached = bench_cached(w, h, body, n_dist, n_warmup);
        println!("{}", fmt_row(&r_cached));
        rows.push(r_cached);
    }

    // Print speedup summary line per size.
    println!();
    println!("# Per-distortion speedup of cached-ref over uncached strip:");
    for chunk in rows.chunks(2) {
        if chunk.len() == 2 {
            let unc = &chunk[0];
            let cac = &chunk[1];
            let speedup = unc.per_dist_ms / cac.per_dist_ms;
            println!(
                "# {w}x{h} body={body}: uncached={u:.2}ms/dist  cached={c:.2}ms/dist  ({s:.2}x)",
                w = unc.w,
                h = unc.h,
                body = unc.h_body,
                u = unc.per_dist_ms,
                c = cac.per_dist_ms,
                s = speedup,
            );
        }
    }

    let date = std::env::var("BENCH_DATE").unwrap_or_else(|_| "2026-05-22".to_string());
    let out_path = format!("benchmarks/iwssim_cachedref_strip_{date}.csv");
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
