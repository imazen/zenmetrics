//! Per-allocation working-set tally for `Iwssim::new` vs
//! `Iwssim::new_strip`. Prints a CSV of (image_size, mode, scale,
//! plane_bytes) to stdout and a summary table to stderr.
//!
//! Build + run:
//! ```bash
//! cargo run --release -p iwssim-gpu --example strip_memory_tally \
//!     --no-default-features --features cubecl-types,cuda
//! ```
//!
//! The numbers are derived from the `Scale::new` allocation count
//! (19 f32 planes per scale, hand-counted from `pipeline.rs::struct
//! Scale`) plus the per-instance scratch (src_u32 staging, partials,
//! sums, cov_partials). No GPU is touched — this is pure arithmetic
//! over the documented buffer sizes.
//!
//! Used to answer: "what does the production sweep worker save by
//! switching to strip mode at 1 / 4 / 12 / 24 MP?"

const PLANES_PER_SCALE: u64 = 19;
const NUM_SCALES: usize = 5;
/// `STRIP_DEFAULT_HALO` from `pipeline.rs`. Hardcoded here because
/// the constant is `pub` but the example would otherwise need to
/// reach into the crate.
const STRIP_DEFAULT_HALO: u32 = 256;

fn scale_dims(w: u32, h: u32) -> Vec<(u32, u32)> {
    let mut dims = Vec::with_capacity(NUM_SCALES);
    let mut hh = h;
    let mut ww = w;
    for _ in 0..NUM_SCALES {
        dims.push((ww, hh));
        ww = ww.div_ceil(2);
        hh = hh.div_ceil(2);
    }
    dims
}

/// Bytes for all `Scale::new` allocations across the 5-scale pyramid.
fn scales_bytes(w: u32, h: u32) -> u64 {
    let mut total: u64 = 0;
    for (ww, hh) in scale_dims(w, h) {
        total += (ww as u64) * (hh as u64) * 4 * PLANES_PER_SCALE;
    }
    total
}

/// Per-instance scratch (src_u32_a/b + partials + sums + cov_partials).
/// src_u32 staging is sized for the upload plane (strip-alloc-h × w
/// in strip mode, full image in whole mode); partials / sums /
/// cov_partials are fixed-size buffers independent of the input.
fn scratch_bytes(upload_w: u32, upload_h: u32) -> u64 {
    let src_u32_bytes = (upload_w as u64) * (upload_h as u64) * 4; // u32 packed
    // Constants verified against pipeline.rs / reduction.rs 2026-05-28:
    // NUM_SLOTS=9, NUM_BLOCKS=16 (reduction.rs), BLOCK_SIZE=256.
    let n_partials = 9 * 16 * 256; // NUM_SLOTS × NUM_BLOCKS × BLOCK_SIZE
    let partials_bytes = (n_partials as u64) * 4;
    let sums_bytes = 9 * 4; // NUM_SLOTS f32 sums
    // COV_MAX_CELLS=100 (pipeline.rs:394), COV_N_THREADS=COV_CUBE_COUNT(64)·COV_CUBE_DIM(256)=16384.
    let cov_partials_bytes: u64 = 100 * 64 * 256 * 4; // COV_MAX_CELLS × COV_N_THREADS
    // Two src_u32 buffers (a + b), one partials, one sums, one cov.
    src_u32_bytes * 2 + partials_bytes + sums_bytes + cov_partials_bytes
}

fn whole_total_mb(w: u32, h: u32) -> f64 {
    let b = scales_bytes(w, h) + scratch_bytes(w, h);
    (b as f64) / 1e6
}

fn strip_total_mb(w: u32, _full_image_h: u32, h_body: u32) -> f64 {
    let strip_alloc_h = h_body + 2 * STRIP_DEFAULT_HALO;
    let b = scales_bytes(w, strip_alloc_h) + scratch_bytes(w, strip_alloc_h);
    (b as f64) / 1e6
}

fn main() {
    // 1 / 4 / 12 / 24 MP image sizes.
    let sizes: &[(u32, u32)] = &[
        (1024, 1024),  // 1 MP
        (2048, 2048),  // 4 MP
        (4000, 3000),  // 12 MP
        (6000, 4000),  // 24 MP
    ];
    // The default sweep configuration uses body=1024. h_body=512 is
    // included for sensitivity to the strip-size knob.
    let bodies: &[u32] = &[512, 1024];

    println!("image_size,mp,mode,h_body,working_set_mb");
    for &(w, h) in sizes {
        let mp = (w as f64 * h as f64) / 1e6;
        let whole_mb = whole_total_mb(w, h);
        println!("{w}x{h},{mp:.2},whole,n/a,{whole_mb:.2}");
        for &body in bodies {
            let strip_mb = strip_total_mb(w, h, body);
            println!("{w}x{h},{mp:.2},strip,{body},{strip_mb:.2}");
        }
    }

    eprintln!();
    eprintln!("Summary (working_set_mb, whole vs strip_body=1024):");
    eprintln!("{:>10}  {:>8}  {:>10}  {:>10}  {:>8}", "size", "MP", "whole", "strip", "ratio");
    eprintln!("{}", "-".repeat(56));
    for &(w, h) in sizes {
        let mp = (w as f64 * h as f64) / 1e6;
        let whole_mb = whole_total_mb(w, h);
        let strip_mb = strip_total_mb(w, h, 1024);
        eprintln!(
            "{w:>5}x{h:<5}  {mp:>6.2}  {whole_mb:>8.1} MB  {strip_mb:>8.1} MB  {ratio:>6.2}×",
            ratio = whole_mb / strip_mb,
        );
    }
}
