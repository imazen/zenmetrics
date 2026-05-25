//! Paired interleaved A/B end-to-end bench for the W44 SIMD CSF
//! chunk. Measures `Cvvdp::score` wall-time before vs after the
//! CSF apply SIMD rewire.
//!
//! Same shape as `time_masking_paired_ab.rs`:
//! - 3 warmup calls
//! - N (default 30) timed calls
//! - Report best / median / mean / stddev
//!
//! Intended invocation: build the pre-Chunk-5 binary on the parent
//! commit, the post-Chunk-5 binary on this commit, run each 5 rounds
//! interleaved per size, aggregate via best/median across rounds.

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use cvvdp_cpu::{Cvvdp, CvvdpParams};

const SIZES: &[(u32, u32)] = &[(256, 256), (512, 512), (1024, 1024), (2048, 2048)];

fn make_image(w: u32, h: u32, seed: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 3];
    let mut s = seed;
    for v in out.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *v = (s >> 16) as u8;
    }
    out
}

fn distort(src: &[u8], seed: u32) -> Vec<u8> {
    let mut out = src.to_vec();
    let mut s = seed;
    for v in out.iter_mut() {
        s = s.wrapping_mul(48271);
        let delta = ((s >> 24) as i32 - 128) / 8;
        *v = (*v as i32 + delta).clamp(0, 255) as u8;
    }
    out
}

fn run_one(w: u32, h: u32, n: usize) -> (f64, f64, f64, f64) {
    let r = make_image(w, h, 1234);
    let d = distort(&r, 9876);
    let mut cv = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();
    for _ in 0..3 {
        let _ = cv.score(&r, &d).unwrap();
    }
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        let _ = cv.score(&r, &d).unwrap();
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let best = samples[0];
    let median = samples[n / 2];
    let mean = samples.iter().sum::<f64>() / n as f64;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let stddev = var.sqrt();
    (best, median, mean, stddev)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut output_path: Option<String> = None;
    let mut iters: usize = 30;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output_path = Some(args[i + 1].clone());
            i += 2;
        } else if args[i] == "--iters" && i + 1 < args.len() {
            iters = args[i + 1].parse().unwrap_or(30);
            i += 2;
        } else {
            i += 1;
        }
    }
    let path = output_path.unwrap_or_else(|| "/tmp/time_csf_paired_ab.tsv".to_string());
    let parent = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).ok();
    let mut out = File::create(&path)?;
    writeln!(
        out,
        "size_w\tsize_h\tpixels\titers\tbest_ms\tmedian_ms\tmean_ms\tstddev_ms"
    )?;
    eprintln!("Writing TSV to {path} (iters={iters})");

    for &(w, h) in SIZES {
        let (best, median, mean, stddev) = run_one(w, h, iters);
        let px = (w as u64) * (h as u64);
        writeln!(
            out,
            "{w}\t{h}\t{px}\t{iters}\t{best:.3}\t{median:.3}\t{mean:.3}\t{stddev:.3}"
        )?;
        out.flush()?;
        eprintln!("  {w}x{h}: best={best:.2} median={median:.2} mean={mean:.2} ± {stddev:.2} ms");
    }
    eprintln!("done");
    Ok(())
}
