//! Size sweep — measures cvvdp-cpu wall time per call across image
//! sizes + content classes, fits `α + β · pixels`, emits a TSV.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p cvvdp-cpu --release --example time_size_sweep -- --output bench.tsv
//! ```
//!
//! Output columns: `size_w  size_h  pixels  content_class  cold_ms  warm_ms`.
//!
//! `content_class ∈ {smooth, photo, screenshot}` — synthetic
//! generation: smooth = linear gradient + low noise, photo =
//! mid-amplitude pseudo-random, screenshot = high-contrast grid.

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use cvvdp_cpu::{Cvvdp, CvvdpParams};

const SIZES: &[(u32, u32)] = &[
    (64, 64),
    (128, 128),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
];
const CONTENT_CLASSES: &[&str] = &["smooth", "photo", "screenshot"];

fn make_image(w: u32, h: u32, class: &str, seed: u32) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    let mut out = vec![0u8; n * 3];
    let mut s = seed;
    match class {
        "smooth" => {
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) as usize) * 3;
                    let gx = (x as f32 / w as f32 * 255.0) as u8;
                    let gy = (y as f32 / h as f32 * 200.0 + 30.0) as u8;
                    s = s.wrapping_mul(48271);
                    let noise = ((s >> 24) as i32 - 128) / 16;
                    out[i] = (gx as i32 + noise).clamp(0, 255) as u8;
                    out[i + 1] = (gy as i32 + noise / 2).clamp(0, 255) as u8;
                    out[i + 2] = ((gx as u32 + gy as u32) / 2).clamp(0, 255) as u8;
                }
            }
        }
        "photo" => {
            for i in 0..n * 3 {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                out[i] = (s >> 16) as u8;
            }
        }
        "screenshot" => {
            for y in 0..h {
                for x in 0..w {
                    let i = ((y * w + x) as usize) * 3;
                    let block = ((x / 16) ^ (y / 16)) & 1;
                    let bg = if block == 1 { 230u8 } else { 32u8 };
                    out[i] = bg;
                    out[i + 1] = bg;
                    out[i + 2] = bg;
                    if (x % 16 == 0) || (y % 16 == 0) {
                        out[i] = 200;
                        out[i + 1] = 200;
                        out[i + 2] = 200;
                    }
                }
            }
        }
        _ => unreachable!(),
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

fn run_one(w: u32, h: u32, class: &str) -> (f64, f64) {
    let r = make_image(w, h, class, 1234);
    let d = distort(&r, 9876);
    let mut cv = Cvvdp::new(w, h, CvvdpParams::default()).unwrap();

    // 3-call warmup.
    let _ = cv.score(&r, &d).unwrap();
    let _ = cv.score(&r, &d).unwrap();
    let _ = cv.score(&r, &d).unwrap();

    // 5 cold-path timings.
    let mut cold = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let _ = cv.score(&r, &d).unwrap();
        cold.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    cold.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cold_ms = cold[cold.len() / 2];

    // 5 warm-path timings (warm_reference once, score N times).
    cv.warm_reference(&r).unwrap();
    let _ = cv.score_with_warm_ref(&d).unwrap(); // warmup
    let mut warm = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        let _ = cv.score_with_warm_ref(&d).unwrap();
        warm.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    warm.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let warm_ms = warm[warm.len() / 2];

    (cold_ms, warm_ms)
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut output_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    let path = output_path.unwrap_or_else(|| {
        let date = std::process::Command::new("date")
            .arg("-u")
            .arg("+%Y-%m-%d")
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown-date".to_string());
        format!("benchmarks/cvvdp_cpu_size_sweep_{date}.tsv")
    });

    let parent = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).ok();
    let mut out = File::create(&path)?;
    writeln!(out, "size_w\tsize_h\tpixels\tcontent_class\tcold_ms\twarm_ms")?;
    eprintln!("Writing TSV to {path}");

    for &(w, h) in SIZES {
        for class in CONTENT_CLASSES {
            let (cold, warm) = run_one(w, h, class);
            let pixels = (w as u64) * (h as u64);
            writeln!(
                out,
                "{w}\t{h}\t{pixels}\t{class}\t{cold:.3}\t{warm:.3}"
            )?;
            out.flush()?;
            eprintln!(
                "  {w}x{h} {class}: cold={cold:.2}ms warm={warm:.2}ms ({:.2} MP)",
                pixels as f64 / 1e6
            );
        }
    }
    eprintln!("done");
    Ok(())
}
