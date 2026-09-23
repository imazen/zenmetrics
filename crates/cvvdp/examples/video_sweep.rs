//! Video sweep — measures `cvvdp::score_video` wall time across clip
//! sizes and frame counts, emits a TSV. Baseline for the V5 SIMD work:
//! the video path is scalar-only as of the V3 port.
//!
//! Run with:
//!
//! ```bash
//! cargo run -p cvvdp --release --example video_sweep -- --output bench.tsv
//! ```
//!
//! Output columns:
//! `size_w  size_h  pixels  n_frames  fps  clip_ms  ms_per_frame`.
//!
//! Clip content: deterministic scrolling gradient + per-frame noise
//! (exercises all temporal channels — static content would skip the
//! transient channel's real work in any real-world clip anyway).

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use cvvdp::{CvvdpParams, DisplayGeometry, score_video};

const SIZES: &[(u32, u32)] = &[(256, 256), (512, 512), (1280, 720), (1920, 1080)];
const FRAMES: &[usize] = &[12, 24];
const FPS: f32 = 30.0;

fn make_frame(w: u32, h: u32, t: usize, seed: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h * 3];
    let mut s = seed.wrapping_add(t as u32);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            // Scrolling texture: phase shifts with t.
            let gx = (((x + t * 3) as f32 / w as f32 * 255.0) as u8) as i32;
            let gy = ((y as f32 / h as f32 * 200.0 + 30.0) as u8) as i32;
            s = s.wrapping_mul(48271);
            let noise = ((s >> 24) as i32 - 128) / 16;
            out[i] = (gx + noise).clamp(0, 255) as u8;
            out[i + 1] = (gy + noise / 2).clamp(0, 255) as u8;
            out[i + 2] = ((gx + gy) / 2 + noise / 4).clamp(0, 255) as u8;
        }
    }
    out
}

fn distort_clip(frames: &[Vec<u8>], seed: u32) -> Vec<Vec<u8>> {
    // Per-frame flicker + noise — hits the transient channel.
    frames
        .iter()
        .enumerate()
        .map(|(t, f)| {
            let mut s = seed.wrapping_add(t as u32 * 7919);
            let gain = 1.0 + 0.06 * ((t % 4) as f32 - 1.5);
            f.iter()
                .map(|&v| {
                    s = s.wrapping_mul(48271);
                    let d = ((s >> 24) as i32 - 128) / 8;
                    ((v as f32 * gain) as i32 + d).clamp(0, 255) as u8
                })
                .collect()
        })
        .collect()
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
        format!("benchmarks/cvvdp_cpu_video_sweep_{date}.tsv")
    });

    let parent = std::path::Path::new(&path)
        .parent()
        .unwrap_or(std::path::Path::new("."));
    std::fs::create_dir_all(parent).ok();
    let mut out = File::create(&path)?;
    writeln!(
        out,
        "size_w\tsize_h\tpixels\tn_frames\tfps\tclip_ms\tms_per_frame"
    )?;
    eprintln!("Writing TSV to {path}");

    for &(w, h) in SIZES {
        for &n in FRAMES {
            let refs: Vec<Vec<u8>> = (0..n).map(|t| make_frame(w, h, t, 1234)).collect();
            let dists = distort_clip(&refs, 9876);

            // Warmup.
            let _ = score_video(
                &refs,
                &dists,
                w,
                h,
                FPS,
                CvvdpParams::default(),
                DisplayGeometry::STANDARD_4K,
            )
            .unwrap();

            let mut times = Vec::new();
            let reps = if w * h >= 1280 * 720 { 3 } else { 5 };
            for _ in 0..reps {
                let t0 = Instant::now();
                let jod = score_video(
                    &refs,
                    &dists,
                    w,
                    h,
                    FPS,
                    CvvdpParams::default(),
                    DisplayGeometry::STANDARD_4K,
                )
                .unwrap();
                assert!(jod.is_finite());
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let clip_ms = times[times.len() / 2];
            let ms_per_frame = clip_ms / n as f64;
            writeln!(
                out,
                "{w}\t{h}\t{}\t{n}\t{FPS}\t{clip_ms:.3}\t{ms_per_frame:.3}",
                w * h
            )?;
            eprintln!("{w}x{h} x{n}f: {clip_ms:.1} ms ({ms_per_frame:.2} ms/frame)");
        }
    }
    Ok(())
}
