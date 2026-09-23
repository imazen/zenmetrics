//! `video_vs_ssim2` — cvvdp video scoring vs fast-ssim2 applied
//! per-frame, on the same deterministic clip. One metric per
//! invocation so wall / user-CPU / peak-RSS attribute cleanly.
//!
//! ```bash
//! cargo run -p cvvdp --release --example video_vs_ssim2 -- \
//!     <cvvdp|ssim2|gen> <w> <h> <n_frames> [fps] [reps]
//! ```
//!
//! `RAYON_NUM_THREADS=1|8` sizes the shared rayon pool both crates
//! use (fast-ssim2's `rayon` feature is enabled for this example).
//! Frames are synthesized lazily inside the timed loop, so peak RSS
//! reflects each metric's own working set — a resident 1080p×24 sRGB
//! clip would add ~150 MB/side and drown the streaming difference.
//! `gen` mode runs only the frame synthesis, so `wall − gen` ≈
//! scoring-only time.
//!
//! stdout TSV row:
//! `metric  w  h  n_frames  fps  threads  reps  wall_ms  gen_ms
//!  user_ms  sys_ms  score  vmhwm_kb`
//! `wall_ms`/`user_ms`/`sys_ms` are medians over `reps` (Linux:
//! utime/stime from /proc/self/stat deltas at USER_HZ=100).

use std::env;
use std::time::Instant;

use cvvdp::{CvvdpParams, DisplayGeometry, VideoScorer};

fn make_frame(w: u32, h: u32, t: usize, seed: u32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let mut out = vec![0u8; w * h * 3];
    let mut s = seed.wrapping_add(t as u32);
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
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

fn distort_frame(f: &[u8], t: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_add(t as u32 * 7919);
    let gain = 1.0 + 0.06 * ((t % 4) as f32 - 1.5);
    f.iter()
        .map(|&v| {
            s = s.wrapping_mul(48271);
            let d = ((s >> 24) as i32 - 128) / 8;
            ((v as f32 * gain) as i32 + d).clamp(0, 255) as u8
        })
        .collect()
}

/// (utime, stime) jiffies from /proc/self/stat (fields 14–15 after
/// the last ')' — comm can contain spaces). Linux USER_HZ is 100.
fn cpu_jiffies() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    let get = |i: usize| f.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    // Index 0 of `after` is field 3 (state) → utime is index 11, stime 12.
    (get(11), get(12))
}

fn vmhwm_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmHWM"))
        .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        .unwrap_or(0)
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Timed rep: returns (score, wall_ms, user_ms, sys_ms).
fn rep<F: FnMut() -> f64>(f: &mut F) -> (f64, f64, f64, f64) {
    let (u0, s0) = cpu_jiffies();
    let t0 = Instant::now();
    let score = f();
    let wall = t0.elapsed().as_secs_f64() * 1000.0;
    let (u1, s1) = cpu_jiffies();
    (
        score,
        wall,
        (u1 - u0) as f64 * 10.0,
        (s1 - s0) as f64 * 10.0,
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <cvvdp|ssim2|gen> <w> <h> <n_frames> [fps=30] [reps=3]",
            args[0]
        );
        std::process::exit(2);
    }
    let metric = args[1].as_str();
    let w: u32 = args[2].parse()?;
    let h: u32 = args[3].parse()?;
    let n: usize = args[4].parse()?;
    let fps: f32 = args.get(5).map(|s| s.parse()).transpose()?.unwrap_or(30.0);
    let reps: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let (wu, hu) = (w as usize, h as usize);

    let mut run = move || -> f64 {
        match metric {
            "gen" => {
                let mut acc = 0u64;
                for t in 0..n {
                    let rf = make_frame(w, h, t, 1234);
                    let df = distort_frame(&rf, t, 9876);
                    acc += rf[0] as u64 + df[0] as u64;
                }
                acc as f64
            }
            "cvvdp" => {
                let mut v = VideoScorer::new(
                    w,
                    h,
                    fps,
                    CvvdpParams::default(),
                    DisplayGeometry::STANDARD_4K,
                )
                .unwrap();
                for t in 0..n {
                    let rf = make_frame(w, h, t, 1234);
                    let df = distort_frame(&rf, t, 9876);
                    v.push_frame(&rf, &df).unwrap();
                }
                v.finish().unwrap() as f64
            }
            "ssim2" => {
                let mut sum = 0.0;
                for t in 0..n {
                    let rf = make_frame(w, h, t, 1234);
                    let df = distort_frame(&rf, t, 9876);
                    let ri = imgref::Img::new(bytemuck::cast_slice::<u8, [u8; 3]>(&rf), wu, hu);
                    let di = imgref::Img::new(bytemuck::cast_slice::<u8, [u8; 3]>(&df), wu, hu);
                    sum += fast_ssim2::compute_ssimulacra2(ri, di).unwrap();
                }
                sum / n as f64
            }
            _ => panic!("unknown metric {metric}"),
        }
    };

    let (mut walls, mut users, mut syss) = (Vec::new(), Vec::new(), Vec::new());
    let mut score = 0.0;
    for _ in 0..reps {
        let (sc, wall, user, sys) = rep(&mut run);
        score = sc;
        walls.push(wall);
        users.push(user);
        syss.push(sys);
    }
    let threads = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "auto".into());
    println!(
        "{metric}\t{w}\t{h}\t{n}\t{fps}\t{threads}\t{reps}\t{:.3}\t{:.3}\t{:.3}\t{:.6}\t{}",
        median(&mut walls),
        median(&mut users),
        median(&mut syss),
        score,
        vmhwm_kb(),
    );
    Ok(())
}
