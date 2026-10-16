//! Wall-time harness for the AIC-4 CPU metrics in this crate.
//! Scores one synthetic RGB8 pair and prints ms/call plus the exact score bits.
//!
//!   cargo run --release -p zenmetrics-cli --example aic_perf_wall -- \
//!       <metric> [w h reps]
//!
//! `<metric>` is any `--metric` name (`ciede2000`, `flip`, `flip-v17`,
//! `cw-ssim`, `ssimulacra1`, `ws-hv`, `psnr-cb`, `psnr-cr`). The pair is the
//! same procedural pattern the crate wall examples use, so numbers are
//! comparable to them.
use clap::ValueEnum;
use std::time::Instant;
use zenmetrics_cli::{
    decode::Rgb8Image,
    metrics::{GpuRuntime, MetricKind, run_metric},
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: aic_perf_wall <metric> [w h reps]");
        std::process::exit(64);
    }
    let kind = MetricKind::from_str(&args[1], false).unwrap();
    let w: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let h: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1024);
    let reps: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(2);

    let mut reference = vec![0u8; w * h * 3];
    let mut distorted = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let v = ((x * 7 + y * 13) ^ (x * y % 251)) as u8;
            reference[i..i + 3].copy_from_slice(&[v, v.wrapping_add(3), v]);
            distorted[i..i + 3].copy_from_slice(&[v.wrapping_add(9), v, v.wrapping_add(5)]);
        }
    }
    let r = Rgb8Image {
        pixels: reference,
        width: w as u32,
        height: h as u32,
    };
    let d = Rgb8Image {
        pixels: distorted,
        width: w as u32,
        height: h as u32,
    };

    let mut out = Vec::new();
    let start = Instant::now();
    for _ in 0..reps {
        out = run_metric(kind, &r, &d, GpuRuntime::Cpu).unwrap();
    }
    let ms = start.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let bits: String = out
        .iter()
        .map(|(n, s)| format!("{n}={s:.17} bits={:016x}", s.to_bits()))
        .collect::<Vec<_>>()
        .join(" ");
    println!("{w}x{h} {reps} reps {}: {ms:.2} ms/call ({bits})", args[1]);
}
