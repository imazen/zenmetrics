//! Benchmark-only successor probe. Compile beside the hash-bound, unmodified
//! production hdrvdp_aic.rs emitted by first_boundary_hdrvdp_plan.py.
#![forbid(unsafe_code)]

use std::{env, fs, hint::black_box, path::Path, time::Instant};

mod decode {
    // The production input carrier's fields; decoding is outside this probe.
    pub(crate) struct Rgb8Image {
        pub(crate) pixels: Vec<u8>,
        pub(crate) width: u32,
        pub(crate) height: u32,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum MetricKind {
    Hdrvdp,
    Hdrvdp3,
    // Compile the complete production module, including unchanged AIC glue.
    Hdrvdp3Aic2026,
}

impl MetricKind {
    fn name(self) -> &'static str {
        match self {
            Self::Hdrvdp => "hdrvdp",
            Self::Hdrvdp3 => "hdrvdp3",
            Self::Hdrvdp3Aic2026 => "hdrvdp3-aic2026",
        }
    }

    fn column_names(self) -> &'static [&'static str] {
        match self {
            Self::Hdrvdp => &[zenmetrics_api::hdrvdp_cpu::HDRVDP_COLUMN_NAME],
            Self::Hdrvdp3 => &[zenmetrics_api::hdrvdp_cpu::v3::HDRVDP3_COLUMN_NAME],
            Self::Hdrvdp3Aic2026 => &["hdrvdp3_aic2026_q", "hdrvdp3_aic2026_jod"],
        }
    }
}

#[allow(dead_code)]
#[path = "first_boundary_production_hdrvdp_aic.rs"]
mod production;

fn read(path: &Path, width: u32, height: u32) -> Result<decode::Rgb8Image, String> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(3))
        .ok_or("RGB byte count overflow")?;
    let pixels = fs::read(path).map_err(|e| e.to_string())?;
    if pixels.len() != expected {
        return Err("raw RGB byte count differs from declared geometry".into());
    }
    Ok(decode::Rgb8Image { pixels, width, height })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 10 {
        return Err("metric width height reference.rgb distorted.rgb threads first|warm samples output-column".into());
    }
    let kind = match args[1].as_str() {
        "hdrvdp" => MetricKind::Hdrvdp,
        "hdrvdp3" => MetricKind::Hdrvdp3,
        _ => return Err("only base sRGB-display routes belong to this successor".into()),
    };
    let width: u32 = args[2].parse()?;
    let height: u32 = args[3].parse()?;
    let threads: usize = args[6].parse()?;
    let samples: usize = args[8].parse()?;
    let first = match args[7].as_str() {
        "first" if samples == 1 => true,
        "warm" if samples > 0 => false,
        _ => return Err("first requires one call; warm requires positive samples".into()),
    };
    if width == 0 || height == 0 || threads == 0 {
        return Err("positive geometry and explicit thread budget required".into());
    }
    // Do not pre-create Rayon workers: first-call lazy setup remains timed.
    if env::var("RAYON_NUM_THREADS")? != threads.to_string() {
        return Err("RAYON_NUM_THREADS must equal the explicit command budget".into());
    }
    if args[9] != kind.column_names()[0] {
        return Err("output column differs from production declared column".into());
    }
    let reference = read(Path::new(&args[4]), width, height)?;
    let distorted = read(Path::new(&args[5]), width, height)?;
    println!("BOUNDARY production=4f89155cd8ef766c285fcfbfed9d40def047a836 ingress=RGB8_sRGB encoding=sRGB-display ppd=30 output={} phase={} threads={threads}", args[9], args[7]);
    let call = || -> Result<f64, Box<dyn std::error::Error>> {
        let result = production::score_srgb_display(kind, &reference, &distorted)?;
        if result.len() != 1 || result[0].0 != kind.column_names()[0] || !result[0].1.is_finite() {
            return Err("invalid production scalar result".into());
        }
        Ok(black_box(result[0].1))
    };
    if !first {
        black_box(call()?); // Explicit warmup; never emitted as a first call.
    }
    for _ in 0..samples {
        let start = Instant::now();
        let value = call()?;
        let elapsed = start.elapsed().as_nanos();
        println!("{} {elapsed} {value:.17}", if first { "COLD" } else { "SAMPLE" });
        println!("SCORE_BITS {:016x}", value.to_bits());
    }
    Ok(())
}
