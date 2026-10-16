//! Compute-only IQA benchmark. File IO/preparation and model warmup are outside zenbench.
#[path = "../iqa_compete_maps.rs"]
mod maps;
use serde_json::{Value, json};
use std::{env, fs, hint::black_box, path::Path, time::Duration};
use zenmetrics_cli::{
    decode::{Rgb8Image, decode_image_to_rgb8},
    metrics::{GpuRuntime, MetricKind, display::parse_display, run_metric_display},
};

fn prepare(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let r = decode_image_to_rgb8(Path::new(&args[2]))?;
    let d = decode_image_to_rgb8(Path::new(&args[3]))?;
    assert_eq!((r.width, r.height), (d.width, d.height));
    let out = Path::new(&args[4]);
    fs::create_dir(out)?;
    let mut rows = Vec::new();
    for (name, w, h) in [
        ("cid22", 512, 512),
        ("aic-small", 840, 944),
        ("aic-medium", 1680, 1888),
        ("aic-large", 3355, 2516),
        ("2k", 2048, 1080),
        ("4k", 3840, 2160),
        ("8k", 7680, 4320),
    ] {
        let down = w <= r.width && h <= r.height;
        for (label, src) in [("reference", &r), ("distorted", &d)] {
            let pixels = if down {
                use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};
                let c = ResizeConfig::builder(r.width, r.height, w, h)
                    .filter(Filter::Lanczos)
                    .format(PixelDescriptor::RGB8_SRGB)
                    .build();
                Resizer::new(&c).resize(&src.pixels)
            } else {
                let mirror = |v: u32, n: u32| {
                    let v = v % (2 * n);
                    if v < n { v } else { 2 * n - 1 - v }
                };
                let mut pixels = vec![0; w as usize * h as usize * 3];
                for y in 0..h {
                    for x in 0..w {
                        let a = (mirror(y, r.height) * r.width + mirror(x, r.width)) as usize * 3;
                        let b = (y * w + x) as usize * 3;
                        pixels[b..b + 3].copy_from_slice(&src.pixels[a..a + 3]);
                    }
                }
                pixels
            };
            fs::write(out.join(format!("{name}-{label}.rgb")), pixels)?;
        }
        rows.push(json!({"size_name":name,"width":w,"height":h,"reference":format!("{name}-reference.rgb"),"distorted":format!("{name}-distorted.rgb"),"method":if down {"zenresize Lanczos RGB8 sRGB"}else{"mirror-tile largest real pair"}}));
    }
    fs::write(out.join("pairs.json"), serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
fn load() -> Vec<f64> {
    fs::read_to_string("/proc/loadavg")
        .ok()
        .map(|s| {
            s.split_whitespace()
                .take(3)
                .filter_map(|v| v.parse().ok())
                .collect()
        })
        .unwrap_or_default()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    if a.get(1).is_some_and(|s| s == "prepare") {
        return prepare(&a);
    }
    if a.get(1).is_some_and(|s| s == "inventory") {
        println!("{}",serde_json::to_string(&MetricKind::all().iter().map(|k|json!({"metric_name":k.name(),"gpu":k.requires_gpu(),"compiled":k.scoring_compiled_in(),"columns":k.column_names()})).collect::<Vec<_>>())?);
        return Ok(());
    }
    let config: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let w = config["width"].as_u64().ok_or("width")? as u32;
    let h = config["height"].as_u64().ok_or("height")? as u32;
    let read = |key: &str| -> Result<Rgb8Image, Box<dyn std::error::Error>> {
        let pixels = fs::read(config[key].as_str().ok_or("input path")?)?;
        if pixels.len() != w as usize * h as usize * 3 {
            return Err("raw RGB byte count".into());
        }
        Ok(Rgb8Image {
            width: w,
            height: h,
            pixels,
        })
    };
    let r: &'static Rgb8Image = Box::leak(Box::new(read("reference")?));
    let d: &'static Rgb8Image = Box::leak(Box::new(read("distorted")?));
    let name = config["metric_name"].as_str().ok_or("metric name")?;
    let kind = *MetricKind::all()
        .iter()
        .find(|k| k.name() == name)
        .ok_or("unknown metric")?;
    let display = Box::leak(Box::new(parse_display("standard_fhd")?));
    let display: &'static _ = display;
    let before = load();
    if before.first().is_some_and(|v| *v > 4.) {
        return Err(format!("busy load {before:?}").into());
    }
    let mut score: maps::Call = if name == "cvvdp" {
        maps::cvvdp_score(r, d, false)?
    } else {
        Box::new(move || {
            run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(display))
                .map(|v| v[0].1)
                .map_err(|e| e.to_string())
        })
    };
    let first = score(); // Model/initialization warmup is outside the timer.
    let mut row = config.clone();
    row["load_before"] = json!(before);
    if let Err(e) = first {
        row["status"] = json!(format!("skipped:{e}"));
        row["n"] = json!(0);
        row["load_after"] = json!(load());
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
        row["mode"] = json!("score+map");
        row["status"] =
            json!("skipped:score-path-refused; map capability not qualified on this input");
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
        return Ok(());
    }
    let first = first?;
    let mut map_call = maps::native(name, r, d);
    let map_first = map_call.as_mut().map(|f| f());
    let mut unsupported = row.clone();
    unsupported["mode"] = json!("score+map");
    unsupported["n"] = json!(0);
    match &map_first {
        None => {
            unsupported["status"] = json!("unsupported");
            unsupported["notes"] =
                json!("No qualified native-map adapter for this metric; see coverage inventory");
            println!("IQA_ROW {}", serde_json::to_string(&unsupported)?);
        }
        Some(Err(e)) => {
            unsupported["status"] = json!(format!("skipped:map-input:{e}"));
            println!("IQA_ROW {}", serde_json::to_string(&unsupported)?);
            map_call = None;
        }
        Some(Ok(_)) => {}
    }
    let result = zenbench::run(|suite| {
        suite.group(name, |g| {
            g.config()
                .min_rounds(10)
                .max_rounds(30)
                .auto_rounds(false)
                .warmup_time(Duration::from_millis(100))
                .max_time(Duration::from_secs(10))
                .max_wall_time(Duration::from_secs(120));
            g.bench("score", move |b| {
                b.iter(|| {
                    let value = score().expect("metric failed after warmup");
                    black_box(value)
                })
            });
            if let Some(mut call) = map_call {
                g.bench("score+map", move |b| {
                    b.iter(|| black_box(call().expect("map failed after warmup")))
                });
            }
        });
    });
    let cmp = result.comparisons.first().ok_or("no comparison")?;
    for (index, bm) in cmp.benchmarks.iter().enumerate() {
        let mut row = row.clone();
        row["mode"] = json!(bm.name);
        let mut samples: Vec<f64> = cmp
            .samples
            .iter()
            .map(|s| s.compensated_ns[index] as f64 / s.iterations as f64)
            .collect();
        samples.sort_by(f64::total_cmp);
        if samples.len() < 10 {
            return Err(format!("insufficient zenbench rounds {}", samples.len()).into());
        }
        let percentile = |p: f64| {
            let i = p * (samples.len() - 1) as f64;
            let lo = i.floor() as usize;
            let hi = i.ceil() as usize;
            samples[lo] + (samples[hi] - samples[lo]) * (i - lo as f64)
        };
        row["zenbench_samples"] = serde_json::to_value(&cmp.samples)?;
        row["rayon_threads_observed"] = json!(rayon::current_num_threads());
        #[cfg(target_arch = "x86_64")]
        {
            row["simd_tier"] = json!(if std::is_x86_feature_detected!("avx512f") {
                "v4-capable"
            } else if std::is_x86_feature_detected!("avx2") {
                "v3-capable"
            } else {
                "scalar-capable"
            });
        }
        #[cfg(target_arch = "aarch64")]
        {
            row["simd_tier"] = json!("neon-capable");
        }
        row["n"] = json!(samples.len());
        row["median_ns"] = json!(percentile(0.5));
        row["p10_ns"] = json!(percentile(0.1));
        row["p90_ns"] = json!(percentile(0.9));
        row["mean_ns"] = json!(bm.summary.mean);
        let measured = if index == 0 {
            first
        } else {
            *map_first.as_ref().unwrap().as_ref().unwrap()
        };
        row["score"] = json!(measured);
        row["ref_score"] = json!(first);
        row["abs_delta"] = json!((measured - first).abs());
        row["relative_delta"] =
            json!((measured - first).abs() / first.abs().max(f64::MIN_POSITIVE));
        row["status"] = json!(if measured.to_bits() == first.to_bits() {
            "ok"
        } else {
            "variant_mismatch"
        });
        row["load_after"] = json!(load());
        row["notes"] = json!(
            "RGB8 resident; CLI compute call incl. metric colour conversion; initialization warmup excluded; zenbench round batch means, not individual-call percentiles"
        );
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
    }
    Ok(())
}
