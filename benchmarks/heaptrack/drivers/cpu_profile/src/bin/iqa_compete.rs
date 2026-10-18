//! IQA benchmark: resident-input cold construction/call or warmed zenbench rounds.
#[cfg(iqa_compete_dvifmish)]
mod iqa_compete_dvifmish;
#[allow(unexpected_cfgs)]
#[path = "../iqa_compete_maps.rs"]
mod maps;
use serde_json::{Value, json};
use std::{
    env, fs,
    hint::black_box,
    path::Path,
    time::{Duration, Instant},
};
#[cfg(iqa_compete_ingress)]
use zenmetrics_cli::decode;
use zenmetrics_cli::{
    decode::{Rgb8Image, decode_image_to_rgb8},
    metrics::{GpuRuntime, MetricKind, display::parse_display, run_metric_display},
};

#[allow(unexpected_cfgs)]
const MODEL_PRELOAD_ADAPTER: bool = cfg!(iqa_compete_cached);

struct ScoreObservations {
    expected: u64,
    last: std::sync::atomic::AtomicU64,
    different: std::sync::atomic::AtomicU64,
    count: std::sync::atomic::AtomicU64,
}
impl ScoreObservations {
    fn new(value: f64) -> Self {
        Self {
            expected: value.to_bits(),
            last: std::sync::atomic::AtomicU64::new(value.to_bits()),
            different: std::sync::atomic::AtomicU64::new(u64::MAX),
            count: std::sync::atomic::AtomicU64::new(0),
        }
    }
}
struct ObservedScore {
    value: f64,
    observations: &'static ScoreObservations,
}
impl Drop for ObservedScore {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let bits = self.value.to_bits();
        self.observations.last.store(bits, Relaxed);
        if bits != self.observations.expected {
            self.observations.count.fetch_add(1, Relaxed);
            let _ = self
                .observations
                .different
                .compare_exchange(u64::MAX, bits, Relaxed, Relaxed);
        }
    }
}

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
    let text = fs::read_to_string("/proc/loadavg").ok();
    #[cfg(target_os = "macos")]
    let text = text.or_else(|| {
        std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()
            .and_then(|v| String::from_utf8(v.stdout).ok())
            .map(|s| s.replace(['{', '}'], ""))
    });
    text.map(|s| {
        s.split_whitespace()
            .take(3)
            .filter_map(|v| v.parse().ok())
            .collect()
    })
    .unwrap_or_default()
}
#[path = "iqa_compete_timing/mod.rs"]
mod timing;
use timing::cold_call;

fn make_call(
    config: &Value,
    name: &str,
    kind: Option<MetricKind>,
    r: &'static Rgb8Image,
    d: &'static Rgb8Image,
    map: bool,
) -> Result<Option<maps::Call>, String> {
    #[cfg(iqa_compete_dvifmish)]
    if iqa_compete_dvifmish::supported(name) {
        let threads = config["threads"]
            .as_u64()
            .ok_or("explicit dvifmish thread budget required")?;
        return iqa_compete_dvifmish::make_call(
            name,
            r,
            d,
            map,
            usize::try_from(threads).map_err(|e| e.to_string())?,
        )
        .map(Some);
    }
    let kind = kind.ok_or("unknown metric")?;
    if name.ends_with("-gpu") {
        #[cfg(any(feature = "study-gpu-cuda", feature = "study-gpu-wgpu"))]
        return maps::gpu_score(
            name,
            r,
            d,
            config["gpu_runtime"]
                .as_str()
                .unwrap_or("explicit-runtime-required"),
            map,
        )
        .map(Some);
        #[cfg(not(any(feature = "study-gpu-cuda", feature = "study-gpu-wgpu")))]
        return Err("GPU metric requires a GPU benchmark build and explicit device".into());
    }
    // This constructor returns a real refusal rather than native()'s Option.
    if name == "cvvdp" {
        return maps::cvvdp_score(r, d, map).map(Some);
    }
    if map {
        return Ok(maps::native(name, r, d));
    }
    let display = parse_display("standard_fhd")?;
    let _ = config; // Used by GPU builds.
    Ok(Some(Box::new(move || {
        run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))
            .map(|v| v[0].1)
            .map_err(|e| e.to_string())
    })))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = env::args().collect();
    if a.get(1).is_some_and(|s| s == "prepare") {
        return prepare(&a);
    }
    if a.get(1).is_some_and(|s| s == "inventory") {
        #[allow(unused_mut)]
        let mut inventory = MetricKind::all().iter().map(|k|json!({"metric_name":k.name(),"gpu":k.requires_gpu(),"compiled":k.scoring_compiled_in(),"columns":k.column_names()})).collect::<Vec<_>>();
        #[cfg(iqa_compete_dvifmish)]
        inventory.extend(iqa_compete_dvifmish::inventory());
        println!("{}", serde_json::to_string(&inventory)?);
        return Ok(());
    }
    #[cfg(iqa_compete_ingress)]
    if a.get(1).is_some_and(|s| s == "prepare-planes") {
        let pairs_path = Path::new(&a[2]);
        let pairs: Vec<Value> = serde_json::from_slice(&fs::read(pairs_path)?)?;
        let out = Path::new(&a[3]);
        fs::create_dir(out)?;
        let mut manifest = Vec::new();
        for pair in pairs {
            for convention in ["709-420", "601-420", "601-444", "601-y", "709-y"] {
                let mut row = pair.clone();
                row["convention"] = json!(convention);
                for role in ["reference", "distorted"] {
                    let image = Rgb8Image {
                        width: pair["width"].as_u64().ok_or("width")? as u32,
                        height: pair["height"].as_u64().ok_or("height")? as u32,
                        pixels: fs::read(
                            pairs_path
                                .parent()
                                .ok_or("parent")?
                                .join(pair[role].as_str().ok_or("input")?),
                        )?,
                    };
                    let (w, h, bytes) = maps::competitor_planes(&image, convention)?;
                    let filename = format!(
                        "{}-{convention}-{role}.yuv",
                        pair["size_name"].as_str().ok_or("size name")?
                    );
                    fs::write(out.join(&filename), bytes)?;
                    row[role] = json!(filename);
                    row["width"] = json!(w);
                    row["height"] = json!(h);
                }
                manifest.push(row);
            }
        }
        fs::write(
            out.join("planes.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
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
    let input_load_start = Instant::now();
    let r: &'static Rgb8Image = Box::leak(Box::new(read("reference")?));
    let d: &'static Rgb8Image = Box::leak(Box::new(read("distorted")?));
    let input_load_ns = input_load_start.elapsed().as_nanos();
    let name = config["metric_name"].as_str().ok_or("metric name")?;
    let only_mode = config["only_mode"].as_str();
    if only_mode.is_some_and(|mode| !["score", "score+map"].contains(&mode)) {
        return Err("unknown isolated mode".into());
    }
    let timing_phase = config["timing_phase"].as_str().unwrap_or("warm");
    if !["cold", "warm"].contains(&timing_phase) {
        return Err("unknown timing phase".into());
    }
    if timing_phase == "cold" && (only_mode.is_none() || config["probe_only"] == true) {
        return Err("cold timing requires only_mode and cannot be a probe".into());
    }
    let only_map = only_mode == Some("score+map");
    let primary_mode = if only_map { "score+map" } else { "score" };
    let primary_arm = usize::from(only_map);
    let kind = MetricKind::all().iter().find(|k| k.name() == name).copied();
    #[allow(unused_mut)]
    let mut known = kind.is_some();
    #[cfg(iqa_compete_dvifmish)]
    {
        known |= iqa_compete_dvifmish::supported(name);
    }
    if !known {
        return Err("unknown metric".into());
    }
    let admitted_unix_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    #[cfg(target_os = "linux")]
    {
        let stat = fs::read_to_string("/proc/self/stat")?;
        let pgid: u32 = stat
            .rsplit_once(')')
            .ok_or("invalid process stat")?
            .1
            .split_whitespace()
            .nth(2)
            .ok_or("missing process group")?
            .parse()?;
        println!(
            "IQA_PROCESS {}",
            json!({"pid":std::process::id(),"pgid":pgid,"unix_s":admitted_unix_ns / 1_000_000_000})
        );
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    let before = load();
    if !config["probe_only"].as_bool().unwrap_or(false)
        && !config["locked_process_accounting"]
            .as_bool()
            .unwrap_or(false)
        && before.first().is_some_and(|v| *v > 4.)
    {
        return Err(format!("busy load {before:?}").into());
    }
    let mut row = config.clone();
    row["mode"] = json!(primary_mode);
    row["timing_phase"] = json!(timing_phase);
    row["input_load_ns"] = json!(input_load_ns);
    row["load_before"] = json!(before);
    row["admitted_unix_ns"] = json!(admitted_unix_ns);
    row["model_preload_adapter"] = json!(MODEL_PRELOAD_ADAPTER);
    let columns = kind
        .map(|k| k.column_names().to_vec())
        .unwrap_or_else(|| vec![name]);
    row["declared_columns"] = json!(columns);
    row["headline_declared_column"] = json!(columns[0]);
    #[cfg(iqa_compete_dvifmish)]
    if iqa_compete_dvifmish::supported(name) {
        row["benchmark_route"] = iqa_compete_dvifmish::inventory()
            .into_iter()
            .find(|route| route["metric_name"] == name)
            .expect("known route");
    }
    // Config fields carry input identity, but cannot supply observed results.
    for field in [
        "median_ns",
        "cold_elapsed_ns",
        "score",
        "score_bits",
        "n",
        "status",
    ] {
        row.as_object_mut()
            .ok_or("config must be an object")?
            .remove(field);
    }
    if timing_phase == "cold" {
        let mut unsupported = false;
        let (result, elapsed_ns) = cold_call(|| {
            make_call(&config, name, kind, r, d, only_map)?.ok_or_else(|| {
                unsupported = true;
                "No qualified native-map adapter for this metric".to_owned()
            })
        });
        row["n"] = json!(0);
        match result {
            Ok(value) if value.is_finite() => {
                row["n"] = json!(1);
                row["status"] = json!("ok");
                row["median_ns"] = json!(elapsed_ns);
                row["cold_elapsed_ns"] = json!(elapsed_ns);
                row["score"] = json!(value);
                row["score_bits"] = json!(format!("{:016x}", value.to_bits()));
            }
            Ok(value) => {
                row["status"] = json!("failed");
                row["error"] = json!("nonfinite_score");
                row["score_bits"] = json!(format!("{:016x}", value.to_bits()));
            }
            Err(error) => {
                row["status"] = json!(if unsupported { "unsupported" } else { "failed" });
                row["error"] = json!(error);
            }
        }
        row["notes"] = json!(
            "First scorer construction and evaluation in this process; raw input loading excluded; no warmup"
        );
        row["load_after"] = json!(load());
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
        return Ok(());
    }
    let mut score = match make_call(&config, name, kind, r, d, only_map) {
        Ok(Some(call)) => call,
        Ok(None) => {
            row["status"] = json!("unsupported");
            row["n"] = json!(0);
            row["load_after"] = json!(load());
            println!("IQA_ROW {}", serde_json::to_string(&row)?);
            return Ok(());
        }
        Err(error) => Box::new(move || Err(error.clone())),
    };
    let display = parse_display("standard_fhd")?;
    if config["probe_only"].as_bool().unwrap_or(false) {
        let kind = kind.ok_or("CLI probe is unavailable for benchmark-only routes")?;
        let warmup = run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))?;
        let values = run_metric_display(kind, r, d, GpuRuntime::Cpu, Some(&display))?;
        assert_eq!(
            warmup.iter().map(|(_, v)| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|(_, v)| v.to_bits()).collect::<Vec<_>>(),
            "probe changed after model warmup"
        );
        let probe = json!({"metric_name":name,"scores":values.iter().map(|(k,v)|(*k,format!("{:016x}",v.to_bits()))).collect::<Vec<_>>(),"model_preload_adapter":MODEL_PRELOAD_ADAPTER});
        #[cfg(iqa_compete_ingress)]
        let probe = if let Some(path) = config["map_dump_dir"].as_str() {
            let mut value = probe;
            value["maps"] = maps::vmaf_probe_dump(r, d, Path::new(path))?;
            value
        } else {
            probe
        };
        println!("IQA_PROBE {}", serde_json::to_string(&probe)?);
        return Ok(());
    }
    let warmup_start = std::time::Instant::now();
    let first = score(); // Model/initialization warmup is outside the timer.
    println!(
        "IQA_WARMUP {}",
        json!({"initialization_and_warmup_ns":warmup_start.elapsed().as_nanos(),"completed":true,"score_bits":first.as_ref().ok().map(|v|format!("{:016x}",v.to_bits()))})
    );
    if let Err(e) = first {
        row["status"] = if config["timing_phase"].is_string() {
            row["error"] = json!(e);
            json!("failed")
        } else {
            json!(format!("skipped:{e}"))
        };
        row["n"] = json!(0);
        row["load_after"] = json!(load());
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
        if only_mode.is_none() {
            row["mode"] = json!("score+map");
            row["status"] =
                json!("skipped:score-path-refused; map capability not qualified on this input");
            println!("IQA_ROW {}", serde_json::to_string(&row)?);
        }
        return Ok(());
    }
    let first = first?;
    if !first.is_finite() {
        row["status"] = if config["timing_phase"].is_string() {
            row["error"] = json!("nonfinite_score");
            json!("failed")
        } else {
            json!("skipped:nonfinite_score")
        };
        row["n"] = json!(0);
        row["score_bits"] = json!(format!("{:016x}", first.to_bits()));
        row["load_after"] = json!(load());
        println!("IQA_ROW {}", serde_json::to_string(&row)?);
        if only_mode.is_none() {
            row["mode"] = json!("score+map");
            row["status"] = json!("skipped:nonfinite_score");
            println!("IQA_ROW {}", serde_json::to_string(&row)?);
        }
        return Ok(());
    }
    let mut map_call = if only_mode.is_none() {
        maps::native(name, r, d)
    } else {
        None
    };
    #[cfg(any(feature = "study-gpu-cuda", feature = "study-gpu-wgpu"))]
    if only_mode.is_none() && name.ends_with("-gpu") {
        map_call = maps::gpu_score(
            name,
            r,
            d,
            config["gpu_runtime"]
                .as_str()
                .unwrap_or("explicit-runtime-required"),
            true,
        )
        .ok();
    }
    let map_first = map_call.as_mut().map(|f| f());
    let mut unsupported = row.clone();
    unsupported["mode"] = json!("score+map");
    unsupported["n"] = json!(0);
    if only_mode.is_none() {
        match &map_first {
            None => {
                unsupported["status"] = json!("unsupported");
                unsupported["notes"] = json!(
                    "No qualified native-map adapter for this metric; see coverage inventory"
                );
                println!("IQA_ROW {}", serde_json::to_string(&unsupported)?);
            }
            Some(Err(e)) => {
                unsupported["status"] = json!(format!("skipped:map-input:{e}"));
                println!("IQA_ROW {}", serde_json::to_string(&unsupported)?);
                map_call = None;
            }
            Some(Ok(_)) => {}
        }
    }
    let chunk_budget_ms = config["chunk_budget_ms"].as_u64().unwrap_or(850_000);
    if !(1..=850_000).contains(&chunk_budget_ms) {
        return Err("chunk budget must be within 1..=850000 ms".into());
    }
    let sample_target_ns = config["sample_target_ns"].as_u64().unwrap_or(1_000_000);
    let inspect_gpu = config["device"].as_str().is_some_and(|v| v != "cpu");
    let observations: &'static [ScoreObservations; 2] = Box::leak(Box::new([
        ScoreObservations::new(first),
        ScoreObservations::new(
            map_first
                .as_ref()
                .and_then(|v| v.as_ref().ok())
                .copied()
                .unwrap_or(first),
        ),
    ]));
    let required_samples = if config["timing_phase"].as_str() == Some("warm") {
        10
    } else {
        config["required_samples"].as_u64().unwrap_or(10) as usize
    };
    if !(1..=10).contains(&required_samples) {
        return Err("required_samples must be within 1..=10".into());
    }
    let coverage_first = config["coverage_first"].as_bool().unwrap_or(false);
    let result = zenbench::run(|suite| {
        suite.group(name, |g| {
            g.config()
                .min_rounds(required_samples)
                .max_rounds(required_samples)
                .auto_rounds(false)
                .warmup_time(if coverage_first {
                    Duration::ZERO
                } else {
                    Duration::from_millis(100)
                })
                .sample_target_ns(sample_target_ns)
                .max_time(Duration::from_millis(chunk_budget_ms))
                .max_wall_time(Duration::from_millis(chunk_budget_ms));
            g.bench(primary_mode, move |b| {
                if inspect_gpu {
                    // Deferred Drop checks every returned score AFTER the clock.
                    b.iter_deferred_drop(|| ObservedScore {
                        value: score().expect("metric failed after warmup"),
                        observations: &observations[primary_arm],
                    });
                } else {
                    b.iter(|| black_box(score().expect("metric failed after warmup")));
                }
            });
            if let Some(mut call) = map_call {
                g.bench("score+map", move |b| {
                    if inspect_gpu {
                        b.iter_deferred_drop(|| ObservedScore {
                            value: call().expect("map failed after warmup"),
                            observations: &observations[1],
                        });
                    } else {
                        b.iter(|| black_box(call().expect("map failed after warmup")));
                    }
                });
            }
        });
    });
    let cmp = result.comparisons.first().ok_or("no comparison")?;
    for (index, bm) in cmp.benchmarks.iter().enumerate() {
        let mut row = row.clone();
        row["mode"] = json!(bm.name);
        let round_means: Vec<f64> = cmp
            .samples
            .iter()
            .map(|s| s.compensated_ns[index] as f64 / s.iterations as f64)
            .collect();
        let mut samples = round_means.clone();
        samples.sort_by(f64::total_cmp);
        if samples.is_empty() {
            return Err("zenbench completed no rounds".into());
        }
        let percentile = |p: f64| {
            let i = p * (samples.len() - 1) as f64;
            let lo = i.floor() as usize;
            let hi = i.ceil() as usize;
            samples[lo] + (samples[hi] - samples[lo]) * (i - lo as f64)
        };
        row["round_means_ns"] = json!(round_means);
        row["zenbench_arm_index"] = json!(index);
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
        let arm = usize::from(bm.name != "score");
        let measured = if inspect_gpu {
            f64::from_bits(
                observations[arm]
                    .last
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
        } else if only_map || arm == 0 {
            first
        } else {
            *map_first.as_ref().unwrap().as_ref().unwrap()
        };
        if inspect_gpu {
            let count = observations[arm]
                .count
                .load(std::sync::atomic::Ordering::Relaxed);
            row["gpu_call_bit_variation_count"] = json!(count);
            row["gpu_call_observation_scope"] = json!(
                "Every zenbench warmup/calibration/timed result; checks run in deferred output Drop after timing"
            );
            row["gpu_one_different_score_bits"] = if count > 0 {
                json!(format!(
                    "{:016x}",
                    observations[arm]
                        .different
                        .load(std::sync::atomic::Ordering::Relaxed)
                ))
            } else {
                Value::Null
            };
        }
        row["score"] = json!(measured);
        row["score_bits"] = json!(format!("{:016x}", measured.to_bits()));
        row["map_score_matches_scalar_bits"] = if only_map {
            Value::Null
        } else {
            json!(measured.to_bits() == first.to_bits())
        };
        if only_map {
            row["scalar_comparison_status"] = json!("Not repeated in isolated map process");
        }
        row["ref_score"] = json!(first);
        row["self_warmup_score"] = json!(first);
        row["abs_delta"] = json!((measured - first).abs());
        row["relative_delta"] =
            json!((measured - first).abs() / first.abs().max(f64::MIN_POSITIVE));
        row["status"] = json!(if measured.to_bits() == first.to_bits()
            && row["gpu_call_bit_variation_count"].as_u64().unwrap_or(0) == 0
        {
            "ok"
        } else {
            "variant_mismatch"
        });
        row["load_after"] = json!(load());
        row["notes"] = json!(
            "RGB8 resident; CLI compute call incl. metric colour conversion; initialization warmup excluded; zenbench round batch means, not individual-call percentiles"
        );
        if samples.len() < required_samples {
            row["status"] = json!("partial");
            row["partial_reason"] =
                json!("chunk time budget reached before required zenbench rounds");
            let prefix = if only_mode.is_some() {
                "IQA_ROW"
            } else {
                "IQA_PARTIAL"
            };
            println!("{prefix} {}", serde_json::to_string(&row)?);
        } else {
            println!("IQA_ROW {}", serde_json::to_string(&row)?);
        }
    }
    Ok(())
}
