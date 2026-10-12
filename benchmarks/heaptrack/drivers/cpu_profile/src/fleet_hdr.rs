//! Explicit native HDR-VDP absolute-nits workloads; no transfer conversion.
use super::{Error, Phase, input_path, require_bundle_inputs};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams, Scores, hdrvdp_cpu as hdrvdp};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    schema: u32,
    metric: String,
    width: u32,
    height: u32,
    stride_bytes: usize,
    reference: PathBuf,
    distorted: PathBuf,
    pixel_format: String,
    reuse: String,
    threads: usize,
    products: String,
    warmup_operations: usize,
    viewing: Viewing,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Viewing {
    model: String,
    pixels_per_degree: f64,
}

fn parameters(case: &Case) -> Result<(MetricKind, MetricParams), Error> {
    let ppd = case.viewing.pixels_per_degree;
    if !ppd.is_finite() || ppd < 4.0 {
        return Err("HDR viewing requires finite pixels_per_degree >= 4".into());
    }
    let (kind, params) = match (case.metric.as_str(), case.viewing.model.as_str()) {
        ("hdrvdp", "hdrvdp2-defaults-bt709") => (
            MetricKind::Hdrvdp,
            MetricParams::Hdrvdp(Box::new(hdrvdp::Params::new(ppd))),
        ),
        ("hdrvdp3", "hdrvdp3-quality-bt709-age24-nosurround-reference") => {
            use hdrvdp::v3;
            let params = v3::Params::new(
                v3::Task::Quality,
                v3::ViewingConditions::new(ppd, v3::Surround::None, 24),
                v3::InputEncoding::RgbBt709,
                v3::Emission::default_for(v3::InputEncoding::RgbBt709),
                v3::Options::reference(v3::Task::Quality),
            )?;
            (MetricKind::Hdrvdp3, MetricParams::Hdrvdp3(Box::new(params)))
        }
        _ => return Err("HDR metric and explicit viewing preset do not match".into()),
    };
    Ok((kind, params))
}

fn input(path: &Path, case: &Case) -> Result<Vec<f32>, Error> {
    let row = (case.width as usize)
        .checked_mul(12)
        .ok_or("row overflow")?;
    let length = case
        .stride_bytes
        .checked_mul(case.height as usize)
        .ok_or("input overflow")?;
    if case.stride_bytes < row || std::fs::metadata(path)?.len() != length as u64 {
        return Err("HDR raw input length/stride differs from declared layout".into());
    }
    let bytes = std::fs::read(path)?;
    let mut values = Vec::new();
    values.try_reserve_exact(
        (case.width as usize)
            .checked_mul(case.height as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or("packed size overflow")?,
    )?;
    for source in bytes.chunks_exact(case.stride_bytes) {
        for channel in source[..row].as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*channel);
            if !value.is_finite() || value < 0.0 {
                return Err("HDR channels require finite nonnegative absolute cd/m2".into());
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn score(case: &Case, reference: &[f32], distorted: &[f32]) -> Result<Scores, Error> {
    // Cold includes construction and destruction of all metric reference state.
    let (kind, params) = parameters(case)?;
    Ok(
        Metric::new(kind, Backend::Cpu, case.width, case.height, params)?
            .compute_pu_nits_interleaved_multi(reference, distorted)?,
    )
}

pub(super) fn run(case_path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let case: Case = serde_json::from_slice(bytes)?;
    if case.schema != 3
        || case.width == 0
        || case.height == 0
        || case.threads == 0
        || case.pixel_format != "rgb_f32le_bt709_linear_nits"
        || case.reuse != "cold"
        || !["primary", "all"].contains(&case.products.as_str())
        || case.warmup_operations > 1024
    {
        return Err("invalid native HDR case schema, format, reuse, products or budget".into());
    }
    // Validate parameters before allocating image buffers.
    drop(parameters(&case)?);
    let operations = std::env::var("ZENBENCH_OPERATIONS")?.parse::<usize>()?;
    if !(1..=1024).contains(&operations)
        || std::env::var("RAYON_NUM_THREADS")?.parse::<usize>()? != case.threads
    {
        return Err("HDR operation/thread budget mismatch".into());
    }
    let base = case_path.parent().ok_or("case has no parent")?;
    let reference_path = input_path(base, &case.reference)?;
    let distorted_path = input_path(base, &case.distorted)?;
    require_bundle_inputs(&[case_path, &reference_path, &distorted_path])?;
    let mut records = Vec::with_capacity(20);
    let preparation = Phase::begin()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(case.threads)
        .build_global()?;
    rayon::broadcast(|_| {});
    let reference = input(&reference_path, &case)?;
    let distorted = input(&distorted_path, &case)?;
    preparation.finish("input_and_pool_preparation", &mut records)?;
    if case.warmup_operations > 0 {
        let phase = Phase::begin()?;
        for _ in 0..case.warmup_operations {
            std::hint::black_box(score(&case, &reference, &distorted)?);
        }
        phase.finish("score_warmup", &mut records)?;
    }
    let mut products = Vec::with_capacity(operations);
    let mut primary = Vec::with_capacity(operations);
    let mut durations = Vec::with_capacity(operations);
    let phase = Phase::begin()?;
    for _ in 0..operations {
        let start = std::time::Instant::now();
        let scores = score(&case, &reference, &distorted)?;
        durations.push(start.elapsed().as_nanos());
        if scores.scores.is_empty() || scores.scores.iter().any(|s| !s.value.is_finite()) {
            return Err("native HDR metric returned empty or nonfinite scalar products".into());
        }
        primary.push(scores.primary().to_bits());
        if case.products == "all" {
            products.push(
                scores
                    .scores
                    .iter()
                    .map(|s| json!({"name":s.name,"bits":s.value.to_bits()}))
                    .collect::<Vec<_>>(),
            );
        }
    }
    phase.finish("score", &mut records)?;
    let correctness = if case.products == "all" {
        json!({"products":products})
    } else {
        json!({"score_bits":primary})
    };
    let result =
        json!({"schema":1,"operations":operations,"correctness":correctness,"memory":records});
    if let Some(path) = std::env::var_os("ZENBENCH_RESULT_FILE") {
        std::fs::write(path, serde_json::to_vec(&result)?)?;
    }
    #[cfg(feature = "heap-phases")]
    let lifetime = serde_json::to_value(zenbench::HeapSnapshot::get()?)?;
    #[cfg(not(feature = "heap-phases"))]
    let lifetime = serde_json::Value::Null;
    println!(
        "{}",
        json!({"driver":result,"operation_ns":durations,"lifetime":lifetime,
        "timing_usable":!cfg!(feature="heap-phases"),"workload":{"metric":case.metric,
        "width":case.width,"height":case.height,"source_stride_bytes":case.stride_bytes,
        "pixel_format":case.pixel_format,"reuse":"cold","backend":"native_cpu",
        "warmup_operations":case.warmup_operations,"products":case.products,
        "viewing_model":case.viewing.model,"pixels_per_degree":case.viewing.pixels_per_degree,
        "rayon_threads":rayon::current_num_threads(),"simd":"runtime dispatch; effective tier not observed",
        "timing_boundary":"native absolute-nits metric construction and scoring; input unpacking excluded"}})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case(metric: &str) -> Case {
        Case {
            schema: 3,
            metric: metric.into(),
            width: 64,
            height: 48,
            stride_bytes: 64 * 12,
            reference: "r".into(),
            distorted: "d".into(),
            pixel_format: "rgb_f32le_bt709_linear_nits".into(),
            reuse: "cold".into(),
            threads: 1,
            products: "all".into(),
            warmup_operations: 0,
            viewing: Viewing {
                model: if metric == "hdrvdp" {
                    "hdrvdp2-defaults-bt709"
                } else {
                    "hdrvdp3-quality-bt709-age24-nosurround-reference"
                }
                .into(),
                pixels_per_degree: 45.0,
            },
        }
    }

    #[test]
    fn native_hdr_routes_match_direct_absolute_nits_entries() {
        let reference: Vec<f32> = (0..64 * 48 * 3).map(|i| 50.0 + (i % 701) as f32).collect();
        let distorted: Vec<f32> = reference.iter().map(|v| v * 0.9).collect();
        let wide = |values: &[f32]| values.iter().map(|v| f64::from(*v)).collect::<Vec<_>>();
        for name in ["hdrvdp", "hdrvdp3"] {
            let case = case(name);
            let (_, params) = parameters(&case).unwrap();
            let direct = match params {
                MetricParams::Hdrvdp(params) => hdrvdp::score(
                    &wide(&distorted),
                    &wide(&reference),
                    64,
                    48,
                    hdrvdp::ColorEncoding::RgbBt709,
                    &params,
                )
                .unwrap(),
                MetricParams::Hdrvdp3(params) => {
                    hdrvdp::v3::hdrvdp3(&wide(&distorted), &wide(&reference), 64, 48, &params)
                        .unwrap()
                        .q_jod
                }
                _ => panic!("wrong parameter variant"),
            };
            assert_eq!(
                score(&case, &reference, &distorted)
                    .unwrap()
                    .primary()
                    .to_bits(),
                direct.to_bits()
            );
        }
    }

    #[test]
    fn padded_nits_preserve_channels_and_invalid_values_are_refused() {
        let scratch = PathBuf::from(std::env::var_os("HOME").unwrap()).join("tmp");
        std::fs::create_dir_all(&scratch).unwrap();
        let dir = tempfile::Builder::new()
            .prefix("hdr-driver-control-")
            .tempdir_in(scratch)
            .unwrap();
        let path = dir.path().join("nits");
        let mut case = case("hdrvdp");
        case.width = 2;
        case.height = 2;
        case.stride_bytes = 27;
        let mut bytes = Vec::new();
        for row in 0..2 {
            for i in 0..6 {
                bytes.extend_from_slice(&((row * 6 + i) as f32 + 0.5).to_le_bytes());
            }
            bytes.extend_from_slice(&[255, 255, 255]);
        }
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            input(&path, &case).unwrap(),
            (0..12).map(|i| i as f32 + 0.5).collect::<Vec<_>>()
        );
        for invalid in [f32::NAN, f32::INFINITY, -1.0] {
            bytes[..4].copy_from_slice(&invalid.to_le_bytes());
            std::fs::write(&path, &bytes).unwrap();
            assert!(input(&path, &case).is_err());
        }
        case.viewing.pixels_per_degree = 0.0;
        assert!(parameters(&case).is_err());
        case.viewing.pixels_per_degree = 45.0;
        case.viewing.model = "unspecified".into();
        assert!(parameters(&case).is_err());
    }
}
