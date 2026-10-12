//! Explicit device backend, full-frame memory and GPU instance lifetime.
use super::{Error, Phase, input_path, read_rgb8, require_bundle_inputs};
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
use zenmetrics_api::{Backend, MemoryMode, Metric, MetricKind, MetricParams, Scores};

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
    backend: String,
    memory_mode: String,
    parameter_preset: String,
    expected_adapter: String,
}

fn parameters(case: &Case) -> Result<(MetricKind, Backend, MetricParams), Error> {
    let backend = match case.backend.as_str() {
        #[cfg(feature = "study-gpu-cuda")]
        "cuda" => Backend::Cuda,
        #[cfg(feature = "study-gpu-wgpu")]
        "wgpu" => Backend::Wgpu,
        _ => return Err("requested explicit GPU backend is not compiled in this driver".into()),
    };
    let kind = match case.metric.as_str() {
        "butteraugli-gpu" => MetricKind::Butter,
        "cvvdp-gpu" => MetricKind::Cvvdp,
        "dssim-gpu" => MetricKind::Dssim,
        "iwssim-gpu" => MetricKind::Iwssim,
        "ssim2-gpu" => MetricKind::Ssim2,
        "zensim-gpu" => MetricKind::Zensim,
        _ => return Err("schema 4 requires one of the six GPU metric variants".into()),
    };
    if case.memory_mode != "full" || case.parameter_preset != "api-defaults-standard-fhd" {
        return Err(
            "GPU workload requires declared full memory and api-defaults-standard-fhd preset"
                .into(),
        );
    }
    let params = if kind == MetricKind::Cvvdp {
        MetricParams::cvvdp(super::super::cvvdp_display().clone())
    } else {
        MetricParams::try_default_for(kind)?
    };
    Ok((kind, backend, params))
}

fn verify_adapter(name: &str, kind: &str, expected: &str) -> Result<(), Error> {
    if !["DiscreteGpu", "IntegratedGpu", "CudaGpu"].contains(&kind) {
        return Err(
            format!("GPU benchmark refuses non-hardware adapter type {kind}: {name}").into(),
        );
    }
    if expected.is_empty() || name != expected {
        return Err(
            format!("GPU adapter mismatch: expected {expected:?}, selected {name:?}").into(),
        );
    }
    Ok(())
}

fn device_identity(case: &Case) -> Result<serde_json::Value, Error> {
    match case.backend.as_str() {
        #[cfg(feature = "study-gpu-wgpu")]
        "wgpu" => {
            // Register the inspected setup under the SAME default-device key that
            // the metric constructors use. Enumerating a separate adapter is not proof.
            let setup = cubecl::wgpu::init_setup::<cubecl::wgpu::AutoGraphicsApi>(
                &Default::default(),
                Default::default(),
            );
            let info = setup.adapter.get_info();
            let kind = format!("{:?}", info.device_type);
            verify_adapter(&info.name, &kind, &case.expected_adapter)?;
            Ok(json!({"name":info.name,"type":kind,"vendor":info.vendor,
                "device":info.device,"graphics_backend":format!("{:?}",info.backend),
                "driver":info.driver,"driver_info":info.driver_info}))
        }
        #[cfg(feature = "study-gpu-cuda")]
        "cuda" => {
            // CubeCL's metric path uses CudaDevice::default(), CUDA ordinal 0.
            // Query the same driver ordinal, including CUDA visibility remapping.
            let context = cudarc::driver::CudaContext::new(0)?;
            let name = context.name()?;
            verify_adapter(&name, "CudaGpu", &case.expected_adapter)?;
            Ok(json!({"name":name,"type":"CudaGpu","cuda_ordinal":context.ordinal()}))
        }
        _ => Err("explicit GPU backend not compiled".into()),
    }
}

fn scorer(case: &Case) -> Result<Metric, Error> {
    let (kind, backend, params) = parameters(case)?;
    // Explicit Cuda/Wgpu never invokes Auto or the orchestrator CPU ladder.
    Ok(Metric::new_with_memory_mode(
        kind,
        backend,
        case.width,
        case.height,
        params,
        MemoryMode::Full,
    )?)
}

pub(super) fn run(case_path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let case: Case = serde_json::from_slice(bytes)?;
    if case.schema != 4
        || case.width == 0
        || case.height == 0
        || case.threads == 0
        || case.pixel_format != "rgb8_srgb"
        || !["cold", "warm_instance"].contains(&case.reuse.as_str())
        || !["primary", "all"].contains(&case.products.as_str())
        || case.warmup_operations > 1024
    {
        return Err("invalid GPU workload schema, format, reuse, products or budget".into());
    }
    drop(parameters(&case)?);
    let operations = std::env::var("ZENBENCH_OPERATIONS")?.parse::<usize>()?;
    if !(1..=1024).contains(&operations)
        || std::env::var("RAYON_NUM_THREADS")?.parse::<usize>()? != case.threads
    {
        return Err("GPU operation/thread budget mismatch".into());
    }
    let base = case_path.parent().ok_or("case has no parent")?;
    let reference_path = input_path(base, &case.reference)?;
    let distorted_path = input_path(base, &case.distorted)?;
    require_bundle_inputs(&[case_path, &reference_path, &distorted_path])?;
    let mut records = Vec::with_capacity(24);
    let phase = Phase::begin()?;
    let device = device_identity(&case)?;
    phase.finish("device_identity_and_context_preparation", &mut records)?;
    let phase = Phase::begin()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(case.threads)
        .build_global()?;
    rayon::broadcast(|_| {});
    let reference = read_rgb8(&reference_path, case.width, case.height, case.stride_bytes)?;
    let distorted = read_rgb8(&distorted_path, case.width, case.height, case.stride_bytes)?;
    phase.finish("input_and_pool_preparation", &mut records)?;
    let mut retained = if case.reuse == "warm_instance" {
        let phase = Phase::begin()?;
        let value = scorer(&case)?;
        phase.finish("instance_preparation", &mut records)?;
        Some(value)
    } else {
        None
    };
    let compute = |retained: &mut Option<Metric>| -> Result<Scores, Error> {
        match retained {
            Some(value) => Ok(value.compute_srgb_u8_multi(&reference, &distorted)?),
            None => Ok(scorer(&case)?.compute_srgb_u8_multi(&reference, &distorted)?),
        }
    };
    if case.warmup_operations > 0 {
        let phase = Phase::begin()?;
        for _ in 0..case.warmup_operations {
            std::hint::black_box(compute(&mut retained)?);
        }
        phase.finish("score_warmup", &mut records)?;
    }
    let mut primary = Vec::with_capacity(operations);
    let mut products = Vec::with_capacity(operations);
    let mut durations = Vec::with_capacity(operations);
    let mut implementation = None;
    let phase = Phase::begin()?;
    for _ in 0..operations {
        let start = std::time::Instant::now();
        // Multi-score completes reduction/readback before returning.
        let scores = compute(&mut retained)?;
        durations.push(start.elapsed().as_nanos());
        if scores.scores.is_empty() || scores.scores.iter().any(|s| !s.value.is_finite()) {
            return Err("GPU metric returned empty or nonfinite scalar products".into());
        }
        implementation = Some(scores.metric_version);
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
    let phase = Phase::begin()?;
    drop(retained);
    phase.finish("instance_release", &mut records)?;
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
        "pixel_format":case.pixel_format,"reuse":case.reuse,"backend":case.backend,
        "device":device,"memory_mode":case.memory_mode,"parameter_preset":case.parameter_preset,
        "implementation":"zenmetrics_api explicit GPU scalar path","metric_version":implementation,
        "products":case.products,"warmup_operations":case.warmup_operations,
        "rayon_threads":rayon::current_num_threads(),"device_memory":"not observed by this driver",
        "device_context":"process-wide context and JIT caches can survive cold instance destruction",
        "timing_boundary":"input upload through reduction/readback; cold includes constructor; warm_instance retains buffers, not cached reference"}})
    );
    super::emit_pair_timings(&durations);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn case() -> Case {
        Case {
            schema: 4,
            metric: "ssim2-gpu".into(),
            width: 256,
            height: 256,
            stride_bytes: 256 * 3,
            reference: "r".into(),
            distorted: "d".into(),
            pixel_format: "rgb8_srgb".into(),
            reuse: "cold".into(),
            threads: 1,
            products: "primary".into(),
            warmup_operations: 0,
            backend: if cfg!(feature = "study-gpu-wgpu") {
                "wgpu"
            } else {
                "cuda"
            }
            .into(),
            memory_mode: "full".into(),
            parameter_preset: "api-defaults-standard-fhd".into(),
            expected_adapter: "test GPU".into(),
        }
    }
    #[test]
    fn hardware_identity_refuses_software_unknown_and_wrong_device() {
        for kind in ["Cpu", "Other", "VirtualGpu", "unknown"] {
            assert!(verify_adapter("test GPU", kind, "test GPU").is_err());
        }
        for kind in ["IntegratedGpu", "DiscreteGpu", "CudaGpu"] {
            assert!(verify_adapter("test GPU", kind, "test GPU").is_ok());
            assert!(verify_adapter("test GPU", kind, "different GPU").is_err());
            assert!(verify_adapter("test GPU", kind, "").is_err());
        }
    }
    #[test]
    fn explicit_gpu_parameters_refuse_auto_cpu_and_implicit_memory() {
        let mut case = case();
        assert!(parameters(&case).is_ok());
        for backend in ["auto", "cpu", "cubecl_cpu", "unknown"] {
            case.backend = backend.into();
            assert!(parameters(&case).is_err());
        }
        case = self::case();
        case.memory_mode = "auto".into();
        assert!(parameters(&case).is_err());
        case = self::case();
        case.metric = "ssim2".into();
        assert!(parameters(&case).is_err());
    }
}
