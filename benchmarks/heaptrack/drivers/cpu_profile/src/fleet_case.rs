//! One explicit SDR CPU case. Kept private to the existing metric driver.
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use zenmetrics_api::{Backend, MetricKind, Warm};
use zenpixels::{PixelDescriptor, PixelSlice};

type Error = Box<dyn std::error::Error>;
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
    #[serde(default)]
    products: Option<String>,
    #[serde(default)]
    warmup_operations: Option<usize>,
}

struct Phase {
    #[cfg(feature = "heap-phases")]
    phase: zenbench::HeapPhase,
}
impl Phase {
    fn begin() -> Result<Self, Error> {
        Ok(Self {
            #[cfg(feature = "heap-phases")]
            phase: zenbench::HeapPhase::begin()?,
        })
    }
    fn finish(self, scope: &str, records: &mut Vec<Value>) -> Result<(), Error> {
        #[cfg(feature = "heap-phases")]
        {
            let m = self.phase.finish()?;
            for (quantity, bytes) in [
                ("requested_live_heap_entry", m.entry_live_bytes),
                ("requested_live_heap_peak", m.peak_live_bytes),
                ("requested_live_heap_exit", m.exit_live_bytes),
                ("allocation_traffic", m.requested_bytes),
            ] {
                records.push(json!({"quantity":quantity,"bytes":bytes,"unavailable_reason":null,
                    "backend":"zenbench_HeapProfiler","backend_version":"77707b68eae6e5245d7f87a78e04a9453bcde4e5",
                    "coverage":"all Rust global-allocator threads; excludes native allocators, child processes, mappings and devices",
                    "scope":scope}));
            }
        }
        #[cfg(not(feature = "heap-phases"))]
        let _ = (scope, records);
        Ok(())
    }
}

// The existing CLI metric dispatch takes packed RGB8. Preserve every source
// row, including padded inputs; packing is charged to input preparation.
fn input(path: &Path, case: &Case) -> Result<Vec<u8>, Error> {
    let row = (case.width as usize).checked_mul(3).ok_or("row overflow")?;
    let length = case
        .stride_bytes
        .checked_mul(case.height as usize)
        .ok_or("input overflow")?;
    if case.stride_bytes < row || std::fs::metadata(path)?.len() != length as u64 {
        return Err("raw input length/stride differs from declared layout".into());
    }
    let bytes = std::fs::read(path)?;
    if row == case.stride_bytes {
        return Ok(bytes);
    }
    let mut packed = Vec::new();
    packed.try_reserve_exact(
        row.checked_mul(case.height as usize)
            .ok_or("packed size overflow")?,
    )?;
    for source in bytes.chunks_exact(case.stride_bytes) {
        packed.extend_from_slice(&source[..row]);
    }
    Ok(packed)
}
fn pixels<'a>(bytes: &'a [u8], case: &Case) -> Result<PixelSlice<'a>, Error> {
    Ok(PixelSlice::new(
        bytes,
        case.width,
        case.height,
        case.width as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )?)
}
fn warm_kind(name: &str) -> Result<MetricKind, Error> {
    match name {
        "ssim2" => Ok(MetricKind::Ssim2),
        "butteraugli" => Ok(MetricKind::Butter),
        "dssim" => Ok(MetricKind::Dssim),
        "zensim" => Ok(MetricKind::Zensim),
        _ => Err("this driver exposes warm-reference cases for ssim2, butteraugli, dssim and zensim; choose cold for other metrics".into()),
    }
}

fn input_path(base: &Path, relative: &Path) -> Result<PathBuf, Error> {
    let name = relative.to_str().ok_or("input path must be UTF-8")?;
    if name.is_empty()
        || name.contains(['\\', ':'])
        || !relative
            .components()
            .all(|p| matches!(p, std::path::Component::Normal(_)))
    {
        return Err("input path must be relative without parent components".into());
    }
    let path = base.join(relative).canonicalize()?;
    if !path.starts_with(base) || !path.is_file() {
        return Err("input path escapes the case directory or is not a file".into());
    }
    Ok(path)
}

fn require_bundle_inputs(paths: &[&Path]) -> Result<(), Error> {
    if std::env::var_os("ZENBENCH_RESULT_FILE").is_none() {
        return Ok(());
    }
    // fleet-run verifies these manifest entries before and after execution.
    // Refuse unlisted inputs even when another file happens to exist nearby.
    let root = std::env::current_dir()?.canonicalize()?;
    let manifest = root.join("bundle.json");
    if std::fs::metadata(&manifest)?.len() > 8 * 1024 * 1024 {
        return Err("bundle manifest exceeds metadata limit".into());
    }
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let files = manifest["files"]
        .as_object()
        .ok_or("missing bundle files")?;
    for path in paths {
        let relative = path
            .strip_prefix(&root)?
            .to_str()
            .ok_or("non-UTF-8 bundle path")?
            .replace('\\', "/");
        if !files.contains_key(&relative) {
            return Err(format!("IQA input is not bound by the bundle: {relative}").into());
        }
    }
    Ok(())
}

pub(super) fn run(args: &[String]) -> Result<(), Error> {
    if args.len() != 3 {
        return Err("usage: new-metrics-wall fleet CASE.json".into());
    }
    let case_path = Path::new(&args[2]).canonicalize()?;
    let case: Case = serde_json::from_slice(&std::fs::read(&case_path)?)?;
    if ![1, 2].contains(&case.schema)
        || case.width == 0
        || case.height == 0
        || case.threads == 0
        || case.pixel_format != "rgb8_srgb"
        || !["cold", "warm_reference"].contains(&case.reuse.as_str())
        || !super::METRICS.iter().any(|(name, _)| *name == case.metric)
    {
        return Err("invalid SDR CPU case schema, metric, layout, threads or reuse mode".into());
    }
    let all_products = match (case.schema, case.products.as_deref(), case.warmup_operations) {
        (1, None, None) => false,
        (2, Some("primary"), Some(n)) if n <= 1024 => false,
        (2, Some("all"), Some(n)) if n <= 1024 && case.reuse == "cold" => true,
        _ => return Err("schema 2 needs products=primary|all and warmup_operations=0..1024; all products require cold scoring; schema 1 has neither field".into()),
    };
    let operations = std::env::var("ZENBENCH_OPERATIONS")?.parse::<usize>()?;
    if !(1..=1024).contains(&operations) {
        return Err("operations must be 1..1024".into());
    }
    if std::env::var("RAYON_NUM_THREADS")?.parse::<usize>()? != case.threads {
        return Err("declared threads differ from RAYON_NUM_THREADS".into());
    }
    let base = case_path.parent().ok_or("case has no parent")?;
    let reference = input_path(base, &case.reference)?;
    let distorted = input_path(base, &case.distorted)?;
    require_bundle_inputs(&[&case_path, &reference, &distorted])?;
    let mut records = Vec::with_capacity(20);
    let mut scores = Vec::with_capacity(operations);
    let mut products = Vec::with_capacity(operations);
    let mut durations = Vec::with_capacity(operations);
    let preparation = Phase::begin()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(case.threads)
        .build_global()?;
    rayon::broadcast(|_| {});
    let r = super::mk_image(input(&reference, &case)?, case.width, case.height);
    let d = super::mk_image(input(&distorted, &case)?, case.width, case.height);
    let _ = super::cvvdp_display();
    preparation.finish("input_and_pool_preparation", &mut records)?;

    let mut warm: Option<Warm> = None;
    if case.reuse == "warm_reference" {
        let phase = Phase::begin()?;
        warm = Some(zenmetrics_api::warm_reference(
            warm_kind(&case.metric)?,
            Backend::Cpu,
            pixels(&r.pixels, &case)?,
        )?);
        phase.finish("reference_preparation", &mut records)?;
        // Explicit score warmup; retain the reference across measured calls.
        let phase = Phase::begin()?;
        for _ in 0..case.warmup_operations.unwrap_or(1) {
            std::hint::black_box(warm.as_mut().unwrap().score(pixels(&d.pixels, &case)?)?);
        }
        phase.finish("score_warmup", &mut records)?;
    }
    let cold_kind = super::METRICS
        .iter()
        .find(|(name, _)| *name == case.metric)
        .unwrap()
        .1;
    if case.reuse == "cold" && case.warmup_operations.unwrap_or(0) > 0 {
        let phase = Phase::begin()?;
        for _ in 0..case.warmup_operations.unwrap() {
            std::hint::black_box(zenmetrics_cli::metrics::run_metric_display(
                cold_kind,
                &r,
                &d,
                zenmetrics_cli::metrics::GpuRuntime::Cpu,
                Some(super::cvvdp_display()),
            )?);
        }
        phase.finish("score_warmup", &mut records)?;
    }
    let phase = Phase::begin()?;
    for _ in 0..operations {
        let start = std::time::Instant::now();
        let columns = match warm.as_mut() {
            Some(warm) => {
                let score = warm.score(pixels(&d.pixels, &case)?)?;
                vec![(score.metric_name, score.value)]
            }
            None => zenmetrics_cli::metrics::run_metric_display(
                cold_kind,
                &r,
                &d,
                zenmetrics_cli::metrics::GpuRuntime::Cpu,
                Some(super::cvvdp_display()),
            )?,
        };
        let elapsed = start.elapsed().as_nanos();
        let value = columns.first().ok_or("metric returned no products")?.1;
        if case.schema == 1 && !value.is_finite() {
            return Err("metric produced a non-finite score".into());
        }
        scores.push(value.to_bits());
        if all_products {
            products.push(
                columns
                    .iter()
                    .map(|(name, value)| json!({"name":name,"bits":value.to_bits()}))
                    .collect::<Vec<_>>(),
            );
        }
        durations.push(elapsed);
    }
    phase.finish("score", &mut records)?;
    let phase = Phase::begin()?;
    drop(warm);
    phase.finish("reference_release", &mut records)?;
    let correctness = if all_products {
        json!({"products":products})
    } else {
        json!({"score_bits":scores})
    };
    let result =
        json!({"schema":1,"operations":operations,"correctness":correctness,"memory":records});
    if let Some(path) = std::env::var_os("ZENBENCH_RESULT_FILE") {
        std::fs::write(path, serde_json::to_vec(&result)?)?;
    }
    #[cfg(feature = "heap-phases")]
    let lifetime = serde_json::to_value(zenbench::HeapSnapshot::get()?)?;
    #[cfg(not(feature = "heap-phases"))]
    let lifetime = Value::Null;
    println!(
        "{}",
        json!({"driver":result,"operation_ns":durations,"lifetime":lifetime,
        "timing_usable":!cfg!(feature="heap-phases"),"workload":{"metric":case.metric,
        "width":case.width,"height":case.height,"source_stride_bytes":case.stride_bytes,
        "dispatch_layout":"packed RGB8 after preparation","pixel_format":case.pixel_format,
        "reuse":case.reuse,"warmup_operations":case.warmup_operations.unwrap_or(if case.reuse=="warm_reference" {1} else {0}),
        "products":if all_products {"all CLI scalar columns"} else {"primary score"},
        "rayon_threads":rayon::current_num_threads(),
        "simd":"metric runtime dispatch; effective tier not observed",
        "display":"standard_fhd for cvvdp","output":"scalar score products; no retained diffmap"}})
    );
    Ok(())
}
