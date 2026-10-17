use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};
use std::collections::HashMap;
use std::path::Path;

use zenmetrics_api::alpha::{composite_rgba_srgb8, compute_rgba_alpha_stats, AlphaPolicy};
use zenmetrics_cli::decode::{decode_image_to_rgb8, decode_image_with_alpha_policy, Rgb8Image};
use zenmetrics_cli::metrics::display::parse_display;
use zenmetrics_cli::metrics::{run_metric_display, GpuRuntime, LumaIngress, MetricKind};

#[pyclass(name = "ScoreResult")]
#[derive(Clone, Debug)]
pub struct PyScoreResult {
    #[pyo3(get)]
    pub metric: String,
    #[pyo3(get)]
    pub value: f64,
    #[pyo3(get)]
    pub scores: HashMap<String, f64>,
}

#[pymethods]
impl PyScoreResult {
    fn __repr__(&self) -> String {
        format!(
            "ScoreResult(metric='{}', value={:.6}, scores={:?})",
            self.metric, self.value, self.scores
        )
    }

    fn __float__(&self) -> f64 {
        self.value
    }

    fn get(&self, col: &str) -> Option<f64> {
        self.scores.get(col).copied()
    }
}

#[pyclass(name = "PanelStats")]
#[derive(Clone, Copy, Debug)]
pub struct PyPanelStats {
    #[pyo3(get)]
    pub srocc: f64,
    #[pyo3(get)]
    pub plcc: f64,
    #[pyo3(get)]
    pub krocc: f64,
    #[pyo3(get)]
    pub outlier_ratio: f64,
    #[pyo3(get)]
    pub pwrc_auc: f64,
    #[pyo3(get)]
    pub z_rmse: f64,
    #[pyo3(get)]
    pub n: usize,
}

#[pymethods]
impl PyPanelStats {
    fn __repr__(&self) -> String {
        format!(
            "PanelStats(srocc={:.4}, plcc={:.4}, krocc={:.4}, outlier_ratio={:.4}, pwrc={:.4}, z_rmse={:.4}, n={})",
            self.srocc, self.plcc, self.krocc, self.outlier_ratio, self.pwrc_auc, self.z_rmse, self.n
        )
    }

    fn to_dict(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("srocc".to_string(), self.srocc);
        m.insert("plcc".to_string(), self.plcc);
        m.insert("krocc".to_string(), self.krocc);
        m.insert("outlier_ratio".to_string(), self.outlier_ratio);
        m.insert("pwrc_auc".to_string(), self.pwrc_auc);
        m.insert("z_rmse".to_string(), self.z_rmse);
        m.insert("n".to_string(), self.n as f64);
        m
    }
}

fn parse_metric_kind(s: &str) -> PyResult<MetricKind> {
    let lower = s.to_ascii_lowercase().replace('_', "-");
    let kind = match lower.as_str() {
        "ssim2" | "ssim-2" | "ssimulacra2" | "ssimulacra-2" => MetricKind::Ssim2,
        "ssim2-gpu" | "ssim-2-gpu" | "ssimulacra2-gpu" | "ssimulacra-2-gpu" => MetricKind::Ssim2Gpu,
        "butteraugli" | "butter" => MetricKind::Butteraugli,
        "butteraugli-gpu" | "butter-gpu" => MetricKind::ButteraugliGpu,
        "dssim" => MetricKind::Dssim,
        "dssim-gpu" => MetricKind::DssimGpu,
        "ssim" => MetricKind::Ssim,
        "psnr" => MetricKind::Psnr,
        "psnr-y" => MetricKind::PsnrY,
        "psnr-y601" => MetricKind::PsnrY601,
        "psnr-y-studio601" => MetricKind::PsnrYStudio601,
        "psnr-y-libvmaf" => MetricKind::PsnrYLibvmaf,
        "cvvdp" => MetricKind::Cvvdp,
        "cvvdp-gpu" => MetricKind::CvvdpGpu,
        "iwssim" | "iw-ssim" => MetricKind::Iwssim,
        "iwssim-gpu" | "iw-ssim-gpu" => MetricKind::IwssimGpu,
        "iwssim-piq" | "iw-ssim-piq" => MetricKind::IwssimPiq,
        "gmsd" => MetricKind::Gmsd,
        "mdsi" => MetricKind::Mdsi,
        "ms-gmsd" | "msgmsd" => MetricKind::MsGmsd,
        "ms-gmsdc" | "msgmsdc" => MetricKind::MsGmsdc,
        "hdrvdp" => MetricKind::Hdrvdp,
        "hdrvdp3" => MetricKind::Hdrvdp3,
        "fsim" => MetricKind::Fsim,
        "fsim-y" => MetricKind::FsimY,
        "haarpsi" => MetricKind::Haarpsi,
        "haarpsi-y" => MetricKind::HaarpsiY,
        "vif" => MetricKind::Vif,
        "vifvec" => MetricKind::VifVec,
        "vmaf" => MetricKind::Vmaf,
        "vmaf-neg" => MetricKind::VmafNeg,
        "vmaf-4k" => MetricKind::Vmaf4k,
        "vmaf-v1" => MetricKind::VmafV1,
        "vsi" => MetricKind::Vsi,
        "msssim" | "ms-ssim" => MetricKind::Msssim,
        "msssim-libvmaf" | "ms-ssim-libvmaf" => MetricKind::MsssimLibvmaf,
        "nlpd" => MetricKind::Nlpd,
        "nlpd-iqa" => MetricKind::NlpdIqa,
        "psnrhvs" => MetricKind::Psnrhvs,
        "psnrhvs-y" => MetricKind::PsnrhvsY,
        "psnrhvs-daala" => MetricKind::PsnrhvsDaala,
        "mad" => MetricKind::Mad,
        "mdctpsnr" => MetricKind::Mdctpsnr,
        "zensim" => MetricKind::Zensim,
        "zensim-gpu" => MetricKind::ZensimGpu,
        "ssim-libvmaf" => MetricKind::SsimLibvmaf,
        "lpips" => {
            return Err(PyValueError::new_err(
                "Metric 'lpips' (Learned Perceptual Image Patch Similarity) is not built-in. Zenmetrics provides 'zensim' as its in-tree learned metric, alongside calibrated psychovisual metrics ('ssim2', 'butteraugli') and classical metrics ('psnr', 'ssim')."
            ))
        }
        unknown => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{unknown}'. Run zm.list_metrics() to see available metrics (e.g. ssim2, butteraugli, psnr, ms-ssim, cvvdp, zensim)."
            )))
        }
    };
    Ok(kind)
}

fn all_metric_names() -> Vec<&'static str> {
    vec![
        "ssim2", "ssim2-gpu", "butteraugli", "butteraugli-gpu", "dssim", "dssim-gpu",
        "ssim", "psnr", "psnr-y", "psnr-y601", "psnr-y-studio601", "psnr-y-libvmaf",
        "cvvdp", "cvvdp-gpu", "iwssim", "iwssim-gpu", "iwssim-piq", "gmsd", "mdsi",
        "ms-gmsd", "ms-gmsdc", "hdrvdp", "hdrvdp3", "fsim", "fsim-y", "haarpsi",
        "haarpsi-y", "vif", "vifvec", "vmaf", "vmaf-neg", "vmaf-4k", "vmaf-v1",
        "vsi", "msssim", "msssim-libvmaf", "nlpd", "nlpd-iqa", "psnrhvs", "psnrhvs-y",
        "psnrhvs-daala", "mad", "mdctpsnr", "zensim", "zensim-gpu", "ssim-libvmaf"
    ]
}

fn parse_gpu_runtime(name: Option<&str>) -> PyResult<GpuRuntime> {
    match name {
        Some(s) => {
            let lower = s.to_ascii_lowercase();
            if lower == "auto" {
                Ok(GpuRuntime::Auto)
            } else if lower.starts_with("cuda") {
                Ok(GpuRuntime::Cuda)
            } else if lower == "wgpu" || lower.starts_with("vulkan") {
                Ok(GpuRuntime::Wgpu)
            } else if lower.starts_with("hip") || lower.starts_with("rocm") {
                Ok(GpuRuntime::Hip)
            } else if lower.starts_with("cpu") {
                Ok(GpuRuntime::Cpu)
            } else {
                Err(PyValueError::new_err(format!("unknown gpu_runtime/device '{s}'")))
            }
        }
        None => Ok(GpuRuntime::Auto),
    }
}

fn parse_luma_ingress(name: Option<&str>) -> PyResult<LumaIngress> {
    match name {
        Some(s) => match s.to_ascii_lowercase().as_str() {
            "house" => Ok(LumaIngress::House),
            "yuv601-studio" | "yuv601_studio" => Ok(LumaIngress::Yuv601Studio),
            other => Err(PyValueError::new_err(format!("unknown luma_ingress '{other}'"))),
        },
        None => Ok(LumaIngress::House),
    }
}

fn parse_alpha_policy(name: Option<&str>, bg: Option<(u8, u8, u8)>) -> PyResult<AlphaPolicy> {
    match name {
        Some(s) => {
            let clean = s.trim().to_ascii_lowercase().replace(['-', ' '], "_");
            if clean == "composite" || clean == "bg" {
                let (r, g, b) = bg.unwrap_or((255, 255, 255));
                return Ok(AlphaPolicy::CompositeOnto { r, g, b });
            }
            AlphaPolicy::from_str_loose(s).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "Unknown alpha_policy '{s}'. Supported policies: 'opaque', 'noise', 'matte_white', 'matte_black', 'composite', 'min_wb', 'discard_unchecked'."
                ))
            })
        }
        None => Ok(AlphaPolicy::RequireOpaque),
    }
}

fn execute_metric(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
    display_model: Option<&str>,
    luma_ingress: LumaIngress,
    gpu_runtime: GpuRuntime,
) -> PyResult<PyScoreResult> {
    if reference.width != distorted.width || reference.height != distorted.height {
        return Err(PyValueError::new_err(format!(
            "dimension mismatch: reference is {}x{}, distorted is {}x{}",
            reference.width, reference.height, distorted.width, distorted.height
        )));
    }

    // CVVDP requires an explicit display calibration (viewing distance, screen luminance, resolution).
    // Silently defaulting this causes reproducibility and science discrepancies.
    if (kind == MetricKind::Cvvdp || kind == MetricKind::CvvdpGpu) && display_model.is_none() {
        return Err(PyValueError::new_err(
            "metric 'cvvdp' requires an explicit display model (e.g. display_model='standard_4k', 'standard_fhd', or custom geometry) because viewing geometry and display luminance fundamentally determine CVVDP contrast sensitivity."
        ));
    }

    let display = match display_model {
        Some(d) => Some(parse_display(d).map_err(|e| PyValueError::new_err(e))?),
        None => None,
    };

    let (r_ing, d_ing) =
        zenmetrics_cli::metrics::luma_ingress_pair(kind, luma_ingress, reference, distorted);
    let rows = run_metric_display(
        kind,
        r_ing.as_ref().unwrap_or(reference),
        d_ing.as_ref().unwrap_or(distorted),
        gpu_runtime,
        display.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(format!("metric execution failed: {e}")))?;

    if rows.is_empty() {
        return Err(PyValueError::new_err("metric produced no scores"));
    }

    let primary_val = rows[0].1;
    let mut scores_map = HashMap::with_capacity(rows.len());
    for (col, val) in rows {
        scores_map.insert(col.to_string(), val);
    }

    Ok(PyScoreResult {
        metric: kind.name().to_string(),
        value: primary_val,
        scores: scores_map,
    })
}

#[pyfunction]
#[pyo3(signature = (ref_path, dist_path, metric="ssim2", alpha_policy=None, alpha_bg=None, display_model=None, luma_ingress=None, gpu_runtime=None))]
pub fn score_files(
    ref_path: &str,
    dist_path: &str,
    metric: &str,
    alpha_policy: Option<&str>,
    alpha_bg: Option<(u8, u8, u8)>,
    display_model: Option<&str>,
    luma_ingress: Option<&str>,
    gpu_runtime: Option<&str>,
) -> PyResult<PyScoreResult> {
    let kind = parse_metric_kind(metric)?;
    let gpu_rt = parse_gpu_runtime(gpu_runtime)?;
    let l_ingress = parse_luma_ingress(luma_ingress)?;
    let policy = parse_alpha_policy(alpha_policy, alpha_bg)?;

    let reference = decode_image_with_alpha_policy(Path::new(ref_path), policy).map_err(|e| {
        PyIOError::new_err(format!(
            "failed to decode reference image '{ref_path}': {e}"
        ))
    })?;
    let distorted = decode_image_with_alpha_policy(Path::new(dist_path), policy).map_err(|e| {
        PyIOError::new_err(format!(
            "failed to decode distorted image '{dist_path}': {e}"
        ))
    })?;

    execute_metric(
        kind,
        &reference,
        &distorted,
        display_model,
        l_ingress,
        gpu_rt,
    )
}

#[pyfunction]
#[pyo3(signature = (ref_bytes, dist_bytes, width, height, metric="ssim2", display_model=None, luma_ingress=None, gpu_runtime=None))]
pub fn score_rgb8(
    ref_bytes: &[u8],
    dist_bytes: &[u8],
    width: u32,
    height: u32,
    metric: &str,
    display_model: Option<&str>,
    luma_ingress: Option<&str>,
    gpu_runtime: Option<&str>,
) -> PyResult<PyScoreResult> {
    let expected = (width as usize) * (height as usize) * 3;
    if ref_bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "reference buffer length {} does not match expected {} bytes for {}x{} RGB",
            ref_bytes.len(),
            expected,
            width,
            height
        )));
    }
    if dist_bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "distorted buffer length {} does not match expected {} bytes for {}x{} RGB",
            dist_bytes.len(),
            expected,
            width,
            height
        )));
    }

    let kind = parse_metric_kind(metric)?;
    let gpu_rt = parse_gpu_runtime(gpu_runtime)?;
    let l_ingress = parse_luma_ingress(luma_ingress)?;

    let reference = Rgb8Image {
        pixels: ref_bytes.to_vec(),
        width,
        height,
    };
    let distorted = Rgb8Image {
        pixels: dist_bytes.to_vec(),
        width,
        height,
    };

    execute_metric(
        kind,
        &reference,
        &distorted,
        display_model,
        l_ingress,
        gpu_rt,
    )
}

#[pyfunction]
#[pyo3(signature = (ref_bytes, dist_bytes, width, height, metric="ssim2", alpha_policy=None, alpha_bg=None, display_model=None, luma_ingress=None, gpu_runtime=None))]
pub fn score_rgba(
    ref_bytes: &[u8],
    dist_bytes: &[u8],
    width: u32,
    height: u32,
    metric: &str,
    alpha_policy: Option<&str>,
    alpha_bg: Option<(u8, u8, u8)>,
    display_model: Option<&str>,
    luma_ingress: Option<&str>,
    gpu_runtime: Option<&str>,
) -> PyResult<PyScoreResult> {
    let expected = (width as usize) * (height as usize) * 4;
    if ref_bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "reference buffer length {} does not match expected {} bytes for {}x{} RGBA",
            ref_bytes.len(),
            expected,
            width,
            height
        )));
    }
    if dist_bytes.len() != expected {
        return Err(PyValueError::new_err(format!(
            "distorted buffer length {} does not match expected {} bytes for {}x{} RGBA",
            dist_bytes.len(),
            expected,
            width,
            height
        )));
    }

    let policy = parse_alpha_policy(alpha_policy, alpha_bg)?;

    // Dual-track alpha fidelity tracking
    let alpha_stats = compute_rgba_alpha_stats(ref_bytes, dist_bytes)
        .map_err(|e| PyValueError::new_err(format!("alpha calculation failed: {e}")))?;

    // Linear-light compositing via the fundamental Rust API
    let ref_rgb = composite_rgba_srgb8(ref_bytes, width, height, policy)
        .map_err(|e| PyValueError::new_err(format!("reference compositing failed: {e}")))?;
    let dist_rgb = composite_rgba_srgb8(dist_bytes, width, height, policy)
        .map_err(|e| PyValueError::new_err(format!("distorted compositing failed: {e}")))?;

    let reference = Rgb8Image {
        pixels: ref_rgb,
        width,
        height,
    };
    let distorted = Rgb8Image {
        pixels: dist_rgb,
        width,
        height,
    };

    let kind = parse_metric_kind(metric)?;
    let gpu_rt = parse_gpu_runtime(gpu_runtime)?;
    let l_ingress = parse_luma_ingress(luma_ingress)?;

    let mut res = execute_metric(
        kind,
        &reference,
        &distorted,
        display_model,
        l_ingress,
        gpu_rt,
    )?;

    res.scores.insert("alpha_max_delta".to_string(), alpha_stats.max_delta as f64);
    res.scores.insert("alpha_pixels_differing".to_string(), alpha_stats.pixels_differing as f64);
    res.scores.insert("alpha_mae".to_string(), alpha_stats.mae);
    res.scores.insert("alpha_psnr".to_string(), alpha_stats.psnr);
    Ok(res)
}

#[pyfunction]
#[pyo3(signature = (ref_path, variant_paths, metrics, display_model=None))]
pub fn compare_files(
    ref_path: &str,
    variant_paths: Vec<String>,
    metrics: Vec<String>,
    display_model: Option<&str>,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let py = unsafe { Python::assume_gil_acquired() };
    let ref_img = decode_image_to_rgb8(Path::new(ref_path)).map_err(|e| {
        PyIOError::new_err(format!(
            "failed to decode reference image '{ref_path}': {e}"
        ))
    })?;

    let metric_kinds: Vec<MetricKind> = metrics
        .iter()
        .map(|m| parse_metric_kind(m))
        .collect::<PyResult<Vec<_>>>()?;

    let mut out = Vec::with_capacity(variant_paths.len());

    for v_path in &variant_paths {
        let mut row_dict = HashMap::new();
        row_dict.insert("reference".to_string(), ref_path.into_py(py));
        row_dict.insert("variant".to_string(), v_path.into_py(py));

        match decode_image_to_rgb8(Path::new(v_path)) {
            Ok(var_img) => {
                for &kind in &metric_kinds {
                    match execute_metric(
                        kind,
                        &ref_img,
                        &var_img,
                        display_model,
                        LumaIngress::House,
                        GpuRuntime::Auto,
                    ) {
                        Ok(res) => {
                            for (col, val) in res.scores {
                                row_dict.insert(col, val.into_py(py));
                            }
                        }
                        Err(e) => {
                            row_dict.insert(
                                format!("{}_error", kind.name()),
                                e.to_string().into_py(py),
                            );
                        }
                    }
                }
            }
            Err(e) => {
                row_dict.insert(
                    "decode_error".to_string(),
                    format!("failed to decode '{v_path}': {e}").into_py(py),
                );
            }
        }

        out.push(row_dict);
    }

    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (ref_dir, dist_dir, metric="ssim2", recursive=false, display_model=None))]
pub fn score_folder(
    ref_dir: &str,
    dist_dir: &str,
    metric: &str,
    recursive: bool,
    display_model: Option<&str>,
) -> PyResult<HashMap<String, PyObject>> {
    let py = unsafe { Python::assume_gil_acquired() };
    let kind = parse_metric_kind(metric)?;

    let r_path = Path::new(ref_dir);
    let d_path = Path::new(dist_dir);

    if !r_path.is_dir() {
        return Err(PyValueError::new_err(format!(
            "reference path is not a directory: '{ref_dir}'"
        )));
    }
    if !d_path.is_dir() {
        return Err(PyValueError::new_err(format!(
            "distorted path is not a directory: '{dist_dir}'"
        )));
    }

    let pairs = scan_and_pair_images(r_path, d_path, recursive)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let mut results = Vec::new();
    let mut col_stats: HashMap<String, Vec<f64>> = HashMap::new();
    let mut failed_count = 0usize;

    for (ref_file, dist_file) in pairs {
        let mut pair_dict = HashMap::new();
        pair_dict.insert("reference".to_string(), ref_file.display().to_string().into_py(py));
        pair_dict.insert("distorted".to_string(), dist_file.display().to_string().into_py(py));

        match (decode_image_to_rgb8(&ref_file), decode_image_to_rgb8(&dist_file)) {
            (Ok(r_img), Ok(d_img)) => {
                match execute_metric(
                    kind,
                    &r_img,
                    &d_img,
                    display_model,
                    LumaIngress::House,
                    GpuRuntime::Auto,
                ) {
                    Ok(score_res) => {
                        let scores_dict = PyDict::new_bound(py);
                        for (col, val) in &score_res.scores {
                            scores_dict.set_item(col, val)?;
                            col_stats.entry(col.clone()).or_default().push(*val);
                        }
                        pair_dict.insert("scores".to_string(), scores_dict.into_py(py));
                        pair_dict.insert("value".to_string(), score_res.value.into_py(py));
                    }
                    Err(e) => {
                        failed_count += 1;
                        pair_dict.insert("error".to_string(), e.to_string().into_py(py));
                    }
                }
            }
            (Err(e), _) => {
                failed_count += 1;
                pair_dict.insert(
                    "error".to_string(),
                    format!("failed to decode reference: {e}").into_py(py),
                );
            }
            (_, Err(e)) => {
                failed_count += 1;
                pair_dict.insert(
                    "error".to_string(),
                    format!("failed to decode distorted: {e}").into_py(py),
                );
            }
        }
        results.push(pair_dict.into_py(py));
    }

    let summary_dict = PyDict::new_bound(py);
    for (col, vals) in &col_stats {
        if !vals.is_empty() {
            let sum: f64 = vals.iter().sum();
            let mean = sum / vals.len() as f64;
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let s = PyDict::new_bound(py);
            s.set_item("count", vals.len())?;
            s.set_item("mean", mean)?;
            s.set_item("min", min)?;
            s.set_item("max", max)?;
            summary_dict.set_item(col, s)?;
        }
    }

    let mut out = HashMap::new();
    out.insert("metric".to_string(), kind.name().into_py(py));
    out.insert("total_pairs".to_string(), (results.len()).into_py(py));
    out.insert("successful_pairs".to_string(), (results.len() - failed_count).into_py(py));
    out.insert("failed_pairs".to_string(), failed_count.into_py(py));
    out.insert("summary".to_string(), summary_dict.into_py(py));
    out.insert("results".to_string(), results.into_py(py));

    Ok(out)
}

fn scan_and_pair_images(
    ref_dir: &Path,
    dist_dir: &Path,
    recursive: bool,
) -> Result<Vec<(std::path::PathBuf, std::path::PathBuf)>, Box<dyn std::error::Error>> {
    fn is_img(p: &Path) -> bool {
        let Some(ext) = p.extension().and_then(|s| s.to_str()) else {
            return false;
        };
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "webp" | "avif" | "jxl" | "gif" | "tiff" | "tif" | "bmp"
        )
    }

    fn scan(root: &Path, curr: &Path, rec: bool, out: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(curr)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_file() && is_img(&p) {
                if let Ok(rel) = p.strip_prefix(root) {
                    out.push(rel.to_path_buf());
                }
            } else if rec && p.is_dir() {
                scan(root, &p, rec, out)?;
            }
        }
        Ok(())
    }

    let mut dist_rels = Vec::new();
    scan(dist_dir, dist_dir, recursive, &mut dist_rels)?;
    dist_rels.sort();

    let mut ref_rels = Vec::new();
    scan(ref_dir, ref_dir, recursive, &mut ref_rels)?;
    ref_rels.sort();

    let mut exact_refs = std::collections::HashSet::new();
    let mut stem_rel_refs: HashMap<std::path::PathBuf, std::path::PathBuf> = HashMap::new();
    let mut basename_refs: HashMap<String, std::path::PathBuf> = HashMap::new();

    for r in &ref_rels {
        exact_refs.insert(r.clone());
        stem_rel_refs.insert(r.with_extension(""), r.clone());
        if let Some(s) = r.file_stem().and_then(|s| s.to_str()) {
            basename_refs.entry(s.to_ascii_lowercase()).or_insert_with(|| r.clone());
        }
    }

    let mut pairs = Vec::new();
    for d in &dist_rels {
        let full_dist = dist_dir.join(d);
        if exact_refs.contains(d) {
            pairs.push((ref_dir.join(d), full_dist));
            continue;
        }
        let d_stem = d.with_extension("");
        if let Some(r) = stem_rel_refs.get(&d_stem) {
            pairs.push((ref_dir.join(r), full_dist));
            continue;
        }
        if let Some(stem) = d.file_stem().and_then(|s| s.to_str()) {
            if let Some(r) = basename_refs.get(&stem.to_ascii_lowercase()) {
                pairs.push((ref_dir.join(r), full_dist));
                continue;
            }
        }
    }

    if pairs.is_empty() {
        return Err(format!(
            "no matching image pairs found between '{}' and '{}'",
            ref_dir.display(),
            dist_dir.display()
        )
        .into());
    }

    Ok(pairs)
}

#[pyfunction]
pub fn list_metrics() -> Vec<String> {
    all_metric_names().into_iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Stats submodule bindings (zenmetrics-stats)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn stats_spearman(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::spearman(&a, &b))
}

#[pyfunction]
pub fn stats_pearson(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::pearson(&a, &b))
}

#[pyfunction]
pub fn stats_kendall_tau(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::kendall_tau(&a, &b))
}

#[pyfunction]
pub fn stats_outlier_ratio(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<f64> {
    if predictions.len() != targets.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::outlier_ratio(&predictions, &targets))
}

#[pyfunction]
pub fn stats_pwrc_auc(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<f64> {
    if predictions.len() != targets.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::pwrc_sa_st_auc(&predictions, &targets))
}

#[pyfunction]
pub fn stats_z_rmse(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<f64> {
    if predictions.len() != targets.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::z_rmse(&predictions, &targets))
}

#[pyfunction]
pub fn stats_rescale_logistic(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<Vec<f64>> {
    if predictions.len() != targets.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    Ok(zenmetrics_stats::rescale_logistic(&predictions, &targets))
}

#[pyfunction]
pub fn stats_compute_panel(predictions: Vec<f64>, targets: Vec<f64>) -> PyResult<PyPanelStats> {
    if predictions.len() != targets.len() {
        return Err(PyValueError::new_err("vectors must have equal length"));
    }
    let p = zenmetrics_stats::compute_panel(&predictions, &targets);
    Ok(PyPanelStats {
        srocc: p.srocc,
        plcc: p.plcc,
        krocc: p.krocc,
        outlier_ratio: p.or_ratio,
        pwrc_auc: p.pwrc,
        z_rmse: p.z_rmse,
        n: p.n,
    })
}

#[pymodule]
fn _zenmetrics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScoreResult>()?;
    m.add_class::<PyPanelStats>()?;

    m.add_function(wrap_pyfunction!(score_files, m)?)?;
    m.add_function(wrap_pyfunction!(score_rgb8, m)?)?;
    m.add_function(wrap_pyfunction!(score_rgba, m)?)?;
    m.add_function(wrap_pyfunction!(compare_files, m)?)?;
    m.add_function(wrap_pyfunction!(score_folder, m)?)?;
    m.add_function(wrap_pyfunction!(list_metrics, m)?)?;

    // Stats functions
    m.add_function(wrap_pyfunction!(stats_spearman, m)?)?;
    m.add_function(wrap_pyfunction!(stats_pearson, m)?)?;
    m.add_function(wrap_pyfunction!(stats_kendall_tau, m)?)?;
    m.add_function(wrap_pyfunction!(stats_outlier_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(stats_pwrc_auc, m)?)?;
    m.add_function(wrap_pyfunction!(stats_z_rmse, m)?)?;
    m.add_function(wrap_pyfunction!(stats_rescale_logistic, m)?)?;
    m.add_function(wrap_pyfunction!(stats_compute_panel, m)?)?;

    Ok(())
}
