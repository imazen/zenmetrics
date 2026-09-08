use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
use zenmetrics_av1_compare::{Backend, Chroma, Config, SvtSource, decode_planar, encode};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    backend: Backend,
    quantizer: u32,
    speed: i32,
    #[serde(default = "eight")]
    bit_depth: u8,
    #[serde(default)]
    chroma: Chroma,
    #[serde(default = "one")]
    threads: u32,
    #[serde(default)]
    tune: Option<u8>,
    #[serde(default)]
    scm: Option<u8>,
    #[serde(default)]
    sb128: bool,
    #[serde(default)]
    svt_reference: Option<SvtSource>,
    #[serde(default)]
    zen_intra_edge_filter: bool,
    #[serde(default)]
    zen_restoration_unit_search: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    inputs: Vec<String>,
    max_edges: Vec<u32>,
    arms: Vec<Arm>,
    repeats: usize,
    output_dir: String,
}
fn eight() -> u8 {
    8
}
fn one() -> u32 {
    1
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn decode_reference_sdr8(
    source: &[u8],
) -> Result<image::RgbImage, Box<dyn std::error::Error>> {
    let decoded = image::load_from_memory(source)?;
    let nonopaque = match &decoded {
        image::DynamicImage::ImageRgb8(_) | image::DynamicImage::ImageLuma8(_) => false,
        image::DynamicImage::ImageRgba8(p) => p.as_raw().chunks_exact(4).any(|p| p[3] != 255),
        image::DynamicImage::ImageLumaA8(p) => p.as_raw().chunks_exact(2).any(|p| p[1] != 255),
        _ => return Err("measurement requires native 8-bit SDR input; high-depth/HDR source scoring is not implemented".into()),
    };
    if nonopaque {
        return Err(
            "measurement does not support transparent source pixels; refusing to discard alpha"
                .into(),
        );
    }
    Ok(decoded.to_rgb8())
}

/// Keep CPU/worker timing cohorts separate without publishing host identifiers.
/// Captured once, outside the timed region. Binary and configuration identities
/// remain separate row keys; neither is a substitute for hardware provenance.
fn timing_environment() -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let worker =
        std::env::var("ZEN_WORKER").or_else(|_| fs::read_to_string("/proc/sys/kernel/hostname"))?;
    let cpuinfo = fs::read_to_string("/proc/cpuinfo")?;
    let mut models: Vec<_> = cpuinfo
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(key, _)| key.trim() == "model name")
        .map(|(_, value)| value.trim().to_owned())
        .collect();
    models.sort();
    models.dedup();
    if models.is_empty() || worker.trim().is_empty() {
        return Err("measurement requires a CPU model and worker identity".into());
    }
    Ok(serde_json::json!({
        "cpu_models": models,
        "worker_identity_sha256": sha(worker.trim().as_bytes()),
        "arch": std::env::consts::ARCH,
        "os": std::env::consts::OS,
    }))
}
fn score(a: &[u8], b: &[u8], w: u32, h: u32) -> Result<f64, Box<dyn std::error::Error>> {
    let image = |p: &[u8]| {
        imgref::Img::new(
            p.as_chunks::<3>()
                .0
                .iter()
                .map(|p| [p[0], p[1], p[2]])
                .collect::<Vec<_>>(),
            w as usize,
            h as usize,
        )
    };
    Ok(fast_ssim2::compute_ssimulacra2(
        image(a).as_ref(),
        image(b).as_ref(),
    )?)
}
pub fn run(req: Measurement, emit_rows: bool) -> Result<(), Box<dyn std::error::Error>> {
    run_checked(req, emit_rows, crate::verify::saved)
}

fn run_checked(
    req: Measurement,
    emit_rows: bool,
    verify: impl FnOnce(&Path, &Path) -> Result<usize, Box<dyn std::error::Error>>,
) -> Result<(), Box<dyn std::error::Error>> {
    if req.inputs.is_empty()
        || req.arms.is_empty()
        || req.max_edges.is_empty()
        || !(2..=20).contains(&req.repeats)
        || req.max_edges.iter().any(|v| !(64..=16384).contains(v))
    {
        return Err("invalid measurement grid".into());
    }
    let expects_svt = req.arms.iter().any(|a| matches!(a.backend, Backend::Svt));
    let hardware = timing_environment()?;
    let hardware_bytes = serde_json::to_vec(&hardware)?;
    let timing_cohort = sha(&hardware_bytes);
    let out = Path::new(&req.output_dir);
    fs::create_dir(out)?;
    fs::write(out.join("timing-environment.json"), hardware_bytes)?;
    fs::create_dir(out.join("obu"))?;
    let mut rows = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("rows.jsonl"))?;
    let binary_sha = sha(&fs::read(std::env::current_exe()?)?);
    for input in req.inputs {
        let source = fs::read(&input)?;
        let original = decode_reference_sdr8(&source)?;
        for edge in &req.max_edges {
            let ratio = f64::min(
                1.0,
                *edge as f64 / original.width().max(original.height()) as f64,
            );
            let w = ((original.width() as f64 * ratio).round() as u32 / 2 * 2).max(64);
            let h = ((original.height() as f64 * ratio).round() as u32 / 2 * 2).max(64);
            let reference =
                image::imageops::resize(&original, w, h, image::imageops::FilterType::Lanczos3);
            reference.save(out.join(format!("{}-{w}x{h}-reference.png", sha(&source))))?;
            let mut prepared = Vec::new();
            for arm in &req.arms {
                let cfg = Config {
                    backend: arm.backend,
                    width: w,
                    height: h,
                    quantizer: arm.quantizer,
                    speed: arm.speed,
                    threads: arm.threads,
                    bit_depth: arm.bit_depth,
                    chroma: arm.chroma,
                    tune: arm.tune,
                    scm: arm.scm,
                    sb128: arm.sb128,
                    svt_reference: arm.svt_reference,
                    zen_intra_edge_filter: arm.zen_intra_edge_filter,
                    zen_restoration_unit_search: arm.zen_restoration_unit_search,
                };
                cfg.validate_configuration()?;
                let pixels = crate::pixels::prepare(reference.as_raw(), cfg)?;
                let converted = crate::pixels::rgb_from_samples(
                    &crate::pixels::unpack(&pixels, cfg.bit_depth),
                    cfg,
                )?;
                let ceiling = score(reference.as_raw(), &converted, w, h)?;
                prepared.push((cfg, pixels, converted, ceiling));
            }
            // Rotate the arm order each round. All arms for this source/size
            // run in this process on the same worker, with independent lifecycles.
            for round in 0..req.repeats {
                for offset in 0..req.arms.len() {
                    let (cfg, pixels, converted, ceiling) =
                        &prepared[(offset + round) % prepared.len()];
                    let cfg = *cfg;
                    cfg.validate(pixels)?;
                    let input_sha = sha(pixels);
                    let start = Instant::now();
                    let obu = encode(cfg, pixels)?;
                    let ns = start.elapsed().as_nanos();
                    let output_sha = sha(&obu);
                    let path = out.join("obu").join(format!("{output_sha}.obu"));
                    if !path.exists() {
                        fs::write(path, &obu)?;
                    }
                    let decoded = decode_planar(&obu, cfg)?;
                    let rgb = crate::pixels::rgb_from_samples(&decoded, cfg)?;
                    let row = serde_json::json!({"protocol":"av1-api-planar-measure-v3","config":cfg,"round":round,
                        "timing_cohort_sha256":timing_cohort,
                        "source":Path::new(&input).file_name().unwrap().to_string_lossy(),"source_sha256":sha(&source),
                        "input_sha256":input_sha,"binary_sha256":binary_sha,"revision":cfg.revision(),"svt_reference":cfg.resolved_svt_reference(),
                        "output_sha256":output_sha,"bytes":obu.len(),"api_elapsed_ns":ns,
                        "ssimulacra2":score(reference.as_raw(),&rgb,w,h)?,
                        "ssimulacra2_codec_only":score(converted,&rgb,w,h)?,"conversion_ceiling_ssimulacra2":ceiling,
                        "matrix":"BT709 limited; same converter for every arm","decoder":"libaom",
                        "timing_scope":"fresh-lifecycle-including-plane-preparation"});
                    writeln!(rows, "{row}")?;
                    rows.flush()?;
                    if emit_rows {
                        println!("{row}");
                    }
                }
            }
        }
    }
    // Replay only after every timed round, keeping verification work out of
    // both the timer and the interleaved measurement order.
    drop(rows);
    let checked = verify(out, &out.join("reconstruction-verification.jsonl"))?;
    if expects_svt && checked == 0 {
        return Err("SVT measurement completed without a reconstruction witness".into());
    }
    fs::write(
        out.join("validation.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "protocol": "av1-api-validation-v1", "complete": true,
            "verified_svt_cells": checked, "svt_reconstruction_required": expects_svt,
            "timing": "verification occurs after all timed rounds"
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_precision_and_alpha_are_not_silently_discarded() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source.png");
        let high = image::ImageBuffer::from_pixel(64, 64, image::Rgb([1u16, 1023, 65535]));
        high.save(&source).unwrap();
        assert!(
            decode_reference_sdr8(&fs::read(&source).unwrap())
                .unwrap_err()
                .to_string()
                .contains("native 8-bit SDR")
        );
        for alpha in [0, 128, 255] {
            image::RgbaImage::from_pixel(64, 64, image::Rgba([17, 83, 211, alpha]))
                .save(&source)
                .unwrap();
            let decoded = decode_reference_sdr8(&fs::read(&source).unwrap());
            if alpha == 255 {
                assert!(decoded.unwrap().pixels().all(|p| p.0 == [17, 83, 211]));
            } else {
                assert!(
                    decoded
                        .unwrap_err()
                        .to_string()
                        .contains("refusing to discard alpha")
                );
            }
        }
    }
    fn request(root: &Path, backend: &str) -> Measurement {
        let source = root.join("correctness-only.png");
        image::RgbImage::from_fn(64, 64, |x, y| {
            image::Rgb([(x * 3) as u8, (y * 3) as u8, ((x + y) * 2) as u8])
        })
        .save(&source)
        .unwrap();
        serde_json::from_value(serde_json::json!({
            "inputs": [source], "max_edges": [64], "repeats": 3,
            "arms": [{"backend": backend, "speed": 9, "quantizer": 32}],
            "output_dir": root.join("result")
        }))
        .unwrap()
    }
    #[test]
    fn reconstruction_runs_after_all_timing_rounds_and_is_required_for_success() {
        let tmp = tempfile::tempdir().unwrap();
        run_checked(request(tmp.path(), "zenav1-svt"), false, |root, output| {
            assert_eq!(
                fs::read_to_string(root.join("rows.jsonl"))?.lines().count(),
                3
            );
            assert!(!root.join("validation.json").exists());
            crate::verify::saved(root, output)
        })
        .unwrap();
        let validation: serde_json::Value =
            serde_json::from_slice(&fs::read(tmp.path().join("result/validation.json")).unwrap())
                .unwrap();
        assert_eq!(validation["verified_svt_cells"], 1);
        assert_eq!(validation["complete"], true);
    }
    #[test]
    fn failed_verification_retains_rows_but_never_marks_complete() {
        let tmp = tempfile::tempdir().unwrap();
        let error = run_checked(request(tmp.path(), "zenav1-svt"), false, |root, _| {
            assert_eq!(
                fs::read_to_string(root.join("rows.jsonl"))?.lines().count(),
                3
            );
            Err("injected reconstruction mismatch".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("reconstruction mismatch"));
        assert!(!tmp.path().join("result/validation.json").exists());
        assert!(tmp.path().join("result/rows.jsonl").is_file());
    }
    #[test]
    fn non_svt_measurement_records_reconstruction_as_not_applicable() {
        let tmp = tempfile::tempdir().unwrap();
        run(request(tmp.path(), "libaom"), false).unwrap();
        let validation: serde_json::Value =
            serde_json::from_slice(&fs::read(tmp.path().join("result/validation.json")).unwrap())
                .unwrap();
        assert_eq!(validation["verified_svt_cells"], 0);
        assert_eq!(validation["svt_reconstruction_required"], false);
        assert_eq!(validation["complete"], true);
    }
}
