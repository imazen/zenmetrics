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
    if req.inputs.is_empty()
        || req.arms.is_empty()
        || req.max_edges.is_empty()
        || !(2..=20).contains(&req.repeats)
        || req.max_edges.iter().any(|v| !(64..=16384).contains(v))
    {
        return Err("invalid measurement grid".into());
    }
    let out = Path::new(&req.output_dir);
    fs::create_dir(out)?;
    fs::create_dir(out.join("obu"))?;
    let mut rows = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("rows.jsonl"))?;
    let binary_sha = sha(&fs::read(std::env::current_exe()?)?);
    for input in req.inputs {
        let source = fs::read(&input)?;
        let original = image::load_from_memory(&source)?.to_rgb8();
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
                    let row = serde_json::json!({"protocol":"av1-api-planar-measure-v2","config":cfg,"round":round,
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
    Ok(())
}
