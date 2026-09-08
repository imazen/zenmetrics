use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path, time::Instant};
use zenmetrics_av1_compare::{Backend, Config, decode_i420, encode};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    backend: Backend,
    quantizer: u32,
    speed: u32,
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
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn rgb_from_i420(data: &[u8], w: u32, h: u32) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let n = (w * h) as usize;
    let (y, uv) = data.split_at(n);
    let (u, v) = uv.split_at(n / 4);
    let planes = yuv::YuvPlanarImage {
        y_plane: y,
        y_stride: w,
        u_plane: u,
        u_stride: w / 2,
        v_plane: v,
        v_stride: w / 2,
        width: w,
        height: h,
    };
    let mut rgb = vec![0; n * 3];
    yuv::yuv420_to_rgb(
        &planes,
        &mut rgb,
        w * 3,
        yuv::YuvRange::Limited,
        yuv::YuvStandardMatrix::Bt709,
    )?;
    Ok(rgb)
}
fn score(a: &[u8], b: &[u8], w: u32, h: u32) -> Result<f64, Box<dyn std::error::Error>> {
    let image = |p: &[u8]| {
        imgref::Img::new(
            p.chunks_exact(3)
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
pub fn run(req: Measurement) -> Result<(), Box<dyn std::error::Error>> {
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
            let mut planes =
                yuv::YuvPlanarImageMut::<u8>::alloc(w, h, yuv::YuvChromaSubsampling::Yuv420);
            yuv::rgb_to_yuv420(
                &mut planes,
                reference.as_raw(),
                w * 3,
                yuv::YuvRange::Limited,
                yuv::YuvStandardMatrix::Bt709,
                yuv::YuvConversionMode::Balanced,
            )?;
            let p = planes.to_fixed();
            let pixels = [p.y_plane, p.u_plane, p.v_plane].concat();
            let converted = rgb_from_i420(&pixels, w, h)?;
            let ceiling = score(reference.as_raw(), &converted, w, h)?;
            let input_sha = sha(&pixels);
            reference.save(out.join(format!("{input_sha}-reference.png")))?;
            // Rotate the arm order each round. All arms for this source/size
            // run in this process on the same worker, with independent lifecycles.
            for round in 0..req.repeats {
                for offset in 0..req.arms.len() {
                    let arm = &req.arms[(offset + round) % req.arms.len()];
                    let cfg = Config {
                        backend: arm.backend,
                        width: w,
                        height: h,
                        quantizer: arm.quantizer,
                        speed: arm.speed,
                        threads: 1,
                    };
                    cfg.validate(&pixels)?;
                    let start = Instant::now();
                    let obu = encode(cfg, &pixels)?;
                    let ns = start.elapsed().as_nanos();
                    let output_sha = sha(&obu);
                    let path = out.join("obu").join(format!("{output_sha}.obu"));
                    if !path.exists() {
                        fs::write(path, &obu)?;
                    }
                    let decoded = decode_i420(&obu, w, h)?;
                    let rgb = rgb_from_i420(&decoded, w, h)?;
                    let row = serde_json::json!({"protocol":"av1-api-i420-measure-v1","config":cfg,"round":round,
                        "source":Path::new(&input).file_name().unwrap().to_string_lossy(),"source_sha256":sha(&source),
                        "input_sha256":input_sha,"binary_sha256":binary_sha,"revision":cfg.revision(),
                        "output_sha256":output_sha,"bytes":obu.len(),"api_elapsed_ns":ns,
                        "ssimulacra2":score(reference.as_raw(),&rgb,w,h)?,
                        "ssimulacra2_codec_only":score(&converted,&rgb,w,h)?,"i420_ceiling_ssimulacra2":ceiling,
                        "matrix":"BT709 limited; same converter for every arm","decoder":"libaom",
                        "timing_scope":"fresh-lifecycle-including-plane-preparation"});
                    writeln!(rows, "{row}")?;
                    rows.flush()?;
                    println!("{row}");
                }
            }
        }
    }
    Ok(())
}
