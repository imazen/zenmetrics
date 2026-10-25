//! Replay calibration frame zero, retaining the actual linear reconstruction.
#![forbid(unsafe_code)]
use jxl_encoder::{LoopTrace, LossyConfig, PixelLayout};
use std::error::Error;
use std::io::{BufWriter, Write};
use std::path::Path;

fn srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

fn save(path: &Path, values: impl Iterator<Item = f32>) -> Result<(), Box<dyn Error>> {
    let mut file = BufWriter::new(std::fs::File::create_new(path)?);
    for v in values {
        file.write_all(&v.to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: margarine-loop-replay PAIRS_DIR NEW_OUTPUT_DIR".into());
    }
    let input = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    std::fs::create_dir(output)?;
    let index = std::fs::read_to_string(input.join("pairs.tsv"))?;
    std::fs::write(output.join("pairs.tsv"), &index)?;
    for line in index.lines().skip(1) {
        let f: Vec<_> = line.split('\t').collect();
        if f[2].contains("validation") {
            continue;
        }
        let (id, w, h, distance) = (
            f[0],
            f[4].parse::<usize>()?,
            f[5].parse::<usize>()?,
            f[3].parse::<f32>()?,
        );
        let rgba = std::fs::read(input.join(format!("{id}.ref.rgba")))?;
        if rgba.len() != w * h * 4 {
            return Err("reference geometry mismatch".into());
        }
        let rgb: Vec<_> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let trace = LoopTrace::new();
        let encoded = LossyConfig::new(distance)
            .with_effort(7)
            .with_butteraugli_iters(1)
            .encode_request(w as u32, h as u32, PixelLayout::Rgb8)
            .with_loop_trace(&trace)
            .encode(&rgb)?;
        let frames = trace.take();
        let frame = frames.first().ok_or("missing frame zero")?;
        let [r, g, b] = &frame.recon;
        if r.len() != w * h || g.len() != r.len() || b.len() != r.len() {
            return Err("reconstruction geometry mismatch".into());
        }
        let rebuilt: Vec<_> = (0..r.len())
            .flat_map(|i| [srgb8(r[i]), srgb8(g[i]), srgb8(b[i]), 255])
            .collect();
        if rebuilt != std::fs::read(input.join(format!("{id}.dist.rgba")))? {
            return Err(format!("{id}: replay changes saved RGBA reconstruction").into());
        }
        std::fs::write(output.join(format!("{id}.jxl")), encoded)?;
        save(
            &output.join(format!("{id}.dist.linear.f32")),
            (0..r.len()).flat_map(|i| [r[i], g[i], b[i]]),
        )?;
        save(
            &output.join(format!("{id}.ref.linear.f32")),
            rgb.iter().map(|&v| butteraugli::srgb_to_linear(v)),
        )?;
        save(
            &output.join(format!("{id}.loop.f32")),
            frame.diffmap.iter().copied(),
        )?;
        println!("{id}\tRGBA replay exact");
        std::io::stdout().flush()?;
    }
    Ok(())
}
