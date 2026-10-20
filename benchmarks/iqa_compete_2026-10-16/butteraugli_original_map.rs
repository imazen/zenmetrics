#![forbid(unsafe_code)]
//! Accuracy-only original Butteraugli map; the caller freezes the source module.
#[path = "butteraugli-original-source.rs"]
mod original;
use std::{
    fs,
    io::{BufWriter, Write},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("width height reference.rgb distorted.rgb worker-budget output.f32".into());
    }
    let (w, h, budget): (usize, usize, usize) = (a[1].parse()?, a[2].parse()?, a[5].parse()?);
    let (r, d) = (fs::read(&a[3])?, fs::read(&a[4])?);
    if budget == 0 || r.len() != w * h * 3 || r.len() != d.len() {
        return Err("RGB geometry/budget".into());
    }
    let map = original::error_map(&r, &d, w, h, w * 3, w * 3)?;
    if map.len() != w * h {
        return Err("map geometry".into());
    }
    let score = map.iter().copied().fold(0.0_f32, f32::max) as f64;
    let mut out = BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&a[6])?,
    );
    for value in map {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()?;
    println!("BUTTER_ORIGINAL_MAP {score:.17e} {w} {h} {budget} 1");
    Ok(())
}
