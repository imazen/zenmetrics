#![forbid(unsafe_code)]
#[allow(dead_code)]
#[path = "frozen-flip.rs"]
mod flip;
use std::{fs, io::Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("width height reference.rgb distorted.rgb threads new-map.f32".into());
    }
    let (w, h, t): (usize, usize, usize) = (a[1].parse()?, a[2].parse()?, a[5].parse()?);
    if w == 0 || h == 0 || t == 0 {
        return Err("positive geometry and workers required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build_global()?;
    let (r, d) = (fs::read(&a[3])?, fs::read(&a[4])?);
    let length = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or("geometry overflow")?;
    if r.len() != length || d.len() != length {
        return Err("RGB geometry".into());
    }
    let map = flip::error_map(&r, &d, w, h, w * 3, w * 3)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&a[6])?;
    for value in &map {
        file.write_all(&value.to_le_bytes())?
    }
    println!(
        "FLIP {:.17e} V17 {:.17e} MAP {w} {h}",
        flip::mean(&map),
        flip::mean_v17(&map)
    );
    Ok(())
}
