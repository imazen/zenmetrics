#![forbid(unsafe_code)]
//! Raw RGB native-map probe linked to the retained, frozen HaarPSI rlib.
use std::{fs, io::Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 7 {
        return Err("width height reference.rgb distorted.rgb threads output.f32".into());
    }
    let (w, h, threads): (usize, usize, usize) = (a[1].parse()?, a[2].parse()?, a[5].parse()?);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let (r, d) = (fs::read(&a[3])?, fs::read(&a[4])?);
    if r.len() != w * h * 3 || d.len() != r.len() {
        return Err("RGB geometry".into());
    }
    let (score, map) = haarpsi::haarpsi_rgb8_with_diffmap(&r, &d, w, h, w * 3)?;
    assert_eq!(score.to_bits(), haarpsi::haarpsi_rgb8(&r, &d, w, h, w * 3)?.to_bits());
    let mut output = fs::OpenOptions::new().write(true).create_new(true).open(&a[6])?;
    for value in map.map() {
        output.write_all(&value.to_le_bytes())?;
    }
    println!("HAARPSI {:.17e} MAP {} {}", score, map.width, map.height);
    Ok(())
}
