#![forbid(unsafe_code)]
//! Raw RGB map probe using the already frozen gmsd and rayon rlibs.
use std::{fs, io::Write, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("width height reference.rgb distorted.rgb threads output.f32".into());
    }
    let (w, h, threads): (usize, usize, usize) =
        (args[1].parse()?, args[2].parse()?, args[5].parse()?);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build_global()?;
    let (r, d) = (fs::read(&args[3])?, fs::read(&args[4])?);
    if r.len() != w * h * 3 || d.len() != r.len() || Path::new(&args[6]).exists() {
        return Err("RGB geometry or existing output".into());
    }
    let mut rg = vec![0.0; w * h];
    let mut dg = rg.clone();
    gmsd::rgb8_to_gray(&r, w, h, w * 3, &mut rg)?;
    gmsd::rgb8_to_gray(&d, w, h, w * 3, &mut dg)?;
    let (mw, mh) = gmsd::map_dims(w, h);
    let mut map = vec![0.0; mw * mh];
    let a = gmsd::GrayImage::packed(&rg, w, h)?;
    let b = gmsd::GrayImage::packed(&dg, w, h)?;
    let first = gmsd::gmsd_with_map(a, b, &mut map)?;
    let second = gmsd::gmsd_with_map(a, b, &mut map)?;
    let direct = gmsd::gmsd_rgb8(&r, &d, w, h, w * 3)?;
    assert_eq!(first, second);
    assert_eq!(first, direct);
    let mut out = fs::OpenOptions::new().write(true).create_new(true).open(&args[6])?;
    for value in &map {
        out.write_all(&value.to_le_bytes())?;
    }
    println!("GMSD {:.17e} MEAN {:.17e} MAP {} {} THREADS {}", first.gmsd, first.mean_gms, mw, mh, threads);
    Ok(())
}
