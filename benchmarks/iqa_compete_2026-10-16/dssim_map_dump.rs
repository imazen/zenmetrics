#![forbid(unsafe_code)]
use std::{fs, io::Write};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err(
            "width height reference.rgb distorted.rgb threads new-output canonical|native".into(),
        );
    }
    let (w, h, t): (usize, usize, usize) = (a[1].parse()?, a[2].parse()?, a[5].parse()?);
    if w == 0 || h == 0 || t == 0 {
        return Err("positive geometry and workers required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(t)
        .build_global()?;
    let expected = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or("geometry overflow")?;
    let read = |path: &str| -> Result<Vec<rgb::RGB<u8>>, Box<dyn std::error::Error>> {
        let b = fs::read(path)?;
        if b.len() != expected {
            return Err("RGB geometry".into());
        }
        Ok(b.chunks_exact(3)
            .map(|p| rgb::RGB::new(p[0], p[1], p[2]))
            .collect())
    };
    let (r, d) = (read(&a[3])?, read(&a[4])?);
    let mut scorer = dssim_core::Dssim::new();
    let rr = scorer
        .create_image_rgb(&r, w, h)
        .ok_or("reference rejected")?;
    let dd = scorer
        .create_image_rgb(&d, w, h)
        .ok_or("distorted rejected")?;
    scorer.set_save_ssim_maps(5);
    let (score, maps) = scorer.compare(&rr, &dd);
    scorer.set_save_ssim_maps(0);
    let (plain, _) = scorer.compare(&rr, &dd);
    let value = f64::from(score);
    let plain = f64::from(plain);
    if a[7] == "native" {
        assert_eq!(
            value.to_bits(),
            plain.to_bits(),
            "native map changes scalar"
        )
    } else if a[7] != "canonical" {
        return Err("explicit role required".into());
    }
    if maps.is_empty() {
        return Err("no maps".into());
    }
    fs::create_dir(&a[6])?;
    println!(
        "SCORE {value:.17e} MAPS {} PLAIN_SCORE {plain:.17e} MAP_BITS {:016x} PLAIN_BITS {:016x}",
        maps.len(),
        value.to_bits(),
        plain.to_bits()
    );
    for (i, map) in maps.iter().enumerate() {
        let path = std::path::Path::new(&a[6]).join(format!("scale-{i}.f32"));
        let mut out = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?;
        for value in map.map.pixels() {
            out.write_all(&value.to_le_bytes())?
        }
        println!(
            "MAP {i} {} {} {:.17e}",
            map.map.width(),
            map.map.height(),
            map.ssim
        );
    }
    Ok(())
}
