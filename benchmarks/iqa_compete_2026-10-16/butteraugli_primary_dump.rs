#![forbid(unsafe_code)]
//! Accuracy-only map probe linked to the frozen CPU Butteraugli rlib.
use std::{fs, io::Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 7 {
        return Err("width height reference.rgb distorted.rgb threads output.f32".into());
    }
    let (w, h, threads): (usize, usize, usize) =
        (args[1].parse()?, args[2].parse()?, args[5].parse()?);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let (r, d) = (fs::read(&args[3])?, fs::read(&args[4])?);
    if r.len() != w * h * 3 || r.len() != d.len() {
        return Err("RGB geometry".into());
    }
    let pixels = |bytes: &[u8]| {
        butteraugli::Img::new(
            bytes
                .chunks_exact(3)
                .map(|p| butteraugli::RGB8::new(p[0], p[1], p[2]))
                .collect::<Vec<_>>(),
            w,
            h,
        )
    };
    let (r, d) = (pixels(&r), pixels(&d));
    let result = butteraugli::butteraugli(
        r.as_ref(),
        d.as_ref(),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(true),
    )?;
    let scalar = butteraugli::butteraugli(
        r.as_ref(),
        d.as_ref(),
        &butteraugli::ButteraugliParams::default().with_compute_diffmap(false),
    )?;
    assert_eq!(result.score.to_bits(), scalar.score.to_bits());
    let map = result.diffmap.ok_or("missing native map")?;
    assert_eq!((map.width(), map.height()), (w, h));
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[6])?;
    for row in map.rows() {
        for value in row {
            output.write_all(&value.to_le_bytes())?;
        }
    }
    println!("BUTTER_MAP {:.17e} {} {}", result.score, w, h);
    Ok(())
}
