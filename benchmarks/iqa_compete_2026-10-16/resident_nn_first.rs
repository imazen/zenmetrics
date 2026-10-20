#![forbid(unsafe_code)]
//! First evaluation after immutable model loading, using frozen CLI helpers.
mod decode {
    pub use zenmetrics_cli::decode::Rgb8Image;
}
mod original_nn;
mod pyiqa_nn;
mod timing;
use decode::Rgb8Image;
use std::fs;
use zenmetrics_cli::metrics::MetricKind;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 8 {
        return Err("original|pyiqa metric width height ref.rgb dist.rgb threads".into());
    }
    let kind = MetricKind::all()
        .iter()
        .copied()
        .find(|k| k.name() == a[2])
        .ok_or("unknown metric")?;
    let (width, height, threads): (u32, u32, usize) = (a[3].parse()?, a[4].parse()?, a[7].parse()?);
    if width == 0 || height == 0 || threads == 0 {
        return Err("positive geometry and threads required".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(3))
        .ok_or("geometry overflow")?;
    let image = |path: &str| -> Result<_, Box<dyn std::error::Error>> {
        let pixels = fs::read(path)?;
        if pixels.len() != expected {
            return Err("RGB geometry".into());
        }
        Ok(Rgb8Image {
            width,
            height,
            pixels,
        })
    };
    let (r, d) = (image(&a[5])?, image(&a[6])?);
    match a[1].as_str() {
        "original" => original_nn::benchmark_prepare(kind)?,
        "pyiqa" => pyiqa_nn::benchmark_prepare(kind)?,
        _ => return Err("explicit model family required".into()),
    }
    let evaluate = || match a[1].as_str() {
        "original" => original_nn::score(kind, &r, &d),
        "pyiqa" => pyiqa_nn::score(kind, &r, &d),
        _ => unreachable!(),
    };
    let (result, elapsed) = timing::cold_call(|| {
        Ok(|| {
            let values = evaluate().map_err(|e| e.to_string())?;
            values
                .first()
                .map(|(_, v)| *v)
                .ok_or("missing score".into())
        })
    });
    let first = result?;
    let warm = evaluate()?[0].1;
    assert_eq!(
        first.to_bits(),
        warm.to_bits(),
        "first and warm score bits differ"
    );
    println!("COLD {elapsed} {first:.17e}");
    println!("WARM_SCORE_BITS {:016x}", warm.to_bits());
    Ok(())
}
