use brisque::Brisque;
use std::io::BufRead;
use za::zenpixels::{PixelDescriptor, PixelSlice};
use za::{EvalOpts, Image, NoReferenceMetric};
use zenmetrics_abstractions as za;

fn main() {
    let root = std::env::var_os("ZENMETRICS_NR_GOLDEN_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var("HOME").unwrap()).join("tmp"));
    let tsv = std::fs::File::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/goldens_brisque.tsv"
    ))
    .unwrap();
    let mut rows = Vec::new();
    for line in std::io::BufReader::new(tsv).lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (p, e) = line.split_once('\t').unwrap();
        rows.push((p.to_string(), e.parse::<f64>().unwrap()));
    }
    let mut res = Vec::new();
    for (p, expected) in &rows {
        let img = image::open(root.join(p)).unwrap().to_rgb8();
        let (w, h) = img.dimensions();
        let slice =
            PixelSlice::new(&img, w, h, w as usize * 3, PixelDescriptor::RGB8_SRGB).unwrap();
        let got = match Brisque::new().evaluate_image(Image::from(slice), EvalOpts::new()) {
            Ok(e) => e.score(),
            Err(_) => {
                res.push((p.clone(), *expected, f64::NAN));
                continue;
            }
        };
        res.push((p.clone(), *expected, got));
    }
    res.sort_by(|a, b| (b.1 - b.2).abs().total_cmp(&(a.1 - a.2).abs()));
    let mut n_exact = 0;
    for (p, e, g) in &res {
        let d = e - g;
        if d == 0.0 {
            n_exact += 1;
        }
        println!("{:+.6e}\t{}\texp={:.9}\tgot={:.9}", d, p, e, g);
    }
    eprintln!("exact: {}/170", n_exact);
}
