//! Score a caller-supplied TSV of ref_path/dist_path with PieAPP.
//! Usage: pieapp_score MODEL_DIR PAIRS_TSV OUTPUT_TSV rten|ort-cuda.
//! The caller explicitly chooses this NC-weight measurement tool.
use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{Engine, PieApp, WeightsConsent};

fn load(path: &str) -> (Vec<u8>, usize, usize) {
    let bytes = std::fs::read(path).expect("read PNG");
    let decoded = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .expect("decode PNG");
    let pixels = decoded.pixels.as_slice();
    assert_eq!(
        pixels.descriptor().bytes_per_pixel(),
        3,
        "expected RGB8: {path}"
    );
    let (w, h) = (pixels.width() as usize, pixels.rows() as usize);
    let mut tight = Vec::with_capacity(w * h * 3);
    for row in pixels.as_strided_bytes().chunks(pixels.stride()).take(h) {
        tight.extend_from_slice(&row[..w * 3]);
    }
    (tight, w, h)
}
fn image(pixels: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(
        PixelSlice::new(
            pixels,
            w as u32,
            h as u32,
            w * 3,
            PixelDescriptor::RGB8_SRGB,
        )
        .unwrap(),
    )
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "MODEL_DIR PAIRS_TSV OUTPUT_TSV rten|ort-cuda"
    );
    let engine = match args[4].as_str() {
        "rten" => Engine::Rten,
        #[cfg(feature = "ort")]
        "ort-cuda" => Engine::OrtCuda,
        other => panic!("unknown or unbuilt engine {other}"),
    };
    let mut model = PieApp::from_dir(
        Path::new(&args[1]),
        engine,
        &WeightsConsent::new().accept(PieApp::WEIGHTS),
    )
    .expect("load model");
    let pairs = std::fs::read_to_string(&args[2]).expect("read pairs");
    let mut lines = pairs.lines();
    let header = lines
        .next()
        .expect("header")
        .split('\t')
        .collect::<Vec<_>>();
    let ri = header
        .iter()
        .position(|&x| x == "ref_path")
        .expect("ref_path column");
    let di = header
        .iter()
        .position(|&x| x == "dist_path")
        .expect("dist_path column");
    let mut grouped: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        grouped.entry(fields[ri]).or_default().push(fields[di]);
    }
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])
        .expect("new output path");
    writeln!(out, "ref_path\tdist_path\tpieapp").unwrap();
    let mut count = 0;
    for (reference, distorted) in grouped {
        let (r, w, h) = load(reference);
        let mut warm = model
            .warm_reference(image(&r, w, h), EvalOpts::new())
            .expect("prepare reference");
        for path in distorted {
            let (d, dw, dh) = load(path);
            assert_eq!((w, h), (dw, dh));
            let mut score = || {
                let evaluation = warm
                    .evaluate(image(&d, w, h), EvalOpts::new())
                    .expect("score");
                evaluation
                    .scores()
                    .find(|(k, _)| k.as_ref() == "pieapp")
                    .expect("column")
                    .1
            };
            let a = score();
            let b = score();
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "nonreproducible score {path}: first={a:.17e} repeated={b:.17e}"
            );
            let value = a;
            assert!(value.is_finite());
            writeln!(out, "{reference}\t{path}\t{value:.17e}").unwrap();
            out.flush().unwrap();
            count += 1;
            eprintln!("PieAPP {count} {path}");
        }
    }
}
