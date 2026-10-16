//! Score a PNG pair TSV with all five Cool-Chic WD columns. Caller supplies
//! model directory, input TSV, output TSV, and `rten` or `ort-cuda`.
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{Engine, WassersteinDistortion};

fn load(path: &str) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).expect("input PNG");
    let decoded = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .expect("PNG decode");
    let s = decoded.pixels.as_slice();
    assert_eq!(s.descriptor().bytes_per_pixel(), 3, "RGB8 required: {path}");
    let (w, h) = (s.width() as usize, s.rows() as usize);
    let bytes = s.as_strided_bytes();
    let mut pixels = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        pixels.extend_from_slice(&bytes[y * s.stride()..y * s.stride() + w * 3]);
    }
    (pixels, w, h)
}
fn image(data: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(
        PixelSlice::new(data, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB)
            .expect("pixel view"),
    )
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 5, "wd_score MODEL_DIR PAIRS_TSV OUT_TSV ENGINE");
    let engine = match args[4].as_str() {
        "rten" => Engine::Rten,
        #[cfg(feature = "ort")]
        "ort-cuda" => Engine::OrtCuda,
        other => panic!("unbuilt engine {other}"),
    };
    let mut metric = WassersteinDistortion::from_dir(&args[1], engine).expect("load WD");
    let input = std::fs::read_to_string(&args[2]).expect("pairs TSV");
    let mut lines = input.lines();
    let header = lines.next().expect("TSV header");
    let columns: Vec<_> = header.split('\t').collect();
    let ri = columns
        .iter()
        .position(|&s| s == "ref_path")
        .expect("ref_path");
    let di = columns
        .iter()
        .position(|&s| s == "dist_path")
        .expect("dist_path");
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in lines {
        let f: Vec<_> = line.split('\t').collect();
        assert!(Path::new(f[di]).exists(), "{}", f[di]);
        groups.entry(f[ri].into()).or_default().push(line.into());
    }
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[3])
        .expect("new output TSV");
    writeln!(output, "{header}\twd_s1\twd_s2\twd_s3\twd_s4\twd_s5").unwrap();
    let mut done = 0;
    for (reference, rows) in groups {
        let (r, w, h) = load(&reference);
        let mut warm = metric
            .warm_reference(image(&r, w, h), EvalOpts::new())
            .expect("reference features");
        for row in rows {
            let fields: Vec<_> = row.split('\t').collect();
            let (d, dw, dh) = load(fields[di]);
            assert_eq!((w, h), (dw, dh));
            let mut observed = Vec::new();
            let accepted = loop {
                let evaluation = warm
                    .evaluate(image(&d, w, h), EvalOpts::new())
                    .expect("WD score");
                let values: Vec<_> = evaluation.scores().map(|(_, v)| v).collect();
                assert_eq!(values.len(), 5);
                assert!(values.iter().all(|x| x.is_finite()));
                if let Some(previous) = observed.iter().find(|p: &&Vec<f64>| {
                    p.iter()
                        .zip(&values)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                }) {
                    break previous.clone();
                }
                if !observed.is_empty() {
                    eprintln!(
                        "repeat mismatch {}: {:?} then {:?}",
                        fields[di],
                        observed.last(),
                        values
                    );
                }
                observed.push(values);
                assert!(observed.len() < 3, "WD did not reproduce in three attempts");
            };
            write!(output, "{row}").unwrap();
            for score in accepted {
                write!(output, "\t{score:.17e}").unwrap();
            }
            writeln!(output).unwrap();
            output.flush().unwrap();
            done += 1;
            eprintln!("WD {done} {}", fields[di]);
        }
    }
}
