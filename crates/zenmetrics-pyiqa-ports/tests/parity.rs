//! Parity of the pyiqa ports against pyiqa 0.1.16's own scores.
//!
//! Inputs (all required, never defaulted here):
//! - `ZENMETRICS_PYIQA_MODEL_DIR`: the `models/*/export.py` output directory
//!   (graphs plus each export's `<model>_goldens.tsv`, keyed by `distorted`).
//! - `ZENMETRICS_PYIQA_PAIRS`: TSV with `ref_path`, `dist_path`, `distorted`
//!   columns; every row must have a golden.
//! - `ZENMETRICS_PYIQA_ENGINE`: `rten`, `ort-cpu` or `ort-cuda` (the last
//!   two need the `ort` feature and `ORT_DYLIB_PATH`).
//! - `ZENMETRICS_PYIQA_MODELS`: comma-separated score columns to check
//!   (`topiq_fr`, `topiq_fr_pipal`).
//!
//! Goldens are pyiqa `metric(dist, ref)` on CPU fp32.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric};
use zenmetrics_nn::WeightsConsent;
use zenmetrics_pyiqa_ports::{Engine, Topiq, TopiqVariant};

/// Tolerance in JND units, as zenmetrics-nn's parity gate: 1/1000 of the
/// column's estimated just-noticeable difference.
const TOL_JND: f64 = 1e-3;

/// One estimated JND per score column, from `models/jnd_scale.tsv`. A column
/// without an entry fails.
fn jnd(col: &str) -> f64 {
    const TABLE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/jnd_scale.tsv"));
    TABLE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("column\t"))
        .find_map(|l| {
            let mut f = l.split('\t');
            (f.next() == Some(col)).then(|| f.next().unwrap().parse::<f64>().unwrap())
        })
        .unwrap_or_else(|| panic!("models/jnd_scale.tsv has no JND for {col}"))
}

fn env(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| panic!("{var} must be set"))
}

fn engine() -> Engine {
    let e = env("ZENMETRICS_PYIQA_ENGINE");
    Engine::parse(&e).unwrap_or_else(|| panic!("engine {e:?} unknown or not built"))
}

fn load_rgb8(path: &Path) -> (Vec<u8>, usize, usize) {
    let data = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let out = zenpng::decode(
        &data,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .expect("png decode");
    let s = out.pixels.as_slice();
    let (w, h) = (s.width() as usize, s.rows() as usize);
    assert_eq!(
        s.descriptor().bytes_per_pixel(),
        3,
        "{}: expected RGB8",
        path.display()
    );
    let bytes = s.as_strided_bytes();
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        px.extend_from_slice(&bytes[y * s.stride()..y * s.stride() + w * 3]);
    }
    (px, w, h)
}

fn image(px: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

fn score(e: &dyn Evaluation, col: &str) -> f64 {
    e.scores().find(|(k, _)| k == col).expect("score column").1
}

/// `(ref_path, dist_path, distorted)` rows.
fn pairs() -> Vec<(PathBuf, PathBuf, String)> {
    let path = env("ZENMETRICS_PYIQA_PAIRS");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let col = |n: &str| {
        header
            .iter()
            .position(|h| *h == n)
            .unwrap_or_else(|| panic!("{path}: no {n}"))
    };
    let (r, d, n) = (col("ref_path"), col("dist_path"), col("distorted"));
    lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (PathBuf::from(f[r]), PathBuf::from(f[d]), f[n].to_string())
        })
        .collect()
}

/// `distorted → value` of `col` in `<dir>/<file>`.
fn goldens(dir: &Path, file: &str, col: &str) -> HashMap<String, f64> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
    let i = header
        .iter()
        .position(|h| *h == col)
        .unwrap_or_else(|| panic!("{}: no column {col}", path.display()));
    lines
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f[0].to_string(), f[i].parse().expect("golden value"))
        })
        .collect()
}

/// The model for score column `col`, and its goldens file.
fn model(dir: &Path, col: &str) -> (Box<dyn Metric>, &'static str) {
    let variant = TopiqVariant::ALL
        .into_iter()
        .find(|v| v.column() == col)
        .unwrap_or_else(|| panic!("unknown model {col}"));
    let consent = WeightsConsent::new().accept(variant.weights());
    (
        Box::new(Topiq::from_dir(dir, variant, engine(), &consent).expect("load")),
        "topiq_goldens.tsv",
    )
}

#[test]
fn parity() {
    let dir = PathBuf::from(env("ZENMETRICS_PYIQA_MODEL_DIR"));
    let pairs = pairs();
    let mut failures = Vec::new();
    for col in env("ZENMETRICS_PYIQA_MODELS")
        .split(',')
        .filter(|c| !c.is_empty())
    {
        let (mut m, file) = model(&dir, col);
        let gold = goldens(&dir, file, col);
        let (j, tol) = (jnd(col), TOL_JND * jnd(col));
        let (mut worst, mut worst_name, mut warm_checked) = (0.0f64, String::new(), 0);
        for (rp, dp, name) in &pairs {
            let g = *gold
                .get(name)
                .unwrap_or_else(|| panic!("{col}: no golden for {name}"));
            let (r, w, h) = load_rgb8(rp);
            let (d, w2, h2) = load_rgb8(dp);
            assert_eq!((w, h), (w2, h2), "{name}");
            let cold = score(
                &*m.evaluate(image(&r, w, h), image(&d, w, h), EvalOpts::new())
                    .unwrap_or_else(|e| panic!("{col} {name}: {e}")),
                col,
            );
            let delta = (cold - g).abs();
            if delta > worst {
                (worst, worst_name) = (delta, name.clone());
            }
            if delta > tol {
                failures.push(format!(
                    "{col} {name}: rust {cold} pyiqa {g} |d| {delta:.3e}"
                ));
            }
            // Warm reference: bit-identical to the cold score, reused twice.
            if warm_checked < 3 && w * h <= 1024 * 1024 {
                let mut warm = m.warm_reference(image(&r, w, h), EvalOpts::new()).unwrap();
                assert!(warm.reuses_reference_work());
                for _ in 0..2 {
                    let s = score(
                        &*warm.evaluate(image(&d, w, h), EvalOpts::new()).unwrap(),
                        col,
                    );
                    assert_eq!(
                        s.to_bits(),
                        cold.to_bits(),
                        "{col} {name}: warm {s} != cold {cold}"
                    );
                }
                warm_checked += 1;
            }
        }
        println!(
            "{col} engine {:?} pairs {} worst |d| {worst:.3e} ({worst_name}) = {:.3e} JND, tol {tol:.3e}",
            engine(),
            pairs.len(),
            worst / j
        );
    }
    assert!(
        failures.is_empty(),
        "{} over tolerance:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
