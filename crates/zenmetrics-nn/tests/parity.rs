//! Parity of every zenmetrics-nn model against its upstream package's goldens.
//!
//! Inputs (set by `just nn-parity`, never defaulted here):
//! - `ZENMETRICS_NN_MODEL_DIR`: the `models/*/export.py` output directory
//!   (`<model>_goldens.tsv` next to each model's files).
//! - `ZENMETRICS_NN_MODELS`: comma-separated models to check:
//!   `lpips_alex`, `lpips_vgg`, `lpips_squeeze`, `stlpips_alex`, `stlpips_vgg`,
//!   `dists`, `deepdc`, `ckdn`.
//! - `ZENMETRICS_NN_LADDER`: the ladder PNGs the goldens were computed on.
//! - `ZENMETRICS_NN_ENGINE`: `rten` or `ort-cuda` (needs the `ort` feature
//!   and `ORT_DYLIB_PATH`).

use std::path::{Path, PathBuf};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric};
use zenmetrics_nn::{Afine, Ckdn, DeepDc, Dists, Engine, Lpips, LpipsNet, WeightsConsent};

/// Tolerance against the upstream CPU fp32 score, in JND units (owner,
/// 2026-10-15: "should be in estimated jnd units, could be looser"): each
/// model may differ from upstream by 1/1000 of its estimated just-noticeable
/// difference. The engines differ from PyTorch only in summation order;
/// measured worst on the 64²–2048² ladder is ~2e-6 JND.
const TOL_JND: f64 = 1e-3;

/// One estimated JND per score column, from `models/jnd_scale.tsv`
/// (KonJND-1k, pooled median at each image's JND threshold). A column
/// without an entry fails: every model needs its JND measured.
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

/// Absolute tolerance for `col`: `TOL_JND` JNDs.
fn tol(col: &str) -> f64 {
    TOL_JND * jnd(col)
}

fn env_dir(var: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(var)
            .unwrap_or_else(|| panic!("{var} must be set (run via `just nn-parity`)")),
    )
}

fn engine() -> Engine {
    match std::env::var("ZENMETRICS_NN_ENGINE").as_deref() {
        Ok("rten") => Engine::Rten,
        #[cfg(feature = "ort")]
        Ok("ort-cuda") => Engine::OrtCuda,
        other => panic!("ZENMETRICS_NN_ENGINE must name a built engine, got {other:?}"),
    }
}

/// The models named in `ZENMETRICS_NN_MODELS`, loaded, with their columns.
fn models() -> Vec<(Box<dyn Metric>, &'static str)> {
    let dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let list = std::env::var("ZENMETRICS_NN_MODELS")
        .expect("ZENMETRICS_NN_MODELS must be set (run via `just nn-parity`)");
    list.split(',')
        .map(|name| -> (Box<dyn Metric>, &'static str) {
            let name = name.trim();
            if name == "dists" {
                return (
                    Box::new(Dists::from_dir(&dir, engine()).expect("load dists")),
                    "dists",
                );
            }
            if name == "deepdc" {
                return (
                    Box::new(DeepDc::from_dir(&dir, engine()).expect("load deepdc")),
                    "deepdc",
                );
            }
            if name == "ckdn" {
                let consent = WeightsConsent::new().accept(Ckdn::WEIGHTS);
                return (
                    Box::new(Ckdn::from_dir(&dir, engine(), &consent).expect("load ckdn")),
                    "ckdn",
                );
            }
            if name == "afine" {
                let consent = WeightsConsent::new().accept(Afine::WEIGHTS);
                return (
                    Box::new(Afine::from_dir(&dir, engine(), &consent).expect("load afine")),
                    "afine",
                );
            }
            let net = *LpipsNet::ALL
                .iter()
                .find(|n| n.column() == name)
                .unwrap_or_else(|| panic!("unknown model {name:?}"));
            (
                Box::new(Lpips::from_dir(&dir, net, engine()).expect("load lpips")),
                net.column(),
            )
        })
        .collect()
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

fn score(e: &dyn Evaluation, col: &str) -> f64 {
    e.scores().find(|(k, _)| k == col).expect("score column").1
}

fn image(px: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

#[test]
fn matches_upstream() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    for (mut m, col) in models() {
        let path = model_dir.join(format!("{col}_goldens.tsv"));
        let goldens =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut checked = 0;
        let mut worst = 0.0f64;
        for line in goldens.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            let (stem, size, want): (&str, usize, f64) =
                (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
            let (rp, w, h) = load_rgb8(&ladder.join(format!("{stem}_{size}_ref.png")));
            let (dp, _, _) = load_rgb8(&ladder.join(format!("{stem}_{size}_dist.png")));
            let got = score(
                &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                    .expect("evaluate"),
                col,
            );
            let err = (got - want).abs();
            worst = worst.max(err);
            eprintln!("{col} {stem} {size}: got {got:.9} want {want:.9} |d| {err:.3e}");
            assert!(
                err <= tol(col),
                "{col} {stem} {size}: |{got} - {want}| = {err:e} > {:e} ({TOL_JND} JND)",
                tol(col)
            );
            checked += 1;
        }
        assert!(checked > 0, "{} had no rows", path.display());
        eprintln!(
            "{col}: {checked} pairs, worst |d| {worst:.3e} = {:.2e} JND",
            worst / jnd(col)
        );
    }
}

#[test]
fn warm_reference_equals_cold() {
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let (rp, w, h) = load_rgb8(&ladder.join("1415_256_ref.png"));
    let (dp, _, _) = load_rgb8(&ladder.join("1415_256_dist.png"));
    for (mut m, col) in models() {
        let cold = score(
            &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                .unwrap(),
            col,
        );
        // CKDN is a quality score, not a distance: an identical pair's dr is
        // sigmoid(fc(csp(qse(x) - dte(x)))) — nonzero because the two encoders
        // differ. For it the identical-pair check below is warm == cold.
        let cold_same = matches!(col, "ckdn" | "afine").then(|| {
            score(
                &*m.evaluate(image(&rp, w, h), image(&rp, w, h), EvalOpts::new())
                    .unwrap(),
                col,
            )
        });
        let mut warm = m.warm_reference(image(&rp, w, h), EvalOpts::new()).unwrap();
        // A trunk engine caches the reference's features; a whole-graph
        // engine (ort-cuda) recomputes them on every candidate.
        assert_eq!(
            warm.reuses_reference_work(),
            engine() == Engine::Rten,
            "{col}"
        );
        for _ in 0..2 {
            let s = score(
                &*warm.evaluate(image(&dp, w, h), EvalOpts::new()).unwrap(),
                col,
            );
            assert_eq!(s.to_bits(), cold.to_bits(), "{col}");
        }
        let same = score(
            &*warm.evaluate(image(&rp, w, h), EvalOpts::new()).unwrap(),
            col,
        );
        if let Some(cold_same) = cold_same {
            assert_eq!(same.to_bits(), cold_same.to_bits(), "{col}");
        } else if engine() == Engine::Rten || col != "dists" {
            assert_eq!(same, 0.0, "{col}");
        } else {
            // On ort-cuda the whole DISTS graph, including 1 - (dist1 + dist2),
            // runs in ONNX Runtime's f32 reduction order: measured 1.19e-7
            // (one f32 step at 1.0) for an identical pair. Bounded by the
            // parity tolerance.
            assert!(
                same.abs() <= tol(col),
                "{col}: identical pair scored {same}"
            );
        }
    }
}
