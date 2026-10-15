//! LPIPS parity against upstream `lpips` 0.1.4 goldens, per network.
//!
//! Inputs (set by `just nn-parity`, never defaulted here):
//! - `ZENMETRICS_NN_MODEL_DIR`: a `models/lpips/export.py` output directory
//!   (`lpips_<net>_{trunk.onnx,full.onnx,lin.f32,goldens.tsv}`, `lpips_<net>.meta`).
//! - `ZENMETRICS_NN_NETS`: comma-separated networks to check (`alex,vgg,squeeze`).
//! - `ZENMETRICS_NN_LADDER`: the ladder PNGs the goldens were computed on.
//! - `ZENMETRICS_NN_ENGINE`: `rten` or `ort-cuda` (needs the `ort` feature
//!   and `ORT_DYLIB_PATH`).

use std::path::{Path, PathBuf};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric};
use zenmetrics_nn::{Engine, Lpips, LpipsNet};

/// Absolute LPIPS tolerance against the upstream CPU fp32 score. Proposed,
/// pending sign-off: the trunk runs in a different engine (rten vs PyTorch
/// MKL-DNN), so convolution summation order differs. Measured worst on the
/// 64²–2048² ladder (3 images, i265): 4.5e-8.
const TOL: f64 = 1e-6;

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

fn nets() -> Vec<LpipsNet> {
    let list = std::env::var("ZENMETRICS_NN_NETS")
        .expect("ZENMETRICS_NN_NETS must be set (run via `just nn-parity`)");
    list.split(',')
        .map(|n| {
            *LpipsNet::ALL
                .iter()
                .find(|x| x.name() == n.trim())
                .unwrap_or_else(|| panic!("unknown LPIPS net {n:?}"))
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
fn lpips_matches_upstream() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    for net in nets() {
        let mut m = Lpips::from_dir(&model_dir, net, engine()).expect("load model");
        let col = net.column();
        let path = model_dir.join(format!("lpips_{}_goldens.tsv", net.name()));
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
                err <= TOL,
                "{col} {stem} {size}: |{got} - {want}| = {err:e} > {TOL:e}"
            );
            checked += 1;
        }
        assert!(checked > 0, "{} had no rows", path.display());
        eprintln!("{col}: {checked} pairs, worst |d| {worst:.3e}");
    }
}

#[test]
fn warm_reference_equals_cold() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let (rp, w, h) = load_rgb8(&ladder.join("1415_256_ref.png"));
    let (dp, _, _) = load_rgb8(&ladder.join("1415_256_dist.png"));
    for net in nets() {
        let col = net.column();
        let mut m = Lpips::from_dir(&model_dir, net, engine()).expect("load model");
        let cold = score(
            &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                .unwrap(),
            col,
        );
        let mut warm = m.warm_reference(image(&rp, w, h), EvalOpts::new()).unwrap();
        // The trunk engine caches the reference's taps; a whole-graph engine
        // (ort-cuda) recomputes them on every candidate.
        assert_eq!(warm.reuses_reference_work(), engine() == Engine::Rten);
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
        assert_eq!(same, 0.0, "{col}");
    }
}
