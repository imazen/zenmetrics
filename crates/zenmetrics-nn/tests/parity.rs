//! LPIPS-vgg parity against upstream `lpips` 0.1.4 goldens.
//!
//! Inputs (set by `just nn-parity`, never defaulted here):
//! - `ZENMETRICS_NN_MODEL_DIR`: an `export.py` output directory
//!   (`lpips_vgg_trunk.onnx`, `lpips_vgg_lin.f32`, `goldens.tsv`).
//! - `ZENMETRICS_NN_LADDER`: the ladder PNGs the goldens were computed on.

use std::path::{Path, PathBuf};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Evaluation, Image, Metric};
use zenmetrics_nn::LpipsVgg;

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

fn lpips(e: &dyn Evaluation) -> f64 {
    e.scores()
        .find(|(k, _)| k == "lpips_vgg")
        .expect("lpips_vgg score")
        .1
}

fn image(px: &[u8], w: usize, h: usize) -> Image<'_> {
    Image::from(PixelSlice::new(px, w as u32, h as u32, w * 3, PixelDescriptor::RGB8_SRGB).unwrap())
}

#[test]
fn lpips_vgg_matches_upstream() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let mut m = LpipsVgg::from_dir(&model_dir).expect("load model");
    let goldens = std::fs::read_to_string(model_dir.join("goldens.tsv")).expect("goldens.tsv");
    let mut checked = 0;
    let mut worst = 0.0f64;
    for line in goldens.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (stem, size, want): (&str, usize, f64) =
            (f[0], f[1].parse().unwrap(), f[2].parse().unwrap());
        let (rp, w, h) = load_rgb8(&ladder.join(format!("{stem}_{size}_ref.png")));
        let (dp, _, _) = load_rgb8(&ladder.join(format!("{stem}_{size}_dist.png")));
        let got = lpips(
            &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
                .expect("evaluate"),
        );
        let err = (got - want).abs();
        worst = worst.max(err);
        eprintln!("{stem} {size}: got {got:.9} want {want:.9} |d| {err:.3e}");
        assert!(
            err <= TOL,
            "{stem} {size}: |{got} - {want}| = {err:e} > {TOL:e}"
        );
        checked += 1;
    }
    assert!(checked > 0, "goldens.tsv had no rows");
    eprintln!("{checked} pairs, worst |d| {worst:.3e}");
}

#[test]
fn warm_reference_equals_cold() {
    let model_dir = env_dir("ZENMETRICS_NN_MODEL_DIR");
    let ladder = env_dir("ZENMETRICS_NN_LADDER");
    let mut m = LpipsVgg::from_dir(&model_dir).expect("load model");
    let (rp, w, h) = load_rgb8(&ladder.join("1415_256_ref.png"));
    let (dp, _, _) = load_rgb8(&ladder.join("1415_256_dist.png"));
    let cold = lpips(
        &*m.evaluate(image(&rp, w, h), image(&dp, w, h), EvalOpts::new())
            .unwrap(),
    );
    let mut warm = m.warm_reference(image(&rp, w, h), EvalOpts::new()).unwrap();
    assert!(warm.reuses_reference_work());
    for _ in 0..2 {
        let s = lpips(&*warm.evaluate(image(&dp, w, h), EvalOpts::new()).unwrap());
        assert_eq!(s.to_bits(), cold.to_bits());
    }
    let same = lpips(&*warm.evaluate(image(&rp, w, h), EvalOpts::new()).unwrap());
    assert_eq!(same, 0.0);
}
