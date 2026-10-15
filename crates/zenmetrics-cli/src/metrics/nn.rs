//! Learned metrics from `zenmetrics-nn` (LPIPS ×3, ST-LPIPS ×2, DISTS, DeepDC).
//!
//! Each model loads once per process from `ZENMETRICS_NN_MODEL_DIR` (a
//! directory written by `crates/zenmetrics-nn/models/*/export.py`) on the
//! engine in `ZENMETRICS_NN_ENGINE`: `rten` (default, CPU) or `ort-cuda`
//! (needs the `nn-ort` feature and `ORT_DYLIB_PATH`).

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{DeepDc, Dists, Engine, Lpips, LpipsNet};

use super::MetricKind;
use crate::decode::Rgb8Image;

type Models = Mutex<HashMap<&'static str, Box<dyn Metric>>>;

fn engine() -> Result<Engine, Box<dyn std::error::Error>> {
    match std::env::var("ZENMETRICS_NN_ENGINE").as_deref() {
        Err(_) | Ok("rten") => Ok(Engine::Rten),
        #[cfg(feature = "nn-ort")]
        Ok("ort-cuda") => Ok(Engine::OrtCuda),
        Ok(other) => Err(format!(
            "ZENMETRICS_NN_ENGINE={other}: expected `rten`{}",
            if cfg!(feature = "nn-ort") {
                " or `ort-cuda`"
            } else {
                " (build with `nn-ort` for `ort-cuda`)"
            }
        )
        .into()),
    }
}

fn load(column: &'static str) -> Result<Box<dyn Metric>, Box<dyn std::error::Error>> {
    let dir = std::env::var_os("ZENMETRICS_NN_MODEL_DIR").ok_or(
        "learned metrics need ZENMETRICS_NN_MODEL_DIR: a directory exported by crates/zenmetrics-nn/models/*/export.py",
    )?;
    let engine = engine()?;
    if column == "dists" {
        return Ok(Box::new(Dists::from_dir(&dir, engine)?));
    }
    if column == "deepdc" {
        return Ok(Box::new(DeepDc::from_dir(&dir, engine)?));
    }
    let net = LpipsNet::ALL
        .into_iter()
        .find(|n| n.column() == column)
        .ok_or_else(|| format!("no learned metric {column}"))?;
    Ok(Box::new(Lpips::from_dir(&dir, net, engine)?))
}

fn image(img: &Rgb8Image) -> Result<Image<'_>, Box<dyn std::error::Error>> {
    let s = PixelSlice::new(
        &img.pixels,
        img.width,
        img.height,
        img.width as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )?;
    Ok(Image::from(s))
}

pub(super) fn score(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    static MODELS: OnceLock<Models> = OnceLock::new();
    let column = kind.column_names()[0];
    let mut models = MODELS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| "learned-metric cache poisoned")?;
    if !models.contains_key(column) {
        models.insert(column, load(column)?);
    }
    let m = models.get_mut(column).ok_or("model cache")?;
    let eval = m.evaluate(image(reference)?, image(distorted)?, EvalOpts::new())?;
    let value = eval
        .scores()
        .find(|(k, _)| k == column)
        .map(|(_, v)| v)
        .ok_or_else(|| format!("{column}: no score"))?;
    Ok(vec![(column, value)])
}
