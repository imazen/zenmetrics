//! Learned metrics from `zenmetrics-nn` (LPIPS ×3, ST-LPIPS ×2, DISTS, DeepDC,
//! CKDN, A-FINE, DreamSim, AHIQ).
//!
//! Each model loads once per process from `ZENMETRICS_NN_MODEL_DIR` (a
//! directory written by `crates/zenmetrics-nn/models/*/export.py`) on the
//! engine in `ZENMETRICS_NN_ENGINE`: `rten` (default, CPU) or `ort-cuda`
//! (needs the `nn-ort` feature and `ORT_DYLIB_PATH`). Models whose weights are
//! not permissively licensed also need `ZENMETRICS_NN_ACCEPT_WEIGHTS=<col>` —
//! a refusal names the terms' URL.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::{
    Afine, Ahiq, Ckdn, DeepDc, Dists, Dmm, Dreamsim, Engine, Lpips, LpipsNet, WeightsConsent,
};

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

/// `ZENMETRICS_NN_ACCEPT_WEIGHTS`: comma-separated score columns whose
/// non-permissive weights terms the user accepts; each name maps to that
/// model's `WEIGHTS` constant. Names without terms to accept are ignored —
/// a load then fails with `WeightsNotAccepted`, which names the terms.
fn consent() -> WeightsConsent {
    let mut consent = WeightsConsent::new();
    if let Ok(list) = std::env::var("ZENMETRICS_NN_ACCEPT_WEIGHTS") {
        for name in list.split(',').map(str::trim).filter(|n| !n.is_empty()) {
            let terms = match name {
                "ckdn" => Ckdn::WEIGHTS,
                "afine" => Afine::WEIGHTS,
                "dreamsim" => Dreamsim::WEIGHTS,
                "ahiq" => Ahiq::WEIGHTS,
                _ => continue,
            };
            consent = consent.accept(terms);
        }
    }
    consent
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
    if column == "ckdn" {
        return Ok(Box::new(Ckdn::from_dir(&dir, engine, &consent())?));
    }
    if column == "afine" {
        return Ok(Box::new(Afine::from_dir(&dir, engine, &consent())?));
    }
    if column == "dreamsim" {
        return Ok(Box::new(Dreamsim::from_dir(&dir, engine, &consent())?));
    }
    if column == "ahiq" {
        return Ok(Box::new(Ahiq::from_dir(&dir, engine, &consent())?));
    }
    if column == "dmm" {
        return Ok(Box::new(Dmm::from_dir(&dir, engine)?));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_permissive_weights_refuse_without_consent() {
        // `from_dir` checks the consent before reading any file, so the
        // directory need not exist.
        let err = match Ckdn::from_dir("/nonexistent-ckdn", Engine::Rten, &WeightsConsent::new()) {
            Err(e) => e,
            Ok(_) => panic!("ckdn's weights have no stated licence"),
        };
        assert!(
            matches!(err, zenmetrics_nn::Error::WeightsNotAccepted(t) if t == Ckdn::WEIGHTS),
            "expected WeightsNotAccepted for ckdn, got {err:?}"
        );
    }
}
