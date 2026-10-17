//! Metrics from `zenmetrics-pyiqa-ports` (feature `pyiqa-ports`): ports
//! whose only reference implementation is pyiqa, under pyiqa's
//! non-commercial terms. TOPIQ-FR (KADID, PIPAL), and the pyiqa conventions
//! of WaDIQaM, AHIQ, CW-SSIM, DMM and PieAPP (`*-pyiqa`).
//!
//! Models load once per process from `ZENMETRICS_NN_MODEL_DIR` (the
//! directory written by `crates/zenmetrics-pyiqa-ports/models/*/export.py`)
//! on the engine in `ZENMETRICS_NN_ENGINE`: `rten` (default), or with
//! `nn-ort` `ort-cpu` / `ort-cuda` (`ORT_DYLIB_PATH`). pieapp-pyiqa reads
//! zenmetrics-nn's `pieapp.onnx` from the same directory. Non-commercial
//! weights need `ZENMETRICS_NN_ACCEPT_WEIGHTS=<col>`; a refusal names the
//! terms' URL. `cw-ssim-pyiqa` has no model files and `dmm-pyiqa` only
//! torchvision's VGG16, so neither needs consent.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric};
use zenmetrics_nn::WeightsConsent;
use zenmetrics_pyiqa_ports::{
    AhiqPyiqa, CwSsimPyiqa, DmmPyiqa, Engine, PieAppPyiqa, Topiq, TopiqVariant, WadiqamPyiqa,
};

use super::MetricKind;
use crate::decode::Rgb8Image;

type Models = Mutex<HashMap<&'static str, Box<dyn Metric>>>;

fn engine() -> Result<Engine, Box<dyn std::error::Error>> {
    let name = std::env::var("ZENMETRICS_NN_ENGINE").unwrap_or_else(|_| "rten".into());
    Engine::parse(&name).ok_or_else(|| {
        format!(
            "ZENMETRICS_NN_ENGINE={name}: pyiqa ports run on `rten`{}",
            if cfg!(feature = "nn-ort") {
                ", `ort-cpu` or `ort-cuda`"
            } else {
                " (build with `nn-ort` for `ort-cpu` and `ort-cuda`)"
            }
        )
        .into()
    })
}

/// `ZENMETRICS_NN_ACCEPT_WEIGHTS`: comma-separated score columns whose
/// weights terms the user accepts. Names that are not this module's are
/// ignored (zenmetrics-nn's models read the same variable).
fn consent() -> WeightsConsent {
    let mut consent = WeightsConsent::new();
    if let Ok(list) = std::env::var("ZENMETRICS_NN_ACCEPT_WEIGHTS") {
        for name in list.split(',').map(str::trim) {
            if let Some(v) = TopiqVariant::ALL.into_iter().find(|v| v.column() == name) {
                consent = consent.accept(v.weights());
            }
            let terms = match name {
                "wadiqam_pyiqa" => WadiqamPyiqa::WEIGHTS,
                "ahiq_pyiqa" => AhiqPyiqa::WEIGHTS,
                "pieapp_pyiqa" => PieAppPyiqa::WEIGHTS,
                _ => continue,
            };
            consent = consent.accept(terms);
        }
    }
    consent
}

fn load(column: &'static str) -> Result<Box<dyn Metric>, Box<dyn std::error::Error>> {
    if column == "cw_ssim_pyiqa" {
        return Ok(Box::new(CwSsimPyiqa::new()));
    }
    let dir = std::env::var_os("ZENMETRICS_NN_MODEL_DIR").ok_or(
        "pyiqa ports need ZENMETRICS_NN_MODEL_DIR: a directory exported by crates/zenmetrics-pyiqa-ports/models/*/export.py",
    )?;
    let engine = engine()?;
    if let Some(v) = TopiqVariant::ALL.into_iter().find(|v| v.column() == column) {
        return Ok(Box::new(Topiq::from_dir(&dir, v, engine, &consent())?));
    }
    match column {
        "wadiqam_pyiqa" => return Ok(Box::new(WadiqamPyiqa::from_dir(&dir, engine, &consent())?)),
        "ahiq_pyiqa" => return Ok(Box::new(AhiqPyiqa::from_dir(&dir, engine, &consent())?)),
        "dmm_pyiqa" => return Ok(Box::new(DmmPyiqa::from_dir(&dir, engine)?)),
        "pieapp_pyiqa" => return Ok(Box::new(PieAppPyiqa::from_dir(&dir, engine, &consent())?)),
        _ => {}
    }
    Err(format!("no pyiqa port {column}").into())
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
        .map_err(|_| "pyiqa-port cache poisoned")?;
    if !models.contains_key(column) {
        models.insert(column, load(column)?);
    }
    let m = models.get_mut(column).ok_or("model cache")?;
    let eval = m.evaluate(image(reference)?, image(distorted)?, EvalOpts::new())?;
    let scores: Vec<(std::borrow::Cow<'static, str>, f64)> = eval.scores().collect();
    kind.column_names()
        .iter()
        .map(|&col| {
            scores
                .iter()
                .find(|(k, _)| *k == col)
                .map(|(_, v)| (col, *v))
                .ok_or_else(|| format!("{col}: no score").into())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topiq_refuses_without_consent() {
        for v in TopiqVariant::ALL {
            let err = match Topiq::from_dir(
                "/nonexistent-topiq",
                v,
                Engine::Rten,
                &WeightsConsent::new(),
            ) {
                Err(e) => e,
                Ok(_) => panic!("loaded without consent"),
            };
            assert!(
                matches!(err, zenmetrics_nn::Error::WeightsNotAccepted(t) if t == v.weights()),
                "{err}"
            );
        }
    }

    #[test]
    fn noncommercial_pyiqa_conventions_refuse_without_consent() {
        // from_dir checks consent before reading any file.
        let none = WeightsConsent::new();
        for (terms, err) in [
            (
                WadiqamPyiqa::WEIGHTS,
                WadiqamPyiqa::from_dir("/nonexistent-wadiqam", Engine::Rten, &none).err(),
            ),
            (
                AhiqPyiqa::WEIGHTS,
                AhiqPyiqa::from_dir("/nonexistent-ahiq", Engine::Rten, &none).err(),
            ),
            (
                PieAppPyiqa::WEIGHTS,
                PieAppPyiqa::from_dir("/nonexistent-pieapp", Engine::Rten, &none).err(),
            ),
        ] {
            let err = err.expect("loaded without consent");
            assert!(
                matches!(err, zenmetrics_nn::Error::WeightsNotAccepted(t) if t == terms),
                "{err}"
            );
        }
    }
}
