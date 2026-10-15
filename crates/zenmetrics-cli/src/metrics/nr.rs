//! Classical no-reference metrics (`piqe`, `niqe`, `brisque`) — in-tree
//! ports driven through [`PairedNoReference`], which scores the distorted
//! image under each output key and the reference under `<key>_ref`. No model
//! files or environment setup: these metrics are stateless, so a fresh
//! `PairedNoReference` per call is cheap (unlike `nn.rs`'s cached models).

use zenmetrics_abstractions::zenpixels::{PixelDescriptor, PixelSlice};
use zenmetrics_abstractions::{EvalOpts, Image, Metric, PairedNoReference};

use super::MetricKind;
use crate::decode::Rgb8Image;

fn image(img: &Rgb8Image) -> Result<Image<'_>, Box<dyn std::error::Error>> {
    Ok(Image::from(PixelSlice::new(
        &img.pixels,
        img.width,
        img.height,
        img.width as usize * 3,
        PixelDescriptor::RGB8_SRGB,
    )?))
}

pub(super) fn score(
    kind: MetricKind,
    reference: &Rgb8Image,
    distorted: &Rgb8Image,
) -> Result<Vec<(&'static str, f64)>, Box<dyn std::error::Error>> {
    let metric: Box<dyn Metric> = match kind {
        MetricKind::Piqe => Box::new(PairedNoReference::new(piqe::Piqe::new())),
        MetricKind::Niqe => Box::new(PairedNoReference::new(niqe::Niqe::new())),
        _ => return Err(format!("nr: {} is not a no-reference metric", kind.name()).into()),
    };
    let mut metric = metric;
    let eval = metric.evaluate(image(reference)?, image(distorted)?, EvalOpts::new())?;
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
