//! Learned full-reference image-quality metrics on exported ONNX graphs.
//!
//! Each model is a network trunk exported to ONNX (by the model's
//! `models/<name>/export.py`, from the upstream reference implementation)
//! plus a small head implemented here in Rust. Keeping the head outside the
//! graph lets a reference's features be computed once and reused across
//! many distorted candidates ([`za::Metric::warm_reference`]).
//!
//! Models so far:
//!
//! - [`Lpips`]: LPIPS v0.1 with the AlexNet, VGG16 or SqueezeNet trunk
//!   ([`LpipsNet`]; Zhang et al., CVPR 2018), parity target the upstream
//!   `lpips` 0.1.4 package.
//! - [`Dists`]: DISTS (Ding, Ma, Wang, Simoncelli, TPAMI 2020), parity target
//!   the upstream `DISTS_pytorch` 0.1 package.
//! - [`DeepDc`]: DeepDC (Zhu et al., arXiv:2211.04927), parity target the
//!   upstream `DeepDC_PyTorch` 0.2 package.
//! - [`Ckdn`]: CKDN (Zheng et al., ICCV 2021), parity target the upstream
//!   `researchmm/CKDN` checkpoint, `cat(dist, ref, dist, ref)` → `dr`.
//! - [`Arniqa`]: ARNIQA (Agnolucci, Galteri et al., WACV 2024) — a [`za::NoReferenceMetric`]
//!   (wrap it in [`za::PairedNoReference`] where a pair metric is expected);
//!   parity target upstream `single_image_inference.py`, kadid10k regressor.
//! - [`Maniqa`]: MANIQA (Yang et al., CVPRW 2022) — a [`za::NoReferenceMetric`];
//!   parity target upstream's koniq10k eval path (224 INTER_CUBIC single
//!   forward, `ckpt_koniq10k.pt`).
//! - [`Liqe`]: LIQE (Zhang et al., CVPR 2023) — a [`za::NoReferenceMetric`];
//!   parity target upstream `LIQE.forward`, `LIQE.pt` over CLIP ViT-B/32
//!   (weights licence unknown: consent-gated).
//! - [`Musiq`]: MUSIQ (Ke et al., ICCV 2021) — a [`za::NoReferenceMetric`];
//!   parity target upstream `run_predict_image.py`, `koniq_ckpt.npz`
//!   (google-research, JAX/Flax).
//! - [`Hyperiqa`]: HyperIQA (Su et al., CVPR 2020) — a [`za::NoReferenceMetric`];
//!   parity target upstream `demo.py`, `koniq_pretrained.pkl` with a
//!   documented deterministic crop grid (upstream crops at random;
//!   weights licence unknown: consent-gated).
//! - [`Paq2piq`]: PaQ-2-PiQ (Ying et al., CVPR 2020) — a
//!   [`za::NoReferenceMetric`]; parity target upstream
//!   `baidut/paq2piq` `InferenceModel.predict`,
//!   `RoIPoolModel-fit.10.bs.120.pth` (CC-BY-NC weights: consent-gated).
//! - [`WadiqamFr`] / [`WadiqamNr`]: WaDIQaM (Bosse, Maniry et al., TIP
//!   2018), `dmaniry/deepIQA` — shared conv trunk over non-overlapping
//!   32×32 patches + relu-weighted pooling, LIVE- or TID2013-trained
//!   (`WadiqamKind`); FR concatenates `(h − h_ref, h, h_ref)` and NR is
//!   a [`za::NoReferenceMetric`]. MIT code and weights.
//!
//! Engines ([`Engine`]): `rten` (pure-Rust CPU, default feature) and, with
//! the `ort` feature, ONNX Runtime on CUDA. Weights are never bundled in the
//! crate; load an exported model directory with [`Lpips::from_dir`].
//!
//! Models whose weights are not permissively licensed ([`WeightsLicence`])
//! load only with a [`WeightsConsent`] naming their `WEIGHTS` terms.

#![forbid(unsafe_code)]

mod afine;
mod ahiq;
mod arniqa;
mod ckdn;
mod common;
mod deepdc;
mod dists;
mod dmm;
mod dreamsim;
mod engine;
mod error;
mod hyperiqa;
mod liqe;
mod lpips;
mod maniqa;
mod musiq;
mod paq2piq;
mod par;
mod pieapp;
mod preprocess;
mod resize;
mod tfresize;
mod wadiqam;
mod wd;
mod weights;

pub use afine::Afine;
pub use ahiq::Ahiq;
pub use arniqa::Arniqa;
pub use ckdn::Ckdn;
pub use deepdc::DeepDc;
pub use dists::Dists;
pub use dmm::Dmm;
pub use dreamsim::Dreamsim;
pub use error::Error;
pub use hyperiqa::Hyperiqa;
pub use liqe::Liqe;
pub use lpips::{Lpips, LpipsNet};
pub use maniqa::Maniqa;
pub use musiq::Musiq;
pub use paq2piq::Paq2piq;
pub use pieapp::PieApp;
pub use wadiqam::{WadiqamFr, WadiqamKind, WadiqamNr};
pub use wd::WassersteinDistortion;
pub use weights::{WeightsConsent, WeightsLicence, WeightsTerms};

/// Which engine runs a model's network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Engine {
    /// rten, pure-Rust CPU (feature `rten`, default).
    Rten,
    /// ONNX Runtime with its CUDA execution provider, TF32 off (feature
    /// `ort`). The library is loaded at run time from `ORT_DYLIB_PATH`.
    #[cfg(feature = "ort")]
    OrtCuda,
    /// ONNX Runtime with the WebGPU plugin execution provider (feature
    /// `ort-webgpu`, implies `ort`). The provider shared library is
    /// registered at run time from `ZENMETRICS_NN_WEBGPU_EP_LIB`; sessions
    /// forbid CPU fallback.
    #[cfg(feature = "ort-webgpu")]
    OrtWebgpu,
}

use zenmetrics_abstractions as za;
