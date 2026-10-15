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
//! - [`Arniqa`]: ARNIQA (Agnolon et al., WACV 2024) — a [`za::NoReferenceMetric`]
//!   (wrap it in [`za::PairedNoReference`] where a pair metric is expected);
//!   parity target upstream `single_image_inference.py`, kadid10k regressor.
//! - [`Maniqa`]: MANIQA (Yang et al., CVPRW 2022) — a [`za::NoReferenceMetric`];
//!   parity target upstream's koniq10k eval path (224 INTER_CUBIC single
//!   forward, `ckpt_koniq10k.pt`).
//! - [`Liqe`]: LIQE (Zhang et al., CVPR 2023) — a [`za::NoReferenceMetric`];
//!   parity target upstream `LIQE.forward`, `LIQE.pt` over CLIP ViT-B/32
//!   (weights licence unknown: consent-gated).
//!
//! Engines ([`Engine`]): `rten` (pure-Rust CPU, default feature) and, with
//! the `ort` feature, ONNX Runtime on CUDA. Weights are never bundled in the
//! crate; load an exported model directory with [`Lpips::from_dir`].
//!
//! Models whose weights are not permissively licensed ([`WeightsLicence`])
//! load only with a [`WeightsConsent`] naming their `WEIGHTS` terms.

#![forbid(unsafe_code)]

mod afine;
mod arniqa;
mod ckdn;
mod common;
mod deepdc;
mod dists;
mod dreamsim;
mod engine;
mod error;
mod liqe;
mod lpips;
mod maniqa;
mod par;
mod preprocess;
mod resize;
mod weights;

pub use afine::Afine;
pub use arniqa::Arniqa;
pub use ckdn::Ckdn;
pub use deepdc::DeepDc;
pub use dists::Dists;
pub use dreamsim::Dreamsim;
pub use error::Error;
pub use liqe::Liqe;
pub use lpips::{Lpips, LpipsNet};
pub use maniqa::Maniqa;
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
}

use zenmetrics_abstractions as za;
