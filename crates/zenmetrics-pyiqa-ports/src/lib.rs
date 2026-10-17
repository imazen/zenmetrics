//! Ports of image-quality metrics whose only reference implementation is pyiqa
//! (IQA-PyTorch), following pyiqa 0.1.16's default conventions.
//!
//! This crate is under pyiqa's non-commercial terms (PolyForm Noncommercial
//! 1.0.0 and the S-Lab License 1.0, see `LICENSE-pyiqa` and `LICENSE-S-Lab`),
//! unlike the rest of the workspace. It is never published, and nothing outside
//! it may copy its code. Every model's weights are non-commercial too, so each
//! constructor takes a [`zenmetrics_nn::WeightsConsent`].
#![forbid(unsafe_code)]

mod ahiq;
mod cw_ssim;
mod dmm;
mod dmm_svd;
mod engine;
mod pieapp;
mod port;
mod topiq;
mod wadiqam;

pub use ahiq::AhiqPyiqa;
pub use cw_ssim::CwSsimPyiqa;
pub use dmm::DmmPyiqa;
pub use engine::Engine;
pub use pieapp::PieAppPyiqa;
pub use topiq::{Topiq, TopiqVariant};
pub use wadiqam::WadiqamPyiqa;

use zenmetrics_abstractions as za;
