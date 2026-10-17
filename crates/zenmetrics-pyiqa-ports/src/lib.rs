//! Ports of image-quality metrics whose only reference implementation is pyiqa
//! (IQA-PyTorch), following pyiqa 0.1.16's default conventions.
//!
//! This crate is under pyiqa's non-commercial terms (PolyForm Noncommercial
//! 1.0.0 and the S-Lab License 1.0, see `LICENSE-pyiqa` and `LICENSE-S-Lab`),
//! unlike the rest of the workspace. It is never published, and nothing outside
//! it may copy its code. Every model's weights are non-commercial too, so each
//! constructor takes a [`zenmetrics_nn::WeightsConsent`].
#![forbid(unsafe_code)]

// Used by the model modules; until the first one lands nothing calls it.
#[allow(dead_code)]
mod engine;

pub use engine::Engine;

use zenmetrics_abstractions as za;
