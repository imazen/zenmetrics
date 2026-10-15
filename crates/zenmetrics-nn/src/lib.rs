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
//! - [`LpipsVgg`]: LPIPS v0.1 with the VGG16 trunk (Zhang et al., CVPR
//!   2018), parity target the upstream `lpips` 0.1.4 package.
//!
//! Engines: `rten` (pure-Rust CPU, default feature). Weights are never
//! bundled in the crate; load an exported model directory with
//! [`LpipsVgg::from_dir`].

#![forbid(unsafe_code)]

mod engine;
mod error;
mod lpips;
mod preprocess;

pub use error::Error;
pub use lpips::LpipsVgg;

use zenmetrics_abstractions as za;
