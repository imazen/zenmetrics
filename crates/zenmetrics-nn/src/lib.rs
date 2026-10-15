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
//!
//! Engines ([`Engine`]): `rten` (pure-Rust CPU, default feature) and, with
//! the `ort` feature, ONNX Runtime on CUDA. Weights are never bundled in the
//! crate; load an exported model directory with [`Lpips::from_dir`].

#![forbid(unsafe_code)]

mod engine;
mod error;
mod lpips;
mod preprocess;

pub use error::Error;
pub use lpips::{Lpips, LpipsNet};

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
