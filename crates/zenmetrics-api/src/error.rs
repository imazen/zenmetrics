//! Unified error type for the umbrella API.
//!
//! Each metric crate has its own `Error` enum; the umbrella wraps them
//! into a single `Error::Metric(name, message)` variant rather than
//! exposing six separate `From<*::Error>` arms (which would force
//! consumers to match on a closed enum that grows whenever a new metric
//! lands).

use core::fmt;

/// Error returned by every [`crate::Metric`] method.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Caller asked for a metric whose Cargo feature is disabled in
    /// this build of `zenmetrics-api`.
    MetricNotEnabled {
        /// Short metric tag (`"cvvdp"`, `"ssim2"`, …).
        kind: &'static str,
    },
    /// Underlying metric crate returned a failure. The wrapped string
    /// is the metric crate's `Display` for its own error.
    Metric {
        /// Short metric tag (`"cvvdp"`, `"ssim2"`, …).
        kind: &'static str,
        /// The metric crate's own error rendered via `Display`.
        message: String,
    },
    /// `Backend` variant requested at runtime is not compiled in for
    /// this build (e.g. asked for `Backend::Cuda` but the umbrella
    /// was built `--no-default-features --features wgpu`).
    BackendNotEnabled {
        /// Short backend tag (`"cuda"`, `"wgpu"`, `"cpu"`).
        backend: &'static str,
    },
    /// Dimensions of a pixel input don't match the scorer's configured
    /// `(width, height)`.
    DimensionMismatch {
        /// Expected `(width, height)` from the scorer.
        expected: (u32, u32),
        /// Actual `(width, height)` from the caller's input.
        got: (u32, u32),
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::MetricNotEnabled { kind } => write!(
                f,
                "metric '{kind}' is not enabled in this build of zenmetrics-api (enable the matching Cargo feature)"
            ),
            Error::Metric { kind, message } => write!(f, "{kind}: {message}"),
            Error::BackendNotEnabled { backend } => write!(
                f,
                "backend '{backend}' is not enabled in this build of zenmetrics-api (enable the matching Cargo feature)"
            ),
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: scorer configured for {}x{}, got {}x{}",
                expected.0, expected.1, got.0, got.1
            ),
        }
    }
}

impl std::error::Error for Error {}
