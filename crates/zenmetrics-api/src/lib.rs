//! # zenmetrics-api — umbrella for the six zen GPU image-quality metrics
//!
//! One enum, one constructor, one `compute_*` method. Behind the scenes
//! every variant is the corresponding per-crate opaque type ([`cvvdp_gpu::CvvdpOpaque`],
//! [`butteraugli_gpu::ButteraugliOpaque`], …) — the umbrella adds no GPU
//! work on its own; it just hides the per-crate type-name churn and the
//! per-crate version-skew bookkeeping behind a single dependency line.
//!
//! ## Three usage regimes
//!
//! 1. **Opaque, single-vendor (default).** Add `zenmetrics-api` with
//!    default features, pick a metric via [`MetricKind`], get a
//!    [`Score`] back. No cubecl types leak into your code. This is the
//!    intended surface for nearly every consumer.
//!
//!    ```no_run
//!    use zenmetrics_api::{Backend, Metric, MetricKind, MetricParams};
//!
//!    let mut m = Metric::new(
//!        MetricKind::Dssim,
//!        Backend::Cuda,
//!        256,
//!        256,
//!        MetricParams::default_for(MetricKind::Dssim),
//!    )?;
//!    let r = vec![128u8; 256 * 256 * 3];
//!    let d = vec![100u8; 256 * 256 * 3];
//!    let score = m.compute_srgb_u8(&r, &d)?;
//!    println!("{} = {:.4} (impl {})", score.metric_name, score.value, score.metric_version);
//!    # Ok::<(), zenmetrics_api::Error>(())
//!    ```
//!
//! 2. **Opaque with `pixels`.** Pass [`zenpixels::PixelSlice`] inputs
//!    via [`Metric::compute_pixels`] — strided + non-sRGB inputs are
//!    converted per-call.
//!
//! 3. **cubecl-types enabled (advanced).** With the `cubecl-types`
//!    feature on, the umbrella additionally re-exports each metric
//!    crate's typed `<Metric><R: Runtime>` and exposes
//!    [`MetricContext`] for sharing one ref/dist upload across
//!    multiple metric invocations on the same GPU. This requires every
//!    enabled metric crate to resolve to the same `cubecl` version
//!    (Cargo's workspace inheritance handles this in-tree; cross-tree
//!    patches must keep cubecl in lockstep).
//!
//! ## Scope of this crate
//!
//! `zenmetrics-api` is **purely additive** over the per-crate opaque
//! shims. It does no GPU work, defines no new kernels, and does not
//! modify the metric crates. Every score it returns is bit-identical
//! to calling the per-crate `<Metric>Opaque::compute_*` directly.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

mod error;
mod memory_mode;
mod metric;
#[cfg(feature = "pixels")]
mod pixels;

#[cfg(feature = "cubecl-types")]
pub mod context;

pub use error::Error;
pub use memory_mode::{CachedRefStripPolicy, MemoryMode};
pub use metric::{reclaim_pooled_vram, Backend, Metric, MetricKind, MetricParams, Score};

#[cfg(feature = "cubecl-types")]
pub use context::MetricContext;

// ---------------------------------------------------------------
// Per-crate re-exports so downstream code doesn't have to declare
// six dependencies. Every per-crate `Params` / public config type
// is reachable through `zenmetrics_api::<crate>` even when the
// host project only depends on `zenmetrics-api`.
// ---------------------------------------------------------------

/// Re-export of the underlying [`cvvdp_gpu`] crate (when the `cvvdp`
/// feature is enabled). Use this to reach per-metric public types
/// (params, version constants, etc.) without adding a direct crate
/// dependency.
#[cfg(feature = "cvvdp")]
pub use cvvdp_gpu as cvvdp;

/// Re-export of the underlying [`butteraugli_gpu`] crate (when the
/// `butter` feature is enabled).
#[cfg(feature = "butter")]
pub use butteraugli_gpu as butter;

/// Re-export of the underlying [`ssim2_gpu`] crate (when the `ssim2`
/// feature is enabled).
#[cfg(feature = "ssim2")]
pub use ssim2_gpu as ssim2;

/// Re-export of the underlying [`dssim_gpu`] crate (when the `dssim`
/// feature is enabled).
#[cfg(feature = "dssim")]
pub use dssim_gpu as dssim;

/// Re-export of the underlying [`iwssim_gpu`] crate (when the `iwssim`
/// feature is enabled).
#[cfg(feature = "iwssim")]
pub use iwssim_gpu as iwssim;

/// Re-export of the underlying [`zensim_gpu`] crate (when the `zensim`
/// feature is enabled).
#[cfg(feature = "zensim")]
pub use zensim_gpu as zensim;

/// Result alias for the umbrella API. Most calls return this directly.
pub type Result<T> = core::result::Result<T, Error>;
