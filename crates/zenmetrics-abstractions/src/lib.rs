//! The object surface every metric crate implements, rebased onto
//! [`zenpsy_api`] and `zenpixels 0.5`.
//!
//! ## The shape
//!
//! - [`Metric`] — a configured, `Send` scorer. [`Metric::id`] returns the
//!   unified [`zenpsy_api::MetricId`] (extended by [`MetricIdExt`] for stable
//!   FNV-1a fingerprints and column names).
//! - [`Metric::evaluate`] takes two [`Image`]s (packed [`zenpixels::PixelView`]
//!   or [`zenpixels::PlanarView`]) and [`EvalOpts`] (map request plus a how-far
//!   [`how_far::Pulse`] for cancellation), and returns a
//!   `Box<dyn Evaluation>`.
//! - [`Evaluation`] — scores keyed as [`Metric::outputs`] declares, maps as
//!   [`MapView`]s with a [`MapPolarity`] and a [`MapGeometry`] (which input
//!   pixels each value describes), on-demand folds ([`MapFold`]), extras,
//!   and a downcast to the concrete result.
//! - [`Metric::warm_reference`] — prepare a reference once for many
//!   candidates; results are bit-identical to cold evaluation.
//! - [`Metric::sequence`] — temporal sessions whose results may trail
//!   their frames ([`Temporal`]).
//! - [`NoReferenceMetric`] — one image in, a quality estimate out;
//!   [`PairedNoReference`] runs it as a pair [`Metric`].
//! - [`PsyEval`] / [`PsyWarm`] / `impl_zenpsy_*!` — direct zero-boilerplate
//!   bridge implementing [`zenpsy_api::Metric`], [`zenpsy_api::NoReferenceMetric`],
//!   [`zenpsy_api::PrepareReference`], and [`zenpsy_api::PlanarMetric`].
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

pub mod conformance;
mod error;
mod eval;
mod identity;
mod input;
mod map;
mod metric;
mod no_reference;
mod psy_bridge;

/// `f32` ↔ byte casts without a per-crate bytemuck dependency.
pub use bytemuck;
/// Cancellation and progress (how-far re-exports enough's `Stop` and
/// `StopReason`).
pub use how_far;
/// Error location tracking.
pub use whereat;
/// The pixel crate the signatures are written in (`zenpixels 0.5`).
pub use zenpixels;
/// The shared still-image quality contract (`zenpsy-api`).
pub use zenpsy_api;

pub use error::{
    ErrorKind, MetricError, MetricResult, ResultMetricExt, SequenceOrderError, fail,
    find_metric_error, is_sequence_order, is_stopped, sequence_order_error, stop_reason,
};
pub use eval::{
    Evaluation, Extras, Resampled, ScalarEval, ScoreKind, ScoreSpec, Scores, apply_map_request,
};
pub use identity::{
    Backend, BackendExt, Implementation, MetricId, MetricIdExt, implementation, is_sanitized_column,
};
pub use input::{
    ChannelOrder, Image, InputFormat, OwnedImage, PlanarBuf, PlanarFormat, PlanarView, PlaneBuf,
    PlaneFormat, PlaneRole, PlaneView, Primaries, Rgb8In, SampleLayout, Transfer, check_same_size,
    contiguous_rgb8, is_rgb8_srgb, rgb8_pair, rgb8_view, unsupported,
};
pub use map::{
    IntoPixelCow, MapFold, MapGeometry, MapPolarity, MapReduce, MapRequest, MapView, PixelRect,
    Support, fold_values, gray_f32_buffer, gray_f32_slice, map_fold, map_subsample_f32,
};
pub use metric::{
    AlwaysStop, ColdReplay, EvalOpts, FrameEvaluation, FrameRate, Metric, Sequence, SequenceInfo,
    SequenceSummary, Temporal, ViewingConditions, WarmReference,
};
pub use no_reference::{NoReferenceMetric, PairedEval, PairedNoReference, REFERENCE_SUFFIX};
pub use psy_bridge::{
    PsyEval, PsyWarm, psy_evaluate, psy_evaluate_image, psy_evaluate_planar, psy_warm_reference,
    score_spec_to_psy,
};

#[cfg(test)]
mod send_tests {
    use super::*;

    #[test]
    fn trait_objects_cross_threads() {
        fn send<T: Send + ?Sized>() {}
        fn sync<T: Sync + ?Sized>() {}
        send::<dyn Metric>();
        send::<dyn Evaluation>();
        sync::<dyn Evaluation>();
        send::<dyn WarmReference>();
        send::<dyn Sequence>();
        send::<OwnedImage>();
        send::<EvalOpts<'static>>();
        sync::<EvalOpts<'static>>();
        send::<MetricError>();
        send::<PsyEval>();
        sync::<PsyEval>();
    }
}
