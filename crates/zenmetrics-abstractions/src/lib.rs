//! The object surface every metric crate implements, so consumers
//! (`zenmetrics-api`, the fleet executors, the browser viewer, batch
//! scorers) talk to *a* metric instead of N crates.
//!
//! ## The shape
//!
//! - [`Metric`] — a configured, `Send` scorer. [`Metric::id`] names it
//!   completely ([`MetricId`]: family, variant, reproduced reference,
//!   implementing crate, score revision, backend, configuration) and gives
//!   every output a fully qualified column name ([`MetricId::column`]).
//! - [`Metric::evaluate`] takes two [`Image`]s (packed [`zenpixels::PixelSlice`]
//!   or validated [`Planes`]) and [`EvalOpts`] (map request plus a how-far
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
//!   [`PairedNoReference`] runs it as a pair [`Metric`] (distorted image
//!   under each key, reference under `<key>_ref`).
//!
//! Errors are [`MetricResult<T>`] = `Result<T, whereat::At<MetricError>>`:
//! a located, classified ([`ErrorKind`]) envelope around the metric's own
//! boxed error. [`conformance::assert_conforms`] checks an implementation
//! against all of the above.
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

/// `f32` ↔ byte casts without a per-crate bytemuck dependency.
pub use bytemuck;
/// Cancellation and progress (how-far re-exports enough's `Stop` and
/// `StopReason`).
pub use how_far;
/// Error location tracking.
pub use whereat;
/// The pixel crate the signatures are written in.
pub use zenpixels;

pub use error::{
    ErrorKind, MetricError, MetricResult, ResultMetricExt, fail, find_metric_error, is_stopped,
    stop_reason,
};
pub use eval::{
    Evaluation, Extras, Resampled, ScalarEval, ScoreKind, ScoreSpec, Scores, apply_map_request,
};
pub use identity::{Backend, Implementation, MetricId, is_sanitized_column};
pub use input::{
    Image, InputFormat, OwnedImage, PlanarFormat, PlaneLayout, PlaneRole, PlaneSpec, Planes,
    Rgb8In, check_same_size, contiguous_rgb8, is_rgb8_srgb, rgb8_pair, rgb8_view, unsupported,
};
pub use map::{
    MapFold, MapGeometry, MapPolarity, MapReduce, MapRequest, MapView, PixelRect, Support,
    fold_values, gray_f32_buffer, gray_f32_slice, map_fold, map_subsample_f32,
};
pub use metric::{
    AlwaysStop, ColdReplay, EvalOpts, FrameEvaluation, FrameRate, Metric, Sequence, SequenceInfo,
    SequenceSummary, Temporal, ViewingConditions, WarmReference,
};
pub use no_reference::{NoReferenceMetric, PairedEval, PairedNoReference, REFERENCE_SUFFIX};

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
    }
}
