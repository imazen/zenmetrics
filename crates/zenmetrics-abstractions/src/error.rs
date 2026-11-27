//! Unified metric error envelope, re-exported directly from [`zenpsy_api`].
//!
//! Temporal sequence-order violations (specific to [`crate::Sequence`]) attach
//! a typed [`SequenceOrderError`] detail on [`ErrorKind::InvalidInput`]; check
//! with [`is_sequence_order`].

use core::fmt;

pub use zenpsy_api::{
    ErrorKind, MetricError, MetricResult, ResultMetricExt, fail, find_metric_error, is_stopped,
    stop_reason,
};

/// Detail error attached to [`ErrorKind::InvalidInput`] when frames are pushed
/// to a [`crate::Sequence`] out of order, after a prior failure, or finished
/// with no frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequenceOrderError(pub &'static str);

impl fmt::Display for SequenceOrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl core::error::Error for SequenceOrderError {}

/// Construct a sequence-order [`MetricError`] stamped with `family`.
pub fn sequence_order_error(family: &'static str, message: &'static str) -> MetricError {
    MetricError::from_detail(ErrorKind::InvalidInput, SequenceOrderError(message))
        .with_metric(family)
}

/// `true` when `error` was produced by [`sequence_order_error`].
pub fn is_sequence_order(error: &MetricError) -> bool {
    error.detail().is_some_and(|d| d.is::<SequenceOrderError>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use alloc::string::ToString;
    use enough::{AsStopReason, StopReason};
    use whereat::At;

    #[derive(Debug)]
    struct Leaf;
    impl fmt::Display for Leaf {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("leaf failed")
        }
    }
    impl core::error::Error for Leaf {}

    #[test]
    fn display_kind_detail_and_source() {
        let e = MetricError::from_detail(ErrorKind::TooSmall, Leaf).with_metric("gmsd");
        assert_eq!(e.to_string(), "gmsd: too_small: leaf failed");
        assert!(core::error::Error::source(&e).is_some());
        assert_eq!(e.kind(), ErrorKind::TooSmall);
        assert!(e.kind().is_deterministic());
    }

    #[test]
    fn stop_reason_round_trips_through_as_stop_reason() {
        let e: MetricError = StopReason::Cancelled.into();
        assert_eq!(e.as_stop_reason(), Some(StopReason::Cancelled));
        assert!(!e.kind().is_deterministic());
        let located = whereat::at(e);
        assert_eq!(stop_reason(&located), Some(StopReason::Cancelled));
        let other = whereat::at(MetricError::new(ErrorKind::NoMap));
        assert_eq!(stop_reason(&other), None);
    }

    #[test]
    fn sequence_order_helper_round_trips() {
        let e = sequence_order_error("vmaf", "push after a failed push");
        assert_eq!(e.kind(), ErrorKind::InvalidInput);
        assert_eq!(e.metric(), Some("vmaf"));
        assert!(is_sequence_order(&e));
        assert!(!is_sequence_order(&MetricError::new(
            ErrorKind::InvalidInput
        )));
    }

    #[test]
    fn recoverable_after_erasure() {
        let r: Result<(), Leaf> = Err(Leaf);
        let err = r.metric_err("fsim", ErrorKind::InvalidInput).unwrap_err();
        let boxed: Box<dyn core::error::Error + Send + Sync> = Box::new(err);
        let found = find_metric_error(&*boxed).expect("envelope recoverable");
        assert_eq!(found.kind(), ErrorKind::InvalidInput);
        assert_eq!(found.metric(), Some("fsim"));
    }

    #[test]
    fn error_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MetricError>();
        assert_send_sync::<At<MetricError>>();
    }
}
