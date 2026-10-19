//! Per-call numerical observations, collected after the measurement clock.
#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

pub(crate) struct ScoreObservations {
    pub(crate) expected: u64,
    pub(crate) last: AtomicU64,
    pub(crate) different: AtomicU64,
    pub(crate) count: AtomicU64,
    pub(crate) total: AtomicU64,
    pub(crate) nonfinite: AtomicU64,
    pub(crate) minimum: AtomicU64,
    pub(crate) maximum: AtomicU64,
    pub(crate) max_abs_delta: AtomicU64,
}

impl ScoreObservations {
    pub(crate) fn new(value: f64) -> Self {
        assert!(value.is_finite());
        Self {
            expected: value.to_bits(),
            last: AtomicU64::new(value.to_bits()),
            different: AtomicU64::new(u64::MAX),
            count: AtomicU64::new(0),
            total: AtomicU64::new(0),
            nonfinite: AtomicU64::new(0),
            minimum: AtomicU64::new(value.to_bits()),
            maximum: AtomicU64::new(value.to_bits()),
            max_abs_delta: AtomicU64::new(0),
        }
    }

    fn observe(&self, value: f64) {
        let bits = value.to_bits();
        self.total.fetch_add(1, Relaxed);
        self.last.store(bits, Relaxed);
        if bits != self.expected {
            self.count.fetch_add(1, Relaxed);
            let _ = self
                .different
                .compare_exchange(u64::MAX, bits, Relaxed, Relaxed);
        }
        if !value.is_finite() {
            self.nonfinite.fetch_add(1, Relaxed);
            return;
        }
        let _ = self.minimum.try_update(Relaxed, Relaxed, |old| {
            Some(f64::from_bits(old).min(value).to_bits())
        });
        let _ = self.maximum.try_update(Relaxed, Relaxed, |old| {
            Some(f64::from_bits(old).max(value).to_bits())
        });
        // Nonnegative f64 bits have the same order as their numerical values.
        // An overflow is retained as +infinity bits and logged explicitly.
        let delta = (value - f64::from_bits(self.expected)).abs();
        self.max_abs_delta.fetch_max(delta.to_bits(), Relaxed);
    }
}

pub(crate) struct ObservedScore {
    pub(crate) value: f64,
    pub(crate) observations: &'static ScoreObservations,
}

impl Drop for ObservedScore {
    fn drop(&mut self) {
        self.observations.observe(self.value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_range_and_delta_including_baseline() {
        let o = ScoreObservations::new(2.0);
        for v in [2.0, 3.0, -4.0, 2.0] {
            o.observe(v);
        }
        assert_eq!(o.total.load(Relaxed), 4);
        assert_eq!(o.count.load(Relaxed), 2);
        assert_eq!(o.nonfinite.load(Relaxed), 0);
        assert_eq!(f64::from_bits(o.minimum.load(Relaxed)), -4.0);
        assert_eq!(f64::from_bits(o.maximum.load(Relaxed)), 3.0);
        assert_eq!(f64::from_bits(o.max_abs_delta.load(Relaxed)), 6.0);
    }

    #[test]
    fn nonfinite_after_finite_alternate_is_not_hidden() {
        let o = ScoreObservations::new(1.0);
        for v in [2.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0] {
            o.observe(v);
        }
        assert_eq!(o.total.load(Relaxed), 5);
        assert_eq!(o.count.load(Relaxed), 4);
        assert_eq!(o.nonfinite.load(Relaxed), 3);
        assert_eq!(f64::from_bits(o.different.load(Relaxed)), 2.0);
        assert_eq!(f64::from_bits(o.last.load(Relaxed)), 1.0);
    }

    #[test]
    fn finite_delta_overflow_is_retained_explicitly() {
        let o = ScoreObservations::new(f64::MAX);
        o.observe(-f64::MAX);
        assert_eq!(o.nonfinite.load(Relaxed), 0);
        assert_eq!(f64::from_bits(o.max_abs_delta.load(Relaxed)), f64::INFINITY);
    }

    #[test]
    fn concurrent_observations_do_not_lose_counts_or_extrema() {
        let o = ScoreObservations::new(0.0);
        std::thread::scope(|scope| {
            for i in 1..=8 {
                let o = &o;
                scope.spawn(move || {
                    for _ in 0..128 {
                        o.observe(i as f64);
                    }
                });
            }
        });
        assert_eq!(o.total.load(Relaxed), 1024);
        assert_eq!(o.count.load(Relaxed), 1024);
        assert_eq!(f64::from_bits(o.maximum.load(Relaxed)), 8.0);
        assert_eq!(f64::from_bits(o.max_abs_delta.load(Relaxed)), 8.0);
    }

    #[test]
    fn deferred_drop_records_the_actual_value() {
        let o = Box::leak(Box::new(ScoreObservations::new(1.0)));
        drop(ObservedScore {
            value: 1.5,
            observations: o,
        });
        assert_eq!(o.total.load(Relaxed), 1);
        assert_eq!(f64::from_bits(o.last.load(Relaxed)), 1.5);
    }
}
