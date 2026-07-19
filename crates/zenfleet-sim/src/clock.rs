//! Virtual time.
//!
//! Every time-dependent fleet decision — lease expiry, claim staleness, idle
//! detection — is a pure function of a `now: u64` (seconds). The simulator owns
//! one clock and advances it explicitly (op latency, settle delays, "wait until
//! the lease goes stale"), so tests are deterministic and never call `sleep`.
//!
//! The clock is a cheap shared handle: cloning it shares the same underlying
//! counter, so the store, the claim layer, and the scenario all read and advance
//! one timeline.

use std::cell::Cell;
use std::rc::Rc;

/// A shared, cloneable virtual clock measured in whole seconds since an
/// arbitrary epoch. Clones share the same counter.
#[derive(Clone, Default)]
pub struct SimClock {
    t: Rc<Cell<u64>>,
}

impl SimClock {
    /// A clock starting at `start` seconds.
    pub fn new(start: u64) -> Self {
        Self {
            t: Rc::new(Cell::new(start)),
        }
    }

    /// Current virtual time in seconds.
    pub fn now(&self) -> u64 {
        self.t.get()
    }

    /// Advance the clock by `secs` seconds. Saturating, so a test can never
    /// wrap time backwards.
    pub fn advance(&self, secs: u64) {
        self.t.set(self.t.get().saturating_add(secs));
    }

    /// Jump the clock to an absolute time (never backwards).
    pub fn set(&self, t: u64) {
        if t > self.t.get() {
            self.t.set(t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advances_and_shares_across_clones() {
        let a = SimClock::new(100);
        let b = a.clone();
        assert_eq!(b.now(), 100);
        a.advance(50);
        assert_eq!(b.now(), 150, "clones share one timeline");
        b.set(140); // backwards is a no-op
        assert_eq!(a.now(), 150);
        b.set(200);
        assert_eq!(a.now(), 200);
    }
}
