//! Lease state machine (goal E): a claimed chunk holds a heartbeat-renewed lease. A dead worker's
//! lease expires in minutes — not the old fixed 600s — so the reconciler/queue re-dispatches fast,
//! while a live worker renews to keep ownership across a long job (no mid-flight steal). Pure
//! transitions; the R2 conditional-write (`If-Match` renewal) or NATS `AckWait` layer drives these.

use serde::{Deserialize, Serialize};

/// A claim's lease. `last_renew_secs` is bumped by the worker's heartbeat; expiry is measured from
/// it, not from acquisition, so long-but-alive jobs keep their claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lease {
    pub holder: String,
    pub acquired_secs: u64,
    pub last_renew_secs: u64,
    pub ttl_secs: u64,
}

impl Lease {
    pub fn new(holder: impl Into<String>, now: u64, ttl_secs: u64) -> Self {
        Self {
            holder: holder.into(),
            acquired_secs: now,
            last_renew_secs: now,
            ttl_secs,
        }
    }

    /// Expired = no heartbeat within `ttl_secs`. An expired lease is reclaimable by any worker.
    pub fn is_expired(&self, now: u64) -> bool {
        now.saturating_sub(self.last_renew_secs) >= self.ttl_secs
    }

    /// A live worker renews (heartbeat) to retain ownership across a long-running job.
    pub fn renew(&mut self, now: u64) {
        if now > self.last_renew_secs {
            self.last_renew_secs = now;
        }
    }

    /// Another worker may steal this claim only once it has expired (holder dead/stalled). This is
    /// what makes a reclaim safe — a live holder's lease can't be stolen out from under it.
    pub fn can_steal(&self, now: u64) -> bool {
        self.is_expired(now)
    }

    /// Seconds until expiry (0 if already expired) — drives the "stalled chunk" dashboard signal.
    pub fn remaining_secs(&self, now: u64) -> u64 {
        let elapsed = now.saturating_sub(self.last_renew_secs);
        self.ttl_secs.saturating_sub(elapsed)
    }
}

/// Recommended lease TTL ≈ 3× expected job duration (heartbeat renews well inside it), floored at
/// 60s so trivially-short jobs still tolerate a missed beat.
pub fn recommended_ttl_secs(expected_job_secs: u64) -> u64 {
    expected_job_secs.saturating_mul(3).max(60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_lease_is_held() {
        let l = Lease::new("w1", 1000, 120);
        assert!(!l.is_expired(1000));
        assert!(!l.is_expired(1119)); // within ttl
        assert!(!l.can_steal(1119));
        assert_eq!(l.remaining_secs(1060), 60);
    }

    #[test]
    fn expires_without_renewal() {
        let l = Lease::new("w1", 1000, 120);
        assert!(l.is_expired(1120)); // exactly ttl later
        assert!(l.can_steal(1120));
        assert_eq!(l.remaining_secs(1200), 0);
    }

    #[test]
    fn renew_keeps_long_job_alive() {
        let mut l = Lease::new("w1", 1000, 120);
        l.renew(1100); // heartbeat before expiry
        assert!(!l.is_expired(1200), "renewed lease survives past the original ttl window");
        assert!(l.is_expired(1221)); // now stale relative to the renewal
    }

    #[test]
    fn renew_never_goes_backwards() {
        let mut l = Lease::new("w1", 1000, 120);
        l.renew(900); // a stale/late heartbeat must not shorten the lease
        assert_eq!(l.last_renew_secs, 1000);
    }

    #[test]
    fn ttl_recommendation_has_floor() {
        assert_eq!(recommended_ttl_secs(10), 60); // floor
        assert_eq!(recommended_ttl_secs(120), 360); // 3×
    }
}
