//! Simulated paid boxes with lifecycle faults — the "infra goes wrong" half.
//!
//! A [`SimFleet`] is a set of [`SimBox`]es, each billing `rate_usd_per_hr` from
//! the moment it boots until it is destroyed. Each box can misbehave via
//! [`BoxFault`]: never boot (image-pull hang / onstart crash), die mid-run
//! (freeze / crash → goes silent), sit idle (heartbeats but does no work), or
//! **refuse to tear down** (the money leak — `destroy()` errors and the box keeps
//! billing).
//!
//! The fleet emits the **real** [`zenfleet_core::WorkerReport`] snapshots that the
//! canonical [`zenfleet_core::detect_idle`] consumes, so the tests assert the
//! production detector actually flags each failure — and that a reaper built on it
//! bounds spend. This is the piece the fleet inventory found missing in
//! production: nothing emits a `WorkerReport`, so `detect_idle` has no live feed
//! and the money-leak guard can't fire. Here it does, against realistic data.

use zenfleet_core::{
    IdleThresholds, IdleWarning, ResourceClass, Severity, WorkerReport, detect_idle,
};

use crate::clock::SimClock;

/// How a box misbehaves over its life. `default()` is a healthy box.
#[derive(Clone, Debug, Default)]
pub struct BoxFault {
    /// Never boots: it bills from creation but never emits a live heartbeat —
    /// its "last seen" stays pinned at boot time. The startup-failure case
    /// (image-pull hang, onstart crash).
    pub never_boots: bool,
    /// Absolute clock time the box goes silent (stops heartbeating and working)
    /// — a mid-run crash/freeze. `None` = healthy the whole run.
    pub dies_at: Option<u64>,
    /// Steady GPU utilization % while working (`None` for a CPU box). A live box
    /// reporting a low value is the "heartbeats but does no GPU work" idle case.
    pub gpu_util_pct: Option<u8>,
    /// Jobs completed per second while working (throughput).
    pub jobs_per_sec: f64,
    /// How many times `destroy()` fails before it finally succeeds — a box that
    /// refuses to tear down and keeps burning money. `0` = tears down first try.
    pub teardown_failures: u32,
}

/// Where a box is in its lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxState {
    /// Alive and billing (booting, working, or silently frozen — the report
    /// distinguishes those via staleness).
    Live,
    /// Torn down; no longer billing or reporting.
    Destroyed,
}

/// One simulated paid box.
pub struct SimBox {
    pub id: String,
    pub provider: String,
    pub class: ResourceClass,
    pub rate_usd_per_hr: f64,
    booted_at: u64,
    fault: BoxFault,
    destroyed_at: Option<u64>,
    teardown_attempts: u32,
}

impl SimBox {
    /// A box that booted at `booted_at` and bills `rate_usd_per_hr`.
    pub fn new(
        id: impl Into<String>,
        provider: impl Into<String>,
        class: ResourceClass,
        rate_usd_per_hr: f64,
        booted_at: u64,
        fault: BoxFault,
    ) -> Self {
        Self {
            id: id.into(),
            provider: provider.into(),
            class,
            rate_usd_per_hr,
            booted_at,
            fault,
            destroyed_at: None,
            teardown_attempts: 0,
        }
    }

    pub fn state(&self) -> BoxState {
        if self.destroyed_at.is_some() {
            BoxState::Destroyed
        } else {
            BoxState::Live
        }
    }

    /// The last time this box emitted a live heartbeat as of `now`. A healthy box
    /// beats continuously (→ `now`); a dead box freezes at its death time; a
    /// never-booted box is stuck at boot.
    fn last_beat(&self, now: u64) -> u64 {
        if self.fault.never_boots {
            return self.booted_at;
        }
        match self.fault.dies_at {
            Some(d) => d.min(now),
            None => now,
        }
    }

    fn jobs_done(&self, now: u64) -> u64 {
        if self.fault.never_boots {
            return 0;
        }
        let worked = self.last_beat(now).saturating_sub(self.booted_at);
        (worked as f64 * self.fault.jobs_per_sec) as u64
    }

    /// The `WorkerReport` snapshot this box would upload at `now`. Panics-free;
    /// mirrors what a real worker's heartbeat carries.
    pub fn report(&self, now: u64) -> WorkerReport {
        WorkerReport {
            worker: self.id.clone(),
            provider: self.provider.clone(),
            class: self.class,
            rate_usd_per_hr: self.rate_usd_per_hr,
            uptime_secs: now.saturating_sub(self.booted_at),
            jobs_done: self.jobs_done(now),
            gpu_util_pct: self.fault.gpu_util_pct,
            cpu_util_pct: None,
            last_report_unix_secs: Some(self.last_beat(now)),
        }
    }

    /// Dollars this box has burned as of `now` (until destroyed, if it was).
    pub fn spend_usd(&self, now: u64) -> f64 {
        let end = self.destroyed_at.unwrap_or(now);
        let secs = end.saturating_sub(self.booted_at);
        self.rate_usd_per_hr * (secs as f64 / 3600.0)
    }

    /// Attempt teardown. Fails (still billing) `teardown_failures` times, then
    /// succeeds. Returns `Ok(())` once the box is actually destroyed.
    pub fn destroy(&mut self, now: u64) -> Result<(), TeardownError> {
        if self.destroyed_at.is_some() {
            return Ok(());
        }
        self.teardown_attempts += 1;
        if self.teardown_attempts <= self.fault.teardown_failures {
            return Err(TeardownError {
                box_id: self.id.clone(),
                attempt: self.teardown_attempts,
            });
        }
        self.destroyed_at = Some(now);
        Ok(())
    }
}

/// A teardown that failed — the box is still alive and still billing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TeardownError {
    pub box_id: String,
    pub attempt: u32,
}

/// What one reap pass did.
#[derive(Clone, Debug, Default)]
pub struct ReapOutcome {
    /// Every idle/dead box the canonical detector flagged this pass.
    pub warnings: Vec<IdleWarning>,
    /// Boxes actually torn down this pass.
    pub destroyed: usize,
    /// Boxes the detector flagged for teardown but whose `destroy()` failed —
    /// still billing, to be retried next pass.
    pub teardown_failed: usize,
}

/// A fleet of simulated boxes sharing one clock.
pub struct SimFleet {
    clock: SimClock,
    pub boxes: Vec<SimBox>,
}

impl SimFleet {
    pub fn new(clock: SimClock) -> Self {
        Self {
            clock,
            boxes: Vec::new(),
        }
    }

    pub fn add(&mut self, b: SimBox) -> &mut Self {
        self.boxes.push(b);
        self
    }

    /// The heartbeat snapshots the still-live boxes would upload right now — the
    /// exact input `detect_idle` runs on in production.
    pub fn reports(&self) -> Vec<WorkerReport> {
        let now = self.clock.now();
        self.boxes
            .iter()
            .filter(|b| b.state() == BoxState::Live)
            .map(|b| b.report(now))
            .collect()
    }

    /// Total dollars burned across every box (destroyed or not) as of now.
    pub fn total_spend_usd(&self) -> f64 {
        let now = self.clock.now();
        self.boxes.iter().map(|b| b.spend_usd(now)).sum()
    }

    /// Boxes still alive (billing).
    pub fn live_count(&self) -> usize {
        self.boxes
            .iter()
            .filter(|b| b.state() == BoxState::Live)
            .count()
    }

    /// One reap pass: run the canonical [`detect_idle`] over the current reports
    /// and tear down every box it flags `Critical` (a paid box that is dead,
    /// frozen, or idle). A box whose teardown fails is counted and left billing —
    /// call again to retry, which is what a real reaper loop does.
    pub fn reap(&mut self, th: &IdleThresholds) -> ReapOutcome {
        let now = self.clock.now();
        let warnings = detect_idle(&self.reports(), now, th);
        let mut out = ReapOutcome {
            destroyed: 0,
            teardown_failed: 0,
            warnings: warnings.clone(),
        };
        for w in warnings.iter().filter(|w| w.severity == Severity::Critical) {
            if let Some(b) = self.boxes.iter_mut().find(|b| b.id == w.worker) {
                match b.destroy(now) {
                    Ok(()) => out.destroyed += 1,
                    Err(_) => out.teardown_failed += 1,
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_box_reports_fresh_and_busy() {
        let clock = SimClock::new(10_000);
        let b = SimBox::new(
            "box-a",
            "vast",
            ResourceClass::Gpu,
            0.40,
            10_000 - 3600, // booted an hour ago
            BoxFault {
                gpu_util_pct: Some(85),
                jobs_per_sec: 0.05,
                ..BoxFault::default()
            },
        );
        let r = b.report(clock.now());
        assert_eq!(r.last_report_unix_secs, Some(10_000), "beats up to now");
        assert!(r.jobs_done > 100, "an hour at 0.05/s ~ 180 jobs");
        assert!((b.spend_usd(clock.now()) - 0.40).abs() < 1e-9, "1 hr * $0.40");
    }

    #[test]
    fn destroy_stops_billing_after_the_configured_failures() {
        let mut b = SimBox::new(
            "box-b",
            "hetzner",
            ResourceClass::CpuHeavy,
            0.10,
            0,
            BoxFault {
                teardown_failures: 2,
                ..BoxFault::default()
            },
        );
        assert!(b.destroy(3600).is_err(), "1st teardown fails");
        assert!(b.destroy(3600).is_err(), "2nd teardown fails");
        assert!(b.destroy(3600).is_ok(), "3rd succeeds");
        assert_eq!(b.state(), BoxState::Destroyed);
        // Billing froze at destruction time, not the ever-advancing clock.
        assert!((b.spend_usd(999_999) - 0.10).abs() < 1e-9);
    }
}
