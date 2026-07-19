//! What can go wrong with the object store, and when.
//!
//! [`FaultSpec`] is the knob panel for [`crate::FaultStore`]. Each field models
//! a specific real-world R2/s5cmd failure we have actually been bitten by (the
//! comments name the incident class). A default spec is a perfectly-behaved
//! store; you turn on exactly the faults a scenario needs, so a test's name and
//! its spec read as the same sentence ("bad credentials", "3-second read-after-
//! write window").

/// A fault schedule for [`crate::FaultStore`]. All rates are probabilities in
/// `[0, 1]` sampled from the store's seeded [`crate::Rng`]; all delays are whole
/// seconds on the [`crate::SimClock`]. A `FaultSpec::default()` never injects
/// anything.
#[derive(Clone, Debug, Default)]
pub struct FaultSpec {
    /// Seconds a freshly-PUT object stays *invisible* to `get`/`head`/`list`
    /// before it becomes readable. This is R2/S3 read-after-write eventual
    /// consistency — the exact reason the vast claim sleeps 1.5 s before reading
    /// its own claim back. `0` = strongly consistent.
    pub consistency_delay_secs: u64,

    /// Whether a writer can read *its own* most-recent write immediately, even
    /// inside the `consistency_delay_secs` window (read-your-writes). Real edge
    /// caches often give this while still hiding *other* writers' puts — which is
    /// precisely what lets a token-race claim double-acquire. Off by default
    /// (uniform eventual consistency).
    pub read_your_writes: bool,

    /// Seconds added to the clock on every store op — models per-request latency.
    /// Makes "N tiny files" cost real wall-time so a scenario can show batching
    /// (tarballs) win.
    pub op_latency_secs: u64,

    /// Probability any single op fails with a transient storage error (R2 503 /
    /// throttle). The caller is expected to retry.
    pub transient_rate: f64,

    /// Probability a `put` *appears* to succeed but stores truncated bytes — the
    /// silent partial upload (s5cmd killed mid-transfer). Content-addressed
    /// consumers must catch this via hash mismatch; a length-only consumer won't.
    pub partial_write_rate: f64,

    /// Probability a `delete` fails. Models a claim file or box that refuses to
    /// be cleaned up — the "fail to teardown" money leak when a stale claim can't
    /// be removed.
    pub delete_fail_rate: f64,

    /// Probability a `list` silently omits an otherwise-visible key. List is the
    /// weakest-consistency S3 op; a dropped key is why LIST-driven dispatch can
    /// miss work or a claim probe can miss a peer.
    pub list_drop_rate: f64,

    /// Credentials are invalid from t=0 (a mis-scoped or wrong-bucket key): every
    /// op fails with a credentials error. Models "bad credentials" outright.
    pub creds_invalid: bool,

    /// Credentials expire at this absolute clock time: ops succeed before it and
    /// fail with a credentials error at/after it. `0` = never expire. Models a
    /// scoped R2 token whose TTL is shorter than the run (the 3h→12h incident).
    pub creds_expire_at: u64,
}

impl FaultSpec {
    /// A clean store — no faults. The baseline every scenario perturbs.
    pub fn perfect() -> Self {
        Self::default()
    }

    /// An eventual-consistency window of `secs` seconds with read-your-writes on
    /// — the configuration under which the token-race claim is unsafe.
    pub fn eventual_consistency(secs: u64) -> Self {
        Self {
            consistency_delay_secs: secs,
            read_your_writes: true,
            ..Self::default()
        }
    }

    /// Credentials that are invalid from the start (403 on every op).
    pub fn bad_credentials() -> Self {
        Self {
            creds_invalid: true,
            ..Self::default()
        }
    }

    /// Credentials that expire at absolute clock time `at`.
    pub fn creds_expiring_at(at: u64) -> Self {
        Self {
            creds_expire_at: at,
            ..Self::default()
        }
    }

    /// A generally flaky store: `rate` transient errors, same-rate partial writes
    /// and list drops, plus 1 s of latency. For soak-style convergence tests.
    pub fn flaky(rate: f64) -> Self {
        Self {
            transient_rate: rate,
            partial_write_rate: rate,
            list_drop_rate: rate,
            op_latency_secs: 1,
            ..Self::default()
        }
    }
}
