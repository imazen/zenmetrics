//! Run control state (goal C: pause / resume / drain without losing state).
//!
//! A tiny object the dashboard writes and workers read before pulling new work. The ledger stays the
//! single source of truth; this only gates whether a worker *claims new jobs*:
//! - `paused`  — pull nothing (a hard stop; resume by clearing it).
//! - `drain`   — pull no new work but let in-flight jobs finish.
//!
//! "Without losing state": pausing/draining never abandons or rewrites ledger rows — it just stops
//! the worker from claiming the next job, so resuming continues exactly where it left off.

use serde::{Deserialize, Serialize};

use crate::epoch::ClaimMode;

// NOT `Copy`/`Eq` since `worker_weights` landed: the map isn't Copy and f64 weights aren't Eq.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RunControl {
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub drain: bool,
    /// Campaign-level claim mode (see [`ClaimMode`]). `None` = the worker's own config decides
    /// (default lease). Setting it here converges the WHOLE fleet on the next pass without
    /// per-box env edits — important because mixed claim modes on one run re-introduce the
    /// duplicate-work tax (each mode's dedup is invisible to the other).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_mode: Option<ClaimMode>,
    /// Campaign override for [`crate::EpochShardCfg::epoch_len_secs`] (used when `claim_mode`
    /// is `epoch_sharded`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epoch_len_secs: Option<u64>,
    /// Campaign override for [`crate::EpochShardCfg::heartbeat_interval_secs`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heartbeat_interval_secs: Option<u64>,
    /// Campaign-level worker speed handicaps (epoch-sharded claiming): worker key →
    /// per-mode multipliers. When present this REPLACES the committed registry
    /// (`fleet/handicaps.toml`) wholesale — highest precedence, converges the fleet on the
    /// next pass, and is the recommended way to change weights on a LIVE campaign (an
    /// image-embedded registry edit only lands at the next image roll, and mixed-image
    /// fleets would shard-diverge until the roll completes).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_weights: Option<std::collections::BTreeMap<String, crate::epoch::WorkerHandicap>>,
}

impl RunControl {
    /// Running normally — pull and execute new work.
    pub const RUNNING: RunControl = RunControl {
        paused: false,
        drain: false,
        claim_mode: None,
        epoch_len_secs: None,
        heartbeat_interval_secs: None,
        worker_weights: None,
    };
    /// Hard stop — claim nothing until resumed.
    pub const PAUSED: RunControl = RunControl {
        paused: true,
        drain: false,
        claim_mode: None,
        epoch_len_secs: None,
        heartbeat_interval_secs: None,
        worker_weights: None,
    };
    /// Claim no new work; let in-flight jobs finish.
    pub const DRAINING: RunControl = RunControl {
        paused: false,
        drain: true,
        claim_mode: None,
        epoch_len_secs: None,
        heartbeat_interval_secs: None,
        worker_weights: None,
    };

    /// A worker should claim no new jobs when paused or draining.
    pub fn claims_blocked(&self) -> bool {
        self.paused || self.drain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_claims_when_paused_or_draining() {
        assert!(!RunControl::RUNNING.claims_blocked());
        assert!(RunControl::PAUSED.claims_blocked());
        assert!(RunControl::DRAINING.claims_blocked());
    }

    #[test]
    fn claim_mode_rides_the_control_object_compatibly() {
        // Old control objects (no claim fields) parse with None — worker config decides.
        let old: RunControl = serde_json::from_str(r#"{"paused":false,"drain":false}"#).unwrap();
        assert_eq!(old.claim_mode, None);
        // A campaign flips the whole fleet by writing the mode (+ optional overrides).
        let c: RunControl =
            serde_json::from_str(r#"{"claim_mode":"epoch_sharded","epoch_len_secs":300}"#).unwrap();
        assert_eq!(c.claim_mode, Some(crate::epoch::ClaimMode::EpochSharded));
        assert_eq!(c.epoch_len_secs, Some(300));
        assert_eq!(c.heartbeat_interval_secs, None);
        assert!(!c.claims_blocked());
        // None fields stay OFF the wire, so old workers see exactly the old shape.
        assert_eq!(
            serde_json::to_string(&RunControl::RUNNING).unwrap(),
            r#"{"paused":false,"drain":false}"#
        );
    }

    #[test]
    fn defaults_to_running_and_tolerates_partial_json() {
        // Absent fields default to false (running) — an empty/old control object never blocks work.
        let c: RunControl = serde_json::from_str("{}").unwrap();
        assert_eq!(c, RunControl::RUNNING);
        let p: RunControl = serde_json::from_str(r#"{"paused":true}"#).unwrap();
        assert!(p.claims_blocked());
    }
}
