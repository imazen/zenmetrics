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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunControl {
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub drain: bool,
}

impl RunControl {
    /// Running normally — pull and execute new work.
    pub const RUNNING: RunControl = RunControl {
        paused: false,
        drain: false,
    };
    /// Hard stop — claim nothing until resumed.
    pub const PAUSED: RunControl = RunControl {
        paused: true,
        drain: false,
    };
    /// Claim no new work; let in-flight jobs finish.
    pub const DRAINING: RunControl = RunControl {
        paused: false,
        drain: true,
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
    fn defaults_to_running_and_tolerates_partial_json() {
        // Absent fields default to false (running) — an empty/old control object never blocks work.
        let c: RunControl = serde_json::from_str("{}").unwrap();
        assert_eq!(c, RunControl::RUNNING);
        let p: RunControl = serde_json::from_str(r#"{"paused":true}"#).unwrap();
        assert!(p.claims_blocked());
    }
}
