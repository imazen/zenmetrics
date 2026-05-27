//! Local `Heartbeat` — a log-only no-op.
//!
//! A heartbeat exists to tell some external monitor "this worker is
//! still alive" (vast.ai writes a file to R2; k8s answers a liveness
//! probe). A single-process local run has no external monitor and no
//! peers racing for its chunks, so there is nothing to signal — the
//! local `Heartbeat` just logs the lifecycle transition at `debug` and
//! returns `Ok`.
//!
//! Keeping it as a real impl (rather than a generic no-op) lets a
//! developer flip on `RUST_LOG=zen_cloud_local=debug` to watch the
//! worker move through `Starting` → `Working` → `Draining` → `Done`
//! while debugging the compute path locally.

use zen_cloud_core::{CloudError, Heartbeat, WorkerId, WorkerStatus};

/// No-op (log-only) heartbeat for the local backend.
#[derive(Default, Clone, Copy, Debug)]
pub struct LocalHeartbeat;

impl Heartbeat for LocalHeartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError> {
        tracing::debug!(worker = %worker, ?status, "local heartbeat (no-op; single-process local run)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_is_always_ok() {
        let hb = LocalHeartbeat;
        let w = WorkerId("localhost".into());
        assert!(hb.beat(&w, WorkerStatus::Starting).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Working { in_flight: 1 }).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Draining).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Done).is_ok());
    }
}
