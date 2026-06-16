//! SaladCloud `Heartbeat` — a thin no-op status reporter.
//!
//! Per spec §1.9 item 5: SaladCloud manages instance liveness natively
//! (the platform health-checks the container + the IMDS `/v1/status`
//! readiness signal), and the `salad-http-job-queue-worker` sidecar
//! handles per-job acknowledgement (`CompleteJob` / `RejectJob`) over
//! its gRPC stream to the managed queue. There is no R2 heartbeat file
//! to write the way the vast.ai backend does, so the Salad `Heartbeat`
//! impl just logs the status transition at `debug` and returns `Ok`.
//!
//! Keeping it as a real impl (rather than reusing a generic no-op) lets
//! operators flip on `RUST_LOG=zenfleet_salad=debug` to watch the
//! worker's lifecycle (`Starting` → `Working` → `Draining` → `Done`)
//! without standing up any external heartbeat sink.

use zenfleet_cloud::{CloudError, Heartbeat, WorkerId, WorkerStatus};

/// No-op (log-only) heartbeat for the Salad backend.
#[derive(Default, Clone, Copy, Debug)]
pub struct SaladHeartbeat;

impl Heartbeat for SaladHeartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError> {
        tracing::debug!(worker = %worker, ?status, "salad heartbeat (no-op; salad manages liveness)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beat_is_always_ok() {
        let hb = SaladHeartbeat;
        let w = WorkerId("machine-abc".into());
        assert!(hb.beat(&w, WorkerStatus::Starting).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Working { in_flight: 1 }).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Draining).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Done).is_ok());
    }
}
