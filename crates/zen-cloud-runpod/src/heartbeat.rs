//! RunPod `Heartbeat` — R2 liveness file (default) or a thin no-op.
//!
//! RunPod's own dashboard + API (`GET /pods/{podId}`) report pod
//! liveness natively, so a heartbeat is not strictly required the way it
//! is on a bring-your-own-monitoring fleet. But the zenmetrics fleet
//! tooling (the launcher's `watch` view) reads per-worker R2 heartbeat
//! files to show progress across a mixed vast.ai + RunPod fleet, so the
//! default RunPod `Heartbeat` writes the SAME R2 liveness object the
//! vast.ai backend does — reusing the shared [`zen_cloud_s3`]
//! `BlobStorage` (no second R2 writer). Operators that rely solely on
//! the RunPod dashboard can use [`NoopHeartbeat`] instead.

use zen_cloud_core::{ArtifactKey, BlobStorage, CloudError, Heartbeat, WorkerId, WorkerStatus};
use zen_cloud_s3::S3BlobStorage;

/// `Heartbeat` that writes a small liveness object to R2 under a
/// per-run heartbeat prefix — fleet-monitoring parity with vast.ai.
pub struct R2Heartbeat {
    storage: S3BlobStorage,
    prefix: String,
}

impl R2Heartbeat {
    /// `prefix` is an `s3://bucket/<run>/heartbeats/`-style location; the
    /// per-worker object is `<prefix><worker>.beat`.
    pub fn new(storage: S3BlobStorage, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        if !prefix.ends_with('/') {
            prefix.push('/');
        }
        Self { storage, prefix }
    }

    /// The per-worker heartbeat object key under the configured prefix.
    fn key(&self, worker: &WorkerId) -> ArtifactKey {
        ArtifactKey(format!("{}{}.beat", self.prefix, worker))
    }
}

impl Heartbeat for R2Heartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let body = format!("{worker}\t{now}\t{status:?}\n");
        self.storage.put(&self.key(worker), body.as_bytes())
    }
}

/// Log-only no-op heartbeat — for operators that rely on RunPod's own
/// dashboard / `GET /pods/{podId}` for liveness and want no R2 writes.
#[derive(Default, Clone, Copy, Debug)]
pub struct NoopHeartbeat;

impl Heartbeat for NoopHeartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError> {
        tracing::debug!(worker = %worker, ?status, "runpod heartbeat (no-op; runpod tracks pod liveness)");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_beat_is_always_ok() {
        let hb = NoopHeartbeat;
        let w = WorkerId("pod-abc".into());
        assert!(hb.beat(&w, WorkerStatus::Starting).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Working { in_flight: 1 }).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Draining).is_ok());
        assert!(hb.beat(&w, WorkerStatus::Done).is_ok());
    }

    #[test]
    fn r2_heartbeat_key_normalises_prefix() {
        let client = zen_cloud_s3::S3Client::new("s5cmd", "https://e.example.com", "r2");
        let storage = S3BlobStorage::new(client);
        // Prefix without trailing slash gets one appended.
        let hb = R2Heartbeat::new(storage, "s3://bucket/run/heartbeats");
        let key = hb.key(&WorkerId("pod-7".into()));
        assert_eq!(key.as_str(), "s3://bucket/run/heartbeats/pod-7.beat");
    }
}
