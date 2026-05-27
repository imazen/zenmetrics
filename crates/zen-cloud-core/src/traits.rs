//! The cloud-agnostic trait surface.
//!
//! These five traits are the entire coupling point between the generic
//! worker and any cloud provider. A provider crate implements them; the
//! worker is generic over them. Signatures track spec §1.5 of
//! `docs/ZEN_CLOUD_AND_CONSOLIDATION_SPEC_2026-05-26.md`.

use crate::error::CloudError;
use crate::types::{
    ArtifactKey, BlobMeta, Chunk, ChunkId, ChunkOutcome, Credentials, WorkerId, WorkerStatus,
};

/// Pull-or-push job source.
///
/// vast.ai is *pull* (an atomic R2-ETag claim happens inside
/// `next_chunk`); GCP Batch / k8s are *push* (the controller assigns a
/// chunk and `next_chunk` just returns it). Both are expressible with
/// this shape — spec §1.6 decision 1.
pub trait JobQueue {
    /// Return the next chunk to work, or `None` when the queue is
    /// drained. Pull impls perform the atomic claim here; push impls
    /// return the pre-assigned chunk.
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError>;

    /// Acknowledge a chunk's terminal outcome (done / skipped / failed
    /// / retryable).
    fn ack_chunk(&mut self, id: &ChunkId, outcome: ChunkOutcome) -> Result<(), CloudError>;
}

/// Object storage. R2 / GCS / S3 / Spaces / local FS all fit this
/// shape — spec §1.6 decision 2.
pub trait BlobStorage {
    fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError>;
    fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError>;
    fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError>;
    fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError>;
    fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError>;
}

/// Liveness signal. vast: a heartbeat file in R2; k8s: a liveness
/// probe. Best-effort by contract — a failed beat does not abort work.
pub trait Heartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError>;
}

/// Credential resolution. vast: `/proc/1/environ`; gcp: metadata
/// server; do: env; local: dotenv — spec §1.6 decision 3.
pub trait CredentialSource {
    fn resolve(&self) -> Result<Credentials, CloudError>;
}

/// Host-environment introspection (node id, scratch dir, GPU count).
pub trait WorkerHost {
    fn worker_id(&self) -> WorkerId;
    fn scratch_dir(&self) -> std::path::PathBuf;
    fn gpu_count(&self) -> usize;
}
