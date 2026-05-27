//! Core value types shared across every cloud backend.
//!
//! These are deliberately plain data — no provider coupling, no async,
//! no IO. A provider crate (`zen-cloud-vastai`, `zen-cloud-local`, …)
//! maps its own internal representation to/from these at the trait
//! boundary, and `zen-sweep-worker` only ever sees these.

use std::collections::HashMap;

/// Stable identifier for one work unit. For vast.ai this is the
/// `chunk_id` field from `chunks.jsonl`; for a push-based backend
/// (GCP Batch / k8s) it is whatever the controller assigned.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ChunkId(pub String);

impl ChunkId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ChunkId {
    fn from(s: String) -> Self {
        ChunkId(s)
    }
}

impl From<&str> for ChunkId {
    fn from(s: &str) -> Self {
        ChunkId(s.to_owned())
    }
}

/// One unit of work handed to the `compute` closure. `id` is the stable
/// identifier; `payload` carries the backend-opaque job description
/// (for vast.ai: the raw `chunks.jsonl` line that the existing chunk
/// processor re-parses). Keeping the payload as a string preserves the
/// "parse lazily in the processor" contract the vastai worker relies on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub id: ChunkId,
    /// Backend-opaque job description. The `compute` closure interprets
    /// it; the core loop never inspects it.
    pub payload: String,
}

/// Terminal outcome of attempting one chunk. Mirrors the
/// claim-aware control flow the vastai worker already implements:
/// a chunk can complete, be skipped (already done / held by a peer /
/// race lost), fail retryably, or fail terminally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChunkOutcome {
    /// Work completed and artifacts were uploaded.
    Done,
    /// Nothing to do — the output already exists, a peer holds a fresh
    /// claim, or we lost the claim race. Not an error; the loop moves on.
    Skipped { reason: String },
    /// Transient failure (network blip, R2 503). The queue may hand the
    /// chunk out again later.
    Retryable { error: String },
    /// Terminal failure for this chunk. Logged + counted; the loop
    /// proceeds to the next chunk (one bad chunk never kills the box).
    Failed { error: String },
}

/// Storage key for a blob (object-storage object / file). The string is
/// the provider-native locator: an `s3://bucket/key` URI for R2/GCS/S3,
/// or a relative path for the local filesystem backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArtifactKey(pub String);

impl ArtifactKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ArtifactKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for ArtifactKey {
    fn from(s: String) -> Self {
        ArtifactKey(s)
    }
}

impl From<&str> for ArtifactKey {
    fn from(s: &str) -> Self {
        ArtifactKey(s.to_owned())
    }
}

/// Metadata returned by `BlobStorage::head`. Size + optional ETag is
/// the common subset across R2 / GCS / S3 / local FS; the ETag is what
/// the vast.ai atomic claim relies on, so it is first-class here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlobMeta {
    /// Object size in bytes.
    pub size: u64,
    /// Provider ETag, when available. For the vast.ai conditional /
    /// read-back claim this is load-bearing; local FS leaves it `None`.
    pub etag: Option<String>,
}

/// Identifier for one worker process. For vast.ai this is the box
/// hostname (or `$WORKER_ID`); it distinguishes peers racing for the
/// same chunk and scopes claim tokens.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WorkerId(pub String);

impl WorkerId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for WorkerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for WorkerId {
    fn from(s: String) -> Self {
        WorkerId(s)
    }
}

impl From<&str> for WorkerId {
    fn from(s: &str) -> Self {
        WorkerId(s.to_owned())
    }
}

/// Coarse liveness status reported via `Heartbeat::beat`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerStatus {
    /// Worker booted and is about to start the dispatch loop.
    Starting,
    /// Worker is actively processing chunks. `in_flight` is the current
    /// in-flight count (the AIMD-controlled concurrency on vast.ai).
    Working { in_flight: usize },
    /// Worker is draining in-flight work and about to exit.
    Draining,
    /// Worker finished cleanly.
    Done,
}

/// Summary returned by the generic `run_worker` loop. Lets the caller
/// log / report without re-deriving the counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerSummary {
    /// Chunks dispatched to the `compute` closure (includes skips —
    /// each consumed a dispatch slot).
    pub dispatched: usize,
    /// Chunks that completed with `ChunkOutcome::Done`.
    pub done: usize,
    /// Chunks skipped (already done / held / race lost).
    pub skipped: usize,
    /// Chunks that failed (terminal or retryable).
    pub failed: usize,
}

/// Credential bundle resolved by a `CredentialSource`. Provider-native
/// `KEY=VALUE` pairs (R2 access keys, GCS service-account, …) the
/// downstream tooling (s5cmd, gsutil) reads from the environment.
pub type Credentials = HashMap<String, String>;
