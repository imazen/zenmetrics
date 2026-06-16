//! The one error type the trait surface speaks. Provider crates map
//! their native errors (anyhow, aws-sdk, io::Error) into this so the
//! generic worker loop has a single error model.

/// Errors surfaced across the cloud trait boundary.
///
/// Deliberately coarse: the worker loop only needs to distinguish
/// "this op failed, here is why" from "the queue is drained". Provider
/// crates attach detail in the message string.
#[derive(Debug, thiserror::Error)]
pub enum CloudError {
    /// A job-queue operation failed (claim, ack, fetch manifest).
    #[error("job queue: {0}")]
    Queue(String),

    /// A blob-storage operation failed (put/get/head/list/delete).
    #[error("blob storage: {0}")]
    Storage(String),

    /// A heartbeat write failed. Usually non-fatal; the caller decides.
    #[error("heartbeat: {0}")]
    Heartbeat(String),

    /// Credential resolution failed (missing env, unreadable metadata).
    #[error("credentials: {0}")]
    Credentials(String),

    /// Host introspection failed (no scratch dir, gpu query error).
    #[error("worker host: {0}")]
    Host(String),

    /// The `compute` closure returned an error the loop couldn't map to
    /// a `ChunkOutcome`. Distinct from a `ChunkOutcome::Failed`, which
    /// is an *expected* per-chunk failure that the loop counts + skips.
    #[error("compute: {0}")]
    Compute(String),

    /// Anything else a provider needs to surface.
    #[error("{0}")]
    Other(String),
}

impl CloudError {
    /// Convenience for provider impls wrapping a `Display`able error as
    /// a storage failure.
    pub fn storage(e: impl std::fmt::Display) -> Self {
        CloudError::Storage(e.to_string())
    }

    /// Convenience for wrapping a `Display`able error as a queue failure.
    pub fn queue(e: impl std::fmt::Display) -> Self {
        CloudError::Queue(e.to_string())
    }
}
