//! `zen-cloud-core` — the cloud-agnostic worker trait layer.
//!
//! The binary deployed on a compute node must NOT know which cloud it
//! runs on (spec §1.1). This crate is the code half of that principle:
//! five traits ([`JobQueue`], [`BlobStorage`], [`Heartbeat`],
//! [`CredentialSource`], [`WorkerHost`]), the plain value types they
//! speak ([`Chunk`], [`ArtifactKey`], [`WorkerSummary`], …), and a
//! generic [`run_worker`] job loop parameterized over them.
//!
//! It has ZERO gpu / cloud-SDK / parquet dependencies, so any consumer
//! (coefficient, jxl-encoder, zensim picker training) can depend on the
//! abstraction without dragging GPU infrastructure (spec §1.6 dec. 5).
//!
//! Provider crates implement the traits:
//! - `zen-cloud-vastai` — vast.ai API + `/proc/1/environ` + Cloudflare R2
//! - `zen-cloud-local`  — localhost + filesystem + sqlite (Phase B)
//! - `zen-cloud-gcp` / `zen-cloud-do` — from coefficient (Phase C)
//!
//! and `zen-sweep-worker` selects one at runtime via cargo features +
//! a `--backend` flag (spec §1.6 decision 4).

#![forbid(unsafe_code)]

mod error;
mod run;
mod traits;
mod types;

pub use error::CloudError;
pub use run::run_worker;
pub use traits::{BlobStorage, CredentialSource, Heartbeat, JobQueue, WorkerHost};
pub use types::{
    ArtifactKey, BlobMeta, Chunk, ChunkId, ChunkOutcome, Credentials, WorkerId, WorkerStatus,
    WorkerSummary,
};
