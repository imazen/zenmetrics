//! `zenfleet-cloud` — the cloud-agnostic worker trait layer.
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
//! (It does carry a thin `reqwest` HTTP dep for [`r2creds`] — the shared,
//! provider-agnostic Cloudflare R2 scoped-credential minter every fleet
//! launcher reuses — but no cloud SDK.)
//!
//! Provider crates implement the traits:
//! - `zenfleet-vastai` — vast.ai API + `/proc/1/environ` + Cloudflare R2
//! - `zenfleet-local`  — localhost + filesystem + sqlite (Phase B)
//! - `zen-cloud-gcp` / `zen-cloud-do` — from coefficient (Phase C)
//!
//! and `zenfleet-sweep` selects one at runtime via cargo features +
//! a `--backend` flag (spec §1.6 decision 4).

#![forbid(unsafe_code)]

mod error;
pub mod r2creds;
mod run;
mod traits;
mod types;

pub use error::CloudError;
pub use r2creds::{
    DEFAULT_TTL_SECONDS, MAX_TTL_SECONDS, MIN_TTL_SECONDS, Permission, ScopedR2Cred,
    mint_scoped_r2_cred, mint_scoped_r2_cred_with,
};
pub use run::run_worker;
pub use traits::{BlobStorage, CredentialSource, Heartbeat, JobQueue, WorkerHost};
pub use types::{
    ArtifactKey, BlobMeta, Chunk, ChunkId, ChunkOutcome, Credentials, WorkerId, WorkerStatus,
    WorkerSummary,
};
