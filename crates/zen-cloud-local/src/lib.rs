//! `zen-cloud-local` — the local (no-cloud) provider for the
//! `zen-cloud-core` worker trait layer (spec §1.7 Phase B).
//!
//! This backend runs the full
//! claim → fetch → compute → upload → ack loop entirely on localhost
//! against the filesystem — no cloud SDK, no network, no spend. It has
//! two jobs (spec §1.4 value-props 2 + 3):
//!
//! 1. **Validate the trait abstraction.** If the five
//!    [`zen_cloud_core`] trait shapes are wrong, this is where it shows
//!    up — cheaply, before more cloud providers are built on top.
//! 2. **Debug the worker compute locally.** Point
//!    `zen-sweep-worker --backend local` at a `chunks.jsonl` on disk +
//!    a filesystem mirror dir and exercise the GPU encode+score path on
//!    this box (which has an RTX 5070) with zero cloud cost.
//!
//! ## What this crate provides
//!
//! - [`queue::LocalDirQueue`] — a [`zen_cloud_core::JobQueue`] over a
//!   `chunks.jsonl` file (or a directory of `*.json` chunk files).
//!   `next_chunk` claims the next unclaimed record by renaming it into a
//!   `claimed/` sub-dir (single-process is fine — no atomic-R2-ETag
//!   needed); `ack_chunk` moves it on to `done/` or `failed/`. The chunk
//!   payload is the same `{"chunk_id":…}` JSON line the vast.ai / runpod
//!   workers parse.
//! - [`storage::LocalFsStorage`] — a [`zen_cloud_core::BlobStorage`]
//!   over a local base dir. Resolves `s3://bucket/key`, `file://…`, and
//!   plain relative paths to `<base>/<bucket>/<key>` (or `<base>/<path>`)
//!   so a chunk that references `s3://…` reads/writes a local mirror.
//! - [`host::DotenvCredentials`] — a [`zen_cloud_core::CredentialSource`]
//!   over the process env plus an optional `.env` file. Local runs need
//!   no remote creds, so this just returns whatever is set.
//! - [`host::LocalWorkerHost`] — a [`zen_cloud_core::WorkerHost`]: worker
//!   id from `$WORKER_ID` (else hostname), scratch dir from `$WORKDIR`
//!   (else a temp dir), GPU count via `nvidia-smi`.
//! - [`heartbeat::LocalHeartbeat`] — a log-only no-op (single-process
//!   local run; nothing to signal liveness to).
//!
//! The `compute` closure the worker runs is identical to the vast.ai
//! one — [`zen_cloud_core::run_worker`] is backend-agnostic. Only the
//! thin glue above is local-specific.

#![forbid(unsafe_code)]

pub mod heartbeat;
pub mod host;
pub mod queue;
pub mod storage;

pub use heartbeat::LocalHeartbeat;
pub use host::{DotenvCredentials, LocalWorkerHost};
pub use queue::{LocalDirQueue, LocalQueueConfig, LocalQueueSource};
pub use storage::LocalFsStorage;
