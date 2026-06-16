//! `zenfleet-salad` — the SaladCloud provider for the `zenfleet-cloud`
//! worker trait layer (spec §1.9).
//!
//! SaladCloud runs containers on distributed consumer GPUs at low $/hr —
//! a commodity-GPU alternative to vast.ai for the same workload class.
//! It is the first real exercise of the "add a provider" path and it
//! validates the push-vs-pull `JobQueue` abstraction: where vast.ai is
//! bring-your-own-queue + *pull* (the worker claims chunks from R2 via
//! atomic ETag writes), SaladCloud is managed-queue + *push* (a sidecar
//! receives jobs from Salad's managed queue and forwards them into the
//! app).
//!
//! ## What this crate provides
//!
//! - [`queue::SaladJobQueue`] — a [`zenfleet_cloud::JobQueue`] backed by
//!   a tiny local HTTP server that the baked-in
//!   `salad-http-job-queue-worker` sidecar POSTs jobs to. `next_chunk`
//!   blocks on the next POST; `ack_chunk` turns the outcome into the
//!   HTTP response the sidecar reads back as the job result. (See
//!   `queue.rs` for why the app side is HTTP, not gRPC — the gRPC
//!   contract is internal to the sidecar.)
//! - [`host::SaladEnvCredentials`] — a [`zenfleet_cloud::CredentialSource`]
//!   over the container-group env (BYO R2/S3 creds + the reserved
//!   `SALAD_MACHINE_ID` / `SALAD_CONTAINER_GROUP_ID`).
//! - [`host::SaladWorkerHost`] — a [`zenfleet_cloud::WorkerHost`] over the
//!   Salad instance identity (machine id, scratch dir, `nvidia-smi`).
//! - [`host::SaladImds`] — a minimal Salad IMDS client (`/v1/status`,
//!   `/v1/token`) for readiness + the workload identity token.
//! - [`storage::SaladBlobStorage`] — re-exported from the shared
//!   `zenfleet-s3` crate (Salad has no native object store; workers
//!   bring their own R2/S3 — spec §1.9 item 4). No second S3 client.
//! - [`heartbeat::SaladHeartbeat`] — a no-op (log-only) heartbeat;
//!   Salad manages liveness + the sidecar handles per-job acks
//!   (spec §1.9 item 5).
//! - [`launch`] — the launcher-side Salad provider: a `reqwest`-based
//!   client for the public API (resolve a GPU class, create a managed
//!   queue, create a container group with the queue attached, push job
//!   chunks). This runs on the operator workstation, NOT the worker.
//!
//! The `compute` closure the worker runs is identical to the vast.ai
//! one — `run_worker` is backend-agnostic. Only the thin glue above is
//! Salad-specific.

#![forbid(unsafe_code)]

pub mod heartbeat;
pub mod host;
pub mod launch;
pub mod queue;
pub mod storage;

// Operator-side modules — only compiled when the launcher binary is
// being built. They depend on sha2 / hmac via `--features launcher`.
#[cfg(feature = "launcher")]
pub mod launcher_support;
#[cfg(feature = "launcher")]
pub mod provider;
#[cfg(feature = "launcher")]
pub mod r2_ops;

pub use heartbeat::SaladHeartbeat;
pub use host::{SaladEnvCredentials, SaladImds, SaladWorkerHost};
pub use queue::{SaladJobQueue, SaladQueueConfig};
pub use storage::{SaladBlobStorage, blob_storage, blob_storage_from_credentials};

#[cfg(feature = "launcher")]
pub use provider::{SaladProviderConfig, SaladProviderHandle};
#[cfg(feature = "launcher")]
pub use r2_ops::{R2OperatorImpl, short_ts, split_s3_uri};
