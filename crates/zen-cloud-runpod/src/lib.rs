//! `zen-cloud-runpod` — the RunPod provider for the `zen-cloud-core`
//! worker trait layer (spec §1.10).
//!
//! RunPod runs containers on rented GPU pods (commodity COMMUNITY-cloud
//! GPUs or datacenter SECURE-cloud GPUs) — another vast.ai / SaladCloud
//! alternative for the same workload class. RunPod offers two product
//! modes; this crate implements the **Pods (pull)** path, which is
//! structurally identical to vast.ai:
//!
//! - **Pods (pull) — implemented here.** A rented GPU pod boots a
//!   generic container with credentials + sweep wiring in its
//!   environment; the worker *pulls* chunks from R2 via the shared
//!   atomic token-race claim. Zero protocol risk: the claim *algorithm*
//!   is reused verbatim from [`zen_cloud_vastai`].
//! - **Serverless (push) — documented follow-on, NOT implemented.** See
//!   `RUNPOD.md` for how a future serverless `JobQueue` would speak the
//!   RunPod serverless job-poll contract (handler-SDK shim vs native).
//!
//! ## What this crate provides
//!
//! - [`queue::RunpodChunkQueue`] — a [`zen_cloud_core::JobQueue`] over an
//!   R2 `chunks.jsonl` manifest with the shared R2-ETag atomic claim
//!   (reuses [`zen_cloud_vastai::worker::claim::try_claim`] — no third
//!   claim impl). `ack_chunk` is a no-op (the claim + sidecar are the
//!   durable state).
//! - [`host::RunpodEnvCredentials`] — a [`zen_cloud_core::CredentialSource`]
//!   over the plain pod environment (RunPod injects the BYO `R2_*` /
//!   `AWS_*` creds + sweep wiring + reserved `RUNPOD_POD_ID` as ordinary
//!   env vars — no IMDS, no pid-1 trick).
//! - [`host::RunpodWorkerHost`] — a [`zen_cloud_core::WorkerHost`] over
//!   `RUNPOD_POD_ID` / `RUNPOD_GPU_COUNT` / `nvidia-smi`.
//! - [`storage::RunpodBlobStorage`] — re-exported from the shared
//!   `zen-cloud-s3` crate (RunPod has no native object store; workers
//!   bring their own R2/S3). No second S3 client (spec §1.10).
//! - [`heartbeat::R2Heartbeat`] / [`heartbeat::NoopHeartbeat`] — an R2
//!   liveness file for cross-fleet monitoring parity with vast.ai, or a
//!   thin no-op (RunPod's dashboard/API tracks pod liveness natively).
//! - [`launch`] — the launcher-side RunPod provider: a `reqwest`-based
//!   client for the v1 REST API (create / get / stop / terminate a pod).
//!   Runs on the operator workstation, NOT the worker.
//!
//! The `compute` closure the worker runs is identical to the vast.ai
//! one — [`zen_cloud_core::run_worker`] is backend-agnostic. Only the
//! thin glue above is RunPod-specific.

#![forbid(unsafe_code)]

pub mod heartbeat;
pub mod host;
pub mod launch;
pub mod queue;
pub mod storage;

pub use heartbeat::{NoopHeartbeat, R2Heartbeat};
pub use host::{RunpodEnvCredentials, RunpodWorkerHost};
pub use launch::{Pod, PodCreateInput, RunpodApi};
pub use queue::{RunpodChunkQueue, RunpodQueueConfig};
pub use storage::{RunpodBlobStorage, blob_storage, blob_storage_from_credentials};
