//! `zencloud-hetzner` — Hetzner Cloud provider for the
//! [`zenfleet_orchestrator::ProviderHandle`] trait.
//!
//! The second clean consumer of the provider-trait extraction (after
//! Salad). Hetzner has no managed job queue and no managed object
//! store, so the provider keeps two pieces alive:
//!
//! 1. **Hetzner Cloud REST client** ([`api::HetznerApi`]) — a tiny
//!    `reqwest`-based wrapper around the four endpoints we need:
//!    `GET /server_types`, `POST /servers`, `GET /servers?label_selector=...`,
//!    `DELETE /servers/{id}`.
//! 2. **Worker bootstrap** — workers don't get a sidecar that POSTs
//!    jobs; instead the cloud-init `user_data` runs the docker worker
//!    image with `WORKER_BACKEND=hetzner` and the worker polls R2 for
//!    chunks (see [`crate::worker_loop`] for the loop spec).
//!
//! Jobs land in R2 at `s3://<bucket>/runs/<run_id>/queue/<chunk_id>.json`
//! and the worker LISTs that prefix, claims one (alphabetic order +
//! existing worker_chunk_start_unix idempotency), processes it via the
//! shared inline pipeline, then DELETEs the queue entry. Duplicate
//! processing is safe — the omni sidecar dedup pattern we validated on
//! vast.ai iter2/3 reconciles it.
//!
//! ## Why a label-selector group, not a project
//!
//! Hetzner's "project" is an organization-level concept; one API token
//! is scoped to ONE project. Inside that project, all servers are
//! peers — there is no nested "container group". We use the
//! `labels={group: <sweep_id>}` field as the equivalent: every server
//! we create carries the label, and the LIST + DELETE paths use
//! `?label_selector=group=<sweep_id>` to scope operations to this
//! sweep without affecting other servers in the project.

#![forbid(unsafe_code)]

pub mod api;
pub mod cloud_init;
pub mod provider;

pub use api::{
    HetznerApi, HetznerLocation, HetznerServer, HetznerServerStatus, HetznerServerType,
    load_token_from_file_or_env,
};
pub use cloud_init::{WorkerBootstrap, build_user_data};
pub use provider::{HetznerProviderConfig, HetznerProviderHandle};
