//! `zen-cloud-vastai` — the vast.ai provider for the `zen-cloud-core`
//! trait layer (renamed from `vastai-fleet`, spec §1.7 Phase A).
//!
//! This crate is the live workhorse for zenmetrics backfill sweeps. It
//! carries two things:
//!
//! 1. **The proven worker + fleet-management code**, unchanged from
//!    `vastai-fleet`: the tolerant `vastai` JSON [`parse`]r, the async
//!    [`worker`] dispatch loop with adaptive concurrency + the R2
//!    token-race claim, and the `self-destroy` / `status` / `destroy`
//!    / `watch` operator CLI (in the `vastai-fleet` binary). This is
//!    the byte-identical production path; Phase A does NOT rewrite it.
//!
//! 2. **Implementations of the [`zen_cloud_core`] traits** ([`cloud`]
//!    module) that wrap (1) behind the cloud-agnostic surface:
//!    - [`cloud::R2BlobStorage`] — `BlobStorage` over the s5cmd-backed
//!      [`worker::r2::R2Client`]. As of Phase C this is a re-export of
//!      the shared [`zen_cloud_s3::S3BlobStorage`] (R2 is
//!      S3-compatible, so one impl serves vast.ai + SaladCloud + DO +
//!      AWS — spec §1.9 item 4); the alias preserves the historical
//!      name for every call site.
//!    - [`cloud::ProcEnvironCredentials`] — `CredentialSource` reading
//!      vast.ai's `/proc/1/environ`.
//!    - [`cloud::VastaiWorkerHost`] — `WorkerHost` over the existing
//!      hostname / scratch-dir / `nvidia-smi` introspection.
//!    - [`cloud::R2Heartbeat`] — `Heartbeat` writing a liveness file to
//!      R2.
//!
//! The new `zen-sweep-worker` binary depends on this crate and dispatches
//! `--backend vastai` to [`worker::cmd_worker`] — the SAME code path the
//! `vastai-fleet` binary runs, so output is byte-identical.

// `deny` rather than `forbid`: the crate is unsafe-free EXCEPT the
// pre-existing `hydrate_pid1_env` env-hydration call (a documented,
// single-threaded `std::env::set_var` that vast.ai pid-1 credential
// copying requires). `deny` lets that one proven call carry a scoped
// `#[allow(unsafe_code)]` with its SAFETY note instead of being
// rewritten in a no-behaviour-change carve. Everything else is gated.
#![deny(unsafe_code)]

pub mod parse;

#[cfg(feature = "worker")]
pub mod worker;

#[cfg(feature = "worker")]
pub mod cloud;
