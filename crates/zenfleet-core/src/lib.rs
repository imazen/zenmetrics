#![forbid(unsafe_code)]
//! # zenfleet-core
//!
//! Phase-0 keystone for the zen job system (see `docs/JOB_SYSTEM_GOAL.md` and
//! `docs/JOB_SYSTEM_SHIFT_SPEC_2026-05-29.md`). Defines the primitives every later phase rests on:
//!
//! - **Content addressing** ([`content`]): blobs are named by SHA-256 of their bytes. Gives free
//!   dedup, the GC handle (goal G), and idempotent enqueue (goal A).
//! - **Job taxonomy** ([`job`]): [`JobKind`] + per-kind [`JobProfile`] (resource class, batching
//!   group key, GC regenerability). The encode cost asymmetry (JPEG cheap, AVIF expensive) and
//!   metric reference-locality live here as *data*, so the engine never special-cases a kind.
//! - **Identity** ([`ids`]): the retained [`CellId`] training tuple + the content-addressed
//!   [`JobId`] that makes duplicate work structurally impossible (goal I).
//! - **Outcomes** ([`status`]): [`JobStatus`] + [`ErrorClass`] — failures are rows, not gaps (goal B),
//!   and drive retry-vs-poison (goal F).
//!
//! This crate intentionally depends on nothing heavy (no GPU, no codec crates): it is the shared
//! vocabulary the queue, reconciler, GC, and dashboard all speak.

pub mod catalog;
pub mod content;
pub mod control;
pub mod cost;
pub mod gc;
pub mod ids;
pub mod job;
pub mod lease;
pub mod ledger;
pub mod reconcile;
pub mod schedule;
pub mod status;

pub use catalog::{CatalogEntry, SemanticId};
pub use content::{BlobRef, ContentError, Sha256Hex, blob_key, sha256};
pub use control::RunControl;
pub use cost::{FleetCost, WorkerReport, aggregate, cost_per_1000_by_tier, over_budget};
pub use gc::{BlobIndexEntry, GcPlan, GcVerdict, Tombstone, gc_plan, lru_cap_evict, verdict};
pub use ids::{CellId, JobId};
pub use job::{GroupBy, JobKind, JobProfile, Regenerability, ResourceClass, worker_serves};
pub use lease::{Lease, recommended_ttl_secs};
pub use ledger::{DesiredJob, LedgerRow, LedgerView, ResourceHint};
pub use reconcile::{ReconcilePlan, RetryPolicy, reconcile};
pub use status::{ErrorClass, JobStatus};
