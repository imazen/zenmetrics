//! `zen-cloud-s3` — the shared S3-compatible storage helper for the
//! `zen-cloud-core` worker trait layer.
//!
//! R2 IS S3-compatible, so one client + one `BlobStorage` impl serve
//! every provider that brings its own object store: vast.ai (Cloudflare
//! R2), SaladCloud (BYO R2/S3 — Salad has no native object store),
//! DigitalOcean (Spaces), AWS (S3), or a self-hosted MinIO. This crate
//! is the single home for that logic so providers reuse it instead of
//! each carrying a copy (spec §1.9 item 4).
//!
//! It carries two things:
//!
//! 1. [`S3Client`] — a thin `s5cmd`-backed object-store client
//!    (`exists` / `cat` / `upload` / `download` / `ls_keys` / `rm` /
//!    `fetch_chunks_jsonl`), relocated behaviour-identical from
//!    `zen-cloud-vastai`'s `worker::r2::R2Client`. The only change is
//!    the constructor, which now takes plain fields so any provider can
//!    build one without a vast.ai `WorkerArgs`.
//! 2. [`S3BlobStorage`] — the [`zen_cloud_core::BlobStorage`] impl over
//!    [`S3Client`], relocated from `zen-cloud-vastai`'s
//!    `cloud::R2BlobStorage`.
//!
//! `zen-cloud-vastai` re-exports both under their historical names
//! (`R2Client`, `R2BlobStorage`) so its 25 unit + 7 CLI tests and every
//! internal call site compile + pass unchanged.

#![forbid(unsafe_code)]

pub mod blob;
pub mod client;

pub use blob::S3BlobStorage;
pub use client::{S3Client, with_retry};
