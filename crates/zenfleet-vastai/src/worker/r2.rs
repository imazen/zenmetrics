//! vast.ai R2 client — a thin re-export of the shared S3-compatible
//! client in [`super::s3_client`].
//!
//! The s5cmd-backed object-store client + retry helper live in
//! `worker::s3_client` (folded in from the former `zenfleet-s3` crate,
//! 2026-06-26; R2 IS S3-compatible, so one client serves vast.ai + any
//! BYO bucket).
//!
//! This module re-exports the shared client under the historical
//! `R2Client` name so every internal call site
//! (`super::r2::R2Client` in `chunk` / `claim` / `inline` /
//! `feature_backfill` / `source_features_only` / `mod`) and every test
//! compiles unchanged. Behaviour is byte-identical to the pre-extraction
//! client.
//!
//! The one constructor difference: the shared `S3Client` takes plain
//! endpoint/profile fields, so the vast.ai `WorkerArgs`-aware
//! constructor lives here as [`new_from_args`]. The two in-crate call
//! sites (`worker::mod` + `cloud`) call it directly.

use anyhow::{Context, Result};

use super::WorkerArgs;

/// The shared S3-compatible client, re-exported under its historical
/// vast.ai name. R2 is S3-compatible, so the same client drives it.
pub use super::s3_client::S3Client as R2Client;

/// The shared bounded-backoff retry helper, re-exported.
pub use super::s3_client::with_retry;

/// Build an [`R2Client`] from vast.ai worker CLI args.
///
/// Derives the endpoint from `$R2_ACCOUNT_ID` when the explicit
/// `--r2-endpoint` flag is unset — matches the bash convention so
/// operators can run the Rust worker with the same env they fed the bash
/// one. This is the constructor that lived on the old in-crate
/// `R2Client`; it now adapts `WorkerArgs` to the shared client's
/// field-based constructor.
pub fn new_from_args(args: &WorkerArgs) -> Result<R2Client> {
    let endpoint = if let Some(ep) = &args.r2_endpoint {
        ep.clone()
    } else {
        let acct = std::env::var("R2_ACCOUNT_ID")
            .context("R2_ACCOUNT_ID env not set and --r2-endpoint not passed")?;
        R2Client::r2_endpoint_for_account(&acct)
    };
    Ok(R2Client::new(
        args.s5cmd_bin.clone(),
        endpoint,
        args.s5cmd_profile.clone(),
    ))
}
