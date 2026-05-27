//! vast.ai R2 client — now a thin re-export of the shared
//! [`zen_cloud_s3`] client.
//!
//! The s5cmd-backed object-store client + retry helper that used to live
//! here were factored out into the standalone `zen-cloud-s3` crate
//! (spec §1.9 item 4) so SaladCloud and other providers reuse the proven
//! R2 logic instead of duplicating it. R2 IS S3-compatible, so one
//! client serves both.
//!
//! This module re-exports the shared client under the historical
//! `R2Client` name so every internal call site
//! (`super::r2::R2Client` in `chunk` / `claim` / `inline` /
//! `feature_backfill` / `source_features_only` / `mod`) and every test
//! compiles unchanged. Behaviour is byte-identical to the pre-extraction
//! client.
//!
//! The one constructor difference: the shared [`zen_cloud_s3::S3Client`]
//! takes plain endpoint/profile fields, so the vast.ai
//! `WorkerArgs`-aware constructor lives here as [`new_from_args`]. The
//! two in-crate call sites (`worker::mod` + `cloud`) call it directly.

use anyhow::{Context, Result};

use super::WorkerArgs;

/// The shared S3-compatible client, re-exported under its historical
/// vast.ai name. R2 is S3-compatible, so the same client drives it.
pub use zen_cloud_s3::S3Client as R2Client;

/// The shared bounded-backoff retry helper, re-exported.
pub use zen_cloud_s3::with_retry;

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
