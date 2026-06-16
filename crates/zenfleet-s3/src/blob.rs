//! `BlobStorage` over any S3-compatible store via the [`S3Client`].
//!
//! This is the `BlobStorage` trait impl relocated verbatim (behaviour-
//! identical) from `zenfleet-vastai`'s `cloud::R2BlobStorage` so every
//! provider — vast.ai (R2), SaladCloud (BYO R2/S3), DigitalOcean
//! (Spaces) — shares one proven impl instead of duplicating the
//! subprocess plumbing (spec §1.9 item 4). vast.ai re-exports it as
//! `R2BlobStorage` so its call sites are unchanged.
//!
//! The core trait surface is synchronous (spec §1.5); the underlying
//! client is async (it parks on `tokio::process::Command`). Each sync
//! trait method drives the async op to completion on a private
//! current-thread runtime — the same bridge the vast.ai `cmd_worker`
//! uses between sync `main` and the async dispatcher.

use zenfleet_cloud::{ArtifactKey, BlobMeta, BlobStorage, CloudError};

use crate::client::S3Client;

/// Build a fresh single-thread tokio runtime for one blocking bridge.
///
/// Cheap (no worker threads) and disposable — used to drive one async
/// object-store op to completion from a sync trait method.
fn block_on<F: std::future::Future>(fut: F) -> Result<F::Output, CloudError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CloudError::Other(format!("build tokio runtime: {e}")))?;
    Ok(rt.block_on(fut))
}

/// `BlobStorage` over an S3-compatible store via the existing s5cmd
/// client.
///
/// Keys are full `s3://bucket/key` URIs (the provider-native locator),
/// matching every tool in the fleet. `head`/`list` are derived from the
/// same `s5cmd ls` the worker already uses.
pub struct S3BlobStorage {
    client: S3Client,
}

impl S3BlobStorage {
    pub fn new(client: S3Client) -> Self {
        Self { client }
    }

    /// Borrow the underlying client (for providers that also need raw
    /// `cat`/`upload`/`fetch_chunks_jsonl` access alongside the trait).
    pub fn client(&self) -> &S3Client {
        &self.client
    }
}

impl BlobStorage for S3BlobStorage {
    fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError> {
        // s5cmd uploads files, not stdin streams — stage to a temp file
        // then `cp`, mirroring the claim writer's temp-file pattern.
        let tmp = std::env::temp_dir().join(format!(
            "zenfleet-s3-put-{}-{}.bin",
            std::process::id(),
            blob_basename(key)
        ));
        std::fs::write(&tmp, bytes).map_err(CloudError::storage)?;
        let r = block_on(self.client.upload(&tmp, key.as_str()))?;
        let _ = std::fs::remove_file(&tmp);
        r.map_err(CloudError::storage)
    }

    fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError> {
        let bytes = block_on(self.client.cat(key.as_str()))?;
        if bytes.is_empty() && !block_on(self.client.exists(key.as_str()))? {
            return Err(CloudError::Storage(format!("object not found: {key}")));
        }
        Ok(bytes)
    }

    fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError> {
        // s5cmd `ls <uri>` prints `<date> <time> <size> <key>` for a
        // single object. We surface existence; ETag is not exposed by
        // the s5cmd ls path. Callers that only need existence use
        // `list` / the client's `exists`.
        let exists = block_on(self.client.exists(key.as_str()))?;
        if !exists {
            return Ok(None);
        }
        // We don't currently parse size out of `ls`; report size 0 + no
        // etag. Callers that only need existence use `list`.
        Ok(Some(BlobMeta {
            size: 0,
            etag: None,
        }))
    }

    fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError> {
        let out = block_on(self.client.ls_keys(prefix))?;
        out.map_err(CloudError::storage)
            .map(|keys| keys.into_iter().map(ArtifactKey).collect())
    }

    fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError> {
        block_on(self.client.rm(key.as_str()))?.map_err(CloudError::storage)
    }
}

/// Sanitise the trailing path segment of a key into a filesystem-safe
/// temp-file basename.
pub(crate) fn blob_basename(key: &ArtifactKey) -> String {
    key.as_str()
        .rsplit('/')
        .next()
        .unwrap_or("blob")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '-' || *c == '_')
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_basename_sanitizes() {
        assert_eq!(
            blob_basename(&ArtifactKey("s3://b/run/omni/abc 123.parquet".into())),
            "abc123.parquet"
        );
    }
}
