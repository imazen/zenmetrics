//! Thin S3-compatible object-store client. Shells out to `s5cmd` to
//! preserve the vast.ai bash worker's tested behaviour and minimise
//! risk — this is the exact client logic that lived in
//! `zenfleet-vastai`'s `worker::r2` module before the Phase C
//! extraction (spec §1.9 item 4). The only change is the constructor:
//! it now takes plain endpoint/profile fields instead of a vast.ai
//! `WorkerArgs`, so any provider (R2, S3, Spaces, MinIO, SaladCloud's
//! BYO bucket) can build one. vast.ai keeps a `from_worker_args`
//! adapter so its call sites are unchanged.
//!
//! The methods here are all `async` even though the s5cmd subprocess is
//! blocking — tokio's `Command::output().await` parks the calling task
//! on the runtime while waiting for the child, so an in-flight chunk
//! does not starve its peers.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use tokio::process::Command;

/// An S3-compatible object-store client backed by `s5cmd`.
///
/// Works against any S3-compatible endpoint (Cloudflare R2, AWS S3,
/// DigitalOcean Spaces, MinIO). The historical name in
/// `zenfleet-vastai` was `R2Client`; that alias is re-exported from
/// the vast.ai crate so existing call sites compile unchanged.
#[derive(Clone, Debug)]
pub struct S3Client {
    /// `s5cmd` binary on PATH. Configurable for offline tests.
    pub bin: String,
    /// Endpoint URL, e.g. `https://<account>.r2.cloudflarestorage.com`.
    pub endpoint: String,
    /// Named profile in `~/.aws/credentials` (default `r2`).
    pub profile: String,
}

impl S3Client {
    /// Build from explicit endpoint/profile/binary fields.
    ///
    /// `endpoint` is the S3-compatible endpoint URL (for R2,
    /// `https://<account>.r2.cloudflarestorage.com`); `profile` is the
    /// named profile in `~/.aws/credentials`; `bin` is the `s5cmd`
    /// binary name/path on PATH.
    pub fn new(
        bin: impl Into<String>,
        endpoint: impl Into<String>,
        profile: impl Into<String>,
    ) -> Self {
        Self {
            bin: bin.into(),
            endpoint: endpoint.into(),
            profile: profile.into(),
        }
    }

    /// Derive an R2 endpoint URL from a Cloudflare account id.
    ///
    /// Convenience for the common case where a caller has
    /// `$R2_ACCOUNT_ID` rather than a full endpoint URL — mirrors the
    /// bash worker convention.
    pub fn r2_endpoint_for_account(account_id: &str) -> String {
        format!("https://{account_id}.r2.cloudflarestorage.com")
    }

    /// Run `s5cmd ls <uri>` and return whether ANY object matched.
    /// Used for sidecar-exists checks and existing-claim probes. We
    /// treat non-zero exit as "not present" (s5cmd 2.x returns 1
    /// on no-match).
    pub async fn exists(&self, uri: &str) -> bool {
        let out = self.cmd(&["ls", uri]).output().await;
        match out {
            Ok(o) => o.status.success() && !o.stdout.is_empty(),
            Err(_) => false,
        }
    }

    /// Read the body of an object to bytes. Returns the empty vector on
    /// s5cmd failure (caller treats "missing" the same as "empty" for
    /// claim files, which is the safe interpretation under
    /// read-modify-write).
    pub async fn cat(&self, uri: &str) -> Vec<u8> {
        let out = self.cmd(&["cat", uri]).output().await;
        match out {
            Ok(o) if o.status.success() => o.stdout,
            _ => Vec::new(),
        }
    }

    /// Read the body of an object to a UTF-8 string, or empty.
    pub async fn cat_string(&self, uri: &str) -> String {
        let buf = self.cat(uri).await;
        String::from_utf8_lossy(&buf).into_owned()
    }

    /// Upload a local file to an object URI. Returns Err on s5cmd
    /// failure so the caller can decide to retry, skip the chunk, or
    /// fail hard.
    pub async fn upload(&self, local: &Path, uri: &str) -> Result<()> {
        let out = self
            .cmd(&["cp", local.to_str().context("non-utf8 path")?, uri])
            .output()
            .await
            .context("spawn s5cmd cp")?;
        if !out.status.success() {
            return Err(anyhow!(
                "s5cmd cp failed: status={} stderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }

    /// Download an object URI to a local path.
    pub async fn download(&self, uri: &str, local: &Path) -> Result<()> {
        let out = self
            .cmd(&["cp", uri, local.to_str().context("non-utf8 path")?])
            .output()
            .await
            .context("spawn s5cmd cp")?;
        if !out.status.success() {
            return Err(anyhow!(
                "s5cmd cp failed: status={} stderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }

    /// List object keys under `prefix` via `s5cmd ls`. Returns the full
    /// key column of each line (the last whitespace-separated field).
    /// Used by the [`crate::blob::S3BlobStorage`] `BlobStorage::list`
    /// impl.
    pub async fn ls_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let out = self
            .cmd(&["ls", prefix])
            .output()
            .await
            .context("spawn s5cmd ls")?;
        if !out.status.success() {
            // s5cmd 2.x returns 1 on no-match; treat as empty rather
            // than an error so `list` of an absent prefix is `[]`.
            return Ok(Vec::new());
        }
        let stdout = String::from_utf8_lossy(&out.stdout);
        let keys = stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            // `s5cmd ls` prints `<date> <time> <size> <key>`; the key is
            // the final whitespace-separated field.
            .filter_map(|l| l.split_whitespace().next_back().map(|s| s.to_owned()))
            .collect();
        Ok(keys)
    }

    /// Remove an object via `s5cmd rm`. Used by the
    /// [`crate::blob::S3BlobStorage`] `BlobStorage::delete` impl.
    pub async fn rm(&self, uri: &str) -> Result<()> {
        let out = self
            .cmd(&["rm", uri])
            .output()
            .await
            .context("spawn s5cmd rm")?;
        if !out.status.success() {
            return Err(anyhow!(
                "s5cmd rm failed: status={} stderr={}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        Ok(())
    }

    /// Fetch a JSON-lines manifest (`chunks.jsonl`) and split into one
    /// record per line. Each returned string is the raw JSON for one
    /// chunk; per-chunk parsing happens lazily downstream.
    pub async fn fetch_chunks_jsonl(&self, uri: &str) -> Result<Vec<String>> {
        let body = self.cat(uri).await;
        if body.is_empty() {
            return Err(anyhow!("empty chunks.jsonl at {uri}"));
        }
        let text = String::from_utf8(body).context("chunks.jsonl is not UTF-8")?;
        Ok(text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.to_owned())
            .collect())
    }

    /// Build a `tokio::process::Command` with the standard s5cmd flags
    /// pre-set. Callers append the subcommand + args.
    fn cmd(&self, args: &[&str]) -> Command {
        let mut c = Command::new(&self.bin);
        c.arg("--endpoint-url")
            .arg(&self.endpoint)
            .arg("--profile")
            .arg(&self.profile);
        for a in args {
            c.arg(a);
        }
        // s5cmd is well-behaved with stdout/stderr piped, but kill on
        // tokio drop to avoid leaking child processes if a task is
        // cancelled mid-call.
        c.kill_on_drop(true);
        c
    }
}

/// Helper to retry a flaky object-store op with bounded exponential
/// backoff. Commodity-GPU networks are often noisy — a single 503
/// should not fail a multi-minute chunk processing job.
pub async fn with_retry<F, Fut, T>(name: &str, max_attempts: u32, mut op: F) -> Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut delay = Duration::from_millis(500);
    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=max_attempts {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                tracing::warn!("{name}: attempt {attempt}/{max_attempts} failed: {e:#}");
                last_err = Some(e);
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(Duration::from_secs(30));
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow!("{name}: all retries failed")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r2_endpoint_for_account_formats() {
        assert_eq!(
            S3Client::r2_endpoint_for_account("abc123"),
            "https://abc123.r2.cloudflarestorage.com"
        );
    }

    #[test]
    fn new_sets_fields() {
        let c = S3Client::new("s5cmd", "https://example.com", "r2");
        assert_eq!(c.bin, "s5cmd");
        assert_eq!(c.endpoint, "https://example.com");
        assert_eq!(c.profile, "r2");
    }
}
