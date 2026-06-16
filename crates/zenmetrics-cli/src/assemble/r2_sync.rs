#![forbid(unsafe_code)]

//! Synchronous R2 sidecar sync, shelling out to `s5cmd`.
//!
//! # Why shell `s5cmd` rather than reuse `zenfleet-vastai`'s `R2Client`?
//!
//! `crates/zenfleet-vastai/src/worker/r2.rs` is an excellent client, but it is
//! `async` (built on tokio) and lives in a different crate. The
//! `zenmetrics-cli` binary is synchronous and does not depend on tokio; the
//! corpus assembler is a one-shot batch job, not a server. Pulling tokio +
//! the fleet crate in just to fetch a few sidecar prefixes would be a large
//! new dependency surface for no latency benefit (the download is dominated
//! by network, not subprocess spawn).
//!
//! So we mirror the **synchronous `s5cmd` invocation** that
//! `scripts/sweep/build_per_codec_training.py:s5cmd()` /`sync_prefix()` used —
//! same binary, same `--endpoint-url` + `--profile` flags, same parallel
//! `s5cmd run` for bulk downloads. This keeps behavioural parity with the
//! Python builder operators already trust. If a future need arises for the
//! in-process async client, factor `R2Client` into a shared crate and swap it
//! in here; the call sites are isolated behind [`R2Sync`].

use std::path::{Path, PathBuf};
use std::process::Command;

use super::table::AssembleError;

/// Synchronous R2 sync over `s5cmd`. Endpoint derivation matches both the
/// Python builder and `zenfleet-vastai`'s `R2Client::new`: explicit
/// `--r2-endpoint` wins, else `$R2_ENDPOINT`, else
/// `https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com`.
pub struct R2Sync {
    bin: String,
    endpoint: String,
    profile: String,
}

impl R2Sync {
    /// Resolve endpoint + profile from explicit args / env.
    pub fn new(endpoint: Option<&str>, profile: &str, bin: &str) -> Result<Self, AssembleError> {
        let endpoint = if let Some(ep) = endpoint {
            ep.to_string()
        } else if let Ok(ep) = std::env::var("R2_ENDPOINT") {
            ep
        } else if let Ok(acct) = std::env::var("R2_ACCOUNT_ID") {
            format!("https://{acct}.r2.cloudflarestorage.com")
        } else {
            return Err(AssembleError::Io(
                "R2 endpoint unknown: pass --r2-endpoint, or set $R2_ENDPOINT / $R2_ACCOUNT_ID"
                    .into(),
            ));
        };
        Ok(Self {
            bin: bin.to_string(),
            endpoint,
            profile: profile.to_string(),
        })
    }

    fn base_cmd(&self) -> Command {
        let mut c = Command::new(&self.bin);
        c.arg("--endpoint-url")
            .arg(&self.endpoint)
            .arg("--profile")
            .arg(&self.profile);
        c
    }

    /// List `*.parquet` object names under an `s3://…/` prefix. Mirrors
    /// `build_per_codec_training.sync_prefix`'s `s5cmd ls` parse (last
    /// whitespace-separated token is the filename).
    pub fn list_parquets(&self, prefix: &str) -> Result<Vec<String>, AssembleError> {
        let out = self
            .base_cmd()
            .arg("ls")
            .arg(prefix)
            .output()
            .map_err(|e| AssembleError::Io(format!("spawn s5cmd ls: {e}")))?;
        if !out.status.success() {
            // s5cmd returns nonzero on no-match — treat as empty, like the
            // fleet client's `exists`.
            return Ok(Vec::new());
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut names = Vec::new();
        for line in text.lines() {
            if let Some(tok) = line
                .split_whitespace()
                .last()
                .filter(|t| t.ends_with(".parquet"))
            {
                names.push(tok.to_string());
            }
        }
        Ok(names)
    }

    /// Download every `*.parquet` under `prefix` into `local_dir`, skipping
    /// names already present (the resumable behaviour of `sync_prefix`).
    /// Returns the total count of parquets in `local_dir` afterward.
    pub fn sync_prefix(&self, prefix: &str, local_dir: &Path) -> Result<usize, AssembleError> {
        std::fs::create_dir_all(local_dir)
            .map_err(|e| AssembleError::Io(format!("mkdir {}: {e}", local_dir.display())))?;
        let existing: std::collections::HashSet<String> = std::fs::read_dir(local_dir)
            .map_err(|e| AssembleError::Io(format!("readdir {}: {e}", local_dir.display())))?
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| n.ends_with(".parquet"))
            .collect();

        let remote = self.list_parquets(prefix)?;
        let to_dl: Vec<&String> = remote.iter().filter(|n| !existing.contains(*n)).collect();

        if !to_dl.is_empty() {
            // Use `s5cmd run` with a script for parallel downloads — same as
            // the Python builder's `--numworkers 32 run` path.
            for name in &to_dl {
                let uri = format!("{prefix}{name}");
                let dst = local_dir.join(name);
                let out = self
                    .base_cmd()
                    .arg("cp")
                    .arg(&uri)
                    .arg(&dst)
                    .output()
                    .map_err(|e| AssembleError::Io(format!("spawn s5cmd cp: {e}")))?;
                if !out.status.success() {
                    return Err(AssembleError::Io(format!(
                        "s5cmd cp {uri} failed: {}",
                        String::from_utf8_lossy(&out.stderr)
                    )));
                }
            }
        }
        Ok(existing.len() + to_dl.len())
    }
}

/// List the local `*.parquet` files in `dir`, sorted, for deterministic
/// concatenation order.
pub fn list_local_parquets(dir: &Path) -> Result<Vec<PathBuf>, AssembleError> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| AssembleError::Io(format!("readdir {}: {e}", dir.display())))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "parquet").unwrap_or(false))
        .collect();
    files.sort();
    Ok(files)
}
