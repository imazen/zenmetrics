//! Local `CredentialSource` + `WorkerHost` — process env + optional
//! `.env`, hostname identity, `nvidia-smi` GPU probe.
//!
//! A local run needs no *remote* credentials — there is no cloud to
//! authenticate to. But the same compute closure the cloud backends run
//! may read storage/sweep env vars (e.g. `R2_ENDPOINT` when a developer
//! points the filesystem mirror at a half-real layout, or `WORKER_ID`),
//! so [`DotenvCredentials`] returns the process environment, optionally
//! seeded from a `.env` file in the working directory. It only ever
//! returns keys that are actually set — it never invents nulls.
//!
//! [`LocalWorkerHost`] reports this box's identity: worker id from
//! `$WORKER_ID` (else the hostname), scratch dir from `$WORKDIR` (else a
//! per-process temp dir), and GPU count from `nvidia-smi` — so a local
//! run on a GPU box (this one has an RTX 5070) exercises the same GPU
//! detection the cloud hosts use.

use std::path::{Path, PathBuf};

use zenfleet_cloud::{CloudError, CredentialSource, Credentials, WorkerHost, WorkerId};

/// `CredentialSource` over the process environment plus an optional
/// `.env` file.
///
/// Local runs need no remote creds, so this resolves to the env map. A
/// `.env` file (if present and readable) seeds entries that are not
/// already set in the process env — the live env always wins. By default
/// it reads `./.env`; an explicit path can be set with
/// [`DotenvCredentials::with_dotenv`].
#[derive(Default, Clone, Debug)]
pub struct DotenvCredentials {
    /// Optional path to a `.env` file. `None` → no file is read (env
    /// only). The default constructor leaves this `None`; the worker
    /// glue opts in with `with_dotenv("./.env")`.
    dotenv_path: Option<PathBuf>,
    /// Which env keys to surface. Empty → surface ALL env vars (the
    /// "local debugging, give me everything" default); non-empty →
    /// surface only the named subset (matching the cloud providers'
    /// allow-list shape).
    keys: Vec<String>,
}

impl DotenvCredentials {
    /// Read the whole process environment (and a `.env` file if set via
    /// [`Self::with_dotenv`]). The local default — a developer debugging
    /// the compute path wants whatever they exported visible.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restrict resolution to the named keys (the same allow-list shape
    /// the vast.ai / runpod / salad credential sources use). Useful when
    /// the local run should mirror a cloud run's exact credential
    /// surface.
    pub fn with_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.keys = keys.into_iter().map(Into::into).collect();
        self
    }

    /// Seed credentials from a `.env` file at `path` (entries that are
    /// not already in the process env). Missing / unreadable file is not
    /// an error — it is simply skipped.
    pub fn with_dotenv(mut self, path: impl Into<PathBuf>) -> Self {
        self.dotenv_path = Some(path.into());
        self
    }

    /// Parse a `.env` body into key/value pairs. Supports `KEY=VALUE`
    /// lines, blank lines, `#` comments, an optional leading `export `,
    /// and single/double-quoted values. Deliberately tiny — no crate dep
    /// for a dev-only convenience.
    fn parse_dotenv(body: &str) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for raw in body.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let line = line.strip_prefix("export ").unwrap_or(line);
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let key = k.trim();
            if key.is_empty() {
                continue;
            }
            let mut val = v.trim();
            // Strip a single matched pair of surrounding quotes.
            if (val.starts_with('"') && val.ends_with('"') && val.len() >= 2)
                || (val.starts_with('\'') && val.ends_with('\'') && val.len() >= 2)
            {
                val = &val[1..val.len() - 1];
            }
            out.push((key.to_string(), val.to_string()));
        }
        out
    }

    fn read_dotenv(path: &Path) -> Vec<(String, String)> {
        match std::fs::read_to_string(path) {
            Ok(body) => Self::parse_dotenv(&body),
            Err(_) => Vec::new(),
        }
    }
}

impl CredentialSource for DotenvCredentials {
    fn resolve(&self) -> Result<Credentials, CloudError> {
        let mut out = Credentials::new();

        // 1. The live process env. When `keys` is empty, surface
        //    everything; otherwise only the named subset.
        if self.keys.is_empty() {
            for (k, v) in std::env::vars() {
                out.insert(k, v);
            }
        } else {
            for k in &self.keys {
                if let Ok(v) = std::env::var(k) {
                    out.insert(k.clone(), v);
                }
            }
        }

        // 2. The `.env` file fills in entries the process env did NOT set
        //    (the live env always wins). When a key allow-list is set,
        //    only those keys are taken from the file.
        if let Some(path) = &self.dotenv_path {
            for (k, v) in Self::read_dotenv(path) {
                if !self.keys.is_empty() && !self.keys.contains(&k) {
                    continue;
                }
                out.entry(k).or_insert(v);
            }
        }

        Ok(out)
    }
}

/// `WorkerHost` over this local box's identity.
///
/// The worker id is `$WORKER_ID` (falling back to the hostname); the
/// scratch dir is `$WORKDIR` (falling back to a per-process temp dir);
/// GPU count comes from `nvidia-smi` (one line per GPU) — identical to
/// the vast.ai / salad / runpod host probes, so a local run on a GPU box
/// sees its real GPUs.
pub struct LocalWorkerHost {
    worker_id: WorkerId,
    scratch: PathBuf,
}

impl LocalWorkerHost {
    /// Build from the environment: `$WORKER_ID` → hostname →
    /// `"localhost"` for the id; `$WORKDIR` → a `zenfleet-local-*` dir
    /// under the system temp dir for scratch.
    pub fn from_env() -> Self {
        let worker_id = std::env::var("WORKER_ID")
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(hostname)
            .unwrap_or_else(|| "localhost".to_string());
        let scratch = std::env::var("WORKDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("zenfleet-local"));
        Self {
            worker_id: WorkerId(worker_id),
            scratch,
        }
    }

    /// Explicit constructor (tests / pinned scratch).
    pub fn new(worker_id: impl Into<String>, scratch: impl Into<PathBuf>) -> Self {
        Self {
            worker_id: WorkerId(worker_id.into()),
            scratch: scratch.into(),
        }
    }
}

impl WorkerHost for LocalWorkerHost {
    fn worker_id(&self) -> WorkerId {
        self.worker_id.clone()
    }

    fn scratch_dir(&self) -> PathBuf {
        self.scratch.clone()
    }

    fn gpu_count(&self) -> usize {
        match std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count(),
            _ => 0,
        }
    }
}

/// Read the system hostname via the `HOSTNAME` env var (set in most
/// shells / container runtimes) falling back to the `hostname` command.
fn hostname() -> Option<String> {
    if let Ok(h) = std::env::var("HOSTNAME")
        && !h.is_empty()
    {
        return Some(h);
    }
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_host_explicit() {
        let h = LocalWorkerHost::new("local-1", "/tmp/scratch");
        assert_eq!(h.worker_id().as_str(), "local-1");
        assert_eq!(h.scratch_dir(), PathBuf::from("/tmp/scratch"));
        // gpu_count is env-dependent (this box has a GPU; CI may not) —
        // it must not panic regardless.
        let _ = h.gpu_count();
    }

    #[test]
    fn from_env_yields_nonempty_identity_and_scratch() {
        let h = LocalWorkerHost::from_env();
        assert!(!h.worker_id().as_str().is_empty());
        assert!(!h.scratch_dir().as_os_str().is_empty());
    }

    #[test]
    fn credentials_resolve_subset_only_returns_present_keys() {
        // With an allow-list, resolve() must never invent keys: every
        // returned key is genuinely set in the ambient env. (Mutating the
        // process env from a test is `unsafe` + racy under the parallel
        // runner, so we assert against whatever the env happens to be.)
        let creds = DotenvCredentials::new()
            .with_keys([
                "WORKER_ID",
                "R2_ENDPOINT",
                "ZEN_CLOUD_LOCAL_DEFINITELY_UNSET_XYZ",
            ])
            .resolve()
            .unwrap();
        for k in creds.keys() {
            assert!(
                std::env::var(k).is_ok(),
                "resolve() returned key {k:?} that is not actually set"
            );
        }
        assert!(!creds.contains_key("ZEN_CLOUD_LOCAL_DEFINITELY_UNSET_XYZ"));
    }

    #[test]
    fn dotenv_parser_handles_comments_quotes_export() {
        let body = "\
# a comment
export FOO=bar
BAZ=\"quoted value\"
QUX='single'
EMPTY=

=novalue
NOEQ
SPACED = trimmed
";
        let parsed: std::collections::HashMap<_, _> =
            DotenvCredentials::parse_dotenv(body).into_iter().collect();
        assert_eq!(parsed.get("FOO").map(String::as_str), Some("bar"));
        assert_eq!(parsed.get("BAZ").map(String::as_str), Some("quoted value"));
        assert_eq!(parsed.get("QUX").map(String::as_str), Some("single"));
        assert_eq!(parsed.get("EMPTY").map(String::as_str), Some(""));
        assert_eq!(parsed.get("SPACED").map(String::as_str), Some("trimmed"));
        // A line with an empty key and a bare `NOEQ` line are skipped.
        assert!(!parsed.contains_key(""));
        assert!(!parsed.contains_key("NOEQ"));
    }

    #[test]
    fn dotenv_fills_only_unset_keys_under_allowlist() {
        let dir = tempfile::tempdir().unwrap();
        let env_path = dir.path().join(".env");
        std::fs::write(
            &env_path,
            "ZEN_CLOUD_LOCAL_TEST_KEY_FROM_DOTENV=from_file\n",
        )
        .unwrap();
        let creds = DotenvCredentials::new()
            .with_keys(["ZEN_CLOUD_LOCAL_TEST_KEY_FROM_DOTENV"])
            .with_dotenv(&env_path)
            .resolve()
            .unwrap();
        // The key is absent from the process env, so the `.env` value is
        // used.
        assert_eq!(
            creds
                .get("ZEN_CLOUD_LOCAL_TEST_KEY_FROM_DOTENV")
                .map(String::as_str),
            Some("from_file")
        );
    }

    #[test]
    fn missing_dotenv_is_not_an_error() {
        let creds = DotenvCredentials::new()
            .with_keys(["WORKER_ID"])
            .with_dotenv("/nonexistent/path/.env")
            .resolve();
        assert!(creds.is_ok());
    }
}
