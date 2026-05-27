//! SaladCloud `CredentialSource` + `WorkerHost` — Salad IMDS + the
//! reserved container-group environment variables.
//!
//! SaladCloud auto-injects two reserved env vars into every container
//! instance: `SALAD_MACHINE_ID` (this instance's machine id) and
//! `SALAD_CONTAINER_GROUP_ID` (the whole group's id). The Instance
//! Metadata Service (IMDS) is a REST API at `http://169.254.169.254:80`
//! (overridable via `SALAD_METADATA_URI`, matching the sidecar's
//! config) reachable only from inside a Salad node; requests MUST carry
//! the `Metadata: true` header and MUST NOT carry any `X-Forwarded-*`
//! header (Salad rejects those with 403). `GET /v1/status` reports
//! `{ready, started}`; `GET /v1/token` returns `{jwt}` (a workload
//! identity token whose claims include `salad_machine_id`).
//!
//! Spec §1.9 item 3: `CredentialSource` + `WorkerHost` read the IMDS +
//! the container-group env. R2/S3 credentials themselves are BYO — set
//! as container-group env vars by the launcher (Salad has no native
//! secret store beyond env vars), so `resolve()` reads the same
//! `R2_*` / `S5CMD_*` keys the vast.ai path uses.

use std::path::PathBuf;

use zen_cloud_core::{CloudError, CredentialSource, Credentials, WorkerHost, WorkerId};

/// Default IMDS base URL on a Salad node.
pub const DEFAULT_IMDS_URI: &str = "http://169.254.169.254:80";

/// Reserved env var: this instance's machine id.
pub const ENV_MACHINE_ID: &str = "SALAD_MACHINE_ID";
/// Reserved env var: the container group id.
pub const ENV_CONTAINER_GROUP_ID: &str = "SALAD_CONTAINER_GROUP_ID";
/// Override for the IMDS base URI (matches the sidecar's
/// `SALAD_METADATA_URI`).
pub const ENV_METADATA_URI: &str = "SALAD_METADATA_URI";

/// `CredentialSource` reading the container-group environment.
///
/// Unlike vast.ai's `/proc/1/environ` trick, Salad injects env vars into
/// the app process normally, so a plain `std::env::var` read suffices.
/// The keys are the BYO object-store creds the launcher set on the
/// container group, plus the reserved Salad identity vars.
#[derive(Default)]
pub struct SaladEnvCredentials;

impl CredentialSource for SaladEnvCredentials {
    fn resolve(&self) -> Result<Credentials, CloudError> {
        let mut out = Credentials::new();
        let keys = [
            // BYO R2 / S3 storage creds (set by the launcher on the
            // container group; same names the vast.ai path expects).
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ENDPOINT",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "S5CMD_PROFILE",
            // Sweep wiring.
            "SWEEP_RUN_ID",
            "CHUNKS_R2",
            "WORKER_ID",
            // Salad-reserved identity vars.
            ENV_MACHINE_ID,
            ENV_CONTAINER_GROUP_ID,
            ENV_METADATA_URI,
        ];
        for k in keys {
            if let Ok(v) = std::env::var(k) {
                out.insert(k.to_string(), v);
            }
        }
        Ok(out)
    }
}

/// `WorkerHost` over Salad's instance identity.
///
/// The worker id is `SALAD_MACHINE_ID` (falling back to `$WORKER_ID`
/// then the hostname); the scratch dir defaults to `/workspace` (the
/// container working dir the deploy image uses); GPU count comes from
/// `nvidia-smi` (one line per GPU), matching the vast.ai host's probe.
pub struct SaladWorkerHost {
    worker_id: WorkerId,
    scratch: PathBuf,
}

impl SaladWorkerHost {
    /// Build from the environment: `SALAD_MACHINE_ID` → `$WORKER_ID` →
    /// hostname for the id; `$WORKDIR` → `/workspace` for scratch.
    pub fn from_env() -> Self {
        let worker_id = std::env::var(ENV_MACHINE_ID)
            .ok()
            .filter(|s| !s.is_empty())
            .or_else(|| std::env::var("WORKER_ID").ok().filter(|s| !s.is_empty()))
            .or_else(hostname)
            .unwrap_or_else(|| "salad-worker".to_string());
        let scratch = std::env::var("WORKDIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/workspace"));
        Self {
            worker_id: WorkerId(worker_id),
            scratch,
        }
    }

    /// Explicit constructor (tests / non-Salad hosts).
    pub fn new(worker_id: impl Into<String>, scratch: impl Into<PathBuf>) -> Self {
        Self {
            worker_id: WorkerId(worker_id.into()),
            scratch: scratch.into(),
        }
    }
}

impl WorkerHost for SaladWorkerHost {
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
/// container runtimes) falling back to the `hostname` command.
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

/// Minimal SaladCloud IMDS client.
///
/// Only the endpoints the worker glue needs: `/v1/status` (readiness)
/// and `/v1/token` (workload identity JWT). The `Metadata: true` header
/// is mandatory and web proxies must be disabled — Salad routes IMDS
/// link-locally and rejects proxied/forwarded requests with 403.
pub struct SaladImds {
    base: String,
    client: reqwest::Client,
}

/// `GET /v1/status` response.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ImdsStatus {
    pub ready: bool,
    pub started: bool,
}

/// `GET /v1/token` response.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ImdsToken {
    pub jwt: String,
}

impl SaladImds {
    /// Build a client targeting the IMDS base URL (from
    /// `SALAD_METADATA_URI`, else the link-local default). Proxies are
    /// disabled per the IMDS contract.
    pub fn from_env() -> Result<Self, CloudError> {
        let base = std::env::var(ENV_METADATA_URI)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_IMDS_URI.to_string());
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .map_err(|e| CloudError::Host(format!("build IMDS client: {e}")))?;
        Ok(Self { base, client })
    }

    /// `GET /v1/status` — instance readiness.
    pub async fn status(&self) -> Result<ImdsStatus, CloudError> {
        self.get_json("/v1/status").await
    }

    /// `GET /v1/token` — workload identity token (JWT).
    pub async fn token(&self) -> Result<ImdsToken, CloudError> {
        self.get_json("/v1/token").await
    }

    async fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T, CloudError> {
        let url = format!("{}{}", self.base.trim_end_matches('/'), path);
        let resp = self
            .client
            .get(&url)
            // Mandatory IMDS header; absence → 403.
            .header("Metadata", "true")
            .send()
            .await
            .map_err(|e| CloudError::Host(format!("IMDS GET {path}: {e}")))?;
        if !resp.status().is_success() {
            return Err(CloudError::Host(format!(
                "IMDS GET {path}: HTTP {}",
                resp.status()
            )));
        }
        resp.json::<T>()
            .await
            .map_err(|e| CloudError::Host(format!("IMDS GET {path}: decode: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_host_explicit() {
        let h = SaladWorkerHost::new("machine-abc", "/workspace");
        assert_eq!(h.worker_id().as_str(), "machine-abc");
        assert_eq!(h.scratch_dir(), PathBuf::from("/workspace"));
        let _ = h.gpu_count(); // env-dependent; must not panic.
    }

    #[test]
    fn imds_default_uri_is_link_local() {
        assert_eq!(DEFAULT_IMDS_URI, "http://169.254.169.254:80");
    }

    #[test]
    fn imds_status_deserializes() {
        let s: ImdsStatus = serde_json::from_str(r#"{"ready":true,"started":true}"#).unwrap();
        assert!(s.ready && s.started);
    }

    #[test]
    fn imds_token_deserializes() {
        let t: ImdsToken = serde_json::from_str(r#"{"jwt":"eyJhbGc"}"#).unwrap();
        assert_eq!(t.jwt, "eyJhbGc");
    }

    #[test]
    fn imds_builds_from_env_default() {
        // Without SALAD_METADATA_URI set, the default link-local base is
        // used and the client builds (proxies disabled).
        let imds = SaladImds::from_env().expect("build IMDS");
        assert_eq!(imds.base, DEFAULT_IMDS_URI);
    }
}
