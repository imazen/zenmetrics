//! SaladCloud launcher-side provider — provisioning over the public API.
//!
//! This is the operator-workstation half (spec §1.3 / §1.9
//! "Launcher-side"): it provisions Salad infrastructure and pushes work,
//! then the worker-side glue in this crate runs on the node. It is NOT
//! baked into the deploy image and is NOT on the hot path.
//!
//! ## What it does
//!
//! 1. Resolve a GPU *class id* from a human name (`GET .../gpu-classes`).
//! 2. Create a managed Job Queue (`POST .../queues`).
//! 3. Create a container group attached to that queue, with the
//!    `salad-http-job-queue-worker` sidecar baked into the image and the
//!    queue connection pointing at the worker's local HTTP receiver
//!    (`POST .../containers`).
//! 4. Push job chunks into the queue (`POST .../queues/{name}/jobs`).
//! 5. Inspect / scale the container group (`GET .../containers/{name}`,
//!    `POST .../start`, `POST .../stop`).
//!
//! ## Auth + endpoints
//!
//! Public API base: `https://api.salad.com/api/public`. Auth header:
//! `Salad-Api-Key: <key>` (portal.salad.com → API Keys). The key is read
//! from `$SALAD_API_KEY` or `~/.config/salad/credentials` (mirroring the
//! R2 creds convention). All bodies are JSON (`serde`). Request/response
//! shapes track the Salad OpenAPI as exposed by the official Python SDK
//! (`salad-cloud-sdk-python`), the source of truth for field names.
//!
//! There is NO Rust SDK (Salad ships Python/Go/Java/JS/.NET), so this
//! hand-rolls the calls with `reqwest` + `serde` per spec §1.9.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use zen_cloud_core::{DEFAULT_TTL_SECONDS, Permission, ScopedR2Cred, mint_scoped_r2_cred};

/// Public API base URL.
pub const API_BASE: &str = "https://api.salad.com/api/public";

/// A configured SaladCloud public-API client scoped to one org+project.
pub struct SaladApi {
    base: String,
    api_key: String,
    organization: String,
    project: String,
    http: reqwest::Client,
}

/// One GPU class as returned by `GET .../gpu-classes`.
#[derive(Debug, Clone, Deserialize)]
pub struct GpuClass {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub is_high_demand: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct GpuClassList {
    items: Vec<GpuClass>,
}

/// Body for `POST .../queues`.
#[derive(Debug, Serialize)]
pub struct CreateQueueRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

/// A managed queue (subset of the `GET .../queues/{name}` response).
#[derive(Debug, Clone, Deserialize)]
pub struct Queue {
    pub name: String,
    #[serde(default)]
    pub current_queue_length: Option<i64>,
}

/// The S3-compatible resource requirements block.
#[derive(Debug, Serialize)]
pub struct ResourceRequirements {
    pub cpu: u32,
    /// Memory in MiB.
    pub memory: u32,
    /// GPU class ids (resolved from names via [`SaladApi::resolve_gpu_class`]).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpu_classes: Vec<String>,
}

/// Container registry auth (for a private image).
#[derive(Debug, Serialize)]
pub struct RegistryAuth {
    pub username: String,
    pub password: String,
}

/// The container block of a container-group creation request.
#[derive(Debug, Serialize)]
pub struct ContainerConfig {
    pub image: String,
    pub resources: ResourceRequirements,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<Vec<String>>,
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub environment_variables: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub registry_authentication: Option<RegistryAuth>,
}

/// The `queue_connection` block: ties the container group's sidecar to a
/// managed queue and tells the sidecar which local HTTP `path`/`port` to
/// forward jobs to. `port` MUST match the worker's
/// [`crate::queue::SaladQueueConfig::bind_addr`] port.
#[derive(Debug, Serialize)]
pub struct QueueConnection {
    pub path: String,
    pub port: u16,
    pub queue_name: String,
}

/// Body for `POST .../containers`.
#[derive(Debug, Serialize)]
pub struct CreateContainerGroupRequest {
    /// DNS-style name (lowercase alphanumeric + hyphens).
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub container: ContainerConfig,
    pub replicas: u32,
    /// `always` is the right policy for a long-running sweep worker.
    pub restart_policy: String,
    /// Start the group immediately on create.
    pub autostart_policy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_connection: Option<QueueConnection>,
}

/// Subset of the container-group resource returned on create / get.
#[derive(Debug, Clone, Deserialize)]
pub struct ContainerGroup {
    pub name: String,
    #[serde(default)]
    pub current_state: Option<serde_json::Value>,
    #[serde(default)]
    pub replicas: Option<u32>,
}

/// Body for `POST .../queues/{name}/jobs` — one job chunk.
#[derive(Debug, Serialize)]
pub struct CreateQueueJobRequest {
    /// The job input — any valid JSON. The sidecar forwards this as the
    /// HTTP POST body to the worker's receiver, which surfaces it as
    /// [`zen_cloud_core::Chunk::payload`].
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
}

/// A created queue job (subset).
#[derive(Debug, Clone, Deserialize)]
pub struct QueueJob {
    pub id: String,
    #[serde(default)]
    pub status: Option<String>,
}

impl SaladApi {
    /// Build a client. `api_key` from `$SALAD_API_KEY` or
    /// `~/.config/salad/credentials` if `None`.
    pub fn new(
        organization: impl Into<String>,
        project: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let api_key = match api_key {
            Some(k) => k,
            None => load_api_key().context("resolve Salad API key")?,
        };
        let http = reqwest::Client::builder()
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            base: API_BASE.to_string(),
            api_key,
            organization: organization.into(),
            project: project.into(),
            http,
        })
    }

    /// Override the API base (testing against a mock server).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn org_url(&self, suffix: &str) -> String {
        format!(
            "{}/organizations/{}{}",
            self.base, self.organization, suffix
        )
    }

    fn proj_url(&self, suffix: &str) -> String {
        format!(
            "{}/organizations/{}/projects/{}{}",
            self.base, self.organization, self.project, suffix
        )
    }

    /// `GET .../gpu-classes` then resolve a class id by (case-insensitive)
    /// name match. Returns the id to put in `gpu_classes`.
    pub async fn resolve_gpu_class(&self, name: &str) -> Result<String> {
        let url = self.org_url("/gpu-classes");
        let resp = self
            .http
            .get(&url)
            .header("Salad-Api-Key", &self.api_key)
            .send()
            .await
            .context("GET gpu-classes")?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("GET gpu-classes: HTTP {status}: {text}");
        }
        let list: GpuClassList =
            serde_json::from_str(&text).context("decode gpu-classes response")?;
        list.items
            .into_iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .map(|c| c.id)
            .with_context(|| format!("no GPU class named {name:?}"))
    }

    /// `POST .../queues` — create a managed job queue.
    pub async fn create_queue(&self, req: &CreateQueueRequest) -> Result<Queue> {
        self.post_json(&self.proj_url("/queues"), req).await
    }

    /// `POST .../containers` — create a container group attached to a
    /// queue, running the sidecar + the cloud-agnostic worker image.
    pub async fn create_container_group(
        &self,
        req: &CreateContainerGroupRequest,
    ) -> Result<ContainerGroup> {
        self.post_json(&self.proj_url("/containers"), req).await
    }

    /// `GET .../containers/{name}` — inspect a container group (state,
    /// replicas) for monitoring.
    pub async fn get_container_group(&self, name: &str) -> Result<ContainerGroup> {
        let url = self.proj_url(&format!("/containers/{name}"));
        self.get_json(&url).await
    }

    /// `POST .../containers/{name}/stop` — stop a container group.
    pub async fn stop_container_group(&self, name: &str) -> Result<()> {
        let url = self.proj_url(&format!("/containers/{name}/stop"));
        self.post_empty(&url).await
    }

    /// `POST .../queues/{queue}/jobs` — push one job chunk into the
    /// managed queue. Salad fans these out to the sidecars.
    pub async fn push_job(&self, queue: &str, req: &CreateQueueJobRequest) -> Result<QueueJob> {
        let url = self.proj_url(&format!("/queues/{queue}/jobs"));
        self.post_json(&url, req).await
    }

    /// Push many chunks; returns the created job ids in order.
    pub async fn push_jobs(
        &self,
        queue: &str,
        chunks: impl IntoIterator<Item = serde_json::Value>,
    ) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for input in chunks {
            let job = self
                .push_job(
                    queue,
                    &CreateQueueJobRequest {
                        input,
                        metadata: None,
                    },
                )
                .await?;
            ids.push(job.id);
        }
        Ok(ids)
    }

    async fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<R> {
        let resp = self
            .http
            .post(url)
            .header("Salad-Api-Key", &self.api_key)
            .json(body)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("POST {url}: HTTP {status}: {text}");
        }
        serde_json::from_str(&text).with_context(|| format!("decode response from {url}"))
    }

    async fn get_json<R: for<'de> Deserialize<'de>>(&self, url: &str) -> Result<R> {
        let resp = self
            .http
            .get(url)
            .header("Salad-Api-Key", &self.api_key)
            .send()
            .await
            .with_context(|| format!("GET {url}"))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            bail!("GET {url}: HTTP {status}: {text}");
        }
        serde_json::from_str(&text).with_context(|| format!("decode response from {url}"))
    }

    async fn post_empty(&self, url: &str) -> Result<()> {
        let resp = self
            .http
            .post(url)
            .header("Salad-Api-Key", &self.api_key)
            .send()
            .await
            .with_context(|| format!("POST {url}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("POST {url}: HTTP {status}: {text}");
        }
        Ok(())
    }
}

/// Resolve the Salad API key: `$SALAD_API_KEY`, else the
/// `~/.config/salad/credentials` file (first non-empty,
/// non-`#`-prefixed line, or a `SALAD_API_KEY=...` line).
pub fn load_api_key() -> Result<String> {
    if let Ok(k) = std::env::var("SALAD_API_KEY")
        && !k.trim().is_empty()
    {
        return Ok(k.trim().to_string());
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = std::path::Path::new(&home).join(".config/salad/credentials");
    let contents =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(v) = line.strip_prefix("SALAD_API_KEY=") {
            return Ok(v.trim().to_string());
        }
        // A bare key on its own line.
        return Ok(line.to_string());
    }
    bail!("no API key in {}", path.display())
}

// ── Per-sweep scoped R2 credential minting + injection ──────────────────
//
// SaladCloud runs containers on hardware the operator does NOT own
// (distributed consumer GPUs). Injecting the root R2 key into a remote
// container exposes the whole R2 account to a hostile node operator who
// can read the container env. Instead the launcher mints a credential
// scoped to ONE bucket (object-read-write, short TTL) and injects the
// minted key+secret+session-token. A compromised node's blast radius is
// then limited to that one bucket. See
// `~/work/claudehints/topics/r2-credentials.md` and the shared minter at
// `zen_cloud_core::r2creds`.

/// Where to read the parent (root) R2 keys + Cloudflare API token the
/// launcher uses to mint scoped creds. These live on the operator box
/// (mirroring `~/.config/cloudflare/r2-credentials`) and are NEVER
/// injected into a worker — only the minted child cred is.
#[derive(Debug, Clone)]
pub struct R2ParentCreds {
    /// Cloudflare REST API bearer token with R2 temp-cred-mint permission.
    pub cf_api_token: String,
    /// Cloudflare account id (the `{account_id}` in the API path).
    pub account_id: String,
    /// Root R2 **S3** access key id (the "parent" key for minting).
    pub parent_access_key_id: String,
    /// Root R2 S3 secret (the "parent" secret).
    pub parent_secret_access_key: String,
}

impl R2ParentCreds {
    /// Resolve the parent creds from the operator-box environment:
    /// `CF_API_TOKEN` || `R2_API_TOKEN` for the bearer token, plus
    /// `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`.
    /// These mirror `~/.config/cloudflare/r2-credentials`. NEVER hardcode
    /// secrets — this only reads them from the environment.
    pub fn from_env() -> Result<Self> {
        let cf_api_token = std::env::var("CF_API_TOKEN")
            .or_else(|_| std::env::var("R2_API_TOKEN"))
            .context("CF_API_TOKEN (or R2_API_TOKEN) not set — needed to mint scoped R2 creds")?;
        let account_id = std::env::var("R2_ACCOUNT_ID").context("R2_ACCOUNT_ID not set")?;
        let parent_access_key_id =
            std::env::var("R2_ACCESS_KEY_ID").context("R2_ACCESS_KEY_ID not set")?;
        let parent_secret_access_key =
            std::env::var("R2_SECRET_ACCESS_KEY").context("R2_SECRET_ACCESS_KEY not set")?;
        for (name, val) in [
            ("CF_API_TOKEN/R2_API_TOKEN", &cf_api_token),
            ("R2_ACCOUNT_ID", &account_id),
            ("R2_ACCESS_KEY_ID", &parent_access_key_id),
            ("R2_SECRET_ACCESS_KEY", &parent_secret_access_key),
        ] {
            if val.trim().is_empty() {
                bail!("{name} is set but empty");
            }
        }
        Ok(Self {
            cf_api_token,
            account_id,
            parent_access_key_id,
            parent_secret_access_key,
        })
    }
}

/// What to scope a per-sweep minted credential to. R2 temp creds are
/// **single-bucket** — see the SALAD.md follow-on note about
/// reads-from-A-writes-to-B sweeps.
#[derive(Debug, Clone)]
pub struct ScopedCredSpec {
    /// The sweep's WORKING bucket (object-read-write).
    pub bucket: String,
    /// Optional tighter prefix scope (e.g. `["runs/<SWEEP_ID>/"]`). Empty
    /// = the whole bucket.
    pub prefixes: Vec<String>,
    /// TTL in seconds. Defaults (via [`ScopedCredSpec::new`]) to 6h; the
    /// minter clamps to Cloudflare's `[900, 604800]` range.
    pub ttl_seconds: u64,
    /// The permission to mint. Workers use [`Permission::ObjectReadWrite`].
    pub permission: Permission,
}

impl ScopedCredSpec {
    /// A spec for `bucket` with the 6h default TTL + object-read-write —
    /// the right shape for a sweep worker.
    pub fn new(bucket: impl Into<String>) -> Self {
        Self {
            bucket: bucket.into(),
            prefixes: Vec::new(),
            ttl_seconds: DEFAULT_TTL_SECONDS,
            permission: Permission::ObjectReadWrite,
        }
    }

    /// Tighten the scope to one or more key prefixes.
    pub fn with_prefixes(mut self, prefixes: Vec<String>) -> Self {
        self.prefixes = prefixes;
        self
    }

    /// Override the TTL (clamped to `[900, 604800]` at mint time).
    pub fn with_ttl_seconds(mut self, ttl: u64) -> Self {
        self.ttl_seconds = ttl;
        self
    }
}

/// Inject a minted scoped R2 credential into a container-group env map.
///
/// Sets the three keys the worker + entrypoint consume:
/// `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`.
/// The session token is REQUIRED — a temp key+secret without it 403s
/// (see `zen_cloud_core::r2creds` gotchas). The existing env-injection
/// mechanism is unchanged; this just inserts these three entries.
pub fn inject_r2_cred_into_env(env: &mut HashMap<String, String>, cred: &ScopedR2Cred) {
    env.insert("R2_ACCESS_KEY_ID".into(), cred.access_key_id.clone());
    env.insert(
        "R2_SECRET_ACCESS_KEY".into(),
        cred.secret_access_key.clone(),
    );
    env.insert("AWS_SESSION_TOKEN".into(), cred.session_token.clone());
}

impl SaladApi {
    /// Mint a scoped, auto-expiring R2 credential for a sweep's working
    /// bucket, using the operator-box parent keys.
    ///
    /// This is the provider-agnostic minter (`zen_cloud_core::r2creds`)
    /// behind a Salad-launcher-shaped convenience: pass the parent creds
    /// (typically [`R2ParentCreds::from_env`]) and a [`ScopedCredSpec`].
    /// Inject the result with [`inject_r2_cred_into_env`] before
    /// [`SaladApi::create_container_group`].
    pub async fn mint_sweep_r2_cred(
        &self,
        parent: &R2ParentCreds,
        spec: &ScopedCredSpec,
    ) -> Result<ScopedR2Cred> {
        mint_scoped_r2_cred(
            &parent.cf_api_token,
            &parent.account_id,
            &parent.parent_access_key_id,
            &parent.parent_secret_access_key,
            &spec.bucket,
            &spec.prefixes,
            spec.permission,
            spec.ttl_seconds,
        )
        .await
        .with_context(|| format!("mint scoped R2 cred for bucket {:?}", spec.bucket))
    }

    /// Create a container group, OPTIONALLY minting + injecting a scoped
    /// R2 credential first.
    ///
    /// When `cred` is `Some((parent, spec))`, mint a scoped cred for the
    /// sweep's working bucket and inject `R2_ACCESS_KEY_ID` /
    /// `R2_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` into the request's
    /// container env (any pre-set values for those three keys are
    /// overwritten by the minted cred). When `None`, the caller's env is
    /// used verbatim — back-compat for callers that supply their own
    /// (e.g. pre-set permanent-token) creds. Minting is NOT forced.
    ///
    /// Returns the created group plus the minted cred (if any) so the
    /// caller can track `expires_at` for re-mint bookkeeping on long
    /// sweeps.
    pub async fn create_container_group_with_scoped_cred(
        &self,
        mut req: CreateContainerGroupRequest,
        cred: Option<(&R2ParentCreds, &ScopedCredSpec)>,
    ) -> Result<(ContainerGroup, Option<ScopedR2Cred>)> {
        let minted = match cred {
            Some((parent, spec)) => {
                let c = self.mint_sweep_r2_cred(parent, spec).await?;
                inject_r2_cred_into_env(&mut req.container.environment_variables, &c);
                Some(c)
            }
            None => None,
        };
        let group = self.create_container_group(&req).await?;
        Ok((group, minted))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn org_and_proj_urls() {
        let api = SaladApi {
            base: "https://api.example".into(),
            api_key: "k".into(),
            organization: "myorg".into(),
            project: "myproj".into(),
            http: reqwest::Client::new(),
        };
        assert_eq!(
            api.org_url("/gpu-classes"),
            "https://api.example/organizations/myorg/gpu-classes"
        );
        assert_eq!(
            api.proj_url("/containers"),
            "https://api.example/organizations/myorg/projects/myproj/containers"
        );
    }

    #[test]
    fn gpu_class_list_decodes() {
        let json = r#"{"items":[{"id":"gpu-1","name":"RTX 4090","is_high_demand":true}]}"#;
        let list: GpuClassList = serde_json::from_str(json).unwrap();
        assert_eq!(list.items[0].id, "gpu-1");
        assert_eq!(list.items[0].name, "RTX 4090");
    }

    #[test]
    fn create_container_group_request_serializes_queue_connection() {
        let req = CreateContainerGroupRequest {
            name: "zen-sweep".into(),
            display_name: None,
            container: ContainerConfig {
                image: "ghcr.io/imazen/zen-sweep:latest".into(),
                resources: ResourceRequirements {
                    cpu: 4,
                    memory: 8192,
                    gpu_classes: vec!["gpu-1".into()],
                },
                command: None,
                environment_variables: HashMap::new(),
                registry_authentication: None,
            },
            replicas: 3,
            restart_policy: "always".into(),
            autostart_policy: true,
            queue_connection: Some(QueueConnection {
                path: "/job".into(),
                port: 80,
                queue_name: "zen-sweep-q".into(),
            }),
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["queue_connection"]["queue_name"], "zen-sweep-q");
        assert_eq!(v["queue_connection"]["port"], 80);
        assert_eq!(v["container"]["resources"]["gpu_classes"][0], "gpu-1");
        assert_eq!(v["replicas"], 3);
        // Empty env map is omitted.
        assert!(v["container"].get("environment_variables").is_none());
    }

    #[test]
    fn create_queue_job_request_serializes_input() {
        let req = CreateQueueJobRequest {
            input: serde_json::json!({"chunk_id": "abc", "codec": "jpeg"}),
            metadata: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(v["input"]["chunk_id"], "abc");
        assert!(v.get("metadata").is_none());
    }

    #[test]
    fn scoped_cred_spec_defaults_to_6h_object_rw() {
        let spec = ScopedCredSpec::new("zen-tuning-ephemeral");
        assert_eq!(spec.bucket, "zen-tuning-ephemeral");
        assert!(spec.prefixes.is_empty());
        assert_eq!(spec.ttl_seconds, 21_600);
        assert_eq!(spec.permission, Permission::ObjectReadWrite);

        let tightened = ScopedCredSpec::new("b")
            .with_prefixes(vec!["runs/s1/".into()])
            .with_ttl_seconds(3600);
        assert_eq!(tightened.prefixes, vec!["runs/s1/".to_string()]);
        assert_eq!(tightened.ttl_seconds, 3600);
    }

    #[test]
    fn inject_sets_the_three_required_env_keys() {
        // Construct a ScopedR2Cred directly (no live Cloudflare call).
        let cred = ScopedR2Cred {
            access_key_id: "CHILD_AK".into(),
            secret_access_key: "CHILD_SK".into(),
            session_token: "SESSION_TOKEN".into(),
            expires_at: Some(1_700_000_000),
        };
        let mut env = HashMap::new();
        // Pre-existing root key must be overwritten by the scoped cred.
        env.insert("R2_ACCESS_KEY_ID".into(), "ROOT_KEY".into());
        env.insert("SWEEP_RUN_ID".into(), "sweep-1".into());

        inject_r2_cred_into_env(&mut env, &cred);

        assert_eq!(env["R2_ACCESS_KEY_ID"], "CHILD_AK");
        assert_eq!(env["R2_SECRET_ACCESS_KEY"], "CHILD_SK");
        // The session token MUST be present (key+secret alone would 403).
        assert_eq!(env["AWS_SESSION_TOKEN"], "SESSION_TOKEN");
        // Unrelated env entries are untouched.
        assert_eq!(env["SWEEP_RUN_ID"], "sweep-1");
    }

    #[test]
    fn injected_cred_serializes_into_container_group_env() {
        let cred = ScopedR2Cred {
            access_key_id: "AK".into(),
            secret_access_key: "SK".into(),
            session_token: "ST".into(),
            expires_at: None,
        };
        let mut env = HashMap::new();
        env.insert("SWEEP_RUN_ID".into(), "s2".into());
        inject_r2_cred_into_env(&mut env, &cred);

        let req = CreateContainerGroupRequest {
            name: "zen-sweep".into(),
            display_name: None,
            container: ContainerConfig {
                image: "ghcr.io/imazen/zen-sweep:latest".into(),
                resources: ResourceRequirements {
                    cpu: 4,
                    memory: 8192,
                    gpu_classes: vec!["gpu-1".into()],
                },
                command: None,
                environment_variables: env,
                registry_authentication: None,
            },
            replicas: 1,
            restart_policy: "always".into(),
            autostart_policy: true,
            queue_connection: None,
        };
        let v = serde_json::to_value(&req).unwrap();
        assert_eq!(
            v["container"]["environment_variables"]["AWS_SESSION_TOKEN"],
            "ST"
        );
        assert_eq!(
            v["container"]["environment_variables"]["R2_ACCESS_KEY_ID"],
            "AK"
        );
    }
}
