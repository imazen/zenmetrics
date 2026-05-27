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
}
