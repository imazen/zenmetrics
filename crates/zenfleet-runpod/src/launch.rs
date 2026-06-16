//! RunPod launcher-side provider — provisioning over the v1 REST API.
//!
//! This is the operator-workstation half (spec §1.3 / §1.10
//! "Launcher-side"): it provisions a GPU pod, pushes work (the worker
//! then PULLs chunks from R2 — see [`crate::queue`]), monitors, and
//! terminates. It is NOT baked into the deploy image and is NOT on the
//! hot path.
//!
//! ## Auth + endpoints (VERIFIED against the live OpenAPI 2026-05-27)
//!
//! RunPod has migrated from GraphQL to a REST API; REST is the
//! go-forward path (the GraphQL API still exists but the docs steer new
//! integrations to REST). Base URL: `https://rest.runpod.io/v1`. Auth
//! header: `Authorization: Bearer <api-key>` (the key from the RunPod
//! console → Settings → API Keys). All bodies are JSON.
//!
//! Endpoints used (from `GET https://rest.runpod.io/v1/openapi.json`):
//! - `POST   /pods`            — create (rent) a pod.
//! - `GET    /pods/{podId}`    — inspect a pod (status, cost, gpu).
//! - `POST   /pods/{podId}/stop` — stop (pause) a pod.
//! - `DELETE /pods/{podId}`    — terminate (delete) a pod.
//!
//! ## GPU-type discovery
//!
//! The REST v1 surface does NOT expose a `/gpu-types` list endpoint;
//! `gpuTypeIds` takes GPU **display-name** ids directly (e.g.
//! `"NVIDIA GeForce RTX 4090"`, `"NVIDIA L40S"`). Discovery of the exact
//! available ids is via the RunPod console or the legacy GraphQL
//! `gpuTypes` query — see `RUNPOD.md`. The launcher therefore takes the
//! GPU id string as-is; there is no `resolve_gpu_class` round-trip the
//! way Salad needs (Salad's REST surface DOES list classes).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

/// Public REST API base URL (v1).
pub const API_BASE: &str = "https://rest.runpod.io/v1";

/// A configured RunPod v1 REST client.
pub struct RunpodApi {
    base: String,
    api_key: String,
    http: reqwest::Client,
}

/// Body for `POST /pods` (subset of `PodCreateInput`, verified against
/// the live OpenAPI). Only the fields a sweep worker needs are modelled;
/// the rest take RunPod's documented defaults. Field names + casing
/// (camelCase) match the spec exactly.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PodCreateInput {
    /// Docker image tag for the baked zenfleet-sweep deploy image.
    pub image_name: String,
    /// GPU type ids, e.g. `["NVIDIA GeForce RTX 4090"]`. Display-name
    /// ids — see the module docs on GPU-type discovery.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub gpu_type_ids: Vec<String>,
    /// Number of GPUs (RunPod default 1).
    pub gpu_count: u32,
    /// `"SECURE"` (datacenter) or `"COMMUNITY"` (commodity, cheaper —
    /// the vast.ai-equivalent tier).
    pub cloud_type: String,
    /// Container disk in GiB (RunPod default 50).
    pub container_disk_in_gb: u32,
    /// Persistent volume size in GiB (RunPod default 20).
    pub volume_in_gb: u32,
    /// Persistent volume mount path (RunPod default `/workspace`).
    pub volume_mount_path: String,
    /// Pod name (shown in the console).
    pub name: String,
    /// Environment variables injected into the pod — the BYO R2/S3
    /// creds + sweep wiring (`SWEEP_RUN_ID`, `CHUNKS_R2`, …) the worker
    /// reads via [`crate::host::RunpodEnvCredentials`].
    #[serde(skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub env: std::collections::HashMap<String, String>,
    /// Override the container entrypoint, e.g.
    /// `["zenfleet-sweep"]`. Empty → use the image's own entrypoint.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub docker_entrypoint: Vec<String>,
    /// Override the container start command, e.g.
    /// `["worker", "--backend", "runpod"]`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub docker_start_cmd: Vec<String>,
    /// Optional private-registry auth id (created via
    /// `/containerregistryauth`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_registry_auth_id: Option<String>,
}

impl PodCreateInput {
    /// A sensible sweep-worker default: COMMUNITY cloud (cheap commodity
    /// GPUs, the vast.ai-equivalent tier), 1 GPU, RunPod's default disk
    /// sizes, `/workspace` scratch, no entrypoint override (the deploy
    /// image's entrypoint runs the pull worker per the BAKE-EVERYTHING
    /// rule).
    pub fn sweep_worker(image_name: impl Into<String>, gpu_type_id: impl Into<String>) -> Self {
        Self {
            image_name: image_name.into(),
            gpu_type_ids: vec![gpu_type_id.into()],
            gpu_count: 1,
            cloud_type: "COMMUNITY".to_string(),
            container_disk_in_gb: 50,
            volume_in_gb: 20,
            volume_mount_path: "/workspace".to_string(),
            name: "zenfleet-sweep".to_string(),
            env: std::collections::HashMap::new(),
            docker_entrypoint: Vec::new(),
            docker_start_cmd: Vec::new(),
            container_registry_auth_id: None,
        }
    }
}

/// Subset of the `Pod` resource returned on create / get (verified
/// against the live OpenAPI).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pod {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    /// The pod's desired status (`RUNNING` / `EXITED` / …).
    #[serde(default)]
    pub desired_status: Option<String>,
    #[serde(default)]
    pub cost_per_hr: Option<f64>,
    /// GPU info object (opaque here; surfaced for monitoring).
    #[serde(default)]
    pub gpu: Option<serde_json::Value>,
    #[serde(default)]
    pub machine_id: Option<String>,
}

impl RunpodApi {
    /// Build a client. `api_key` from the argument, else `$RUNPOD_API_KEY`
    /// or `~/.config/runpod/credentials`.
    pub fn new(api_key: Option<String>) -> Result<Self> {
        let api_key = match api_key {
            Some(k) => k,
            None => load_api_key().context("resolve RunPod API key")?,
        };
        let http = reqwest::Client::builder()
            .build()
            .context("build reqwest client")?;
        Ok(Self {
            base: API_BASE.to_string(),
            api_key,
            http,
        })
    }

    /// Override the API base (testing against a mock server).
    pub fn with_base(mut self, base: impl Into<String>) -> Self {
        self.base = base.into();
        self
    }

    fn url(&self, suffix: &str) -> String {
        format!("{}{}", self.base, suffix)
    }

    /// `POST /pods` — create (rent) a GPU pod running the deploy image.
    pub async fn create_pod(&self, req: &PodCreateInput) -> Result<Pod> {
        self.post_json(&self.url("/pods"), req).await
    }

    /// `GET /pods/{podId}` — inspect a pod (status, cost, gpu) for
    /// monitoring.
    pub async fn get_pod(&self, pod_id: &str) -> Result<Pod> {
        self.get_json(&self.url(&format!("/pods/{pod_id}"))).await
    }

    /// `POST /pods/{podId}/stop` — stop (pause) a pod without deleting it.
    pub async fn stop_pod(&self, pod_id: &str) -> Result<()> {
        self.post_empty(&self.url(&format!("/pods/{pod_id}/stop")))
            .await
    }

    /// `DELETE /pods/{podId}` — terminate (delete) a pod, releasing the
    /// GPU and stopping all billing.
    pub async fn terminate_pod(&self, pod_id: &str) -> Result<()> {
        let resp = self
            .http
            .delete(self.url(&format!("/pods/{pod_id}")))
            .bearer_auth(&self.api_key)
            .send()
            .await
            .with_context(|| format!("DELETE pods/{pod_id}"))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            bail!("DELETE pods/{pod_id}: HTTP {status}: {text}");
        }
        Ok(())
    }

    async fn post_json<B: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        url: &str,
        body: &B,
    ) -> Result<R> {
        let resp = self
            .http
            .post(url)
            .bearer_auth(&self.api_key)
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
            .bearer_auth(&self.api_key)
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
            .bearer_auth(&self.api_key)
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

/// Resolve the RunPod API key: `$RUNPOD_API_KEY`, else the
/// `~/.config/runpod/credentials` file (first non-empty,
/// non-`#`-prefixed line, or a `RUNPOD_API_KEY=...` line). Mirrors the
/// R2 + Salad credential-file conventions.
pub fn load_api_key() -> Result<String> {
    if let Ok(k) = std::env::var("RUNPOD_API_KEY")
        && !k.trim().is_empty()
    {
        return Ok(k.trim().to_string());
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = std::path::Path::new(&home).join(".config/runpod/credentials");
    let contents =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(v) = line.strip_prefix("RUNPOD_API_KEY=") {
            return Ok(v.trim().to_string());
        }
        return Ok(line.to_string());
    }
    bail!("no API key in {}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api() -> RunpodApi {
        RunpodApi {
            base: "https://api.example".into(),
            api_key: "k".into(),
            http: reqwest::Client::new(),
        }
    }

    #[test]
    fn url_builds() {
        let a = api();
        assert_eq!(a.url("/pods"), "https://api.example/pods");
        assert_eq!(a.url("/pods/abc/stop"), "https://api.example/pods/abc/stop");
    }

    #[test]
    fn api_base_is_rest_v1() {
        assert_eq!(API_BASE, "https://rest.runpod.io/v1");
    }

    #[test]
    fn sweep_worker_defaults_are_community_single_gpu() {
        let req = PodCreateInput::sweep_worker(
            "ghcr.io/imazen/zen-sweep:latest",
            "NVIDIA GeForce RTX 4090",
        );
        assert_eq!(req.cloud_type, "COMMUNITY");
        assert_eq!(req.gpu_count, 1);
        assert_eq!(req.gpu_type_ids, vec!["NVIDIA GeForce RTX 4090"]);
        assert_eq!(req.volume_mount_path, "/workspace");
    }

    #[test]
    fn pod_create_input_serializes_camel_case() {
        let mut req = PodCreateInput::sweep_worker("img:tag", "NVIDIA L40S");
        req.env.insert("SWEEP_RUN_ID".into(), "run-1".into());
        req.docker_start_cmd = vec!["worker".into(), "--backend".into(), "runpod".into()];
        let v = serde_json::to_value(&req).unwrap();
        // camelCase field names match the RunPod OpenAPI exactly.
        assert_eq!(v["imageName"], "img:tag");
        assert_eq!(v["gpuTypeIds"][0], "NVIDIA L40S");
        assert_eq!(v["gpuCount"], 1);
        assert_eq!(v["cloudType"], "COMMUNITY");
        assert_eq!(v["containerDiskInGb"], 50);
        assert_eq!(v["volumeMountPath"], "/workspace");
        assert_eq!(v["env"]["SWEEP_RUN_ID"], "run-1");
        assert_eq!(v["dockerStartCmd"][1], "--backend");
        // Empty optional vectors are omitted.
        assert!(v.get("dockerEntrypoint").is_none());
        assert!(v.get("containerRegistryAuthId").is_none());
    }

    #[test]
    fn pod_response_decodes() {
        let json = r#"{
            "id": "pod-xyz",
            "name": "zenfleet-sweep",
            "image": "img:tag",
            "desiredStatus": "RUNNING",
            "costPerHr": 0.34,
            "machineId": "m-1",
            "gpu": {"count": 1}
        }"#;
        let pod: Pod = serde_json::from_str(json).unwrap();
        assert_eq!(pod.id, "pod-xyz");
        assert_eq!(pod.desired_status.as_deref(), Some("RUNNING"));
        assert_eq!(pod.cost_per_hr, Some(0.34));
        assert_eq!(pod.machine_id.as_deref(), Some("m-1"));
    }

    #[test]
    fn pod_response_tolerates_minimal_body() {
        // Only `id` is required; everything else defaults to None.
        let pod: Pod = serde_json::from_str(r#"{"id":"p1"}"#).unwrap();
        assert_eq!(pod.id, "p1");
        assert!(pod.desired_status.is_none());
    }
}
