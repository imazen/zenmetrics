//! `SaladProviderHandle` — wires Salad's REST API into the
//! provider-generic [`zenfleet_orchestrator::ProviderHandle`] trait.
//!
//! Only Salad-specific behavior lives here:
//!   * `resources.gpu_classes` ID resolution (names come from the
//!     driver; the launcher binary resolves names → ids before
//!     handing the spec over).
//!   * `queue_connection { path, port, queue_name }` — Salad routes
//!     queue jobs to the sidecar on a per-group basis.
//!   * `instance_status_counts` JSON shape — exposed as-is in
//!     [`GroupStatus`].
//!
//! Everything else (poll loop, TTL, speculative, fleet_summary
//! stitch) lives in the orchestrator crate.

#![cfg(feature = "launcher")]

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use zenfleet_orchestrator::{
    GroupId, GroupStatus, InstanceStatus, ProviderHandle, ProvisionSpec, QueueJob,
};

use crate::launch::{
    ContainerConfig, CreateContainerGroupRequest, CreateQueueJobRequest, CreateQueueRequest,
    QueueConnection, RegistryAuth, ResourceRequirements, SaladApi,
};

/// Salad-specific configuration that doesn't belong on the generic
/// [`ProvisionSpec`]. The launcher fills these in once, before
/// constructing the handle.
pub struct SaladProviderConfig {
    /// Salad organization slug.
    pub organization: String,
    /// Salad project slug.
    pub project: String,
    /// Container group display name (DNS-style).
    pub group_name: String,
    /// Queue name (managed Salad job queue).
    pub queue_name: String,
    /// Pre-resolved GPU class ids (from `SaladApi::resolve_gpu_class`
    /// or `gpu_classes_under_price`). Stays parallel with
    /// `ProvisionSpec::gpu_classes` (which carries the human names
    /// for the summary).
    pub gpu_class_ids: Vec<String>,
    /// CPU cores per replica.
    pub cpu: u32,
    /// Memory per replica in MiB.
    pub memory_mib: u32,
    /// Optional registry auth (username / password).
    pub registry_auth: Option<RegistryAuth>,
    /// Restart policy (`"always"` / `"on_failure"` / `"never"`).
    pub restart_policy: String,
    /// Whether the group autostarts on create.
    pub autostart: bool,
    /// Queue HTTP path (worker's local job receiver).
    pub queue_path: String,
    /// Queue HTTP port.
    pub queue_port: u16,
}

/// Salad implementation of [`ProviderHandle`].
pub struct SaladProviderHandle {
    api: SaladApi,
    cfg: SaladProviderConfig,
}

impl SaladProviderHandle {
    /// Construct from a configured `SaladApi` + Salad-specific knobs.
    pub fn new(api: SaladApi, cfg: SaladProviderConfig) -> Self {
        Self { api, cfg }
    }

    /// Portal URL for the operator to click into.
    pub fn portal_url(&self) -> String {
        format!(
            "https://portal.salad.com/organizations/{}/projects/{}/containers/{}",
            self.cfg.organization, self.cfg.project, self.cfg.group_name
        )
    }

    /// Idempotent queue create. Salad's POST returns 409 if the queue
    /// already exists; we treat that as success.
    pub async fn ensure_queue_exists(&self) -> Result<()> {
        let req = CreateQueueRequest {
            name: self.cfg.queue_name.clone(),
            display_name: Some(format!("zen-salad-sweep {}", self.cfg.group_name)),
            description: Some("zen-salad-sweep managed queue".into()),
        };
        match self.api.create_queue(&req).await {
            Ok(_) => Ok(()),
            Err(e) => {
                let msg = format!("{e:#}");
                if msg.contains("409") || msg.to_lowercase().contains("already exists") {
                    Ok(())
                } else {
                    Err(e.context("create queue"))
                }
            }
        }
    }

    /// Borrow the underlying API (for pre-flight class resolution etc.
    /// that the launcher does outside the trait surface).
    pub fn api(&self) -> &SaladApi {
        &self.api
    }
}

impl ProviderHandle for SaladProviderHandle {
    async fn provision(&mut self, spec: &ProvisionSpec) -> Result<GroupId> {
        // The queue must exist before the container group attaches
        // to it. Idempotent.
        self.ensure_queue_exists()
            .await
            .context("ensure_queue_exists")?;

        // Map BTreeMap → HashMap for the request type.
        let env: std::collections::HashMap<String, String> = spec
            .env
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let req = CreateContainerGroupRequest {
            name: self.cfg.group_name.clone(),
            display_name: Some(format!("zen-salad-sweep {}", self.cfg.group_name)),
            container: ContainerConfig {
                image: spec.image.clone(),
                resources: ResourceRequirements {
                    cpu: self.cfg.cpu,
                    memory: self.cfg.memory_mib,
                    gpu_classes: self.cfg.gpu_class_ids.clone(),
                },
                command: None,
                environment_variables: env,
                registry_authentication: self.cfg.registry_auth.clone(),
            },
            replicas: spec.replicas,
            restart_policy: self.cfg.restart_policy.clone(),
            autostart_policy: self.cfg.autostart,
            queue_connection: Some(QueueConnection {
                path: self.cfg.queue_path.clone(),
                port: self.cfg.queue_port,
                queue_name: self.cfg.queue_name.clone(),
            }),
        };
        let _group = self
            .api
            .create_container_group(&req)
            .await
            .with_context(|| {
                format!(
                    "POST container group {} (image={})",
                    self.cfg.group_name, spec.image
                )
            })?;
        Ok(GroupId(self.cfg.group_name.clone()))
    }

    async fn poll_instances(&self, group: &GroupId) -> Result<(GroupStatus, Vec<InstanceStatus>)> {
        let cg = self.api.get_container_group(&group.0).await.ok();
        let state = cg
            .as_ref()
            .and_then(|g| g.current_state.as_ref())
            .and_then(|s| s.get("status"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let counts = cg
            .as_ref()
            .and_then(|g| g.current_state.as_ref())
            .and_then(|s| s.get("instance_status_counts"))
            .cloned()
            .unwrap_or_else(|| JsonValue::Object(Default::default()));
        let instances_val = self.api.list_container_group_instances(&group.0).await.ok();

        let mut instances: Vec<InstanceStatus> = Vec::new();
        if let Some(v) = instances_val {
            let arr = v
                .as_array()
                .cloned()
                .or_else(|| v.get("instances").and_then(|x| x.as_array().cloned()));
            if let Some(arr) = arr {
                for inst in arr {
                    let machine_id = inst
                        .get("machine_id")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let st = inst
                        .get("state")
                        .and_then(|x| x.as_str())
                        .unwrap_or("")
                        .to_string();
                    let gpu_class = inst
                        .get("gpu_class")
                        .and_then(|x| x.as_str())
                        .map(|s| s.to_string());
                    instances.push(InstanceStatus {
                        machine_id,
                        state: st,
                        gpu_class,
                    });
                }
            }
        }

        Ok((
            GroupStatus {
                state,
                instance_status_counts: counts,
                portal_url: Some(self.portal_url()),
            },
            instances,
        ))
    }

    async fn teardown(&mut self, group: &GroupId) -> Result<()> {
        self.api
            .stop_container_group(&group.0)
            .await
            .with_context(|| format!("stop container group {}", group.0))
    }

    async fn push_jobs(&mut self, _group: &GroupId, jobs: &[QueueJob]) -> Result<()> {
        for j in jobs {
            self.api
                .push_job(
                    &self.cfg.queue_name,
                    &CreateQueueJobRequest {
                        input: j.payload.clone(),
                        metadata: None,
                    },
                )
                .await
                .with_context(|| format!("push chunk {}", j.chunk_id))?;
        }
        Ok(())
    }
}
