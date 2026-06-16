//! `HetznerProviderHandle` — wires Hetzner Cloud's REST API into the
//! provider-generic [`zenfleet_orchestrator::ProviderHandle`] trait.
//!
//! Hetzner-specific behavior lives here:
//!
//! * **No managed queue.** Workers poll R2 for jobs (`runs/<sweep_id>/
//!   queue/<chunk_id>.json`); `push_jobs` writes each `QueueJob.payload`
//!   to R2 and DELETEs them as the worker drains. The orchestrator's
//!   speculative-execution / TTL re-dispatch logic stays unchanged —
//!   re-pushing rewrites the queue file, and the worker idempotency
//!   on the omni sidecar reconciles dupes.
//! * **No container group.** Each replica is one server with
//!   `labels={group: <sweep_id>}`. `poll_instances` + `teardown` use
//!   `?label_selector=group=<sweep_id>` to scope.
//! * **No provider-supplied portal.** We point the operator at the
//!   Hetzner Cloud console.
//!
//! Everything else (poll loop, TTL, speculative, fleet_summary stitch)
//! lives in the orchestrator crate.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde_json::Value as JsonValue;
use zenfleet_orchestrator::{
    GroupId, GroupStatus, InstanceStatus, ProviderHandle, ProvisionSpec, QueueJob, R2Operator,
};
use zenfleet_salad::r2_ops::R2OperatorImpl;

use crate::api::{HetznerApi, HetznerLocation};
use crate::cloud_init::{WorkerBootstrap, build_user_data};

/// Hetzner-specific configuration that doesn't belong on the generic
/// [`ProvisionSpec`]. The launcher fills these in once.
pub struct HetznerProviderConfig {
    /// DNS-style group name (`hetzner-iter1-2026-05-28`); used as the
    /// server name prefix AND as the `group=<value>` label value.
    pub group_name: String,
    /// Server type slug — e.g. `cax21` (ARM 4 cores, €0.0152/hr) or
    /// `ccx13` (AMD 2 cores dedicated, €0.032/hr).
    pub server_type: String,
    /// Image slug — `ubuntu-24.04`. Hetzner serves both x86 and ARM
    /// variants under the same slug; the API selects the right one for
    /// the server_type's architecture.
    pub image: String,
    /// Location slug (`fsn1`, `nbg1`, `hel1`, `ash`, `hil`, `sin`).
    pub location: String,
    /// R2 bucket workers read/write sidecars + queue from.
    pub r2_bucket: String,
    /// R2 account id (for `<acct>.r2.cloudflarestorage.com` endpoint).
    pub r2_account_id: String,
    /// SSH key ids/names on the project. Empty = no inbound SSH (fine
    /// for production workers; the launcher logs the public IP for
    /// debug-via-Hetzner-console if a worker hangs).
    pub ssh_keys: Vec<String>,
    /// Optional registry username (for ghcr.io private images).
    pub registry_username: Option<String>,
    /// Optional registry password (or PAT).
    pub registry_password: Option<String>,
    /// Optional registry server (default `ghcr.io`).
    pub registry_server: Option<String>,
    /// R2 operator handle (the launcher's parent-cred-signed R2 client).
    /// We use it inside `push_jobs` to upload the queue files. Wrapped
    /// in an `Arc` so cloning the provider is cheap.
    pub r2: std::sync::Arc<R2OperatorImpl>,
    /// Optional SSH ed25519 public key (one line) injected into every
    /// worker's `/root/.ssh/authorized_keys` via cloud-init. Enables
    /// the launcher to SSH in and pull diagnostic logs when a worker
    /// reaches Hetzner-reported `running` but never claims a chunk.
    pub ssh_authorized_pubkey: Option<String>,
    /// Ordered `(server_type, location)` fallbacks tried when the
    /// primary placement returns HTTP 412 `resource_unavailable`
    /// (capacity drought — e.g. the 2026-06-12 Hetzner-wide CAX/ARM
    /// drought where `available` was empty in every datacenter).
    /// Empty = no fallback, fail on the first 412 (prior behavior).
    pub placement_fallbacks: Vec<(String, String)>,
}

impl HetznerProviderConfig {
    /// Build a config with default Ubuntu image + FSN1 location.
    pub fn new(
        group_name: impl Into<String>,
        server_type: impl Into<String>,
        r2_bucket: impl Into<String>,
        r2_account_id: impl Into<String>,
        r2: std::sync::Arc<R2OperatorImpl>,
    ) -> Self {
        Self {
            group_name: group_name.into(),
            server_type: server_type.into(),
            image: "ubuntu-24.04".into(),
            location: HetznerLocation::Fsn1.as_str().into(),
            r2_bucket: r2_bucket.into(),
            r2_account_id: r2_account_id.into(),
            ssh_keys: Vec::new(),
            registry_username: None,
            registry_password: None,
            registry_server: None,
            r2,
            ssh_authorized_pubkey: None,
            placement_fallbacks: Vec::new(),
        }
    }

    /// Override the location.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = location.into();
        self
    }

    /// Override the image slug.
    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = image.into();
        self
    }

    /// Set the SSH key list.
    pub fn with_ssh_keys(mut self, keys: Vec<String>) -> Self {
        self.ssh_keys = keys;
        self
    }

    /// Set the diagnostic SSH public key (injected into cloud-init).
    pub fn with_ssh_authorized_pubkey(mut self, pubkey: impl Into<String>) -> Self {
        self.ssh_authorized_pubkey = Some(pubkey.into());
        self
    }

    /// Set the `(server_type, location)` placement-fallback ladder.
    pub fn with_placement_fallbacks(mut self, fallbacks: Vec<(String, String)>) -> Self {
        self.placement_fallbacks = fallbacks;
        self
    }

    /// Set registry creds.
    pub fn with_registry_auth(
        mut self,
        user: impl Into<String>,
        pass: impl Into<String>,
        server: Option<String>,
    ) -> Self {
        self.registry_username = Some(user.into());
        self.registry_password = Some(pass.into());
        self.registry_server = server;
        self
    }
}

/// Hetzner implementation of [`ProviderHandle`].
pub struct HetznerProviderHandle {
    api: HetznerApi,
    cfg: HetznerProviderConfig,
    /// Server ids created in this group — populated by `provision`,
    /// used as a fast-path on teardown if the label selector ever
    /// disagrees with what we created.
    created_ids: Vec<i64>,
    /// R2 prefix the queue files land under, derived from
    /// `SWEEP_RUN_ID` at provision time (matches the cloud-init's
    /// CHUNKS_QUEUE_PREFIX). Set by `provision`; consumed by
    /// `push_jobs`.
    cached_sweep_prefix: Option<String>,
}

impl HetznerProviderHandle {
    /// Construct from a configured `HetznerApi` + Hetzner-specific knobs.
    pub fn new(api: HetznerApi, cfg: HetznerProviderConfig) -> Self {
        Self {
            api,
            cfg,
            created_ids: Vec::new(),
            cached_sweep_prefix: None,
        }
    }

    /// Hetzner Cloud console URL — the operator-facing equivalent of
    /// Salad's portal_url.
    pub fn console_url(&self) -> String {
        "https://console.hetzner.cloud/projects".to_string()
    }

    /// Borrow the API client (for pre-flight catalog probes etc.).
    pub fn api(&self) -> &HetznerApi {
        &self.api
    }

    /// Build the label selector string for this group.
    fn label_selector(&self) -> String {
        format!("group={}", self.cfg.group_name)
    }
}

impl ProviderHandle for HetznerProviderHandle {
    async fn provision(&mut self, spec: &ProvisionSpec) -> Result<GroupId> {
        // Build the cloud-init script the worker boots into. Reads R2
        // creds + sweep id from spec.env (the orchestrator already
        // plumbed scoped creds there via inject_r2_cred_into_env).
        let sweep_id = spec
            .env
            .get("SWEEP_RUN_ID")
            .context("SWEEP_RUN_ID missing from ProvisionSpec.env — orchestrator must inject it")?
            .clone();
        let r2_access = spec
            .env
            .get("R2_ACCESS_KEY_ID")
            .context("R2_ACCESS_KEY_ID missing from ProvisionSpec.env")?
            .clone();
        let r2_secret = spec
            .env
            .get("R2_SECRET_ACCESS_KEY")
            .context("R2_SECRET_ACCESS_KEY missing from ProvisionSpec.env")?
            .clone();
        let r2_session = spec
            .env
            .get("AWS_SESSION_TOKEN")
            .cloned()
            .or_else(|| spec.env.get("R2_SESSION_TOKEN").cloned())
            .context("AWS_SESSION_TOKEN missing from ProvisionSpec.env (scoped R2 creds REQUIRE a session token)")?;

        // Filter out the keys we already place explicitly so we don't
        // duplicate them via `extra_env`.
        let mut extra_env: std::collections::BTreeMap<String, String> = Default::default();
        for (k, v) in &spec.env {
            if matches!(
                k.as_str(),
                "SWEEP_RUN_ID"
                    | "R2_ACCESS_KEY_ID"
                    | "R2_SECRET_ACCESS_KEY"
                    | "AWS_SESSION_TOKEN"
                    | "R2_SESSION_TOKEN"
                    | "R2_ACCOUNT_ID"
                    | "BUCKET"
                    | "CHUNKS_R2"
                    | "CHUNKS_QUEUE_PREFIX"
                    | "WORKER_BACKEND"
                    | "RUST_LOG"
            ) {
                continue;
            }
            extra_env.insert(k.clone(), v.clone());
        }

        let bootstrap = WorkerBootstrap {
            image: spec.image.clone(),
            sweep_id: sweep_id.clone(),
            r2_account_id: self.cfg.r2_account_id.clone(),
            r2_bucket: self.cfg.r2_bucket.clone(),
            r2_access_key_id: r2_access,
            r2_secret_access_key: r2_secret,
            r2_session_token: r2_session,
            registry_username: self.cfg.registry_username.clone(),
            registry_password: self.cfg.registry_password.clone(),
            registry_server: self.cfg.registry_server.clone(),
            extra_env,
            chunks_queue_prefix: format!("runs/{sweep_id}/queue/"),
            ssh_authorized_pubkey: self.cfg.ssh_authorized_pubkey.clone(),
        };
        let user_data = build_user_data(&bootstrap);

        // Provision N replicas, one server each.
        let mut labels = HashMap::new();
        labels.insert("group".to_string(), self.cfg.group_name.clone());
        labels.insert("sweep_id".to_string(), sweep_id.clone());

        // Placement ladder: primary `(server_type, location)` first,
        // then any configured fallbacks. The index is STICKY across
        // replicas — capacity exhaustion (HTTP 412 resource_unavailable)
        // is a property of the (type, location) pair, so a placement
        // that 412'd once is never retried within this provision call.
        let placements: Vec<(String, String)> =
            std::iter::once((self.cfg.server_type.clone(), self.cfg.location.clone()))
                .chain(self.cfg.placement_fallbacks.iter().cloned())
                .collect();
        let mut placement_idx = 0usize;

        for i in 0..spec.replicas {
            let name = format!("{}-{:03}", self.cfg.group_name, i);
            let srv = loop {
                let (server_type, location) = &placements[placement_idx];
                tracing::info!(
                    server_type = %server_type,
                    location = %location,
                    name = %name,
                    ladder_rung = placement_idx,
                    "creating Hetzner server"
                );
                match self
                    .api
                    .create_server(
                        &name,
                        server_type,
                        &self.cfg.image,
                        location,
                        &user_data,
                        &self.cfg.ssh_keys,
                        &labels,
                    )
                    .await
                {
                    Ok(srv) => break srv,
                    Err(e)
                        if is_placement_unavailable(&e) && placement_idx + 1 < placements.len() =>
                    {
                        tracing::warn!(
                            server_type = %server_type,
                            location = %location,
                            error = %format!("{e:#}"),
                            "placement unavailable (HTTP 412); advancing ladder"
                        );
                        placement_idx += 1;
                    }
                    Err(e) => {
                        let (st, loc) = &placements[placement_idx];
                        return Err(e).with_context(|| {
                            format!(
                                "create Hetzner server {name} ({st} in {loc}; \
                                 ladder rung {} of {})",
                                placement_idx + 1,
                                placements.len()
                            )
                        });
                    }
                }
            };
            tracing::info!(
                server_id = %srv.id,
                ipv4 = %srv.ipv4().unwrap_or_default(),
                "Hetzner server created"
            );
            self.created_ids.push(srv.id);
        }
        self.cached_sweep_prefix = Some(format!("runs/{sweep_id}/queue/"));
        Ok(GroupId(self.cfg.group_name.clone()))
    }

    async fn poll_instances(&self, _group: &GroupId) -> Result<(GroupStatus, Vec<InstanceStatus>)> {
        let servers = self
            .api
            .list_servers_by_label(&self.label_selector())
            .await
            .context("list_servers_by_label")?;

        // Build instance status list.
        let mut instances = Vec::with_capacity(servers.len());
        let mut counts: std::collections::BTreeMap<String, u64> = Default::default();
        for s in &servers {
            let state = s.parsed_status().as_orchestrator_state();
            *counts.entry(state.to_string()).or_insert(0) += 1;
            instances.push(InstanceStatus {
                machine_id: s.id.to_string(),
                state: state.to_string(),
                gpu_class: None,
            });
        }

        // Group-state: "stopped" iff every server is stopping/off (or
        // the list is empty); "running" iff at least one is running;
        // else "pending".
        let group_state = if servers.is_empty() {
            "stopped".to_string()
        } else if servers
            .iter()
            .all(|s| matches!(s.parsed_status().as_orchestrator_state(), "stopping"))
        {
            "stopped".to_string()
        } else if servers
            .iter()
            .any(|s| matches!(s.parsed_status().as_orchestrator_state(), "running"))
        {
            "running".to_string()
        } else {
            "pending".to_string()
        };

        let counts_json = serde_json::to_value(&counts).unwrap_or(JsonValue::Null);

        Ok((
            GroupStatus {
                state: group_state,
                instance_status_counts: counts_json,
                portal_url: Some(self.console_url()),
            },
            instances,
        ))
    }

    async fn teardown(&mut self, _group: &GroupId) -> Result<()> {
        // Source of truth: the label selector. We accumulate the union
        // of `self.created_ids` + the live list and DELETE everything.
        let live = self
            .api
            .list_servers_by_label(&self.label_selector())
            .await
            .context("list_servers_by_label (teardown)")?;
        let mut to_delete: Vec<i64> = live.iter().map(|s| s.id).collect();
        for id in &self.created_ids {
            if !to_delete.contains(id) {
                to_delete.push(*id);
            }
        }
        let mut errors: Vec<String> = Vec::new();
        for id in &to_delete {
            tracing::info!(server_id = %id, "deleting Hetzner server");
            if let Err(e) = self.api.delete_server(*id).await {
                let msg = format!("delete server {id}: {e:#}");
                tracing::warn!("{msg}");
                errors.push(msg);
            }
        }

        // Verify zero remaining via re-list.
        let remaining = self
            .api
            .list_servers_by_label(&self.label_selector())
            .await
            .unwrap_or_default();
        if !remaining.is_empty() {
            let ids: Vec<String> = remaining.iter().map(|s| s.id.to_string()).collect();
            anyhow::bail!(
                "teardown left {} servers alive in group {}: ids=[{}]",
                remaining.len(),
                self.cfg.group_name,
                ids.join(",")
            );
        }
        if !errors.is_empty() {
            anyhow::bail!("delete errors during teardown: {}", errors.join("; "));
        }
        Ok(())
    }

    async fn push_jobs(&mut self, _group: &GroupId, jobs: &[QueueJob]) -> Result<()> {
        // Workers poll R2 at `runs/<sweep_id>/queue/<chunk_id>.json`.
        // Push = write one JSON object per chunk under that prefix.
        //
        // We use the FIRST job's chunk_id's prefix to determine the
        // sweep id — pulling it from the cached prefix the provider
        // stored at provision-time. This keeps push_jobs consistent
        // with the cloud-init's CHUNKS_QUEUE_PREFIX env var (which is
        // built from SWEEP_RUN_ID, not from the group_name).
        //
        // The sweep_id was injected into ProvisionSpec.env by the
        // launcher; we cached it in `provision` so push_jobs (called
        // after provision) sees it. If provision wasn't called yet
        // (e.g. test harness pushing before provision), fall back to
        // group_name so writes at least land somewhere predictable.
        let sweep_prefix = self
            .cached_sweep_prefix
            .clone()
            .unwrap_or_else(|| format!("runs/{}/queue/", self.cfg.group_name));
        for j in jobs {
            let key = format!("{sweep_prefix}{}.json", j.chunk_id);
            // The payload IS the chunk JSON; the worker reads it raw.
            let body = serde_json::to_vec(&j.payload)
                .with_context(|| format!("serialize chunk payload for {}", j.chunk_id))?;
            self.cfg
                .r2
                .upload(&self.cfg.r2_bucket, &key, &body)
                .await
                .with_context(|| {
                    format!("upload chunk queue file s3://{}/{key}", self.cfg.r2_bucket)
                })?;
        }
        Ok(())
    }
}

/// Whether an error from [`HetznerApi::create_server`] is the HTTP 412
/// `resource_unavailable` placement failure ("error during placement" —
/// no capacity for that server_type in that location). Matched on the
/// error-body code Hetzner returns, which `create_server` embeds in its
/// bail string; only this error advances the placement ladder — auth,
/// quota, and validation errors must keep failing fast.
fn is_placement_unavailable(err: &anyhow::Error) -> bool {
    format!("{err:#}").contains("resource_unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn placement_unavailable_matches_hetzner_412_body() {
        // Shape produced by HetznerApi::create_server on the real
        // 2026-06-12 capacity-drought failure (iter-6 launch log).
        let e = anyhow::anyhow!(
            "POST /servers (g-000): HTTP 412 Precondition Failed: \
             {{\"error\":{{\"code\":\"resource_unavailable\",\
             \"message\":\"error during placement\",\"details\":{{}}}}}}"
        );
        assert!(is_placement_unavailable(&e));
    }

    #[test]
    fn placement_unavailable_ignores_other_errors() {
        for msg in [
            "POST /servers (g-000): HTTP 401 Unauthorized: {\"error\":{\"code\":\"unauthorized\"}}",
            "POST /servers (g-000): HTTP 422: {\"error\":{\"code\":\"resource_limit_exceeded\"}}",
            "POST /servers: connection reset by peer",
        ] {
            let e = anyhow::anyhow!("{msg}");
            assert!(!is_placement_unavailable(&e), "false positive on: {msg}");
        }
    }

    #[test]
    fn placement_unavailable_sees_through_context_chain() {
        let inner = anyhow::anyhow!(
            "POST /servers (g-001): HTTP 412 Precondition Failed: \
             {{\"error\":{{\"code\":\"resource_unavailable\"}}}}"
        );
        let wrapped = inner.context("create Hetzner server g-001 (cax11 in fsn1)");
        assert!(is_placement_unavailable(&wrapped));
    }

    fn mock_spec() -> ProvisionSpec {
        let mut env = BTreeMap::new();
        env.insert("SWEEP_RUN_ID".into(), "sweep-test".into());
        env.insert("R2_ACCESS_KEY_ID".into(), "key".into());
        env.insert("R2_SECRET_ACCESS_KEY".into(), "secret".into());
        env.insert("AWS_SESSION_TOKEN".into(), "session".into());
        env.insert("METRICS".into(), "ssim2-gpu".into());
        ProvisionSpec {
            image: "ghcr.io/imazen/zenmetrics-sweep-salad:v6".into(),
            replicas: 5,
            gpu_classes: vec![],
            env,
            max_price_per_hour: 0.02,
            extra: JsonValue::Null,
        }
    }

    /// Smoke test: the WorkerBootstrap we'd hand to `build_user_data`
    /// from a real provision call carries every required env var.
    #[test]
    fn provision_builds_correct_bootstrap_from_spec() {
        let spec = mock_spec();
        // We can't call provision() without a real API mock + R2, but
        // we can verify the env-extraction logic matches: every key
        // in mock_spec().env should be either explicitly placed OR
        // forwarded via extra_env.
        let sweep_id = spec.env.get("SWEEP_RUN_ID").unwrap();
        let extra: std::collections::BTreeMap<String, String> = spec
            .env
            .iter()
            .filter(|(k, _)| {
                !matches!(
                    k.as_str(),
                    "SWEEP_RUN_ID"
                        | "R2_ACCESS_KEY_ID"
                        | "R2_SECRET_ACCESS_KEY"
                        | "AWS_SESSION_TOKEN"
                        | "R2_SESSION_TOKEN"
                        | "R2_ACCOUNT_ID"
                        | "BUCKET"
                        | "CHUNKS_R2"
                        | "CHUNKS_QUEUE_PREFIX"
                        | "WORKER_BACKEND"
                        | "RUST_LOG"
                )
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(sweep_id, "sweep-test");
        assert!(extra.contains_key("METRICS"));
        assert!(!extra.contains_key("R2_ACCESS_KEY_ID"));
    }
}
