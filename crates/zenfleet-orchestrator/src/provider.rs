//! Provider-generic traits + value types for the `FleetSweep` driver.
//!
//! A compute provider (Salad / RunPod / Hetzner / Vast.ai) implements
//! [`ProviderHandle`] to plug into the generic poll loop. A blob store
//! (R2 / S3 / GCS) implements [`R2Operator`] for the operator-side
//! reads/writes the driver does (poll for sidecars, upload snapshots,
//! upload the final fleet_summary.json).
//!
//! Both traits use Rust 2024 edition's async-fn-in-trait — no
//! `async-trait` macro, no boxed futures, no GAT plumbing. The
//! `Future` types must be `Send` so the driver can run on a
//! multi-thread tokio runtime.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

/// One job's worth of provider-pushable input.
///
/// The driver hands these off to [`ProviderHandle::push_jobs`] and to
/// the TTL / speculative re-dispatch loop. The provider's queue
/// implementation forwards `payload` to the worker as the job body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueJob {
    /// Stable chunk identifier. Worker-side idempotency keys off this
    /// (the omni sidecar filename includes `chunk_id`).
    pub chunk_id: String,
    /// The provider-opaque payload (typically the chunk JSON the
    /// worker's inline pipeline parses).
    pub payload: JsonValue,
}

/// Provisioning request handed to [`ProviderHandle::provision`].
///
/// `extra` is a free-form JSON blob for provider-specific knobs that
/// shouldn't leak into the generic trait surface (e.g. Salad's
/// `gpu_class_ids` vector, RunPod's pod-template id, Hetzner's
/// cloud-init payload).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvisionSpec {
    /// Container image (registry path with tag).
    pub image: String,
    /// Provisioned replica count. The driver computes this from
    /// `SweepConfig.replicas × overshoot` capped to
    /// `provider_replica_quota`.
    pub replicas: u32,
    /// Human-readable GPU class names the provider scheduler should
    /// consider. Names are provider-defined.
    pub gpu_classes: Vec<String>,
    /// Environment variables to inject into the worker container.
    pub env: BTreeMap<String, String>,
    /// Per-replica $/hr upper bound for spend estimation. The
    /// provider need not enforce this; the driver uses it for the
    /// summary line.
    pub max_price_per_hour: f64,
    /// Provider-specific extras. Opaque to the driver.
    pub extra: JsonValue,
}

/// Per-instance status sample returned by
/// [`ProviderHandle::poll_instances`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceStatus {
    /// Provider's machine id (Salad's `machine_id`, RunPod's pod id,
    /// Hetzner's server id). Empty when not yet assigned.
    pub machine_id: String,
    /// Provider-defined state string (`allocating` / `downloading` /
    /// `creating` / `running` / `stopping` / `stopped` / `unknown`).
    pub state: String,
    /// Optional GPU class name (when the provider exposes it).
    pub gpu_class: Option<String>,
}

/// Snapshot of the container group as a whole. Returned alongside
/// per-instance statuses to feed into the launcher's R2 snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroupStatus {
    /// Top-level group state (`pending` / `running` / `stopped` / ...).
    pub state: String,
    /// Provider-specific instance-state counts (`{"running": 5,
    /// "downloading": 2, ...}`). Stitched into the R2 snapshot as-is.
    pub instance_status_counts: JsonValue,
    /// Optional URL the operator can click to see the group in the
    /// provider's portal.
    pub portal_url: Option<String>,
}

/// Opaque provider-side group identifier returned from
/// [`ProviderHandle::provision`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupId(pub String);

impl std::fmt::Display for GroupId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// The provider-specific surface. Implementors wrap their REST client
/// (Salad's `SaladApi`, RunPod's REST API, Hetzner's `hcloud`).
///
/// All methods are `async` and return `Send` futures so the driver
/// can run on a multi-thread runtime.
pub trait ProviderHandle: Send {
    /// Provision the container group + queue with the given spec.
    /// Returns the group identifier the driver passes to subsequent
    /// calls.
    fn provision(
        &mut self,
        spec: &ProvisionSpec,
    ) -> impl std::future::Future<Output = Result<GroupId>> + Send;

    /// Poll instance + group state. Best-effort; transient errors
    /// should be surfaced as `Err(_)` so the driver can log + retry
    /// on the next tick.
    fn poll_instances(
        &self,
        group: &GroupId,
    ) -> impl std::future::Future<Output = Result<(GroupStatus, Vec<InstanceStatus>)>> + Send;

    /// Tear down the group. Implementations should retry internally
    /// (the driver only calls this once at the end). Idempotent.
    fn teardown(&mut self, group: &GroupId)
    -> impl std::future::Future<Output = Result<()>> + Send;

    /// Push N jobs onto the provider's queue. The driver calls this
    /// once for the initial dispatch and again per chunk for
    /// TTL / speculative re-dispatch.
    fn push_jobs(
        &mut self,
        group: &GroupId,
        jobs: &[QueueJob],
    ) -> impl std::future::Future<Output = Result<()>> + Send;
}

/// Operator-side blob storage interface — the driver lists / reads /
/// writes objects in the bucket where workers drop their sidecars.
///
/// This is intentionally NOT [`zen_cloud_core::traits::BlobStorage`]:
/// that one is sync + worker-side. This one is async + operator-side
/// and only exposes the four call shapes the driver needs.
pub trait R2Operator: Send + Sync {
    /// List keys under `prefix` (no trailing wildcard).
    fn list(
        &self,
        bucket: &str,
        prefix: &str,
    ) -> impl std::future::Future<Output = Result<Vec<String>>> + Send;

    /// Upload `body` to `<bucket>/<key>`. PUT semantics.
    fn upload(
        &self,
        bucket: &str,
        key: &str,
        body: &[u8],
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// GET `<bucket>/<key>`, returning the raw bytes.
    fn get_bytes(
        &self,
        bucket: &str,
        key: &str,
    ) -> impl std::future::Future<Output = Result<Vec<u8>>> + Send;
}

/// Outcome of the polling loop. Returned to the launcher so it can
/// fold this into the final summary.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PollResult {
    /// Seconds from `t_post` to the first omni sidecar landing in R2.
    pub t_first_sidecar_secs: Option<f64>,
    /// Seconds to N distinct sidecars (proxy for "all replicas booted
    /// and processed at least one chunk").
    pub t_all_n_sidecars_secs: Option<f64>,
    /// Seconds to the last sidecar (or wall-cap hit).
    pub t_done_secs: Option<f64>,
    /// Distinct worker count observed (= unique omni sidecars).
    pub distinct_workers_observed: u32,
    /// Total omni sidecars at exit.
    pub omni_sidecars: u32,
    /// Total error sidecars at exit.
    pub error_sidecars: u32,
    /// Chunks re-pushed by the TTL re-dispatch logic.
    pub chunks_redispatched: u32,
    /// Chunks re-pushed by the speculative-execution logic.
    pub chunks_speculatively_dispatched: u32,
}

/// Final summary the driver emits. The launcher binary stitches this
/// + provider-specific fields into its stdout JSON.
#[derive(Debug, Clone, Default, Serialize)]
pub struct FleetSummary {
    /// Identifier used in R2 prefixes (`runs/<sweep_id>/`).
    pub sweep_id: String,
    /// Provider-side group name.
    pub group_name: String,
    /// Container image deployed.
    pub image: String,
    /// Provisioned replica count.
    pub replicas_provisioned: u32,
    /// Total chunks pushed (initial dispatch only).
    pub chunks: u32,
    /// Wall-clock seconds from group create to driver exit.
    pub wall_secs: f64,
    /// Whether the teardown call returned Ok.
    pub teardown_ok: bool,
    /// Whatever the poll loop produced.
    pub poll: PollResult,
}

/// Marker constants used by the driver to standardize R2 prefixes.
///
/// Layout under `runs/<sweep_id>/`:
///   - `chunks.jsonl`         — initial chunk list (uploaded by launcher)
///   - `boot/<machine>.txt`   — one per replica that booted
///   - `instances/<ts>.json`  — periodic snapshots from the driver
///   - `omni/<chunk>.parquet` — completed chunks (worker uploads)
///   - `errors/<chunk>.txt`   — error sidecars (worker uploads)
///   - `encoded/<chunk>/...`  — encoded artifacts (worker uploads)
///   - `fleet_summary.json`   — driver uploads at end
pub mod r2_layout {
    /// Per-sweep root prefix template. `format!(ROOT, sweep_id)` →
    /// `runs/<sweep_id>/`.
    pub const ROOT: &str = "runs/{sweep_id}/";

    /// Build the per-sweep prefix.
    pub fn sweep_prefix(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/")
    }

    /// Build the omni-sidecars prefix.
    pub fn omni_prefix(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/omni/")
    }

    /// Build the error-sidecars prefix.
    pub fn errors_prefix(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/errors/")
    }

    /// Build the instances-snapshots prefix.
    pub fn instances_prefix(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/instances/")
    }

    /// Build the boot-records prefix.
    pub fn boot_prefix(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/boot/")
    }

    /// Build the final fleet_summary key.
    pub fn fleet_summary_key(sweep_id: &str) -> String {
        format!("runs/{sweep_id}/fleet_summary.json")
    }
}

/// Re-export so the launcher can reference `Instant` without pulling
/// `std::time::Instant` directly (keeps imports consistent).
pub type DriverInstant = Instant;
