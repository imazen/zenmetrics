//! `FleetSweep<P, R>` — the provider-generic poll loop.
//!
//! The driver alternates between three things on each tick:
//!
//! 1. Ask the [`R2Operator`] for the current omni / error sidecar
//!    keys + (optionally) read prior snapshots — these reflect what
//!    workers have already produced.
//! 2. Ask the [`ProviderHandle`] for the current per-instance state
//!    (used for the snapshot JSON + for the `state == "stopped"`
//!    early-exit). Upload a snapshot JSON to R2 so a forensic
//!    session can replay the run.
//! 3. Feed the latest completed-chunk set into [`SpeculativeState`]
//!    and call [`ttl_redispatch_decisions`] for the TTL path. Both
//!    re-dispatch paths push back through `ProviderHandle::push_jobs`.
//!
//! Exit conditions:
//!   - All expected chunks completed (`omni + error >= expected`)
//!   - Container group reached `state == "stopped"` (provider already
//!     drained → no more sidecars coming)
//!   - Wall-time cap hit
//!
//! After the loop, the driver stitches `fleet_summary.json` from the
//! R2 artifacts and uploads it.

use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use tokio::time::sleep;

use crate::provider::{
    FleetSummary, GroupId, GroupStatus, PollResult, ProviderHandle, ProvisionSpec, QueueJob,
    R2Operator, r2_layout,
};
use crate::{
    SpeculativeState, SweepConfig, compute_provisioned_replicas, ttl_redispatch_decisions,
};

/// Provider-generic fleet sweep.
///
/// Created with [`FleetSweep::new`], then `run()` executes the
/// provision → push → poll → teardown lifecycle, returning the final
/// summary.
pub struct FleetSweep<P: ProviderHandle, R: R2Operator> {
    provider: P,
    storage: R,
    config: SweepConfig,
    /// Bucket workers + the driver share for sidecars / snapshots.
    bucket: String,
    /// Unique sweep id (`runs/<sweep_id>/...`).
    sweep_id: String,
    /// Provider-side group name (for the summary).
    group_name: String,
    /// Image the provider should deploy.
    image: String,
    /// GPU class names to pass to the provider.
    gpu_classes: Vec<String>,
    /// Env vars to inject into the worker container.
    env: BTreeMap<String, String>,
    /// Per-replica $/hr ceiling.
    max_price_per_hour: f64,
    /// Provider-specific extras (Salad's `gpu_class_ids`, queue_name,
    /// path, port; RunPod's pod-template; etc.).
    provision_extra: JsonValue,
    /// Wall-time cap for the poll loop.
    max_wall_secs: u64,
    /// Polling interval.
    poll_secs: u64,
    /// Skip teardown (debug flag — billing exposure).
    keep_running: bool,
}

impl<P: ProviderHandle, R: R2Operator> FleetSweep<P, R> {
    /// Construct a new sweep driver. All R2 writes happen under
    /// `runs/<sweep_id>/` in `bucket`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provider: P,
        storage: R,
        config: SweepConfig,
        bucket: impl Into<String>,
        sweep_id: impl Into<String>,
        group_name: impl Into<String>,
        image: impl Into<String>,
        gpu_classes: Vec<String>,
        env: BTreeMap<String, String>,
        max_price_per_hour: f64,
        provision_extra: JsonValue,
        max_wall_secs: u64,
        poll_secs: u64,
        keep_running: bool,
    ) -> Self {
        Self {
            provider,
            storage,
            config,
            bucket: bucket.into(),
            sweep_id: sweep_id.into(),
            group_name: group_name.into(),
            image: image.into(),
            gpu_classes,
            env,
            max_price_per_hour,
            provision_extra,
            max_wall_secs,
            poll_secs,
            keep_running,
        }
    }

    /// Execute one sweep cycle. Returns the final summary regardless
    /// of teardown outcome (callers check `summary.teardown_ok`).
    ///
    /// `chunks` is the initial dispatch set. The driver records each
    /// `chunk_id` for speculative-execution tracking and re-pushes
    /// them through `ProviderHandle::push_jobs` for TTL / speculative
    /// re-dispatch.
    pub async fn run(mut self, chunks: Vec<QueueJob>) -> Result<FleetSummary> {
        let t_start = Instant::now();

        // ─ Provision ──────────────────────────────────────────────
        let replicas_provisioned = compute_provisioned_replicas(
            self.config.replicas,
            self.config.replicas_overshoot,
            self.config.provider_replica_quota,
        );
        let spec = ProvisionSpec {
            image: self.image.clone(),
            replicas: replicas_provisioned,
            gpu_classes: self.gpu_classes.clone(),
            env: self.env.clone(),
            max_price_per_hour: self.max_price_per_hour,
            extra: self.provision_extra.clone(),
        };
        let t_post = Instant::now();
        let group: GroupId = self
            .provider
            .provision(&spec)
            .await
            .context("provider.provision")?;
        eprintln!("[driver] group provisioned: {group}");

        // ─ Initial dispatch ───────────────────────────────────────
        self.provider
            .push_jobs(&group, &chunks)
            .await
            .context("provider.push_jobs (initial)")?;

        // ─ Poll loop ──────────────────────────────────────────────
        let poll = self
            .poll_until_done(&group, &chunks, t_post, replicas_provisioned)
            .await
            .unwrap_or_else(|e| {
                eprintln!("[driver] poll loop returned error: {e:#}");
                PollResult::default()
            });

        // ─ Fleet summary (best-effort) ────────────────────────────
        if let Err(e) = self.build_and_upload_fleet_summary(t_post).await {
            eprintln!("[driver] fleet_summary build/upload failed (non-fatal): {e:#}");
        }

        // ─ Teardown ───────────────────────────────────────────────
        let teardown_ok = if !self.keep_running {
            let mut attempts: u32 = 0;
            let mut ok = false;
            while attempts < 3 {
                match self.provider.teardown(&group).await {
                    Ok(()) => {
                        eprintln!("[driver] teardown OK");
                        ok = true;
                        break;
                    }
                    Err(e) => {
                        eprintln!(
                            "[driver] teardown failed (attempt {}): {:#}",
                            attempts + 1,
                            e
                        );
                        attempts += 1;
                        sleep(Duration::from_secs(3)).await;
                    }
                }
            }
            ok
        } else {
            eprintln!("[driver] keep_running=true; SKIPPING teardown");
            false
        };

        let wall = t_start.elapsed().as_secs_f64();
        Ok(FleetSummary {
            sweep_id: self.sweep_id,
            group_name: self.group_name,
            image: self.image,
            replicas_provisioned,
            chunks: chunks.len() as u32,
            wall_secs: wall,
            teardown_ok,
            poll,
        })
    }

    async fn poll_until_done(
        &mut self,
        group: &GroupId,
        chunks: &[QueueJob],
        t_post: Instant,
        replicas_provisioned: u32,
    ) -> Result<PollResult> {
        let omni_prefix = r2_layout::omni_prefix(&self.sweep_id);
        let error_prefix = r2_layout::errors_prefix(&self.sweep_id);
        let instances_prefix = r2_layout::instances_prefix(&self.sweep_id);
        eprintln!("[poll]   watching s3://{}/{omni_prefix}", self.bucket);
        eprintln!("[poll]   snapshots s3://{}/{instances_prefix}", self.bucket);

        let mut out = PollResult::default();
        let cap = Duration::from_secs(self.max_wall_secs);
        let interval = Duration::from_secs(self.poll_secs.max(2));
        let expected_chunks = chunks.len();

        let chunk_ids: Vec<String> = chunks.iter().map(|c| c.chunk_id.clone()).collect();
        let mut already_redispatched: HashSet<String> = HashSet::new();
        let mut spec_state = SpeculativeState::new();
        for cid in &chunk_ids {
            spec_state.record_dispatched(cid, 0.0);
        }

        let spec_cfg = self.config.speculative.clone();
        if spec_cfg.enabled {
            eprintln!(
                "[spec] enabled (factor={:.2} min_completed={} cap_per_chunk={})",
                spec_cfg.straggler_factor,
                spec_cfg.min_completed_for_stats,
                spec_cfg.speculation_cap_per_chunk,
            );
        } else {
            eprintln!("[spec] disabled");
        }

        let mut tick: u32 = 0;
        loop {
            let elapsed = t_post.elapsed();
            if elapsed >= cap {
                eprintln!(
                    "[poll] wall-time cap {}s hit; stopping poll loop",
                    self.max_wall_secs
                );
                out.t_done_secs.get_or_insert(elapsed.as_secs_f64());
                break;
            }
            tick += 1;

            let omni = self
                .storage
                .list(&self.bucket, &omni_prefix)
                .await
                .unwrap_or_default();
            let errs = self
                .storage
                .list(&self.bucket, &error_prefix)
                .await
                .unwrap_or_default();
            out.omni_sidecars = omni.len() as u32;
            out.error_sidecars = errs.len() as u32;
            out.distinct_workers_observed = out.omni_sidecars;

            if !omni.is_empty() && out.t_first_sidecar_secs.is_none() {
                out.t_first_sidecar_secs = Some(elapsed.as_secs_f64());
                eprintln!(
                    "[poll] FIRST sidecar at t={:.1}s ({})",
                    elapsed.as_secs_f64(),
                    omni[0]
                );
            }
            if (omni.len() as u32) >= replicas_provisioned && out.t_all_n_sidecars_secs.is_none() {
                out.t_all_n_sidecars_secs = Some(elapsed.as_secs_f64());
                eprintln!(
                    "[poll] all-N sidecars at t={:.1}s (n={} >= replicas_provisioned={})",
                    elapsed.as_secs_f64(),
                    omni.len(),
                    replicas_provisioned
                );
            }

            // Completed set (chunk_id stems extracted from omni keys).
            let mut completed: HashSet<String> = HashSet::new();
            for k in &omni {
                if let Some(stem) = k
                    .rsplit('/')
                    .next()
                    .and_then(|f| f.strip_suffix(".parquet"))
                {
                    completed.insert(stem.to_string());
                }
            }
            let elapsed_secs = elapsed.as_secs_f64();
            for cid in &completed {
                spec_state.record_completed(cid, elapsed_secs);
            }

            // TTL re-dispatch.
            let ttl_redispatches = ttl_redispatch_decisions(
                elapsed_secs,
                &self.config,
                &chunk_ids,
                &completed,
                &mut already_redispatched,
            );
            for cid in &ttl_redispatches {
                if let Some(chunk) = chunks.iter().find(|c| &c.chunk_id == cid) {
                    let one = [chunk.clone()];
                    match self.provider.push_jobs(group, &one).await {
                        Ok(_) => {
                            out.chunks_redispatched += 1;
                            eprintln!(
                                "[poll] TTL re-dispatch: chunk_id={} (elapsed={:.1}s > ttl={}s)",
                                cid, elapsed_secs, self.config.chunk_ttl_secs
                            );
                        }
                        Err(e) => {
                            eprintln!("[poll] TTL re-dispatch FAILED chunk_id={cid}: {e:#}");
                        }
                    }
                }
            }

            // Speculative re-dispatch.
            if spec_cfg.enabled {
                let p95 = spec_state.p95_completion_secs();
                for cid in &chunk_ids {
                    if let Some(spec_elapsed) =
                        spec_state.decide_speculative(cid, elapsed_secs, &spec_cfg)
                    {
                        if let Some(chunk) = chunks.iter().find(|c| &c.chunk_id == cid) {
                            let one = [chunk.clone()];
                            match self.provider.push_jobs(group, &one).await {
                                Ok(_) => {
                                    spec_state.record_speculative_dispatched(cid);
                                    out.chunks_speculatively_dispatched += 1;
                                    eprintln!(
                                        "[spec] re-dispatch: chunk_id={} elapsed={:.1}s p95={:.1}s factor={:.2}",
                                        cid,
                                        spec_elapsed,
                                        p95.unwrap_or(0.0),
                                        spec_cfg.straggler_factor,
                                    );
                                }
                                Err(e) => {
                                    eprintln!("[spec] re-dispatch FAILED chunk_id={cid}: {e:#}");
                                }
                            }
                        }
                    }
                }
            }

            // Provider status snapshot.
            let (group_status, instances) = self
                .provider
                .poll_instances(group)
                .await
                .unwrap_or_else(|_| (GroupStatus::default(), Vec::new()));

            let ts_unix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let snapshot = json!({
                "unix_ts": ts_unix,
                "elapsed_secs": elapsed_secs,
                "tick": tick,
                "state": group_status.state,
                "instance_status_counts": group_status.instance_status_counts,
                "instances": instances,
                "omni_count": omni.len(),
                "error_count": errs.len(),
            });
            let snapshot_bytes = serde_json::to_vec(&snapshot).unwrap_or_default();
            let snapshot_key = format!("{instances_prefix}{ts_unix}.json");
            if let Err(e) = self
                .storage
                .upload(&self.bucket, &snapshot_key, &snapshot_bytes)
                .await
            {
                eprintln!("[poll]   snapshot upload failed (non-fatal): {e:#}");
            }

            eprintln!(
                "[poll t={:>5.1}s tick={tick:>3}] state={} counts={} omni={} err={}",
                elapsed_secs,
                snapshot["state"].as_str().unwrap_or("unknown"),
                snapshot["instance_status_counts"],
                omni.len(),
                errs.len(),
            );

            if (omni.len() + errs.len()) >= expected_chunks {
                out.t_done_secs = Some(elapsed_secs);
                eprintln!(
                    "[poll] DONE at t={elapsed_secs:.1}s (omni={} err={})",
                    omni.len(),
                    errs.len()
                );
                break;
            }
            if snapshot["state"] == "stopped" {
                out.t_done_secs = Some(elapsed_secs);
                eprintln!(
                    "[poll] container group is stopped at t={elapsed_secs:.1}s; short-circuiting"
                );
                break;
            }

            sleep(interval).await;
        }
        Ok(out)
    }

    /// Stitch the per-replica picture from R2 artifacts and upload
    /// the result as `runs/<sweep_id>/fleet_summary.json`.
    ///
    /// Best-effort: errors are logged but never block teardown.
    async fn build_and_upload_fleet_summary(&self, t_post: Instant) -> Result<usize> {
        let boot_prefix = r2_layout::boot_prefix(&self.sweep_id);
        let instances_prefix = r2_layout::instances_prefix(&self.sweep_id);
        let omni_prefix = r2_layout::omni_prefix(&self.sweep_id);
        let errors_prefix = r2_layout::errors_prefix(&self.sweep_id);

        let boots = self
            .storage
            .list(&self.bucket, &boot_prefix)
            .await
            .unwrap_or_default();
        let snaps = self
            .storage
            .list(&self.bucket, &instances_prefix)
            .await
            .unwrap_or_default();
        let omnis = self
            .storage
            .list(&self.bucket, &omni_prefix)
            .await
            .unwrap_or_default();
        let errs = self
            .storage
            .list(&self.bucket, &errors_prefix)
            .await
            .unwrap_or_default();

        #[derive(Default, Debug, Clone, Serialize)]
        struct Replica {
            machine_id: String,
            gpu_class: String,
            gpu_uuid: String,
            driver: String,
            warmup_seconds: Option<f64>,
            boot_unix_ts: Option<u64>,
            t_first_seen_status: Option<String>,
            t_first_running_unix: Option<u64>,
            chunks_processed: u32,
            last_status_seen: Option<String>,
        }
        let mut replicas: BTreeMap<String, Replica> = BTreeMap::new();

        for key in &boots {
            let body = match self.storage.get_bytes(&self.bucket, key).await {
                Ok(b) => b,
                Err(_) => continue,
            };
            let body_str = String::from_utf8_lossy(&body).to_string();
            let mut rec = Replica::default();
            for line in body_str.lines() {
                let Some((k, v)) = line.split_once(':') else {
                    continue;
                };
                let v = v.trim();
                match k.trim() {
                    "machine_id" => rec.machine_id = v.to_string(),
                    "gpu_class" => rec.gpu_class = v.to_string(),
                    "gpu_uuid" => rec.gpu_uuid = v.to_string(),
                    "driver" => rec.driver = v.to_string(),
                    "warmup_seconds" => rec.warmup_seconds = v.parse().ok(),
                    "boot_unix_ts" => rec.boot_unix_ts = v.parse().ok(),
                    _ => {}
                }
            }
            let key_id = if rec.machine_id.is_empty() {
                key.rsplit('/').next().unwrap_or(key).to_string()
            } else {
                rec.machine_id.clone()
            };
            replicas.insert(key_id, rec);
        }

        for key in &snaps {
            let body = match self.storage.get_bytes(&self.bucket, key).await {
                Ok(b) => b,
                Err(_) => continue,
            };
            let Ok(v) = serde_json::from_slice::<serde_json::Value>(&body) else {
                continue;
            };
            let ts = v.get("unix_ts").and_then(|x| x.as_u64()).unwrap_or(0);
            let state = v
                .get("state")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let instances = v.get("instances").cloned().unwrap_or(JsonValue::Null);
            let arr = instances
                .as_array()
                .cloned()
                .or_else(|| {
                    instances
                        .get("instances")
                        .and_then(|x| x.as_array().cloned())
                })
                .unwrap_or_default();
            for inst in arr {
                let m_id = inst
                    .get("machine_id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                if m_id.is_empty() {
                    continue;
                }
                let entry = replicas.entry(m_id.clone()).or_insert_with(|| Replica {
                    machine_id: m_id.clone(),
                    ..Default::default()
                });
                let inst_state = inst
                    .get("state")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                if entry.t_first_seen_status.is_none() && inst_state.is_some() {
                    entry.t_first_seen_status = inst_state.clone();
                }
                if entry.t_first_running_unix.is_none() && inst_state.as_deref() == Some("running")
                {
                    entry.t_first_running_unix = Some(ts);
                }
                entry.last_status_seen = inst_state.or(Some(state.clone()));
            }
        }

        let total_chunks = omnis.len() as u32;
        let mut replica_rows: Vec<Replica> = replicas.into_values().collect();
        if !replica_rows.is_empty() {
            let n = replica_rows.len() as u32;
            let each = total_chunks / n;
            let rem = total_chunks % n;
            for (i, r) in replica_rows.iter_mut().enumerate() {
                r.chunks_processed = each + if (i as u32) < rem { 1 } else { 0 };
            }
        }

        let wall = t_post.elapsed().as_secs_f64();
        eprintln!(
            "[fleet] === fleet_summary ({} replicas, wall={:.1}s) ===",
            replica_rows.len(),
            wall
        );
        eprintln!(
            "[fleet] {:<24}  {:<24}  {:<12}  {:<10}  {:<8}",
            "machine_id_or_filename", "gpu_class", "warmup_s", "chunks", "status"
        );
        for r in &replica_rows {
            eprintln!(
                "[fleet] {:<24}  {:<24}  {:<12}  {:<10}  {:<8}",
                short24(&r.machine_id),
                short24(&r.gpu_class),
                r.warmup_seconds
                    .map(|x| format!("{x:.1}"))
                    .unwrap_or_default(),
                r.chunks_processed,
                r.last_status_seen.as_deref().unwrap_or("-"),
            );
        }
        eprintln!(
            "[fleet] totals: boot={} snapshots={} omni={} errors={}",
            boots.len(),
            snaps.len(),
            omnis.len(),
            errs.len()
        );

        let summary = json!({
            "sweep_id": self.sweep_id,
            "bucket": self.bucket,
            "wall_secs": wall,
            "boot_records": boots.len(),
            "snapshot_count": snaps.len(),
            "omni_count": omnis.len(),
            "error_count": errs.len(),
            "replicas": replica_rows,
        });
        let body = serde_json::to_vec_pretty(&summary).unwrap_or_default();
        let key = r2_layout::fleet_summary_key(&self.sweep_id);
        self.storage
            .upload(&self.bucket, &key, &body)
            .await
            .with_context(|| format!("upload {key}"))?;
        eprintln!(
            "[fleet] fleet_summary.json uploaded to s3://{}/{}",
            self.bucket, key
        );
        Ok(replica_rows.len())
    }
}

fn short24(s: &str) -> String {
    if s.chars().count() <= 24 {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(21).collect();
        out.push_str("...");
        out
    }
}
