//! `hetzner-lifecycle-smoke` — provision → poll → push-jobs → teardown
//! validator that runs against the real Hetzner API.
//!
//! Differs from `zenfleet-hetzner-sweep` in that it does NOT wait for
//! worker sidecars to land. The existing `:v6-visibility-b` image
//! doesn't include `--backend hetzner` in its zenfleet-sweep build,
//! so the cloud-init will succeed but the docker container will exit
//! with "unknown backend hetzner". That's fine for validating the
//! PROVIDER TRAIT IMPL (provision/poll/teardown/push_jobs) without
//! waiting for the chunk-processing tail.
//!
//! Outputs JSON to stdout with per-phase timing.
//!
//! HARD CAPS (do NOT remove):
//! - 20 min wall-time
//! - $1 spend (5 × CAX21 × 1h = ~$0.10 even at full hour minimum)
//! - mandatory teardown via DELETE + re-list verification

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use serde_json::Value as JsonValue;
use zenfleet_hetzner::api::{HetznerApi, load_token_from_file_or_env};
use zenfleet_hetzner::provider::{HetznerProviderConfig, HetznerProviderHandle};
use zenfleet_orchestrator::{GroupId, ProviderHandle, ProvisionSpec, QueueJob, R2Operator};
use zenfleet_salad::launch::{ScopedCredSpec, inject_r2_cred_into_env};
use zenfleet_salad::launcher_support::{
    ChunkLayout, generate_chunks, load_r2_parent_creds_or_env, short_id_for_name,
};
use zenfleet_salad::r2_ops::{R2OperatorImpl, short_ts};

#[derive(Debug, Parser)]
#[command(name = "hetzner-lifecycle-smoke")]
struct Args {
    #[arg(long, default_value = "zen-tuning-ephemeral")]
    bucket: String,
    #[arg(long, default_value = "cax21")]
    server_type: String,
    #[arg(long, default_value = "fsn1")]
    location: String,
    #[arg(long, default_value_t = 5)]
    replicas: u32,
    #[arg(long, default_value_t = 24)]
    chunks: u32,
    #[arg(long, default_value_t = 12)]
    cells_per_chunk: u32,
    #[arg(
        long,
        default_value = "s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/input/smoke.parquet"
    )]
    input_parquet_r2: String,
    #[arg(
        long,
        default_value = "s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/sources"
    )]
    source_dir_r2: String,
    #[arg(long, default_value = "graph.png")]
    image_basename: String,
    #[arg(
        long,
        default_value = "ghcr.io/imazen/zenmetrics-sweep-salad:v6-visibility-b"
    )]
    docker_image: String,
    /// Wall cap for the poll loop (seconds).
    #[arg(long, default_value_t = 1080)]
    poll_wall_secs: u64,
    /// Poll interval (seconds).
    #[arg(long, default_value_t = 15)]
    poll_interval_secs: u64,
    /// Stop waiting once at least one server is `running`.
    #[arg(long, default_value_t = true)]
    stop_at_first_running: bool,
    /// SKIP teardown (DANGEROUS — billing exposure).
    #[arg(long, default_value_t = false)]
    skip_teardown: bool,
}

#[derive(Debug, Serialize)]
struct LifecycleReport {
    sweep_id: String,
    group_name: String,
    server_type: String,
    location: String,
    replicas: u32,
    t_provision_secs: f64,
    t_first_status_running_secs: Option<f64>,
    t_all_status_running_secs: Option<f64>,
    t_push_jobs_secs: f64,
    t_teardown_secs: f64,
    t_total_secs: f64,
    queue_files_landed: u32,
    teardown_ok: bool,
    servers_alive_after_teardown: u32,
    poll_ticks: u32,
    final_status_counts: HashMap<String, u32>,
    per_replica_boot_secs: Vec<f64>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let t0 = Instant::now();
    let sweep_id = format!("hetzner-smoke-{}", short_ts());
    let group_name = short_id_for_name(&sweep_id);
    eprintln!(
        "[smoke] sweep_id={sweep_id} group={group_name} replicas={} server_type={} location={}",
        args.replicas, args.server_type, args.location
    );

    let token = load_token_from_file_or_env()?;
    let api = HetznerApi::new(&token);
    let parent = load_r2_parent_creds_or_env()?;
    let r2 = Arc::new(R2OperatorImpl::new(parent.clone()));

    // Scoped R2 cred.
    let mut cred_prefixes: Vec<String> = vec![format!("runs/{sweep_id}/")];
    let bp = format!("s3://{}/", args.bucket);
    for uri in [&args.input_parquet_r2, &args.source_dir_r2] {
        if let Some(rest) = uri.strip_prefix(&bp) {
            let leading = rest.split('/').next().unwrap_or("");
            if !leading.is_empty() {
                let p = format!("{leading}/");
                if !cred_prefixes.contains(&p) {
                    cred_prefixes.push(p);
                }
            }
        }
    }
    let salad = zenfleet_salad::launch::SaladApi::new("placeholder", "placeholder", None)?;
    let scoped = salad
        .mint_sweep_r2_cred(
            &parent,
            &ScopedCredSpec::new(&args.bucket)
                .with_prefixes(cred_prefixes)
                .with_ttl_seconds(3600),
        )
        .await?;

    // ── Generate + upload chunks.jsonl ─────────────────────────────
    let chunks = generate_chunks(&ChunkLayout {
        n: args.chunks,
        cells_per_chunk: args.cells_per_chunk,
        bucket: args.bucket.clone(),
        sweep_id: sweep_id.clone(),
        input_parquet_r2: args.input_parquet_r2.clone(),
        source_dir_r2: args.source_dir_r2.clone(),
        image_basenames: vec![args.image_basename.clone()],
    });
    let chunks_key = format!("runs/{sweep_id}/chunks.jsonl");
    let chunks_jsonl = chunks
        .iter()
        .map(|c| serde_json::to_string(c).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    r2.upload(&args.bucket, &chunks_key, chunks_jsonl.as_bytes())
        .await
        .context("upload chunks.jsonl")?;
    eprintln!("[smoke] chunks.jsonl uploaded ({} chunks)", chunks.len());

    // ── Build provider + provision ────────────────────────────────
    let mut env: std::collections::BTreeMap<String, String> = Default::default();
    env.insert("SWEEP_RUN_ID".into(), sweep_id.clone());
    env.insert("R2_ACCOUNT_ID".into(), parent.account_id.clone());
    env.insert("BUCKET".into(), args.bucket.clone());
    env.insert(
        "CHUNKS_R2".into(),
        format!("s3://{}/{}", args.bucket, chunks_key),
    );
    env.insert(
        "CHUNKS_QUEUE_PREFIX".into(),
        format!("runs/{sweep_id}/queue/"),
    );
    env.insert("WORKER_BACKEND".into(), "hetzner".into());
    env.insert("RUST_LOG".into(), "info".into());
    env.insert("METRICS".into(), "ssim2-gpu".into());
    let mut env_h: HashMap<String, String> =
        env.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    inject_r2_cred_into_env(&mut env_h, &scoped);
    let env: std::collections::BTreeMap<String, String> = env_h.into_iter().collect();

    let provider_cfg = HetznerProviderConfig::new(
        group_name.clone(),
        args.server_type.clone(),
        args.bucket.clone(),
        parent.account_id.clone(),
        r2.clone(),
    )
    .with_location(args.location.clone());
    let mut provider = HetznerProviderHandle::new(api.clone(), provider_cfg);
    let group = GroupId(group_name.clone());

    let spec = ProvisionSpec {
        image: args.docker_image.clone(),
        replicas: args.replicas,
        gpu_classes: vec![],
        env,
        max_price_per_hour: 0.02,
        extra: JsonValue::Null,
    };

    let t_prov_start = Instant::now();
    let _g = match provider.provision(&spec).await {
        Ok(g) => g,
        Err(e) => {
            eprintln!("[smoke] PROVISION FAILED: {e:#}");
            // Best-effort teardown if any servers got created.
            best_effort_teardown(&mut provider, &group, &api, &group_name).await;
            return Err(e.context("provision"));
        }
    };
    let t_provision_secs = t_prov_start.elapsed().as_secs_f64();
    eprintln!("[smoke] provision OK in {t_provision_secs:.1}s");

    // ── Push N queue jobs to R2 ───────────────────────────────────
    let t_push = Instant::now();
    let queue_jobs: Vec<QueueJob> = chunks
        .iter()
        .take(args.chunks as usize)
        .map(|c| QueueJob {
            chunk_id: c.chunk_id.clone(),
            payload: serde_json::to_value(c).unwrap_or(JsonValue::Null),
        })
        .collect();
    if let Err(e) = provider.push_jobs(&group, &queue_jobs).await {
        eprintln!("[smoke] push_jobs FAILED: {e:#}");
        best_effort_teardown(&mut provider, &group, &api, &group_name).await;
        return Err(e.context("push_jobs"));
    }
    let t_push_jobs_secs = t_push.elapsed().as_secs_f64();
    eprintln!(
        "[smoke] push_jobs OK in {t_push_jobs_secs:.1}s (uploaded {} queue files)",
        queue_jobs.len()
    );

    // Verify queue files landed.
    let queue_prefix = format!("runs/{sweep_id}/queue/");
    let queue_files = r2
        .list(&args.bucket, &queue_prefix)
        .await
        .unwrap_or_default();
    let queue_files_landed = queue_files.len() as u32;
    eprintln!("[smoke] queue prefix listed: {} files", queue_files_landed);

    // ── Poll loop ─────────────────────────────────────────────────
    let mut t_first_running: Option<f64> = None;
    let mut t_all_running: Option<f64> = None;
    let mut tick: u32 = 0;
    let mut final_counts: HashMap<String, u32> = Default::default();
    let mut boot_times: Vec<f64> = Vec::with_capacity(args.replicas as usize);
    let mut seen_running: std::collections::HashSet<String> = Default::default();

    let poll_deadline = Instant::now() + Duration::from_secs(args.poll_wall_secs);
    while Instant::now() < poll_deadline {
        tick += 1;
        match provider.poll_instances(&group).await {
            Ok((_, instances)) => {
                let elapsed = t_prov_start.elapsed().as_secs_f64();
                let mut counts: HashMap<String, u32> = Default::default();
                let mut running = 0u32;
                for i in &instances {
                    *counts.entry(i.state.clone()).or_insert(0) += 1;
                    if i.state == "running" {
                        running += 1;
                        if !seen_running.contains(&i.machine_id) {
                            seen_running.insert(i.machine_id.clone());
                            boot_times.push(elapsed);
                        }
                    }
                }
                eprintln!(
                    "[smoke t={elapsed:>6.1}s tick={tick:>3}] instances={} running={running} counts={counts:?}",
                    instances.len()
                );
                if running >= 1 && t_first_running.is_none() {
                    t_first_running = Some(elapsed);
                    eprintln!("[smoke] FIRST replica running at t={elapsed:.1}s");
                    if args.stop_at_first_running {
                        // Wait briefly to capture the next tick (for boot_times signal).
                    }
                }
                if running >= args.replicas && t_all_running.is_none() {
                    t_all_running = Some(elapsed);
                    eprintln!("[smoke] ALL {running} replicas running at t={elapsed:.1}s");
                }
                final_counts = counts;
                // Exit condition: either all-running, or first-running + 30s grace.
                if t_all_running.is_some() {
                    eprintln!("[smoke] all running; stopping poll loop");
                    break;
                }
                if args.stop_at_first_running
                    && let Some(t_first) = t_first_running
                {
                    let since_first = elapsed - t_first;
                    if since_first >= 60.0 {
                        eprintln!(
                            "[smoke] stop_at_first_running + 60s grace elapsed; stopping poll loop"
                        );
                        break;
                    }
                }
            }
            Err(e) => {
                eprintln!("[smoke] poll error tick={tick}: {e:#}");
            }
        }
        tokio::time::sleep(Duration::from_secs(args.poll_interval_secs)).await;
    }
    if t_first_running.is_none() {
        eprintln!("[smoke] WARNING: no replica reached `running` within wall cap");
    }

    // ── Teardown ─────────────────────────────────────────────────
    let t_td_start = Instant::now();
    let teardown_ok = if args.skip_teardown {
        eprintln!("[smoke] skip_teardown=true; LEAVING servers running (BILLING EXPOSURE)");
        false
    } else {
        match provider.teardown(&group).await {
            Ok(()) => {
                eprintln!("[smoke] teardown OK");
                true
            }
            Err(e) => {
                eprintln!("[smoke] teardown FAILED: {e:#}");
                best_effort_teardown(&mut provider, &group, &api, &group_name).await;
                false
            }
        }
    };
    let t_teardown_secs = t_td_start.elapsed().as_secs_f64();

    // Verify zero remaining.
    let label = format!("group={group_name}");
    let remaining = api.list_servers_by_label(&label).await.unwrap_or_default();
    let servers_alive_after_teardown = remaining.len() as u32;
    if servers_alive_after_teardown > 0 {
        eprintln!(
            "[smoke] WARNING: {servers_alive_after_teardown} servers STILL alive after teardown"
        );
    }

    let report = LifecycleReport {
        sweep_id,
        group_name,
        server_type: args.server_type,
        location: args.location,
        replicas: args.replicas,
        t_provision_secs,
        t_first_status_running_secs: t_first_running,
        t_all_status_running_secs: t_all_running,
        t_push_jobs_secs,
        t_teardown_secs,
        t_total_secs: t0.elapsed().as_secs_f64(),
        queue_files_landed,
        teardown_ok,
        servers_alive_after_teardown,
        poll_ticks: tick,
        final_status_counts: final_counts,
        per_replica_boot_secs: boot_times,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if servers_alive_after_teardown > 0 {
        bail!("teardown left {servers_alive_after_teardown} servers alive");
    }
    Ok(())
}

async fn best_effort_teardown(
    provider: &mut HetznerProviderHandle,
    group: &GroupId,
    api: &HetznerApi,
    group_name: &str,
) {
    eprintln!("[smoke] best-effort teardown after error");
    if let Err(e) = provider.teardown(group).await {
        eprintln!("[smoke] best-effort teardown via provider failed: {e:#}");
        // Direct API fallback.
        let label = format!("group={group_name}");
        if let Ok(live) = api.list_servers_by_label(&label).await {
            for s in live {
                eprintln!("[smoke] direct DELETE server id={}", s.id);
                let _ = api.delete_server(s.id).await;
            }
        }
    }
}
