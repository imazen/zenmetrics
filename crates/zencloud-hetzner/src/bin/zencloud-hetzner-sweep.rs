//! `zencloud-hetzner-sweep` — Hetzner-flavoured launcher driving the
//! provider-generic [`zenfleet_orchestrator::FleetSweep`] driver.
//!
//! Mirrors the Salad bin (`zen-salad-sweep`); the only differences are
//! Hetzner-specific knobs (server type / location) and the R2-queue
//! polling worker model (no managed queue).

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use zen_cloud_salad::launch::{ScopedCredSpec, inject_r2_cred_into_env};
use zen_cloud_salad::launcher_support::{
    ChunkLayout, generate_chunks, load_r2_parent_creds_or_env, short_id_for_name,
};
use zen_cloud_salad::r2_ops::{R2OperatorImpl, short_ts};
use zencloud_hetzner::api::{HetznerApi, load_token_from_file_or_env};
use zencloud_hetzner::provider::{HetznerProviderConfig, HetznerProviderHandle};
use zenfleet_orchestrator::{
    FleetSweep, QueueJob, R2Operator, SpeculativeConfig, SweepConfig,
    compute_provisioned_replicas,
};

const DEFAULT_BUCKET: &str = "zen-tuning-ephemeral";
const DEFAULT_SOURCE_DIR_R2: &str =
    "s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/sources";
const DEFAULT_INPUT_PARQUET_R2: &str =
    "s3://zen-tuning-ephemeral/salad-smoke-2026-05-28-24cell/inputs.parquet";
const DEFAULT_IMAGE_BASENAME: &str = "graph.png";
const DEFAULT_IMAGE: &str = "ghcr.io/imazen/zen-metrics-sweep-salad:v6-visibility-b";
const DEFAULT_SERVER_TYPE: &str = "cax21";
const DEFAULT_LOCATION: &str = "fsn1";
/// Hetzner per-project server quota is generous (10+); pick a safe
/// number for the validation cap. Operator can override.
const HETZNER_PROJECT_REPLICA_QUOTA: u32 = 10;

/// Default per-replica $/hr for cax21 (€0.0152 ≈ $0.017 at 1.10
/// USD/EUR — round up to $0.018 for safety in spend-cap math).
const DEFAULT_PRICE_PER_HOUR_USD: f64 = 0.018;

#[derive(Debug, Parser)]
#[command(
    name = "zencloud-hetzner-sweep",
    about = "Hetzner-flavoured launcher (provider-generic via zenfleet-orchestrator)"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_BUCKET)]
    bucket: String,
    #[arg(long)]
    sweep_id: Option<String>,
    #[arg(long)]
    group_name: Option<String>,
    #[arg(long, default_value = DEFAULT_IMAGE)]
    image: String,
    /// Number of worker replicas (per-sweep). Capped by
    /// `--provider-quota`.
    #[arg(long, default_value_t = 5)]
    replicas: u32,
    #[arg(long)]
    chunks: Option<u32>,
    /// Hetzner server type slug. Defaults to `cax21` (ARM 4 cores,
    /// ~€0.0152/hr). For dedicated AMD use `ccx13`.
    #[arg(long, default_value = DEFAULT_SERVER_TYPE)]
    server_type: String,
    /// Hetzner location (`fsn1`, `nbg1`, `hel1`, `ash`, `hil`, `sin`).
    #[arg(long, default_value = DEFAULT_LOCATION)]
    location: String,
    /// Per-replica $/hr estimate (USD). Used for the spend summary.
    #[arg(long, default_value_t = DEFAULT_PRICE_PER_HOUR_USD)]
    price_per_hour: f64,
    /// Provider replica quota (server-creation cap inside the
    /// project).
    #[arg(long, default_value_t = HETZNER_PROJECT_REPLICA_QUOTA)]
    provider_quota: u32,
    #[arg(long, default_value_t = 1800)]
    max_wall_secs: u64,
    #[arg(long, default_value_t = 15)]
    poll_secs: u64,
    #[arg(long, default_value = DEFAULT_SOURCE_DIR_R2)]
    source_dir_r2: String,
    #[arg(long, default_value = DEFAULT_INPUT_PARQUET_R2)]
    input_parquet_r2: String,
    #[arg(long = "image-basename", default_values_t = [DEFAULT_IMAGE_BASENAME.to_string()])]
    image_basenames: Vec<String>,
    #[arg(long, env = "GHCR_USERNAME")]
    registry_username: Option<String>,
    #[arg(long, env = "GHCR_PASSWORD")]
    registry_password: Option<String>,
    #[arg(long)]
    keep_running: bool,
    #[arg(long)]
    skip_preflight: bool,
    #[arg(long)]
    chunks_key: Option<String>,
    #[arg(long)]
    dry_run: bool,
    #[arg(long, default_value_t = 12)]
    cells_per_chunk: u32,
    #[arg(long, default_value_t = 1.0)]
    replicas_overshoot: f64,
    #[arg(long, default_value_t = 360)]
    chunk_ttl_secs: u64,
    #[arg(long)]
    prior_fleet_summary: Option<String>,
    #[arg(long, default_value_t = 600.0)]
    max_warmup_secs: f64,
    #[arg(long, default_value_t = 2.0)]
    min_productive_chunks: f64,
    #[arg(long, default_value_t = false)]
    no_speculative: bool,
    #[arg(long, default_value_t = 1.5)]
    speculative_straggler_factor: f64,
    #[arg(long, default_value_t = 3)]
    speculative_min_completed: u32,
    #[arg(long, default_value_t = 1)]
    speculative_cap_per_chunk: u32,
}

#[derive(Debug, Serialize)]
struct LauncherSummary {
    sweep_id: String,
    group_name: String,
    image: String,
    replicas: u32,
    chunks: u32,
    server_type: String,
    location: String,
    wall_secs: f64,
    t_first_sidecar_secs: Option<f64>,
    t_all_n_sidecars_secs: Option<f64>,
    t_done_secs: Option<f64>,
    distinct_workers_observed: u32,
    throughput_chunks_per_sec: Option<f64>,
    estimated_spend_usd: f64,
    teardown_ok: bool,
    error_sidecars: u32,
    omni_sidecars: u32,
    cells_per_chunk: u32,
    replicas_requested: u32,
    replicas_provisioned: u32,
    chunks_redispatched: u32,
    chunks_speculatively_dispatched: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    let sweep_id = args
        .sweep_id
        .clone()
        .unwrap_or_else(|| format!("hetzner-{}", short_ts()));
    let group_name = args
        .group_name
        .clone()
        .unwrap_or_else(|| short_id_for_name(&sweep_id));
    let chunks_count = args.chunks.unwrap_or(args.replicas * 3);
    let chunks_key = args
        .chunks_key
        .clone()
        .unwrap_or_else(|| format!("runs/{sweep_id}/chunks.jsonl"));
    eprintln!(
        "[launcher] sweep_id={sweep_id} group={group_name} bucket={} server_type={} location={}",
        args.bucket, args.server_type, args.location
    );

    let token = load_token_from_file_or_env().context(
        "load Hetzner token (set $HETZNER_API_TOKEN or write ~/.config/hetzner/credentials)",
    )?;
    let api = HetznerApi::new(&token);

    let replicas_provisioned = compute_provisioned_replicas(
        args.replicas,
        args.replicas_overshoot,
        args.provider_quota,
    );

    if args.dry_run {
        // Preview: dump the synthesized POST body + cloud-init for one
        // replica + the queue-push paths.
        let parent =
            load_r2_parent_creds_or_env().context("load R2 parent cred (for cloud-init preview)")?;
        let preview_env = preview_env_block(&sweep_id, &parent.account_id, &args.bucket);
        let user_data_preview =
            zencloud_hetzner::cloud_init::build_user_data(&zencloud_hetzner::cloud_init::WorkerBootstrap {
                image: args.image.clone(),
                sweep_id: sweep_id.clone(),
                r2_account_id: parent.account_id.clone(),
                r2_bucket: args.bucket.clone(),
                r2_access_key_id: "<scoped-key-id>".into(),
                r2_secret_access_key: "<scoped-secret>".into(),
                r2_session_token: "<scoped-session>".into(),
                registry_username: args.registry_username.clone(),
                registry_password: args.registry_password.clone(),
                registry_server: None,
                extra_env: preview_env,
                chunks_queue_prefix: format!("runs/{sweep_id}/queue/"),
            });
        let synth_post = json!({
            "name": format!("{}-000", group_name),
            "server_type": args.server_type,
            "image": "ubuntu-24.04",
            "location": args.location,
            "labels": {"group": group_name, "sweep_id": sweep_id},
            "user_data": "<see preview below>",
            "ssh_keys": JsonValue::Array(Vec::new()),
            "start_after_create": true,
        });
        let queue_paths: Vec<String> = (0..chunks_count)
            .map(|i| {
                format!(
                    "s3://{}/runs/{}/queue/scaleup-{:03}.json",
                    args.bucket, sweep_id, i
                )
            })
            .take(3)
            .collect();
        let preview = json!({
            "replicas_requested": args.replicas,
            "replicas_provisioned": replicas_provisioned,
            "server_type": args.server_type,
            "location": args.location,
            "synthesized_post_body": synth_post,
            "queue_paths_first_3": queue_paths,
            "cells_per_chunk": args.cells_per_chunk,
        });
        println!("{}", serde_json::to_string_pretty(&preview)?);
        eprintln!("\n--- cloud-init user_data preview ---");
        eprintln!("{user_data_preview}");
        eprintln!("[launcher] dry-run: NO Hetzner/R2 calls; exiting 0");
        return Ok(());
    }

    let parent = load_r2_parent_creds_or_env()?;
    let r2 = Arc::new(R2OperatorImpl::new(parent.clone()));

    // Scoped R2 cred (per-sweep, never the parent root key).
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
    // Salad's `SaladApi` carries the scoped-cred minter; we don't need
    // a SaladApi instance to use it though — the underlying minter is
    // standalone. For simplicity here we reuse the wrapper.
    let salad = zen_cloud_salad::launch::SaladApi::new("placeholder", "placeholder", None)?;
    let scoped = salad
        .mint_sweep_r2_cred(
            &parent,
            &ScopedCredSpec::new(&args.bucket)
                .with_prefixes(cred_prefixes)
                .with_ttl_seconds(3600),
        )
        .await
        .context("mint scoped R2 cred")?;

    if !args.skip_preflight {
        r2.head_uri(&args.input_parquet_r2)
            .await
            .with_context(|| format!("HEAD {}", args.input_parquet_r2))?;
        for b in &args.image_basenames {
            let uri = format!("{}/{}", args.source_dir_r2.trim_end_matches('/'), b);
            r2.head_uri(&uri)
                .await
                .with_context(|| format!("HEAD {uri}"))?;
        }
    }

    // Chunks → JSONL → R2 (manifest, plus one per-chunk queue file
    // gets written later via push_jobs).
    let chunks = generate_chunks(&ChunkLayout {
        n: chunks_count,
        cells_per_chunk: args.cells_per_chunk,
        bucket: args.bucket.clone(),
        sweep_id: sweep_id.clone(),
        input_parquet_r2: args.input_parquet_r2.clone(),
        source_dir_r2: args.source_dir_r2.clone(),
        image_basenames: args.image_basenames.clone(),
    });
    let chunks_jsonl = chunks
        .iter()
        .map(|c| serde_json::to_string(c).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    r2.upload(&args.bucket, &chunks_key, chunks_jsonl.as_bytes())
        .await
        .with_context(|| format!("upload chunks.jsonl to s3://{}/{}", args.bucket, chunks_key))?;

    // Worker env. Hetzner workers don't have a sidecar — they poll R2.
    let mut env_h: HashMap<String, String> = HashMap::new();
    env_h.insert("SWEEP_RUN_ID".into(), sweep_id.clone());
    env_h.insert("R2_ACCOUNT_ID".into(), parent.account_id.clone());
    env_h.insert("BUCKET".into(), args.bucket.clone());
    env_h.insert(
        "CHUNKS_R2".into(),
        format!("s3://{}/{}", args.bucket, chunks_key),
    );
    env_h.insert(
        "CHUNKS_QUEUE_PREFIX".into(),
        format!("runs/{sweep_id}/queue/"),
    );
    env_h.insert("WORKER_BACKEND".into(), "hetzner".into());
    env_h.insert("RUST_LOG".into(), "info,zencloud_hetzner=info".into());
    env_h.insert("METRICS".into(), "ssim2-gpu".into());
    inject_r2_cred_into_env(&mut env_h, &scoped);
    let env: BTreeMap<String, String> = env_h.into_iter().collect();

    let mut provider_cfg = HetznerProviderConfig::new(
        group_name.clone(),
        args.server_type.clone(),
        args.bucket.clone(),
        parent.account_id.clone(),
        r2.clone(),
    )
    .with_location(args.location.clone());
    if let (Some(u), Some(p)) = (
        args.registry_username.as_ref(),
        args.registry_password.as_ref(),
    ) {
        provider_cfg = provider_cfg.with_registry_auth(u.clone(), p.clone(), None);
    }

    let provider = HetznerProviderHandle::new(api, provider_cfg);

    let sweep_cfg = SweepConfig {
        replicas: args.replicas,
        replicas_overshoot: args.replicas_overshoot,
        provider_replica_quota: args.provider_quota,
        chunk_ttl_secs: args.chunk_ttl_secs,
        cells_per_chunk: args.cells_per_chunk,
        max_warmup_secs: args.max_warmup_secs as u32,
        min_productive_chunks: args.min_productive_chunks as f32,
        speculative: SpeculativeConfig {
            enabled: !args.no_speculative,
            straggler_factor: args.speculative_straggler_factor,
            min_completed_for_stats: args.speculative_min_completed,
            speculation_cap_per_chunk: args.speculative_cap_per_chunk,
        },
    };
    let queue_jobs: Vec<QueueJob> = chunks
        .iter()
        .map(|c| QueueJob {
            chunk_id: c.chunk_id.clone(),
            payload: serde_json::to_value(c).unwrap_or(JsonValue::Null),
        })
        .collect();

    // The FleetSweep takes ownership of the R2Operator. We hold an Arc
    // on the same data inside HetznerProviderConfig — so the orchestrator's
    // moves operate on a CLONE of the wrapper, not the same instance.
    let r2_for_driver = R2OperatorImpl::new(parent.clone());
    let fleet = FleetSweep::new(
        provider,
        r2_for_driver,
        sweep_cfg,
        args.bucket.clone(),
        sweep_id.clone(),
        group_name.clone(),
        args.image.clone(),
        vec![args.server_type.clone()],
        env,
        args.price_per_hour,
        json!({"server_type": args.server_type, "location": args.location}),
        args.max_wall_secs,
        args.poll_secs,
        args.keep_running,
    )
    .run(queue_jobs)
    .await?;

    let wall = t_start.elapsed().as_secs_f64();
    let throughput = match (
        fleet.poll.t_done_secs,
        fleet.poll.t_first_sidecar_secs,
        fleet.poll.omni_sidecars,
    ) {
        (Some(td), Some(tf), n) if n > 0 && td > tf => Some(f64::from(n) / (td - tf).max(0.001)),
        _ => None,
    };
    // Hetzner bills per-hour with a 1-hour minimum.
    let billed_hours = (wall / 3600.0).max(1.0);
    let spend = args.price_per_hour * f64::from(replicas_provisioned) * billed_hours;
    let summary = LauncherSummary {
        sweep_id: fleet.sweep_id.clone(),
        group_name: fleet.group_name.clone(),
        image: fleet.image.clone(),
        replicas: replicas_provisioned,
        chunks: fleet.chunks,
        server_type: args.server_type.clone(),
        location: args.location.clone(),
        wall_secs: wall,
        t_first_sidecar_secs: fleet.poll.t_first_sidecar_secs,
        t_all_n_sidecars_secs: fleet.poll.t_all_n_sidecars_secs,
        t_done_secs: fleet.poll.t_done_secs,
        distinct_workers_observed: fleet.poll.distinct_workers_observed,
        throughput_chunks_per_sec: throughput,
        estimated_spend_usd: spend,
        teardown_ok: fleet.teardown_ok,
        error_sidecars: fleet.poll.error_sidecars,
        omni_sidecars: fleet.poll.omni_sidecars,
        cells_per_chunk: args.cells_per_chunk,
        replicas_requested: args.replicas,
        replicas_provisioned,
        chunks_redispatched: fleet.poll.chunks_redispatched,
        chunks_speculatively_dispatched: fleet.poll.chunks_speculatively_dispatched,
    };
    println!("{}", serde_json::to_string(&summary).unwrap());

    if !fleet.teardown_ok && !args.keep_running {
        bail!(
            "teardown of Hetzner server group {} did not succeed",
            fleet.group_name
        );
    }
    Ok(())
}

fn preview_env_block(
    _sweep_id: &str,
    _r2_account_id: &str,
    _bucket: &str,
) -> std::collections::BTreeMap<String, String> {
    // Only "extra" envs the user might inject — the cloud-init builder
    // already places BUCKET / CHUNKS_R2 / R2_ACCOUNT_ID explicitly.
    let mut m = std::collections::BTreeMap::new();
    m.insert("METRICS".into(), "ssim2-gpu".into());
    m
}
