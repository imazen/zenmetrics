//! `zenfleet-salad-sweep` — Salad-flavoured launcher driving the
//! provider-generic [`zenfleet_orchestrator::FleetSweep`] driver.
//!
//! The binary owns ONLY:
//!   * CLI argument parsing.
//!   * Wiring helpers (in `zenfleet_salad::launcher_support`) into
//!     a `SaladProviderHandle` + `FleetSweep` invocation.
//!   * Final-summary JSON emit.
//!
//! Every algorithm (TTL re-dispatch, replica overshoot, speculative
//! execution, class-aware filter, poll loop, fleet_summary stitch)
//! lives in `zenfleet-orchestrator`. Salad-specific helpers (GPU class
//! resolution, R2 SigV4, R2 cred load, chunk synthesis) live in
//! `zenfleet_salad::{launcher_support, provider, r2_ops}`.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Serialize;
use serde_json::{Value as JsonValue, json};
use zenfleet_orchestrator::{
    FleetSweep, QueueJob, R2Operator, SpeculativeConfig, SweepConfig, compute_provisioned_replicas,
};
use zenfleet_salad::launch::{RegistryAuth, SaladApi, ScopedCredSpec, inject_r2_cred_into_env};
use zenfleet_salad::launcher_support::{
    ChunkLayout, apply_class_filter, generate_chunks, load_r2_parent_creds_or_env,
    resolve_gpu_classes, short_id_for_name,
};
use zenfleet_salad::provider::{SaladProviderConfig, SaladProviderHandle};
use zenfleet_salad::r2_ops::{R2OperatorImpl, short_ts};

const DEFAULT_BUCKET: &str = "zen-tuning-ephemeral";
const DEFAULT_SOURCE_DIR_R2: &str = "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/sources";
const DEFAULT_INPUT_PARQUET_R2: &str =
    "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/input/smoke.parquet";
const DEFAULT_IMAGE_BASENAME: &str = "graph.png";
const DEFAULT_IMAGE: &str = "ghcr.io/imazen/zenmetrics-sweep-salad:v6-visibility-b";
const DEFAULT_GPU_CLASS: &str = "RTX 3060";
const SALAD_ORG_REPLICA_QUOTA: u32 = 10;

#[derive(Debug, Parser)]
#[command(
    name = "zenfleet-salad-sweep",
    about = "Salad-flavoured launcher (provider-generic via zenfleet-orchestrator)"
)]
struct Args {
    #[arg(long, default_value = "imazen", env = "SALAD_ORGANIZATION")]
    organization: String,
    #[arg(long, default_value = "zenmetrics", env = "SALAD_PROJECT")]
    project: String,
    #[arg(long, default_value = DEFAULT_BUCKET)]
    bucket: String,
    #[arg(long)]
    sweep_id: Option<String>,
    #[arg(long)]
    group_name: Option<String>,
    #[arg(long)]
    queue_name: Option<String>,
    #[arg(long, default_value = DEFAULT_IMAGE)]
    image: String,
    #[arg(long, default_value_t = 10)]
    replicas: u32,
    #[arg(long)]
    chunks: Option<u32>,
    #[arg(long, default_value = DEFAULT_GPU_CLASS)]
    gpu_class: String,
    #[arg(long, value_delimiter = ',')]
    gpu_classes: Vec<String>,
    #[arg(long, default_value_t = 0.10)]
    max_price_per_hour: f64,
    #[arg(long, default_value = "high")]
    price_priority: String,
    #[arg(long, default_value_t = 4)]
    cpu: u32,
    #[arg(long, default_value_t = 8192)]
    memory_mib: u32,
    #[arg(long, default_value_t = 0.20)]
    price_per_hour: f64,
    #[arg(long, default_value_t = 900)]
    max_wall_secs: u64,
    #[arg(long, default_value_t = 15)]
    poll_secs: u64,
    #[arg(long, default_value = DEFAULT_SOURCE_DIR_R2)]
    source_dir_r2: String,
    #[arg(long, default_value = DEFAULT_INPUT_PARQUET_R2)]
    input_parquet_r2: String,
    #[arg(long = "image-basename", default_values_t = [DEFAULT_IMAGE_BASENAME.to_string()])]
    image_basenames: Vec<String>,
    #[arg(long, env = "DOCKER_USERNAME")]
    registry_username: Option<String>,
    #[arg(long, env = "DOCKER_PASSWORD")]
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
    #[arg(long, default_value_t = 1.7)]
    replicas_overshoot: f64,
    #[arg(long, default_value_t = 360)]
    chunk_ttl_secs: u64,
    #[arg(long)]
    prior_fleet_summary: Option<String>,
    #[arg(long, default_value_t = 60.0)]
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
    gpu_class: String,
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
    classes_dropped_by_filter: Vec<String>,
    classes_kept_after_filter: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let t_start = Instant::now();

    let sweep_id = args
        .sweep_id
        .clone()
        .unwrap_or_else(|| format!("scaleup-{}", short_ts()));
    let group_name = args
        .group_name
        .clone()
        .unwrap_or_else(|| format!("scaleup-{}", short_id_for_name(&sweep_id)));
    let queue_name = args
        .queue_name
        .clone()
        .unwrap_or_else(|| group_name.clone());
    let chunks_count = args.chunks.unwrap_or(args.replicas * 3);
    let chunks_key = args
        .chunks_key
        .clone()
        .unwrap_or_else(|| format!("runs/{sweep_id}/chunks.jsonl"));
    eprintln!(
        "[launcher] sweep_id={sweep_id} group={group_name} bucket={}",
        args.bucket
    );

    let api = SaladApi::new(&args.organization, &args.project, None)
        .context("build SaladApi (set SALAD_API_KEY or ~/.config/salad/credentials)")?;

    let gpu = resolve_gpu_classes(
        &api,
        &args.gpu_classes,
        args.max_price_per_hour,
        &args.price_priority,
        &args.gpu_class,
    )
    .await?;
    if gpu.ids.is_empty() {
        bail!("no GPU classes resolved");
    }
    let replicas_provisioned = compute_provisioned_replicas(
        args.replicas,
        args.replicas_overshoot,
        SALAD_ORG_REPLICA_QUOTA,
    );

    if args.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "name": group_name,
                "container": {
                    "image": args.image,
                    "resources": {
                        "cpu": args.cpu, "memory": args.memory_mib,
                        "gpu_classes": gpu.ids,
                    },
                },
                "replicas": replicas_provisioned,
                "replicas_requested": args.replicas,
                "replicas_overshoot": args.replicas_overshoot,
                "cells_per_chunk": args.cells_per_chunk,
                "gpu_class_names": gpu.names,
            }))?
        );
        eprintln!("[launcher] dry-run: NO Salad/R2 calls; exiting 0");
        return Ok(());
    }

    let parent = load_r2_parent_creds_or_env()?;
    let r2 = R2OperatorImpl::new(parent.clone());
    let filtered = apply_class_filter(
        &r2,
        args.prior_fleet_summary.as_deref(),
        gpu.names,
        gpu.ids,
        args.max_warmup_secs as u32,
        args.min_productive_chunks as f32,
    )
    .await;
    let (gpu_class_names, gpu_class_ids, classes_dropped_by_filter) =
        (filtered.names, filtered.ids, filtered.dropped);
    let classes_kept_after_filter = gpu_class_names.clone();

    // Scoped R2 cred (Salad-specific; never put root key in worker env).
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
    let scoped = api
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

    // Chunks → JSONL → R2.
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

    // Worker env.
    let mut env_h: HashMap<String, String> = HashMap::new();
    env_h.insert("SWEEP_RUN_ID".into(), sweep_id.clone());
    env_h.insert("R2_ACCOUNT_ID".into(), parent.account_id.clone());
    env_h.insert(
        "CHUNKS_R2".into(),
        format!("s3://{}/{}", args.bucket, chunks_key),
    );
    env_h.insert("SALAD_JOB_PORT".into(), "80".into());
    env_h.insert("RUST_LOG".into(), "info,zenfleet_salad=info".into());
    env_h.insert("METRICS".into(), "ssim2-gpu".into());
    inject_r2_cred_into_env(&mut env_h, &scoped);
    let env: BTreeMap<String, String> = env_h.into_iter().collect();

    let provider = SaladProviderHandle::new(
        api,
        SaladProviderConfig {
            organization: args.organization.clone(),
            project: args.project.clone(),
            group_name: group_name.clone(),
            queue_name,
            gpu_class_ids: gpu_class_ids.clone(),
            cpu: args.cpu,
            memory_mib: args.memory_mib,
            registry_auth: match (
                args.registry_username.as_ref(),
                args.registry_password.as_ref(),
            ) {
                (Some(u), Some(p)) => Some(RegistryAuth {
                    username: u.clone(),
                    password: p.clone(),
                }),
                _ => None,
            },
            restart_policy: "always".into(),
            autostart: true,
            queue_path: "/".into(),
            queue_port: 80,
        },
    );

    let sweep_cfg = SweepConfig {
        replicas: args.replicas,
        replicas_overshoot: args.replicas_overshoot,
        provider_replica_quota: SALAD_ORG_REPLICA_QUOTA,
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

    let fleet = FleetSweep::new(
        provider,
        r2,
        sweep_cfg,
        args.bucket.clone(),
        sweep_id.clone(),
        group_name.clone(),
        args.image.clone(),
        gpu_class_names.clone(),
        env,
        args.price_per_hour,
        json!({"gpu_class_ids": gpu_class_ids}),
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
    let spend = args.price_per_hour * f64::from(replicas_provisioned) * wall / 3600.0;
    let summary = LauncherSummary {
        sweep_id: fleet.sweep_id.clone(),
        group_name: fleet.group_name.clone(),
        image: fleet.image.clone(),
        replicas: replicas_provisioned,
        chunks: fleet.chunks,
        gpu_class: gpu_class_names.join("|"),
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
        classes_dropped_by_filter,
        classes_kept_after_filter,
    };
    println!("{}", serde_json::to_string(&summary).unwrap());

    if !fleet.teardown_ok && !args.keep_running {
        bail!(
            "teardown of container group {} did not succeed",
            fleet.group_name
        );
    }
    Ok(())
}
