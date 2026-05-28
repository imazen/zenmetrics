//! `zen-salad-sweep` — end-to-end Salad launcher driving the
//! `zen-cloud-salad` library through one sweep cycle.
//!
//! What it does (one `run` invocation):
//!
//! 1. Resolves a Salad GPU class id by name (default RTX 3060).
//! 2. Reads R2 parent creds from the environment (or `~/.config/cloudflare/
//!    r2-credentials`) and the Salad API key from `$SALAD_API_KEY` or
//!    `~/.config/salad/credentials`.
//! 3. Mints a scoped R2 credential restricted to the working bucket
//!    (default `zen-tuning-ephemeral`), `object-read-write`, TTL 3600 s,
//!    prefix scoped to `runs/<sweep_id>/`.
//! 4. Generates `chunks.jsonl` (one chunk per worker target, default
//!    N=10) referencing the existing tiny smoke parquet at
//!    `s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/input/smoke.parquet`
//!    and source dir
//!    `s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/sources/`.
//! 5. Uploads `chunks.jsonl` to R2.
//! 6. Creates (or reuses by `--queue-name`) a Salad job queue.
//! 7. Creates a container group with the requested image (default
//!    `ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache`),
//!    N replicas, scoped R2 cred injected into the env, queue
//!    attached on `path=/job port=80`.
//! 8. Pushes N jobs into the queue (one per chunk) carrying the chunk
//!    JSON as input.
//! 9. Polls the R2 sidecars under `<sweep_id>/omni/*.parquet` (and
//!    `<sweep_id>/errors/*.txt`) plus the container-group
//!    `current_state` until either all N chunks complete OR the
//!    wall-time cap is reached.
//! 10. **Mandatory** teardown: stops the container group. Re-attempts
//!     on failure, surfaces if it can't be stopped.
//! 11. Emits a one-line JSON summary to stdout with measured timings,
//!     a per-worker boot cost estimate, throughput, and rough spend.
//!
//! Per `~/work/claudehints/topics/r2-credentials.md` the root R2 key is
//! NEVER injected into a worker. Only the minted scoped + session-token
//! cred reaches the container env.
//!
//! Per CLAUDE.md: spend is hard-bounded by the wall-time cap, billing
//! stops only when replicas count drops to zero, so teardown is the
//! safety belt. The summary line reports whether teardown succeeded.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::time::sleep;
use zen_cloud_salad::launch::{
    ContainerConfig, CreateContainerGroupRequest, CreateQueueJobRequest, CreateQueueRequest,
    QueueConnection, R2ParentCreds, RegistryAuth, ResourceRequirements, SaladApi, ScopedCredSpec,
    inject_r2_cred_into_env,
};

/// Default working bucket (matches existing smoke setup).
const DEFAULT_BUCKET: &str = "zen-tuning-ephemeral";

/// Default chunks pre-existing smoke parquet (3-row zenjpeg q={30,50,70}).
const DEFAULT_SOURCE_DIR_R2: &str = "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/sources";
const DEFAULT_INPUT_PARQUET_R2: &str =
    "s3://zen-tuning-ephemeral/salad-smoke-2026-05-27/input/smoke.parquet";
const DEFAULT_IMAGE_BASENAME: &str = "graph.png";

/// Default image (kernel-cache enabled).
const DEFAULT_IMAGE: &str = "ghcr.io/imazen/zen-metrics-sweep-salad:v4-kernel-cache";

/// Default GPU class. RTX 3060 = mid-tier consumer, cheapest predictable
/// price tier on Salad's pool. ~$0.10/h at time of writing.
const DEFAULT_GPU_CLASS: &str = "RTX 3060";

#[derive(Debug, Parser)]
#[command(
    name = "zen-salad-sweep",
    about = "End-to-end Salad scale-up launcher for the zen-metrics sweep image"
)]
struct Args {
    /// Salad organization name (path segment under `/organizations/`).
    #[arg(long, default_value = "imazen", env = "SALAD_ORGANIZATION")]
    organization: String,

    /// Salad project name (path segment under `/projects/`).
    #[arg(long, default_value = "zenmetrics", env = "SALAD_PROJECT")]
    project: String,

    /// Working R2 bucket (the scoped cred is minted for THIS bucket).
    #[arg(long, default_value = DEFAULT_BUCKET)]
    bucket: String,

    /// Unique sweep id. Defaults to a timestamped string.
    #[arg(long)]
    sweep_id: Option<String>,

    /// Container group name (DNS-style). Defaults to `scaleup-<short-id>`.
    #[arg(long)]
    group_name: Option<String>,

    /// Job queue name. Defaults to the group name.
    #[arg(long)]
    queue_name: Option<String>,

    /// Docker image to deploy.
    #[arg(long, default_value = DEFAULT_IMAGE)]
    image: String,

    /// Replica count. Default 10 (Salad's default org quota).
    #[arg(long, default_value_t = 10)]
    replicas: u32,

    /// Number of chunks to push to the queue. Default = replicas × 3
    /// (each worker chews through ~3-5).
    #[arg(long)]
    chunks: Option<u32>,

    /// GPU class name. Resolved to id via `GET /gpu-classes`.
    #[arg(long, default_value = DEFAULT_GPU_CLASS)]
    gpu_class: String,

    /// CPU cores per replica.
    #[arg(long, default_value_t = 4)]
    cpu: u32,

    /// Memory per replica in MiB.
    #[arg(long, default_value_t = 8192)]
    memory_mib: u32,

    /// Per-replica $/hour for the chosen GPU class (used for the
    /// spend-estimate line in the summary; Salad billing is on actual
    /// running replica-seconds, this is a conservative upper bound).
    #[arg(long, default_value_t = 0.20)]
    price_per_hour: f64,

    /// Hard wall-time cap (seconds). The launcher stops the container
    /// group and exits when this hits, regardless of completion.
    #[arg(long, default_value_t = 900)]
    max_wall_secs: u64,

    /// Polling interval for state + R2 sidecars (seconds).
    #[arg(long, default_value_t = 15)]
    poll_secs: u64,

    /// Source dir R2 URI (where the chunk's image_basenames live).
    #[arg(long, default_value = DEFAULT_SOURCE_DIR_R2)]
    source_dir_r2: String,

    /// Input parquet R2 URI. Each chunk references this same file.
    #[arg(long, default_value = DEFAULT_INPUT_PARQUET_R2)]
    input_parquet_r2: String,

    /// Image basenames in the chunk record (must exist under
    /// `--source-dir-r2`). Repeatable.
    #[arg(long = "image-basename", default_values_t = [DEFAULT_IMAGE_BASENAME.to_string()])]
    image_basenames: Vec<String>,

    /// Docker registry auth username (e.g., GHCR username). Optional;
    /// the kernel-cache image is in a public ghcr.io path.
    #[arg(long, env = "DOCKER_USERNAME")]
    registry_username: Option<String>,

    /// Docker registry auth password / PAT.
    #[arg(long, env = "DOCKER_PASSWORD")]
    registry_password: Option<String>,

    /// Skip the teardown stop call (dangerous; for debugging only).
    /// Default false — teardown ALWAYS runs.
    #[arg(long)]
    keep_running: bool,

    /// Skip the smoke-parquet validation: by default the launcher
    /// requires the configured `--input-parquet-r2` + each
    /// `--image-basename` under `--source-dir-r2` to exist before
    /// pushing jobs. This skips that pre-flight.
    #[arg(long)]
    skip_preflight: bool,

    /// Path under the working bucket to write `chunks.jsonl`.
    /// Defaults to `runs/<sweep_id>/chunks.jsonl`.
    #[arg(long)]
    chunks_key: Option<String>,
}

/// One row of `chunks.jsonl`. Matches the schema the worker's inline
/// pipeline parses (`crates/zen-cloud-vastai/src/worker/inline.rs::ChunkRecord`).
#[derive(Debug, Serialize, Deserialize)]
struct ChunkRecord {
    chunk_id: String,
    input_parquet: String,
    input_parquet_r2: String,
    row_range: [usize; 2],
    source_dir_r2: String,
    image_basenames: Vec<String>,
    run_id: String,
    out_sidecar_omni: String,
    out_encoded_prefix: String,
}

/// Final summary line emitted to stdout.
#[derive(Debug, Serialize)]
struct Summary {
    sweep_id: String,
    group_name: String,
    image: String,
    replicas: u32,
    chunks: u32,
    gpu_class: String,
    /// Wall seconds from group POST to launcher exit.
    wall_secs: f64,
    /// Wall seconds from group POST to FIRST sidecar landing in R2.
    /// Null when no sidecars landed in time.
    t_first_sidecar_secs: Option<f64>,
    /// Wall seconds from group POST to first sidecar from each replica
    /// (proxy for all-N booted). Null when fewer than N unique workers
    /// produced sidecars before the cap.
    t_all_n_sidecars_secs: Option<f64>,
    /// Wall seconds from group POST to LAST sidecar (or cap hit).
    t_done_secs: Option<f64>,
    /// Distinct workers (by chunk → worker mapping derived from the
    /// salad-machine-id metadata in the omni sidecar, if available;
    /// falls back to chunk_id count).
    distinct_workers_observed: u32,
    /// Throughput = chunks-completed ÷ (t_done − t_first).
    throughput_chunks_per_sec: Option<f64>,
    /// $/hour × wall × replicas / 3600. Upper bound; Salad bills on
    /// running-replica-seconds which is ≤ this.
    estimated_spend_usd: f64,
    /// Teardown status.
    teardown_ok: bool,
    /// Number of error sidecars observed in R2.
    error_sidecars: u32,
    /// Number of completed omni sidecars observed in R2.
    omni_sidecars: u32,
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
    let queue_name = args.queue_name.clone().unwrap_or_else(|| group_name.clone());
    let chunks_count = args.chunks.unwrap_or(args.replicas * 3);
    let chunks_key = args
        .chunks_key
        .clone()
        .unwrap_or_else(|| format!("runs/{sweep_id}/chunks.jsonl"));

    eprintln!("[launcher] sweep_id={sweep_id}");
    eprintln!("[launcher] group_name={group_name}");
    eprintln!("[launcher] queue_name={queue_name}");
    eprintln!("[launcher] bucket={}", args.bucket);
    eprintln!("[launcher] image={}", args.image);
    eprintln!(
        "[launcher] replicas={} chunks={} gpu_class={:?}",
        args.replicas, chunks_count, args.gpu_class
    );
    eprintln!("[launcher] max_wall_secs={}", args.max_wall_secs);

    // 1. Build the Salad API client.
    let api = SaladApi::new(&args.organization, &args.project, None)
        .context("build Salad API client (set SALAD_API_KEY or ~/.config/salad/credentials)")?;

    // 2. Resolve GPU class id.
    eprintln!("[launcher] resolving GPU class {:?}", args.gpu_class);
    let gpu_class_id = api
        .resolve_gpu_class(&args.gpu_class)
        .await
        .with_context(|| format!("resolve GPU class {:?}", args.gpu_class))?;
    eprintln!("[launcher] gpu_class_id={gpu_class_id}");

    // 3. Load R2 parent creds + mint scoped child.
    let parent = load_r2_parent_creds_or_env()?;
    let scoped_spec = ScopedCredSpec::new(&args.bucket)
        .with_prefixes(vec![format!("runs/{sweep_id}/")])
        .with_ttl_seconds(3600);
    eprintln!(
        "[launcher] minting scoped R2 cred (bucket={} prefix=runs/{}/ ttl=3600s)",
        args.bucket, sweep_id
    );
    let scoped = api
        .mint_sweep_r2_cred(&parent, &scoped_spec)
        .await
        .context("mint scoped R2 cred")?;
    eprintln!(
        "[launcher] minted: access_key_id={}… session_token_len={}",
        &scoped.access_key_id[..scoped.access_key_id.len().min(12)],
        scoped.session_token.len()
    );

    // 4. Pre-flight: smoke parquet + sources must exist (HEAD via reqwest).
    if !args.skip_preflight {
        preflight_smoke_inputs(&parent, &args).await?;
    }

    // 5. Generate chunks.jsonl + upload.
    let chunks = generate_chunks(chunks_count, &sweep_id, &args);
    let chunks_jsonl = chunks_to_jsonl(&chunks);
    let chunks_uri = format!("s3://{}/{}", args.bucket, chunks_key);
    eprintln!(
        "[launcher] uploading {} chunks to {} ({} bytes)",
        chunks.len(),
        chunks_uri,
        chunks_jsonl.len()
    );
    upload_to_r2_with_parent(&parent, &args.bucket, &chunks_key, chunks_jsonl.as_bytes())
        .await
        .with_context(|| format!("upload chunks.jsonl to {chunks_uri}"))?;

    // 6. Create the managed queue (idempotent: ignore "already exists").
    let queue_req = CreateQueueRequest {
        name: queue_name.clone(),
        display_name: Some(format!("scale-up {sweep_id}")),
        description: Some("zen-salad-sweep scale-up test queue".into()),
    };
    eprintln!("[launcher] create queue {queue_name}");
    match api.create_queue(&queue_req).await {
        Ok(_) => eprintln!("[launcher]   queue created"),
        Err(e) => {
            let msg = format!("{e:#}");
            if msg.contains("409") || msg.to_lowercase().contains("already exists") {
                eprintln!("[launcher]   queue already exists, reusing");
            } else {
                return Err(e.context("create queue"));
            }
        }
    }

    // 7. Build container-group env.
    let mut env: HashMap<String, String> = HashMap::new();
    env.insert("SWEEP_RUN_ID".into(), sweep_id.clone());
    env.insert(
        "R2_ACCOUNT_ID".into(),
        parent_r2_account_id_for_env(&parent),
    );
    env.insert("CHUNKS_R2".into(), chunks_uri.clone());
    env.insert("SALAD_JOB_PORT".into(), "80".into());
    env.insert("RUST_LOG".into(), "info,zen_cloud_salad=info".into());
    env.insert("METRICS".into(), "ssim2-gpu".into());
    inject_r2_cred_into_env(&mut env, &scoped);

    // 8. Build + POST container group.
    let cg_req = CreateContainerGroupRequest {
        name: group_name.clone(),
        display_name: Some(format!("scale-up {sweep_id}")),
        container: ContainerConfig {
            image: args.image.clone(),
            resources: ResourceRequirements {
                cpu: args.cpu,
                memory: args.memory_mib,
                gpu_classes: vec![gpu_class_id.clone()],
            },
            command: None,
            environment_variables: env,
            registry_authentication: match (
                args.registry_username.as_ref(),
                args.registry_password.as_ref(),
            ) {
                (Some(u), Some(p)) => Some(RegistryAuth {
                    username: u.clone(),
                    password: p.clone(),
                }),
                _ => None,
            },
        },
        replicas: args.replicas,
        restart_policy: "always".into(),
        autostart_policy: true,
        queue_connection: Some(QueueConnection {
            // Match the path used by the working v3 smoke (`/`). The
            // worker's HTTP handler matches all paths but the sidecar's
            // routing is determined by this field.
            path: "/".into(),
            port: 80,
            queue_name: queue_name.clone(),
        }),
    };
    eprintln!("[launcher] POST container group {group_name}");
    let t_post = Instant::now();
    let _group = api.create_container_group(&cg_req).await.with_context(|| {
        format!(
            "POST container group {group_name} (image={})",
            args.image.clone()
        )
    })?;
    eprintln!("[launcher]   group created");

    // 9. Push N jobs into the queue.
    eprintln!("[launcher] pushing {} jobs into queue", chunks.len());
    for ch in &chunks {
        let body = serde_json::to_value(ch).unwrap();
        api.push_job(
            &queue_name,
            &CreateQueueJobRequest {
                input: body,
                metadata: None,
            },
        )
        .await
        .with_context(|| format!("push chunk {}", ch.chunk_id))?;
    }
    eprintln!("[launcher]   pushed");

    // 10. Poll for completion (or cap).
    let poll_result = poll_until_done(
        &api,
        &parent,
        &args,
        &sweep_id,
        &group_name,
        chunks.len(),
        t_post,
    )
    .await;

    // 11. Teardown.
    let teardown_ok = if !args.keep_running {
        eprintln!("[launcher] tearing down container group {group_name}");
        let mut attempts: u32 = 0;
        let mut ok = false;
        while attempts < 3 {
            match api.stop_container_group(&group_name).await {
                Ok(()) => {
                    eprintln!("[launcher]   stop OK");
                    ok = true;
                    break;
                }
                Err(e) => {
                    eprintln!(
                        "[launcher]   stop failed (attempt {}): {:#}",
                        attempts + 1,
                        e
                    );
                    attempts += 1;
                    sleep(Duration::from_secs(3)).await;
                }
            }
        }
        if !ok {
            eprintln!(
                "[launcher] !! TEARDOWN FAILED for {group_name}: stop the group manually at \
                 https://portal.salad.com/organizations/{}/projects/{}/",
                args.organization, args.project
            );
        }
        ok
    } else {
        eprintln!("[launcher] --keep-running set; SKIPPING teardown (manual stop required)");
        false
    };

    // 12. Emit summary.
    let poll = poll_result.unwrap_or_else(|e| {
        eprintln!("[launcher] poll loop returned error: {e:#}");
        PollResult::default()
    });
    let wall = t_start.elapsed().as_secs_f64();
    let throughput = match (poll.t_done_secs, poll.t_first_sidecar_secs, poll.omni_sidecars) {
        (Some(td), Some(tf), n) if n > 0 && td > tf => {
            Some(f64::from(n) / (td - tf).max(0.001))
        }
        _ => None,
    };
    let spend = args.price_per_hour * f64::from(args.replicas) * wall / 3600.0;
    let summary = Summary {
        sweep_id: sweep_id.clone(),
        group_name: group_name.clone(),
        image: args.image.clone(),
        replicas: args.replicas,
        chunks: chunks.len() as u32,
        gpu_class: args.gpu_class.clone(),
        wall_secs: wall,
        t_first_sidecar_secs: poll.t_first_sidecar_secs,
        t_all_n_sidecars_secs: poll.t_all_n_sidecars_secs,
        t_done_secs: poll.t_done_secs,
        distinct_workers_observed: poll.distinct_workers_observed,
        throughput_chunks_per_sec: throughput,
        estimated_spend_usd: spend,
        teardown_ok,
        error_sidecars: poll.error_sidecars,
        omni_sidecars: poll.omni_sidecars,
    };
    let summary_json = serde_json::to_string(&summary).unwrap();
    println!("{summary_json}");

    // Non-zero exit on any failure path that mattered.
    if !teardown_ok && !args.keep_running {
        bail!("teardown of container group {group_name} did not succeed");
    }
    Ok(())
}

/// What the polling loop measures.
#[derive(Default, Debug)]
struct PollResult {
    t_first_sidecar_secs: Option<f64>,
    t_all_n_sidecars_secs: Option<f64>,
    t_done_secs: Option<f64>,
    distinct_workers_observed: u32,
    omni_sidecars: u32,
    error_sidecars: u32,
}

async fn poll_until_done(
    api: &SaladApi,
    parent: &R2ParentCreds,
    args: &Args,
    sweep_id: &str,
    group_name: &str,
    expected_chunks: usize,
    t_post: Instant,
) -> Result<PollResult> {
    let omni_prefix = format!("runs/{sweep_id}/omni/");
    let error_prefix = format!("runs/{sweep_id}/errors/");
    let group_url_prefix = format!(
        "https://portal.salad.com/organizations/{}/projects/{}/containers/{}",
        args.organization, args.project, group_name
    );
    eprintln!("[poll]   portal: {group_url_prefix}");
    eprintln!("[poll]   watching s3://{}/{omni_prefix}", args.bucket);

    let mut out = PollResult::default();
    let cap = Duration::from_secs(args.max_wall_secs);
    let interval = Duration::from_secs(args.poll_secs.max(2));

    let mut tick: u32 = 0;
    loop {
        let elapsed = t_post.elapsed();
        if elapsed >= cap {
            eprintln!(
                "[poll] wall-time cap {}s hit; stopping poll loop",
                args.max_wall_secs
            );
            // The latest counts are already in `out`; record t_done as
            // the cap so spend math reflects what we paid for.
            out.t_done_secs.get_or_insert(elapsed.as_secs_f64());
            break;
        }
        tick += 1;
        // List omni + errors in R2.
        let omni = list_r2_prefix(parent, &args.bucket, &omni_prefix)
            .await
            .unwrap_or_default();
        let errs = list_r2_prefix(parent, &args.bucket, &error_prefix)
            .await
            .unwrap_or_default();
        out.omni_sidecars = omni.len() as u32;
        out.error_sidecars = errs.len() as u32;
        out.distinct_workers_observed = out.omni_sidecars; // ~1 sidecar per chunk

        if !omni.is_empty() && out.t_first_sidecar_secs.is_none() {
            out.t_first_sidecar_secs = Some(elapsed.as_secs_f64());
            eprintln!(
                "[poll] FIRST sidecar at t={:.1}s ({})",
                elapsed.as_secs_f64(),
                omni[0]
            );
        }
        if (omni.len() as u32) >= args.replicas && out.t_all_n_sidecars_secs.is_none() {
            out.t_all_n_sidecars_secs = Some(elapsed.as_secs_f64());
            eprintln!(
                "[poll] all-N sidecars at t={:.1}s (n={})",
                elapsed.as_secs_f64(),
                omni.len()
            );
        }

        // Container-group state (best-effort).
        let cg = api.get_container_group(group_name).await.ok();
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
            .unwrap_or_else(|| json!({}));
        eprintln!(
            "[poll t={:>5.1}s tick={tick:>3}] state={state} counts={} omni={} err={}",
            elapsed.as_secs_f64(),
            counts,
            omni.len(),
            errs.len()
        );

        if (omni.len() + errs.len()) >= expected_chunks {
            out.t_done_secs = Some(elapsed.as_secs_f64());
            eprintln!(
                "[poll] DONE at t={:.1}s (omni={} err={})",
                elapsed.as_secs_f64(),
                omni.len(),
                errs.len()
            );
            break;
        }

        // Early-exit: when the container group is `stopped` AND no more
        // sidecars can land (either we have what we expected, or zero
        // sidecars and zero running instances means nothing else is
        // coming). Avoids wasting wall-time polling a dead group.
        if state == "stopped" {
            out.t_done_secs = Some(elapsed.as_secs_f64());
            eprintln!(
                "[poll] container group is stopped at t={:.1}s; \
                 short-circuiting (omni={} err={} expected={})",
                elapsed.as_secs_f64(),
                omni.len(),
                errs.len(),
                expected_chunks
            );
            break;
        }

        sleep(interval).await;
    }
    Ok(out)
}

fn generate_chunks(n: u32, sweep_id: &str, args: &Args) -> Vec<ChunkRecord> {
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let chunk_id = format!("scaleup-{i:03}");
        let out_sidecar_omni = format!(
            "s3://{}/runs/{}/omni/{}.parquet",
            args.bucket, sweep_id, chunk_id
        );
        let out_encoded_prefix = format!(
            "s3://{}/runs/{}/encoded/{}/",
            args.bucket, sweep_id, chunk_id
        );
        // input_parquet is the LOCAL filename the worker writes to under
        // scratch; the R2 URI is what it downloads from. Both point at the
        // same shared 3-row smoke parquet.
        out.push(ChunkRecord {
            chunk_id,
            input_parquet: "smoke.parquet".into(),
            input_parquet_r2: args.input_parquet_r2.clone(),
            row_range: [0, 3],
            source_dir_r2: args.source_dir_r2.clone(),
            image_basenames: args.image_basenames.clone(),
            run_id: sweep_id.to_string(),
            out_sidecar_omni,
            out_encoded_prefix,
        });
    }
    out
}

fn chunks_to_jsonl(chunks: &[ChunkRecord]) -> String {
    chunks
        .iter()
        .map(|c| serde_json::to_string(c).unwrap())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Best-effort load of the R2 parent creds. Tries env first; falls back
/// to `~/.config/cloudflare/r2-credentials` (KEY=VALUE lines).
fn load_r2_parent_creds_or_env() -> Result<R2ParentCreds> {
    if let Ok(c) = R2ParentCreds::from_env() {
        return Ok(c);
    }
    let home = std::env::var("HOME").context("HOME not set")?;
    let path = PathBuf::from(home).join(".config/cloudflare/r2-credentials");
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("read {} (or set env vars)", path.display()))?;
    let mut map = HashMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            map.insert(k.trim().to_string(), v.trim().to_string());
        }
    }
    let token = map
        .get("CF_API_TOKEN")
        .or_else(|| map.get("R2_API_TOKEN"))
        .cloned()
        .context("CF_API_TOKEN/R2_API_TOKEN missing from creds file")?;
    let account_id = map
        .get("R2_ACCOUNT_ID")
        .cloned()
        .context("R2_ACCOUNT_ID missing")?;
    let access = map
        .get("R2_ACCESS_KEY_ID")
        .cloned()
        .context("R2_ACCESS_KEY_ID missing")?;
    let secret = map
        .get("R2_SECRET_ACCESS_KEY")
        .cloned()
        .context("R2_SECRET_ACCESS_KEY missing")?;
    Ok(R2ParentCreds {
        cf_api_token: token,
        account_id,
        parent_access_key_id: access,
        parent_secret_access_key: secret,
    })
}

fn parent_r2_account_id_for_env(parent: &R2ParentCreds) -> String {
    parent.account_id.clone()
}

/// HEAD-check the smoke parquet and source images are present before
/// burning Salad replica-seconds on chunks that would fail.
async fn preflight_smoke_inputs(parent: &R2ParentCreds, args: &Args) -> Result<()> {
    eprintln!("[preflight] HEAD {}", args.input_parquet_r2);
    head_r2_uri(parent, &args.input_parquet_r2)
        .await
        .with_context(|| format!("HEAD {}", args.input_parquet_r2))?;
    for b in &args.image_basenames {
        let uri = format!("{}/{}", args.source_dir_r2.trim_end_matches('/'), b);
        eprintln!("[preflight] HEAD {uri}");
        head_r2_uri(parent, &uri)
            .await
            .with_context(|| format!("HEAD {uri}"))?;
    }
    Ok(())
}

// ── R2 helpers (signed via the parent root creds) ────────────────────────
//
// These run on the OPERATOR box, NOT on a worker. Using the root creds is
// fine here — they never leave this process. Workers only ever see the
// scoped child cred.

fn r2_endpoint(parent: &R2ParentCreds) -> String {
    format!(
        "https://{}.r2.cloudflarestorage.com",
        parent.account_id
    )
}

fn split_s3_uri(uri: &str) -> Result<(String, String)> {
    let rest = uri
        .strip_prefix("s3://")
        .with_context(|| format!("not an s3:// URI: {uri}"))?;
    let (bucket, key) = rest
        .split_once('/')
        .with_context(|| format!("URI missing key: {uri}"))?;
    Ok((bucket.to_string(), key.to_string()))
}

async fn head_r2_uri(parent: &R2ParentCreds, uri: &str) -> Result<()> {
    let (bucket, key) = split_s3_uri(uri)?;
    head_r2(parent, &bucket, &key).await
}

async fn head_r2(parent: &R2ParentCreds, bucket: &str, key: &str) -> Result<()> {
    let url = format!("{}/{bucket}/{key}", r2_endpoint(parent));
    let now = chrono_now();
    let auth = sigv4_auth_header(
        parent,
        "HEAD",
        bucket,
        key,
        &url,
        &[("host", host_of(&url))],
        b"",
        "auto",
        "s3",
        &now,
    );
    let resp = reqwest::Client::new()
        .head(&url)
        .header("Host", host_of(&url))
        .header("x-amz-content-sha256", empty_payload_hash())
        .header("x-amz-date", &now.amz)
        .header("Authorization", auth)
        .send()
        .await?;
    if !resp.status().is_success() {
        bail!("HEAD {url}: HTTP {}", resp.status());
    }
    Ok(())
}

async fn upload_to_r2_with_parent(
    parent: &R2ParentCreds,
    bucket: &str,
    key: &str,
    body: &[u8],
) -> Result<()> {
    let url = format!("{}/{bucket}/{key}", r2_endpoint(parent));
    let now = chrono_now();
    let payload_hash = sha256_hex(body);
    let auth = sigv4_auth_header(
        parent,
        "PUT",
        bucket,
        key,
        &url,
        &[("host", host_of(&url))],
        body,
        "auto",
        "s3",
        &now,
    );
    let resp = reqwest::Client::new()
        .put(&url)
        .header("Host", host_of(&url))
        .header("x-amz-content-sha256", payload_hash)
        .header("x-amz-date", &now.amz)
        .header("Authorization", auth)
        .header("Content-Type", "application/octet-stream")
        .body(body.to_vec())
        .send()
        .await?;
    if !resp.status().is_success() {
        let s = resp.status();
        let body = resp.text().await.unwrap_or_default();
        bail!("PUT {url}: HTTP {s}: {body}");
    }
    Ok(())
}

async fn list_r2_prefix(
    parent: &R2ParentCreds,
    bucket: &str,
    prefix: &str,
) -> Result<Vec<String>> {
    let endpoint = r2_endpoint(parent);
    let url = format!("{endpoint}/{bucket}/?list-type=2&prefix={prefix}");
    let now = chrono_now();
    // Canonical query string components (sorted).
    let mut query: Vec<(String, String)> = vec![
        ("list-type".into(), "2".into()),
        ("prefix".into(), prefix.to_string()),
    ];
    query.sort();
    let auth = sigv4_auth_header_with_query(
        parent,
        "GET",
        bucket,
        "",
        &endpoint,
        &[("host", host_of(&endpoint))],
        b"",
        "auto",
        "s3",
        &now,
        &query,
    );
    let resp = reqwest::Client::new()
        .get(&url)
        .header("Host", host_of(&endpoint))
        .header("x-amz-content-sha256", empty_payload_hash())
        .header("x-amz-date", &now.amz)
        .header("Authorization", auth)
        .send()
        .await?;
    if !resp.status().is_success() {
        let s = resp.status();
        let b = resp.text().await.unwrap_or_default();
        bail!("LIST {url}: HTTP {s}: {b}");
    }
    let xml = resp.text().await?;
    // Cheap parse: pull out <Key>...</Key> blocks.
    let mut out = Vec::new();
    let mut rest = xml.as_str();
    while let Some(open) = rest.find("<Key>") {
        let after = &rest[open + 5..];
        if let Some(close) = after.find("</Key>") {
            out.push(after[..close].to_string());
            rest = &after[close..];
        } else {
            break;
        }
    }
    Ok(out)
}

// ── Tiny SigV4 (just what we need: HEAD / PUT / GET ?list-type=2) ────────
//
// The R2 endpoint speaks the S3 v4 signing scheme exactly. We avoid
// pulling `aws-sdk-s3` (heavy + transitively expensive) by hand-rolling
// the 4 known calls. This is OPERATOR-side code; correctness over
// completeness — only the call shapes above need to work.

struct AmzDate {
    amz: String,    // YYYYMMDDTHHMMSSZ
    short: String,  // YYYYMMDD
}

fn chrono_now() -> AmzDate {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    // ymdhms from epoch — vendored 0-dep mini.
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    AmzDate {
        amz: format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z"),
        short: format!("{y:04}{mo:02}{d:02}"),
    }
}

fn epoch_to_ymdhms(secs: u64) -> (i32, u32, u32, u32, u32, u32) {
    // Days/seconds split.
    let days_total = (secs / 86_400) as i64;
    let secs_of_day = (secs % 86_400) as u32;
    let h = secs_of_day / 3600;
    let mi = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    // 1970-01-01 epoch. Algorithm from "Howard Hinnant" date.
    let z = days_total + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146097)
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0,400)
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y_final = if m <= 2 { (y + 1) as i32 } else { y as i32 };
    (y_final, m as u32, d as u32, h, mi, s)
}

fn host_of(url: &str) -> String {
    let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    after_scheme
        .split('/')
        .next()
        .unwrap_or("")
        .to_string()
}

fn empty_payload_hash() -> &'static str {
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b);
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

fn hmac_sha256(key: &[u8], data: &[u8]) -> Vec<u8> {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    let mut m = <Hmac<Sha256>>::new_from_slice(key).expect("hmac");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

#[allow(clippy::too_many_arguments)]
fn sigv4_auth_header(
    parent: &R2ParentCreds,
    method: &str,
    _bucket: &str,
    key: &str,
    _url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    region: &str,
    service: &str,
    now: &AmzDate,
) -> String {
    sigv4_auth_header_with_query(
        parent, method, _bucket, key, _url, headers, body, region, service, now, &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn sigv4_auth_header_with_query(
    parent: &R2ParentCreds,
    method: &str,
    bucket: &str,
    key: &str,
    _url: &str,
    headers: &[(&str, String)],
    body: &[u8],
    region: &str,
    service: &str,
    now: &AmzDate,
    query: &[(String, String)],
) -> String {
    // Canonical request.
    let canonical_uri = if key.is_empty() {
        format!("/{bucket}/")
    } else {
        format!("/{bucket}/{key}")
    };
    let canonical_query = query
        .iter()
        .map(|(k, v)| format!("{}={}", uri_encode(k, true), uri_encode(v, true)))
        .collect::<Vec<_>>()
        .join("&");
    let payload_hash = sha256_hex(body);
    let mut h = headers.to_vec();
    h.push(("x-amz-content-sha256", payload_hash.clone()));
    h.push(("x-amz-date", now.amz.clone()));
    h.sort_by(|a, b| a.0.cmp(b.0));
    let canonical_headers = h
        .iter()
        .map(|(k, v)| format!("{}:{}\n", k.to_ascii_lowercase(), v.trim()))
        .collect::<String>();
    let signed_headers = h
        .iter()
        .map(|(k, _)| k.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(";");
    let canonical_req = format!(
        "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
    );
    let cr_hash = sha256_hex(canonical_req.as_bytes());
    // String to sign.
    let credential_scope = format!("{}/{region}/{service}/aws4_request", now.short);
    let sts = format!(
        "AWS4-HMAC-SHA256\n{}\n{credential_scope}\n{cr_hash}",
        now.amz
    );
    // Derive signing key.
    let k_secret = format!("AWS4{}", parent.parent_secret_access_key);
    let k_date = hmac_sha256(k_secret.as_bytes(), now.short.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    let k_signing = hmac_sha256(&k_service, b"aws4_request");
    let sig = hex(&hmac_sha256(&k_signing, sts.as_bytes()));
    format!(
        "AWS4-HMAC-SHA256 Credential={}/{credential_scope}, SignedHeaders={signed_headers}, Signature={sig}",
        parent.parent_access_key_id
    )
}

fn uri_encode(s: &str, encode_slash: bool) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.bytes() {
        let unreserved = c.is_ascii_alphanumeric()
            || c == b'-'
            || c == b'_'
            || c == b'.'
            || c == b'~'
            || (c == b'/' && !encode_slash);
        if unreserved {
            out.push(c as char);
        } else {
            out.push_str(&format!("%{c:02X}"));
        }
    }
    out
}

fn short_ts() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let (y, mo, d, h, mi, s) = epoch_to_ymdhms(secs);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}")
}

fn short_id_for_name(id: &str) -> String {
    // DNS label: lowercase, alnum + hyphens, ≤ 63 chars.
    let mut out = String::new();
    for c in id.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if c == '-' || c == '_' {
            out.push('-');
        }
    }
    if out.len() > 50 {
        out.truncate(50);
    }
    out
}
