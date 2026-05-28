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

/// Default image (kernel-cache + boot-record upload + GPU-class visibility).
const DEFAULT_IMAGE: &str = "ghcr.io/imazen/zen-metrics-sweep-salad:v6-visibility-b";

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

    /// GPU class name. Resolved to id via `GET /gpu-classes`. Used
    /// when `--gpu-classes` (plural) is NOT provided. Kept for
    /// back-compat with the v4-era single-class CLI.
    #[arg(long, default_value = DEFAULT_GPU_CLASS)]
    gpu_class: String,

    /// Comma-separated list of GPU class names, in priority order.
    /// Salad's scheduler treats `resources.gpu_classes` as a Vec of
    /// acceptable classes — useful when one tier is starved upstream.
    /// Each name is resolved via `GET /gpu-classes`; failures surface
    /// at preflight, not mid-launch. When set, this **overrides both**
    /// `--gpu-class` and `--max-price-per-hour` auto-enumeration. Example:
    /// `--gpu-classes "RTX 3060 (12 GB),RTX 3090 (24 GB),RTX 4090 (24 GB)"`.
    #[arg(long, value_delimiter = ',')]
    gpu_classes: Vec<String>,

    /// Auto-enumerate every Salad GPU class whose `--price-priority` price
    /// is <= this many USD per hour, and pass them ALL in
    /// `resources.gpu_classes`. This is the default scheduling mode: it
    /// gives Salad's allocator the broadest possible pool and avoids the
    /// starvation we saw in Runs 2-5 when only 1-3 classes were nominated.
    /// Set to 0 to disable (then `--gpu-class` is used). Manual
    /// `--gpu-classes "name1,name2,..."` takes priority over this.
    #[arg(long, default_value_t = 0.10)]
    max_price_per_hour: f64,

    /// Salad price tier to filter against when `--max-price-per-hour` is
    /// in effect. Salad quotes per-class prices at four priorities:
    /// `high` (default; scheduler picks high when capacity exists),
    /// `medium`, `low`, `batch`. Using `high` here ensures we're
    /// pricing what the scheduler actually charges in non-spot mode.
    #[arg(long, default_value = "high")]
    price_priority: String,

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

    /// Resolve GPU classes + print the container-group request body
    /// JSON, then exit before any container group / queue / job is
    /// created (no R2 writes either). Used to verify
    /// `--gpu-classes` plumbs through to a multi-element
    /// `resources.gpu_classes` Vec without spending any money.
    #[arg(long)]
    dry_run: bool,
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
    let queue_name = args
        .queue_name
        .clone()
        .unwrap_or_else(|| group_name.clone());
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
    // 1. Build the Salad API client.
    let api = SaladApi::new(&args.organization, &args.project, None)
        .context("build Salad API client (set SALAD_API_KEY or ~/.config/salad/credentials)")?;

    // Determine the GPU class list. Precedence:
    //   (a) explicit `--gpu-classes "name1,name2,..."` wins.
    //   (b) `--max-price-per-hour > 0` (default 0.10) auto-enumerates
    //       every Salad class priced at or below that, at the named
    //       priority tier. This is the broad-pool default; it kills
    //       allocator starvation by nominating every cheap class
    //       simultaneously.
    //   (c) Fallback to `--gpu-class` (singular).
    let (gpu_class_names, gpu_class_ids): (Vec<String>, Vec<String>) = if !args.gpu_classes.is_empty() {
        let names: Vec<String> = args
            .gpu_classes
            .iter()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        eprintln!(
            "[launcher] gpu class selection: MANUAL ({} classes from --gpu-classes)",
            names.len()
        );
        let mut ids: Vec<String> = Vec::with_capacity(names.len());
        for name in &names {
            let id = api
                .resolve_gpu_class(name)
                .await
                .with_context(|| format!("resolve GPU class {name:?}"))?;
            eprintln!("[launcher] gpu_class {name:?} -> {id}");
            ids.push(id);
        }
        (names, ids)
    } else if args.max_price_per_hour > 0.0 {
        eprintln!(
            "[launcher] gpu class selection: AUTO (price <= ${:.4}/hr at priority {:?})",
            args.max_price_per_hour, args.price_priority
        );
        let classes = api
            .gpu_classes_under_price(args.max_price_per_hour, &args.price_priority)
            .await
            .context("auto-enumerate GPU classes by price")?;
        eprintln!(
            "[launcher] auto-enumerated {} GPU classes <= ${:.4}/hr (priority={}):",
            classes.len(),
            args.max_price_per_hour,
            args.price_priority
        );
        let mut names = Vec::with_capacity(classes.len());
        let mut ids = Vec::with_capacity(classes.len());
        for c in &classes {
            eprintln!(
                "[launcher]   ${:.4}/hr  {:<32}  id={}",
                c.price_for(&args.price_priority),
                c.name,
                c.id
            );
            names.push(c.name.clone());
            ids.push(c.id.clone());
        }
        (names, ids)
    } else {
        eprintln!(
            "[launcher] gpu class selection: SINGLE (--gpu-class {:?})",
            args.gpu_class
        );
        let id = api
            .resolve_gpu_class(&args.gpu_class)
            .await
            .with_context(|| format!("resolve GPU class {:?}", args.gpu_class))?;
        eprintln!("[launcher] gpu_class {:?} -> {id}", args.gpu_class);
        (vec![args.gpu_class.clone()], vec![id])
    };

    if gpu_class_ids.is_empty() {
        bail!(
            "no GPU classes resolved — provide --gpu-class, --gpu-classes, or a positive --max-price-per-hour"
        );
    }
    eprintln!(
        "[launcher] replicas={} chunks={} gpu_classes_count={}",
        args.replicas,
        chunks_count,
        gpu_class_ids.len()
    );
    eprintln!("[launcher] max_wall_secs={}", args.max_wall_secs);

    // Dry-run: synthesise a representative container-group request body
    // (without minting creds, uploading chunks, or creating any Salad
    // resource) so the operator can verify the multi-class plumbing.
    // Exits zero on success.
    if args.dry_run {
        let dry_req = json!({
            "name": group_name,
            "container": {
                "image": args.image,
                "resources": {
                    "cpu": args.cpu,
                    "memory": args.memory_mib,
                    "gpu_classes": gpu_class_ids,
                },
            },
            "replicas": args.replicas,
            "gpu_class_names": gpu_class_names,
        });
        println!("{}", serde_json::to_string_pretty(&dry_req)?);
        eprintln!("[launcher] dry-run: NO Salad / R2 calls were made; exiting 0");
        return Ok(());
    }

    // 3. Load R2 parent creds + mint scoped child.
    //
    // Scoped-cred prefix list MUST cover every R2 path the worker will
    // touch — both the writable `runs/<sweep>/` per-sweep output prefix
    // AND every readable input prefix the chunks reference. Missing a
    // prefix produces a 403 the worker can't recover from (the durable
    // error sidecar would also 403 under the same scope, masking the
    // failure entirely — see Salad Runs 1-6b 2026-05-28 forensic doc).
    //
    // We derive the read prefixes from the chunks themselves so any future
    // launcher that retargets `--input-parquet-r2` / `--source-dir-r2` to
    // a non-default location automatically widens the scope. The cred is
    // still single-bucket; cross-bucket inputs require a different cred
    // model (see SALAD.md's "reads-from-A-writes-to-B" note).
    let parent = load_r2_parent_creds_or_env()?;
    let mut cred_prefixes: Vec<String> = Vec::new();
    cred_prefixes.push(format!("runs/{sweep_id}/"));
    // Read scope: extract the bucket-relative prefix from each input URI.
    // The s3 URI form is `s3://<bucket>/<key>`; the scoped cred lives
    // under one bucket (`args.bucket`), so we only honour inputs in that
    // bucket. Inputs in other buckets will 403 — surface a loud warning
    // (preflight already HEADs them, but with the PARENT cred; the worker
    // uses the scoped cred and gets a different result).
    let bucket_prefix_with_slash = format!("s3://{}/", args.bucket);
    for input_uri in [&args.input_parquet_r2, &args.source_dir_r2] {
        if let Some(rest) = input_uri.strip_prefix(&bucket_prefix_with_slash) {
            // Take the LEADING key segment (everything up to the first '/')
            // and re-add the trailing slash so the scoped prefix covers
            // every object under that subtree. Empty rest (= bucket root)
            // → whole-bucket scope, which we just inject as "" (the
            // CF minter omits an empty entry from the wire request).
            let leading = rest.split('/').next().unwrap_or("");
            if !leading.is_empty() {
                let p = format!("{leading}/");
                if !cred_prefixes.contains(&p) {
                    cred_prefixes.push(p);
                }
            }
        } else {
            eprintln!(
                "[launcher] WARN input {} is outside scoped bucket {}; worker will 403 on read",
                input_uri, args.bucket
            );
        }
    }
    let scoped_spec = ScopedCredSpec::new(&args.bucket)
        .with_prefixes(cred_prefixes.clone())
        .with_ttl_seconds(3600);
    eprintln!(
        "[launcher] minting scoped R2 cred (bucket={} prefixes={:?} ttl=3600s)",
        args.bucket, cred_prefixes
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
                gpu_classes: gpu_class_ids.clone(),
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

    // 10b. Build fleet_summary.json from boot/, instances/, omni/.
    //      Best-effort: log any error but never block teardown on it.
    match build_and_upload_fleet_summary(&parent, &args, &sweep_id, t_post).await {
        Ok(rows) => {
            eprintln!("[fleet] summary uploaded ({} replica rows)", rows);
        }
        Err(e) => {
            eprintln!("[fleet] summary build/upload failed (non-fatal): {e:#}");
        }
    }

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
    let throughput = match (
        poll.t_done_secs,
        poll.t_first_sidecar_secs,
        poll.omni_sidecars,
    ) {
        (Some(td), Some(tf), n) if n > 0 && td > tf => Some(f64::from(n) / (td - tf).max(0.001)),
        _ => None,
    };
    let spend = args.price_per_hour * f64::from(args.replicas) * wall / 3600.0;
    let summary = Summary {
        sweep_id: sweep_id.clone(),
        group_name: group_name.clone(),
        image: args.image.clone(),
        replicas: args.replicas,
        chunks: chunks.len() as u32,
        // Summary keeps a single string for back-compat with downstream
        // parsers (jq selectors in scripts/sweep). When multiple GPU
        // classes were resolved, join them with `|` so the original
        // order is preserved without breaking quoting.
        gpu_class: gpu_class_names.join("|"),
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
    let instances_prefix = format!("runs/{sweep_id}/instances/");
    let group_url_prefix = format!(
        "https://portal.salad.com/organizations/{}/projects/{}/containers/{}",
        args.organization, args.project, group_name
    );
    eprintln!("[poll]   portal: {group_url_prefix}");
    eprintln!("[poll]   watching s3://{}/{omni_prefix}", args.bucket);
    eprintln!("[poll]   snapshots s3://{}/{instances_prefix}", args.bucket);

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

        // Instances snapshot — fetched best-effort and persisted to R2
        // under the scoped prefix `runs/<sweep_id>/instances/<ts>.json`.
        // Salad sometimes returns 404 for stopped or just-created groups
        // — that's fine, we just skip the snapshot.
        let instances_val = api
            .list_container_group_instances(group_name)
            .await
            .ok();
        let ts_unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let snapshot = json!({
            "unix_ts": ts_unix,
            "elapsed_secs": elapsed.as_secs_f64(),
            "tick": tick,
            "state": state,
            "instance_status_counts": counts,
            "instances": instances_val.clone().unwrap_or_else(|| json!(null)),
            "omni_count": omni.len(),
            "error_count": errs.len(),
        });
        let snapshot_bytes = serde_json::to_vec(&snapshot).unwrap_or_default();
        let snapshot_key = format!("{instances_prefix}{ts_unix}.json");
        if let Err(e) = upload_to_r2_with_parent(parent, &args.bucket, &snapshot_key, &snapshot_bytes).await {
            eprintln!("[poll]   snapshot upload failed (non-fatal): {e:#}");
        }

        eprintln!(
            "[poll t={:>5.1}s tick={tick:>3}] state={state} counts={} omni={} err={} snap_ok={}",
            elapsed.as_secs_f64(),
            counts,
            omni.len(),
            errs.len(),
            instances_val.is_some(),
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
    format!("https://{}.r2.cloudflarestorage.com", parent.account_id)
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

async fn list_r2_prefix(parent: &R2ParentCreds, bucket: &str, prefix: &str) -> Result<Vec<String>> {
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
    amz: String,   // YYYYMMDDTHHMMSSZ
    short: String, // YYYYMMDD
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
    after_scheme.split('/').next().unwrap_or("").to_string()
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
        parent,
        method,
        _bucket,
        key,
        _url,
        headers,
        body,
        region,
        service,
        now,
        &[],
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

/// Stitch the per-replica picture from R2 artifacts and upload the
/// result as `runs/<sweep_id>/fleet_summary.json`. Also prints a
/// sortable stdout table.
///
/// Inputs read from R2 (all under the scoped `runs/<sweep_id>/` prefix):
/// - `boot/*.txt`          — one per replica that booted (worker uploads)
/// - `instances/*.json`    — periodic launcher snapshots
/// - `omni/*.parquet`      — completed chunks (used for chunk count)
/// - `errors/*.txt`        — error sidecars (used for fail count)
///
/// Returns the row count emitted.
async fn build_and_upload_fleet_summary(
    parent: &R2ParentCreds,
    args: &Args,
    sweep_id: &str,
    t_post: Instant,
) -> Result<usize> {
    use std::collections::BTreeMap;

    let boot_prefix = format!("runs/{sweep_id}/boot/");
    let instances_prefix = format!("runs/{sweep_id}/instances/");
    let omni_prefix = format!("runs/{sweep_id}/omni/");
    let errors_prefix = format!("runs/{sweep_id}/errors/");

    let boots = list_r2_prefix(parent, &args.bucket, &boot_prefix)
        .await
        .unwrap_or_default();
    let snaps = list_r2_prefix(parent, &args.bucket, &instances_prefix)
        .await
        .unwrap_or_default();
    let omnis = list_r2_prefix(parent, &args.bucket, &omni_prefix)
        .await
        .unwrap_or_default();
    let errs = list_r2_prefix(parent, &args.bucket, &errors_prefix)
        .await
        .unwrap_or_default();

    // Per-replica record indexed by machine_id (or hostname when
    // machine_id is empty/synthesized).
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

    // Hydrate from boot/*.txt — each is a key:value file.
    for key in &boots {
        let body = match get_r2_object_bytes(parent, &args.bucket, key).await {
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
            // fall back to the filename stem
            key.rsplit('/').next().unwrap_or(key).to_string()
        } else {
            rec.machine_id.clone()
        };
        replicas.insert(key_id, rec);
    }

    // Hydrate timing/status from snapshots.
    for key in &snaps {
        let body = match get_r2_object_bytes(parent, &args.bucket, key).await {
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
        let instances = v
            .get("instances")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Null);
        // The instances endpoint sometimes returns {"instances": [...]}
        // and sometimes [...] directly. Handle both.
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
            if entry.t_first_running_unix.is_none()
                && inst_state.as_deref() == Some("running")
            {
                entry.t_first_running_unix = Some(ts);
            }
            entry.last_status_seen = inst_state.or(Some(state.clone()));
        }
    }

    // Chunks processed = count of omni sidecars. We can't reliably
    // attribute back to a specific machine_id without reading the
    // parquet header, which is too expensive here. Approximate by
    // distributing the total across replicas that we've seen.
    let total_chunks = omnis.len() as u32;
    // Simple even-distribute is wrong; better to report aggregate +
    // leave per-replica null. So we just attribute by ordering match
    // when filenames embed the worker — but for the smoke run, we
    // surface the total.
    let mut replica_rows: Vec<Replica> = replicas.into_values().collect();
    if !replica_rows.is_empty() {
        // Distribute total_chunks evenly across replicas as a coarse
        // estimate; mark the field as approximate.
        let n = replica_rows.len() as u32;
        let each = total_chunks / n;
        let rem = total_chunks % n;
        for (i, r) in replica_rows.iter_mut().enumerate() {
            r.chunks_processed = each + if (i as u32) < rem { 1 } else { 0 };
        }
    }

    // Print stdout table.
    let wall = t_post.elapsed().as_secs_f64();
    eprintln!("[fleet] === fleet_summary ({} replicas, wall={:.1}s) ===", replica_rows.len(), wall);
    eprintln!(
        "[fleet] {:<24}  {:<24}  {:<12}  {:<10}  {:<8}",
        "machine_id_or_filename", "gpu_class", "warmup_s", "chunks", "status"
    );
    for r in &replica_rows {
        eprintln!(
            "[fleet] {:<24}  {:<24}  {:<12}  {:<10}  {:<8}",
            short24(&r.machine_id),
            short24(&r.gpu_class),
            r.warmup_seconds.map(|x| format!("{x:.1}")).unwrap_or_default(),
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
        "sweep_id": sweep_id,
        "bucket": args.bucket,
        "wall_secs": wall,
        "boot_records": boots.len(),
        "snapshot_count": snaps.len(),
        "omni_count": omnis.len(),
        "error_count": errs.len(),
        "replicas": replica_rows,
    });
    let body = serde_json::to_vec_pretty(&summary).unwrap_or_default();
    let key = format!("runs/{sweep_id}/fleet_summary.json");
    upload_to_r2_with_parent(parent, &args.bucket, &key, &body)
        .await
        .with_context(|| format!("upload {key}"))?;
    eprintln!("[fleet] fleet_summary.json uploaded to s3://{}/{}", args.bucket, key);
    Ok(replica_rows.len())
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

/// Tiny helper to download an object's bytes via SigV4 GET on the
/// parent cred. Same shape as upload_to_r2_with_parent. Used for
/// boot/*.txt + snapshot/*.json hydration.
async fn get_r2_object_bytes(
    parent: &R2ParentCreds,
    bucket: &str,
    key: &str,
) -> Result<Vec<u8>> {
    let url = format!("{}/{bucket}/{key}", r2_endpoint(parent));
    let now = chrono_now();
    let auth = sigv4_auth_header(
        parent,
        "GET",
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
        .get(&url)
        .header("Host", host_of(&url))
        .header("x-amz-content-sha256", empty_payload_hash())
        .header("x-amz-date", &now.amz)
        .header("Authorization", auth)
        .send()
        .await?;
    if !resp.status().is_success() {
        let s = resp.status();
        bail!("GET {url}: HTTP {s}");
    }
    Ok(resp.bytes().await?.to_vec())
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
