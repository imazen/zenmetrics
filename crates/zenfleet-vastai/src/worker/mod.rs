//! Worker mode — replaces the bash `onstart_omni_backfill.sh` dispatch
//! loop with a Rust-native async one.
//!
//! ## What it does
//!
//! The worker runs ONE-per-box inside the v21+ Docker image. Its job:
//!
//! 1. Fetch `chunks.jsonl` from R2 (the run's manifest of work units).
//! 2. Shuffle deterministically per-worker so 6+ boxes don't all
//!    stampede the same first chunk at boot.
//! 3. For each chunk: check the sidecar exists already (skip),
//!    otherwise race for a claim (token-based, stale-aware), then
//!    run the chunk through the existing [`crate::worker::chunk`]
//!    processor.
//! 4. Bound the number of in-flight chunks with a tokio
//!    [`Semaphore`]. The bound is initially set by
//!    [`adapt::auto_parallel_chunks`] (heuristic from host specs)
//!    and adjusted at runtime by an AIMD loop sampling
//!    `nvidia-smi`.
//!
//! ## Why Rust here
//!
//! The bash version had three structural issues this module fixes:
//!
//! - **Stale-claim races** were ad-hoc (the bash version
//!   sleep-then-read-back pattern works but is hard to reason about
//!   under load). We model the claim explicitly here.
//! - **`wait -n` is fixed-bound**. With adaptive concurrency we need
//!   to grow/shrink the in-flight count between chunks; a Semaphore's
//!   `add_permits` / `forget_permits` is the right primitive.
//! - **Subprocess fan-out from bash leaks file descriptors** when one
//!   chunk panics mid-encode; tokio's structured concurrency means
//!   we drop tasks cleanly on shutdown.
//!
//! ## Scope of phase A
//!
//! Phase A keeps the inner chunk-processor as a subprocess call to the
//! existing `omni_backfill_chunk_worker.sh` bash script. The Rust
//! worker is just the *outer* dispatcher. Phase B will move the chunk
//! processor in-process by calling `zenmetrics_cli::sweep::run_sweep`
//! directly, eliminating the 30× cubecl init per chunk (the biggest
//! remaining latency on a warm box).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Args;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{error, info, warn};

pub mod adapt;
mod chunk;
#[cfg(feature = "inline-sweep")]
mod chunk_input;
#[cfg(feature = "inline-sweep")]
mod chunk_output;
pub mod claim;
#[cfg(feature = "inline-sweep")]
mod feature_backfill;
#[cfg(feature = "inline-sweep")]
mod inline;
#[cfg(feature = "inline-sweep")]
pub mod r2_queue_loop;
#[cfg(feature = "source-features")]
mod source_features;
#[cfg(feature = "source-features")]
mod source_features_only;

#[cfg(feature = "inline-sweep")]
pub use feature_backfill::backfill_features_for_chunk;
// Re-exported so cloud-agnostic backends (e.g. `zenfleet-salad`) can
// reuse the SAME encode+score compute the vast.ai worker runs per chunk
// — the spec's "the compute closure is backend-agnostic". Takes the raw
// chunk line (the `Chunk.payload`), an R2 client, and the worker args.
#[cfg(feature = "inline-sweep")]
pub use inline::process_chunk_inline;
#[cfg(feature = "source-features")]
pub use source_features_only::backfill_source_features_for_chunk;
pub mod r2;
#[cfg(feature = "inline-sweep")]
mod sweep_runner;
mod util;

/// CLI arguments for the `zenfleet-vastai worker` subcommand.
///
/// Mirrors (and supersedes) the env-var contract the bash onstart
/// script reads from `/proc/1/environ`. CLI flags take precedence over
/// env vars; env vars fall back to compatible defaults.
#[derive(Args, Debug, Clone)]
pub struct WorkerArgs {
    /// Sweep run ID (e.g. `cvvdp-v15rc-2026-05-18`). Used to scope
    /// the claim namespace and the sidecar output path.
    #[arg(long, env = "SWEEP_RUN_ID")]
    pub run_id: String,

    /// R2 URI of the chunks manifest (one JSON object per line).
    /// Example: `s3://coefficient/jobs/<run>/chunks.jsonl`.
    #[arg(long, env = "CHUNKS_R2")]
    pub chunks_r2: String,

    /// Unique worker identifier for claim tokens. Distinguishes 6+
    /// boxes in one fleet. Defaults to the box's hostname or
    /// `$WORKER_ID` if set.
    #[arg(long, env = "WORKER_ID")]
    pub worker_id: Option<String>,

    /// Initial parallel-chunks-per-box budget. If unset,
    /// [`adapt::auto_parallel_chunks`] derives it from `nproc` +
    /// `nvidia-smi --query-gpu=memory.total`. Override e.g. for
    /// smoke runs.
    #[arg(long, env = "PARALLEL_CHUNKS")]
    pub parallel_chunks: Option<usize>,

    /// Hard ceiling on parallel-chunks (for the AIMD loop). If unset,
    /// derived from `nproc / 2` clamped to [1, 8].
    #[arg(long, env = "PARALLEL_CHUNKS_MAX")]
    pub parallel_chunks_max: Option<usize>,

    /// What kind of work to do per chunk. `omni` (default) re-encodes
    /// + scores + writes the omni sidecar. `feature-backfill` reads
    /// the existing omni sidecar from R2, downloads the already-saved
    /// encoded variants, computes zensim 300-feature vectors, writes
    /// a feature parquet to s3://zentrain/<run>/zensim_features/.
    /// Use feature-backfill when the omni sidecars exist already and
    /// you just want the features — saves the encode cost entirely.
    #[arg(long, env = "WORKER_MODE", default_value = "omni")]
    pub mode: String,

    /// Skip the R2 token-race claim. Used only by single-instance
    /// smoke runs that want to bypass claim contention with an
    /// already-running fleet. Production fleets MUST set this false.
    ///
    /// Accepts the union of bool forms our launchers historically
    /// passed: `0`/`1`, `yes`/`no`, `true`/`false`. The bash worker
    /// did `[[ "$SKIP_CLAIMS" != "0" ]]`; clap's default bool parser
    /// is strict (only "true"/"false"), so we use BoolishValueParser
    /// to match the bash semantics exactly.
    #[arg(
        long,
        env = "SKIP_CLAIMS",
        default_value_t = false,
        value_parser = clap::builder::BoolishValueParser::new()
    )]
    pub skip_claims: bool,

    /// Path to the bash chunk-processor script. Phase A subproc-call
    /// preserves the battle-tested bash worker. Phase B will inline
    /// the equivalent Rust via `zenmetrics_cli::sweep::run_sweep`.
    #[arg(
        long,
        env = "OMNI_WORKER_BIN",
        default_value = "/usr/local/bin/omni_backfill_chunk_worker.sh"
    )]
    pub chunk_worker_bin: PathBuf,

    /// Working directory for downloaded chunks + per-chunk scratch.
    /// Defaults to `/workspace/omni-backfill`. The chunk worker
    /// allocates a `<workdir>/<chunk_id>` subdir per chunk.
    #[arg(long, env = "WORKDIR", default_value = "/workspace/omni-backfill")]
    pub workdir: PathBuf,

    /// Adaptive concurrency: AIMD sample interval in seconds. Set 0
    /// to disable the AIMD loop entirely (PC stays at the initial
    /// value).
    #[arg(long, env = "ADAPT_INTERVAL_SEC", default_value_t = 60)]
    pub adapt_interval_sec: u64,

    /// Per-process chunk cap. After dispatching this many chunks the
    /// worker exits cleanly (code 0); the outer onstart loop is
    /// expected to respawn it. 0 = no cap (process until chunks
    /// exhausted). Default 20.
    ///
    /// Why: even with cubecl `memory_cleanup` between source images,
    /// the cubecl-cuda pool footprint creeps up over long runs (NVRTC
    /// PTX cache, fragmented allocations, retained planes from per-
    /// metric features). A bounded process lifetime resets that
    /// footprint to zero on respawn — much cheaper than fighting the
    /// pool growth in-process. The atomic claim discipline (sidecar
    /// existence + R2 claim file) ensures no chunk gets reprocessed
    /// across respawns.
    ///
    /// Skipped chunks (sidecar present, peer holds, claim race lost)
    /// DO count toward the cap — they each consumed a dispatch slot,
    /// and capping by "real work done" would unbounded the process
    /// lifetime when the corpus is sparse.
    #[arg(long, env = "MAX_CHUNKS_PER_PROCESS", default_value_t = 20)]
    pub max_chunks_per_process: usize,

    /// GPU util threshold for ramp-up. When the average GPU util
    /// over the last interval drops below this, the AIMD loop
    /// increments PC. Default 30%.
    #[arg(long, env = "ADAPT_RAMP_UP_BELOW", default_value_t = 30)]
    pub adapt_ramp_up_below: u32,

    /// GPU util threshold for back-off. When avg goes above this,
    /// AIMD decrements PC. Default 90%.
    #[arg(long, env = "ADAPT_BACK_OFF_ABOVE", default_value_t = 90)]
    pub adapt_back_off_above: u32,

    /// Path to s5cmd binary. Default `s5cmd` on PATH.
    #[arg(long, env = "S5CMD_BIN", default_value = "s5cmd")]
    pub s5cmd_bin: String,

    /// R2 endpoint URL. If unset, derive from $R2_ACCOUNT_ID
    /// (`https://<account>.r2.cloudflarestorage.com`). The deferred
    /// derivation is what every other tool in the fleet does, so we
    /// match for consistency.
    #[arg(long, env = "R2_ENDPOINT")]
    pub r2_endpoint: Option<String>,

    /// s5cmd profile name. Default `r2` (matches the bash convention
    /// + ~/.aws/credentials shape we ship).
    #[arg(long, env = "S5CMD_PROFILE", default_value = "r2")]
    pub s5cmd_profile: String,
}

/// Entry point for `zenfleet-vastai worker`. Sets up tracing, derives the
/// effective worker id, builds the R2 client, fetches chunks, then
/// hands off to the async dispatcher.
pub fn cmd_worker(args: WorkerArgs) -> Result<()> {
    init_tracing();

    // Hydrate env from /proc/1/environ — vast.ai sets R2 creds, run
    // id, etc. into the container's pid-1 environment but those vars
    // don't reach a non-pid-1 process unless we copy them out. The
    // bash equivalent did this in onstart's header.
    hydrate_pid1_env();

    // Defensive: write ~/.aws/credentials from env so s5cmd's `--profile`
    // lookup succeeds even on backends whose onstart didn't write the
    // file. vast.ai's `onstart_v3.sh` writes static creds via bash
    // already; this is a no-op when the env vars are absent and
    // overwrites identically when present.
    if let Err(e) = provision_aws_credentials_file(&args.s5cmd_profile) {
        warn!(error = %e, "provision_aws_credentials_file failed; s5cmd may 403");
    }

    let worker_id = args.worker_id.clone().unwrap_or_else(default_worker_id);
    let r2 = r2::new_from_args(&args)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    rt.block_on(run_worker_async(args, worker_id, r2))
}

async fn run_worker_async(args: WorkerArgs, worker_id: String, r2: r2::R2Client) -> Result<()> {
    info!(worker_id = %worker_id, run_id = %args.run_id, "worker starting");

    // Boot report — upload /var/run/zen-boot.txt (written by
    // entrypoint_salad.sh) to <scoped-prefix>/boot/<machine_id>.txt
    // for fleet visibility. Best-effort; never blocks the worker.
    upload_boot_record(&args, &worker_id, &r2).await;

    let initial_pc = args
        .parallel_chunks
        .unwrap_or_else(adapt::auto_parallel_chunks);
    let pc_max = args
        .parallel_chunks_max
        .unwrap_or_else(adapt::derive_pc_max);
    info!(initial_pc, pc_max, "concurrency configured");

    // Pull chunks.jsonl from R2 into memory. Even at 2568 lines × 1KB,
    // that's <3 MB — fits cleanly in RAM. Avoids holding an fd to a
    // temp file across the dispatcher's lifetime.
    let chunks = r2
        .fetch_chunks_jsonl(&args.chunks_r2)
        .await
        .context("download chunks.jsonl")?;
    info!(n_chunks = chunks.len(), "chunks downloaded");

    // Seeded shuffle: same worker_id always sees the same ordering
    // (useful for resume) but different workers see different orders
    // (kills claim contention at boot).
    let shuffled = util::seeded_shuffle(&chunks, &worker_id);

    // The Semaphore is the in-flight bound. add_permits grows it; the
    // AIMD loop calls add_permits / forget_permits via the
    // adapt::PcController interface (see adapt.rs).
    let sem = Arc::new(Semaphore::new(initial_pc));
    let pc_ctrl = Arc::new(adapt::PcController::new(initial_pc, pc_max, sem.clone()));

    // Spawn the AIMD loop if interval > 0.
    let adapt_handle = if args.adapt_interval_sec > 0 {
        let ctrl = pc_ctrl.clone();
        let interval = Duration::from_secs(args.adapt_interval_sec);
        let ramp_up = args.adapt_ramp_up_below;
        let back_off = args.adapt_back_off_above;
        Some(tokio::spawn(async move {
            adapt::run_aimd_loop(ctrl, interval, ramp_up, back_off).await;
        }))
    } else {
        None
    };

    // Per-chunk dispatcher. We `JoinSet` instead of `tokio::spawn`
    // bare so a SIGTERM / panic in main drops outstanding tasks
    // cleanly (no leaked subprocesses).
    let mut tasks: JoinSet<()> = JoinSet::new();
    let mut shutdown = util::shutdown_signal();
    let mut chunk_iter = shuffled.into_iter();

    let chunk_cap = args.max_chunks_per_process;
    if chunk_cap > 0 {
        info!(
            chunk_cap,
            "per-process chunk cap enabled; process will exit after \
             dispatching N chunks so the outer onstart loop respawns \
             and resets the cubecl pool footprint to zero"
        );
    }
    let mut dispatched: usize = 0;
    let mut cap_hit = false;

    'dispatch: loop {
        if chunk_cap > 0 && dispatched >= chunk_cap {
            info!(
                chunk_cap,
                dispatched,
                "per-process chunk cap reached; stopping dispatch (outer \
                 onstart loop will respawn worker with fresh cubecl pool)"
            );
            cap_hit = true;
            break 'dispatch;
        }
        tokio::select! {
            biased;
            _ = &mut shutdown => {
                warn!("shutdown signal received; draining in-flight chunks");
                break 'dispatch;
            }
            Some(line) = std::future::ready(chunk_iter.next()) => {
                // Acquire permit. `acquire_owned` ties the permit's
                // lifetime to the task, so when the task drops the
                // permit returns to the semaphore.
                let permit = sem.clone().acquire_owned().await
                    .context("semaphore closed")?;
                let r2 = r2.clone();
                let args = args.clone();
                let worker_id = worker_id.clone();
                tasks.spawn(async move {
                    if let Err(e) = chunk::process_chunk(
                        &args, &worker_id, &r2, &line,
                    ).await {
                        error!(error = %e, "process_chunk failed");
                    }
                    drop(permit);
                });
                dispatched += 1;
            }
            else => break 'dispatch,
        }
    }

    // Drain.
    while let Some(_res) = tasks.join_next().await {}
    if let Some(h) = adapt_handle {
        h.abort();
    }
    if cap_hit {
        info!(
            dispatched,
            "chunk cap drained; worker exiting cleanly for respawn"
        );
    } else {
        info!(dispatched, "all chunks processed; worker exiting");
    }
    Ok(())
}

fn default_worker_id() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("WORKER_ID"))
        .unwrap_or_else(|_| {
            // No HOSTNAME — synthesize from pid + nanos so each
            // process is at least distinguishable in logs.
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            format!("worker-{pid}-{nanos}")
        })
}

fn init_tracing() {
    // RUST_LOG=info by default. Operator can crank to debug via env.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();
}

/// vast.ai puts R2 credentials + the sweep run id into the container's
/// pid-1 environment, not into every spawned process. The bash
/// onstart copies them out by reading `/proc/1/environ` (which is
/// NUL-separated `KEY=VALUE` strings). We do the same.
//
// The single `unsafe { set_var }` below is the crate's only unsafe;
// it is pre-existing, documented, and called before the tokio runtime
// spins up (single-threaded). Scoped allow keeps the crate-level
// `deny(unsafe_code)` gate active everywhere else.
#[allow(unsafe_code)]
pub fn hydrate_pid1_env() {
    let Ok(buf) = std::fs::read("/proc/1/environ") else {
        return;
    };
    for entry in buf.split(|b| *b == 0) {
        let s = String::from_utf8_lossy(entry);
        let Some((k, v)) = s.split_once('=') else {
            continue;
        };
        // Only copy variables we care about — don't pollute env.
        if matches!(
            k,
            "R2_ACCOUNT_ID"
                | "R2_ACCESS_KEY_ID"
                | "R2_SECRET_ACCESS_KEY"
                // Scoped R2 creds (minted per-sweep on Hetzner +
                // anywhere else we don't trust the worker host) carry
                // a session token. s5cmd reads it from
                // `~/.aws/credentials` -> `provision_aws_credentials_file`
                // writes it from these env vars. Either name is accepted
                // because some backends inject one or the other.
                | "AWS_SESSION_TOKEN"
                | "R2_SESSION_TOKEN"
                | "SWEEP_RUN_ID"
                | "WORKER_ID"
                | "PARALLEL"
                | "PARALLEL_CHUNKS"
                | "PARALLEL_CHUNKS_MAX"
                | "GPU_RUNTIME"
                | "METRICS"
                | "CHUNKS_R2"
                | "SKIP_CLAIMS"
                | "CONTAINER_ID"
                | "CONTAINER_API_KEY"
                | "ADAPT_INTERVAL_SEC"
                | "ZENSIM_FEATURES_REGIME"
                | "JOBS"
                | "MAX_CHUNKS_PER_PROCESS"
                // Hetzner + any future R2-queue-poll backend reads
                // BUCKET + CHUNKS_QUEUE_PREFIX from env; without these
                // in the allowlist the propagation gap reappears for
                // non-pid-1 callers.
                | "BUCKET"
                | "CHUNKS_QUEUE_PREFIX"
                | "R2_QUEUE_POLL_SECS"
                | "R2_QUEUE_IDLE_EXIT_SECS"
        ) && std::env::var_os(k).is_none()
        {
            // SAFETY: we're single-threaded at this point (called
            // before tokio runtime starts).
            // Soundness: writing to env from one thread is fine.
            // The MT-unsafety warning applies when other threads are
            // reading env concurrently.
            // v is a Cow<str> from from_utf8_lossy. Convert to owned String
            // so set_var doesn't borrow from a buffer we're about to drop.
            let v_str: String = v.to_string();
            unsafe {
                std::env::set_var(k, v_str);
            }
        }
    }
}

/// Write the s5cmd credentials profile to `~/.aws/credentials` from
/// env vars.
///
/// s5cmd is invoked with `--profile <name>` (default `r2`, see
/// [`WorkerArgs::s5cmd_profile`]) and reads creds from the standard
/// AWS credentials file. Backends that inject creds via container env
/// vars only (Hetzner cloud-init's `--env-file`, scoped per-sweep R2
/// tokens, anything not running `entrypoint_hetzner.sh` /
/// `entrypoint_salad.sh`) leave that file unwritten — every s5cmd call
/// then 403s silently because the named profile doesn't exist. This
/// closes that gap: read `R2_ACCESS_KEY_ID` /
/// `R2_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN` (or `R2_SESSION_TOKEN`
/// as fallback) from env, write the profile.
///
/// Idempotent — overwrites the file if present so a re-spawned worker
/// always sees the env's view of the world (e.g. after the launcher
/// rotates the scoped token).
///
/// `profile_name` matches `WorkerArgs::s5cmd_profile` (default `r2`).
///
/// Returns `Ok(true)` if the file was written, `Ok(false)` if there
/// were no creds in env (vast.ai's `cmd_worker` calls this defensively
/// — its onstart already wrote the file via bash, so a no-op return is
/// fine), and `Err` for filesystem failures.
pub fn provision_aws_credentials_file(profile_name: &str) -> Result<bool> {
    let Ok(access) = std::env::var("R2_ACCESS_KEY_ID") else {
        return Ok(false);
    };
    let Ok(secret) = std::env::var("R2_SECRET_ACCESS_KEY") else {
        return Ok(false);
    };
    // Either env name is accepted; AWS_SESSION_TOKEN wins. The session
    // token is REQUIRED when scoped (temporary) R2 creds are in use; it
    // is absent for static creds.
    let session = std::env::var("AWS_SESSION_TOKEN")
        .ok()
        .or_else(|| std::env::var("R2_SESSION_TOKEN").ok());

    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("HOME env not set; cannot write ~/.aws/credentials")?;
    let aws_dir = home.join(".aws");
    std::fs::create_dir_all(&aws_dir)
        .with_context(|| format!("create dir {}", aws_dir.display()))?;
    let cred_path = aws_dir.join("credentials");

    let mut body = String::new();
    body.push_str(&format!("[{profile_name}]\n"));
    body.push_str(&format!("aws_access_key_id = {access}\n"));
    body.push_str(&format!("aws_secret_access_key = {secret}\n"));
    if let Some(token) = session.as_deref() {
        body.push_str(&format!("aws_session_token = {token}\n"));
    }

    std::fs::write(&cred_path, body).with_context(|| format!("write {}", cred_path.display()))?;
    // 0600 on Unix — credentials file shouldn't be world-readable.
    // Best-effort; not a failure if chmod isn't available (e.g. tests on
    // a non-Unix host, though this binary is Linux-only in practice).
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        let _ = std::fs::set_permissions(&cred_path, perms);
    }
    info!(
        path = %cred_path.display(),
        profile = profile_name,
        has_session_token = session.is_some(),
        "wrote AWS credentials file for s5cmd"
    );
    Ok(true)
}

/// Derive the bucket-scoped prefix from `chunks_r2`.
///
/// `chunks_r2` is `s3://<bucket>/<prefix>/chunks.jsonl`. Returns
/// `(bucket, prefix_with_slash)` — e.g. `("zen-tuning-ephemeral",
/// "runs/scaleup-2026-05-28T010203/")`.
pub(crate) fn split_chunks_uri(chunks_r2: &str) -> Option<(String, String)> {
    let rest = chunks_r2.strip_prefix("s3://")?;
    let (bucket, key) = rest.split_once('/')?;
    // Strip the trailing `chunks.jsonl` (or whatever filename) leaving
    // the prefix with a trailing `/`.
    let prefix = match key.rfind('/') {
        Some(idx) => &key[..=idx],
        None => "",
    };
    Some((bucket.to_string(), prefix.to_string()))
}

/// Public wrapper so backends like Salad (which run their own worker
/// loop instead of `cmd_worker`) can fire boot-record upload at
/// startup. Reads `/var/run/zen-boot.txt`, synthesizes a minimal one
/// if absent, uploads to `<scoped-prefix>/boot/<worker_id>.txt`.
/// Best-effort — never fails the caller.
pub async fn fire_boot_upload(args: &WorkerArgs, worker_id: &str, r2: &r2::R2Client) {
    upload_boot_record(args, worker_id, r2).await
}

/// Best-effort: read /var/run/zen-boot.txt (written by
/// entrypoint_salad.sh) and upload it to R2 at
/// `s3://<bucket>/<prefix>boot/<worker_id>.txt`. Failures are logged
/// at `warn` and never block worker startup — boot records are
/// observational, not load-bearing.
async fn upload_boot_record(args: &WorkerArgs, worker_id: &str, r2: &r2::R2Client) {
    // 1. Find the local boot file (env wins; default to the path
    //    the entrypoint writes).
    let local_path =
        std::env::var("ZEN_BOOT_INFO_FILE").unwrap_or_else(|_| "/var/run/zen-boot.txt".to_string());
    let local = std::path::PathBuf::from(&local_path);
    if !local.exists() {
        // Not Salad (no entrypoint wrote it) — synthesize a minimal
        // record so the launcher always sees SOMETHING per replica.
        let synth = format!(
            "machine_id: {worker_id}\n\
             hostname: {hostname}\n\
             salad_machine_id: {salad}\n\
             gpu_class: {gpu_class}\n\
             warmup_seconds: {warm}\n\
             boot_unix_ts: {ts}\n\
             synthesized: true\n",
            worker_id = worker_id,
            hostname = std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".into()),
            salad = std::env::var("SALAD_MACHINE_ID").unwrap_or_default(),
            gpu_class = std::env::var("ZEN_BOOT_GPU_CLASS").unwrap_or_else(|_| "unknown".into()),
            warm = std::env::var("ZEN_BOOT_WARMUP_SECONDS").unwrap_or_default(),
            ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        );
        if tokio::fs::write(&local, synth.as_bytes()).await.is_err() {
            // Fall back to a writable tmp file.
            let tmp = std::env::temp_dir().join("zen-boot.txt");
            if let Err(e) = tokio::fs::write(&tmp, synth.as_bytes()).await {
                warn!(error = %e, "could not write synthesized boot record");
                return;
            }
            return upload_boot_to_r2(args, worker_id, r2, &tmp).await;
        }
    }
    upload_boot_to_r2(args, worker_id, r2, &local).await;
}

async fn upload_boot_to_r2(
    args: &WorkerArgs,
    worker_id: &str,
    r2: &r2::R2Client,
    local: &std::path::Path,
) {
    let Some((bucket, prefix)) = split_chunks_uri(&args.chunks_r2) else {
        warn!(chunks_r2 = %args.chunks_r2, "could not derive bucket/prefix; skipping boot upload");
        return;
    };
    // Sanitize worker_id for an object key (hostnames can have ':' etc.).
    let safe: String = worker_id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let uri = format!("s3://{bucket}/{prefix}boot/{safe}.txt");
    match r2.upload(local, &uri).await {
        Ok(()) => info!(uri = %uri, "boot record uploaded"),
        Err(e) => warn!(uri = %uri, error = %e, "boot record upload failed (non-fatal)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct unit on the file-writing logic: given a tempdir as `HOME`,
    /// write all three env vars, expect a complete profile body.
    ///
    /// We can't simply spam `std::env::set_var` in unit tests because
    /// cargo-test runs them in parallel threads sharing the process env;
    /// two concurrent tests that twiddle `R2_*` clobber each other.
    /// Serialize the env-mutating tests with a mutex.
    use std::sync::Mutex;
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn clear_env() {
        // Single-threaded under ENV_MUTEX; safe to set_var.
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("R2_ACCESS_KEY_ID");
            std::env::remove_var("R2_SECRET_ACCESS_KEY");
            std::env::remove_var("AWS_SESSION_TOKEN");
            std::env::remove_var("R2_SESSION_TOKEN");
        }
    }

    #[test]
    fn provision_writes_full_profile_with_session_token() {
        let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("HOME");
        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("HOME", tmp.path());
            std::env::set_var("R2_ACCESS_KEY_ID", "AKIAFAKE");
            std::env::set_var("R2_SECRET_ACCESS_KEY", "SECRETFAKE");
            std::env::set_var("AWS_SESSION_TOKEN", "SESSIONFAKE");
        }
        let wrote = provision_aws_credentials_file("r2").expect("provision");
        assert!(wrote, "should report wrote=true when creds present");
        let body = std::fs::read_to_string(tmp.path().join(".aws").join("credentials")).unwrap();
        assert!(body.contains("[r2]"), "profile header missing: {body}");
        assert!(
            body.contains("aws_access_key_id = AKIAFAKE"),
            "akid missing: {body}"
        );
        assert!(
            body.contains("aws_secret_access_key = SECRETFAKE"),
            "secret missing: {body}"
        );
        assert!(
            body.contains("aws_session_token = SESSIONFAKE"),
            "session token missing — this is the iter-5 bug: {body}"
        );

        // Restore env.
        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            match prev_home {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn provision_accepts_r2_session_token_as_fallback() {
        let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("HOME");
        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("HOME", tmp.path());
            std::env::set_var("R2_ACCESS_KEY_ID", "AKIAFAKE");
            std::env::set_var("R2_SECRET_ACCESS_KEY", "SECRETFAKE");
            // No AWS_SESSION_TOKEN; only R2_SESSION_TOKEN.
            std::env::set_var("R2_SESSION_TOKEN", "R2SESSIONFAKE");
        }
        let wrote = provision_aws_credentials_file("r2").expect("provision");
        assert!(wrote);
        let body = std::fs::read_to_string(tmp.path().join(".aws").join("credentials")).unwrap();
        assert!(
            body.contains("aws_session_token = R2SESSIONFAKE"),
            "R2_SESSION_TOKEN fallback didn't land: {body}"
        );

        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            match prev_home {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn provision_no_op_when_creds_absent() {
        let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("HOME");
        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("HOME", tmp.path());
        }
        let wrote = provision_aws_credentials_file("r2").expect("no-op should not error");
        assert!(!wrote, "should report wrote=false when no creds");
        assert!(
            !tmp.path().join(".aws").join("credentials").exists(),
            "no file should be written when creds absent"
        );

        #[allow(unsafe_code)]
        unsafe {
            match prev_home {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }

    #[test]
    fn provision_custom_profile_name() {
        let _g = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().expect("tempdir");
        let prev_home = std::env::var_os("HOME");
        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("HOME", tmp.path());
            std::env::set_var("R2_ACCESS_KEY_ID", "AKIAFAKE");
            std::env::set_var("R2_SECRET_ACCESS_KEY", "SECRETFAKE");
        }
        provision_aws_credentials_file("custom-profile").expect("provision");
        let body = std::fs::read_to_string(tmp.path().join(".aws").join("credentials")).unwrap();
        assert!(
            body.contains("[custom-profile]"),
            "custom profile header missing: {body}"
        );
        assert!(
            !body.contains("aws_session_token"),
            "no session token expected"
        );

        clear_env();
        #[allow(unsafe_code)]
        unsafe {
            match prev_home {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}
