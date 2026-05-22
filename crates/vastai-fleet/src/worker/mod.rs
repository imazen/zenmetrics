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
//! processor in-process by calling `zen_metrics_cli::sweep::run_sweep`
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

mod adapt;
mod chunk;
#[cfg(feature = "inline-sweep")]
mod chunk_input;
#[cfg(feature = "inline-sweep")]
mod chunk_output;
mod claim;
#[cfg(feature = "inline-sweep")]
mod feature_backfill;
#[cfg(feature = "inline-sweep")]
mod inline;
#[cfg(feature = "source-features")]
mod source_features;
#[cfg(feature = "source-features")]
mod source_features_only;

#[cfg(feature = "inline-sweep")]
pub use feature_backfill::backfill_features_for_chunk;
#[cfg(feature = "source-features")]
pub use source_features_only::backfill_source_features_for_chunk;
mod r2;
#[cfg(feature = "inline-sweep")]
mod sweep_runner;
mod util;

/// CLI arguments for the `vastai-fleet worker` subcommand.
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
    /// the equivalent Rust via `zen_metrics_cli::sweep::run_sweep`.
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

/// Entry point for `vastai-fleet worker`. Sets up tracing, derives the
/// effective worker id, builds the R2 client, fetches chunks, then
/// hands off to the async dispatcher.
pub fn cmd_worker(args: WorkerArgs) -> Result<()> {
    init_tracing();

    // Hydrate env from /proc/1/environ — vast.ai sets R2 creds, run
    // id, etc. into the container's pid-1 environment but those vars
    // don't reach a non-pid-1 process unless we copy them out. The
    // bash equivalent did this in onstart's header.
    hydrate_pid1_env();

    let worker_id = args.worker_id.clone().unwrap_or_else(default_worker_id);
    let r2 = r2::R2Client::new(&args)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;

    rt.block_on(run_worker_async(args, worker_id, r2))
}

async fn run_worker_async(
    args: WorkerArgs,
    worker_id: String,
    r2: r2::R2Client,
) -> Result<()> {
    info!(worker_id = %worker_id, run_id = %args.run_id, "worker starting");

    let initial_pc = args.parallel_chunks.unwrap_or_else(adapt::auto_parallel_chunks);
    let pc_max = args
        .parallel_chunks_max
        .unwrap_or_else(adapt::derive_pc_max);
    info!(initial_pc, pc_max, "concurrency configured");

    // Pull chunks.jsonl from R2 into memory. Even at 2568 lines × 1KB,
    // that's <3 MB — fits cleanly in RAM. Avoids holding an fd to a
    // temp file across the dispatcher's lifetime.
    let chunks = r2.fetch_chunks_jsonl(&args.chunks_r2).await
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

    'dispatch: loop {
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
            }
            else => break 'dispatch,
        }
    }

    // Drain.
    while let Some(_res) = tasks.join_next().await {}
    if let Some(h) = adapt_handle {
        h.abort();
    }
    info!("all chunks processed; worker exiting");
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
fn hydrate_pid1_env() {
    let Ok(buf) = std::fs::read("/proc/1/environ") else {
        return;
    };
    for entry in buf.split(|b| *b == 0) {
        let s = String::from_utf8_lossy(entry);
        let Some((k, v)) = s.split_once('=') else { continue };
        // Only copy variables we care about — don't pollute env.
        if matches!(
            k,
            "R2_ACCOUNT_ID"
                | "R2_ACCESS_KEY_ID"
                | "R2_SECRET_ACCESS_KEY"
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
