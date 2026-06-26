//! `zenfleet-sweep` — the cloud-agnostic deployed compute binary.
//!
//! This is the single compute binary baked into the deploy docker
//! image (spec §1.2/§1.3). It runs the "claim job → fetch inputs →
//! compute → upload artifacts → heartbeat" loop, parameterized over the
//! [`zenfleet_cloud`] trait layer, and selects which cloud provider to
//! use at runtime via `--backend <name>`.
//!
//! ## Backends
//!
//! - `vastai` (Phase A): `zenfleet-sweep worker --backend vastai`
//!   dispatches to the proven async worker in `zenfleet-vastai` — the
//!   SAME code path the `zenfleet-vastai worker` binary runs — so the
//!   produced sweep artifacts are byte-identical to today's worker.
//!
//! Backend selection is cargo features + trait objects, NOT dlopen
//! (spec §1.6 decision 4).

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand, ValueEnum};

/// Cloud backend selector — the runtime side of the cargo-feature-gated
/// backend set.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    /// vast.ai + Cloudflare R2 + `/proc/1/environ` credentials.
    Vastai,
    /// Localhost (no cloud) + filesystem queue + filesystem storage +
    /// env/.env credentials. The no-spend dev / abstraction-validation
    /// backend (Phase B).
    Local,
    /// Hetzner Cloud + R2-queue polling (no managed queue) + BYO R2.
    Hetzner,
}

/// Cloud-agnostic sweep worker.
#[derive(Parser, Debug)]
#[command(
    name = "zenfleet-sweep",
    version,
    about = "Cloud-agnostic sweep worker — claim → fetch → compute → upload → beat over a pluggable cloud backend"
)]
struct Cli {
    /// Which compiled-in cloud backend to run against. Defaults to
    /// `vastai`.
    #[arg(long, value_enum, default_value_t = Backend::Vastai, global = true)]
    backend: Backend,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the sweep worker loop. The compute closure is the selected
    /// backend's encode+score sweep — byte-identical to the legacy
    /// `zenfleet-vastai worker` for `--backend vastai`.
    //
    // The args are vast.ai's `WorkerArgs`, which are backend-agnostic
    // enough (run id, chunks manifest, workdir, s5cmd/R2 config, mode)
    // to drive every backend. Available under any backend that pulls
    // `zenfleet-vastai/worker`.
    #[cfg(feature = "_vastai-backend")]
    Worker(zenfleet_vastai::worker::WorkerArgs),
}

// Without a backend feature the `Cmd` enum is empty, so `cli`/the match arm
// look unused/unreachable; the real builds always enable a backend.
#[allow(unused_variables, unreachable_code, unreachable_patterns)]
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "_vastai-backend")]
        Cmd::Worker(wargs) => run_worker_backend(cli.backend, wargs),
        #[cfg(not(feature = "_vastai-backend"))]
        _ => anyhow::bail!(
            "no cloud backend compiled in; build with --features vastai \
             (Phase A)"
        ),
    }
}

/// Dispatch the worker to the selected backend.
#[cfg(feature = "_vastai-backend")]
fn run_worker_backend(
    backend: Backend,
    wargs: zenfleet_vastai::worker::WorkerArgs,
) -> anyhow::Result<()> {
    match backend {
        // vast.ai: call the proven `cmd_worker` directly — the exact
        // code path the legacy `zenfleet-vastai worker` binary runs, so the
        // sweep output is byte-identical.
        Backend::Vastai => zenfleet_vastai::worker::cmd_worker(wargs),

        #[cfg(feature = "_local-backend")]
        Backend::Local => local::run(wargs),

        #[cfg(not(feature = "_local-backend"))]
        Backend::Local => anyhow::bail!(
            "--backend local selected but the local backend is not compiled \
             in; rebuild with --features local (glue) or --features local-sweep \
             (full encode+score)"
        ),

        #[cfg(feature = "_hetzner-backend")]
        Backend::Hetzner => hetzner::run(wargs),

        #[cfg(not(feature = "_hetzner-backend"))]
        Backend::Hetzner => anyhow::bail!(
            "--backend hetzner selected but the hetzner backend is not compiled \
             in; rebuild with --features hetzner"
        ),
    }
}

/// Local (no-cloud) backend: drive the generic `run_worker` loop with the
/// local filesystem traits + the shared inline encode+score compute.
///
/// This is the no-spend dev / abstraction-validation path (spec §1.7
/// Phase B). The worker reads chunks from a local `chunks.jsonl` (or a
/// queue dir) instead of R2, mirrors blobs under a local base dir, and
/// runs the SAME backend-agnostic `run_worker` loop the cloud backends
/// use. The `local` glue build wires + typechecks the loop, queue, host,
/// creds, and storage GPU-free; the `local-sweep` build adds the shared
/// inline encode+score compute (which can be pointed at a real R2 bucket
/// to debug the GPU path on a local GPU box).
///
/// ## Arg mapping
///
/// The shared [`WorkerArgs`] are reused, reinterpreted for localhost:
/// - `--chunks-r2` (`CHUNKS_R2`): the local `chunks.jsonl` path or queue
///   dir. A `file://` URI, a plain path to a `*.jsonl` file (jsonl mode),
///   or a path to a directory (dir mode of `*.json` chunk files).
/// - `--workdir` (`WORKDIR`): the filesystem blob-storage base (the local
///   mirror dir) — `s3://bucket/key` artifacts land at
///   `<workdir>/bucket/key`.
#[cfg(feature = "_local-backend")]
mod local {
    use anyhow::{Context, Result};
    use std::path::PathBuf;
    use zenfleet_cloud::{Chunk, ChunkOutcome, CloudError, run_worker};
    use zenfleet_local::{
        LocalDirQueue, LocalFsStorage, LocalHeartbeat, LocalQueueConfig, LocalQueueSource,
        LocalWorkerHost,
    };
    use zenfleet_vastai::worker::WorkerArgs;

    /// Run the local sweep worker.
    ///
    /// Wires the five local traits and runs the backend-agnostic
    /// [`run_worker`] loop entirely on the filesystem — no cloud, no
    /// spend. The `compute` closure runs the SAME inline encode+score the
    /// vast.ai worker runs (in the `local-sweep` build).
    pub fn run(args: WorkerArgs) -> Result<()> {
        init_tracing();

        // Resolve the local chunk source from `--chunks-r2`: a directory
        // (dir mode) or a `*.jsonl` file (jsonl mode). A `file://` URI is
        // accepted and stripped to a plain path.
        let source = chunk_source(&args.chunks_r2).context("resolve local chunk source")?;
        let mut queue =
            LocalDirQueue::open(LocalQueueConfig::new(source)).context("open local job queue")?;

        // The filesystem storage base is the workdir — `s3://bucket/key`
        // artifacts mirror to `<workdir>/bucket/key`.
        let storage = LocalFsStorage::new(&args.workdir);
        let heartbeat = LocalHeartbeat;
        // Honour `--worker-id` (else env/hostname via the host).
        let host = match args.worker_id.as_deref().filter(|s| !s.is_empty()) {
            Some(id) => LocalWorkerHost::new(id, &args.workdir),
            None => LocalWorkerHost::from_env(),
        };

        tracing::info!(
            run_id = %args.run_id,
            chunks = %args.chunks_r2,
            workdir = %args.workdir.display(),
            "local sweep worker starting; reading chunks from the filesystem"
        );

        let summary = run_worker(&mut queue, &storage, &heartbeat, &host, |chunk, _s, _h| {
            compute_chunk(&args, chunk)
        })
        .map_err(|e| anyhow::anyhow!("local run_worker loop: {e}"))?;

        tracing::info!(
            dispatched = summary.dispatched,
            done = summary.done,
            skipped = summary.skipped,
            failed = summary.failed,
            "local sweep worker finished"
        );
        Ok(())
    }

    /// Map `--chunks-r2` to a [`LocalQueueSource`]: a `file://` URI is
    /// stripped to a path; a directory becomes dir mode; anything else
    /// (a `*.jsonl` file) becomes jsonl mode.
    fn chunk_source(chunks: &str) -> Result<LocalQueueSource> {
        let path = PathBuf::from(chunks.strip_prefix("file://").unwrap_or(chunks));
        if path.is_dir() {
            Ok(LocalQueueSource::Dir(path))
        } else {
            Ok(LocalQueueSource::Jsonl(path))
        }
    }

    /// Per-chunk compute. `chunk.payload` is the raw chunk record the
    /// local queue surfaced (the same `{chunk_id}` shape vast.ai reads
    /// from `chunks.jsonl`).
    fn compute_chunk(args: &WorkerArgs, chunk: &Chunk) -> Result<ChunkOutcome, CloudError> {
        #[cfg(feature = "_local-sweep")]
        {
            // Real encode+score path (the `local-sweep` build): reuse the
            // exact inline compute the vast.ai worker runs per chunk. It
            // shells s5cmd against the R2/S3 bucket referenced by the
            // chunk + args, so a developer debugging the GPU path points
            // it at a real bucket with local creds.
            run_inline_sweep(args, &chunk.payload)
        }
        #[cfg(not(feature = "_local-sweep"))]
        {
            let _ = (args, chunk);
            // GPU-free `local` glue build: the loop, queue, host, creds,
            // and storage are all live and exercised, but the encode+score
            // tree was not compiled in. Surface a terminal failure so the
            // operator rebuilds with `--features local-sweep` rather than
            // silently dropping the chunk. (The abstraction-validation
            // loop is covered end-to-end by the `zenfleet-local`
            // integration test with a stub compute closure.)
            Err(CloudError::Compute(
                "local glue build has no encode+score compute; \
                 rebuild zenfleet-sweep with --features local-sweep"
                    .into(),
            ))
        }
    }

    #[cfg(feature = "_local-sweep")]
    fn run_inline_sweep(args: &WorkerArgs, payload: &str) -> Result<ChunkOutcome, CloudError> {
        use zenfleet_vastai::worker::{process_chunk_inline, r2::new_from_args};

        let r2 = new_from_args(args)
            .map_err(|e| CloudError::Storage(format!("build R2 client: {e}")))?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| CloudError::Compute(format!("build compute runtime: {e}")))?;
        match rt.block_on(process_chunk_inline(args, &r2, payload)) {
            Ok(()) => Ok(ChunkOutcome::Done),
            Err(e) => Ok(ChunkOutcome::Failed {
                error: format!("{e:#}"),
            }),
        }
    }

    fn init_tracing() {
        use tracing_subscriber::EnvFilter;
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| EnvFilter::new("info,zenfleet_local=info")),
            )
            .try_init();
    }
}

/// Hetzner backend: poll R2 for chunks, run the inline pipeline, delete
/// the queue entry. No managed queue, no managed object store — workers
/// BYO R2 and the launcher's `push_jobs` writes one `<chunk_id>.json`
/// queue file per chunk under `runs/<sweep_id>/queue/`.
#[cfg(feature = "_hetzner-backend")]
mod hetzner {
    use anyhow::{Context, Result};
    use zenfleet_vastai::worker::{
        WorkerArgs, hydrate_pid1_env, provision_aws_credentials_file,
        r2::new_from_args as new_r2,
        r2_queue_loop::{R2QueueLoopConfig, run_r2_queue_loop},
    };

    pub fn run(args: WorkerArgs) -> Result<()> {
        init_tracing();
        // Hetzner cloud-init injects R2 creds + sweep wiring as docker
        // env vars on the container's pid 1 (the worker binary itself).
        // `std::env::var()` reads them directly in that case, but we
        // still call `hydrate_pid1_env` for symmetry with vast.ai and to
        // protect against shapes where the worker isn't pid 1 (e.g. when
        // wrapped by `entrypoint_hetzner.sh`).
        hydrate_pid1_env();
        // ⚡ THE iter-5 bug fix: write `~/.aws/credentials` from the env
        // vars BEFORE building the R2 client. The cloud-init `docker run`
        // bypasses `entrypoint_hetzner.sh`, so without this call no
        // credentials file exists, every `s5cmd --profile r2 ...` 403s,
        // and the worker silently spins on an empty LIST. See
        // `crates/zenfleet-vastai/src/worker/mod.rs` for the helper.
        provision_aws_credentials_file(&args.s5cmd_profile)
            .context("write ~/.aws/credentials from env for s5cmd (hetzner backend)")?;
        let r2 = new_r2(&args).context("build R2 client for hetzner backend")?;
        let cfg = R2QueueLoopConfig::from_env()
            .context("R2QueueLoopConfig::from_env (BUCKET + CHUNKS_QUEUE_PREFIX)")?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("build tokio runtime")?;
        let processed = rt.block_on(run_r2_queue_loop(&args, &r2, &cfg))?;
        tracing::info!(processed, "hetzner sweep worker finished");
        Ok(())
    }

    fn init_tracing() {
        use tracing_subscriber::EnvFilter;
        let _ = tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::new("info,zenfleet_vastai=info,zenfleet_hetzner=info")
            }))
            .try_init();
    }
}
