//! `zen-sweep-worker` — the cloud-agnostic deployed compute binary.
//!
//! This is the single compute binary baked into the deploy docker
//! image (spec §1.2/§1.3). It runs the "claim job → fetch inputs →
//! compute → upload artifacts → heartbeat" loop, parameterized over the
//! [`zen_cloud_core`] trait layer, and selects which cloud provider to
//! use at runtime via `--backend <name>`.
//!
//! ## Backends
//!
//! - `vastai` (Phase A): `zen-sweep-worker worker --backend vastai`
//!   dispatches to the proven async worker in `zen-cloud-vastai` — the
//!   SAME code path the `vastai-fleet worker` binary runs — so the
//!   produced sweep artifacts are byte-identical to today's worker.
//! - `salad` (Phase C, spec §1.9): `--backend salad` drives the generic
//!   [`zen_cloud_core::run_worker`] loop with the SaladCloud traits
//!   (HTTP job receiver fed by the baked-in sidecar, container-group
//!   env + IMDS credentials/host, shared S3 BlobStorage, no-op
//!   heartbeat). The encode+score `compute` closure is the SAME one
//!   vast.ai runs (`zen_cloud_vastai::worker::process_chunk_inline`),
//!   available in the `salad-sweep` build; the GPU-free `salad` build
//!   wires + typechecks the glue without the codec tree.
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
    /// SaladCloud + managed queue (sidecar→HTTP) + IMDS + BYO R2/S3.
    Salad,
}

/// Cloud-agnostic sweep worker.
#[derive(Parser, Debug)]
#[command(
    name = "zen-sweep-worker",
    version,
    about = "Cloud-agnostic sweep worker — claim → fetch → compute → upload → beat over a pluggable cloud backend"
)]
struct Cli {
    /// Which compiled-in cloud backend to run against. Defaults to
    /// `vastai`; `salad` selects the SaladCloud backend (Phase C).
    #[arg(long, value_enum, default_value_t = Backend::Vastai, global = true)]
    backend: Backend,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the sweep worker loop. The compute closure is the selected
    /// backend's encode+score sweep — byte-identical to the legacy
    /// `vastai-fleet worker` for `--backend vastai`, and the same
    /// inline compute for `--backend salad`.
    //
    // The args are vast.ai's `WorkerArgs`, which are backend-agnostic
    // enough (run id, chunks manifest, workdir, s5cmd/R2 config, mode)
    // to drive both backends. Available under any backend that pulls
    // `zen-cloud-vastai/worker` (both `vastai*` and `salad*` do).
    #[cfg(feature = "_vastai-backend")]
    Worker(zen_cloud_vastai::worker::WorkerArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        #[cfg(feature = "_vastai-backend")]
        Cmd::Worker(wargs) => run_worker_backend(cli.backend, wargs),
        #[cfg(not(feature = "_vastai-backend"))]
        _ => anyhow::bail!(
            "no cloud backend compiled in; build with --features vastai \
             (Phase A) or --features salad (Phase C)"
        ),
    }
}

/// Dispatch the worker to the selected backend.
#[cfg(feature = "_vastai-backend")]
fn run_worker_backend(
    backend: Backend,
    wargs: zen_cloud_vastai::worker::WorkerArgs,
) -> anyhow::Result<()> {
    match backend {
        // vast.ai: call the proven `cmd_worker` directly — the exact
        // code path the legacy `vastai-fleet worker` binary runs, so the
        // sweep output is byte-identical.
        Backend::Vastai => zen_cloud_vastai::worker::cmd_worker(wargs),

        #[cfg(feature = "_salad-backend")]
        Backend::Salad => salad::run(wargs),

        #[cfg(not(feature = "_salad-backend"))]
        Backend::Salad => anyhow::bail!(
            "--backend salad selected but the salad backend is not compiled \
             in; rebuild with --features salad (glue) or --features salad-sweep \
             (full encode+score)"
        ),
    }
}

/// SaladCloud backend: drive the generic `run_worker` loop with the
/// Salad traits + the shared inline encode+score compute.
#[cfg(feature = "_salad-backend")]
mod salad {
    use anyhow::{Context, Result};
    use zen_cloud_core::{Chunk, ChunkOutcome, CloudError, CredentialSource, run_worker};
    use zen_cloud_salad::{
        SaladEnvCredentials, SaladHeartbeat, SaladJobQueue, SaladQueueConfig, SaladWorkerHost,
        blob_storage_from_credentials,
    };
    use zen_cloud_vastai::worker::WorkerArgs;

    /// Run the SaladCloud sweep worker.
    ///
    /// Wires the five Salad traits and runs the backend-agnostic
    /// [`run_worker`] loop. The `compute` closure parses each chunk's
    /// payload (the raw queue job body the sidecar POSTed) and runs the
    /// SAME inline encode+score the vast.ai worker runs.
    pub fn run(args: WorkerArgs) -> Result<()> {
        init_tracing();

        // Resolve BYO object-store credentials from the container-group
        // env, then build the shared S3 BlobStorage from them.
        let creds = SaladEnvCredentials
            .resolve()
            .context("resolve salad container-group credentials")?;
        let storage = blob_storage_from_credentials(&creds)
            .context("build salad blob storage from credentials")?;

        let host = SaladWorkerHost::from_env();
        let heartbeat = SaladHeartbeat;

        // Bind the local HTTP job receiver. The port must match the
        // container group's `queue_connection.port` set by the launcher;
        // the worker args carry it via the (optional) bind override, else
        // the default :80 (matching the upstream sample).
        let mut queue =
            SaladJobQueue::bind(salad_queue_config(&args)).context("bind salad job queue")?;

        tracing::info!(
            run_id = %args.run_id,
            "salad sweep worker starting; awaiting jobs from the sidecar"
        );

        let summary = run_worker(&mut queue, &storage, &heartbeat, &host, |chunk, _s, _h| {
            compute_chunk(&args, chunk)
        })
        .map_err(|e| anyhow::anyhow!("salad run_worker loop: {e}"))?;

        tracing::info!(
            dispatched = summary.dispatched,
            done = summary.done,
            skipped = summary.skipped,
            failed = summary.failed,
            "salad sweep worker finished"
        );
        Ok(())
    }

    /// Derive the local job-receiver bind config. Salad's sidecar POSTs
    /// to `localhost:<queue_connection.port>`; we read that port from
    /// `$SALAD_JOB_PORT` (set by the launcher to match the container
    /// group's `queue_connection.port`), defaulting to the sample's :80.
    fn salad_queue_config(_args: &WorkerArgs) -> SaladQueueConfig {
        match std::env::var("SALAD_JOB_PORT")
            .ok()
            .and_then(|p| p.parse::<u16>().ok())
        {
            Some(port) => SaladQueueConfig {
                bind_addr: std::net::SocketAddr::from(([0, 0, 0, 0], port)),
            },
            None => SaladQueueConfig::default(),
        }
    }

    /// Per-chunk compute. `chunk.payload` is the raw queue-job body the
    /// sidecar forwarded (the chunk descriptor JSON, same shape vast.ai
    /// reads from `chunks.jsonl`).
    fn compute_chunk(args: &WorkerArgs, chunk: &Chunk) -> Result<ChunkOutcome, CloudError> {
        #[cfg(feature = "_salad-sweep")]
        {
            // Real encode+score path (the `salad-sweep` build): reuse the
            // exact inline compute the vast.ai worker runs per chunk.
            run_inline_sweep(args, &chunk.payload)
        }
        #[cfg(not(feature = "_salad-sweep"))]
        {
            let _ = (args, chunk);
            // GPU-free `salad` glue build: the loop, queue, host, creds,
            // and storage are all live and exercised, but the encode+score
            // tree was not compiled in. Surface a terminal failure so the
            // operator rebuilds with `--features salad-sweep` rather than
            // silently dropping the chunk.
            Err(CloudError::Compute(
                "salad glue build has no encode+score compute; \
                 rebuild zen-sweep-worker with --features salad-sweep"
                    .into(),
            ))
        }
    }

    #[cfg(feature = "_salad-sweep")]
    fn run_inline_sweep(args: &WorkerArgs, payload: &str) -> Result<ChunkOutcome, CloudError> {
        use zen_cloud_vastai::worker::{process_chunk_inline, r2::new_from_args};

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
                    .unwrap_or_else(|_| EnvFilter::new("info,zen_cloud_salad=info")),
            )
            .try_init();
    }
}
