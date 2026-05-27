//! `zen-sweep-worker` — the cloud-agnostic deployed compute binary.
//!
//! This is the single compute binary baked into the deploy docker
//! image (spec §1.2/§1.3). It runs the "claim job → fetch inputs →
//! compute → upload artifacts → heartbeat" loop, parameterized over the
//! [`zen_cloud_core`] trait layer, and selects which cloud provider to
//! use at runtime via `--backend <name>`.
//!
//! ## Phase A
//!
//! Only the `vastai` backend exists. `zen-sweep-worker worker
//! --backend vastai <args>` dispatches to the proven async worker in
//! `zen-cloud-vastai` — the SAME code path the `vastai-fleet worker`
//! binary runs — so the produced sweep artifacts are byte-identical to
//! today's worker. The carve introduces the indirection without
//! changing the compute behaviour (spec §1.7 Phase A).
//!
//! ## Later phases
//!
//! Phase B adds `zen-cloud-local` (filesystem + sqlite), Phase C adds
//! `zen-cloud-gcp` / `zen-cloud-do`. Each becomes a feature-gated dep
//! and an additional `Backend` arm; `--backend` picks among the
//! compiled-in set. Selection is cargo features + trait objects, NOT
//! dlopen (spec §1.6 decision 4).

#![forbid(unsafe_code)]

use clap::{Parser, Subcommand, ValueEnum};

/// Cloud backend selector. Only `vastai` is compiled in for Phase A;
/// the enum is the runtime side of the cargo-feature-gated backend set.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    /// vast.ai + Cloudflare R2 + `/proc/1/environ` credentials.
    Vastai,
}

/// Cloud-agnostic sweep worker.
#[derive(Parser, Debug)]
#[command(
    name = "zen-sweep-worker",
    version,
    about = "Cloud-agnostic sweep worker — claim → fetch → compute → upload → beat over a pluggable cloud backend"
)]
struct Cli {
    /// Which compiled-in cloud backend to run against. Phase A ships
    /// only `vastai`; the flag exists so launchers + the docker
    /// entrypoint are forward-compatible with Phase B/C backends.
    #[arg(long, value_enum, default_value_t = Backend::Vastai, global = true)]
    backend: Backend,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run the sweep worker loop. The compute closure is the selected
    /// backend's encode+score sweep — byte-identical to the legacy
    /// `vastai-fleet worker` for `--backend vastai`.
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
            "no cloud backend compiled in; build with --features vastai (Phase A) \
             or a later-phase backend feature"
        ),
    }
}

/// Dispatch the worker to the selected backend. For `vastai` this calls
/// the proven `zen_cloud_vastai::worker::cmd_worker` directly — the
/// exact code path the legacy `vastai-fleet worker` binary runs, so the
/// sweep output is byte-identical.
#[cfg(feature = "_vastai-backend")]
fn run_worker_backend(
    backend: Backend,
    wargs: zen_cloud_vastai::worker::WorkerArgs,
) -> anyhow::Result<()> {
    match backend {
        Backend::Vastai => zen_cloud_vastai::worker::cmd_worker(wargs),
    }
}
