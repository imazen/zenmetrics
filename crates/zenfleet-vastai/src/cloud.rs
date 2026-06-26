//! vast.ai implementations of the [`zenfleet_cloud`] trait surface.
//!
//! These wrap the crate's existing, proven building blocks — the
//! s5cmd-backed [`crate::worker::r2::R2Client`], the R2 token-race
//! [`crate::worker::claim`], the `nvidia-smi` introspection in
//! [`crate::worker::adapt`], and the `/proc/1/environ` credential
//! convention — behind the cloud-agnostic traits so the generic
//! `zenfleet_cloud::run_worker` loop (and the Phase-B `local` backend)
//! can drive vast.ai with no provider knowledge.
//!
//! Phase A does NOT route the production worker through these — the
//! `zenfleet-vastai`/`zenfleet-sweep` deployed worker stays on the
//! battle-tested async [`crate::worker::cmd_worker`] path for
//! byte-identical output. This module is the trait bridge that proves
//! the carve fits vast.ai and that Phase B/C/D extend.
//!
//! The core trait surface is synchronous (spec §1.5); the underlying
//! R2 client is async (it parks on `tokio::process::Command`). Each
//! sync trait method drives the async op to completion on a private
//! current-thread runtime — the same pattern the existing
//! `cmd_worker` uses to bridge sync `main` to the async dispatcher.

use std::path::PathBuf;

use zenfleet_cloud::{
    ArtifactKey, BlobStorage, Chunk, ChunkId, ChunkOutcome, CloudError, CredentialSource,
    Credentials, Heartbeat, JobQueue, WorkerHost, WorkerId, WorkerStatus,
};

use crate::worker::adapt;
use crate::worker::claim::{ClaimConfig, ClaimOutcome, try_claim};
use crate::worker::r2::R2Client;

/// Build a fresh single-thread tokio runtime for one blocking bridge.
///
/// Cheap (no worker threads) and disposable — used to drive one async
/// R2 op to completion from a sync trait method.
fn block_on<F: std::future::Future>(fut: F) -> Result<F::Output, CloudError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CloudError::Other(format!("build tokio runtime: {e}")))?;
    Ok(rt.block_on(fut))
}

// ───────────────────────── BlobStorage ─────────────────────────

/// `BlobStorage` over Cloudflare R2.
///
/// The impl itself lives in `worker::s3_blob` (folded in from the former
/// `zenfleet-s3` crate, 2026-06-26; R2 is S3-compatible, so one impl
/// serves vast.ai + any BYO bucket). `R2BlobStorage` is a vast.ai-named
/// alias for that shared [`crate::worker::s3_blob::S3BlobStorage`], kept
/// so every call site that referenced `R2BlobStorage` compiles unchanged.
pub use crate::worker::s3_blob::S3BlobStorage as R2BlobStorage;

/// Convenience constructor: build an [`R2BlobStorage`] from the same
/// args the worker uses. The shared `S3BlobStorage` takes an
/// already-built client, so this bridges from vast.ai `WorkerArgs`.
pub fn r2_blob_storage_from_worker_args(
    args: &crate::worker::WorkerArgs,
) -> Result<R2BlobStorage, CloudError> {
    let client = crate::worker::r2::new_from_args(args).map_err(CloudError::storage)?;
    Ok(R2BlobStorage::new(client))
}

// ───────────────────────── CredentialSource ─────────────────────────

/// `CredentialSource` reading vast.ai's pid-1 environment.
///
/// vast.ai injects R2 creds + the sweep run id into the *container's*
/// pid-1 environment, not into every spawned process. This reads
/// `/proc/1/environ` (NUL-separated `KEY=VALUE`) — the same source the
/// worker's `hydrate_pid1_env` copies from.
#[derive(Default)]
pub struct ProcEnvironCredentials;

impl CredentialSource for ProcEnvironCredentials {
    fn resolve(&self) -> Result<Credentials, CloudError> {
        let mut out = Credentials::new();
        // Best-effort: prefer the live process env, fall back to
        // /proc/1/environ for the vars vast.ai only puts on pid 1.
        let keys = [
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ENDPOINT",
            "SWEEP_RUN_ID",
            "WORKER_ID",
            "CHUNKS_R2",
            "CONTAINER_ID",
            "CONTAINER_API_KEY",
        ];
        for k in keys {
            if let Ok(v) = std::env::var(k) {
                out.insert(k.to_string(), v);
            }
        }
        if let Ok(buf) = std::fs::read("/proc/1/environ") {
            for entry in buf.split(|b| *b == 0) {
                let s = String::from_utf8_lossy(entry);
                if let Some((k, v)) = s.split_once('=') {
                    out.entry(k.to_string()).or_insert_with(|| v.to_string());
                }
            }
        }
        Ok(out)
    }
}

// ───────────────────────── WorkerHost ─────────────────────────

/// `WorkerHost` over vast.ai box introspection: hostname / `$WORKER_ID`
/// for the id, the worker workdir for scratch, and `nvidia-smi` for the
/// GPU count.
pub struct VastaiWorkerHost {
    worker_id: WorkerId,
    scratch: PathBuf,
}

impl VastaiWorkerHost {
    pub fn new(worker_id: impl Into<String>, scratch: impl Into<PathBuf>) -> Self {
        Self {
            worker_id: WorkerId(worker_id.into()),
            scratch: scratch.into(),
        }
    }
}

impl WorkerHost for VastaiWorkerHost {
    fn worker_id(&self) -> WorkerId {
        self.worker_id.clone()
    }

    fn scratch_dir(&self) -> PathBuf {
        self.scratch.clone()
    }

    fn gpu_count(&self) -> usize {
        // Reuse the worker's nvidia-smi probe. `--query-gpu=memory.total`
        // returns one line per GPU; count the lines. Absent nvidia-smi
        // (dev laptop) -> 0.
        match std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
            .output()
        {
            Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
                .lines()
                .filter(|l| !l.trim().is_empty())
                .count(),
            _ => 0,
        }
    }
}

/// Re-export the worker's GPU-memory probe so consumers can size
/// concurrency the same way the adaptive controller does.
pub use adapt::nvidia_smi_total_memory_mb;

// ───────────────────────── Heartbeat ─────────────────────────

/// `Heartbeat` that writes a small liveness object to R2 under a
/// per-run heartbeat prefix.
pub struct R2Heartbeat {
    storage: R2BlobStorage,
    prefix: String,
}

impl R2Heartbeat {
    /// `prefix` is an `s3://bucket/<run>/heartbeats/`-style location; the
    /// per-worker object is `<prefix><worker>.beat`.
    pub fn new(storage: R2BlobStorage, prefix: impl Into<String>) -> Self {
        let mut prefix = prefix.into();
        if !prefix.ends_with('/') {
            prefix.push('/');
        }
        Self { storage, prefix }
    }
}

impl Heartbeat for R2Heartbeat {
    fn beat(&self, worker: &WorkerId, status: WorkerStatus) -> Result<(), CloudError> {
        let key = ArtifactKey(format!("{}{}.beat", self.prefix, worker));
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let body = format!("{worker}\t{now}\t{status:?}\n");
        self.storage.put(&key, body.as_bytes())
    }
}

// ───────────────────────── JobQueue ─────────────────────────

/// Pull-based `JobQueue` over an R2 `chunks.jsonl` manifest with the
/// existing atomic R2 token-race claim.
///
/// `next_chunk` pops the next manifest line and performs the same
/// `try_claim` race the async worker uses; a chunk that is already done
/// / held by a peer / lost is returned to the caller as the chunk with
/// the loop deciding to skip via the closure, OR — to match the
/// existing worker's "skip silently and move on" semantics — we keep
/// claiming until we acquire one (or drain). This keeps the trait-level
/// pull queue consistent with the production dispatcher's behaviour.
pub struct R2ChunkQueue {
    storage_client: R2Client,
    worker_id: String,
    run_id: String,
    lines: std::vec::IntoIter<String>,
    skip_claims: bool,
}

impl R2ChunkQueue {
    /// Fetch + parse the manifest, then hand chunks out one at a time.
    /// `chunks_r2` is the `s3://…/chunks.jsonl` URI.
    pub fn fetch(
        client: R2Client,
        worker_id: impl Into<String>,
        run_id: impl Into<String>,
        chunks_r2: &str,
        skip_claims: bool,
    ) -> Result<Self, CloudError> {
        let lines = block_on(client.fetch_chunks_jsonl(chunks_r2))?.map_err(CloudError::queue)?;
        Ok(Self {
            storage_client: client,
            worker_id: worker_id.into(),
            run_id: run_id.into(),
            lines: lines.into_iter(),
            skip_claims,
        })
    }

    /// Parse the `chunk_id` + optional sidecar URL out of one manifest
    /// line, deriving the default sidecar/claim URIs the worker uses.
    fn chunk_uris(&self, line: &str) -> Result<(String, String, String), CloudError> {
        #[derive(serde::Deserialize)]
        struct Rec {
            chunk_id: String,
            #[serde(default)]
            out_sidecar_omni: Option<String>,
        }
        let rec: Rec = serde_json::from_str(line)
            .map_err(|e| CloudError::Queue(format!("parse chunk: {e}")))?;
        let sidecar = rec.out_sidecar_omni.unwrap_or_else(|| {
            format!(
                "s3://zentrain/{}/omni/{}.parquet",
                self.run_id, rec.chunk_id
            )
        });
        let claim = format!(
            "s3://coefficient/claims/{}/{}.claim",
            self.run_id, rec.chunk_id
        );
        Ok((rec.chunk_id, sidecar, claim))
    }
}

impl JobQueue for R2ChunkQueue {
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError> {
        // Pull semantics: keep popping + claiming until we win a chunk
        // or the manifest drains. Already-done / peer-held / lost chunks
        // are skipped here (the async production worker does the same).
        loop {
            let Some(line) = self.lines.next() else {
                return Ok(None);
            };
            let (chunk_id, sidecar_uri, claim_uri) = self.chunk_uris(&line)?;

            if self.skip_claims {
                return Ok(Some(Chunk {
                    id: ChunkId(chunk_id),
                    payload: line,
                }));
            }

            let cfg = ClaimConfig::default();
            let outcome = block_on(try_claim(
                &self.storage_client,
                &self.worker_id,
                &chunk_id,
                &sidecar_uri,
                &claim_uri,
                &cfg,
            ))?
            .map_err(CloudError::queue)?;

            match outcome {
                ClaimOutcome::Acquired { .. } => {
                    return Ok(Some(Chunk {
                        id: ChunkId(chunk_id),
                        payload: line,
                    }));
                }
                // Not ours — try the next manifest line.
                ClaimOutcome::AlreadyDone
                | ClaimOutcome::HeldByPeer
                | ClaimOutcome::LostRace
                | ClaimOutcome::Errored => continue,
            }
        }
    }

    fn ack_chunk(&mut self, _id: &ChunkId, _outcome: ChunkOutcome) -> Result<(), CloudError> {
        // vast.ai's claim is the only durable per-chunk state and the
        // sidecar's existence is the completion record — there is no
        // separate ack channel to write. Done/Skipped need no action;
        // a Failed chunk intentionally leaves its claim so a peer can
        // steal it once stale (the existing worker's behaviour). So ack
        // is a no-op for this backend.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(unsafe_code)] // single-threaded test env mutation
    fn proc_environ_credentials_resolves_present_env() {
        // SAFETY: single-threaded test; no other thread reads env here.
        unsafe {
            std::env::set_var("SWEEP_RUN_ID", "test-run-xyz");
        }
        let creds = ProcEnvironCredentials.resolve().unwrap();
        assert_eq!(
            creds.get("SWEEP_RUN_ID").map(String::as_str),
            Some("test-run-xyz")
        );
        unsafe {
            std::env::remove_var("SWEEP_RUN_ID");
        }
    }

    #[test]
    fn worker_host_reports_id_and_scratch() {
        let host = VastaiWorkerHost::new("box-7", "/workspace/scratch");
        assert_eq!(host.worker_id().as_str(), "box-7");
        assert_eq!(host.scratch_dir(), PathBuf::from("/workspace/scratch"));
        // gpu_count is environment-dependent (0 on a dev box without
        // nvidia-smi); we only assert it does not panic.
        let _ = host.gpu_count();
    }

    #[test]
    fn r2_blob_storage_alias_is_the_shared_s3_impl() {
        // The R2 `BlobStorage` impl + temp-file basename sanitisation
        // were factored into the shared `zenfleet-s3` crate (spec §1.9
        // item 4); `R2BlobStorage` here is the re-exported alias of
        // `zenfleet_s3::S3BlobStorage`. This asserts the alias really
        // is the shared impl, exposes its client, and still satisfies
        // the core `BlobStorage` trait. (The basename-sanitisation
        // behaviour itself is unit-tested in `zenfleet-s3`'s `blob`
        // module, which is where that code now lives.)
        let client = R2Client::new("s5cmd", "https://example.r2.cloudflarestorage.com", "r2");
        let storage = R2BlobStorage::new(client);
        assert_eq!(
            storage.client().endpoint,
            "https://example.r2.cloudflarestorage.com"
        );
        // It satisfies the core `BlobStorage` trait (compile-time check).
        fn _assert_blob_storage<T: BlobStorage>(_: &T) {}
        _assert_blob_storage(&storage);
    }
}
