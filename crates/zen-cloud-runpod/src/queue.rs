//! RunPod `JobQueue` — the Pods (pull) path over an R2 `chunks.jsonl`
//! manifest, using the shared R2-ETag atomic chunk-claim.
//!
//! ## Why pull (and why this reuses vast.ai's claim verbatim)
//!
//! RunPod offers two product modes (spec §1.10): **Pods** (you rent a
//! GPU pod that boots a container, exactly like vast.ai) and
//! **Serverless** (Salad-style managed push, documented as the follow-on
//! in `RUNPOD.md`, not implemented here). The Pods path is structurally
//! identical to vast.ai: a generic container with credentials in its
//! environment, bring-your-own object store, and the worker *pulls*
//! work. So the RunPod `JobQueue` is the SAME R2 token-race claim the
//! vast.ai backend uses — and rather than copy-paste a third claim impl,
//! this module reuses [`zen_cloud_vastai::worker::claim::try_claim`] and
//! [`zen_cloud_vastai::worker::r2::R2Client`] verbatim. The claim
//! *algorithm* lives in exactly one place (vast.ai's `claim.rs`); this
//! crate only owns the RunPod-specific URI conventions + the
//! [`zen_cloud_core::JobQueue`] mapping.
//!
//! ## Mapping onto the [`zen_cloud_core::JobQueue`] trait
//!
//! - [`RunpodChunkQueue::next_chunk`] pops the next manifest line and
//!   races for its claim; already-done / peer-held / lost / errored
//!   chunks are skipped (keep popping) so the loop only ever receives a
//!   chunk this worker owns — matching the production vast.ai
//!   dispatcher's "skip silently and move on" behaviour. When the
//!   manifest drains, returns `None`.
//! - [`RunpodChunkQueue::ack_chunk`] is a no-op: the R2 claim + the
//!   sidecar's existence are the only durable per-chunk state (there is
//!   no separate ack channel), and a `Failed` chunk intentionally leaves
//!   its claim so a peer can steal it once stale. This is identical to
//!   the vast.ai backend's ack semantics.

use std::future::Future;

use zen_cloud_core::{Chunk, ChunkId, ChunkOutcome, CloudError, JobQueue};
use zen_cloud_vastai::worker::claim::{ClaimConfig, ClaimOutcome, try_claim};
use zen_cloud_vastai::worker::r2::R2Client;

/// URI conventions for the pull queue: how to derive a chunk's sidecar
/// (completion record) and claim-file URIs from its `chunk_id`.
///
/// Defaults match the live zenmetrics sweep layout (the same paths the
/// vast.ai backend's `R2ChunkQueue` uses), so a RunPod pod joins an
/// existing sweep fleet with zero reconfiguration. Templates use
/// `{run}` and `{chunk}` placeholders.
#[derive(Clone, Debug)]
pub struct RunpodQueueConfig {
    /// Sweep run id — scopes the claim namespace + sidecar output path.
    pub run_id: String,
    /// Sidecar (completion record) URI template. A chunk is considered
    /// already-done when this object exists. `{run}` / `{chunk}` are
    /// substituted. Used only when a manifest line omits an explicit
    /// `out_sidecar_omni`.
    pub sidecar_template: String,
    /// Claim-file URI template. The token-race writes/reads this object.
    /// `{run}` / `{chunk}` are substituted.
    pub claim_template: String,
    /// Skip the R2 token-race claim entirely. Single-instance smoke runs
    /// set this to bypass claim contention; production fleets leave it
    /// false. Mirrors the vast.ai `--skip-claims` flag.
    pub skip_claims: bool,
    /// Claim race tunables (stale window, read-back settle delay).
    pub claim: ClaimConfig,
}

impl RunpodQueueConfig {
    /// Build the default config for a run id: the live-sweep sidecar +
    /// claim conventions, claims enabled.
    pub fn for_run(run_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            sidecar_template: "s3://zentrain/{run}/omni/{chunk}.parquet".to_string(),
            claim_template: "s3://coefficient/claims/{run}/{chunk}.claim".to_string(),
            skip_claims: false,
            claim: ClaimConfig::default(),
        }
    }

    fn sidecar_uri(&self, chunk_id: &str) -> String {
        self.sidecar_template
            .replace("{run}", &self.run_id)
            .replace("{chunk}", chunk_id)
    }

    fn claim_uri(&self, chunk_id: &str) -> String {
        self.claim_template
            .replace("{run}", &self.run_id)
            .replace("{chunk}", chunk_id)
    }
}

/// Parse one `chunks.jsonl` manifest line into `(chunk_id, sidecar_uri)`.
///
/// Honours an explicit `out_sidecar_omni` on the record (the manifest
/// can override the default sidecar location); otherwise derives it from
/// the config template. Shared by the real queue and tests.
fn parse_manifest_line(
    line: &str,
    cfg: &RunpodQueueConfig,
) -> Result<(String, String), CloudError> {
    #[derive(serde::Deserialize)]
    struct Rec {
        chunk_id: String,
        #[serde(default)]
        out_sidecar_omni: Option<String>,
    }
    let rec: Rec =
        serde_json::from_str(line).map_err(|e| CloudError::Queue(format!("parse chunk: {e}")))?;
    let sidecar = rec
        .out_sidecar_omni
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| cfg.sidecar_uri(&rec.chunk_id));
    Ok((rec.chunk_id, sidecar))
}

/// Drive an async future to completion on a disposable current-thread
/// tokio runtime, mapping a build failure into a [`CloudError`].
///
/// The same blocking bridge the vast.ai + S3 backends use between the
/// synchronous core trait surface (spec §1.5) and the async R2 client.
fn block_on<F: Future>(fut: F) -> Result<F::Output, CloudError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| CloudError::Other(format!("build tokio runtime: {e}")))?;
    Ok(rt.block_on(fut))
}

/// Pull-based `JobQueue` over an R2 `chunks.jsonl` manifest with the
/// shared atomic R2 token-race claim — the RunPod Pods path.
pub struct RunpodChunkQueue {
    client: R2Client,
    worker_id: String,
    cfg: RunpodQueueConfig,
    lines: std::vec::IntoIter<String>,
}

impl RunpodChunkQueue {
    /// Build from already-fetched manifest lines. Used by tests and any
    /// caller that fetched the manifest itself.
    pub fn from_lines(
        client: R2Client,
        worker_id: impl Into<String>,
        cfg: RunpodQueueConfig,
        lines: Vec<String>,
    ) -> Self {
        Self {
            client,
            worker_id: worker_id.into(),
            cfg,
            lines: lines.into_iter(),
        }
    }

    /// Fetch + parse the `chunks.jsonl` manifest from R2, then hand
    /// chunks out one at a time. `chunks_r2` is the `s3://…/chunks.jsonl`
    /// URI (the same manifest the vast.ai fleet reads).
    pub fn fetch(
        client: R2Client,
        worker_id: impl Into<String>,
        cfg: RunpodQueueConfig,
        chunks_r2: &str,
    ) -> Result<Self, CloudError> {
        let lines = block_on(client.fetch_chunks_jsonl(chunks_r2))?.map_err(CloudError::queue)?;
        Ok(Self::from_lines(client, worker_id, cfg, lines))
    }

    /// The claim decision for one chunk, factored out so it can be unit-
    /// tested with a fake claim function (the real path uses `try_claim`
    /// against R2). Returns `Ok(Some(chunk))` if this worker owns the
    /// chunk, `Ok(None)` to skip it (done / peer-held / lost / errored).
    fn decide<Claim, Fut>(&self, line: String, claim_fn: Claim) -> Result<Option<Chunk>, CloudError>
    where
        Claim: FnOnce(String, String) -> Fut,
        Fut: Future<Output = Result<ClaimOutcome, CloudError>>,
    {
        let (chunk_id, sidecar_uri) = parse_manifest_line(&line, &self.cfg)?;

        if self.cfg.skip_claims {
            return Ok(Some(Chunk {
                id: ChunkId(chunk_id),
                payload: line,
            }));
        }

        let claim_uri = self.cfg.claim_uri(&chunk_id);
        let outcome = block_on(claim_fn(sidecar_uri, claim_uri))??;
        match outcome {
            ClaimOutcome::Acquired { .. } => Ok(Some(Chunk {
                id: ChunkId(chunk_id),
                payload: line,
            })),
            // Not ours — skip and let the caller pop the next line.
            ClaimOutcome::AlreadyDone
            | ClaimOutcome::HeldByPeer
            | ClaimOutcome::LostRace
            | ClaimOutcome::Errored => Ok(None),
        }
    }
}

impl JobQueue for RunpodChunkQueue {
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError> {
        // Pull semantics: keep popping + claiming until we win a chunk or
        // the manifest drains. Already-done / peer-held / lost chunks are
        // skipped here (the production vast.ai worker does the same).
        loop {
            let Some(line) = self.lines.next() else {
                return Ok(None);
            };
            let client = &self.client;
            let worker_id = &self.worker_id;
            let claim_cfg = &self.cfg.claim;
            // The real claim: the shared R2 token-race. `decide` owns the
            // skip-claims short-circuit and the outcome→chunk mapping.
            let chunk_id_for_claim = {
                // parse once for the closure's chunk_id arg; decide
                // re-parses for the returned Chunk, but parsing a <1 KB
                // JSON line twice is negligible vs the R2 round-trip.
                let (cid, _) = parse_manifest_line(&line, &self.cfg)?;
                cid
            };
            let decided = self.decide(line.clone(), |sidecar_uri, claim_uri| async move {
                try_claim(
                    client,
                    worker_id,
                    &chunk_id_for_claim,
                    &sidecar_uri,
                    &claim_uri,
                    claim_cfg,
                )
                .await
                .map_err(CloudError::queue)
            })?;
            if let Some(chunk) = decided {
                return Ok(Some(chunk));
            }
            // else: skipped — pop the next line.
        }
    }

    fn ack_chunk(&mut self, _id: &ChunkId, _outcome: ChunkOutcome) -> Result<(), CloudError> {
        // The R2 claim + the sidecar's existence are the only durable
        // per-chunk state — there is no separate ack channel. Done /
        // Skipped need no action; a Failed chunk intentionally leaves its
        // claim so a peer can steal it once stale (the vast.ai backend's
        // behaviour). So ack is a no-op for the pull path.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> RunpodQueueConfig {
        RunpodQueueConfig::for_run("test-run")
    }

    fn client() -> R2Client {
        R2Client::new("s5cmd", "https://acct.r2.cloudflarestorage.com", "r2")
    }

    #[test]
    fn config_substitutes_run_and_chunk() {
        let c = cfg();
        assert_eq!(
            c.sidecar_uri("chunk-7"),
            "s3://zentrain/test-run/omni/chunk-7.parquet"
        );
        assert_eq!(
            c.claim_uri("chunk-7"),
            "s3://coefficient/claims/test-run/chunk-7.claim"
        );
    }

    #[test]
    fn parse_manifest_line_derives_default_sidecar() {
        let c = cfg();
        let (id, sidecar) = parse_manifest_line(r#"{"chunk_id":"abc"}"#, &c).unwrap();
        assert_eq!(id, "abc");
        assert_eq!(sidecar, "s3://zentrain/test-run/omni/abc.parquet");
    }

    #[test]
    fn parse_manifest_line_honours_explicit_sidecar() {
        let c = cfg();
        let (id, sidecar) = parse_manifest_line(
            r#"{"chunk_id":"abc","out_sidecar_omni":"s3://other/x.parquet"}"#,
            &c,
        )
        .unwrap();
        assert_eq!(id, "abc");
        assert_eq!(sidecar, "s3://other/x.parquet");
    }

    #[test]
    fn parse_manifest_line_rejects_garbage() {
        let c = cfg();
        assert!(parse_manifest_line("not json", &c).is_err());
        assert!(parse_manifest_line(r#"{"no_chunk_id":1}"#, &c).is_err());
    }

    #[test]
    fn skip_claims_returns_chunk_without_racing() {
        let mut c = cfg();
        c.skip_claims = true;
        let q = RunpodChunkQueue::from_lines(client(), "w1", c, vec![]);
        // decide must NOT call the claim fn when skip_claims is set.
        let chunk = q
            .decide(r#"{"chunk_id":"sc-1"}"#.to_string(), |_s, _c| async {
                panic!("claim fn must not be called when skip_claims=true");
                #[allow(unreachable_code)]
                Ok(ClaimOutcome::Errored)
            })
            .unwrap()
            .expect("skip-claims yields the chunk directly");
        assert_eq!(chunk.id.as_str(), "sc-1");
        assert!(chunk.payload.contains("sc-1"));
    }

    #[test]
    fn decide_returns_chunk_on_acquired() {
        let q = RunpodChunkQueue::from_lines(client(), "w1", cfg(), vec![]);
        let chunk = q
            .decide(
                r#"{"chunk_id":"acq-1"}"#.to_string(),
                |sidecar, claim| async move {
                    // The URIs handed to the claim fn are the derived ones.
                    assert_eq!(sidecar, "s3://zentrain/test-run/omni/acq-1.parquet");
                    assert_eq!(claim, "s3://coefficient/claims/test-run/acq-1.claim");
                    Ok(ClaimOutcome::Acquired {
                        token: "tok".into(),
                    })
                },
            )
            .unwrap()
            .expect("acquired claim yields the chunk");
        assert_eq!(chunk.id.as_str(), "acq-1");
    }

    #[test]
    fn decide_skips_on_already_done_peer_lost_errored() {
        let q = RunpodChunkQueue::from_lines(client(), "w1", cfg(), vec![]);
        for outcome in [
            ClaimOutcome::AlreadyDone,
            ClaimOutcome::HeldByPeer,
            ClaimOutcome::LostRace,
            ClaimOutcome::Errored,
        ] {
            let oc = outcome.clone();
            let decided = q
                .decide(r#"{"chunk_id":"skip-1"}"#.to_string(), move |_s, _c| {
                    let oc = oc.clone();
                    async move { Ok(oc) }
                })
                .unwrap();
            assert!(decided.is_none(), "outcome {outcome:?} must skip the chunk");
        }
    }

    #[test]
    fn ack_chunk_is_noop() {
        let mut q = RunpodChunkQueue::from_lines(client(), "w1", cfg(), vec![]);
        // No in-flight state to manage; every outcome acks Ok.
        assert!(
            q.ack_chunk(&ChunkId("x".into()), ChunkOutcome::Done)
                .is_ok()
        );
        assert!(
            q.ack_chunk(
                &ChunkId("x".into()),
                ChunkOutcome::Failed { error: "e".into() }
            )
            .is_ok()
        );
    }
}
