//! The generic job loop — the heart of the worker, backend-agnostic.
//!
//! `run_worker` ties the five traits together into one
//! claim → fetch → compute → upload → ack → beat sequence and loops
//! until the queue drains. Everything provider-specific lives behind
//! the trait impls; the `compute` closure is whatever the sweep does
//! (encode + score, picker train, IQA panel batch). Spec §1.5.
//!
//! This loop is intentionally synchronous and dependency-free. It is
//! the abstraction Phase B's `zenfleet-local` backend validates and
//! the shape Phase C/D providers reuse. The proven vast.ai async
//! worker (adaptive concurrency, seeded shuffle, claim races) keeps
//! its own entry point in `zenfleet-vastai` for byte-identical
//! production behaviour during Phase A; this loop is the single-flight
//! reference the other backends build on.

use crate::error::CloudError;
use crate::traits::{BlobStorage, Heartbeat, JobQueue, WorkerHost};
use crate::types::{Chunk, ChunkOutcome, WorkerStatus, WorkerSummary};

/// Drive a worker over the trait surface: pull chunks from `queue`,
/// run `compute` on each (which uses `storage` + `host` as needed),
/// acknowledge the outcome, and emit heartbeats around the loop.
///
/// `compute` returns a [`ChunkOutcome`]; a returned [`CloudError`] from
/// `compute` is treated as a terminal per-chunk failure and recorded as
/// `ChunkOutcome::Failed` (the loop never dies on one chunk). The only
/// way the loop returns `Err` is a queue/ack failure it cannot make
/// progress past.
pub fn run_worker<Q, S, H, W, F>(
    queue: &mut Q,
    storage: &S,
    heartbeat: &H,
    host: &W,
    mut compute: F,
) -> Result<WorkerSummary, CloudError>
where
    Q: JobQueue,
    S: BlobStorage,
    H: Heartbeat,
    W: WorkerHost,
    F: FnMut(&Chunk, &S, &W) -> Result<ChunkOutcome, CloudError>,
{
    let worker = host.worker_id();
    // Best-effort starting beat — never abort work on a missed beat.
    let _ = heartbeat.beat(&worker, WorkerStatus::Starting);

    let mut summary = WorkerSummary::default();

    while let Some(chunk) = queue.next_chunk()? {
        summary.dispatched += 1;
        let _ = heartbeat.beat(&worker, WorkerStatus::Working { in_flight: 1 });

        // A `CloudError` out of `compute` is an *unexpected* failure we
        // record as a terminal chunk failure — distinct from an
        // *expected* `ChunkOutcome::Failed` the closure returns itself.
        let outcome = match compute(&chunk, storage, host) {
            Ok(o) => o,
            Err(e) => ChunkOutcome::Failed {
                error: e.to_string(),
            },
        };

        match &outcome {
            ChunkOutcome::Done => summary.done += 1,
            ChunkOutcome::Skipped { .. } => summary.skipped += 1,
            ChunkOutcome::Failed { .. } | ChunkOutcome::Retryable { .. } => summary.failed += 1,
        }

        queue.ack_chunk(&chunk.id, outcome)?;
    }

    let _ = heartbeat.beat(&worker, WorkerStatus::Draining);
    let _ = heartbeat.beat(&worker, WorkerStatus::Done);
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ArtifactKey, BlobMeta, ChunkId, WorkerId};
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// A pull queue that hands out a fixed list once, then drains.
    struct VecQueue {
        chunks: Vec<Chunk>,
        idx: usize,
        acks: RefCell<Vec<(ChunkId, ChunkOutcome)>>,
    }
    impl JobQueue for VecQueue {
        fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError> {
            if self.idx < self.chunks.len() {
                let c = self.chunks[self.idx].clone();
                self.idx += 1;
                Ok(Some(c))
            } else {
                Ok(None)
            }
        }
        fn ack_chunk(&mut self, id: &ChunkId, outcome: ChunkOutcome) -> Result<(), CloudError> {
            self.acks.borrow_mut().push((id.clone(), outcome));
            Ok(())
        }
    }

    /// In-memory blob store backing `put`/`get`.
    #[derive(Default)]
    struct MemStore {
        map: RefCell<HashMap<String, Vec<u8>>>,
    }
    impl BlobStorage for MemStore {
        fn put(&self, key: &ArtifactKey, bytes: &[u8]) -> Result<(), CloudError> {
            self.map.borrow_mut().insert(key.0.clone(), bytes.to_vec());
            Ok(())
        }
        fn get(&self, key: &ArtifactKey) -> Result<Vec<u8>, CloudError> {
            self.map
                .borrow()
                .get(&key.0)
                .cloned()
                .ok_or_else(|| CloudError::Storage(format!("missing {key}")))
        }
        fn head(&self, key: &ArtifactKey) -> Result<Option<BlobMeta>, CloudError> {
            Ok(self.map.borrow().get(&key.0).map(|b| BlobMeta {
                size: b.len() as u64,
                etag: None,
            }))
        }
        fn list(&self, prefix: &str) -> Result<Vec<ArtifactKey>, CloudError> {
            Ok(self
                .map
                .borrow()
                .keys()
                .filter(|k| k.starts_with(prefix))
                .map(|k| ArtifactKey(k.clone()))
                .collect())
        }
        fn delete(&self, key: &ArtifactKey) -> Result<(), CloudError> {
            self.map.borrow_mut().remove(&key.0);
            Ok(())
        }
    }

    struct NoopBeat;
    impl Heartbeat for NoopBeat {
        fn beat(&self, _w: &WorkerId, _s: WorkerStatus) -> Result<(), CloudError> {
            Ok(())
        }
    }

    struct TestHost;
    impl WorkerHost for TestHost {
        fn worker_id(&self) -> WorkerId {
            WorkerId("test-worker".into())
        }
        fn scratch_dir(&self) -> std::path::PathBuf {
            std::env::temp_dir()
        }
        fn gpu_count(&self) -> usize {
            0
        }
    }

    fn mk_chunks(ids: &[&str]) -> Vec<Chunk> {
        ids.iter()
            .map(|i| Chunk {
                id: ChunkId((*i).into()),
                payload: format!("{{\"chunk_id\":\"{i}\"}}"),
            })
            .collect()
    }

    #[test]
    fn drains_queue_and_counts_outcomes() {
        let mut q = VecQueue {
            chunks: mk_chunks(&["a", "b", "c", "d"]),
            idx: 0,
            acks: RefCell::new(Vec::new()),
        };
        let store = MemStore::default();
        let beat = NoopBeat;
        let host = TestHost;

        // compute: a=Done, b=Skipped, c=Failed, d=Done. It also writes
        // a blob per Done chunk to exercise storage threading.
        let summary = run_worker(&mut q, &store, &beat, &host, |chunk, s, _h| {
            let outcome = match chunk.id.as_str() {
                "a" | "d" => {
                    s.put(&ArtifactKey(format!("out/{}", chunk.id)), b"ok")?;
                    ChunkOutcome::Done
                }
                "b" => ChunkOutcome::Skipped {
                    reason: "exists".into(),
                },
                _ => ChunkOutcome::Failed {
                    error: "boom".into(),
                },
            };
            Ok(outcome)
        })
        .expect("loop should not error");

        assert_eq!(summary.dispatched, 4);
        assert_eq!(summary.done, 2);
        assert_eq!(summary.skipped, 1);
        assert_eq!(summary.failed, 1);
        // Two Done chunks wrote two blobs.
        assert_eq!(store.list("out/").unwrap().len(), 2);
        // Every chunk was acked.
        assert_eq!(q.acks.borrow().len(), 4);
    }

    #[test]
    fn compute_error_becomes_failed_not_loop_abort() {
        let mut q = VecQueue {
            chunks: mk_chunks(&["x"]),
            idx: 0,
            acks: RefCell::new(Vec::new()),
        };
        let store = MemStore::default();
        let beat = NoopBeat;
        let host = TestHost;

        let summary = run_worker(&mut q, &store, &beat, &host, |_c, _s, _h| {
            Err(CloudError::Compute("kaboom".into()))
        })
        .expect("a compute error must not abort the loop");

        assert_eq!(summary.dispatched, 1);
        assert_eq!(summary.failed, 1);
        // The ack recorded a Failed outcome carrying the message.
        let acks = q.acks.borrow();
        match &acks[0].1 {
            ChunkOutcome::Failed { error } => assert!(error.contains("kaboom")),
            other => panic!("expected Failed, got {other:?}"),
        }
    }

    #[test]
    fn empty_queue_yields_empty_summary() {
        let mut q = VecQueue {
            chunks: Vec::new(),
            idx: 0,
            acks: RefCell::new(Vec::new()),
        };
        let store = MemStore::default();
        let beat = NoopBeat;
        let host = TestHost;
        let summary = run_worker(&mut q, &store, &beat, &host, |_c, _s, _h| {
            Ok(ChunkOutcome::Done)
        })
        .unwrap();
        assert_eq!(summary, WorkerSummary::default());
    }
}
