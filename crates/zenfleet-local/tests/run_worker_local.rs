//! End-to-end abstraction-validation test (spec §1.7 Phase B).
//!
//! This is the no-cloud test the spec calls for: run the GENERIC
//! [`zenfleet_cloud::run_worker`] loop against the local backend's
//! [`LocalDirQueue`] + [`LocalFsStorage`] + [`LocalWorkerHost`] +
//! [`LocalHeartbeat`], with a TRIVIAL compute closure (NOT the GPU
//! encode+score — a stub that just echoes each chunk back to storage).
//! It proves the full claim → fetch → compute → upload → ack loop works
//! end-to-end on the filesystem with zero cloud spend, which is exactly
//! what validates the `zenfleet-cloud` trait shapes.

use std::path::Path;

use zenfleet_cloud::{
    ArtifactKey, BlobStorage, Chunk, ChunkOutcome, CloudError, WorkerHost, run_worker,
};
use zenfleet_local::{LocalDirQueue, LocalFsStorage, LocalHeartbeat, LocalWorkerHost};

fn write_jsonl(dir: &Path, lines: &[&str]) -> std::path::PathBuf {
    let path = dir.join("chunks.jsonl");
    std::fs::write(&path, lines.join("\n")).unwrap();
    path
}

/// The trivial "compute": parse the chunk's `chunk_id`, write a small
/// echo artifact to the local FS storage at an `s3://`-style key, and
/// report the outcome. This stands in for the real encode+score; the
/// point of the test is the LOOP, not the work.
fn echo_compute(
    chunk: &Chunk,
    storage: &LocalFsStorage,
    host: &LocalWorkerHost,
) -> Result<ChunkOutcome, CloudError> {
    // A chunk whose id starts with "skip-" simulates the already-done /
    // race-lost case; "fail-" simulates a per-chunk failure. Everything
    // else does the echo + Done.
    let id = chunk.id.as_str();
    if let Some(rest) = id.strip_prefix("skip-") {
        let _ = rest;
        return Ok(ChunkOutcome::Skipped {
            reason: "simulated already-done".into(),
        });
    }
    if id.starts_with("fail-") {
        return Ok(ChunkOutcome::Failed {
            error: "simulated per-chunk failure".into(),
        });
    }

    // Upload an artifact keyed under an s3://-style URI — LocalFsStorage
    // mirrors it under its base dir. Include the worker id to prove the
    // host is threaded through.
    let key = ArtifactKey(format!("s3://zentrain/local-test/omni/{id}.txt"));
    let body = format!(
        "worker={} chunk={} payload={}",
        host.worker_id(),
        id,
        chunk.payload
    );
    storage.put(&key, body.as_bytes())?;

    // Read it straight back to prove get/put round-trip on the FS store.
    let got = storage.get(&key)?;
    assert_eq!(got, body.as_bytes());

    Ok(ChunkOutcome::Done)
}

#[test]
fn run_worker_drives_local_backend_end_to_end() {
    let tmp = tempfile::tempdir().unwrap();
    let queue_root = tmp.path().join("queue");
    std::fs::create_dir_all(&queue_root).unwrap();
    let storage_base = tmp.path().join("blobs");

    // A 4-chunk manifest exercising every outcome path: two Done, one
    // Skipped, one Failed.
    let manifest = write_jsonl(
        &queue_root,
        &[
            r#"{"chunk_id":"chunk-0","codec":"jpeg","q":80}"#,
            r#"{"chunk_id":"skip-1","codec":"webp"}"#,
            r#"{"chunk_id":"fail-2","codec":"avif"}"#,
            r#"{"chunk_id":"chunk-3","codec":"jxl","q":40}"#,
        ],
    );

    let mut queue = LocalDirQueue::open_jsonl(&manifest).unwrap();
    let storage = LocalFsStorage::new(&storage_base);
    let heartbeat = LocalHeartbeat;
    let host = LocalWorkerHost::new("local-test-worker", tmp.path().join("scratch"));

    let summary = run_worker(&mut queue, &storage, &heartbeat, &host, echo_compute)
        .expect("the local loop must not error");

    // Loop accounting.
    assert_eq!(summary.dispatched, 4);
    assert_eq!(summary.done, 2);
    assert_eq!(summary.skipped, 1);
    assert_eq!(summary.failed, 1);

    // The two Done chunks each wrote one artifact to the FS store.
    let listed = storage.list("s3://zentrain/local-test/omni/").unwrap();
    let mut keys: Vec<String> = listed.into_iter().map(|k| k.0).collect();
    keys.sort();
    assert_eq!(
        keys,
        vec![
            "s3://zentrain/local-test/omni/chunk-0.txt".to_string(),
            "s3://zentrain/local-test/omni/chunk-3.txt".to_string(),
        ]
    );

    // The artifact content proves the host + payload were threaded into
    // the compute closure.
    let body = storage
        .get(&ArtifactKey(
            "s3://zentrain/local-test/omni/chunk-0.txt".into(),
        ))
        .unwrap();
    let body = String::from_utf8(body).unwrap();
    assert!(body.contains("worker=local-test-worker"));
    assert!(body.contains("chunk=chunk-0"));
    assert!(body.contains("\"codec\":\"jpeg\""));

    // The queue's on-disk state reflects each terminal outcome.
    let state = queue_root.join(".zen-queue-state");
    assert!(state.join("done/chunk-0.json").exists());
    assert!(state.join("done/skip-1.json").exists()); // Skipped → done/
    assert!(state.join("failed/fail-2.json").exists());
    assert!(state.join("done/chunk-3.json").exists());

    // The skipped chunk wrote no artifact.
    assert!(
        storage
            .head(&ArtifactKey(
                "s3://zentrain/local-test/omni/skip-1.txt".into()
            ))
            .unwrap()
            .is_none()
    );
}

/// A queue-directory source (one `*.json` per chunk) drives the same
/// loop — the dir-mode claim-by-rename path.
#[test]
fn run_worker_drives_dir_mode_queue() {
    let tmp = tempfile::tempdir().unwrap();
    let qdir = tmp.path().join("queue");
    std::fs::create_dir_all(&qdir).unwrap();
    std::fs::write(qdir.join("0.json"), r#"{"chunk_id":"chunk-a"}"#).unwrap();
    std::fs::write(qdir.join("1.json"), r#"{"chunk_id":"chunk-b"}"#).unwrap();

    let mut queue = LocalDirQueue::open_dir(&qdir).unwrap();
    let storage = LocalFsStorage::new(tmp.path().join("blobs"));
    let heartbeat = LocalHeartbeat;
    let host = LocalWorkerHost::new("dir-worker", tmp.path().join("scratch"));

    let summary = run_worker(&mut queue, &storage, &heartbeat, &host, echo_compute)
        .expect("dir-mode loop must not error");

    assert_eq!(summary.dispatched, 2);
    assert_eq!(summary.done, 2);
    // Both source files were consumed out of the queue dir.
    assert!(!qdir.join("0.json").exists());
    assert!(!qdir.join("1.json").exists());
    let state = qdir.join(".zen-queue-state");
    assert!(state.join("done/chunk-a.json").exists());
    assert!(state.join("done/chunk-b.json").exists());
}
