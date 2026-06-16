//! Local `JobQueue` — a plain-filesystem queue, no cloud, single-process.
//!
//! The local backend pulls work from one of two on-disk sources
//! ([`LocalQueueSource`]):
//!
//! 1. **A `chunks.jsonl` file** — the SAME manifest the vast.ai / runpod
//!    fleet reads: one chunk record (`{"chunk_id":…}`) per line. The
//!    queue snapshots the lines on construction and hands them out in
//!    order. A claim sub-dir under the queue's state dir records which
//!    records were claimed so a re-run resumes (skipping already-acked
//!    ones).
//! 2. **A queue directory of `*.json` files** — one chunk record per
//!    file. `next_chunk` claims the lexicographically-next unclaimed
//!    file by *renaming* it into a `claimed/` sub-dir (an atomic rename
//!    on the same filesystem — sufficient for a single process; no
//!    R2-ETag race needed). `ack_chunk` then moves it to `done/` or
//!    `failed/`.
//!
//! In both modes the chunk's stable id is the `chunk_id` field parsed
//! from the record, and the `Chunk::payload` is the raw record text —
//! exactly what the cloud workers' inline compute re-parses. The queue
//! NEVER inspects the payload beyond extracting `chunk_id`.
//!
//! ## Outcome → on-disk state
//!
//! | [`ChunkOutcome`] | jsonl mode | dir mode |
//! |---|---|---|
//! | `Done`     | claim marker → `done/<id>.json`   | `claimed/` → `done/`   |
//! | `Skipped`  | claim marker → `done/<id>.json`   | `claimed/` → `done/`   |
//! | `Retryable`| claim marker removed (re-runnable)| `claimed/` → queue dir |
//! | `Failed`   | claim marker → `failed/<id>.json` | `claimed/` → `failed/` |
//!
//! A `Retryable` outcome releases the claim so a later run re-attempts
//! the chunk; `Done`/`Skipped`/`Failed` are terminal for this state dir.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use zenfleet_cloud::{Chunk, ChunkId, ChunkOutcome, CloudError, JobQueue};

/// Where the local queue reads chunk records from.
#[derive(Clone, Debug)]
pub enum LocalQueueSource {
    /// A `chunks.jsonl` manifest file — one chunk record per line.
    Jsonl(PathBuf),
    /// A directory of `*.json` files — one chunk record per file.
    Dir(PathBuf),
}

/// Configuration for a [`LocalDirQueue`].
#[derive(Clone, Debug)]
pub struct LocalQueueConfig {
    /// The chunk-record source (a `chunks.jsonl` file or a queue dir).
    pub source: LocalQueueSource,
    /// State dir holding the `claimed/`, `done/`, and `failed/`
    /// sub-dirs. Defaults (via [`LocalQueueConfig::new`]) to a
    /// `.zen-queue-state` dir alongside the source. The dir-mode queue
    /// also moves files between these; the jsonl-mode queue uses them as
    /// claim markers.
    pub state_dir: PathBuf,
}

impl LocalQueueConfig {
    /// Build a config from a source, placing the state dir alongside it.
    ///
    /// For a `chunks.jsonl` at `<p>/chunks.jsonl` the state dir is
    /// `<p>/.zen-queue-state`; for a queue dir `<d>` it is
    /// `<d>/.zen-queue-state`.
    pub fn new(source: LocalQueueSource) -> Self {
        let anchor = match &source {
            LocalQueueSource::Jsonl(p) => p.parent().map(Path::to_path_buf).unwrap_or_default(),
            LocalQueueSource::Dir(d) => d.clone(),
        };
        Self {
            state_dir: anchor.join(".zen-queue-state"),
            source,
        }
    }

    /// Override the state dir (e.g. point it at a scratch location so the
    /// queue source dir stays read-only).
    pub fn with_state_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.state_dir = dir.into();
        self
    }

    fn claimed_dir(&self) -> PathBuf {
        self.state_dir.join("claimed")
    }
    fn done_dir(&self) -> PathBuf {
        self.state_dir.join("done")
    }
    fn failed_dir(&self) -> PathBuf {
        self.state_dir.join("failed")
    }
}

/// Parse the stable `chunk_id` out of a chunk record (jsonl line or
/// `*.json` file body). The rest of the record is opaque to the queue.
fn parse_chunk_id(record: &str) -> Result<String, CloudError> {
    #[derive(Deserialize)]
    struct Rec {
        chunk_id: String,
    }
    let rec: Rec = serde_json::from_str(record)
        .map_err(|e| CloudError::Queue(format!("parse chunk record: {e}")))?;
    if rec.chunk_id.is_empty() {
        return Err(CloudError::Queue("chunk record has empty chunk_id".into()));
    }
    Ok(rec.chunk_id)
}

/// The state-dir filename for a chunk id's marker / moved file. Chunk
/// ids are sweep-generated (`chunk-0`, hex hashes) so they are
/// path-safe; we still guard against a separator sneaking in.
fn marker_name(chunk_id: &str) -> Result<String, CloudError> {
    if chunk_id.contains('/') || chunk_id.contains('\\') || chunk_id.contains("..") {
        return Err(CloudError::Queue(format!(
            "chunk_id is not a safe filename: {chunk_id:?}"
        )));
    }
    Ok(format!("{chunk_id}.json"))
}

/// Plain-filesystem [`JobQueue`] for the local backend.
pub struct LocalDirQueue {
    cfg: LocalQueueConfig,
    /// Pending records to hand out, each as `(chunk_id, payload,
    /// origin)`. For jsonl mode `origin` is `None` (the marker lives in
    /// the state dir); for dir mode it is the source file path so
    /// `ack_chunk` can move it.
    pending: std::vec::IntoIter<PendingChunk>,
}

#[derive(Clone, Debug)]
struct PendingChunk {
    chunk_id: String,
    payload: String,
    /// In dir mode, the source file's path within the queue dir (used to
    /// move it to claimed/done/failed). `None` in jsonl mode.
    origin: Option<PathBuf>,
}

impl LocalDirQueue {
    /// Open a queue over `cfg.source`, creating the state sub-dirs.
    ///
    /// In jsonl mode the manifest lines are read once; records whose
    /// `done/` or `failed/` marker already exists are skipped (resume
    /// support). In dir mode the queue dir is scanned for `*.json`
    /// files (already-`claimed/`/`done/`/`failed/` files live in the
    /// state dir, not the queue dir, so they are naturally excluded).
    pub fn open(cfg: LocalQueueConfig) -> Result<Self, CloudError> {
        for d in [cfg.claimed_dir(), cfg.done_dir(), cfg.failed_dir()] {
            std::fs::create_dir_all(&d)
                .map_err(|e| CloudError::Queue(format!("create {}: {e}", d.display())))?;
        }

        let pending = match &cfg.source {
            LocalQueueSource::Jsonl(path) => Self::scan_jsonl(&cfg, path)?,
            LocalQueueSource::Dir(dir) => Self::scan_dir(&cfg, dir)?,
        };

        Ok(Self {
            cfg,
            pending: pending.into_iter(),
        })
    }

    /// Convenience: open a `chunks.jsonl`-backed queue.
    pub fn open_jsonl(path: impl Into<PathBuf>) -> Result<Self, CloudError> {
        Self::open(LocalQueueConfig::new(LocalQueueSource::Jsonl(path.into())))
    }

    /// Convenience: open a queue-directory-backed queue.
    pub fn open_dir(dir: impl Into<PathBuf>) -> Result<Self, CloudError> {
        Self::open(LocalQueueConfig::new(LocalQueueSource::Dir(dir.into())))
    }

    fn scan_jsonl(cfg: &LocalQueueConfig, path: &Path) -> Result<Vec<PendingChunk>, CloudError> {
        let body = std::fs::read_to_string(path)
            .map_err(|e| CloudError::Queue(format!("read {}: {e}", path.display())))?;
        let mut out = Vec::new();
        for raw in body.lines() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let chunk_id = parse_chunk_id(line)?;
            let name = marker_name(&chunk_id)?;
            // Resume: skip records already terminal in this state dir.
            if cfg.done_dir().join(&name).exists() || cfg.failed_dir().join(&name).exists() {
                tracing::debug!(chunk_id = %chunk_id, "skip: already terminal in state dir");
                continue;
            }
            out.push(PendingChunk {
                chunk_id,
                payload: line.to_string(),
                origin: None,
            });
        }
        Ok(out)
    }

    fn scan_dir(_cfg: &LocalQueueConfig, dir: &Path) -> Result<Vec<PendingChunk>, CloudError> {
        let mut files: Vec<PathBuf> = Vec::new();
        let entries = std::fs::read_dir(dir)
            .map_err(|e| CloudError::Queue(format!("read dir {}: {e}", dir.display())))?;
        for entry in entries {
            let entry =
                entry.map_err(|e| CloudError::Queue(format!("read dir {}: {e}", dir.display())))?;
            let path = entry.path();
            // Only top-level `*.json` files are chunk records; the
            // `.zen-queue-state` sub-dir and anything else is ignored.
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("json") {
                files.push(path);
            }
        }
        // Lexicographic order → deterministic, "next" is well-defined.
        files.sort();

        let mut out = Vec::with_capacity(files.len());
        for path in files {
            let body = std::fs::read_to_string(&path)
                .map_err(|e| CloudError::Queue(format!("read {}: {e}", path.display())))?;
            let chunk_id = parse_chunk_id(body.trim())?;
            out.push(PendingChunk {
                chunk_id,
                payload: body.trim().to_string(),
                origin: Some(path),
            });
        }
        Ok(out)
    }

    /// Move a file, falling back to copy+remove across filesystems
    /// (rename fails with `EXDEV` when src/dst are on different mounts).
    fn move_file(from: &Path, to: &Path) -> Result<(), CloudError> {
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| CloudError::Queue(format!("create {}: {e}", parent.display())))?;
        }
        match std::fs::rename(from, to) {
            Ok(()) => Ok(()),
            Err(_) => {
                std::fs::copy(from, to).map_err(|e| {
                    CloudError::Queue(format!("copy {} -> {}: {e}", from.display(), to.display()))
                })?;
                std::fs::remove_file(from)
                    .map_err(|e| CloudError::Queue(format!("remove {}: {e}", from.display())))?;
                Ok(())
            }
        }
    }
}

impl JobQueue for LocalDirQueue {
    fn next_chunk(&mut self) -> Result<Option<Chunk>, CloudError> {
        let Some(pending) = self.pending.next() else {
            return Ok(None);
        };
        let name = marker_name(&pending.chunk_id)?;

        // Record the claim. In dir mode the record file is MOVED out of
        // the queue dir into `claimed/` (so a concurrent re-scan can't
        // pick it up); in jsonl mode a claim marker file is written.
        match &pending.origin {
            Some(src) => {
                let dst = self.cfg.claimed_dir().join(&name);
                Self::move_file(src, &dst)?;
            }
            None => {
                let marker = self.cfg.claimed_dir().join(&name);
                std::fs::write(&marker, pending.payload.as_bytes()).map_err(|e| {
                    CloudError::Queue(format!("write claim {}: {e}", marker.display()))
                })?;
            }
        }

        tracing::debug!(chunk_id = %pending.chunk_id, "claimed");
        Ok(Some(Chunk {
            id: ChunkId(pending.chunk_id),
            payload: pending.payload,
        }))
    }

    fn ack_chunk(&mut self, id: &ChunkId, outcome: ChunkOutcome) -> Result<(), CloudError> {
        let name = marker_name(id.as_str())?;
        let claimed = self.cfg.claimed_dir().join(&name);

        let dest = match &outcome {
            ChunkOutcome::Done | ChunkOutcome::Skipped { .. } => {
                Some(self.cfg.done_dir().join(&name))
            }
            ChunkOutcome::Failed { .. } => Some(self.cfg.failed_dir().join(&name)),
            // Retryable: release the claim so a later run re-attempts it.
            // In dir mode the file goes back to the queue dir; in jsonl
            // mode the claim marker is removed (the manifest line will be
            // re-read next run).
            ChunkOutcome::Retryable { .. } => None,
        };

        match (dest, &outcome) {
            (Some(dest), _) => {
                // Move the claimed record to its terminal location. If the
                // claim file is missing (already moved / never claimed),
                // record the outcome anyway so the terminal marker exists.
                if claimed.exists() {
                    Self::move_file(&claimed, &dest)?;
                } else {
                    std::fs::write(&dest, b"")
                        .map_err(|e| CloudError::Queue(format!("write {}: {e}", dest.display())))?;
                }
            }
            (None, ChunkOutcome::Retryable { .. }) => {
                // Release: dir-mode → back to the queue dir; jsonl-mode →
                // drop the claim marker.
                if let LocalQueueSource::Dir(qdir) = &self.cfg.source {
                    if claimed.exists() {
                        Self::move_file(&claimed, &qdir.join(&name))?;
                    }
                } else if claimed.exists() {
                    std::fs::remove_file(&claimed).map_err(|e| {
                        CloudError::Queue(format!("remove claim {}: {e}", claimed.display()))
                    })?;
                }
            }
            (None, _) => unreachable!("only Retryable maps to no destination"),
        }

        tracing::debug!(chunk_id = %id, ?outcome, "acked");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_jsonl(dir: &Path, lines: &[&str]) -> PathBuf {
        let path = dir.join("chunks.jsonl");
        std::fs::write(&path, lines.join("\n")).unwrap();
        path
    }

    #[test]
    fn parse_chunk_id_extracts_and_rejects() {
        assert_eq!(parse_chunk_id(r#"{"chunk_id":"c0","x":1}"#).unwrap(), "c0");
        assert!(parse_chunk_id("not json").is_err());
        assert!(parse_chunk_id(r#"{"no_id":1}"#).is_err());
        assert!(parse_chunk_id(r#"{"chunk_id":""}"#).is_err());
    }

    #[test]
    fn marker_name_rejects_unsafe_ids() {
        assert_eq!(marker_name("chunk-0").unwrap(), "chunk-0.json");
        assert!(marker_name("a/b").is_err());
        assert!(marker_name("../escape").is_err());
    }

    #[test]
    fn jsonl_drains_in_order_and_acks_to_done() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_jsonl(
            dir.path(),
            &[
                r#"{"chunk_id":"chunk-0","codec":"jpeg"}"#,
                r#"{"chunk_id":"chunk-1","codec":"webp"}"#,
            ],
        );
        let mut q = LocalDirQueue::open_jsonl(&path).unwrap();

        let c0 = q.next_chunk().unwrap().expect("c0");
        assert_eq!(c0.id.as_str(), "chunk-0");
        assert!(c0.payload.contains("jpeg"));
        // Claim marker exists after claiming.
        let state = path.parent().unwrap().join(".zen-queue-state");
        assert!(state.join("claimed/chunk-0.json").exists());
        q.ack_chunk(&c0.id, ChunkOutcome::Done).unwrap();
        assert!(state.join("done/chunk-0.json").exists());
        assert!(!state.join("claimed/chunk-0.json").exists());

        let c1 = q.next_chunk().unwrap().expect("c1");
        assert_eq!(c1.id.as_str(), "chunk-1");
        q.ack_chunk(
            &c1.id,
            ChunkOutcome::Failed {
                error: "boom".into(),
            },
        )
        .unwrap();
        assert!(state.join("failed/chunk-1.json").exists());

        assert!(q.next_chunk().unwrap().is_none(), "drained");
    }

    #[test]
    fn jsonl_resume_skips_terminal_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_jsonl(
            dir.path(),
            &[r#"{"chunk_id":"c0"}"#, r#"{"chunk_id":"c1"}"#],
        );
        // First pass: ack c0 Done.
        {
            let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
            let c0 = q.next_chunk().unwrap().unwrap();
            q.ack_chunk(&c0.id, ChunkOutcome::Done).unwrap();
        }
        // Second pass: c0 is terminal → only c1 is handed out.
        let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
        let next = q.next_chunk().unwrap().expect("c1 remains");
        assert_eq!(next.id.as_str(), "c1");
        assert!(q.next_chunk().unwrap().is_none());
    }

    #[test]
    fn jsonl_retryable_releases_claim_for_rerun() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_jsonl(dir.path(), &[r#"{"chunk_id":"c0"}"#]);
        let state = dir.path().join(".zen-queue-state");
        {
            let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
            let c0 = q.next_chunk().unwrap().unwrap();
            q.ack_chunk(
                &c0.id,
                ChunkOutcome::Retryable {
                    error: "net".into(),
                },
            )
            .unwrap();
            // Claim released; not terminal.
            assert!(!state.join("claimed/c0.json").exists());
            assert!(!state.join("done/c0.json").exists());
        }
        // Re-run hands c0 out again.
        let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
        assert_eq!(q.next_chunk().unwrap().unwrap().id.as_str(), "c0");
    }

    #[test]
    fn dir_mode_claims_by_rename_and_acks() {
        let dir = tempfile::tempdir().unwrap();
        let qdir = dir.path().join("queue");
        std::fs::create_dir_all(&qdir).unwrap();
        std::fs::write(qdir.join("a.json"), r#"{"chunk_id":"chunk-a"}"#).unwrap();
        std::fs::write(qdir.join("b.json"), r#"{"chunk_id":"chunk-b"}"#).unwrap();

        let mut q = LocalDirQueue::open_dir(&qdir).unwrap();
        let state = qdir.join(".zen-queue-state");

        let first = q.next_chunk().unwrap().expect("first");
        // Lexicographic file order → a.json first.
        assert_eq!(first.id.as_str(), "chunk-a");
        // The source file was moved out of the queue dir into claimed/.
        assert!(!qdir.join("a.json").exists());
        assert!(state.join("claimed/chunk-a.json").exists());
        q.ack_chunk(&first.id, ChunkOutcome::Done).unwrap();
        assert!(state.join("done/chunk-a.json").exists());

        let second = q.next_chunk().unwrap().expect("second");
        assert_eq!(second.id.as_str(), "chunk-b");
        q.ack_chunk(&second.id, ChunkOutcome::Retryable { error: "x".into() })
            .unwrap();
        // Retryable in dir mode puts the file back in the queue dir.
        assert!(qdir.join("chunk-b.json").exists());

        assert!(q.next_chunk().unwrap().is_none());
    }

    #[test]
    fn empty_jsonl_yields_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_jsonl(dir.path(), &[]);
        let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
        assert!(q.next_chunk().unwrap().is_none());
    }

    #[test]
    fn jsonl_ignores_blank_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_jsonl(dir.path(), &[r#"{"chunk_id":"c0"}"#, "", "   "]);
        let mut q = LocalDirQueue::open_jsonl(&path).unwrap();
        assert_eq!(q.next_chunk().unwrap().unwrap().id.as_str(), "c0");
        assert!(q.next_chunk().unwrap().is_none());
    }
}
