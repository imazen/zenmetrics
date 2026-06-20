//! The ledger row + an in-memory latest-wins view. The ledger is the single source of truth
//! (columnar Parquet at rest; latest-wins on `(job_id, ts)`). These are the shapes the reconciler
//! and GC reason over — a FAILED row is a first-class record (goal B), never a gap.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::content::Sha256Hex;
use crate::ids::{CellId, JobId};
use crate::job::JobKind;
use crate::status::{ErrorClass, JobStatus};

/// One recorded outcome of one work item. Appended (never mutated); latest `ts` wins at read time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerRow {
    pub job_id: JobId,
    /// The job kind — carried in the row so the dashboard can group/drill-down by kind & metric
    /// (goal B) without re-deriving it from the queue.
    pub kind: JobKind,
    pub cell: CellId,
    /// The produced blob's content hash, if the job emitted one (None on failure).
    pub output_sha: Option<Sha256Hex>,
    pub status: JobStatus,
    pub error_class: Option<ErrorClass>,
    pub attempts: u32,
    /// Unix seconds. Latest-wins ordering key.
    pub ts: u64,
    pub worker: String,
    pub provider: String,
}

/// An advisory per-job resource estimate, attached to a [`DesiredJob`] at
/// declare time (from the codec's own `estimate_encode_resources` against the
/// source dimensions) so a worker can pack concurrent jobs under its RAM + core
/// budget via [`crate::schedule::BoxBudget`] instead of a fixed N-per-core
/// fan-out — the lever for "some encodes need 8 GB, some 80 MB; some are serial,
/// some self-thread to dozens of cores".
///
/// **Advisory only, and deliberately *not* part of the content-addressed
/// [`JobId`]** (which hashes kind + inputs). Attaching, refining, or dropping a
/// hint never changes a job's identity or its dedup/idempotence — two declares
/// of the same work with different hints still resolve to one job.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceHint {
    /// Conservative upper-bound peak memory for this job, in bytes
    /// (`zencodec::estimate::ResourceEstimate::peak_memory_bytes_max`). Feeds
    /// the memory term of [`BoxBudget::can_admit`](crate::schedule::BoxBudget::can_admit).
    pub peak_mem_bytes: u64,
    /// Threads the job can usefully occupy at the box's core count
    /// (`ResourceEstimate::threading().effective_threads(cores)`). Feeds the
    /// core term of admission control; `1` for a serial encode.
    pub threads: u32,
}

/// A job an agent or the reconciler *wants* to exist: its kind + content-addressed inputs + the
/// human identity tuple it maps to. Its [`JobId`] is content-addressed, so declaring the same work
/// twice resolves to the same id (idempotent — goals A & I).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesiredJob {
    pub kind: JobKind,
    pub inputs: Vec<Sha256Hex>,
    pub cell: CellId,
    /// Optional scheduling hint (see [`ResourceHint`]). `None` when the declarer
    /// couldn't estimate (e.g. dims unknown, or a non-encode kind) — the worker
    /// then falls back to its default fan-out. Not part of [`JobId`], and
    /// `#[serde(default)]` so manifests written before this field deserialize
    /// cleanly as `None`.
    #[serde(default)]
    pub hint: Option<ResourceHint>,
}

impl DesiredJob {
    /// A desired job with no scheduling hint. Attach one with [`Self::with_hint`].
    pub fn new(kind: JobKind, inputs: Vec<Sha256Hex>, cell: CellId) -> Self {
        Self {
            kind,
            inputs,
            cell,
            hint: None,
        }
    }

    /// Attach a scheduling [`ResourceHint`] (builder style). Does not affect
    /// [`Self::job_id`].
    pub fn with_hint(mut self, hint: ResourceHint) -> Self {
        self.hint = Some(hint);
        self
    }

    pub fn job_id(&self) -> JobId {
        JobId::of(&self.kind, &self.inputs)
    }
}

/// Latest-wins view of the ledger: the current state of each `job_id`.
#[derive(Clone, Debug, Default)]
pub struct LedgerView {
    latest: HashMap<JobId, LedgerRow>,
}

impl LedgerView {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fold in a row, keeping the one with the greatest `ts` (ties broken by status rank, so a
    /// terminal verdict can't be regressed by a same-second in-flight row).
    pub fn apply(&mut self, row: LedgerRow) {
        let keep = match self.latest.get(&row.job_id) {
            None => true,
            Some(cur) => {
                row.ts > cur.ts || (row.ts == cur.ts && row.status.rank() > cur.status.rank())
            }
        };
        if keep {
            self.latest.insert(row.job_id.clone(), row);
        }
    }

    pub fn from_rows<I: IntoIterator<Item = LedgerRow>>(rows: I) -> Self {
        let mut v = Self::new();
        for r in rows {
            v.apply(r);
        }
        v
    }

    pub fn get(&self, id: &JobId) -> Option<&LedgerRow> {
        self.latest.get(id)
    }

    pub fn len(&self) -> usize {
        self.latest.len()
    }

    pub fn is_empty(&self) -> bool {
        self.latest.is_empty()
    }

    pub fn rows(&self) -> impl Iterator<Item = &LedgerRow> {
        self.latest.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::sha256;

    fn row(id: JobId, status: JobStatus, ts: u64) -> LedgerRow {
        LedgerRow {
            job_id: id,
            kind: JobKind::Metric {
                metric: "cvvdp".into(),
            },
            cell: CellId {
                image_path: "x.png".into(),
                codec: "zenjpeg".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
            output_sha: None,
            status,
            error_class: None,
            attempts: 1,
            ts,
            worker: "w1".into(),
            provider: "local".into(),
        }
    }

    #[test]
    fn latest_ts_wins() {
        let id = JobId::of(
            &JobKind::Metric {
                metric: "cvvdp".into(),
            },
            &[sha256(b"e")],
        );
        let mut v = LedgerView::new();
        v.apply(row(id.clone(), JobStatus::Failed, 100));
        v.apply(row(id.clone(), JobStatus::Done, 200)); // newer wins
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
        v.apply(row(id.clone(), JobStatus::Failed, 150)); // older loses
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
    }

    #[test]
    fn same_ts_terminal_wins() {
        let id = JobId::of(
            &JobKind::Metric {
                metric: "ssim2".into(),
            },
            &[sha256(b"e")],
        );
        let mut v = LedgerView::new();
        v.apply(row(id.clone(), JobStatus::Claimed, 100));
        v.apply(row(id.clone(), JobStatus::Done, 100)); // same ts, higher rank wins
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
    }

    fn sample_desired() -> DesiredJob {
        DesiredJob::new(
            JobKind::Encode {
                codec: "zenjxl".into(),
                q: 90,
                knobs: "{\"effort\":9}".into(),
            },
            vec![sha256(b"src")],
            CellId {
                image_path: "x".into(),
                codec: "zenjxl".into(),
                q: 90,
                knob_tuple_json: "{}".into(),
            },
        )
    }

    #[test]
    fn desired_job_id_is_content_addressed() {
        let d = sample_desired();
        assert_eq!(d.job_id(), d.clone().job_id());
        // new() leaves the hint unset.
        assert_eq!(d.hint, None);
    }

    #[test]
    fn hint_does_not_change_job_id() {
        let base = sample_desired();
        let hinted = base.clone().with_hint(ResourceHint {
            peak_mem_bytes: 8 << 30,
            threads: 4,
        });
        // Identity hashes kind + inputs only — the hint is advisory metadata.
        assert_eq!(base.job_id(), hinted.job_id());
        assert_eq!(
            hinted.hint,
            Some(ResourceHint {
                peak_mem_bytes: 8 << 30,
                threads: 4
            })
        );
    }

    #[test]
    fn hint_serde_roundtrips_and_defaults_to_none_for_legacy_manifests() {
        // A hinted job survives a serde round-trip intact.
        let hinted = sample_desired().with_hint(ResourceHint {
            peak_mem_bytes: 80 << 20,
            threads: 1,
        });
        let json = serde_json::to_string(&hinted).unwrap();
        assert_eq!(serde_json::from_str::<DesiredJob>(&json).unwrap(), hinted);
        // A manifest written before the `hint` field existed (no key) still
        // deserializes — `#[serde(default)]` fills it with None.
        let legacy = r#"{"kind":{"kind":"encode","codec":"zenjpeg","q":80,"knobs":"{}"},"inputs":["aa"],"cell":{"image_path":"x","codec":"zenjpeg","q":80,"knob_tuple_json":"{}"}}"#;
        let parsed: DesiredJob = serde_json::from_str(legacy).unwrap();
        assert_eq!(parsed.hint, None);
    }
}
