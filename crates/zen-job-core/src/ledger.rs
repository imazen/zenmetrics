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

/// A job an agent or the reconciler *wants* to exist: its kind + content-addressed inputs + the
/// human identity tuple it maps to. Its [`JobId`] is content-addressed, so declaring the same work
/// twice resolves to the same id (idempotent — goals A & I).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DesiredJob {
    pub kind: JobKind,
    pub inputs: Vec<Sha256Hex>,
    pub cell: CellId,
}

impl DesiredJob {
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
            kind: JobKind::Metric { metric: "cvvdp".into() },
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
        let id = JobId::of(&JobKind::Metric { metric: "cvvdp".into() }, &[sha256(b"e")]);
        let mut v = LedgerView::new();
        v.apply(row(id.clone(), JobStatus::Failed, 100));
        v.apply(row(id.clone(), JobStatus::Done, 200)); // newer wins
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
        v.apply(row(id.clone(), JobStatus::Failed, 150)); // older loses
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
    }

    #[test]
    fn same_ts_terminal_wins() {
        let id = JobId::of(&JobKind::Metric { metric: "ssim2".into() }, &[sha256(b"e")]);
        let mut v = LedgerView::new();
        v.apply(row(id.clone(), JobStatus::Claimed, 100));
        v.apply(row(id.clone(), JobStatus::Done, 100)); // same ts, higher rank wins
        assert_eq!(v.get(&id).unwrap().status, JobStatus::Done);
    }

    #[test]
    fn desired_job_id_is_content_addressed() {
        let d = DesiredJob {
            kind: JobKind::Metric { metric: "cvvdp".into() },
            inputs: vec![sha256(b"enc")],
            cell: CellId {
                image_path: "x".into(),
                codec: "zenjpeg".into(),
                q: 1,
                knob_tuple_json: "{}".into(),
            },
        };
        assert_eq!(d.job_id(), d.clone().job_id());
    }
}
