//! Per-item outcome vocabulary. Failures are first-class values (rows in the ledger), never gaps —
//! this is what gives "exactly what failed" visibility (goal B) and drives retry-vs-poison (goal F).

use serde::{Deserialize, Serialize};

/// Terminal (and in-flight) state of one work item, recorded as a row in the Parquet ledger.
/// `Pending`/`Claimed` live in the queue; `Done`/`Failed`/`Poison` are durable rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Claimed,
    Done,
    Failed,
    Poison,
}

impl JobStatus {
    /// A terminal state needs no further scheduling.
    pub fn is_terminal(self) -> bool {
        matches!(self, JobStatus::Done | JobStatus::Poison)
    }

    /// Tie-break precedence when two ledger rows share a timestamp (higher wins). A success or
    /// poison verdict beats an in-flight state so latest-wins can't regress a finished job.
    pub fn rank(self) -> u8 {
        match self {
            JobStatus::Pending => 0,
            JobStatus::Claimed => 1,
            JobStatus::Failed => 2,
            JobStatus::Poison => 3,
            JobStatus::Done => 4,
        }
    }
}

/// Classified failure cause — a small enum so millions of failures are *aggregatable*
/// (`GROUP BY error_class`) instead of a wall of free text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorClass {
    Timeout,
    Oom,
    DecodeError,
    EncoderPanic,
    MetricNan,
    UploadFail,
    WorkerLost,
    /// Fetching the job's SOURCE bytes (corpus image / persisted variant) failed — network, cred, or
    /// object-missing. Transient by default: a retry on another box (or after a cred refresh)
    /// usually succeeds; a deterministic config gap (the G-Z2 corpus-cred incident, 2026-08-06,
    /// where every fetch failed and was mislabeled `encoder_panic`) still poisons at the retry cap.
    SourceFetch,
    /// `ENOSPC` — no space on the box's scratch/output disk. Box-level and transient: another box
    /// (or the same box after the between-pass scratch sweep) succeeds. Previously mislabeled
    /// `encoder_panic` (the hdrgrid ENOSPC incident, `9d30a00b`) which poisoned the cells.
    DiskFull,
    Unknown,
}

impl ErrorClass {
    /// Transient failures are worth retrying (different box, transient load, lost worker); a job that
    /// keeps hitting these is capped into `Poison` by the reconciler. Deterministic failures
    /// (bad bytes, encoder panic, NaN score) go straight to `Poison` after the cap — retrying them
    /// only burns money (goal F).
    pub fn is_transient(self) -> bool {
        matches!(
            self,
            ErrorClass::Timeout
                | ErrorClass::Oom
                | ErrorClass::UploadFail
                | ErrorClass::WorkerLost
                | ErrorClass::SourceFetch
                | ErrorClass::DiskFull
                | ErrorClass::Unknown
        )
    }

    /// Strict parse of the snake_case wire/ledger string (`"oom"`, `"source_fetch"`, …).
    /// `None` for anything this binary doesn't know — callers decide whether unknown means
    /// "ignore the claim" (executor marker lines) or "degrade to [`ErrorClass::Unknown`]"
    /// (ledger rows written by a newer binary — see [`ErrorClass::parse_lossy`]).
    pub fn parse_strict(s: &str) -> Option<ErrorClass> {
        serde_json::from_value(serde_json::Value::String(s.to_string())).ok()
    }

    /// Forgiving parse for cross-version reads: an error-class string this binary doesn't know
    /// (written by a newer worker/executor) degrades to [`ErrorClass::Unknown`] — which is
    /// TRANSIENT, so the reconciler retries rather than poisons — instead of failing the whole
    /// ledger read. Classification is advisory metadata; refusing to load a million-row ledger
    /// over one unknown label is the wrong trade (rolling-upgrade hazard: a mixed-rev fleet
    /// shares snapshots).
    pub fn parse_lossy(s: &str) -> ErrorClass {
        Self::parse_strict(s).unwrap_or(ErrorClass::Unknown)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_vs_deterministic() {
        assert!(ErrorClass::Timeout.is_transient());
        assert!(ErrorClass::WorkerLost.is_transient());
        assert!(ErrorClass::SourceFetch.is_transient());
        assert!(ErrorClass::DiskFull.is_transient());
        assert!(!ErrorClass::DecodeError.is_transient());
        assert!(!ErrorClass::MetricNan.is_transient());
        assert!(!ErrorClass::EncoderPanic.is_transient());
    }

    #[test]
    fn error_class_parse_strict_and_lossy() {
        assert_eq!(ErrorClass::parse_strict("oom"), Some(ErrorClass::Oom));
        assert_eq!(
            ErrorClass::parse_strict("source_fetch"),
            Some(ErrorClass::SourceFetch)
        );
        assert_eq!(
            ErrorClass::parse_strict("disk_full"),
            Some(ErrorClass::DiskFull)
        );
        // Unknown string: strict refuses (caller ignores the claim)…
        assert_eq!(ErrorClass::parse_strict("gpu_meltdown_v2"), None);
        // …lossy degrades to the transient Unknown (cross-version ledger read).
        assert_eq!(
            ErrorClass::parse_lossy("gpu_meltdown_v2"),
            ErrorClass::Unknown
        );
        assert_eq!(ErrorClass::parse_lossy("disk_full"), ErrorClass::DiskFull);
    }

    #[test]
    fn terminality() {
        assert!(JobStatus::Done.is_terminal());
        assert!(JobStatus::Poison.is_terminal());
        assert!(!JobStatus::Failed.is_terminal()); // Failed may still be retried up to the cap
        assert!(!JobStatus::Pending.is_terminal());
    }

    #[test]
    fn status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&JobStatus::Poison).unwrap(),
            "\"poison\""
        );
        assert_eq!(
            serde_json::to_string(&ErrorClass::DecodeError).unwrap(),
            "\"decode_error\""
        );
    }
}
