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
                | ErrorClass::Unknown
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_vs_deterministic() {
        assert!(ErrorClass::Timeout.is_transient());
        assert!(ErrorClass::WorkerLost.is_transient());
        assert!(!ErrorClass::DecodeError.is_transient());
        assert!(!ErrorClass::MetricNan.is_transient());
        assert!(!ErrorClass::EncoderPanic.is_transient());
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
        assert_eq!(serde_json::to_string(&JobStatus::Poison).unwrap(), "\"poison\"");
        assert_eq!(serde_json::to_string(&ErrorClass::DecodeError).unwrap(), "\"decode_error\"");
    }
}
