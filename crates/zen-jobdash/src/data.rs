//! The state the dashboard renders, loaded from the columnar ledger + blob index + worker
//! heartbeats. Loads local Parquet/JSON today; the R2 adapter is a thin future addition (fetch the
//! same objects from R2 before reading). No worker logic here.

use std::path::Path;

use zen_job_core::{BlobIndexEntry, LedgerRow, WorkerReport};

#[derive(Clone, Debug, Default)]
pub struct DashData {
    pub rows: Vec<LedgerRow>,
    pub blobs: Vec<BlobIndexEntry>,
    pub workers: Vec<WorkerReport>,
}

impl DashData {
    pub fn new(rows: Vec<LedgerRow>, blobs: Vec<BlobIndexEntry>, workers: Vec<WorkerReport>) -> Self {
        Self { rows, blobs, workers }
    }

    /// Load from local files: one or more ledger sidecars, an optional blob-index parquet, and an
    /// optional `workers.json` array of [`WorkerReport`].
    pub fn from_local(
        ledger_paths: &[&Path],
        blob_index: Option<&Path>,
        workers_json: Option<&Path>,
    ) -> Result<Self, DashError> {
        let mut rows = Vec::new();
        for p in ledger_paths {
            rows.extend(zen_ledger::read_ledger(p)?);
        }
        let blobs = match blob_index {
            Some(p) => zen_ledger::read_blob_index(p)?,
            None => Vec::new(),
        };
        let workers = match workers_json {
            Some(p) => {
                let bytes = std::fs::read(p)
                    .map_err(|e| DashError::Io(format!("read {}: {e}", p.display())))?;
                serde_json::from_slice(&bytes).map_err(|e| DashError::Json(e.to_string()))?
            }
            None => Vec::new(),
        };
        Ok(Self { rows, blobs, workers })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DashError {
    #[error("ledger {0}")]
    Ledger(String),
    #[error("io {0}")]
    Io(String),
    #[error("json {0}")]
    Json(String),
}

impl From<zen_ledger::LedgerError> for DashError {
    fn from(e: zen_ledger::LedgerError) -> Self {
        DashError::Ledger(e.to_string())
    }
}
