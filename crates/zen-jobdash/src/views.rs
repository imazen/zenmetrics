//! Monitoring views (goal B) — pure aggregations over the ledger / blob index / worker reports into
//! serde DTOs the dashboard renders. No I/O; fully testable offline.

use std::collections::BTreeMap;

use serde::Serialize;

use zen_job_core::{aggregate, cost_per_1000_by_tier, BlobIndexEntry, JobKind, JobStatus, LedgerRow, WorkerReport};

/// Short, stable label for a job kind, e.g. `metric:cvvdp`, `encode:zenjpeg`.
pub fn kind_label(k: &JobKind) -> String {
    match k {
        JobKind::Encode { codec, .. } => format!("encode:{codec}"),
        JobKind::Metric { metric } => format!("metric:{metric}"),
        JobKind::Feature { regime } => format!("feature:{regime}"),
        JobKind::Diffmap { metric } => format!("diffmap:{metric}"),
        JobKind::Resample { kernel, .. } => format!("resample:{kernel}"),
        JobKind::Bake { view } => format!("bake:{view}"),
    }
}

/// Progress per job kind: total + status breakdown (goal B).
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct KindProgress {
    pub kind: String,
    pub total: usize,
    pub done: usize,
    pub failed: usize,
    pub poison: usize,
    pub in_flight: usize,
}

pub fn progress(rows: &[LedgerRow]) -> Vec<KindProgress> {
    let mut m: BTreeMap<String, KindProgress> = BTreeMap::new();
    for r in rows {
        let label = kind_label(&r.kind);
        let e = m.entry(label.clone()).or_insert(KindProgress {
            kind: label,
            total: 0,
            done: 0,
            failed: 0,
            poison: 0,
            in_flight: 0,
        });
        e.total += 1;
        match r.status {
            JobStatus::Done => e.done += 1,
            JobStatus::Failed => e.failed += 1,
            JobStatus::Poison => e.poison += 1,
            JobStatus::Pending | JobStatus::Claimed => e.in_flight += 1,
        }
    }
    m.into_values().collect()
}

/// Failure drill-down by (error_class, codec, kind) — "exactly what failed" (goal B). Only rows that
/// carry an `error_class` (Failed/Poison) contribute.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct FailureCell {
    pub error_class: String,
    pub codec: String,
    pub kind: String,
    pub count: usize,
}

pub fn failures(rows: &[LedgerRow]) -> Vec<FailureCell> {
    let mut m: BTreeMap<(String, String, String), usize> = BTreeMap::new();
    for r in rows {
        if let Some(ec) = r.error_class {
            let key = (format!("{ec:?}"), r.cell.codec.clone(), kind_label(&r.kind));
            *m.entry(key).or_insert(0) += 1;
        }
    }
    m.into_iter()
        .map(|((error_class, codec, kind), count)| FailureCell { error_class, codec, kind, count })
        .collect()
}

/// Cost view (goal B): fleet totals + cost-per-1000-jobs per tier (the measured cheapest-tier number).
#[derive(Serialize, Debug)]
pub struct TierCost {
    pub tier: String,
    pub cost_per_1000_jobs: Option<f64>,
}

#[derive(Serialize, Debug)]
pub struct CostView {
    pub total_spent_usd: f64,
    pub burn_usd_per_hr: f64,
    pub jobs_done: u64,
    pub cost_per_1000_jobs: Option<f64>,
    pub per_tier: Vec<TierCost>,
}

pub fn cost_view(workers: &[WorkerReport]) -> CostView {
    let fleet = aggregate(workers);
    let per_tier = cost_per_1000_by_tier(workers)
        .into_iter()
        .map(|(t, c)| TierCost { tier: format!("{t:?}"), cost_per_1000_jobs: c })
        .collect();
    CostView {
        total_spent_usd: fleet.total_spent_usd,
        burn_usd_per_hr: fleet.burn_usd_per_hr,
        jobs_done: fleet.jobs_done,
        cost_per_1000_jobs: fleet.cost_per_1000_jobs(),
        per_tier,
    }
}

/// Storage per regenerability tier (goal B: storage $/mo proxy = bytes per tier).
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct TierStorage {
    pub tier: String,
    pub blobs: usize,
    pub bytes: u64,
}

pub fn storage(blobs: &[BlobIndexEntry]) -> Vec<TierStorage> {
    let mut m: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for b in blobs {
        let e = m.entry(format!("{:?}", b.regenerability)).or_insert((0, 0));
        e.0 += 1;
        e.1 += b.size;
    }
    m.into_iter().map(|(tier, (n, bytes))| TierStorage { tier, blobs: n, bytes }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_job_core::{sha256, CellId, ErrorClass, JobId, Regenerability, ResourceClass};

    fn row(kind: JobKind, codec: &str, status: JobStatus, err: Option<ErrorClass>) -> LedgerRow {
        let input = sha256(codec.as_bytes());
        LedgerRow {
            job_id: JobId::of(&kind, std::slice::from_ref(&input)),
            kind,
            cell: CellId { image_path: "x".into(), codec: codec.into(), q: 80, knob_tuple_json: "{}".into() },
            output_sha: None,
            status,
            error_class: err,
            attempts: 1,
            ts: 1,
            worker: "w".into(),
            provider: "p".into(),
        }
    }

    #[test]
    fn progress_groups_by_kind() {
        let rows = vec![
            row(JobKind::Metric { metric: "cvvdp".into() }, "zenjpeg", JobStatus::Done, None),
            row(JobKind::Metric { metric: "cvvdp".into() }, "zenavif", JobStatus::Failed, Some(ErrorClass::Timeout)),
            row(JobKind::Encode { codec: "zenjpeg".into(), q: 80, knobs: "{}".into() }, "zenjpeg", JobStatus::Done, None),
        ];
        let p = progress(&rows);
        let cvvdp = p.iter().find(|k| k.kind == "metric:cvvdp").unwrap();
        assert_eq!(cvvdp.total, 2);
        assert_eq!(cvvdp.done, 1);
        assert_eq!(cvvdp.failed, 1);
        assert!(p.iter().any(|k| k.kind == "encode:zenjpeg"));
    }

    #[test]
    fn failures_drill_down() {
        let rows = vec![
            row(JobKind::Metric { metric: "cvvdp".into() }, "zenavif", JobStatus::Failed, Some(ErrorClass::Timeout)),
            row(JobKind::Metric { metric: "cvvdp".into() }, "zenavif", JobStatus::Failed, Some(ErrorClass::Timeout)),
            row(JobKind::Metric { metric: "ssim2".into() }, "zenjpeg", JobStatus::Done, None),
        ];
        let f = failures(&rows);
        assert_eq!(f.len(), 1, "only the two matching failures aggregate; the Done row has no error");
        assert_eq!(f[0].count, 2);
        assert_eq!(f[0].error_class, "Timeout");
        assert_eq!(f[0].codec, "zenavif");
    }

    #[test]
    fn cost_view_has_per_tier() {
        let workers = vec![WorkerReport {
            worker: "g1".into(),
            provider: "vast".into(),
            class: ResourceClass::Gpu,
            rate_usd_per_hr: 0.50,
            uptime_secs: 3600,
            jobs_done: 100,
        }];
        let cv = cost_view(&workers);
        assert!((cv.total_spent_usd - 0.50).abs() < 1e-9);
        let gpu = cv.per_tier.iter().find(|t| t.tier == "Gpu").unwrap();
        assert!((gpu.cost_per_1000_jobs.unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn storage_per_tier() {
        let blobs = vec![
            BlobIndexEntry { sha: sha256(b"a"), size: 100, regenerability: Regenerability::CheapRegenerable, last_ref_secs: 0 },
            BlobIndexEntry { sha: sha256(b"b"), size: 200, regenerability: Regenerability::CheapRegenerable, last_ref_secs: 0 },
            BlobIndexEntry { sha: sha256(b"c"), size: 9000, regenerability: Regenerability::NotRegenerable, last_ref_secs: 0 },
        ];
        let s = storage(&blobs);
        let cheap = s.iter().find(|t| t.tier == "CheapRegenerable").unwrap();
        assert_eq!(cheap.blobs, 2);
        assert_eq!(cheap.bytes, 300);
    }
}
