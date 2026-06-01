#![forbid(unsafe_code)]
//! # zen-jobctl
//!
//! The enqueue + discovery surface, all over the local ledger (no accounts, fully testable):
//! - [`declare`] expands a high-level spec into desired jobs — goal A ("one call declares desired
//!   artifacts").
//! - [`coverage`] reports done/poison/gap per (codec, metric) from the ledger — goal I ("catalog
//!   derived from ledger, can't drift").
//! - [`gap`] returns only the not-yet-done jobs — idempotent enqueue / "enqueue only the gap" (goals
//!   A & I). Because identity is content-addressed, re-declaring already-done work yields an empty
//!   gap (a structural no-op).

use serde::{Deserialize, Serialize};

use zen_job_core::{
    CellId, DesiredJob, JobKind, JobStatus, LedgerView, RetryPolicy, Sha256Hex, reconcile,
};

fn empty_knobs() -> String {
    "{}".into()
}

/// One thing to score: the cell identity (image/codec/q/knobs) + the content hash of its encode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclareItem {
    pub image_path: String,
    pub codec: String,
    pub q: i64,
    #[serde(default = "empty_knobs")]
    pub knob_tuple_json: String,
    /// Content hash (sha256 hex) of the encode to score.
    pub encode_sha: String,
}

/// A high-level declaration: score these items with these metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclareSpec {
    pub items: Vec<DeclareItem>,
    pub metrics: Vec<String>,
}

/// Expand a declaration into desired metric jobs (one per item × metric). Goal A.
pub fn declare(spec: &DeclareSpec) -> Result<Vec<DesiredJob>, String> {
    let mut out = Vec::with_capacity(spec.items.len() * spec.metrics.len());
    for it in &spec.items {
        let sha = Sha256Hex::parse(it.encode_sha.clone())
            .map_err(|e| format!("item {}: {e}", it.image_path))?;
        for m in &spec.metrics {
            out.push(DesiredJob {
                kind: JobKind::Metric { metric: m.clone() },
                inputs: vec![sha.clone()],
                cell: CellId {
                    image_path: it.image_path.clone(),
                    codec: it.codec.clone(),
                    q: it.q,
                    knob_tuple_json: it.knob_tuple_json.clone(),
                },
            });
        }
    }
    Ok(out)
}

fn metric_label(kind: &JobKind) -> String {
    match kind {
        JobKind::Metric { metric } => metric.clone(),
        JobKind::Diffmap { metric } => format!("diffmap:{metric}"),
        JobKind::Feature { regime } => format!("feature:{regime}"),
        JobKind::Encode { .. } => "encode".into(),
        JobKind::Resample { .. } => "resample".into(),
        JobKind::Bake { .. } => "bake".into(),
    }
}

/// Coverage per (codec, metric): done / poison / still-a-gap, derived purely from the ledger
/// (goal I — same source the dashboard reads, so it can't drift).
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CoverageRow {
    pub codec: String,
    pub metric: String,
    pub total: usize,
    pub done: usize,
    pub poison: usize,
    pub gap: usize,
}

pub fn coverage(desired: &[DesiredJob], view: &LedgerView) -> Vec<CoverageRow> {
    use std::collections::BTreeMap;
    let mut m: BTreeMap<(String, String), CoverageRow> = BTreeMap::new();
    for d in desired {
        let codec = d.cell.codec.clone();
        let metric = metric_label(&d.kind);
        let row = m
            .entry((codec.clone(), metric.clone()))
            .or_insert(CoverageRow {
                codec,
                metric,
                total: 0,
                done: 0,
                poison: 0,
                gap: 0,
            });
        row.total += 1;
        match view.get(&d.job_id()).map(|r| r.status) {
            Some(JobStatus::Done) => row.done += 1,
            Some(JobStatus::Poison) => row.poison += 1,
            _ => row.gap += 1,
        }
    }
    m.into_values().collect()
}

/// The not-yet-done subset of `desired` — what an agent should actually enqueue. Excludes Done and
/// Poison; keeps never-seen + retryable. Re-declaring fully-done work returns an empty gap.
pub fn gap(desired: &[DesiredJob], view: &LedgerView, policy: RetryPolicy) -> Vec<DesiredJob> {
    use std::collections::HashSet;
    let plan = reconcile(desired, view, policy);
    let enq: HashSet<_> = plan.enqueue.into_iter().collect();
    desired
        .iter()
        .filter(|d| enq.contains(&d.job_id()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_job_core::{LedgerRow, sha256};

    fn spec() -> DeclareSpec {
        DeclareSpec {
            items: vec![
                DeclareItem {
                    image_path: "a.png".into(),
                    codec: "zenjpeg".into(),
                    q: 80,
                    knob_tuple_json: "{}".into(),
                    encode_sha: sha256(b"enc-a").as_str().into(),
                },
                DeclareItem {
                    image_path: "b.png".into(),
                    codec: "zenavif".into(),
                    q: 50,
                    knob_tuple_json: "{}".into(),
                    encode_sha: sha256(b"enc-b").as_str().into(),
                },
            ],
            metrics: vec!["cvvdp".into(), "ssim2".into()],
        }
    }

    #[test]
    fn declare_expands_items_times_metrics() {
        let d = declare(&spec()).unwrap();
        assert_eq!(d.len(), 4, "2 items × 2 metrics");
    }

    #[test]
    fn declare_rejects_bad_sha() {
        let mut s = spec();
        s.items[0].encode_sha = "not-a-valid-sha".into();
        assert!(declare(&s).is_err());
    }

    #[test]
    fn coverage_and_gap_reflect_ledger() {
        let d = declare(&spec()).unwrap();
        let done_id = d[0].job_id();
        let row = LedgerRow {
            job_id: done_id.clone(),
            kind: d[0].kind.clone(),
            cell: d[0].cell.clone(),
            output_sha: Some(sha256(b"score")),
            status: JobStatus::Done,
            error_class: None,
            attempts: 1,
            ts: 1,
            worker: "w".into(),
            provider: "local".into(),
        };
        let view = LedgerView::from_rows([row]);

        let cov = coverage(&d, &view);
        assert_eq!(cov.iter().map(|c| c.done).sum::<usize>(), 1);
        assert_eq!(cov.iter().map(|c| c.gap).sum::<usize>(), 3);

        let g = gap(&d, &view, RetryPolicy::default());
        assert_eq!(g.len(), 3, "the done job drops out of the gap");
        assert!(!g.iter().any(|j| j.job_id() == done_id));
    }

    #[test]
    fn re_declaring_done_work_is_empty_gap() {
        let d = declare(&spec()).unwrap();
        // mark ALL done
        let rows: Vec<LedgerRow> = d
            .iter()
            .map(|j| LedgerRow {
                job_id: j.job_id(),
                kind: j.kind.clone(),
                cell: j.cell.clone(),
                output_sha: Some(sha256(b"s")),
                status: JobStatus::Done,
                error_class: None,
                attempts: 1,
                ts: 1,
                worker: "w".into(),
                provider: "local".into(),
            })
            .collect();
        let view = LedgerView::from_rows(rows);
        assert!(
            gap(&d, &view, RetryPolicy::default()).is_empty(),
            "fully-done declaration → no-op"
        );
    }
}
