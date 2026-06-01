//! Monitoring views (goal B) — pure aggregations over the ledger / blob index / worker reports into
//! serde DTOs the dashboard renders. No I/O; fully testable offline.

use std::collections::{BTreeMap, HashSet};

use serde::Serialize;

use zen_job_core::{
    BlobIndexEntry, JobKind, JobStatus, LedgerRow, SemanticId, WorkerReport, aggregate,
    cost_per_1000_by_tier,
};

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
        .map(|((error_class, codec, kind), count)| FailureCell {
            error_class,
            codec,
            kind,
            count,
        })
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
        .map(|(t, c)| TierCost {
            tier: format!("{t:?}"),
            cost_per_1000_jobs: c,
        })
        .collect();
    CostView {
        total_spent_usd: fleet.total_spent_usd,
        burn_usd_per_hr: fleet.burn_usd_per_hr,
        jobs_done: fleet.jobs_done,
        cost_per_1000_jobs: fleet.cost_per_1000_jobs(),
        per_tier,
    }
}

/// Run summary (goal B: "progress … + ETA" and "cost … projected total"). ETA is estimated from the
/// remaining (not-done, not-poison) job count divided by the *current* fleet throughput (sum of
/// per-worker jobs/min from heartbeats) — a snapshot estimate, null when nothing is running. The
/// projected total adds the burn over the ETA window to what's already spent.
#[derive(Serialize, Debug)]
pub struct RunSummary {
    pub total: usize,
    pub done: usize,
    pub remaining: usize,
    pub poison: usize,
    pub fleet_jobs_per_min: f64,
    pub eta_secs: Option<u64>,
    pub spent_usd: f64,
    pub burn_usd_per_hr: f64,
    pub projected_total_usd: Option<f64>,
}

pub fn run_summary(rows: &[LedgerRow], workers: &[WorkerReport]) -> RunSummary {
    let mut total = 0usize;
    let mut done = 0usize;
    let mut poison = 0usize;
    for r in rows {
        total += 1;
        match r.status {
            JobStatus::Done => done += 1,
            JobStatus::Poison => poison += 1,
            _ => {}
        }
    }
    // Remaining = work that can still complete (excludes done and poison).
    let remaining = total.saturating_sub(done).saturating_sub(poison);
    let fleet_jpm: f64 = workers_view(workers).iter().map(|w| w.jobs_per_min).sum();
    let eta_secs = if fleet_jpm > 0.0 && remaining > 0 {
        Some((remaining as f64 / (fleet_jpm / 60.0)).ceil() as u64)
    } else {
        None
    };
    let fleet = aggregate(workers);
    let projected_total_usd =
        eta_secs.map(|s| fleet.total_spent_usd + fleet.burn_usd_per_hr * (s as f64 / 3600.0));
    RunSummary {
        total,
        done,
        remaining,
        poison,
        fleet_jobs_per_min: fleet_jpm,
        eta_secs,
        spent_usd: fleet.total_spent_usd,
        burn_usd_per_hr: fleet.burn_usd_per_hr,
        projected_total_usd,
    }
}

/// One catalog/coverage row (goal I: "canonical queryable catalog by semantic identity"). Derived
/// from the ledger — a result *set* grouped by (codec, kind, config), with its content-addressed
/// [`SemanticId`] key, distinct-image count (corpus-size proxy), q range, and done/total coverage.
/// This is the dashboard side of "find-by-description, consult coverage before enqueuing".
#[derive(Serialize, Debug, PartialEq)]
pub struct CatalogRow {
    /// Content-addressed key of the semantic description (same description → same key).
    pub key: String,
    pub codec: String,
    pub kind: String,
    pub metric: String,
    /// Config descriptor (the knob tuple).
    pub config: String,
    /// Distinct images covered (corpus-size proxy — the ledger names images, not a corpus).
    pub images: usize,
    pub q_min: i64,
    pub q_max: i64,
    pub total: usize,
    pub done: usize,
}

pub fn catalog_view(rows: &[LedgerRow]) -> Vec<CatalogRow> {
    struct Agg {
        metric: String,
        images: HashSet<String>,
        q_min: i64,
        q_max: i64,
        total: usize,
        done: usize,
    }
    let mut m: BTreeMap<(String, String, String), Agg> = BTreeMap::new();
    for r in rows {
        let codec = r.cell.codec.clone();
        let kind = kind_label(&r.kind);
        let config = r.cell.knob_tuple_json.clone();
        let metric = match &r.kind {
            JobKind::Metric { metric } => metric.clone(),
            _ => String::new(),
        };
        let e = m.entry((codec, kind, config)).or_insert_with(|| Agg {
            metric: metric.clone(),
            images: HashSet::new(),
            q_min: r.cell.q,
            q_max: r.cell.q,
            total: 0,
            done: 0,
        });
        e.images.insert(r.cell.image_path.clone());
        e.q_min = e.q_min.min(r.cell.q);
        e.q_max = e.q_max.max(r.cell.q);
        e.total += 1;
        if r.status == JobStatus::Done {
            e.done += 1;
        }
    }
    m.into_iter()
        .map(|((codec, kind, config), a)| {
            let sid = SemanticId {
                corpus: "(ledger-derived)".to_string(),
                codec: codec.clone(),
                q_grid: format!("{}..={}", a.q_min, a.q_max),
                metric: a.metric.clone(),
                config: config.clone(),
            };
            CatalogRow {
                key: sid.key().to_string(),
                codec,
                kind,
                metric: a.metric,
                config,
                images: a.images.len(),
                q_min: a.q_min,
                q_max: a.q_max,
                total: a.total,
                done: a.done,
            }
        })
        .collect()
}

/// A single ledger row, flattened for the ad-hoc query view (goal B: "ad-hoc parquet query"). The
/// ledger *is* the Parquet table; this is a structured filter over it (kind/codec/status/image),
/// the common slice without shipping a SQL engine.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct QueryRow {
    pub kind: String,
    pub codec: String,
    pub image_path: String,
    pub q: i64,
    pub status: String,
    pub error_class: Option<String>,
    pub output_sha: Option<String>,
    pub worker: String,
    pub ts: u64,
}

/// Filter the ledger by optional kind/codec/status substrings + image substring, newest-first, capped.
/// Empty filters match everything. `status` matches the `JobStatus` debug name case-insensitively.
pub fn query_view(
    rows: &[LedgerRow],
    kind: Option<&str>,
    codec: Option<&str>,
    status: Option<&str>,
    image: Option<&str>,
    limit: usize,
) -> Vec<QueryRow> {
    let ci = |hay: &str, needle: Option<&str>| {
        needle.is_none_or(|n| hay.to_lowercase().contains(&n.to_lowercase()))
    };
    let mut out: Vec<&LedgerRow> = rows
        .iter()
        .filter(|r| {
            ci(&kind_label(&r.kind), kind)
                && ci(&r.cell.codec, codec)
                && ci(&format!("{:?}", r.status), status)
                && ci(&r.cell.image_path, image)
        })
        .collect();
    out.sort_by(|a, b| b.ts.cmp(&a.ts));
    out.into_iter()
        .take(limit)
        .map(|r| QueryRow {
            kind: kind_label(&r.kind),
            codec: r.cell.codec.clone(),
            image_path: r.cell.image_path.clone(),
            q: r.cell.q,
            status: format!("{:?}", r.status),
            error_class: r.error_class.map(|e| format!("{e:?}")),
            output_sha: r.output_sha.as_ref().map(|s| s.to_string()),
            worker: r.worker.clone(),
            ts: r.ts,
        })
        .collect()
}

/// A completed result (goal B: "peek results in-browser"). Done rows that produced an output blob —
/// the dashboard lists these and fetches the score blob by `output_sha` on demand.
#[derive(Serialize, Debug, PartialEq, Eq)]
pub struct ResultRow {
    pub kind: String,
    pub codec: String,
    pub image_path: String,
    pub q: i64,
    pub output_sha: String,
    pub worker: String,
}

/// The most recent `limit` Done rows that carry an output blob, newest first.
pub fn results_view(rows: &[LedgerRow], limit: usize) -> Vec<ResultRow> {
    let mut done: Vec<&LedgerRow> = rows
        .iter()
        .filter(|r| r.status == JobStatus::Done && r.output_sha.is_some())
        .collect();
    done.sort_by(|a, b| b.ts.cmp(&a.ts));
    done.into_iter()
        .take(limit)
        .map(|r| ResultRow {
            kind: kind_label(&r.kind),
            codec: r.cell.codec.clone(),
            image_path: r.cell.image_path.clone(),
            q: r.cell.q,
            output_sha: r
                .output_sha
                .as_ref()
                .map(|s| s.to_string())
                .unwrap_or_default(),
            worker: r.worker.clone(),
        })
        .collect()
}

/// Per-worker live view (goal B: "live fleet per worker"). Derived from worker heartbeat reports —
/// provider, tier, rate, uptime, jobs, and throughput. `jobs_per_min` is `jobs_done / uptime`.
#[derive(Serialize, Debug, PartialEq)]
pub struct WorkerStat {
    pub worker: String,
    pub provider: String,
    pub tier: String,
    pub rate_usd_per_hr: f64,
    pub uptime_secs: u64,
    pub jobs_done: u64,
    pub jobs_per_min: f64,
    pub spent_usd: f64,
}

pub fn workers_view(workers: &[WorkerReport]) -> Vec<WorkerStat> {
    workers
        .iter()
        .map(|w| WorkerStat {
            worker: w.worker.clone(),
            provider: w.provider.clone(),
            tier: format!("{:?}", w.class),
            rate_usd_per_hr: w.rate_usd_per_hr,
            uptime_secs: w.uptime_secs,
            jobs_done: w.jobs_done,
            jobs_per_min: if w.uptime_secs > 0 {
                w.jobs_done as f64 / (w.uptime_secs as f64 / 60.0)
            } else {
                0.0
            },
            spent_usd: w.spent_usd(),
        })
        .collect()
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
    m.into_iter()
        .map(|(tier, (n, bytes))| TierStorage {
            tier,
            blobs: n,
            bytes,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zen_job_core::{CellId, ErrorClass, JobId, Regenerability, ResourceClass, sha256};

    fn row(kind: JobKind, codec: &str, status: JobStatus, err: Option<ErrorClass>) -> LedgerRow {
        let input = sha256(codec.as_bytes());
        LedgerRow {
            job_id: JobId::of(&kind, std::slice::from_ref(&input)),
            kind,
            cell: CellId {
                image_path: "x".into(),
                codec: codec.into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
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
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenavif",
                JobStatus::Failed,
                Some(ErrorClass::Timeout),
            ),
            row(
                JobKind::Encode {
                    codec: "zenjpeg".into(),
                    q: 80,
                    knobs: "{}".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
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
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenavif",
                JobStatus::Failed,
                Some(ErrorClass::Timeout),
            ),
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenavif",
                JobStatus::Failed,
                Some(ErrorClass::Timeout),
            ),
            row(
                JobKind::Metric {
                    metric: "ssim2".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
        ];
        let f = failures(&rows);
        assert_eq!(
            f.len(),
            1,
            "only the two matching failures aggregate; the Done row has no error"
        );
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
    fn run_summary_eta_and_projection() {
        let rows = vec![
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Pending,
                None,
            ),
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Pending,
                None,
            ),
        ];
        // one paid worker doing 60 jobs/min → 2 remaining = 2s ETA.
        let workers = vec![WorkerReport {
            worker: "w".into(),
            provider: "hetzner".into(),
            class: ResourceClass::CpuArm,
            rate_usd_per_hr: 3.6, // = 0.001/sec
            uptime_secs: 60,
            jobs_done: 60, // 60/min
        }];
        let s = run_summary(&rows, &workers);
        assert_eq!(s.remaining, 2);
        assert!((s.fleet_jobs_per_min - 60.0).abs() < 1e-9);
        assert_eq!(s.eta_secs, Some(2), "2 remaining / 60 per min = 2s");
        // projected = spent (3.6*60/3600=0.06) + burn(3.6/hr)*2s = 0.06 + 0.002 = 0.062
        let proj = s.projected_total_usd.unwrap();
        assert!((proj - 0.062).abs() < 1e-6, "got {proj}");
    }

    #[test]
    fn run_summary_no_fleet_no_eta() {
        let rows = vec![row(
            JobKind::Metric { metric: "x".into() },
            "c",
            JobStatus::Pending,
            None,
        )];
        let s = run_summary(&rows, &[]);
        assert_eq!(s.eta_secs, None, "nothing running → no ETA");
        assert_eq!(s.projected_total_usd, None);
    }

    #[test]
    fn catalog_groups_by_semantic_identity() {
        let rows = vec![
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
            row(
                JobKind::Metric {
                    metric: "cvvdp".into(),
                },
                "zenjpeg",
                JobStatus::Pending,
                None,
            ),
            row(
                JobKind::Metric {
                    metric: "ssim2".into(),
                },
                "zenjpeg",
                JobStatus::Done,
                None,
            ),
        ];
        let c = catalog_view(&rows);
        // cvvdp×zenjpeg and ssim2×zenjpeg are distinct semantic identities → distinct keys.
        assert_eq!(c.len(), 2);
        let cvvdp = c.iter().find(|r| r.metric == "cvvdp").unwrap();
        assert_eq!(cvvdp.total, 2);
        assert_eq!(cvvdp.done, 1, "coverage = done/total");
        let ssim2 = c.iter().find(|r| r.metric == "ssim2").unwrap();
        assert_ne!(
            cvvdp.key, ssim2.key,
            "different description → different content-addressed key"
        );
    }

    #[test]
    fn query_view_filters_and_orders() {
        let mut done = row(
            JobKind::Metric {
                metric: "cvvdp".into(),
            },
            "zenjpeg",
            JobStatus::Done,
            None,
        );
        done.ts = 10;
        let mut failed = row(
            JobKind::Metric {
                metric: "cvvdp".into(),
            },
            "zenavif",
            JobStatus::Failed,
            Some(ErrorClass::Timeout),
        );
        failed.ts = 20;
        let rows = vec![done, failed];
        // filter by status=failed → only the avif one
        let f = query_view(&rows, None, None, Some("fail"), None, 100);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].codec, "zenavif");
        assert_eq!(f[0].error_class.as_deref(), Some("Timeout"));
        // no filter → both, newest (ts=20) first
        let all = query_view(&rows, None, None, None, None, 100);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].ts, 20);
        // codec filter
        assert_eq!(
            query_view(&rows, None, Some("jpeg"), None, None, 100).len(),
            1
        );
    }

    #[test]
    fn results_view_lists_done_with_output_newest_first() {
        let mut a = row(
            JobKind::Metric {
                metric: "cvvdp".into(),
            },
            "zenjpeg",
            JobStatus::Done,
            None,
        );
        a.output_sha = Some(sha256(b"score-a"));
        a.ts = 100;
        let mut b = row(
            JobKind::Metric {
                metric: "ssim2".into(),
            },
            "zenavif",
            JobStatus::Done,
            None,
        );
        b.output_sha = Some(sha256(b"score-b"));
        b.ts = 200;
        // a done row with no output blob, and a pending row — both excluded.
        let no_out = row(
            JobKind::Metric { metric: "x".into() },
            "c",
            JobStatus::Done,
            None,
        );
        let pending = row(
            JobKind::Metric { metric: "y".into() },
            "c",
            JobStatus::Pending,
            None,
        );
        let r = results_view(&[a, no_out, pending, b], 10);
        assert_eq!(r.len(), 2, "only done rows that produced an output blob");
        assert_eq!(r[0].codec, "zenavif", "newest (ts=200) first");
        assert_eq!(r[1].codec, "zenjpeg");
    }

    #[test]
    fn workers_view_computes_throughput() {
        let workers = vec![WorkerReport {
            worker: "arm-iter3-001".into(),
            provider: "hetzner".into(),
            class: ResourceClass::CpuArm,
            rate_usd_per_hr: 0.006,
            uptime_secs: 600, // 10 min
            jobs_done: 300,
        }];
        let w = workers_view(&workers);
        assert_eq!(w[0].tier, "CpuArm");
        assert!(
            (w[0].jobs_per_min - 30.0).abs() < 1e-9,
            "300 jobs / 10 min = 30/min"
        );
        assert!(
            (w[0].spent_usd - 0.001).abs() < 1e-6,
            "0.006/hr * (600/3600)"
        );
    }

    #[test]
    fn storage_per_tier() {
        let blobs = vec![
            BlobIndexEntry {
                sha: sha256(b"a"),
                size: 100,
                regenerability: Regenerability::CheapRegenerable,
                last_ref_secs: 0,
            },
            BlobIndexEntry {
                sha: sha256(b"b"),
                size: 200,
                regenerability: Regenerability::CheapRegenerable,
                last_ref_secs: 0,
            },
            BlobIndexEntry {
                sha: sha256(b"c"),
                size: 9000,
                regenerability: Regenerability::NotRegenerable,
                last_ref_secs: 0,
            },
        ];
        let s = storage(&blobs);
        let cheap = s.iter().find(|t| t.tier == "CheapRegenerable").unwrap();
        assert_eq!(cheap.blobs, 2);
        assert_eq!(cheap.bytes, 300);
    }
}
