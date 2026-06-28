#![forbid(unsafe_code)]
//! # zenfleet-ctl
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

use zenfleet_core::{
    CellId, DesiredJob, JobKind, JobStatus, LedgerView, ResourceHint, RetryPolicy, Sha256Hex,
    reconcile,
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

/// One encode to declare: the cell identity plus the content hash of the SOURCE image (the
/// encode job's input blob). This is the line format `zenmetrics sweep --plan … --dry-run
/// --emit-cells <path>` writes (JSON-lines, one item per line); the two sides are coupled by field
/// name only, mirroring the jobexec stdin contract's deliberate decoupling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeDeclareItem {
    pub image_path: String,
    pub codec: String,
    pub q: i64,
    pub knob_tuple_json: String,
    /// sha256 hex of the source image bytes (`blobs/<sha>`).
    pub source_sha: String,
    /// Optional scheduling hint (peak mem + useful threads) computed by the
    /// codec-linked emitter (`zenmetrics sweep --emit-cells` via
    /// `PlannedConfig::estimate_resources`). `#[serde(default)]` so manifests
    /// written before this field — or by a codec build without the estimate —
    /// parse as `None`. Propagated verbatim onto the [`DesiredJob`].
    #[serde(default)]
    pub hint: Option<ResourceHint>,
    /// Resolved-encode fingerprint (`zenjxl::sweep::encode_fingerprint`, 16-hex)
    /// — the COMPUTE-dedup key: cells with equal `encode_fp` produce
    /// byte-identical output (verified byte-safe; see
    /// `zenmetrics-cli/examples/encode_fp_byte_safety.rs`). When present,
    /// [`declare_encodes`] declares ONE encode job per `(codec, source_sha,
    /// encode_fp)` group (compute once); the per-knobset identity rows are NOT
    /// collapsed — they are preserved by the score-side fan-out, which MUST
    /// carry `encode_fp → encode_sha` so every cell rejoins the shared blob
    /// (the row-preservation requirement — `scripts/jobsys/build_score_spec.py`
    /// skips cells whose encode is unindexed, so a deduped sweep without that
    /// fan-out fix would DROP the non-representative rows). `#[serde(default)]`:
    /// absent (the default) ⇒ NO dedup, one encode per knobset (the row-safe
    /// baseline). The emitter populates it only once encode-dedup is activated
    /// (after the fan-out carries `encode_fp` AND the end-to-end row-count test
    /// confirms N rows out == N rows in).
    #[serde(default)]
    pub encode_fp: Option<String>,
}

/// Expand encode declarations into desired encode jobs. Plan-cell identity
/// (`{"cell":…,"fp":…,"plan":…}`) rides into `JobKind::Encode.knobs`, so the JobId is
/// content-addressed over the cell — re-declaring the same plan is a structural no-op and [`gap`]
/// returns exactly the unfinished cells. The executor side resolves the id back to a config and
/// verifies the fingerprint (`zenmetrics jobexec`), so a stored item is runnable years later with
/// no plan spec in hand.
pub fn declare_encodes(items: &[EncodeDeclareItem]) -> Result<Vec<DesiredJob>, String> {
    use std::collections::HashSet;
    let mut out = Vec::with_capacity(items.len());
    // Encode-COMPUTE dedup: items carrying an `encode_fp` (the resolved-encode
    // fingerprint — equal fp ⇒ byte-identical output) share ONE encode job per
    // `(codec, source_sha, encode_fp)` group, so the byte-identical encode runs
    // once instead of N times. Items WITHOUT an `encode_fp` (the default) are
    // never deduped — one encode job each (the row-safe baseline). This
    // collapses only the ENCODE COMPUTE, never the per-knobset rows: the omni
    // keeps every input knobset, and the score-side fan-out
    // (`build_score_spec.py` / `writeback_scores.py`) must rejoin every cell to
    // the shared blob via `encode_fp → encode_sha` (the row-preservation
    // requirement enforced by the end-to-end row-count test).
    let mut seen_encode: HashSet<(String, String, String)> = HashSet::new();
    for it in items {
        let sha = Sha256Hex::parse(it.source_sha.clone())
            .map_err(|e| format!("item {}: {e}", it.image_path))?;
        if let Some(fp) = &it.encode_fp {
            // Subsequent members of an (codec, source, encode_fp) group reuse
            // the first member's encode job (its content-addressed blob).
            if !seen_encode.insert((it.codec.clone(), it.source_sha.clone(), fp.clone())) {
                continue;
            }
        }
        out.push(DesiredJob {
            kind: JobKind::Encode {
                codec: it.codec.clone(),
                q: it.q,
                knobs: it.knob_tuple_json.clone(),
            },
            inputs: vec![sha],
            cell: CellId {
                image_path: it.image_path.clone(),
                codec: it.codec.clone(),
                q: it.q,
                knob_tuple_json: it.knob_tuple_json.clone(),
            },
            // Resource hint rides through from the emit-cells item (computed by
            // the codec-linked emitter via PlannedConfig::estimate_resources);
            // zenfleet-ctl stays codec-free and just propagates it. `None` when
            // the emitter couldn't estimate.
            hint: it.hint,
        });
    }
    Ok(out)
}

/// Parse a `--emit-cells` manifest (JSON-lines of [`EncodeDeclareItem`]).
pub fn parse_emit_cells(text: &str) -> Result<Vec<EncodeDeclareItem>, String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("emit-cells line {}: {e}", i + 1))
        })
        .collect()
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
                // Metric jobs are GPU-routed; the per-encode RAM/thread hint is
                // an encoder concept, so metric declares carry none.
                hint: None,
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
        JobKind::ScoreFile { metrics } => format!("scorefile:{}", metrics.join("+")),
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
    #[test]
    fn declare_encodes_is_idempotent_and_carries_plan_identity() {
        use super::*;
        let sha = "a".repeat(64);
        let items = vec![EncodeDeclareItem {
            image_path: "corpus/x.png".into(),
            codec: "zenjpeg".into(),
            q: 85,
            knob_tuple_json:
                r#"{"cell":"jp3_t0_small_420","fp":"0123456789abcdef","plan":"rd_core"}"#.into(),
            source_sha: sha.clone(),
            hint: None,
            encode_fp: None,
        }];
        let a = declare_encodes(&items).unwrap();
        let b = declare_encodes(&items).unwrap();
        assert_eq!(a.len(), 1);
        // Same declaration twice -> same content-addressed JobId (gap is a no-op).
        assert_eq!(a[0].job_id(), b[0].job_id());
        match &a[0].kind {
            zenfleet_core::JobKind::Encode { codec, q, knobs } => {
                assert_eq!(codec, "zenjpeg");
                assert_eq!(*q, 85);
                assert!(knobs.contains("rd_core"));
            }
            other => panic!("expected Encode kind, got {other:?}"),
        }
        // Bad sha rejected.
        let mut bad = items.clone();
        bad[0].source_sha = "nope".into();
        assert!(declare_encodes(&bad).is_err());
    }

    #[test]
    fn encode_dedup_collapses_compute_but_preserves_every_row() {
        use super::*;
        use std::collections::HashMap;
        let src = "a".repeat(64);
        let mk = |knob: &str, fp: Option<&str>| EncodeDeclareItem {
            image_path: "corpus/x.png".into(),
            codec: "zenjxl".into(),
            q: 50,
            knob_tuple_json: knob.into(),
            source_sha: src.clone(),
            hint: None,
            encode_fp: fp.map(str::to_string),
        };
        // 5 input knobsets: A,B,E share encode_fp "f1" (byte-identical);
        // C is "f2"; D carries NO encode_fp (un-deduped baseline).
        let items = vec![
            mk(r#"{"cell":"A"}"#, Some("f1")),
            mk(r#"{"cell":"B"}"#, Some("f1")),
            mk(r#"{"cell":"C"}"#, Some("f2")),
            mk(r#"{"cell":"D"}"#, None),
            mk(r#"{"cell":"E"}"#, Some("f1")),
        ];
        let jobs = declare_encodes(&items).unwrap();
        // COMPUTE dedup: f1 → 1 job, f2 → 1, D (no fp) → 1 = 3 encode jobs
        // (B and E reuse A's job — 2 encodes saved).
        assert_eq!(jobs.len(), 3, "encode jobs must dedup the f1 group to one");

        // ROW PRESERVATION (the N→N gate at the declare/fan-out logic level):
        // each declared encode job yields a content-addressed blob sha; build
        // the (codec, source, encode_fp) → sha map the score-side fan-out must
        // carry, then assert EVERY input knobset resolves to a sha (no cell is
        // dropped — N rows out == N rows in).
        let mut group_sha: HashMap<(String, String, String), String> = HashMap::new();
        for (i, job) in jobs.iter().enumerate() {
            if let JobKind::Encode { codec, q, knobs } = &job.kind {
                // recover the representative item this job came from
                if let Some(rep) = items
                    .iter()
                    .find(|it| it.knob_tuple_json == *knobs && it.codec == *codec && it.q == *q)
                {
                    if let Some(fp) = &rep.encode_fp {
                        group_sha
                            .insert((codec.clone(), rep.source_sha.clone(), fp.clone()), format!("blob{i}"));
                    }
                }
            }
        }
        let mut rows_out = 0usize;
        for it in &items {
            let resolvable = match &it.encode_fp {
                // deduped cell: must rejoin its group's shared blob
                Some(fp) => {
                    group_sha.contains_key(&(it.codec.clone(), it.source_sha.clone(), fp.clone()))
                }
                // no-fp cell: its own encode job → its own blob (always resolvable)
                None => true,
            };
            if resolvable {
                rows_out += 1;
            }
        }
        assert_eq!(
            rows_out,
            items.len(),
            "every input knobset must map to a blob sha — N rows out must equal N rows in"
        );
    }

    #[test]
    fn declare_encodes_propagates_resource_hint_and_survives_jsonl_roundtrip() {
        use super::*;
        let hint = ResourceHint {
            peak_mem_bytes: 8 << 30,
            threads: 4,
        };
        let item = EncodeDeclareItem {
            image_path: "corpus/x.png".into(),
            codec: "zenjxl".into(),
            q: 90,
            knob_tuple_json: r#"{"cell":"c","fp":"f","plan":"rd_core"}"#.into(),
            source_sha: "a".repeat(64),
            hint: Some(hint),
            encode_fp: None,
        };
        // Emit-cells writes JSON lines; parse_emit_cells reads them back. The
        // hint must survive that round-trip and land on the DesiredJob.
        let line = serde_json::to_string(&item).unwrap();
        let parsed = parse_emit_cells(&line).unwrap();
        let jobs = declare_encodes(&parsed).unwrap();
        assert_eq!(jobs[0].hint, Some(hint));
        // A legacy emit-cells line (no `hint` key) declares with hint = None.
        let legacy = serde_json::json!({
            "image_path": "x",
            "codec": "zenjpeg",
            "q": 80,
            "knob_tuple_json": "{}",
            "source_sha": "b".repeat(64),
        })
        .to_string();
        let jobs = declare_encodes(&parse_emit_cells(&legacy).unwrap()).unwrap();
        assert_eq!(jobs[0].hint, None);
    }

    #[test]
    fn emit_cells_manifest_parses() {
        use super::*;
        let line = format!(
            r#"{{"image_path":"a.png","codec":"zenjpeg","q":50,"knob_tuple_json":"{{}}","source_sha":"{}"}}"#,
            "b".repeat(64)
        );
        let text = format!("{line}\n\n{line}\n");
        let items = parse_emit_cells(&text).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].q, 50);
        assert!(parse_emit_cells("not json").is_err());
    }

    use super::*;
    use zenfleet_core::{LedgerRow, sha256};

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
