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
    CellId, DesiredJob, JobKind, JobStatus, LedgerRow, LedgerView, ResourceHint, RetryPolicy,
    Sha256Hex, reconcile,
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
    /// HDR encode cell (the HDR-corpus B1 extension): rides onto
    /// [`JobKind::Encode::hdr`]. `#[serde(default)]` so every pre-HDR
    /// emit-cells manifest parses unchanged (SDR).
    #[serde(default)]
    pub hdr: bool,
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
            requires: vec![],
            kind: JobKind::Encode {
                codec: it.codec.clone(),
                q: it.q,
                knobs: it.knob_tuple_json.clone(),
                hdr: it.hdr,
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
    // Anti-wedge invariant 5: stamp kind-derived executor capability requirements
    // so a stale-image worker self-excludes instead of grinding failures.
    for d in &mut out {
        d.requires = d.kind.required_capabilities();
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
                requires: vec![],
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
    // Anti-wedge invariant 5: stamp kind-derived executor capability requirements
    // so a stale-image worker self-excludes instead of grinding failures.
    for d in &mut out {
        d.requires = d.kind.required_capabilities();
    }
    Ok(out)
}

/// Expand a declaration into desired DIFFMAP jobs (one per item × metric) —
/// the HDR-corpus B2 declare: per-pixel map persistence over the SAME
/// (cell, encode_sha) items a [`declare`] scoring pass uses, so a diffmap
/// wave is a re-declare of the score spec with the map-owning metric names
/// (`butteraugli`, `cvvdp` — the executor rejects names with no in-tree
/// per-pixel map). `hdr` rides onto [`JobKind::Diffmap::hdr`] for every
/// job in the spec (a corpus is scored in one mode, never mixed).
pub fn declare_diffmaps(spec: &DeclareSpec, hdr: bool) -> Result<Vec<DesiredJob>, String> {
    let mut out = Vec::with_capacity(spec.items.len() * spec.metrics.len());
    for it in &spec.items {
        let sha = Sha256Hex::parse(it.encode_sha.clone())
            .map_err(|e| format!("item {}: {e}", it.image_path))?;
        for m in &spec.metrics {
            out.push(DesiredJob {
                requires: vec![],
                kind: JobKind::Diffmap {
                    metric: m.clone(),
                    hdr,
                },
                inputs: vec![sha.clone()],
                cell: CellId {
                    image_path: it.image_path.clone(),
                    codec: it.codec.clone(),
                    q: it.q,
                    knob_tuple_json: it.knob_tuple_json.clone(),
                },
                // Diffmap jobs are metric-class work; no encoder hint.
                hint: None,
            });
        }
    }
    // Anti-wedge invariant 5: stamp kind-derived executor capability requirements
    // so a stale-image worker self-excludes instead of grinding failures.
    for d in &mut out {
        d.requires = d.kind.required_capabilities();
    }
    Ok(out)
}

fn metric_label(kind: &JobKind) -> String {
    match kind {
        JobKind::Metric { metric } => metric.clone(),
        JobKind::Diffmap { metric, .. } => format!("diffmap:{metric}"),
        JobKind::Feature { regime } => format!("feature:{regime}"),
        JobKind::Encode { .. } => "encode".into(),
        JobKind::Resample { .. } => "resample".into(),
        JobKind::Bake { .. } => "bake".into(),
        JobKind::ScoreFile { metrics, .. } => format!("scorefile:{}", metrics.join("+")),
    }
}

/// Per-run reconcile accounting (the `jobctl report` core; migrated 2026-08-27
/// from `scripts/jobsys/pool_reconcile_report.py` — Python read parquet ledgers
/// and re-derived done/failed sets, duplicating [`LedgerView`] semantics).
/// `distinct_done` = distinct job_ids with a done row; `failed_only` = distinct
/// job_ids with a failed row and NO done row; `gap` = declared − distinct_done
/// (can go negative on re-declared runs — reported as 0-floored in totals by
/// the caller, matching the Python).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RunAccounting {
    pub declared: usize,
    /// EVER-done: distinct job_ids with a done row anywhere in history.
    /// Historical/accounting semantics — a later Failed flip (audit-blobs
    /// --scan-errors) does NOT reduce this. Use `live_done` for queue truth.
    pub distinct_done: usize,
    pub failed_only: usize,
    /// declared − EVER-done. Kept for continuity; can hide re-opened work.
    pub gap: i64,
    /// LATEST-WINS done: job_ids whose newest row is Done — the same
    /// resolution reconcile/workers see. An audit flip reduces this.
    pub live_done: usize,
    /// declared − live_done: the queue-truth gap. `--auto-pause` and any
    /// "is this run finished" decision MUST use this, never `gap` (the
    /// 2026-08-27 hdrgrid lesson: 7,449 error-carrying done cells were
    /// invisible to ever-done gap and auto-pause froze their re-drain).
    pub gap_live: i64,
    pub raw_rows: usize,
}

/// Account raw ledger rows against a declared count. Pure; the bin owns I/O.
pub fn account_rows(declared: usize, rows: &[LedgerRow]) -> RunAccounting {
    use std::collections::HashSet;
    let mut done: HashSet<&str> = HashSet::new();
    let mut failed: HashSet<&str> = HashSet::new();
    for r in rows {
        match r.status {
            JobStatus::Done => {
                done.insert(r.job_id.as_str());
            }
            JobStatus::Failed => {
                failed.insert(r.job_id.as_str());
            }
            _ => {}
        }
    }
    let failed_only = failed.iter().filter(|j| !done.contains(**j)).count();
    // Latest-wins resolution — the queue truth the workers see.
    let mut view = LedgerView::new();
    for r in rows {
        view.apply(r.clone());
    }
    let live_done = view.rows().filter(|r| r.status == JobStatus::Done).count();
    RunAccounting {
        declared,
        distinct_done: done.len(),
        failed_only,
        gap: declared as i64 - done.len() as i64,
        live_done,
        gap_live: declared as i64 - live_done as i64,
        raw_rows: rows.len(),
    }
}

#[cfg(test)]
mod accounting_flip_tests {
    use super::*;
    use zenfleet_core::{CellId, JobId, JobKind};

    fn row(id: &str, status: JobStatus, ts: u64) -> LedgerRow {
        LedgerRow {
            job_id: JobId::of(&JobKind::Metric { metric: id.into() }, &[]),
            kind: JobKind::Metric {
                metric: "cvvdp".into(),
            },
            cell: CellId {
                image_path: "x.png".into(),
                codec: "test".into(),
                q: 1,
                knob_tuple_json: "{}".into(),
            },
            output_sha: None,
            status,
            error_class: None,
            attempts: 0,
            ts,
            worker: String::new(),
            provider: String::new(),
        }
    }

    /// Snapshot counterpart of the flip class: the snapshot must carry the
    /// flip's failed row and DROP the stale done row, so a worker booting
    /// from the snapshot alone sees the re-opened job.
    #[test]
    fn snapshot_surfaces_flip_over_stale_done() {
        let rows = vec![
            row("a", JobStatus::Done, 100),
            row("a", JobStatus::Failed, 200), // audit flip
            row("b", JobStatus::Done, 100),
        ];
        let (snap, n_done, n_failed) = snapshot_rows(rows);
        assert_eq!((n_done, n_failed), (1, 1));
        let a_rows: Vec<_> = snap.iter().filter(|r| r.ts == 200).collect();
        assert_eq!(a_rows.len(), 1, "flip's failed row present");
        assert!(
            !snap.iter().any(|r| r.ts == 100
                && r.status == JobStatus::Done
                && snap.iter().any(|x| x.ts == 200 && x.job_id == r.job_id)),
            "stale done row for the flipped job must be dropped"
        );
    }

    /// The 2026-08-27 hdrgrid class: done at t1, audit-flipped Failed at t2.
    /// Ever-done stays 1 (accounting continuity); live_done drops to 0 and
    /// gap_live re-opens — the number auto-pause must consult.
    #[test]
    fn audit_flip_reopens_live_gap_but_not_ever_done() {
        let rows = vec![
            row("a", JobStatus::Done, 100),
            row("a", JobStatus::Failed, 200),
            row("b", JobStatus::Done, 100),
        ];
        let acc = account_rows(2, &rows);
        assert_eq!(acc.distinct_done, 2, "ever-done keeps the flipped job");
        assert_eq!(acc.gap, 0);
        assert_eq!(acc.live_done, 1, "latest-wins sees the flip");
        assert_eq!(acc.gap_live, 1, "queue truth re-opens");
    }
}

/// Build a snapshot row set from raw ledger rows (the `jobctl compact` core;
/// migrated 2026-08-27 from `scripts/jobsys/compact_ledgers.py`): every
/// status==done row FIRST-WINS per job_id, PLUS — anti-wedge invariant 4 —
/// the NEWEST (max ts) failed row for every job_id with no done row, so the
/// worker's `--ledger-in` view preserves attempt history and the
/// retry→Poison ladder actually fires. Returns (snapshot, n_done, n_failed).
pub fn snapshot_rows(rows: Vec<LedgerRow>) -> (Vec<LedgerRow>, usize, usize) {
    use std::collections::{HashMap, HashSet};
    // Queue-truth first (2026-08-27): resolve every job by LATEST-WINS before
    // deciding what the snapshot keeps. A job whose latest row is Failed —
    // including an audit-blobs flip OVER an older Done (error-carrying blob)
    // — must surface its newest failed row, NOT its stale done row; the old
    // "done first-wins unconditionally" shape silently buried flips and left
    // correctness riding on the sidecar fold window.
    let mut view = LedgerView::new();
    for r in &rows {
        view.apply(r.clone());
    }
    let mut latest_done: HashSet<&str> = HashSet::new();
    let mut latest_failed: HashSet<&str> = HashSet::new();
    for r in view.rows() {
        match r.status {
            JobStatus::Done => {
                latest_done.insert(r.job_id.as_str());
            }
            JobStatus::Failed | JobStatus::Poison => {
                latest_failed.insert(r.job_id.as_str());
            }
            _ => {}
        }
    }
    // done rows: FIRST-WINS per job, only for jobs still done under latest-wins
    let mut done_seen: HashSet<String> = HashSet::new();
    let mut snap: Vec<LedgerRow> = Vec::new();
    for r in &rows {
        if r.status == JobStatus::Done
            && latest_done.contains(r.job_id.as_str())
            && done_seen.insert(r.job_id.as_str().to_string())
        {
            snap.push(r.clone());
        }
    }
    let n_done = snap.len();
    // failed rows: NEWEST per job, for jobs failed/poison under latest-wins
    let mut best: HashMap<&str, &LedgerRow> = HashMap::new();
    for r in &rows {
        if matches!(r.status, JobStatus::Failed | JobStatus::Poison)
            && latest_failed.contains(r.job_id.as_str())
        {
            let e = best.entry(r.job_id.as_str()).or_insert(r);
            if r.ts > e.ts {
                *e = r;
            }
        }
    }
    let n_failed = best.len();
    let mut failed_rows: Vec<LedgerRow> = best.into_values().cloned().collect();
    failed_rows.sort_by(|a, b| a.job_id.as_str().cmp(b.job_id.as_str()));
    snap.extend(failed_rows);
    (snap, n_done, n_failed)
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
            hdr: false,
            hint: None,
            encode_fp: None,
        }];
        let a = declare_encodes(&items).unwrap();
        let b = declare_encodes(&items).unwrap();
        assert_eq!(a.len(), 1);
        // Same declaration twice -> same content-addressed JobId (gap is a no-op).
        assert_eq!(a[0].job_id(), b[0].job_id());
        match &a[0].kind {
            zenfleet_core::JobKind::Encode {
                codec,
                q,
                knobs,
                hdr,
            } => {
                assert_eq!(codec, "zenjpeg");
                assert_eq!(*q, 85);
                assert!(knobs.contains("rd_core"));
                assert!(!hdr, "emit-cells without an hdr field must declare SDR");
            }
            other => panic!("expected Encode kind, got {other:?}"),
        }
        // Bad sha rejected.
        let mut bad = items.clone();
        bad[0].source_sha = "nope".into();
        assert!(declare_encodes(&bad).is_err());
    }

    #[test]
    fn declare_diffmaps_expands_per_metric_and_hdr_changes_ids() {
        use super::*;
        let sha = "b".repeat(64);
        let spec = DeclareSpec {
            items: vec![DeclareItem {
                image_path: "hdr/x.hdr.png".into(),
                codec: "zenav1-svt".into(),
                q: 50,
                knob_tuple_json: r#"{"preset":6}"#.into(),
                encode_sha: sha,
            }],
            metrics: vec!["butteraugli".into(), "cvvdp".into()],
        };
        let hdr = declare_diffmaps(&spec, true).unwrap();
        assert_eq!(hdr.len(), 2, "one Diffmap job per (item x metric)");
        for j in &hdr {
            match &j.kind {
                zenfleet_core::JobKind::Diffmap { metric, hdr } => {
                    assert!(["butteraugli", "cvvdp"].contains(&metric.as_str()));
                    assert!(*hdr);
                }
                other => panic!("expected Diffmap kind, got {other:?}"),
            }
        }
        // hdr:true vs hdr:false must never dedup against each other.
        let sdr = declare_diffmaps(&spec, false).unwrap();
        assert_ne!(hdr[0].job_id(), sdr[0].job_id());
        // Idempotent (content-addressed): same spec twice -> same ids.
        let again = declare_diffmaps(&spec, true).unwrap();
        assert_eq!(hdr[0].job_id(), again[0].job_id());
        // SDR Diffmap serialization carries no `hdr` key (append-only schema).
        let json = serde_json::to_string(&sdr[0].kind).unwrap();
        assert_eq!(json, r#"{"kind":"diffmap","metric":"butteraugli"}"#);
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
            hdr: false,
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
            if let JobKind::Encode {
                codec, q, knobs, ..
            } = &job.kind
            {
                // recover the representative item this job came from
                if let Some(rep) = items
                    .iter()
                    .find(|it| it.knob_tuple_json == *knobs && it.codec == *codec && it.q == *q)
                {
                    if let Some(fp) = &rep.encode_fp {
                        group_sha.insert(
                            (codec.clone(), rep.source_sha.clone(), fp.clone()),
                            format!("blob{i}"),
                        );
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
            vram_bytes: None,
        };
        let item = EncodeDeclareItem {
            image_path: "corpus/x.png".into(),
            codec: "zenjxl".into(),
            q: 90,
            knob_tuple_json: r#"{"cell":"c","fp":"f","plan":"rd_core"}"#.into(),
            source_sha: "a".repeat(64),
            hdr: false,
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

// ─── the `pairs` projection: the Encode→ScoreFile bridge ─────────────────────

/// Join a prefix onto a ledger value, leaving an already-absolute URI verbatim.
///
/// Full-URI declares store ABSOLUTE `s3://` paths in `cell.image_path` — blindly
/// prefixing those double-prefixes them (hdrgrid diffmap endgame, 2026-08-27).
pub fn pairs_join_prefix(prefix: &str, value: &str) -> String {
    if value.starts_with("s3://") || prefix.is_empty() {
        value.to_string()
    } else {
        format!("{prefix}/{value}")
    }
}

/// The encode content hash of a DONE row (empty when the row carries none).
pub fn pairs_encode_sha(row: &LedgerRow) -> String {
    row.output_sha
        .as_ref()
        .map(|s| s.as_str().to_string())
        .unwrap_or_default()
}

/// Metric column: a diffmap/metric run has one DONE row per (cell x metric)
/// sharing the cell identity — without this column the join table cannot tell a
/// butteraugli map from a cvvdp map (hdrgrid-diffmap endgame, 2026-08-27). Empty
/// for encode rows.
pub fn pairs_metric(row: &LedgerRow) -> String {
    match &row.kind {
        JobKind::Diffmap { metric, .. } | JobKind::Metric { metric } => metric.clone(),
        JobKind::ScoreFile { metrics, .. } => metrics.join("+"),
        _ => String::new(),
    }
}

/// The DONE rows of a ledger view in a TOTAL, content-derived order, so the
/// emitted pairs table is a pure function of the ledger's DONE set.
///
/// **ROW ORDER IS LOAD-BEARING, and it used to be random.** [`LedgerView`] stores
/// rows in a `HashMap<JobId, _>`, so `view.rows()` yields them in Rust's
/// per-process randomized hash order: two `pairs` invocations over a *frozen*
/// ledger emitted the same SET of rows in a different ORDER.
///
/// That order becomes job IDENTITY downstream. `declare-scorefiles` groups the
/// pairs table by ref and cuts each ref's member list into contiguous
/// `--chunk`-sized slices, stamping `job_id = JobId::of(kind, inputs)` per slice.
/// [`JobId::of`] sorts and dedups its inputs, so member order *within* a chunk
/// cannot move an id — what moves it is which members SHARE a chunk. A
/// permutation re-cuts that membership, and therefore re-mints every id, whenever
/// a ref has MORE members than `--chunk`. Every round of a recurring gap-fill
/// loop then re-declared the whole run as fresh jobs and the workers re-did cells
/// that had already been scored. (This is a correction to the Stage-A writeup,
/// which attributed the churn to input order directly; the distinction is what
/// decides whether a given run is exposed — see
/// `only_refs_larger_than_the_chunk_can_remint`.)
///
/// MEASURED on the AVIF-DOE score wave (`benchmarks/avif_doe_stageA_2026-09-02.md`
/// §1.4): `declared=4,128` against `ever_done=16,476` — a 4.0x re-work multiplier
/// with `errors=0`. It was still compounding when this landed: over rounds 37-40
/// the encode side sat frozen at 49,120 DONE cells and `declared` stayed pinned at
/// exactly 4,128, while `ever_done` reached **29,664 — 7.19x** the declared work,
/// i.e. the fleet had run ~7 full passes over the same 4,128 jobs. (Score blobs
/// track distinct completed `job_id`s about 1:1, which is why they accumulated in
/// step: 29,608 blobs for one pass worth of work.) Such a run never settles — the
/// gap closes each round and the next round re-mints it.
///
/// It is a WASTE bug, not a correctness one — every blob is valid and the ledger
/// converges, because identity is content-addressed — but it multiplies the cost
/// of exactly the pattern the job system recommends: a recurring, idempotent
/// declaration loop.
///
/// The key leads with the emitted cell identity (meaningful and diff-friendly, and
/// it lands a ref's members in q order so chunks group related cells) and ends
/// with `job_id` as the tie-break. `job_id` is unique by construction — it is the
/// `LedgerView` map key — so no two rows compare equal, the order is total, and
/// the result cannot depend on iteration or insertion order.
pub fn pairs_done_sorted(view: &LedgerView) -> Vec<&LedgerRow> {
    /// `(image_path, codec, q, knob_tuple_json, metric, encode_sha, job_id)`.
    /// The trailing `job_id` is what makes the order TOTAL.
    type SortKey<'a> = (&'a str, &'a str, i64, &'a str, String, String, &'a str);
    // Decorate-sort-undecorate: the metric/sha key parts allocate, so build each
    // key once rather than once per comparison.
    let mut keyed: Vec<(SortKey<'_>, &LedgerRow)> = view
        .rows()
        .filter(|r| r.status == JobStatus::Done)
        .map(|r| {
            (
                (
                    r.cell.image_path.as_str(),
                    r.cell.codec.as_str(),
                    r.cell.q,
                    r.cell.knob_tuple_json.as_str(),
                    pairs_metric(r),
                    pairs_encode_sha(r),
                    r.job_id.0.as_str(),
                ),
                r,
            )
        })
        .collect();
    // `sort_unstable` is safe precisely because the key is total (job_id is unique).
    keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    keyed.into_iter().map(|(_, r)| r).collect()
}

/// Render the pairs TSV. The single owner of the emitted column set, shared with
/// the Parquet writer in `jobctl` so the two representations cannot drift.
pub fn pairs_tsv(rows: &[&LedgerRow], refs_prefix: &str, blobs_prefix: &str) -> String {
    use std::fmt::Write as _;
    let mut tsv = String::from(
        "ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\tencode_sha\tmetric\tworker\tprovider\n",
    );
    for r in rows {
        let sha = pairs_encode_sha(r);
        let _ = writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            pairs_join_prefix(refs_prefix, &r.cell.image_path),
            pairs_join_prefix(blobs_prefix, &sha),
            r.cell.image_path,
            r.cell.codec,
            r.cell.q,
            r.cell.knob_tuple_json,
            sha,
            pairs_metric(r),
            r.worker,
            r.provider
        );
    }
    tsv
}

/// One pairs row for [`declare_scorefile_jobs`] (migrated 2026-08-27 from
/// scripts/jobsys/declare_direct_objects.py). `ref_key` is the grouping key
/// (full ref URI, or basename); `member` the variant object name/URI; the
/// identity triple is carried when the pairs parquet has it.
#[derive(Debug, Clone)]
pub struct PairRow {
    pub ref_key: String,
    pub member: String,
    pub identity: Option<(String, i64, String)>, // (codec, q, knob_tuple_json)
}

/// Build ScoreFile jobs (CHUNKed per ref) or Diffmap jobs (one per variant x
/// metric) from pairs rows — the DesiredJob emission the Python did by hand-
/// rolling wire JSON, which silently drifted from the schema the moment
/// invariant 5 added `requires` (hand-emitted manifests carried none). Here
/// the REAL types build the jobs and `required_capabilities()` stamps them.
#[allow(clippy::too_many_arguments)]
pub fn declare_scorefile_jobs(
    rows: &[PairRow],
    metrics: &[String],
    chunk: usize,
    cell_codec: &str,
    cell_knobs: &str,
    hdr: bool,
    hdr_transfer: Option<&str>,
    diffmap: bool,
    hint: Option<ResourceHint>,
) -> Vec<DesiredJob> {
    use std::collections::BTreeMap;
    // BTreeMap: deterministic ref order (the Python used dict insertion order;
    // sorted order is the stable, diff-friendly choice — noted in the parity gate).
    let mut by_ref: BTreeMap<&str, Vec<&PairRow>> = BTreeMap::new();
    for r in rows {
        by_ref.entry(r.ref_key.as_str()).or_default().push(r);
    }
    let mut out = Vec::new();
    for (rk, members) in by_ref {
        if diffmap {
            for m in members {
                let cell = match &m.identity {
                    Some((codec, q, knobs)) => CellId {
                        image_path: rk.to_string(),
                        codec: codec.clone(),
                        q: *q,
                        knob_tuple_json: knobs.clone(),
                    },
                    None => CellId {
                        image_path: rk.to_string(),
                        codec: cell_codec.to_string(),
                        q: -1,
                        knob_tuple_json: "diffmap".into(),
                    },
                };
                for metric in metrics {
                    let kind = JobKind::Diffmap {
                        metric: metric.clone(),
                        hdr,
                    };
                    let j = DesiredJob {
                        requires: kind.required_capabilities(),
                        kind,
                        inputs: vec![Sha256Hex::raw_object_key(m.member.clone())],
                        cell: cell.clone(),
                        hint,
                    };
                    out.push(j);
                }
            }
        } else {
            let names: Vec<&PairRow> = members;
            // Per-ref codec label: the rows' identity codec when the pairs carry
            // one (multi-codec corpora — build_scorefile_from_pairs), else the
            // flag (single-codec runs).
            let ref_codec = names
                .iter()
                .find_map(|m| m.identity.as_ref().map(|(c, _, _)| c.clone()))
                .unwrap_or_else(|| cell_codec.to_string());
            for ch in names.chunks(chunk.max(1)) {
                let kind = JobKind::ScoreFile {
                    metrics: metrics.to_vec(),
                    hdr,
                    hdr_transfer: hdr_transfer.map(str::to_string),
                };
                let j = DesiredJob {
                    requires: kind.required_capabilities(),
                    kind,
                    inputs: ch
                        .iter()
                        .map(|m| Sha256Hex::raw_object_key(m.member.clone()))
                        .collect(),
                    cell: CellId {
                        image_path: rk.to_string(),
                        codec: ref_codec.clone(),
                        q: -1,
                        knob_tuple_json: cell_knobs.to_string(),
                    },
                    hint,
                };
                out.push(j);
            }
        }
    }
    out
}

#[cfg(test)]
mod declare_scorefile_tests {
    use super::*;

    fn rows() -> Vec<PairRow> {
        vec![
            PairRow {
                ref_key: "a.png".into(),
                member: "a_q10.avif".into(),
                identity: Some(("zenavif".into(), 10, "{}".into())),
            },
            PairRow {
                ref_key: "a.png".into(),
                member: "a_q20.avif".into(),
                identity: Some(("zenavif".into(), 20, "{}".into())),
            },
            PairRow {
                ref_key: "a.png".into(),
                member: "a_q30.avif".into(),
                identity: None,
            },
            PairRow {
                ref_key: "b.png".into(),
                member: "b_q10.avif".into(),
                identity: None,
            },
        ]
    }

    #[test]
    fn scorefile_chunks_per_ref_and_stamps_requires() {
        let jobs = declare_scorefile_jobs(
            &rows(),
            &["ssim2-gpu".to_string()],
            2,
            "fallback",
            "scorefile",
            false,
            None,
            false,
            None,
        );
        // a: 3 members / chunk 2 -> 2 jobs; b: 1 job
        assert_eq!(jobs.len(), 3);
        assert!(
            jobs.iter()
                .all(|j| matches!(j.kind, JobKind::ScoreFile { .. }))
        );
        // invariant 5: the capability tokens the Python never stamped
        assert!(
            jobs.iter()
                .all(|j| j.requires == vec!["gpu-ssim2".to_string()])
        );
        assert_eq!(jobs[0].inputs.len(), 2);
        assert_eq!(jobs[1].inputs.len(), 1);
        assert_eq!(jobs[0].cell.knob_tuple_json, "scorefile");
        // per-ref identity codec wins over the flag; refs with no identity fall back
        let a_job = jobs.iter().find(|j| j.cell.image_path == "a.png").unwrap();
        assert_eq!(a_job.cell.codec, "zenavif");
        let b_job = jobs.iter().find(|j| j.cell.image_path == "b.png").unwrap();
        assert_eq!(b_job.cell.codec, "fallback");
    }

    #[test]
    fn diffmap_one_job_per_variant_x_metric_with_true_identity() {
        let jobs = declare_scorefile_jobs(
            &rows(),
            &["butteraugli".to_string(), "cvvdp".to_string()],
            12,
            "zenjpeg",
            "scorefile",
            true,
            None,
            true,
            None,
        );
        assert_eq!(jobs.len(), 8); // 4 variants x 2 metrics
        let with_id = jobs
            .iter()
            .find(|j| j.cell.q == 10)
            .expect("true identity carried");
        assert_eq!(with_id.cell.codec, "zenavif");
        let no_id = jobs
            .iter()
            .find(|j| j.cell.knob_tuple_json == "diffmap")
            .expect("fallback cell");
        assert_eq!(no_id.cell.q, -1);
        // hdr:true rides the kind and requires carries the hdr arm + gpu metric class
        assert!(
            jobs.iter()
                .all(|j| matches!(j.kind, JobKind::Diffmap { hdr: true, .. }))
        );
        assert!(jobs.iter().all(|j| j.requires.contains(&"hdr".to_string())));
    }
}

#[cfg(test)]
mod migrate_tests {
    use super::*;
    use zenfleet_core::{CellId, JobKind, JobStatus, LedgerRow};

    fn row(tag: &str, status: JobStatus, ts: u64) -> LedgerRow {
        let job = DesiredJob::new(
            JobKind::Metric {
                metric: "cvvdp".into(),
            },
            vec![zenfleet_core::sha256(tag.as_bytes())],
            CellId {
                image_path: format!("{tag}.png"),
                codec: "zenjxl".into(),
                q: 80,
                knob_tuple_json: "{}".into(),
            },
        );
        LedgerRow {
            job_id: job.job_id(),
            kind: job.kind.clone(),
            cell: job.cell.clone(),
            output_sha: None,
            status,
            error_class: None,
            attempts: 1,
            ts,
            worker: "t".into(),
            provider: "t".into(),
        }
    }

    #[test]
    fn account_rows_python_semantics() {
        // a: done (twice) — counts once; b: failed only; c: failed then done — done wins
        let rows = vec![
            row("a", JobStatus::Done, 10),
            row("a", JobStatus::Done, 11),
            row("b", JobStatus::Failed, 5),
            row("c", JobStatus::Failed, 6),
            row("c", JobStatus::Done, 7),
        ];
        let acc = account_rows(4, &rows);
        assert_eq!(
            acc,
            RunAccounting {
                declared: 4,
                distinct_done: 2,
                failed_only: 1,
                gap: 2,
                live_done: 2,
                gap_live: 2,
                raw_rows: 5
            }
        );
        // negative gap on re-declared runs is preserved, not clamped here
        assert_eq!(account_rows(1, &rows).gap, -1);
    }

    #[test]
    fn snapshot_rows_first_wins_done_plus_newest_failed() {
        let rows = vec![
            row("a", JobStatus::Done, 10), // kept (first)
            row("a", JobStatus::Done, 99), // dropped (first-wins)
            row("b", JobStatus::Failed, 5),
            row("b", JobStatus::Failed, 9), // kept (newest failed, no done)
            row("c", JobStatus::Failed, 6), // dropped (c has a done row)
            row("c", JobStatus::Done, 7),   // kept
        ];
        let (snap, n_done, n_failed) = snapshot_rows(rows);
        assert_eq!((n_done, n_failed), (2, 1));
        assert_eq!(snap.len(), 3);
        let done_ts: Vec<u64> = snap
            .iter()
            .filter(|r| r.status == JobStatus::Done)
            .map(|r| r.ts)
            .collect();
        assert_eq!(done_ts, vec![10, 7], "first done row wins, in scan order");
        let failed: Vec<&LedgerRow> = snap
            .iter()
            .filter(|r| r.status == JobStatus::Failed)
            .collect();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0].ts, 9, "newest failed row carried");
    }
}

#[cfg(test)]
mod pairs_order_tests {
    use super::*;
    use zenfleet_core::{CellId, JobKind, JobStatus, LedgerRow};

    /// A DONE encode row. `tag` varies the content hash so every row is a
    /// distinct `job_id`, exactly as a real ledger's DONE set is.
    fn done_row(image: &str, q: i64, tag: &str) -> LedgerRow {
        let job = DesiredJob::new(
            JobKind::Encode {
                codec: "zenavif".into(),
                q,
                knobs: "{}".into(),
                hdr: false,
            },
            vec![zenfleet_core::sha256(tag.as_bytes())],
            CellId {
                image_path: image.into(),
                codec: "zenavif".into(),
                q,
                knob_tuple_json: "{}".into(),
            },
        );
        LedgerRow {
            job_id: job.job_id(),
            kind: job.kind.clone(),
            cell: job.cell.clone(),
            output_sha: Some(zenfleet_core::sha256(format!("blob-{tag}").as_bytes())),
            status: JobStatus::Done,
            error_class: None,
            attempts: 1,
            ts: 1,
            worker: "w".into(),
            provider: "local".into(),
        }
    }

    /// 30 rows across 5 refs — large enough that a HashMap ordering could not
    /// coincidentally come out sorted.
    fn corpus() -> Vec<LedgerRow> {
        let mut v = Vec::new();
        for i in 0..5 {
            for q in [20i64, 40, 60, 80, 90, 95] {
                let img = format!("s3://refs/img{i}.png");
                v.push(done_row(&img, q, &format!("{i}-{q}")));
            }
        }
        v
    }

    /// Deterministically permute, so the test never depends on HashMap luck.
    fn rotated(rows: &[LedgerRow], by: usize) -> Vec<LedgerRow> {
        let n = rows.len();
        (0..n).map(|i| rows[(i + by) % n].clone()).collect()
    }

    fn view_of(rows: &[LedgerRow]) -> LedgerView {
        let mut v = LedgerView::new();
        for r in rows {
            v.apply(r.clone());
        }
        v
    }

    #[test]
    fn pairs_output_is_byte_identical_across_invocations() {
        let base = corpus();
        // Eight independently-built views. Each `LedgerView` owns a fresh
        // `HashMap`, and `RandomState::new()` perturbs its hasher per map, so
        // these genuinely differ in `view.rows()` order within one process —
        // which is the nondeterminism that reached the emitted table.
        let renders: Vec<String> = (0..8)
            .map(|k| {
                let v = view_of(&rotated(&base, k * 3));
                pairs_tsv(&pairs_done_sorted(&v), "s3://refs", "s3://blobs")
            })
            .collect();
        for (i, r) in renders.iter().enumerate() {
            assert_eq!(
                r, &renders[0],
                "pairs TSV differs between invocation 0 and {i} — the emitted \
                 table is not a pure function of the ledger's DONE set"
            );
        }
        // Teeth: the output really is in the documented order, so deleting the
        // sort fails here and not only by luck of the hash seed.
        let v = view_of(&base);
        let done = pairs_done_sorted(&v);
        assert_eq!(done.len(), 30);
        let keys: Vec<(&str, i64)> = done
            .iter()
            .map(|r| (r.cell.image_path.as_str(), r.cell.q))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort_unstable();
        assert_eq!(keys, sorted, "emitted rows are not in cell-identity order");
    }

    /// The rows a `--full-uri` declare builds from the pairs table.
    fn pair_rows(view: &LedgerView) -> Vec<PairRow> {
        pairs_done_sorted(view)
            .into_iter()
            .map(|r| PairRow {
                ref_key: pairs_join_prefix("s3://refs", &r.cell.image_path),
                member: pairs_join_prefix("s3://blobs", &pairs_encode_sha(r)),
                identity: Some((
                    r.cell.codec.clone(),
                    r.cell.q,
                    r.cell.knob_tuple_json.clone(),
                )),
            })
            .collect()
    }

    fn job_ids(rows: &[PairRow], chunk: usize) -> Vec<String> {
        declare_scorefile_jobs(
            rows,
            &["ssim2".to_string(), "zensim".to_string()],
            chunk,
            "zenavif",
            "scorefile",
            false,
            None,
            false,
            None,
        )
        .iter()
        .map(|j| j.job_id().0.as_str().to_string())
        .collect()
    }

    #[test]
    fn chunk_boundaries_are_stable_across_ledger_insertion_order() {
        let base = corpus();
        // chunk=2 against 6 members per ref: three chunks per ref, so a
        // reordering genuinely RE-CUTS membership. At chunk >= members/ref the
        // property holds trivially and the test would have no teeth — see
        // `only_refs_larger_than_the_chunk_can_remint`.
        for chunk in [2usize, 4] {
            let ids: Vec<Vec<String>> = (0..8)
                .map(|k| job_ids(&pair_rows(&view_of(&rotated(&base, k * 3))), chunk))
                .collect();
            assert!(ids[0].len() >= 10, "chunk {chunk} should cut many jobs");
            for (i, got) in ids.iter().enumerate() {
                assert_eq!(
                    got, &ids[0],
                    "chunk={chunk}: declare-scorefiles job identity differs between \
                     ledger insertion order 0 and {i} — a recurring declare would \
                     re-mint every job_id and the fleet would redo finished work"
                );
            }
        }
    }

    #[test]
    fn permuted_pairs_rows_would_remint_job_ids() {
        // The deterministic negative control that gives the tests above their
        // teeth: chunk identity IS a function of row order, which is exactly why
        // `pairs` must impose a total one.
        let sorted = pair_rows(&view_of(&corpus()));
        let permuted = rotated_pairs(&sorted, 1);
        assert_ne!(
            job_ids(&sorted, 2),
            job_ids(&permuted, 2),
            "if this ever passes, chunk membership stopped depending on row order \
             and the pairs sort is no longer load-bearing"
        );
    }

    #[test]
    fn only_refs_larger_than_the_chunk_can_remint() {
        // PINS THE PRECISE MECHANISM. `JobId::of` sorts and dedups its inputs, so
        // member ORDER inside a chunk cannot move an id; what moves it is which
        // members SHARE a chunk. A chunk is a contiguous slice of one ref's member
        // list, so a permutation re-cuts membership only when a ref has MORE
        // members than `--chunk`.
        //
        // This is why the AVIF-DOE run churned so hard: ~49,120 pairs rows over a
        // 32-image corpus is ~1,535 members per ref at `--chunk 12`, i.e. ~128
        // chunks per ref, every one re-cut on every round. It is also why the
        // job COUNT stayed pinned at 4,128 while the identities rotated — a
        // permutation changes which members pair up, never how many chunks fall out.
        let sorted = pair_rows(&view_of(&corpus())); // 6 members per ref
        let permuted = rotated_pairs(&sorted, 1);
        assert_ne!(
            job_ids(&sorted, 2),
            job_ids(&permuted, 2),
            "chunk 2 < 6 members/ref: membership must re-cut"
        );
        assert_eq!(
            job_ids(&sorted, 12),
            job_ids(&permuted, 12),
            "chunk 12 >= 6 members/ref: one chunk per ref, so membership is \
             invariant under permutation and ids must NOT move"
        );
    }

    fn rotated_pairs(rows: &[PairRow], by: usize) -> Vec<PairRow> {
        let n = rows.len();
        (0..n).map(|i| rows[(i + by) % n].clone()).collect()
    }
}
