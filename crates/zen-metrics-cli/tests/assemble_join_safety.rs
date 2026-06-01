#![forbid(unsafe_code)]
#![cfg(feature = "assemble")]

//! Integration tests encoding the 2026-05-25 corpus-corruption prevention
//! guarantees for `zen-metrics assemble`.
//!
//! Each test corresponds to one item in the task's mandatory list and to one
//! guarantee ported from `zensim/scripts/canonical_corpus/join_safety.py`.
//! The point of these tests is structural: they prove the corruption modes
//! root-caused in `DATA_INTEGRITY_root_cause_2026-05-25.md` cannot be
//! reintroduced through this code path.
//!
//! Run: `cargo test -p zen-metrics-cli --features sweep --test assemble_join_safety`

use zen_metrics_cli::assemble::join::{
    JoinHow, assert_no_leaked_columns, assert_not_constant_per_ref_tuned, attach_positional,
    safe_join,
};
use zen_metrics_cli::assemble::key::PairKey;
use zen_metrics_cli::assemble::parquet_io::{read_parquet, write_parquet};
use zen_metrics_cli::assemble::table::{Column, Table};

/// Helper: a per-pair metric table keyed on the full PairKey.
fn metric_table(
    image_paths: &[&str],
    codecs: &[&str],
    qs: &[i64],
    knobs: &[&str],
    col: &str,
    vals: &[f64],
) -> Table {
    Table::from_columns(vec![
        (
            "image_path".into(),
            Column::Str(image_paths.iter().map(|s| Some(s.to_string())).collect()),
        ),
        (
            "codec".into(),
            Column::Str(codecs.iter().map(|s| Some(s.to_string())).collect()),
        ),
        ("q".into(), Column::I64(qs.to_vec())),
        (
            "knob_tuple_json".into(),
            Column::Str(knobs.iter().map(|s| Some(s.to_string())).collect()),
        ),
        (col.into(), Column::F64(vals.to_vec())),
    ])
    .unwrap()
}

/// Test 1 — A 3-way join on synthetic tables produces correct per-cell rows
/// (golden). We exercise the full-key `safe_join` twice (omni × zensim ×
/// source-style) and verify each cell keeps its own per-pair value.
#[test]
fn test1_three_way_join_golden_per_cell() {
    // Target (omni-like): 4 cells across 2 references, 2 distortions each.
    let target = Table::from_columns(vec![
        (
            "image_path".into(),
            Column::Str(
                ["A/q50.png", "A/q60.png", "B/q50.png", "B/q60.png"]
                    .iter()
                    .map(|s| Some(s.to_string()))
                    .collect(),
            ),
        ),
        ("codec".into(), Column::Str(vec![Some("jpeg".into()); 4])),
        ("q".into(), Column::I64(vec![50, 60, 50, 60])),
        (
            "knob_tuple_json".into(),
            Column::Str(vec![Some("{}".into()); 4]),
        ),
        (
            "ref_basename".into(),
            Column::Str(
                ["A.png", "A.png", "B.png", "B.png"]
                    .iter()
                    .map(|s| Some(s.to_string()))
                    .collect(),
            ),
        ),
    ])
    .unwrap();

    // First metric: zensim feature/score, distinct per cell.
    let zsm = metric_table(
        &["A/q50.png", "A/q60.png", "B/q50.png", "B/q60.png"],
        &["jpeg", "jpeg", "jpeg", "jpeg"],
        &[50, 60, 50, 60],
        &["{}", "{}", "{}", "{}"],
        "zsm_feat_0",
        &[10.0, 11.0, 12.0, 13.0],
    );
    let step1 = safe_join(&target, &zsm, "zsm_feat_0", JoinHow::Inner).unwrap();
    assert_eq!(step1.num_rows(), 4);

    // Second metric: ssim2 per cell — the column Mode B corrupted. Each cell
    // MUST keep its own value (NOT a per-ref mean).
    let ssim2 = metric_table(
        &["A/q50.png", "A/q60.png", "B/q50.png", "B/q60.png"],
        &["jpeg", "jpeg", "jpeg", "jpeg"],
        &[50, 60, 50, 60],
        &["{}", "{}", "{}", "{}"],
        "ssim2_gpu",
        &[80.0, 70.0, 90.0, 60.0],
    );
    let step2 = safe_join(&step1, &ssim2, "ssim2_gpu", JoinHow::Inner).unwrap();
    assert_eq!(step2.num_rows(), 4);

    // Golden: the two distortions of reference A have DIFFERENT ssim2 (80 vs
    // 70) — proving no ref-broadcast collapse happened.
    let ssim2_col = step2.column("ssim2_gpu").unwrap();
    let zsm_col = step2.column("zsm_feat_0").unwrap();
    // Rows may be reordered by the inner join; map by image_path.
    let ip = step2.column("image_path").unwrap();
    let mut got: std::collections::HashMap<String, (f64, f64)> = Default::default();
    for i in 0..step2.num_rows() {
        got.insert(ip.key_at(i), (zsm_col.f64_at(i), ssim2_col.f64_at(i)));
    }
    assert_eq!(got["A/q50.png"], (10.0, 80.0));
    assert_eq!(got["A/q60.png"], (11.0, 70.0));
    assert_eq!(got["B/q50.png"], (12.0, 90.0));
    assert_eq!(got["B/q60.png"], (13.0, 60.0));
}

/// Test 2 — Attempting to join a per-pair metric onto a target that has ONLY
/// `ref_basename` is an explicit error. The `PairKey` type makes the ref-only
/// CALL unrepresentable (no constructor with fewer fields); `safe_join`'s
/// `require_columns` is the runtime backstop that fires here because the
/// target's *schema* (a runtime parquet) lacks the per-pair columns.
#[test]
fn test2_ref_only_target_is_rejected() {
    // A features table carrying ONLY ref_basename — the exact Mode-B shape.
    let ref_only = Table::from_columns(vec![
        (
            "ref_basename".into(),
            Column::Str(
                ["A.png", "A.png", "B.png"]
                    .iter()
                    .map(|s| Some(s.to_string()))
                    .collect(),
            ),
        ),
        ("human_score".into(), Column::F64(vec![10.0, 20.0, 30.0])),
    ])
    .unwrap();
    let metric = metric_table(
        &["A/q50.png", "A/q60.png", "B/q50.png"],
        &["jpeg", "jpeg", "jpeg"],
        &[50, 60, 50],
        &["{}", "{}", "{}"],
        "ssim2_gpu",
        &[80.0, 70.0, 90.0],
    );

    let err = safe_join(&ref_only, &metric, "ssim2_gpu", JoinHow::Left).unwrap_err();
    let msg = err.to_string();
    // Must name the missing per-pair keys + the Mode-B bug, never silently
    // collapse.
    assert!(msg.contains("per-pair"), "msg: {msg}");
    assert!(
        msg.contains("ref-misjoin") || msg.contains("ref-only"),
        "msg: {msg}"
    );
    // And it must mention ref_basename (what the table DOES carry).
    assert!(msg.contains("ref_basename"), "msg: {msg}");

    // Compile-time half of the defense: there is no PairKey constructor that
    // takes only ref_basename. `require_columns` on a ref-only table is the
    // observable proxy for that — it cannot produce PairKeys.
    assert!(PairKey::require_columns("ref_only", &ref_only).is_err());
}

/// Test 3 — Duplicate keys on the metric side are an error, NOT a silent
/// average. This is the precise mechanism (`groupby().mean()`) that destroyed
/// the per-pair signal in Mode B.
#[test]
fn test3_duplicate_metric_keys_error_not_average() {
    let target = metric_table(
        &["A/q50.png"],
        &["jpeg"],
        &[50],
        &["{}"],
        "human_score",
        &[42.0],
    );
    // Metric side has TWO rows for the same (image_path, codec, q, knob).
    let dup_metric = metric_table(
        &["A/q50.png", "A/q50.png"],
        &["jpeg", "jpeg"],
        &[50, 50],
        &["{}", "{}"],
        "ssim2_gpu",
        &[80.0, 60.0], // a naive mean would be 70.0 — we must NOT do that
    );
    let err = safe_join(&target, &dup_metric, "ssim2_gpu", JoinHow::Left).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("NOT unique"), "msg: {msg}");
    assert!(msg.contains("averaging"), "msg: {msg}");
}

/// Test 4a — A `*mock*` column is rejected by the leak detector.
#[test]
fn test4a_mock_column_rejected() {
    let n = 150usize; // > 100 so the human-score comparison would engage
    let t = Table::from_columns(vec![
        (
            "human_score".into(),
            Column::F64((0..n).map(|i| i as f64).collect()),
        ),
        (
            "iwssim_mock".into(),
            Column::F64((0..n).map(|i| (i as f64) * 0.5).collect()),
        ),
    ])
    .unwrap();
    let err = assert_no_leaked_columns("kadid_training", &t).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("MOCK"), "msg: {msg}");
}

/// Test 4b — A raw-metric column bit-identical to `human_score` is rejected
/// (Mode A target leak), while a `mix_*` column equal to the anchor and a
/// linearly-rescaled metric are BOTH allowed.
#[test]
fn test4b_human_copy_rejected_mix_and_rescale_allowed() {
    let n = 200usize;
    let human: Vec<f64> = (0..n).map(|i| 10.0 + i as f64 * 0.3).collect();

    // iwssim == human_score, bit-identical → leak.
    let leak = Table::from_columns(vec![
        ("human_score".into(), Column::F64(human.clone())),
        ("iwssim".into(), Column::F64(human.clone())),
    ])
    .unwrap();
    let err = assert_no_leaked_columns("tid_training", &leak).unwrap_err();
    assert!(err.to_string().contains("bit-identical"), "{err}");

    // mix_* == human_score is LEGITIMATE (the anchor IS the active mix target
    // for konjnd-dense / LARGE) — must NOT be flagged.
    let mix_ok = Table::from_columns(vec![
        ("human_score".into(), Column::F64(human.clone())),
        ("mix_cv40_iw60".into(), Column::F64(human.clone())),
    ])
    .unwrap();
    assert!(
        assert_no_leaked_columns("konjnd_dense", &mix_ok).is_ok(),
        "mix_* equal to anchor must be allowed"
    );

    // ssim2_gpu = human_score / 100 — a perfect linear rescale (corr 1.0) but
    // NOT bit-identical. safesyn's anchor is exactly this. Must NOT be flagged
    // (only value bit-identity is the leak signature).
    let rescale = Table::from_columns(vec![
        ("human_score".into(), Column::F64(human.clone())),
        (
            "ssim2_gpu".into(),
            Column::F64(human.iter().map(|x| x / 100.0).collect()),
        ),
    ])
    .unwrap();
    assert!(
        assert_no_leaked_columns("safesyn", &rescale).is_ok(),
        "linear rescale (corr 1.0, not bit-identical) must be allowed"
    );
}

/// Test 4c — the post-hoc constant-per-ref detector flags a ref-broadcast and
/// does NOT false-positive on a per-pair sidecar (mean group size ≈ 1).
#[test]
fn test4c_constant_per_ref_detector() {
    // Broadcast shape: 6 refs × 3 distortions, ssim2 CONSTANT within each ref.
    let mut refs: Vec<Option<String>> = Vec::new();
    let mut ssim2: Vec<f64> = Vec::new();
    for r in 0..6 {
        for _ in 0..3 {
            refs.push(Some(format!("ref{r}")));
            ssim2.push(50.0 + r as f64); // same within ref, differs across
        }
    }
    let broadcast = Table::from_columns(vec![
        ("ref_basename".into(), Column::Str(refs)),
        ("ssim2_gpu".into(), Column::F64(ssim2)),
    ])
    .unwrap();
    // mean group size = 3.0 > 1.5 → the gate engages and the broadcast fires.
    let err =
        assert_not_constant_per_ref_tuned("kadid", "ref_basename", "ssim2_gpu", &broadcast, 5, 1.5)
            .unwrap_err();
    assert!(err.to_string().contains("constant within every"), "{err}");

    // Per-pair sidecar shape: each ref appears once → mean group size 1.0 ≤
    // 1.5 → the false-positive gate suppresses the (trivially true) "one value
    // per ref" condition. Must be OK.
    let perpair = Table::from_columns(vec![
        (
            "ref_basename".into(),
            Column::Str((0..6).map(|r| Some(format!("ref{r}"))).collect()),
        ),
        (
            "ssim2_gpu".into(),
            Column::F64((0..6).map(|i| 50.0 + i as f64).collect()),
        ),
    ])
    .unwrap();
    assert!(
        assert_not_constant_per_ref_tuned("scores", "ref_basename", "ssim2_gpu", &perpair, 5, 1.5)
            .is_ok(),
        "per-pair sidecar (group size 1) must not false-positive"
    );
}

/// Test 5 — positional attach with a length mismatch is an error.
#[test]
fn test5_positional_length_mismatch_error() {
    let target = Table::from_columns(vec![
        (
            "ref_basename".into(),
            Column::Str((0..3).map(|i| Some(format!("r{i}"))).collect()),
        ),
        ("human_score".into(), Column::F64(vec![1.0, 2.0, 3.0])),
    ])
    .unwrap();
    // 2 values vs 3 rows → must error.
    let err = attach_positional(&target, &[10.0, 20.0], "ssim2_gpu").unwrap_err();
    assert!(err.to_string().contains("EXACT row-count"), "{err}");

    // Exact match succeeds and attaches positionally.
    let ok = attach_positional(&target, &[10.0, 20.0, 30.0], "ssim2_gpu").unwrap();
    assert_eq!(ok.column("ssim2_gpu").unwrap().f64_at(2), 30.0);
}

/// Test 6 — round-trip a small sidecar trio through the parquet writer/reader
/// and confirm byte-stable output. Uses committed in-test fixtures (no /mnt/v
/// dependency) so the test runs in minimal CI.
#[test]
fn test6_parquet_round_trip_byte_stable() {
    let dir = tempfile::tempdir().unwrap();

    // Build a tiny joined corpus and write it, then read + rewrite and assert
    // the bytes are identical (deterministic writer config).
    let t = Table::from_columns(vec![
        (
            "image_path".into(),
            Column::Str(vec![Some("A/q50.png".into()), Some("B/q50.png".into())]),
        ),
        ("codec".into(), Column::Str(vec![Some("jpeg".into()); 2])),
        ("q".into(), Column::I64(vec![50, 50])),
        (
            "knob_tuple_json".into(),
            Column::Str(vec![Some("{}".into()); 2]),
        ),
        ("ssim2_gpu".into(), Column::F64(vec![80.0, 90.0])),
        ("zsm_feat_0".into(), Column::F64(vec![1.0, 2.0])),
    ])
    .unwrap();

    let p1 = dir.path().join("jpeg_training.parquet");
    let p2 = dir.path().join("jpeg_training_2.parquet");
    write_parquet(&t, &p1).unwrap();
    let read_back = read_parquet(&p1).unwrap();
    write_parquet(&read_back, &p2).unwrap();

    let b1 = std::fs::read(&p1).unwrap();
    let b2 = std::fs::read(&p2).unwrap();
    assert_eq!(b1, b2, "round-trip parquet must be byte-stable");

    // Logical content survived the round-trip with per-pair signal intact.
    assert_eq!(read_back.num_rows(), 2);
    assert_eq!(read_back.column("q").unwrap().key_at(0), "50");
    assert_eq!(read_back.column("ssim2_gpu").unwrap().f64_at(1), 90.0);
}
