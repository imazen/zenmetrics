//! Join-safety tests for `assemble --mode flat-picker` (the strict
//! encode_sha-keyed harvest->picker-shape builder). Mirrors the
//! assemble_join_safety suite's stance: corruption modes must be ERRORS,
//! never silent output.
#![cfg(feature = "assemble")]

use zenmetrics_cli::assemble::flat_picker::run_flat_picker;
use zenmetrics_cli::assemble::parquet_io::{read_parquet, write_parquet};
use zenmetrics_cli::assemble::table::{Column, Table};

fn scores_fixture(dir: &std::path::Path, shas: &[&str]) -> std::path::PathBuf {
    let n = shas.len();
    let t = Table::from_columns(vec![
        (
            "image_path".into(),
            Column::Str(vec![Some("77.scale64x48.png".into()); n]),
        ),
        ("codec".into(), Column::Str(vec![Some("zenavif".into()); n])),
        (
            "q".into(),
            Column::I64((0..n as i64).map(|i| i * 10).collect()),
        ),
        (
            "knob_tuple_json".into(),
            Column::Str(vec![
                Some(
                    r#"{"cell":"s2-420","fp":"ab","plan":"rd_core"}"#.into()
                );
                n
            ]),
        ),
        (
            "encode_sha".into(),
            Column::Str(shas.iter().map(|s| Some((*s).into())).collect()),
        ),
        (
            "ssim2_gpu".into(),
            Column::F64((0..n).map(|i| 50.0 + i as f64).collect()),
        ),
    ])
    .unwrap();
    let p = dir.join("scores.parquet");
    write_parquet(&t, &p).unwrap();
    p
}

fn sizes_fixture(dir: &std::path::Path, rows: &[(&str, u64)]) -> std::path::PathBuf {
    let p = dir.join("sizes.tsv");
    let body: String = rows.iter().map(|(k, v)| format!("{k}\t{v}\n")).collect();
    std::fs::write(&p, body).unwrap();
    p
}

#[test]
fn duplicate_sha_in_scores_is_legal_but_in_sidecars_is_an_error() {
    // scores side: many cells may share one byte-identical encode — legal.
    let d = tempfile::tempdir().unwrap();
    let s = scores_fixture(d.path(), &["aaa", "aaa"]);
    let z = sizes_fixture(d.path(), &[("aaa", 10)]);
    run_flat_picker(
        &s,
        &z,
        &[],
        &[],
        &[],
        "lossy",
        false,
        &d.path().join("o.parquet"),
    )
    .unwrap();
    // sha-keyed sidecar side: duplicates are corruption — hard error.
    let extra = Table::from_columns(vec![
        (
            "encode_sha".into(),
            Column::Str(vec![Some("aaa".into()), Some("aaa".into())]),
        ),
        ("zensim_c944".into(), Column::F64(vec![1.0, 2.0])),
    ])
    .unwrap();
    let xp = d.path().join("extra.parquet");
    write_parquet(&extra, &xp).unwrap();
    let e = run_flat_picker(
        &s,
        &z,
        &[xp],
        &[],
        &[],
        "lossy",
        false,
        &d.path().join("o2.parquet"),
    )
    .unwrap_err();
    assert!(e.to_string().contains("DUPLICATE encode_sha"), "{e}");
}

#[test]
fn missing_bytes_errors_unless_allowed() {
    let d = tempfile::tempdir().unwrap();
    let s = scores_fixture(d.path(), &["aaa", "bbb"]);
    let z = sizes_fixture(d.path(), &[("aaa", 10)]);
    let out = d.path().join("o.parquet");
    let e = run_flat_picker(&s, &z, &[], &[], &[], "lossy", false, &out).unwrap_err();
    assert!(e.to_string().contains("NO bytes"), "{e}");
    // allowed: succeeds, NaN recorded for the missing row
    run_flat_picker(&s, &z, &[], &[], &[], "lossy", true, &out).unwrap();
    let t = read_parquet(&out).unwrap();
    match t.column("encoded_bytes").unwrap() {
        Column::F64(v) => {
            assert_eq!(v.len(), 2);
            assert_eq!(v[0], 10.0);
            assert!(v[1].is_nan());
        }
        _ => panic!("encoded_bytes not f64"),
    }
}

#[test]
fn happy_path_derives_identity_and_joins_extra_scores() {
    let d = tempfile::tempdir().unwrap();
    let s = scores_fixture(d.path(), &["aaa", "bbb"]);
    let z = sizes_fixture(d.path(), &[("aaa", 10), ("bbb", 20)]);
    let extra = Table::from_columns(vec![
        ("encode_sha".into(), Column::Str(vec![Some("bbb".into())])),
        ("zensim_c944".into(), Column::F64(vec![88.5])),
    ])
    .unwrap();
    let xp = d.path().join("extra.parquet");
    write_parquet(&extra, &xp).unwrap();
    let out = d.path().join("o.parquet");
    run_flat_picker(
        &s,
        &z,
        &[xp],
        &[
            "ssim2_gpu=score_ssim2".into(),
            "zensim_c944=score_zensim".into(),
        ],
        &["sweep_run=avifgen-test".into()],
        "lossy",
        false,
        &out,
    )
    .unwrap();
    let t = read_parquet(&out).unwrap();
    assert_eq!(t.num_rows(), 2);
    for c in [
        "origin_id",
        "width",
        "height",
        "cell",
        "knob_plan",
        "encoded_bytes",
        "encoded_filename",
        "mode",
        "score_ssim2",
        "score_zensim",
        "sweep_run",
    ] {
        assert!(t.has_column(c), "missing {c}");
    }
    match t.column("score_zensim").unwrap() {
        Column::F64(v) => {
            assert!(v[0].is_nan(), "unmatched row must be NaN");
            assert_eq!(v[1], 88.5);
        }
        _ => panic!(),
    }
    match t.column("width").unwrap() {
        Column::I64(v) => assert_eq!(v, &vec![64, 64]),
        _ => panic!(),
    }
}

#[test]
fn rename_collision_is_an_error() {
    let d = tempfile::tempdir().unwrap();
    let s = scores_fixture(d.path(), &["aaa"]);
    let z = sizes_fixture(d.path(), &[("aaa", 10)]);
    let e = run_flat_picker(
        &s,
        &z,
        &[],
        &["ssim2_gpu=codec".into()],
        &[],
        "lossy",
        false,
        &d.path().join("o.parquet"),
    )
    .unwrap_err();
    assert!(e.to_string().contains("already exists"), "{e}");
}
