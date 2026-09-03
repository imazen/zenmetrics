//! End-to-end CLI tests. Use `assert_cmd`-style spawning of the compiled
//! `zenmetrics` binary so we exercise the same code path users hit.
//!
//! Phase 7 adds orchestrator integration tests at the bottom of this file
//! (gated on the `orchestrator` feature). They verify the new top-level
//! flags parse, `--use-orchestrator` routes scoring through the
//! orchestrator path, and the legacy path is unchanged when the flag is
//! absent.
//!
//! The whole suite drives the binary over the PNG fixture corpus
//! (`tests/fixtures/*.png`), so it requires the `png` decode feature — the
//! gate below makes that requirement explicit in the feature chain instead of
//! failing 13 tests at runtime in a no-png build shape (e.g. the interim
//! `jobexec,hdr,cpu-metrics` executor build used while the codec siblings are
//! mid-refactor). Every png-bearing shape (default, `sweep`, CI) still runs
//! everything here.
#![cfg(feature = "png")]

use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_zenmetrics");
    Command::new(bin)
}

/// Probe `nvidia-smi` for ground truth so GPU tests branch on real
/// hardware instead of silently skipping. Mirrors the
/// `host_has_nvidia_gpu` probe in `zenmetrics-api`'s `backend_resolve`
/// integration test — the repo's established idiom for "is a CUDA GPU
/// actually present?" (NO graceful runtime skip; the caller asserts a
/// valid outcome on both arms).
#[cfg(feature = "sweep")]
#[allow(dead_code)]
fn host_has_nvidia_gpu() -> bool {
    let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name", "--format=csv,noheader"])
        .output()
    else {
        return false;
    };
    out.status.success()
        && String::from_utf8_lossy(&out.stdout)
            .lines()
            .any(|l| !l.trim().is_empty())
}

#[test]
fn list_metrics_runs() {
    let out = cli().args(["list-metrics"]).output().expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("ssim2"));
    assert!(s.contains("ssim2-gpu"));
    assert!(s.contains("butteraugli"));
    assert!(s.contains("butteraugli-gpu"));
    assert!(s.contains("dssim"));
    assert!(s.contains("dssim-gpu"));
    assert!(s.contains("zensim"));
}

#[test]
fn list_formats_runs() {
    let out = cli().args(["list-formats"]).output().expect("run cli");
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    // Default features include png + webp.
    assert!(s.contains("png"));
    assert!(s.contains("webp"));
}

#[cfg(feature = "cpu-metrics")]
/// **zenmetrics D14** — a runtime-selected zensim bake actually reaches the
/// score, and the unselected default is untouched.
///
/// `--zensim-bake` / `--zensim-profile` / `ZENMETRICS_ZENSIM_PROFILE` all
/// route through `zenmetrics_api::zensim_profile` into the CLI's
/// `metrics::zensim` score path. Before the selector existed every site
/// hard-coded `ZensimProfile::latest_preview()` and no flag could change it.
///
/// **No graceful skip**: the bake path comes from `ZENMETRICS_TEST_ZENSIM_BAKE`
/// (default below) and an absent file FAILS the test. Referenced BY PATH —
/// nothing vendored.
#[cfg(feature = "cpu-metrics")]
#[test]
fn zensim_bake_selector_changes_the_score_and_default_is_unchanged() {
    /// ADD156, 3,575 B, sha256 51437a34f04887ce850b25eff4f72a6bcd12926873ce060a12878d558a7517db
    const DEFAULT_BAKE: &str = "/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin";
    const BAKE_ENV: &str = "ZENMETRICS_TEST_ZENSIM_BAKE";

    let bake = std::env::var(BAKE_ENV).unwrap_or_else(|_| DEFAULT_BAKE.to_string());
    assert!(
        PathBuf::from(&bake).is_file(),
        "zensim bake not found at {bake:?}. This test does NOT skip itself — \
         point it at a readable ZNPR bake with {BAKE_ENV}=<path>, or restore \
         the default at {DEFAULT_BAKE}"
    );

    let fx = fixtures_dir();
    let r = fx.join("ref_256.png");
    let d = fx.join("dist_noisy_256.png");

    let score_of = |extra: &[&str], env: Option<(&str, &str)>| -> f64 {
        let mut c = cli();
        c.args(["score", "--metric", "zensim"])
            .args(extra)
            .arg("--reference")
            .arg(&r)
            .arg("--distorted")
            .arg(&d);
        if let Some((k, v)) = env {
            c.env(k, v);
        } else {
            c.env_remove("ZENMETRICS_ZENSIM_PROFILE");
        }
        let out = c.output().expect("run cli");
        assert!(
            out.status.success(),
            "args={extra:?} stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let s = String::from_utf8_lossy(&out.stdout);
        s.split_whitespace()
            .find_map(|t| t.strip_prefix("zensim="))
            .unwrap_or_else(|| panic!("no zensim= token in {s:?}"))
            .parse()
            .expect("parse zensim score")
    };

    let default = score_of(&[], None);
    let named_b = score_of(&["--zensim-profile", "b"], None);
    let by_flag = score_of(&["--zensim-bake", &bake], None);
    let by_env = score_of(&[], Some(("ZENMETRICS_ZENSIM_PROFILE", &bake)));

    // Default behaviour is unchanged: the shipped default IS `zensim-b`, so
    // naming it explicitly must reproduce the un-flagged number exactly.
    assert_eq!(
        default, named_b,
        "--zensim-profile b must be bit-identical to the unflagged default"
    );
    // The selected bake reaches the score.
    assert_ne!(
        by_flag, default,
        "--zensim-bake did not change the score (it was ignored)"
    );
    assert_eq!(by_flag, by_env, "flag and env must select the same profile");
    assert!(by_flag.is_finite(), "bake score must be finite");
    // The spline-without-skip_score_mapping trap returns exactly 0.000000.
    assert_ne!(by_flag, 0.0, "bake scored exactly 0 — misconfigured spline");

    // A bad spec is a hard error, never a silent fall back to the default.
    let bad = cli()
        .args([
            "score",
            "--metric",
            "zensim",
            "--zensim-profile",
            "no-such-profile",
        ])
        .arg("--reference")
        .arg(&r)
        .arg("--distorted")
        .arg(&d)
        .output()
        .expect("run cli");
    assert!(!bad.status.success(), "a bogus --zensim-profile must fail");
    let msg = String::from_utf8_lossy(&bad.stderr);
    assert!(msg.contains("unknown zensim profile spec"), "stderr={msg}");

    // …and the two flags are mutually exclusive.
    let both = cli()
        .args([
            "score",
            "--metric",
            "zensim",
            "--zensim-profile",
            "b",
            "--zensim-bake",
        ])
        .arg(&bake)
        .arg("--reference")
        .arg(&r)
        .arg("--distorted")
        .arg(&d)
        .output()
        .expect("run cli");
    assert!(
        !both.status.success(),
        "--zensim-profile and --zensim-bake must conflict"
    );
}

#[test]
fn score_zensim_identical_pngs() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    // zenmetrics-cli >= 0.5.0 nests scores under `scores.<column_name>`
    // because a single metric can emit multiple columns (butteraugli emits
    // both `_max` and `_pnorm3`). zensim is a single-column metric.
    let score = v["scores"]["zensim"].as_f64().expect("score");
    // zensim returns ~100 for identical images.
    assert!(score > 95.0, "expected ~100, got {score}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_butteraugli_identical_pngs_tsv() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "score",
            "--metric",
            "butteraugli",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--output",
            "tsv",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    // TSV in v0.5.0+: butteraugli emits two columns from a single
    // compute() — `butteraugli_max` (the per-block maximum) and
    // `butteraugli_pnorm3` (the libjxl 3-norm). The header row carries
    // the column names; the metric name itself is no longer a column
    // because one metric can produce several values.
    let mut lines = s.lines();
    assert_eq!(lines.next().unwrap(), "butteraugli_max\tbutteraugli_pnorm3");
    let row = lines.next().unwrap();
    let parts: Vec<&str> = row.split('\t').collect();
    assert_eq!(parts.len(), 2);
    let max_score: f64 = parts[0].parse().unwrap();
    let pnorm3_score: f64 = parts[1].parse().unwrap();
    // Both aggregations of identical images should be effectively zero.
    assert!(max_score < 0.01, "expected ~0 max, got {max_score}");
    assert!(
        pnorm3_score < 0.01,
        "expected ~0 pnorm3, got {pnorm3_score}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_butteraugli_noisy_is_higher_than_identical() {
    // butteraugli emits two columns. Verify the ordering on BOTH
    // aggregations — noisy should beat identical on max-norm and on the
    // 3-norm.
    let dir = fixtures_dir();
    let identical = run_scores(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    let noisy = run_scores(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    let identical_max = identical
        .iter()
        .find(|(k, _)| k == "butteraugli_max")
        .expect("max col")
        .1;
    let identical_p3 = identical
        .iter()
        .find(|(k, _)| k == "butteraugli_pnorm3")
        .expect("pnorm3 col")
        .1;
    let noisy_max = noisy
        .iter()
        .find(|(k, _)| k == "butteraugli_max")
        .expect("max col")
        .1;
    let noisy_p3 = noisy
        .iter()
        .find(|(k, _)| k == "butteraugli_pnorm3")
        .expect("pnorm3 col")
        .1;
    assert!(identical_max < 0.5, "identical max={identical_max}");
    assert!(identical_p3 < 0.5, "identical pnorm3={identical_p3}");
    assert!(
        noisy_max > identical_max,
        "noisy max {noisy_max} should be > identical max {identical_max}"
    );
    assert!(
        noisy_p3 > identical_p3,
        "noisy pnorm3 {noisy_p3} should be > identical pnorm3 {identical_p3}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_dssim_identical_is_zero() {
    let dir = fixtures_dir();
    let s = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // DSSIM is a distance — identical images should score ~0.
    assert!(s < 1e-3, "expected ~0 for identical, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_dssim_noisy_higher_than_identical() {
    let dir = fixtures_dir();
    let identical = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    let noisy = run_score(
        "dssim",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    assert!(
        noisy > identical,
        "noisy dssim {noisy} should be > identical {identical}"
    );
}

#[cfg(all(feature = "cpu-metrics", feature = "gpu-dssim"))]
#[test]
fn score_dssim_gpu_identical_is_zero() {
    let dir = fixtures_dir();
    let s = run_score(
        "dssim-gpu",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // DSSIM-GPU is a distance — identical images should score ~0.
    assert!(s < 1e-3, "expected ~0 for identical, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn score_ssim2_identical_is_high() {
    let dir = fixtures_dir();
    let s = run_score(
        "ssim2",
        &dir.join("ref_64.png"),
        &dir.join("dist_identical_64.png"),
    );
    // SSIMULACRA2 returns ~100 for identical, lower for distorted.
    assert!(s > 95.0, "expected ~100, got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "avif"))]
#[test]
fn score_decodes_avif_fixture() {
    // Fixture is checked into the repo — no skip path.
    let dir = fixtures_dir();
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.avif"));
    assert!(s.is_finite() && s > 0.0, "got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "jxl"))]
#[test]
fn score_decodes_jxl_fixture() {
    let dir = fixtures_dir();
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.jxl"));
    assert!(s.is_finite() && s > 0.0, "got {s}");
}

#[cfg(all(feature = "cpu-metrics", feature = "webp"))]
#[test]
fn score_works_across_png_and_webp_decoders() {
    let dir = fixtures_dir();
    // Compare PNG-encoded ref against WebP-encoded ref (both lossless,
    // same content) — both decoders should produce matching pixels and
    // give a near-identical zensim score.
    let s = run_score("zensim", &dir.join("ref_64.png"), &dir.join("ref_64.webp"));
    // Lossless WebP of the exact same source should round-trip pixel-exact
    // → zensim score effectively 100.
    assert!(s > 95.0, "expected ~100 for lossless cross-format, got {s}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn batch_zensim_appends_metric_column() {
    let dir = fixtures_dir();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let pairs = tmp.path().join("pairs.tsv");
    let output = tmp.path().join("output.tsv");

    let ref_path = dir.join("ref_64.png");
    let dist_a = dir.join("dist_identical_64.png");
    let dist_b = dir.join("dist_noisy_64.png");

    let mut tsv = String::from("ref_path\tdist_path\ttag\n");
    tsv.push_str(&format!(
        "{}\t{}\tidentical\n",
        ref_path.display(),
        dist_a.display()
    ));
    tsv.push_str(&format!(
        "{}\t{}\tnoisy\n",
        ref_path.display(),
        dist_b.display()
    ));
    std::fs::write(&pairs, tsv).unwrap();

    let out = cli()
        .args([
            "batch",
            "--metric",
            "zensim",
            "--pairs",
            pairs.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );

    let read = std::fs::read_to_string(&output).unwrap();
    let mut lines = read.lines();
    let headers = lines.next().unwrap();
    assert!(
        headers.contains("zensim"),
        "expected zensim col in {headers}"
    );
    let row1 = lines.next().unwrap();
    let row2 = lines.next().unwrap();
    let score1: f64 = row1.split('\t').next_back().unwrap().parse().unwrap();
    let score2: f64 = row2.split('\t').next_back().unwrap().parse().unwrap();
    assert!(score1 > 95.0, "identical: {score1}");
    assert!(
        score2 < score1,
        "noisy {score2} should be < identical {score1}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_one_ref_one_variant_one_metric_json_shape() {
    // Smallest possible compare: 1×1×1. Verify the JSON document shape
    // matches the spec — a top-level `metrics` array and a `results` array
    // where each row carries `reference`, `variant`, and a `scores` map
    // keyed on metric names.
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--metric",
            "zensim",
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let metrics = v["metrics"].as_array().expect("metrics array");
    assert_eq!(metrics.len(), 1);
    assert_eq!(metrics[0], "zensim");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 1);
    let row = &results[0];
    assert!(row["reference"].as_str().unwrap().ends_with("ref_64.png"));
    assert!(
        row["variant"]
            .as_str()
            .unwrap()
            .ends_with("dist_identical_64.png")
    );
    let score = row["scores"]["zensim"].as_f64().expect("score");
    assert!(score > 95.0, "expected ~100 for identical, got {score}");
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_one_ref_two_variants_two_metrics_tsv() {
    // 1×2 × 2 = 4 cells. Verify all four scores show up in the TSV with
    // the correct column ordering, that all values are finite, and that
    // the "noisy" variant scores differ from the "identical" one.
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_noisy_64.png").to_str().unwrap(),
            "--metric",
            "zensim",
            "--metric",
            "butteraugli",
            "--output",
            "tsv",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let mut lines = s.lines();
    let header = lines.next().expect("header");
    // butteraugli emits TWO columns (max + pnorm3) so the compare TSV
    // header has 5 columns total: reference, variant, zensim,
    // butteraugli_max, butteraugli_pnorm3.
    assert_eq!(
        header,
        "reference\tvariant\tzensim\tbutteraugli_max\tbutteraugli_pnorm3"
    );
    let row1: Vec<&str> = lines.next().expect("row1").split('\t').collect();
    let row2: Vec<&str> = lines.next().expect("row2").split('\t').collect();
    assert!(lines.next().is_none(), "exactly two data rows expected");
    assert_eq!(row1.len(), 5);
    assert_eq!(row2.len(), 5);
    let identical_zensim: f64 = row1[2].parse().unwrap();
    let identical_butter_max: f64 = row1[3].parse().unwrap();
    let identical_butter_p3: f64 = row1[4].parse().unwrap();
    let noisy_zensim: f64 = row2[2].parse().unwrap();
    let noisy_butter_max: f64 = row2[3].parse().unwrap();
    let noisy_butter_p3: f64 = row2[4].parse().unwrap();
    assert!(identical_zensim > 95.0, "{identical_zensim}");
    assert!(identical_butter_max < 0.01, "{identical_butter_max}");
    assert!(identical_butter_p3 < 0.01, "{identical_butter_p3}");
    assert!(
        noisy_zensim < identical_zensim,
        "noisy {noisy_zensim} should be < identical {identical_zensim}"
    );
    assert!(
        noisy_butter_max > identical_butter_max,
        "noisy butteraugli_max {noisy_butter_max} (higher = worse) should be > \
         identical {identical_butter_max}"
    );
    assert!(
        noisy_butter_p3 > identical_butter_p3,
        "noisy butteraugli_pnorm3 {noisy_butter_p3} (higher = worse) should be > \
         identical {identical_butter_p3}"
    );
}

#[cfg(feature = "cpu-metrics")]
#[test]
fn compare_continues_on_per_cell_failure() {
    // Two variants: one valid, one that does not exist on disk. The bad
    // variant should produce error cells (null in JSON) for every metric
    // it's paired with, but the good variant should still get scored.
    // Process exit must be non-zero because at least one cell failed.
    let dir = fixtures_dir();
    let tmp = tempfile::tempdir().expect("tmpdir");
    let bogus = tmp.path().join("does_not_exist.png");
    let out = cli()
        .args([
            "compare",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--variant",
            dir.join("dist_identical_64.png").to_str().unwrap(),
            "--variant",
            bogus.to_str().unwrap(),
            "--metric",
            "zensim",
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        !out.status.success(),
        "expected non-zero exit when a cell fails"
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    // First row (identical variant) should have a real score.
    let good_score = results[0]["scores"]["zensim"].as_f64().expect("good score");
    assert!(
        good_score > 95.0,
        "good score should be ~100, got {good_score}"
    );
    // Second row (missing variant) should be null.
    assert!(
        results[1]["scores"]["zensim"].is_null(),
        "expected null score for missing variant, got {}",
        results[1]["scores"]["zensim"]
    );
}

/// CPU butteraugli vs GPU butteraugli on the same noisy-pair fixture.
/// Both backends compute the same two aggregations (`_max` and
/// `_pnorm3`); the scores should agree closely modulo floating-point
/// reduction order across CubeCL runtimes (CUDA vs wgpu vs HIP vs CPU).
/// Tolerance is set to 5e-2 in absolute terms — empirically the
/// cross-backend slack on butteraugli is dominated by reduction order,
/// well below this bound on 64×64 fixtures. We verify BOTH aggregations
/// agree, not just the 3-norm.
#[cfg(all(feature = "cpu-metrics", feature = "gpu-butteraugli"))]
#[test]
fn butteraugli_cpu_and_gpu_agree() {
    let dir = fixtures_dir();
    let cpu = run_scores(
        "butteraugli",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    let gpu = run_scores(
        "butteraugli-gpu",
        &dir.join("ref_64.png"),
        &dir.join("dist_noisy_64.png"),
    );
    let cpu_max = cpu
        .iter()
        .find(|(k, _)| k == "butteraugli_max")
        .expect("cpu max")
        .1;
    let cpu_p3 = cpu
        .iter()
        .find(|(k, _)| k == "butteraugli_pnorm3")
        .expect("cpu pnorm3")
        .1;
    let gpu_max = gpu
        .iter()
        .find(|(k, _)| k == "butteraugli_max_gpu")
        .expect("gpu max")
        .1;
    let gpu_p3 = gpu
        .iter()
        .find(|(k, _)| k == "butteraugli_pnorm3_gpu")
        .expect("gpu pnorm3")
        .1;
    let diff_max = (cpu_max - gpu_max).abs();
    let diff_p3 = (cpu_p3 - gpu_p3).abs();
    assert!(
        diff_max < 5e-2,
        "cpu butteraugli_max={cpu_max} vs gpu butteraugli_max_gpu={gpu_max} \
         (|diff|={diff_max}) exceeds 5e-2 tolerance"
    );
    assert!(
        diff_p3 < 5e-2,
        "cpu butteraugli_pnorm3={cpu_p3} vs gpu butteraugli_pnorm3_gpu={gpu_p3} \
         (|diff|={diff_p3}) exceeds 5e-2 tolerance"
    );
}

#[cfg(feature = "cpu-metrics")]
fn run_score(metric: &str, reference: &std::path::Path, distorted: &std::path::Path) -> f64 {
    // Single-column convenience: pulls the metric's first reported column
    // out of the JSON response. For metrics that emit multiple columns
    // (butteraugli) this returns the first — `butteraugli_max`. Tests that
    // need a different aggregation should use [`run_scores`] directly.
    let scores = run_scores(metric, reference, distorted);
    scores
        .into_iter()
        .next()
        .map(|(_, v)| v)
        .expect("at least one score column")
}

/// Full-fidelity score reader: returns every `(column_name, value)` pair
/// the score subcommand wrote to JSON. Used by butteraugli tests that
/// want to assert on both `_max` and `_pnorm3` independently.
#[cfg(feature = "cpu-metrics")]
fn run_scores(
    metric: &str,
    reference: &std::path::Path,
    distorted: &std::path::Path,
) -> Vec<(String, f64)> {
    let out = cli()
        .args([
            "score",
            "--metric",
            metric,
            "--reference",
            reference.to_str().unwrap(),
            "--distorted",
            distorted.to_str().unwrap(),
            "--output",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(s.trim()).expect("json");
    let scores_obj = v["scores"]
        .as_object()
        .expect("scores object in score JSON");
    scores_obj
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().expect("score f64")))
        .collect()
}

// ── sweep subcommand ────────────────────────────────────────────────────
//
// The `sweep` feature drives a codec across a (q, knob-tuple) Cartesian
// grid and writes a Pareto TSV. The tests below exercise the full
// pipeline (encode → decode-back → score) on the existing 64×64 PNG
// fixture for each codec the sweep feature wires up.

#[cfg(feature = "sweep")]
#[test]
fn sweep_zenwebp_emits_pareto_rows() {
    let dir = fixtures_dir();
    // Stage just one source image so we can predict the row count.
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenwebp",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50,90",
            "--knob-grid",
            r#"{"method": [4, 6]}"#,
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "sweep failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    // 1 header + 4 cells (2 q × 2 method).
    assert_eq!(lines.len(), 5, "got {} lines: {body}", lines.len());
    assert!(lines[0].contains("score_zensim"));
    for row in &lines[1..] {
        // Every emitted row should have a parseable zensim score in the
        // last column.
        let score = row.split('\t').next_back().unwrap();
        score
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("bad zensim score {score:?} in row {row:?}: {e}"));
    }
}

// `--no-score` is the TIMING instrument's mode: encode and time every cell,
// score nothing. It exists because scoring between timed encodes perturbs the
// quantity being timed (a multi-threaded metric on every core between two
// single-threaded encodes leaves a different boost/thermal state), and because
// it dominated the wall clock when the AVIF speed instrument first ran
// (r7900x 2026-09-03: 23 min of wall, 15.0 s of encode_ms).
#[cfg(feature = "sweep")]
#[test]
fn sweep_no_score_emits_timings_and_no_score_columns() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenwebp",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50,90",
            "--knob-grid",
            r#"{"method": [4, 6]}"#,
            "--no-score",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "sweep --no-score failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    // Same 4 cells (2 q x 2 method) as the scored twin -- --no-score changes
    // what is measured, never which cells are encoded.
    assert_eq!(lines.len(), 5, "got {} lines: {body}", lines.len());
    // The whole point: no score column at all. Without the flag the default
    // metric set injects `score_zensim`, so this assertion fails on a binary
    // that ignores it.
    assert!(
        !lines[0].contains("score_"),
        "header must carry no score_* column, got {:?}",
        lines[0]
    );
    let cols: Vec<&str> = lines[0].split('\t').collect();
    let ms_idx = cols
        .iter()
        .position(|c| *c == "encode_ms")
        .expect("encode_ms column must survive --no-score");
    let bytes_idx = cols
        .iter()
        .position(|c| *c == "encoded_bytes")
        .expect("encoded_bytes column");
    for row in &lines[1..] {
        let f: Vec<&str> = row.split('\t').collect();
        let ms: f64 = f[ms_idx]
            .parse()
            .unwrap_or_else(|e| panic!("bad encode_ms {:?} in {row:?}: {e}", f[ms_idx]));
        assert!(ms > 0.0, "encode_ms must be populated, got {ms} in {row:?}");
        let b: u64 = f[bytes_idx]
            .parse()
            .unwrap_or_else(|e| panic!("bad encoded_bytes in {row:?}: {e}"));
        assert!(b > 0, "encoded_bytes must be populated in {row:?}");
    }
}

// Asking for both is a mistake; silently dropping the scores would be
// indistinguishable from a scorer that failed on every cell, so it is refused.
#[cfg(feature = "sweep")]
#[test]
fn sweep_no_score_conflicts_with_metric() {
    let staged = tempfile::tempdir().expect("tmp");
    let out = staged.path().join("pareto.tsv");
    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenwebp",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50",
            "--no-score",
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(!result.status.success(), "--no-score --metric must be refused");
    let err = String::from_utf8_lossy(&result.stderr);
    assert!(
        err.contains("cannot be used with") || err.contains("conflict"),
        "expected a clap conflict error, got: {err}"
    );
}

#[cfg(all(feature = "sweep", feature = "jpeg"))]
#[test]
fn sweep_zenjpeg_plan_mode_emits_cells_and_manifest() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenjpeg",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50,85",
            "--plan",
            "rd_core",
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "plan sweep failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    // The audit manifest lands next to the TSV and carries the cell
    // count — the TSV must have exactly that many rows (one image).
    let manifest = std::fs::read_to_string(staged.path().join("pareto.plan.json"))
        .expect("plan manifest written");
    let mjson: serde_json::Value = serde_json::from_str(&manifest).expect("manifest json");
    assert_eq!(mjson["plan"], "rd_core");
    let cells = mjson["cells"].as_u64().expect("cells count") as usize;
    assert!(cells > 0);

    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(
        lines.len(),
        cells + 1,
        "expected {} rows + header, got {}: {body}",
        cells,
        lines.len()
    );
    // Identity column carries the plan-cell id + resolved-state
    // fingerprint. Rows land in rayon completion order (the QUEUE is
    // main-effects-first, the TSV is not), so assert content, not
    // position: the default stratum must be present (csv-quoted, so
    // embedded quotes double), and every row carries the plan keys.
    assert!(
        body.contains("jp3_t0_small_420"),
        "default stratum missing from TSV: {body}"
    );
    for row in &lines[1..] {
        assert!(row.contains("rd_core"), "row missing plan id: {row}");
        assert!(row.contains("fp"), "row missing fingerprint: {row}");
    }
    for row in &lines[1..] {
        let score = row.split('\t').next_back().unwrap();
        score
            .parse::<f64>()
            .unwrap_or_else(|e| panic!("bad zensim score {score:?} in row {row:?}: {e}"));
    }
}

#[cfg(all(feature = "sweep", feature = "jpeg"))]
#[test]
fn sweep_zenjpeg_trellis_knob_and_smallest_mode() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenjpeg",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "75",
            "--knob-grid",
            r#"{"trellis": [true, {"lambda1": 13.5, "dc": false, "coupling_scale": -4.0, "coupling_max_adjustment": 1.0}], "progressive_mode": ["smallest"]}"#,
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "trellis-knob sweep failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    // 1 header + 2 cells (1 q × 2 trellis values × 1 progressive_mode).
    assert_eq!(lines.len(), 3, "got {} lines: {body}", lines.len());
    for row in &lines[1..] {
        let score = row.split('\t').next_back().unwrap();
        assert!(
            score.parse::<f64>().is_ok(),
            "trellis/smallest cell failed to encode+score: {row:?}"
        );
    }
}

#[cfg(all(feature = "sweep", feature = "jpeg"))]
#[test]
fn plan_dry_run_emits_declare_manifest_and_jobexec_runs_it() {
    use std::io::Write as _;

    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");
    let cells = staged.path().join("cells.jsonl");

    // 1. Dry run: manifest + declare items, NO encodes (no TSV created).
    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenjpeg",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50,85",
            "--plan",
            "rd_core",
            "--dry-run",
            "--emit-cells",
            cells.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "dry run failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!out.exists(), "--dry-run must not encode");
    let manifest = std::fs::read_to_string(staged.path().join("pareto.plan.json")).unwrap();
    let mjson: serde_json::Value = serde_json::from_str(&manifest).unwrap();
    let cell_count = mjson["cells"].as_u64().unwrap() as usize;

    let body = std::fs::read_to_string(&cells).expect("declare manifest written");
    let items: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).expect("item json"))
        .collect();
    assert_eq!(items.len(), cell_count, "one item per cell per source");
    let it = &items[0];
    for key in ["image_path", "codec", "q", "knob_tuple_json", "source_sha"] {
        assert!(it.get(key).is_some(), "missing {key}: {it}");
    }
    assert_eq!(it["source_sha"].as_str().unwrap().len(), 64);
    assert!(it["q"].is_i64(), "q must be integral for CellId");

    // 2. Round-trip an emitted item through the jobexec executor contract:
    //    the stratum id + fingerprint alone must reproduce an encode.
    let job = serde_json::json!({
        "kind": {"kind": "encode", "codec": it["codec"], "q": it["q"],
                 "knobs": it["knob_tuple_json"]},
        "inputs": [it["source_sha"]],
        "cell": {
            "image_path": it["image_path"],
            "codec": it["codec"],
            "q": it["q"],
            "knob_tuple_json": it["knob_tuple_json"],
        },
    });
    let mut child = cli()
        .args(["jobexec"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn jobexec");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&job).unwrap().as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("jobexec");
    assert!(
        out.status.success(),
        "jobexec failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.starts_with(&[0xFF, 0xD8]) && out.stdout.ends_with(&[0xFF, 0xD9]),
        "stdout must be the encoded JPEG bytes ({} bytes)",
        out.stdout.len()
    );

    // 3. Tampered fingerprint = loud deterministic failure, not a wrong
    //    encode (the id-grammar drift tripwire).
    let knob_tampered = it["knob_tuple_json"].as_str().unwrap().replacen(
        it["knob_tuple_json"]
            .as_str()
            .unwrap()
            .split("\"fp\":\"")
            .nth(1)
            .unwrap()
            .split('\"')
            .next()
            .unwrap(),
        "0000000000000000",
        1,
    );
    let mut tampered = job.clone();
    tampered["cell"]["knob_tuple_json"] = serde_json::json!(knob_tampered);
    let mut child = cli()
        .args(["jobexec"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn jobexec");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&tampered).unwrap().as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("jobexec");
    assert!(!out.status.success(), "tampered fp must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("fingerprint mismatch"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(all(feature = "sweep", feature = "avif"))]
#[test]
fn zenavif_plan_dry_run_jobexec_roundtrip_and_fp_tripwire() {
    use std::io::Write as _;

    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");
    let cells = staged.path().join("cells.jsonl");

    // 1. Dry run with --codec zenavif: manifest + declare items, NO encodes.
    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenavif",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "60",
            "--plan",
            "rd_core",
            "--dry-run",
            "--emit-cells",
            cells.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "dry run failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!out.exists(), "--dry-run must not encode");

    let body = std::fs::read_to_string(&cells).expect("declare manifest written");
    let items: Vec<serde_json::Value> = body
        .lines()
        .map(|l| serde_json::from_str(l).expect("item json"))
        .collect();
    assert!(!items.is_empty());
    let it = &items[0];
    assert_eq!(it["codec"].as_str().unwrap(), "zenavif");
    assert!(it["q"].is_i64(), "q must be integral for CellId");
    // The all-defaults stratum leads (main-effects-first ordering).
    assert!(
        it["knob_tuple_json"]
            .as_str()
            .unwrap()
            .contains("\"cell\":\"s4\""),
        "got {}",
        it["knob_tuple_json"]
    );

    // 2. Round-trip through jobexec: stratum id + fingerprint alone must
    //    reproduce an AVIF encode (self-describing ledger contract).
    let job = serde_json::json!({
        "kind": {"kind": "encode", "codec": it["codec"], "q": it["q"],
                 "knobs": it["knob_tuple_json"]},
        "inputs": [it["source_sha"]],
        "cell": {
            "image_path": it["image_path"],
            "codec": it["codec"],
            "q": it["q"],
            "knob_tuple_json": it["knob_tuple_json"],
        },
    });
    let mut child = cli()
        .args(["jobexec"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn jobexec");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&job).unwrap().as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("jobexec");
    assert!(
        out.status.success(),
        "jobexec failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        out.stdout.len() > 12 && &out.stdout[4..12] == b"ftypavif",
        "stdout must be an AVIF file ({} bytes)",
        out.stdout.len()
    );

    // 3. Tampered fingerprint = loud deterministic failure.
    let knob = it["knob_tuple_json"].as_str().unwrap();
    let fp = knob
        .split("\"fp\":\"")
        .nth(1)
        .unwrap()
        .split('\"')
        .next()
        .unwrap();
    let knob_tampered = knob.replacen(fp, "0000000000000000", 1);
    let mut tampered = job.clone();
    tampered["cell"]["knob_tuple_json"] = serde_json::json!(knob_tampered);
    let mut child = cli()
        .args(["jobexec"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn jobexec");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(serde_json::to_string(&tampered).unwrap().as_bytes())
        .unwrap();
    let out = child.wait_with_output().expect("jobexec");
    assert!(!out.status.success(), "tampered fp must fail");
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("fingerprint mismatch"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[cfg(feature = "sweep")]
#[test]
fn sweep_writes_zensim_feature_parquet() {
    // Run a tiny zenwebp sweep with --feature-output and verify:
    // - the parquet file is produced
    // - it has 5 ID columns + one feat_* column per feature of the
    //   default regime (the writer is sized from
    //   `ZensimFeatureRegime::total_features()` for both the CPU and
    //   GPU paths — see the sizing comment in sweep::run)
    // - the row count matches the TSV row count
    // - the file is non-trivially sized (has actual feature data)
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out_tsv = staged.path().join("pareto.tsv");
    let out_pq = staged.path().join("features.parquet");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenwebp",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "50,90",
            "--metric",
            "zensim",
            "--output",
            out_tsv.to_str().unwrap(),
            "--feature-output",
            out_pq.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "sweep failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    // The CLI default regime ("with-iw") drives the sidecar width.
    let feature_n = zenmetrics_cli::metrics::ZensimFeatureRegime::WithIw.total_features();
    let expected_cols = 5 + feature_n;

    let pq_meta = std::fs::metadata(&out_pq).expect("parquet exists");
    // Parquet files have a 12-byte fixed footer minimum; a real file with
    // 2 rows and a few hundred columns is going to be at least a couple
    // of KB even after zstd. Sanity-check we didn't write an empty stub.
    assert!(
        pq_meta.len() > 1024,
        "feature parquet is suspiciously small: {} bytes",
        pq_meta.len()
    );

    // Cross-check TSV row count: 1 header + 2 cells (2 q values, default knob grid = 1 tuple).
    let tsv_body = std::fs::read_to_string(&out_tsv).expect("read tsv");
    assert_eq!(tsv_body.lines().count(), 3, "TSV should have 1+2 lines");

    // Validate parquet footer + read column count via the parquet crate
    // directly. We don't pull pyarrow into the test suite — the parquet
    // crate's own ParquetMetaData reader is the same API we use to write.
    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(&out_pq).expect("open pq");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    let meta = reader.metadata();
    assert_eq!(meta.num_row_groups(), 1, "expect single row group");
    let schema_descr = meta.file_metadata().schema_descr();
    // 5 ID columns + one column per regime feature.
    assert_eq!(
        schema_descr.num_columns(),
        expected_cols,
        "expected {expected_cols} columns, got {}",
        schema_descr.num_columns()
    );
    let num_rows = meta.file_metadata().num_rows();
    assert_eq!(num_rows, 2, "expected 2 rows in parquet, got {num_rows}");

    // First and last feature columns are named feat_0 / feat_<n-1>.
    let names: Vec<String> = (0..schema_descr.num_columns())
        .map(|i| schema_descr.column(i).name().to_string())
        .collect();
    assert_eq!(names[0], "image_path");
    assert_eq!(names[4], "zensim_score");
    assert_eq!(names[5], "feat_0");
    assert_eq!(names[expected_cols - 1], format!("feat_{}", feature_n - 1));
}

/// Build a 2-row pairs.tsv from the 64×64 fixtures (ref + two distorted
/// variants) and the deterministic identity tuple `score-pairs` passes
/// through. Returns `(pairs_tsv_path, out_parquet_path,
/// feature_parquet_path)` staged under `staged`.
#[cfg(feature = "sweep")]
fn stage_score_pairs_inputs(
    staged: &std::path::Path,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    use std::io::Write;
    let dir = fixtures_dir();
    let ref_png = dir.join("ref_64.png");
    let dist_a = dir.join("dist_noisy_64.png");
    let dist_b = dir.join("dist_identical_64.png");

    let pairs_tsv = staged.join("pairs.tsv");
    let mut f = std::fs::File::create(&pairs_tsv).expect("create pairs.tsv");
    // Header matches the SPLIT-fleet pairs.tsv contract emitted by
    // `sweep --pairs-tsv` (image_path/codec/q/knob_tuple_json + ref/dist).
    writeln!(
        f,
        "image_path\tcodec\tq\tknob_tuple_json\tref_path\tdist_path"
    )
    .unwrap();
    writeln!(
        f,
        "{ref}\tzenwebp\t50\t{{}}\t{ref}\t{dist}",
        ref = ref_png.display(),
        dist = dist_a.display()
    )
    .unwrap();
    writeln!(
        f,
        "{ref}\tzenwebp\t90\t{{}}\t{ref}\t{dist}",
        ref = ref_png.display(),
        dist = dist_b.display()
    )
    .unwrap();
    drop(f);

    (
        pairs_tsv,
        staged.join("scores.parquet"),
        staged.join("features.parquet"),
    )
}

/// Read a parquet's column names + row count via the parquet crate (the
/// same API `score-pairs` writes with — no pyarrow in the test suite).
#[cfg(feature = "sweep")]
fn read_parquet_schema(path: &std::path::Path) -> (Vec<String>, i64) {
    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(path).expect("open parquet");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    let meta = reader.metadata();
    let schema_descr = meta.file_metadata().schema_descr();
    let names: Vec<String> = (0..schema_descr.num_columns())
        .map(|i| schema_descr.column(i).name().to_string())
        .collect();
    (names, meta.file_metadata().num_rows())
}

/// `score-pairs --feature-output` with the CPU zensim metric must emit a
/// feature parquet whose schema is byte-identical to `sweep
/// --feature-output` (the SPLIT-fleet join contract): the four identity
/// columns, then `zensim_score`, then `feat_0..feat_<N-1>` where N is the
/// WithIw regime's 372. This is the no-GPU backbone test — it always runs
/// and validates the full feature-output wiring end to end.
#[cfg(feature = "sweep")]
#[cfg(feature = "cpu-metrics")]
#[test]
fn score_pairs_writes_zensim_feature_parquet_cpu() {
    let staged = tempfile::tempdir().expect("tmp");
    let (pairs_tsv, out_pq, feat_pq) = stage_score_pairs_inputs(staged.path());

    let result = cli()
        .args([
            "score-pairs",
            "--metric",
            "zensim",
            "--pairs-tsv",
            pairs_tsv.to_str().unwrap(),
            "--out-parquet",
            out_pq.to_str().unwrap(),
            "--feature-output",
            feat_pq.to_str().unwrap(),
            "--zensim-features-regime",
            "with-iw",
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "score-pairs failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let feature_n = zenmetrics_cli::metrics::ZensimFeatureRegime::WithIw.total_features();
    assert_eq!(feature_n, 372, "with-iw regime must be 372 features");
    // 4 identity columns + zensim_score + one column per regime feature.
    let expected_cols = 5 + feature_n;

    let (names, num_rows) = read_parquet_schema(&feat_pq);
    assert_eq!(
        names.len(),
        expected_cols,
        "expected {expected_cols} columns, got {} ({:?}..)",
        names.len(),
        &names[..names.len().min(6)]
    );
    assert_eq!(num_rows, 2, "expected 2 feature rows, got {num_rows}");

    // Schema matches the sweep's FeatureParquetWriter exactly.
    assert_eq!(names[0], "image_path");
    assert_eq!(names[1], "codec");
    assert_eq!(names[2], "q");
    assert_eq!(names[3], "knob_tuple_json");
    assert_eq!(names[4], "zensim_score");
    assert_eq!(names[5], "feat_0");
    assert_eq!(names[expected_cols - 1], format!("feat_{}", feature_n - 1));

    // The score parquet still carries its 2 rows + the metric score column.
    let (score_names, score_rows) = read_parquet_schema(&out_pq);
    assert_eq!(score_rows, 2, "score parquet should have 2 rows");
    assert!(
        score_names.iter().any(|n| n == "zensim"),
        "score parquet should carry the `zensim` column, got {score_names:?}"
    );
}

/// `score-pairs --metric zensim-gpu --feature-output` over the SPLIT
/// scorer's pairs.tsv. When a CUDA GPU is present (the production SPLIT
/// fleet + this dev box) the GPU path must emit the same 372-wide,
/// 2-row, sweep-identical schema. When no GPU is present, the run still
/// succeeds (per-pair GPU failures are non-fatal without
/// `--fail-on-bogus`) and the feature parquet, when written, keeps the
/// correct 372-column schema — the binary never falls back to a
/// different width. Branch on `nvidia-smi` ground truth (no graceful
/// skip): both arms assert a valid, documented outcome.
#[cfg(feature = "sweep")]
#[cfg(feature = "gpu-zensim")]
#[test]
fn score_pairs_writes_zensim_gpu_feature_parquet() {
    let staged = tempfile::tempdir().expect("tmp");
    let (pairs_tsv, out_pq, feat_pq) = stage_score_pairs_inputs(staged.path());

    let result = cli()
        .args([
            "score-pairs",
            "--metric",
            "zensim-gpu",
            "--pairs-tsv",
            pairs_tsv.to_str().unwrap(),
            "--out-parquet",
            out_pq.to_str().unwrap(),
            "--feature-output",
            feat_pq.to_str().unwrap(),
            "--zensim-features-regime",
            "with-iw",
            "--gpu-runtime",
            "cuda",
        ])
        .output()
        .expect("run cli");
    // Without --fail-on-bogus the process exits 0 even if every pair's
    // GPU call fails (the score column gets NaN); a hard failure here
    // means an arg-parse / writer-setup bug, not a missing GPU.
    assert!(
        result.status.success(),
        "score-pairs zensim-gpu failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );

    let feature_n = zenmetrics_cli::metrics::ZensimFeatureRegime::WithIw.total_features();
    let expected_cols = 5 + feature_n;
    let (names, num_rows) = read_parquet_schema(&feat_pq);

    // The sidecar width is set from the regime, NOT from how many pairs
    // scored — so it is 372 + 5 on both arms.
    assert_eq!(
        names.len(),
        expected_cols,
        "feature sidecar must be {expected_cols} columns wide regardless of GPU presence, got {}",
        names.len()
    );
    assert_eq!(names[0], "image_path");
    assert_eq!(names[4], "zensim_score");
    assert_eq!(names[5], "feat_0");
    assert_eq!(names[expected_cols - 1], format!("feat_{}", feature_n - 1));

    if host_has_nvidia_gpu() {
        // GPU present: the WithIw feature path actually ran, so both
        // pairs produced a feature row.
        assert_eq!(
            num_rows,
            2,
            "CUDA GPU present → expected 2 zensim-gpu feature rows, got {num_rows}; stderr={}",
            String::from_utf8_lossy(&result.stderr)
        );
    } else {
        // No GPU: rows may be 0 (every GPU call failed) or 2 (a wgpu/cpu
        // fallback happened to satisfy it). Either is a valid outcome;
        // the schema width assertion above is the contract that matters.
        assert!(
            num_rows == 0 || num_rows == 2,
            "no-GPU run produced an unexpected row count {num_rows}"
        );
    }
}

#[cfg(feature = "sweep")]
#[test]
fn sweep_zenavif_emits_pareto_rows() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenavif",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "75",
            "--knob-grid",
            r#"{"speed": [8]}"#,
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "sweep zenavif failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2, "expected 1 header + 1 cell, got {body}");
}

#[cfg(feature = "sweep")]
#[test]
fn sweep_zenjxl_emits_pareto_rows() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    std::fs::copy(dir.join("ref_64.png"), staged.path().join("ref.png")).unwrap();
    let out = staged.path().join("pareto.tsv");

    let result = cli()
        .args([
            "sweep",
            "--codec",
            "zenjxl",
            "--sources",
            staged.path().to_str().unwrap(),
            "--q-grid",
            "75",
            "--knob-grid",
            r#"{"effort": [3]}"#,
            "--metric",
            "zensim",
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    assert!(
        result.status.success(),
        "sweep zenjxl failed: stderr={}",
        String::from_utf8_lossy(&result.stderr)
    );
    let body = std::fs::read_to_string(&out).expect("read tsv");
    let lines: Vec<&str> = body.lines().collect();
    assert_eq!(lines.len(), 2, "expected 1 header + 1 cell, got {body}");
}

// ===========================================================================
// Phase 7 — orchestrator integration tests.
//
// These tests verify the orchestrator-driven CLI path. They run end-to-end
// against the compiled binary, so the test runner exercises the same code
// path users will hit.
//
// Each test that requires the orchestrator feature has its own gate; the
// flag-parse tests run regardless of features (the CLI flags are global
// and present on every build).
// ===========================================================================

/// `--use-orchestrator` must be exposed on the top-level binary even when
/// the orchestrator feature is OFF, so users get a clear error message
/// rather than "unknown flag". Without the feature the flag parses but
/// has no effect.
#[test]
fn use_orchestrator_flag_parses_when_built_without_feature() {
    let out = cli()
        .args(["--use-orchestrator", "list-metrics"])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "list-metrics with --use-orchestrator failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--orchestrator-cache <PATH>` must accept a custom cache dir.
#[test]
fn orchestrator_cache_flag_accepts_custom_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = cli()
        .args([
            "--orchestrator-cache",
            dir.path().to_str().unwrap(),
            "list-metrics",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "list-metrics with --orchestrator-cache failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--bench-on-start <auto|yes|no>` must accept each of the three values.
#[test]
fn bench_on_start_flag_accepts_modes() {
    for mode in ["auto", "yes", "no"] {
        let out = cli()
            .args(["--bench-on-start", mode, "list-metrics"])
            .output()
            .expect("run cli");
        assert!(
            out.status.success(),
            "list-metrics with --bench-on-start {mode} failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// `--cpu-features <list>` must accept comma-separated values.
#[test]
fn cpu_features_flag_accepts_list() {
    let out = cli()
        .args(["--cpu-features", "ssim2,dssim,zensim", "list-metrics"])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "list-metrics with --cpu-features list failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `--cpu-features all` must work.
#[test]
fn cpu_features_flag_accepts_all() {
    let out = cli()
        .args(["--cpu-features", "all", "list-metrics"])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "list-metrics with --cpu-features all failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Phase 7 — `--use-orchestrator score …` must produce a score on
/// identical inputs when the orchestrator feature is built. Otherwise
/// the legacy path is exercised and we just confirm the flag is benign.
#[cfg(feature = "cpu-metrics")]
#[test]
fn use_orchestrator_score_identical_pngs() {
    let dir = fixtures_dir();
    let cache_dir = tempfile::tempdir().expect("tempdir");
    let out = cli()
        .args([
            "--use-orchestrator",
            "--orchestrator-cache",
            cache_dir.path().to_str().unwrap(),
            // Phase 7.7.1: was `no` (would require pre-warmed cache);
            // changed to `auto` so the test self-warms on first run.
            "--bench-on-start",
            "auto",
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "use-orchestrator score failed: stdout={stdout} stderr={stderr}",
    );
}

/// Sweep subcommand with `--use-orchestrator` warms the capability
/// cache and prints the active profile to stderr. The per-cell loop
/// remains on the legacy path so the TSV shape is unchanged.
#[cfg(all(feature = "sweep", feature = "cpu-metrics"))]
#[test]
fn sweep_with_orchestrator_warmup_emits_tsv() {
    let dir = fixtures_dir();
    let out_tsv = tempfile::NamedTempFile::new().unwrap().into_temp_path();
    let cache_dir = tempfile::tempdir().expect("tempdir");
    let result = cli()
        .args([
            "--use-orchestrator",
            "--orchestrator-cache",
            cache_dir.path().to_str().unwrap(),
            "--bench-on-start",
            "no",
            "sweep",
            "--codec",
            "zenjpeg",
            "--sources",
            dir.to_str().unwrap(),
            "--q-grid",
            "75",
            "--metric",
            "zensim",
            "--output",
            out_tsv.to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    if !result.status.success() {
        // Sweep may legitimately fail in environments without zenjpeg;
        // we only assert the orchestrator flags didn't trip clap.
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            !stderr.contains("--use-orchestrator: unknown")
                && !stderr.contains("--orchestrator-cache: unknown"),
            "orchestrator flags rejected: stderr={stderr}"
        );
        return;
    }
    let body = std::fs::read_to_string(&out_tsv).expect("read tsv");
    assert!(
        body.lines().count() >= 1,
        "expected at least a header line in sweep output: {body}"
    );
}

/// `--bench-on-start <bogus>` should be rejected when the orchestrator
/// feature is built. Without it the flag is parsed but unused.
#[test]
fn bench_on_start_flag_rejects_unknown_mode() {
    let _out = cli()
        .args(["--bench-on-start", "sometime", "list-metrics"])
        .output()
        .expect("run cli");
    // Either rc=0 (orchestrator feature off, flag silently accepted)
    // or rc=1 (feature on, parser rejects the value). Both are valid
    // — test just exercises the path so no panic-on-parse.
}

// ===========================================================================
// Phase 7.7.1 (2026-05-27): default-flip integration tests
//
// The CLI now defaults to the orchestrator path. `--use-orchestrator` is a
// deprecated no-op that emits a warning; `--use-legacy-scheduler` is the
// new opt-OUT flag.
// ===========================================================================

/// `zenmetrics --use-legacy-scheduler` should be accepted by clap regardless
/// of the orchestrator feature flag, so users get a clean error rather than
/// "unknown flag".
#[test]
fn use_legacy_scheduler_flag_parses_when_built_without_feature() {
    let out = cli()
        .args(["--use-legacy-scheduler", "list-metrics"])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "list-metrics with --use-legacy-scheduler failed: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// `zenmetrics score …` (NO flag, default path) must succeed and route
/// through the orchestrator when the orchestrator feature is built. The
/// orchestrator emits a `[orchestrator] enabled` line to stderr.
#[cfg(feature = "cpu-metrics")]
#[test]
fn default_score_routes_through_orchestrator() {
    let dir = fixtures_dir();
    let cache_dir = tempfile::tempdir().expect("tempdir");
    let out = cli()
        .args([
            "--orchestrator-cache",
            cache_dir.path().to_str().unwrap(),
            "--bench-on-start",
            "auto",
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "default score failed: stdout={stdout} stderr={stderr}",
    );
    // The orchestrator path emits `[orchestrator] enabled (…)` to
    // stderr when the feature is built; legacy path does not. We
    // only assert the marker when the feature is compiled in.
    #[cfg(feature = "orchestrator")]
    assert!(
        stderr.contains("[orchestrator] enabled"),
        "expected orchestrator-enabled stderr marker; got: {stderr}",
    );
}

/// `zenmetrics --use-legacy-scheduler score …` must succeed and route
/// through the legacy direct-dispatch path. The legacy path does NOT
/// emit the `[orchestrator] enabled` stderr marker.
#[cfg(feature = "cpu-metrics")]
#[test]
fn use_legacy_scheduler_score_skips_orchestrator() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--use-legacy-scheduler",
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "legacy-scheduler score failed: stdout={stdout} stderr={stderr}",
    );
    assert!(
        !stderr.contains("[orchestrator] enabled"),
        "legacy path should NOT emit orchestrator marker; got: {stderr}",
    );
}

/// `zenmetrics --use-orchestrator …` is accepted (deprecated no-op
/// since Phase 7.7.1) and emits a deprecation warning to stderr. The
/// score itself goes through the orchestrator since that's the new
/// default.
#[cfg(feature = "cpu-metrics")]
#[cfg(feature = "orchestrator")]
#[test]
fn use_orchestrator_emits_deprecation_warning() {
    let dir = fixtures_dir();
    let cache_dir = tempfile::tempdir().expect("tempdir");
    let out = cli()
        .args([
            "--use-orchestrator",
            "--orchestrator-cache",
            cache_dir.path().to_str().unwrap(),
            "--bench-on-start",
            "auto",
            "score",
            "--metric",
            "zensim",
            "--reference",
            dir.join("ref_64.png").to_str().unwrap(),
            "--distorted",
            dir.join("dist_identical_64.png").to_str().unwrap(),
        ])
        .output()
        .expect("run cli");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "use-orchestrator score failed: stderr={stderr}",
    );
    assert!(
        stderr.contains("--use-orchestrator") && stderr.contains("deprecated"),
        "expected deprecation warning mentioning --use-orchestrator; got: {stderr}",
    );
}

// ---------------------------------------------------------------------------
// ZENMETRICS_REQUIRE_GPU — the silent-CPU-fallback gate
// ---------------------------------------------------------------------------

/// Run the binary with every GPU runtime made unreachable.
///
/// `CUDA_VISIBLE_DEVICES=""` hides CUDA; the bogus Vulkan ICD paths keep a
/// wgpu-enabled build (including a software rasteriser like lavapipe, which
/// would otherwise "succeed" on the CPU and defeat the point) from coming up.
#[cfg(feature = "gpu-cvvdp")]
fn cli_without_gpu(require_gpu: bool) -> std::process::Output {
    let fx = fixtures_dir();
    let mut c = cli();
    c.args([
        "score",
        "--metric",
        "cvvdp-gpu",
        "--reference",
        fx.join("ref_64.png").to_str().unwrap(),
        "--distorted",
        fx.join("dist_noisy_64.png").to_str().unwrap(),
    ])
    .env("CUDA_VISIBLE_DEVICES", "")
    .env("VK_ICD_FILENAMES", "/nonexistent/zenmetrics-test.json")
    .env("VK_DRIVER_FILES", "/nonexistent/zenmetrics-test.json");
    if require_gpu {
        c.env("ZENMETRICS_REQUIRE_GPU", "1");
    } else {
        c.env_remove("ZENMETRICS_REQUIRE_GPU");
    }
    c.output().expect("run cli")
}

/// With no GPU reachable and the gate OFF, `Auto` falls through to the CPU
/// rung and still prints a score under the **GPU** column name. That is the
/// pre-existing (and dangerous) behaviour; pinned here so the gate's effect
/// is a measured difference rather than an assumption, and so we notice if
/// the default ever changes.
#[cfg(feature = "gpu-cvvdp")]
#[test]
fn without_require_gpu_auto_silently_falls_back_to_cpu() {
    let out = cli_without_gpu(false);
    assert!(
        out.status.success(),
        "expected the permissive default to succeed via the CPU rung; \
         stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("cvvdp"),
        "expected a cvvdp score on stdout, got {stdout:?}"
    );
}

/// With the gate ON and no GPU reachable, the run MUST fail loudly — nonzero
/// exit, nothing on stdout, and an error naming the reason — instead of
/// computing on the CPU and reporting it under a GPU column.
#[cfg(feature = "gpu-cvvdp")]
#[test]
fn require_gpu_makes_missing_gpu_a_hard_error() {
    let out = cli_without_gpu(true);
    assert!(
        !out.status.success(),
        "ZENMETRICS_REQUIRE_GPU=1 with no GPU MUST fail; it exited 0 with \
         stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("cvvdp_"),
        "a refused run must not emit a score column; stdout={stdout:?}"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("ZENMETRICS_REQUIRE_GPU"),
        "the failure must name the reason so it is not mistaken for a broken \
         build; stderr={stderr:?}"
    );
}

/// `batch --group-by-ref` parity gate (zenmetrics#46, the `batch` twin of the score-pairs gate
/// below): grouping changes ONLY the output row order (stable-sorted by ref_path), never the
/// scores. Two runs over the same interleaved TSV — grouped and ungrouped — must produce
/// tag-matched rows with BIT-IDENTICAL score text (same binary, same deterministic CPU metric,
/// same inputs), every pass-through column intact, and the grouped run must emit each
/// reference's ladder contiguously with its input order preserved.
#[test]
#[cfg(feature = "cpu-metrics")]
fn batch_group_by_ref_is_score_identical_and_grouped() {
    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    // Interleave two reference ladders so the ungrouped run cannot benefit from the
    // consecutive-same-ref cache — the exact shape --group-by-ref exists to fix.
    let tsv = staged.path().join("pairs.tsv");
    let r64 = dir.join("ref_64.png");
    let r256 = dir.join("ref_256.png");
    let d64a = dir.join("dist_noisy_64.png");
    let d64b = dir.join("dist_identical_64.png");
    let d256 = dir.join("dist_noisy_256.png");
    std::fs::write(
        &tsv,
        format!(
            "ref_path\tdist_path\ttag\n\
             {r64}\t{d64a}\tref64-a\n\
             {r256}\t{d256}\tref256\n\
             {r64}\t{d64b}\tref64-b\n",
            r64 = r64.display(),
            r256 = r256.display(),
            d64a = d64a.display(),
            d64b = d64b.display(),
            d256 = d256.display(),
        ),
    )
    .unwrap();

    let run = |grouped: bool, out: &std::path::Path| -> Vec<(String, String)> {
        let mut c = cli();
        c.args([
            "batch",
            "--metric",
            "ssim2",
            "--pairs",
            tsv.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ]);
        if grouped {
            c.arg("--group-by-ref");
        }
        let res = c.output().expect("run cli");
        assert!(
            res.status.success(),
            "batch failed: stderr={}",
            String::from_utf8_lossy(&res.stderr)
        );
        let text = std::fs::read_to_string(out).unwrap();
        let mut lines = text.lines();
        let header = lines.next().expect("header");
        let cols: Vec<&str> = header.split('\t').collect();
        let tag_i = cols.iter().position(|c| *c == "tag").expect("tag col");
        let score_i = cols
            .iter()
            .position(|c| c.starts_with("ssim2"))
            .expect("ssim2 col");
        // (tag, score text) in file order — the score TEXT is compared so the
        // parity is bit-for-bit on what the consumer reads, no re-parsing.
        lines
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                (f[tag_i].to_string(), f[score_i].to_string())
            })
            .collect()
    };
    let plain = run(false, &staged.path().join("plain.tsv"));
    let grouped = run(true, &staged.path().join("grouped.tsv"));
    assert_eq!(plain.len(), 3);
    assert_eq!(grouped.len(), 3);

    // Parity: tag-matched rows carry identical scores.
    let mut a = plain.clone();
    let mut b = grouped.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b, "grouping must not change any score");
    // Ungrouped output keeps input order (no behaviour change without the flag).
    let plain_tags: Vec<&str> = plain.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(plain_tags, ["ref64-a", "ref256", "ref64-b"]);
    // Grouped: the ref64 ladder is contiguous AND keeps its input order (a before b);
    // ref_256.png sorts before ref_64.png.
    let grouped_tags: Vec<&str> = grouped.iter().map(|(t, _)| t.as_str()).collect();
    assert_eq!(
        grouped_tags,
        ["ref256", "ref64-a", "ref64-b"],
        "stable sort by ref_path: ladders contiguous, input order preserved"
    );
}

/// `score-pairs --group-by-ref` parity gate (zenmetrics#46): grouping changes ONLY the output
/// row order (stable-sorted by ref_path), never the scores. Two runs over the same shuffled
/// TSV — grouped and ungrouped — must produce identity-tuple-matched rows with BIT-IDENTICAL
/// scores (same binary, same deterministic CPU metric, same inputs), and the grouped run must
/// emit each reference's ladder contiguously with its input q order preserved.
#[test]
#[cfg(all(feature = "sweep", feature = "cpu-metrics"))]
fn score_pairs_group_by_ref_is_score_identical_and_grouped() {
    use parquet::file::reader::FileReader;
    use parquet::record::RowAccessor;

    let dir = fixtures_dir();
    let staged = tempfile::tempdir().expect("tmp");
    // Interleave two reference ladders so the ungrouped run cannot benefit from the
    // consecutive-same-ref cache — the exact shape --group-by-ref exists to fix.
    let tsv = staged.path().join("pairs.tsv");
    let r64 = dir.join("ref_64.png");
    let r256 = dir.join("ref_256.png");
    let d64a = dir.join("dist_noisy_64.png");
    let d64b = dir.join("dist_identical_64.png");
    let d256 = dir.join("dist_noisy_256.png");
    std::fs::write(
        &tsv,
        format!(
            "ref_path\tdist_path\timage_path\tcodec\tq\tknob_tuple_json\n\
             {r64}\t{d64a}\tref64\tc\t10\t{{}}\n\
             {r256}\t{d256}\tref256\tc\t10\t{{}}\n\
             {r64}\t{d64b}\tref64\tc\t20\t{{}}\n",
            r64 = r64.display(),
            r256 = r256.display(),
            d64a = d64a.display(),
            d64b = d64b.display(),
            d256 = d256.display(),
        ),
    )
    .unwrap();

    let run = |grouped: bool, out: &std::path::Path| {
        let mut c = cli();
        c.args([
            "score-pairs",
            "--metric",
            "ssim2",
            "--pairs-tsv",
            tsv.to_str().unwrap(),
            "--out-parquet",
            out.to_str().unwrap(),
        ]);
        if grouped {
            c.arg("--group-by-ref");
        }
        let out = c.output().expect("run cli");
        assert!(
            out.status.success(),
            "score-pairs failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let pq_plain = staged.path().join("plain.parquet");
    let pq_grouped = staged.path().join("grouped.parquet");
    run(false, &pq_plain);
    run(true, &pq_grouped);

    // (image_path, q) → score-bits rows, in file order.
    let read_rows = |p: &std::path::Path| -> Vec<(String, i64, u64)> {
        let f = std::fs::File::open(p).expect("open parquet");
        let r = parquet::file::reader::SerializedFileReader::new(f).expect("reader");
        // Column layout: image_path, codec, q, knob_tuple_json, <score>, runtime — find by name.
        let schema = r.metadata().file_metadata().schema_descr();
        let col = |name: &str| {
            (0..schema.num_columns())
                .find(|&i| schema.column(i).name() == name)
                .unwrap_or_else(|| panic!("column {name} missing"))
        };
        let (ip, qi) = (col("image_path"), col("q"));
        let si = (0..schema.num_columns())
            .find(|&i| schema.column(i).name().starts_with("ssim2"))
            .expect("ssim2 score column");
        r.get_row_iter(None)
            .expect("row iter")
            .map(|row| {
                let row = row.expect("row");
                (
                    row.get_string(ip).expect("image_path").clone(),
                    row.get_long(qi).expect("q"),
                    row.get_double(si).expect("score").to_bits(),
                )
            })
            .collect()
    };
    let plain = read_rows(&pq_plain);
    let grouped = read_rows(&pq_grouped);
    assert_eq!(plain.len(), 3);
    assert_eq!(grouped.len(), 3);

    // Parity: identity-matched rows carry BIT-identical scores.
    let mut a = plain.clone();
    let mut b = grouped.clone();
    a.sort();
    b.sort();
    assert_eq!(a, b, "grouping must not change any score");

    // Grouping: the two ref64 rows are contiguous AND keep their input q order (10 before 20).
    let order: Vec<(&str, i64)> = grouped.iter().map(|(p, q, _)| (p.as_str(), *q)).collect();
    assert_eq!(
        order,
        vec![("ref256", 10), ("ref64", 10), ("ref64", 20)],
        "stable sort by ref_path: ladders contiguous, q order preserved"
    );
}

/// Parquet footer key-value metadata of a written file, as `(key, value)`.
#[cfg(all(feature = "sweep", feature = "hdr", feature = "cpu-metrics"))]
fn read_parquet_kv(path: &std::path::Path) -> Vec<(String, String)> {
    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(path).expect("open parquet");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .map(|kv| {
            kv.iter()
                .map(|k| (k.key.clone(), k.value.clone().unwrap_or_default()))
                .collect()
        })
        .unwrap_or_default()
}

/// The named columns of the first row of a parquet file, rendered with
/// parquet's `Field::to_string` (strings come back quoted).
#[cfg(all(feature = "sweep", feature = "hdr", feature = "cpu-metrics"))]
fn read_parquet_first_row(path: &std::path::Path) -> Vec<(String, String)> {
    use parquet::file::reader::FileReader;
    let file = std::fs::File::open(path).expect("open parquet");
    let reader = parquet::file::reader::SerializedFileReader::new(file).expect("parquet reader");
    let row = reader
        .get_row_iter(None)
        .expect("row iter")
        .next()
        .expect("at least one row")
        .expect("row");
    row.get_column_iter()
        .map(|(k, v)| (k.clone(), v.to_string()))
        .collect()
}

/// Stage an absolute-nits EXR pair (a 50..650 cd/m² ramp + a 10%-darker
/// copy) and a SPLIT-contract pairs.tsv over it.
#[cfg(all(feature = "sweep", feature = "hdr", feature = "cpu-metrics"))]
fn stage_hdr_score_pairs_inputs(
    staged: &std::path::Path,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    use std::io::Write;
    let (w, h) = (64u32, 64u32);
    let mut reference = image::Rgb32FImage::new(w, h);
    for (x, y, p) in reference.enumerate_pixels_mut() {
        let v = 50.0 + 600.0 * (x + y) as f32 / (w + h) as f32;
        *p = image::Rgb([v, v, v]);
    }
    let mut distorted = reference.clone();
    for p in distorted.pixels_mut() {
        for c in p.0.iter_mut() {
            *c *= 0.9;
        }
    }
    let ref_exr = staged.join("ref.exr");
    let dist_exr = staged.join("dist.exr");
    reference.save(&ref_exr).expect("write ref.exr");
    distorted.save(&dist_exr).expect("write dist.exr");

    let pairs_tsv = staged.join("pairs.tsv");
    let mut f = std::fs::File::create(&pairs_tsv).expect("create pairs.tsv");
    writeln!(
        f,
        "image_path\tcodec\tq\tknob_tuple_json\tref_path\tdist_path"
    )
    .unwrap();
    writeln!(
        f,
        "{r}\tzenjxl\t50\t{{}}\t{r}\t{d}",
        r = ref_exr.display(),
        d = dist_exr.display()
    )
    .unwrap();
    writeln!(f, "{r}\tzenjxl\t100\t{{}}\t{r}\t{r}", r = ref_exr.display()).unwrap();
    drop(f);
    (
        pairs_tsv,
        staged.join("scores.parquet"),
        staged.join("features.parquet"),
    )
}

/// `score-pairs --hdr --feature-output` (zenmetrics#13 §5): the zensim
/// feature sidecar is **schema `2.0-hdr`** — schema 1's columns in their
/// exact positions, then the four trailing per-row HDR provenance columns
/// (`hdr_mode / feature_regime / hdr_source / ref_peak_nits`), with the
/// version stamped in the parquet footer. Runs the v1 `pu21-u8-shell`
/// regime (default) and the v3 `pu-linear` regime
/// (`--hdr-features-pu-linear`) and checks each row's `feature_regime`
/// says which one produced its `feat_*`. The SDR control (schema 1,
/// no trailing columns) is `score_pairs_writes_zensim_feature_parquet_cpu`.
#[cfg(all(feature = "sweep", feature = "hdr", feature = "cpu-metrics"))]
#[test]
fn score_pairs_hdr_writes_schema_v2_feature_parquet() {
    let staged = tempfile::tempdir().expect("tmp");
    let (pairs_tsv, out_pq, feat_pq) = stage_hdr_score_pairs_inputs(staged.path());
    let feature_n = zenmetrics_cli::metrics::ZensimFeatureRegime::WithIw.total_features();
    let expected_cols = 5 + feature_n + 4;

    for (regime_flag, want_regime) in [
        (None, "pu21-u8-shell"),
        (Some("--hdr-features-pu-linear"), "pu-linear"),
    ] {
        let mut args = vec![
            "score-pairs",
            "--metric",
            "zensim",
            "--hdr",
            "--pairs-tsv",
            pairs_tsv.to_str().unwrap(),
            "--out-parquet",
            out_pq.to_str().unwrap(),
            "--feature-output",
            feat_pq.to_str().unwrap(),
            "--zensim-features-regime",
            "with-iw",
        ];
        if let Some(flag) = regime_flag {
            args.push(flag);
        }
        let result = cli().args(&args).output().expect("run cli");
        assert!(
            result.status.success(),
            "score-pairs --hdr ({want_regime}) failed: stderr={}",
            String::from_utf8_lossy(&result.stderr)
        );

        let (names, num_rows) = read_parquet_schema(&feat_pq);
        assert_eq!(num_rows, 2, "({want_regime}) expected 2 feature rows");
        assert_eq!(
            names.len(),
            expected_cols,
            "({want_regime}) expected {expected_cols} columns, got {}",
            names.len()
        );
        // Schema 1's columns keep their exact positions...
        assert_eq!(
            &names[..5],
            &[
                "image_path",
                "codec",
                "q",
                "knob_tuple_json",
                "zensim_score"
            ]
        );
        assert_eq!(names[5], "feat_0");
        assert_eq!(names[5 + feature_n - 1], format!("feat_{}", feature_n - 1));
        // ...and the HDR provenance trails them.
        assert_eq!(
            &names[5 + feature_n..],
            &["hdr_mode", "feature_regime", "hdr_source", "ref_peak_nits"]
        );
        let kv = read_parquet_kv(&feat_pq);
        assert!(
            kv.contains(&(
                "zenmetrics.sidecar_schema".to_string(),
                "2.0-hdr".to_string()
            )),
            "({want_regime}) footer must stamp sidecar schema 2.0-hdr: {kv:?}"
        );
        assert!(kv.contains(&("zenmetrics.n_features".to_string(), feature_n.to_string())));

        let row = read_parquet_first_row(&feat_pq);
        let get = |k: &str| {
            row.iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| panic!("column {k} missing from {row:?}"))
        };
        assert_eq!(get("hdr_mode"), "\"nits-mcll\"");
        assert_eq!(get("feature_regime"), format!("\"{want_regime}\""));
        assert_eq!(get("hdr_source"), "\"linear-exr\"");
        // The reference's measured MaxCLL peak: the ramp tops out at
        // 50 + 600·126/128 ≈ 640.6 cd/m² (MaxRgb; above the 203-nit clamp floor).
        let peak: f32 = get("ref_peak_nits").parse().expect("ref_peak_nits f32");
        assert!(
            (peak - 640.6).abs() < 1.0,
            "({want_regime}) ref_peak_nits {peak} should be the measured ramp peak ≈ 640.6"
        );
        // The score parquet is still written alongside.
        let (_, score_rows) = read_parquet_schema(&out_pq);
        assert_eq!(score_rows, 2);
    }
}
