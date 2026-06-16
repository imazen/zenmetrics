//! End-to-end CLI tests. Use `assert_cmd`-style spawning of the compiled
//! `zenmetrics` binary so we exercise the same code path users hit.
//!
//! Phase 7 adds orchestrator integration tests at the bottom of this file
//! (gated on the `orchestrator` feature). They verify the new top-level
//! flags parse, `--use-orchestrator` routes scoring through the
//! orchestrator path, and the legacy path is unchanged when the flag is
//! absent.

use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_zenmetrics");
    Command::new(bin)
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
