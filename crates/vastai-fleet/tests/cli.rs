//! End-to-end CLI tests for `vastai-fleet`. These shell out to the built
//! binary via `--raw-input` to avoid any real vast.ai API calls.
//!
//! The fixtures cover the failure modes from the 2026-05-17 / 2026-05-18
//! destroyer sessions:
//!   - empty v1 envelope (the "no instances" case the bash destroyer
//!     STILL crashed on)
//!   - mixed fleet with one malformed row (the `null` element) that
//!     the parser must skip
//!   - deprecation banner glued onto the JSON body
//!   - truncated JSON (genuine network failure mid-fetch)

use std::path::PathBuf;
use std::process::Command;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn cli() -> Command {
    let bin = env!("CARGO_BIN_EXE_vastai-fleet");
    Command::new(bin)
}

#[test]
fn status_empty_fleet_does_not_crash() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("empty_v1.json").to_str().unwrap(),
            "status",
            "--label-prefix",
            "anything",
        ])
        .output()
        .expect("run cli");
    assert!(
        out.status.success(),
        "status should not crash on empty fleet; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("instances:     0"));
}

#[test]
fn status_mixed_fleet_filters_by_label() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("mixed_fleet_v1.json").to_str().unwrap(),
            "status",
            "--label-prefix",
            "ssim2-backfill-2026-05-18",
        ])
        .output()
        .expect("run cli");
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Expect 5 ssim2 boxes (w1..w5).
    assert!(stdout.contains("instances:     5"), "expected 5 ssim2 instances; got: {stdout}");
    // Burn rate: 0.0852 + 0.0911 + 0.1023 + 0.0 + 0.095 = 0.3736
    // (w4 is exited but still listed; dph 0.0).
    assert!(stdout.contains("0.374") || stdout.contains("0.373"), "expected ~0.374/hr; got: {stdout}");
    assert!(stdout.contains("running"));
    assert!(stdout.contains("loading"));
    assert!(stdout.contains("exited"));
    // One warning expected for the null row.
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("WARN parser"), "expected warning on null row; got stderr: {stderr}");
}

#[test]
fn status_with_deprecation_banner_strips_preamble() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("with_deprecation_banner.json").to_str().unwrap(),
            "status",
            "--label-prefix",
            "test-fleet",
        ])
        .output()
        .expect("run cli");
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("instances:     1"));
}

#[test]
fn status_truncated_json_is_error_not_panic() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("truncated.json").to_str().unwrap(),
            "status",
            "--label-prefix",
            "anything",
        ])
        .output()
        .expect("run cli");
    // Truncated JSON should produce a clean error, not a panic.
    assert!(!out.status.success(), "truncated json should fail cleanly");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not JSON") || stderr.contains("EOF") || stderr.contains("expected") || stderr.contains("error"),
        "expected a structured error message; got: {stderr}"
    );
}

#[test]
fn destroy_dry_run_does_not_call_vastai() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("mixed_fleet_v1.json").to_str().unwrap(),
            "destroy",
            "--label-prefix",
            "ssim2-backfill-2026-05-18",
            "--dry-run",
        ])
        .output()
        .expect("run cli");
    assert!(out.status.success(), "destroy --dry-run should succeed; stderr={}", String::from_utf8_lossy(&out.stderr));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("DRY-RUN destroy"));
    // Should mention 5 destroy ops.
    let n_dry = stderr.matches("DRY-RUN destroy").count();
    assert_eq!(n_dry, 5, "expected 5 dry-run destroys; got {n_dry}; stderr: {stderr}");
}

#[test]
fn destroy_no_match_exits_cleanly() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("empty_v1.json").to_str().unwrap(),
            "destroy",
            "--label-prefix",
            "nothing",
        ])
        .output()
        .expect("run cli");
    assert!(out.status.success(), "destroy with no match should succeed");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("nothing to destroy"));
}

#[test]
fn status_json_format() {
    let dir = fixtures_dir();
    let out = cli()
        .args([
            "--raw-input",
            dir.join("mixed_fleet_v1.json").to_str().unwrap(),
            "status",
            "--label-prefix",
            "ssim2-backfill-2026-05-18",
            "--format",
            "json",
        ])
        .output()
        .expect("run cli");
    assert!(out.status.success(), "stderr={}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("output should be valid JSON");
    assert_eq!(v["count"].as_u64(), Some(5));
    assert_eq!(v["label_prefix"].as_str(), Some("ssim2-backfill-2026-05-18"));
    assert!(v["status_breakdown"].is_object());
}
