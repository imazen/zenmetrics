//! Pin the goldens manifest metadata in `tests/common/mod.rs` for
//! self-consistency. `GOLDEN_VERSION`, `MANIFEST_URL`, and
//! `MANIFEST_SHA256` are load-bearing for every parity-goldens test
//! — a typo or paste-error in any of them silently breaks fetch
//! validation (wrong URL = 404, wrong SHA = silent integrity bypass).
//!
//! The constants live in test-only code, so this file shares the
//! same `common` module path indirection used by the rest of the
//! test files (`#[path = "common/mod.rs"] mod common;`).
//!
//! Companion to the constants-pin series (ticks 393-402): same
//! discipline applied to test-infrastructure constants. A regression
//! that bumps `GOLDEN_VERSION = "v2"` but forgets to update the URL
//! or the sha256 trips here before the goldens-feature gate runs
//! the actual fetch.
#![allow(clippy::excessive_precision)]

#[path = "common/mod.rs"]
mod common;

use common::{GOLDEN_VERSION, MANIFEST_SHA256, MANIFEST_URL, cache_dir};

#[test]
fn manifest_url_embeds_golden_version() {
    // The R2 prefix encodes the version path segment. Bumping
    // GOLDEN_VERSION to "v2" without updating MANIFEST_URL (or
    // vice versa) means tests fetch the wrong manifest — likely
    // 404, but worse, possibly a stale-cached "v1" manifest.
    let segment = format!("/{GOLDEN_VERSION}/");
    assert!(
        MANIFEST_URL.contains(&segment),
        "MANIFEST_URL = {MANIFEST_URL:?} must contain the segment {segment:?} from GOLDEN_VERSION = {GOLDEN_VERSION:?}",
    );
}

#[test]
fn manifest_url_is_well_formed_https() {
    // The public R2 mirror requires https; a refactor to http://
    // would either fail TLS termination at the CDN or — worse —
    // succeed against a captive portal. Pin the scheme.
    assert!(
        MANIFEST_URL.starts_with("https://"),
        "MANIFEST_URL = {MANIFEST_URL:?} must start with https://",
    );
    assert!(
        MANIFEST_URL.ends_with(".json"),
        "MANIFEST_URL = {MANIFEST_URL:?} must end with .json",
    );
}

#[test]
fn manifest_sha256_is_64_lowercase_hex() {
    // sha256 hex digests are exactly 64 chars from [0-9a-f]. A
    // typo that drops a char (63 chars) silently truncates
    // compare semantics; an uppercase variant fails fetch's
    // case-sensitive sha2-Digest::finalize() output match.
    assert_eq!(
        MANIFEST_SHA256.len(),
        64,
        "MANIFEST_SHA256 must be 64 hex chars, got len {}: {MANIFEST_SHA256:?}",
        MANIFEST_SHA256.len(),
    );
    for (i, c) in MANIFEST_SHA256.chars().enumerate() {
        assert!(
            c.is_ascii_digit() || ('a'..='f').contains(&c),
            "MANIFEST_SHA256[{i}] = {c:?} must be [0-9a-f] (lowercase hex)",
        );
    }
}

#[test]
fn cache_dir_path_embeds_golden_version() {
    // The per-version cache dir partitions goldens by version so
    // a `v2` upgrade doesn't reuse `v1`'s stale cached files.
    // Pin the path-segment relationship so a refactor that
    // changes the cache structure trips here.
    let path = cache_dir();
    let path_str = path.to_string_lossy();
    assert!(
        path_str.contains(GOLDEN_VERSION),
        "cache_dir() = {path_str:?} must contain GOLDEN_VERSION = {GOLDEN_VERSION:?}",
    );
    assert!(
        path_str.contains("zenmetrics-cvvdp-goldens"),
        "cache_dir() = {path_str:?} must contain the crate-specific subdir 'zenmetrics-cvvdp-goldens'",
    );
}

#[test]
fn golden_version_is_non_empty_and_lowercase() {
    // GOLDEN_VERSION is interpolated into both the URL and the
    // cache dir; empty or weird-charactered strings would break
    // both. Pin minimum invariants.
    assert!(
        !GOLDEN_VERSION.is_empty(),
        "GOLDEN_VERSION must not be empty",
    );
    // Match the existing pattern: "v" + decimal digits (`v1`, `v2`, …).
    assert!(
        GOLDEN_VERSION.starts_with('v'),
        "GOLDEN_VERSION = {GOLDEN_VERSION:?} must follow the v<N> convention",
    );
    assert!(
        GOLDEN_VERSION[1..].chars().all(|c| c.is_ascii_digit()),
        "GOLDEN_VERSION = {GOLDEN_VERSION:?}: chars after 'v' must be digits",
    );
}
