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

use common::{
    CACHE_DIR_SUBDIR, GOLDEN_VERSION, MANIFEST_SHA256, MANIFEST_URL, cache_dir, const_str,
};

// Tick 578 (refactored tick 584): compile-time pins for the goldens-
// metadata structural invariants. The const-byte-loop primitives are
// encapsulated in `common::const_str::{starts_with, ends_with, contains}`.
//
// These promote the previously-runtime checks in:
//   - `manifest_url_is_well_formed_https` (https:// prefix + .json suffix)
//   - `manifest_url_uses_documented_r2_host` (canonical R2 host prefix)
//   - `manifest_sha256_is_64_lowercase_hex` (length 64)
//   - `golden_version_is_non_empty_and_lowercase` (non-empty + v-prefix)
const _: () = {
    let url = MANIFEST_URL.as_bytes();
    assert!(
        const_str::starts_with(url, b"https://"),
        "MANIFEST_URL must start with https://",
    );
    assert!(
        const_str::ends_with(url, b".json"),
        "MANIFEST_URL must end with .json",
    );
    assert!(
        const_str::starts_with(url, b"https://coefficient.r2.imazen.org/"),
        "MANIFEST_URL must start with canonical R2 host https://coefficient.r2.imazen.org/",
    );
    assert!(
        MANIFEST_SHA256.len() == 64,
        "MANIFEST_SHA256 must be 64 hex chars (a typo that truncates one char silently breaks fetch validation)",
    );
    assert!(
        !GOLDEN_VERSION.is_empty(),
        "GOLDEN_VERSION must not be empty"
    );
    let gv = GOLDEN_VERSION.as_bytes();
    assert!(
        gv[0] == b'v',
        "GOLDEN_VERSION must follow the v<N> convention (first byte = 'v')",
    );
    // Tick 586: pin GOLDEN_VERSION to its exact current value so a
    // version bump (e.g. to "v2") forces updating MANIFEST_URL's
    // "/v1/" segment AND the runtime test fixtures in lockstep.
    // Without this pin, a refactor that bumps GOLDEN_VERSION but
    // forgets the URL would pass the "starts-with-v" check above
    // and the version-segment contains-check below (because both
    // would still report v1) — silently fetching the wrong manifest.
    assert!(
        const_str::bytes_eq(gv, b"v1"),
        "GOLDEN_VERSION pinned to current value 'v1'; bump in lockstep with MANIFEST_URL `/v1/` segment",
    );
};

// Tick 579 (refactored tick 584): `.contains(...)` invariants via
// the shared `const_str::contains` substring-search helper.
//
// Bucket subpath: catches a refactor that swapped cvvdp-gpu's
// MANIFEST_URL to a sibling crate's bucket (e.g. /zensim-goldens/).
// Same load-bearing semantic as the runtime test
// `manifest_url_uses_cvvdp_goldens_bucket_subpath` (tick 520).
const _: () = assert!(
    const_str::contains(MANIFEST_URL.as_bytes(), b"/cvvdp-goldens/"),
    "MANIFEST_URL must contain bucket subpath /cvvdp-goldens/",
);

// Version segment: catches a refactor that bumps GOLDEN_VERSION
// to v2 but forgets to update the URL (or vice versa). When
// GOLDEN_VERSION bumps, this pin and the GOLDEN_VERSION value pin
// both need updating in the same commit — by design.
const _: () = assert!(
    const_str::contains(MANIFEST_URL.as_bytes(), b"/v1/"),
    "MANIFEST_URL must contain version path segment /v1/ (matches current GOLDEN_VERSION)",
);

// Tick 580: per-byte lowercase-hex validation on MANIFEST_SHA256.
// `char::is_ascii_digit` and `RangeInclusive::contains` aren't
// const fn yet, but raw u8 comparison is — and the sha256 string
// is pure ASCII so the byte-loop covers every char correctly.
// A uppercase variant (e.g. "EE52F5B...") fails the sha2-Digest
// case-sensitive comparison silently; a stray non-hex char (e.g.
// "g") would fetch the wrong manifest. Pin at compile time so
// either typo class trips before the goldens-feature fetch runs.
const _: () = {
    let bytes = MANIFEST_SHA256.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        assert!(
            (c >= b'0' && c <= b'9') || (c >= b'a' && c <= b'f'),
            "MANIFEST_SHA256 must be lowercase hex (0-9, a-f) — found non-hex byte",
        );
        i += 1;
    }
};

// Tick 581: CACHE_DIR_SUBDIR structural invariants. The constant
// is the per-crate cache-dir name "zenmetrics-cvvdp-goldens" used
// by `cache_dir()`. Pin:
//   - non-empty
//   - contains "cvvdp" (so sibling crates' cache dirs don't collide
//     with this one)
//   - all-ASCII alphanumerics or hyphen (filesystem-portable)
const _: () = {
    assert!(
        !CACHE_DIR_SUBDIR.is_empty(),
        "CACHE_DIR_SUBDIR must not be empty"
    );
    assert!(
        const_str::contains(CACHE_DIR_SUBDIR.as_bytes(), b"cvvdp"),
        "CACHE_DIR_SUBDIR must contain 'cvvdp' to disambiguate from sibling crates",
    );
    let bytes = CACHE_DIR_SUBDIR.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        assert!(
            (c >= b'0' && c <= b'9')
                || (c >= b'a' && c <= b'z')
                || (c >= b'A' && c <= b'Z')
                || c == b'-',
            "CACHE_DIR_SUBDIR must be filesystem-portable (ASCII alphanumerics + hyphen)",
        );
        i += 1;
    }
};

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
fn manifest_url_uses_documented_r2_host() {
    // Tick 519: the existing structure checks (https + .json + version
    // segment) would still pass if a refactor changed the HOST — e.g.
    // pointed to a different CDN bucket on a different cloud, or a
    // localhost dev mirror. Pin the canonical public host so a
    // host-swap surfaces as a test failure rather than a silent
    // fetch redirect / 404.
    //
    // If the canonical host migrates (e.g. to a different Cloudflare
    // account or off-R2 entirely), update this pin in the same commit
    // as the URL change.
    const CANONICAL_HOST: &str = "https://coefficient.r2.imazen.org/";
    assert!(
        MANIFEST_URL.starts_with(CANONICAL_HOST),
        "MANIFEST_URL = {MANIFEST_URL:?} must start with {CANONICAL_HOST:?}",
    );
}

#[test]
fn manifest_url_uses_cvvdp_goldens_bucket_subpath() {
    // Tick 520: the canonical host check pins the domain but not the
    // bucket subpath. Sibling crates (zensim-gpu, butteraugli-gpu,
    // dssim-gpu, ssim2-gpu) all publish their own goldens to the same
    // host under different subpaths. A refactor that accidentally
    // pointed cvvdp-gpu's MANIFEST_URL at — say — `/zensim-goldens/`
    // would still pass the host check and the version-segment check,
    // but fetch the wrong manifest. Pin the crate-specific bucket
    // subpath so misrouting trips here.
    const CANONICAL_BUCKET_SUBPATH: &str = "/cvvdp-goldens/";
    assert!(
        MANIFEST_URL.contains(CANONICAL_BUCKET_SUBPATH),
        "MANIFEST_URL = {MANIFEST_URL:?} must contain bucket subpath \
         {CANONICAL_BUCKET_SUBPATH:?}",
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
    // Tick 582: use the CACHE_DIR_SUBDIR const directly (extracted
    // in tick 581) instead of a duplicate magic string. If
    // CACHE_DIR_SUBDIR is renamed, this test follows automatically
    // and the static asserts on it (also tick 581) still cover the
    // "must contain 'cvvdp'" invariant at compile time.
    assert!(
        path_str.contains(CACHE_DIR_SUBDIR),
        "cache_dir() = {path_str:?} must contain the crate-specific subdir {CACHE_DIR_SUBDIR:?}",
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
