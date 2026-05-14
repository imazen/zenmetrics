//! Fetch + cache helpers for cvvdp parity goldens.
//!
//! Goldens are produced by `scripts/cvvdp_goldens/build_goldens.py`
//! running the pinned pycvvdp v0.5.4 reference, and uploaded to R2
//! under `s3://coefficient/cvvdp-goldens/<version>/`. The same bucket
//! is exposed publicly at `https://coefficient.r2.imazen.org/`, so
//! tests fetch without credentials. Same pattern as zensim's
//! `zentrain-r2.imazen.org` corpus mirror.
//!
//! Tests using this module must be compiled with the
//! `parity-goldens` feature so the network code path isn't built into
//! the default `cargo test` run.
//!
//! Cache layout: `$XDG_CACHE_HOME/zenmetrics-cvvdp-goldens/<version>/<file>`.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pin label for the current cvvdp golden set. Bump in lockstep with
/// the R2 prefix and the pycvvdp version pin in
/// `scripts/cvvdp_goldens/requirements.txt`.
pub const GOLDEN_VERSION: &str = "v1";

/// Public R2 URL for the manifest. The bucket is the same
/// `s3://coefficient/` that the sweep infrastructure uses; its public
/// mirror is configured at `coefficient.r2.imazen.org`.
pub const MANIFEST_URL: &str = "https://coefficient.r2.imazen.org/cvvdp-goldens/v1/manifest.json";

/// sha256 of the manifest, captured at upload time
/// (`2026-05-14`, v0.5.4 reference, zenmetrics-corpus 256×256 q-grid).
/// Bump alongside `GOLDEN_VERSION` when the goldens are regenerated.
pub const MANIFEST_SHA256: &str =
    "9b8638c4cee15b79240acd8116f54d417f2a641999ca0146567b1bea5aa594c5";

/// Returns the per-version cache directory, creating it if needed.
pub fn cache_dir() -> PathBuf {
    let base = if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        std::env::temp_dir()
    };
    let dir = base.join("zenmetrics-cvvdp-goldens").join(GOLDEN_VERSION);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Fetch `name` (e.g. `"src_vs_q70.final.json"`) into the cache and
/// return the local path. Panics on failure — meant for use in tests
/// where the right behavior is loud failure, not silent skip.
pub fn fetch(name: &str, sha256: &str) -> PathBuf {
    let local = cache_dir().join(name);
    if local.exists() {
        if let Ok(hex) = file_sha256_hex(&local) {
            if hex == sha256 {
                return local;
            }
        }
        // Stale or corrupt — drop it and refetch.
        let _ = fs::remove_file(&local);
    }

    let base = MANIFEST_URL
        .rsplit_once('/')
        .map(|(b, _)| b)
        .expect("MANIFEST_URL must have at least one '/'");
    let url = format!("{base}/{name}");

    let mut resp = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("GET {url}: {e}"));
    let mut bytes = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("read {url}: {e}"));

    let got = sha256_hex(&bytes);
    assert_eq!(got, sha256, "sha256 mismatch for {name}");
    fs::write(&local, &bytes).unwrap_or_else(|e| panic!("write {local:?}: {e}"));
    local
}

fn file_sha256_hex(path: &Path) -> std::io::Result<String> {
    let bytes = fs::read(path)?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in out.iter() {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
    }
    s
}
