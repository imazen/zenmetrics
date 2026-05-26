//! Fetch + cache helper for the conformance goldens.
//!
//! Mirrors `cvvdp-gpu/tests/common/mod.rs` — goldens live in the same
//! public R2 bucket (`coefficient.r2.imazen.org`) under the
//! `cvvdp-goldens/conformance-v1/` prefix and are fetched without
//! credentials. A local override (`CVVDP_CONFORMANCE_GOLDENS` env var
//! pointing at a `conformance_goldens.json`) short-circuits the fetch
//! for development against freshly-built goldens.

#![cfg(feature = "conformance-goldens")]
#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

/// Golden-set version pin. Bump in lockstep with the R2 prefix.
pub const GOLDEN_VERSION: &str = "conformance-v1";

/// Public R2 URL for the conformance goldens JSON.
pub const GOLDENS_URL: &str =
    "https://coefficient.r2.imazen.org/cvvdp-goldens/conformance-v1/conformance_goldens.json";

/// sha256 of `conformance_goldens.json` captured at upload time.
/// Bump alongside `GOLDEN_VERSION` when goldens are regenerated.
pub const GOLDENS_SHA256: &str = "8f7d69dc6b98272b8425c2245cf7878e5b397878f8717056715f65bd606940bc";

const CACHE_DIR_SUBDIR: &str = "zenmetrics-cvvdp-conformance-goldens";

fn cache_dir() -> PathBuf {
    let base = if let Some(dir) = std::env::var_os("XDG_CACHE_HOME") {
        PathBuf::from(dir)
    } else if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".cache")
    } else {
        std::env::temp_dir()
    };
    let dir = base.join(CACHE_DIR_SUBDIR).join(GOLDEN_VERSION);
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Load the conformance goldens JSON, either from the
/// `CVVDP_CONFORMANCE_GOLDENS` local override or by fetching from R2
/// (sha256-verified + cached).
pub fn load_goldens() -> serde_json::Value {
    let bytes = if let Ok(path) = std::env::var("CVVDP_CONFORMANCE_GOLDENS") {
        fs::read(&path).unwrap_or_else(|e| panic!("read local goldens {path}: {e}"))
    } else {
        fetch_goldens()
    };
    serde_json::from_slice(&bytes).expect("parse conformance_goldens.json")
}

fn fetch_goldens() -> Vec<u8> {
    let local = cache_dir().join("conformance_goldens.json");
    if local.exists()
        && let Ok(bytes) = fs::read(&local)
        && sha256_hex(&bytes) == GOLDENS_SHA256
    {
        return bytes;
    }
    let mut resp = ureq::get(GOLDENS_URL)
        .call()
        .unwrap_or_else(|e| panic!("GET {GOLDENS_URL}: {e}"));
    let mut bytes = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|e| panic!("read {GOLDENS_URL}: {e}"));
    let got = sha256_hex(&bytes);
    assert_eq!(
        got, GOLDENS_SHA256,
        "conformance goldens sha256 mismatch — regenerate or bump GOLDENS_SHA256"
    );
    fs::write(&local, &bytes).unwrap_or_else(|e| panic!("cache write: {e}"));
    bytes
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let out = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in &out {
        use std::fmt::Write;
        write!(s, "{b:02x}").unwrap();
    }
    s
}
