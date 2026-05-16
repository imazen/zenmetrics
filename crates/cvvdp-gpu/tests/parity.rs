//! Parity tests against the pinned pycvvdp v0.5.4 reference.
//!
//! Gated behind the `parity-goldens` cargo feature so the default
//! `cargo test -p cvvdp-gpu` invocation doesn't touch the network. To
//! run:
//!
//! ```bash
//! cargo test -p cvvdp-gpu --features parity-goldens
//! ```
//!
//! The feature pulls in `ureq` + `sha2` + `serde_json` and the helpers
//! in `tests/common/mod.rs`.

#![cfg(feature = "parity-goldens")]

mod common;

#[test]
fn manifest_fetches() {
    let path = common::fetch("manifest.json", common::MANIFEST_SHA256);
    let contents = std::fs::read_to_string(&path).expect("read manifest");
    let json: serde_json::Value = serde_json::from_str(&contents).expect("parse manifest");
    assert_eq!(
        json["reference_version"].as_str(),
        Some("v0.5.4"),
        "manifest must reference pycvvdp v0.5.4"
    );
}
