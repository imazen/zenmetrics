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

use crate::common;

use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;

// Tick 595: lockstep `const _:` pins for PYCVVDP_REFERENCE_VERSION
// vs requirements.txt / LUT header / docs / README / Cargo.toml
// have been moved to the new `tests/version_lockstep.rs` test
// file, which is NOT gated behind `parity-goldens`. That way the
// pins fire on every `cargo check / test` rather than only when
// the goldens feature is enabled. The runtime manifest-fetch
// check below stays here because it actually needs the goldens
// fetch.

#[test]
fn manifest_fetches() {
    let path = common::fetch("manifest.json", common::MANIFEST_SHA256);
    let contents = std::fs::read_to_string(&path).expect("read manifest");
    let json: serde_json::Value = serde_json::from_str(&contents).expect("parse manifest");
    // Tick 588: source the expected version from the central
    // `PYCVVDP_REFERENCE_VERSION` const so this test follows
    // automatically when the reference is bumped.
    assert_eq!(
        json["reference_version"].as_str(),
        Some(PYCVVDP_REFERENCE_VERSION),
        "manifest must reference pycvvdp {PYCVVDP_REFERENCE_VERSION}"
    );
}
