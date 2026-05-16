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

use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;

// Tick 588: PYCVVDP_REFERENCE_VERSION format pin. The constant
// drives this test's runtime version check plus PORT_STATUS.md
// + the `kernels/csf_lut/v0_5_4.rs` filename + the vendored LUT
// module name. Pin format invariants at compile time so a bump
// that breaks the v<X>.<Y>.<Z> convention trips immediately:
//   - non-empty
//   - first byte == 'v'
//   - contains at least one '.' (so v0.5.4-like, not v054 etc.)
const _: () = {
    use common::const_str;
    assert!(
        !PYCVVDP_REFERENCE_VERSION.is_empty(),
        "PYCVVDP_REFERENCE_VERSION must not be empty",
    );
    assert!(
        const_str::starts_with(PYCVVDP_REFERENCE_VERSION.as_bytes(), b"v"),
        "PYCVVDP_REFERENCE_VERSION must follow v<X>.<Y>.<Z> convention (start with 'v')",
    );
    assert!(
        const_str::contains(PYCVVDP_REFERENCE_VERSION.as_bytes(), b"."),
        "PYCVVDP_REFERENCE_VERSION must contain at least one '.' (semver-like format)",
    );
};

// Tick 589: pin `scripts/cvvdp_goldens/requirements.txt` against
// PYCVVDP_REFERENCE_VERSION at compile time. The requirements
// file pins `cvvdp==<X>.<Y>.<Z>` (note: PyPI package is `cvvdp`,
// importable as `pycvvdp`); the const stores `v<X>.<Y>.<Z>`.
// Strip the leading 'v' from the const and confirm the
// requirements file contains the resulting `X.Y.Z` substring.
//
// `slice::split_first().unwrap()` is const-callable since Rust
// 1.83, and `include_str!` evaluates at compile time. So when
// the reference is bumped, this pin forces requirements.txt to
// be updated in the same commit as PYCVVDP_REFERENCE_VERSION.
//
// Closes the 6th lockstep site documented in PYCVVDP_REFERENCE_VERSION's
// docstring (the other 5 are: the const itself, the LUT filename,
// the csf_lut_v0_5_4 module name, PORT_STATUS.md, and the
// parity-test manifest check above).
const _REQUIREMENTS: &str = include_str!("../../../scripts/cvvdp_goldens/requirements.txt");
const _: () = {
    use common::const_str;
    let v_bytes = PYCVVDP_REFERENCE_VERSION.as_bytes();
    let stripped: &[u8] = v_bytes.split_first().unwrap().1;
    assert!(
        const_str::contains(_REQUIREMENTS.as_bytes(), stripped),
        "scripts/cvvdp_goldens/requirements.txt must contain `cvvdp==<version>` \
         matching PYCVVDP_REFERENCE_VERSION (strip leading 'v')",
    );
};

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
