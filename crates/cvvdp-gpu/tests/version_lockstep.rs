//! Compile-time lockstep pins binding `PYCVVDP_REFERENCE_VERSION`
//! to every current-state external file that references the
//! pycvvdp reference version. Moved out of `tests/parity.rs`
//! (tick 595) so the pins fire on every `cargo check / test`
//! rather than only when `--features parity-goldens` is on.
//!
//! Pinned files (ticks 588-594 in the include_str!() lockstep arc):
//! - `parity.rs::manifest_fetches` — runtime check against const
//!   (stays in parity.rs because it needs the goldens manifest)
//! - `scripts/cvvdp_goldens/requirements.txt`
//! - `src/kernels/csf_lut/v0_5_4.rs` (LUT header comment)
//! - `docs/PORT_STATUS.md`
//! - `README.md`
//! - `Cargo.toml` (parity-goldens feature comment)
//! - `docs/CVVDP_SIDECAR_SCHEMA.md` (reserved column-name tags)
//!
//! Files INTENTIONALLY NOT pinned:
//! - `docs/CHROMA_DRIFT_INVESTIGATION.md` — historical bug-hunt
//!   audit, not current-state. Pinning would cement an old
//!   investigation log against future reference bumps.
//! - `docs/BURN_PORT_PLAN.md` — ABANDONED project plan, historical.
//!
//! Format pins on the const itself:
//! - non-empty
//! - first byte == 'v'
//! - contains at least one '.'
//!
//! Tick 595 also adds the const-format pin block (previously in
//! parity.rs under the parity-goldens gate).

#[path = "common/mod.rs"]
#[allow(dead_code)]
mod common;

use cvvdp_gpu::PYCVVDP_REFERENCE_VERSION;

// Format invariants on the const itself.
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

// requirements.txt: strip leading 'v' to match pip's bare-version format.
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

// Auto-generated LUT file header comment.
const _LUT_V0_5_4: &str = include_str!("../src/kernels/csf_lut/v0_5_4.rs");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_LUT_V0_5_4.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "src/kernels/csf_lut/v0_5_4.rs header must contain PYCVVDP_REFERENCE_VERSION",
    );
};

// PORT_STATUS.md "Reference version pin" section.
const _PORT_STATUS: &str = include_str!("../docs/PORT_STATUS.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_PORT_STATUS.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "docs/PORT_STATUS.md must contain PYCVVDP_REFERENCE_VERSION (Reference version pin section)",
    );
};

// README.md (4 references: algorithm-parity claim, PerfMode::Strict, parity-goldens feature, Status).
const _README: &str = include_str!("../README.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_README.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "crates/cvvdp-gpu/README.md must contain PYCVVDP_REFERENCE_VERSION",
    );
};

// Cargo.toml parity-goldens feature comment.
const _CARGO_TOML: &str = include_str!("../Cargo.toml");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_CARGO_TOML.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "crates/cvvdp-gpu/Cargo.toml must contain PYCVVDP_REFERENCE_VERSION (parity-goldens feature comment)",
    );
};

// CVVDP_SIDECAR_SCHEMA.md reserved column-name tags table.
const _SIDECAR_SCHEMA: &str = include_str!("../docs/CVVDP_SIDECAR_SCHEMA.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_SIDECAR_SCHEMA.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "docs/CVVDP_SIDECAR_SCHEMA.md must contain PYCVVDP_REFERENCE_VERSION (reserved column-name tags)",
    );
};

// Runtime touchpoint so the test runner can name this file in coverage output.
#[test]
fn pycvvdp_reference_version_lockstep_pins_compile() {
    // All the load-bearing work is in the const _: () = ...
    // blocks above. This fn exists so `cargo test` reports
    // a named pass for the lockstep-pins file.
    assert!(!PYCVVDP_REFERENCE_VERSION.is_empty());
}
