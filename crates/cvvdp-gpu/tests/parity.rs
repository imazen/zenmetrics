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

// Tick 590: pin the vendored LUT file's header comment against
// PYCVVDP_REFERENCE_VERSION. The auto-generated header in
// `src/kernels/csf_lut/v0_5_4.rs` reads
//   "Auto-generated from pycvvdp v0.5.4's csf_lut_weber_fixed_size.json."
// so it should contain the FULL `v0.5.4` string (matches the const
// exactly, no v-stripping needed). When the reference bumps, the
// LUT file regen procedure updates the header — this pin catches a
// version mismatch between the const and the vendored data.
//
// `include_str!` reads the whole LUT at compile time (~1000+ lines
// of f32 literals), but the substring search is O(n·m) on a small
// needle — fast enough.
const _LUT_V0_5_4: &str = include_str!("../src/kernels/csf_lut/v0_5_4.rs");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_LUT_V0_5_4.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "src/kernels/csf_lut/v0_5_4.rs header must contain PYCVVDP_REFERENCE_VERSION \
         (the auto-generated header comment references the source pycvvdp version)",
    );
};

// Tick 591: pin PORT_STATUS.md against PYCVVDP_REFERENCE_VERSION.
// The "Reference version pin" section in `docs/PORT_STATUS.md`
// reads "gfxdisp/ColorVideoVDP **v0.5.4** (latest tag as of ...)"
// — when bumping the reference, this prose doc must update in
// the same commit. `include_str!` reads it at compile time;
// `const_str::contains` finds the substring.
//
// This closes the prose-documentation site listed in
// PYCVVDP_REFERENCE_VERSION's docstring. The two remaining sites
// (Rust module name `csf_lut_v0_5_4` and filesystem path
// `csf_lut/v0_5_4.rs`) are identifier / path entities that can't
// be string-matched at compile time via const_str — they'd need
// a build.rs that introspects the source tree, which isn't worth
// the complexity for these documentation-only references.
const _PORT_STATUS: &str = include_str!("../docs/PORT_STATUS.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_PORT_STATUS.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "docs/PORT_STATUS.md must contain PYCVVDP_REFERENCE_VERSION (Reference version pin section)",
    );
};

// Tick 592: pin crate-level README.md against PYCVVDP_REFERENCE_VERSION.
// The README references the pycvvdp v0.5.4 reference in multiple
// places (algorithm-parity claim, PerfMode::Strict semantics,
// parity-goldens feature description, Status section). Pinning at
// compile time forces user-facing docs to update in lockstep with
// the const + parity-test + requirements.txt + LUT header +
// PORT_STATUS.md.
const _README: &str = include_str!("../README.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_README.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "crates/cvvdp-gpu/README.md must contain PYCVVDP_REFERENCE_VERSION",
    );
};

// Tick 593: pin Cargo.toml against PYCVVDP_REFERENCE_VERSION. The
// `parity-goldens` feature comment reads "Enables integration
// tests that fetch the pycvvdp v0.5.4 goldens from R2 ..."; users
// reading the feature list see this version and expect it to
// match the actual reference. Pinning forces the comment to
// update in lockstep when the const bumps.
const _CARGO_TOML: &str = include_str!("../Cargo.toml");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_CARGO_TOML.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "crates/cvvdp-gpu/Cargo.toml must contain PYCVVDP_REFERENCE_VERSION (parity-goldens feature comment)",
    );
};

// Tick 594: pin docs/CVVDP_SIDECAR_SCHEMA.md against
// PYCVVDP_REFERENCE_VERSION. The "Reserved column-name tags" table
// documents that `cvvdp_pycvvdp_v054` corresponds to "upstream
// pycvvdp v0.5.4" — load-bearing for current-state sweep tooling
// docs.
//
// (CHROMA_DRIFT_INVESTIGATION.md is intentionally NOT pinned —
// its v0.5.4 references are historical (tick-200-era bug-hunt
// audit), not current-state. Pinning would cement that the doc
// must always reference v0.5.4 even after future reference bumps,
// which isn't right for an historical investigation log.)
const _SIDECAR_SCHEMA: &str = include_str!("../docs/CVVDP_SIDECAR_SCHEMA.md");
const _: () = {
    use common::const_str;
    assert!(
        const_str::contains(_SIDECAR_SCHEMA.as_bytes(), PYCVVDP_REFERENCE_VERSION.as_bytes()),
        "docs/CVVDP_SIDECAR_SCHEMA.md must contain PYCVVDP_REFERENCE_VERSION (reserved column-name tags)",
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
