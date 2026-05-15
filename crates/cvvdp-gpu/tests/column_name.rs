//! Regression tests pinning the `CVVDP_COLUMN_NAME` contract.
//!
//! Downstream sweep tooling lands cvvdp scores in parquet sidecars
//! keyed on this string (see `docs/CVVDP_SIDECAR_SCHEMA.md` and the
//! "Reserved implementation tags" table). A typo, a version bump
//! that produces an unexpected layout, or an env-override regression
//! would silently mis-name new columns and break joins against
//! historical sidecars. These tests catch that at the crate's own
//! `cargo test`, before the misbehavior propagates to a fleet run.
//!
//! The build-time `CVVDP_IMPL_TAG` env override is exercised
//! end-to-end by `zen-metrics-cli`'s build pipeline (see the
//! `CVVDP_IMPL_TAG=...` invocation in `CVVDP_SIDECAR_SCHEMA.md`);
//! these tests cover the default (un-overridden) path only.

use cvvdp_gpu::CVVDP_COLUMN_NAME;

#[test]
fn column_name_is_not_empty() {
    assert!(
        !CVVDP_COLUMN_NAME.is_empty(),
        "CVVDP_COLUMN_NAME must be non-empty; sweep harnesses use it as a parquet column key"
    );
}

#[test]
fn column_name_has_cvvdp_prefix() {
    assert!(
        CVVDP_COLUMN_NAME.starts_with("cvvdp_"),
        "CVVDP_COLUMN_NAME must start with `cvvdp_` so downstream tooling \
         can discriminate cvvdp columns from other metric families; got {CVVDP_COLUMN_NAME:?}"
    );
}

#[test]
fn column_name_uses_only_parquet_safe_chars() {
    // ASCII letters, digits, and underscore are the only chars that
    // survive every downstream tool (parquet column names, TSV
    // headers, R2 filename derivations, Python attribute access). No
    // whitespace, no path separators, no shell metachars.
    for c in CVVDP_COLUMN_NAME.chars() {
        assert!(
            c.is_ascii_alphanumeric() || c == '_',
            "CVVDP_COLUMN_NAME must contain only ASCII alphanumerics and underscores; \
             got disallowed char {c:?} in {CVVDP_COLUMN_NAME:?}"
        );
    }
}

#[test]
fn default_column_name_encodes_crate_version() {
    // When `CVVDP_IMPL_TAG` is NOT set at compile time, the default
    // form derives from `CARGO_PKG_VERSION`:
    //     cvvdp_imazen_v<MAJOR>_<MINOR>_<PATCH>
    // Skip the assertion if the build overrode `CVVDP_IMPL_TAG` —
    // that path is exercised by the `zen-metrics-cli` release build
    // (see `docs/CVVDP_SIDECAR_SCHEMA.md`).
    if option_env!("CVVDP_IMPL_TAG").is_some() {
        eprintln!(
            "skipping default-form assertion; CVVDP_IMPL_TAG was set at compile time \
             (override path is exercised by the zen-metrics release build)"
        );
        return;
    }

    let major = env!("CARGO_PKG_VERSION_MAJOR");
    let minor = env!("CARGO_PKG_VERSION_MINOR");
    let patch = env!("CARGO_PKG_VERSION_PATCH");
    let expected = format!("cvvdp_imazen_v{major}_{minor}_{patch}");

    assert_eq!(
        CVVDP_COLUMN_NAME, expected,
        "default-form CVVDP_COLUMN_NAME should encode the crate version; \
         expected {expected:?}, got {CVVDP_COLUMN_NAME:?}"
    );
}

#[test]
fn column_name_starts_with_cvvdp_imazen_in_default_build() {
    // Reserved-tag discriminator: the canonical pycvvdp reference
    // uses `cvvdp_pycvvdp_v054`; a future Burn port reserved
    // `cvvdp_burn_v*`. This crate must always claim
    // `cvvdp_imazen_*` to avoid collisions with those.
    //
    // Only enforced in the default (un-overridden) build because
    // the env override path is intentionally a free-form escape
    // hatch — e.g., baking a git short hash in CI is a legitimate
    // override.
    if option_env!("CVVDP_IMPL_TAG").is_some() {
        eprintln!(
            "skipping cvvdp_imazen_ prefix assertion; CVVDP_IMPL_TAG was set at compile time"
        );
        return;
    }

    assert!(
        CVVDP_COLUMN_NAME.starts_with("cvvdp_imazen_"),
        "default-form CVVDP_COLUMN_NAME must start with `cvvdp_imazen_` to claim \
         this crate's reserved tag; got {CVVDP_COLUMN_NAME:?}"
    );
}
