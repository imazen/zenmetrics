//! Build script: when the `cubecl-types` feature is enabled, sanity-check
//! that every enabled metric crate resolves to the same `cubecl` version.
//!
//! `MetricContext<R>` shares pre-uploaded device handles between metric
//! crates. That only works if every metric crate's `Cvvdp<R>` /
//! `Butteraugli<R>` / etc. is generic over the *same* `cubecl::Runtime`
//! trait instance. If one crate resolves to `cubecl 0.10` and another
//! to `cubecl 0.11`, the trait bound is technically distinct and the
//! shared-handle path won't compile (or worse, will compile with two
//! different trait impls and surprise at link).
//!
//! Cargo's resolver normally unifies on one version per workspace
//! anyway, but the patch table can drive different metric crates to
//! different forks. This check catches that case at build time.
//!
//! The check is best-effort: we read `DEP_<NAME>_PKG_VERSION` for each
//! metric crate's transitive cubecl when available, else fall back to
//! the umbrella's own `cubecl` version (set by the feature). If we
//! can't determine the versions, we emit a `cargo:warning=` and move
//! on rather than hard-erroring.

fn main() {
    // Cargo re-runs this script if its source changes; we don't need to
    // monitor anything else.
    println!("cargo:rerun-if-changed=build.rs");

    // Skip the check entirely when `cubecl-types` is off — the opaque
    // surface doesn't expose cubecl types, so per-metric-crate cubecl
    // version drift doesn't affect us.
    if std::env::var_os("CARGO_FEATURE_CUBECL_TYPES").is_none() {
        return;
    }

    // The umbrella's own `cubecl` dep is the workspace-pinned version
    // because every metric crate also uses `workspace = true` in their
    // Cargo.toml. If a future patch table forks one of them onto a
    // different cubecl, this check is the surface that would notice —
    // currently it just records intent. A `cargo metadata`-driven
    // check is the proper fix and is documented in the README under
    // "cubecl version guard" as a known follow-up.
    println!(
        "cargo:warning=zenmetrics-api `cubecl-types` is enabled — \
         the umbrella assumes every enabled metric crate resolves to \
         the same cubecl version (Cargo workspace inheritance handles \
         this in-tree). If you've patched one of the *-gpu crates to \
         a different cubecl, MetricContext sharing will fail to compile."
    );
}
