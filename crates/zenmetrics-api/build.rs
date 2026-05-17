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
//! different forks. This check is intended to catch that case at
//! build time.
//!
//! ## Why this is an advisory warning, not a hard error
//!
//! A *real* check would shell out to `cargo metadata --format-version 1`
//! and parse the resulting JSON to enumerate every transitive
//! `cubecl-*` package and verify they all resolve to one version.
//! Implementing that correctly requires either:
//!   - adding `cargo_metadata` as a build-dep (~10 transitive crates,
//!     including `serde_json` and `semver`, just to drive a sanity
//!     check that almost never trips), or
//!   - hand-rolling a JSON parser in the build script (brittle and
//!     adds a new failure mode), or
//!   - calling `cargo tree --duplicates -i cubecl` and parsing
//!     human-readable output (brittle — `cargo tree`'s formatting is
//!     not stable across Cargo versions).
//!
//! The workspace's `cubecl = { workspace = true }` line in every
//! metric crate's `Cargo.toml` already enforces single-version
//! unification in-tree. The advisory below is what the cross-tree
//! patch case (a downstream `[patch.crates-io]` entry forking one
//! metric crate onto a different cubecl) needs to surface — and that
//! case is rare enough that a warning + README note is the right
//! tradeoff vs. a build-dep that bloats every check by ~10 crates.
//!
//! If this check trips often enough to matter, upgrade to the real
//! `cargo metadata` parse and bail with `cargo:error=` instead.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_CUBECL_TYPES").is_none() {
        return;
    }

    println!(
        "cargo:warning=zenmetrics-api `cubecl-types` is enabled — \
         the umbrella assumes every enabled metric crate resolves to \
         the same cubecl version (Cargo workspace inheritance handles \
         this in-tree). If you've patched one of the *-gpu crates to \
         a different cubecl, MetricContext sharing will fail to compile."
    );
}
