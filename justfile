# zenmetrics justfile — common dev/CI commands.
# `just` (1.x) required. Run `just` with no args to list recipes.

# Feature combo CI uses for the GPU-less zenmetrics-api CPU job: wgpu so
# the -gpu crates build without the CUDA SDK, but `cuda` OFF so the
# matrix's CPU-vs-CUDA parity layer compiles out (see ci.yml
# cpu-metrics-tests). Kept here verbatim so `just test-cpu` reproduces CI.
CPU_FEATURES := "all-metrics,cpu-metrics,wgpu,pixels,encoded"

# List recipes.
default:
    @just --list

# NEVER `cargo fmt --all` — rustfmt can follow `mod`/path-dep edges into the
# patched sibling repos (../zensim, ../../butteraugli, …) and rewrite files
# we don't own. `cargo metadata --no-deps` lists exactly this workspace's
# members (verified to exclude every sibling path-dep); we expand it to a
# `-p NAME` list so only in-repo crates are touched.

# Format in-repo workspace packages only (sibling-safe) + regenerate the
# public-API surface snapshots (docs/public-api/). The snapshot runner lives
# in the workspace-excluded apidoc/ package, so it is never built or run by
# plain `cargo test` or any CI job.
fmt:
    cargo fmt $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# Formatting check over the same in-repo package set (sibling-safe).
fmt-check:
    cargo fmt --check $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')

# zenmetrics-api optimized-CPU backend tests — EXACT mirror of CI's
# `cpu-metrics-tests` job (Backend::Cpu dispatch + backend×metric×size
# matrix, GPU-less: cuda off → CPU-vs-CUDA parity layer gated out).

# Run the GPU-less CPU-backend test suite (mirrors CI).
test-cpu:
    cargo test -p zenmetrics-api --no-default-features --features {{CPU_FEATURES}} \
        --test it -- backend_matrix cpu_dispatch

# The full matrix including the CPU-vs-CUDA parity layer needs an NVIDIA
# GPU; default features turn `cuda` on so the `#[cfg(feature = "cuda")]`
# parity tests compile in and run.

# Run the full backend matrix locally, including CPU-vs-CUDA parity (needs GPU).
test-matrix-gpu:
    cargo test -p zenmetrics-api --features cpu-metrics --test it backend_matrix

# Quick default-feature check of the umbrella crate.
check:
    cargo check -p zenmetrics-api

# ghcr package-name guard: fail if any active-infra file references a
# ghcr.io/imazen/<name> that isn't a canonical package in ghcr-packages.json.
# One package per artifact; variants are TAGS. See docs/GHCR_PACKAGES.md.
ghcr-check:
    python3 scripts/ci/check_ghcr_packages.py

# Strict: also fail on grandfathered splinters still referenced in infra.
# Flip CI to this once ghcr-packages.json's `deprecated` map is empty.
ghcr-check-strict:
    python3 scripts/ci/check_ghcr_packages.py --strict

# Audit the LIVE ghcr.io/imazen packages against the manifest (needs `gh`
# authed with read:packages). Prints orphans + a commented migrate/delete recipe.
ghcr-audit:
    python3 scripts/ci/audit_ghcr_org.py

# Fleet-tooling guard: fail if a new launch_*/onstart_*/fleet*/*_watch script
# appears outside the canonical set in fleet-tools.json. ONE tool per concern —
# add a subcommand to `scripts/jobsys/fleet`, not a new script.
fleet-check:
    python3 scripts/ci/check_fleet_tools.py

# Strict: also fail on grandfathered forks (the post-Phase-E gate).
fleet-check-strict:
    python3 scripts/ci/check_fleet_tools.py --strict
