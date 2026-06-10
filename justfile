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
# public-API surface snapshots (docs/public-api/).
fmt:
    cargo fmt $(cargo metadata --no-deps --format-version 1 | jq -r '.packages[].name | "-p " + .')
    cargo test -p zenmetrics-api --test public_api_doc

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test -p zenmetrics-api --test public_api_doc

# Verify the committed snapshots are current (what CI runs)
api-doc-check:
    ZEN_API_DOC=check cargo test -p zenmetrics-api --test public_api_doc

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
