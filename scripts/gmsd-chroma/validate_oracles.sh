#!/usr/bin/env bash
# One validation job; invoke through the shared heavy wrapper.
set -euo pipefail
python3 scripts/gmsd-chroma/preflight.py
# The caller decides the 116-pair author-score gate: it is required here.
export GMSD_MDSI_GATE=require GMSD_MDSI_TARGETS="${GMSD_MDSI_TARGETS:-/var/tmp/gmsd-chroma/mdsi_gate_bundle/target.tsv}"
python3 scripts/gmsd-chroma/ensure_cargo_patch.py
cargo fmt -p gmsd
cargo fmt -p gmsd -- --check
cargo test -p gmsd --all-features
cargo test -p gmsd --release --all-features
cargo check -p gmsd --no-default-features
cargo test -p gmsd --no-default-features
cargo clippy -p gmsd --all-targets --all-features -- -D warnings
python3 scripts/gmsd-chroma/build_examples.py
python3 scripts/gmsd-chroma/verify_build.py mdsi_oracle
root=/var/tmp/gmsd-chroma
test -s "$root/oracle_inputs.tsv"
test -s "$root/octave/scores.tsv"
test -s "$root/ms_numpy/scores.tsv"
test ! -e "$root/rust"
binary="$root/target/release/examples/mdsi_oracle"
sha256sum "$binary"
export RAYON_NUM_THREADS=8
"$binary" "$root/oracle_inputs.tsv" "$root/rust"
# Preserve both reports if one gate fails; never weaken either tolerance.
status=0
python3 scripts/gmsd-chroma/compare_oracle.py || status=1
python3 scripts/gmsd-chroma/compare_ms.py || status=1
exit "$status"
