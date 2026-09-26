#!/usr/bin/env bash
# Invoke only through /home/lilith/tmp/devin/heavy and record_command.py.
set -euo pipefail
root=/var/tmp/gmsd-chroma
test ! -e /home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md
python3 scripts/gmsd-chroma/preflight.py
python3 scripts/gmsd-chroma/verify_zenbench_snapshot.py
export CARGO_HOME="$root/cache/cargo"
export CARGO_TARGET_DIR="$root/target"
export TMPDIR="$root/tmp"
export XDG_CACHE_HOME="$root/cache"
mkdir -p "$CARGO_HOME"
# Copy dependency caches, never credentials/config or mutable hardlinks.
# Existing caches are read-only inputs; all cargo writes stay under /var/tmp.
if [[ ! -f "$CARGO_HOME/.seeded" ]]; then
    for kind in registry git; do
        cp -a --reflink=auto "/home/lilith/.cargo/$kind" "$CARGO_HOME/"
    done
    touch "$CARGO_HOME/.seeded"
fi
python3 scripts/gmsd-chroma/ensure_cargo_patch.py
rustc --version
cargo --version
cargo fmt -p gmsd -p zenmetrics-cli
# Published 0.1.9 omits raw rounds. Freeze the inspected imazen harness with
# raw-round retention externally and apply it only for this lane's checks.
lane_cargo() {
    cargo --config 'patch.crates-io.zenbench.path="/var/tmp/gmsd-chroma/zenbench-snapshot"' "$@"
}
lane_cargo test -p gmsd --all-features
lane_cargo check -p gmsd --no-default-features
python3 scripts/gmsd-chroma/build_examples.py
lane_cargo clippy -p gmsd --all-targets --all-features -- -D warnings
lane_cargo check -p zenmetrics-cli --no-default-features --features cpu-gmsd,png,jpeg,sweep
lane_cargo clippy -p zenmetrics-cli --no-deps --no-default-features --features cpu-gmsd,png,jpeg,sweep -- -D warnings
lane_cargo build -p zenmetrics-cli --release --no-default-features --features cpu-gmsd,png,jpeg,sweep
cargo fmt -p gmsd -p zenmetrics-cli -- --check
