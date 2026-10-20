#!/usr/bin/env bash
# Build the canonical scalar CPU worker (iqa-compete) at REV in a fresh clone and pinned snapshot, and record
# its build receipt for canonical_bench.py / prepare_durable.py.
#
#   build_canonical_worker.sh NEW_ROOT REV
#
# Features match the 2026-10-19 canonical scalar campaigns (compete,metrics-parallel,zenmetrics-cli/nr). On macOS
# the private zenbench cumulative-counter correction those campaigns used is overlaid on the snapshot
# (patch_mac_cpu_counters.py; it refuses if the pinned zenbench is no longer ac675d3). Run it under the host's
# heavy lock and resource wrapper. Writes NEW_ROOT/{source,snapshot,build-command.json,build.json}.
set -euo pipefail
root=${1:?NEW_ROOT}; rev=${2:?REV}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
if [ -e "$root" ]; then echo "build_canonical_worker: $root exists; choose a new root" >&2; exit 2; fi
mkdir -p "$root"; root=$(cd "$root" && pwd)
export TMPDIR="$HOME/tmp" LOCK_SNAP_DIR="$root/snapshot" CARGO_TARGET_DIR="$root/snapshot/target"
jj git clone --colocate https://github.com/imazen/zenmetrics "$root/source"
cd "$root/source"
jj new "$rev" -m "bench: canonical worker build" >/dev/null
scripts/ci/lock.sh --check --rev "$rev"
if [ "$(uname -s)" = Darwin ]; then
  python3 benchmarks/iqa_compete_2026-10-16/patch_mac_cpu_counters.py "$LOCK_SNAP_DIR/work/zenmetrics" "$root/source" \
    "$LOCK_SNAP_DIR/mac-cpu-counters.meta.json"
fi
features=compete,metrics-parallel,zenmetrics-cli/nr
cd "$LOCK_SNAP_DIR/work/zenmetrics"
cargo build --locked --release -p cpu-profile --features "$features" --bin iqa-compete
printf '["cargo","build","--locked","--release","-p","cpu-profile","--features","%s","--bin","iqa-compete"]\n' "$features" \
  > "$root/build-command.json"
cd "$root/source"
python3 benchmarks/iqa_compete_2026-10-16/record_durable_build.py "$CARGO_TARGET_DIR/release/iqa-compete" "$root/build.json" \
  --source-commit "$rev" --features "$features" --snapshot "$LOCK_SNAP_DIR" --command-file "$root/build-command.json"
