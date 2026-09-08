#!/usr/bin/env bash
# Build portable fleet tools and verify linkage, under the shared resource cap.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ${CARGO_ENCODED_RUSTFLAGS+x} ]]; then
  export CARGO_ENCODED_RUSTFLAGS="${CARGO_ENCODED_RUSTFLAGS}"$'\x1f-C\x1ftarget-feature=+crt-static'
else
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+crt-static"
fi
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "$TMPDIR"
av1_heavy="${RUN_HEAVY:-$HOME/work/claudehints/scripts/run-heavy}"
"$av1_heavy" --mem 16G --jobs 4 -- cargo test --release --target x86_64-unknown-linux-gnu --lib --bin zenmetrics-av1-compare -j 4
"$av1_heavy" --mem 16G --jobs 4 -- cargo build --release --target x86_64-unknown-linux-gnu --bins -j 4
for av1_name in zenmetrics-av1-compare zenfleet-worker zenfleet-ctl; do
  av1_binary="target/x86_64-unknown-linux-gnu/release/$av1_name"
  av1_dynamic=$(readelf -d "$av1_binary")
  if [[ "$av1_dynamic" == *NEEDED* ]]; then
    echo "refusing fleet artifact with dynamic dependencies: $av1_binary" >&2
    exit 1
  fi
  file "$av1_binary"
  sha256sum "$av1_binary"
done
