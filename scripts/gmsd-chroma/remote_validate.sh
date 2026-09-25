#!/usr/bin/env bash
# The caller holds worker 2's shared heavy lock.
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
cd "$root/remote-src"
mkdir "$root/remote-results"
exec >"$root/remote-results/validation.log" 2>&1
date -u +%FT%TZ >"$root/remote-results/start.txt"
trap 'code=$?; date -u +%FT%TZ >"$root/remote-results/end.txt"; echo "$code" >"$root/remote-results/exit.txt"; for name in parity.json ms_parity.json build_examples_latest.json; do if test -f "$root/$name"; then cp "$root/$name" "$root/remote-results/$name"; fi; done; cp Cargo.lock "$root/remote-results/Cargo.lock"; exit "$code"' EXIT
rustc -vV
cargo -V
python3 - <<'PY'
import hashlib,json
from pathlib import Path
r=Path('/var/tmp/gmsd-chroma')
m=json.loads((r/'remote_source_manifest.json').read_text())
for name,digest in m['files'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
print('frozen_source_files_verified',len(m['files']),flush=True)
PY
bash scripts/gmsd-chroma/validate_oracles.sh
