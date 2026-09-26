#!/usr/bin/env bash
set -euo pipefail
python3 scripts/gmsd-chroma/preflight.py
python3 scripts/gmsd-chroma/ensure_cargo_patch.py
# Crate tests/no_std/clippy and CLI check already passed in build_checks_v2.
# The full CLI clippy dependency walk stopped on inherited GPU warnings.
clippy_status=0
cargo clippy -p zenmetrics-cli --no-deps --no-default-features --features cpu-gmsd,png,jpeg,sweep -- -D warnings || clippy_status=$?
python3 - "$clippy_status" <<'PY'
import json,sys
from pathlib import Path
p=Path('/var/tmp/gmsd-chroma/cli_clippy_status.json')
p.write_text(json.dumps(dict(exit_code=int(sys.argv[1]),passed=int(sys.argv[1])==0))+'\n')
print('CLI_CLIPPY_EXIT',sys.argv[1],flush=True)
PY
# Build/ingress checks are independent of the inherited CLI lint failures.
# Retain their failure status above; never present it as a clippy pass.
cargo build -p zenmetrics-cli --release --no-default-features --features cpu-gmsd,png,jpeg,sweep
cargo fmt -p gmsd -p zenmetrics-cli -- --check
