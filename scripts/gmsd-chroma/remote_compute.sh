#!/usr/bin/env bash
# Invoke only through worker 2's shared heavy wrapper.
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
cd "$root/remote-src"
python3 scripts/gmsd-chroma/preflight.py
case "$1" in
speed)
    python3 - <<'PY'
import json
from pathlib import Path
r=Path('/var/tmp/gmsd-chroma')
for n in ['parity.json','ms_parity.json']:
 assert json.loads((r/n).read_text())['passed']
PY
    python3 scripts/gmsd-chroma/verify_zenbench_snapshot.py
    python3 scripts/gmsd-chroma/verify_build.py chroma_speed
    test ! -e "$root/speed"
    bash scripts/gmsd-chroma/speed_namespace.sh "$root/target/release/examples/chroma_speed" "$root/speed"
    ;;
peer-build)
    python3 scripts/gmsd-chroma/build_fast_peer.py
    ;;
predict)
    python3 scripts/gmsd-chroma/score_colour.py --source-manifest "$root/remote_source_manifest.json"
    ;;
stats-deps)
    "$root/bin-uv" --no-config venv --no-python-downloads --python /usr/bin/python3 "$root/stats-env"
    "$root/bin-uv" --no-config pip install --python "$root/stats-env/bin/python" numpy==2.5.2
    ;;
colour-report)
    "$root/stats-env/bin/python" scripts/gmsd-chroma/report_colour.py \
        --table "$root/colour_v2/report_inputs.tsv" \
        --manifest "$root/colour_v2/report_inputs.manifest.json" \
        --parity "$root/parity.json" --ms-parity "$root/ms_parity.json" \
        --panel "$root/panel" --owner "$root/stats-owner" \
        --out "$root/colour_v2/report.json"
    ;;
*) exit 2;;
esac
