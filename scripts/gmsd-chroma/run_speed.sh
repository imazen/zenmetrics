#!/usr/bin/env bash
# Invoke through heavy. The benchmark uses a neutral private UTS hostname.
set -euo pipefail
python3 scripts/gmsd-chroma/preflight.py
python3 scripts/gmsd-chroma/verify_zenbench_snapshot.py
python3 scripts/gmsd-chroma/verify_build.py chroma_speed
root=/var/tmp/gmsd-chroma
test ! -e "$root/speed"
sha256sum "$root/target/release/examples/chroma_speed"
bash scripts/gmsd-chroma/speed_namespace.sh "$root/target/release/examples/chroma_speed" "$root/speed"
python3 scripts/gmsd-chroma/report_speed.py "$root/speed/zenbench.json" "$root/speed/report.json"
