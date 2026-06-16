#!/usr/bin/env bash
# Live demo of capability routing (goal H: "capability-routed (GPU/CPU/ARM)"). One mixed queue;
# workers claim ONLY jobs whose kind's ResourceClass their hardware serves. No R2, no boxes — local.
#
#   GPU worker  (--capability gpu)                  → only Metric jobs   (class Gpu)
#   CPU worker  (--capability cpu_light cpu_heavy)  → only Encode jobs   (CpuLight/CpuHeavy)
#
# Requires: built zenfleet-worker (debug or release), python3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WK="$ROOT/target/debug/zenfleet-worker"; [ -x "$WK" ] || WK="$ROOT/target/release/zenfleet-worker"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

python3 - > "$W/mixed.json" <<'PY'
import json,hashlib
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
jobs =[{"kind":{"kind":"metric","metric":"cvvdp"},"inputs":[sha("m%d"%i)],"cell":{"image_path":"m/%d.png"%i,"codec":"zenjpeg","q":80,"knob_tuple_json":"{}"}} for i in range(6)]
jobs+=[{"kind":{"kind":"encode","codec":"zenjpeg","q":80,"knobs":"{}"},"inputs":[sha("j%d"%i)],"cell":{"image_path":"j/%d.png"%i,"codec":"zenjpeg","q":80,"knob_tuple_json":"{}"}} for i in range(5)]
jobs+=[{"kind":{"kind":"encode","codec":"zenavif","q":50,"knobs":"{}"},"inputs":[sha("a%d"%i)],"cell":{"image_path":"a/%d.png"%i,"codec":"zenavif","q":50,"knob_tuple_json":"{}"}} for i in range(4)]
json.dump(jobs, open("/dev/stdout","w"))
PY
echo "### mixed queue: 6 Gpu(metric) + 5 CpuLight(jpeg) + 4 CpuHeavy(avif) = 15 jobs"
echo -n "[GPU worker --capability gpu] "
"$WK" --manifest "$W/mixed.json" --ledger-out "$W/gpu.parquet" --blobs "$W/blobs" --exec /bin/cat \
  --worker gpu-box --provider gpu --capability gpu 2>&1 | tail -1
echo -n "[CPU worker --capability cpu_light cpu_heavy] "
"$WK" --manifest "$W/mixed.json" --ledger-out "$W/cpu.parquet" --blobs "$W/blobs" --exec /bin/cat \
  --worker cpu-box --provider cpu --capability cpu_light --capability cpu_heavy 2>&1 | tail -1
echo "### Expect GPU done=6 (only metrics), CPU done=9 (only encodes) — capability-routed."
