#!/usr/bin/env bash
# One native zenfleet pass. All selection, claims, heartbeats and output writes
# stay in zenfleet-worker. Caller supplies a manifest with unchanged native IDs.
set -euo pipefail
native_root=$(realpath "${1:?runtime directory}")
native_arm=${2:?arm}
native_index=${3:?worker index}
[[ "$native_arm" =~ ^[A-Za-z][A-Za-z0-9_+-]*$ && "$native_index" =~ ^[1-9][0-9]*$ ]]
native_repo=$(cd "$(dirname "$0")/../.." && pwd)
if [[ "${4:-}" != --inside ]]; then
  exec bash "$native_repo/scripts/jobsys/native_fit_namespace.sh" \
    "$native_root/ns-$native_arm" "$native_root/program" "$native_root/scratch" \
    bash "$(realpath "$0")" "$native_root" "$native_arm" "$native_index" --inside
fi
# Resolve credentials as the calling account after the namespace drops privilege.
# shellcheck disable=SC1091
source "$native_repo/scripts/lib/s3env.sh"
export TMPDIR=/scratch/tmp ZEN_FIT_PROGRAM_SHA=423259e7dcb70ec6fee36711bc3e40512b3dc97b1c9f6114cb0adfffbad3734a
export ZEN_BUCKET=zentrain ZEN_RUN="jobs/byc-$native_arm-r2-20261025" ZEN_R2_ENDPOINT="$EP"
if [[ -f "$native_root/program.sha256" ]]; then
  IFS= read -r ZEN_FIT_PROGRAM_SHA < "$native_root/program.sha256"
  [[ "$ZEN_FIT_PROGRAM_SHA" =~ ^[0-9a-f]{64}$ ]]
fi
native_run="byc-$native_arm-r2-20261025"
if [[ -f "$native_root/run-$native_arm" ]]; then
  IFS= read -r native_run < "$native_root/run-$native_arm"
  [[ "$native_run" =~ ^[A-Za-z0-9_-]+$ ]]
  export ZEN_RUN="jobs/$native_run"
fi
export ZENSIM_MAX_TIER=v3 ZEN_FIT_RAYON_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Five GiB is an admission envelope, not a physical-RAM claim: 75% / 3.75GiB hint = one slot.
native_worker_bin=${ZEN_FIT_WORKER_BIN:-$native_root/../../zenfleet-worker}
"$native_worker_bin" capabilities | python3 -c 'import json,sys; assert json.load(sys.stdin).get("max_chunks_per_pass") is True, "refresh native worker before launching bounded fit passes"'
export ZEN_MAX_CHUNKS_PER_PASS=1
export ZEN_RAM_BYTES=5368709120 ZEN_CORE_OVERSUBSCRIBE=1 ZEN_CHUNK_WALL_SEC=300 ZEN_CLAIM_ORDER=manifest
if [[ -f "$native_root/worker-ram-bytes" ]]; then
  IFS= read -r ZEN_RAM_BYTES < "$native_root/worker-ram-bytes"
  [[ "$ZEN_RAM_BYTES" =~ ^[1-9][0-9]*$ ]]
fi
native_worker="$(hostname)-byc-native-$native_arm-$native_index"
native_control=control.json
if [[ -f "$native_root/control-key-suffix" ]]; then
  IFS= read -r native_control < "$native_root/control-key-suffix"
fi
[[ "$native_control" =~ ^control(-[a-zA-Z0-9_-]+)?\.json$ ]]
native_snapshot="$native_root/$native_arm/snapshot-$native_index"
native_manifest="$native_root/$native_arm/manifest.json"
native_subset="$native_root/$native_arm/manifest-$native_index.json"
native_ttl=${ZEN_CLAIM_TTL_SECS:-86400}
if [[ -f "$native_root/claim-ttl-secs" ]]; then
  IFS= read -r native_ttl < "$native_root/claim-ttl-secs"
  [[ "$native_ttl" =~ ^[1-9][0-9]*$ && "$native_ttl" -ge 600 ]]
fi
if [[ -f "$native_subset" ]]; then
  python3 - "$native_manifest" "$native_subset" <<'PYSUBSET'
import json, sys
from pathlib import Path
full, subset = (json.loads(Path(p).read_text()) for p in sys.argv[1:])
if not subset or any(full.count(job) != 1 for job in subset) or len({job['cell']['image_path'] for job in subset}) != len(subset):
    raise ValueError('native retry must be an unchanged subset of the original manifest')
PYSUBSET
  native_manifest="$native_subset"
  if [[ -f "$native_root/$native_arm/claim-ttl-$native_index" ]]; then
    IFS= read -r native_ttl < "$native_root/$native_arm/claim-ttl-$native_index"
    [[ "$native_ttl" =~ ^[1-9][0-9]*$ && "$native_ttl" -ge 600 ]]
  fi
fi
mkdir -p "$native_snapshot"
"$native_root/../../zenfleet-ctl" compact --run "$native_run" \
  --endpoint "$EP" --out-dir "$native_snapshot" --upload
native_ledger=()
native_snapfile="$native_snapshot/snap_$native_run.parquet"
[[ ! -f "$native_snapfile" ]] || native_ledger=(--ledger-in "$native_snapfile")
mkdir -p "$TMPDIR"
native_scope_receipt() {
  native_rc=$?
  python3 - "$native_root" "$native_worker" "$native_rc" <<'PYMEM'
from pathlib import Path
from datetime import datetime, timezone
import json, sys
root, worker, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
path = Path('/sys/fs/cgroup') / (root/'scope.cgroup').read_text().strip().split('::', 1)[1].lstrip('/')
report = dict(observed_at=datetime.now(timezone.utc).isoformat(), worker=worker, exit_code=rc,
              accounting='aggregate enclosing run-heavy scope after native pass',
              **{k: (path/k).read_text().strip() for k in ('memory.current', 'memory.peak', 'memory.max', 'memory.events')})
with (root/f'scope-after-{worker}.json').open('x') as stream:
    json.dump(report, stream, indent=2)
    stream.write('\n')
PYMEM
  return "$native_rc"
}
trap native_scope_receipt EXIT
"$native_worker_bin" \
  --manifest "$native_manifest" "${native_ledger[@]}" \
  --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/pass-$native_worker.parquet" \
  --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
  --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims" \
  --claim-ttl-secs "$native_ttl" --control-r2-key "$ZEN_RUN/$native_control" --capability cpu_heavy \
  --r2-endpoint "$EP" --exec /opt/fleet-fits/program/fit_cell_exec.py \
  --worker "$native_worker" --provider local
