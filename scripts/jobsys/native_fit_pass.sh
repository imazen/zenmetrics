#!/usr/bin/env bash
# One native zenfleet pass. All selection, claims, heartbeats and output writes
# stay in zenfleet-worker. Caller supplies a manifest with unchanged native IDs.
set -euo pipefail
native_root=$(realpath "${1:?runtime directory}")
native_arm=${2:?arm}
native_index=${3:?worker index}
[[ "$native_arm" =~ ^(Mw100|Mmos|Mpair)$ && "$native_index" =~ ^[1-9][0-9]*$ ]]
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
export ZENSIM_MAX_TIER=v3 ZEN_FIT_RAYON_THREADS=1 RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Five GiB is an admission envelope, not a physical-RAM claim: 75% / 3.75GiB hint = one slot.
export ZEN_RAM_BYTES=5368709120 ZEN_CORE_OVERSUBSCRIBE=1 ZEN_CHUNK_WALL_SEC=300 ZEN_CLAIM_ORDER=manifest
native_worker="$(hostname)-byc-native-$native_arm-$native_index"
native_snapshot="$native_root/$native_arm/snapshot-$native_index"
mkdir -p "$native_snapshot"
"$native_root/../../zenfleet-ctl" compact --run "byc-$native_arm-r2-20261025" \
  --endpoint "$EP" --out-dir "$native_snapshot" --upload
native_ledger=()
native_snapfile="$native_snapshot/snap_byc-$native_arm-r2-20261025.parquet"
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
"$native_root/../../zenfleet-worker" \
  --manifest "$native_root/$native_arm/manifest.json" "${native_ledger[@]}" \
  --ledger-out "s3://$ZEN_BUCKET/$ZEN_RUN/ledger/pass-$native_worker.parquet" \
  --blobs-r2-bucket "$ZEN_BUCKET" --blobs-r2-prefix "$ZEN_RUN/blobs" \
  --claims-r2-bucket "$ZEN_BUCKET" --claims-prefix "$ZEN_RUN/claims" \
  --claim-ttl-secs 600 --control-r2-key "$ZEN_RUN/control.json" --capability cpu_heavy \
  --r2-endpoint "$EP" --exec /opt/fleet-fits/program/fit_cell_exec.py \
  --worker "$native_worker" --provider local
