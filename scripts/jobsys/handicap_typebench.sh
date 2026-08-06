#!/usr/bin/env bash
# handicap_typebench.sh — measure ONE box's per-encoder-type throughput for fleet/handicaps.toml.
#
# "Registered" handicap rows are MEASURED, and measured PER TYPE: per-encode concurrency differs
# (process-parallel single-threaded encoders scale with cores; internally-threaded encoders run
# their own pool and compress the many-core advantage; memory-bound encoders re-rank boxes), so
# a box gets one multiplier per encoder type — never extrapolated across types.
#
# The bench IS the production path: each type's registered cell set runs through the ordinary
# zenfleet-worker pass (chunked executor, BoxBudget admission caps, real exec processes, real
# blob/ledger I/O). Run it INSIDE the box's worker container against the box's baked executor
# for production-realistic numbers; a bare-metal run measures the bare-metal deployment.
#
# REGISTERED CELL SETS — deterministic slices of committed campaign manifests, so every box
# times identical work (same sources, sizes, q ladder per type):
#   zenavif       : first N cells of the avifgen encode manifest
#                   (s3://<bucket>/jobs/avifgen-enc-20260806/manifest.json[.gz])
#   zenav1-svt /
#   jpeg-gainmap /
#   zenjxl (hdr)  : first N cells of that type filtered from the HDR corpus encode manifest
#                   (the hdr-corpus wave's jobs/<run>/manifest.json; filter: .kind.codec)
# Slicing is `jq '[.[] | select(.kind.codec == $type)][0:N]'` — order is the manifest's, which
# is committed/content-addressed, so the slice is reproducible forever.
#
# UPDATE PROCEDURE (also in fleet/handicaps.toml's header):
#   1. On the box (inside its worker container if that is how it deploys):
#        ./handicap_typebench.sh --exec '<executor-cmd>' \
#            --type zenavif=/path/avifgen_manifest.json \
#            --type zenav1-svt=/path/hdr_manifest.json --cells 96
#   2. Normalize each type's cells/min against the current 1.0 reference box for that type.
#   3. Edit fleet/handicaps.toml (cite this run: hostname, date, cells, manifest), commit.
#   4. For a LIVE campaign also write the RunControl worker_weights override (next-pass
#      convergence; the committed file reaches boxes at their next image roll).
#
# Output: cells/min per type + a ready-to-paste TOML fragment (raw rates in comments; you do
# the normalization consciously against the column's reference box).
set -euo pipefail

CELLS=96
EXEC=""
declare -a TYPES=()
WORKER_BIN="${ZB_WORKER_BIN:-zenfleet-worker}"
OUT_DIR="${ZB_OUT_DIR:-$HOME/tmp/handicap-typebench-$(date -u +%Y%m%dT%H%M%SZ)}"

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --exec) EXEC="$2"; shift 2 ;;
    --type) TYPES+=("$2"); shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --worker-bin) WORKER_BIN="$2"; shift 2 ;;
    *) usage ;;
  esac
done
[ -n "$EXEC" ] && [ "${#TYPES[@]}" -gt 0 ] || usage
command -v jq >/dev/null || { echo "handicap_typebench: jq required" >&2; exit 3; }
command -v "$WORKER_BIN" >/dev/null || { echo "handicap_typebench: $WORKER_BIN not on PATH (set ZB_WORKER_BIN)" >&2; exit 3; }

mkdir -p "$OUT_DIR"
HOST="$(hostname)"
echo "handicap_typebench: box=$HOST cells/type=$CELLS out=$OUT_DIR"
TOML_ROWS=""

for spec in "${TYPES[@]}"; do
  ty="${spec%%=*}"; mf="${spec#*=}"
  [ -s "$mf" ] || { echo "handicap_typebench: manifest '$mf' for type '$ty' missing/empty" >&2; exit 4; }
  slice="$OUT_DIR/$ty.manifest.json"
  # The registered slice: first $CELLS cells of this type, manifest order (reproducible).
  jq --arg ty "$ty" --argjson n "$CELLS" \
     '[.[] | select((.kind.codec // "") == $ty)][0:$n]' "$mf" > "$slice"
  n_actual=$(jq 'length' "$slice")
  [ "$n_actual" -gt 0 ] || { echo "handicap_typebench: manifest has no '$ty' cells" >&2; exit 4; }
  run="$OUT_DIR/$ty"
  mkdir -p "$run/blobs"
  echo "── $ty: $n_actual cells through the ordinary worker pass…"
  t0=$(date +%s)
  # The ORDINARY pass: chunked executor + admission caps + fresh exec processes + real I/O.
  "$WORKER_BIN" --manifest "$slice" --ledger-out "$run/bench.parquet" \
      --blobs "$run/blobs" --exec "$EXEC" \
      --worker "typebench-$HOST" --provider typebench \
      2>&1 | tee "$run/pass.log" | tail -1
  t1=$(date +%s)
  done_n=$(grep -oE 'done=[0-9]+' "$run/pass.log" | tail -1 | cut -d= -f2)
  wall=$(( t1 - t0 )); [ "$wall" -gt 0 ] || wall=1
  if [ "${done_n:-0}" -ne "$n_actual" ]; then
    echo "handicap_typebench: $ty did $done_n/$n_actual cells — FIX FAILURES before registering a rate" >&2
    exit 5
  fi
  rate=$(awk -v d="$done_n" -v w="$wall" 'BEGIN { printf "%.2f", d * 60.0 / w }')
  echo "   $ty: $done_n cells in ${wall}s = $rate cells/min"
  TOML_ROWS="$TOML_ROWS
$ty = CHANGEME  # $rate cells/min raw — typebench $HOST $(date -u +%F) n=$n_actual; normalize vs the column's 1.0 box"
done

cat <<EOF

Paste into fleet/handicaps.toml under [workers.$HOST.encode] after normalizing
(divide each raw rate by the column's 1.0-reference box's raw rate for that type):
[workers.$(echo "$HOST" | tr -c 'A-Za-z0-9._\n-' '_').encode]$TOML_ROWS
default = 1.00  # UNMEASURED types fall back here — never extrapolate across types
EOF
echo "handicap_typebench: raw logs + slices kept in $OUT_DIR"
