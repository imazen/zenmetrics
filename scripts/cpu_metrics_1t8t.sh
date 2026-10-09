#!/usr/bin/env bash
# Every CPU metric the CLI scores from sRGB8, timed at 1 thread and 8 threads.
#
# Drives benchmarks/heaptrack/drivers/cpu_profile's `new-metrics-wall` binary
# (the real `zenmetrics_cli::metrics::run_metric_display` dispatch) on the
# zenmetrics-corpus photo pair (source.png vs q20.jpg, mirror-tiled):
#
#   1024        zenbench lat, 1t (taskset 1 core, RAYON=1) and 8t (taskset 0-7, RAYON=8)
#   other sizes zenbench lat at 8t; 1t as one serial cold+warm call per metric,
#               each metric in its own process under /usr/bin/time -v (max RSS),
#               because single-threaded mdctpsnr alone is ~1 min/call at 8.4 MP.
#
# Usage: scripts/cpu_metrics_1t8t.sh <out_dir> [sizes...]   (default: 1024 3355x2516)
# Progress: <out_dir>/progress.log. Analysis: scripts/cpu_metrics_1t8t_analyze.py <out_dir>
# Build first: cargo build --release -p cpu-profile --bin new-metrics-wall
set -uo pipefail

OUT="${1:?usage: cpu_metrics_1t8t.sh <out_dir> [sizes...]}"
shift
SIZES="${*:-1024 3355x2516}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/new-metrics-wall"
SRC="$ROOT/benchmarks/heaptrack/drivers/cpu_profile/src/bin/new_metrics_wall.rs"
MARKER="$ROOT/.workongoing"
RH="${RUN_HEAVY:-$HOME/work/claudehints/scripts/run-heavy}"
ONE_CORE="${ONE_CORE:-2}"
EIGHT_CORES="${EIGHT_CORES:-0-7}"

[ -x "$BIN" ] || { echo "missing $BIN (cargo build --release -p cpu-profile --bin new-metrics-wall)" >&2; exit 1; }
mkdir -p "$OUT/logs"
LOG="$OUT/progress.log"
note() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
claim() { printf '%s cpu_metrics_1t8t %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$MARKER"; }

{
  echo "commit=$(git -C "$ROOT" rev-parse HEAD) dirty=$(git -C "$ROOT" status --porcelain | wc -l)"
  echo "host=$(hostname -s) cpu=$(lscpu | sed -n 's/^Model name: *//p') nproc=$(nproc)"
  echo "bin_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
  echo "content=photo (zenmetrics-corpus source.png vs q20.jpg, mirror-tiled)"
  echo "sizes=$SIZES one_core=$ONE_CORE eight_cores=$EIGHT_CORES"
  echo "command=$0 $OUT $SIZES"
} > "$OUT/run.meta"

# name size wall_s min_rounds cpus threads
leg() {
  local name=$1 size=$2 wall=$3 minr=$4 cpus=$5 threads=$6
  claim "leg $name"
  note "leg $name begin (size=$size wall=${wall}s cpus=$cpus threads=$threads)"
  TMPDIR="$HOME/tmp" NMW_CONTENT=photo NMW_MODES=lat NMW_GROUP_WALL_S=$wall NMW_CELL_MAX_S=$wall NMW_MIN_ROUNDS=$minr \
    "$RH" --mem 16G -- env RAYON_NUM_THREADS=$threads OMP_NUM_THREADS=$threads \
    taskset -c "$cpus" "$BIN" "$size" "$OUT/$name.tsv" > "$OUT/logs/$name.log" 2>&1
  note "leg $name rc=$? $(grep 'run-heavy: done' "$OUT/logs/$name.log")"
}

for s in $SIZES; do
  if [ "$s" = "1024" ]; then
    leg "st_photo_$s" "$s" 150 3 "$ONE_CORE" 1
    leg "mt8_photo_$s" "$s" 150 3 "$EIGHT_CORES" 8
  else
    leg "mt8_photo_$s" "$s" 300 2 "$EIGHT_CORES" 8
    for m in $(grep -o '("[a-z0-9-]*", MetricKind' "$SRC" | cut -d'"' -f2); do
      claim "1t heap $s $m"
      TMPDIR="$HOME/tmp" NMW_CONTENT=photo "$RH" --mem 16G -- env RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
        /usr/bin/time -v taskset -c "$ONE_CORE" "$BIN" heap "$s" "$m" > "$OUT/logs/heap_st_${s}_$m.log" 2>&1
      note "1t $s $m rc=$? $(grep -E 'rep[01]' "$OUT/logs/heap_st_${s}_$m.log" | tr '\n' ' ')"
    done
  fi
done
note "DONE"
