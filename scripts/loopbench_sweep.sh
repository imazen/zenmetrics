#!/usr/bin/env bash
# Task #163 — encoder-search-loop size sweep (linear f32 planar).
# Usage: scripts/loopbench_sweep.sh <bin> <out_tsv> <progress_log> <sizes...>
set -uo pipefail
BIN="${1:?bin}"; OUT="${2:?out}"; LOG="${3:?log}"; shift 3
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
note() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }
note "START bin=$BIN out=$OUT sizes=$* load=$(uptime | sed 's/.*averages: //')"
for s in "$@"; do
  note "cell size=$s begin"
  printf '%s claude-metricloop-agent %s sweep cell size=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(basename "$BIN")" "$s" > "$ROOT/.workongoing"
  CPU_WALL_NO_GATE=1 LOOP_WALL_NO_GATE=1 nice -n 19 "$BIN" "$s" "$OUT" >>"$LOG" 2>&1
  note "cell size=$s rc=$? rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
done
note "DONE rows=$(wc -l < "$OUT")"
