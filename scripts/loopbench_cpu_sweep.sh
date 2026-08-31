#!/usr/bin/env bash
# Task #163 — CPU warm-reference size sweep for the encoder-search-loop question.
#
# Drives benchmarks/heaptrack/drivers/cpu_profile's `cpu-wall` binary across the
# full tiny→large size ladder so `total = alpha + beta*pixels` can be FIT rather
# than assumed. The prior sweep (cpu_wall_all_metrics_2026-05-29.tsv) started at
# 512^2, which is above the regime where the intercept dominates — exactly the
# regime an encoder quality-search loop lives in for small renditions.
#
# Usage: scripts/loopbench_cpu_sweep.sh <out_tsv> <progress_log>
#
# Progress streams to <progress_log> continuously (CLAUDE.md: every long tool
# must emit heartbeat lines you can tail, not go silent then report "done").
set -uo pipefail

OUT="${1:?usage: loopbench_cpu_sweep.sh <out_tsv> <progress_log>}"
LOG="${2:?usage: loopbench_cpu_sweep.sh <out_tsv> <progress_log>}"
BIN="$(cd "$(dirname "$0")/.." && pwd)/target/release/cpu-wall"
MARKER="$(cd "$(dirname "$0")/.." && pwd)/.workongoing"

[ -x "$BIN" ] || { echo "missing $BIN — build with: cargo build --release -p cpu-profile --bin cpu-wall" >&2; exit 1; }

note() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG"; }

# Small sizes: all six metrics in ONE interleaved group (cheap, and the
# cross-metric interleave is what makes zenbench's paired stats meaningful).
# Large sizes: per-metric groups for the three the encoder loop actually uses,
# so peak RAM stays bounded and a 16 MP cvvdp cell can't eat the wall budget.
SMALL_SIZES="64 128 256 512"
LARGE_SIZES="1024 2K 4096"
LARGE_METRICS="zensim ssim2 butter"

note "START out=$OUT bin=$BIN host=$(hostname -s) load=$(uptime | sed 's/.*averages: //')"

for s in $SMALL_SIZES; do
  note "cell size=$s metrics=ALL6 begin"
  # Refresh the repo claim before each cell — cells are minutes long and the
  # 5-minute staleness window would otherwise lapse mid-sweep.
  printf '%s claude-metricloop-agent cpu-wall sweep cell size=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$s" > "$MARKER"
  CPU_WALL_NO_GATE=1 nice -n 19 "$BIN" "$s" "$OUT" >>"$LOG" 2>&1
  note "cell size=$s metrics=ALL6 rc=$? rows=$(wc -l < "$OUT")"
done

for s in $LARGE_SIZES; do
  for m in $LARGE_METRICS; do
    note "cell size=$s metric=$m begin"
    printf '%s claude-metricloop-agent cpu-wall sweep cell size=%s metric=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$s" "$m" > "$MARKER"
    CPU_WALL_NO_GATE=1 nice -n 19 "$BIN" "$s" "$OUT" "$m" >>"$LOG" 2>&1
    note "cell size=$s metric=$m rc=$? rows=$(wc -l < "$OUT")"
  done
done

note "DONE rows=$(wc -l < "$OUT")"
