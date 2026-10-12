#!/usr/bin/env bash
set -euo pipefail
binary="$1"; threads="$2"; prefix="$3"; metrics="$4"; size="${5:-1024}"
mkdir -p "$(dirname "$prefix")" "$HOME/tmp"
export TMPDIR="$HOME/tmp" RAYON_NUM_THREADS="$threads"
export NMW_CONTENT=synth NMW_MODES="${NMW_MODES:-lat}"
export NMW_MIN_ROUNDS="${NMW_MIN_ROUNDS:-30}"
export NMW_GROUP_WALL_S="${NMW_GROUP_WALL_S:-120}"
export NMW_CELL_MAX_S="${NMW_CELL_MAX_S:-30}"
heavy="$HOME/work/claudehints/scripts/run-heavy"
{
  printf 'binary_sha256: '; sha256sum "$binary"
  printf 'host: '; hostname
  printf 'rayon_threads: %s\nsize: %s\nmetrics: %s\n' "$threads" "$size" "$metrics"
  printf 'modes: %s\nmin_rounds: %s\ngroup_wall_s: %s\ncell_max_s: %s\n' \
    "$NMW_MODES" "$NMW_MIN_ROUNDS" "$NMW_GROUP_WALL_S" "$NMW_CELL_MAX_S"
  if [ "$threads" = 1 ]; then
    "$heavy" --mem 16G -- perf stat -e instructions,cycles -- taskset -c 2 \
      "$binary" "$size" "$prefix.tsv" "$metrics"
  else
    "$heavy" --mem 16G -- "$binary" "$size" "$prefix.tsv" "$metrics"
  fi
} 2>&1 | tee "$prefix.log"
