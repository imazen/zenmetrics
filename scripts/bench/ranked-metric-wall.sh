#!/usr/bin/env bash
set -euo pipefail
binary="$1"; threads="$2"; output="$3"; shift 3
mkdir -p "$(dirname "$output")" "$HOME/tmp"
export TMPDIR="$HOME/tmp" RAYON_NUM_THREADS="$threads"
heavy="$HOME/work/claudehints/scripts/run-heavy"
{
  printf 'binary_sha256: '; sha256sum "$binary"
  printf 'host: '; hostname
  printf 'rayon_threads: %s\nargv:' "$threads"
  printf ' %q' "$binary" "$@"; printf '\n'
  if [ "$threads" = 1 ]; then
    "$heavy" --mem 16G -- perf stat -e instructions,cycles -- taskset -c 2 "$binary" "$@"
  else
    "$heavy" --mem 16G -- "$binary" "$@"
  fi
} 2>&1 | tee "$output"
