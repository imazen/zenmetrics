#!/usr/bin/env bash
set -euo pipefail
binary="$1"; threads="$2"; output="$3"; shift 3
mkdir -p "$(dirname "$output")" "$HOME/tmp"
export TMPDIR="$HOME/tmp" RAYON_NUM_THREADS="$threads"
heavy="${RANKED_RUN_HEAVY:-$HOME/work/claudehints/scripts/run-heavy}"
perf_command=(perf)
if [ "${RANKED_PERF_SUDO:-0}" = 1 ]; then
  perf_command=(sudo -n --preserve-env=TMPDIR,RAYON_NUM_THREADS,IWSSIM_WALL_STRIP_HEIGHT perf)
fi
{
  printf 'binary_sha256: '; sha256sum "$binary"
  printf 'host: '; hostname
  printf 'rayon_threads: %s\nargv:' "$threads"
  printf ' %q' "$binary" "$@"; printf '\n'
  if [ "$threads" = 1 ]; then
    "$heavy" --mem 16G -- "${perf_command[@]}" stat -e instructions,cycles -- taskset -c 2 /usr/bin/time -v "$binary" "$@"
  else
    "$heavy" --mem 16G -- /usr/bin/time -v "$binary" "$@"
  fi
} 2>&1 | tee "$output"
