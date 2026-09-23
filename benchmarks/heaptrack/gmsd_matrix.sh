#!/usr/bin/env bash
# GMSD peak-memory matrix (heaptrack), 2026-09-22 gmsd lane.
#   gmsd_matrix.sh <cpu-profile-binary> <out-dir> <tsv>
# One heaptrack process per (metric, mode, size, threads) cell, the driver
# calling the metric exactly once. Reports heaptrack's "peak heap memory
# consumption" and "peak RSS (including heaptrack overhead)", plus the input
# buffers the driver itself allocates (2 × w·h·3 bytes, sRGB8 ref + dist) so
# the metric's own share can be read as peak_heap − inputs.
set -uo pipefail
BIN=${1:?cpu-profile binary}; OUT=${2:?out dir}; TSV=${3:?tsv}
mkdir -p "$OUT"
echo -e "metric\tmode\tw\th\tthreads\tinputs_bytes\tpeak_heap\tpeak_rss\tscore_line" > "$TSV"
cell() { # metric mode w h threads
  local m=$1 mode=$2 w=$3 h=$4 t=$5 f="$OUT/${1}_${2}_${3}x${4}_t${5}"
  rm -f "$f.zst"
  RAYON_NUM_THREADS=$t heaptrack --output "$f" "$BIN" "$m" "$mode" "$w" "$h" > "$f.log" 2>&1
  local p; p=$(heaptrack_print "$f.zst" 2>/dev/null | grep -E "^peak heap memory consumption|^peak RSS" | sed 's/.*: //' | tr '\n' '\t')
  echo -e "$m\t$mode\t$w\t$h\t$t\t$((2*w*h*3))\t${p}$(grep -m1 '^OK\|^GAP\|rror' "$f.log" | cut -c1-120)" | tee -a "$TSV"
}
for sz in "64 64" "256 256" "1024 1024" "4096 4096" "7000 5728"; do
  set -- $sz
  for t in 1 8; do
    cell gmsd full "$1" "$2" $t
    cell gmsd map "$1" "$2" $t
  done
done
for sz in "1024 1024" "4096 4096" "7000 5728"; do
  set -- $sz
  for m in ssim2 butter zensim iwssim; do cell "$m" full "$1" "$2" 8; done
done
echo "gmsd_matrix done"
