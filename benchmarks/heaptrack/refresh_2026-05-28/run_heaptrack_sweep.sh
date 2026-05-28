#!/usr/bin/env bash
# Task #139 — heaptrack PROCESS-PEAK memory sweep for the 6 CPU metrics.
#
# One heaptrack run per (metric, mode, size). Serial. Captures the
# "peak heap memory consumption" line from heaptrack_print (PROCESS
# peak — NOT top-callstack, per task #130 which found top-callstack
# misreports). Raw .zst traces persisted alongside.
#
# heaptrack reports peak in base-1024 K/M/G suffixes. We record both the
# human string and the converted byte value (1024-based), plus peak RSS
# for reference.
#
# Usage: run_heaptrack_sweep.sh <out_dir> [metric_filter] [size_filter]
set -uo pipefail

OUT="${1:?out dir}"
FILTER="${2:-}"
SIZEFILTER="${3:-}"
BIN="/home/lilith/work/zen/zenmetrics--cpu-bench-refresh/target/release/cpu-profile"
TSV="$OUT/heaptrack_peaks.tsv"
mkdir -p "$OUT"

# sizes: label w h
SIZES=(
  "512 512 512"
  "1024 1024 1024"
  "2K 2048 2048"
  "12MP 4000 3000"
  "30MP 6000 5000"
)
METRICS=(cvvdp ssim2 dssim butter iwssim zensim)
MODES=(full warm_ref strip warm_ref_strip)

# Convert heaptrack human (e.g. "1.23G", "456.7M", "789K", "12B") to bytes (1024 base).
to_bytes() {
  local h="$1"
  local num unit
  num=$(echo "$h" | grep -oE '^[0-9.]+')
  unit=$(echo "$h" | grep -oE '[KMGTB]$')
  case "$unit" in
    K) awk -v n="$num" 'BEGIN{printf "%d", n*1024}' ;;
    M) awk -v n="$num" 'BEGIN{printf "%d", n*1024*1024}' ;;
    G) awk -v n="$num" 'BEGIN{printf "%d", n*1024*1024*1024}' ;;
    T) awk -v n="$num" 'BEGIN{printf "%d", n*1024*1024*1024*1024}' ;;
    B|"") awk -v n="$num" 'BEGIN{printf "%d", n}' ;;
    *) echo "0" ;;
  esac
}

if [[ ! -f "$TSV" ]]; then
  printf 'metric\tmode\tsize_label\tw\th\tpeak_heap_bytes\tpeak_heap_human\tpeak_rss_human\tscore\trc\ttrace\n' > "$TSV"
fi

for m in "${METRICS[@]}"; do
  [[ -n "$FILTER" && "$m" != "$FILTER" ]] && continue
  for mode in "${MODES[@]}"; do
    for sz in "${SIZES[@]}"; do
      read -r label w h <<< "$sz"
      [[ -n "$SIZEFILTER" && "$label" != "$SIZEFILTER" ]] && continue
      trace="$OUT/ht_${m}_${mode}_${label}.zst"
      if grep -qP "^${m}\t${mode}\t${label}\t" "$TSV" 2>/dev/null; then
        echo "SKIP (done): $m $mode $label"
        continue
      fi
      echo "=== heaptrack $m $mode ${w}x${h} ($label) ==="
      rm -f "${trace%.zst}".*
      HT_OUT=$(heaptrack -o "${trace%.zst}" "$BIN" "$m" "$mode" "$w" "$h" 2>&1)
      rc=$?
      real_trace=$(ls -t "${trace%.zst}".* 2>/dev/null | head -1)
      score=$(echo "$HT_OUT" | grep -oE "score=[0-9.eE+-]+" | head -1 | sed 's/score=//')
      gap=$(echo "$HT_OUT" | grep -oE "GAP:[a-z_:]+" | head -1)
      if [[ -n "$gap" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\tNOT_SUPPORTED\tNOT_SUPPORTED\tNOT_SUPPORTED\t-\t2\t-\n' \
          "$m" "$mode" "$label" "$w" "$h" >> "$TSV"
        echo "  GAP: $gap"
        rm -f "$real_trace"
        continue
      fi
      PRINT=$(heaptrack_print "$real_trace" 2>/dev/null)
      peak_human=$(echo "$PRINT" | grep "peak heap memory consumption" | head -1 | sed -E 's/.*consumption: *//')
      rss_human=$(echo "$PRINT" | grep "peak RSS" | head -1 | sed -E 's/.*: *//')
      peak_bytes=$(to_bytes "$peak_human")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\n' \
        "$m" "$mode" "$label" "$w" "$h" \
        "$peak_bytes" "$peak_human" "$rss_human" "${score:--}" "$rc" "$(basename "$real_trace")" >> "$TSV"
      echo "  rc=$rc score=${score:--} peak=$peak_human (${peak_bytes}B) rss=$rss_human"
    done
  done
done
echo "DONE. TSV at $TSV"
