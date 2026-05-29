#!/usr/bin/env bash
# Path A recovery measurement harness (task #127 re-verify).
# Measures cvvdp full / strip / warm_ref / warm_ref_strip:
#   - PROCESS peak heap via heaptrack (separate run, with overhead)
#   - score-only wall (t_score_ms) via N timed release runs, median (no heaptrack)
# Sizes: 16 MP (4096x4096) and 30 MP (6000x5000).
# NO target-cpu=native. NO extrapolation — every number is a real run.
set -euo pipefail

BIN="${BIN:-target/release/cpu-profile}"
OUT_TSV="${OUT_TSV:-crates/cvvdp/benchmarks/cpu_path_a_recovered_2026-05-29.tsv}"
N_WALL="${N_WALL:-7}"
TMPDIR_HT="$(mktemp -d /tmp/patha_ht.XXXXXX)"

declare -a SIZES=("4096 4096 16" "6000 5000 30")
declare -a MODES=("full" "strip" "warm_ref" "warm_ref_strip")

median() {
  # median of stdin numbers (one per line)
  sort -n | awk '{a[NR]=$1} END{ if(NR%2==1){print a[(NR+1)/2]} else {print (a[NR/2]+a[NR/2+1])/2} }'
}

echo -e "size_mp\twidth\theight\tmode\tpeak_heap_bytes\tpeak_heap_GB\twall_score_ms_median\twall_score_ms_min\twall_score_ms_all\tjod_score\tn_wall" > "$OUT_TSV"

for sz in "${SIZES[@]}"; do
  read -r W H MP <<< "$sz"
  for mode in "${MODES[@]}"; do
    echo "=== ${MP}MP ${W}x${H} ${mode} ===" >&2

    # --- HEAP: heaptrack process peak (separate run, has overhead) ---
    HT_OUT="$TMPDIR_HT/cvvdp_${mode}_${MP}mp"
    rm -f "${HT_OUT}".* 2>/dev/null || true
    heaptrack -o "$HT_OUT" "$BIN" cvvdp "$mode" "$W" "$H" >/dev/null 2>&1
    TRACE=$(ls "${HT_OUT}".zst 2>/dev/null || ls "${HT_OUT}".*.zst 2>/dev/null | head -1)
    PEAK_LINE=$(heaptrack_print "$TRACE" 2>/dev/null | grep "peak heap memory consumption" | head -1)
    # format: "peak heap memory consumption: 1.58G"
    PEAK_HUMAN=$(echo "$PEAK_LINE" | sed 's/.*consumption: //')
    PEAK_BYTES=$(heaptrack_print "$TRACE" 2>/dev/null | grep -A2 "peak heap memory consumption" | grep -oE "peak RSS|peak heap" >/dev/null 2>&1; echo "")
    # extract bytes precisely from the human value
    PEAK_BYTES=$(python3 -c "
v='$PEAK_HUMAN'.strip()
mult={'B':1,'K':1024,'M':1024**2,'G':1024**3,'T':1024**4}
if v and v[-1] in mult:
    print(int(float(v[:-1])*mult[v[-1]]))
else:
    print(int(float(v)) if v else 0)
")
    PEAK_GB=$(python3 -c "print(f'{$PEAK_BYTES/1024**3:.3f}')")
    echo "   heap peak: $PEAK_HUMAN ($PEAK_BYTES bytes, $PEAK_GB GB) trace=$TRACE" >&2

    # --- WALL: N timed runs, take median of t_score_ms (no heaptrack) ---
    WALLS=""
    JOD=""
    for i in $(seq 1 "$N_WALL"); do
      LINE=$("$BIN" cvvdp "$mode" "$W" "$H" 2>&1)
      MS=$(echo "$LINE" | grep -oE "t_score_ms=[0-9.]+" | cut -d= -f2)
      JOD=$(echo "$LINE" | grep -oE "score=[0-9.]+" | cut -d= -f2)
      WALLS="${WALLS}${MS}"$'\n'
    done
    WALL_MED=$(printf "%s" "$WALLS" | grep -v '^$' | median)
    WALL_MIN=$(printf "%s" "$WALLS" | grep -v '^$' | sort -n | head -1)
    WALL_ALL=$(printf "%s" "$WALLS" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
    echo "   wall median: $WALL_MED ms (min $WALL_MIN, all=[$WALL_ALL]) jod=$JOD" >&2

    echo -e "${MP}\t${W}\t${H}\t${mode}\t${PEAK_BYTES}\t${PEAK_GB}\t${WALL_MED}\t${WALL_MIN}\t${WALL_ALL}\t${JOD}\t${N_WALL}" >> "$OUT_TSV"
  done
done

echo "=== DONE -> $OUT_TSV ===" >&2
cat "$OUT_TSV" >&2
rm -rf "$TMPDIR_HT"
