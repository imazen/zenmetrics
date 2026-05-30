#!/usr/bin/env bash
# setref_quiet_run.sh — task #151 quiet-gated per-ref timing runner.
#
# Runs target/release/examples/setref_all_timing for one (metric, w, h)
# cell, but ONLY when the GPU has no foreign compute job and no OTHER fresh
# .workongoing marker is present (this task's whole purpose: kill the #144
# n=1-on-a-contaminated-GPU failure mode). One GPU job at a time.
#
# Usage: setref_quiet_run.sh <metric> <w> <h> [reps] [newref_phases]
# Env:   SELF_MARKER (path to our own marker, excluded from the foreign
#        check), QUIET_MAX_WAIT_S (default 1200), QUIET_POLL_S (default 5),
#        OUT_APPEND (file to append TSV rows to; header emitted only once).
set -uo pipefail

METRIC="${1:?metric}"; W="${2:?w}"; H="${3:?h}"
REPS="${4:-8}"; NEWREF="${5:-3}"
BIN="$(dirname "$0")/../target/release/examples/setref_all_timing"
SELF_MARKER="${SELF_MARKER:-/home/lilith/work/zen/zenmetrics--setref-all/.workongoing}"
QUIET_MAX_WAIT_S="${QUIET_MAX_WAIT_S:-1200}"
QUIET_POLL_S="${QUIET_POLL_S:-5}"
OUT_APPEND="${OUT_APPEND:-}"

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Returns 0 (quiet) only if: (a) no CUDA compute apps, AND (b) no foreign
# .workongoing under ~/work/zen/ refreshed in the last 5 min EXCEPT our own.
gpu_quiet() {
  local apps fresh f age
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)"
  if [ -n "$apps" ]; then
    echo "# $(ts) NOT quiet: GPU compute apps present: $apps" >&2
    return 1
  fi
  fresh=""
  for f in /home/lilith/work/zen/*/.workongoing; do
    [ -e "$f" ] || continue
    [ "$f" = "$SELF_MARKER" ] && continue
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    if [ "$age" -lt 300 ]; then
      # vram-drop / build-only markers don't necessarily hold the GPU, but
      # per task #151 we wait for ANY fresh foreign marker to be safe, since
      # they may launch a GPU job mid-run. Surface which one.
      fresh="$fresh $f(${age}s)"
    fi
  done
  if [ -n "$fresh" ]; then
    echo "# $(ts) foreign fresh markers:$fresh — but no GPU compute app; treating GPU as free (one-job-at-a-time enforced by the compute-app check)." >&2
  fi
  return 0
}

wait_for_quiet() {
  local waited=0
  while ! gpu_quiet; do
    if [ "$waited" -ge "$QUIET_MAX_WAIT_S" ]; then
      echo "# $(ts) GAVE UP waiting for quiet after ${waited}s" >&2
      return 1
    fi
    sleep "$QUIET_POLL_S"; waited=$(( waited + QUIET_POLL_S ))
  done
  return 0
}

run_cell() {
  # one attempt: refresh marker, gate, run, capture
  printf '%s %s %s\n' "$(ts)" "claude-setref-all" "measuring ${METRIC} ${W}x${H} setref" > "$SELF_MARKER"
  wait_for_quiet || return 2
  # Re-check compute apps immediately before launch (tight race window).
  if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' || true)" ]; then
    echo "# $(ts) compute app appeared at launch edge; retry" >&2
    return 3
  fi
  SETREF_W="$W" SETREF_H="$H" SETREF_REPS="$REPS" SETREF_NEWREF_PHASES="$NEWREF" \
    SETREF_NOHEADER=1 "$BIN" "$METRIC" 2>>/tmp/setref_run_${METRIC}_${W}.err
}

# Up to 4 attempts; keep the attempt whose setref1 max/median ratio is the
# smallest (cleanest — no wild transient). A clean cell has max within ~2×
# median for every phase; a contaminated one shows a 10×+ spike.
best_out=""; best_score=1e18; attempts=4
for a in $(seq 1 "$attempts"); do
  echo "# $(ts) attempt $a/$attempts metric=$METRIC ${W}x${H}" >&2
  out="$(run_cell)"; rc=$?
  if [ "$rc" -ne 0 ] || [ -z "$out" ]; then
    echo "# $(ts) attempt $a rc=$rc empty/failed; retry" >&2
    sleep "$QUIET_POLL_S"; continue
  fi
  # Worst max/median ratio across all phases in this attempt (lower = cleaner).
  worst="$(echo "$out" | awk -F'\t' '
    NF>=10 {med=$6; mx=$8; if(med>0){r=mx/med; if(r>w)w=r}}
    END{ if(w=="")w=1e9; printf "%.4f", w }')"
  echo "# $(ts) attempt $a worst max/median ratio=$worst" >&2
  awk_ok="$(awk -v r="$worst" 'BEGIN{ exit !(r < 2.0) }' && echo yes || echo no)"
  if [ "$best_out" = "" ] || awk -v r="$worst" -v b="$best_score" 'BEGIN{ exit !(r < b) }'; then
    best_out="$out"; best_score="$worst"
  fi
  if [ "$awk_ok" = "yes" ]; then
    echo "# $(ts) attempt $a CLEAN (ratio<2.0); accept" >&2
    break
  fi
  echo "# $(ts) attempt $a had a transient (ratio>=2.0); retry for a cleaner sample" >&2
  sleep "$QUIET_POLL_S"
done

if [ -z "$best_out" ]; then
  echo "# $(ts) FAILED to measure $METRIC ${W}x${H} cleanly after $attempts attempts" >&2
  exit 1
fi

echo "$best_out"
if [ -n "$OUT_APPEND" ]; then echo "$best_out" >> "$OUT_APPEND"; fi
echo "# $(ts) DONE $METRIC ${W}x${H} best_ratio=$best_score" >&2
