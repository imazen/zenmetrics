#!/usr/bin/env bash
#
# cvvdp_backfill/status.sh — at-a-glance fleet status for an in-flight
# cvvdp-backfill run.
#
# Aggregates the same R2 paths a human would manually inspect with
# s5cmd:
#
#   - s3://coefficient/heartbeats/<run>/        per-worker boot/run/done markers
#   - s3://coefficient/logs/<run>/              per-chunk failure logs (only on fail)
#   - s3://zentrain/<run>/cvvdp_imazen/         per-chunk sidecars (imazen impl)
#   - s3://zentrain/<run>/cvvdp_pycvvdp_v054/   per-chunk sidecars (pycvvdp impl)
#
# Plus the chunks.jsonl total for a completion percentage.
#
# USAGE
#
#   SWEEP_RUN_ID=cvvdp-backfill-2026-05-15-half \
#       bash scripts/sweep/cvvdp_backfill/status.sh
#
#   # Periodic poll (Ctrl-C to stop):
#   SWEEP_RUN_ID=... watch -n 60 'bash scripts/sweep/cvvdp_backfill/status.sh'
#
# REQUIRES
#   - s5cmd on PATH
#   - ~/.config/cloudflare/r2-credentials sourced (R2_ACCOUNT_ID etc)
#
# Does NOT call vastai — that's slow + rate-limited. Heartbeat
# timestamps stand in for "still alive" without the rate-limit cost.

set -euo pipefail
source ~/.config/cloudflare/r2-credentials

SWEEP_RUN_ID="${SWEEP_RUN_ID:-cvvdp-backfill-$(date -u +%Y-%m-%d)}"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
S5="s5cmd --endpoint-url ${ENDPOINT} --profile r2"

# Fail-fast on missing chunks.jsonf manifest — if this isn't on R2 the
# fleet has nothing to claim.
CHUNKS_KEY="s3://coefficient/jobs/${SWEEP_RUN_ID}/chunks.jsonl"
if ! $S5 ls "${CHUNKS_KEY}" >/dev/null 2>&1; then
    echo "ERROR: ${CHUNKS_KEY} missing on R2 — run not bootstrapped." >&2
    exit 1
fi

echo "================================================================"
echo "cvvdp-backfill status: ${SWEEP_RUN_ID}"
echo "  at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "================================================================"

# 1. Manifest size — derives the denominator for completion %.
N_CHUNKS_TOTAL=$($S5 cat "${CHUNKS_KEY}" 2>/dev/null | wc -l)
echo
echo "Manifest:"
echo "  chunks.jsonl total: ${N_CHUNKS_TOTAL}"

# 2. Heartbeats — three markers per worker:
#    .boot  = onstart bootstrap reached docker-pull
#    .run   = first chunk claimed and started
#    .done  = chunks.jsonl exhausted, worker shutting down
#
# Counts can lag reality by the heartbeat refresh interval (~60s).
echo
echo "Heartbeats:"
HB_LS=$($S5 ls "s3://coefficient/heartbeats/${SWEEP_RUN_ID}/" 2>/dev/null || true)
if [[ -z "$HB_LS" ]]; then
    echo "  none yet (workers still booting; pycvvdp 6.5 GB image pull dominates)"
else
    N_BOOT=$(echo "$HB_LS" | grep -c '\.boot$' || true)
    N_RUN=$(echo "$HB_LS" | grep -c '\.run$' || true)
    N_DONE=$(echo "$HB_LS" | grep -c '\.done$' || true)
    echo "  boot  : ${N_BOOT}"
    echo "  run   : ${N_RUN}"
    echo "  done  : ${N_DONE}"

    # The newest heartbeat across all workers — staleness signal.
    NEWEST=$(echo "$HB_LS" | awk '{print $1, $2}' | sort | tail -1)
    [[ -n "$NEWEST" ]] && echo "  newest: ${NEWEST}"
fi

# 3. Sidecar counts per impl — the real progress signal.
# s5cmd ls on an empty prefix exits non-zero, so wrap with || true.
echo
echo "Sidecars:"
for IMPL in cvvdp_imazen cvvdp_pycvvdp_v054; do
    N=$(($S5 ls "s3://zentrain/${SWEEP_RUN_ID}/${IMPL}/" 2>/dev/null || true) | wc -l)
    if [[ "$N_CHUNKS_TOTAL" -gt 0 ]]; then
        PCT=$(awk -v n="$N" -v t="$N_CHUNKS_TOTAL" 'BEGIN { printf "%.1f", 100*n/t }')
    else
        PCT="?"
    fi
    printf "  %-22s %5d / %d  (%s%%)\n" "${IMPL}:" "$N" "$N_CHUNKS_TOTAL" "$PCT"
done

# 4. Failures — non-empty .fail.log entries are real failures (the
# worker dumps stderr only when a step failed, so any object here
# means at least one chunk crashed).
echo
echo "Failures:"
FAIL_LS=$($S5 ls "s3://coefficient/logs/${SWEEP_RUN_ID}/" 2>/dev/null || true)
if [[ -z "$FAIL_LS" ]]; then
    echo "  none"
else
    N_FAIL=$(echo "$FAIL_LS" | grep -c '\.fail\.log$' || true)
    echo "  fail logs: ${N_FAIL}"
    # Print the first 3 fail-log names so the user can grab them.
    echo "$FAIL_LS" | awk '/\.fail\.log$/ {print "    " $NF}' | head -3
    [[ "$N_FAIL" -gt 3 ]] && echo "    ... ($N_FAIL total)"
fi

echo
echo "================================================================"
echo "  ${S5} ls 's3://coefficient/heartbeats/${SWEEP_RUN_ID}/'"
echo "  vastai logs <instance_id>"
echo "================================================================"
