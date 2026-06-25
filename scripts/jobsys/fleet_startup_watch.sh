#!/usr/bin/env bash
#
# fleet_startup_watch.sh — know WITHIN ~2 MINUTES if a launched box isn't
# getting started successfully working.
#
# This is the launch-time complement to the steady-state idle detector
# (zenfleet-core::idle / fleet_util_snapshot.sh, which deliberately spare a box
# during its 120 s warmup). This watchdog watches that warmup window: it polls a
# freshly-launched fleet and ALERTS the moment a box has been up past the startup
# deadline without signalling that it actually started working — i.e. it caught
# the image-pull hang, the onstart crash, or the 6-80 s fast-crash cascade
# (scripts/sweep/CLAUDE.md "EXP-LARGER-LARGE") while you can still react.
#
# "Started working" signal, in priority order (any one ⇒ the box is alive):
#   1. a boot record in R2            s3://<bucket>/<run>/boot/<worker>.txt
#   2. a chunk claim in R2            s3://coefficient/claims/<run>/<chunk>.claim  (stronger)
#   3. (vast boxes) non-zero GPU/CPU utilization
# A box that is provider-"running" past the deadline with NONE of these = failed
# to start. A box still "loading" past the deadline = stuck pulling the image.
#
# Usage:
#   fleet_startup_watch.sh --label <vast-label> [opts]      # vast fleet (per-box detail)
#   fleet_startup_watch.sh --run <RUN> --expected <N> [opts] # any fleet (R2 boot-record count)
#
# Options:
#   --label <p>      vast.ai label prefix of the fleet
#   --run <id>       run id (for R2 boot/claim signal + bucket prefix)
#   --bucket <b>     R2 bucket for boot records (default: zen-tuning-ephemeral)
#   --expected <N>   number of boxes you launched (for the R2 aggregate check)
#   --deadline <s>   a box should have started by now (default 90)
#   --poll <s>       poll interval (default 20 → detect within deadline+poll ≈ 110 s)
#   --max-wait <s>   give up watching after this (default 600)
#   --destroy        auto-destroy a vast box that failed to start (else alert only)
#
# Exit: 0 all started (or graceful stop) · 2 one or more failed to start · 3 bad args
set -uo pipefail

LABEL=""; RUN=""; BUCKET="${ZEN_FLEET_BUCKET:-zen-tuning-ephemeral}"
EXPECTED=""; DEADLINE=90; POLL=20; MAX_WAIT=600; DESTROY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --label) LABEL="$2"; shift 2;;
        --run) RUN="$2"; shift 2;;
        --bucket) BUCKET="$2"; shift 2;;
        --expected) EXPECTED="$2"; shift 2;;
        --deadline) DEADLINE="$2"; shift 2;;
        --poll) POLL="$2"; shift 2;;
        --max-wait) MAX_WAIT="$2"; shift 2;;
        --destroy) DESTROY=1; shift;;
        -h|--help) sed -n '2,40p' "$0" >&2; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 3;;
    esac
done
if [[ -z "$LABEL" && -z "$RUN" ]]; then
    echo "need --label (vast) and/or --run (R2). See --help." >&2; exit 3
fi

# R2 startup-signal count for the run (boot records + claims), provider-agnostic.
started_signals_r2() {
    [[ -z "$RUN" ]] && { echo 0; return; }
    local ep="https://${R2_ACCOUNT_ID:-}.r2.cloudflarestorage.com" n=0 b c
    [[ -z "${R2_ACCOUNT_ID:-}" ]] && { echo 0; return; }
    b=$(s5cmd --endpoint-url "$ep" ls "s3://${BUCKET}/${RUN}/boot/" 2>/dev/null | grep -c '\.txt' || true)
    c=$(s5cmd --endpoint-url "$ep" ls "s3://coefficient/claims/${RUN}/" 2>/dev/null | grep -c '\.claim' || true)
    n=$(( ${b:-0} > ${c:-0} ? ${b:-0} : ${c:-0} ))
    echo "$n"
}

echo "startup-watch: deadline=${DEADLINE}s poll=${POLL}s  (a box flagged within ~$((DEADLINE+POLL))s of launch)"
[[ -n "$LABEL" ]] && echo "  vast label: $LABEL"
[[ -n "$RUN" ]] && echo "  run: $RUN  (R2 boot/claim signal, expected=${EXPECTED:-?})"

START_TS=$(date +%s)
ANY_FAILED=0
while :; do
    NOW=$(date +%s); ELAPSED=$((NOW - START_TS))
    R2_STARTED=$(started_signals_r2)
    OUT=""
    if [[ -n "$LABEL" ]]; then
        OUT=$(vastai show instances --raw 2>/dev/null | \
            LABEL="$LABEL" DEADLINE="$DEADLINE" R2_STARTED="$R2_STARTED" EXPECTED="${EXPECTED:-0}" \
            python3 -c '
import json, sys, os
d = json.loads(sys.stdin.read() or "[]")
insts = d if isinstance(d, list) else d.get("instances", [])
label = os.environ["LABEL"]; deadline = int(os.environ["DEADLINE"])
r2_started = int(os.environ.get("R2_STARTED", "0") or 0)
m = [i for i in insts if i and label in (i.get("label") or "")]
failed, working, starting = [], 0, 0
for i in m:
    age = int(i.get("duration") or 0)
    status = (i.get("actual_status") or i.get("cur_state") or "").lower()
    gu = i.get("gpu_util", 0) or 0; cu = i.get("cpu_util", 0) or 0
    dph = float(i.get("dph_total", 0) or 0)
    iid = i.get("id")
    if gu > 0 or cu > 0:
        working += 1; continue
    if age < deadline:
        starting += 1; continue
    if status in ("running", "active"):
        failed.append((iid, f"running {age}s, 0% util - booted but not working", dph))
    else:
        st = status or "?"
        failed.append((iid, f"stuck in {st} {age}s - image-pull/boot hang", dph))
n = len(m)
print(f"BOXES {n} working={working} starting={starting} failed={len(failed)} r2_boot={r2_started}")
waste = sum(d for _, _, d in failed)
for iid, why, dph in failed:
    print(f"FAIL {iid} ${dph:.2f}/hr {why}")
print(f"WASTE {waste:.2f}")
' 2>/dev/null) || OUT="ERR vast query failed"
    fi

    echo "[$(date -u +%H:%M:%SZ) +${ELAPSED}s] ${OUT%%$'\n'*}"
    # R2 aggregate check for heterogeneous fleets (hetzner/basement aren't in vast).
    if [[ -n "$EXPECTED" && "$R2_STARTED" -lt "$EXPECTED" && "$ELAPSED" -gt "$DEADLINE" ]]; then
        echo "  ⚠ only ${R2_STARTED}/${EXPECTED} boxes have a boot record/claim after ${ELAPSED}s — $((EXPECTED-R2_STARTED)) not started."
        ANY_FAILED=1
    fi

    # Per-box vast failures.
    FAIL_IDS=$(printf '%s\n' "$OUT" | awk '/^FAIL /{print $2}')
    if [[ -n "$FAIL_IDS" ]]; then
        ANY_FAILED=1
        echo "  ⚠ FAILED TO START:"
        printf '%s\n' "$OUT" | awk '/^FAIL /{$1="";print "    "$0}'
        WASTE=$(printf '%s\n' "$OUT" | awk '/^WASTE /{print $2}')
        echo "  ⚠ burning \$${WASTE:-?}/hr on boxes that never started."
        if [[ "$DESTROY" == "1" ]]; then
            echo "  → --destroy: tearing them down."
            echo "$FAIL_IDS" | xargs -n1 -I{} sh -c 'echo y | vastai destroy instance {} >/dev/null 2>&1 && echo "    destroyed {}"'
        else
            IDS1=$(printf '%s ' $FAIL_IDS)
            echo "  → destroy:  for i in ${IDS1}; do echo y | vastai destroy instance \$i; done"
        fi
    fi

    # Done?  Everything that's up is working/starting and none failed, or we hit max-wait.
    if printf '%s\n' "$OUT" | grep -q '^BOXES ' && ! printf '%s\n' "$OUT" | grep -q '^FAIL '; then
        STILL=$(printf '%s\n' "$OUT" | awk '/^BOXES /{print $4}' | sed 's/starting=//')
        if [[ "${STILL:-0}" == "0" ]]; then
            echo "✓ all launched boxes started working."
            break
        fi
    fi
    if (( ELAPSED >= MAX_WAIT )); then
        echo "max-wait ${MAX_WAIT}s reached; stopping watch."
        break
    fi
    sleep "$POLL"
done

[[ "$ANY_FAILED" == "1" ]] && exit 2 || exit 0
