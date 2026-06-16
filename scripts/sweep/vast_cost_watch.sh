#!/usr/bin/env bash
#
# vast_cost_watch.sh — continuous burn-rate + balance monitor for
# vast.ai fleets. Alerts (and optionally auto-destroys everything) if
# the active fleet's hourly burn exceeds a budget OR the account
# credit drops below a threshold.
#
# Usage:
#   scripts/sweep/vast_cost_watch.sh [--max-burn-dph 1.0] [--min-credit 5.0]
#                                    [--label-prefix '<run-id>']
#                                    [--auto-destroy] [--poll 60]
#
# Defaults:
#   --max-burn-dph 1.0    Alert when active fleet exceeds $1/hour
#   --min-credit 5.0      Alert when account credit drops below $5
#   --label-prefix ''     Match every instance on the account
#   --poll 60             Poll every 60 s
#   (no --auto-destroy)   Alert only; manual kill required
#
# Exit:
#   0 — graceful Ctrl+C
#   2 — budget alert tripped without --auto-destroy
#   3 — credit alert tripped
#   4 — auto-destroy ran

set -uo pipefail

MAX_BURN_DPH="1.0"
MIN_CREDIT="5.0"
LABEL_PREFIX=""
POLL="60"
AUTO_DESTROY="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-burn-dph) MAX_BURN_DPH="$2"; shift 2;;
        --min-credit) MIN_CREDIT="$2"; shift 2;;
        --label-prefix) LABEL_PREFIX="$2"; shift 2;;
        --poll) POLL="$2"; shift 2;;
        --auto-destroy) AUTO_DESTROY=1; shift;;
        -h|--help) sed -n '2,24p' "$0" >&2; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { printf '[%s] [cost-watch] %s\n' "$(ts)" "$*"; }

log "starting (max_burn=\$${MAX_BURN_DPH}/hr, min_credit=\$${MIN_CREDIT}, poll=${POLL}s, label_prefix='${LABEL_PREFIX}', auto_destroy=${AUTO_DESTROY})"

# Per-iteration: count active instances of interest, sum dph, fetch credit.
TICK=0
while true; do
    TICK=$((TICK + 1))
    INSTANCES_JSON=$(vastai show instances-v1 --raw 2>/dev/null)
    if [[ -z "$INSTANCES_JSON" || "$INSTANCES_JSON" == "[]" ]]; then
        INSTANCES_JSON="[]"
    fi
    USER_JSON=$(vastai show user --raw 2>/dev/null)

    SUMMARY=$(echo "$INSTANCES_JSON" | python3 -c "
import json, sys
data = sys.stdin.read()
try:
    d = json.loads(data)
except Exception:
    print('0|0.0|||')
    sys.exit(0)
if isinstance(d, dict) and 'instances' in d: d = d['instances']
if not isinstance(d, list): d = [d]
prefix = '${LABEL_PREFIX}'
matched = [i for i in d if not prefix or (i.get('label') or '').find(prefix) != -1]
n = len(matched)
dph = sum(float(i.get('dph_total', 0) or 0) for i in matched)
running = sum(1 for i in matched if i.get('actual_status') == 'running')
ids = ','.join(str(i.get('id')) for i in matched[:8])
print(f'{n}|{dph:.4f}|{running}|{ids}')
")
    N_TOTAL="${SUMMARY%%|*}"; REST="${SUMMARY#*|}"
    BURN_DPH="${REST%%|*}"; REST="${REST#*|}"
    N_RUNNING="${REST%%|*}"; IDS="${REST#*|}"

    CREDIT=$(echo "$USER_JSON" | python3 -c "
import json, sys
try:
    print(f\"{json.load(sys.stdin).get('credit', 0):.2f}\")
except Exception:
    print('?')
")

    log "instances=${N_TOTAL} (running=${N_RUNNING}) burn=\$${BURN_DPH}/hr credit=\$${CREDIT} ids=${IDS:0:60}"

    if [[ "${BURN_DPH%.*}" -ge "${MAX_BURN_DPH%.*}" ]] \
       || awk -v a="$BURN_DPH" -v b="$MAX_BURN_DPH" 'BEGIN{exit !(a > b)}'; then
        log "ALERT: burn \$${BURN_DPH}/hr exceeds budget \$${MAX_BURN_DPH}/hr"
        if [[ "$AUTO_DESTROY" == "1" ]]; then
            log "AUTO-DESTROY: triggering destroy on ${IDS}"
            if [[ -n "$LABEL_PREFIX" ]]; then
                zenfleet-vastai destroy --label-prefix "$LABEL_PREFIX" 2>&1 | head -5
            else
                echo "$IDS" | tr ',' '\n' | while read -r id; do
                    [[ -n "$id" ]] && (yes y | vastai destroy instance "$id" 2>&1 | head -1)
                done
            fi
            exit 4
        fi
        # Without --auto-destroy, just keep alerting every iteration but
        # don't exit — operator may want to investigate before killing.
    fi

    if awk -v c="$CREDIT" -v m="$MIN_CREDIT" 'BEGIN{exit !(c+0 < m+0)}'; then
        log "ALERT: credit \$${CREDIT} below threshold \$${MIN_CREDIT}"
        if [[ "$AUTO_DESTROY" == "1" ]]; then
            log "AUTO-DESTROY: low credit safeguard"
            if [[ -n "$LABEL_PREFIX" ]]; then
                zenfleet-vastai destroy --label-prefix "$LABEL_PREFIX" 2>&1 | head -5
            fi
            exit 3
        fi
    fi

    sleep "$POLL"
done
