#!/usr/bin/env bash
# Hourly self-renewal for the POOL backfill (runs from cron at the top of each hour).
# Enforces the operator's rule: at most ONE <=$2 batch of EU boxes started per hour, each self-destructs
# at 55 min. Stops itself when the backfill is complete or if a previous batch is still alive.
set -uo pipefail
cd /home/lilith/work/zen/zenmetrics || exit 1
export HCLOUD_TOKEN="$(grep -E '^api_token=' ~/.config/hetzner/credentials 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \r')"
LOG=~/tmp/hz720/pool_cron.log; mkdir -p ~/tmp/hz720
log(){ echo "[$(date -u +%H:%M:%SZ)] $*" >> "$LOG"; }

# 1. Complete? -> stop renewing (leave a marker so future ticks are instant no-ops).
if [ -f ~/tmp/hz720/pool_done.marker ]; then log "done marker present — no launch"; exit 0; fi
DONE=$(timeout 180 python3 scripts/jobsys/pool_done_check.py 2>/dev/null || echo "NOTDONE ?")
log "done-check: $DONE"
case "$DONE" in
  DONE*) log "backfill COMPLETE — writing marker, not launching"; touch ~/tmp/hz720/pool_done.marker; exit 0 ;;
esac

# 2. Don't stack: a prior batch should have self-destructed (55min < 60min tick). If many remain, skip.
NUP=$(hcloud server list -o columns=name 2>/dev/null | grep -c hzpool || echo 0)
if [ "${NUP:-0}" -gt 20 ]; then log "prior batch still up ($NUP hzpool) — skip this hour"; exit 0; fi

# 3. Launch exactly one <=$2 EU pool batch.
log "launching pool batch (prior up=$NUP, $DONE)"
bash scripts/jobsys/pool_launch.sh >> "$LOG" 2>&1
log "launched: $(hcloud server list -o columns=name 2>/dev/null | grep -c hzpool) hzpool up"
