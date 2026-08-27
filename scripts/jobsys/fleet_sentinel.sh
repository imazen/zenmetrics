#!/bin/bash
# fleet_sentinel.sh — ONE watcher, EVERY condition exits so the supervising
# Claude session is invoked (a background task's exit fires a task
# notification; a log line nobody reads does not). Anti-wedge companion:
# invariants make workers refuse to wedge; this makes the OPERATOR loop
# refuse to sleep through outages. Heartbeats go to stdout every tick.
#
#   fleet_sentinel.sh [--runlist <runlist.tsv>] [--gate-gb 80] [--interval 300]
#                     [--budget-min 720]
#
# Exit codes (each a distinct condition for the supervisor):
#   10 SPACE-GATE-MET       tower cache avail >= gate → run the relaunch runbook
#   11 STORE-WRITABLE-AGAIN write probe recovered while cache below gate
#   12 MOVER-DONE-LOW-SPACE mover finished but the gate is still unmet
#   13 TOWER-UNREACHABLE    ssh failed 3 consecutive ticks
#   14 DRAIN-STALL          --runlist gap unchanged 4 ticks while store writable
#   15 BUDGET-EXHAUSTED     watch window ended with no condition
#   16 PLEX-HARD-DOWN       media server stopped answering (3 ticks)
#   17 STORE-WRITE-DIED     write probe newly failing (3 ticks) after being ok
set -u
RUNLIST=""; GATE=80; IVL=300; BUDGET=720
while [ $# -gt 0 ]; do case "$1" in
  --runlist) RUNLIST=$2; shift 2;;
  --gate-gb) GATE=$2; shift 2;;
  --interval) IVL=$2; shift 2;;
  --budget-min) BUDGET=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;;
esac; done
END=$((SECONDS + BUDGET*60))
ssh_fail=0; plex_fail=0; write_fail=0; write_was_ok=0; prev_write=""; write_streak_fails=0
last_gap=""; gap_same=0
STORE_HOST=${ZEN_STORE_HOST:-192.168.50.170}; STORE_PORT=${ZEN_STORE_PORT:-3900}
probe_write() {
  # s3-independent minimal probe: can the store accept a write? Uses s5cmd if
  # creds are sourced, else falls back to a TCP connect check (reach-only).
  if command -v s5cmd >/dev/null && [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo probe | timeout 20 s5cmd --endpoint-url "${ZEN_S3_ENDPOINT:-http://$STORE_HOST:$STORE_PORT}" \
      pipe "s3://${ZEN_SENTINEL_BUCKET:-zentrain}/_probe/sentinel-$(date +%s)" >/dev/null 2>&1
  else
    timeout 8 bash -c "exec 3<>/dev/tcp/$STORE_HOST/$STORE_PORT" 2>/dev/null
  fi
}
while [ $SECONDS -lt $END ]; do
  TS=$(date -u +%H:%M:%SZ)
  AVAIL=$(ssh -o ConnectTimeout=10 root@tower "df --output=avail -BG /mnt/cache 2>/dev/null | tail -1 | tr -dc 0-9" 2>/dev/null)
  if [ -z "$AVAIL" ]; then
    ssh_fail=$((ssh_fail+1))
    echo "$TS tower-ssh FAIL ($ssh_fail/3)"
    [ $ssh_fail -ge 3 ] && { echo "CONDITION: TOWER-UNREACHABLE"; exit 13; }
    sleep "$IVL"; continue
  fi
  ssh_fail=0
  MOVER=$(ssh -o ConnectTimeout=10 root@tower "pgrep -cx move 2>/dev/null" 2>/dev/null || echo "?")
  PLEX=$(ssh -o ConnectTimeout=10 root@tower "curl -sm 6 -o /dev/null -w %{http_code} http://localhost:32400/identity" 2>/dev/null || echo 000)
  if [ "$PLEX" != "200" ]; then plex_fail=$((plex_fail+1)); else plex_fail=0; fi
  [ $plex_fail -ge 3 ] && { echo "CONDITION: PLEX-HARD-DOWN (http=$PLEX x3)"; exit 16; }
  if probe_write; then W=ok; write_was_ok=1; write_fail=0; else W=fail; write_fail=$((write_fail+1)); fi
  GAPTXT=""
  if [ -n "$RUNLIST" ] && [ "$W" = ok ]; then
    JOBCTL="${ZEN_JOBCTL:-$(cd "$(dirname "$0")/../.." && pwd)/target/release/zenfleet-ctl}"
    G=$(timeout 170 "$JOBCTL" report --runlist "$RUNLIST" 2>/dev/null | grep -a "^TOTAL" | grep -oE "gap=[0-9]+" | head -1)
    GAPTXT=" $G"
    if [ -n "$G" ]; then
      if [ "$G" = "$last_gap" ]; then gap_same=$((gap_same+1)); else gap_same=0; fi
      last_gap=$G
      [ "$G" = "gap=0" ] && { echo "CONDITION: DRAIN-COMPLETE"; exit 0; }
      [ $gap_same -ge 4 ] && { echo "CONDITION: DRAIN-STALL ($G unchanged x5 ticks, store writable)"; exit 14; }
    fi
  fi
  echo "$TS avail=${AVAIL}G mover=$MOVER plex=$PLEX write=$W$GAPTXT"
  [ "$AVAIL" -ge "$GATE" ] && { echo "CONDITION: SPACE-GATE-MET (${AVAIL}G >= ${GATE}G) — run the relaunch runbook"; exit 10; }
  # TRANSITION-based signals (2026-08-27 fixes, both bitten live):
  #  - -eq 3 fired only on EXACTLY the third fail; a streak that passed that
  #    tick unnoticed never fired again -> -ge.
  #  - the writable-again signal was guarded out under --runlist, so a parked
  #    worker had no restart wake -> fire on the fail->ok transition always.
  if [ "$W" = ok ] && [ "$prev_write" = fail ]; then
    echo "CONDITION: STORE-WRITABLE-AGAIN (write recovered after $write_streak_fails failing tick(s)) — restart parked workers"
    exit 11
  fi
  if [ "$W" = fail ]; then write_streak_fails=$((write_streak_fails+1)); else write_streak_fails=0; fi
  prev_write=$W
  [ $write_fail -ge 3 ] && [ "$write_was_ok" = 1 ] && { echo "CONDITION: STORE-WRITE-DIED (probe failing x$write_fail after being ok)"; exit 17; }
  [ "$MOVER" = "0" ] && [ "$AVAIL" -lt "$GATE" ] && { echo "CONDITION: MOVER-DONE-LOW-SPACE (${AVAIL}G < ${GATE}G)"; exit 12; }
  sleep "$IVL"
done
echo "CONDITION: BUDGET-EXHAUSTED (${BUDGET}min, last avail=${AVAIL:-?}G)"; exit 15
