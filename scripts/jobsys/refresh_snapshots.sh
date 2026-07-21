#!/usr/bin/env bash
# refresh_snapshots.sh — (re)compact EVERY pool run's ledger into a done-set snapshot and upload it
# to jobs/<run>/ledger_snapshot.parquet, which the POOL worker feeds to zenfleet-worker as --ledger-in
# (see fleet-entrypoint.sh) so its gap = only-undone instead of re-scoring done cells (the ~2x tax).
#
# Run once to seed, then on a cron (~30min) to keep the done-set current as workers make progress:
#   */30 * * * * bash /home/lilith/work/zen/zenmetrics/scripts/jobsys/refresh_snapshots.sh >> ~/tmp/zen-snaps/refresh.log 2>&1
#
# Read-only vs the live ledger/ dir; writes only the ledger_snapshot.parquet key. Niced, parallel-capped.
# Runs on the box holding the R2 creds (dev box / launcher) — never bakes creds onto a remote worker.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; . "$HOME/.config/cloudflare/r2-credentials"; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
BUCKET="${ZEN_BUCKET:-zentrain}"
SNAP_DIR="$HOME/tmp/zen-snaps"; mkdir -p "$SNAP_DIR"
LOG="$SNAP_DIR/refresh.log"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

r2 cp "s3://$BUCKET/jobs/_pool/runlist.tsv" /tmp/runlist_refresh.tsv >/dev/null 2>&1 \
  || { echo "$(date -u +%FT%TZ) cannot fetch runlist" | tee -a "$LOG"; exit 1; }
runs=$(cut -f1 /tmp/runlist_refresh.tsv | grep -E '^bf-' | sort -u)
echo "$(date -u +%FT%TZ) refresh START: $(printf '%s\n' "$runs" | grep -c .) runs" | tee -a "$LOG"

compact_one(){
  local run="$1" snap="$HOME/tmp/zen-snaps/snap_${1}.parquet"
  if nice -n 19 python3 "$HERE/compact_ledgers.py" "$run" >/dev/null 2>&1 && [ -s "$snap" ]; then
    if AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
       s5cmd --endpoint-url "$EP" cp "$snap" "s3://$BUCKET/jobs/${run}/ledger_snapshot.parquet" >/dev/null 2>&1; then
      echo "OK $run"
    else echo "UPLOAD-FAIL $run"; fi
  else echo "COMPACT-FAIL $run"; fi
}
export -f compact_one; export EP R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY HOME HERE BUCKET
printf '%s\n' "$runs" | xargs -P 6 -I{} bash -c 'compact_one "$@"' _ {} | tee -a "$LOG"
echo "$(date -u +%FT%TZ) refresh DONE" | tee -a "$LOG"
