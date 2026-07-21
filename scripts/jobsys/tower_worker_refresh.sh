#!/usr/bin/env bash
# tower_worker_refresh.sh — keep the basement tower's zensim-720 backfill worker alive.
#
# R2 temp creds max out at 12h, and this worker runs for days, so a cron on the DEV box (which holds the
# CF token — never put it on the tower) re-mints a 12h scoped cred and recreates the container. Twice a
# day, ~11h apart, stays comfortably inside the 12h TTL. The ledger preserves progress, so the brief
# recreate only re-does the one in-flight pass. Niced: 24 of 32 cores (8 reserved for the NAS),
# cpu-shares 256 (yields under load), 40G mem cap, --restart unless-stopped.
set -euo pipefail
set -a; . "$HOME/.config/cloudflare/r2-credentials"; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
LOG="$HOME/tmp/hz720/tower_refresh.log"; mkdir -p "$HOME/tmp/hz720"

body=$(python3 -c "import json,os;print(json.dumps({'bucket':'zentrain','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':604800,'prefixes':['jobs/','jxl-lossy/runs/','canonical/2026-06-27/','refs/']}))")
J=$(curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
     "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials")
read -r AK SK ST < <(printf '%s' "$J" | python3 -c 'import json,sys;r=json.load(sys.stdin)["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "$(date -u +%FT%TZ) mint FAILED: $J" >>"$LOG"; exit 1; }

SSHT="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes root@tower"
$SSHT "docker rm -f zen720-basement 2>/dev/null; docker run -d --name zen720-basement --restart unless-stopped \
  --cpuset-cpus=0-23 --cpu-shares=256 --memory=40g \
  -e AWS_ACCESS_KEY_ID='$AK' -e AWS_SECRET_ACCESS_KEY='$SK' -e AWS_SESSION_TOKEN='$ST' -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT='$EP' -e ZEN_BUCKET=zentrain \
  -e ZEN_POOL_RUNLIST=s3://zentrain/jobs/_pool/runlist.tsv \
  -e ZEN_CORPUS_PREFIX=refs/clean-picker-corpus-2026-06-26 \
  -e ZEN_MAX_MIN=700 -e ZEN_CORE_OVERSUBSCRIBE=1 -e ZEN_PERSISTENT_EXEC=1 \
  -e RAYON_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e ZEN_CHUNK_WALL_SEC=20 -e ZEN_PASS_TIMEOUT=5400 \
  -e ZEN_PROVIDER=basement -e ZEN_WORKER=tower-unraid \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh ghcr.io/imazen/zenfleet-worker:exec" >/dev/null
echo "$(date -u +%FT%TZ) tower worker refreshed (7-day cred)" >>"$LOG"
