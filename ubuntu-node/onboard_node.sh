#!/usr/bin/env bash
# onboard_node.sh <tailnet-ip-or-host> — start the zensim-720 backfill worker on a booted zen-node.
#
# Run from the DEV box (holds the CF token). Mints a fresh 12h scoped R2 cred and starts the worker
# container on the node over SSH. A dedicated node, so it uses ALL cores (the worker's resource-aware
# admission bounds concurrency); --restart keeps it alive. Re-run within 12h to refresh the cred, or add
# a cron line:  13 4,15 * * *  bash .../onboard_node.sh <node>   (mirrors the tower refresh cadence).
set -euo pipefail
NODE="${1:?usage: onboard_node.sh <tailnet-ip-or-host>}"
set -a; . "$HOME/.config/cloudflare/r2-credentials"; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

body=$(python3 -c "import json,os;print(json.dumps({'bucket':'zentrain','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':43200,'prefixes':['jobs/','jxl-lossy/runs/','canonical/2026-06-27/','refs/']}))")
J=$(curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
     "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials")
read -r AK SK ST < <(printf '%s' "$J" | python3 -c 'import json,sys;r=json.load(sys.stdin)["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "cred mint failed: $J"; exit 1; }

SSHN="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o BatchMode=yes zen@$NODE"
$SSHN "sudo docker rm -f zen720 2>/dev/null; sudo docker run -d --name zen720 --restart unless-stopped \
  -e AWS_ACCESS_KEY_ID='$AK' -e AWS_SECRET_ACCESS_KEY='$SK' -e AWS_SESSION_TOKEN='$ST' -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT='$EP' -e ZEN_BUCKET=zentrain \
  -e ZEN_POOL_RUNLIST=s3://zentrain/jobs/_pool/runlist.tsv \
  -e ZEN_CORPUS_PREFIX=refs/clean-picker-corpus-2026-06-26 \
  -e ZEN_MAX_MIN=700 -e ZEN_CORE_OVERSUBSCRIBE=1 -e ZEN_PERSISTENT_EXEC=1 \
  -e RAYON_NUM_THREADS=1 -e OMP_NUM_THREADS=1 -e ZEN_CHUNK_WALL_SEC=20 -e ZEN_PASS_TIMEOUT=5400 \
  -e ZEN_PROVIDER=basement -e ZEN_WORKER='node-$NODE' \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh ghcr.io/imazen/zenfleet-worker:exec" >/dev/null
echo "worker started on $NODE (12h cred). Verify: ssh zen@$NODE 'sudo docker top zen720 | grep -c zenmetrics'"
