#!/usr/bin/env bash
# Emit a ready-to-paste `docker run` for the BASEMENT tier (an Unraid box behind NAT, outbound-only).
# Run THIS on the workstation (it has the root R2 key); it mints a SCOPED, time-boxed R2 credential
# (never the root key — the Unraid box only ever sees the scoped one) and prints the docker command +
# Unraid "Add Container" field values to run a worker against a given run.
#
# The worker is PULL-BASED: it only makes outbound HTTPS to R2 (claim + blob + ledger). No inbound
# ports, no port-forward, no reverse tunnel — which is exactly why a NAT'd basement box works as a
# first-class tier. It claims its share off the ONE shared R2 lease-queue alongside every other tier.
#
# Usage:  bash scripts/jobsys/unraid_worker.sh <RUN> [TTL_DAYS=7] [CAPABILITY]
#   <RUN>        the run prefix under the bucket (the same RUN the launcher/declare used)
#   TTL_DAYS     scoped-cred lifetime, 1..7 (CF temp-cred max is 7d). Re-run this to refresh.
#   CAPABILITY   optional: cpu_light,cpu_heavy,gpu,cpu_arm,high_ram (comma-sep) to restrict what it pulls
#
# For a TRULY persistent basement worker (not re-minting weekly), create a long-lived R2 API token in
# the Cloudflare dashboard (R2 -> Manage API Tokens -> Object Read & Write, scoped to the bucket) and
# hand-edit AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY in the printed command (drop AWS_SESSION_TOKEN).
set -euo pipefail
RUN="${1:?usage: unraid_worker.sh <RUN> [TTL_DAYS] [CAPABILITY]}"
TTL_DAYS="${2:-7}"; CAP="${3:-}"
IMAGE="${ZEN_WORKER_IMAGE:-ghcr.io/imazen/zen-jobworker:latest}"   # multi-arch (amd64+arm64)
BUCKET="${ZEN_FLEET_BUCKET:-zen-tuning-ephemeral}"
EXEC="${ZEN_EXEC:-/bin/cat}"   # /bin/cat = synthetic. For real work bake an executor + set this to it.
TTL=$(( TTL_DAYS * 86400 )); [ "$TTL" -ge 900 ] && [ "$TTL" -le 604800 ] || { echo "TTL_DAYS must be 1..7"; exit 1; }

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Mint a scoped, object-read-write cred limited to this run's prefix (never the root key on the box).
body=$(RUN="$RUN" BUCKET="$BUCKET" TTL="$TTL" python3 -c "import json,os;print(json.dumps({'bucket':os.environ['BUCKET'],'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':int(os.environ['TTL']),'prefixes':[os.environ['RUN']+'/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/unraid_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/unraid_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "FAILED to mint scoped creds:"; cat /tmp/unraid_cred.json; exit 1; }

MANIFEST="s3://$BUCKET/$RUN/manifest-unraid.json"
# The basement worker gets its own shuffled manifest (decorrelated claim order — see launch_fleet.sh).
if [ -f /tmp/fleet_manifest.json ]; then
  python3 -c 'import json,random;j=json.load(open("/tmp/fleet_manifest.json"));random.seed("unraid");random.shuffle(j);json.dump(j,open("/tmp/fleet_manifest_unraid.json","w"))'
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
    s5cmd --endpoint-url "$EP" cp /tmp/fleet_manifest_unraid.json "$MANIFEST" >/dev/null && echo "uploaded $MANIFEST"
else
  echo "NOTE: /tmp/fleet_manifest.json not found — set ZEN_MANIFEST_URI below to the run's manifest."
  MANIFEST="s3://$BUCKET/$RUN/manifest.json"
fi

cat <<EOF

================  RUN THIS ON THE UNRAID BOX (basement tier)  ================
Scoped creds expire in ${TTL_DAYS}d. Outbound HTTPS only — no inbound ports.

docker run -d --name zen-worker-$RUN --restart no \\
  -e AWS_ACCESS_KEY_ID=$AK \\
  -e AWS_SECRET_ACCESS_KEY=$SK \\
  -e AWS_SESSION_TOKEN=$ST \\
  -e AWS_REGION=auto \\
  -e ZEN_R2_ENDPOINT=$EP \\
  -e ZEN_BUCKET=$BUCKET \\
  -e ZEN_RUN=$RUN \\
  -e ZEN_MANIFEST_URI=$MANIFEST \\
  -e ZEN_PROVIDER=basement \\
  -e ZEN_WORKER=unraid \\
  -e ZEN_EXEC=$EXEC \\
  -e ZEN_CONTROL_KEY=$RUN/control.json \\
  -e ZEN_SPEC_THRESHOLD_SECS=20 \\
  -e ZEN_IDLE_PASSES=8 \\${CAP:+
  -e ZEN_CAPABILITY=$CAP \\}
  $IMAGE

Unraid GUI ("Docker" tab -> "Add Container"):
  Repository: $IMAGE
  Network:    bridge (no published ports needed)
  Restart:    No  (the worker drains its share of this run, then exits cleanly)
  Variables:  add each -e KEY=VALUE above as a Container Variable
==============================================================================
EOF
echo "verify it's working from the workstation:  bash scripts/jobsys/watch_fleet.sh $RUN   (look for provider=basement rows)"
