#!/usr/bin/env bash
# Emit a ready-to-paste `docker run` for the BASEMENT tier (an Unraid box behind NAT, outbound-only).
# Run THIS on the workstation (it has the root R2 key); it mints SCOPED, time-boxed R2 credentials
# (never the root key — the Unraid box only ever sees the scoped ones) and prints the docker command +
# Unraid "Add Container" field values to run a worker against a given run.
#
# TWO BUCKETS, TWO CREDS (this is the corrected design — codec-corpus is READ-ONLY):
#   - RUN bucket   (ZEN_FLEET_BUCKET, default `coefficient`): the worker WRITES blobs / ledger / claims /
#     manifest here, under $RUN/. Scoped object-READ-WRITE cred limited to that prefix.
#   - CORPUS bucket (ZEN_CORPUS_BUCKET, default `codec-corpus`): the executor READS source images here.
#     A SEPARATE object-READ-ONLY cred (R2 temp creds are single-bucket, so two buckets need two creds).
#     jobexec uses it via ZEN_CORPUS_AWS_*; the run cred never touches the read-only corpus.
#
# The worker is PULL-BASED: it only makes outbound HTTPS to R2 (claim + blob + ledger). No inbound
# ports, no port-forward, no reverse tunnel — which is exactly why a NAT'd basement box works as a
# first-class tier. It claims its share off the ONE shared R2 lease-queue alongside every other tier.
#
# Usage:  bash scripts/jobsys/unraid_worker.sh <RUN> [TTL_DAYS=7] [CAPABILITY]
#   <RUN>        the run prefix under the RUN bucket (the same RUN the launcher/declare used)
#   TTL_DAYS     scoped-cred lifetime, 1..7 (CF temp-cred max is 7d). Re-run this to refresh.
#   CAPABILITY   optional: cpu_light,cpu_heavy,gpu,cpu_arm,high_ram (comma-sep) to restrict what it pulls
# Env:
#   ZEN_FLEET_BUCKET  run-write bucket (default coefficient)
#   ZEN_CORPUS_BUCKET corpus read-only bucket (default codec-corpus); set == run bucket for single-bucket
#   ZEN_CORPUS_PREFIX R2 prefix under the corpus bucket where source images live (real jobs)
#   ZEN_WORKER_IMAGE  worker image (default the canonical exec image)
#   ZEN_EXEC          executor (the exec image already defaults this to the real `zenmetrics jobexec` shim)
#
# For a TRULY persistent basement worker (not re-minting weekly), create long-lived R2 API tokens in
# the Cloudflare dashboard (R2 -> Manage API Tokens) — one Object Read & Write scoped to the run bucket,
# one Object Read-only scoped to the corpus bucket — and hand-edit the AWS_*/ZEN_CORPUS_AWS_* keys in the
# printed command (drop the AWS_SESSION_TOKEN / ZEN_CORPUS_AWS_SESSION_TOKEN lines).
set -euo pipefail
RUN="${1:?usage: unraid_worker.sh <RUN> [TTL_DAYS] [CAPABILITY]}"
TTL_DAYS="${2:-7}"; CAP="${3:-}"
# Canonical worker image via the single source of truth — never a hard-coded ghcr name.
. "$(dirname "$0")/fleet.env"
IMAGE="${ZEN_WORKER_IMAGE:-$ZEN_FLEET_IMAGE_CPU}"   # exec image bakes `zenmetrics jobexec`
RUN_BUCKET="${ZEN_FLEET_BUCKET:-coefficient}"        # run-WRITE: blobs / ledger / claims / manifest
CORPUS_BUCKET="${ZEN_CORPUS_BUCKET:-codec-corpus}"   # corpus READ-ONLY: source images
EXEC="${ZEN_EXEC:-/bin/cat}"   # /bin/cat = synthetic; the exec image defaults this to the real executor.
CORPUS="${ZEN_CORPUS_PREFIX:-}"   # R2 prefix under the CORPUS bucket where source images live (real jobs)
TTL=$(( TTL_DAYS * 86400 )); [ "$TTL" -ge 900 ] && [ "$TTL" -le 604800 ] || { echo "TTL_DAYS must be 1..7"; exit 1; }

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# 1. RUN cred: scoped object-READ-WRITE, limited to this run's prefix in the RUN bucket (never root).
body=$(RUN="$RUN" BUCKET="$RUN_BUCKET" TTL="$TTL" python3 -c "import json,os;print(json.dumps({'bucket':os.environ['BUCKET'],'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':int(os.environ['TTL']),'prefixes':[os.environ['RUN']+'/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/unraid_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/unraid_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { echo "FAILED to mint scoped RUN creds:"; cat /tmp/unraid_cred.json; exit 1; }

# 2. CORPUS cred: a SEPARATE object-READ-ONLY cred for the corpus bucket, only when it differs from the
#    run bucket (temp creds are single-bucket). jobexec uses ZEN_CORPUS_AWS_* for the source fetch.
CAK=""; CSK=""; CST=""; CORPUS_ENV=""
if [ "$CORPUS_BUCKET" != "$RUN_BUCKET" ]; then
  cbody=$(CB="$CORPUS_BUCKET" CP="$CORPUS" TTL="$TTL" python3 -c "import json,os;p=os.environ.get('CP','').strip('/');print(json.dumps({'bucket':os.environ['CB'],'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-only','ttlSeconds':int(os.environ['TTL']),**({'prefixes':[p+'/']} if p else {})}))")
  curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$cbody" \
    "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/unraid_corpus_cred.json
  read -r CAK CSK CST < <(python3 -c 'import json;r=json.load(open("/tmp/unraid_corpus_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
  [ -n "${CAK:-}" ] || { echo "FAILED to mint scoped CORPUS creds:"; cat /tmp/unraid_corpus_cred.json; exit 1; }
  CORPUS_ENV="-e ZEN_CORPUS_AWS_ACCESS_KEY_ID=$CAK -e ZEN_CORPUS_AWS_SECRET_ACCESS_KEY=$CSK -e ZEN_CORPUS_AWS_SESSION_TOKEN=$CST"
  echo "minted RUN(rw $RUN_BUCKET/$RUN) + CORPUS(ro $CORPUS_BUCKET/${CORPUS:-<all>}) creds (${TTL_DAYS}d)"
else
  echo "minted RUN(rw $RUN_BUCKET/$RUN) cred (${TTL_DAYS}d); corpus == run bucket → single cred"
fi

MANIFEST="s3://$RUN_BUCKET/$RUN/manifest-unraid.json"
# The basement worker gets its own shuffled manifest (decorrelated claim order — see launch_fleet.sh).
if [ -f /tmp/fleet_manifest.json ]; then
  python3 -c 'import json,random;j=json.load(open("/tmp/fleet_manifest.json"));random.seed("unraid");random.shuffle(j);json.dump(j,open("/tmp/fleet_manifest_unraid.json","w"))'
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
    s5cmd --endpoint-url "$EP" cp /tmp/fleet_manifest_unraid.json "$MANIFEST" >/dev/null && echo "uploaded $MANIFEST"
else
  echo "NOTE: /tmp/fleet_manifest.json not found — set ZEN_MANIFEST_URI below to the run's manifest."
  MANIFEST="s3://$RUN_BUCKET/$RUN/manifest.json"
fi

cat <<EOF

================  RUN THIS ON THE UNRAID BOX (basement tier)  ================
Scoped creds expire in ${TTL_DAYS}d. Outbound HTTPS only — no inbound ports.
RUN bucket (read-write): $RUN_BUCKET    CORPUS bucket (read-only): $CORPUS_BUCKET

docker run -d --name zen-worker-$RUN --restart no \\
  -e AWS_ACCESS_KEY_ID=$AK \\
  -e AWS_SECRET_ACCESS_KEY=$SK \\
  -e AWS_SESSION_TOKEN=$ST \\
  -e AWS_REGION=auto \\
  -e ZEN_R2_ENDPOINT=$EP \\
  -e ZEN_BUCKET=$RUN_BUCKET \\
  -e ZEN_RUN=$RUN \\
  -e ZEN_MANIFEST_URI=$MANIFEST \\
  -e ZEN_PROVIDER=basement \\
  -e ZEN_WORKER=unraid \\
  -e ZEN_EXEC=$EXEC \\
  -e ZEN_CORPUS_BUCKET=$CORPUS_BUCKET \\
  -e ZEN_CORPUS_PREFIX=$CORPUS \\
  $CORPUS_ENV \\
  -e ZEN_CONTROL_KEY=$RUN/control.json \\
  -e ZEN_SPEC_THRESHOLD_SECS=20 \\
  -e ZEN_IDLE_PASSES=8 \\
  ${CAP:+-e ZEN_CAPABILITY=$CAP \\}
  $IMAGE

Unraid GUI ("Docker" tab -> "Add Container"):
  Repository: $IMAGE
  Network:    bridge (no published ports needed)
  Restart:    No  (the worker drains its share of this run, then exits cleanly)
  Variables:  add each -e KEY=VALUE above as a Container Variable
==============================================================================
EOF
echo "verify it's working from the workstation:  scripts/jobsys/fleet status $RUN   (boxes, util, idle, progress)"
