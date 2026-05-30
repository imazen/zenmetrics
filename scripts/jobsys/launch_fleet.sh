#!/usr/bin/env bash
# Launch a heterogeneous ≥3-tier fleet (goal H) on ONE R2 conditional-write-lease queue, all running
# the SAME baked zen-jobworker image (ghcr.io/imazen/zen-jobworker). Provider-agnostic: adding/removing
# a tier never touches job logic — only this launcher differs per provider.
#
#   THIS SPENDS MONEY (Hetzner + vast boxes). It is the billable step; run it deliberately.
#   Tears down with: scripts/jobsys/teardown_fleet.sh <RUN>  (or the dashboard Kill controls).
#
# Prerequisites (one-time):
#   - The image is built + pushed by CI (.github/workflows/jobworker-image.yml) and the ghcr package
#     is PUBLIC (Settings → Packages → zen-jobworker → change visibility), so boxes pull without creds.
#   - Env: R2 creds at ~/.config/cloudflare/r2-credentials; HCLOUD token at ~/.config/hetzner/credentials;
#     vastai CLI authed; ssh key zen-arm-dev-20260528 on the Hetzner project.
#
# Usage: bash scripts/jobsys/launch_fleet.sh [N_JOBS] [HETZNER_BOXES] [VAST_BOXES]
set -euo pipefail
IMAGE="${ZEN_WORKER_IMAGE:-ghcr.io/imazen/zen-jobworker:latest}"
N_JOBS="${1:-200}"; N_HZ="${2:-1}"; N_VAST="${3:-1}"
BUCKET="${ZEN_FLEET_BUCKET:-zen-tuning-ephemeral}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="fleet-$(date -u +%Y%m%d-%H%M%S)"
echo "### launching fleet on s3://$BUCKET/$RUN/  image=$IMAGE  jobs=$N_JOBS  hetzner=$N_HZ vast=$N_VAST"

# 1. mint SCOPED temp creds (object-read-write to this run only; never the root key on remote boxes)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':10800,'prefixes':['$RUN/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/fleet_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/fleet_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
echo "minted scoped creds (3h)"

# 2. manifest → R2 (root creds for the upload)
python3 - "$N_JOBS" > /tmp/fleet_spec.json <<'PY'
import json,sys,hashlib
n=int(sys.argv[1])
print(json.dumps({"items":[{"image_path":"fleet/img-%05d.png"%i,"codec":"zenjpeg","q":80,
  "encode_sha":hashlib.sha256(("fleet-%d"%i).encode()).hexdigest()} for i in range(n)],"metrics":["cvvdp"]}))
PY
"$ROOT/target/release/zen-jobctl" declare --spec /tmp/fleet_spec.json --out /tmp/fleet_manifest.json
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/fleet_manifest.json "s3://$BUCKET/$RUN/manifest.json" >/dev/null
MANIFEST="s3://$BUCKET/$RUN/manifest.json"
echo "uploaded $N_JOBS-job manifest"

# Pause-orchestration (so all tiers overlap regardless of boot time): start the run PAUSED, bring up
# every tier (each idles on the RunControl while it boots — the entrypoint waits, doesn't exit), then
# RESUME once all are up so they race the queue simultaneously. Workers honor it via --control-r2-key.
CTLKEY="$RUN/control.json"
printf '{"paused":true}' > /tmp/fleet_ctl.json
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/fleet_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
echo "run starts PAUSED (control=$CTLKEY); will resume after boot"

# common env block the entrypoint reads (scoped creds + queue coordinates + pause control)
envblock() { cat <<EOF
-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto
-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUN -e ZEN_MANIFEST_URI=$MANIFEST
-e ZEN_PROVIDER=$1 -e ZEN_EXEC=/bin/cat -e ZEN_SPEC_THRESHOLD_SECS=20 -e ZEN_CONTROL_KEY=$CTLKEY -e ZEN_IDLE_PASSES=8
EOF
}

# 3a. local tier (this machine) — docker run the baked image
docker run -d --label group=$RUN --name "$RUN-local" $(envblock local) -e ZEN_WORKER=workstation "$IMAGE" >/dev/null && echo "local worker started"

# 3b. Hetzner burst tier(s) — Docker-CE app image; cloud-init docker-runs the baked image
export HCLOUD_TOKEN=$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')
for i in $(seq 1 "$N_HZ"); do
  ci=$(printf '#!/bin/bash\ncurl -fsSL https://get.docker.com | sh\ndocker run -d --restart no %s -e ZEN_WORKER=hetzner-%s %s\n' "$(envblock hetzner | tr '\n' ' ')" "$i" "$IMAGE")
  hcloud server create --name "$RUN-hetzner-$i" --type cpx22 --image ubuntu-24.04 --location fsn1 \
    --ssh-key zen-arm-dev-20260528 --label group=$RUN --user-data-from-string "$ci" >/dev/null 2>&1 \
    && echo "hetzner-$i launched" || echo "hetzner-$i FAILED (cpx22/fsn1 unavailable? try nbg1/hel1)"
done

# 3c. vast burst tier(s) — the baked image IS the instance image; entrypoint runs with env
for i in $(seq 1 "$N_VAST"); do
  OFFER=$(vastai search offers 'num_gpus=0 cpu_cores>=2 disk_space>=14 rentable=true verified=true' -o 'dph+' --raw 2>/dev/null | python3 -c 'import json,sys;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')
  [ -z "$OFFER" ] && { echo "vast-$i: no offer"; continue; }
  vastai create instance "$OFFER" --image "$IMAGE" --label group=$RUN --disk 16 \
    --env "$(envblock vast-$i | tr '\n' ' ') -e ZEN_WORKER=vast-$i" >/dev/null 2>&1 \
    && echo "vast-$i launched on offer $OFFER" || echo "vast-$i FAILED on $OFFER"
done

# 4. resume — let every tier race the queue at once. Wait for boxes to boot first (Hetzner ~60s,
# vast ~2-3min); workers idle on the pause until now, then all start claiming together.
WAIT="${ZEN_BOOT_WAIT_SECS:-200}"
echo "all tiers launched; waiting ${WAIT}s for boot, then RESUME (concurrent start)…"
sleep "$WAIT"
printf '{"paused":false}' > /tmp/fleet_ctl.json
AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
  s5cmd --endpoint-url "$EP" cp /tmp/fleet_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
echo "### RESUMED — RUN=$RUN — watch: scripts/jobsys/watch_fleet.sh $RUN ; teardown: scripts/jobsys/teardown_fleet.sh $RUN"
echo "$RUN" > /tmp/fleet_run.txt
