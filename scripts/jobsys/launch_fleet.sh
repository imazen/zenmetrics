#!/usr/bin/env bash
# Launch a heterogeneous ≥3-tier fleet (goal H) on ONE R2 conditional-write-lease queue, all running
# the SAME baked zenfleet-worker image (ghcr.io/imazen/zenfleet-worker). Provider-agnostic: adding/removing
# a tier never touches job logic — only this launcher differs per provider.
#
#   THIS SPENDS MONEY (Hetzner + vast boxes). It is the billable step; run it deliberately.
#   Tears down with: scripts/jobsys/teardown_fleet.sh <RUN>  (or the dashboard Kill controls).
#
# Prerequisites (one-time):
#   - The image is built + pushed by CI (.github/workflows/jobworker-image.yml) and the ghcr package
#     is PUBLIC (Settings → Packages → zenfleet-worker → change visibility), so boxes pull without creds.
#   - Env: R2 creds at ~/.config/cloudflare/r2-credentials; HCLOUD token at ~/.config/hetzner/credentials;
#     vastai CLI authed; ssh key zen-arm-dev-20260528 on the Hetzner project.
#
# Usage: bash scripts/jobsys/launch_fleet.sh [N_JOBS] [HETZNER_X86] [VAST] [HETZNER_ARM]
#   Tiers are interchangeable + provider-agnostic; pass 0 for any you don't want. local + Hetzner cpx
#   (x86 burst) + Hetzner cax (arm64 capability tier, needs the multi-arch image) + vast (burst) —
#   any ≥2 = a heterogeneous fleet on one R2 queue. e.g. `… 60 1 1 0` = local + Hetzner-x86 + vast.
set -euo pipefail
# Default image runs the SYNTHETIC executor (/bin/cat) — for the demos/proofs. For REAL jobs set
# ZEN_WORKER_IMAGE=ghcr.io/imazen/zenfleet-worker:exec (bakes zenmetrics jobexec; its image-level
# ZEN_EXEC default is the real executor) and ZEN_CORPUS_PREFIX=<R2 prefix of your source images>.
IMAGE="${ZEN_WORKER_IMAGE:-ghcr.io/imazen/zenfleet-worker:latest}"
EXEC="${ZEN_EXEC:-/bin/cat}"           # override for real work (or rely on the exec image's ZEN_EXEC default)
CORPUS="${ZEN_CORPUS_PREFIX:-}"        # R2 prefix under the CORPUS bucket where source images live (real jobs)
CORPUS_BUCKET="${ZEN_CORPUS_BUCKET:-codec-corpus}"  # corpus READ-ONLY bucket (source images)
N_JOBS="${1:-200}"; N_HZ="${2:-1}"; N_VAST="${3:-1}"; N_HZ_ARM="${4:-0}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"   # run-WRITE bucket: manifest / claims / ledger / blobs
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
RUN="fleet-$(date -u +%Y%m%d-%H%M%S)"
echo "### launching fleet on s3://$BUCKET/$RUN/  image=$IMAGE  jobs=$N_JOBS  hetzner-x86=$N_HZ hetzner-arm=$N_HZ_ARM vast=$N_VAST"

# 1. mint SCOPED temp creds (object-read-write to this run only; never the root key on remote boxes)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':10800,'prefixes':['$RUN/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/fleet_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/fleet_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
echo "minted scoped run creds (3h, rw $BUCKET/$RUN)"

# 1b. Corpus READ-ONLY cred — a SECOND single-bucket cred for the corpus when it differs from the run
# bucket (R2 temp creds are single-bucket). jobexec reads codec-corpus with this via ZEN_CORPUS_AWS_*;
# the run cred above (rw, scoped to $RUN/) never touches the read-only corpus.
CAK=""; CSK=""; CST=""; CORPUS_CRED_ENV=""
if [ "$CORPUS_BUCKET" != "$BUCKET" ]; then
  cbody=$(CB="$CORPUS_BUCKET" CP="$CORPUS" python3 -c "import json,os;p=os.environ.get('CP','').strip('/');print(json.dumps({'bucket':os.environ['CB'],'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-only','ttlSeconds':10800,**({'prefixes':[p+'/']} if p else {})}))")
  curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$cbody" \
    "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/fleet_corpus_cred.json
  read -r CAK CSK CST < <(python3 -c 'import json;r=json.load(open("/tmp/fleet_corpus_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
  [ -n "${CAK:-}" ] || { echo "FAILED to mint scoped corpus creds:"; cat /tmp/fleet_corpus_cred.json; exit 1; }
  CORPUS_CRED_ENV="-e ZEN_CORPUS_AWS_ACCESS_KEY_ID=$CAK -e ZEN_CORPUS_AWS_SECRET_ACCESS_KEY=$CSK -e ZEN_CORPUS_AWS_SESSION_TOKEN=$CST"
  echo "minted scoped corpus read-only cred (ro $CORPUS_BUCKET/${CORPUS:-<all>})"
fi

# 2. manifest → R2 (root creds for the upload)
python3 - "$N_JOBS" > /tmp/fleet_spec.json <<'PY'
import json,sys,hashlib
n=int(sys.argv[1])
print(json.dumps({"items":[{"image_path":"fleet/img-%05d.png"%i,"codec":"zenjpeg","q":80,
  "encode_sha":hashlib.sha256(("fleet-%d"%i).encode()).hexdigest()} for i in range(n)],"metrics":["cvvdp"]}))
PY
"$ROOT/target/release/zenfleet-ctl" declare --spec /tmp/fleet_spec.json --out /tmp/fleet_manifest.json
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

# Per-worker shuffled manifest: same job set + same R2 claims namespace (ONE queue), but each worker
# iterates the gap in a different order so they don't all hammer job 0 first. Without this the
# lowest-latency node monopolizes every conditional-write race ({local:60, others:0}); with it, work
# distributes and the fast node still pulls more (goal H "fast nodes pull more"). Seeded by worker name.
shuf_manifest() {  # worker -> echoes the uploaded shuffled-manifest URI
  local w="$1"
  W="$w" python3 -c 'import json,random,os
w=os.environ["W"]; j=json.load(open("/tmp/fleet_manifest.json")); random.seed(w); random.shuffle(j)
json.dump(j, open("/tmp/fleet_manifest_"+w+".json","w"))'
  AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
    s5cmd --endpoint-url "$EP" cp "/tmp/fleet_manifest_$w.json" "s3://$BUCKET/$RUN/manifest-$w.json" >/dev/null 2>&1
  echo "s3://$BUCKET/$RUN/manifest-$w.json"
}

# common env block the entrypoint reads (scoped creds + queue coordinates + pause control).
# $1 = provider label, $2 = ZEN_MANIFEST_URI (defaults to the shared manifest if omitted).
envblock() { cat <<EOF
-e AWS_ACCESS_KEY_ID=$AK -e AWS_SECRET_ACCESS_KEY=$SK -e AWS_SESSION_TOKEN=$ST -e AWS_REGION=auto
-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$BUCKET -e ZEN_RUN=$RUN -e ZEN_MANIFEST_URI=${2:-$MANIFEST}
-e ZEN_PROVIDER=$1 -e ZEN_EXEC=$EXEC -e ZEN_CORPUS_BUCKET=$CORPUS_BUCKET -e ZEN_CORPUS_PREFIX=$CORPUS $CORPUS_CRED_ENV -e ZEN_SPEC_THRESHOLD_SECS=20 -e ZEN_CONTROL_KEY=$CTLKEY -e ZEN_IDLE_PASSES=8
EOF
}

# 3a. local tier (this machine) — docker run the baked image
docker run -d --label group=$RUN --name "$RUN-local" $(envblock local "$(shuf_manifest workstation)") -e ZEN_WORKER=workstation "$IMAGE" >/dev/null && echo "local worker started"

# 3b. Hetzner burst tier(s) — Docker-CE on Ubuntu; cloud-init docker-runs the baked (multi-arch) image.
# hcloud takes user-data ONLY from a file (--user-data-from-file; there is no --user-data-from-string),
# so each box's cloud-init is written to a temp file first. cax* = ARM (Ampere), cpx* = x86 (AMD).
export HCLOUD_TOKEN=$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')
hz_box() {  # name  server-type  provider-label  worker-id
  local name="$1" stype="$2" provider="$3" worker="$4" cif m; cif="$(mktemp)"; m="$(shuf_manifest "$worker")"
  printf '#!/bin/bash\ncurl -fsSL https://get.docker.com | sh\ndocker run -d --restart no %s -e ZEN_WORKER=%s %s\n' \
    "$(envblock "$provider" "$m" | tr '\n' ' ')" "$worker" "$IMAGE" > "$cif"
  hcloud server create --name "$name" --type "$stype" --image ubuntu-24.04 --location fsn1 \
    --ssh-key zen-arm-dev-20260528 --label group=$RUN --user-data-from-file "$cif" >/dev/null 2>&1 \
    && echo "$worker launched ($stype)" || echo "$worker FAILED ($stype/fsn1 unavailable? try nbg1/hel1)"
  rm -f "$cif"
}
for i in $(seq 1 "$N_HZ"); do hz_box "$RUN-hetzner-$i" cpx22 hetzner "hetzner-$i"; done
# ARM capability tier (Ampere cax) — same image, arm64 slice of the manifest.
for i in $(seq 1 "$N_HZ_ARM"); do hz_box "$RUN-hetzner-arm-$i" cax11 hetzner-arm "hetzner-arm-$i"; done

# 3c. vast burst tier(s) — the baked image IS the instance image; entrypoint runs with env
for i in $(seq 1 "$N_VAST"); do
  OFFER=$(vastai search offers 'num_gpus=0 cpu_cores>=2 disk_space>=14 rentable=true verified=true' -o 'dph+' --raw 2>/dev/null | python3 -c 'import json,sys;o=json.load(sys.stdin);print(o[0]["id"] if o else "")')
  [ -z "$OFFER" ] && { echo "vast-$i: no offer"; continue; }
  vastai create instance "$OFFER" --image "$IMAGE" --label group=$RUN --disk 16 \
    --env "$(envblock vast-$i "$(shuf_manifest vast-$i)" | tr '\n' ' ') -e ZEN_WORKER=vast-$i" >/dev/null 2>&1 \
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
echo "### RESUMED — RUN=$RUN — watch: scripts/jobsys/fleet watch $RUN ; teardown: scripts/jobsys/fleet kill $RUN"
echo "$RUN" > /tmp/fleet_run.txt

# 5. Startup watchdog (background): KNOW within ~2 min if any launched box never starts working
#    (image-pull hang / onstart crash / 6-80s fast-crash). Provider-agnostic via R2 boot records
#    (boot/<worker>.txt vs expected count); vast boxes also get per-box util/status detail. It
#    self-stops once every box is working. Disable with ZEN_NO_STARTUP_WATCH=1.
if [ "${ZEN_NO_STARTUP_WATCH:-0}" != "1" ]; then
  FLEET="$(dirname "$0")/fleet"
  echo "### fleet watch (bg): flags any box idle / not-started-within-2min → /tmp/$RUN-startup.log"
  ( AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto \
    R2_ACCOUNT_ID="$R2_ACCOUNT_ID" ZEN_FLEET_BUCKET="$BUCKET" \
    bash "$FLEET" watch "$RUN" --label "$RUN" --max-wait 600 \
      >"/tmp/$RUN-startup.log" 2>&1 & )
  echo "    tail -f /tmp/$RUN-startup.log   (or:  scripts/jobsys/fleet status $RUN)"
fi
