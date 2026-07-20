#!/usr/bin/env bash
# POOL launcher — the hourly-efficient, cost-capped fleet for the zensim-720 byte-range backfill.
#
# Design (fixes the churn that hourly billing punished):
#   * Each box runs POOL mode (fleet-entrypoint.sh): it works the WHOLE undone-tar pool in a round-robin,
#     coordinating with peers via the R2 claim/ledger, then SELF-DESTRUCTS at ZEN_MAX_MIN (55) minutes —
#     one full paid hour, no idle, no churn, never a second billed hour.
#   * EU SHARED boxes only (cx43 > cx33 > cx23). cpx (dedicated/expensive) and US regions are BANNED.
#   * HARD $2/hr START CAP: this launches ONE batch of N boxes whose worst-case hourly price is <= MAX_EUR
#     (default 1.85 EUR ~= $2). Run it once per hour (see pool_cron) and total starts stay <= $2/hr.
#
#   usage: pool_launch.sh [N]      (N default sized to the cap; env MAX_EUR, ZEN_MAX_MIN)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
. "$(dirname "$0")/fleet.env" 2>/dev/null || true
IMAGE="${ZEN_CPU_IMAGE:-ghcr.io/imazen/zenfleet-worker:exec}"
BUCKET="zentrain"
MAX_EUR="${MAX_EUR:-1.85}"                       # worst-case hourly price ceiling for the batch (~$2)
ZEN_MAX_MIN="${ZEN_MAX_MIN:-55}"                 # box lifetime — one paid hour minus a safety margin
TYPES="${TYPES:-cx43 cx33 cx23}"                 # EU SHARED only; NO cpx, NO US
LOCATIONS="${LOCATIONS:-nbg1 fsn1 hel1}"         # EU only
CX43=0.0296                                      # priciest type we allow -> worst-case sizing
N="${1:-$(python3 -c "import math;print(min(60, int($MAX_EUR/$CX43)))")}"  # <= MAX_EUR even if all cx43
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
HCLOUD_TOKEN="${HCLOUD_TOKEN:-$(grep -E '^api_token=' ~/.config/hetzner/credentials 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \r')}"
[ -n "$HCLOUD_TOKEN" ] || { echo "FATAL: no hcloud token"; exit 1; }
export HCLOUD_TOKEN
LOG=~/tmp/hz720/pool_launch.log; mkdir -p ~/tmp/hz720; log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

# ── SAFETY: refuse if starting this batch would exceed the cap even at the priciest allowed type.
WORST=$(python3 -c "print(f'{$N*$CX43:.2f}')")
python3 -c "exit(0 if $N*$CX43 <= $MAX_EUR + 1e-9 else 1)" \
  || { log "REFUSED: $N boxes * ${CX43}EUR = ${WORST}EUR > cap ${MAX_EUR}EUR"; exit 2; }

# ── Runlist: every undone tar-run (byte-range) + zenjpeg (direct-object). run<TAB>src<TAB>mode.
gen_runlist(){
  local out=~/tmp/hz720/runlist.tsv; : > "$out"
  # tag  sweep  ntars
  local rows=(
    "zavif mandfix4-zenavif-1782593621 8"
    "zjxll jxl-lossy-vardct-1782609551 24"
    "zwebp mandfix2-zenwebp-1782584881 9"
    "zjxlm jxl-modular-1782596759 10"
    "zpng  mandfix2-zenpng-1782584881 2"
  )
  for row in "${rows[@]}"; do
    read -r tag sweep ntars <<<"$row"
    for ((i=0;i<ntars;i++)); do
      printf 'bf-%s-t%d\ts3://%s/jxl-lossy/runs/%s/variants/box-%d.tar\ttar\n' "$tag" "$i" "$BUCKET" "$sweep" "$i" >>"$out"
    done
  done
  # zenjpeg: direct-object (individual encodes)
  printf 'bf-zjl2\tcanonical/2026-06-27/zenjpeg_lossy/encodes\tenc\n' >>"$out"
  echo "$out"
}
RL=$(gen_runlist)
r2 cp "$RL" "s3://$BUCKET/jobs/_pool/runlist.tsv" >/dev/null
log "runlist: $(grep -c . "$RL") runs -> s3://$BUCKET/jobs/_pool/runlist.tsv"

# ── Scoped RW cred covering EVERY run's jobs/, all tars, all encodes, and refs — one cred, 90 min TTL.
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':5400,'prefixes':['jobs/','jxl-lossy/runs/','canonical/2026-06-27/','refs/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > ~/tmp/hz720/pool_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/home/lilith/tmp/hz720/pool_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { log "cred mint failed"; cat ~/tmp/hz720/pool_cred.json | tee -a "$LOG"; exit 1; }

log "launching $N EU pool boxes (worst-case ${WORST}EUR/hr <= ${MAX_EUR}), ${ZEN_MAX_MIN}min lifetime, types='$TYPES'"
STAMP=$(date -u +%H%M%S)
launch_box(){
  local wk="$1"; local name="hzpool-${STAMP}-${wk}" ci typ loc ok=0
  ci=$(mktemp)
  cat > "$ci" <<EOF
#cloud-config
runcmd:
  - |
    exec >> /root/ci.log 2>&1
    HCTOKEN='$HCLOUD_TOKEN'
    destroy_self(){
      ID=\$(curl -s --max-time 10 http://169.254.169.254/hetzner/v1/metadata/instance-id 2>/dev/null || true)
      [ -n "\$ID" ] || return 1
      for i in 1 2 3 4 5; do curl -s --max-time 20 -X DELETE -H "Authorization: Bearer \$HCTOKEN" "https://api.hetzner.cloud/v1/servers/\$ID" && return 0; sleep 5; done
    }
    # Hard backstop: kill no matter what a few min after the intended lifetime (hung pull / worker).
    ( sleep $(( (ZEN_MAX_MIN + 5) * 60 )); destroy_self ) &
    # POOL worker in the FOREGROUND: works the pool for ~${ZEN_MAX_MIN}min then EXITS -> self-destruct.
    docker run --name zpool --restart no \
      -e AWS_ACCESS_KEY_ID='$AK' -e AWS_SECRET_ACCESS_KEY='$SK' -e AWS_SESSION_TOKEN='$ST' -e AWS_REGION=auto \
      -e ZEN_R2_ENDPOINT='$EP' -e ZEN_BUCKET='$BUCKET' \
      -e ZEN_POOL_RUNLIST='s3://$BUCKET/jobs/_pool/runlist.tsv' -e ZEN_MAX_MIN='$ZEN_MAX_MIN' \
      -e ZEN_CORPUS_PREFIX='refs/clean-picker-corpus-2026-06-26' \
      -e RAYON_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
      -e ZEN_CHUNK_WALL_SEC=20 -e ZEN_CORE_OVERSUBSCRIBE=2 -e ZEN_PASS_TIMEOUT=5400 \
      -e ZEN_PERSISTENT_EXEC=1 -e ZEN_PROVIDER=hetzner-pool -e ZEN_WORKER='pool-$wk' \
      --entrypoint /usr/local/bin/fleet-entrypoint.sh '$IMAGE'
    echo "[pool] worker exited rc=\$? -> self-destroy"
    destroy_self
EOF
  for typ in $TYPES; do for loc in $LOCATIONS; do
    hcloud server create --name "$name" --type "$typ" --image docker-ce --location "$loc" \
      --ssh-key "$SSH_KEY" --label group="pool" --user-data-from-file "$ci" >/dev/null 2>&1 \
      && { ok=1; break 2; } || true
  done; done
  rm -f "$ci"
  [ "$ok" = 1 ] || log "  $name FAILED all type/loc (capacity)"
}
for w in $(seq 1 "$N"); do launch_box "$w" & done
wait
UP=$(hcloud server list -o columns=name 2>/dev/null | grep -c hzpool)
log "### batch launched: $UP hzpool boxes up (target $N). They self-destruct at ${ZEN_MAX_MIN}min."
