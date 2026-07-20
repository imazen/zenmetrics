#!/usr/bin/env bash
# Launch Hetzner CX/CPX (cheap shared-vCPU) boxes to run a ScoreFile manifest — the CPU counterpart
# of gpu_scorefile_launch.sh for decode-bound fills (avif decode dominates; GPUs sat idle on vast).
# Same job-system contract: workers claim chunk jobs off jobs/<run>/manifest.json via R2-lease,
# byte-range-fetch variants out of the existing tar via variant_index.tsv, score CPU metrics, write
# blobs/. Boxes are cloud-init docker-ce; the worker is the baked CPU exec image (fleet.env).
#   usage: hetzner_scorefile_launch.sh <run_id> <N_boxes>
#   env:   ZEN_TAR_OVERRIDE (required — tar URI), ZEN_CORPUS_PREFIX_OVERRIDE (refs prefix on
#          codec-corpus), TYPES ("cx53 cx43 cpx62 cpx52"), LOCATIONS ("fsn1 nbg1 hel1"), RESUME=1 (skip
#          pause/resume dance — run already live)
set -uo pipefail
RUN="${1:?run id}"; N="${2:-2}"
. "$(dirname "$0")/fleet.env"
IMAGE="${ZEN_CPU_IMAGE:-$ZEN_FLEET_IMAGE_CPU}"
# ZEN_BUCKET — the ONE bucket holding run + tar + refs (R2 temp creds are single-bucket,
# so all three must share it). Default codec-corpus; set to `zentrain` for the canonical
# corpora whose variant tars live in zentrain (declare with ZEN_JOBS_BUCKET=zentrain +
# copy the refs into zentrain). A cross-bucket run 403s every variant fetch.
BUCKET="${ZEN_BUCKET:-codec-corpus}"; RUNP="jobs/$RUN"
TAR="${ZEN_TAR_OVERRIDE:?set ZEN_TAR_OVERRIDE}"
CORPUS_PREFIX="${ZEN_CORPUS_PREFIX_OVERRIDE:?set ZEN_CORPUS_PREFIX_OVERRIDE}"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
# Idle AUTOSHUTDOWN (Hetzner has NO built-in one — idle.rs is a dashboard alarm with
# zero side effects, fleet watch --destroy is vast-only, and the old cloud-init ran
# the worker --restart unless-stopped so a drained box looped + billed forever). Each
# box now SELF-DESTRUCTS via the hcloud API: (1) drain-exit — when the gap drains the
# worker exits clean and the box deletes itself; (2) a hard-runtime backstop for a
# hung worker / stuck pull. Token is HOST-only (never enters the container), read from
# the same creds file the other hetzner scripts use. ZEN_SELF_DESTRUCT=0 opts out
# (e.g. to keep a box for debugging — then teardown_fleet.sh is the only stop).
SELF_DESTRUCT="${ZEN_SELF_DESTRUCT:-1}"
MAX_RUNTIME_MIN="${ZEN_MAX_RUNTIME_MIN:-720}"   # backstop ceiling (min); default 12h = cred TTL
HCLOUD_TOKEN="${HCLOUD_TOKEN:-$(grep -E '^api_token=' ~/.config/hetzner/credentials 2>/dev/null | head -1 | cut -d= -f2- | tr -d ' \r')}"
[ -n "$HCLOUD_TOKEN" ] || { echo "FATAL: no hcloud api_token (~/.config/hetzner/credentials) — needed for box self-destruct + create"; exit 1; }
export HCLOUD_TOKEN
LOG=/tmp/hz_scorefile_launch.log; log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }
# scoped RW creds: run prefix + datagen (tar) + corpus refs + ENCODES prefix (direct-object
# mode reads variants there — omitting it 403s every fetch, producing all-error blobs) — 12h.
TARPFX=$(python3 -c "import sys;u='$TAR';print('/'.join(u.split('/')[3:-1])+'/')")
# The encodes prefix a direct-object run reads; add its PARENT so one cred covers many codecs.
ENCPFX="${ZEN_ENCODES_PREFIX:+${ZEN_ENCODES_PREFIX%/}/}"
body=$(ZEN_ENCPFX="$ENCPFX" python3 -c "import json,os;pfx=['$RUNP/','$TARPFX','$CORPUS_PREFIX/'];e=os.environ.get('ZEN_ENCPFX','');
[pfx.append(e) for _ in [0] if e];
print(json.dumps({'bucket':'$BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':43200,'prefixes':pfx}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/hzsf_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/hzsf_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "${AK:-}" ] || { log "cred mint failed"; cat /tmp/hzsf_cred.json | tee -a "$LOG"; exit 1; }
MANIFEST="s3://$BUCKET/$RUNP/manifest.json"; CTLKEY="$RUNP/control.json"
IDX="s3://$BUCKET/$RUNP/variant_index.tsv"
if [ "${RESUME:-0}" = "0" ]; then
  printf '{"paused":false}' > /tmp/hzsf_ctl.json; r2 cp /tmp/hzsf_ctl.json "s3://$BUCKET/$CTLKEY" >/dev/null
fi
NJOBS=$(r2 cat "$MANIFEST" 2>/dev/null | python3 -c "import json,sys;print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
log "run=$RUN jobs=$NJOBS image=$IMAGE boxes=$N types=${TYPES:-cx53 cx43 cpx62 cpx52}"
launch_box(){
  local wk="$1" name="hzsf-${RUN//[^a-zA-Z0-9-]/}-$1" ci typ loc ok=0 err
  ci=$(mktemp)
  cat > "$ci" <<EOF
#cloud-config
runcmd:
  - |
    exec >> /root/ci.log 2>&1
    # ── Idle autoshutdown: self-destruct via the hcloud API. Token is HOST-only.
    HCTOKEN='$HCLOUD_TOKEN'
    destroy_self(){
      ID=\$(curl -s --max-time 10 http://169.254.169.254/hetzner/v1/metadata/instance-id 2>/dev/null || true)
      [ -n "\$ID" ] || ID=\$(curl -s --max-time 10 http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || true)
      [ -n "\$ID" ] || { echo "[destroy] no instance-id from metadata — cannot self-delete"; return 1; }
      for i in 1 2 3 4 5; do
        curl -s --max-time 20 -X DELETE -H "Authorization: Bearer \$HCTOKEN" \
          "https://api.hetzner.cloud/v1/servers/\$ID" && { echo "[destroy] deleted server \$ID"; return 0; }
        sleep 5
      done
      echo "[destroy] FAILED to delete server \$ID after 5 tries"
    }
    SELF_DESTRUCT='$SELF_DESTRUCT'
    # Hard-runtime backstop: destroy no matter what after ${MAX_RUNTIME_MIN}m (hung
    # worker / stuck image pull / lost R2) — the drain-exit below is the normal path.
    [ "\$SELF_DESTRUCT" = 1 ] && ( sleep $(( MAX_RUNTIME_MIN * 60 )); echo "[backstop] ${MAX_RUNTIME_MIN}m runtime ceiling -> destroy"; destroy_self ) &
    # Worker in the FOREGROUND with --restart no: when the gap drains
    # (ZEN_IDLE_PASSES consecutive no-work passes) the entrypoint exits clean and we
    # fall through to destroy_self. (Was -d --restart unless-stopped = looped forever.)
    docker run --name zensf --restart no \
      -e AWS_ACCESS_KEY_ID='$AK' -e AWS_SECRET_ACCESS_KEY='$SK' -e AWS_SESSION_TOKEN='$ST' -e AWS_REGION=auto \
      -e ZEN_R2_ENDPOINT='$EP' -e ZEN_BUCKET='$BUCKET' -e ZEN_RUN='$RUNP' \
      -e ZEN_MANIFEST_URI='$MANIFEST' -e ZEN_CONTROL_KEY='$CTLKEY' \
      -e ZEN_CORPUS_PREFIX='$CORPUS_PREFIX' -e ZEN_VARIANTS_TAR_URI='$TAR' -e ZEN_VARIANT_INDEX_URI='$IDX' \
      ${ZEN_ENCODES_PREFIX:+-e ZEN_ENCODES_PREFIX='$ZEN_ENCODES_PREFIX'} ${ZEN_ENCODES_BUCKET:+-e ZEN_ENCODES_BUCKET='$ZEN_ENCODES_BUCKET'} \
      -e RAYON_NUM_THREADS='${ZEN_RAYON_THREADS:-1}' -e OMP_NUM_THREADS='${ZEN_RAYON_THREADS:-1}' \
      ${ZEN_CHUNK_WALL_SEC:+-e ZEN_CHUNK_WALL_SEC='$ZEN_CHUNK_WALL_SEC'} \
      -e ZEN_PERSISTENT_EXEC=1 -e ZEN_PROVIDER=hetzner-cpu -e ZEN_IDLE_PASSES='${ZEN_IDLE_PASSES:-10}' -e ZEN_WORKER='hz-$wk' \
      --entrypoint /usr/local/bin/fleet-entrypoint.sh '$IMAGE'
    echo "[drain] worker exited rc=\$? -> self-destroy (SELF_DESTRUCT=\$SELF_DESTRUCT)"
    [ "\$SELF_DESTRUCT" = 1 ] && destroy_self || echo "[drain] SELF_DESTRUCT=0 — leaving box up (teardown_fleet.sh to stop billing)"
EOF
  for typ in ${TYPES:-cx53 cx43 cpx62 cpx52}; do
    for loc in ${LOCATIONS:-fsn1 nbg1 hel1}; do
      err=$(hcloud server create --name "$name" --type "$typ" --image docker-ce --location "$loc" \
        --ssh-key "$SSH_KEY" --label group="$RUN" --user-data-from-file "$ci" 2>&1) \
        && { log "$name launched ($typ/$loc)"; ok=1; break 2; } || true
    done
  done
  [ "$ok" = 1 ] || { log "$name FAILED all type/loc: $(echo "$err" | head -2)"; }
  rm -f "$ci"
}
for w in $(seq 1 "$N"); do launch_box "$w" & done
wait
log "### launched $N boxes onto run=$RUN (blobs: s3://$BUCKET/$RUNP/blobs/)"
if [ "$SELF_DESTRUCT" = 1 ]; then
  log "autoshutdown: ON — each box SELF-DELETES on drain (gap empty) or after ${MAX_RUNTIME_MIN}m backstop. Confirm: hcloud server list -l group=$RUN"
else
  log "autoshutdown: OFF (ZEN_SELF_DESTRUCT=0) — boxes bill until you tear down"
fi
log "manual teardown: bash scripts/jobsys/teardown_fleet.sh $RUN   (or: fleet kill $RUN)"
