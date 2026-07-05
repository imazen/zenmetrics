#!/usr/bin/env bash
# Hetzner ML-TRAINING launcher — one box per codec runs the per-codec dual-model
# picker pipeline (picker_tree_ab A/B + train_hybrid) via the zen-train:hybrid-cpu
# image. The workstation can't run concurrent heavy ML (documented lockup risk),
# so this fans the per-codec trains out to dedicated Hetzner boxes IN PARALLEL.
#
# Each box: pulls the public image, runs dualmodel_runner.sh for ONE codec
# (canonical parquets + omni/features TSVs from R2 -> results back to R2), and
# SELF-DESTRUCTS on success AND failure (host cloud-init deletes the server via
# the Hetzner metadata-id + API token, which stays HOST-side — never in the
# container; the container only ever holds SCOPED, short-TTL R2 creds). A local
# background monitor tails progress and is the teardown backstop.
#
#   CODEC=zenwebp_lossy bash scripts/train/hetzner_ml_train.sh
#
# Env:
#   CODEC        one codec (zenwebp_lossy | zenavif_lossy | zenjpeg_lossy |
#                zenjxl_lossy | zenjxl_lossless | zenpng_lossless | zenwebp_lossless)
#   STYPE        hcloud server type (default cpx51 = 16 vCPU / 32 GB)
#   IMAGE        worker image (default ghcr.io/imazen/zen-train:hybrid-cpu)
#   MAXMIN       hard self-destruct backstop in minutes (default 120)
#   SKIP_TRAIN_HYBRID=1   run only Stage A (picker_tree_ab)
set -u
CODEC="${CODEC:?set CODEC=zenwebp_lossy (or another per-codec task)}"
STYPE="${STYPE:-cpx51}"
IMAGE="${IMAGE:-ghcr.io/imazen/zen-train:hybrid-cpu}"
MAXMIN="${MAXMIN:-90}"          # per-box self-destruct backstop (task: 90 min/box)
RUN="${RUN:-mltrain-$(date +%s)}"
# Metric axis: zensim (default) or ssim2. PICKER_TARGET names the Stage B MLP output;
# METRIC_COL = the omni column for the MLP; SCORE_COL = the canonical-parquet column
# for the Stage A CART (prep_combined renames it to score_zensim so picker_tree_ab,
# which hardcodes that column, computes reach/oracle on it).
PICKER_TARGET="${PICKER_TARGET:-zensim_a}"
METRIC_COL="${METRIC_COL:-score_zensim}"
SCORE_COL="${SCORE_COL:-score_zensim}"
# Default OUT_PREFIX namespaces ssim2 runs so they don't overwrite the zensim outputs.
case "$PICKER_TARGET" in *ssim2*) _MTAG="__ssim2";; *) _MTAG="";; esac
OUT_PREFIX="${OUT_PREFIX:-dualmodel-2026-06-28/${CODEC}${_MTAG}}"
# Hetzner server names must be valid hostnames (no underscores) — sanitize codec+metric
# so the zensim and ssim2 boxes for one codec don't collide on the same name.
NAME="$RUN-$(echo "${CODEC}-${PICKER_TARGET}" | tr '_' '-')"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
SKIP_TRAIN_HYBRID="${SKIP_TRAIN_HYBRID:-0}"
# picker_tree_ab is single-threaded; the runner caps it (--skip-mlp/--max-train/
# --perm-val-cap/--skip-rf + timeout) so the GBDT dump + CART + train_hybrid all
# finish well inside MAXMIN. SKIP_TEST_SPLIT=1 (val only) trims it further.
SKIP_TEST_SPLIT="${SKIP_TEST_SPLIT:-1}"
SKIP_RF="${SKIP_RF:-1}"

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
HCLOUD_TOKEN="$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')"
export HCLOUD_TOKEN
[ -n "$HCLOUD_TOKEN" ] || { echo "FATAL: no hcloud api_token"; exit 1; }
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto aws s3 "$@" --endpoint-url "$EP"; }

# keyed on the unique box NAME (codec+metric) so the zensim and ssim2 boxes for
# one codec don't clobber each other's monitor log.
MON_LOG="/tmp/hetzner_mltrain_${NAME}.log"
echo "### $RUN  codec=$CODEC type=$STYPE image=$IMAGE  monitor->$MON_LOG"

# Clear STALE terminal markers from a prior run in this OUT_PREFIX. Without this the
# per-box monitor (which tears the box down on the FIRST _DONE/_FAILED it sees in the
# prefix) instantly kills a fresh box on a leftover marker — observed 2026-06-28 when
# zenpng_lossless/zenwebp_lossless (completed a prior run) had stale _DONE and the new
# boxes were destroyed within ~2 min before doing any work. The runner re-uploads its
# own deliverables; we only purge the markers so the monitor reacts to THIS run.
r2 rm "s3://zentrain/$OUT_PREFIX/_DONE"   >/dev/null 2>&1 || true
r2 rm "s3://zentrain/$OUT_PREFIX/_FAILED" >/dev/null 2>&1 || true

# 1) scoped temp R2 cred — RW on zentrain, scoped to the two prefixes the box
#    touches (canonical read + dualmodel-2026-06-28 read/write). Never the root key.
body=$(python3 -c "import json,os;print(json.dumps({
  'bucket':'zentrain',
  'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],
  'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],
  'permission':'object-read-write',
  'ttlSeconds':$((MAXMIN*60+1800)),
  'prefixes':['canonical/2026-06-27/','dualmodel-2026-06-28/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/mltrain_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/mltrain_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])' 2>/dev/null)
[ -n "${AK:-}" ] || { echo "FATAL: R2 cred mint failed"; cat /tmp/mltrain_cred.json; exit 1; }
echo "minted scoped RW cred (ttl ${MAXMIN}m+30m): zentrain/{canonical/2026-06-27,dualmodel-2026-06-28}"

# 2) cloud-init (HOST). Writes the container env-file (scoped R2 creds, NO hcloud
#    token), docker-runs the image (= the JOB), uploads the host log, self-destructs.
CI="$(mktemp)"
cat > "$CI" <<EOF
#!/bin/bash
exec > /root/ci.log 2>&1
set -x
set +x   # trace OFF across the secret assignment — xtrace would print the literal
         # token into ci.log, which is uploaded to R2 below (security hole, fixed
         # 2026-07-05; token already rotated).
HCLOUD_TOKEN='$HCLOUD_TOKEN'   # HOST-only — for self-destruct; never passed to the container
set -x
EP='$EP'
IMG='$IMAGE'
OUTP='$OUT_PREFIX'
cat > /root/cenv <<ENV
CODEC=$CODEC
R2_ENDPOINT=$EP
RUN_BUCKET=zentrain
CANON_PREFIX=canonical/2026-06-27
OUT_PREFIX=$OUT_PREFIX
INPUTS_PREFIX=dualmodel-2026-06-28/inputs
PICKER_TARGET=$PICKER_TARGET
METRIC_COL=$METRIC_COL
SCORE_COL=$SCORE_COL
RUN_ID=$RUN
SKIP_TRAIN_HYBRID=$SKIP_TRAIN_HYBRID
SKIP_TEST_SPLIT=$SKIP_TEST_SPLIT
SKIP_RF=$SKIP_RF
AWS_ACCESS_KEY_ID=$AK
AWS_SECRET_ACCESS_KEY=$SK
AWS_SESSION_TOKEN=$ST
AWS_REGION=auto
ENV

destroy_self(){
  set +x   # trace OFF for the whole function — the DELETE call below carries the
           # token in an Authorization header; xtrace would print it into ci.log.
  local ID
  ID=\$(curl -s --max-time 10 http://169.254.169.254/hetzner/v1/metadata/instance-id || true)
  [ -n "\$ID" ] || ID=\$(curl -s --max-time 10 http://169.254.169.254/latest/meta-data/instance-id || true)
  for i in 1 2 3 4 5; do
    curl -s --max-time 20 -X DELETE -H "Authorization: Bearer \$HCLOUD_TOKEN" \
      "https://api.hetzner.cloud/v1/servers/\$ID" && break
    sleep 5
  done
  set -x
}
# hard-timeout backstop — destroy no matter what after ${MAXMIN}m
( sleep $((MAXMIN*60)); echo "BACKSTOP timeout firing"; destroy_self ) &

docker pull "\$IMG" || true
docker run --rm --env-file /root/cenv "\$IMG"
rc=\$?
echo "container exited rc=\$rc"
# upload the host ci.log via the image's baked s5cmd (creds via env-file)
docker run --rm --env-file /root/cenv -v /root/ci.log:/ci.log --entrypoint /usr/local/bin/s5cmd \
  "\$IMG" --endpoint-url="\$EP" cp /ci.log "s3://zentrain/\$OUTP/ci.host.log" || true
destroy_self
EOF

# 3) provision — EU ONLY (fsn1/nbg1/hel1: cpx51 ~EUR0.1338/hr, ~3.3x cheaper than
#    the ash/hil US fallback). Prefer the cheap shared cpx tiers; smaller type if
#    cpx51 is capacity-out in every EU DC. NEVER fall to US.
# picker_tree_ab peaks ~9 GB RSS (measured, zenjpeg 1.48M rows) so every fallback
# type must have >=16 GB; cpx31/cpx21 (8 GB) would OOM mid-dump. Prefer the cheap
# shared cpx51 (~EUR0.1338/hr, 32 GB); when those are EU-capacity-out (common),
# fall to ccx33 (8 vCPU/32 GB, ~EUR0.288/hr) before the pricier ccx43.
launched=0; lasterr=""
for typ in "$STYPE" cpx51 cpx41 ccx33 ccx43; do
  for loc in fsn1 nbg1 hel1; do
    lasterr=$(hcloud server create --name "$NAME" --type "$typ" --image docker-ce --location "$loc" \
      --ssh-key "$SSH_KEY" --label group="$RUN" --label codec="$CODEC" \
      --user-data-from-file "$CI" 2>&1) \
      && { echo "launched $NAME ($typ/$loc)"; launched=1; ACTUAL_TYPE="$typ"; break 2; } || true
  done
done
rm -f "$CI"
[ "$launched" = 1 ] || { echo "FATAL: server create failed all type/loc"; echo "$lasterr" | tail -4; exit 1; }

# price (best-effort, for the cost report)
PRICE=$(hcloud server-type describe "$ACTUAL_TYPE" -o json 2>/dev/null | python3 -c '
import json,sys
try:
  d=json.load(sys.stdin); p=d["prices"][0]["price_hourly"]["gross"]; print(f"{float(p):.4f}")
except Exception: print("?")' 2>/dev/null)
echo "type=$ACTUAL_TYPE  ~EUR ${PRICE}/hr (gross, ex-IPv4)"

# 4) background monitor — streams progress + is the teardown backstop.
(
  start=$(date +%s)
  echo "[monitor] $NAME ($ACTUAL_TYPE) launched $(date -u +%FT%TZ); image=$IMAGE; out=s3://zentrain/$OUT_PREFIX/"
  while :; do
    now=$(date +%s); el=$(( (now-start)/60 ))
    status=$(hcloud server describe "$NAME" -o json 2>/dev/null | python3 -c 'import json,sys;print(json.load(sys.stdin).get("status","gone"))' 2>/dev/null || echo gone)
    have=$(r2 ls "s3://zentrain/$OUT_PREFIX/" 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')
    echo "[monitor +${el}m] box=$status  r2_keys=[ $have ]"
    if echo "$have" | grep -qE '_DONE|_FAILED'; then
      echo "[monitor] marker seen ([ $have ]) — ensuring box is destroyed"
      hcloud server delete "$NAME" 2>/dev/null && echo "[monitor] deleted $NAME" || echo "[monitor] box already gone (self-destructed)"
      break
    fi
    if [ "$status" = "gone" ] && [ "$el" -ge 2 ]; then
      echo "[monitor] box gone (self-destructed) — checking final markers"
      sleep 10
      r2 ls "s3://zentrain/$OUT_PREFIX/" 2>/dev/null | awk '{print $NF}'
      break
    fi
    if [ "$el" -ge "$MAXMIN" ]; then
      echo "[monitor] MAXMIN ${MAXMIN}m exceeded — force-deleting $NAME"
      hcloud server delete "$NAME" 2>/dev/null || true
      break
    fi
    sleep 30
  done
  echo "[monitor] === FINAL: results at s3://zentrain/$OUT_PREFIX/ ==="
  r2 ls "s3://zentrain/$OUT_PREFIX/" 2>/dev/null | awk '{print "  "$0}'
  echo "[monitor] done $(date -u +%FT%TZ)"
) > "$MON_LOG" 2>&1 &
MONPID=$!
echo "monitor PID=$MONPID -> tail -f $MON_LOG"
echo
echo "### SCALE-UP (after smoke review): one box per codec, in parallel —"
echo "  for c in zenavif_lossy zenjpeg_lossy zenjxl_lossy zenjxl_lossless zenpng_lossless zenwebp_lossless; do CODEC=\$c bash scripts/train/hetzner_ml_train.sh; done"
echo "  (each codec's omni TSV must be uploaded to s3://zentrain/dualmodel-2026-06-28/inputs/<family>.zensim.combined.tsv first for Stage B)"
echo "teardown (manual, if needed): hcloud server delete $NAME"
