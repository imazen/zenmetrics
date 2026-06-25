#!/usr/bin/env bash
# 1-pass Hetzner-CPU chunk-sweep for picker training data.
#
# Each cheap cpx box fetches its chunk of renditions from R2 and runs
# `zenmetrics sweep --metric ssim2 --metric zensim` (the deadlock-fixed CPU
# binary baked in the PUBLIC exec image) -> an omni TSV (image,codec,q,
# knob_tuple_json,encoded_bytes,score_ssim2,score_zensim) = exactly the picker
# training format -> uploads it. The launcher mints SCOPED temp R2 creds (3h,
# object-read-write to the run prefix only — never a root key on a box), bakes a
# box-worker script via cloud-init (docker-ce image -> docker is pre-installed,
# no apt at boot), polls R2 for each box's DONE marker, and destroys the box.
#
# Usage:
#   RUN=smoke CODEC=zenjpeg PLAN=rd_core N_BOXES=1 IMAGES=10 \
#     bash scripts/sweep/hetzner_cpu_sweep.sh
# Env:
#   RUN          run id (default fleet-cpu-<unix>)        CODEC   one zen codec (zenjpeg…)
#   PLAN         rd_core|modes_full|scalar_dense          QG      q-grid (default web-weighted 7)
#   N_BOXES      cpx boxes to launch                      IMAGES  cap renditions (0 = all 1482)
#   STYPE        hcloud server type (default cpx41)       BUDGET  cell budget for modes_full/scalar_dense
set -u
SRC_BUCKET="${SRC_BUCKET:-codec-corpus}"
SRC_PREFIX="${SRC_PREFIX:-picker-sweep-2026-06-22/renditions}"
RUN="${RUN:-fleet-cpu-$(date +%s)}"
RUN_PREFIX="picker-sweep-2026-06-22/runs/$RUN"
CODEC="${CODEC:-zenjpeg}"; PLAN="${PLAN:-rd_core}"
QG="${QG:-5,15,30,50,70,85,95}"; N_BOXES="${N_BOXES:-1}"; IMAGES="${IMAGES:-0}"
STYPE="${STYPE:-cpx41}"; BUDGET="${BUDGET:-600}"
# Canonical image via the single source of truth (scripts/jobsys/fleet.env) — no hard-coded ghcr name.
. "$(dirname "$0")/../jobsys/fleet.env"
IMAGE="${IMAGE:-$ZEN_FLEET_IMAGE_CPU}"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export HCLOUD_TOKEN="$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')"
 r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

echo "### $RUN  codec=$CODEC plan=$PLAN boxes=$N_BOXES images=${IMAGES:-all} type=$STYPE"

# 1. scoped temp creds (read renditions + write run outputs; one bucket/prefix)
body=$(python3 -c "import json,os;print(json.dumps({'bucket':'$SRC_BUCKET','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':10800,'prefixes':['picker-sweep-2026-06-22/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/hz_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/hz_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "$AK" ] || { echo "cred mint failed"; cat /tmp/hz_cred.json; exit 1; }
echo "minted scoped creds (3h)"

# 2. chunk lists from the R2 rendition listing (cap + size-skip the >4MP monsters by name)
r2 ls "s3://$SRC_BUCKET/$SRC_PREFIX/" | awk '{print $NF}' | grep '\.png$' > /tmp/hz_all.txt
# keep renditions <= MAXPX (default 4.2 MP) by parsing scaleWxH from the name —
# matches the local picker's 4 MP cap; the corpus has up to 100 MP monsters.
MAXPX="${MAXPX:-4200000}" MINPX="${MINPX:-0}" python3 -c '
import re,os,sys
cap=int(os.environ["MAXPX"]); lo=int(os.environ["MINPX"])
for l in open("/tmp/hz_all.txt"):
    m=re.search(r"scale(\d+)x(\d+)",l)
    if m:
        px=int(m.group(1))*int(m.group(2))
        if lo < px <= cap: sys.stdout.write(l)  # MINPX..MAXPX window -> big-image tier sets MINPX=4200000
' > /tmp/hz_ok.txt
[ "$IMAGES" -gt 0 ] 2>/dev/null && head -n "$IMAGES" /tmp/hz_ok.txt > /tmp/hz_sel.txt || cp /tmp/hz_ok.txt /tmp/hz_sel.txt
total=$(wc -l < /tmp/hz_sel.txt); per=$(( (total + N_BOXES - 1) / N_BOXES ))
echo "selected $total renditions; $per per box"
split -d -l "$per" /tmp/hz_sel.txt /tmp/hz_chunk_
i=0
for cf in /tmp/hz_chunk_*; do
  r2 cp "$cf" "s3://$SRC_BUCKET/$RUN_PREFIX/chunks/chunk-$i.txt" >/dev/null
  i=$((i+1))
done
echo "uploaded $i chunk lists"

# 3. launch boxes — cloud-init writes env+worker, docker-runs the public exec image
launch_box(){ local idx="$1" name="$RUN-cpu-$1" ci; ci="$(mktemp)"
  cat > "$ci" <<EOF
#!/bin/bash
set -e
mkdir -p /root/r
cat > /root/r/env <<ENV
AWS_ACCESS_KEY_ID=$AK
AWS_SECRET_ACCESS_KEY=$SK
AWS_SESSION_TOKEN=$ST
AWS_REGION=auto
EP=$EP
BUCKET=$SRC_BUCKET
SRC_PREFIX=$SRC_PREFIX
CHUNK_KEY=$RUN_PREFIX/chunks/chunk-$idx.txt
OUT_KEY=$RUN_PREFIX/omni/box-$idx.omni.tsv
MANIFEST_KEY=$RUN_PREFIX/manifests/box-$idx.plan.json
DONE_KEY=$RUN_PREFIX/done/box-$idx.done
ENC_KEY=$RUN_PREFIX/variants/box-$idx.tar
FEAT_KEY=$RUN_PREFIX/features/box-$idx.feat.parquet
CODEC=$CODEC
PLAN=$PLAN
QG=$QG
BUDGET=$BUDGET
SWEEP_JOBS=${SWEEP_JOBS:-4}
${THREADS:+RAYON_NUM_THREADS=$THREADS}
ENV
cat > /root/r/worker.sh <<'WORK'
set -e
mkdir -p /data
s5cmd --endpoint-url=\$EP cp "s3://\$BUCKET/\$CHUNK_KEY" /data/chunk.txt
while read -r f; do [ -n "\$f" ] && s5cmd --endpoint-url=\$EP cp "s3://\$BUCKET/\$SRC_PREFIX/\$f" "/data/\$f"; done < /data/chunk.txt
rm -f /data/chunk.txt
PB=""; [ "\$PLAN" != "rd_core" ] && PB="--plan-budget \$BUDGET"
mkdir -p /enc
zenmetrics sweep --codec "\$CODEC" --sources /data --q-grid "\$QG" --plan "\$PLAN" \$PB \
  --jobs "\${SWEEP_JOBS:-4}" \
  --metric ssim2 --metric zensim --encoded-out-dir /enc --feature-output /feat.parquet --output /omni.tsv
s5cmd --endpoint-url=\$EP cp /omni.tsv "s3://\$BUCKET/\$OUT_KEY"
# codec-commit provenance (the plan manifest carries codec_commits) — lands WITH the blobs
s5cmd --endpoint-url=\$EP cp /omni.plan.json "s3://\$BUCKET/\$MANIFEST_KEY" 2>/dev/null || true
s5cmd --endpoint-url=\$EP cp /feat.parquet "s3://\$BUCKET/\$FEAT_KEY"
# persist encoded variants (the master record): 372 zensim features re-extractable
# on GPU (zensim-gpu fixed), plus diffmaps / any future metric, with NO re-encode.
# Variants are already-compressed codec bytes -> tar without recompression.
tar -cf /enc.tar -C /enc . && s5cmd --endpoint-url=\$EP cp /enc.tar "s3://\$BUCKET/\$ENC_KEY"
printf 'rows=%s\n' "\$(wc -l < /omni.tsv)" > /done.txt
s5cmd --endpoint-url=\$EP cp /done.txt "s3://\$BUCKET/\$DONE_KEY"
WORK
docker run --rm --env-file /root/r/env -v /root/r/worker.sh:/worker.sh \
  --entrypoint /bin/bash $IMAGE /worker.sh > /root/r/log 2>&1 || \
  s5cmd --endpoint-url=$EP cp /root/r/log "s3://$SRC_BUCKET/$RUN_PREFIX/done/box-$idx.FAILED" 2>/dev/null || true
EOF
  local typ loc ok=0 err
  # ccx (dedicated AMD) first — not phased out like cpx41; then cpx shared fallbacks.
  # biggest-first: when slot-limited, max cores/box. ccx (dedicated) > cpx (shared) for CPU sweep.
  for typ in ${TYPES:-$STYPE ccx63 ccx53 ccx43 cpx51 ccx33 cpx31}; do
    for loc in ${LOCATIONS:-fsn1 nbg1 hel1 ash hil}; do
      err=$(hcloud server create --name "$name" --type "$typ" --image docker-ce --location "$loc" \
        --ssh-key "$SSH_KEY" --label group="$RUN" --user-data-from-file "$ci" 2>&1) \
        && { echo "$name launched ($typ/$loc)"; ok=1; break 2; } || true
    done
  done
  [ "$ok" = 1 ] || { echo "$name FAILED all type/loc"; printf '%s\n' "$err" | grep -iE 'unavailable|unsupported|limit|invalid' | head -1; }
  rm -f "$ci"
}
for n in ${CHUNKS:-$(seq 0 $((N_BOXES-1)))}; do launch_box "$n" & done
wait
echo "### launched. poll: bash scripts/sweep/hetzner_cpu_sweep.sh ... then watch s3://$SRC_BUCKET/$RUN_PREFIX/done/"
echo "RUN=$RUN  out=s3://$SRC_BUCKET/$RUN_PREFIX/omni/  done=s3://$SRC_BUCKET/$RUN_PREFIX/done/"
echo "teardown: hcloud server list -l group=$RUN -o noheader | awk '{print \$2}' | xargs -r hcloud server delete"
