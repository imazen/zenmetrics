#!/usr/bin/env bash
# REMOTE chunked `zenmetrics sweep` for the hqfill-A remainder (chunks 11-29).
#
# WHY THIS SHAPE (coordinator-confirmed path, 2026-07-02): the salvaged chunks 0-10
# were made by `zenmetrics sweep --feature-output` — so running the SAME tool on the
# remainder yields COLUMN-IDENTICAL output (7 metric cols + 372 feat + variants) by
# construction, no score_file/tar plumbing, no metric-vs-feature gap. No rebuild
# needed (the salvage binary is baked in the GPU image) → the concurrent-agent ravif
# build-block is irrelevant. All-remote (vast GPU) → honors "no local encode".
#
# The chunk_11 OOM was chunk-SIZE (2086 LARGEST-rendition cells in one process tipped
# the allocator high-water), NOT the tool. FIX: bound cells-per-process — size-stratify
# so big-rendition chunks are SMALL; run on a >=32 GB box; and the salvage's ~2100
# normal-rendition cells peaked <40 GB so normal chunks stay safe.
#
# Backends MATCH the salvage EXACTLY (7 cols, byte-identical names):
#   ssim2=CPU, dssim=CPU, zensim=GPU(profile A, 372 WithIw feat),
#   butteraugli(max+pnorm3)=GPU, cvvdp=GPU, iwssim=GPU.
# --use-legacy-scheduler dodges the cubecl orchestrator warm-bench descriptor race that
# a real vast card hits (the local WSL2 5070 masked it); same kernels → same scores.
#
# Modeled on hetzner_cpu_sweep.sh (scoped creds, chunk lists, per-box DONE, self-destruct)
# but targets vast GPU via --onstart-cmd (vast ignores the image ENTRYPOINT).
set -u

RUN="${RUN:-hqfillA-$(date +%s)}"
RUN_BUCKET="${RUN_BUCKET:-zentrain}"                 # run-WRITE (codec-corpus is RO)
CORPUS_PREFIX="${CORPUS_PREFIX:-jxl-lossy-hqfill-A/2026-07-01/corpus}"   # renditions live in RUN bucket (uploaded by launcher)
RUN_PREFIX="jxl-lossy-hqfill-A/2026-07-01/remote/$RUN"
IMAGE="${IMAGE:-ghcr.io/imazen/zenfleet-worker:exec-gpu-hqfillA-d5a142e0e166}"
DIST='0.05,0.08,0.11,0.14,0.17,0.2,0.25,0.3,0.35,0.45,0.6,0.8,1.0,1.3'
METRICS="${METRICS:-zensim-gpu ssim2 butteraugli-gpu cvvdp-gpu dssim iwssim-gpu}"
# Size-stratified chunking: renditions >= BIG_PX go in SMALL chunks (OOM-safe); the rest larger.
BIG_PX="${BIG_PX:-400000}"           # >=0.4MP (640x640+) = the chunk_11 OOM class
BIG_PER="${BIG_PER:-40}"             # 40 big renditions x 14 = 560 cells/chunk
SMALL_PER="${SMALL_PER:-120}"        # 120 small renditions x 14 = 1680 cells/chunk (well under the salvage's safe 2100)
RENDS="${RENDS:-/tmp/remaining_rends.txt}"
N_SMOKE="${N_SMOKE:-1}"              # stage-1 smoke box count

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
export VASTAI_API_KEY="$(cat ~/.config/vastai/vast_api_key 2>/dev/null | tr -d ' \r\n')"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto aws s3 "$@" --endpoint-url "$EP"; }

echo "### hqfill-A REMOTE run=$RUN image=$IMAGE"
echo "    corpus=s3://$RUN_BUCKET/$CORPUS_PREFIX  out=s3://$RUN_BUCKET/$RUN_PREFIX"

# 1. scoped temp creds — RW on the run+corpus prefixes (renditions are in the RUN bucket here, so one cred covers both).
body=$(B="$RUN_BUCKET" python3 -c "import json,os;print(json.dumps({'bucket':os.environ['B'],'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':43200,'prefixes':['jxl-lossy-hqfill-A/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/hqa_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/hqa_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
[ -n "$AK" ] || { echo "cred mint failed"; cat /tmp/hqa_cred.json; exit 1; }
echo "minted scoped 12h creds (RW $RUN_BUCKET/jxl-lossy-hqfill-A/)"

# 2. size-stratified chunk lists
python3 - "$RENDS" "$BIG_PX" "$BIG_PER" "$SMALL_PER" > /tmp/hqa_chunk_manifest.txt <<'PY'
import re,sys
rends_file, big_px, big_per, small_per = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def px(p):
    m=re.search(r'scale(\d+)x(\d+)',p); return int(m.group(1))*int(m.group(2)) if m else 0
rends=[l.strip() for l in open(rends_file) if l.strip()]
big=sorted((r for r in rends if px(r)>=big_px), key=px, reverse=True)
small=[r for r in rends if px(r)<big_px]
chunks=[]
for i in range(0,len(big),big_per): chunks.append(big[i:i+big_per])
for i in range(0,len(small),small_per): chunks.append(small[i:i+small_per])
import os
os.makedirs('/tmp/hqa_chunks',exist_ok=True)
for ci,c in enumerate(chunks):
    open(f'/tmp/hqa_chunks/chunk-{ci}.txt','w').write('\n'.join(os.path.basename(r) for r in c)+'\n')
    print(f'chunk-{ci} n={len(c)} cells={len(c)*14}')
print(f'TOTAL_CHUNKS {len(chunks)}', file=sys.stderr)
PY
NCHUNKS=$(ls /tmp/hqa_chunks/chunk-*.txt 2>/dev/null | wc -l)
echo "built $NCHUNKS size-stratified chunk lists (big: $BIG_PER rends/chunk, small: $SMALL_PER)"
for cf in /tmp/hqa_chunks/chunk-*.txt; do
  ci=$(basename "$cf" .txt); r2 cp "$cf" "s3://$RUN_BUCKET/$RUN_PREFIX/chunks/$ci.txt" >/dev/null
done
echo "uploaded $NCHUNKS chunk lists to R2"

# 3. box-side worker = the COMMITTED standalone script (no heredoc escaping traps).
WORKER_SRC="$(dirname "$0")/hqfill_A_box_worker.sh"
[ -f "$WORKER_SRC" ] || { echo "FATAL: $WORKER_SRC missing"; exit 1; }
r2 cp "$WORKER_SRC" "s3://$RUN_BUCKET/$RUN_PREFIX/worker.sh" >/dev/null
echo "uploaded committed worker.sh (hqfill_A_box_worker.sh)"

# 4. launch vast GPU boxes (>=32GB RAM so no chunk can OOM the box; cuda>=12.6; fast net)
launch_box(){ local idx="$1"
  local OFFERS OFFER
  OFFERS=$(vastai search offers 'num_gpus=1 cuda_max_good>=12.6 gpu_ram>=8 cpu_ram>=32 disk_space>=40 rentable=true inet_down>300' -o 'dph+' --raw 2>/dev/null)
  OFFER=$(echo "$OFFERS" | python3 -c "import json,sys;o=json.load(sys.stdin);print(o[$idx]['id'] if len(o)>$idx else '')")
  [ -z "$OFFER" ] && { echo "no offer for box $idx"; return 1; }
  local ONSTART="set +e
export PATH=/usr/local/sbin:/usr/sbin:/sbin:\$PATH
env | grep -E '^(AWS_|ZEN_)' >> /etc/environment
s5cmd --endpoint-url \$ZEN_R2_ENDPOINT cp s3://$RUN_BUCKET/$RUN_PREFIX/worker.sh /usr/local/bin/hqa_worker.sh
bash /usr/local/bin/hqa_worker.sh 2>&1 | tee /var/log/hqa.log"
  local ENVB="-e ZEN_R2_ENDPOINT=$EP -e ZEN_BUCKET=$RUN_BUCKET -e ZEN_RUN_PREFIX=$RUN_PREFIX -e ZEN_CORPUS_PREFIX=$CORPUS_PREFIX -e ZEN_AK=$AK -e ZEN_SK=$SK -e ZEN_ST=$ST -e ZEN_DIST=$DIST -e ZEN_METRICS=\"$METRICS\" -e ZEN_BOX=$idx -e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$AK -e R2_SECRET_ACCESS_KEY=$SK"
  vastai create instance "$OFFER" --image "$IMAGE" --label "group=$RUN" --disk 40 --env "$ENVB" --onstart-cmd "$ONSTART" 2>&1 | grep -iE 'new_contract|success' | head -1 && echo "box $idx launched (offer $OFFER)"
}

STAGE="${STAGE:-smoke}"
if [ "$STAGE" = "smoke" ]; then
  echo "=== STAGE 1: SMOKE ($N_SMOKE box) ==="
  for i in $(seq 0 $((N_SMOKE-1))); do launch_box "$i"; done
  echo "RUN=$RUN — smoke launched. Verify a chunk lands + columns match salvage + no OOM, THEN: STAGE=scale N_BOXES=<n> RUN=$RUN bash $0"
else
  N_BOXES="${N_BOXES:-6}"
  echo "=== STAGE 2: SCALE ($N_BOXES boxes) ==="
  for i in $(seq 0 $((N_BOXES-1))); do launch_box "$i" & done
  wait
  echo "RUN=$RUN — $N_BOXES boxes launched."
fi
echo "monitor: r2 ls s3://$RUN_BUCKET/$RUN_PREFIX/done/  |  teardown: vastai show instances (label group=$RUN) -> destroy"
echo "$RUN" > /tmp/hqfillA_remote_run.txt
