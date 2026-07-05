#!/usr/bin/env bash
# Autonomous fleet leave-one-out (LOO) feature-ablation launcher.
#
# Fans the per-feature retrains of a (codec, metric) picker across many cheap EU
# Hetzner boxes so the ablation runs ~Nx faster than serial. NO encoding — pure
# retraining on the EXISTING swept data, so it's embarrassingly parallel.
#
#   CODEC=zenjpeg_lossy METRIC=ssim2 N_BOXES=6 bash scripts/train/loo_fleet.sh
#   CODEC=zenjpeg_lossy METRIC=ssim2 SMOKE=1   bash scripts/train/loo_fleet.sh   # 1 box, 2-feat smoke
#   CODEC=zenjpeg_lossy METRIC=ssim2 N_BOXES=4 ROUND=2 bash scripts/train/loo_fleet.sh   # pair-aware verify
#
# Pipeline:
#   1. Build the pareto ONCE locally (omni→pareto for jpeg/webp/avif; canonical→pareto
#      for jxl) + upload the small parquet — boxes pull a few MB, not the 900MB omni.
#   2. ROUND 1: jobs = baseline + one --drop-features per KEEP_FEATURES feature.
#      ROUND 2 (pair-aware): from round-1 results, JOINT-drop the droppable-looking set
#      + bisection subsets + top-correlated pairwise drops (single-LOO under-estimates
#      pair value; the verified safe-to-drop set is proven by joint drop, not singles).
#   3. Split jobs into N balanced batches; each box pulls its batch, pulls the pareto
#      ONCE, then LOOPS train_hybrid --drop-features over its whole batch (amortize the
#      ~10-min boot/pull fixed cost over many runs — the explicit efficiency ask).
#   4. Each box SELF-DESTRUCTS on done/fail (host cloud-init); a local monitor tails
#      progress, is the teardown backstop, enforces the €/time watchdog, then COLLECTS
#      results → ranked loo_<codec>_<metric>.tsv. Verifies 0 boxes remain.
#
# Box lifecycle / scoped-cred / EU-only / self-destruct patterns COPY hetzner_ml_train.sh.
# Does NOT touch train_hybrid/bake_picker/zenpredict (read-only); only new scripts.
set -u
REPO="${REPO:-/home/lilith/work/zen/zenmetrics}"
CODEC="${CODEC:?set CODEC=zenjpeg_lossy|zenwebp_lossy|zenjxl_lossy|zenavif_lossy}"
METRIC="${METRIC:-ssim2}"
ROUND="${ROUND:-1}"
SMOKE="${SMOKE:-0}"
SEED="${SEED:-12345}"
HIDDEN="${HIDDEN:-192,192,192}"
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-25m}"
STYPE="${STYPE:-ccx33}"                 # 8 vCPU / 32 GB dedicated, EUR0.266/hr EU — cheapest 32GB-adequate
                                        # (cpx51 is EUR0.448/hr — MORE expensive; cpx41 is EUR0.227 but only 16GB).
                                        # train_hybrid loads the full pareto (jpeg 3.8M rows ~3-4GB) + models; 32GB is safe.
IMAGE="${IMAGE:-ghcr.io/imazen/zen-train:hybrid-cpu}"
MAXMIN="${MAXMIN:-80}"                  # per-box self-destruct backstop (min)
MAX_BURN_EUR="${MAX_BURN_EUR:-15}"      # fleet €-cap watchdog
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
DATE="${DATE:-$(date +%Y-%m-%d)}"
WORKDIR="${WORKDIR:-$REPO/scripts/train/loo_work}"; mkdir -p "$WORKDIR"

# ── per-codec wiring: family / config module / pareto-file stem / data source ──
# config = the picker config module train_hybrid imports; pareto_stem = its CODEC var
# (names {stem}.{target}.pareto.parquet). source = how the pareto is built.
case "$CODEC" in
  zenjpeg_lossy) FAMILY=zenjpeg; CONFIG_MODULE=zenjpeg_picker;       PARETO_STEM=zenjpeg;            SRC=omni;      OMNI=zenjpeg.zensim.combined.tsv;;
  zenwebp_lossy) FAMILY=zenwebp; CONFIG_MODULE=zenwebp_picker;       PARETO_STEM=zenwebp;            SRC=omni;      OMNI=zenwebp.zensim.combined.tsv;;
  zenavif_lossy) FAMILY=zenavif; CONFIG_MODULE=zenavif_picker;       PARETO_STEM=zenavif;            SRC=omni;      OMNI=zenavif.zensim.combined.tsv;;
  zenjxl_lossy)  FAMILY=zenjxl;  CONFIG_MODULE=zenjxl_lossy_dense;   PARETO_STEM=zenjxl_lossy_dense; SRC=canonical; OMNI=;;
  *) echo "FATAL unknown CODEC=$CODEC"; exit 2;;
esac
PICKER_TARGET="${PICKER_TARGET:-${METRIC}_a}"   # ssim2 -> ssim2_a (matches the shipped picker)
METRIC_COL="${METRIC_COL:-score_${METRIC}}"     # omni column -> pareto 'zensim' slot
RUN_PREFIX="loo-$DATE/${CODEC}_${METRIC}"
FEATURES_TSV_NAME=combined_features_vn_tiled.tsv
PARQ_LOCAL="$WORKDIR/${PARETO_STEM}.${PICKER_TARGET}.pareto.parquet"
FEAT_LOCAL="$WORKDIR/${PARETO_STEM}.features.tsv"

echo "### LOO fleet: codec=$CODEC family=$FAMILY config=$CONFIG_MODULE stem=$PARETO_STEM"
echo "    metric=$METRIC target=$PICKER_TARGET source=$SRC round=$ROUND smoke=$SMOKE"
echo "    run_prefix=s3://zentrain/$RUN_PREFIX  type=$STYPE image=$IMAGE"

set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
HCLOUD_TOKEN="$(grep -E '^api_token=' ~/.config/hetzner/credentials | head -1 | cut -d= -f2- | tr -d ' \r')"
export HCLOUD_TOKEN
[ -n "$HCLOUD_TOKEN" ] || { echo "FATAL: no hcloud api_token"; exit 1; }
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto aws s3 "$@" --endpoint-url "$EP"; }

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — build + upload the pareto (once)
# ══════════════════════════════════════════════════════════════════════════════
build_pareto(){
  if [ -s "$PARQ_LOCAL" ] && [ -s "$FEAT_LOCAL" ]; then
    echo "    pareto already built locally: $PARQ_LOCAL"; return 0
  fi
  if [ "$SRC" = omni ]; then
    echo "### building pareto via omni_to_pareto (--metric-col $METRIC_COL)"
    [ -f "$WORKDIR/$OMNI" ] || { echo "    pull omni $OMNI"; r2 cp "s3://zentrain/dualmodel-2026-06-28/inputs/$OMNI" "$WORKDIR/$OMNI" || return 1; }
    [ -f "$WORKDIR/$FEATURES_TSV_NAME" ] || r2 cp "s3://zentrain/dualmodel-2026-06-28/inputs/$FEATURES_TSV_NAME" "$WORKDIR/$FEATURES_TSV_NAME" || return 1
    nice -n 19 python3 "$REPO/scripts/picker/omni_to_pareto.py" \
      --omni "$WORKDIR/$OMNI" --features-tsv "$WORKDIR/$FEATURES_TSV_NAME" \
      --metric-col "$METRIC_COL" --out-pareto "$PARQ_LOCAL" --out-features "$FEAT_LOCAL" || return 1
  else
    echo "### building pareto from CANONICAL (jxl: no omni TSV) via column projection"
    build_canonical_pareto || return 1
  fi
}

# canonical → pareto adapter (jxl). Reads ONLY the cols we need (the 97 omni
# keep_features + cell/bytes/score) via pyarrow column projection — never the 469-feat
# full row. Output schema == omni_to_pareto's (image_path/size_class/width/height/
# config_id/config_name/q/bytes/zensim/encode_ms/total_ms/effective_max_zensim).
build_canonical_pareto(){
  local CANON_PREFIX="canonical/2026-06-27/$CODEC"
  local CDIR="$WORKDIR/canon_$CODEC"; mkdir -p "$CDIR"
  local f
  for f in train validate test; do
    [ -f "$CDIR/$f.parquet" ] || { echo "    pull canonical $f.parquet"; r2 cp "s3://zentrain/$CANON_PREFIX/$f.parquet" "$CDIR/$f.parquet" || return 1; }
  done
  [ -f "$WORKDIR/$FEATURES_TSV_NAME" ] || r2 cp "s3://zentrain/dualmodel-2026-06-28/inputs/$FEATURES_TSV_NAME" "$WORKDIR/$FEATURES_TSV_NAME" || return 1
  nice -n 19 python3 - "$CDIR" "$WORKDIR/$FEATURES_TSV_NAME" "$METRIC_COL" "$PARQ_LOCAL" "$FEAT_LOCAL" <<'PY' || return 1
import sys, csv, glob, os
import pyarrow.parquet as pq
import pandas as pd
cdir, feat_tsv, metric_col, out_parq, out_feat = sys.argv[1:6]
# the 97 features the omni-path pickers use (keep_features over the combined features TSV)
sys.path.insert(0, "/home/lilith/work/zen/zenmetrics/scripts/picker/configs")
import picker_config_common as cc
keep = cc.keep_features(feat_tsv)
need = ["variant_name", "cell", "encoded_bytes", "encode_ms", "width", "height", "q", metric_col]
parts = []
for fp in sorted(glob.glob(os.path.join(cdir, "*.parquet"))):
    schema = set(f.name for f in pq.ParquetFile(fp).schema_arrow)
    cols = [c for c in need if c in schema] + [k for k in keep if k in schema]
    parts.append(pq.read_table(fp, columns=cols).to_pandas())
df = pd.concat(parts, ignore_index=True)
df["image_path"] = df["variant_name"]
df["config_name"] = df["cell"].astype(str)
df["bytes"] = df["encoded_bytes"].astype("int64")
df["zensim"] = df[metric_col].astype("float64")
if "encode_ms" not in df: df["encode_ms"] = 0.0
df["total_ms"] = df["encode_ms"].astype("float64")
def size_class(px):
    return "tiny" if px <= 64*64 else "small" if px <= 256*256 else "medium" if px <= 1024*1024 else "large"
df["size_class"] = (df["width"]*df["height"]).map(size_class)
cfg_index = {c: i for i, c in enumerate(sorted(df["config_name"].unique()))}
df["config_id"] = df["config_name"].map(cfg_index).astype("int64")
df["effective_max_zensim"] = df.groupby(["variant_name","size_class"])["zensim"].transform("max")
pcols = ["image_path","size_class","width","height","config_id","config_name","q","bytes","zensim","encode_ms","total_ms","effective_max_zensim"]
import pyarrow as pa, pyarrow.parquet as pqw
pqw.write_table(pa.Table.from_pandas(df[pcols], preserve_index=False), out_parq)
fcols = [k for k in keep if k in df.columns]
feat = df[["variant_name","size_class","width","height",*fcols]].drop_duplicates(["variant_name","size_class"]).rename(columns={"variant_name":"image_path"})
feat.to_csv(out_feat, sep="\t", index=False)
print(f"canonical->pareto: {len(df)} rows, {len(cfg_index)} configs, {len(fcols)} feat cols, "
      f"sizes={sorted(df['size_class'].unique())}, zensim(={metric_col}) [{df['zensim'].min():.1f},{df['zensim'].max():.1f}]")
PY
}

if [ -n "${RELAUNCH_GROUP:-}" ]; then
  # RELAUNCH: recover specific boxes (e.g. one that lost a create race to capacity) into an
  # EXISTING group, reusing the pareto+batches already in R2 — NO prefix clear, no rebuild.
  echo "### RELAUNCH mode: group=$RELAUNCH_GROUP boxes=[${RELAUNCH_BOXES:?set RELAUNCH_BOXES=comma,ids}] — reuse R2 pareto+batches (NO clear)"
  LAUNCH_IDS="$(echo "$RELAUNCH_BOXES" | tr ',' ' ')"; N_BOXES="$(echo $LAUNCH_IDS | wc -w)"
else
build_pareto || { echo "FATAL: pareto build failed"; exit 1; }
echo "    pareto: $(du -h "$PARQ_LOCAL" | cut -f1)  features: $(du -h "$FEAT_LOCAL" | cut -f1)"
echo "### upload pareto + features to s3://zentrain/$RUN_PREFIX/inputs/"
r2 cp "$PARQ_LOCAL" "s3://zentrain/$RUN_PREFIX/inputs/${PARETO_STEM}.${PICKER_TARGET}.pareto.parquet" >/dev/null || { echo "FATAL upload pareto"; exit 1; }
r2 cp "$FEAT_LOCAL" "s3://zentrain/$RUN_PREFIX/inputs/${PARETO_STEM}.features.tsv" >/dev/null || { echo "FATAL upload features"; exit 1; }

# feature list (KEEP_FEATURES) from the features TSV — the LOO grid
mapfile -t FEATURES < <(PYTHONPATH="$REPO/scripts/picker/configs" python3 -c \
  "import picker_config_common as c;print('\n'.join(c.keep_features('$FEAT_LOCAL')))")
NFEAT="${#FEATURES[@]}"
[ "$NFEAT" -gt 0 ] || { echo "FATAL: empty feature list"; exit 1; }
echo "    KEEP_FEATURES: $NFEAT"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — generate jobs (round 1 single-drops, or round 2 group/pairwise)
# ══════════════════════════════════════════════════════════════════════════════
JOBS_FILE="$WORKDIR/jobs_${CODEC}_${METRIC}_r${ROUND}.jsonl"; : > "$JOBS_FILE"
jobline(){ printf '{"tag":"%s","drop":"%s"}\n' "$1" "$2" >> "$JOBS_FILE"; }

if [ "$ROUND" = 1 ]; then
  jobline baseline ""
  if [ "$SMOKE" = 1 ]; then
    # smoke: baseline + first 2 features only (validate the whole chain cheaply)
    jobline "${FEATURES[0]}" "${FEATURES[0]}"
    jobline "${FEATURES[1]}" "${FEATURES[1]}"
  else
    for feat in "${FEATURES[@]}"; do jobline "$feat" "$feat"; done
  fi
else
  # ROUND 2 — pair-aware verification. Needs the round-1 collected TSV locally.
  R1_TSV="$WORKDIR/collected_${CODEC}_${METRIC}/loo_${CODEC}_${METRIC}.tsv"
  [ -s "$R1_TSV" ] || { echo "FATAL: round-2 needs round-1 results at $R1_TSV (run ROUND=1 + collect first)"; exit 1; }
  KEEP_TH="${KEEP_THRESHOLD_PP:-0.05}"
  # generate group-drop (all droppable-looking), bisection subsets, and top-correlated
  # pairwise drops among the droppable set (the redundancy suspects). Correlation comes
  # from the local features TSV (cheap). Emitted as drop=csv jobs.
  python3 - "$R1_TSV" "$FEAT_LOCAL" "$KEEP_TH" "$JOBS_FILE" <<'PY'
import sys, csv, json, itertools
r1_tsv, feat_tsv, keep_th, jobs_file = sys.argv[1:5]
keep_th = float(keep_th)
droppable, must_keep = [], []
with open(r1_tsv) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        d = row["val_delta_pp"]
        if d == "NA": continue
        (droppable if float(d) <= keep_th else must_keep).append(row["feature"])
jobs = [("baseline", "")]
if droppable:
    jobs.append(("group_all_droppable", ",".join(droppable)))
    # bisection: halves + quarters of the droppable set → find the largest safe subset
    def chunks(lst, n):
        k = (len(lst) + n - 1)//n
        return [lst[i:i+k] for i in range(0, len(lst), k)]
    for parts, lbl in ((chunks(droppable,2),"half"), (chunks(droppable,4),"quarter")):
        for i, p in enumerate(parts):
            if p: jobs.append((f"group_{lbl}{i}", ",".join(p)))
    # top-correlated pairwise drops among droppable (redundancy suspects)
    try:
        import pandas as pd
        df = pd.read_csv(feat_tsv, sep="\t")
        cols = [c for c in droppable if c in df.columns]
        if len(cols) >= 2:
            corr = df[cols].corr().abs()
            pairs = []
            for a, b in itertools.combinations(cols, 2):
                pairs.append((corr.loc[a, b], a, b))
            pairs.sort(reverse=True)
            for c, a, b in pairs[:20]:
                jobs.append((f"pair_{a}__{b}", f"{a},{b}"))
    except Exception as e:
        print(f"  WARN pairwise gen skipped: {e}")
with open(jobs_file, "w") as f:
    for tag, drop in jobs:
        f.write(json.dumps({"tag": tag, "drop": drop}) + "\n")
print(f"round2 jobs: {len(jobs)} (droppable={len(droppable)} must_keep={len(must_keep)})")
PY
fi
NJOBS="$(grep -c . "$JOBS_FILE")"
echo "    jobs (round $ROUND): $NJOBS"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — size the fleet + split into balanced batches
# ══════════════════════════════════════════════════════════════════════════════
if [ "$SMOKE" = 1 ]; then
  N_BOXES=1
else
  N_BOXES="${N_BOXES:-6}"
fi
# never more boxes than jobs
[ "$N_BOXES" -gt "$NJOBS" ] && N_BOXES="$NJOBS"
echo "### N_BOXES=$N_BOXES  (~$(( (NJOBS + N_BOXES - 1) / N_BOXES )) variants/box)"

# clear any stale batches/markers/results from a prior run in this prefix
r2 rm "s3://zentrain/$RUN_PREFIX/batches/"  --recursive >/dev/null 2>&1 || true
r2 rm "s3://zentrain/$RUN_PREFIX/markers/"  --recursive >/dev/null 2>&1 || true
r2 rm "s3://zentrain/$RUN_PREFIX/results/"  --recursive >/dev/null 2>&1 || true

# round-robin assignment → balanced load (every variant costs ~the same T) so boxes
# finish together (no straggler idling). batch i = jobs[i], jobs[i+N], jobs[i+2N], ...
BATCHDIR="$WORKDIR/batches_${CODEC}_${METRIC}_r${ROUND}"; rm -rf "$BATCHDIR"; mkdir -p "$BATCHDIR"
awk -v n="$N_BOXES" -v d="$BATCHDIR" 'NF{print >> (d"/box-" (NR-1)%n ".jsonl")}' "$JOBS_FILE"
for i in $(seq 0 $((N_BOXES-1))); do
  [ -f "$BATCHDIR/box-$i.jsonl" ] || : > "$BATCHDIR/box-$i.jsonl"
  r2 cp "$BATCHDIR/box-$i.jsonl" "s3://zentrain/$RUN_PREFIX/batches/box-$i.jsonl" >/dev/null
done
echo "    uploaded $N_BOXES batch file(s)"
LAUNCH_IDS="$(seq 0 $((N_BOXES-1)))"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — scoped R2 cred + per-box cloud-init (embedded runner + self-destruct)
# ══════════════════════════════════════════════════════════════════════════════
TTL=$((MAXMIN*60+1800))
body=$(python3 -c "import json,os;print(json.dumps({
  'bucket':'zentrain',
  'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],
  'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],
  'permission':'object-read-write','ttlSeconds':$TTL,
  'prefixes':['$RUN_PREFIX/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/loo_cred.json
read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/loo_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])' 2>/dev/null)
[ -n "${AK:-}" ] || { echo "FATAL: R2 cred mint failed"; cat /tmp/loo_cred.json; exit 1; }
echo "    minted scoped RW cred (ttl ${MAXMIN}m+30m) -> zentrain/$RUN_PREFIX/"

# Separate scoped temp R2 cred for the PRIVATE zenfleet-logs bucket — the
# per-box host ci.log/ci.host.box-$i.log upload is the only thing that goes
# here (owner directive 2026-07-05: host logs move off the public zentrain
# bucket; batches/markers/results stay on zentrain). A separate mint is
# required because R2 temp-access-credentials are scoped to exactly one
# bucket — the zentrain-scoped cred above cannot write to a different bucket.
logbody=$(python3 -c "import json,os;print(json.dumps({
  'bucket':'zenfleet-logs',
  'parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],
  'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],
  'permission':'object-read-write','ttlSeconds':$TTL,
  'prefixes':['$RUN_PREFIX/']}))")
curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$logbody" \
  "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/loo_log_cred.json
read -r LOG_AK LOG_SK LOG_ST < <(python3 -c 'import json;r=json.load(open("/tmp/loo_log_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])' 2>/dev/null)
[ -n "${LOG_AK:-}" ] || { echo "FATAL: R2 log-bucket cred mint failed"; cat /tmp/loo_log_cred.json; exit 1; }
echo "    minted scoped RW cred (ttl ${MAXMIN}m+30m) -> zenfleet-logs/$RUN_PREFIX/"

RUNNER_B64="$(base64 -w0 "$REPO/scripts/train/loo_box_runner.sh")"
# Bind-mount CURRENT-master picker configs over the (possibly stale) baked /opt/picker/configs
# so the box's KEEP_FEATURES matches what the launcher computed (the 2026-06-28 image predates
# the 51->97 keep_features expansion → would silently no-op the extra-feature drops). Configs
# are committed master code, unmodified — not a logic change, just version alignment.
CONFIGS_B64="$(tar czf - --exclude='__pycache__' --exclude='*.bak' --exclude='*.pre-expand-bak' \
  -C "$REPO/scripts/picker" configs | base64 -w0)"
RUN="${RELAUNCH_GROUP:-loo-$(echo "${CODEC}-${METRIC}-r${ROUND}" | tr '_' '-')-$(date +%s)}"
MON_LOG="/tmp/loo_fleet_${RUN}.log"

launch_box(){
  local i="$1" NAME="$RUN-box$i"
  local CI; CI="$(mktemp)"
  cat > "$CI" <<EOF
#!/bin/bash
exec > /root/ci.log 2>&1
set -x
set +x   # trace OFF across the secret assignment — xtrace would print the literal
         # token into ci.log, which is uploaded to R2 below (security hole, fixed
         # 2026-07-05; token already rotated).
HCLOUD_TOKEN='$HCLOUD_TOKEN'          # HOST-only (self-destruct); NEVER passed to container
set -x
EP='$EP'; IMG='$IMAGE'; OUTP='$RUN_PREFIX'
echo '$RUNNER_B64' | base64 -d > /root/loo_box_runner.sh
chmod +x /root/loo_box_runner.sh
mkdir -p /root/pc && echo '$CONFIGS_B64' | base64 -d | tar xzf - -C /root/pc   # current-master picker configs
cat > /root/cenv <<ENV
R2_ENDPOINT=$EP
RUN_BUCKET=zentrain
RUN_PREFIX=$RUN_PREFIX
BOX_ID=$i
CONFIG_MODULE=$CONFIG_MODULE
PARETO_STEM=$PARETO_STEM
PICKER_TARGET=$PICKER_TARGET
METRIC_TAG=$METRIC
SEED=$SEED
HIDDEN=$HIDDEN
PER_RUN_TIMEOUT=$PER_RUN_TIMEOUT
AWS_ACCESS_KEY_ID=$AK
AWS_SECRET_ACCESS_KEY=$SK
AWS_SESSION_TOKEN=$ST
AWS_REGION=auto
ENV
cat > /root/logenv <<ENV
AWS_ACCESS_KEY_ID=$LOG_AK
AWS_SECRET_ACCESS_KEY=$LOG_SK
AWS_SESSION_TOKEN=$LOG_ST
AWS_REGION=auto
ENV
destroy_self(){
  set +x   # trace OFF for the whole function — the DELETE call below carries the
           # token in an Authorization header; xtrace would print it into ci.log.
  local ID
  ID=\$(curl -s --max-time 10 http://169.254.169.254/hetzner/v1/metadata/instance-id || true)
  for x in 1 2 3 4 5; do
    curl -s --max-time 20 -X DELETE -H "Authorization: Bearer \$HCLOUD_TOKEN" \
      "https://api.hetzner.cloud/v1/servers/\$ID" && break
    sleep 5
  done
  set -x
}
( sleep $((MAXMIN*60)); echo "BACKSTOP timeout"; destroy_self ) &     # hard backstop
docker pull "\$IMG" || true
docker run --rm --env-file /root/cenv \
  -v /root/loo_box_runner.sh:/usr/local/bin/loo_box_runner.sh \
  -v /root/pc/configs:/opt/picker/configs \
  --entrypoint bash "\$IMG" /usr/local/bin/loo_box_runner.sh
rc=\$?
echo "container exited rc=\$rc"
# Host log goes to the PRIVATE zenfleet-logs bucket (owner directive
# 2026-07-05) — uses the /root/logenv cred minted for that bucket, NOT
# /root/cenv (zentrain).
docker run --rm --env-file /root/logenv -v /root/ci.log:/ci.log --entrypoint /usr/local/bin/s5cmd \
  "\$IMG" --endpoint-url="\$EP" cp /ci.log "s3://zenfleet-logs/\$OUTP/logs/ci.host.box-$i.log" || true
destroy_self
EOF
  local launched=0 lasterr="" typ loc
  for typ in "$STYPE" ccx33 cpx51 ccx43; do   # cheapest-adequate (32GB) first; cpx41(16GB) omitted (RAM risk)
    for loc in fsn1 nbg1 hel1; do
      lasterr=$(hcloud server create --name "$NAME" --type "$typ" --image docker-ce --location "$loc" \
        --ssh-key "$SSH_KEY" --label group="$RUN" --label codec="$CODEC" \
        --user-data-from-file "$CI" 2>&1) \
        && { echo "  launched $NAME ($typ/$loc)"; launched=1; ACTUAL_TYPE="$typ"; break 2; } || true
    done
  done
  rm -f "$CI"
  [ "$launched" = 1 ] || { echo "  FAIL launch $NAME: $(echo "$lasterr" | tail -1)"; return 1; }
}

echo "### launching $N_BOXES EU box(es) [group=$RUN]  (parallel creates)"
ACTUAL_TYPE="$STYPE"
# `hcloud server create` BLOCKS ~1-2 min until the box provisions; launching 10 boxes
# serially wastes ~20 min. Background the creates (2s stagger to be gentle on the API)
# and wait — all provision in parallel (~2 min total). ACTUAL_TYPE is set racily by each
# (cosmetic: only the price display / burn estimate read it; all boxes are ~same type).
for i in $LAUNCH_IDS; do launch_box "$i" & sleep 2; done
wait
PRICE=$(hcloud server-type describe "$ACTUAL_TYPE" -o json 2>/dev/null | python3 - <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(sys.stdin)
    print("%.4f" % float(d["prices"][0]["price_hourly"]["gross"]))
except Exception:
    print("")
PY
)
# robust numeric fallback (EU shared/dedicated approx) so the €-watchdog math never breaks
case "$PRICE" in ""|*[!0-9.]*) case "$ACTUAL_TYPE" in cpx51) PRICE=0.4479;; cpx41) PRICE=0.2267;; ccx33) PRICE=0.2660;; ccx43) PRICE=0.5300;; *) PRICE=0.30;; esac;; esac
echo "    type=$ACTUAL_TYPE ~EUR ${PRICE}/hr/box  (fleet cap EUR $MAX_BURN_EUR / ${MAXMIN}m/box)"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — monitor: tail progress, €-watchdog, teardown backstop, collect
# ══════════════════════════════════════════════════════════════════════════════
# Teardown helper — SCOPED: list ONLY servers carrying this run's unique group label,
# then delete them by explicit id (delete one-by-one). `hcloud server delete` has NO
# label selector (-l is list-only), and a blanket/prefix delete could catch a pre-existing
# box — so we resolve ids from the unique label and delete exactly those. Never touches
# servers without group=$RUN (the persistent zen-arm-* / zen-fuzz-* boxes are safe).
kill_group(){
  local ids
  ids=$(HCLOUD_TOKEN="$HCLOUD_TOKEN" hcloud server list -l group="$RUN" -o noheader 2>/dev/null | awk '{print $1}')
  [ -n "$ids" ] && echo "$ids" | xargs -r -n1 -I{} env HCLOUD_TOKEN="$HCLOUD_TOKEN" hcloud server delete {} 2>/dev/null || true
}
COLLECT_DIR="$WORKDIR/collected_${CODEC}_${METRIC}"
# In RELAUNCH mode the group is SHARED with the original fleet's still-running boxes, so this
# monitor's group-teardown (kill_group) must NOT run — it would delete the in-flight boxes.
# The relaunched box self-destructs via cloud-init on done; the original fleet monitor + the
# external monitor own group teardown. So skip the monitor entirely when relaunching.
if [ -z "${RELAUNCH_GROUP:-}" ]; then
(
  start=$(date +%s)
  echo "[mon] group=$RUN boxes=$N_BOXES type=$ACTUAL_TYPE price=$PRICE/hr launched $(date -u +%FT%TZ)"
  while :; do
    now=$(date +%s); el=$(( (now-start)/60 ))
    alive=$(HCLOUD_TOKEN="$HCLOUD_TOKEN" hcloud server list -l group="$RUN" -o noheader 2>/dev/null | wc -l)
    done_n=$(r2 ls "s3://zentrain/$RUN_PREFIX/markers/" 2>/dev/null | grep -c _DONE); done_n=${done_n:-0}
    fail_n=$(r2 ls "s3://zentrain/$RUN_PREFIX/markers/" 2>/dev/null | grep -c _FAILED); fail_n=${fail_n:-0}
    res_n=$(r2 ls "s3://zentrain/$RUN_PREFIX/results/" 2>/dev/null | grep -c 'box-.*json'); res_n=${res_n:-0}
    burn=$(python3 -c "print(f'{$alive*$PRICE*$el/60:.2f}')" 2>/dev/null || echo "?")
    echo "[mon +${el}m] alive=$alive done=$done_n failed=$fail_n results=$res_n  est_burn=EUR${burn}"
    # all boxes reported a terminal marker, or all gone → finish
    if [ $((done_n + fail_n)) -ge "$N_BOXES" ] || { [ "$alive" = 0 ] && [ "$el" -ge 3 ]; }; then
      echo "[mon] terminal: done=$done_n failed=$fail_n alive=$alive — tearing down any stragglers"
      kill_group
      break
    fi
    # €-watchdog
    if python3 -c "import sys; sys.exit(0 if $alive*$PRICE*$el/60 > $MAX_BURN_EUR else 1)" 2>/dev/null; then
      echo "[mon] €-CAP EUR$MAX_BURN_EUR exceeded (est EUR$burn) — KILLING fleet"
      kill_group
      break
    fi
    if [ "$el" -ge "$MAXMIN" ]; then
      echo "[mon] MAXMIN ${MAXMIN}m — force-killing fleet"
      kill_group
      break
    fi
    sleep 30
  done
  # final teardown verification — 0 boxes must remain
  sleep 8
  left=$(HCLOUD_TOKEN="$HCLOUD_TOKEN" hcloud server list -l group="$RUN" -o noheader 2>/dev/null | wc -l)
  echo "[mon] boxes remaining in group=$RUN: $left"
  [ "$left" -gt 0 ] && { echo "[mon] FORCE delete remaining"; kill_group; }
  # collect results → ranked TSV
  echo "[mon] collecting results -> $COLLECT_DIR"
  rm -rf "$COLLECT_DIR"; mkdir -p "$COLLECT_DIR/results"
  r2 cp "s3://zentrain/$RUN_PREFIX/results/" "$COLLECT_DIR/results/" --recursive >/dev/null 2>&1 || true
  ngot=$(ls "$COLLECT_DIR/results/"box-*.json 2>/dev/null | wc -l)
  echo "[mon] downloaded $ngot box result JSON(s)"
  if [ "$ngot" -gt 0 ]; then
    python3 "$REPO/scripts/train/loo_collect.py" --results-dir "$COLLECT_DIR/results" \
      --codec "$CODEC" --metric "$METRIC" --out-dir "$COLLECT_DIR" 2>&1 || echo "[mon] collect failed"
    # persist the ranked TSV to the repo benchmarks dir (committable)
    if [ -s "$COLLECT_DIR/loo_${CODEC}_${METRIC}.tsv" ]; then
      cp "$COLLECT_DIR/loo_${CODEC}_${METRIC}.tsv" "$REPO/benchmarks/loo_${CODEC}_${METRIC}_${DATE}.tsv" 2>/dev/null || true
      [ -s "$COLLECT_DIR/loo_${CODEC}_${METRIC}_summary.md" ] && cp "$COLLECT_DIR/loo_${CODEC}_${METRIC}_summary.md" "$REPO/benchmarks/loo_${CODEC}_${METRIC}_${DATE}.summary.md" 2>/dev/null || true
    fi
  fi
  echo "[mon] === DONE group=$RUN at $(date -u +%FT%TZ); results s3://zentrain/$RUN_PREFIX/results/ ==="
) > "$MON_LOG" 2>&1 &
MONPID=$!
echo "### monitor PID=$MONPID  ->  tail -f $MON_LOG"
echo "### teardown (manual if needed): hcloud server list -l group=$RUN -o noheader | awk '{print \$1}' | xargs -r -n1 hcloud server delete"
else
  echo "### RELAUNCH: relaunched box(es) [$LAUNCH_IDS] self-destruct on done; NO group-teardown monitor here (original fleet still running in group=$RUN)"
fi
echo "$RUN" > "$WORKDIR/.last_run_group"
