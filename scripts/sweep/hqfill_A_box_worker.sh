#!/usr/bin/env bash
# hqfill-A box worker — runs ON a vast GPU box (uploaded to R2 by hqfill_A_remote.sh,
# fetched by the box onstart). Committed standalone (NOT a heredoc) to avoid the
# nested-heredoc escaping traps noted in scripts/sweep/CLAUDE.md.
#
# Loops: atomic-claim a chunk (R2 If-None-Match) -> pull its renditions from R2 ->
# `zenmetrics sweep --knob-grid {distance...} --feature-output --encoded-out-dir`
# (THE tool that made the salvage; GPU metrics + CPU ssim2/dssim + profile-A zensim,
# --use-legacy-scheduler to dodge the cubecl orchestrator warm-bench descriptor race)
# -> upload TSV + 372-feat parquet + variants tar + DONE. Self-destroys at the end.
#
# Env (vast injects into /proc/1/environ; onstart also sets them): ZEN_R2_ENDPOINT
# ZEN_BUCKET ZEN_RUN_PREFIX ZEN_CORPUS_PREFIX ZEN_AK ZEN_SK ZEN_ST ZEN_DIST ZEN_METRICS ZEN_BOX
set -uo pipefail
if [[ -r /proc/1/environ ]]; then
  while IFS='=' read -r -d '' k v; do
    case "$k" in R2_*|ZEN_*|AWS_*|CONTAINER_*) export "$k=$v";; esac
  done < /proc/1/environ
fi
EP="${ZEN_R2_ENDPOINT}"; BUCKET="${ZEN_BUCKET}"; PRE="${ZEN_RUN_PREFIX}"; CORPUS="${ZEN_CORPUS_PREFIX}"
export AWS_ACCESS_KEY_ID="${ZEN_AK}" AWS_SECRET_ACCESS_KEY="${ZEN_SK}" AWS_SESSION_TOKEN="${ZEN_ST}" AWS_REGION=auto
DIST="${ZEN_DIST}"; METRICS="${ZEN_METRICS//,/ }"; BOX="${ZEN_BOX:-0}"
s5(){ s5cmd --endpoint-url "$EP" "$@"; }
LOG=/var/log/hqa_worker.log
exec > >(tee -a "$LOG") 2>&1
trap 's5 cp "$LOG" "s3://$BUCKET/$PRE/logs/worker-$BOX.log" 2>/dev/null || true' EXIT

echo "=== hqfill-A box worker $BOX starting $(date -u +%FT%TZ) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1 || echo "no nvidia-smi"
command -v zenmetrics >/dev/null || { echo "FATAL: zenmetrics not baked"; exit 1; }
command -v s5cmd >/dev/null || { echo "FATAL: s5cmd not baked"; exit 1; }

MFLAGS=""; for m in $METRICS; do MFLAGS="$MFLAGS --metric $m"; done

process_chunk(){
  local ci="$1"
  echo "$BOX $(date -u +%FT%TZ)" > /tmp/claim.txt
  # atomic claim (exactly-once): If-None-Match fails if another box already claimed
  if ! s5 cp --if-none-match /tmp/claim.txt "s3://$BUCKET/$PRE/claims/$ci.claim" >/dev/null 2>&1; then
    return 2
  fi
  # idempotent: skip if a DONE already exists (survives re-runs / new passes)
  if s5 ls "s3://$BUCKET/$PRE/done/$ci.done" >/dev/null 2>&1; then
    echo "chunk $ci already done — skip"; return 3
  fi
  echo "--- chunk $ci: claimed, fetching renditions ---"
  rm -rf /data/src /enc; mkdir -p /data/src /enc
  s5 cp "s3://$BUCKET/$PRE/chunks/$ci.txt" /data/chunk.txt
  local n=0
  while read -r f; do
    [ -n "$f" ] && s5 cp "s3://$BUCKET/$CORPUS/$f" "/data/src/$f" && n=$((n+1))
  done < /data/chunk.txt
  echo "chunk $ci: $n renditions fetched, running sweep (14 distances x $n = $((n*14)) cells)"
  # THE salvage-consistent invocation. --use-legacy-scheduler dodges the cubecl
  # orchestrator warm-bench descriptor race on a real card; same kernels => same scores.
  ZENSIM_FEATURES_REGIME=with-iw zenmetrics sweep \
    --codec zenjxl --sources /data/src --q-grid 90 \
    --knob-grid "{\"distance\":[$DIST],\"effort\":[7]}" \
    $MFLAGS --zensim-features-regime with-iw \
    --feature-output "/data/$ci.features.parquet" \
    --encoded-out-dir /enc \
    --output "/data/$ci.tsv" \
    --gpu-runtime cuda --use-legacy-scheduler --jobs 1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "chunk $ci: sweep FAILED rc=$rc"; return 1; fi
  local ncells nfeat
  ncells=$(( $(wc -l < "/data/$ci.tsv") - 1 ))
  nfeat=$(python3 -c "import pyarrow.parquet as pq;print(pq.read_metadata('/data/$ci.features.parquet').num_rows)" 2>/dev/null || echo 0)
  if [ "$ncells" != "$nfeat" ]; then echo "chunk $ci: GATE-FAIL cells=$ncells feat=$nfeat"; return 1; fi
  s5 cp "/data/$ci.tsv" "s3://$BUCKET/$PRE/tsv/$ci.tsv"
  s5 cp "/data/$ci.features.parquet" "s3://$BUCKET/$PRE/features/$ci.features.parquet"
  tar -cf "/data/$ci.enc.tar" -C /enc . && s5 cp "/data/$ci.enc.tar" "s3://$BUCKET/$PRE/variants/$ci.tar"
  printf 'cells=%s feat=%s rc=%s box=%s\n' "$ncells" "$nfeat" "$rc" "$BOX" > "/data/$ci.done"
  s5 cp "/data/$ci.done" "s3://$BUCKET/$PRE/done/$ci.done"
  rm -f "/data/$ci.tsv" "/data/$ci.features.parquet" "/data/$ci.enc.tar"
  echo "chunk $ci: OK cells=$ncells feat=$nfeat"
  return 0
}

did=0; passes=0
# Multi-pass: re-iterate the chunk list; claim+run any not-yet-claimed/done chunk.
# Converges even if boxes join/leave (content-addressed done markers gate re-work).
while : ; do
  progressed=0
  for ci in $(s5 ls "s3://$BUCKET/$PRE/chunks/" | awk '{print $NF}' | sed 's/\.txt$//'); do
    process_chunk "$ci"; r=$?
    if [ $r -eq 0 ]; then did=$((did+1)); progressed=1; fi
  done
  passes=$((passes+1))
  # stop when a full pass claimed nothing new (all chunks claimed/done) or after a safety cap
  [ $progressed -eq 0 ] && break
  [ $passes -ge 50 ] && break
done
echo "=== box $BOX done: processed $did chunks over $passes passes ==="

# best-effort self-destroy on success (run_with_error_trap covers the failure path)
if [ -n "${CONTAINER_ID:-}" ] && command -v zenfleet-vastai >/dev/null 2>&1; then
  echo "self-destroying box (CONTAINER_ID=$CONTAINER_ID)"
  zenfleet-vastai self-destroy 2>/dev/null || true
fi
