#!/usr/bin/env bash
# _lan_gpu_seq_driver.sh — the REMOTE loop for lan_gpu_sequence.sh. Runs ON a fleet
# box (scp'd there, launched under setsid). Drains each bucket ($@) sequentially in
# blocking single-run mode. Env: ZM_KIND (gpu|cpu), ZM_IMG, ZM_S3BUCKET.
# Do not invoke directly — use lan_gpu_sequence.sh.
set -uo pipefail

CREDS="$HOME/.config/cloudflare/r2-credentials"
[ -r "$CREDS" ] || { echo "no R2 creds at $CREDS on $(hostname)" >&2; exit 3; }
set -a; . "$CREDS"; set +a
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing from creds}"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

LOG="$HOME/lan_gpu_seq.log"; : > "$LOG"
rm -f "$HOME/lan_gpu_seq.COMPLETE"
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

GPU_ARGS=(); REQ=()
if [ "${ZM_KIND:-gpu}" = gpu ]; then GPU_ARGS=(--gpus all); REQ=(-e ZEN_REQUIRE_GPU=1); fi
sudo -n docker pull "$ZM_IMG" >/dev/null 2>&1 || true

log "sequencer start on $(hostname): $# buckets ($*)"
for js in "$@"; do
  case "$js" in *-small-*) role=small;; *-huge-*) role=huge;; *) role=med;; esac
  ctr="zen-seq-$role"
  log "START $js (role=$role, worker=$(hostname)-$role)"
  sudo -n docker rm -f "$ctr" >/dev/null 2>&1 || true
  # blocking run (no -d): returns when the worker self-exits on drain (idle passes)
  sudo -n docker run --rm --name "$ctr" "${GPU_ARGS[@]}" \
    -e AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" -e AWS_REGION=auto \
    -e ZEN_R2_ENDPOINT="$EP" -e ZEN_BUCKET="${ZM_S3BUCKET:-zentrain}" \
    -e ZEN_RUN="jobs/$js" \
    -e ZEN_MANIFEST_URI="s3://${ZM_S3BUCKET:-zentrain}/jobs/$js/manifest.json" \
    -e ZEN_CONTROL_KEY="jobs/$js/control.json" \
    "${REQ[@]}" -e ZEN_WORKER="$(hostname)-$role" -e ZEN_PROVIDER=basement \
    -e ZEN_MAX_MIN=1400 -e ZEN_IDLE_PASSES=8 -e ZEN_CORE_OVERSUBSCRIBE=2 \
    --entrypoint /usr/local/bin/fleet-entrypoint.sh "$ZM_IMG" >> "$LOG" 2>&1
  log "DONE $js rc=$?"
done
log "ALL BUCKETS DRAINED"
echo "ALL BUCKETS DRAINED $(date -u +%FT%TZ)" > "$HOME/lan_gpu_seq.COMPLETE"
