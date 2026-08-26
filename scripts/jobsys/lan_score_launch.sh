#!/usr/bin/env bash
# lan_score_launch.sh — launch ONE LAN fleet box as a zenfleet score/feature worker
# against a declared job set, in single-run (direct-manifest) mode.
#
# The existing GPU-score launchers (gpu_scorefile_launch.sh, gpu_e2e_proof.sh) are
# VAST.AI-oriented (`vastai create instance`). Since the fleet moved to the LAN
# (2026-08-25), score jobs run on the household GPU/CPU boxes via a direct docker
# worker. This is that launcher — the LAN counterpart, one box per call.
#
# Usage:
#   lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]
#     host     ssh host (r7900x, lianli, i265, ...); creds read from the box's
#              own ~/.config/cloudflare/r2-credentials (distribute first).
#     job-set  the run prefix under s3://zentrain/jobs/ (e.g. hdrgrid-sf-gpu-20260807)
#     role     a short label for THIS worker's run (small/huge/medium/cpu/...)
#     gpu|cpu  gpu (default): --gpus all + ZEN_REQUIRE_GPU=1 + the CUDA exec image.
#              cpu: no GPU, the CPU exec image.
#     image    override the docker image (else the role default).
#
# The worker name is derived from the REMOTE hostname + role
# (`<hostname>-<role>`), so two boxes NEVER collide on ZEN_WORKER (the 2026-08-26
# heredoc-interpolation bug that gave both 6 GB boxes `lianli-hdr`). Container is
# `zen-score-<role>`, `--restart unless-stopped`; it self-exits on drain
# (ZEN_IDLE_PASSES) and is torn down with `docker rm -f zen-score-<role>`.
set -uo pipefail

HOST="${1:?usage: lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]}"
JOBSET="${2:?job-set (run prefix under s3://zentrain/jobs/)}"
ROLE="${3:?role label (small/huge/medium/cpu/...)}"
KIND="${4:-gpu}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"

case "$KIND" in
  gpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-gpu-avifgen-66e3c417"
       GPU_ARGS="--gpus all"; REQ_GPU="-e ZEN_REQUIRE_GPU=1" ;;
  cpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-zensim944-57b7b9ad"
       GPU_ARGS=""; REQ_GPU="" ;;
  *) echo "lan_score_launch: KIND must be gpu|cpu (got '$KIND')" >&2; exit 2 ;;
esac
IMG="${5:-$DEF_IMG}"
CTR="zen-score-${ROLE}"

# Config is passed through the environment; creds are read on the REMOTE box (they
# were distributed to ~/.config/cloudflare/r2-credentials), never sent over the wire.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  ZM_JOBSET="$JOBSET" ZM_BUCKET="$BUCKET" ZM_ROLE="$ROLE" ZM_CTR="$CTR" \
  ZM_IMG="$IMG" ZM_GPU_ARGS="$GPU_ARGS" ZM_REQ_GPU="$REQ_GPU" 'bash -s' <<'REMOTE'
set -euo pipefail
CREDS="$HOME/.config/cloudflare/r2-credentials"
[ -r "$CREDS" ] || { echo "no R2 creds at $CREDS on $(hostname) — distribute first" >&2; exit 3; }
set -a; . "$CREDS"; set +a
: "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing from creds}"
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
WORKER="$(hostname)-${ZM_ROLE}"   # collision-proof: hostname is unique per box

sudo -n docker pull "$ZM_IMG" >/dev/null 2>&1 || true
sudo -n docker rm -f "$ZM_CTR" >/dev/null 2>&1 || true
# shellcheck disable=SC2086  # ZM_GPU_ARGS / ZM_REQ_GPU are intentional word-splits
sudo -n docker run -d --name "$ZM_CTR" --restart unless-stopped $ZM_GPU_ARGS \
  -e AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT="$EP" -e ZEN_BUCKET="$ZM_BUCKET" \
  -e ZEN_RUN="jobs/$ZM_JOBSET" \
  -e ZEN_MANIFEST_URI="s3://$ZM_BUCKET/jobs/$ZM_JOBSET/manifest.json" \
  -e ZEN_CONTROL_KEY="jobs/$ZM_JOBSET/control.json" \
  $ZM_REQ_GPU -e ZEN_WORKER="$WORKER" -e ZEN_PROVIDER=basement \
  -e ZEN_MAX_MIN=1400 -e ZEN_IDLE_PASSES=8 -e ZEN_CORE_OVERSUBSCRIBE=2 \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh "$ZM_IMG" >/dev/null
echo "$(hostname) -> $ZM_JOBSET as $WORKER : $(sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep "$ZM_CTR" || echo FAILED-TO-START)"
REMOTE
