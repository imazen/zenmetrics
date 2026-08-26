#!/usr/bin/env bash
# lan_score_launch.sh — launch ONE LAN fleet box as a zenfleet score/feature worker
# against a declared job set, in single-run (direct-manifest) mode.
#
# The existing GPU-score launchers (gpu_scorefile_launch.sh, gpu_e2e_proof.sh) are
# VAST.AI-oriented (`vastai create instance`). Since the fleet moved to the LAN
# (2026-08-25), score jobs run on the household GPU/CPU boxes via a direct docker
# worker. This is that launcher — the LAN counterpart, one box per call.
#
# STORE (2026-08-26): writes go to the LAN store — tower SeaweedFS — BY DEFAULT
# (`ZEN_STORE=tower`, the migration target; R2 is being retired for ops cost). The
# box reads its store creds locally: LAN store from ~/.config/zen/lanstore.env,
# R2 from ~/.config/cloudflare/r2-credentials. `ZEN_STORE=r2` is the EXPLICIT
# opt-out, for working the legacy R2-hosted jobs only. This resolver mirrors
# scripts/lib/s3env.sh. NOTE: the worker's `--r2-endpoint` / `ZEN_R2_ENDPOINT` is
# misnamed — it is just "the S3 endpoint"; the worker reads and writes there
# regardless of whether it points at R2 or SeaweedFS.
#
# Usage:
#   lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]
#     host     ssh host (lilith, r5900xt, i265, r5600g, node-2, tower, ...)
#     job-set  the run prefix under s3://<bucket>/jobs/ (e.g. hdrgrid-sf2-cpu-20260807)
#     role     a short label for THIS worker's run (small/huge/medium/cpu/...)
#     gpu|cpu  gpu (default): --gpus all + ZEN_REQUIRE_GPU=1 + the CUDA exec image.
#              cpu: no GPU, the CPU exec image.
#     image    override the docker image (else the role default).
#   env: ZEN_STORE=tower|r2 (default tower), ZEN_FLEET_BUCKET (default zentrain).
#
# The worker name is derived from the REMOTE hostname + role (`<hostname>-<role>`),
# so two boxes NEVER collide on ZEN_WORKER. Container is `zen-score-<role>`,
# `--restart unless-stopped`; it self-exits on drain (ZEN_IDLE_PASSES) and is torn
# down with `docker rm -f zen-score-<role>`.
set -uo pipefail

HOST="${1:?usage: lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]}"
JOBSET="${2:?job-set (run prefix under s3://<bucket>/jobs/)}"
ROLE="${3:?role label (small/huge/medium/cpu/...)}"
KIND="${4:-gpu}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"
STORE="${ZEN_STORE:-tower}"   # LAN SeaweedFS by default; 'r2' selects legacy R2

case "$KIND" in
  gpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-gpu-avifgen-66e3c417" ;;
  cpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-zensim944-57b7b9ad" ;;
  *) echo "lan_score_launch: KIND must be gpu|cpu (got '$KIND')" >&2; exit 2 ;;
esac
IMG="${5:-$DEF_IMG}"
CTR="zen-score-${ROLE}"

# Config is passed through the environment; store creds are read on the REMOTE box
# (distributed to ~/.config/zen/lanstore.env for the LAN store, or
# ~/.config/cloudflare/r2-credentials for R2), never sent over the wire.
# NOTE: every passed value is SPACE-FREE — ssh joins `VAR=val` args into the remote
# command line, so a space would split into a bogus command (the 2026-08-26
# `--gpus all` → "all: command not found" bug). GPU flags are rebuilt on the remote.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  ZM_JOBSET="$JOBSET" ZM_BUCKET="$BUCKET" ZM_ROLE="$ROLE" ZM_CTR="$CTR" \
  ZM_IMG="$IMG" ZM_KIND="$KIND" ZM_STORE="$STORE" 'bash -s' <<'REMOTE'
set -euo pipefail
# Resolve the object store on the box (mirror of scripts/lib/s3env.sh): the LAN
# store (tower SeaweedFS) by default, R2 only when ZM_STORE=r2. The worker gets
# the resolved endpoint via ZEN_R2_ENDPOINT (misnamed = the S3 endpoint) and the
# resolved creds via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
case "${ZM_STORE:-tower}" in
  r2|R2)
    CREDS="$HOME/.config/cloudflare/r2-credentials"
    [ -r "$CREDS" ] || { echo "no R2 creds at $CREDS on $(hostname) — distribute first" >&2; exit 3; }
    set -a; . "$CREDS"; set +a
    : "${R2_ACCOUNT_ID:?R2_ACCOUNT_ID missing from creds}"
    EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    AK="$R2_ACCESS_KEY_ID"; SK="$R2_SECRET_ACCESS_KEY" ;;
  *)
    CREDS="$HOME/.config/zen/lanstore.env"
    [ -r "$CREDS" ] || { echo "no LAN store creds at $CREDS on $(hostname) — distribute lanstore.env first" >&2; exit 3; }
    set -a; . "$CREDS"; set +a
    : "${ZEN_S3_ENDPOINT:?ZEN_S3_ENDPOINT missing from lanstore.env}"
    EP="$ZEN_S3_ENDPOINT"; AK="$ZEN_S3_ACCESS_KEY_ID"; SK="$ZEN_S3_SECRET_ACCESS_KEY" ;;
esac
WORKER="$(hostname)-${ZM_ROLE}"   # collision-proof: hostname is unique per box

# Rebuild multi-token docker args HERE (remote), where quoting is intact.
GPU_ARGS=(); REQ_GPU=()
if [ "$ZM_KIND" = "gpu" ]; then GPU_ARGS=(--gpus all); REQ_GPU=(-e ZEN_REQUIRE_GPU=1); fi

sudo -n docker pull "$ZM_IMG" >/dev/null 2>&1 || true
sudo -n docker rm -f "$ZM_CTR" >/dev/null 2>&1 || true
sudo -n docker run -d --name "$ZM_CTR" --restart unless-stopped "${GPU_ARGS[@]}" \
  -e AWS_ACCESS_KEY_ID="$AK" -e AWS_SECRET_ACCESS_KEY="$SK" -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT="$EP" -e ZEN_BUCKET="$ZM_BUCKET" \
  -e ZEN_RUN="jobs/$ZM_JOBSET" \
  -e ZEN_MANIFEST_URI="s3://$ZM_BUCKET/jobs/$ZM_JOBSET/manifest.json" \
  -e ZEN_CONTROL_KEY="jobs/$ZM_JOBSET/control.json" \
  "${REQ_GPU[@]}" -e ZEN_WORKER="$WORKER" -e ZEN_PROVIDER=basement \
  -e ZEN_MAX_MIN=1400 -e ZEN_IDLE_PASSES=8 -e ZEN_CORE_OVERSUBSCRIBE=2 \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh "$ZM_IMG" >/dev/null
echo "$(hostname) -> $ZM_JOBSET as $WORKER [store=${ZM_STORE:-tower} ep=$EP] : $(sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep "$ZM_CTR" || echo FAILED-TO-START)"
REMOTE
