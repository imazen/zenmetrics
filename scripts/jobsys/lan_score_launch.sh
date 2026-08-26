#!/usr/bin/env bash
# lan_score_launch.sh — launch ONE LAN fleet box as a zenfleet score/feature worker
# against a declared job set, in single-run (direct-manifest) mode.
#
# STORE (2026-08-26 fix): writes go to whatever `scripts/lib/s3env.sh` resolves —
# the LAN store (tower SeaweedFS) BY DEFAULT (`ZEN_STORE=tower`, the migration
# target as of 2026-08-10). R2 (`ZEN_STORE=r2`) is the EXPLICIT opt-out, for the
# few legacy R2-hosted jobs only. The box SOURCES the canonical resolver
# (distributed to ~/.config/zen/s3env.sh alongside ~/.config/zen/lanstore.env) so
# creds stay on the box and there is ONE resolver, not a re-implementation.
#
# This replaces the old version that HARDCODED the R2 endpoint — the bug that had
# the whole LAN fleet writing ledgers to R2 (2026-08-26). The worker's
# `ZEN_R2_ENDPOINT` is misnamed: it is just "the S3 endpoint"; the worker reads and
# writes there regardless of R2 vs SeaweedFS.
#
# Usage:
#   lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]
#   env: ZEN_STORE=tower|r2 (default tower), ZEN_FLEET_BUCKET (default zentrain).
#
# Worker name = <hostname>-<role> (collision-proof). Container zen-score-<role>,
# --restart unless-stopped; self-exits on drain (ZEN_IDLE_PASSES); torn down with
# `docker rm -f zen-score-<role>`.
set -uo pipefail

HOST="${1:?usage: lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]}"
JOBSET="${2:?job-set (run prefix under s3://<bucket>/jobs/)}"
ROLE="${3:?role label (small/huge/medium/cpu/...)}"
KIND="${4:-gpu}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"
STORE="${ZEN_STORE:-tower}"   # LAN SeaweedFS by default; 'r2' = legacy opt-out

case "$KIND" in
  gpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-gpu-2af6dbc3" ;;
  cpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-zensim944-2af6dbc3" ;;
  *) echo "lan_score_launch: KIND must be gpu|cpu (got '$KIND')" >&2; exit 2 ;;
esac
IMG="${5:-$DEF_IMG}"
CTR="zen-score-${ROLE}"

# The box resolves the store by sourcing the CANONICAL s3env.sh (never a copy of
# its logic). Creds are read on the box (~/.config/zen/lanstore.env for the LAN
# store, ~/.config/cloudflare/r2-credentials for R2), never sent over the wire.
# Only SPACE-FREE single tokens cross the ssh line (a space would split the remote
# command — the 2026-08-26 `--gpus all` bug); GPU flags are rebuilt on the remote.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  ZM_JOBSET="$JOBSET" ZM_BUCKET="$BUCKET" ZM_ROLE="$ROLE" ZM_CTR="$CTR" \
  ZM_IMG="$IMG" ZM_KIND="$KIND" ZM_STORE="$STORE" ZM_VRAM_CAP="${ZEN_VRAM_CAP:-}" ZM_ENC_PREFIX="${ZEN_ENCODES_PREFIX:-}" ZM_CORPUS_PREFIX="${ZEN_CORPUS_PREFIX:-}" ZM_PASS_TIMEOUT="${ZEN_PASS_TIMEOUT:-}" 'bash -s' <<'REMOTE'
set -euo pipefail
export ZEN_STORE="${ZM_STORE:-tower}"
S3ENV="$HOME/.config/zen/s3env.sh"
[ -r "$S3ENV" ] || { echo "no s3env.sh at $S3ENV on $(hostname) — distribute scripts/lib/s3env.sh + lanstore.env first" >&2; exit 3; }
# shellcheck disable=SC1090
. "$S3ENV"   # exports EP, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ZEN_S3_STORE
WORKER="$(hostname)-${ZM_ROLE}"

GPU_ARGS=(); REQ_GPU=()
if [ "$ZM_KIND" = "gpu" ]; then GPU_ARGS=(--gpus all); REQ_GPU=(-e ZEN_REQUIRE_GPU=1); fi
# ZENMETRICS_VRAM_CAP_BYTES makes the GPU metric's auto mode pick Strip (bounded
# VRAM) instead of Full, so large HDR images don't OOM the card (2026-08-26). Set
# ZEN_VRAM_CAP to the card's usable VRAM (e.g. 5500000000 for 8 GB, 9500000000 for 12 GB).
VRAM=(); [ -n "${ZM_VRAM_CAP:-}" ] && VRAM=(-e "ZENMETRICS_VRAM_CAP_BYTES=$ZM_VRAM_CAP")
# Encode jobs resolve bare source names at <ZEN_ENCODES_PREFIX>/<name> (jobexec);
# forward it when the operator sets it (space-free, e.g. refs/imazen-26-hdr-grid-2026-06-14).
ENCP=(); [ -n "${ZM_ENC_PREFIX:-}" ] && ENCP=(-e "ZEN_ENCODES_PREFIX=$ZM_ENC_PREFIX")
# Encode jobs resolve bare cell.image_path at s3://$ZEN_BUCKET/$ZEN_CORPUS_PREFIX/<name>.
[ -n "${ZM_CORPUS_PREFIX:-}" ] && ENCP+=(-e "ZEN_CORPUS_PREFIX=$ZM_CORPUS_PREFIX")
# Pass budget override: big-cell workloads (HDR diffmaps) exceed the 1800s default and get
# rc=124-killed each pass, wasting the in-flight tail chunk (observed node-2 2026-08-26).
[ -n "${ZM_PASS_TIMEOUT:-}" ] && ENCP+=(-e "ZEN_PASS_TIMEOUT=$ZM_PASS_TIMEOUT")

sudo -n docker pull "$ZM_IMG" >/dev/null 2>&1 || true
sudo -n docker rm -f "$ZM_CTR" >/dev/null 2>&1 || true
sudo -n docker run -d --name "$ZM_CTR" --restart unless-stopped "${GPU_ARGS[@]}" "${VRAM[@]}" "${ENCP[@]}" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT="$EP" -e ZEN_BUCKET="$ZM_BUCKET" \
  -e ZEN_RUN="jobs/$ZM_JOBSET" \
  -e ZEN_MANIFEST_URI="s3://$ZM_BUCKET/jobs/$ZM_JOBSET/manifest.json" \
  -e ZEN_CONTROL_KEY="jobs/$ZM_JOBSET/control.json" \
  "${REQ_GPU[@]}" -e ZEN_WORKER="$WORKER" -e ZEN_PROVIDER=basement \
  -e ZEN_MAX_MIN=1400 -e ZEN_IDLE_PASSES=8 -e ZEN_CORE_OVERSUBSCRIBE=2 \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh "$ZM_IMG" >/dev/null
echo "$(hostname) -> $ZM_JOBSET as $WORKER [store=$ZEN_S3_STORE ep=$EP] : $(sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep "$ZM_CTR" || echo FAILED-TO-START)"
REMOTE
