#!/usr/bin/env bash
# lan_gpu_sequence.sh <host> <kind> <bucket1> [bucket2 ...]
#
# Drain several DIRECT-MANIFEST score buckets SEQUENTIALLY on ONE fleet box, in
# blocking single-run mode: each bucket's zenfleet worker runs to drain (idle
# self-exit) before the next launches, so one box (one GPU) chews through many
# buckets with no operator babysitting and no VRAM contention (never two GPU
# workers on the same card). This is the one-GPU-box counterpart to pool mode —
# pool mode (fleet-entrypoint.sh) sources variants from a tar/encodes prefix,
# which does not fit HDR direct-blob score_file manifests (inputs are s3:// URIs).
#
# Implementation: a standalone driver is scp'd to the box and run under setsid
# (survives ssh disconnect) with the buckets as positional args — no nested
# heredocs, and nothing space-bearing is passed through the ssh command line
# (the 2026-08-26 "command not found" class of bug). It logs to ~/lan_gpu_seq.log
# and drops ~/lan_gpu_seq.COMPLETE at the end.
#
#   kind = gpu (--gpus all + ZEN_REQUIRE_GPU=1 + CUDA image) | cpu
#   ZEN_WORKER = <hostname>-<role>, role from the bucket (…-small-→small,
#   …-huge-→huge, else med). Creds read on the box.
set -uo pipefail

HOST="${1:?usage: lan_gpu_sequence.sh <host> <gpu|cpu> <bucket>...}"
KIND="${2:?gpu|cpu}"; shift 2
[ "$#" -ge 1 ] || { echo "need >=1 bucket" >&2; exit 2; }
case "$KIND" in
  gpu) IMG="ghcr.io/imazen/zenfleet-worker:exec-gpu-avifgen-66e3c417" ;;
  cpu) IMG="ghcr.io/imazen/zenfleet-worker:exec-zensim944-2af6dbc3" ;;
  *) echo "kind must be gpu|cpu" >&2; exit 2 ;;
esac
S3BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/_lan_gpu_seq_driver.sh"   # standalone; committed alongside

scp -o BatchMode=yes -o ConnectTimeout=10 "$DRIVER" "$HOST:lan_gpu_seq_run.sh" >/dev/null \
  || { echo "scp driver to $HOST failed" >&2; exit 4; }
# env vars are single-token (safe on the ssh line); buckets are positional args.
# ZM_VRAM_CAP (bytes) → ZENMETRICS_VRAM_CAP_BYTES in the worker, so the GPU metric
# auto-picks Strip mode for large images instead of OOMing. Set it to the card's
# real VRAM (e.g. 5500000000 for a 6 GB card, 7500000000 for 8 GB). Space-free.
ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
  "chmod +x lan_gpu_seq_run.sh; setsid nohup env ZM_KIND='$KIND' ZM_IMG='$IMG' ZM_S3BUCKET='$S3BUCKET' ZM_VRAM_CAP='${ZM_VRAM_CAP:-}' ZM_STORE='${ZEN_STORE:-tower}' bash lan_gpu_seq_run.sh $* >/dev/null 2>&1 & echo \"\$(hostname): sequencer launched over: $* (store=${ZEN_STORE:-tower} vram_cap=${ZM_VRAM_CAP:-none})\""
