#!/usr/bin/env bash
# Build (+ optionally push) the GPU real-executor fleet image: the CUDA sweep image
# (ghcr.io/imazen/zenmetrics-sweep:v29-*, which already ships a GPU `zenmetrics` with `jobexec`
# + the CUDA runtime) with the fleet-worker pieces COPYed in (zenfleet-worker, zenfleet-gc,
# aws-cli, fleet-entrypoint.sh, zenfleet-exec). This is the GPU counterpart of
# scripts/jobsys/build_executor_image.sh — it closes the gap that forced ad-hoc SPLIT scoring:
# the stock exec image's GPU metrics fail because its base is non-CUDA.
#
# No local cargo build needed: the GPU `zenmetrics` is already inside the v29 GPU_BASE, and the
# zenfleet-worker/-gc binaries + aws-cli come from the WORKER_BASE via a multi-stage COPY. So
# this script does NOT require the codec sibling crates to be present.
#
#   bash scripts/jobsys/build_executor_image_gpu.sh                 # build + smoke only
#   PUSH=1 bash scripts/jobsys/build_executor_image_gpu.sh          # build + smoke + push (ghcr login)
#   GPU_BASE=ghcr.io/imazen/zenmetrics-sweep:v29-split-feat \
#     bash scripts/jobsys/build_executor_image_gpu.sh               # different GPU base
#
# Usage: [PUSH=1] [WORKER_BASE=…] [GPU_BASE=…] build_executor_image_gpu.sh [IMAGE]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${1:-ghcr.io/imazen/zenfleet-worker-exec-gpu:latest}"
WORKER_BASE="${WORKER_BASE:-ghcr.io/imazen/zenfleet-worker:latest}"
GPU_BASE="${GPU_BASE:-ghcr.io/imazen/zenmetrics-sweep:v29-2026-06-23}"

# Stage under target/ (gitignored; on a mount the snap docker daemon can read — unlike /tmp).
CTX="$ROOT/target/.exec-ctx-gpu"; rm -rf "$CTX"; mkdir -p "$CTX"; trap 'rm -rf "$CTX"' EXIT
cp "$ROOT/scripts/jobsys/Dockerfile.executor.gpu" "$CTX/Dockerfile"
cp "$ROOT/crates/zenfleet-worker/fleet-entrypoint.sh" "$CTX/fleet-entrypoint.sh"
cp "$ROOT/scripts/jobsys/zenfleet-exec" "$CTX/zenfleet-exec"
# Freshly-built binaries baked over the GPU_BASE: the persistent `--serve` executor lives in BOTH
# zenmetrics (the warm loop) and zenfleet-worker (the warm-child handler). Build them first with
# `cargo build --release -p zenmetrics-cli --no-default-features --features sweep,png,jpeg,webp,avif,jxl,gpu,gpu-cuda`
# and `-p zenfleet-worker`.
[ -x "$ROOT/target/release/zenmetrics" ] || { echo "FATAL: target/release/zenmetrics missing — build it first"; exit 1; }
[ -x "$ROOT/target/release/zenfleet-worker" ] || { echo "FATAL: target/release/zenfleet-worker missing — build it first"; exit 1; }
cp "$ROOT/target/release/zenmetrics" "$CTX/zenmetrics"
cp "$ROOT/target/release/zenfleet-worker" "$CTX/zenfleet-worker"

echo "building $IMAGE"
echo "  WORKER_BASE = $WORKER_BASE   (source of zenfleet-worker/-gc + aws-cli + entrypoint)"
echo "  GPU_BASE    = $GPU_BASE      (CUDA runtime + GPU zenmetrics + jobexec)"
docker build \
  --build-arg WORKER_BASE="$WORKER_BASE" \
  --build-arg GPU_BASE="$GPU_BASE" \
  -t "$IMAGE" "$CTX"

echo "=== smoke ==="
# 1) the GPU zenmetrics + its jobexec --serve (persistent warm executor) are baked in
docker run --rm --entrypoint /usr/local/bin/zenmetrics "$IMAGE" jobexec --help 2>&1 \
  | grep -q -- '--serve' && echo "OK: zenmetrics jobexec --serve present (persistent executor)"
# 2) GPU metrics are compiled in (list-metrics shows requires_gpu=yes rows -> GPU build)
docker run --rm --entrypoint /usr/local/bin/zenmetrics "$IMAGE" list-metrics 2>&1 \
  | grep -qE 'cvvdp .*GPU .*yes' && echo "OK: GPU metrics (cvvdp) compiled in"
# 3) the fleet-worker binary loads (glibc 2.36 binary on the v29 2.39 rootfs)
docker run --rm --entrypoint /usr/local/bin/zenfleet-worker "$IMAGE" --help >/dev/null \
  && echo "OK: zenfleet-worker loads"
# 4) aws-cli v2 loads (its conditional-write lease is what makes GPU jobs claimable)
docker run --rm --entrypoint /usr/local/bin/aws "$IMAGE" --version >/dev/null \
  && echo "OK: aws-cli loads"
# 5) the ZEN_EXEC default + entrypoint are wired
docker run --rm --entrypoint sh "$IMAGE" -c \
  'test "$ZEN_EXEC" = /usr/local/bin/zenfleet-exec && test -x /usr/local/bin/zenfleet-exec && test -x /usr/local/bin/fleet-entrypoint.sh' \
  && echo "OK: ZEN_EXEC default + entrypoint shim wired"

if [ "${PUSH:-0}" = "1" ]; then docker push "$IMAGE" && echo "pushed $IMAGE"; fi
