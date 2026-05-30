#!/usr/bin/env bash
# Build (+ optionally push) the real-executor fleet image: worker base + prebuilt zen-metrics (with
# `jobexec`) + the zen-jobexec shim. Run on a box that has already built zen-metrics (the codec sibling
# crates must be present — i.e. the workstation). amd64 only (the workstation arch); an arm64 image
# needs an arm64 zen-metrics binary built where the siblings live.
#
#   cargo build --release -p zen-metrics-cli --no-default-features \
#     --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics      # produces target/release/zen-metrics
#   PUSH=1 bash scripts/jobsys/build_executor_image.sh         # build + push (needs ghcr login)
#
# Usage: [PUSH=1] [ZEN_METRICS_BIN=path] build_executor_image.sh [IMAGE]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${1:-ghcr.io/imazen/zen-jobworker-exec:latest}"
BIN="${ZEN_METRICS_BIN:-$ROOT/target/release/zen-metrics}"
[ -x "$BIN" ] || { echo "build zen-metrics first (see header); not found at $BIN"; exit 1; }

# Stage under target/ (gitignored, and on a mount the docker daemon can read — unlike some sandbox
# TMPDIRs). Cleaned on exit.
CTX="$ROOT/target/.exec-ctx"; rm -rf "$CTX"; mkdir -p "$CTX"; trap 'rm -rf "$CTX"' EXIT
cp "$BIN" "$CTX/zen-metrics"
cp "$ROOT/scripts/jobsys/zen-jobexec" "$CTX/zen-jobexec"
cp "$ROOT/crates/zen-jobworker/Dockerfile.executor" "$CTX/Dockerfile"
echo "building $IMAGE (base = ghcr.io/imazen/zen-jobworker:latest + zen-metrics $(du -h "$BIN" | cut -f1))"
docker build -t "$IMAGE" "$CTX"
# Smoke: the binary loads + jobexec is present.
docker run --rm --entrypoint /usr/local/bin/zen-metrics "$IMAGE" jobexec --help >/dev/null \
  && echo "OK: jobexec present in $IMAGE"
if [ "${PUSH:-0}" = "1" ]; then docker push "$IMAGE" && echo "pushed $IMAGE"; fi
