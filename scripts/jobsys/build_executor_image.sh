#!/usr/bin/env bash
# Build (+ optionally push) the real-executor fleet image: worker base + prebuilt zenmetrics (with
# `jobexec`) + the zenfleet-exec shim. Run on a box that has already built zenmetrics (the codec sibling
# crates must be present — i.e. the workstation). amd64 only (the workstation arch); an arm64 image
# needs an arm64 zenmetrics binary built where the siblings live.
#
#   cargo build --release -p zenmetrics-cli --no-default-features \
#     --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics      # produces target/release/zenmetrics
#   PUSH=1 bash scripts/jobsys/build_executor_image.sh         # build + push (needs ghcr login)
#
# Usage: [PUSH=1] [ZEN_METRICS_BIN=path] build_executor_image.sh [IMAGE]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${1:-ghcr.io/imazen/zenfleet-worker:exec}"
BIN="${ZEN_METRICS_BIN:-$ROOT/target/release/zenmetrics}"
[ -x "$BIN" ] || { echo "build zenmetrics first (see header); not found at $BIN"; exit 1; }

# Stage under target/ (gitignored, and on a mount the docker daemon can read — unlike some sandbox
# TMPDIRs). Cleaned on exit.
CTX="$ROOT/target/.exec-ctx"; rm -rf "$CTX"; mkdir -p "$CTX"; trap 'rm -rf "$CTX"' EXIT
cp "$BIN" "$CTX/zenmetrics"
cp "$ROOT/scripts/jobsys/zenfleet-exec" "$CTX/zenfleet-exec"
# Overlay the CURRENT zenfleet-worker too (base :latest bakes a stale one). Built by
# `cargo build --release -p zenfleet-worker`. Skipped if absent (keeps base's).
WK="$ROOT/target/release/zenfleet-worker"
[ -f "$WK" ] && cp "$WK" "$CTX/zenfleet-worker" || echo "note: no local zenfleet-worker — image keeps base's"
# Overlay the CURRENT entrypoint — the :latest base bakes a stale one, so without this
# copy an edited fleet-entrypoint.sh never reaches :exec (this bit us on the tar-prefetch).
cp "$ROOT/crates/zenfleet-worker/fleet-entrypoint.sh" "$CTX/fleet-entrypoint.sh"
cp "$ROOT/crates/zenfleet-worker/Dockerfile.executor" "$CTX/Dockerfile"
echo "building $IMAGE (base = ghcr.io/imazen/zenfleet-worker:latest + zenmetrics $(du -h "$BIN" | cut -f1))"
docker build -t "$IMAGE" "$CTX"
# Smoke: the binary loads + jobexec is present.
docker run --rm --entrypoint /usr/local/bin/zenmetrics "$IMAGE" jobexec --help >/dev/null \
  && echo "OK: jobexec present in $IMAGE"
if [ "${PUSH:-0}" = "1" ]; then docker push "$IMAGE" && echo "pushed $IMAGE"; fi
