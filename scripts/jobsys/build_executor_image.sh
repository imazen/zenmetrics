#!/usr/bin/env bash
# Build (+ optionally push) the real-executor fleet image: worker base + prebuilt zenmetrics (with
# `jobexec`) + the zenfleet-exec shim. Run on a box that has already built zenmetrics (the codec sibling
# crates must be present — i.e. the workstation). amd64 only (the workstation arch); an arm64 image
# needs an arm64 zenmetrics binary built where the siblings live.
#
# USE THE MUSL TARGET (found + fixed 2026-08-24, see fleet.env's dated note): a dynamically-linked
# `cargo build --release` binary requires whatever glibc the BUILD box happens to run — the base
# image (`:latest`) is Debian bookworm / glibc 2.36, and a build box that has since drifted to a newer
# OS (e.g. Ubuntu 26.04 / glibc 2.43) produces a binary that BUILDS and even smoke-tests fine on the
# build box itself, then refuses to start inside the actual image (`GLIBC_2.43' not found`). The musl
# target sidesteps this permanently — a fully static binary, immune to the base image's libc version:
#
#   rustup target add x86_64-unknown-linux-musl   # + `apt install musl-tools` for musl-gcc, once
#   cargo build --release --target x86_64-unknown-linux-musl -p zenmetrics-cli --no-default-features \
#     --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics
#   # HDR waves (score_file hdr:true) need MORE: add `hdr-gainmap` (pulls `hdr`) or
#   # jpeg-gainmap variants fail to decode ("needs the hdr-gainmap build feature") —
#   # found the hard way on the 2026-08-26 hdrfeat944 first-cell gate. The HDR image
#   # convention: tag exec-zensim944hdr-<sha>, features
#   #   sweep,png,jpeg,webp,avif,jxl,cpu-metrics,hdr-gainmap
#     # produces target/x86_64-unknown-linux-musl/release/zenmetrics
#   cargo build --release --target x86_64-unknown-linux-musl -p zenfleet-worker
#     # produces target/x86_64-unknown-linux-musl/release/zenfleet-worker (this script overlays it
#     # automatically if present at the plain target/release/ path — pass ZEN_WORKER_BIN to point it
#     # at the musl path instead, or symlink/copy it there)
#   PUSH=1 ZEN_METRICS_BIN=target/x86_64-unknown-linux-musl/release/zenmetrics \
#     ZEN_WORKER_BIN=target/x86_64-unknown-linux-musl/release/zenfleet-worker \
#     bash scripts/jobsys/build_executor_image.sh         # build + push (needs ghcr login)
#
# Verify before shipping: `docker run --rm --entrypoint /bin/sh <image> -c 'ldd /usr/local/bin/zenmetrics'`
# must say "statically linked", never a glibc version list.
#
# Usage: [PUSH=1] [ZEN_METRICS_BIN=path] [ZEN_WORKER_BIN=path] build_executor_image.sh [IMAGE]
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
WK="${ZEN_WORKER_BIN:-$ROOT/target/release/zenfleet-worker}"
[ -f "$WK" ] && cp "$WK" "$CTX/zenfleet-worker" || echo "note: no local zenfleet-worker — image keeps base's"
# Overlay the CURRENT entrypoint — the :latest base bakes a stale one, so without this
# copy an edited fleet-entrypoint.sh never reaches :exec (this bit us on the tar-prefetch).
cp "$ROOT/crates/zenfleet-worker/fleet-entrypoint.sh" "$CTX/fleet-entrypoint.sh"
cp "$ROOT/crates/zenfleet-worker/Dockerfile.executor" "$CTX/Dockerfile"
echo "building $IMAGE (base = ghcr.io/imazen/zenfleet-worker:latest + zenmetrics $(du -h "$BIN" | cut -f1))"
# --chmod on COPY needs BuildKit (`docker buildx version` to check); if the build box's docker
# lacks the buildx plugin (seen on a fresh docker.io install, 2026-08-24), either install it
# (https://docs.docker.com/go/buildx/) or drop --chmod from Dockerfile.executor's COPY lines and add
# a trailing `RUN chmod 0755 ...` instead — the source files are already executable, but classic
# (non-BuildKit) `docker build` doesn't reliably preserve that without an explicit chmod step.
docker build -t "$IMAGE" "$CTX"
# Smoke: the binary loads + jobexec is present.
docker run --rm --entrypoint /usr/local/bin/zenmetrics "$IMAGE" jobexec --help >/dev/null \
  && echo "OK: jobexec present in $IMAGE"
if [ "${PUSH:-0}" = "1" ]; then docker push "$IMAGE" && echo "pushed $IMAGE"; fi
