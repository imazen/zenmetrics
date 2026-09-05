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
# 2026-08-26 (2nd instance, GPU side): the DIFFMAP route decodes variants too —
#   the exec-gpu image built without `hdr-gainmap` failed 100% of jpeg-gainmap
#   diffmap cells (46,680 rows, hdrgrid-diffmap-20260807) as encoder_panic while
#   zenjxl/svt sailed. EVERY image that decodes HDR variants (features, scores,
#   diffmaps; CPU or GPU) needs `hdr-gainmap` in its cargo features.
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
# `cargo build --release -p zenfleet-worker`. REQUIRED, not optional: Dockerfile.executor
# COPYs `zenfleet-worker` unconditionally, so "skip it and keep the base's" was never a
# reachable outcome — it just turned a clear precondition into an opaque `COPY failed: file
# not found` two minutes into a docker build. Requiring it is also the safer half of the
# disagreement: a silently-inherited stale worker is exactly the failure that cost the
# AVIF-DOE wave 93 % of its main-effects arm (plan section 11.5/11.7).
# Prefer the STATIC musl worker when it exists: the glibc build inherits the
# build box's libc (Ubuntu 26.04 -> GLIBC_2.38/2.39 symbols) and refuses to start
# inside the 24.04 base ("version `GLIBC_2.39' not found", 2026-08-30 first-chunk
# gate on r5900xt: every pass rc=1 before a single cell ran). Same trap the
# header describes for zenmetrics; the overlay had the glibc default.
WK_MUSL="$ROOT/target/x86_64-unknown-linux-musl/release/zenfleet-worker"
WK="${ZEN_WORKER_BIN:-$([ -f "$WK_MUSL" ] && echo "$WK_MUSL" || echo "$ROOT/target/release/zenfleet-worker")}"
[ -f "$WK" ] || {
  echo "build zenfleet-worker first; not found at $WK" >&2
  echo "  cargo build --release --target x86_64-unknown-linux-musl -p zenfleet-worker" >&2
  echo "  (or point ZEN_WORKER_BIN at one). Dockerfile.executor COPYs it unconditionally," >&2
  echo "  so there is no 'keep the base's worker' path to fall back to." >&2
  exit 1
}
if file "$WK" 2>/dev/null | grep -q "dynamically linked"; then
  echo "WARNING: overlaying a DYNAMICALLY linked zenfleet-worker ($WK) — it must not need a newer glibc than the base image; build the musl target instead (cargo build --release --target x86_64-unknown-linux-musl -p zenfleet-worker)" >&2
fi
cp "$WK" "$CTX/zenfleet-worker"
# Overlay the CURRENT entrypoint — the :latest base bakes a stale one, so without this
# copy an edited fleet-entrypoint.sh never reaches :exec (this bit us on the tar-prefetch).
cp "$ROOT/crates/zenfleet-worker/fleet-entrypoint.sh" "$CTX/fleet-entrypoint.sh"
# fleet-entrypoint.sh sources this at boot (TMPDIR discipline guard) — must travel with it.
cp "$ROOT/crates/zenfleet-worker/tmpdir_discipline.sh" "$CTX/tmpdir_discipline.sh"
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
