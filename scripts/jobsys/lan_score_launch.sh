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
#        ZEN_CORPUS_BUCKET / ZEN_CORPUS_PREFIX -> where encode jobs resolve cell.image_path
#        ZEN_CPUSET / ZEN_CPU_SHARES / ZEN_MEMORY -> docker --cpuset-cpus / --cpu-shares / --memory
#        (the tower rule: e.g. ZEN_CPUSET=0-23 ZEN_CPU_SHARES=256 ZEN_MEMORY=24g — before 2026-08-30
#        the remote read these but the ssh line never forwarded them, so tower launches were uncapped
#        unless hand-run).
#        ZEN_TMPDIR_HOST_DIR -> override the host-side scratch dir bind-mounted at /scratch (TMPDIR
#        discipline, 2026-09-05: every launch gets a disk-backed TMPDIR, never bare /tmp). Default
#        auto-detects: /mnt/user/coefficient/scratch when the remote has an Unraid array mounted
#        (tower), else $HOME/tmp/zfw-scratch on the remote.
#
# Worker name = <hostname>-<role> (collision-proof). Container zen-score-<role>;
# self-exits on drain (ZEN_IDLE_PASSES, default 8) or ZEN_MAX_MIN budget; torn down with
#   ZEN_CORE_OVERSUBSCRIBE (default 2) sets concurrent cells per core. 2 is right for
#   IO- or stall-heavy work; for a memory-bandwidth-bound codec it thrashes (measured
#   on zenrav1e: 34.8 CPU-s/cell oversubscribed vs the intrinsic ~13.8). Set 1 there.
#   ⛔ ZEN_LONG_LIVED=1 PINS THE MANIFEST. fleet-entrypoint.sh fetches
#   ZEN_MANIFEST_URI ONCE, outside the pass loop, so a long-lived worker never sees
#   jobs declared after it started. That is fine for a FIXED manifest (an encode run
#   whose cells are all declared up front) and WRONG for a GROWING one -- a
#   gap-fill-fed SCORE run, whose manifest is re-declared every round as encodes
#   land. Measured on avifdoe-br-sf-cpu-20260903: the scorer sat idle at 0% CPU for
#   3 hours on a manifest it had fetched exactly once, while 9,628 pairs waited.
#   The same trap bites a DECLARATION SHRINK: swapping manifest.json (the sanctioned
#   de-scope path) does NOT reach a running long-lived worker -- measured, both hosts
#   kept reporting the pre-shrink 4,425,398-byte manifest until restarted. ANY
#   manifest change requires restarting long-lived workers to take effect.
#   For a gap-fill-fed scorer use ZEN_LONG_LIVED=0 with a large ZEN_IDLE_PASSES and
#   `docker update --restart unless-stopped`: there the drain-exit-restart cycle IS
#   the manifest-refresh mechanism, and it is the one case where unless-stopped is
#   correct rather than the restart-loop bug the 2026-09-03 fix removed.
#   ZEN_LONG_LIVED=1 is the right knob for a SCORE worker on a FIXED manifest:
#   such a worker legitimately runs out of scorable cells during any lull on the encode
#   side, drains, and -- since the 2026-09-03 --restart on-failure:5 fix, correctly --
#   stays stopped, leaving nothing scoring while the wave keeps producing. Under
#   ZEN_LONG_LIVED=1 idle passes back off exponentially instead of counting toward the
#   drain, which is exactly this case; the entrypoint documents it as "for the always-on
#   LAN fleet only".
#   ZEN_IDLE_PASSES also raises the drain threshold but is NOT the right lever here and
#   was tried first: it counts PASSES, and an idle pass costs ZEN_PASS_SLEEP (0.2s), so
#   even 200 drained in about two minutes on avifdoe-br-sf-cpu-20260903. Both defaults
#   are unchanged.
# `docker rm -f zen-score-<role>`.
#
# --restart on-failure:5 (2026-09-03 fix; was `unless-stopped`). The worker's
# own clean drain-exit (fleet-entrypoint.sh, both the ZEN_IDLE_PASSES and the
# ZEN_MAX_MIN paths) is `exit 0` with nothing after it in the script — and
# `unless-stopped` restarts a container on ANY exit until it is explicitly
# `docker stop`/`rm`'d, so a worker whose run had already gone COMPLETE looped
# forever, re-fetching its manifest every cycle. Found on 10 containers across
# 3 hosts, up to 7,457 restarts on one. Docker's `on-failure` policy restarts
# ONLY on a non-zero exit, so a clean drain now stays stopped, while a genuine
# crash/hang/fail-fast (exit 143/3/4/5) still gets bounded retries instead of
# an infinite loop — the same policy already used for the Hetzner worker
# (crates/zenfleet-hetzner/src/cloud_init.rs). Record: docs/RUNNING_JOBS.md
# "Restart-loop on clean drain", benchmarks/avif_doe_plan_2026-09-01.md §18
# point 2.
set -uo pipefail

HOST="${1:?usage: lan_score_launch.sh <host> <job-set> <role> [gpu|cpu] [image]}"
JOBSET="${2:?job-set (run prefix under s3://<bucket>/jobs/)}"
ROLE="${3:?role label (small/huge/medium/cpu/...)}"
KIND="${4:-gpu}"
BUCKET="${ZEN_FLEET_BUCKET:-zentrain}"
STORE="${ZEN_STORE:-tower}"   # LAN SeaweedFS by default; 'r2' = legacy opt-out

case "$KIND" in
  gpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-gpu-6d4f9963" ;;
  cpu) DEF_IMG="ghcr.io/imazen/zenfleet-worker:exec-zensim944hdr-9dffa5ca" ;;
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
  ZM_IMG="$IMG" ZM_KIND="$KIND" ZM_STORE="$STORE" ZM_VRAM_CAP="${ZEN_VRAM_CAP:-}" ZM_CPUSET="${ZEN_CPUSET:-}" ZM_CPU_SHARES="${ZEN_CPU_SHARES:-}" ZM_MEMORY="${ZEN_MEMORY:-}" ZM_ENC_PREFIX="${ZEN_ENCODES_PREFIX:-}" ZM_CORPUS_PREFIX="${ZEN_CORPUS_PREFIX:-}" ZM_CORPUS_BUCKET="${ZEN_CORPUS_BUCKET:-}" ZM_PASS_TIMEOUT="${ZEN_PASS_TIMEOUT:-}" ZM_CHUNK_WALL="${ZEN_CHUNK_WALL_SEC:-}" ZM_IDLE_PASSES="${ZEN_IDLE_PASSES:-}" ZM_LONG_LIVED="${ZEN_LONG_LIVED:-}" ZM_OVERSUB="${ZEN_CORE_OVERSUBSCRIBE:-}" ZM_CAPABILITY="${ZEN_CAPABILITY:-}" ZM_REQ_SNAP="${ZEN_REQUIRE_SNAPSHOT:-1}" ZM_CPUSET="${ZEN_CPUSET:-}" ZM_CPU_SHARES="${ZEN_CPU_SHARES:-}" ZM_MEMORY="${ZEN_MEMORY:-}" ZM_TMPDIR_HOST_DIR="${ZEN_TMPDIR_HOST_DIR:-}" 'bash -s' <<'REMOTE'
set -euo pipefail
export ZEN_STORE="${ZM_STORE:-tower}"
S3ENV="$HOME/.config/zen/s3env.sh"
[ -r "$S3ENV" ] || { echo "no s3env.sh at $S3ENV on $(hostname) — distribute scripts/lib/s3env.sh + lanstore.env first" >&2; exit 3; }
# shellcheck disable=SC1090
. "$S3ENV"   # exports EP, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, ZEN_S3_STORE
WORKER="$(hostname)-${ZM_ROLE}"

# TMPDIR discipline (ban RAM-backed tmp everywhere, 2026-09-05): bind-mount a disk-backed
# scratch dir at /scratch and export TMPDIR to it — fleet-entrypoint.sh's
# check_tmpdir_discipline refuses to boot without this. Auto-detect an Unraid array (the
# tower convention: /mnt/user/<share>/scratch, never the RAM-booted host root) vs a plain
# box (this account's ~/tmp, per the workspace-wide "/tmp is banned, use ~/tmp" rule);
# override with ZEN_TMPDIR_HOST_DIR for a box with its own convention.
if [ -n "${ZM_TMPDIR_HOST_DIR:-}" ]; then
  SCRATCH_HOST_DIR="$ZM_TMPDIR_HOST_DIR"
elif [ -d /mnt/user/coefficient ]; then
  SCRATCH_HOST_DIR="/mnt/user/coefficient/scratch"
else
  SCRATCH_HOST_DIR="$HOME/tmp/zfw-scratch"
fi
mkdir -p "$SCRATCH_HOST_DIR"
TMPDIR_ARGS=(-e TMPDIR=/scratch -v "$SCRATCH_HOST_DIR:/scratch")

GPU_ARGS=(); REQ_GPU=()
if [ "$ZM_KIND" = "gpu" ]; then GPU_ARGS=(--gpus all); REQ_GPU=(-e ZEN_REQUIRE_GPU=1); fi
# ZENMETRICS_VRAM_CAP_BYTES makes the GPU metric's auto mode pick Strip (bounded
# VRAM) instead of Full, so large HDR images don't OOM the card (2026-08-26). Set
# ZEN_VRAM_CAP to the card's usable VRAM (e.g. 5500000000 for 8 GB, 9500000000 for 12 GB).
VRAM=(); [ -n "${ZM_VRAM_CAP:-}" ] && VRAM=(-e "ZENMETRICS_VRAM_CAP_BYTES=$ZM_VRAM_CAP")
# Encode jobs resolve bare source names at <ZEN_ENCODES_PREFIX>/<name> (jobexec);
# forward it when the operator sets it (space-free, e.g. refs/imazen-26-hdr-grid-2026-06-14).
ENCP=(); [ -n "${ZM_ENC_PREFIX:-}" ] && ENCP=(-e "ZEN_ENCODES_PREFIX=$ZM_ENC_PREFIX")
# Encode jobs resolve bare cell.image_path at
# s3://${ZEN_CORPUS_BUCKET:-$ZEN_BUCKET}/$ZEN_CORPUS_PREFIX/<name> (jobexec.rs:400).
# ZEN_CORPUS_BUCKET is REQUIRED whenever the corpus lives in its own read-only bucket
# (the normal case: codec-corpus). Without it jobexec falls back to the run bucket and
# every cell fails to resolve its source — added 2026-09-02 after this script could not
# launch the AVIF-DOE encode arms, whose corpus is in codec-corpus.
[ -n "${ZM_CORPUS_PREFIX:-}" ] && ENCP+=(-e "ZEN_CORPUS_PREFIX=$ZM_CORPUS_PREFIX")
[ -n "${ZM_CORPUS_BUCKET:-}" ] && ENCP+=(-e "ZEN_CORPUS_BUCKET=$ZM_CORPUS_BUCKET")
# Pass budget override: big-cell workloads (HDR diffmaps) exceed the 1800s default and get
# rc=124-killed each pass, wasting the in-flight tail chunk (observed node-2 2026-08-26).
[ -n "${ZM_PASS_TIMEOUT:-}" ] && ENCP+=(-e "ZEN_PASS_TIMEOUT=$ZM_PASS_TIMEOUT")
[ -n "${ZM_CHUNK_WALL:-}" ] && ENCP+=(-e "ZEN_CHUNK_WALL_SEC=$ZM_CHUNK_WALL")
[ -n "${ZM_CAPABILITY:-}" ] && ENCP+=(-e "ZEN_CAPABILITY=$ZM_CAPABILITY")
ENCP+=(-e "ZEN_REQUIRE_SNAPSHOT=${ZM_REQ_SNAP}")   # strict by default for single-run queues; ZEN_REQUIRE_SNAPSHOT=0 opts out
# Resource caps for shared boxes (tower rule: never an uncapped worker on the media server).
CAPS=()
[ -n "${ZM_CPUSET:-}" ] && CAPS+=(--cpuset-cpus "$ZM_CPUSET")
[ -n "${ZM_CPU_SHARES:-}" ] && CAPS+=(--cpu-shares "$ZM_CPU_SHARES")
[ -n "${ZM_MEMORY:-}" ] && CAPS+=(--memory "$ZM_MEMORY")

sudo -n docker pull "$ZM_IMG" >/dev/null 2>&1 || true
sudo -n docker rm -f "$ZM_CTR" >/dev/null 2>&1 || true
sudo -n docker run -d --name "$ZM_CTR" ${CAPS[@]+"${CAPS[@]}"} --restart on-failure:5 "${GPU_ARGS[@]}" "${VRAM[@]}" "${ENCP[@]}" "${TMPDIR_ARGS[@]}" \
  -e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" -e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT="$EP" -e ZEN_BUCKET="$ZM_BUCKET" \
  -e ZEN_RUN="jobs/$ZM_JOBSET" \
  -e ZEN_MANIFEST_URI="s3://$ZM_BUCKET/jobs/$ZM_JOBSET/manifest.json" \
  -e ZEN_CONTROL_KEY="jobs/$ZM_JOBSET/control.json" \
  "${REQ_GPU[@]}" -e ZEN_WORKER="$WORKER" -e ZEN_PROVIDER=basement \
  -e ZEN_MAX_MIN=1400 -e ZEN_IDLE_PASSES="${ZM_IDLE_PASSES:-8}" -e ZEN_CORE_OVERSUBSCRIBE="${ZM_OVERSUB:-2}" \
  -e ZEN_LONG_LIVED="${ZM_LONG_LIVED:-0}" \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh "$ZM_IMG" >/dev/null
echo "$(hostname) -> $ZM_JOBSET as $WORKER [store=$ZEN_S3_STORE ep=$EP] : $(sudo -n docker ps --format '{{.Names}} {{.Status}}' | grep "$ZM_CTR" || echo FAILED-TO-START)"
REMOTE
