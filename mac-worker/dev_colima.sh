#!/usr/bin/env bash
# dev_colima.sh -- fastest "worker running on a Mac" path: NO native build, NO sibling repos.
# Runs the SAME ghcr.io/imazen/zenfleet-worker:exec container the tower runs, under headless
# colima (Docker without Docker Desktop). Reads mac-worker/worker.env for creds + pool settings.
#
#   brew install colima docker
#   cp worker.env.example worker.env    # fill in scoped creds (ubuntu-node/mint_cred.sh on the dev box)
#   bash dev_colima.sh
#
# Apple Silicon runs the amd64 :exec image EMULATED (works; slower) until an arm64 :exec is published
# (scripts/jobsys/build_executor_image.sh is amd64-only today). Intel Macs run it native-in-VM.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/worker.env" ] || { echo "FATAL: copy worker.env.example -> worker.env and fill it in first"; exit 1; }
set -a; . "$HERE/worker.env"; set +a
: "${ZEN_R2_ENDPOINT:?worker.env must set ZEN_R2_ENDPOINT}"
: "${AWS_ACCESS_KEY_ID:?worker.env must set AWS_ACCESS_KEY_ID}" "${AWS_SESSION_TOKEN:?scoped creds need AWS_SESSION_TOKEN}"

command -v colima >/dev/null || { echo "FATAL: brew install colima docker"; exit 2; }
command -v docker >/dev/null || { echo "FATAL: brew install docker"; exit 2; }

CPUS="${ZEN_THREADS:-4}"; MEM="${ZEN_COLIMA_MEM:-8}"
colima status >/dev/null 2>&1 || colima start --cpu "$CPUS" --memory "$MEM"

PLATFORM=""; [ "$(uname -m)" = "arm64" ] && PLATFORM="--platform linux/amd64"   # amd64 image, emulated on Apple Silicon
IMG="ghcr.io/imazen/zenfleet-worker:exec"
docker pull $PLATFORM "$IMG"
docker rm -f zen720-mac 2>/dev/null || true
docker run -d --name zen720-mac --restart unless-stopped $PLATFORM \
  --cpus "$CPUS" --memory "${MEM}g" \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_SESSION_TOKEN -e AWS_REGION="${AWS_REGION:-auto}" \
  -e ZEN_R2_ENDPOINT -e ZEN_BUCKET="${ZEN_BUCKET:-zentrain}" \
  -e ZEN_POOL_RUNLIST="${ZEN_POOL_RUNLIST:-s3://zentrain/jobs/_pool/runlist.tsv}" \
  -e ZEN_CORPUS_PREFIX="${ZEN_CORPUS_PREFIX:-refs/clean-picker-corpus-2026-06-26}" \
  -e ZEN_PROVIDER="${ZEN_PROVIDER:-mac}" -e ZEN_WORKER="${ZEN_WORKER:-mac-$(hostname -s)}" \
  -e RAYON_NUM_THREADS="${ZEN_THREADS:-1}" -e OMP_NUM_THREADS="${ZEN_THREADS:-1}" -e ZEN_CORE_OVERSUBSCRIBE=3 \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh "$IMG"
echo "started zen720-mac. logs: docker logs -f zen720-mac"
echo "stop:   docker rm -f zen720-mac   (colima stop  to halt the VM)"
